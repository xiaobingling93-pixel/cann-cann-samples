/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file block_scheduler_mx.h
 * \brief
 */

#ifndef BLOCK_SCHEDULER_MX_H
#define BLOCK_SCHEDULER_MX_H

#include "./block_scheduler_utils.h"
#include "./block_scheduler_policy.h"
#include "kernel_utils/common_utils.h"

namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
class BlockSchedulerQuantMatmulMx {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t mBaseNormCnt_{0};
    int64_t nBaseNormCnt_{0};
    int64_t mBaseTailMain_{0};
    int64_t nBaseTailMain_{0};
    int64_t mBaseTailLast_{0};
    int64_t nBaseTailLast_{0};
    int64_t mCoreNum_{0};
    int64_t mTailCoreNum_{0};
    int64_t blockIdx_{AscendC::GetBlockIdx() / AscendC::GetTaskRation()};
    int64_t blockNum_{AscendC::GetBlockNum()};
    int64_t startBlockIdx_{0};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t mTailTile_{1}; // init value must be 1
    int64_t nTailTile_{1}; // init value must be 1
    int64_t totalTailTile_{1}; // init value must be 1
    int64_t mSplitAddrOffset_{0};
    int64_t nSplitAddrOffset_{0};
    int64_t mainRow_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t baseM;
        int64_t baseN;
        int64_t mTailTile;
        int64_t nTailTile;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
    };

    const int64_t WINDOW_LEN = 4;

public:
    __aicore__ inline BlockSchedulerQuantMatmulMx(const ProblemShape &shape, const Params &params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mCnt_ = CeilDiv(m_, baseM_);
        nCnt_ = CeilDiv(n_, baseN_);
        totalCnt_ = mCnt_ * nCnt_;
        mCoreNum_ = Min(WINDOW_LEN, mCnt_);
        mainRow_ = mCnt_ / mCoreNum_ - 1;
        mTailCoreNum_ = mCnt_ - mCoreNum_ * mainRow_;
        endBlockIdx_ = (totalCnt_ - 1) % blockNum_;
        round_ = CeilDiv(totalCnt_, blockNum_);
        if (blockIdx_ > endBlockIdx_) {
            round_ -= 1;
        }
        mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
        int64_t mMergeSize = m_ - mBaseNormCnt_ * baseM_;
        mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
        mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;

        nBaseNormCnt_ = nCnt_ - params.nBaseTailSplitCnt;
        int64_t nMergeSize = n_ - nBaseNormCnt_ * baseN_;
        nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nMergeSize : params.nTailMain;
        nBaseTailLast_ = nMergeSize - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
    {
        mTailTile_ = mTailTile;
        nTailTile_ = nTailTile;
        totalTailTile_ = mTailTile * nTailTile;
        uint64_t tailOriCnt = AscendC::Std::min(totalCnt_, endBlockIdx_ + 1);
        int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (totalTailTile_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailTile_ = 1;
            nTailTile_ = 1;
            totalTailTile_ = 1;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline int64_t GetTotalCnt()
    {
        return totalCnt_;
    }

    __aicore__ inline int64_t GetEndBlockIdx()
    {
        return endBlockIdx_;
    }

    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t singleCoreM = baseM_;
        int64_t singleCoreN = baseN_;

        if (Get<MNK_M>(blockCoord) >= mBaseNormCnt_) {
            singleCoreM = Get<MNK_M>(blockCoord) < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
        }
        if (Get<MNK_N>(blockCoord) >= nBaseNormCnt_) {
            singleCoreN = Get<MNK_N>(blockCoord) < nCnt_ - 1 ? nBaseTailMain_ : nBaseTailLast_;
        }

        if (totalTailTile_ == 1 || roundIdx_ < round_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = CeilDiv(singleCoreM, mTailTile_);
        int64_t singleCoreNSplit = CeilDiv(singleCoreN, nTailTile_);
        int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
        int64_t nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
        mSplitAddrOffset_ = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset_ = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset_ >= singleCoreM || nSplitAddrOffset_ >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset_, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset_, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
    }

    __aicore__ inline AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLoadBalanceInfo()
    {
        return {static_cast<uint32_t>(mBaseNormCnt_), static_cast<uint32_t>(mBaseTailMain_),
                static_cast<uint32_t>(nBaseNormCnt_), static_cast<uint32_t>(nBaseTailMain_)};
    }

    __aicore__ inline bool GetTileIdx(BlockCoord &blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }

        int64_t newBlockIdx = (roundIdx_ == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
        int64_t tileIdx = newBlockIdx + roundIdx_ * blockNum_;
        if (blockIdx_ < startBlockIdx_) {
            tileIdx += blockNum_ - startBlockIdx_;
        } else if (endBlockIdx_ + 1 >= totalTailTile_ * totalCnt_) {
            tileIdx -= startBlockIdx_ / totalTailTile_;
        } else {
            tileIdx -= startBlockIdx_;
        }
        int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
        if (rowIdx < mainRow_) {
            Get<MNK_M>(blockCoord) = rowIdx * mCoreNum_ + tileIdx % mCoreNum_;
            Get<MNK_N>(blockCoord) = (tileIdx / mCoreNum_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
            Get<MNK_M>(blockCoord) = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
            Get<MNK_N>(blockCoord) = (tailIdx / mTailCoreNum_) % nCnt_;
        }
        if (rowIdx & 1) {
            Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
        }
        roundIdx_++;
        return true;
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, QuantMatmulMxAswtScheduler,
                              TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerQuantMatmulMx<ProblemShape_, L1TileShape_, L0TileShape_, TransA_, TransB_>;
};
}  // namespace Block
#endif