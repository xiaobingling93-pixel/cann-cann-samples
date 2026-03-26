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
 * \brief MXFP4 BlockMmad (Te API): K-tile MMAD uses unitFlag; L0C→GM uses FixpipeParams (see recipe).
 */
#ifndef MATMUL_BLOCK_MMAD_MX_SWAT_H
#define MATMUL_BLOCK_MMAD_MX_SWAT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../../common/kernel_utils/layout_utils.h"
#include "../../../../common/kernel_utils/common_utils.h"
#include "../../../../common/kernel_utils/tuple_utils.h"
#include "include/tensor.h"
#include "../utils/quant_matmul_constant.h"
#include "../../../common/include/tile/tile_mmad_mx.h"
#include "../../../common/include/tile/copy_scale_l1_to_l0a.h"
#include "../../../common/include/tile/copy_scale_l1_to_l0b.h"
#include "../../../common/include/tile/copy_scale_gm_to_l1.h"

namespace Block {
template <class AType_, class BType_, class CType_>
class BlockMmadMx {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = false;
    static constexpr bool transB = true;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1{0};
        uint64_t scaleKL1{0};
        uint64_t l1BufNum{2};
    };

    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{256};
    uint64_t kL1_{1024};
    uint64_t kL1Iter_{0};
    uint64_t l1BufNum_{2};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;

    __aicore__ inline BlockMmadMx()
    {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params)
    {
        m_ = static_cast<uint64_t>(Get<IDX_M_IDX>(problemShape));
        n_ = static_cast<uint64_t>(Get<IDX_N_IDX>(problemShape));
        k_ = static_cast<uint64_t>(Get<IDX_K_IDX>(problemShape));
        baseM_ = static_cast<uint64_t>(Get<IDX_M_IDX>(l0TileShape));
        baseN_ = static_cast<uint64_t>(Get<IDX_N_IDX>(l0TileShape));
        baseK_ = static_cast<uint64_t>(Get<IDX_K_IDX>(l0TileShape));
        if (baseK_ == 0) {
            baseK_ = 128 / sizeof(fp4x2_e2m1_t);
        }
        kL1_ = (l1Params.kL1 > 0) ? l1Params.kL1 : (baseK_ * 4);
        l1BufNum_ = l1Params.l1BufNum;
        kL1Iter_ = CeilDiv(k_, kL1_);

        // Packed fp4 halves A/B footprint; L1 bank is half of L1_SIZE per ping-pong side.
        bL1OneBuffer_ = (baseN_ * kL1_) >> 1;
        aL1OneBuffer_ = (baseM_ * Align(kL1_, MXFP_DIVISOR_SIZE)) >> 1;
        scaleAL1OneBuffer_ = baseM_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        scaleBL1OneBuffer_ = baseN_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

        const uint64_t l1BankSize = static_cast<uint64_t>(L1_SIZE >> 1);
        for (int32_t bufferId = 0; bufferId < static_cast<int32_t>(l1BufNum_); ++bufferId) {
            uint64_t l1Offset = l1BankSize * static_cast<uint64_t>(bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * static_cast<uint64_t>(bufferId >> 1);
            l1BufferBOffset_[bufferId] =
                l1Offset + aL1OneBuffer_ * static_cast<uint64_t>(l1BufNum_ >> 1) + bL1OneBuffer_ * static_cast<uint64_t>(bufferId >> 1);
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * static_cast<uint64_t>(l1BufNum_ >> 1);
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TensorScaleA gmScaleA, TensorScaleB gmScaleB, TensorC gmC, const BlockShape& singleShape)
    {
        auto curM = static_cast<uint64_t>(Get<IDX_M_TILEIDX>(singleShape));
        auto curN = static_cast<uint64_t>(Get<IDX_N_TILEIDX>(singleShape));

        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);

        uint64_t offsetA = 0;
        uint64_t offsetB = 0;
        uint64_t offsetScaleA = 0;
        uint64_t offsetScaleB = 0;

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t curGmKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            uint64_t curPadKL1 = CeilAlign(curGmKL1, MXFP_DIVISOR_SIZE);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t offsetAL1 = l1BufferAOffset_[l1BufId];
            uint64_t offsetBL1 = l1BufferBOffset_[l1BufId];
            uint64_t offsetScaleAL1 = l1BufferScaleAOffset_[l1BufId];
            uint64_t offsetScaleBL1 = l1BufferScaleBOffset_[l1BufId];

            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeNzLayout<AType>(curM, curGmKL1);
            auto tensorAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(offsetAL1), layoutAL1);
            auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, offsetA), AscendC::Te::MakeShape(curM, curGmKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

            auto layoutBL1 = AscendC::Te::MakeZnLayout<BType>(curGmKL1, curN);
            auto tensorBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(offsetBL1), layoutBL1);
            auto gmBlockB = gmB(AscendC::Te::MakeCoord(offsetB, 0), AscendC::Te::MakeShape(curGmKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            auto gmBlockScaleA = gmScaleA(
                AscendC::Te::MakeCoord(0, offsetScaleA),
                AscendC::Te::MakeShape(curM, CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
            auto gmBlockScaleB = gmScaleB(
                AscendC::Te::MakeCoord(offsetScaleB, 0),
                AscendC::Te::MakeShape(CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
            auto copyScaleGM2L1 = AscendC::Te::MakeCopy(::Tile::CopyScaleGM2L1{});
            auto layoutScaleAL1 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
                curM, CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(offsetScaleAL1), layoutScaleAL1);
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto layoutScaleBL1 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
            auto tensorScaleBL1 =
                AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(offsetScaleBL1), layoutScaleBL1);
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);

            offsetA += curGmKL1;
            offsetScaleA += CeilDiv(curGmKL1, MXFP_GROUP_SIZE);
            offsetB += curGmKL1;
            offsetScaleB += CeilDiv(curGmKL1, MXFP_GROUP_SIZE);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            uint64_t kL0TileNum = CeilDiv(curGmKL1, baseK_);
            uint64_t tailKL0 = curGmKL1 - (kL0TileNum - 1) * baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                uint64_t curKL0 = (iter1 == (kL0TileNum - 1)) ? tailKL0 : baseK_;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t kL0Offset = iter1 * baseK_;
                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 = tensorAL1(
                    AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorBlockAL1);

                auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 = tensorBL1(
                    AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBlockBL1);

                auto layoutScaleAL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
                    curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
                auto tensorScaleAL0 =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(l0Offset), layoutScaleAL0);
                auto tensorBlockScaleAL1 = tensorScaleAL1(
                    AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(curM, CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
                auto copyL12L0MxScaleA = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleA3510{});
                AscendC::Te::Copy(
                    copyL12L0MxScaleA, tensorScaleAL0, tensorBlockScaleAL1, AscendC::Te::MakeCoord(0, kL0Offset));

                auto layoutScaleBL0 = AscendC::Te::MakeNnLayout<fp8_e8m0_t>(
                    CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN);
                auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeL0BmemPtr<>((__cb__ fp8_e8m0_t*)(l0Offset)), layoutScaleBL0);
                auto tensorBlockScaleBL1 = tensorScaleBL1(
                    AscendC::Te::MakeCoord(0, 0),
                    AscendC::Te::MakeShape(CeilDiv(curGmKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, curN));
                auto copyL12L0MxScaleB = AscendC::Te::MakeCopy(::Tile::CopyL12L0MxScaleB3510{});
                AscendC::Te::Copy(
                    copyL12L0MxScaleB, tensorScaleBL0, tensorBlockScaleBL1, AscendC::Te::MakeCoord(kL0Offset, 0));

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

                uint8_t mmadUnitFlag = ((iter0 + 1 == kL1Iter_) && (iter1 + 1 == kL0TileNum))
                    ? static_cast<uint8_t>(FINAL_ACCUMULATION)
                    : static_cast<uint8_t>(NON_FINAL_ACCUMULATION);
                bool mmadCmatrixInitVal = (iter0 == 0 && iter1 == 0);
                AscendC::Te::Mad(
                    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<::Tile::MmadMx>>{}.with(
                        static_cast<uint16_t>(curM),
                        static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE)),
                        static_cast<uint16_t>(curN), mmadUnitFlag, false, mmadCmatrixInitVal),
                    tensorL0C, tensorAL0, tensorBL0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }

        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(
            copyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});  // required with MMAD unitFlag
    }

private:
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t scaleAL1OneBuffer_{0};
    uint64_t scaleBL1OneBuffer_{0};
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t l1BufferScaleAOffset_[4] = {0UL};
    uint64_t l1BufferScaleBOffset_[4] = {0UL};
};
}
#endif
