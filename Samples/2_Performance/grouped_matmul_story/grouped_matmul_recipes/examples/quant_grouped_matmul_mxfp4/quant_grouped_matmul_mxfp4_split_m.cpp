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
 * \file quant_grouped_matmul_mxfp4_split_m.cpp
 * \brief Minimal launcher for the grouped MXFP4 split-M recipe.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "host_utils/common_utils.h"
#include "host_utils/io_utils.h"
#include "kernel/quant_grouped_matmul_mxfp4_kernel_split_m.h"
#include "policy/dispatch_policy.h"
#include "tiling/quant_grouped_matmul_mxfp4_tiling_data.h"
#include "tiling/quant_grouped_matmul_mxfp4_tiling_split_m.h"

// mxfp4: only for cube core
__global__ __aicore__ __cube__ void QuantGroupedMatmulMxfp4SplitMKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR y,
    const QuantGroupedMatmulMxfp4TilingData tilingData)
{
    using AType = fp4x2_e2m1_t;
    using BType = fp4x2_e2m1_t;
    using CType = bfloat16_t;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    using ProblemShape = MatmulShape;
    using DispatchPolicy = QuantMatmulMxMultiBlockMmad;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>;
    using BlockScheduler = Block::BlockSchedulerGmmAswtWithTailSplit<ProblemShape, false, true>;
    using GroupedKernel = Kernel::QuantGroupedMatmulMxfp4KernelSplitM<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename GroupedKernel::Params;

    Params params = {
        {static_cast<int64_t>(tilingData.maxM), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k), 1},
        {x, weight, groupList, xScale, weightScale, y},
        &tilingData,
    };
    GroupedKernel kernel;
    kernel(params);
}

namespace {
struct AclRtSession {
    explicit AclRtSession(int32_t deviceId) : deviceId_(deviceId) {}

    void Init()
    {
        CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "Failed to initialize ACL runtime.");
        aclInit_ = true;
        CHECK_COND(aclrtSetDevice(deviceId_) == ACL_SUCCESS, "Failed to set ACL device.");
        deviceSet_ = true;
        CHECK_COND(aclrtCreateStream(&stream_) == ACL_SUCCESS, "Failed to create ACL stream.");
    }

    aclrtStream GetStream() const
    {
        return stream_;
    }

    ~AclRtSession()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInit_) {
            (void)aclFinalize();
            aclInit_ = false;
        }
    }

    AclRtSession(const AclRtSession&) = delete;
    AclRtSession& operator=(const AclRtSession&) = delete;

private:
    int32_t deviceId_;
    aclrtStream stream_{nullptr};
    bool deviceSet_{false};
    bool aclInit_{false};
};

struct AclDeviceBuffers {
    ~AclDeviceBuffers()
    {
        for (GM_ADDR gmAddr : gmAddrList) {
            (void)aclrtFree(gmAddr);
        }
    }
    void Push(GM_ADDR gmAddr)
    {
        if (gmAddr != nullptr) {
            gmAddrList.push_back(gmAddr);
        }
    }

private:
    std::vector<GM_ADDR> gmAddrList{};
};

std::vector<uint32_t> ParseGroupList(const std::vector<int64_t>& groupListHost)
{
    CHECK_COND(!groupListHost.empty(), "group_list must not be empty.");
    std::vector<uint32_t> groupMList;
    groupMList.reserve(groupListHost.size());
    for (int64_t groupM : groupListHost) {
        CHECK_COND(groupM >= 0, "Each group M value must be greater than or equal to zero.");
        CHECK_COND(groupM <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()), "Each group M value must not exceed INT32_MAX.");
        groupMList.push_back(static_cast<uint32_t>(groupM));
    }
    return groupMList;
}

uint64_t ParsePositiveUint64(const char* arg, const char* name)
{
    std::string value(arg);
    CHECK_COND(!value.empty(), std::string(name) + " must not be empty.");
    unsigned long long parsed = std::stoull(value);
    CHECK_COND(parsed > 0ULL, std::string(name) + " must be greater than zero.");
    return static_cast<uint64_t>(parsed);
}

void PrintUsage(const char* program)
{
    std::cerr << "Usage: " << program << " group_num m k n" << std::endl;
    std::cerr << "Example: " << program << " 2 256 4096 1024" << std::endl;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 5) {
        PrintUsage(argv[0]);
        return 1;
    }

    constexpr int32_t deviceId = 0;

    try {
        uint64_t groupNum = ParsePositiveUint64(argv[1], "group_num");
        uint64_t m = ParsePositiveUint64(argv[2], "m");
        uint64_t k = ParsePositiveUint64(argv[3], "k");
        uint64_t n = ParsePositiveUint64(argv[4], "n");
        constexpr uint64_t int32MaxU64 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
        CHECK_COND(m <= int32MaxU64, "m must not exceed INT32_MAX.");
        CHECK_COND(k <= int32MaxU64, "k must not exceed INT32_MAX.");
        CHECK_COND(n <= int32MaxU64, "n must not exceed INT32_MAX.");
        CHECK_COND(k % 2UL == 0UL, "k must be even for fp4 packed storage.");
        CHECK_COND(
            groupNum <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
            "group_num exceeds addressable size on this platform.");
        uint64_t groupListBytesU64 = groupNum * sizeof(int64_t);
        CHECK_COND(
            groupListBytesU64 <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
            "group list byte size exceeds addressable size on this platform.");
        const size_t groupListBytes = static_cast<size_t>(groupListBytesU64);
        std::string inputDir = "./input";
        std::string outputDir = "./output";
        std::vector<int64_t> groupListHost(static_cast<size_t>(groupNum), 0);

        size_t groupListFileSize = groupListBytes;
        CHECK_COND(
            ReadFile(inputDir + "/input_groupList.bin", groupListFileSize, groupListHost.data(), groupListBytes),
            "Failed to read input_groupList.bin.");
        CHECK_COND(
            groupListFileSize == groupListBytes, "input_groupList.bin size does not match the expected tensor size.");

        std::vector<uint32_t> groupMList = ParseGroupList(groupListHost);
        uint64_t sumGroupM = std::accumulate(groupMList.begin(), groupMList.end(), 0ULL);
        CHECK_COND(sumGroupM <= m, "m must be greater than or equal to sum(group_m_list).");

        QuantGroupedMatmulMxfp4TilingData tilingData;
        QuantGroupedMatmulMxfp4TilingSplitM tiler;
        tiler.GetTilingData(static_cast<uint32_t>(groupMList.size()), static_cast<uint32_t>(m),
                            static_cast<uint32_t>(n), static_cast<uint32_t>(k), tilingData);

        uint64_t scaleK = CeilDiv<uint64_t>(k, 64UL) * 2UL;
        // Tensor and GM sizes follow the declared M budget (m), not sum(group_m_list), so buffers stay valid when
        // some groups contribute zero rows.
        size_t xByteSize = (m * k / 2UL) * sizeof(uint8_t);
        size_t weightByteSize = (static_cast<uint64_t>(groupMList.size()) * n * k / 2UL) * sizeof(uint8_t);
        size_t xScaleByteSize = m * scaleK * sizeof(uint8_t);
        size_t weightScaleByteSize = static_cast<uint64_t>(groupMList.size()) * n * scaleK * sizeof(uint8_t);
        size_t yByteSize = m * n * sizeof(uint16_t);

        std::vector<uint8_t> hostX(xByteSize, 0);
        std::vector<uint8_t> hostWeight(weightByteSize, 0);
        std::vector<uint8_t> hostXScale(xScaleByteSize, 0);
        std::vector<uint8_t> hostWeightScale(weightScaleByteSize, 0);
        std::vector<uint8_t> hostY(yByteSize, 0);
        size_t xFileSize = xByteSize;
        size_t weightFileSize = weightByteSize;
        size_t xScaleFileSize = xScaleByteSize;
        size_t weightScaleFileSize = weightScaleByteSize;

        CHECK_COND(ReadFile(inputDir + "/input_a.bin", xFileSize, hostX.data(), xByteSize), "Failed to read input_a.bin.");
        CHECK_COND(xFileSize == xByteSize, "input_a.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_b.bin", weightFileSize, hostWeight.data(), weightByteSize),
            "Failed to read input_b.bin.");
        CHECK_COND(weightFileSize == weightByteSize, "input_b.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_scaleA.bin", xScaleFileSize, hostXScale.data(), xScaleByteSize),
            "Failed to read input_scaleA.bin.");
        CHECK_COND(xScaleFileSize == xScaleByteSize, "input_scaleA.bin size does not match the expected tensor size.");
        CHECK_COND(
            ReadFile(inputDir + "/input_scaleB.bin", weightScaleFileSize, hostWeightScale.data(), weightScaleByteSize),
            "Failed to read input_scaleB.bin.");
        CHECK_COND(
            weightScaleFileSize == weightScaleByteSize, "input_scaleB.bin size does not match the expected tensor size.");

        AclRtSession aclSession(deviceId);
        aclSession.Init();

        {
            AclDeviceBuffers dBuffers;
            GM_ADDR dX = nullptr;
            GM_ADDR dWeight = nullptr;
            GM_ADDR dXScale = nullptr;
            GM_ADDR dWeightScale = nullptr;
            GM_ADDR dGroupList = nullptr;
            GM_ADDR dY = nullptr;
            CHECK_COND(aclrtMalloc(reinterpret_cast<void**>(&dX), xByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                       "Failed to allocate device X.");
            dBuffers.Push(dX);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeight), weightByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device weight.");
            dBuffers.Push(dWeight);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dXScale), xScaleByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                "Failed to allocate device xScale.");
            dBuffers.Push(dXScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dWeightScale), weightScaleByteSize, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device weightScale.");
            dBuffers.Push(dWeightScale);
            CHECK_COND(
                aclrtMalloc(reinterpret_cast<void**>(&dGroupList), groupListBytes, ACL_MEM_MALLOC_HUGE_ONLY) ==
                    ACL_SUCCESS,
                "Failed to allocate device groupList.");
            dBuffers.Push(dGroupList);
            CHECK_COND(aclrtMalloc(reinterpret_cast<void**>(&dY), yByteSize, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS,
                       "Failed to allocate device Y.");
            dBuffers.Push(dY);

            aclrtStream stream = aclSession.GetStream();
            CHECK_COND(
                aclrtMemcpyAsync(dX, xByteSize, hostX.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
                "Failed to copy X to device.");
            CHECK_COND(
                aclrtMemcpyAsync(dWeight, weightByteSize, hostWeight.data(), weightByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy weight to device.");
            CHECK_COND(
                aclrtMemcpyAsync(dXScale, xScaleByteSize, hostXScale.data(), xScaleByteSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy xScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dWeightScale, weightScaleByteSize, hostWeightScale.data(), weightScaleByteSize, ACL_MEMCPY_HOST_TO_DEVICE,
                    stream) == ACL_SUCCESS,
                "Failed to copy weightScale to device.");
            CHECK_COND(
                aclrtMemcpyAsync(
                    dGroupList, groupListBytes, groupListHost.data(), groupListBytes, ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
                    ACL_SUCCESS,
                "Failed to copy groupList to device.");

            QuantGroupedMatmulMxfp4SplitMKernel<<<tilingData.usedCoreNum, nullptr, stream>>>(
                dX, dWeight, dXScale, dWeightScale, dGroupList, dY, tilingData);
            CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "Failed to synchronize stream.");
            CHECK_COND(
                aclrtMemcpy(hostY.data(), yByteSize, dY, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
                "Failed to copy Y back to host.");
            CHECK_COND(WriteFile(outputDir + "/npu_out.bin", hostY.data(), yByteSize), "Failed to write npu_out.bin.");
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
