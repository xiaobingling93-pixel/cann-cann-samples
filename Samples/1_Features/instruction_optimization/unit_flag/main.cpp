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
 * \file main.cpp
 * \brief Main implementation file for Ascend matrix multiplication kernel
 *        This version includes unit flag optimization for cube unit pipelining
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <iomanip>

#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "tiling/platform/platform_ascendc.h"
#include "include/tensor.h"

namespace AscendC::Te {
// Data loading trait definitions for matrix B
constexpr LoadDataTrait LOAD_DATA_B_TRAIT{true};
struct LoadData2BTrait {
    using TraitType = LoadDataTrait;
    static constexpr const TraitType value = LOAD_DATA_B_TRAIT;
};

// Layout configuration for matrices A and B (NZ format by default)
static constexpr bool transA = false;  // Don't transpose matrix A
static constexpr bool transB = false;  // Don't transpose matrix B
template <typename T>
using MakeLayoutAL1 =
    AscendC::Std::conditional_t<transA, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>;
template <typename T>
using MakeLayoutBL1 =
    AscendC::Std::conditional_t<transB, AscendC::Te::ZnLayoutFormat<T>, AscendC::Te::NzLayoutFormat<T>>;
} // namespace AscendC::Te

namespace tool {
// Basic configuration constants
constexpr static uint16_t CUBE_BLOCK_SIZE = 16;  // Cube unit block size
constexpr static uint16_t ZERO_FLAG = 0;         // Zero flag for synchronization

// Unit flag values for cube unit pipelining
constexpr uint32_t UNITFLAG_DISABLE = 0;         // Disable unit flag
constexpr uint32_t NO_FINAL_ACCUMULATION = 2;          // Enable unit flag (inner loops)
constexpr uint32_t FINAL_ACCUMULATION = 3;   // Enable unit flag for outer last iteration

// Utility function declarations
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);   // Ceiling division
__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b);   // Ceiling Align
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max);      // Fill vector with random data
template <typename T>
void ComputeGolden(    // Compute reference result on CPU
    int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight, std::vector<T>& goldenOutput);
template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput);  // Compare results
} // namespace tool

namespace matmul {
/**
 * @brief Matrix multiplication kernel with unit flag optimization
 * 
 * This kernel implements C = A * B with optimizations:
 * - Tiled computation for memory hierarchy
 * - Unit flag support for cube unit pipelining
 * - Multi-core parallelization
 * 
 * @tparam T Data type (float in this implementation)
 * @param aGm Global memory pointer to matrix A (size m*k)
 * @param bGm Global memory pointer to matrix B (size k*n)
 * @param cGm Global memory pointer to output matrix C (size m*n)
 * @param m Rows of A and C
 * @param k Columns of A, rows of B
 * @param n Columns of B and C
 */
template <typename T>
__global__ __aicore__ void MatmulKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, uint32_t m, uint32_t k, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);  // Specify AI Core task type

    // Initialize tiling parameters based on memory sizes
    uint64_t baseM = 256;
    uint64_t baseN = 256;
    uint64_t baseK = 128 / sizeof(T);
    uint64_t kL1 = 512 / sizeof(T);
    uint64_t mTileNum = tool::CeilDiv(m, baseM);
    uint64_t nTileNum = tool::CeilDiv(n, baseN);
    uint64_t tileNum = mTileNum * nTileNum;
    uint64_t kL1TileNum = tool::CeilDiv(k, kL1);
    uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;
    uint64_t tailBaseM = tool::CeilAlign(m - (mTileNum - 1) * baseM, 16);
    uint64_t tailBaseN = tool::CeilAlign(n - (nTileNum - 1) * baseN, 16);

    uint64_t curBlockIdx = AscendC::GetBlockIdx();
    uint64_t blockNum = AscendC::GetBlockNum();

    // Construct GM tensors with ND layout
    auto layoutA = AscendC::Te::MakeNDLayout<T>(m, k);
    auto layoutB = AscendC::Te::MakeNDLayout<T>(k, n);
    auto layoutC = AscendC::Te::MakeNDLayout<T>(m, n);

    auto tensorAgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(aGm)), layoutA);
    auto tensorBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(bGm)), layoutB);
    auto tensorCgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(cGm)), layoutC);

    // Initialize synchronization flags for data movement
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);

    // Multi-core tile processing loop - each core processes different tiles
    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        uint64_t mTileIdx = tileIdx / nTileNum;
        uint64_t nTileIdx = tileIdx % nTileNum;
        int64_t curM = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        int64_t curN = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;
        int64_t realM = mTileIdx == (mTileNum - 1) ? (m - (mTileNum - 1) * baseM) : baseM;
        int64_t realN = nTileIdx == (nTileNum - 1) ? (n - (nTileNum - 1) * baseN) : baseN;

        // Extract current tile from global memory
        uint64_t l0cOffset = 0;

        auto tensorAGmBlock = tensorAgm(AscendC::Te::MakeCoord(mTileIdx * baseM, 0L), AscendC::Te::MakeShape(curM, k));
        auto tensorBGmBlock = tensorBgm(AscendC::Te::MakeCoord(0L, nTileIdx * baseN), AscendC::Te::MakeShape(k, curN));
        auto tensorCGmBlock =
            tensorCgm(AscendC::Te::MakeCoord(mTileIdx * baseM, nTileIdx * baseN), AscendC::Te::MakeShape(realM, realN));

        // Setup L0C tensor for accumulation
        auto layoutL0C = AscendC::Te::MakeL0CLayout(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(l0cOffset), layoutL0C);
        
        // Loop over K dimension tiles for L1 buffer
        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            // Wait for previous GM->L1 transfer to complete
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);

            // Determine current K tile size for L1
            auto curGmBKL1 = (iter0 + 1 == kL1TileNum) ? tool::CeilAlign((k - iter0 * kL1), 16) : kL1;
            auto curGmAKL1 = curGmBKL1;

            // Copy GM to L1 buffers (single buffer version)
            uint64_t l1BufferAOffset = 0;
            uint64_t l1BufferBOffset = baseM * kL1 * sizeof(T);

            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto layoutAL1 = AscendC::Te::MakeLayoutAL1<T>{}(curM, curGmAKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferAOffset), layoutAL1);
            // Copy A tile from GM to L1
            auto tensorAGmTile = tensorAGmBlock(AscendC::Te::MakeCoord(0, iter0 * kL1), AscendC::Te::MakeShape(curM, curGmAKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, tensorAGmTile);

            auto layoutBL1 = AscendC::Te::MakeLayoutBL1<T>{}(curGmBKL1, curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<T>(l1BufferBOffset), layoutBL1);
            // Copy B tile from GM to L1
            auto tensorBGmTile = tensorBGmBlock(AscendC::Te::MakeCoord(iter0 * kL1, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, tensorBGmTile);

            // Signal L1 data ready and wait for it
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(tool::ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(tool::ZERO_FLAG);

            // Further tile K dimension for L0 buffer
            uint64_t kL0IterNum = tool::CeilDiv(curGmBKL1, baseK);
            uint64_t tailKL0 = curGmBKL1 - (kL0IterNum - 1) * baseK;
            
            for (uint16_t iter1 = 0; iter1 < kL0IterNum; ++iter1) {
                // Wait for previous L1->L0 transfer to complete
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);

                uint64_t curKL0 = (iter1 + 1 == kL0IterNum) ? tailKL0 : baseK;

                // Copy L1 to L0 buffers
                auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
                auto layoutAL0 = AscendC::Te::MakeNzLayout<T>(curM, curKL0);
                auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<T>(0), layoutAL0);
                // Copy A sub-tile from L1 to L0
                auto tensorAL1Tile =
                    tensorAL1(AscendC::Te::MakeCoord(0, iter1 * baseK), AscendC::Te::MakeShape(curM, curKL0));
                AscendC::Te::Copy(copyL12L0, tensorAL0, tensorAL1Tile);

                auto layoutBL0 = AscendC::Te::MakeZnLayout<T>(curKL0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<T>(0), layoutBL0);
                // Copy B sub-tile from L1 to L0
                auto tensorBL1Tile =
                    tensorBL1(AscendC::Te::MakeCoord(iter1 * baseK, 0), AscendC::Te::MakeShape(curKL0, curN));
                AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBL1Tile);

                // Signal L0 data ready and wait for it
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(tool::ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(tool::ZERO_FLAG);

                // Execute M-MAD operation with unit flag optimization
                AscendC::MmadParams para;
                para.cmatrixInitVal = (iter1 == 0 && iter0 == 0);
                para.m = curM;
                para.n = curN;
                para.k = curKL0;

                // Configure unit flag for cube unit pipelining
                if (iter1 == (kL0IterNum - 1) && iter0 == (kL1TileNum - 1)) {
                    // Last K tile - special flag for outer loop termination
                    para.unitFlag = tool::FINAL_ACCUMULATION;
                } else {
                    // Not the last K tile - enable pipelining
                    para.unitFlag = tool::NO_FINAL_ACCUMULATION;
                }
                
                // Perform MAD operation: C += A * B
                auto MadOp = AscendC::Te::MakeMad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
                AscendC::Te::Mad(MadOp, tensorL0C, tensorAL0, tensorBL0, para);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
        }
        
        // Copy result from L0C to global memory with unit flag
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM, tensorCGmBlock, tensorL0C, AscendC::Te::FixpipeParams{tool::FINAL_ACCUMULATION});
    }
    
    // Final synchronization waits
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(tool::ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(tool::ZERO_FLAG);
}

} // namespace matmul

// Utility macro for condition checking with error message
#define CHECK_COND(cond, message, return_expr)              \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "ERROR: " << message << std::endl; \
            return_expr;                                    \
        }                                                   \
    } while (0)

// Print command-line usage help
void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// Brief parses and validates command-line arguments
void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 4) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("ERROR: m k n must be Integer");
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::invalid_argument("ERROR: m k n must be positive");
    }
}

/**
 * @brief Main function - host-side setup and execution
 * 
 * This function handles:
 * 1. Command line argument parsing
 * 2. ACL resource initialization
 * 3. Memory allocation and data transfer
 * 4. Kernel launch
 * 5. Result verification
 * 6. Resource cleanup
 */
int main(int argc, char* argv[])
{
    using namespace tool;
    int m, k, n;
    try {
        parseArguments(argc, argv, m, k, n);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Initialize ACL resources
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtEvent kernelStartEvent = nullptr;
    aclrtEvent kernelEndEvent = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_COND(ret == ACL_SUCCESS, "aclInit failed.", return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSetDevice failed.", return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtCreateStream failed.", return 1);

    // Allocate host memory and fill with random data
    std::vector<float> hostInput(m * k, 0);
    std::vector<float> hostWeight(k * n, 0);
    std::vector<float> hostOutput(m * n, 0);
    std::vector<float> goldenOutput(m * n, 0);
    FillRandomData<float>(hostInput, -2.0f, 2.0f);
    FillRandomData<float>(hostWeight, -2.0f, 2.0f);

    // Allocate device memory
    GM_ADDR deviceInput = nullptr;
    GM_ADDR deviceWeight = nullptr;
    GM_ADDR deviceOutput = nullptr;
    auto sizeInput = hostInput.size() * sizeof(float);
    auto sizeWeight = hostWeight.size() * sizeof(float);
    auto sizeOutput = hostOutput.size() * sizeof(float);
    
    // Use RAII for automatic cleanup
    std::unique_ptr<void, aclError (*)(void*)> deviceInputPtr(nullptr, aclrtFree);
    ret = aclrtMalloc((void**)&deviceInput, sizeInput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceInput failed.", return 1);
    deviceInputPtr.reset(deviceInput);

    std::unique_ptr<void, aclError (*)(void*)> deviceWeightPtr(nullptr, aclrtFree);
    ret = aclrtMalloc((void**)&deviceWeight, sizeWeight, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceWeight failed.", return 1);
    deviceWeightPtr.reset(deviceWeight);

    std::unique_ptr<void, aclError (*)(void*)> deviceOutputPtr(nullptr, aclrtFree);
    ret = aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMalloc deviceOutput failed.", return 1);
    deviceOutputPtr.reset(deviceOutput);

    // Copy data from host to device
    ret = aclrtMemcpy(deviceInput, sizeInput, hostInput.data(), sizeInput, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceInput failed.", return 1);
    ret = aclrtMemcpy(deviceWeight, sizeWeight, hostWeight.data(), sizeWeight, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceWeight failed.", return 1);

    // Get platform instance to access AI core information and setup kernel timing events
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "get ascendcPlatform failed.", return 1);
    uint32_t numBlocks = ascendcPlatform->GetCoreNumAic();  // Number of AI cores
    ret = aclrtCreateEvent(&kernelStartEvent);
    CHECK_COND(ret == ACL_SUCCESS, "Failed to create the start event for kernel timing.", return 1);
    ret = aclrtCreateEvent(&kernelEndEvent);
    CHECK_COND(ret == ACL_SUCCESS, "Failed to create the end event for kernel timing.", return 1);
    ret = aclrtRecordEvent(kernelStartEvent, stream);
    CHECK_COND(ret == ACL_SUCCESS, "Failed to record the start event for kernel timing.", return 1);

    // Launch kernel on all available AI cores
    matmul::MatmulKernel<float><<<numBlocks, nullptr, stream>>>(deviceInput, deviceWeight, deviceOutput, m, k, n);

    ret = aclrtRecordEvent(kernelEndEvent, stream);
    CHECK_COND(ret == ACL_SUCCESS, "Failed to record the end event for kernel timing.", return 1);

    // Wait for kernel completion
    ret = aclrtSynchronizeStream(stream);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed.", return 1);
    float kernelElapsedMs = 0.0F;
    ret = aclrtEventElapsedTime(&kernelElapsedMs, kernelStartEvent, kernelEndEvent);
    CHECK_COND(ret == ACL_SUCCESS, "Failed to query the kernel elapsed time.", return 1);
    double kernelElapsedUs = static_cast<double>(kernelElapsedMs) * 1000.0;

    // Copy result back from device to host
    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_COND(ret == ACL_SUCCESS, "aclrtMemcpy deviceOutput failed.", return 1);

    // Verify results against CPU golden reference
    ComputeGolden<float>(m, k, n, hostInput, hostWeight, goldenOutput);
    std::vector<uint64_t> errorIndices = Compare<float>(hostOutput, goldenOutput);
    if (errorIndices.size() == 0) {
        std::cout << "matmul run successfully!" << std::endl;
    } else {
        for (uint64_t i : errorIndices) {
            std::cout << "error index: " << i << ", output: " << hostOutput[i]
                      << ", golden: " << goldenOutput[i] << std::endl;
        }
        std::cout << "matmul run failed!" << std::endl;
    }

    std::cout << std::fixed << std::setprecision(3) << "Kernel elapsed time: "
        << kernelElapsedUs << " us" << std::endl;
    std::cout << "Timing note: event-based timing may be skewed when the NPU is hared. "
                 "If the device is not exclusively owned, or the reported time is unstable, "
                 "use the `msprof` command for precise profiling."
              << std::endl;

    // Cleanup resources (unique_ptr handles device memory)
    std::unique_ptr<void, aclError (*)(void*)> DeviceOutputAddr(deviceOutput, aclrtFree);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

namespace tool {
/**
 * @brief Ceiling division for integer arithmetic
 * 
 * @param a Numerator
 * @param b Denominator
 * @return Ceiling of a/b
 */
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * @brief Ceiling alignment: returns smallest multiple of b that is >= a
 */
__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}

/**
 * @brief Fill a vector with random data
 * 
 * @tparam T Data type (integral or floating point)
 * @param data Vector to fill
 * @param min Minimum value
 * @param max Maximum value
 */
template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (auto& elem : data)
            elem = dist(gen);
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        for (auto& elem : data)
            elem = dist(gen);
    }
}

/**
 * @brief Compute matrix multiplication on CPU as golden reference
 * 
 * @tparam T Data type
 * @param m Rows of A
 * @param k Columns of A / Rows of B
 * @param n Columns of B
 * @param hostInput Matrix A
 * @param hostWeight Matrix B
 * @param goldenOutput Output matrix C (reference)
 */
template <typename T>
void ComputeGolden(
    int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight, std::vector<T>& goldenOutput)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            size_t offsetGolden = row * n + col;
            T sum = 0;
            for (uint32_t iter = 0; iter < k; ++iter) {
                size_t offsetInput = row * k + iter;
                size_t offsetWeight = iter * n + col;
                sum += hostInput[offsetInput] * hostWeight[offsetWeight];
            }
            goldenOutput[offsetGolden] = sum;
        }
    }
}

/**
 * @brief Compare kernel output with golden reference
 * 
 * @tparam T Data type
 * @param hostOutput Kernel output
 * @param goldenOutput CPU reference
 * @return std::vector<uint64_t> Indices where values differ beyond tolerance
 */
template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput)
{
    std::vector<uint64_t> errorIndices;
    const float rtol = 1.0f / 256;  // Relative tolerance for float comparison
    for (uint64_t i = 0; i < hostOutput.size(); ++i) {
        T actualValue = hostOutput[i];
        T expectValue = goldenOutput[i];
        T diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

} // namespace tool