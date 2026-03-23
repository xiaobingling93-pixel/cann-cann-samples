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
 * \file block_scheduler_policy.h
 * \brief Scheduler policy definitions for SWAT quantized matmul kernels.
 */

#ifndef BLOCK_SCHEDULER_POLICY_H
#define BLOCK_SCHEDULER_POLICY_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Streaming SWAT path: both A and B move through the pipeline chunk by chunk.
constexpr uint64_t SWAT_NO_FULL_LOAD_MODE = 0UL;
// A-full-load SWAT path: the A tile stays resident while B keeps streaming.
constexpr uint64_t SWAT_A_FULL_LOAD_MODE = 1UL;

// This policy tag is intentionally tiny: it only carries the path mode so the
// kernel, scheduler, and block pipeline can agree on the selected SWAT flavor
// at compile time.
template <uint64_t FULL_LOAD_MODE_ = SWAT_NO_FULL_LOAD_MODE>
struct QuantMatmulMxSwatScheduler {
    // `fullLoadMode` is consumed by trait selection only; no runtime state is
    // stored in this tag type.
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
#endif
