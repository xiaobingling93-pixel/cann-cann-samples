#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def write_artifacts(base_dir, a_fp8, b_fp8, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # MXFP8 keeps one value per byte, so there is no fp4 nibble packing.
    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_simple(m, k, n):
    M = m
    K = k
    N = n

    # Generate MXFP8 quantized inputs.
    a_ori = np.random.uniform(1, 8, (M, K)).astype(float8_e4m3fn)
    # Keep B in (N, K) order so the binary file matches the sample's
    # column-major/filter-major interpretation on the device side.
    b_ori = np.random.uniform(1, 8, (N, K)).astype(float8_e4m3fn)
    a_scale = np.random.uniform(1, 8, size=(M, math.ceil(K / 64), 2)).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(float8_e8m0)

    # Transpose and broadcast scale tensors.
    a_scale_reshape = a_scale.reshape(M, -1)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :K]

    b_ori_transpose = np.swapaxes(b_ori, -1, -2)
    b_scale_reshape = b_scale.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)

    # Dequantize inputs.
    a_dequant = a_ori.astype(np.float32) * a_scale_broadcast.astype(np.float32)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    a_cpu = torch.from_numpy(a_dequant)
    b_cpu = torch.from_numpy(b_dequant)
    out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_ori, b_ori, a_scale, b_scale, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The script may be called from either the source tree or the installed
    # sample directory. When those differ, emit artifacts to both locations.
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_ori, b_ori, a_scale, b_scale, out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py m k n")
        sys.exit(1)

    # Parse command-line arguments.
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    gen_golden_data_simple(m, k, n)
