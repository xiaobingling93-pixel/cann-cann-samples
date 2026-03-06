# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

add_executable(quant_matmul_template_aswt quant_matmul_template_aswt.cpp)

target_include_directories(quant_matmul_template_aswt PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(quant_matmul_template_aswt PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(quant_matmul_template_aswt PRIVATE
    "-xasc"
    "--npu-arch=dav-3101"
    "-w"
    "-O3"
)

target_link_libraries(quant_matmul_template_aswt
    PRIVATE
    m
    dl
    platform
    tiling_api
)

install(TARGETS quant_matmul_template_aswt
    RUNTIME DESTINATION 2_Performance/matmul_story/quant_matmul_mxfp4/quant_matmul_template_aswt
)

install(DIRECTORY
    ../../../../../Common/golden/golden_quant_matmul_mxfp4/
    DESTINATION 2_Performance/matmul_story/quant_matmul_mxfp4/quant_matmul_template_aswt
    FILES_MATCHING
    PATTERN "*.py"
)
