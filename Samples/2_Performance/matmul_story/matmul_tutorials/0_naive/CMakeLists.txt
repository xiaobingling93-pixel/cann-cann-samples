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

add_executable(matmul_tutorial_mxfp4_base ${CMAKE_CURRENT_SOURCE_DIR}/matmul_tutorial_mxfp4_base.cpp)

target_include_directories(matmul_tutorial_mxfp4_base PRIVATE ${ASCEND_INCLUDE_DIRS})
target_link_directories(matmul_tutorial_mxfp4_base PRIVATE ${ASCEND_DIR}/lib64)

target_compile_options(matmul_tutorial_mxfp4_base PRIVATE
    "-xasc"
    "--npu-arch=dav-3510"
    "-w"
    "-O3"
)

target_link_libraries(matmul_tutorial_mxfp4_base
    PRIVATE
    m
    dl
    platform
    tiling_api
)

install(TARGETS matmul_tutorial_mxfp4_base
    RUNTIME DESTINATION 2_Performance/matmul_story/matmul_tutorials/0_naive
)

