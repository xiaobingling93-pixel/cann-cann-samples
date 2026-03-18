# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(TARGET cann_samples_ascend_base)
    return()
endif()

add_library(cann_samples_ascend_base INTERFACE)
add_library(cann_samples::ascend_base ALIAS cann_samples_ascend_base)

target_include_directories(cann_samples_ascend_base INTERFACE
    ${ASCEND_INCLUDE_DIRS}
)

target_link_directories(cann_samples_ascend_base INTERFACE
    ${ASCEND_DIR}/lib64
)

target_link_libraries(cann_samples_ascend_base INTERFACE
    m
    dl
    platform
    tiling_api
)
