# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(TARGET cann_samples_tensor_api)
    return()
endif()

add_library(cann_samples_tensor_api INTERFACE)
add_library(cann_samples::tensor_api ALIAS cann_samples_tensor_api)

target_include_directories(cann_samples_tensor_api INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/tensor_api
)
