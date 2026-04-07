## ----------------------------------------------------------------------------
## This program is free software, you can redistribute it and/or modify it.
## Copyright (c) 2026 Huawei Technologies Co., Ltd.
## This file is a part of the CANN Open Software.
## Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
## Please refer to the License for details. You may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850
## ----------------------------------------------------------------------------

function(add_grouped_matmul_recipe target_name source variant)
    add_executable(${target_name}
        ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/examples/${source}
    )
    target_include_directories(${target_name} PRIVATE
        ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/include
        ${GROUPED_MATMUL_STORY_DIR}/common
    )
    target_compile_options(${target_name} PRIVATE ${GMM_STORY_ASCENDC_COMPILE_OPTS})
    target_link_libraries(${target_name} PRIVATE
        cann_samples::ascend_base
        cann_samples::tensor_api
    )
    install(TARGETS ${target_name}
        RUNTIME DESTINATION ${GROUPED_MATMUL_INSTALL_PREFIX}/grouped_matmul_recipes/examples/${variant}
    )
endfunction()

add_grouped_matmul_recipe(
    quant_grouped_matmul_mxfp4
    quant_grouped_matmul_mxfp4/quant_grouped_matmul_mxfp4_split_m.cpp
    quant_grouped_matmul_mxfp4
)

install(FILES
    ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/examples/quant_grouped_matmul_mxfp4/scripts/gen_data.py
    ${GROUPED_MATMUL_STORY_DIR}/grouped_matmul_recipes/examples/quant_grouped_matmul_mxfp4/scripts/verify_result.py
    DESTINATION ${GROUPED_MATMUL_INSTALL_PREFIX}/grouped_matmul_recipes/examples/quant_grouped_matmul_mxfp4
)

