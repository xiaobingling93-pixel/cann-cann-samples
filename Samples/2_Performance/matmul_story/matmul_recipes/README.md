# 矩阵乘算子变体总览

## 概述

本目录包含矩阵乘算子在昇腾AI处理器上的多种优化实现变体，每个变体针对不同数据类型或量化策略提供完整的算子实现与运行脚本。

## 目录结构

```text
matmul_recipes/
├── CMakeLists.txt
├── README.md
├── common/                         # 公共工具函数（host/kernel）
├── include/                        # 共享头文件（block, kernel, policy, tile, tiling, utils）
└── examples/
│   ├── quant_matmul_mxfp4/             # MXFP4 量化矩阵乘变体
│   ├── README.md
│   ├── quant_matmul_mxfp4_swat.cpp
│   ├── quant_matmul_mxfp4_a_full_load.cpp
│   ├── quant_matmul_mxfp4_algorithm_recommend.py
│   ├── gen_data.py
│   └── verify_result.py
    └── matmul_a16w16/                  # A16W16 非量化矩阵乘变体
    └── README.md
```

## 变体列表

| 变体 | 数据类型 | 说明 |
|------|----------|------|
| [quant_matmul_mxfp4](examples/quant_matmul_mxfp4/README.md) | MXFP4 | 4 位浮点量化矩阵乘，含 SWAT 与 A 全载两种模板 |
| [matmul_a16w16](examples/matmul_a16w16/README.md) | Float16 | A16W16 非量化矩阵乘 |

## 性能优化指南

各变体涉及的模板实现及优化策略详见 [MXFP4 量化矩阵乘算子性能优化指南](docs/quant_matmul_mxfp4_performance.md)。
