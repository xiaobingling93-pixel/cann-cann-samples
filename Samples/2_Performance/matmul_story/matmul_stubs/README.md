# matmul_stubs

Kernel implementations and sample applications for MatMul performance optimization.

## Directory structure

```
matmul_stubs/
├── include/           # Header files (block scheduler, MMAD, kernel impl, policy, utils)
└── examples/          # Sample applications
    └── quant_matmul_mxfp4/   # MXFP4 quantized MatMul sample
```

## Dependencies

- `../common/` – shared host/kernel utilities and golden scripts
