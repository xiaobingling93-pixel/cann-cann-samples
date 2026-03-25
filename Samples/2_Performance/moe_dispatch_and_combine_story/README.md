# moe dispatch 和 combine 通信算子性能优化实践与效果分析

## 构建与运行

在项目根目录启动构建，执行：

```bash
cmake -S . -B build
cmake --build build --target moe_dispatch_and_combine_story
```

先生成一组测试数据：

```bash
python3 Samples/2_Performance/moe_dispatch_and_combine_story/scripts/gen_data.py --chip-num-per-server 2 --bs 8
```

再运行相应可执行文件：

```bash
./build/Samples/2_Performance/moe_dispatch_and_combine_story/moe_dispatch_and_combine_story 2 8
```

执行精度校验：

```bash
python3 Samples/2_Performance/moe_dispatch_and_combine_story/scripts/verify_result.py
```