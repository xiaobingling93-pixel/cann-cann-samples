# 测试清单说明

## 定位

本文件只说明 `tests/` 下测试清单的维护规则和字段契约。
执行入口、构建命令、artifact 输出统一见 [`.ci/README.md`](../.ci/README.md)。

## 文件

- 主清单：[`ci_functional_test.yaml`](./ci_functional_test.yaml)
- 执行入口：[`../.ci/run_ci_functional.sh`](../.ci/run_ci_functional.sh)
- Python runner：[`../.ci/run_ci_functional.py`](../.ci/run_ci_functional.py)

## YAML 契约

顶层字段：

- `version`
- `samples`

每个 `samples` 条目只允许包含：

- `id`
- `setup`
- `steps`
- `pass_criteria`

每个 `setup` / `steps` 条目包含：

- `name`
- `cwd`
- `cmd`

约束：

- `cmd` 必须是 argv 数组，不要写 shell 拼接命令
- `cwd` 使用仓库相对路径
- `id` 保持稳定，不要随意改名
- 不能自动判定 PASS/FAIL 的样例，不要先放进清单

最小示例：

```yaml
- id: hif8_quantize
  setup:
    - name: gen_input
      cwd: build_out/1_Features/hardware_features/hif8
      cmd: ["python3", "scripts/gen_data.py"]
  steps:
    - name: run_demo
      cwd: build_out/1_Features/hardware_features/hif8
      cmd: ["./quantize_hif8_demo"]
  pass_criteria:
    all_steps_exit_zero: true
```

## 补充清单时只做四件事

1. 选定稳定的运行目录和命令
2. 把前置动作写进 `setup`
3. 把执行和校验写进 `steps`
4. 用 `pass_criteria` 定义可机器判断的通过条件

## 当前边界

- 当前只支持最小功能测试契约
- 当前不导出 JUnit
- 当前不包含性能测试
