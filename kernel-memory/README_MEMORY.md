# Memory 统一输入规范

每个使用者提交同一种 **`kernel-memory-input` v1 JSON**，程序校验后生成固定的 memory 目录。
原始实验来自什么项目、用什么日志格式，由人工、AI 或项目适配器负责转换。
README 给人和 AI 说明规则，JSON Schema 与导入程序负责实际执行这些规则。

给同事使用时，提供 `kernel-memory/` 文件夹即可，本说明已经在其中；同事还需要提供自己的实验记录。
保留 `pyproject.toml`、`README.md`、`src/`、`scripts/` 和 `examples/`，按下文安装依赖后生成结果。
`tests/`、`fixtures/` 和 `docs/` 用于测试与维护，仓库中继续保留。
打包时不带 `.venv/`、`__pycache__/`、`.pytest_cache/` 或临时演示输出；
已有 `.demo/memory/` 可作为阅读示例，生成新项目的 memory 不依赖它。

- [完整输入示例](examples/memory_input.json)：可直接校验和运行；其中数值明确标注为合成示例。
- [Project2 真实输入示例](examples/project2_memory_input.json)：你当前打开的 dense-aligned-dma-attention 方法，4 个历史候选的相同指标。
- [机器可读 Schema](src/kernel_memory/contracts/memory_input.v1.schema.json)：字段、类型、单位和允许的范围。
- [Project2 适配说明](README_PROJECT2.md)：已有 Project2 JSON/CSV 的批量转换及旧字段解释。

## 输入包含什么

```text
input.json
└── kernels[]
    ├── kernel_id、description
    └── algorithms[]
        ├── algorithm_id、method_summary、summary_author
        └── shapes[]
            ├── shape_id、problem
            ├── baseline_run_id（可选）
            └── attempts[]
                ├── attempt_id、candidate、parameters（可选）
                ├── pr_url、commit、code（可选）
                └── runs[]
                    ├── run_id
                    ├── recorded_at、environment、measurement（可选）
                    └── result
```

一个最小合法输入如下。它只演示格式，**没有测量数据**：

```json
{
  "format": "kernel-memory-input",
  "schema_version": "1.0.0",
  "kernels": [{
    "kernel_id": "vector_add",
    "description": "两个向量逐元素相加。",
    "algorithms": [{
      "algorithm_id": "elementwise-add",
      "method_summary": "逐元素计算 y[i] = a[i] + b[i]。",
      "summary_author": "human",
      "shapes": [{
        "shape_id": "n4096-f32",
        "problem": {
          "input_signature": {
            "a": {"shape": [4096], "dtype": "float32"},
            "b": {"shape": [4096], "dtype": "float32"}
          },
          "output_signature": {
            "y": {"shape": [4096], "dtype": "float32"}
          }
        },
        "attempts": [{
          "attempt_id": "trial-1",
          "candidate": "reference",
          "runs": [{
            "run_id": "run-1",
            "result": {
              "execution_status": "not_run",
              "correctness": "not_run",
              "latency_us": null
            }
          }]
        }]
      }]
    }]
  }]
}
```

### 身份、方法与 shape

| 字段 | 规则 |
|---|---|
| 各级 `*_id` | 小写字母、数字、`-`、`_`，最多 80 字符；同一级作用域内唯一。Windows 保留文件名也不允许。 |
| `algorithm_id` | 方法名，可以由 AI 根据实现归类，例如 `paged-attention`。不同 tile 通常是同一方法下的不同尝试。 |
| `method_summary` | 方法的实际做法；AI 可以写，不能只填句号，也不能凭空写优化收益。 |
| `summary_author` | `human`、`agent` 或 `unknown`。 |
| `problem` | 固定的输入/输出张量 shape、dtype，以及可选的 `parameters`，如 scale、causal、布局。 |
| `attempt.parameters` | 调优参数，如 `block_kv: 2048`、`dma2: true`；与固定问题参数分开。 |
| `baseline_run_id` | 本 shape 内已有的一个 `run_id`；必须成功、正确性通过且有耗时。未知就省略或填 null。 |
| `pr_url`、`commit` | 实际 PR 的 HTTPS 地址、实际 Git commit；未知就 null/省略，不要拿源码摘要冒充 commit。 |
| `code` | 可选的源码位置：`repository_url`、`path`、`entrypoint`；只保存字符串，不下载或执行。 |
| `environment` | 可选的 `device` 和 `software` 版本表。 |
| `measurement` | 可选的 `scope`、`timing_method`、`warmup`。 |

例如 `problem.parameters` 的 `causal` 影响计算问题；`attempt.parameters` 的 `block_kv` 是实现选择。
相同方法下相同 `problem` 的尝试应放在一个 shape 中，程序会拒绝重复 problem。
输入只需提供可读 ID；`source_config_ref`、关联 ID 和内容 hash 由程序生成。

### 结果与单位

`result` 必填 `execution_status`、`correctness`、`latency_us`。
其余指标可以省略，生成时补齐为 `null`；`hlo/llo` 缺省为 `not_collected`。

| 字段 | 含义与单位 |
|---|---|
| `execution_status` | `succeeded`、`compile_error`、`runtime_error`、`timeout`、`cancelled`、`not_run`。 |
| `correctness` | `pass`、`fail`、`not_run`；没有正确性记录就填 `not_run`。 |
| `latency_us`、`p90_latency_us` | 中位耗时、p90 耗时，单位微秒；未知填 null。 |
| `samples` | 样本数量；未知填 null。 |
| `mfu` | 有效计算量相对于峰值算力的利用率，`0.7484 = 74.84%`。 |
| `mfu_executed_estimate` | 根据估算执行量得到的利用率；与有效 MFU 分开。 |
| `mxu` | 运行时矩阵单元利用率；没有实测就 null。 |
| `attention_mxu_static` | attention 的编译器静态 MXU 利用率，不能填进 `mxu`。 |
| `roofline_efficiency` | 实际性能相对于所用 roofline 模型上限的比例。 |
| `logical_bandwidth_gbps`、`logical_bandwidth_utilization` | 根据逻辑数据量估算的 GB/s 及其峰值比例。 |
| `hbm_bandwidth_gbps`、`hbm_bandwidth_utilization` | 根据实测 HBM 流量得到的 GB/s 及其峰值比例。 |
| `hlo`、`llo` | 仅状态：`recorded`、`not_collected`、`unavailable`，不填文件内容或路径。 |
| `failure_reason` | 失败原因简述；没有则 null。 |

所有利用率统一 `1 = 100%`。`mfu`、实测 MXU/HBM 利用率、静态 MXU 的范围为 0～1；
执行量估算、roofline 和逻辑带宽模型的比例允许超过 1。
输出比例通常保留最多 4 位小数，耗时/带宽最多 3 位；极小正数如果会被舍入为零，则保留原精度。
`null` 表示未知，`0` 表示已有结果为零。
执行失败或未运行时不能填写耗时或 `correctness: pass`；p90 不能小于中位耗时。

## LLO 只存结果

原始 LLO/HLO、编译器 dump、profile 大文件、源码正文和附件不进入公开输入或最终 memory。
在本地解析这些文件后，只提交指标及短说明，例如：

```json
{
  "execution_status": "succeeded",
  "correctness": "pass",
  "latency_us": 503.94,
  "mfu": 0.2786,
  "mxu": null,
  "attention_mxu_static": 0.528,
  "llo": "recorded"
}
```

`llo: recorded` 只说明提供了相关信息/结果，不表示 memory 内有原文件。
公开入口只读取 `input.json`，不会打开里面提到的源码位置或 LLO 文件。
Project2 forward 适配器同样只读取 LLO 汇总 CSV，保留小型 `compile.csv` 的成功编译检查；不再依赖原始 LLO 目录。

## 校验与生成

Python 3.11+，在 `kernel-memory/` 中安装：

```bash
python -m pip install -e .
```

先验证，再生成：

```bash
python scripts/import_memory_input.py input.json --validate-only
python scripts/import_memory_input.py input.json ./memory
```

安装后也可用 `kmem-import input.json --validate-only` 和 `kmem-import input.json ./memory`。
本机 WSL 已有环境可把 `python` 替换为 `.venv/bin/python`。
试用仓库提供的合成示例：

```bash
.venv/bin/python scripts/import_memory_input.py examples/memory_input.json --validate-only
.venv/bin/python scripts/import_memory_input.py examples/memory_input.json .demo/input-example
```

要使用真实历史数据，可将示例路径替换为 `examples/project2_memory_input.json`，
并选择单独的输出目录（例如 `.demo/project2-input-example`）。

输入错误会返回非零退出码并说明字段位置。校验包括字段类型、单位范围、重复 ID、
baseline 引用和执行状态一致性；格式校验通过不代表真实性已经复测。
所有用户导入结果标记为 `imported_unverified`，不伪造比较协议或环境指纹。

**一次输入描述一个完整输出快照。** 重复导入到同一目标会刷新它，不做增量合并；
要累积多个用户的数据，先把它们整理进同一个输入，或分别选择输出目录。
输入文件放在输出目录外。输出目录中的无关文件会阻止覆盖。

输出保持：

```text
memory/
├── manifest.json
└── <kernel>/
    ├── kernel.json
    └── <algorithm>/
        ├── method_summary.json
        ├── baseline.json
        └── shape-<shape_id>-<hash>/
            ├── config.json
            └── trajectory/
                ├── attempt.json
                ├── code.json       # 提供源码位置时才有
                └── result.json
```

公开输入的 attempt 直接保存 `pr_url`、`commit` 和调优参数，不再让本地尝试看起来像真实 PR。
`run_ref` 在 attempt/code/result 之间关联同一次运行；输入的 `run_id` 保留在 attempt 中。
`result.json` 始终是固定的简单指标字段。新项目都使用同一入口；旧 Project2 输出字段见专用说明。

## 可以直接交给 AI 的提示词

```text
请将我提供的实验记录整理为 kernel-memory-input v1 JSON。

先阅读仓库 kernel-memory/README_MEMORY.md、
kernel-memory/src/kernel_memory/contracts/memory_input.v1.schema.json，
并参考 kernel-memory/examples/memory_input.json 的结构，不使用其合成测量值。

1. 根据实际实现给 kernel、algorithm、shape 分类，填写简短 method_summary。
2. 把固定输入张量和计算语义放在 problem，tile 等实现选项放在 attempt.parameters。
3. 每次尝试和每次运行分开保留；失败也保留。不要把不同 shape 或实验合为一条。
4. 只从我提供的记录中提取指标；耗时转成微秒，利用率转成比例（1=100%）。
5. 缺失指标填 null。静态 MXU 与运行时 MXU 分开，逻辑带宽与实测 HBM 带宽分开。
6. LLO/HLO 只提取结果；不嵌入正文、附件、日志或编码后的文件。
7. 没有真实 PR/Git commit 就填 null；没有 baseline 就不指定。不要编造记录或测量。
8. 输出 input.json，运行 import_memory_input.py input.json --validate-only，
   修正格式错误后再生成 memory。原始资料缺少必需事实时，明确报告缺项。
```

AI 负责理解来源并填入统一结构；程序负责检查结构并生成目录。
这两个步骤共用同一份版本化 Schema，避免每个人产出不同的 memory 格式。
