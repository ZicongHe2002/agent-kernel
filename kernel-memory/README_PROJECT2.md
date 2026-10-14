# Project2 适配器：历史输入格式与结果阅读

所有项目共用的公开输入规范见 [Memory 输入 README](README_MEMORY.md)。
本页只说明如何适配现有 Project2 目录，以及该历史示例中沿用的字段名称。

当前入口接收 **Project2 实验结果目录**，整理其中已经完成的实验，输出
[`.demo/memory/`](.demo/memory/) 中的固定目录结构。它不会重新运行 TPU 实验。
只想看结果时，依次打开 `method_summary.json`、`config.json` 的 `problem`、`trajectory/result.json`。

这份 README 对应 `build_project2_memory.py` 的 `project2-v1` 导入规则和
`kernel-memory-readable` v2 输出。示例包含 2 个 kernel、11 个方法、29 个 shape、128 条运行记录。
它覆盖下面列出的五组输入，并不代表已经收录 Project2 的全部后续实验。

## 1. 输入是什么

生成命令接收两个位置参数：

```text
python scripts/build_project2_memory.py <project2目录> <memory输出目录>
```

例如输入 `C:\Users\fengluo\Desktop\project2`，输出仓库内的 `kernel-memory/.demo/memory`。
现有的 `config.json`、`attempt.json`、`code.json`、`result.json` 都是生成结果，不用手工填写后交给这个命令。

输入目录需要保留这五组实验的路径和内部结构：

```text
project2/
└── result/
    ├── decode/
    │   ├── result-decode-2/          # 递归寻找 results.json 所在的实验目录
    │   ├── result-decode-3/          # 同上
    │   └── mla_decode_single/
    │       └── rerun/               # 单组 decode 结果
    ├── baseline_unoptimized_tpu_20260907/
    └── result_6_tpu/
```

**ZIP 需要先解压。** 当前生成脚本不直接读取 ZIP，也不会自动收集其他目录。
输入中的原始文件保持不变；输出目录必须位于 Project2 目录之外。

### Decode 输入文件

以下路径相对于每个包含 `results.json` 的实验目录；矩阵目录顶层只有
`matrix_results.json` 或 `matrix_performance.csv` 不够。

| 文件 | 用途与要求 |
|---|---|
| `results.json` | 必需。提供 `configuration`、`candidates`、源代码标识及实验环境。 |
| `performance.csv` | 必需。按 `name` 对应候选；提供 `device_median_us` 及已有 MFU、带宽、roofline 等指标。 |
| `accuracy.csv` | 必需。按 `name`、`gate` 对应正确性测试；候选声明 PASS 时必须有对应的通过记录。 |
| `source_snapshot/<文件>` | `results.json.source_sha256` 列出的文件必须存在，内容 SHA-256 必须一致。 |
| `<候选名>/before_opt.hlo` 等编译输出 | 用于读取已有接口信息及记录 HLO 状态；保留原实验中的这些文件即可，最终 memory 不复制它们。 |

`configuration` 记录 `batch`、`heads`、`seq` 等问题尺寸，以及已有的 `length`、
采样次数、seed 等配置。`candidates[]` 记录候选的 `name`、精度分类、实现参数、
`status`、`compile_status`、`correctness_status`、`accuracy` 和 `device_samples_ms`。

对于 `status = "MEASURED"` 的候选，编译与正确性都必须通过；设备样本必须有效，
样本数及其中位数必须与配置和 CSV 一致。失败候选也会保留，其耗时可以是 `null`。
不要只改汇总 CSV 中的一个数值：导入器会与原始样本交叉检查。

### Forward 输入文件

以下文件分别放在两个 forward 实验目录下：

| 文件 | 用途与要求 |
|---|---|
| `manifest.json` | 必需。包含 `created_utc`、源码标识以及已有的 `environment`、`run_plan`。 |
| `device_event_performance_mfu.csv` | 必需。选择 `scope = whole` 的设备测量；这是最终耗时与 MFU 的来源。 |
| `compile.csv` | 必须能按 `run_id` 找到 `result = PASS` 的编译记录。 |
| `accuracy.csv` | 可选；按 `run_id` 查找正确性结果，缺少对应测量的记录时输出 `correctness: "not_run"`。 |
| `performance_mfu.csv` | 可选的 host 测量；不替代设备耗时。 |
| `device_event_raw_samples.csv` | 可选；有内容时，按 `run_id` 和 `event_scope` 与设备样本逐项核对。 |
| `llo_static_utilization.csv` | 可选；按 `run_id`、`bundle_owner`、`resource` 读取编译器静态利用率。 |
| `llo_pressure_summary.csv` | 可选的编译器资源压力摘要。 |
| 原始 HLO/LLO 目录 | 不需要。Forward 导入不再读取、扫描或复制这些文件，`compile.csv` 中的目录字段也不使用。 |

性能表中的关键列如下：

| 列 | 含义 |
|---|---|
| `run_id`、`implementation` | 运行编号与候选名称；`run_id` 用于关联其他表。 |
| `scope`、`output_mode` | 测量范围与输出模式；当前完整生成入口选择 `whole`。 |
| `batch`、`heads`、`seq_len`、`q_rank`、`kv_rank`、`d_nope`、`d_rope`、`d_value`、`sm_scale` | 固定输入问题及张量尺寸。 |
| `round_medians_ms` | CSV 单元格中的 JSON 数字数组，单位毫秒。 |
| `sample_count`、`median_ms`、`p90_ms` | 必须与上述样本数组一致。 |
| `useful_mfu_percent`、`executed_mfu_percent_estimate` | 原始百分数；最终输出转换成 `1 = 100%` 的比例。 |
| `options_json`、`block_q`、`block_kv` 等 | 候选的实现选项和分块参数。 |
| `is_leader_aligned_original` | 值为 `True` 时，标识输入中声明的原始基线。 |

`manifest.json` 需要能识别被测源码：有效的 Git commit 或源码归档 SHA-256，
并提供有效的源码摘要。缺少实测数据、编译记录、源码标识，或样本与统计不一致时会报错。
正确性不能用其他 shape 的测试代替，静态 MXU 也不能当成运行时 MXU。

### 原始单位与输出单位

| 数据 | 输入 | 输出 |
|---|---|---|
| Decode 样本 | `device_samples_ms`：毫秒 | `latency_us`：微秒 |
| Forward 样本 | `round_medians_ms`：毫秒 | `latency_us`：微秒 |
| MFU 等百分数 | `27.8567` 表示 27.8567% | `mfu: 0.2786` |
| Forward 静态利用率 | `average_static_utilization` 本身为比例 | `attention_mxu_static` 仍为比例 |
| 带宽 | GB/s | `*_bandwidth_gbps`：GB/s，单位是字节，不是 bit |
| 缺失指标 | 缺列、空值或未采集 | `null`，不当成 0 |

比例保留最多 4 位小数，耗时与带宽保留最多 3 位小数。估算结果超过 1 时保留原义，不强行截为 1。

## 2. 如何生成

在 WSL 中，使用当前项目的 Linux Python 环境：

```bash
cd /mnt/c/Users/fengluo/Desktop/agent-kernel/kernel-memory
.venv/bin/python scripts/build_project2_memory.py \
  /mnt/c/Users/fengluo/Desktop/project2 \
  .demo/memory
```

新环境先在 `kernel-memory/` 下执行 `python3 -m venv .venv`，再运行
`.venv/bin/python -m pip install -e ".[dev]"`。这条历史数据导入流程不需要 TPU 或 JAX。

如果在 Windows PowerShell 中操作，可以直接调用上面已经准备好的 WSL 环境：

```powershell
wsl.exe --distribution Ubuntu --exec `
  /mnt/c/Users/fengluo/Desktop/agent-kernel/kernel-memory/.venv/bin/python `
  /mnt/c/Users/fengluo/Desktop/agent-kernel/kernel-memory/scripts/build_project2_memory.py `
  /mnt/c/Users/fengluo/Desktop/project2 `
  /mnt/c/Users/fengluo/Desktop/agent-kernel/kernel-memory/.demo/memory
```

不要用 Windows Python 执行 WSL 的 `.venv/bin/python`，也不要覆盖已有的 WSL 环境。
生成成功时终端会显示 `validation.ok: true`、`output_verified: true` 和记录数量；
`measurements_reexecuted: false` 表示这些是历史测量。

重复运行会刷新输出。输出目录如果有生成清单之外的文件，程序会拒绝覆盖；
因此说明文档放在这里，不放进 `.demo/memory/`。

## 3. 输出里的每个文件

```text
memory/
├── manifest.json
└── <kernel>/
    ├── kernel.json
    └── <algorithm>/
        ├── method_summary.json
        ├── baseline.json
        └── shape-.../
            ├── config.json
            └── trajectory/
                ├── attempt.json
                ├── code.json
                └── result.json
```

| 文件 | 回答的问题 |
|---|---|
| `manifest.json` | 总共有多少记录、文件是否齐全、指标采用什么单位？ |
| `kernel.json` | 这个 kernel 做什么，有哪些方法？ |
| `method_summary.json` | 这个方法的思路是什么？ |
| `baseline.json` | 各 shape 的原始基线是什么？没有来源声明的基线时保留空列表。 |
| `config.json` | 这组实验在计算什么固定输入问题？ |
| `attempt.json` | 试过哪些候选，各自对应哪些运行？ |
| `code.json` | 各次运行使用哪份源码、哪个入口、哪些实现参数？ |
| `result.json` | 各次运行是否正确、耗时多少、利用率多少？ |

`baseline.json` 中的 `shared_baselines` 是其他方法在相同固定输入问题下已有的基线引用。
共享 shape 本身不保证环境、计时范围或正确性协议可直接比较。

方法分类和总结已经写进两个导入脚本的映射中，生成时会复用这些 AI 整理的说明，
不会每次再调用模型。指标来自输入记录，不由 AI 猜测。

## 4. 用 dense-aligned-dma-attention 这一组读一次

打开
[`mla_decode/dense-aligned-dma-attention/shape-b1-h16-n4096-727741dca6d4/`](.demo/memory/mla_decode/dense-aligned-dma-attention/shape-b1-h16-n4096-727741dca6d4/)。
这个目录有 4 个候选，共用固定输入：

| 张量 | shape | dtype |
|---|---|---|
| `q` | `[1, 16, 576]` | bfloat16 |
| `kv` | `[1, 4096, 576]` | bfloat16 |
| `lengths` | `[1]` | int32 |
| 输出 `output` | `[1, 16, 512]` | bfloat16 |

这里的 `lengths` 表示有效 token 数，计时配置 `timed_length` 为 4096。
`576 → 640` 是方法内部的 padding，外部输入仍是 576 维。

### attempt.json：尝试记录

`attempts[]` 的一项表示一个候选尝试，里面的 `runs[]` 指向它对应的运行。
当前这组数据为每个候选记录了一次运行：

| `title` / 候选 | KV tile | DMA2 | `run_ref` | 中位耗时 µs |
|---|---:|---|---|---:|
| `A_K1024` | 1024 | false | `run-5ef4bf1796f8` | 13.999 |
| `A_K2048` | 2048 | false | `run-f1ebdce8be35` | 13.230 |
| `A_K2048_DMA2` | 2048 | true | `run-cddf2d296e0b` | 13.046 |
| `A_K4096_DMA2` | 4096 | true | `run-d707a68a1a74` | 12.746 |

耗时取自 `result.json`；`attempt.json` 中重点看 `title`、
`runs[].implementation_overrides.implementation_parameters` 和 `runs[].run_ref`。

| 其他字段 | 这组数据里的含义 |
|---|---|
| `pr.provider: "local"`、`pr.number: null` | 本地实验上下文；原始记录没有关联 GitHub PR。 |
| `subject_ref` | 对应实现记录的内部 ID，例如 `commit-5ef4bf1796f8`。 |
| `context_ref` / `pr_ref` | 本地尝试上下文的内部 ID。 |
| `commit_oid` | 此例为源码快照的 SHA-256 内容标识，不是 GitHub commit。看 `source_identity_kind` 区分。 |
| `changes: []` | 没有提取源码 diff；不能理解成代码一定没改。 |
| `status: "recorded"` | 已有运行记录；是否成功、是否正确要看 `result.json`。 |
| `attempt_no`、`rerun_of` | 同一请求的执行次数及重跑关联；不是这个方法下候选的排名。 |

### code.json：源码信息与参数

`entries[]` 按运行保存源码描述。以 `A_K1024` 为例：

```text
run_ref       = run-5ef4bf1796f8
entrypoint    = project2:source_snapshot/aligned_kernels.py#candidate=A_K1024
block_kv      = 1024
dma2          = false
content       = null
```

`entrypoint` 表示原实验源码快照中的文件与候选名称；这是来源标识，不是可以直接执行的命令。
`content: null` 和 `content_status: "not_retained_in_compact_output"` 表示最终 memory
没有复制源码正文。需要看实现时，回到原实验的 `source_snapshot/aligned_kernels.py`。

`source_digest` 标识源码内容；`variant_digest` 还结合入口和实现选项来区分变体。
因此同一份源码文件可以对应多个不同的 candidate。
`target_commit`、`tested_commit`、`checkout_mode` 等名称沿用了内部记录结构；
此例应结合 `source_identity_kind: "source_snapshot_content_address"` 阅读，不能据此认定存在 Git 提交或执行了新的 checkout。

`config_implementation_parameters` 保存从原始 config 提取出来的调优参数；
`source.implementation_overrides` 保存实际候选参数。此例前者叫 `block_size: 1024`，
后者叫 `block_kv: 1024`，来自两种记录方式。
`attempt.json` 也保留了 `implementation_overrides`，所以两个文件当前有部分重复。
其中 `native_tpu_status` 等文字是原始实现描述，实际运行结论以 `result.json` 为准。

### config.json：problem 与 source 开头的字段

先看 `problem`：它就是本目录共用的输入/输出张量、尺寸、dtype、精度、生成规则和计算语义。

| 字段 | 含义 |
|---|---|
| `source_config_refs` | 导入阶段生成、随后汇总到本 shape 的内部 config ID 列表。`source` 指导出前的配置来源，不是输入张量，也不要求额外准备 source 文件。 |
| `source_configs` | 对这些 ID 的补充说明，列出各自的配置摘要和使用的格式规则。 |
| `source_configs[].config_ref` | 与列表中的 ID，以及 attempt/code/result 中的 `source_config_ref` 对应。 |
| `source_configs[].config_hash` | 原始配置内容的指纹；可能包括 tile 等实现设置。 |
| `problem_schema_id` | 原始配置遵循的规则版本，例如 `mla-decode:v2`。 |
| `problem_schema_digest` | 该规则文件的指纹，防止同名规则内容变了却未被发现。 |
| `shape_hash` | 按 kernel 和导出的固定 `problem` 计算的指纹，目录末尾是前 12 位。 |

本例有 **4 个候选，但导入阶段只生成了 3 个 config**：tile 分别为 1024、2048、4096。
`A_K2048` 与 `A_K2048_DMA2` 共用 `cfg-dma-b1-h16-n4096-01944e306469`，
它们的 DMA2 差别记录在候选实现参数中。
导出时不把 tile 等调优参数作为独立 shape，因此这 3 个 config 被汇总到同一个目录。
这也不表示只有 batch/head/length 相同就会合并：dtype、张量布局、精度和其余 `problem` 字段也参与 shape 标识。

例如，用 `run-5ef4bf1796f8` 可以在 attempt、code、result 三个文件中找到同一次运行；
它的 `source_config_ref` 是 `cfg-dma-b1-h16-n4096-3330c0051469`，可在 config 中找到来源配置。
阅读性能时无需手工计算这些 hash。

## 5. 当前输入格式的边界

目前“固定输入”指的是上面的 **Project2 目录结构及已有 JSON/CSV 字段**。
公开入口现在接收 [统一 JSON 格式](README_MEMORY.md)，可以手工填写，也可以让 AI 整理。
直接读取新项目的原始日志时仍需适配器，或让 AI 先转成统一 JSON；仅改 Project2 文件夹名称不够。

最终 `.demo/memory` 是精简阅读输出，不是 `kmem init` 创建的内部 MemoryStore，
因此不应把它直接交给要求内部 store 的 `kmem validate --deep`、`import-bundle` 等命令。
生成脚本已先验证内部记录，再检查最终输出的文件 hash 和引用关系。
这些检查确认记录一致，`imported_unverified` 仍表示历史测量尚未由本次导入独立复现。
