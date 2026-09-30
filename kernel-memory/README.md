# Kernel Memory (`kernel_memory`, CLI `kmem`)

面向内核优化智能体、以文件为基础的权威记忆系统（Memory）。它按照以下层级存储、校验、
检索、比较和重建内核优化历史：

```text
Memory
└── kernel（name / kernel_id）
    ├── trajectory/                  # 自动生成的整核视图：覆盖全部 algorithm 与 shape；不保存独有事实，可重建
    ├── algorithm_1                  # 一种优化方法；method_summary 为人工/智能体撰写的说明（数据，不是测量）
    │   ├── shape_1                  # 一个固定的计算问题（= config 记录，config_hash 与算法无关）
    │   │   ├── attempt/             # 此 shape 下所有 PR 的集合 → PR → commit（changes + summary）→ run
    │   │   ├── code                 # 生成视图：commit 的 diff 证据 + 各 run 的源码/变体摘要
    │   │   └── result               # 生成视图：各 run 的执行状态、正确性、耗时、分析指标、HLO/LLO 证据槽
    │   ├── shape_2
    │   └── shape_3
    ├── algorithm_2
    └── algorithm_3
```

`attempt` 表示 PR 的集合，而非单个 PR。未经测试的提交始终保留未测试状态。测试夹具数据
不会进入生产排名。同一 shape 在不同 algorithm 下拥有相同的 `config_hash`，因此可以比较；
v0.2 数据导入时会为每个 kernel 补上占位算法 `unspecified`（其 `method_summary` 明确写明未描述方法）。
设计文档、架构决策记录（ADR，本层级见 ADR-0004）、实现状态和操作指南见 `docs/`。

## 环境配置

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"      # 此处使用的固定依赖版本见 requirements.lock.txt
.venv/bin/python -m pytest tests -q
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,cpu-demo]"
$env:PYTHONPATH = "$PWD\src"
& .\.venv\Scripts\python.exe -m pytest tests -q
& .\.venv\Scripts\python.exe -m kernel_memory.cli.main --help
```

macOS 注意事项：如果在虚拟环境中执行 `import kernel_memory` 失败，请运行
`chflags nohidden .venv/lib/python3.11/site-packages/*.pth`（Python 会跳过隐藏的 `.pth` 文件），
或使用不依赖 `.pth` 的备用方式：
`ln -s ../../../../src/kernel_memory .venv/lib/python3.11/site-packages/kernel_memory`。

## 演示

两个脚本都在当前目录下使用 `.venv` 运行，无需网络，仅向 `.demo/` 写入文件
（该目录已被 Git 忽略；每次运行都会删除并重建目标存储）。

* `bash scripts/demo_p0.sh [STORE]` — 使用合成的交接数据包（原样的 v0.2 数据包，导入时在内存中升级到 v0.3，
  为 kernel 补上占位算法），离线演示 P0 流程：`init`、两次 `import-bundle --allow-fixture`（第二次导入满足幂等性，
  19 条记录）、`validate --deep`、确定性的 `trajectory --kernel demo_vector_add --rebuild` / `--verify`、
  `query`（按算法查询 shape、查询未经测试的提交和分块变化）、
  `compare run-demo-a run-demo-baseline`（夹具数据的计算结果：加速比为 100/90，耗时降低 10%）、
  `decide --dry-run`（被阻止，原因是 `FIXTURE_NOT_ELIGIBLE`）、`export-context`、`status`。
  默认存储为 `.demo/p0-memory`。其中所有数值均来自测试夹具，并非实际测量结果。
* `REPS=50 WARMUP=10 bash scripts/demo_cpu.sh [STORE]` — 在本机 CPU 上实际执行演示用向量加法内核：
  注册内核、算法（`register-algorithm`，附人工撰写的方法说明）、shape 和基线，执行并测量基线及一个覆盖运行时参数的变体（`provenance=trusted_worker`），
  重放同一请求 ID 而不再次执行，将一个故意出错的候选记录为执行状态 `succeeded`、正确性 `fail`，
  进行比较，用一组配对数据作出决策（结果为 `inconclusive`），展示 `jax_tpu` 后端明确拒绝执行，
  进行深度校验，重建轨迹，并在小额预算下运行 MockPlanner。`REPS` 和 `WARMUP` 分别设置基准测试
  重复次数和预热迭代次数（默认分别为 50 和 10）。默认存储为 `.demo/cpu-memory`；
  协议、验证器、配对数据和预算文件保存在 `.demo/cpu-demo-files/`。
  此演示仅展示 CPU 上的执行集成，不代表 MLA 或 TPU 的性能。

## 文档

`docs/README.md`（文档索引）、`docs/IMPLEMENTATION_STATUS.md`（已完成内容、实际执行记录和待办事项）、
`docs/ACCEPTANCE.md`（验收场景 T01–T32 与实际执行测试的对应关系及状态）、`docs/DESIGN.md`、
`docs/CONFIGURATION.md`、`docs/RECOVERY.md`、`docs/MIGRATION.md`、`docs/MEMORY_GUIDE.zh-CN.md`、`docs/adr/`。
