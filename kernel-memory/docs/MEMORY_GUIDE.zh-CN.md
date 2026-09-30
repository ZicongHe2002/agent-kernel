# Kernel Memory 使用指南（Windows）

## 1. 先在 Windows 上安装

在 PowerShell 中执行：

```powershell
cd C:\Users\fengluo\Desktop\agent-kernel\kernel-memory
python -m venv .venv
\.venv\Scripts\python.exe -m pip install -e ".[dev,cpu-demo]"
\.venv\Scripts\python.exe -m pytest tests -q
```

如果系统没有 `python`，请先安装 Python 3.11+，并重新打开 PowerShell。需要记录真实 Git commit 时，还要安装 Git for Windows；只注册 problem、导入 bundle 和查询已有 Memory 不依赖 Git。之后可以直接用下面的方式调用 CLI：

```powershell
$env:PYTHONPATH = "$PWD\src"
$kmem = "$PWD\.venv\Scripts\python.exe"
& $kmem -m kernel_memory.cli.main --help
```

仓库中的 `*.sh` 是 Bash 演示脚本；原生 PowerShell 下可直接调用上面的 Python CLI，或在 WSL/Git Bash 中运行脚本。
Windows 锁已经使用 `msvcrt.locking`，POSIX 系统仍使用 `fcntl.flock`。

## 2. Memory 里到底存什么

Memory 的输入不是一段可以直接执行的自然语言 prompt，而是经过 schema 校验的事实记录。一个内核的轨迹按下面的层级组织：

```text
kernel
└─ algorithm（优化方法）
   └─ config / shape（固定计算问题）
      └─ PR 或 local trial
         └─ commit（代码变化、Change[]、diff）
            └─ run（环境、协议、正确性、耗时、分析证据）
```

因此通常有三类输入：

1. **问题输入**：`config/shape` 的 `problem` JSON，描述内核计算什么。它会被规范化并生成 `config_hash`。这不是运行时 tensor 的原始字节；例如 CPU demo 会由 adapter 根据 `n` 和固定 seed 生成输入，并在 Run 中记录 `input_suite_hash`。
2. **优化输入**：PR、commit、变化项 `changes`、方法说明、基线和候选实现。
3. **运行证据**：`run` 的 protocol、verifier、backend、源码身份、正确性结果、样本、耗时和 artifact。

`decision` 和 `annotation` 用于记录策略结论与经验。每条已发布记录不可修改；JSON 记录是事实源，`trajectory`、`memory_records.jsonl` 和 SQLite 索引都是可重建视图。

## 3. 直接输入一个 problem

当前内置的 `demo_vector_add` 支持：

```json
{"n": 4096, "dtype": "f32", "operation": "vector_add", "outputs": ["y"]}
```

PowerShell 示例：

```powershell
$root = "$PWD\memory"
& $kmem -m kernel_memory.cli.main --root $root init --json

$problem = @'
{
  "n": 4096,
  "dtype": "f32",
  "operation": "vector_add",
  "outputs": ["y"]
}
'@
# PowerShell 5 的 `Set-Content -Encoding utf8` 会写入 BOM；严格 JSON 读取时使用无 BOM UTF-8。
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $PWD 'problem.json'), $problem, $utf8NoBom)

& $kmem -m kernel_memory.cli.main --root $root register-kernel `
  --kernel-id demo_vector_add --display-name "CPU demo vector add" `
  --adapter-id cpu-demo-v1 --notes "demo kernel" --json

& $kmem -m kernel_memory.cli.main --root $root register-algorithm `
  --kernel-id demo_vector_add --algorithm-id numpy-add `
  --method-summary "Elementwise vector addition with NumPy." --json

$configResult = & $kmem -m kernel_memory.cli.main --root $root register-config `
  --kernel-id demo_vector_add --algorithm numpy-add `
  --problem '@.\problem.json' --json | ConvertFrom-Json
$configId = $configResult.config.record_id
$configId
```

命令输出中的 `config.record_id` 是后续查询的 shape ID。`register-config` 会校验和规范化 problem；未知字段、类型错误或不完整的 kernel contract 会被拒绝。当前 `mla_forward` 仍然明确处于不完整状态，必须先根据真实 MLA 代码补齐 ABI、problem schema、reference 和 verifier，不能凭历史参数猜测。

## 4. 已有完整数据如何直接写入

如果手里已经有完整的记录集合，可以使用：

```powershell
& $kmem -m kernel_memory.cli.main --root $root import-bundle `
  .\fixtures\handoff\examples\demo_bundle.json `
  --artifact-root .\fixtures\handoff --allow-fixture --json
```

bundle 顶层需要有 `bundle_version`、`records`，每条 record 都必须包含 `schema_version`、`record_type`、`record_id`、RFC3339 `created_at` 和完整 `payload`；记录之间的引用必须闭合，artifact 需要能按声明的 hash 找到。`--allow-fixture` 只适合合成数据，fixture 或未经认证导入的运行结果不会成为生产候选。

新数据更适合使用 `register-*`、`record-commit`、`run`、`annotate` 和 `decide`，因为这些入口会自动做引用检查、规范化和幂等处理。

如果必须保存一份可重放的原始输入，应把输入序列化成 artifact，并在对应 Run 的 artifact 引用中登记；只把输入写入任意 JSON 文件而不建立 artifact 引用，后续校验和查询都不会把它当作运行证据。

## 5. 后续开发和优化时怎么读 Memory

### 面向智能体的推荐入口：`export-context`

```powershell
& $kmem -m kernel_memory.cli.main --root $root `
  export-context --config <config_record_id> --max-records 30 --json `
  > .\context.json
```

输出是纯读的 `context-v2`，包含当前算法说明、基线、已确认候选、临时候选、失败分支、未测试提交、近期变化、经验注释、阻断决策、证据引用和 `record_refs_included`。它是后续 planner 或人工开发最适合消费的上下文。

### 结构化查询

```powershell
# 查看某个 shape 的全部记录
& $kmem -m kernel_memory.cli.main --root $root query --config <config_record_id> --json

# 只看还没有运行记录的 commit
& $kmem -m kernel_memory.cli.main --root $root query `
  --config <config_record_id> --record-type commit --run-status not_run --json

# 按代码变化组件查询
& $kmem -m kernel_memory.cli.main --root $root query `
  --config <config_record_id> --component tiling --json
```

`query` 支持按 kernel、algorithm、config_hash、record_type、component、parameter、subject、PR、运行状态、正确性、provenance、comparison key、decision outcome 和 reason code 过滤。`not_run` 是根据没有 Run 推导出来的状态，不会伪造一条运行记录。

### 读取轨迹视图或权威 JSON

```powershell
& $kmem -m kernel_memory.cli.main --root $root `
  trajectory --config <config_record_id> --rebuild --print-view --json

& $kmem -m kernel_memory.cli.main --root $root `
  trajectory --kernel demo_vector_add --rebuild --print-view --json
```

需要注意：`trajectory/` 下的文件可以删除并重建，不能当作唯一事实源。要读取原始事实，应读取 `kernels\...\kernel.json`、`algorithm.json`、`config.json`、`commit.json`、`run.json` 等 JSON 文件，或使用 Python API：

```python
from pathlib import Path
from kernel_memory.storage import MemoryStore
from kernel_memory.services.context import export_context
from kernel_memory.services.query import QueryFilters, query_memory

store = MemoryStore.open(Path(r".\memory"))
record = store.get("<config_record_id>")
runs = store.records("run")
context = export_context(store, "<config_record_id>", max_records=30)
results = query_memory(
    store,
    QueryFilters(config_ref="<config_record_id>", record_type="run"),
).to_dict()
```

一个实际的优化循环是：读取 `export-context` → 选择基线和未测试 commit → 修改 kernel → 用 `record-commit` 或 `run --overrides` 记录候选 → 运行正确性和性能验证 → `compare` → `decide` → 再次导出 context。Memory 保存的是事实和证据，优化器是否接受候选由明确的 policy 和 decision 记录决定。
