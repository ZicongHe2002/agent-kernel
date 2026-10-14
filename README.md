# agent-kernel

**公共 Memory 输入：** [统一 JSON 输入规范及 AI 整理提示词](kernel-memory/README_MEMORY.md)。
现有 [`.demo/memory/`](kernel-memory/.demo/memory/) 示例的生成和阅读方式见 [Project2 适配说明](kernel-memory/README_PROJECT2.md)。
该示例采用精简阅读结构；下文的存储层级描述内部 MemoryStore。

面向计算内核（kernel）自动优化的智能体项目。当前已实现的组件是 **Kernel Memory**
（Python 包名 `kernel_memory`，命令行工具 `kmem`）：以文件为权威数据源的记忆系统，
用于存储、校验、检索、比较和重建内核优化历史，按以下层级组织数据：

`kernel（内核）→ { trajectory（整核生成视图）, algorithm（算法/方法）→ shape（= config，固定计算问题）→ attempt（PR 集合）→ PR → commit（提交）→ run（运行）}`
（2026-09-28 起，见 `kernel-memory/docs/adr/ADR-0004-algorithm-shape-hierarchy.md`）。

## 目录结构

| 路径 | 用途 |
|---|---|
| `kernel-memory/` | 主项目：`src/kernel_memory/` 为源码包，`tests/` 为测试，`fixtures/handoff/` 保存数据契约和合成演示数据的副本，`scripts/` 提供演示及验收报告脚本，`docs/` 为项目文档。 |
| `LICENSE`、`.gitignore` | 许可证及 Git 忽略规则。 |

给同事生成 Memory 时，交付 `kernel-memory/` 即可，使用说明已经包含在
`kernel-memory/README_MEMORY.md` 中。同事提供自己的实验记录，整理成 `input.json` 后生成 `memory/`。
打包时排除 `.venv/`、Python/测试缓存和临时演示输出；`.demo/memory/` 是可选的真实历史结果示例。
早期的 `kernel_memory_ai_handoff/` 交付包和旧演示输出已清理；程序和测试使用项目内的契约及 fixtures。

## 快速开始

以下命令从仓库根目录开始执行，进入 `kernel-memory/` 后完成环境配置、测试和演示。
使用 Python 3.11；本项目环境所用的固定依赖版本记录在 `kernel-memory/requirements.lock.txt` 中。

```bash
cd kernel-memory
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,jax]"
# 仅限 macOS：若 `.venv/bin/python -c "import kernel_memory"` 因 .pth 文件被隐藏而失败，执行下一行：
if [ "$(uname -s)" = Darwin ]; then chflags nohidden .venv/lib/python3.11/site-packages/*.pth; fi
.venv/bin/python -m pytest tests -q
bash scripts/demo_p0.sh                     # 使用合成数据包进行离线演示
REPS=50 WARMUP=10 bash scripts/demo_cpu.sh   # 在本机 CPU 上实际运行向量加法演示内核
```

Windows PowerShell 可使用原生 Python CLI（演示脚本本身是 Bash）：

```powershell
cd kernel-memory
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,jax]"
$env:PYTHONPATH = "$PWD\src"
& .\.venv\Scripts\python.exe -m pytest tests -q
& .\.venv\Scripts\python.exe -m kernel_memory.cli.main --help
```

演示脚本仅向 `kernel-memory/.demo/` 写入数据。普通演示目录已被 Git 忽略，
其中 `kernel-memory/.demo/memory/` 是单独保留并提交的 Project2 阅读示例。
运行 `.venv/bin/kmem --help` 可查看命令行帮助。

WSL 中先进入 `/mnt/c/Users/fengluo/Desktop/agent-kernel`，再执行上面的 Bash 命令。
Windows 和 WSL 的 `.venv` 不能共用；如果两边都要运行，请使用不同 checkout。
`dev` 包含 pytest、NumPy 和 PyYAML；`jax` 用于真实 JAX CPU 测试，无需 TPU。
只安装 `.[dev]` 也可以测试 Memory，但真实 JAX 测试会明确跳过。

WSL 中需要保留完整输出时，推荐在仓库根目录运行：

```bash
bash kernel-memory/scripts/check_wsl.sh
```

脚本会创建或使用项目的 Linux `.venv`、安装 `.[dev]`（包含 PyYAML），再依次运行测试、
P0 演示和 CPU 演示。这些阶段的标准输出和报错都保存到 `kernel-memory/.demo/checks/latest.txt`，
终端仅显示阶段、退出码和日志路径；失败或 Ctrl+C 中断后停止，保留已写入的日志。
每次运行覆盖该文件，需要保留旧结果时先另存。之后可以直接让 Codex 读取这个文件，无需复制终端内容。
加 `--with-jax` 会安装 `.[dev,jax]` 并包含真实 JAX CPU 测试。

`.gitattributes` 要求 Bash、fixture 和契约文件使用 LF。已有 CRLF 文件不会因为添加该配置而自动修复；
若出现大量 `artifact checksum mismatch`、legacy schema hash 不符或 Bash `pipefail` 报错，
请看 [WSL 测试与换行排查](kernel-memory/docs/MEMORY_GUIDE.zh-CN.md#wsl-测试与换行排查)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [文档索引](kernel-memory/docs/README.md) | 项目文档入口。 |
| [实现状态](kernel-memory/docs/IMPLEMENTATION_STATUS.md) | 已完成的功能、实际执行过的验证、待完成事项及原因；继续开发前建议先阅读。 |
| [验收报告](kernel-memory/docs/ACCEPTANCE.md) | 验收场景 T01–T32 与已执行测试的对应关系及状态。 |
| [设计文档](kernel-memory/docs/DESIGN.md) | 系统架构、模块划分、公共 API、标识符及测试约定。 |
| [配置指南](kernel-memory/docs/CONFIGURATION.md) | 项目设置、GitHub / TPU / 模型 / LLO 配置、权限和预算。 |
| [Memory 使用指南](kernel-memory/docs/MEMORY_GUIDE.zh-CN.md) | Windows 安装、problem 输入、bundle 导入，以及 query/context/trajectory 读取方式。 |
| [恢复指南](kernel-memory/docs/RECOVERY.md) | 存储完整性模型、数据发布协议，以及 `kmem recover`、`kmem validate --deep` 的使用。 |
| [迁移指南](kernel-memory/docs/MIGRATION.md) | v0.1 → v0.2 的输入格式、映射规则、禁止推断的信息及验证方式。 |
| [架构决策记录](kernel-memory/docs/adr/) | 重要架构决策及其背景。 |

## 功能边界与验证范围

* 所有测试样例（fixture）均为合成数据。演示数据中的耗时（100/90/88 µs）仅用于验证计算和展示逻辑，
  不代表 CPU、GPU 或 TPU 实测性能，样例记录也不会作为生产结果。
* 尚未执行真实 TPU 任务、在线 GitHub 采集、模型驱动的规划或 LLO 解析。
  当硬件、令牌、凭据、格式样例或授权等前置条件缺失时，相应功能会明确报告不可用、未授权或不支持；
  这些处理路径已进行测试。具体情况见[实现状态](kernel-memory/docs/IMPLEMENTATION_STATUS.md)。
* CPU 演示在本机测量向量加法，用于验证执行流程的集成情况，其结果不代表 MLA 或 TPU 性能。

## 仓库维护约定

* `.venv/`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.DS_Store`、普通 `kernel-memory/.demo/` 子目录和 `*.sqlite`
  已加入 Git 忽略规则。`kernel-memory/.demo/memory/` 是用于阅读的已提交示例；不要提交虚拟环境或其他临时演示存储。
* 仅使用 `kernel-memory/.venv` 中的 `pip` 安装依赖；运行测试所需的合成数据在 `kernel-memory/fixtures/`，应随源码保留。
* 提交和推送由仓库所有者执行，详见[工作区边界约定](kernel-memory/docs/adr/ADR-0001-workspace-boundary.md)。
