# agent-kernel

面向计算内核（kernel）自动优化的智能体项目。当前已实现的组件是 **Kernel Memory**
（Python 包名 `kernel_memory`，命令行工具 `kmem`）：以文件为权威数据源的记忆系统，
用于存储、校验、检索、比较和重建内核优化历史，按以下层级组织数据：

`kernel（内核）→ config（配置）→ attempt（PR 集合）→ PR → commit（提交）→ run（运行）`。

## 目录结构

| 路径 | 用途 |
|---|---|
| `kernel-memory/` | 主项目：`src/kernel_memory/` 为源码包，`tests/` 为测试，`fixtures/handoff/` 保存数据契约和合成演示数据的副本，`scripts/` 提供演示及验收报告脚本，`docs/` 为项目文档。 |
| `kernel_memory_ai_handoff/` | 只读交付包，包含实现规范、机器可读的数据契约、合成示例和交付包校验工具。禁止向此目录写入内容；`MANIFEST.sha256` 必须保持校验通过，可运行 `cd kernel_memory_ai_handoff && shasum -a 256 -c MANIFEST.sha256` 检查。 |
| `LICENSE`、`.gitignore` | 许可证及 Git 忽略规则。 |

## 快速开始

以下命令从仓库根目录开始执行，进入 `kernel-memory/` 后完成环境配置、测试和演示。
使用 Python 3.11；本项目环境所用的固定依赖版本记录在 `kernel-memory/requirements.lock.txt` 中。

```bash
cd kernel-memory
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
# 仅限 macOS：若 `.venv/bin/python -c "import kernel_memory"` 因 .pth 文件被隐藏而失败，执行下一行：
chflags nohidden .venv/lib/python3.11/site-packages/*.pth
.venv/bin/python -m pytest tests -q
bash scripts/demo_p0.sh                     # 使用合成数据包进行离线演示
REPS=50 WARMUP=10 bash scripts/demo_cpu.sh   # 在本机 CPU 上实际运行向量加法演示内核
```

演示脚本仅向 `kernel-memory/.demo/` 写入数据，该目录已被 Git 忽略。
运行 `.venv/bin/kmem --help` 可查看命令行帮助。

## 文档导航

| 文档 | 内容 |
|---|---|
| [文档索引](kernel-memory/docs/README.md) | 项目文档入口。 |
| [实现状态](kernel-memory/docs/IMPLEMENTATION_STATUS.md) | 已完成的功能、实际执行过的验证、待完成事项及原因；继续开发前建议先阅读。 |
| [验收报告](kernel-memory/docs/ACCEPTANCE.md) | 验收场景 T01–T32 与已执行测试的对应关系及状态。 |
| [设计文档](kernel-memory/docs/DESIGN.md) | 系统架构、模块划分、公共 API、标识符及测试约定。 |
| [配置指南](kernel-memory/docs/CONFIGURATION.md) | 项目设置、GitHub / TPU / 模型 / LLO 配置、权限和预算。 |
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

* `.venv/`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`、`.DS_Store`、`kernel-memory/.demo/` 和 `*.sqlite`
  已加入 Git 忽略规则。不要提交虚拟环境或演示生成的存储数据。
* 仅使用 `kernel-memory/.venv` 中的 `pip` 安装依赖；不要在 `kernel_memory_ai_handoff/` 内运行 `pytest`，
  以免在交付包中生成 `__pycache__`，破坏只读输入的约定。
* 提交和推送由仓库所有者执行，详见[工作区边界约定](kernel-memory/docs/adr/ADR-0001-workspace-boundary.md)。
