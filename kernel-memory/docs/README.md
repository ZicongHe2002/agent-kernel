# 文档索引

| 文档 | 用途 |
|---|---|
| [README_MEMORY.md](../README_MEMORY.md) | 当前公共入口：统一 input JSON、AI 整理提示词，以及生成精简 memory 的命令。 |
| [README_PROJECT2.md](../README_PROJECT2.md) | 现有 Project2 数据适配器与示例的生成方式。 |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | 2026-09-24 的历史实现和验收记录；不代表当前运行环境或最新测试结果。 |
| [DESIGN.md](DESIGN.md) | 内部 MemoryStore 的架构、共享 API、运行／比较／决策规则；公共输入与精简输出见 README_MEMORY。 |
| [RECOVERY.md](RECOVERY.md) | 存储完整性模型、发布协议、`kmem recover` / `kmem validate --deep`、锁和备份。 |
| [CONFIGURATION.md](CONFIGURATION.md) | 项目设置，GitHub / TPU / 模型 / LLO 配置指南，以及权限和预算。 |
| [MEMORY_GUIDE.zh-CN.md](MEMORY_GUIDE.zh-CN.md) | Windows 安装、Memory 输入、写入记录，以及后续优化时的 context/query/trajectory 读取方式。 |
| [MIGRATION.md](MIGRATION.md) | v0.1 → v0.2 数据迁移，以及 v0.2 存储/数据包 → v0.3（算法层级）升级：预期输入、映射规则，以及绝不通过推断补全的内容。 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 验收场景 T01–T32 与测试的对应关系及执行状态。 |
| [adr/](adr/) | 架构决策记录；[ADR-0004](adr/ADR-0004-algorithm-shape-hierarchy.md) 记录 kernel → algorithm → shape 层级与整核 `trajectory/`。 |

旧实现交付包和历史运行日志已按用户确认从工作目录移除，可在 Git 历史中查阅；当前公共输入的校验和生成不依赖它们。
当前机器可读输入规则见 [`memory_input.v1.schema.json`](../src/kernel_memory/contracts/memory_input.v1.schema.json)。
