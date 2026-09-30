# 文档索引

| 文档 | 用途 |
|---|---|
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | 已完成内容、实际执行记录（命令与结果）、待办事项及原因。恢复工作时请优先阅读。 |
| [DESIGN.md](DESIGN.md) | 架构、模块关系、共享 API、标识符约定、运行／比较／决策规则、安全与测试约定。 |
| [RECOVERY.md](RECOVERY.md) | 存储完整性模型、发布协议、`kmem recover` / `kmem validate --deep`、锁和备份。 |
| [CONFIGURATION.md](CONFIGURATION.md) | 项目设置，GitHub / TPU / 模型 / LLO 配置指南，以及权限和预算。 |
| [MEMORY_GUIDE.zh-CN.md](MEMORY_GUIDE.zh-CN.md) | Windows 安装、Memory 输入、写入记录，以及后续优化时的 context/query/trajectory 读取方式。 |
| [MIGRATION.md](MIGRATION.md) | v0.1 → v0.2 数据迁移，以及 v0.2 存储/数据包 → v0.3（算法层级）升级：预期输入、映射规则，以及绝不通过推断补全的内容。 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 验收场景 T01–T32 与测试的对应关系及执行状态。 |
| [adr/](adr/) | 架构决策记录；[ADR-0004](adr/ADR-0004-algorithm-shape-hierarchy.md) 记录 kernel → algorithm → shape 层级与整核 `trajectory/`。 |

上游输入（只读，不属于本项目）：[`../../kernel_memory_ai_handoff/`](../../kernel_memory_ai_handoff/)。
