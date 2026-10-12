# project2 的 MLA decode 输入与 Kernel Memory 映射

本文记录对 `C:\\Users\\fengluo\\Desktop\\project2` 的只读检查结果。`project2` 没有被修改。

## 先看结论

project2 的 native decode 逻辑接口都是四个输入，但物理 cache 布局取决于
`interface`。连续 cache 的 signoff 版本和分页 cache 的实际单项 TPU 记录必须分开登记。

连续 cache（`mla_decode.yaml` 的 non-paged signoff 版本）：

```text
q:       bfloat16 [B, H, 576]
new_kv:  bfloat16 [B, 576]
cache:   bfloat16 [B, capacity, 640]
lengths: int32    [B]
```

分页 cache（`result\\decode\\mla_decode_single\\rerun` 的实际 native 记录，B=8、容量=16384）：

```text
q:       bfloat16 [8, 16, 576]
new_kv:  bfloat16 [8, 576]
cache:   bfloat16 [1024, 64, 2, 640]
lengths: int32    [8]
```

这里每个 page 是 `[64, 2, 640]`，8 个 page 组成一个 1024-token KV tile；分页 cache
的第一维是总 page 数，而不是 batch。`mla.export.json` 的 entry layout 是判断实际
测量 ABI 的权威证据：该记录明确导出为
`(bf16[8,16,576], bf16[8,576], bf16[1024,64,2,640], s32[8])`。

返回两个输出：

```text
attention:      bfloat16 [B, H, 512]
updated_cache:  与输入 cache 相同的物理 shape
```

其中 `new_kv` 会追加到 cache。连续布局对正长度请求的逻辑写入位置是
`cache[b, clip(lengths[b]) - 1, :576]`；分页布局由 page/page-slot 映射到四维 cache。
非正长度请求返回零 attention。连续版本返回独立的 updated cache；分页 native 版本的
cache 可能被 donation/alias，调用方应始终使用返回的 cache 进入下一步。

连续 signoff 配置的大型逻辑 shape 是：

```yaml
B: 8
H: 16
latent_dim: 512
rope_dim: 64
q_dim: 576
cache_feature_dim: 640
capacity: 16384
dtype: bfloat16
block_size: 1024
sm_scale: 0.07216878364870323
sliding_window: 0
soft_cap: 0.0
```

实际 `mla_decode_single/rerun` 记录使用同一逻辑 shape（B=8、H=16、latent=512、
RoPE=64、容量/长度=16384），但 physical cache 是上面的分页四维布局。不要只看
`mla_decode.yaml` 的 `params.interface: contiguous` 就把这条结果导入成连续 cache；应
同时保存 `mla.export.json`、source snapshot 和 runner 配置，并把 interface 写成
`paged_native`（或项目内约定的等价名称）。

`project2` 也保留了一个两输入的 dense/reference 接口：

```text
q:       [B, H, kv_dim]
kv:      [B, buffer_len, kv_dim]
lkv_dim: static argument
output:  [B, H, padded_lkv_dim]
```

所以导入 Memory 时必须记录 `interface`。建议使用 `contiguous_native`、`paged_native`、`dense_reference` 等明确名称，而不是把三种接口都叫作 `mla_decode`。

## input、output 和 evidence 的区别

这里有三种容易混淆的“输入”：

1. **kernel 运行时 input**：上面的 `q`、`new_kv`、`cache`、`lengths` 四个数组，以及 dtype、shape、layout、scale、长度语义等协议字段。
2. **benchmark input**：shape/config、候选实现、warmup/repetition、随机数据生成方式、设备和软件版本。project2 的数据生成记录是 standard normal 后精确量化为 BF16，seed 为 `0`。
3. **Memory import input**：用于构造记录的 bundle（kernel、algorithm、config、baseline、run、artifact 等）。它不是把四个大数组直接塞进一个 JSON。

性能和正确性文件是 output/evidence：

```text
accuracy.csv
performance.csv
candidate_status.csv
results.json
mla.export.json
*.hlo
*.stablehlo.mlir
profile / device events / console log
source_snapshot/
config YAML 和 source/hash manifest
```

默认完整证据模式下，这些文件应作为 artifact 登记，并在 run 中通过 artifact reference 关联。对原始数组只记录 `shape`、`dtype`、`layout`、`seed`、生成器和 sha256；如果确实要保存数组，将其作为外部 artifact 保存，不要内嵌到 record JSON。默认 shape 的 cache 约有 75M 个元素，直接内嵌会使 record 难以读取和校验。

## project2 里已有结果的边界

`result\\decode\\mla_decode_single\\rerun` 是一个主要结果目录，候选为 `G_T1_X0_F32S`，shape 为 `B=8,H=16,capacity=length=16384`。记录显示 correctness PASS，典型 device median 约 `123.453594 us`，custom call 约 `113.176484 us`，outside custom call 约 `10.215704 us`，logical bandwidth 约 `1225.35 GB/s`，effective MFU 约 `4.0266%`。

该结果还记录了 TPU v6 lite、Python 3.13.15、JAX/JAXLIB 0.8.1、libtpu 0.0.30，以及最大相对 L2 误差约 `0.0019056`、容差 `0.01`。硬件 HBM byte counter 不可用，因此 logical bandwidth 不能当成真实 HBM utilization。

`result\\decode\\result-decode-3\\matrix_performance.csv` 包含多个 shape，例如：

```text
B=1,H=16,capacity=length=4096
B=3,H=16,capacity=length=4096
B=8,H=16,capacity=length=4096
B=8,H=16,capacity=length=16384
```

每个 shape、候选实现和测量批次都应单独建 run；不能把整个 CSV 目录压成一个“成功”记录。`G_T1_X0_F32S`、`M_T1_K2048_B2` 等更适合表示候选/variant；只有在实现语义确实不同且有稳定方法说明时，才拆成不同 algorithm。

## 建议的 Memory 层级

```text
kernel: mla_decode
└── algorithm: contiguous-native / paged-native / dense-reference
    └── config: 固定 shape + interface + dtype + cache layout + protocol
        ├── baseline: G_T1_X0_F32S
        ├── commit/PR: source snapshot 或 Git commit
        ├── run: 每个候选、shape、测量批次
        ├── decision: 候选与 baseline 配对后的 policy 结论
        └── annotation: 人工经验或解释，附 evidence_refs
```

`config.problem` 应表达逻辑契约。例如默认大型 native shape 可以写成：

```json
{
  "operation": "mla_decode",
  "interface": "contiguous_native",
  "batch": 8,
  "num_q_heads": 16,
  "query_tokens_per_request": 1,
  "capacity": 16384,
  "timed_length": 16384,
  "latent_dim": 512,
  "rope_dim": 64,
  "q_dim": 576,
  "cache_feature_dim": 640,
  "dtype": "bfloat16",
  "score_dtype": "float32",
  "probability_dtype": "bfloat16",
  "block_size": 1024,
  "sm_scale": 0.07216878364870323,
  "sliding_window": 0,
  "soft_cap": 0.0,
  "output": ["attention", "updated_cache"],
  "length_semantics": "append_then_attend"
}
```

如果导入的是 paged 变体，还应增加 `page_size=128`、`pages_per_block=8`、KV block、
internal decode batch、是否转置 cache、是否 two-step 等字段。不能把这些字段省略后再用
一个模糊的 `mla_decode` 名称代替。

## run 至少需要的证据

完整证据模式的 run 至少应保存：

1. source repo、commit/tree 或 snapshot hash、entrypoint、candidate 名称；
2. `config_ref`，并明确 shape/interface；
3. TPU、JAX、JAXLIB、libtpu、设备拓扑等环境；
4. timing boundary（是否包含 append、attention、wrapper）、warmup、samples、统计量和 input suite hash；
5. correctness gate、通过数、误差、reference 和 tolerance；
6. device median/p10/p90、custom/outside、bandwidth/MFU 及其计数器可用性；
7. `accuracy.csv`、`performance.csv`、`results.json`、HLO/MLIR、profile、log 等 artifact references。

只有一个历史 median 数字时，不要伪造完整的已验证 run。可以先写成 `annotation` 或 `imported_unverified`，标明缺少的 source、samples、protocol 或 verifier；证据补齐后再建立可比较的 run。

## 从 project2 导入的实际步骤

现有 `scripts/import_project2_mla.py` 可转换单个 project2 result/rerun 目录。默认输出包含附件的 bundle；使用 `--store --summary-only` 可直接生成标准 Memory 目录，仅保留历史结果摘要，并清理临时 bundle。默认完整证据模式的流程是：

1. 从 canonical source/config/result 文件提取上面的逻辑契约；
2. 为每个 shape/interface 建立 `kernel`、`algorithm`、`config` 和 baseline；
3. 把每个候选与测量批次转换成独立 run；
4. 把 CSV、JSON、HLO、profile、source snapshot 等登记为 artifacts；
5. 计算并保存 input suite、source snapshot 和 artifact 的 sha256；
6. 通过 bundle 导入，再执行深度校验和 trajectory 重建；
7. 只有拥有完整 comparison key、samples、correctness 和可信来源时，才运行 compare/decide。

摘要模式显式设置 `evidence_mode=summary_only` 和 `provenance=imported_unverified`，不保留附件或附件引用，但仍校验记录结构和数值约束；它不代表重新执行或独立验证实验。

如果暂时没有完整 schema adapter，仍可导入历史记录，但必须提供符合 record contract 的 kernel/algorithm/config/run 引用链。`imported_unverified` 只表示来源或证据尚未完成验证，不会免除必填字段，也不能补造 samples、digest 或 verifier。信息不足的材料先作为 `confidence=unverified` annotation 保存更安全。

## 重要限制

- 本文只读检查了 `project2` 中的源码、配置、测试和已有结果，没有运行其 TPU 代码。
- project2 约有 73,845 个文件、约 30.39 GB，其中大量是缓存、压缩包、MLIR/HLO、profile 和历史产物；“扫描所有文件”在这里指完成枚举并重点核对 canonical 输入/输出/结果入口，不把每个生成文件都当作独立事实。
- `result` 中的快照通常重复同一长度的 decode 测量，不应自动解释为持续增长的生成序列。
