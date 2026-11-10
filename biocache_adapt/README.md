# MuKV BioCache streaming adapter

该包是 MuKV 的独立在线评测适配层，不修改官方推理路径。调用方必须逐段提供已经解码并带时间戳的外部帧；适配器不接收视频路径、不打开视频，也不会预读完整视频。

## 契约

- `open_stream(stream_id)` / `reset()`：建立或重置单流状态，并调用官方 cache 初始化。
- `push_frames(TimestampedFrame(...))`：仅缓存至多 `segment_frames` 个外部帧，然后调用官方 `encode_video`；多粒度 KV 路径保持不变。
- `answer()` / `query()`：刷新最后一段后调用官方 `question_answering`。
- `close_stream()`：返回报告并清理官方 cache。
- `state_report()`：将状态分为 `persistent`、`static`、`temporary`，递归按 tensor storage/ndarray nbytes 计算真实字节；persistent 完整列出 CPU 历史 KV、representative vectors、token indices、compression stats、初始化 KV。
- 分阶段计时包括 ingest、encode、query、reset、close。

`linear-growth` 模式报告累计 persistent bytes 与 bytes/frame，不声称内存有界。`fixed-budget` 模式要求显式 `fixed_budget_bytes`，报告是否越界；开启 `strict_budget` 时越界立即报错，但不会在适配层静默删除 MuKV 历史。预算档为 `12.5%`、`25%`、`50%`，是跨方法报告标签，不会暗改官方粒度或保留率。

## Dry-run

```bash
python -m biocache_adapt.cli --dry-run --report-mode linear-growth
python -m biocache_adapt.cli --dry-run --report-mode fixed-budget \
  --fixed-budget-bytes 1073741824 --budget-tier 25%
```

可追加 `--benchmark OVO --annotations /local/path/data.jsonl` 检查统一 schema。支持 OVO、StreamingBench、TwinPast、VNBench、EgoLifeQA 的本地 JSON/JSONL；不会下载数据。每次运行写 `manifest.json`、`metrics.jsonl` 和可选 schema preview。PyTorch、NumPy 与 MuKV 模型依赖均在实际编码时延迟使用。
