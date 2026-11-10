"""无模型 dry-run 与 benchmark schema 检查入口。"""

from __future__ import annotations

import argparse
import json
import os

from .adapter import MuKVBioCacheAdapter, MuKVConfig
from .common import BUDGET_TIERS, TimestampedFrame, iter_benchmark, write_jsonl, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuKV BioCache streaming adapter")
    parser.add_argument("--dry-run", action="store_true", help="不加载模型且不执行推理")
    parser.add_argument("--stream-id", default="dry-run")
    parser.add_argument("--segment-frames", type=int, default=32)
    parser.add_argument("--budget-tier", choices=sorted(BUDGET_TIERS), default="25%")
    parser.add_argument("--report-mode", choices=["fixed-budget", "linear-growth"], default="linear-growth")
    parser.add_argument("--fixed-budget-bytes", type=int)
    parser.add_argument("--benchmark", choices=["OVO", "StreamingBench", "TwinPast", "VNBench", "EgoLifeQA"])
    parser.add_argument("--annotations", help="本地 JSON/JSONL；适配器不会下载数据")
    parser.add_argument("--run-dir", default="runs/biocache_dry_run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit("CLI 当前仅负责 --dry-run；真实运行请在宿主中注入已加载的 MuKV model")
    if bool(args.benchmark) != bool(args.annotations):
        raise SystemExit("--benchmark 与 --annotations 必须同时提供")
    config = MuKVConfig(
        segment_frames=args.segment_frames,
        budget_tier=args.budget_tier,
        report_mode=args.report_mode,
        fixed_budget_bytes=args.fixed_budget_bytes,
        dry_run=True,
    )
    manifest = write_manifest(args.run_dir, vars(args), os.path.dirname(os.path.dirname(__file__)))
    adapter = MuKVBioCacheAdapter(config=config)
    adapter.open_stream(args.stream_id)
    adapter.push_frames(TimestampedFrame(float(i), {"synthetic_frame": i}) for i in range(3))
    adapter.answer("dry-run question")
    report = adapter.close_stream()
    output = os.path.join(args.run_dir, "metrics.jsonl")
    write_jsonl(output, [{"event": "dry_run", **report}])
    schema_count = 0
    if args.annotations:
        samples = []
        for sample in iter_benchmark(args.annotations, args.benchmark):
            schema_count += 1
            if len(samples) < 3:
                samples.append(sample)
        with open(os.path.join(args.run_dir, "schema_preview.json"), "w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"manifest": manifest, "metrics": output, "schema_records": schema_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
