"""MuKV 的 BioCache 在线流适配器。重依赖只在真正编码时导入。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .common import BUDGET_TIERS, PhaseTimer, TimestampedFrame, describe_tree, object_nbytes


@dataclass
class MuKVConfig:
    segment_frames: int = 32
    encode_chunk_size: int = 64
    budget_tier: str = "25%"
    report_mode: str = "linear-growth"
    fixed_budget_bytes: Optional[int] = None
    strict_budget: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.segment_frames < 1:
            raise ValueError("segment_frames must be positive")
        if self.budget_tier not in BUDGET_TIERS:
            raise ValueError(f"budget_tier must be one of {sorted(BUDGET_TIERS)}")
        if self.report_mode not in {"fixed-budget", "linear-growth"}:
            raise ValueError("report_mode must be fixed-budget or linear-growth")
        if self.report_mode == "fixed-budget" and not self.fixed_budget_bytes:
            raise ValueError("fixed-budget mode requires fixed_budget_bytes")


class MuKVBioCacheAdapter:
    """包装官方 ``encode_video``/``question_answering``，不拥有或预读视频源。"""

    def __init__(self, model: Any = None, config: Optional[MuKVConfig] = None) -> None:
        self.model = model
        self.config = config or MuKVConfig()
        self.timer = PhaseTimer()
        self.stream_id: Optional[str] = None
        self._pending: list[TimestampedFrame] = []
        self._timestamps: list[float] = []
        self._frames_pushed = 0
        self._segments_pushed = 0
        self._answers = 0
        self._opened_at: Optional[float] = None
        self._temporary_peaks: dict[str, int] = {}

    def open_stream(self, stream_id: Optional[str] = None) -> str:
        if self.stream_id is not None:
            raise RuntimeError("stream already open")
        self.stream_id = stream_id or f"mukv-{uuid.uuid4().hex[:12]}"
        self._opened_at = time.time()
        self.reset()
        return self.stream_id

    def reset(self) -> None:
        self._pending.clear()
        self._timestamps.clear()
        self._frames_pushed = self._segments_pushed = self._answers = 0
        self._temporary_peaks.clear()
        self.timer = PhaseTimer()
        if self.model is not None and not self.config.dry_run:
            with self.timer.measure("reset"):
                clear = getattr(self.model, "clear_cache", None)
                if clear is not None:
                    clear()
                init = getattr(self.model, "encode_init_prompt", None)
                if init is not None:
                    init()

    def push_frames(self, frames: Iterable[TimestampedFrame | tuple[float, Any]]) -> dict[str, Any]:
        """消费调用方提供的外部帧，最多仅保留一个 segment，绝不打开视频文件。"""
        self._require_open()
        accepted = 0
        for item in frames:
            with self.timer.measure("ingest"):
                stamped = item if isinstance(item, TimestampedFrame) else TimestampedFrame(float(item[0]), item[1])
                if self._timestamps and stamped.timestamp < self._timestamps[-1]:
                    raise ValueError("frame timestamps must be monotonic")
                self._pending.append(stamped)
                self._timestamps.append(stamped.timestamp)
                self._frames_pushed += 1
                accepted += 1
            if len(self._pending) >= self.config.segment_frames:
                self._flush_segment()
        return {"accepted_frames": accepted, "total_frames": self._frames_pushed,
                "pending_frames": len(self._pending)}

    def _flush_segment(self) -> None:
        if not self._pending:
            return
        segment, self._pending = self._pending, []
        if not self.config.dry_run:
            if self.model is None:
                raise RuntimeError("a MuKV model is required outside dry-run")
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("torch is required to encode MuKV frames") from exc
            payloads = [item.frame for item in segment]
            if all(isinstance(frame, torch.Tensor) for frame in payloads):
                video = torch.stack(payloads)
            else:
                try:
                    import numpy as np
                except ImportError as exc:
                    raise RuntimeError("NumPy is required for non-tensor MuKV frames") from exc
                video = np.stack(payloads)
            self._record_temporary("encoded_segment", video)
            with self.timer.measure("encode"):
                self.model.encode_video(video, encode_chunk_size=self.config.encode_chunk_size)
        self._segments_pushed += 1
        self._check_budget()

    def answer(self, question: str | dict[str, Any], **kwargs: Any) -> Any:
        self._require_open()
        if self._pending:
            self._flush_segment()
        if self.config.dry_run:
            result: Any = {"dry_run": True, "question": question}
        else:
            if self.model is None:
                raise RuntimeError("a MuKV model is required outside dry-run")
            request = question if isinstance(question, dict) else {"question": question}
            self._record_temporary("query_input", request)
            with self.timer.measure("query"):
                result = self.model.question_answering(request, **kwargs)
            self._record_temporary("query_output", result)
        self._answers += 1
        return result

    query = answer

    def close_stream(self) -> dict[str, Any]:
        self._require_open()
        if self._pending:
            self._flush_segment()
        report = self.report()
        if self.model is not None and not self.config.dry_run:
            with self.timer.measure("close"):
                clear = getattr(self.model, "clear_cache", None)
                if clear is not None:
                    clear()
        report["timing"] = self.timer.report()
        self.stream_id = None
        return report

    def _persistent_objects(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        store_obj = getattr(self.model, "multigran_kv_store", None)
        stores = getattr(store_obj, "kv_stores", {}) if store_obj is not None else {}
        # 包括 CPU 历史 KV、代表向量、token 索引和 compression_stats。
        return {
            "cpu_history": {gran: data.get("keys", []) for gran, data in stores.items()},
            "representative_vectors": {gran: data.get("repr_vectors", []) for gran, data in stores.items()},
            "token_indices": {gran: data.get("token_indices", []) for gran, data in stores.items()},
            "compression_stats": {gran: data.get("compression_stats", []) for gran, data in stores.items()},
            "initial_prompt_kv": getattr(self.model, "init_kv_cache", None),
            "legacy_kv_cache": getattr(self.model, "kv_cache", None),
            "external_timestamps": self._timestamps,
        }

    def _static_objects(self) -> dict[str, Any]:
        if self.model is None:
            return {"config": self.config}
        parameters = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
        return {
            "model_parameters": parameters,
            "model_config": getattr(self.model, "config", None),
            "processor": getattr(self.model, "processor", None),
            "adapter_config": self.config,
        }

    def state_report(self) -> dict[str, Any]:
        persistent = self._persistent_objects()
        static = self._static_objects()
        temporary = {"pending_external_frames": self._pending,
                     "observed_peak_bytes": self._temporary_peaks}
        persistent_bytes = object_nbytes(persistent)
        cap = self.config.fixed_budget_bytes if self.config.report_mode == "fixed-budget" else None
        return {
            "persistent": {"bytes": persistent_bytes, "objects": describe_tree(persistent, "persistent")},
            "static": {"bytes": object_nbytes(static), "objects": describe_tree(static, "static")},
            "temporary": {"bytes": object_nbytes(temporary), "objects": describe_tree(temporary, "temporary")},
            "budget": {
                "tier": self.config.budget_tier,
                "ratio": BUDGET_TIERS[self.config.budget_tier]["ratio"],
                "report_mode": self.config.report_mode,
                "fixed_budget_bytes": cap,
                "within_budget": None if cap is None else persistent_bytes <= cap,
                "bytes_per_frame": persistent_bytes / max(self._frames_pushed, 1),
            },
        }

    def report(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "opened_at": self._opened_at,
            "frames": self._frames_pushed,
            "segments": self._segments_pushed,
            "answers": self._answers,
            "timing": self.timer.report(),
            "state": self.state_report(),
        }

    def _check_budget(self) -> None:
        if self.config.report_mode != "fixed-budget":
            return
        budget = self.state_report()["budget"]
        if self.config.strict_budget and not budget["within_budget"]:
            raise MemoryError(
                f"MuKV persistent state exceeds fixed budget: "
                f"{self.state_report()['persistent']['bytes']} > {self.config.fixed_budget_bytes}"
            )

    def _require_open(self) -> None:
        if self.stream_id is None:
            raise RuntimeError("call open_stream() first")

    def _record_temporary(self, name: str, value: Any) -> None:
        self._temporary_peaks[name] = max(
            self._temporary_peaks.get(name, 0), object_nbytes(value)
        )
