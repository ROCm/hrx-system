# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Local benchmarks with the metric vocabulary a Loom card publishes.

The rows mirror the publisher's measurement record: decode at a short prompt and
at 2k and 32k context, prefill over 1k through 32k documents, four concurrent
streams and draft acceptance when the server reports it. Each row records the
samples it came from, so a stored run can be read the way a card is read: this
engine against the baseline it was measured with, and against the hardware
ceiling.
"""

from __future__ import annotations

import concurrent.futures
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loom.launch.client import ChatResult, Client
from loom.launch.errors import BenchmarkError
from loom.launch.profile import METRIC_LABELS, METRIC_ORDER, Profile
from loom.launch.store import BenchmarkStore, new_run_id
from loom.launch.values import number_value, str_value
from loom.launch.workload import (
    WORKLOAD_IDENTITY,
    decode_prompt,
    prefill_prompt,
)

#: Metric sets a caller can request, cheapest first.
SUITES: dict[str, tuple[str, ...]] = {
    "quick": ("decode", "prefill_1k", "prefill_2k"),
    "standard": (
        "decode",
        "decode_2k",
        "prefill_1k",
        "prefill_2k",
        "prefill_8k",
        "streams4",
    ),
    "full": (
        "decode",
        "decode_2k",
        "decode_32k",
        "prefill_1k",
        "prefill_2k",
        "prefill_8k",
        "prefill_16k",
        "prefill_32k",
        "streams4",
    ),
}

#: Output budget and document size each metric measures.
_METRIC_SHAPE = {
    "decode": (0, 300),
    "decode_2k": (2000, 256),
    "decode_32k": (32000, 128),
    "prefill_1k": (1000, 4),
    "prefill_2k": (2000, 4),
    "prefill_8k": (8000, 4),
    "prefill_16k": (16000, 4),
    "prefill_32k": (32000, 4),
}


@dataclass
class MetricResult:
    """One measured row, with its samples and the protocol that produced it."""

    metric: str
    value: float | None = None
    unit: str = "tok/s"
    samples: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_seconds: float | None = None
    duration_seconds: float | None = None
    note: str | None = None

    def describe(self) -> dict:
        return {
            "metric": self.metric,
            "label": METRIC_LABELS.get(self.metric, self.metric),
            "value": _round(self.value),
            "unit": self.unit,
            "samples": [_round(sample) for sample in self.samples],
            "runs": len(self.samples),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": _round(self.ttft_seconds),
            "duration_seconds": _round(self.duration_seconds),
            "note": self.note,
        }


class Runner:
    """Measures one suite against a live endpoint."""

    def __init__(
        self,
        client: Client,
        profile: Profile,
        *,
        model: str | None = None,
        repeats: int = 1,
        thinking: bool | None = False,
        on_progress: Callable[[str], None] | None = None,
        warmup: bool = True,
        session: str | None = None,
        max_sessions: int | None = None,
    ):
        self.client = client
        self.profile = profile
        self.model = model or _default_model(profile)
        self.repeats = max(1, repeats)
        self.thinking = thinking
        self.on_progress = on_progress
        self.warmup = warmup
        self.session = session
        self.max_sessions = max_sessions

    # -- helpers ------------------------------------------------------------
    def _notify(self, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(message)

    def _headers(self) -> dict[str, str] | None:
        if self.session:
            return {"X-Loom-Session": self.session}
        return None

    def _chat(self, content: str, max_tokens: int) -> ChatResult:
        fields: dict = {"max_tokens": max_tokens, "temperature": 0}
        if self.thinking is not None:
            fields["enable_thinking"] = self.thinking
        return self.client.chat(
            [{"role": "user", "content": content}],
            model=self.model,
            stream=True,
            headers=self._headers(),
            **fields,
        )

    # -- measurement --------------------------------------------------------
    def run(self, suite: str = "standard") -> dict:
        metrics = SUITES.get(suite)
        if metrics is None:
            raise BenchmarkError(
                f"unknown suite {suite!r}; choose one of {', '.join(SUITES)}"
            )
        started_at = time.time()
        started = time.monotonic()
        if self.warmup:
            self._notify("warm-up")
            self._chat(decode_prompt(0), 32)
        results: dict[str, MetricResult] = {}
        for metric in metrics:
            results[metric] = self._measure(metric)
        elapsed = time.monotonic() - started
        record = {
            "id": new_run_id(),
            "kind": "loom-launch-benchmark",
            "suite": suite,
            "profile": self.profile.repo,
            "revision": self.profile.revision,
            "model": self.model,
            "started_at": _isoformat(started_at),
            "elapsed_seconds": round(elapsed, 3),
            "harness": {
                "workload": WORKLOAD_IDENTITY,
                "repeats": self.repeats,
                "thinking": self.thinking,
                "warmup": self.warmup,
            },
            "settings": self.profile.execution.describe()
            if self.profile.execution
            else None,
            "device": self.profile.device_hint,
            "metrics": {name: result.describe() for name, result in results.items()},
            "comparison": compare(self.profile, results),
            "ok": any(result.value is not None for result in results.values()),
        }
        return record

    def _measure(self, metric: str) -> MetricResult:
        if metric == "streams4":
            return self._measure_streams()
        if metric == "spec_acceptance":
            return MetricResult(
                metric=metric,
                unit="fraction",
                note="the server does not report drafted-token acceptance here",
            )
        shape = _METRIC_SHAPE.get(metric)
        if shape is None:
            return MetricResult(metric=metric, note="unknown metric")
        document_tokens, max_tokens = shape
        result = MetricResult(metric=metric)
        prefill = metric.startswith("prefill")
        for repeat in range(self.repeats):
            if prefill:
                completion = self._chat(prefill_prompt(document_tokens), max_tokens)
                prompt_tokens = _usage_prompt(completion)
                ttft = completion.ttft_seconds
                if not prompt_tokens or not ttft:
                    result.note = completion.error or "no prompt token count reported"
                    continue
                rate = prompt_tokens / ttft
                result.prompt_tokens = prompt_tokens
                result.ttft_seconds = ttft
            else:
                completion = self._chat(decode_prompt(document_tokens), max_tokens)
                rate = completion.decode_tokens_per_second
                completed = _usage_completion(completion)
                result.prompt_tokens = _usage_prompt(completion)
                result.completion_tokens = completed
                result.ttft_seconds = completion.ttft_seconds
                result.duration_seconds = completion.duration_seconds
                if completion.error:
                    result.note = completion.error
                    continue
                if completed is not None and completed < max_tokens // 2:
                    result.note = (
                        f"stopped after {completed} of {max_tokens} tokens; "
                        "not a decode measurement"
                    )
                    continue
            if rate is None or rate <= 0:
                result.note = result.note or "no usable measurement"
                continue
            result.samples.append(rate)
            self._notify(
                f"{metric}: {rate:,.1f} tok/s (repeat {repeat + 1}/{self.repeats})"
            )
        if result.samples:
            result.value = statistics.median(result.samples)
        return result

    def _measure_streams(self, streams: int = 4, max_tokens: int = 128) -> MetricResult:
        result = MetricResult(metric="streams4")
        tokens: list[int] = []
        if self.max_sessions is not None and self.max_sessions < streams:
            # The row is measured, but it answers a narrower question than the
            # published one: four clients against fewer resident sessions queue.
            result.note = (
                f"deployment admits {self.max_sessions} session(s); the published "
                "row assumes four concurrent streams"
            )
        window_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
            futures = [
                pool.submit(self._chat, decode_prompt(0), max_tokens)
                for _ in range(streams)
            ]
            completions = [future.result() for future in futures]
        window = time.monotonic() - window_start
        for completion in completions:
            completed = _usage_completion(completion)
            if completion.error:
                result.note = completion.error
                continue
            if completed:
                tokens.append(completed)
        if not tokens or window <= 0:
            result.note = result.note or "no usable measurements"
            return result
        # Aggregate decode: every stream's decoded tokens over the shared window.
        aggregate = sum(max(count - 1, 1) for count in tokens) / window
        result.samples.append(aggregate)
        result.value = aggregate
        result.completion_tokens = sum(tokens)
        result.duration_seconds = window
        self._notify(f"streams4: {aggregate:,.1f} tok/s aggregate over {len(tokens)}")
        return result


def _default_model(profile: Profile) -> str:
    return profile.model_id


def _usage_prompt(result: ChatResult) -> int | None:
    return result.usage.prompt_tokens if result.usage else None


def _usage_completion(result: ChatResult) -> int | None:
    return result.usage.completion_tokens if result.usage else None


def compare(profile: Profile, results: dict[str, MetricResult]) -> list[dict]:
    """Compares measured rows with the card's published rows and ceiling."""
    rows: list[dict] = []
    published = profile.performance.by_metric() if profile.performance else {}
    for metric in METRIC_ORDER:
        result = results.get(metric)
        if result is None or result.value is None:
            continue
        reference = published.get(metric)
        published_value = reference.value if reference else None
        ceiling = reference.ceiling if reference else None
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "measured": _round(result.value),
                "published": _round(published_value),
                "ratio": _round(result.value / published_value)
                if published_value
                else None,
                "ceiling": _round(ceiling),
                "roofline_fraction": _round(result.value / ceiling)
                if ceiling
                else None,
            }
        )
    return rows


def format_metrics(record: dict) -> list[str]:
    """Renders one stored run as a table a terminal can show."""
    lines = [
        f"{'metric':<22} {'value':>10} {'published':>10} {'ratio':>7} {'roofline':>9}"
    ]
    lines.append("-" * 62)
    comparison = {
        str_value(row.get("metric")): row
        for row in record.get("comparison") or []
        if isinstance(row, dict)
    }
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return lines
    for metric in METRIC_ORDER:
        entry = metrics.get(metric)
        if not isinstance(entry, dict):
            continue
        value = number_value(entry.get("value"))
        if value is None:
            continue
        reference = comparison.get(metric, {})
        published = number_value(reference.get("published"))
        ratio = number_value(reference.get("ratio"))
        fraction = number_value(reference.get("roofline_fraction"))
        lines.append(
            f"{metric:<22} {value:>10,.1f} "
            f"{(f'{published:,.1f}' if published else '-'):>10} "
            f"{(f'{ratio:.2f}x' if ratio else '-'):>7} "
            f"{(f'{fraction * 100:.0f}%' if fraction else '-'):>9}"
        )
    return lines


def format_published(profile: Profile) -> list[str]:
    """Renders a card's stored measurements, with the baseline and ceiling."""
    lines = [
        f"{'metric':<22} {'loom':>10} {'baseline':>10} {'ratio':>7} {'roofline':>9}"
    ]
    lines.append("-" * 62)
    if profile.performance is None:
        return lines + ["no measurements published"]
    for measurement in profile.performance.measurements:
        if measurement.value is None:
            continue
        ratio = measurement.baseline_ratio
        fraction = measurement.roofline_fraction
        lines.append(
            f"{measurement.metric:<22} {measurement.value:>10,.1f} "
            f"{(f'{measurement.baseline:,.1f}' if measurement.baseline else '-'):>10} "
            f"{(f'{ratio:.2f}x' if ratio else '-'):>7} "
            f"{(f'{fraction * 100:.0f}%' if fraction else '-'):>9}"
        )
    return lines


def store_record(store: BenchmarkStore, record: dict) -> str:
    path = store.save(record)
    return str(path)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _isoformat(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
