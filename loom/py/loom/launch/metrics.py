# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Request instrumentation recorded by the launcher's fronting proxy.

One record describes one HTTP request the proxy carried: its route, status,
token counts, time to first token and decode rate. Records feed a bounded
in-memory ring for percentiles and a JSONL log for later analysis, and the
counters are exposed as Prometheus text so an ordinary scrape finds them.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

#: Local endpoints the proxy answers itself.
LAUNCHER_PREFIX = "/loom/"
#: How many recent records feed percentile estimates.
DEFAULT_WINDOW = 1024


def route_name(path: str) -> str:
    """Groups request paths into stable route labels for metrics."""
    path = path.split("?", 1)[0]
    if path.startswith(LAUNCHER_PREFIX):
        return "launcher"
    known = {
        "/v1/chat/completions": "chat_completions",
        "/v1/completions": "completions",
        "/v1/embeddings": "embeddings",
        "/v1/models": "models",
        "/v1/status": "status",
        "/v1/requests": "requests",
        "/v1/sessions": "sessions",
        "/v1/observe": "observe",
        "/metrics": "metrics",
        "/logz": "logz",
        "/tokenize": "tokenize",
        "/detokenize": "detokenize",
        "/healthz": "health",
        "/readyz": "health",
    }
    if path in known:
        return known[path]
    if path.startswith("/v1/models/"):
        return "model_control"
    if path.startswith("/v1/checkpoints/"):
        return "checkpoints"
    if path.startswith("/dashboard"):
        return "dashboard"
    return "other"


@dataclass(frozen=True)
class RequestRecord:
    """One measured request."""

    request_id: str
    method: str
    path: str
    route: str
    started_at: float
    status: int = 0
    stream: bool = False
    model: str | None = None
    ttft_seconds: float | None = None
    duration_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    chunks: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    client: str | None = None
    error: str | None = None

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Rate over the decode interval after the first token arrived."""
        if not self.completion_tokens or self.completion_tokens < 2:
            return None
        if self.duration_seconds is None or self.ttft_seconds is None:
            return None
        interval = self.duration_seconds - self.ttft_seconds
        if interval <= 0:
            return None
        return (self.completion_tokens - 1) / interval

    def describe(self) -> dict:
        return {
            "id": self.request_id,
            "time": _isoformat(self.started_at),
            "method": self.method,
            "path": self.path,
            "route": self.route,
            "model": self.model,
            "status": self.status,
            "stream": self.stream,
            "duration_seconds": _round(self.duration_seconds),
            "ttft_seconds": _round(self.ttft_seconds),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "decode_tokens_per_second": _round(self.decode_tokens_per_second),
            "chunks": self.chunks,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "client": self.client,
            "error": self.error,
        }


@dataclass
class _Window:
    """Bounded recent observations for percentile estimates."""

    values: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW))

    def add(self, value: float | None) -> None:
        if value is not None and not math.isnan(value):
            self.values.append(value)

    def quantile(self, fraction: float) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def mean(self) -> float | None:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)


class Metrics:
    """Cumulative counters, a recent-request window and Prometheus rendering."""

    def __init__(self, *, window: int = DEFAULT_WINDOW):
        self._lock = threading.Lock()
        self._counts: Counter = Counter()
        self._requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_tokens = 0
        self._request_bytes = 0
        self._response_bytes = 0
        self._errors = 0
        self._started_at = time.time()
        self._in_flight = 0
        self._upstream_up = False
        self._ttft = _Window(deque(maxlen=window))
        self._duration = _Window(deque(maxlen=window))
        self._decode = _Window(deque(maxlen=window))
        self._recent: deque = deque(maxlen=window)

    # -- accounting ---------------------------------------------------------
    def begin(self) -> None:
        with self._lock:
            self._in_flight += 1

    def end(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def set_upstream_up(self, up: bool) -> None:
        with self._lock:
            self._upstream_up = up

    def observe(self, record: RequestRecord) -> None:
        key = (record.route, str(record.status), "stream" if record.stream else "json")
        with self._lock:
            self._requests += 1
            self._counts[key] += 1
            self._prompt_tokens += record.prompt_tokens or 0
            self._completion_tokens += record.completion_tokens or 0
            self._cached_tokens += record.cached_tokens or 0
            self._request_bytes += record.request_bytes
            self._response_bytes += record.response_bytes
            if record.error or record.status >= 500:
                self._errors += 1
            self._ttft.add(record.ttft_seconds)
            self._duration.add(record.duration_seconds)
            self._decode.add(record.decode_tokens_per_second)
            self._recent.append(record)

    # -- views --------------------------------------------------------------
    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            records = list(self._recent)[-limit:]
        return [record.describe() for record in reversed(records)]

    def snapshot(self) -> dict:
        with self._lock:
            counts = {
                "|".join(key): value for key, value in sorted(self._counts.items())
            }
            return {
                "uptime_seconds": round(time.time() - self._started_at, 3),
                "requests": self._requests,
                "errors": self._errors,
                "in_flight": self._in_flight,
                "upstream_up": self._upstream_up,
                "by_route": counts,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_tokens": self._cached_tokens,
                "request_bytes": self._request_bytes,
                "response_bytes": self._response_bytes,
                "ttft_seconds": {
                    "p50": _round(self._ttft.quantile(0.5)),
                    "p95": _round(self._ttft.quantile(0.95)),
                },
                "duration_seconds": {
                    "p50": _round(self._duration.quantile(0.5)),
                    "p95": _round(self._duration.quantile(0.95)),
                },
                "decode_tokens_per_second": {
                    "mean": _round(self._decode.mean()),
                    "p50": _round(self._decode.quantile(0.5)),
                },
            }

    def prometheus(self) -> str:
        """Renders the current counters in the Prometheus text format."""
        lines: list[str] = []
        lines.append(
            "# HELP loom_launch_requests_total Requests carried by the launcher"
        )
        lines.append("# TYPE loom_launch_requests_total counter")
        with self._lock:
            counts = dict(self._counts)
            totals = {
                "requests": self._requests,
                "errors": self._errors,
                "in_flight": self._in_flight,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_tokens": self._cached_tokens,
                "request_bytes": self._request_bytes,
                "response_bytes": self._response_bytes,
                "upstream_up": 1 if self._upstream_up else 0,
                "uptime_seconds": time.time() - self._started_at,
            }
            ttft = {
                "0.5": self._ttft.quantile(0.5),
                "0.95": self._ttft.quantile(0.95),
            }
            decode = {"0.5": self._decode.quantile(0.5)}
            decode_mean = self._decode.mean()
        for (route, status, stream), count in sorted(counts.items()):
            lines.append(
                f'loom_launch_requests_total{{route="{route}",status="{status}",'
                f'stream="{stream}"}} {count}'
            )
        for name, value in (
            ("loom_launch_errors_total", totals["errors"]),
            ("loom_launch_in_flight", totals["in_flight"]),
            ("loom_launch_upstream_up", totals["upstream_up"]),
            ("loom_launch_prompt_tokens_total", totals["prompt_tokens"]),
            ("loom_launch_completion_tokens_total", totals["completion_tokens"]),
            ("loom_launch_cached_tokens_total", totals["cached_tokens"]),
            ("loom_launch_request_bytes_total", totals["request_bytes"]),
            ("loom_launch_response_bytes_total", totals["response_bytes"]),
            ("loom_launch_uptime_seconds", totals["uptime_seconds"]),
        ):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        lines.append("# TYPE loom_launch_ttft_seconds gauge")
        for quantile, value in ttft.items():
            if value is not None:
                lines.append(
                    f'loom_launch_ttft_seconds{{quantile="{quantile}"}} {value:.6f}'
                )
        lines.append("# TYPE loom_launch_decode_tokens_per_second gauge")
        if decode_mean is not None:
            lines.append(
                "loom_launch_decode_tokens_per_second"
                f'{{quantile="mean"}} {decode_mean:.6f}'
            )
        for quantile, value in decode.items():
            if value is not None:
                lines.append(
                    "loom_launch_decode_tokens_per_second"
                    f'{{quantile="{quantile}"}} {value:.6f}'
                )
        return "\n".join(lines) + "\n"


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _isoformat(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp)) + (
        f".{int((timestamp % 1) * 1000):03d}Z"
    )
