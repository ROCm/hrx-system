# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.launch.metrics import Metrics, RequestRecord, route_name


def record(**overrides):
    arguments = {
        "request_id": "r1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "route": "chat_completions",
        "started_at": 0.0,
        "status": 200,
        "stream": True,
        "model": "test/model",
        "ttft_seconds": 0.2,
        "duration_seconds": 1.2,
        "prompt_tokens": 40,
        "completion_tokens": 11,
        "cached_tokens": 0,
        "chunks": 12,
        "request_bytes": 100,
        "response_bytes": 900,
        "client": "127.0.0.1:5000",
    }
    arguments.update(overrides)
    return RequestRecord(**arguments)


def test_routes_group_known_paths_and_keep_unknown_ones():
    assert route_name("/v1/chat/completions") == "chat_completions"
    assert route_name("/v1/models?full=true") == "models"
    assert route_name("/v1/models/variants") == "model_control"
    assert route_name("/v1/checkpoints/scout") == "checkpoints"
    assert route_name("/loom/metrics") == "launcher"
    assert route_name("/healthz") == "health"
    assert route_name("/something/else") == "other"


def test_decode_rate_uses_the_interval_after_the_first_token():
    assert record().decode_tokens_per_second == 10.0
    assert record(completion_tokens=1).decode_tokens_per_second is None
    assert record(ttft_seconds=None).decode_tokens_per_second is None
    assert record(ttft_seconds=1.2).decode_tokens_per_second is None


def test_records_describe_themselves_with_iso_times():
    document = record().describe()
    assert document["id"] == "r1"
    assert document["time"].endswith("Z")
    assert document["decode_tokens_per_second"] == 10.0
    assert document["stream"] is True


def test_counters_accumulate_over_observations():
    metrics = Metrics()
    metrics.observe(record())
    metrics.observe(record(request_id="r2", status=500, route="other"))
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["errors"] == 1
    assert snapshot["prompt_tokens"] == 80
    assert snapshot["completion_tokens"] == 22
    assert snapshot["by_route"]["chat_completions|200|stream"] == 1
    assert snapshot["ttft_seconds"]["p50"] == 0.2


def test_in_flight_and_upstream_health_are_gauges():
    metrics = Metrics()
    metrics.begin()
    metrics.begin()
    metrics.set_upstream_up(True)
    assert metrics.snapshot()["in_flight"] == 2
    metrics.end()
    assert metrics.snapshot()["in_flight"] == 1
    assert metrics.snapshot()["upstream_up"] is True


def test_prometheus_text_reports_counters_and_quantiles():
    metrics = Metrics()
    metrics.observe(record())
    metrics.set_upstream_up(True)
    text = metrics.prometheus()
    assert (
        'loom_launch_requests_total{route="chat_completions",status="200",stream="stream"} 1'
        in text
    )
    assert "loom_launch_completion_tokens_total 11" in text
    assert 'loom_launch_ttft_seconds{quantile="0.5"} 0.200000' in text
    assert 'loom_launch_decode_tokens_per_second{quantile="mean"} 10.000000' in text
    assert "loom_launch_upstream_up 1" in text
    assert text.endswith("\n")


def test_recent_records_are_newest_first_and_bounded():
    metrics = Metrics(window=2)
    for index in range(4):
        metrics.observe(record(request_id=f"r{index}"))
    ids = [entry["id"] for entry in metrics.recent()]
    assert ids == ["r3", "r2"]


def test_an_empty_metrics_registry_renders_a_valid_document():
    text = Metrics().prometheus()
    assert "loom_launch_requests_total" in text
    assert "quantile" not in text
