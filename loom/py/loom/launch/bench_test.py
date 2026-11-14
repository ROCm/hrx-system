# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.launch.bench import (
    SUITES,
    Runner,
    compare,
    format_metrics,
    format_published,
)
from loom.launch.client import Client
from loom.launch.profile import Profile
from loom.launch.testing import FakeModelConfig, FakeOpenAIServer


def profile():
    card = {
        "schema_version": 1,
        "model": {"id": "test/model", "name": "Test model"},
        "services": {"chat": {"kind": "text", "version": 1}},
        "presets": {
            "chat": {
                "service": "chat",
                "execution": {
                    "rows": 1,
                    "context_capacity": 65536,
                    "prefill_capacity": 1024,
                    "max_tokens": 512,
                    "draft_depth": 0,
                },
            }
        },
    }
    measurements = {
        "card": {"memory_bytes": 1000000},
        "performance": {
            "baseline": {"name": "llama.cpp", "version": "b1", "args": "-ngl 99"},
            "sessions": [
                {
                    "date": "2026-11-11",
                    "metrics": {
                        "decode": {"loom": 10.0, "baseline": 5.0},
                        "prefill_1k": {"loom": 200.0},
                    },
                }
            ],
        },
        "roofline": {
            "device": {"name": "gfx1151"},
            "metrics": {"decode": {"ceiling": 100.0}},
        },
    }
    return Profile.from_documents(
        "loom-fun/test-loom",
        card,
        info=measurements,
        llm_config={
            "arch": "test",
            "gguf": "test-Q4_K_M.gguf",
            "context_length": 65536,
        },
    )


def served(config):
    server = FakeOpenAIServer(config=config)
    url = server.start()
    return server, Client(url)


def test_the_quick_suite_measures_every_row_it_names():
    server, client = served(
        FakeModelConfig(completion_tokens=160, first_token_delay=0.001)
    )
    try:
        runner = Runner(client, profile(), warmup=False, thinking=False)
        record = runner.run("quick")
        assert record["ok"] is True
        assert record["suite"] == "quick"
        assert record["harness"]["workload"] == "loom-launch/text-v1"
        measured = record["metrics"]
        assert set(measured) == set(SUITES["quick"])
        for entry in measured.values():
            assert entry["value"] is not None, entry
            assert entry["runs"] == 1
        assert measured["decode"]["completion_tokens"] == 160
        assert measured["prefill_1k"]["prompt_tokens"] > 500
        assert record["comparison"][0]["metric"] == "decode"
        assert record["comparison"][0]["published"] == 10.0
    finally:
        server.stop()


def test_short_completions_are_not_reported_as_decode_measurements():
    server, client = served(FakeModelConfig(completion_tokens=4))
    try:
        runner = Runner(client, profile(), warmup=False)
        result = runner._measure("decode")
        assert result.value is None
        assert "not a decode measurement" in result.note
    finally:
        server.stop()


def test_four_streams_report_an_aggregate_rate():
    server, client = served(
        FakeModelConfig(completion_tokens=32, first_token_delay=0.001)
    )
    try:
        runner = Runner(client, profile(), warmup=False)
        result = runner._measure_streams()
        assert result.metric == "streams4"
        assert result.value is not None
        assert result.value > 0
        assert result.completion_tokens == 32 * 4
        assert len(server.completions()) == 4
    finally:
        server.stop()


def test_comparison_pairs_measured_rows_with_published_ones():
    server, client = served(
        FakeModelConfig(completion_tokens=160, first_token_delay=0.001)
    )
    try:
        runner = Runner(client, profile(), warmup=False)
        results = {name: runner._measure(name) for name in ("decode", "prefill_1k")}
        rows = compare(profile(), results)
        by_metric = {row["metric"]: row for row in rows}
        assert by_metric["decode"]["published"] == 10.0
        assert by_metric["decode"]["ceiling"] == 100.0
        assert by_metric["decode"]["roofline_fraction"] is not None
        assert by_metric["prefill_1k"]["published"] == 200.0
    finally:
        server.stop()


def test_reports_render_measured_and_published_tables():
    server, client = served(
        FakeModelConfig(completion_tokens=160, first_token_delay=0.001)
    )
    try:
        runner = Runner(client, profile(), warmup=False)
        record = runner.run("quick")
        lines = format_metrics(record)
        assert lines[0].startswith("metric")
        assert any(line.startswith("decode") for line in lines)
        assert any(line.startswith("prefill_1k") for line in lines)
        published = format_published(profile())
        assert any(line.startswith("decode") for line in published)
        assert any(
            "no measurements published" in line
            for line in format_published(
                Profile.from_documents("x/y", {"model": {"id": "x/y"}})
            )
        )
    finally:
        server.stop()


def test_an_unknown_suite_is_refused_before_measuring():
    server, client = served(FakeModelConfig())
    try:
        runner = Runner(client, profile(), warmup=False)
        try:
            runner.run("absent")
        except Exception as error:  # BenchmarkError
            assert "unknown suite" in str(error)
        else:
            raise AssertionError("an unknown suite must be refused")
    finally:
        server.stop()
