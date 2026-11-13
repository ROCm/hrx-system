# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import urllib.parse

from loom.launch.client import Client
from loom.launch.proxy import Proxy, Upstream
from loom.launch.store import JsonlWriter
from loom.launch.testing import FakeModelConfig, FakeOpenAIServer


class Instrumented:
    """A fake model behind an instrumented proxy, for one test."""

    def __init__(self, tmp_path, *, token=None, config=None):
        self.model = FakeOpenAIServer(config=config or FakeModelConfig())
        model_url = self.model.start()
        parsed = urllib.parse.urlsplit(model_url)
        self.log_path = tmp_path / "requests.jsonl"
        self.writer = JsonlWriter(self.log_path)
        self.proxy = Proxy(
            Upstream(parsed.hostname, parsed.port),
            request_log=self.writer,
            status_provider=lambda: {"launch": {"repo": "loom-fun/a-loom"}},
            auth_token=token,
        )
        host, port = self.proxy.start()
        self.base_url = f"http://{host}:{port}"

    def close(self):
        self.proxy.stop()
        self.model.stop()
        self.writer.close()

    def records(self):
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def test_a_streamed_turn_passes_through_and_is_recorded(tmp_path):
    instrumented = Instrumented(tmp_path)
    try:
        client = Client(instrumented.base_url)
        result = client.chat(
            [{"role": "user", "content": "hello there friend"}], model="test/model"
        )
        assert result.status == 200
        assert result.text.startswith(" token0")
        assert result.usage.completion_tokens == 8
        assert result.ttft_seconds is not None
        records = instrumented.records()
        assert len(records) == 1
        record = records[0]
        assert record["route"] == "chat_completions"
        assert record["status"] == 200
        assert record["stream"] is True
        assert record["prompt_tokens"] == 3
        assert record["completion_tokens"] == 8
        assert record["chunks"] >= 9
        assert record["response_bytes"] > 0
    finally:
        instrumented.close()


def test_a_json_turn_is_recorded_with_usage_and_latency(tmp_path):
    instrumented = Instrumented(tmp_path)
    try:
        client = Client(instrumented.base_url)
        result = client.chat(
            [{"role": "user", "content": "two words"}], model="test/model", stream=False
        )
        assert result.text.startswith(" token0")
        record = instrumented.records()[0]
        assert record["stream"] is False
        assert record["completion_tokens"] == 8
        assert record["duration_seconds"] is not None
        snapshot = instrumented.proxy.metrics.snapshot()
        assert snapshot["requests"] == 1
        assert snapshot["prompt_tokens"] == 2
    finally:
        instrumented.close()


def test_discovery_routes_are_forwarded_untouched(tmp_path):
    instrumented = Instrumented(tmp_path)
    try:
        client = Client(instrumented.base_url)
        assert client.model_names() == ["test/model"]
        status, document = client.get_json("/v1/status")
        assert status == 200
        assert document["models"]["test/model"]["state"] == "ready"
        record = instrumented.records()[0]
        assert record["route"] == "models"
        assert record["method"] == "GET"
    finally:
        instrumented.close()


def test_launcher_routes_are_served_locally(tmp_path):
    instrumented = Instrumented(tmp_path)
    try:
        client = Client(instrumented.base_url)
        status, document = client.get_json("/loom/status")
        assert status == 200
        assert document["launch"]["repo"] == "loom-fun/a-loom"
        assert document["proxy"]["upstream"]["port"] > 0
        response = client.request("GET", "/loom/metrics")
        assert response[0] == 200
        assert b"loom_launch_requests_total" in response[1]
        response = client.request("GET", "/loom/requests")
        assert response[0] == 200
        assert json.loads(response[1])["object"] == "list"
        assert client.request("GET", "/loom/absent")[0] == 404
        # Launcher routes never reach the model, so nothing is logged as inference.
        assert instrumented.records() == []
    finally:
        instrumented.close()


def test_a_configured_token_is_required_and_accepted(tmp_path):
    instrumented = Instrumented(tmp_path, token="s3cret")
    try:
        anonymous = Client(instrumented.base_url)
        status, document = anonymous.get_json("/v1/models")
        assert status == 401
        assert document["error"]["code"] == "PERMISSION_DENIED"
        authorized = Client(instrumented.base_url, token="s3cret")
        assert authorized.model_names() == ["test/model"]
    finally:
        instrumented.close()


def test_a_failed_upstream_becomes_a_recorded_bad_gateway(tmp_path):
    proxy = Proxy(Upstream("127.0.0.1", 1))
    host, port = proxy.start()
    try:
        client = Client(f"http://{host}:{port}", timeout=5.0)
        status, document = client.get_json("/v1/models")
        assert status == 502
        assert document["error"]["type"] == "loom_launch_proxy_error"
        snapshot = proxy.metrics.snapshot()
        assert snapshot["errors"] == 1
    finally:
        proxy.stop()


def test_model_errors_are_forwarded_with_their_status(tmp_path):
    instrumented = Instrumented(tmp_path, config=FakeModelConfig(fail_status=429))
    try:
        client = Client(instrumented.base_url)
        status, document = client.post_json(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert status == 429
        assert document["error"]["code"] == "FAILED_PRECONDITION"
    finally:
        instrumented.close()


def test_upstream_health_follows_the_requests_that_reached_it(tmp_path):
    instrumented = Instrumented(tmp_path)
    try:
        client = Client(instrumented.base_url, timeout=10.0)
        assert instrumented.proxy.metrics.snapshot()["upstream_up"] is False
        assert client.model_names(timeout=10.0)
        assert instrumented.proxy.metrics.snapshot()["upstream_up"] is True
    finally:
        instrumented.close()
