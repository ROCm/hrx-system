# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path

from loom.launch.client import Client
from loom.launch.deployment import context_settings
from loom.launch.profile import Profile
from loom.launch.server import (
    LaunchOptions,
    _entry_state,
    _error_text,
    _parse_event,
    _status_entries,
    clear_current,
    describe_state,
    plan_launch,
    read_current,
    running_endpoint,
    write_current,
)
from loom.launch.store import RunStore
from loom.launch.testing import FakeOpenAIServer


def profile():
    card = {
        "schema_version": 1,
        "model": {"id": "test/model", "name": "Test"},
        "services": {"chat": {"kind": "text", "version": 1}},
        "presets": {
            "chat": {
                "service": "chat",
                "execution": {
                    "rows": 2,
                    "context_capacity": 8192,
                    "prefill_capacity": 256,
                    "max_tokens": 512,
                    "draft_depth": 0,
                },
            }
        },
    }
    return Profile.from_documents("loom-fun/test-loom", card)


def planned(tmp_path, **option_overrides):
    run = RunStore(tmp_path).create("run-1")
    card = tmp_path / "model.json"
    card.write_text(json.dumps({"model": {"id": "test/model"}}), encoding="utf-8")
    parsed = profile()
    options = {"loom_serve": "/bin/true", "port": 9100, "device": "amdgpu://"}
    options.update(option_overrides)
    plan = plan_launch(
        parsed,
        card,
        context_settings(parsed, context_length=4096),
        run,
        LaunchOptions(**options),
    )
    return plan, run


def test_a_plan_writes_the_deployment_and_builds_the_command(tmp_path):
    plan, run = planned(tmp_path)
    assert plan.command[0] == "/bin/true"
    assert f"--model={run.deployment_path}" in plan.command
    assert "--listen=127.0.0.1:9100" in plan.command
    assert "--device=amdgpu://" in plan.command
    document = json.loads(run.deployment_path.read_text(encoding="utf-8"))
    # The server resolves model.path against the deployment document's own
    # directory, so it is always written absolute.
    assert Path(document["model"]["path"]).is_absolute()
    assert document["name"] == "loom-fun-test-loom"
    assert document["model"]["preset"] == "chat"
    assert document["definition"]["context_capacity"] == 4096
    assert document["definition"]["max_tokens"] == 512
    assert plan.settings.context_length == 4096


def test_launch_options_add_server_resources(tmp_path):
    plan, _ = planned(
        tmp_path,
        workers=4,
        pool_backing="elastic",
        memory_bytes=1024,
        extra_arguments=("--wait_mode=blocking",),
    )
    assert "--workers=4" in plan.command
    assert "--pool_backing=elastic" in plan.command
    assert "--memory_bytes=1024" in plan.command
    assert "--wait_mode=blocking" in plan.command


def test_the_card_memory_budget_is_used_when_none_is_chosen(tmp_path):
    run = RunStore(tmp_path).create("run-1")
    card = tmp_path / "model.json"
    card.write_text(json.dumps({"model": {"id": "test/model"}}), encoding="utf-8")
    parsed = Profile.from_documents(
        "loom-fun/test-loom",
        {"model": {"id": "test/model"}},
        info={"card": {"memory_bytes": 49000000000}},
    )
    plan = plan_launch(
        parsed,
        card,
        context_settings(parsed),
        run,
        LaunchOptions(loom_serve="/bin/true"),
    )
    assert "--memory_bytes=49000000000" in plan.command


def test_the_default_device_is_detected_when_none_is_given(tmp_path):
    plan, _ = planned(tmp_path, device=None)
    assert plan.device_uri in ("amdgpu://", "task")


def test_server_events_are_parsed_only_when_they_are_json_objects():
    assert _parse_event('{"event":"ready","address":"127.0.0.1:8080"}') == {
        "event": "ready",
        "address": "127.0.0.1:8080",
    }
    assert _parse_event("plain output") is None
    assert _parse_event("[1,2]") is None
    assert _parse_event("{not json}") is None


def test_status_entries_cover_every_shape_the_server_reports():
    document = {
        "models": {"scout": {"state": "ready"}},
        "data": [{"name": "other", "status": "Failed"}],
    }
    entries = _status_entries(document)
    names = sorted(str(entry.get("name")) for entry in entries)
    assert names == ["other", "scout"]
    assert _entry_state({"registration": "READY"}) == "ready"
    assert _entry_state({}) is None


def test_the_current_record_round_trips_and_reports_state(tmp_path):
    assert read_current(tmp_path) is None
    write_current(
        tmp_path,
        {"pid": os.getpid(), "endpoint": "http://127.0.0.1:1", "repo": "a/b"},
    )
    record = read_current(tmp_path)
    assert record["repo"] == "a/b"
    assert describe_state(tmp_path)["state_directory"] == str(tmp_path)
    clear_current(tmp_path)
    assert read_current(tmp_path) is None


def test_a_running_endpoint_is_only_reported_while_it_answers(tmp_path):
    server = FakeOpenAIServer()
    url = server.start()
    try:
        write_current(tmp_path, {"pid": os.getpid(), "endpoint": url, "repo": "a/b"})
        found = running_endpoint(tmp_path)
        assert found is not None
        assert found[0] == url
        assert found[1]["repo"] == "a/b"
        # A dead process invalidates the record even if its port still answers.
        write_current(tmp_path, {"pid": 999999999, "endpoint": url, "repo": "a/b"})
        assert running_endpoint(tmp_path) is None
        # A live process with an unreachable endpoint is not reported either.
        write_current(
            tmp_path,
            {"pid": os.getpid(), "endpoint": "http://127.0.0.1:1", "repo": "a/b"},
        )
        assert running_endpoint(tmp_path) is None
    finally:
        server.stop()


def test_the_client_sees_a_prepared_deployment(tmp_path):
    server = FakeOpenAIServer()
    url = server.start()
    try:
        client = Client(url)
        assert client.healthy()
        assert client.model_names() == ["test/model"]
        assert client.server_status()["models"]["test/model"]["state"] == "ready"
    finally:
        server.stop()


def test_a_nested_registration_failure_is_visible_to_the_launcher():
    # The server reports registration under a nested loom object, and a wait that
    # cannot see it stalls until the startup timeout instead of failing fast.
    document = {
        "object": "list",
        "data": [
            {
                "object": "model",
                "id": "Qwen-Qwen2.5-0.5B-Instruct",
                "loom": {
                    "registration": "failed",
                    "error": {"code": 5, "message": "no such file"},
                },
            }
        ],
    }
    entries = _status_entries(document)
    failed = [entry for entry in entries if _entry_state(entry) == "failed"]
    assert failed
    assert failed[0]["name"] == "Qwen-Qwen2.5-0.5B-Instruct"
    assert _error_text(failed[0]) == "no such file"
    assert _error_text({"error": "plain message"}) == "plain message"
    assert _error_text({}) is None


def test_a_still_preparing_registration_is_not_a_failure():
    document = {"data": [{"id": "m", "loom": {"registration": "preparing"}}]}
    states = {_entry_state(entry) for entry in _status_entries(document)}
    assert "failed" not in states
