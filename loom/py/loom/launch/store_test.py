# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from pathlib import Path
from unittest.mock import patch

from loom.launch.store import (
    BenchmarkStore,
    InstalledModel,
    InstallRegistry,
    JsonlWriter,
    ProfileCache,
    RunStore,
    read_json,
    state_directory,
    write_json,
)


def test_state_directory_follows_explicit_environment_then_default(tmp_path):
    with patch.dict("os.environ", {"LOOM_LAUNCH_STATE": str(tmp_path)}):
        assert state_directory() == tmp_path
    assert state_directory(tmp_path / "x") == tmp_path / "x"
    with patch.dict("os.environ", {}, clear=True):
        assert state_directory().name == "loom-launch"


def test_json_documents_round_trip_whole(tmp_path):
    path = write_json(tmp_path / "a" / "b.json", {"value": 1})
    assert read_json(path) == {"value": 1}
    assert read_json(tmp_path / "absent.json") == {}
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert read_json(tmp_path / "broken.json") == {}


def test_jsonl_logs_append_one_record_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlWriter(path)
    writer.write({"event": "ready"})
    writer.write({"event": "serving", "endpoint": "http://x"})
    writer.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["ready", "serving"]


def test_run_directories_are_created_and_enumerated(tmp_path):
    store = RunStore(tmp_path)
    first = store.create("20260101T000000-1", metadata={"repo": "a/b"})
    second = store.create("20260102T000000-1")
    first.save()
    assert first.serve_log.name == "serve.log"
    assert first.deployment_path.name == "serve.json"
    assert store.open("20260101T000000-1").read_metadata()["repo"] == "a/b"
    assert [run.run_id for run in store.list()] == [
        "20260102T000000-1",
        "20260101T000000-1",
    ]
    assert store.latest().run_id == "20260102T000000-1"
    assert second.describe()["id"] == "20260102T000000-1"


def test_installs_are_recorded_and_can_be_forgotten(tmp_path):
    registry = InstallRegistry(tmp_path)
    model = InstalledModel(
        repo="loom-fun/a-loom",
        path=str(tmp_path / "models" / "a"),
        preset="chat",
        revision="abc",
        installed_at="2026-11-12T00:00:00Z",
        size_bytes=1024,
    )
    (tmp_path / "models" / "a").mkdir(parents=True)
    registry.record(model)
    assert registry.get("loom-fun/a-loom").preset == "chat"
    assert registry.find_by_path(tmp_path / "models" / "a").repo == "loom-fun/a-loom"
    assert registry.list()[0].describe()["size"] == "1.0 KiB"
    assert registry.forget("loom-fun/a-loom") is True
    assert registry.forget("loom-fun/a-loom") is False
    assert registry.list() == []


def test_profile_cache_expires_and_clears(tmp_path):
    cache = ProfileCache(tmp_path, max_age_seconds=3600)
    assert cache.get("loom-fun/a-loom") is None
    cache.put("loom-fun/a-loom", {"model_card": {"model": {"id": "a"}}})
    assert cache.get("loom-fun/a-loom")["model_card"]["model"]["id"] == "a"
    assert cache.clear() == 1
    assert cache.get("loom-fun/a-loom") is None


def test_benchmark_records_are_stored_indexed_and_loadable(tmp_path):
    store = BenchmarkStore(tmp_path)
    assert store.list() == []
    record = {
        "id": "run-1",
        "profile": "loom-fun/a-loom",
        "suite": "quick",
        "started_at": "2026-11-12T00:00:00Z",
        "ok": True,
        "metrics": {"decode": {"value": 10.0}},
    }
    path = store.save(record)
    assert path.is_file()
    assert store.index()[0]["id"] == "run-1"
    assert [entry["id"] for entry in store.list("loom-fun/a-loom")] == ["run-1"]
    assert store.list("other") == []
    assert store.latest("loom-fun/a-loom")["metrics"]["decode"]["value"] == 10.0
    assert store.load("run-1")["suite"] == "quick"
    assert store.load("absent") is None


def test_a_relative_state_directory_resolves_absolutely(tmp_path):
    # Every run record, deployment document and install is persisted with paths
    # derived from this one, so a relative state root cannot be stored as typed.
    assert state_directory("relative/state").is_absolute()
    with patch.dict("os.environ", {"LOOM_LAUNCH_STATE": "relative/state"}):
        assert state_directory().is_absolute()


def test_a_legacy_relative_install_path_still_resolves(tmp_path):
    state = tmp_path / "state"
    card = state / "models" / "a-loom" / "model.json"
    card.parent.mkdir(parents=True)
    card.write_text("{}", encoding="utf-8")
    # A record written before the state directory was made absolute.
    write_json(
        state / "installs.json",
        {"installs": {"a/loom": {"path": "state/models/a-loom"}}},
    )
    model = InstallRegistry(state).get("a/loom")
    assert model is not None
    assert Path(model.path) == state / "models" / "a-loom"
    assert model.card_path == card
