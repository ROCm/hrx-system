# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import sys
from unittest.mock import patch

import pytest

from loom.launch.errors import InstallError
from loom.launch.hub import Hub, HubUnreachable
from loom.launch.install import (
    DOCUMENTS_ABSENT,
    DOCUMENTS_CACHE,
    DOCUMENTS_FAILED,
    DOCUMENTS_HUB,
    DOCUMENTS_INCOMPLETE,
    DOCUMENTS_INVALID,
    DOCUMENTS_NOT_HUB,
    DOCUMENTS_OFFLINE,
    DOCUMENTS_PRESENT,
    acquire_scenario_documents,
    default_models_directory,
    fetch_command,
    install,
    install_and_record,
    result_revision,
    selector,
)
from loom.launch.scenario import is_scenario
from loom.launch.store import InstallRegistry

REVISION = "3d5342e3317741e55995fa568ba907e528d811e9"

#: The GPU/NPU hybrid's card and its two scenario documents (no payloads).
HYBRID_CARD = {
    "schema_version": 2,
    "model": {"id": "qwen3.8-27b", "name": "Qwen3.8 27B hybrid"},
    "services": {"chat": {"kind": "text", "version": 1}},
    "presets": {
        "chat": {
            "service": "chat",
            "execution": {"rows": 1, "context_capacity": 40960, "max_tokens": 4096},
        }
    },
}
HYBRID_DEPLOYMENT = {
    "name": "qwen3.8-27b",
    "model": {"path": "model.json", "preset": "chat"},
    "definition": {"kind": "text"},
    "device": "amdgpu",
    "pool": "model",
}
HYBRID_DEVICES = {
    "devices": {
        "amdgpu": {"uri": "amdgpu://0"},
        "xdna": {"uri": "xdna://0?columns=8"},
    },
    "pools": {
        "model": {"devices": ["amdgpu"], "memory_bytes": 52000000000},
        "exchange": {
            "devices": ["amdgpu", "xdna"],
            "uncached": True,
            "uncached_devices": ["amdgpu"],
            "pool_backing": "fixed",
        },
    },
}


class FakeHub:
    """A hub transport answering files by URL suffix (else 404), recording URLs."""

    def __init__(self, files=None, *, unreachable=False):
        self.files = {
            name: json.dumps(body).encode() for name, body in (files or {}).items()
        }
        self.unreachable = unreachable
        self.urls = []

    def __call__(self, url, headers):
        self.urls.append(url)
        if self.unreachable:
            raise HubUnreachable(f"cannot reach {url}: connection refused")
        for suffix, body in self.files.items():
            if url.endswith(suffix):
                return 200, body
        return 404, b"Entry not found"

    def hub(self):
        return Hub(endpoint="http://hub.test", token="", transport=self)


def fetched(directory):
    """An installed directory as loom-fetch leaves it: the card only."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.json").write_text(json.dumps(HYBRID_CARD), encoding="utf-8")
    return directory


FETCH_SCRIPT = """#!{python}
import json, os, sys
output = [item.split("=", 1)[1] for item in sys.argv[1:]
          if item.startswith("--output=")][0]
if "fail" in os.environ.get("FAKE_FETCH_MODE", ""):
    print("download failed")
    raise SystemExit(3)
if "nocreate" not in os.environ.get("FAKE_FETCH_MODE", ""):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "model.json"), "w", encoding="utf-8") as file:
        file.write("{{\\"model\\": {{\\"id\\": \\"a/b\\"}}}}")
print("fetching 1/2 files")
print(json.dumps({{"preset": "chat", "revision": "abc123"}}))
"""


def fetch_script(tmp_path):
    script = tmp_path / "fake-fetch"
    script.write_text(FETCH_SCRIPT.format(python=sys.executable), encoding="utf-8")
    script.chmod(0o755)
    return script


def test_an_explicit_command_wins_over_every_discovery_path():
    assert fetch_command("/opt/loom-fetch") == ["/opt/loom-fetch"]


def test_the_environment_can_name_the_command():
    with patch.dict("os.environ", {"LOOM_FETCH": "/srv/loom-fetch"}):
        assert fetch_command() == ["/srv/loom-fetch"]


def test_a_command_on_path_is_used():
    with patch.dict("os.environ", {}, clear=True):
        with patch("loom.launch.install.shutil.which", return_value="/usr/bin/x"):
            assert fetch_command() == ["/usr/bin/x"]


def test_a_missing_command_explains_the_installation(tmp_path):
    with patch.dict("os.environ", {}, clear=True):
        with patch("loom.launch.install.shutil.which", return_value=None):
            with patch("loom.launch.install.sys.executable", str(tmp_path / "python")):
                with patch("loom.launch.install.Path.is_file", return_value=False):
                    with patch.dict("sys.modules", {"loom.model.fetch_main": None}):
                        with pytest.raises(InstallError) as error:
                            fetch_command()
    assert "hrx-loom[hf]" in str(error.value) or "pip install" in str(error.value)


def test_installation_runs_the_fetch_command_and_reads_its_result(tmp_path):
    script = fetch_script(tmp_path)
    output = tmp_path / "models" / "a-b"
    log = tmp_path / "fetch.log"
    lines = []
    result = install(
        "loom-fun/a-loom",
        output,
        preset="chat",
        explicit_fetch=str(script),
        log_path=log,
        on_line=lines.append,
    )
    assert result.path == output
    assert result.card_path.is_file()
    assert result.preset == "chat"
    assert result.revision == "abc123"
    assert result.command[1].startswith("huggingface:loom-fun/a-loom")
    assert "--preset" not in " ".join(result.command)
    assert ":chat" in result.command[1]
    assert "fetching 1/2 files" in lines
    assert "fetching 1/2 files" in log.read_text(encoding="utf-8")


def test_a_failed_installation_reports_the_exit_status(tmp_path):
    script = fetch_script(tmp_path)
    with patch.dict("os.environ", {"FAKE_FETCH_MODE": "fail"}):
        with pytest.raises(InstallError) as error:
            install("loom-fun/a-loom", tmp_path / "out", explicit_fetch=str(script))
    assert "exit code 3" in str(error.value)
    assert "download failed" in str(error.value)


def test_installation_without_a_card_is_refused(tmp_path):
    script = fetch_script(tmp_path)
    with patch.dict("os.environ", {"FAKE_FETCH_MODE": "nocreate"}):
        with pytest.raises(InstallError) as error:
            install("loom-fun/a-loom", tmp_path / "out", explicit_fetch=str(script))
    assert "did not install" in str(error.value)


def test_install_and_record_keeps_the_installation_for_later_launches(tmp_path):
    script = fetch_script(tmp_path)
    fake = FakeHub()
    with patch.dict("os.environ", {"HF_HUB_CACHE": str(tmp_path / "hf")}):
        result, model = install_and_record(
            "loom-fun/a-loom",
            state_root=tmp_path,
            explicit_fetch=str(script),
            hub=fake.hub(),
        )
    # No scenario documents published: asked once, at the pinned revision, and
    # the record is exactly as before.
    assert result.documents.status == DOCUMENTS_ABSENT
    assert fake.urls[0] == "http://hub.test/loom-fun/a-loom/raw/abc123/deployment.json"
    assert not (result.path / "deployment.json").exists()
    entry = json.loads((tmp_path / "installs.json").read_text())["installs"]
    assert sorted(entry["loom-fun/a-loom"]) == [
        "installed_at",
        "path",
        "preset",
        "repo",
        "revision",
        "size_bytes",
    ]
    assert model.path == str(default_models_directory(tmp_path) / "loom-fun-a-loom")
    assert model.size_bytes is not None and model.size_bytes > 0
    assert model.installed_at
    assert InstallRegistry(tmp_path).get("loom-fun/a-loom").path == model.path
    assert result.card_path.is_file()


def test_selectors_carry_preset_and_revision():
    assert selector("a/b", None, None) == "huggingface:a/b"
    assert selector("a/b", "chat", "v1") == "huggingface:a/b:chat@v1"
    assert selector("https://x/y", "chat", None) == "https://x/y"


def test_the_pinned_revision_is_read_from_file_transports():
    document = {
        "files": [
            {"transport": {"repository": "other/x", "revision": "fff"}},
            {"transport": {"repository": "a/b", "revision": REVISION}},
        ]
    }
    assert result_revision(document, "a/b") == REVISION
    assert result_revision(document, "c/d") == "fff"
    assert result_revision({"revision": "r1", **document}, "a/b") == "r1"


def test_published_scenario_documents_are_installed_at_the_pinned_revision(tmp_path):
    directory = fetched(tmp_path / "installed")
    fake = FakeHub(
        {
            f"/{REVISION}/deployment.json": HYBRID_DEPLOYMENT,
            f"/{REVISION}/devices.json": HYBRID_DEVICES,
        }
    )
    found = acquire_scenario_documents(
        directory,
        "ilintar/hybrid",
        REVISION,
        offline=False,
        hub=fake.hub(),
        cache=tmp_path / "empty-cache",
    )
    assert found.status == DOCUMENTS_HUB
    assert found.installed and found.warning is None
    assert json.loads((directory / "deployment.json").read_text()) == HYBRID_DEPLOYMENT
    assert json.loads((directory / "devices.json").read_text()) == HYBRID_DEVICES
    assert found.summary() == (
        "deployment qwen3.8-27b on amdgpu + xdna (deployment.json + devices.json "
        f"from the hub at {REVISION})"
    )
    assert fake.urls == [
        f"http://hub.test/ilintar/hybrid/raw/{REVISION}/deployment.json",
        f"http://hub.test/ilintar/hybrid/raw/{REVISION}/devices.json",
    ]
    assert not list(directory.glob(".*.partial"))


def test_documents_already_present_or_cached_never_reach_the_hub(tmp_path):
    fake = FakeHub()
    directory = fetched(tmp_path / "present")
    (directory / "deployment.json").write_text(json.dumps(HYBRID_DEPLOYMENT))
    (directory / "devices.json").write_text(json.dumps(HYBRID_DEVICES))
    found = acquire_scenario_documents(
        directory, "a/b", REVISION, offline=False, hub=fake.hub(), cache=None
    )
    assert found.status == DOCUMENTS_PRESENT
    assert found.summary().endswith("shipped with the card)")
    cache = tmp_path / "hf"
    snapshot = cache / "models--a--c" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (cache / "models--a--c" / "refs").mkdir()
    (cache / "models--a--c" / "refs" / "main").write_text(REVISION)
    (snapshot / "deployment.json").write_text(json.dumps(HYBRID_DEPLOYMENT))
    (snapshot / "devices.json").write_text(json.dumps(HYBRID_DEVICES))
    second = fetched(tmp_path / "second")
    found = acquire_scenario_documents(
        second, "a/c", None, offline=True, hub=fake.hub(), cache=cache
    )
    assert found.status == DOCUMENTS_CACHE
    assert is_scenario(second)
    third = fetched(tmp_path / "third")
    found = acquire_scenario_documents(
        third, "a/d", REVISION, offline=True, hub=fake.hub(), cache=cache
    )
    assert found.status == DOCUMENTS_OFFLINE
    assert found.warning is None
    assert fake.urls == []


def test_a_profile_without_documents_is_silent_and_a_partial_one_is_explained(
    tmp_path,
):
    directory = fetched(tmp_path / "installed")
    fake = FakeHub()
    found = acquire_scenario_documents(
        directory, "a/b", "v1", offline=False, hub=fake.hub(), cache=None
    )
    assert found.status == DOCUMENTS_ABSENT
    assert found.warning is None and found.summary() is None
    assert len(fake.urls) == 2
    partial = FakeHub({"/deployment.json": HYBRID_DEPLOYMENT})
    found = acquire_scenario_documents(
        directory, "a/b", "v1", offline=False, hub=partial.hub(), cache=None
    )
    assert found.status == DOCUMENTS_INCOMPLETE
    assert found.warning == (
        "a/b@v1 publishes deployment.json but not devices.json; installed as an "
        "ordinary profile (a scenario needs both)"
    )
    assert not (directory / "deployment.json").exists()
    found = acquire_scenario_documents(
        directory, "https://x/y", None, offline=False, hub=fake.hub(), cache=None
    )
    assert found.status == DOCUMENTS_NOT_HUB


def test_a_hub_failure_is_reported_without_failing_the_install(tmp_path):
    script = fetch_script(tmp_path)
    fake = FakeHub(unreachable=True)
    with patch.dict("os.environ", {"HF_HUB_CACHE": str(tmp_path / "hf")}):
        result, model = install_and_record(
            "loom-fun/a-loom",
            state_root=tmp_path,
            explicit_fetch=str(script),
            hub=fake.hub(),
        )
    assert result.card_path.is_file()
    assert result.documents.status == DOCUMENTS_FAILED
    assert result.documents.warning.startswith(
        "could not check loom-fun/a-loom@abc123 for deployment.json and devices.json: "
        "cannot reach http://hub.test/loom-fun/a-loom/raw/abc123/deployment.json"
    )
    assert "The install itself succeeded" in result.documents.warning
    assert model.scenario is None
    assert InstallRegistry(tmp_path).get("loom-fun/a-loom").path == model.path
    # Documents that would not serve are not left behind.
    directory = fetched(tmp_path / "invalid")
    broken = FakeHub(
        {
            "/deployment.json": {"name": "x", "model": {"path": "absent.json"}},
            "/devices.json": {"devices": {"amdgpu": {"uri": "amdgpu://0"}}},
        }
    )
    found = acquire_scenario_documents(
        directory, "a/b", REVISION, offline=False, hub=broken.hub(), cache=None
    )
    assert found.status == DOCUMENTS_INVALID
    assert not (directory / "deployment.json").exists()
    assert not (directory / "devices.json").exists()
