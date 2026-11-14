# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import sys
from unittest.mock import patch

import pytest

from loom.launch.cli import (
    _cmd_bench,
    _cmd_chat,
    _cmd_hardware,
    _cmd_install,
    _cmd_search,
    _cmd_serve,
    _cmd_show,
    _parser,
    _search_order,
    main,
)
from loom.launch.errors import LaunchError
from loom.launch.hardware import Compatibility
from loom.launch.install_test import (
    HYBRID_DEPLOYMENT,
    HYBRID_DEVICES,
    REVISION,
    FakeHub,
)


def test_every_verb_parses_with_its_handler():
    parser = _parser()
    handlers = {
        "search": _cmd_search,
        "show": _cmd_show,
        "install": _cmd_install,
        "serve": _cmd_serve,
        "chat": _cmd_chat,
        "bench": _cmd_bench,
        "hardware": _cmd_hardware,
    }
    for verb, handler in handlers.items():
        target = ["search", "loom"] if verb == "search" else [verb]
        if verb in ("show", "install", "serve", "chat", "bench"):
            target = [verb, "loom-fun/a-loom"]
        assert parser.parse_args(target).handler is handler


def test_serving_defaults_are_loopback_and_instrumented():
    arguments = _parser().parse_args(["serve", "loom-fun/a-loom"])
    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8080
    assert arguments.no_proxy is False
    assert arguments.detach is False
    assert arguments.wait == 1800.0
    assert arguments.loom_serve is None


def test_context_and_parallel_are_available_on_every_launching_verb():
    for verb in ("serve", "chat", "bench"):
        arguments = _parser().parse_args([verb, "loom-fun/a-loom", "--context", "8192"])
        assert arguments.context == 8192
    served = _parser().parse_args(
        ["serve", "x/y", "--parallel", "2", "--draft-depth", "0"]
    )
    assert (served.parallel, served.draft_depth) == (2, 0)


def test_bench_defaults_measure_a_comparable_suite():
    arguments = _parser().parse_args(["bench", "loom-fun/a-loom"])
    assert arguments.suite == "standard"
    assert arguments.repeats == 1
    assert arguments.think == "off"
    assert arguments.no_store is False


def test_chat_defaults_follow_the_package_and_allow_an_endpoint():
    arguments = _parser().parse_args(
        ["chat", "--base-url", "http://127.0.0.1:8080", "--prompt", "hi"]
    )
    assert arguments.think == "auto"
    assert arguments.base_url == "http://127.0.0.1:8080"
    assert arguments.prompt == "hi"


def test_search_reads_profile_details_unless_told_not_to():
    arguments = _parser().parse_args(["search", "loom"])
    assert arguments.no_details is False
    assert arguments.limit == 20
    assert arguments.tag == []


def test_without_a_verb_the_parser_refuses():
    with pytest.raises(SystemExit):
        main([])


def test_status_reports_nothing_running_in_a_fresh_state(tmp_path, capsys):
    code = main(["--state", str(tmp_path), "status"])
    assert code == 1
    assert "no launch is running" in capsys.readouterr().out


def test_search_order_prefers_the_best_fit():
    class Row:
        def __init__(self, repo, compat):
            self.repo = repo
            self.compat = compat

    rows = [
        {"model": Row("a", None), "compat": None},
        {"model": Row("b", None), "compat": Compatibility(score=0.9)},
    ]
    ordered = sorted(rows, key=_search_order)
    assert ordered[0]["compat"].score == 0.9
    assert ordered[0]["model"].repo == "b"


def _hybrid_package(directory):
    directory.mkdir()
    (directory / "model.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": {"id": "qwen3.8-27b", "name": "Qwen3.8 27B hybrid"},
                "services": {"chat": {"kind": "text", "version": 1}},
                "presets": {
                    "chat": {
                        "service": "chat",
                        "execution": {
                            "rows": 1,
                            "context_capacity": 40960,
                            "max_tokens": 4096,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "llm_config.json").write_text(
        json.dumps({"arch": "qwen35", "context_length": 262144}), encoding="utf-8"
    )
    return directory


def test_show_reports_the_served_context_and_suggests_the_local_path(tmp_path, capsys):
    package = _hybrid_package(tmp_path / "hybrid")
    code = main(
        [
            "--state",
            str(tmp_path / "state"),
            "--hub",
            "http://127.0.0.1:9",
            "show",
            str(package),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "  context      40960 tokens (model native 262144)\n" in out
    assert f"serve with: loom-launch serve {os.path.abspath(package)}\n" in out
    assert "install with" not in out


def test_serve_refuses_a_context_beyond_the_native_one(tmp_path, capsys):
    package = _hybrid_package(tmp_path / "hybrid")
    code = main(
        [
            "--state",
            str(tmp_path / "state"),
            "--hub",
            "http://127.0.0.1:9",
            "serve",
            str(package),
            "--context",
            "300000",
            "--loom-serve",
            "/bin/false",
        ]
    )
    assert code == 1
    assert "exceeds the model's 262144 tokens" in capsys.readouterr().err


HYBRID_REPO = "ilintar/Qwen3.8-27B-hybrid-loom"

#: A loom-fetch that installs what the card declares (never deployment.json or
#: devices.json) and reports the pinned revision per file transport.
SCENARIO_FETCH = """#!{python}
import json, shutil, sys
from pathlib import Path
output = Path([a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--output=")][0])
Path({argv!r}).write_text("\\n".join(sys.argv[1:]))
output.mkdir(parents=True, exist_ok=True)
for name in ("model.json", "llm_config.json"):
    source = Path({source!r}) / name
    if source.is_file():
        shutil.copy(source, output / name)
print(json.dumps({{"preset": "chat", "files": [{{"path": "model.json", "transport":
    {{"kind": "hf", "repository": {repo!r}, "revision": {revision!r}}}}}]}}))
"""


def _scenario_fetch(tmp_path, source):
    script = tmp_path / "loom-fetch"
    script.write_text(
        SCENARIO_FETCH.format(
            python=sys.executable,
            argv=str(tmp_path / "fetch-argv.txt"),
            source=str(source),
            repo=HYBRID_REPO,
            revision=REVISION,
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _install(tmp_path, hub, script, *extra):
    state = tmp_path / "state"
    with patch("loom.launch.cli._hub", return_value=hub):
        with patch.dict(os.environ, {"HF_HUB_CACHE": str(tmp_path / "hf")}):
            code = main(
                [
                    "--state",
                    str(state),
                    "install",
                    HYBRID_REPO,
                    "--loom-fetch",
                    str(script),
                    *extra,
                ]
            )
    return code, state


def _planned(state, *serve_arguments):
    from loom.launch.cli import SERVE_CAPACITY_OPTIONS, _plan, _prepare, _resolve
    from loom.launch.server import LaunchOptions
    from loom.launch.store import RunStore

    args = _parser().parse_args(
        ["--state", str(state), "--hub", "http://127.0.0.1:9", "serve", HYBRID_REPO]
        + list(serve_arguments)
    )
    resolved = _resolve(args, install=False)
    prepared = _prepare(args, resolved, SERVE_CAPACITY_OPTIONS)
    run = RunStore(state).create(metadata={"repo": HYBRID_REPO, "source": "serve"})
    options = LaunchOptions(
        device=args.device,
        port=8123,
        workers=4,
        memory_bytes=args.memory_bytes,
        pool_backing=args.pool_backing,
        loom_serve="/opt/loom-serve",
    )
    return _plan(prepared, resolved, run, options), run


def test_an_installed_hub_scenario_carries_its_own_documents_and_serves_them(
    tmp_path, capsys
):
    published = _hybrid_package(tmp_path / "published")
    script = _scenario_fetch(tmp_path, published)
    fake = FakeHub(
        {
            f"/{REVISION}/deployment.json": HYBRID_DEPLOYMENT,
            f"/{REVISION}/devices.json": HYBRID_DEVICES,
        }
    )
    code, state = _install(
        tmp_path,
        fake.hub(),
        script,
        "--revision",
        REVISION,
        "--materialize",
        "share",
    )
    assert code == 0
    captured = capsys.readouterr()
    assert (
        "  scenario  deployment qwen3.8-27b on amdgpu + xdna (deployment.json + "
        f"devices.json from the hub at {REVISION})\n"
    ) in captured.out
    assert "warning" not in captured.err
    fetch_argv = (tmp_path / "fetch-argv.txt").read_text().splitlines()
    assert f"huggingface:{HYBRID_REPO}@{REVISION}" in fetch_argv
    assert "--materialize=share" in fetch_argv
    installed = state / "models" / "ilintar-Qwen3.8-27B-hybrid-loom"
    assert json.loads((installed / "deployment.json").read_text()) == HYBRID_DEPLOYMENT
    assert json.loads((installed / "devices.json").read_text()) == HYBRID_DEVICES
    record = json.loads((state / "installs.json").read_text())["installs"][HYBRID_REPO]
    assert record["scenario"] is True
    assert record["scenario_revision"] == REVISION
    assert record["revision"] == REVISION

    # `show` of the installed name explains the installed scenario.
    assert (
        main(
            ["--state", str(state), "--hub", "http://127.0.0.1:9", "show", HYBRID_REPO]
        )
        == 0
    )
    assert (
        "scenario     deployment qwen3.8-27b on amdgpu + xdna"
        in capsys.readouterr().out
    )
    main(
        [
            "--state",
            str(state),
            "--hub",
            "http://127.0.0.1:9",
            "--json",
            "show",
            HYBRID_REPO,
        ]
    )
    assert json.loads(capsys.readouterr().out)["scenario"]["devices"] == "amdgpu + xdna"

    # `serve` plans the scenario: the installed documents, no --device.
    plan, run = _planned(state)
    assert plan.kind == "scenario"
    assert plan.command == (
        "/opt/loom-serve",
        f"--devices={installed}/devices.json",
        f"--model={installed}/deployment.json",
        "--listen=127.0.0.1:8123",
        "--workers=4",
    )
    assert plan.device_uri == "amdgpu + xdna"
    assert plan.settings.context_length == 40960
    assert not run.deployment_path.exists()
    # --memory-bytes changes a run-local copy of devices.json, never the package's.
    plan, run = _planned(state, "--memory-bytes", "1000")
    assert plan.command[:2] == ("/opt/loom-serve", f"--devices={run.root}/devices.json")
    # Options the scenario owns are refused, naming both documents.
    with pytest.raises(LaunchError) as error:
        _planned(state, "--context", "8192", "--device", "amdgpu://1")
    message = str(error.value)
    assert "is a prepared scenario" in message and "--context, --device" in message
    assert f"{installed}/devices.json" in message


def test_a_profile_without_scenario_documents_installs_as_before(tmp_path, capsys):
    published = _hybrid_package(tmp_path / "published")
    script = _scenario_fetch(tmp_path, published)
    fake = FakeHub({"/model.json": {"model": {"id": "x"}}})
    code, state = _install(tmp_path, fake.hub(), script)
    assert code == 0
    captured = capsys.readouterr()
    assert "  scenario  " not in captured.out
    assert "warning" not in captured.err
    installed = state / "models" / "ilintar-Qwen3.8-27B-hybrid-loom"
    assert not (installed / "deployment.json").exists()
    assert not (installed / "devices.json").exists()
    record = json.loads((state / "installs.json").read_text())["installs"][HYBRID_REPO]
    assert sorted(record) == [
        "installed_at",
        "path",
        "preset",
        "repo",
        "revision",
        "size_bytes",
    ]
    plan, run = _planned(state, "--device", "amdgpu://0")
    assert plan.kind == "composed"
    assert plan.command[1:4] == (
        f"--model={run.deployment_path}",
        "--listen=127.0.0.1:8123",
        "--device=amdgpu://0",
    )


def test_a_hub_failure_after_the_fetch_keeps_the_install_and_says_so(tmp_path, capsys):
    published = _hybrid_package(tmp_path / "published")
    script = _scenario_fetch(tmp_path, published)
    code, state = _install(tmp_path, FakeHub(unreachable=True).hub(), script)
    assert code == 0
    err = capsys.readouterr().err
    assert (
        f"warning: could not check {HYBRID_REPO}@{REVISION} for deployment.json and "
        f"devices.json: cannot reach http://hub.test/{HYBRID_REPO}/raw/{REVISION}/"
        "deployment.json"
    ) in err
    assert "The install itself succeeded" in err
    record = json.loads((state / "installs.json").read_text())["installs"][HYBRID_REPO]
    assert "scenario" not in record
