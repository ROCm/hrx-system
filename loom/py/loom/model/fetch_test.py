# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import http.server
import json
import os
import threading
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch import check_output, install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_main import _cache_directory, main
from loom.model.fetch_source import Document, FetchError
from loom.model.fetch_storage import ContentCache
from loom.model.http_test_server import serving_http
from loom.model.lock import FileIdentity, ModelLock


def test_platform_cache_directory_does_not_resolve_an_unused_home(tmp_path):
    for platform_name, environment_name in (
        ("nt", "LOCALAPPDATA"),
        ("posix", "XDG_CACHE_HOME"),
    ):
        platform_cache = tmp_path / platform_name
        with (
            patch(
                "loom.model.fetch_main.os",
                SimpleNamespace(
                    environ={environment_name: str(platform_cache)},
                    name=platform_name,
                ),
            ),
            patch(
                "loom.model.fetch_main.Path.home",
                side_effect=AssertionError("unused home directory lookup"),
            ),
        ):
            assert _cache_directory() == platform_cache / "loom"


def test_catalog_receipt_is_owned_verified_and_published_with_the_card(tmp_path):
    publication = tmp_path / "publication"
    publication.mkdir()
    (publication / "code.loom").write_bytes(b"code")
    (publication / "weights.data").write_bytes(b"weights")
    text = json.dumps(_card())
    card = publication / "model.json"
    card.write_text(text)
    cache = ContentCache(tmp_path / "cache")
    snapshot = {"source": "github:fixture/library@compatible", "revision": "a" * 40}
    plan = plan_fetch(
        cache.read_document(str(card), offline=True),
        ModelCard.parse(text).select(),
        cache,
        no_lock=True,
        catalog=snapshot,
    )
    snapshot["revision"] = "b" * 40
    output = tmp_path / "installed"
    install(plan, cache, output)
    receipt = json.loads((output / ".loom-fetch.json").read_text())
    assert receipt["catalog"]["revision"] == "a" * 40
    assert receipt["card"] == {
        "location": str(card),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    check_output(plan, output)
    (output / ".loom-fetch.json").write_text("{}")
    with pytest.raises(FetchError, match="conflicting installation document"):
        check_output(plan, output)


def test_catalog_receipt_destination_is_reserved(tmp_path):
    document = _card()
    document["programs"]["p"]["inputs"] = [".loom-fetch.json"]
    card = ModelCard(document)
    cache = ContentCache(tmp_path / "cache")
    with pytest.raises(ManifestError, match="reserved installation path"):
        plan_fetch(
            Document(str(tmp_path / "model.json"), json.dumps(document)),
            card.select(),
            cache,
            no_lock=True,
        )


def test_card_publication_failure_rolls_back_the_catalog_receipt(tmp_path):
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    card = ModelCard(_card())
    plan = plan_fetch(
        Document(str(tmp_path / "source.json"), json.dumps(card.document)),
        card.select(),
        cache,
        no_lock=True,
        catalog={"source": "local catalog"},
    )
    output = tmp_path / "output"
    link = os.link

    def fail_card_publication(source, destination, **options):
        if destination == output / "model.json":
            assert (output / ".loom-fetch.json").is_file()
            raise OSError("publication failed")
        return link(source, destination, **options)

    with (
        patch("loom.model.fetch.os.link", side_effect=fail_card_publication),
        pytest.raises(OSError, match="publication failed"),
    ):
        install(plan, cache, output)
    assert not (output / "model.json").exists()
    assert not (output / ".loom-fetch.json").exists()
    assert not (output / "code.loom").exists()
    assert not (output / "weights.data").exists()


def _card():
    return {
        "schema_version": 1,
        "model": {"id": "fixture/fetch", "name": "Fetch fixture"},
        "programs": {"p": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"text": {"kind": "text", "version": 1}},
        "variants": {
            "v": {
                "services": {
                    "text": {
                        "programs": ["p"],
                        "parameters": {"scope": {"files": ["weights.data"]}},
                    }
                }
            },
            "unused": {
                "services": {
                    "text": {
                        "programs": ["p"],
                        "parameters": {"scope": {"files": ["large.data"]}},
                    }
                }
            },
        },
        "presets": {"check": {"variant": "v", "service": "text", "programs": "p"}},
    }


def _lock(files):
    return ModelLock.snapshot(
        {},
        [
            FileIdentity(name, len(content), hashlib.sha256(content).hexdigest())
            for name, content in files.items()
        ],
    )


def test_memory_guidance_is_owned_metadata_in_listing_and_fetch_plan(tmp_path, capsys):
    document = _card()
    memory = {
        "weights_bytes": 7,
        "workload": "Original parameter payload; overhead excluded.",
    }
    document["presets"]["check"]["memory"] = memory
    path = tmp_path / "model.json"
    path.write_text(json.dumps(document))
    assert main([str(path), "--list"]) == 0
    assert (
        "Estimated memory: weights 7 bytes; transients unknown."
        in capsys.readouterr().out
    )
    assert main([str(path), "--list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["presets"]["check"]["memory"] == memory
    for name, content in (
        ("code.loom", b"code"),
        ("weights.data", b"weights"),
        ("large.data", b"alternate"),
    ):
        (tmp_path / name).write_bytes(content)
    result = main(
        [
            str(path),
            "--preset=check",
            "--no-lock",
            "--no-input",
            "--dry-run",
            "--output=" + str(tmp_path / "preview"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0, captured.err
    assert "overhead excluded" in captured.err
    cache = ContentCache(tmp_path / "cache")
    source = Document(str(path), path.read_text())
    card = ModelCard(document)
    plan = plan_fetch(source, card.select(), cache, no_lock=True)
    summary = plan.summary()
    assert summary["memory"] == memory
    summary["memory"]["weights_bytes"] = 999
    assert plan.selection.settings["memory"] == memory
    changed = plan_fetch(
        source, card.select("check", variant="unused"), cache, no_lock=True
    )
    assert "memory" not in changed.summary()


@contextmanager
def _server(card, files, *, parallel=False):
    requests = []
    resources = {
        "/model.json": json.dumps(card).encode(),
        "/model.json.lock": json.dumps(_lock(files)).encode(),
        **{"/" + name: content for name, content in files.items()},
    }
    rendezvous = threading.Barrier(2) if parallel else None

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            content = resources.get(self.path)
            if content is None:
                self.send_error(404)
                return
            if rendezvous and self.path in ("/code.loom", "/weights.data"):
                rendezvous.wait()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    with (
        http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server,
        serving_http(server),
    ):
        yield f"http://127.0.0.1:{server.server_port}", requests, resources


def _remote_plan(base, cache, **kwargs):
    document = cache.read_document(
        base + "/model.json", offline=kwargs.get("offline", False)
    )
    return plan_fetch(
        document, ModelCard.parse(document.text).select(), cache, **kwargs
    )


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_progress_preserves_json_stdout_and_quiet_mode(tmp_path, capsys, quiet):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_card(), files, parallel=True) as (base, _, _):
        output = tmp_path / "installed"
        arguments = [
            base + "/model.json",
            "--no-input",
            "--json",
            "--output=" + str(output),
            "--cache=" + str(tmp_path / "cache"),
        ]
        assert main(arguments + (["--quiet"] if quiet else [])) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["card"] == str(output / "model.json")
        if quiet:
            assert captured.err == ""
        else:
            for label in (
                "Metadata",
                "Fetch",
                "Download",
                "Verify",
                "Install",
                "Complete",
            ):
                assert "loom-fetch  " + label in captured.err
            assert "2/2 files" in captured.err
            assert base not in captured.err
            assert "\x1b" not in captured.err


def test_progress_failure_never_reports_completion_or_publishes_card(tmp_path, capsys):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_card(), files) as (base, _, resources):
        resources["/weights.data"] = b"corrupt"
        output = tmp_path / "installed"
        assert (
            main(
                [
                    base + "/model.json",
                    "--no-input",
                    "--json",
                    "--output=" + str(output),
                    "--cache=" + str(tmp_path / "cache"),
                ]
            )
            == 1
        )
        captured = capsys.readouterr()
        assert json.loads(captured.out)["kind"] == "error"
        assert "SHA-256" in captured.err
        assert "Complete" not in captured.err
        assert not (output / "model.json").exists()
        assert not (output / "code.loom").exists()


def test_progress_tracks_unique_sources_then_installed_destinations(tmp_path):
    events = []
    card = _card()
    card["programs"]["p"]["libraries"] = [{"path": "copy.loom", "origin": "code.loom"}]
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(card, files, parallel=True) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache", progress=events.append)
        plan = _remote_plan(base, cache)
        install(plan, cache, tmp_path / "installed")
        acquired = [event for event in events if event.phase == "acquire"]
        assert [(event.completed_files, event.total_files) for event in acquired] == [
            (0, 2),
            (1, 2),
            (2, 2),
        ]
        assert events[-1].phase == "complete"
        assert events[-1].completed_files == events[-1].total_files == 3
        assert requests.count("/code.loom") == 1
        events.clear()
        offline = _remote_plan(base, cache, offline=True)
        install(offline, cache, tmp_path / "offline")
        assert not any(event.phase in ("download", "transfer") for event in events)
        assert any(event.phase == "verify" for event in events)
        assert events[-1].phase == "complete"


def _publication_card():
    document = _card()
    document["programs"]["alternate"] = {
        "inputs": [{"path": "renamed.loom", "origin": "code.loom"}],
        "libraries": [],
    }
    for variant in document["variants"].values():
        variant["services"]["text"]["programs"].append("alternate")
    document["variants"]["unused"]["services"]["text"]["parameters"]["scope"][
        "files"
    ] = [{"path": "weights.data", "origin": "alternate.data"}]
    return document


def test_developer_parameters_keep_code_lock_and_record_observed_bytes(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_card(), files) as (base, _, resources):
        original_lock = resources["/model.json.lock"]
        resources["/weights.data"] = b"developer values of a different size"
        cache = ContentCache(tmp_path / "cache")
        plan = _remote_plan(base, cache, no_verify_parameters=True)
        assert plan.unverified_parameters == {"weights.data"}
        assert set(plan.identities) == {"code.loom"}
        output = tmp_path / "installed"
        install(plan, cache, output)
        assert (output / "weights.data").read_bytes() == resources["/weights.data"]
        installed_card = ModelCard.parse((output / "model.json").read_text())
        lock = ModelLock.parse((output / "model.json.lock").read_text())
        for name, identity in lock.cover(installed_card.select()).items():
            identity.verify(output / name)
        receipt = json.loads((output / ".loom-fetch.json").read_text())
        assert receipt["unverified_parameters"] == ["weights.data"]
        assert resources["/model.json.lock"] == original_lock
        assert plan.summary()["unverified_parameters"] == ["weights.data"]
        with pytest.raises(FetchError, match=r"another output|existing destination"):
            install(plan, cache, output)
        strict = ContentCache(tmp_path / "strict")
        with pytest.raises(
            (ManifestError, FetchError), match=r"size|byte count|SHA-256"
        ):
            install(_remote_plan(base, strict), strict, tmp_path / "rejected")
        assert not (tmp_path / "rejected/model.json").exists()


@pytest.mark.parametrize("role", ["inputs", "libraries", "auxiliary", "tokenizer"])
def test_parameter_exception_cannot_weaken_another_role_at_another_path(tmp_path, role):
    card = _card()
    shared = {"path": "other-name.data", "origin": "weights.data"}
    if role in ("inputs", "libraries"):
        card["programs"]["p"][role].append(shared)
    elif role == "auxiliary":
        card["programs"]["p"]["inputs"] = [{"path": "code.loom", "data": [shared]}]
    else:
        card["variants"]["v"]["services"]["text"]["tokenizers"] = {"text": shared}
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(card, files) as (base, _, resources):
        resources["/weights.data"] = b"changed"
        cache = ContentCache(tmp_path / "cache")
        plan = _remote_plan(base, cache, no_verify_parameters=True)
        assert not plan.unverified_parameters
        assert set(plan.identities) == {"code.loom", "weights.data", "other-name.data"}
        with pytest.raises(ManifestError, match="SHA-256"):
            install(plan, cache, tmp_path / "output")
        assert not (tmp_path / "output/model.json").exists()


@pytest.mark.parametrize("conflict", ["no_lock", "update_lock"])
def test_parameter_verification_exception_does_not_compose_with_other_lock_modes(
    tmp_path, conflict
):
    cache = ContentCache(tmp_path / "cache")
    with pytest.raises(ManifestError, match="ordinary locked acquisition"):
        plan_fetch(
            Document(str(tmp_path / "model.json"), json.dumps(_card())),
            ModelCard(_card()).select(),
            cache,
            no_verify_parameters=True,
            **{conflict: True},
        )


def test_publication_lock_unions_sources_without_merging_destinations(tmp_path):
    events = []
    files = {
        "code.loom": b"code",
        "weights.data": b"weights",
        "alternate.data": b"different weights",
    }
    with _server(_publication_card(), files, parallel=True) as (
        base,
        requests,
        resources,
    ):
        cache = ContentCache(tmp_path / "cache", progress=events.append)
        document = cache.read_document(base + "/model.json", offline=False)
        card = ModelCard.parse(document.text)
        lock_path = tmp_path / "publication.json.lock"
        lock_path.write_text('{"stale":"record"}')
        plan = plan_lock(
            document, card, cache, lock_location=str(lock_path), lock_scope="all"
        )
        assert requests == ["/model.json"]
        assert len(plan.selections) == 4
        assert len(plan.sources) == 3
        assert update_lock(plan, cache) == lock_path
        assert {event.path for event in events if event.path is not None} <= {
            "code.loom",
            "renamed.loom",
            "weights.data",
        }
        acquired = [event for event in events if event.phase == "acquire"]
        assert [(event.completed_files, event.total_files) for event in acquired] == [
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
        ]
        assert events[-1].phase == "complete"
        assert events[-1].completed_files == events[-1].total_files == 1
        assert set(requests) == {"/model.json", *("/" + name for name in files)}
        assert len(requests) == 4
        lock = ModelLock.parse(lock_path.read_text())
        for choice in plan.selections:
            selected = card.select(**choice)
            assert set(lock.cover(selected)) == set(selected.files)
        # A clean client reads the broad publication lock but acquires one choice.
        resources["/model.json.lock"] = lock_path.read_bytes()
        requests.clear()
        client = ContentCache(tmp_path / "client-cache")
        selected = _remote_plan(base, client)
        install(selected, client, tmp_path / "installed")
        assert "/alternate.data" not in requests
        assert (tmp_path / "installed/weights.data").read_bytes() == b"weights"
        assert not (tmp_path / "installed/renamed.loom").exists()


def test_publication_lock_failure_preserves_the_previous_snapshot(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_publication_card(), files) as (base, _, _):
        cache = ContentCache(tmp_path / "cache")
        document = cache.read_document(base + "/model.json", offline=False)
        path = tmp_path / "model.json.lock"
        before = json.dumps(_lock(files))
        path.write_text(before)
        plan = plan_lock(
            document, ModelCard.parse(document.text), cache, lock_location=str(path)
        )
        with pytest.raises(FetchError, match="HTTP 404"):
            update_lock(plan, cache)
        assert path.read_text() == before
        assert not list((tmp_path / "cache/temporary").iterdir())


def test_publication_lock_cancellation_never_changes_previous_bytes(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights", "alternate.data": b"w"}
    with _server(_publication_card(), files) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache")
        document = cache.read_document(base + "/model.json", offline=False)
        path = tmp_path / "model.json.lock"
        path.write_text("previous bytes")
        plan = plan_lock(
            document, ModelCard.parse(document.text), cache, lock_location=str(path)
        )
        cache.cancel()
        with pytest.raises(FetchError, match="cancelled"):
            update_lock(plan, cache)
        assert path.read_text() == "previous bytes"
        assert requests == ["/model.json"]


def test_publication_preflight_checks_every_source_against_the_lock_path(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights", "alternate.data": b"w"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    document = Document(str(tmp_path / "model.json"), json.dumps(_publication_card()))
    for name in ("code.loom", "weights.data", "alternate.data", "model.json"):
        with pytest.raises(ManifestError, match="overlaps a source file"):
            plan_lock(
                document,
                ModelCard.parse(document.text),
                ContentCache(tmp_path / "cache"),
                lock_location=str(tmp_path / name),
                offline=True,
            )
    for name, content in files.items():
        assert (tmp_path / name).read_bytes() == content


def test_publication_cli_dry_run_and_offline_exact_update(tmp_path, capsys):
    card = tmp_path / "model.json"
    card.write_text(json.dumps(_publication_card()))
    for name in ("code.loom", "weights.data", "alternate.data"):
        (tmp_path / name).write_bytes(name.encode())
    arguments = [
        str(card),
        "--update-lock",
        "--lock-scope=all",
        "--all-variants",
        "--offline",
        "--json",
    ]
    assert main([*arguments, "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary["files"]) == 3
    assert not (tmp_path / "model.json.lock").exists()
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["lockfile"] == str(card) + ".lock"
    lock = ModelLock.parse((tmp_path / "model.json.lock").read_text())
    assert len(lock.files) == 3
    assert json.loads(card.read_text()) == _publication_card()


def test_publication_program_filter_locks_every_compatible_variant(tmp_path, capsys):
    authored = _publication_card()
    authored["programs"]["alternate"]["inputs"] = ["unbuilt.loombc"]
    authored["variants"]["compiled-only"] = {
        "services": {"text": {"programs": ["alternate"]}}
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(authored))
    for name in ("code.loom", "weights.data", "alternate.data"):
        (tmp_path / name).write_bytes(name.encode())
    lock_path = tmp_path / "model.json.lock"
    lock_path.write_text(json.dumps(_lock({"obsolete.loombc": b"old output"})))
    assert (
        main(
            [
                str(path),
                "--update-lock",
                "--all-variants",
                "--programs=p",
                "--lock-scope=all",
                "--offline",
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["selections"] == [
        {"variant": name, "service": "text", "programs": "p"}
        for name in ("v", "unused")
    ]
    lock = ModelLock.parse(lock_path.read_text())
    assert len(lock.files) == 3
    cache = ContentCache(tmp_path / "client-cache")
    document = cache.read_document(str(path), offline=True)
    card = ModelCard.parse(document.text)
    for variant, expected in (("v", b"weights.data"), ("unused", b"alternate.data")):
        selected = card.select(variant=variant, service="text", programs="p")
        plan = plan_fetch(document, selected, cache, offline=True)
        destination = tmp_path / variant
        install(plan, cache, destination)
        assert (destination / "weights.data").read_bytes() == expected
        assert (destination / "code.loom").read_bytes() == b"code.loom"
    assert json.loads(path.read_text()) == authored


@pytest.mark.parametrize("programs", ["unknown", "unreferenced"])
def test_publication_program_filter_rejects_empty_selection_before_io(
    tmp_path, programs
):
    authored = _publication_card()
    authored["programs"]["unreferenced"] = {
        "inputs": ["absent.loombc"],
        "libraries": [],
    }
    with _server(authored, {}) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache")
        document = cache.read_document(base + "/model.json", offline=False)
        with pytest.raises(ManifestError, match="no compatible choices"):
            plan_lock(
                document,
                ModelCard.parse(document.text),
                cache,
                programs=programs,
                lock_location=str(tmp_path / "model.json.lock"),
            )
        assert requests == ["/model.json"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--all-variants"],
        ["--all-variants", "--no-lock"],
        ["--all-variants", "--update-lock", "--variant=v"],
        ["--all-variants", "--update-lock", "--partition=node-0"],
        ["--all-variants", "--update-lock", "--output=installed"],
        ["--all-variants", "--update-lock", "--list"],
        ["--all-variants", "--update-lock", "--materialize=copy"],
        ["--all-variants", "--update-lock", "--materialize=share"],
    ],
)
def test_publication_cli_rejects_ambiguous_modes_before_io(arguments):
    with pytest.raises(SystemExit) as error:
        main(["unopened.json", *arguments])
    assert error.value.code == 2


def test_remote_publication_requires_local_lock_destination_before_payload_io(tmp_path):
    with _server(_publication_card(), {}) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache")
        document = cache.read_document(base + "/model.json", offline=False)
        with pytest.raises(ManifestError, match="local --lockfile"):
            plan_lock(document, ModelCard.parse(document.text), cache)
        assert requests == ["/model.json"]


def test_cancelled_operation_does_not_publish_an_installation(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_card(), files) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache")
        plan = _remote_plan(base, cache)
        before = list(requests)
        cache.cancel()
        with pytest.raises(FetchError, match="cancelled"):
            install(plan, cache, tmp_path / "output")
        assert requests == before
        assert not (tmp_path / "output").exists()


def test_parallel_verified_install_and_offline_reuse(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    output = tmp_path / "first"
    files = {"code.loom": b"code", "weights.data": b"weights", "large.data": b"unused"}
    with _server(_card(), files, parallel=True) as (base, requests, _):
        plan = _remote_plan(base, cache)
        result = install(plan, cache, output)
        assert result == output / "model.json"
        assert sorted(requests) == [
            "/code.loom",
            "/model.json",
            "/model.json.lock",
            "/weights.data",
        ]
    offline = _remote_plan(base, cache, offline=True)
    install(offline, cache, tmp_path / "second")
    assert os.path.samefile(output / "weights.data", tmp_path / "second/weights.data")
    # The installed document retains the original HTTP coordinates, and its
    # selected lock is a usable projection of the publication's superset lock.
    installed = ModelCard.parse(result.read_text()).select()
    locked = ModelLock.parse((output / "model.json.lock").read_text())
    assert len(locked.cover(installed)) == 2
    assert installed.files["code.loom"].origin == "@document/code.loom"
    assert installed.document["origins"]["document"]["url"] == base + "/"
    assert len(locked.files) == 2
    legacy = json.loads((output / "model.json.lock").read_text())
    legacy["schema_version"] = 1
    legacy.pop("scope")
    (output / "model.json.lock").write_text(json.dumps(legacy))
    install(offline, cache, output)


@pytest.mark.parametrize("remote", [False, True])
def test_sibling_programs_follow_snapshot_and_survive_relocation(tmp_path, remote):
    card = _card()
    source = "../../../program/shared.loom"
    card["programs"]["p"]["inputs"] = [
        {"path": "library/shared.loom", "origin": source}
    ]
    directory = "snapshot/model/publisher/example/"
    files = {
        directory + "model.json": json.dumps(card).encode(),
        directory + "model.json.lock": json.dumps(
            _lock({source: b"shared code", "weights.data": b"weights"})
        ).encode(),
        directory + "weights.data": b"weights",
        "snapshot/program/shared.loom": b"shared code",
    }
    cache = ContentCache(tmp_path / "cache")
    with ExitStack() as stack:
        if remote:
            base, requests, resources = stack.enter_context(_server({}, {}))
            resources.update({"/" + name: content for name, content in files.items()})
            location = base + "/" + directory + "model.json"
        else:
            for name, content in files.items():
                path = tmp_path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            location = str(tmp_path / directory / "model.json")
        document = cache.read_document(location, offline=not remote)
        plan = plan_fetch(document, ModelCard.parse(document.text).select(), cache)
        output = tmp_path / "installed"
        result = install(plan, cache, output)
        assert (output / "library/shared.loom").read_bytes() == b"shared code"
        assert (output / "weights.data").read_bytes() == b"weights"
        if remote:
            assert sorted(requests) == sorted("/" + name for name in files)

    installed = cache.read_document(str(result), offline=True)
    selected = ModelCard.parse(installed.text).select()
    locked = ModelLock.parse((output / "model.json.lock").read_text())
    assert len(locked.cover(selected)) == 2
    assert selected.files["library/shared.loom"].origin == "@document/" + source
    offline = plan_fetch(installed, selected, cache, offline=True)
    second = tmp_path / "refetched"
    install(offline, cache, second)
    assert (second / "library/shared.loom").read_bytes() == b"shared code"
    assert (second / "weights.data").read_bytes() == b"weights"


def test_tampered_code_does_not_publish_partial_installation(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        _,
        resources,
    ):
        resources["/code.loom"] = b"evil"
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        with pytest.raises(ManifestError, match="SHA-256"):
            install(plan, cache, output)
        assert not (output / "model.json").exists()
        assert not (output / "code.loom").exists()
        assert not (output / "weights.data").exists()


def test_conflicting_output_fails_before_payload_request(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        (output / "code.loom").write_bytes(b"mine")
        with pytest.raises(ManifestError, match="integrity mismatch"):
            install(plan, cache, output)
        assert requests == ["/model.json", "/model.json.lock"]
        assert (output / "code.loom").read_bytes() == b"mine"


def test_dry_run_does_not_create_output_or_read_payloads(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        check_output(plan, tmp_path / "output")
        assert not (tmp_path / "output").exists()
        assert requests == ["/model.json", "/model.json.lock"]
        assert {item["size"] for item in plan.summary()["files"]} == {4, 7}


def test_conflicting_installation_lock_fails_in_plan_and_execution(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        (output / "model.json.lock").write_text(json.dumps(_lock({"other": b"other"})))
        for operation in (
            check_output,
            lambda plan, output: install(plan, cache, output),
        ):
            with pytest.raises(FetchError, match="conflicting installation document"):
                operation(plan, output)
        assert requests == ["/model.json", "/model.json.lock"]


def test_no_lock_neither_reads_nor_writes_any_lock(tmp_path):
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(_card()))
    source_lock = tmp_path / "card.json.lock"
    source_lock.write_text("deliberately invalid lock")
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    document = Document(str(card_path), card_path.read_text())
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(
        document,
        ModelCard.parse(document.text).select(),
        cache,
        no_lock=True,
        offline=True,
    )
    install(plan, cache, tmp_path / "output")
    assert source_lock.read_text() == "deliberately invalid lock"
    assert not (tmp_path / "output/model.json.lock").exists()
    assert os.path.samefile(tmp_path / "weights.data", tmp_path / "output/weights.data")


def test_update_replaces_exact_set_without_accumulating_records(tmp_path):
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(_card()))
    lock_path = tmp_path / "card.json.lock"
    lock_path.write_text(json.dumps(_lock({"old": b"old", "large.data": b"large"})))
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    document = Document(str(card_path), card_path.read_text())
    plan = plan_fetch(
        document,
        ModelCard.parse(document.text).select(),
        cache,
        update_lock=True,
        offline=True,
    )
    install(plan, cache, tmp_path / "output")
    updated = json.loads(lock_path.read_text())
    assert updated["scope"] == "assets"
    assert {record["origin"] for record in updated["files"]} == {"weights.data"}
    assert updated["origins"] == {}


def test_asset_lock_update_never_reads_programs_or_retains_their_hashes(
    tmp_path, capsys
):
    document = _card()
    document["origins"] = {
        "library": {"kind": "hf", "repository": "fixture/library", "revision": "main"}
    }
    document["programs"]["p"]["libraries"] = [
        {"path": "absent.loombc", "origin": "@library/absent.loombc"}
    ]
    card = tmp_path / "model.json"
    card.write_text(json.dumps(document))
    for name in ("weights.data", "large.data"):
        (tmp_path / name).write_bytes(name.encode())
    arguments = [
        str(card),
        "--update-lock",
        "--all-variants",
        "--offline",
        "--json",
        "--cache=" + str(tmp_path / "cache"),
    ]
    assert main(arguments) == 0, capsys.readouterr().err
    lock_path = tmp_path / "model.json.lock"
    original = lock_path.read_bytes()
    locked = ModelLock.parse(original.decode())
    assert locked.scope == "assets"
    assert locked.origins == {}
    assert {file.origin for file in locked.files.values()} == {
        "weights.data",
        "large.data",
    }
    document["programs"]["p"]["inputs"].append("another-absent.loom")
    card.write_text(json.dumps(document))
    assert main(arguments) == 0, capsys.readouterr().err
    assert lock_path.read_bytes() == original


def test_asset_lock_allows_aliased_source_edits_and_offline_reuse(tmp_path):
    card = _card()
    card["origins"] = {
        "library": {"kind": "url", "url": "https://example.org/library/"}
    }
    card["programs"]["p"]["origin"] = "@library/"
    card["programs"]["p"]["libraries"] = [
        {"path": "copy.loom", "origin": "@library/code.loom"}
    ]
    files = {"code.loom": b"code", "weights.data": b"weights"}
    cache = ContentCache(tmp_path / "cache")
    with _server(card, files) as (base, requests, resources):
        asset_lock = _lock({"weights.data": b"weights"})
        asset_lock["scope"] = "assets"
        resources["/model.json.lock"] = json.dumps(asset_lock).encode()
        original_lock = resources["/model.json.lock"]
        overrides = {"library": base}
        plan = _remote_plan(base, cache, overrides=overrides)
        assert set(plan.identities) == {"weights.data"}
        first = tmp_path / "first"
        second = tmp_path / "second"
        install(plan, cache, first)
        installed_lock = (first / "model.json.lock").read_bytes()
        (first / "code.loom").write_bytes(b"locally edited")
        (first / "copy.loom").unlink()
        install(plan, cache, first)
        assert (first / "code.loom").read_bytes() == b"locally edited"
        assert (first / "copy.loom").read_bytes() == b"code"
        assert (first / "model.json.lock").read_bytes() == installed_lock
        install(plan, cache, second)
        assert (second / "code.loom").read_bytes() == b"code"
        assert os.path.samefile(first / "weights.data", second / "weights.data")
        assert resources["/model.json.lock"] == original_lock
        assert requests.count("/weights.data") == 1
        resources["/code.loom"] = b"upstream edit"
        install(plan, cache, tmp_path / "third")
        assert (tmp_path / "third/code.loom").read_bytes() == b"upstream edit"
        assert (first / "code.loom").read_bytes() == b"locally edited"
        before = list(requests)
    offline = _remote_plan(base, cache, overrides=overrides, offline=True)
    install(offline, cache, tmp_path / "offline")
    assert requests == before
    assert (tmp_path / "offline/code.loom").read_bytes() == b"upstream edit"
    assert os.path.samefile(first / "weights.data", tmp_path / "offline/weights.data")


def test_program_only_asset_lock_is_empty_and_repeatable(tmp_path):
    document = _card()
    document["variants"]["v"]["services"]["text"].pop("parameters")
    (tmp_path / "code.loom").write_bytes(b"code")
    lock = ModelLock.snapshot({}, [], scope="assets")
    (tmp_path / "model.json.lock").write_text(json.dumps(lock))
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(
        Document(str(tmp_path / "model.json"), json.dumps(document)),
        ModelCard(document).select(),
        cache,
        offline=True,
    )
    assert plan.summary()["lock"] == "verify"
    install(plan, cache, tmp_path / "output")
    (tmp_path / "output/code.loom").write_bytes(b"edited")
    install(plan, cache, tmp_path / "output")
    assert (tmp_path / "output/code.loom").read_bytes() == b"edited"
    assert json.loads((tmp_path / "output/model.json.lock").read_text()) == lock


def test_override_retains_logical_hf_identity_in_installed_card(tmp_path):
    card = _card()
    card["origins"] = {
        "weights": {"kind": "hf", "repository": "org/gated", "revision": "a" * 40}
    }
    card["variants"]["v"]["services"]["text"]["parameters"]["scope"]["origin"] = (
        "@weights/"
    )
    (tmp_path / "code.loom").write_bytes(b"code")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "weights.data").write_bytes(b"weights")
    document = Document(str(tmp_path / "card.json"), json.dumps(card))
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(
        document,
        ModelCard(card).select(),
        cache,
        overrides={"weights": str(mirror)},
        offline=True,
        update_lock=True,
    )
    result = install(plan, cache, tmp_path / "output")
    installed = ModelCard.parse(result.read_text()).select()
    assert installed.document["origins"]["weights"] == card["origins"]["weights"]
    assert installed.files["weights.data"].origin == "@weights/weights.data"
    lock = ModelLock.parse((result.parent / "model.json.lock").read_text())
    assert lock.origins["weights"]["resolved"] == {"revision": "a" * 40}
    lock.cover(installed)


def test_parent_symlink_cannot_redirect_installation(tmp_path):
    card = _card()
    card["programs"]["p"]["inputs"] = ["program/code.loom"]
    cache = ContentCache(tmp_path / "cache")
    with _server(card, {"program/code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (output / "program").symlink_to(outside, target_is_directory=True)
        with pytest.raises(FetchError, match="ordinary directory"):
            install(plan, cache, output)
        assert requests == ["/model.json", "/model.json.lock"]
        assert list(outside.iterdir()) == []


def test_copy_is_an_explicit_independent_materialization(tmp_path):
    card = _card()
    document = Document(str(tmp_path / "card.json"), json.dumps(card))
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(document, ModelCard(card).select(), cache, no_lock=True)
    install(plan, cache, tmp_path / "output", materialize="copy")
    assert not os.path.samefile(
        tmp_path / "weights.data", tmp_path / "output/weights.data"
    )
    (tmp_path / "weights.data").write_bytes(b"changed")
    assert (tmp_path / "output/weights.data").read_bytes() == b"weights"


@pytest.mark.parametrize("materialize", [None, "auto", "share", "copy"])
def test_program_edits_are_independent_by_default_while_assets_share(
    tmp_path, materialize
):
    card = _card()
    card["programs"]["p"]["inputs"] = [{"path": "code.loom", "data": ["source.inc"]}]
    card["programs"]["p"]["libraries"] = ["library.loombc"]
    files = {
        "code.loom": b"code",
        "source.inc": b"auxiliary",
        "library.loombc": b"library",
        "weights.data": b"weights",
    }
    cache = ContentCache(tmp_path / "cache")
    with _server(card, files) as (base, requests, _):
        plan = _remote_plan(base, cache)
        options = {} if materialize is None else {"materialize": materialize}
        first = tmp_path / "first"
        second = tmp_path / "second"
        install(plan, cache, first, **options)
        install(plan, cache, second, **options)
        for name, content in files.items():
            assert requests.count("/" + name) == 1
            cached = cache.blob_path(hashlib.sha256(content).hexdigest())
            shared = materialize == "share" or (
                materialize != "copy" and name == "weights.data"
            )
            assert os.path.samefile(first / name, cached) == shared
            assert os.path.samefile(second / name, cached) == shared
            (first / name).write_bytes(b"edited")
            expected = b"edited" if shared else content
            assert (second / name).read_bytes() == expected
            assert cached.read_bytes() == expected


def test_cli_default_copies_programs_from_a_local_checkout(tmp_path, capsys):
    (tmp_path / "model.json").write_text(json.dumps(_card()))
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    output = tmp_path / "output"
    arguments = [
        str(tmp_path / "model.json"),
        "--no-lock",
        "--offline",
        "--no-input",
        "--cache=" + str(tmp_path / "cache"),
        "--output=" + str(output),
    ]
    assert main(arguments) == 0, capsys.readouterr().err
    (output / "code.loom").write_bytes(b"edited")
    assert (tmp_path / "code.loom").read_bytes() == b"code"
    assert os.path.samefile(output / "weights.data", tmp_path / "weights.data")
    assert main(arguments) == 0, capsys.readouterr().err
    assert (output / "code.loom").read_bytes() == b"edited"


def test_shared_logical_source_is_downloaded_only_once(tmp_path):
    card = _card()
    card["programs"]["p"]["libraries"] = [{"path": "copy.loom", "origin": "code.loom"}]
    cache = ContentCache(tmp_path / "cache")
    with _server(card, {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        install(_remote_plan(base, cache), cache, tmp_path / "output")
        assert requests.count("/code.loom") == 1
        assert not os.path.samefile(
            tmp_path / "output/code.loom", tmp_path / "output/copy.loom"
        )
        assert (tmp_path / "output/code.loom").read_bytes() == b"code"
        assert (tmp_path / "output/copy.loom").read_bytes() == b"code"


def test_voice_recording_lock_and_fetch_preserve_original_bytes_and_content_type(
    tmp_path,
):
    document = _card()
    document["variants"]["v"]["services"]["text"]["voices"] = {
        "narrator": {
            "file": {"path": "voice.data", "origin": "original.pcm"},
            "content_type": "audio/pcm;rate=44100;channels=2;format=f32le",
        }
    }
    for name, payload in [
        ("code.loom", b"code"),
        ("weights.data", b"weights"),
        ("original.pcm", b"original recording"),
    ]:
        (tmp_path / name).write_bytes(payload)
    card = ModelCard(document)
    cache = ContentCache(tmp_path / "cache")
    source = Document(str(tmp_path / "model.json"), json.dumps(document))
    plan = plan_fetch(source, card.select(), cache, update_lock=True)
    output = tmp_path / "installed"
    install(plan, cache, output)
    selected = ModelCard.parse((output / "model.json").read_text()).select()
    assert (
        selected.voices["narrator"]["content_type"]
        == "audio/pcm;rate=44100;channels=2;format=f32le"
    )
    assert (output / "voice.data").read_bytes() == b"original recording"
    assert selected.files["voice.data"].roles == frozenset({"voice"})
    lock = ModelLock.parse((output / "model.json.lock").read_text())
    lock.cover(selected)
