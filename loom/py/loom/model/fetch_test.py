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
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch import check_output, install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_main import main
from loom.model.fetch_source import Document, FetchError
from loom.model.fetch_storage import ContentCache
from loom.model.lock import FileIdentity, ModelLock


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

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests, resources
        finally:
            server.shutdown()
            thread.join()


def _remote_plan(base, cache, **kwargs):
    document = cache.read_document(
        base + "/model.json", offline=kwargs.get("offline", False)
    )
    return plan_fetch(
        document, ModelCard.parse(document.text).select(), cache, **kwargs
    )


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
        cache = ContentCache(tmp_path / "cache")
        document = cache.read_document(base + "/model.json", offline=False)
        card = ModelCard.parse(document.text)
        lock_path = tmp_path / "publication.json.lock"
        lock_path.write_text('{"stale":"record"}')
        plan = plan_lock(document, card, cache, lock_location=str(lock_path))
        assert requests == ["/model.json"]
        assert len(plan.selections) == 4
        assert len(plan.sources) == 3
        assert update_lock(plan, cache) == lock_path
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
    arguments = [str(card), "--update-lock", "--all-variants", "--offline", "--json"]
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
    install(offline, cache, output)


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
    assert {record["origin"] for record in updated["files"]} == {
        "code.loom",
        "weights.data",
    }
    assert updated["origins"] == {}


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
        assert os.path.samefile(
            tmp_path / "output/code.loom", tmp_path / "output/copy.loom"
        )


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
