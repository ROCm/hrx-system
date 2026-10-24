# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from loom.model.fetch_hf import AccessRequired, HuggingFace, hf_url_source
from loom.model.fetch_source import FetchError
from loom.model.hf_test.hub import running_hub


def test_gate_refreshes_credentials_and_preserves_selected_revision(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        actions = []

        def wait(action, delay, cancelled):
            actions.append(action.document())
            assert delay == 5
            assert not cancelled.is_set()
            hub.credentials.write_text("fixture-second")
            hub.approved = True

        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        )
        requested = {"repository": hub.repository, "revision": "main"}
        assert provider.resolve_revision(requested, offline=False) == hub.revision
        assert provider.resolve_revision(requested, offline=False) == hub.revision
        path = provider.download(hub.source(), offline=False)
        assert path.read_bytes() == b"code"
        assert actions[0]["reason"] == "gate"
        assert actions[0]["account"] == "first"
        assert actions[0]["url"] == hub.endpoint + "/fixture/model"
        assert "fixture-first" not in json.dumps(actions)
        assert "fixture-second" not in json.dumps(actions)
        assert sum("/revision/main" in path for _, path in hub.requests) == 1
        before = list(hub.requests)
        hub.approved = False
        assert provider.download(hub.source(), offline=True) == path
        assert hub.requests == before


def test_headless_gate_returns_action_without_polling_or_payload(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        with pytest.raises(AccessRequired) as denied:
            provider.download(hub.source(), offline=False)
        assert denied.value.reason == "gate"
        assert hub.requests == [
            ("GET", "/api/models/fixture/model/auth-check"),
            ("GET", "/api/whoami-v2"),
        ]
        assert not hub.cache_dir.exists()


@pytest.mark.parametrize(
    ("status", "reason"),
    [(401, "credentials"), (403, "permissions"), (429, "rate_limit")],
)
def test_generic_http_failures_are_not_approval_gates(tmp_path, status, reason):
    with running_hub(tmp_path) as hub:
        hub.denial = status, ""
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        with pytest.raises(AccessRequired) as denied:
            provider.ensure_access(hub.repository)
        assert denied.value.reason == reason
        assert len(hub.requests) == 1


def test_inaccessible_repository_keeps_its_ambiguity(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.denial = 404, "RepoNotFound"
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        with pytest.raises(FetchError, match="missing or inaccessible"):
            provider.ensure_access(hub.repository)
        assert len(hub.requests) == 1


def test_invalid_environment_credential_is_identified_without_disclosure(tmp_path):
    with running_hub(tmp_path) as hub:
        os.environ["HF_TOKEN"] = "fixture-invalid"
        hub.credentials.write_text("fixture-first")
        actions = []

        def wait(action, delay, cancelled):
            actions.append(action.document())
            assert action.reason == "credentials"
            assert action.credential_source == "HF_TOKEN"
            os.environ["HF_TOKEN"] = ""
            hub.approved = True

        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        )
        provider.ensure_access(hub.repository)
        assert len(actions) == 1
        assert "fixture-invalid" not in json.dumps(actions)


def test_concurrent_shards_share_one_access_coordinator(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        actions = []

        def wait(action, delay, cancelled):
            actions.append(action)
            hub.approved = True

        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(provider.download, hub.source(name), offline=False)
                for name in hub.files
            ]
            assert [future.result().read_bytes() for future in futures] == [
                b"code",
                b"weights",
            ]
        assert len(actions) == 1
        assert hub.requests.count(("GET", "/api/models/fixture/model/auth-check")) == 2


def test_cancelled_gate_stops_before_payload(tmp_path):
    with running_hub(tmp_path) as hub:

        def wait(action, delay, cancelled):
            cancelled.set()

        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        )
        with pytest.raises(FetchError, match="cancelled"):
            provider.download(hub.source(), offline=False)
        assert hub.requests == [("GET", "/api/models/fixture/model/auth-check")]


@pytest.mark.parametrize("second_reader", [False, True])
def test_cancel_releases_body_and_sdk_cache_waiters(tmp_path, second_reader):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        hub.approved = True
        hub.held_file = "code.loom"
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        second_metadata = threading.Event()

        def observe(method, path):
            if method == "HEAD" and sum(m == "HEAD" for m, _ in hub.requests) == 2:
                second_metadata.set()

        hub.request_observer = observe
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                first = executor.submit(provider.download, hub.source(), offline=False)
                first.add_done_callback(lambda _: hub.body_started.set())
                hub.body_started.wait()
                assert not first.done(), first.result()
                futures = [first]
                if second_reader:
                    second = executor.submit(
                        provider.download, hub.source(), offline=False
                    )
                    second.add_done_callback(lambda _: second_metadata.set())
                    second_metadata.wait()
                    assert not second.done(), second.result()
                    futures.append(second)
                provider.cancelled.set()
                for future in futures:
                    with pytest.raises(FetchError, match="cancelled"):
                        future.result()
                assert not hub.release_body.is_set()
                assert provider.cached_file(hub.source()) is None
            finally:
                provider.cancelled.set()
                hub.release_body.set()
        # A new operation can reuse the exact SDK cache after its writers stop.
        replacement = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        path = replacement.download(hub.source(), offline=False)
        assert path.read_bytes() == b"code"


def test_transfer_process_preserves_gate_semantics(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        hub.approved = True
        actions = []

        def revoke(method, path):
            if method == "HEAD":
                hub.approved = False
                hub.request_observer = None

        def approve(action, delay, cancelled):
            actions.append(action.document())
            hub.approved = True

        hub.request_observer = revoke
        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=approve
        )
        assert provider.download(hub.source(), offline=False).read_bytes() == b"code"
        assert len(actions) == 1
        assert actions[0]["reason"] == "gate"
        assert actions[0]["account"] == "first"
        assert "fixture-first" not in json.dumps(actions)


@pytest.mark.parametrize("operation", ["access", "account", "revision", "metadata"])
def test_cancel_stops_metadata_and_identity_requests(tmp_path, operation):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        hub.approved = operation != "account"
        started, release = threading.Event(), threading.Event()

        def hold(method, path):
            selected = {
                "access": path.endswith("/auth-check"),
                "account": path == "/api/whoami-v2",
                "revision": "/revision/main" in path,
                "metadata": method == "HEAD",
            }[operation]
            if selected:
                hub.allow_disconnect = True
                started.set()
                release.wait()

        hub.request_observer = hold
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        call = {
            "access": lambda: provider.ensure_access(hub.repository),
            "account": lambda: provider.ensure_access(hub.repository),
            "revision": lambda: provider.resolve_revision(
                {"repository": hub.repository, "revision": "main"}, offline=False
            ),
            "metadata": lambda: provider.metadata(hub.source()),
        }[operation]
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(call)
                future.add_done_callback(lambda _: started.set())
                started.wait()
                assert not future.done(), future.result()
                provider.cancelled.set()
                with pytest.raises(FetchError, match="cancelled"):
                    future.result()
                assert not release.is_set()
            finally:
                provider.cancelled.set()
                release.set()


def test_offline_miss_does_not_check_access(tmp_path):
    with running_hub(tmp_path) as hub:
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        with pytest.raises(FetchError, match="--offline"):
            provider.download(hub.source(), offline=True)
        with pytest.raises(FetchError, match="--offline"):
            provider.resolve_revision(
                {"repository": hub.repository, "revision": "main"}, offline=True
            )
        assert hub.requests == []


def test_concurrent_revocation_does_not_duplicate_approval_waits(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.credentials.write_text("fixture-first")
        hub.approved = True
        actions = []

        def wait(action, delay, cancelled):
            actions.append(action)
            hub.approved = True

        provider = HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        )
        provider.ensure_access(hub.repository)
        hub.approved = False
        hub.denial_barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(provider.metadata, hub.source(name))
                for name in hub.files
            ]
            assert [future.result().size for future in futures] == [4, 7]
        assert len(actions) == 1
        assert hub.requests.count(("GET", "/api/models/fixture/model/auth-check")) == 2


@pytest.mark.parametrize("date_header", [False, True])
def test_rate_limit_retains_provider_retry_guidance(tmp_path, date_header):
    with running_hub(tmp_path) as hub:
        hub.denial = 429, ""
        hub.retry_after = (
            format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)
            if date_header
            else "17"
        )
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        with pytest.raises(AccessRequired) as denied:
            provider.ensure_access(hub.repository)
        assert denied.value.reason == "rate_limit"
        if date_header:
            assert 3500 < denied.value.retry_after <= 3600
        else:
            assert denied.value.retry_after == 17


def test_url_recognition_keeps_credentials_on_the_configured_provider():
    recognized = hf_url_source(
        "https://huggingface.co/owner/model/blob/feature%2Fbranch/path/file.json?download=true"
    )
    assert recognized == (
        {
            "kind": "hf",
            "repository": "owner/model",
            "revision": "feature/branch",
            "path": "path/file.json",
        },
        "https://huggingface.co",
    )
    assert (
        hf_url_source("https://example.com/owner/model/resolve/main/model.json") is None
    )
    assert (
        hf_url_source("https://huggingface.co.evil/owner/model/resolve/main/model.json")
        is None
    )


def test_invalid_repository_fails_before_network_access(tmp_path):
    with running_hub(tmp_path) as hub:
        provider = HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir)
        source = {**hub.source(), "repository": "bad/identity/shape"}
        for operation in (
            lambda: provider.resolve_revision(source, offline=False),
            lambda: provider.download(source, offline=False),
            lambda: provider.ensure_access(source["repository"]),
        ):
            with pytest.raises(FetchError, match="repository identity"):
                operation()
        assert hub.requests == []
