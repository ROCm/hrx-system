# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import hashlib
import http.server
import json
import os
import re
import struct
import threading

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch import install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_hf import HuggingFace
from loom.model.fetch_source import FetchError, Source
from loom.model.fetch_storage import ContentCache
from loom.model.hf_test.hub import running_hub
from loom.model.lock import FileIdentity, ModelLock


def _card():
    return {
        "schema_version": 1,
        "model": {"id": "fixture/fetch", "name": "HF fetch fixture"},
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
            }
        },
        "presets": {"check": {"variant": "v", "service": "text", "programs": "p"}},
    }


def _publication(hub):
    files = {
        path: FileIdentity(path, len(content), hashlib.sha256(content).hexdigest())
        for path, content in hub.files.items()
    }
    hub.files["model.json"] = json.dumps(_card()).encode()
    hub.files["model.json.lock"] = json.dumps(
        ModelLock.snapshot({}, list(files.values()))
    ).encode()


def _cache(root, hub, wait=None):
    return ContentCache(
        root / "cache",
        huggingface=HuggingFace(
            endpoint=hub.endpoint, cache_dir=hub.cache_dir, wait_for_access=wait
        ),
    )


def test_range_redirect_strips_credentials_and_never_persists_signed_url(tmp_path):
    header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
    content = struct.pack("<Q", len(header)) + header + b"abcd"
    credentials = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            credentials.append(self.headers.get("Authorization") is not None)
            first, last = map(
                int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups()
            )
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(content)}")
            self.send_header("Content-Length", str(last - first + 1))
            self.end_headers()
            self.wfile.write(content[first : last + 1])

        def log_message(self, *_):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with running_hub(tmp_path, {"weights.data": content}) as hub:
                hub.approved = True
                hub.credentials.write_text("fixture-first")
                hub.redirects["weights.data"] = (
                    f"http://127.0.0.1:{server.server_port}/payload?signature=private-fixture-value"
                )
                cache = _cache(tmp_path, hub)
                source = Source("weights.data", hub.source("weights.data"))
                result = cache.acquire_transformed(
                    source,
                    {"kind": "safetensors", "version": 1, "alignment": 4096},
                    None,
                    offline=False,
                    preserve_sources=True,
                )
                assert result.identity.size == 4100
                assert credentials == [False, False, False]
                receipt = next((cache.root / "preserved").iterdir()).read_text()
                assert "signature" not in receipt and "fixture-first" not in receipt
        finally:
            server.shutdown()
            thread.join()


def test_streaming_transform_reuses_hub_auth_without_downloading_parent(tmp_path):
    unused = 64 * 1024
    header = json.dumps(
        {
            "unused": {"dtype": "U8", "shape": [unused], "data_offsets": [0, unused]},
            "bias": {
                "dtype": "I32",
                "shape": [1],
                "data_offsets": [unused, unused + 4],
            },
        },
        separators=(",", ":"),
    ).encode()
    content = (
        struct.pack("<Q", len(header)) + header + b"x" * unused + struct.pack("<i", 2)
    )
    with running_hub(tmp_path, {"code.loom": b"code", "weights.data": content}) as hub:
        hub.approved = True
        hub.credentials.write_text("fixture-first")
        authored = _card()
        authored["origins"] = {
            "checkpoint": {
                "kind": "hf",
                "repository": hub.repository,
                "revision": "main",
            }
        }
        authored["programs"]["p"]["origin"] = "@checkpoint/"
        scope = authored["variants"]["v"]["services"]["text"]["parameters"]["scope"]
        scope["origin"] = "@checkpoint/"
        scope["files"] = [
            {
                "path": "weights.data",
                "format": "safetensors",
                "transform": {
                    "kind": "safetensors",
                    "version": 1,
                    "alignment": 4096,
                    "tensors": ["bias"],
                },
            }
        ]
        card_path = tmp_path / "model.json"
        card_path.write_text(json.dumps(authored))
        cache = _cache(tmp_path, hub)
        document = cache.read_document(str(card_path), offline=False)
        selection = ModelCard.parse(document.text).select()
        plan = plan_fetch(document, selection, cache, update_lock=True)
        install(plan, cache, tmp_path / "first", preserve_sources=True)
        assert hub.range_requests == [
            ("weights.data", 0, 7),
            ("weights.data", 8, 7 + len(header)),
            ("weights.data", len(content) - 4, len(content) - 1),
        ]
        assert cache.huggingface.cached_file(hub.source("weights.data")) is None
        result = (tmp_path / "first/weights.data").read_bytes()
        assert len(result) == 4100 and result[-4:] == struct.pack("<i", 2)
        count = len(hub.requests)
        offline = plan_fetch(document, selection, cache, offline=True)
        install(offline, cache, tmp_path / "second", preserve_sources=True)
        assert len(hub.requests) == count
        assert os.path.samefile(
            tmp_path / "first/weights.data", tmp_path / "second/weights.data"
        )
        hub.files["weights.data"] = content[:-4] + struct.pack("<i", 3)
        fresh = ContentCache(tmp_path / "fresh", huggingface=cache.huggingface)
        plan = plan_fetch(document, selection, fresh)
        with pytest.raises(ManifestError, match="transformed parameter"):
            install(plan, fresh, tmp_path / "rejected")
        assert not (tmp_path / "rejected/model.json").exists()
        # The developer exception still resolves the publisher's immutable
        # source revision before dropping only the expected parameter bytes.
        developer = _cache(tmp_path / "developer", hub)
        plan = plan_fetch(document, selection, developer, no_verify_parameters=True)
        assert plan.sources["weights.data"].transport["revision"] == hub.revision
        assert plan.unverified_parameters == {"weights.data"}
        install(plan, developer, tmp_path / "experiment")
        assert (tmp_path / "experiment/weights.data").read_bytes()[-4:] == struct.pack(
            "<i", 3
        )
        receipt = json.loads((tmp_path / "experiment/.loom-fetch.json").read_text())
        assert receipt["unverified_parameters"] == ["weights.data"]


def test_publication_resolves_one_snapshot_and_checks_each_shared_source_once(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.approved = True
        hub.credentials.write_text("fixture-first")
        hub.files["alternate.data"] = b"other weights"
        authored = _card()
        authored["origins"] = {
            "checkpoint": {
                "kind": "hf",
                "repository": hub.repository,
                "revision": "main",
            }
        }
        authored["programs"]["p"]["origin"] = "@checkpoint/"
        authored["programs"]["alias"] = {
            "inputs": [{"path": "renamed.loom", "origin": "@checkpoint/code.loom"}],
            "libraries": [],
        }
        service = authored["variants"]["v"]["services"]["text"]
        service["programs"].append("alias")
        service["parameters"]["scope"]["origin"] = "@checkpoint/"
        authored["variants"]["alternate"] = copy.deepcopy(authored["variants"]["v"])
        alternate = authored["variants"]["alternate"]["services"]["text"]
        alternate["parameters"]["scope"]["files"] = [
            {"path": "weights.data", "origin": "@checkpoint/alternate.data"}
        ]
        card_path = tmp_path / "model.json"
        card_path.write_text(json.dumps(authored))
        cache = _cache(tmp_path, hub)
        document = cache.read_document(str(card_path), offline=False)
        card = ModelCard.parse(document.text)
        plan = plan_lock(document, card, cache)
        assert len(plan.selections) == 4
        assert len(plan.sources) == 3
        assert sum("/revision/main" in path for _, path in hub.requests) == 1
        for name in ("code.loom", "weights.data", "alternate.data"):
            assert (
                hub.requests.count(
                    ("HEAD", f"/fixture/model/resolve/{hub.revision}/{name}")
                )
                == 1
            )
        update_lock(plan, cache)
        for name in ("code.loom", "weights.data", "alternate.data"):
            assert (
                hub.requests.count(
                    ("GET", f"/fixture/model/resolve/{hub.revision}/{name}")
                )
                == 1
            )
        lock = ModelLock.parse(plan.path.read_text())
        assert lock.origins["checkpoint"]["resolved"] == {"revision": hub.revision}
        before = list(hub.requests)
        selected = plan_fetch(document, card.select("check"), cache, offline=True)
        installed = install(selected, cache, tmp_path / "selected")
        assert hub.requests == before
        assert not (installed.parent / "alternate.data").exists()
        cached = cache.huggingface.cached_file(hub.source("weights.data"))
        assert (installed.parent / "weights.data").samefile(cached)


def test_hf_document_pins_its_closure_and_reuses_shared_cache(tmp_path):
    with running_hub(tmp_path) as hub:
        _publication(hub)
        hub.credentials.write_text("fixture-first")
        actions = []

        def wait(action, delay, cancelled):
            actions.append(action)
            hub.approved = True

        cache = _cache(tmp_path, hub, wait)
        url = hub.endpoint + "/fixture/model/blob/main/model.json?download=true"
        document = cache.read_document(url, offline=False)
        assert document.location == (
            hub.endpoint + f"/fixture/model/resolve/{hub.revision}/model.json"
        )
        selection = ModelCard.parse(document.text).select()
        plan = plan_fetch(document, selection, cache)
        assert not any(
            method == "GET" and path.endswith(("code.loom", "weights.data"))
            for method, path in hub.requests
        )
        first = install(plan, cache, tmp_path / "first")
        assert len(actions) == 1
        assert not (cache.root / "blob").exists()
        for filename in ("code.loom", "weights.data"):
            cached = cache.huggingface.cached_file(hub.source(filename))
            assert os.path.samefile(first.parent / filename, cached)
        before = list(hub.requests)
        hub.approved = False
        # A fresh operation keeps the document's resolved snapshot while offline.
        cache = _cache(tmp_path, hub)
        offline_document = cache.read_document(url, offline=True)
        assert offline_document == document
        offline = plan_fetch(offline_document, selection, cache, offline=True)
        second = install(offline, cache, tmp_path / "second")
        assert os.path.samefile(
            first.parent / "weights.data", second.parent / "weights.data"
        )
        assert hub.requests == before
        assert sum("/revision/main" in path for _, path in hub.requests) == 1


def test_hf_origin_update_records_revision_and_only_selected_files(tmp_path):
    with running_hub(tmp_path) as hub:
        hub.approved = True
        hub.credentials.write_text("fixture-first")
        card = _card()
        root = {"kind": "hf", "repository": hub.repository, "revision": "main"}
        card["origins"] = {"hub": root}
        card["programs"]["p"]["origin"] = "@hub/"
        card["variants"]["v"]["services"]["text"]["parameters"]["scope"]["origin"] = (
            "@hub/"
        )
        hub.files["unused.data"] = b"unused"
        authored = tmp_path / "card.json"
        authored.write_text(json.dumps(card))
        cache = _cache(tmp_path, hub)
        document = cache.read_document(str(authored), offline=False)
        selection = ModelCard.parse(document.text).select()
        plan = plan_fetch(document, selection, cache, update_lock=True)
        installed = install(plan, cache, tmp_path / "installed")
        lock = ModelLock.parse((tmp_path / "card.json.lock").read_text())
        assert lock.origins == {
            "hub": {"requested": root, "resolved": {"revision": hub.revision}}
        }
        assert len(lock.cover(selection)) == 2
        assert not any("unused.data" in path for _, path in hub.requests)
        assert (installed.parent / "weights.data").read_bytes() == b"weights"


def test_hf_mismatched_size_fails_before_weight_transfer(tmp_path):
    with running_hub(tmp_path) as hub:
        _publication(hub)
        hub.approved = True
        hub.credentials.write_text("fixture-first")
        hub.files["weights.data"] = b"wrong size"
        cache = _cache(tmp_path, hub)
        document = cache.read_document(
            hub.endpoint + "/fixture/model/resolve/main/model.json", offline=False
        )
        with pytest.raises(FetchError, match="size does not match"):
            plan_fetch(document, ModelCard.parse(document.text).select(), cache)
        assert not any(
            method == "GET" and path.endswith("weights.data")
            for method, path in hub.requests
        )


def test_hf_cached_content_is_verified_before_publication(tmp_path):
    with running_hub(tmp_path) as hub:
        _publication(hub)
        hub.approved = True
        hub.credentials.write_text("fixture-first")
        cache = _cache(tmp_path, hub)
        document = cache.read_document(
            hub.endpoint + "/fixture/model/resolve/main/model.json", offline=False
        )
        cached = cache.huggingface.download(hub.source("weights.data"), offline=False)
        cached.write_bytes(b"changed")
        plan = plan_fetch(document, ModelCard.parse(document.text).select(), cache)
        with pytest.raises(ManifestError, match="integrity mismatch"):
            install(plan, cache, tmp_path / "output")
        assert not (tmp_path / "output/model.json").exists()


def test_hf_offline_document_miss_never_checks_access(tmp_path):
    with running_hub(tmp_path) as hub:
        cache = _cache(tmp_path, hub)
        with pytest.raises(FetchError, match="--offline"):
            cache.read_document(
                hub.endpoint + f"/fixture/model/resolve/{hub.revision}/model.json",
                offline=True,
            )
        assert hub.requests == []
