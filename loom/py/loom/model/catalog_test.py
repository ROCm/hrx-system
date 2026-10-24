# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
from pathlib import Path

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.catalog import ModelCatalog, generate
from loom.model.fetch_catalog import load_catalog
from loom.model.fetch_source import Document, FetchError
from loom.model.fetch_storage import ContentCache


def _card():
    return ModelCard.parse(
        (Path(__file__).parent / "testdata/moonshine.json").read_text()
    )


def _catalog():
    return generate({"model/moonshine/model.json": _card()}).document


class Documents:
    """Network/document-cache dependency with controlled GitHub responses."""

    def __init__(self, root):
        self.root = root
        self.remote = {}
        self.cached = {}
        self.requests = []

    def read_document(self, location, *, offline):
        if offline:
            if location not in self.cached:
                raise FetchError("--offline document is not cached")
            return self.cached[location]
        self.requests.append(location)
        if location not in self.remote:
            raise FetchError("source returned HTTP 404")
        document = Document(location, json.dumps(self.remote[location]))
        self.cached[location] = document
        return document


_REFERENCE = "https://api.github.com/repos/fixture/library/git/ref/tags/compatible"
_RAW = "https://raw.githubusercontent.com/fixture/library/"
_SOURCE = "github:fixture/library@compatible"


def _publish(documents, revision, catalog=None):
    documents.remote[_REFERENCE] = {
        "ref": "refs/tags/compatible",
        "object": {"type": "commit", "sha": revision},
    }
    documents.remote[_RAW + revision + "/catalog.json"] = catalog or _catalog()
    documents.remote[_RAW + revision + "/model/moonshine/model.json"] = _card().document


def test_generator_reads_model_facts_and_not_parameters():
    document = _catalog()
    entry = document["models"]["moonshine-ai/moonshine-streaming-tiny"]
    assert entry["card"] == "model/moonshine/model.json"
    assert entry["services"] == {"transcription": {"kind": "recognition"}}
    assert entry["presets"] == ["streaming"]
    assert "parameters" not in entry
    with pytest.raises(ManifestError, match="duplicate catalog model"):
        generate({"a.json": _card(), "b.json": _card()})


def test_catalog_owns_its_input_and_exposed_document():
    document = _catalog()
    catalog = ModelCatalog(document)
    document["models"].clear()
    catalog.document["models"].clear()
    assert (
        catalog.card_path("moonshine-ai/moonshine-streaming-tiny")
        == "model/moonshine/model.json"
    )


@pytest.mark.parametrize(
    "path",
    [
        "../card.json",
        "/card.json",
        "https://example.com/card.json",
        "a/../card.json",
        "C:\\card.json",
    ],
)
def test_catalog_cards_stay_inside_the_publication(path):
    document = _catalog()
    next(iter(document["models"].values()))["card"] = path
    with pytest.raises(ManifestError, match="relative path"):
        ModelCatalog(document)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_unknown_catalog_versions_are_rejected(version):
    document = _catalog()
    document["schema_version"] = version
    with pytest.raises(ManifestError, match="unsupported model-catalog"):
        ModelCatalog(document)


def test_local_catalog_discovery_does_not_read_cards(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog()))
    cache = ContentCache(tmp_path / "cache")
    snapshot = load_catalog(str(path), cache, offline=True)
    assert snapshot.revision is None
    assert snapshot.catalog.document == _catalog()
    assert not (tmp_path / "cache").exists()


def test_moved_tag_changes_next_discovery_without_changing_current_selection(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    first = load_catalog(_SOURCE, documents, offline=False)
    assert documents.requests == [_REFERENCE, _RAW + "a" * 40 + "/catalog.json"]
    added = _catalog()
    added["models"]["fixture/another"] = copy.deepcopy(
        next(iter(added["models"].values()))
    )
    _publish(documents, "b" * 40, added)
    second = load_catalog(_SOURCE, documents, offline=False)
    assert second.revision == "b" * 40
    assert "fixture/another" in second.catalog.document["models"]
    assert "fixture/another" not in first.catalog.document["models"]
    document, _ = first.read_model(
        "moonshine-ai/moonshine-streaming-tiny", documents, offline=False
    )
    assert document.location == _RAW + "a" * 40 + "/model/moonshine/model.json"
    requests = list(documents.requests)
    assert load_catalog(_SOURCE, documents, offline=True).revision == "b" * 40
    assert documents.requests == requests


def test_failed_refresh_preserves_previous_complete_offline_snapshot(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    load_catalog(_SOURCE, documents, offline=False)
    _publish(documents, "b" * 40, {"schema_version": 55, "models": {}})
    with pytest.raises(ManifestError, match="unsupported model-catalog"):
        load_catalog(_SOURCE, documents, offline=False)
    assert load_catalog(_SOURCE, documents, offline=True).revision == "a" * 40
    del documents.remote[_REFERENCE]
    with pytest.raises(FetchError, match="404"):
        load_catalog(_SOURCE, documents, offline=False)


def test_annotated_tags_are_peeled_without_changing_repository(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "c" * 40)
    documents.remote[_REFERENCE]["object"] = {
        "type": "tag",
        "sha": "a" * 40,
        "url": "https://untrusted.example/tag",
    }
    tag_root = "https://api.github.com/repos/fixture/library/git/tags/"
    documents.remote[tag_root + "a" * 40] = {
        "sha": "a" * 40,
        "object": {"type": "tag", "sha": "b" * 40},
    }
    documents.remote[tag_root + "b" * 40] = {
        "sha": "b" * 40,
        "object": {"type": "commit", "sha": "c" * 40},
    }
    assert load_catalog(_SOURCE, documents, offline=False).revision == "c" * 40
    assert documents.requests == [
        _REFERENCE,
        tag_root + "a" * 40,
        tag_root + "b" * 40,
        _RAW + "c" * 40 + "/catalog.json",
    ]


@pytest.mark.parametrize(
    "target",
    [{}, {"type": "tree", "sha": "a" * 40}, {"type": "commit", "sha": "../bad"}],
)
def test_invalid_reference_objects_do_not_select_another_ref(tmp_path, target):
    documents = Documents(tmp_path)
    documents.remote[_REFERENCE] = {"ref": "refs/tags/compatible", "object": target}
    with pytest.raises(FetchError):
        load_catalog(_SOURCE, documents, offline=False)
    assert documents.requests == [_REFERENCE]


def test_missing_offline_snapshot_performs_no_requests(tmp_path):
    documents = Documents(tmp_path)
    with pytest.raises(FetchError, match="no cached snapshot"):
        load_catalog(_SOURCE, documents, offline=True)
    assert documents.requests == []


def test_reference_response_must_match_the_requested_tag(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    documents.remote[_REFERENCE]["ref"] = "refs/heads/main"
    with pytest.raises(FetchError, match="different catalog reference"):
        load_catalog(_SOURCE, documents, offline=False)
    assert documents.requests == [_REFERENCE]


def test_cyclic_annotated_tag_is_rejected(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    documents.remote[_REFERENCE]["object"]["type"] = "tag"
    location = "https://api.github.com/repos/fixture/library/git/tags/" + "a" * 40
    documents.remote[location] = {
        "sha": "a" * 40,
        "object": {"type": "tag", "sha": "a" * 40},
    }
    with pytest.raises(FetchError, match="does not resolve to a commit"):
        load_catalog(_SOURCE, documents, offline=False)
    assert documents.requests == [_REFERENCE, location]


def test_offline_evicted_document_does_not_refresh_the_tag(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    load_catalog(_SOURCE, documents, offline=False)
    requests = list(documents.requests)
    documents.cached.clear()
    with pytest.raises(FetchError, match="not cached"):
        load_catalog(_SOURCE, documents, offline=True)
    assert documents.requests == requests


def test_unpinned_remote_catalog_is_not_treated_as_a_snapshot(tmp_path):
    documents = Documents(tmp_path)
    with pytest.raises(ManifestError, match="catalog discovery requires"):
        load_catalog("https://example.com/catalog.json", documents, offline=False)
    assert documents.requests == []


def test_explicit_commit_does_not_resolve_a_tag(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    source = "github:fixture/library@" + "a" * 40
    assert load_catalog(source, documents, offline=False).revision == "a" * 40
    assert documents.requests == [_RAW + "a" * 40 + "/catalog.json"]


def test_selected_card_identity_is_verified(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    document = documents.remote[_RAW + "a" * 40 + "/model/moonshine/model.json"]
    document["model"]["id"] = "fixture/different"
    snapshot = load_catalog(_SOURCE, documents, offline=False)
    with pytest.raises(ManifestError, match="identity differs"):
        snapshot.read_model(
            "moonshine-ai/moonshine-streaming-tiny", documents, offline=False
        )
