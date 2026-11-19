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
from loom.model.catalog import main as generate_catalog
from loom.model.fetch_publication import load_publication
from loom.model.fetch_source import Document, FetchError, SourceNotFound
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
            raise SourceNotFound("source returned HTTP 404")
        if isinstance(self.remote[location], Exception):
            raise self.remote[location]
        document = Document(location, json.dumps(self.remote[location]))
        self.cached[location] = document
        return document


_REFERENCE = (
    "https://api.github.com/repos/fixture/library/commits/compatible?per_page=1"
)
_RAW = "https://raw.githubusercontent.com/fixture/library/"
_SOURCE = "github:fixture/library@compatible"


def _publish(documents, revision, catalog=None):
    documents.remote[_REFERENCE] = {"sha": revision}
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


def test_regenerating_catalog_retains_annotations_on_current_entries(tmp_path):
    card = _card().document
    card["model"]["x-label"] = ["speech"]
    card["services"]["transcription"]["x-icon"] = "microphone"
    identity = card["model"]["id"]
    previous = _catalog()
    previous["x-index"] = {"language": "en"}
    entry = previous["models"][identity]
    entry["x-rank"] = 1
    entry["services"]["transcription"]["x-display"] = True
    previous["models"]["removed"] = entry
    index = tmp_path / "catalog.json"
    index.write_text(json.dumps(previous))
    path = tmp_path / "model.json"
    path.write_text(json.dumps(card))
    assert (
        generate_catalog(["--card", f"model.json={path}", "--output", str(index)]) == 0
    )
    result = ModelCatalog.parse(index.read_text()).document
    assert result["x-index"] == {"language": "en"}
    assert set(result["models"]) == {identity}
    entry = result["models"][identity]
    assert entry["x-label"] == ["speech"]
    assert entry["x-rank"] == 1
    assert entry["services"]["transcription"] == {
        "kind": "recognition",
        "x-icon": "microphone",
        "x-display": True,
    }


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
    snapshot = load_publication(str(path), cache, offline=True, catalog=True)
    assert snapshot.revision is None
    assert snapshot.content.document == _catalog()
    assert not (tmp_path / "cache").exists()


def test_moved_tag_changes_next_discovery_without_changing_current_selection(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    first = load_publication(_SOURCE, documents, offline=False, catalog=True)
    assert documents.requests == [_REFERENCE, _RAW + "a" * 40 + "/catalog.json"]
    added = _catalog()
    added["models"]["fixture/another"] = copy.deepcopy(
        next(iter(added["models"].values()))
    )
    _publish(documents, "b" * 40, added)
    second = load_publication(_SOURCE, documents, offline=False, catalog=True)
    assert second.revision == "b" * 40
    assert "fixture/another" in second.content.document["models"]
    assert "fixture/another" not in first.content.document["models"]
    document, _ = first.read_model(
        "moonshine-ai/moonshine-streaming-tiny", documents, offline=False
    )
    assert document.location == _RAW + "a" * 40 + "/model/moonshine/model.json"
    requests = list(documents.requests)
    assert (
        load_publication(_SOURCE, documents, offline=True, catalog=True).revision
        == "b" * 40
    )
    assert documents.requests == requests


def test_failed_refresh_preserves_previous_complete_offline_snapshot(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    load_publication(_SOURCE, documents, offline=False, catalog=True)
    _publish(documents, "b" * 40, {"schema_version": 55, "models": {}})
    with pytest.raises(ManifestError, match="unsupported model-catalog"):
        load_publication(_SOURCE, documents, offline=False, catalog=True)
    assert (
        load_publication(_SOURCE, documents, offline=True, catalog=True).revision
        == "a" * 40
    )
    del documents.remote[_REFERENCE]
    with pytest.raises(FetchError, match="404"):
        load_publication(_SOURCE, documents, offline=False, catalog=True)


def test_default_branch_is_resolved_without_assuming_its_name(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "c" * 40)
    location = "https://api.github.com/repos/fixture/library/commits?per_page=1"
    documents.remote[location] = [{"sha": "c" * 40}]
    assert (
        load_publication(
            "github:fixture/library", documents, offline=False, catalog=True
        ).revision
        == "c" * 40
    )
    assert documents.requests == [
        location,
        _RAW + "c" * 40 + "/catalog.json",
    ]


@pytest.mark.parametrize(
    "target",
    [{}, {"sha": None}, {"sha": "../bad"}, [], {"sha": "a" * 39}],
)
def test_invalid_reference_objects_do_not_select_another_ref(tmp_path, target):
    documents = Documents(tmp_path)
    documents.remote[_REFERENCE] = target
    with pytest.raises(FetchError):
        load_publication(_SOURCE, documents, offline=False, catalog=True)
    assert documents.requests == [_REFERENCE]


def test_missing_offline_snapshot_performs_no_requests(tmp_path):
    documents = Documents(tmp_path)
    with pytest.raises(FetchError, match="no cached snapshot"):
        load_publication(_SOURCE, documents, offline=True, catalog=True)
    assert documents.requests == []


def test_root_card_precedes_catalog_and_reads_only_its_snapshot(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    location = _RAW + "a" * 40 + "/model.json"
    documents.remote[location] = _card().document
    snapshot = load_publication(_SOURCE, documents, offline=False)
    assert isinstance(snapshot.content, ModelCard)
    assert documents.requests == [_REFERENCE, location]
    _publish(documents, "b" * 40)
    document, card = snapshot.read_model(
        _card().document["model"]["id"], documents, offline=False
    )
    assert document.location == location
    assert card.document == _card().document
    requests = list(documents.requests)
    assert (
        load_publication(_SOURCE, documents, offline=True).document.location == location
    )
    assert documents.requests == requests


def test_root_catalog_discovery_is_remembered_offline(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    snapshot = load_publication(_SOURCE, documents, offline=False)
    assert isinstance(snapshot.content, ModelCatalog)
    assert documents.requests == [
        _REFERENCE,
        _RAW + "a" * 40 + "/model.json",
        _RAW + "a" * 40 + "/catalog.json",
    ]
    requests = list(documents.requests)
    documents.remote.clear()
    assert (
        load_publication(_SOURCE, documents, offline=True).revision == snapshot.revision
    )
    assert documents.requests == requests


@pytest.mark.parametrize(
    "failure",
    [FetchError("HTTP 403"), FetchError("transport failed"), {"schema_version": 99}],
)
def test_only_genuine_card_absence_can_discover_a_catalog(tmp_path, failure):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    location = _RAW + "a" * 40 + "/model.json"
    documents.remote[location] = failure
    with pytest.raises((FetchError, ManifestError)):
        load_publication(_SOURCE, documents, offline=False)
    assert documents.requests == [_REFERENCE, location]
    with pytest.raises(FetchError, match="no cached snapshot"):
        load_publication(_SOURCE, documents, offline=True)


def test_missing_repository_is_not_treated_as_missing_card(tmp_path):
    documents = Documents(tmp_path)
    source = "github:fixture/library@" + "a" * 40
    with pytest.raises(SourceNotFound):
        load_publication(source, documents, offline=False)
    assert documents.requests == [
        "https://api.github.com/repos/fixture/library/commits/"
        + "a" * 40
        + "?per_page=1"
    ]


def test_missing_entry_points_have_a_direct_diagnostic(tmp_path):
    documents = Documents(tmp_path)
    documents.remote[_REFERENCE] = {"sha": "a" * 40}
    with pytest.raises(
        FetchError, match=r"neither a root model\.json nor catalog\.json"
    ):
        load_publication(_SOURCE, documents, offline=False)


def test_revision_slashes_are_not_publication_directories(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    location = (
        "https://api.github.com/repos/fixture/library/commits/release%2Fv1?per_page=1"
    )
    documents.remote[location] = {"sha": "a" * 40}
    snapshot = load_publication(
        "github:fixture/library:streaming@release/v1",
        documents,
        offline=False,
        catalog=True,
    )
    assert snapshot.preset == "streaming"
    assert documents.requests == [location, _RAW + "a" * 40 + "/catalog.json"]


def test_offline_evicted_document_does_not_refresh_the_tag(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    load_publication(_SOURCE, documents, offline=False, catalog=True)
    requests = list(documents.requests)
    documents.cached.clear()
    with pytest.raises(FetchError, match="not cached"):
        load_publication(_SOURCE, documents, offline=True, catalog=True)
    assert documents.requests == requests


def test_unpinned_remote_catalog_is_not_treated_as_a_snapshot(tmp_path):
    documents = Documents(tmp_path)
    with pytest.raises(ManifestError, match="catalog discovery requires"):
        load_publication(
            "https://example.com/catalog.json", documents, offline=False, catalog=True
        )
    assert documents.requests == []


def test_explicit_commit_does_not_resolve_a_tag(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    source = "github:fixture/library@" + "a" * 40
    assert (
        load_publication(source, documents, offline=False, catalog=True).revision
        == "a" * 40
    )
    assert documents.requests == [_RAW + "a" * 40 + "/catalog.json"]


def test_selected_card_identity_is_verified(tmp_path):
    documents = Documents(tmp_path)
    _publish(documents, "a" * 40)
    document = documents.remote[_RAW + "a" * 40 + "/model/moonshine/model.json"]
    document["model"]["id"] = "fixture/different"
    snapshot = load_publication(_SOURCE, documents, offline=False, catalog=True)
    with pytest.raises(ManifestError, match="identity differs"):
        snapshot.read_model(
            "moonshine-ai/moonshine-streaming-tiny", documents, offline=False
        )
