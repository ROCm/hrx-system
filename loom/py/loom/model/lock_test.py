# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import hashlib
import threading

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.lock import FileIdentity, ModelLock


def _card():
    return {
        "schema_version": 1,
        "model": {"id": "fixture/model", "name": "Fixture"},
        "origins": {
            "weights": {"kind": "hf", "repository": "org/model", "revision": "a" * 40}
        },
        "programs": {"mixed": {"inputs": ["prepare.loom"], "libraries": []}},
        "services": {"chat": {"kind": "text", "version": 1}},
        "variants": {
            "tiny": {
                "services": {
                    "chat": {
                        "programs": ["mixed"],
                        "parameters": {
                            "model": {"origin": "@weights/", "files": ["shard.bin"]}
                        },
                    }
                }
            }
        },
    }


def _lock():
    origins = {
        "weights": {
            "requested": _card()["origins"]["weights"],
            "resolved": {"revision": "a" * 40},
        }
    }
    identities = [
        FileIdentity("prepare.loom", 4, hashlib.sha256(b"code").hexdigest()),
        FileIdentity("@weights/shard.bin", 6, hashlib.sha256(b"weight").hexdigest()),
    ]
    return ModelLock.snapshot(origins, identities)


def test_matching_lock_covers_content_independently_of_names_and_config():
    card = _card()
    variant = card["variants"]["tiny"]["services"]["chat"]
    variant["config"] = {"width": 42}
    variant["parameters"]["model"]["files"] = [
        {"path": "renamed/weights.data", "origin": "@weights/shard.bin"}
    ]
    coverage = ModelLock(_lock()).cover(ModelCard(card).select())
    assert set(coverage) == {"prepare.loom", "renamed/weights.data"}
    assert coverage["renamed/weights.data"].size == 6


def test_snapshot_deduplicates_and_does_not_retain_other_variants():
    lock = _lock()
    lock["origins"]["unused"] = {
        "requested": {"kind": "url", "url": "https://example.org/"}
    }
    lock["files"].append(
        {"origin": "@unused/large.bin", "size": 400_000_000_000, "sha256": "0" * 64}
    )
    parsed = ModelLock(lock)
    coverage = parsed.cover(ModelCard(_card()).select())
    snapshot = ModelLock.snapshot(parsed.origins, list(coverage.values()) * 2)
    assert snapshot == _lock()
    assert len(parsed.files) == 3


@pytest.mark.parametrize("version", [1, 2])
def test_lock_keeps_opaque_recipe_payloads_distinct(version):
    lock = _lock()
    if version == 1:
        lock["schema_version"] = 1
        lock.pop("scope")
    records = [
        {
            "origin": "blob",
            "size": 1,
            "sha256": str(i) * 64,
            "transform": {"kind": "vendor.codec", "version": 2, "order": order},
        }
        for i, order in enumerate(([0, 1], [1, 0]))
    ]
    lock["files"].extend(records)
    parsed = ModelLock(lock)
    snapshot = ModelLock.snapshot(parsed.origins, list(parsed.files.values()))
    assert [item for item in snapshot["files"] if item["origin"] == "blob"] == records


def test_changed_authored_origin_is_not_a_transport_override():
    card = _card()
    card["origins"]["weights"]["repository"] = "different/model"
    with pytest.raises(ManifestError, match="does not match origin"):
        ModelLock(_lock()).cover(ModelCard(card).select())


def test_incomplete_lock_rejects_before_any_file_is_opened():
    lock = _lock()
    lock["files"].pop()
    with pytest.raises(ManifestError, match="does not cover"):
        ModelLock(lock).cover(ModelCard(_card()).select())


def test_asset_scope_excludes_local_and_aliased_programs_and_auxiliary_files():
    card = _card()
    card["origins"]["library"] = {
        "kind": "url",
        "url": "https://example.org/library/revision/",
    }
    program = card["programs"]["mixed"]
    program["inputs"] = [{"path": "prepare.loom", "data": ["header.h"]}]
    program["libraries"] = [
        {"path": "library.loombc", "origin": "@library/bundle.loombc"}
    ]
    document = _lock()
    document["scope"] = "assets"
    document["files"] = [
        file for file in document["files"] if file["origin"] != "prepare.loom"
    ]
    lock = ModelLock(document)
    assert set(lock.cover(ModelCard(card).select())) == {"shard.bin"}
    program["inputs"].append("new-program.loom")
    assert set(lock.cover(ModelCard(card).select())) == {"shard.bin"}
    document["files"] = []
    with pytest.raises(ManifestError, match=r"does not cover 'shard\.bin'"):
        ModelLock(document).cover(ModelCard(card).select())


def test_asset_used_as_program_at_another_destination_remains_locked():
    card = _card()
    card["programs"]["mixed"]["libraries"] = [
        {"path": "same.loom", "origin": "@weights/shard.bin"}
    ]
    document = _lock()
    document["scope"] = "assets"
    document["files"] = [
        file for file in document["files"] if file["origin"] != "prepare.loom"
    ]
    coverage = ModelLock(document).cover(ModelCard(card).select())
    assert set(coverage) == {"shard.bin", "same.loom"}
    assert coverage["shard.bin"] == coverage["same.loom"]


def test_version_one_always_requires_complete_program_coverage():
    document = _lock()
    document["schema_version"] = 1
    document.pop("scope")
    assert ModelLock(document).scope == "all"
    assert set(ModelLock(document).cover(ModelCard(_card()).select())) == {
        "prepare.loom",
        "shard.bin",
    }
    document["files"] = [
        file for file in document["files"] if file["origin"] != "prepare.loom"
    ]
    with pytest.raises(ManifestError, match=r"does not cover 'prepare\.loom'"):
        ModelLock(document).cover(ModelCard(_card()).select())


@pytest.mark.parametrize("scope", [None, "programs", True])
def test_version_two_requires_an_explicit_valid_scope(scope):
    document = _lock()
    if scope is None:
        document.pop("scope")
    else:
        document["scope"] = scope
    with pytest.raises(ManifestError, match="scope"):
        ModelLock(document)


def test_conflicting_logical_sources_are_rejected():
    identities = [FileIdentity("a", 1, "0" * 64), FileIdentity("a", 2, "1" * 64)]
    with pytest.raises(ManifestError, match="conflicting content"):
        ModelLock.snapshot({}, identities)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size", True),
        ("size", -1),
        ("size", 2**64),
        ("sha256", "0" * 63),
        ("sha256", "A" * 64),
    ],
)
def test_invalid_commitments_fail(field, value):
    lock = _lock()
    lock["files"][0][field] = value
    with pytest.raises(ManifestError):
        ModelLock(lock)


def test_duplicate_json_and_duplicate_outputs_fail():
    with pytest.raises(ManifestError, match="duplicate object key"):
        ModelLock.parse('{"schema_version":1,"schema_version":1}')
    lock = _lock()
    lock["files"].append(copy.deepcopy(lock["files"][0]))
    with pytest.raises(ManifestError, match="duplicate logical output"):
        ModelLock(lock)


def test_byte_verification_detects_same_length_substitution(tmp_path):
    path = tmp_path / "library.loombc"
    identity = FileIdentity("library.loombc", 4, hashlib.sha256(b"code").hexdigest())
    path.write_bytes(b"code")
    identity.verify(path)
    path.write_bytes(b"evil")
    with pytest.raises(ManifestError, match="integrity mismatch"):
        identity.verify(path)


def test_verification_observes_cancelled_operation(tmp_path):
    path = tmp_path / "weights.data"
    path.write_bytes(b"weights")
    cancelled = threading.Event()
    cancelled.set()
    identity = FileIdentity("weights.data", 7, hashlib.sha256(b"weights").hexdigest())
    with pytest.raises(InterruptedError, match="verification cancelled"):
        identity.verify(path, cancelled=cancelled)
