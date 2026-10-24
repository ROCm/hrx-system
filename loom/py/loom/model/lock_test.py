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
