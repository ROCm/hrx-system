# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selected output commitments, independent of placement and runtime settings."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .manifest import (
    ManifestError,
    Selection,
    _json_value,
    _mapping,
    _object,
    _origin,
    _origin_object,
    _string,
    _transform,
    parse_json,
)


def file_key(origin: str | dict, transform: dict | None = None) -> str:
    """Logical output identity, independent of placement and transport override."""
    value = {"origin": origin, "transform": transform} if transform else origin
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_digest(path: Path, *, cancelled: Event | None = None) -> tuple[int, str]:
    """Hashes bytes with bounded working storage, including empty inputs."""
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("file verification cancelled")
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("file verification cancelled")
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _resolved(value: dict, path: str) -> None:
    _object(value, path, ("revision",))
    revision = _string(value["revision"], path + ".revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ManifestError(f"{path}.revision: expected an immutable repository commit")


@dataclass(frozen=True)
class FileIdentity:
    # Logical authored origin, independent of destination and effective mirror.
    origin: str | dict
    # Exact output byte count, after any declared acquisition transform.
    size: int
    # Lowercase SHA-256 of the entire file.
    sha256: str
    # Immutable revision for an explicitly described repository file, if any.
    resolved: dict | None = None
    # Normalized operation producing these bytes from the authored origin.
    transform: dict | None = None

    def verify(self, path: Path, *, cancelled: Event | None = None) -> None:
        size, digest = file_digest(path, cancelled=cancelled)
        if size != self.size or digest != self.sha256:
            raise ManifestError(
                f"integrity mismatch for {path}: expected {self.size} bytes / "
                f"sha256 {self.sha256}, got {size} bytes / sha256 {digest}"
            )

    def document(self) -> dict:
        result = {
            "origin": copy.deepcopy(self.origin),
            "size": self.size,
            "sha256": self.sha256,
        }
        if self.resolved is not None:
            result["resolved"] = copy.deepcopy(self.resolved)
        if self.transform is not None:
            result["transform"] = copy.deepcopy(self.transform)
        return result


class ModelLock:
    """An owned publication lock; selection never mutates or expands it."""

    def __init__(self, document: dict):
        doc = copy.deepcopy(document)
        _json_value(doc, "lock")
        _mapping(doc, "lock")
        if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1:
            raise ManifestError(
                "lock.schema_version: unsupported lock version; supported: 1"
            )
        _object(doc, "lock", ("schema_version", "origins", "files"), ("$schema",))
        if "$schema" in doc:
            _string(doc["$schema"], "lock.$schema")
        self.origins = _mapping(doc["origins"], "lock.origins", empty=True)
        for name, record in self.origins.items():
            path = "lock.origins." + name
            _object(record, path, ("requested",), ("resolved",))
            _origin_object(record["requested"], path + ".requested", root=True)
            if "resolved" in record:
                _resolved(record["resolved"], path + ".resolved")
                if record["requested"]["kind"] != "hf":
                    raise ManifestError(
                        f"{path}: only repository origins have revisions"
                    )
            elif record["requested"]["kind"] == "hf":
                raise ManifestError(
                    f"{path}: repository origin requires a resolved revision"
                )
        if not isinstance(doc["files"], list):
            raise ManifestError("lock.files: expected an array")
        self.files: dict[str, FileIdentity] = {}
        for index, record in enumerate(doc["files"]):
            path = f"lock.files[{index}]"
            _object(
                record, path, ("origin", "size", "sha256"), ("resolved", "transform")
            )
            transform = (
                _transform(record["transform"], path + ".transform")
                if "transform" in record
                else None
            )
            _origin(record["origin"], self.origins, path + ".origin")
            if type(record["size"]) is not int or not 0 <= record["size"] < 2**64:
                raise ManifestError(
                    f"{path}.size: expected an unsigned 64-bit byte count"
                )
            digest = _string(record["sha256"], path + ".sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ManifestError(
                    f"{path}.sha256: expected a lowercase SHA-256 digest"
                )
            repository_file = (
                isinstance(record["origin"], dict)
                and record["origin"].get("kind") == "hf"
            )
            if "resolved" in record:
                _resolved(record["resolved"], path + ".resolved")
                if not repository_file:
                    raise ManifestError(f"{path}: only repository files have revisions")
            elif repository_file:
                raise ManifestError(
                    f"{path}: repository file requires a resolved revision"
                )
            key = file_key(record["origin"], transform)
            if key in self.files:
                raise ManifestError(f"{path}: duplicate logical output")
            self.files[key] = FileIdentity(
                record["origin"],
                record["size"],
                digest,
                record.get("resolved"),
                transform,
            )

    @classmethod
    def parse(cls, source: str) -> ModelLock:
        return cls(parse_json(source))

    def cover(self, selection: Selection) -> dict[str, FileIdentity]:
        """Checks complete selected coverage before any source payload is read."""
        roots = selection.document.get("origins", {})
        covered = {}
        for name, file in selection.files.items():
            if isinstance(file.origin, str) and file.origin.startswith("@"):
                alias = file.origin[1:].split("/", 1)[0]
                if (
                    alias not in self.origins
                    or self.origins[alias]["requested"] != roots[alias]
                ):
                    raise ManifestError(
                        f"lock does not match origin {alias!r}; "
                        "use --update-lock or --no-lock"
                    )
            identity = self.files.get(file_key(file.origin, file.transform))
            if identity is None:
                raise ManifestError(
                    f"lock does not cover {name!r}; use --update-lock or --no-lock"
                )
            covered[name] = identity
        return covered

    @staticmethod
    def snapshot(origins: dict, identities: list[FileIdentity]) -> dict:
        """Writes exactly this verified set, with no historical accumulation."""
        files = {}
        used_origins = set()
        for identity in identities:
            key = file_key(identity.origin, identity.transform)
            if key in files and files[key] != identity:
                raise ManifestError(
                    "selected sources have conflicting content identities"
                )
            files[key] = identity
            if isinstance(identity.origin, str) and identity.origin.startswith("@"):
                used_origins.add(identity.origin[1:].split("/", 1)[0])
        return {
            "schema_version": 1,
            "origins": {
                name: copy.deepcopy(origins[name]) for name in sorted(used_origins)
            },
            "files": [files[key].document() for key in sorted(files)],
        }
