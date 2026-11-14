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
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .derived import derived_files, is_loader_derived, parameter_label
from .manifest import (
    ManifestError,
    ModelFile,
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


def lock_files(selection: Selection, scope: str) -> dict[str, ModelFile]:
    """Selects required commitments by declared use, independent of file format."""
    if scope == "all":
        return selection.files
    if scope != "assets":
        raise ManifestError("lock scope must be 'assets' or 'all'")
    assets = {
        file_key(file.origin, file.transform)
        for file in selection.files.values()
        if file.roles - {"program", "auxiliary"}
    }
    # A logical source used as an asset remains protected at every destination,
    # including an additional program or auxiliary use of the same bytes.
    return {
        name: file
        for name, file in selection.files.items()
        if file_key(file.origin, file.transform) in assets
    }


def file_digest(
    path: Path, *, cancelled: Event | None = None, progress=None
) -> tuple[int, str]:
    """Hashes bytes with bounded working storage, including empty inputs."""
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("file verification cancelled")
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        total = os.fstat(stream.fileno()).st_size if progress is not None else None
        if progress is not None:
            progress(0, total)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("file verification cancelled")
            size += len(chunk)
            digest.update(chunk)
            if progress is not None:
                progress(size, total)
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

    def verify(
        self, path: Path, *, cancelled: Event | None = None, progress=None
    ) -> None:
        size, digest = file_digest(path, cancelled=cancelled, progress=progress)
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


@dataclass(frozen=True)
class DerivedIdentity:
    """A loader-derived output: committed recipe and source, optional bytes."""

    # Destination path; acquisition never transfers derived bytes.
    origin: str | dict
    # Card transform implemented by the native runtime loader.
    transform: dict
    # Source origin and loader recipe identifiers; None in records without them.
    derived: dict | None = None
    # Expected payload bytes when a publisher observed them, otherwise None.
    size: int | None = None
    # Lowercase SHA-256 of those payload bytes, present exactly with size.
    sha256: str | None = None

    def document(self) -> dict:
        result = {"origin": copy.deepcopy(self.origin)}
        if self.size is not None:
            result["size"] = self.size
            result["sha256"] = self.sha256
        result["transform"] = copy.deepcopy(self.transform)
        if self.derived is not None:
            result["derived"] = copy.deepcopy(self.derived)
        return result


def _byte_identity(record: dict, path: str) -> tuple[int, str]:
    if type(record["size"]) is not int or not 0 <= record["size"] < 2**64:
        raise ManifestError(f"{path}.size: expected an unsigned 64-bit byte count")
    digest = _string(record["sha256"], path + ".sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ManifestError(f"{path}.sha256: expected a lowercase SHA-256 digest")
    return record["size"], digest


def _derived_identity(record: dict, origins: dict, path: str) -> DerivedIdentity:
    _object(record, path, ("origin", "transform"), ("size", "sha256", "derived"))
    transform = _transform(record["transform"], path + ".transform")
    _origin(record["origin"], origins, path + ".origin")
    size = digest = None
    if "size" in record or "sha256" in record:
        _object(record, path, ("origin", "transform", "size", "sha256"), ("derived",))
        size, digest = _byte_identity(record, path)
    derived = None
    if "derived" in record:
        derived = _object(
            record["derived"],
            path + ".derived",
            ("source", "layout_version", "recipe_version"),
        )
        _origin(derived["source"], origins, path + ".derived.source")
        for key in ("layout_version", "recipe_version"):
            _string(derived[key], f"{path}.derived.{key}")
    return DerivedIdentity(
        record["origin"], transform, copy.deepcopy(derived), size, digest
    )


class ModelLock:
    """Explicit asset or complete coverage; selection never expands the lock."""

    def __init__(self, document: dict):
        doc = copy.deepcopy(document)
        _json_value(doc, "lock")
        _mapping(doc, "lock")
        version = doc.get("schema_version")
        if type(version) is not int or version not in (1, 2):
            raise ManifestError(
                "lock.schema_version: unsupported lock version; supported: 1, 2"
            )
        required = ("schema_version", "origins", "files")
        _object(
            doc, "lock", required + (("scope",) if version == 2 else ()), ("$schema",)
        )
        # Version 1 committed to every file. Only an explicit version 2 scope
        # changes coverage; a missing identity never implies editable content.
        self.scope = doc["scope"] if version == 2 else "all"
        if self.scope not in ("assets", "all"):
            raise ManifestError("lock.scope: expected 'assets' or 'all'")
        if "$schema" in doc:
            _string(doc["$schema"], "lock.$schema")
        self.origins = _mapping(doc["origins"], "lock.origins", empty=True)
        requested_origins = {}
        for name, record in self.origins.items():
            path = "lock.origins." + name
            _object(record, path, ("requested",), ("resolved",))
            _origin_object(record["requested"], path + ".requested", root=True)
            requested_origins[name] = record["requested"]
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
        self.files: dict[str, FileIdentity | DerivedIdentity] = {}
        for index, record in enumerate(doc["files"]):
            path = f"lock.files[{index}]"
            if isinstance(record, dict) and is_loader_derived(record.get("transform")):
                identity = _derived_identity(record, requested_origins, path)
                key = file_key(identity.origin, identity.transform)
                if key in self.files:
                    raise ManifestError(f"{path}: duplicate logical output")
                self.files[key] = identity
                continue
            _object(
                record, path, ("origin", "size", "sha256"), ("resolved", "transform")
            )
            transform = (
                _transform(record["transform"], path + ".transform")
                if "transform" in record
                else None
            )
            _origin(record["origin"], requested_origins, path + ".origin")
            _, digest = _byte_identity(record, path)
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

    def cover(self, selection: Selection) -> dict[str, FileIdentity | DerivedIdentity]:
        """Checks complete coverage within scope before reading source payloads.

        A loader-derived output is covered by its recipe record. Its source must
        be committed too, because the loader keys its cache by the source bytes.
        """
        roots = selection.document.get("origins", {})
        files = lock_files(selection, self.scope)
        derived = {
            name: output
            for name, output in derived_files(selection).items()
            if name in files
        }
        for name, output in derived.items():
            source = selection.files[output.source]
            if file_key(source.origin, source.transform) not in self.files:
                raise ManifestError(
                    f"lock does not cover {parameter_label(selection, output.source)}, "
                    f"the source the native loader derives {name!r} from; "
                    "use --update-lock or --no-lock"
                )
        covered = {}
        for name, file in files.items():
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
            if name in derived:
                expected = derived[name].facts()
                if identity.derived is not None and identity.derived != expected:
                    raise ManifestError(
                        f"lock records derivation {identity.derived} for {name!r}, "
                        f"but the native loader derives {expected}; "
                        "use --update-lock"
                    )
            covered[name] = identity
        return covered

    @staticmethod
    def snapshot(
        origins: dict,
        identities: list[FileIdentity | DerivedIdentity],
        *,
        scope: str = "all",
    ) -> dict:
        """Writes exactly this verified set, with no historical accumulation."""
        if scope not in ("assets", "all"):
            raise ManifestError("lock.scope: expected 'assets' or 'all'")
        files = {}
        used_origins = set()
        for identity in identities:
            key = file_key(identity.origin, identity.transform)
            if key in files and files[key] != identity:
                raise ManifestError(
                    "selected sources have conflicting content identities"
                )
            files[key] = identity
            references = [identity.origin]
            if isinstance(identity, DerivedIdentity) and identity.derived is not None:
                references.append(identity.derived["source"])
            for origin in references:
                if isinstance(origin, str) and origin.startswith("@"):
                    used_origins.add(origin[1:].split("/", 1)[0])
        return {
            "schema_version": 2,
            "scope": scope,
            "origins": {
                name: copy.deepcopy(origins[name]) for name in sorted(used_origins)
            },
            "files": [files[key].document() for key in sorted(files)],
        }
