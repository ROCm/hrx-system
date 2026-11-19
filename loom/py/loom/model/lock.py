# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selected output commitments, independent of placement and runtime settings."""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

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
    operation_fields,
    parse_json,
    project_fields,
)
from .manifest import (
    file_key as file_key,
)


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


def _check_resolution(record: dict, origin: str | dict, path: str) -> None:
    kind = origin.get("kind") if isinstance(origin, dict) else None
    if kind == "hf":
        if "resolved" not in record:
            raise ManifestError(
                f"{path}: repository origin requires a resolved revision"
            )
        value = _mapping(record["resolved"], path + ".resolved")
        revision = _string(value.get("revision"), path + ".resolved.revision")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ManifestError(
                f"{path}.resolved.revision: expected an immutable repository commit"
            )
    elif "resolved" in record:
        if kind in (None, "local", "url", "inline"):
            raise ManifestError(
                f"{path}: this origin does not have a resolved revision"
            )
        _mapping(record["resolved"], path + ".resolved", empty=True)


def require_supported_revision(resolution: dict | None, destination: str) -> None:
    if resolution:
        unknown = operation_fields(resolution).keys() - {"revision"}
        if unknown:
            raise ManifestError(
                f"{destination}: unsupported Hugging Face resolution options: "
                + ", ".join(sorted(unknown))
            )


def _annotated_record(record: dict, prior: dict, descriptors: tuple[str, ...]) -> dict:
    result = copy.deepcopy(project_fields(prior)) | record
    for name in descriptors:
        if isinstance(record.get(name), dict) and isinstance(prior.get(name), dict):
            result[name] = copy.deepcopy(project_fields(prior[name])) | record[name]
    return result


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
    # Optional publisher annotations carried by this lock entry.
    extensions: dict = field(default_factory=dict, compare=False)

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
            **copy.deepcopy(self.extensions),
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
        self.extensions = project_fields(doc)
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
            _check_resolution(record, record["requested"], path)
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
            _origin(record["origin"], requested_origins, path + ".origin")
            if type(record["size"]) is not int or not 0 <= record["size"] < 2**64:
                raise ManifestError(
                    f"{path}.size: expected an unsigned 64-bit byte count"
                )
            digest = _string(record["sha256"], path + ".sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ManifestError(
                    f"{path}.sha256: expected a lowercase SHA-256 digest"
                )
            _check_resolution(record, record["origin"], path)
            key = file_key(record["origin"], transform)
            if key in self.files:
                raise ManifestError(f"{path}: duplicate logical output")
            self.files[key] = FileIdentity(
                record["origin"],
                record["size"],
                digest,
                record.get("resolved"),
                transform,
                project_fields(record),
            )

    @classmethod
    def parse(cls, source: str) -> ModelLock:
        return cls(parse_json(source))

    @classmethod
    def read_if_present(cls, path: Path) -> ModelLock | None:
        try:
            return cls.parse(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def matching_root(self, name: str, requested: dict) -> dict | None:
        record = self.origins.get(name)
        if record and operation_fields(record["requested"]) == operation_fields(
            requested
        ):
            return record
        return None

    def project_root(self, name: str, record: dict) -> dict:
        prior = self.matching_root(name, record["requested"])
        return (
            _annotated_record(record, prior, ("requested", "resolved"))
            if prior
            else record
        )

    def cover(
        self, selection: Selection, *, assets_only: bool = False
    ) -> dict[str, FileIdentity]:
        """Checks coverage before acquisition. Publishers may carry only assets
        forward when rebuilding the program portion of a complete lock.
        """
        roots = selection.document.get("origins", {})
        covered = {}
        for name, file in lock_files(
            selection, "assets" if assets_only else self.scope
        ).items():
            if isinstance(file.origin, str) and file.origin.startswith("@"):
                alias = file.origin[1:].split("/", 1)[0]
                if not self.matching_root(alias, roots[alias]):
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
    def snapshot(
        origins: dict,
        identities: list[FileIdentity],
        *,
        scope: str = "all",
        previous: ModelLock | None = None,
        relocated: dict[str, str | dict] | None = None,
    ) -> dict:
        """Writes exactly this verified set, with no historical accumulation."""
        if scope not in ("assets", "all"):
            raise ManifestError("lock.scope: expected 'assets' or 'all'")
        files = {}
        used_origins = set()
        for identity in identities:
            key = file_key(identity.origin, identity.transform)
            record = identity.document()
            if previous and key in previous.files:
                record = _annotated_record(
                    record,
                    previous.files[key].document(),
                    ("origin", "transform", "resolved"),
                )
            # An installation rewrites source coordinates. Match the publisher's
            # annotations before relocation, and deduplicate the installed coordinates.
            if relocated is not None:
                origin = copy.deepcopy(relocated[key])
                if isinstance(origin, dict) and isinstance(record["origin"], dict):
                    origin.update(project_fields(record["origin"]))
                record["origin"] = origin
                key = file_key(origin, identity.transform)
            other = files.get(key)
            if other and (
                other["size"] != record["size"]
                or other["sha256"] != record["sha256"]
                or operation_fields(other.get("resolved"))
                != operation_fields(record.get("resolved"))
            ):
                raise ManifestError(
                    "selected sources have conflicting content identities"
                )
            files[key] = record
            origin = record["origin"]
            if isinstance(origin, str) and origin.startswith("@"):
                used_origins.add(origin[1:].split("/", 1)[0])
        named_roots = {}
        for name in sorted(used_origins):
            record = copy.deepcopy(origins[name])
            named_roots[name] = (
                previous.project_root(name, record) if previous else record
            )
        return {
            **copy.deepcopy(previous.extensions if previous else {}),
            "schema_version": 2,
            "scope": scope,
            "origins": named_roots,
            "files": [files[key] for key in sorted(files)],
        }
