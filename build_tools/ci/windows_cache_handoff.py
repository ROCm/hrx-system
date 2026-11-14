# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Moves Bazel-selected Windows test runfiles through a shared CAS."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import pprint
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
MANIFEST_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_MANIFEST"
PYTHON_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_PYTHON"
CAS_DIRECTORY_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_CAS_DIR"
RUNFILES_PATH_BEGIN = "__IREE_BAZEL_RUNFILE_PATH_BEGIN__"
RUNFILES_PATH_END = "__IREE_BAZEL_RUNFILE_PATH_END__"
HANDOFF_PATH_BEGIN = "__IREE_CACHE_HANDOFF_RUNFILE_BEGIN__"
HANDOFF_PATH_END = "__IREE_CACHE_HANDOFF_RUNFILE_END__"


class HandoffError(RuntimeError):
    """Raised when the handoff contract cannot be represented or verified."""


@dataclass(frozen=True)
class Runfile:
    logical_path: str
    physical_path: Path | None


@dataclass(frozen=True)
class BlobSource:
    path: Path
    size: int


class CasClient:
    """Streams immutable blobs to and from bazel-remote's HTTP CAS surface."""

    def __init__(self, base_url: str, *, timeout_seconds: int = 60):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise HandoffError(f"invalid CAS base URL: {base_url!r}")
        if parsed.query or parsed.fragment:
            raise HandoffError("CAS base URL cannot contain a query or fragment")
        self._parsed = parsed
        self._timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")

    def _connection(self) -> http.client.HTTPConnection:
        connection_type = (
            http.client.HTTPSConnection
            if self._parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        return connection_type(
            self._parsed.hostname,
            self._parsed.port,
            timeout=self._timeout_seconds,
        )

    def _path(self, digest: str) -> str:
        prefix = self._parsed.path.rstrip("/")
        return f"{prefix}/cas/{digest}"

    def put(self, source: Path, digest: str, size: int) -> None:
        connection = self._connection()
        try:
            connection.putrequest("PUT", self._path(digest))
            connection.putheader("Content-Length", str(size))
            connection.putheader("Content-Type", "application/octet-stream")
            connection.endheaders()
            with source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    connection.send(chunk)
            response = connection.getresponse()
            body = response.read(4096)
            if response.status not in (200, 201, 204):
                raise HandoffError(
                    f"CAS PUT {digest} failed with HTTP {response.status}: "
                    f"{body.decode(errors='replace')}"
                )
        finally:
            connection.close()

    def contains(self, digest: str, size: int) -> bool:
        connection = self._connection()
        try:
            connection.request("HEAD", self._path(digest))
            response = connection.getresponse()
            if response.status == 404:
                return False
            if response.status != 200:
                body = response.read(4096)
                raise HandoffError(
                    f"CAS HEAD {digest} failed with HTTP {response.status}: "
                    f"{body.decode(errors='replace')}"
                )
            content_length = response.getheader("Content-Length")
            if content_length is not None and int(content_length) != size:
                raise HandoffError(
                    f"CAS blob {digest} has length {content_length}, expected {size}"
                )
            return True
        finally:
            connection.close()

    def get(self, destination: Path, digest: str, size: int) -> None:
        connection = self._connection()
        temporary_path = destination.with_name(destination.name + ".partial")
        hasher = hashlib.sha256()
        written = 0
        try:
            connection.request(
                "GET",
                self._path(digest),
                headers={"Accept-Encoding": "identity"},
            )
            response = connection.getresponse()
            if response.status != 200:
                body = response.read(4096)
                raise HandoffError(
                    f"CAS GET {digest} failed with HTTP {response.status}: "
                    f"{body.decode(errors='replace')}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > size:
                        raise HandoffError(
                            f"CAS blob {digest} exceeded declared size {size}"
                        )
                    hasher.update(chunk)
                    stream.write(chunk)
            actual_digest = hasher.hexdigest()
            if written != size or actual_digest != digest:
                raise HandoffError(
                    f"CAS blob {digest} verified as {actual_digest}/{written}, "
                    f"expected {digest}/{size}"
                )
            temporary_path.replace(destination)
        finally:
            connection.close()
            temporary_path.unlink(missing_ok=True)


def _decode_runfiles_value(value: str) -> str:
    return value.replace(r"\s", " ").replace(r"\n", "\n").replace(r"\b", "\\")


def parse_runfiles_manifest(path: Path) -> list[Runfile]:
    """Parses Bazel's escaped two-column runfiles manifest format."""
    runfiles = []
    logical_paths = set()
    casefolded_paths: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="\n") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            escaped = line.startswith(" ")
            logical_path, separator, physical_path = line[escaped:].partition(" ")
            if not separator:
                raise HandoffError(
                    f"{path}:{line_number}: invalid runfiles manifest entry"
                )
            if escaped:
                logical_path = _decode_runfiles_value(logical_path)
                physical_path = _decode_runfiles_value(physical_path)
            _validate_logical_path(logical_path)
            if logical_path in logical_paths:
                raise HandoffError(f"duplicate runfile path {logical_path!r}")
            logical_paths.add(logical_path)
            casefolded = logical_path.casefold()
            other_path = casefolded_paths.get(casefolded)
            if other_path is not None:
                raise HandoffError(
                    f"case-colliding runfile paths {other_path!r} and {logical_path!r}"
                )
            casefolded_paths[casefolded] = logical_path
            runfiles.append(
                Runfile(
                    logical_path=logical_path,
                    physical_path=Path(physical_path) if physical_path else None,
                )
            )
    return runfiles


def _validate_logical_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
    ):
        raise HandoffError(f"invalid logical runfile path: {value!r}")


def _sha256(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _load_json_values(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HandoffError(f"failed to read {path}: {exc}") from exc
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise HandoffError(f"failed to read {path}: {exc}") from exc
        values.append(value)
    return values


def _load_json_object(path: Path) -> dict[str, Any]:
    values = _load_json_values(path)
    if len(values) != 1:
        raise HandoffError(f"{path} must contain exactly one JSON value")
    value = values[0]
    if not isinstance(value, dict):
        raise HandoffError(f"{path} must contain a JSON object")
    return value


def _configured_rules(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = _load_json_object(path)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise HandoffError(f"{path} must describe configured targets")
    configured_rules = {}
    for result in results:
        rule = (
            result.get("target", {}).get("rule") if isinstance(result, dict) else None
        )
        if not isinstance(rule, dict):
            raise HandoffError(f"{path} does not describe configured rules")
        label = rule.get("name")
        attributes = rule.get("attribute")
        if not isinstance(label, str) or not isinstance(attributes, list):
            raise HandoffError(f"{path} has malformed configured rule metadata")
        label = _normalize_main_repository_label(label)
        if label in configured_rules:
            raise HandoffError(f"{path} contains duplicate configured target {label}")
        configured_rules[label] = {
            attribute["name"]: attribute
            for attribute in attributes
            if isinstance(attribute, dict) and isinstance(attribute.get("name"), str)
        }
    return configured_rules


def _normalize_main_repository_label(label: str) -> str:
    return label[2:] if label.startswith("@@//") else label


def _main_repository_label_parts(label: str) -> tuple[str, str]:
    label = _normalize_main_repository_label(label)
    if not label.startswith("//"):
        raise HandoffError(f"test target is not in the main repository: {label}")
    body = label[2:]
    package, separator, name = body.partition(":")
    if not separator:
        name = PurePosixPath(package).name
    package_path = PurePosixPath(package)
    name_path = PurePosixPath(name)
    if (
        (
            package
            and (
                package_path.is_absolute()
                or package_path.as_posix() != package
                or ".." in package_path.parts
            )
        )
        or not name
        or name_path.is_absolute()
        or name_path.as_posix() != name
        or ".." in name_path.parts
        or ":" in name
        or "\\" in label
    ):
        raise HandoffError(f"invalid main-repository test label: {label}")
    return package, name


def _string_attribute(attributes: dict[str, dict[str, Any]], name: str) -> str:
    value = attributes.get(name, {}).get("stringValue", "")
    if not isinstance(value, str):
        raise HandoffError(f"configured attribute {name} is not a string")
    return value


def _string_list_attribute(
    attributes: dict[str, dict[str, Any]], name: str
) -> list[str]:
    value = attributes.get(name, {}).get("stringListValue", [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HandoffError(f"configured attribute {name} is not a string list")
    return value


def _string_dict_attribute(
    attributes: dict[str, dict[str, Any]], name: str
) -> dict[str, str]:
    entries = attributes.get(name, {}).get("stringDictValue", [])
    if not isinstance(entries, list):
        raise HandoffError(f"configured attribute {name} is not a string dictionary")
    result = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("key"), str)
            or not isinstance(entry.get("value"), str)
        ):
            raise HandoffError(f"configured attribute {name} is malformed")
        result[entry["key"]] = entry["value"]
    return result


def _bool_attribute(attributes: dict[str, dict[str, Any]], name: str) -> bool:
    value = attributes.get(name, {}).get("booleanValue", False)
    if not isinstance(value, bool):
        raise HandoffError(f"configured attribute {name} is not boolean")
    return value


def _int_attribute(attributes: dict[str, dict[str, Any]], name: str) -> int:
    value = attributes.get(name, {}).get("intValue", 0)
    if not isinstance(value, int):
        raise HandoffError(f"configured attribute {name} is not an integer")
    return value


def _physical_to_logical(runfiles: Iterable[Runfile]) -> dict[Path, list[str]]:
    result: dict[Path, list[str]] = {}
    for runfile in runfiles:
        if runfile.physical_path is None:
            continue
        path = runfile.physical_path.resolve(strict=True)
        result.setdefault(path, []).append(runfile.logical_path)
    return result


def _provider_runfiles(
    provider_metadata: dict[str, Any], execution_root: Path
) -> list[Runfile]:
    entries = provider_metadata.get("runfiles")
    if not isinstance(entries, list):
        raise HandoffError("configured target has malformed runfiles metadata")
    runfiles = []
    logical_paths = set()
    casefolded_paths: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise HandoffError("configured target has malformed runfiles metadata")
        logical_path = entry.get("logical_path")
        physical_value = entry.get("physical_path")
        if not isinstance(logical_path, str) or not isinstance(physical_value, str):
            raise HandoffError("configured target has malformed runfiles metadata")
        _validate_logical_path(logical_path)
        if logical_path in logical_paths:
            raise HandoffError(f"duplicate runfile path {logical_path!r}")
        logical_paths.add(logical_path)
        casefolded = logical_path.casefold()
        other_path = casefolded_paths.get(casefolded)
        if other_path is not None:
            raise HandoffError(
                f"case-colliding runfile paths {other_path!r} and {logical_path!r}"
            )
        casefolded_paths[casefolded] = logical_path
        physical_path = None
        if physical_value:
            physical_path = Path(physical_value)
            if not physical_path.is_absolute():
                physical_path = execution_root / physical_path
        runfiles.append(Runfile(logical_path, physical_path))
    return runfiles


def _logical_path_for_marked_runfile(
    value: str,
    *,
    execution_root: Path,
    physical_to_logical: dict[Path, list[str]],
    logical_paths: set[str],
) -> str:
    # $(rootpath) expands to a path relative to the runfiles workspace, while
    # $(location) normally expands to an execution-root path. The provider
    # marker proves that either form names a declared runfile, but deliberately
    # does not bake the original Make variable into the configured value.
    # Resolve an exact runfiles spelling first, then fall back to the physical
    # execution-root identity used by location expansions.
    logical_candidates = []
    if not Path(value).is_absolute() and "\\" not in value:
        if value.startswith("../"):
            logical_candidates.append(value[3:])
        else:
            logical_candidates.extend([value, "_main/" + value])
    matching_logical_paths = {
        candidate for candidate in logical_candidates if candidate in logical_paths
    }
    if len(matching_logical_paths) == 1:
        return matching_logical_paths.pop()
    if len(matching_logical_paths) > 1:
        raise HandoffError(
            f"configured runfile path {value!r} maps to multiple logical paths: "
            f"{sorted(matching_logical_paths)}"
        )

    path = Path(value)
    if not path.is_absolute():
        path = execution_root / path
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        resolved_path = None
    physical_matches = (
        physical_to_logical.get(resolved_path, []) if resolved_path is not None else []
    )
    if len(physical_matches) != 1:
        raise HandoffError(
            f"configured runfile path {value!r} maps to "
            f"{len(physical_matches)} logical paths"
        )
    return physical_matches[0]


def _convert_marked_value(
    value: str,
    *,
    execution_root: Path,
    physical_to_logical: dict[Path, list[str]],
    logical_paths: set[str],
) -> str:
    result = ""
    cursor = 0
    while cursor < len(value):
        start = value.find(RUNFILES_PATH_BEGIN, cursor)
        if start < 0:
            result += value[cursor:]
            break
        result += value[cursor:start]
        path_start = start + len(RUNFILES_PATH_BEGIN)
        end = value.find(RUNFILES_PATH_END, path_start)
        if end < 0:
            raise HandoffError("configured runfile path marker is unterminated")
        logical_path = _logical_path_for_marked_runfile(
            value[path_start:end],
            execution_root=execution_root,
            physical_to_logical=physical_to_logical,
            logical_paths=logical_paths,
        )
        result += HANDOFF_PATH_BEGIN + logical_path + HANDOFF_PATH_END
        cursor = end + len(RUNFILES_PATH_END)
    return result


def _manifest_entry(
    runfile: Runfile,
    blobs: dict[str, BlobSource],
) -> dict[str, Any]:
    if runfile.physical_path is None:
        return {"path": runfile.logical_path, "empty": True}
    path = runfile.physical_path.resolve(strict=True)
    if not path.is_file():
        raise HandoffError(f"runfile is not a regular file: {path}")
    digest, size = _sha256(path)
    prior_source = blobs.get(digest)
    if prior_source is None:
        blobs[digest] = BlobSource(path=path, size=size)
    elif prior_source.size != size:
        raise HandoffError(f"CAS digest {digest} has inconsistent byte lengths")
    return {
        "path": runfile.logical_path,
        "sha256": digest,
        "size": size,
        "executable": os.access(path, os.X_OK),
    }


def _provider_metadata_by_label(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for value in _load_json_values(path):
        if not isinstance(value, dict):
            raise HandoffError(f"{path} must contain JSON objects")
        label = value.get("label")
        if not isinstance(label, str):
            raise HandoffError(f"{path} contains provider metadata without a label")
        label = _normalize_main_repository_label(label)
        if label in result:
            raise HandoffError(
                f"{path} contains duplicate provider metadata for {label}"
            )
        result[label] = value
    if not result:
        raise HandoffError(f"{path} must describe configured target providers")
    return result


def _emit_target_manifest(
    *,
    label: str,
    attributes: dict[str, dict[str, Any]],
    provider_metadata: dict[str, Any],
    execution_root: Path,
    cas_url: str,
    blobs: dict[str, BlobSource],
) -> tuple[dict[str, Any], dict[str, Any]]:
    executable_value = provider_metadata.get("executable")
    if not isinstance(executable_value, str) or not executable_value:
        raise HandoffError("configured target has no executable")
    executable_path = Path(executable_value)
    if not executable_path.is_absolute():
        executable_path = execution_root / executable_path
    executable_path = executable_path.resolve(strict=True)
    executable_provider_count = provider_metadata.get("executable_provider_count")
    if executable_provider_count not in (0, 1):
        raise HandoffError("configured target has ambiguous executable providers")
    runfiles = _provider_runfiles(provider_metadata, execution_root)
    physical_to_logical = _physical_to_logical(runfiles)
    logical_paths = {runfile.logical_path for runfile in runfiles}
    executable_logical_path = provider_metadata.get("executable_logical_path")
    if not isinstance(executable_logical_path, str):
        raise HandoffError("configured target has no logical executable path")
    _validate_logical_path(executable_logical_path)
    executable_runfiles = [
        runfile
        for runfile in runfiles
        if runfile.logical_path == executable_logical_path
    ]
    if (
        len(executable_runfiles) != 1
        or executable_runfiles[0].physical_path is None
        or executable_runfiles[0].physical_path.resolve(strict=True) != executable_path
    ):
        raise HandoffError(
            f"configured executable {executable_logical_path!r} does not match its runfile"
        )

    argument_provider_count = provider_metadata.get("argument_provider_count")
    if argument_provider_count not in (0, 1):
        raise HandoffError("configured target has ambiguous argument providers")
    if argument_provider_count:
        arguments = provider_metadata.get("arguments")
        marked_arguments = provider_metadata.get("marked_arguments")
        if (
            not isinstance(arguments, list)
            or not isinstance(marked_arguments, list)
            or len(arguments) != len(marked_arguments)
        ):
            raise HandoffError("configured argument provider is malformed")
        converted_arguments = [
            _convert_marked_value(
                value,
                execution_root=execution_root,
                physical_to_logical=physical_to_logical,
                logical_paths=logical_paths,
            )
            for value in marked_arguments
        ]
        if [
            value.replace(RUNFILES_PATH_BEGIN, "").replace(RUNFILES_PATH_END, "")
            for value in marked_arguments
        ] != arguments:
            raise HandoffError("configured argument markers do not match arguments")
    else:
        converted_arguments = _string_list_attribute(attributes, "args")

    environment = provider_metadata.get("environment")
    marked_environment = provider_metadata.get("marked_environment")
    inherited_environment = provider_metadata.get("inherited_environment")
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise HandoffError("configured run environment is malformed")
    if not isinstance(marked_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in marked_environment.items()
    ):
        raise HandoffError("configured marked environment is malformed")
    if not isinstance(inherited_environment, list) or any(
        not isinstance(value, str) for value in inherited_environment
    ):
        raise HandoffError("configured inherited environment is malformed")
    environment = dict(environment)
    for name, marked_value in marked_environment.items():
        if name not in environment:
            raise HandoffError(f"marked environment variable {name} is not configured")
        environment[name] = _convert_marked_value(
            marked_value,
            execution_root=execution_root,
            physical_to_logical=physical_to_logical,
            logical_paths=logical_paths,
        )

    entries = [_manifest_entry(runfile, blobs) for runfile in runfiles]
    shard_count = _int_attribute(attributes, "shard_count")
    test_metadata = {
        "arguments": converted_arguments,
        "environment": environment,
        "inherited_environment": inherited_environment,
        "size": _string_attribute(attributes, "size"),
        "timeout": _string_attribute(attributes, "timeout"),
        "flaky": _bool_attribute(attributes, "flaky"),
        "local": _bool_attribute(attributes, "local"),
        "tags": _string_list_attribute(attributes, "tags"),
        "exec_properties": _string_dict_attribute(attributes, "exec_properties"),
    }
    if shard_count >= 0:
        test_metadata["shard_count"] = shard_count
    workspace = executable_logical_path.split("/", 1)[0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_label": label,
        "cas_url": cas_url,
        "workspace": workspace,
        "executable": executable_logical_path,
        "test": test_metadata,
        "runfiles": entries,
    }
    build_attributes: dict[str, Any] = {
        "env_inherit": inherited_environment,
        "size": test_metadata["size"],
        "timeout": test_metadata["timeout"],
        "flaky": test_metadata["flaky"],
        "local": test_metadata["local"],
        "tags": test_metadata["tags"],
        "exec_properties": test_metadata["exec_properties"],
    }
    if "shard_count" in test_metadata:
        build_attributes["shard_count"] = test_metadata["shard_count"]
    return manifest, build_attributes


def _publish_blobs(cas: CasClient, blobs: dict[str, BlobSource], jobs: int = 8) -> None:
    """Ensures each unique product blob exists in the shared CAS."""

    def publish(entry: tuple[str, BlobSource]) -> None:
        digest, source = entry
        if not cas.contains(digest, source.size):
            cas.put(source.path, digest, source.size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(publish, blobs.items()))


def _render_build_file(rules: list[dict[str, Any]]) -> str:
    lines = [
        'load("//handoff:build_defs.bzl", "windows_cache_handoff_test")',
        "",
    ]
    for attributes in rules:
        lines.append("windows_cache_handoff_test(")
        for name, value in attributes.items():
            lines.append(f"    {name} = {pprint.pformat(value, sort_dicts=True)},")
        lines.extend([")", ""])
    return "\n".join(lines)


def _emit_standalone_module(output_dir: Path) -> None:
    """Writes the source-revision runner and its dependency-free Bazel module."""
    output_dir.joinpath("MODULE.bazel").write_text(
        'module(name = "hrx_windows_cache_handoff")\n', encoding="utf-8"
    )
    handoff_dir = output_dir / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(Path(__file__), handoff_dir / "runner.py")
    shutil.copyfile(
        source_root / "windows" / "cache_handoff" / "build_defs.bzl",
        handoff_dir / "build_defs.bzl",
    )
    handoff_dir.joinpath("BUILD.bazel").write_text(
        'exports_files(["runner.py"], visibility = ["//visibility:public"])\n',
        encoding="utf-8",
    )


def emit_handoff_packages(
    *,
    target_metadata_path: Path,
    provider_metadata_path: Path,
    execution_root: Path,
    cas_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    configured_rules = _configured_rules(target_metadata_path)
    providers = _provider_metadata_by_label(provider_metadata_path)
    if configured_rules.keys() != providers.keys():
        missing_providers = sorted(configured_rules.keys() - providers.keys())
        missing_rules = sorted(providers.keys() - configured_rules.keys())
        raise HandoffError(
            "configured rules and providers differ: "
            f"missing providers={missing_providers}, missing rules={missing_rules}"
        )

    cas = CasClient(cas_url)
    blobs: dict[str, BlobSource] = {}
    build_rules: dict[str, list[dict[str, Any]]] = {}
    incompatible_labels = []
    non_native_labels = []
    test_summaries = []
    runfile_reference_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_standalone_module(output_dir)
    for label in sorted(configured_rules):
        provider_metadata = providers[label]
        compatible = provider_metadata.get("compatible")
        if not isinstance(compatible, bool):
            raise HandoffError(f"provider metadata for {label} has no compatibility")
        if not compatible:
            incompatible_labels.append(label)
            continue
        executable_value = provider_metadata.get("executable")
        if not isinstance(
            executable_value, str
        ) or not executable_value.lower().endswith(".exe"):
            non_native_labels.append(label)
            continue
        package, name = _main_repository_label_parts(label)
        try:
            manifest, build_attributes = _emit_target_manifest(
                label=label,
                attributes=configured_rules[label],
                provider_metadata=provider_metadata,
                execution_root=execution_root,
                cas_url=cas.base_url,
                blobs=blobs,
            )
        except (HandoffError, OSError) as exc:
            raise HandoffError(f"failed to emit {label}: {exc}") from exc
        package_dir = output_dir / PurePosixPath(package)
        package_dir.mkdir(parents=True, exist_ok=True)
        manifest_name = (
            "handoff-" + hashlib.sha256(label.encode()).hexdigest()[:16] + ".json"
        )
        manifest_path = package_dir / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        build_rules.setdefault(package, []).append(
            {
                "name": name,
                "manifest": manifest_name,
                **build_attributes,
            }
        )
        runfile_reference_count += len(manifest["runfiles"])
        test_summaries.append(
            {
                "source_label": label,
                "package": package,
                "name": name,
                "manifest": manifest_path.relative_to(output_dir).as_posix(),
                "runfile_count": len(manifest["runfiles"]),
            }
        )

    if not test_summaries:
        raise HandoffError("configured selection contains no compatible tests")
    for package, rules in build_rules.items():
        package_dir = output_dir / PurePosixPath(package)
        (package_dir / "BUILD.bazel").write_text(
            _render_build_file(rules), encoding="utf-8"
        )
    _publish_blobs(cas, blobs)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cas_url": cas.base_url,
        "blobs": [
            {"sha256": digest, "size": source.size}
            for digest, source in sorted(blobs.items())
        ],
        "selected_test_count": len(configured_rules),
        "compatible_test_count": len(test_summaries),
        "incompatible_test_count": len(incompatible_labels),
        "incompatible_labels": incompatible_labels,
        "non_native_test_count": len(non_native_labels),
        "non_native_labels": non_native_labels,
        "runfile_reference_count": runfile_reference_count,
        "unique_blob_count": len(blobs),
        "unique_blob_bytes": sum(source.size for source in blobs.values()),
        "tests": test_summaries,
    }
    (output_dir / "handoff-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def prefetch_handoff_blobs(
    summary_path: Path, content_root: Path, *, jobs: int = 8
) -> None:
    """Materializes and verifies one shared content store for a test batch."""
    summary = _load_json_object(summary_path)
    cas_url = summary.get("cas_url")
    blobs = summary.get("blobs")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise HandoffError("unsupported handoff summary schema")
    if not isinstance(cas_url, str) or not isinstance(blobs, list):
        raise HandoffError("handoff summary has malformed CAS metadata")
    if jobs < 1:
        raise HandoffError("prefetch jobs must be positive")
    entries = []
    digests = set()
    for entry in blobs:
        if not isinstance(entry, dict):
            raise HandoffError("handoff summary has a malformed blob")
        digest = entry.get("sha256")
        size = entry.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or size < 0
        ):
            raise HandoffError("handoff summary has invalid blob metadata")
        if digest in digests:
            raise HandoffError(f"handoff summary repeats CAS blob {digest}")
        digests.add(digest)
        entries.append((digest, size))

    content_root.mkdir(parents=True, exist_ok=True)
    cas = CasClient(cas_url)

    def fetch(entry: tuple[str, int]) -> None:
        digest, size = entry
        destination = content_root / digest
        if destination.exists():
            actual_digest, actual_size = _sha256(destination)
            if actual_digest != digest or actual_size != size:
                raise HandoffError(
                    f"prefetched CAS blob {digest} verified as "
                    f"{actual_digest}/{actual_size}, expected {digest}/{size}"
                )
            return
        cas.get(destination, digest, size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(fetch, entries))


def _resolve_handoff_value(value: str, runfiles_root: Path) -> str:
    result = ""
    cursor = 0
    while cursor < len(value):
        start = value.find(HANDOFF_PATH_BEGIN, cursor)
        if start < 0:
            result += value[cursor:]
            break
        result += value[cursor:start]
        path_start = start + len(HANDOFF_PATH_BEGIN)
        end = value.find(HANDOFF_PATH_END, path_start)
        if end < 0:
            raise HandoffError("handoff runfile path marker is unterminated")
        logical_path = value[path_start:end]
        _validate_logical_path(logical_path)
        result += str(runfiles_root / PurePosixPath(logical_path))
        cursor = end + len(HANDOFF_PATH_END)
    return result


def _write_runfiles_manifest(
    path: Path, entries: list[dict[str, Any]], root: Path
) -> None:
    """Writes the legacy manifest format consumed by Bazel's Windows launcher."""
    lines = []
    for entry in entries:
        logical_path = entry["path"]
        physical_path = str(root / PurePosixPath(logical_path))
        if " " in logical_path or "\n" in logical_path or "\r" in logical_path:
            raise HandoffError(
                "Bazel's native Windows launcher cannot represent runfile path "
                f"{logical_path!r} in its manifest"
            )
        if "\n" in physical_path or "\r" in physical_path:
            raise HandoffError(
                "Bazel's native Windows launcher cannot represent physical path "
                f"{physical_path!r} in its manifest"
            )
        # Bazel's native Windows launcher splits each line at its first space.
        # Backslashes and subsequent spaces in the physical path are literal.
        lines.append(f"{logical_path} {physical_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_runfiles(manifest: dict[str, Any], root: Path) -> None:
    cas_url = manifest.get("cas_url")
    entries = manifest.get("runfiles")
    if not isinstance(cas_url, str) or not isinstance(entries, list):
        raise HandoffError("handoff manifest has malformed CAS metadata")
    shared_content_root_value = os.environ.get(CAS_DIRECTORY_ENVIRONMENT_VARIABLE)
    shared_content_root = (
        Path(shared_content_root_value) if shared_content_root_value else None
    )
    cas = CasClient(cas_url)
    content_root = shared_content_root or root / ".cas"
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise HandoffError("handoff manifest has a malformed runfile")
        logical_path = entry["path"]
        _validate_logical_path(logical_path)
        destination = root / PurePosixPath(logical_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("empty") is True:
            destination.touch()
            continue
        digest = entry.get("sha256")
        size = entry.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or size < 0
        ):
            raise HandoffError(f"runfile {logical_path!r} has invalid CAS metadata")
        content_path = content_root / digest
        if shared_content_root is not None:
            if not content_path.is_file() or content_path.stat().st_size != size:
                raise HandoffError(
                    f"CAS blob {digest} was not prefetched into {content_root}"
                )
        elif not content_path.exists():
            cas.get(content_path, digest, size)
        if shared_content_root is not None:
            try:
                os.link(content_path, destination)
            except OSError as exc:
                raise HandoffError(
                    f"cannot hard-link prefetched CAS blob {digest} into "
                    f"the test runfiles tree: {exc}"
                ) from exc
        else:
            shutil.copyfile(content_path, destination)
        if entry.get("executable") is True:
            destination.chmod(destination.stat().st_mode | stat.S_IXUSR)


def _find_manifest_from_environment() -> Path:
    value = os.environ.get(MANIFEST_ENVIRONMENT_VARIABLE)
    if not value:
        raise HandoffError(f"{MANIFEST_ENVIRONMENT_VARIABLE} is not set")
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    candidates = [Path.cwd() / path]
    test_srcdir = os.environ.get("TEST_SRCDIR")
    test_workspace = os.environ.get("TEST_WORKSPACE")
    if test_srcdir and test_workspace:
        candidates.append(Path(test_srcdir) / test_workspace / path)
    runfiles_manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if runfiles_manifest and test_workspace:
        logical_path = f"{test_workspace}/{path.as_posix()}"
        _validate_logical_path(logical_path)
        candidates.extend(
            runfile.physical_path
            for runfile in parse_runfiles_manifest(Path(runfiles_manifest))
            if runfile.logical_path == logical_path
            and runfile.physical_path is not None
        )
    matches = list(
        dict.fromkeys(candidate for candidate in candidates if candidate.is_file())
    )
    if len(matches) != 1:
        raise HandoffError(f"cannot resolve {MANIFEST_ENVIRONMENT_VARIABLE}={value!r}")
    return matches[0]


def run_handoff_test(
    manifest_path: Path, additional_arguments: Iterable[str] = ()
) -> int:
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise HandoffError("unsupported handoff manifest schema")
    test = manifest.get("test")
    entries = manifest.get("runfiles")
    executable = manifest.get("executable")
    workspace = manifest.get("workspace")
    if (
        not isinstance(test, dict)
        or not isinstance(entries, list)
        or not isinstance(executable, str)
        or not isinstance(workspace, str)
    ):
        raise HandoffError("handoff manifest is malformed")
    _validate_logical_path(executable)
    _validate_logical_path(workspace)
    test_tmpdir = Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
    with tempfile.TemporaryDirectory(
        prefix="iree-windows-cache-handoff-", dir=test_tmpdir
    ) as temp_dir:
        runfiles_root = Path(temp_dir) / "runfiles"
        runfiles_root.mkdir()
        _materialize_runfiles(manifest, runfiles_root)
        subject_manifest = Path(temp_dir) / "MANIFEST"
        _write_runfiles_manifest(subject_manifest, entries, runfiles_root)
        executable_path = runfiles_root / PurePosixPath(executable)
        workspace_root = runfiles_root / PurePosixPath(workspace)
        if not executable_path.is_file() or not workspace_root.is_dir():
            raise HandoffError("materialized test executable or workspace is missing")
        arguments = test.get("arguments")
        environment = test.get("environment")
        if not isinstance(arguments, list) or any(
            not isinstance(value, str) for value in arguments
        ):
            raise HandoffError("handoff test arguments are malformed")
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise HandoffError("handoff test environment is malformed")
        child_environment = dict(os.environ)
        child_environment.pop(PYTHON_ENVIRONMENT_VARIABLE, None)
        child_environment.update(
            {
                name: _resolve_handoff_value(value, runfiles_root)
                for name, value in environment.items()
            }
        )
        child_environment.update(
            {
                "RUNFILES_DIR": str(runfiles_root),
                "RUNFILES_MANIFEST_FILE": str(subject_manifest),
                "RUNFILES_MANIFEST_ONLY": "1" if os.name == "nt" else "0",
                "TEST_SRCDIR": str(runfiles_root),
                "TEST_WORKSPACE": workspace,
            }
        )
        completed = subprocess.run(
            [
                str(executable_path),
                *(_resolve_handoff_value(value, runfiles_root) for value in arguments),
                *additional_arguments,
            ],
            cwd=workspace_root,
            env=child_environment,
            check=False,
        )
        return completed.returncode


def audit_consumer_execution(execution_log_path: Path, output_path: Path) -> None:
    """Proves that the native consumer did not compile or link test products."""
    entries = [
        entry
        for entry in _load_json_values(execution_log_path)
        if isinstance(entry, dict)
    ]
    forbidden = [
        entry
        for entry in entries
        if entry.get("mnemonic") in ("CppCompile", "CppLink", "CppArchive")
    ]
    if forbidden:
        details = [
            {
                "mnemonic": entry.get("mnemonic"),
                "target": entry.get("targetLabel"),
                "runner": entry.get("runner"),
            }
            for entry in forbidden
        ]
        raise HandoffError(
            "native consumer executed C/C++ build actions: "
            + json.dumps(details, sort_keys=True)
        )
    receipt = {
        "entry_count": len(entries),
        "mnemonics": sorted(
            {
                entry.get("mnemonic")
                for entry in entries
                if isinstance(entry.get("mnemonic"), str)
            }
        ),
        "native_cpp_action_count": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser(
        "emit", help="Upload runfiles and emit a Bazel test package"
    )
    emit_parser.add_argument("--target-metadata", type=Path, required=True)
    emit_parser.add_argument("--provider-metadata", type=Path, required=True)
    emit_parser.add_argument("--execution-root", type=Path, required=True)
    emit_parser.add_argument("--cas-url", required=True)
    emit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser = subparsers.add_parser(
        "audit", help="Verify that the native consumer ran no C/C++ build actions"
    )
    audit_parser.add_argument("--execution-log", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    prefetch_parser = subparsers.add_parser(
        "prefetch", help="Materialize and verify the shared CAS content store"
    )
    prefetch_parser.add_argument("--summary", type=Path, required=True)
    prefetch_parser.add_argument("--output-dir", type=Path, required=True)
    prefetch_parser.add_argument("--jobs", type=int, default=8)
    run_parser = subparsers.add_parser(
        "run", help="Materialize and execute one handoff test"
    )
    run_parser.add_argument("test_arguments", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.command == "emit":
            emit_handoff_packages(
                target_metadata_path=arguments.target_metadata,
                provider_metadata_path=arguments.provider_metadata,
                execution_root=arguments.execution_root.resolve(strict=True),
                cas_url=arguments.cas_url,
                output_dir=arguments.output_dir,
            )
            return 0
        if arguments.command == "audit":
            audit_consumer_execution(arguments.execution_log, arguments.output)
            return 0
        if arguments.command == "prefetch":
            prefetch_handoff_blobs(
                arguments.summary,
                arguments.output_dir,
                jobs=arguments.jobs,
            )
            return 0
        test_arguments = arguments.test_arguments
        if test_arguments[:1] == ["--"]:
            test_arguments = test_arguments[1:]
        return run_handoff_test(_find_manifest_from_environment(), test_arguments)
    except (HandoffError, OSError, ValueError) as exc:
        print(f"windows cache handoff: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
