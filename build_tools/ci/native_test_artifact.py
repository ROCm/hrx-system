#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packages native Bazel tests and tools for execution on CI workers."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

PAYLOAD_OUTPUT_GROUP = "native_artifact_payload"
DEBUG_OUTPUT_GROUP = "native_artifact_debug"
METADATA_SUFFIX = ".native-artifact.json"

_LOOM_TOOL_ROLES = {
    "benchmark": "//loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom",
    "compile": "//loom/src/loom/tools/loom-compile:loom-compile",
    "format": "//loom/src/loom/tools/loom-format:loom-format",
    "link": "//loom/src/loom/tools/loom-link:loom-link",
    "test": "//loom/src/loom/tools/iree-test-loom:iree-test-loom",
}


@dataclasses.dataclass(frozen=True)
class ProducerRoots:
    workspace: Path
    execution_root: Path
    external_root: Path


@dataclasses.dataclass(frozen=True)
class TargetExport:
    label: str
    configuration: str
    payload_files: frozenset[Path]
    debug_files: frozenset[Path]


@dataclasses.dataclass(frozen=True)
class BuildExport:
    roots: ProducerRoots
    targets: tuple[TargetExport, ...]


@dataclasses.dataclass(frozen=True)
class CopyResult:
    primary: Path
    payload_files: frozenset[Path]
    elf_files: frozenset[Path]


@dataclasses.dataclass
class PackagedTarget:
    label: str
    configuration: str
    name: str
    is_tool: bool
    metadata: dict[str, object]
    executable_path: str
    payload_files: set[Path]
    elf_files: set[Path]
    runfiles: dict[str, str]


def _normalize_label(label: str) -> str:
    if label.startswith("@@//"):
        return label[2:]
    if label.startswith("@//"):
        return label[1:]
    if label.startswith("@@"):
        return "@" + label[2:]
    return label


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected a file URI from Bazel, got {uri}")
    if parsed.netloc:
        raise ValueError(f"file URI has an unsupported host component: {uri}")
    return Path(url2pathname(parsed.path))


def _named_set_paths(
    *,
    build_events_path: Path,
    named_sets: dict[str, dict[str, object]],
    root_ids: Iterable[str],
) -> frozenset[Path]:
    seen_ids: set[str] = set()
    paths: set[Path] = set()

    def visit(set_id: str) -> None:
        if set_id in seen_ids:
            return
        seen_ids.add(set_id)
        named_set = named_sets.get(set_id)
        if named_set is None:
            raise ValueError(f"{build_events_path}: missing named set {set_id}")
        for child in named_set.get("fileSets", []):
            visit(child["id"])
        for entry in named_set.get("files", []):
            uri = entry.get("uri")
            if not uri:
                raise ValueError(
                    f"{build_events_path}: output {entry.get('name')} has no file URI"
                )
            paths.add(_file_uri_to_path(uri))

    for root_id in root_ids:
        visit(root_id)
    return frozenset(paths)


def load_build_export(build_events_path: Path) -> BuildExport:
    """Loads typed native artifact outputs from one Bazel JSON BEP file."""
    named_sets: dict[str, dict[str, object]] = {}
    grouped_roots: dict[tuple[str, str], dict[str, list[str]]] = {}
    workspace: Path | None = None
    execution_root: Path | None = None
    build_success: bool | None = None

    with build_events_path.open(encoding="utf-8") as build_events_file:
        for line_number, line in enumerate(build_events_file, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{build_events_path}:{line_number}: invalid JSON"
                ) from exc
            started = event.get("started")
            if started is not None:
                workspace_value = started.get("workspaceDirectory")
                if workspace_value:
                    workspace = Path(workspace_value)
            workspace_info = event.get("workspaceInfo")
            if workspace_info is not None:
                execution_root_value = workspace_info.get("localExecRoot")
                if execution_root_value:
                    execution_root = Path(execution_root_value)
            named_set_id = event.get("id", {}).get("namedSet", {}).get("id")
            if named_set_id is not None:
                named_sets[named_set_id] = event.get("namedSetOfFiles", {})

            target_id = event.get("id", {}).get("targetCompleted")
            completed = event.get("completed")
            if target_id is not None and completed is not None:
                groups = [
                    group
                    for group in completed.get("outputGroup", [])
                    if group.get("name") in (PAYLOAD_OUTPUT_GROUP, DEBUG_OUTPUT_GROUP)
                ]
                if groups and not completed.get("success", False):
                    raise ValueError(
                        f"{build_events_path}: artifact aspect failed for "
                        f"{target_id.get('label')}"
                    )
                if groups:
                    key = (
                        _normalize_label(target_id["label"]),
                        target_id.get("configuration", {}).get("id", ""),
                    )
                    target_groups = grouped_roots.setdefault(key, {})
                    for group in groups:
                        target_groups.setdefault(group["name"], []).extend(
                            file_set["id"] for file_set in group.get("fileSets", [])
                        )
            finished = event.get("finished")
            if finished is not None:
                build_success = bool(finished.get("overallSuccess"))

    if build_success is not True:
        raise ValueError(
            f"{build_events_path}: Bazel build did not finish successfully"
        )
    if workspace is None or execution_root is None:
        raise ValueError(f"{build_events_path}: missing Bazel workspace roots")
    workspace = workspace.resolve(strict=True)
    execution_root = execution_root.resolve(strict=True)
    external_root = execution_root.parent.parent / "external"
    external_root = external_root.resolve(strict=True)

    targets = []
    for (label, configuration), groups in sorted(grouped_roots.items()):
        payload_files = _named_set_paths(
            build_events_path=build_events_path,
            named_sets=named_sets,
            root_ids=groups.get(PAYLOAD_OUTPUT_GROUP, []),
        )
        debug_files = _named_set_paths(
            build_events_path=build_events_path,
            named_sets=named_sets,
            root_ids=groups.get(DEBUG_OUTPUT_GROUP, []),
        )
        if payload_files:
            targets.append(
                TargetExport(
                    label=label,
                    configuration=configuration,
                    payload_files=payload_files,
                    debug_files=debug_files,
                )
            )
    return BuildExport(
        roots=ProducerRoots(
            workspace=workspace,
            execution_root=execution_root,
            external_root=external_root,
        ),
        targets=tuple(targets),
    )


def parse_runfiles_manifest(manifest_path: Path) -> list[tuple[str, str | None]]:
    """Parses Bazel's escaped runfiles manifest format."""
    entries = []
    with manifest_path.open(encoding="utf-8", newline="\n") as manifest_file:
        for line_number, line in enumerate(manifest_file, 1):
            line = line.removesuffix("\n")
            try:
                if line.startswith(" "):
                    escaped_logical, escaped_physical = line[1:].split(" ", 1)
                    logical = (
                        escaped_logical.replace(r"\s", " ")
                        .replace(r"\n", "\n")
                        .replace(r"\b", "\\")
                    )
                    physical = escaped_physical.replace(r"\n", "\n").replace(
                        r"\b", "\\"
                    )
                else:
                    logical, physical = line.split(" ", 1)
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: invalid runfiles entry"
                ) from exc
            entries.append((logical, physical or None))
    return entries


def _is_safe_relative_path(path: Path) -> bool:
    return not path.is_absolute() and path.parts and ".." not in path.parts


def _is_elf(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(4) == b"\x7fELF"


class ArtifactTree:
    """Copies producer files while preserving loader paths and content sharing."""

    def __init__(
        self,
        *,
        destination: Path,
        roots: ProducerRoots,
        llvm_objcopy: Path,
    ) -> None:
        self.destination = destination
        self.roots = (
            (roots.execution_root, "execroot"),
            (roots.workspace, "source"),
            (roots.external_root, "external"),
        )
        self.llvm_objcopy = llvm_objcopy
        self.blobs: dict[tuple[str, int], Path] = {}
        self.copies: dict[Path, Path] = {}
        self.elf_files: set[Path] = set()
        self.original_unique_bytes = 0
        self.packaged_unique_bytes = 0

    def _relocated(self, path: Path) -> Path:
        for root, name in self.roots:
            if path.is_relative_to(root):
                return self.destination / "files" / name / path.relative_to(root)
        raise ValueError(f"runtime file is outside the producer roots: {path}")

    @staticmethod
    def link_alias(output: Path, target: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(output):
            if output.is_symlink() and output.resolve() == target.resolve():
                return
            raise FileExistsError(output)
        output.symlink_to(os.path.relpath(target, output.parent))

    @staticmethod
    def _declared(source: Path, declared_files: frozenset[Path]) -> bool:
        normalized = Path(os.path.abspath(source))
        return normalized in declared_files

    def copy_file(
        self,
        source: Path,
        *,
        declared_files: frozenset[Path],
    ) -> CopyResult:
        if not source.is_absolute():
            source = self.roots[0][0] / source
        source = Path(os.path.abspath(source))
        if not self._declared(source, declared_files):
            raise ValueError(
                f"runtime file was not exported by its Bazel target: {source}"
            )
        if source.is_dir():
            raise ValueError(
                f"directory runfile needs an explicit packaging rule: {source}"
            )
        canonical = source.resolve(strict=True)
        copied = self.copies.get(canonical)
        source_is_elf = _is_elf(canonical)
        if copied is None:
            mode = canonical.stat().st_mode & 0o777
            with canonical.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            key = (digest, mode)
            copied = self._relocated(canonical)
            copied.parent.mkdir(parents=True, exist_ok=True)
            existing_blob = self.blobs.get(key)
            if existing_blob is not None:
                os.link(existing_blob, copied)
            else:
                self.original_unique_bytes += canonical.stat().st_size
                if source_is_elf:
                    subprocess.run(
                        [
                            self.llvm_objcopy,
                            "--remove-section=.gdb_index",
                            "--remove-section=.symtab",
                            "--remove-section=.strtab",
                            canonical,
                            copied,
                        ],
                        check=True,
                    )
                    copied.chmod(mode)
                else:
                    shutil.copy2(canonical, copied)
                self.packaged_unique_bytes += copied.stat().st_size
                self.blobs[key] = copied
            self.copies[canonical] = copied
            if source_is_elf:
                self.elf_files.add(copied)

        alias = self._relocated(source)
        if alias != copied:
            self.link_alias(alias, copied)
        payload_files = {copied, alias}
        elf_files = {copied} if source_is_elf else set()
        return CopyResult(
            primary=copied,
            payload_files=frozenset(payload_files),
            elf_files=frozenset(elf_files),
        )


def _producer_path(path_value: object, execution_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"invalid producer path: {path_value!r}")
    path = Path(path_value)
    if not path.is_absolute():
        path = execution_root / path
    return Path(os.path.abspath(path))


def _target_name(label: str, *, is_tool: bool) -> str:
    if is_tool:
        target_name = label.rsplit(":", 1)[-1]
        return "prebuilt-" + re.sub(r"[^A-Za-z0-9_.+-]", "-", target_name)
    return "native_" + re.sub(r"[^A-Za-z0-9_]", "_", label).strip("_")


def _load_target_metadata(
    build_exports: Sequence[BuildExport],
) -> tuple[ProducerRoots, list[tuple[TargetExport, dict[str, object]]]]:
    if not build_exports:
        raise ValueError("at least one Bazel build-event file is required")
    roots = build_exports[0].roots
    for build_export in build_exports[1:]:
        if build_export.roots != roots:
            raise ValueError(
                "all Bazel build-event files must use the same producer roots"
            )

    targets_by_label: dict[str, tuple[TargetExport, dict[str, object]]] = {}
    for build_export in build_exports:
        for target in build_export.targets:
            metadata_paths = [
                path
                for path in target.payload_files
                if path.name.endswith(METADATA_SUFFIX)
            ]
            if len(metadata_paths) != 1:
                raise ValueError(
                    f"{target.label}: expected one {METADATA_SUFFIX} output, "
                    f"found {len(metadata_paths)}"
                )
            metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
            metadata_label = _normalize_label(str(metadata.get("label", "")))
            if metadata_label != target.label:
                raise ValueError(
                    f"{target.label}: metadata names a different label {metadata_label}"
                )
            previous = targets_by_label.get(target.label)
            if previous is not None:
                if previous[1] != metadata:
                    raise ValueError(
                        f"{target.label}: conflicting metadata across Bazel builds"
                    )
                previous_target = previous[0]
                if previous_target.configuration != target.configuration:
                    raise ValueError(
                        f"{target.label}: conflicting configurations across Bazel builds"
                    )
                targets_by_label[target.label] = (
                    TargetExport(
                        label=target.label,
                        configuration=target.configuration,
                        payload_files=previous_target.payload_files
                        | target.payload_files,
                        debug_files=previous_target.debug_files | target.debug_files,
                    ),
                    metadata,
                )
                continue
            targets_by_label[target.label] = (target, metadata)
    return roots, [targets_by_label[label] for label in sorted(targets_by_label)]


def _starlark(value: object, indentation: int = 0) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = [
            " " * (indentation + 4) + _starlark(item, indentation + 4) + ","
            for item in value
        ]
        return "[\n" + "\n".join(inner) + "\n" + " " * indentation + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = [
            " " * (indentation + 4)
            + _starlark(str(key))
            + ": "
            + _starlark(value[key], indentation + 4)
            + ","
            for key in sorted(value)
        ]
        return "{\n" + "\n".join(inner) + "\n" + " " * indentation + "}"
    raise TypeError(f"cannot render Starlark value {value!r}")


def _render_rule(rule: str, attributes: dict[str, object]) -> str:
    lines = [f"{rule}("]
    for name, value in attributes.items():
        rendered = _starlark(value, 4)
        lines.append(f"    {name} = {rendered},")
    lines.append(")")
    return "\n".join(lines)


def _write_build_file(
    *,
    package_directory: Path,
    targets: Sequence[PackagedTarget],
    loom_toolchain: bool,
    platform_constraints: Sequence[str],
) -> None:
    loads = [
        'load("//build_tools/bazel:native_test_artifact.bzl", "native_artifact_test", "native_artifact_tool")'
    ]
    if loom_toolchain:
        loads.append(
            'load("//loom/build_tools/bazel:loom_toolchain.bzl", "loom_tools_toolchains")'
        )
    sections = [*loads, "", 'package(default_visibility = ["//visibility:public"])']
    test_names = []
    tool_targets: dict[str, str] = {}
    for target in targets:
        attributes: dict[str, object] = {
            "name": target.name,
            "executable_path": target.executable_path,
            "payload": sorted(
                path.relative_to(package_directory).as_posix()
                for path in target.payload_files
            ),
            "environment": target.metadata.get("environment", {}),
            "marked_environment": target.metadata.get("marked_environment", {}),
            "inherited_environment": target.metadata.get("inherited_environment", []),
            "source_arguments": target.metadata.get(
                "marked_arguments", target.metadata.get("arguments", [])
            ),
            "target_compatible_with": list(platform_constraints),
        }
        if target.is_tool:
            rule = "native_artifact_tool"
            tool_targets[target.label] = target.name
            for name in ("tags", "exec_properties"):
                if name in target.metadata:
                    attributes[name] = target.metadata[name]
        else:
            rule = "native_artifact_test"
            test_names.append(target.name)
            for name in (
                "tags",
                "size",
                "timeout",
                "shard_count",
                "flaky",
                "exec_properties",
            ):
                value = target.metadata.get(name)
                if value is not None and not (name == "shard_count" and value == -1):
                    attributes[name] = value
        sections.extend(["", _render_rule(rule, attributes)])

    sections.extend(
        [
            "",
            _render_rule(
                "test_suite",
                {
                    "name": "native_tests",
                    "tests": [f":{name}" for name in test_names],
                },
            ),
        ]
    )
    if loom_toolchain:
        missing_labels = sorted(set(_LOOM_TOOL_ROLES.values()) - tool_targets.keys())
        if missing_labels:
            raise ValueError(
                "cannot declare the Loom artifact toolchain without: "
                + ", ".join(missing_labels)
            )
        attributes = {
            "name": "artifact",
            **{
                f"{role}_tool": f":{tool_targets[label]}"
                for role, label in _LOOM_TOOL_ROLES.items()
            },
            "lint_tool": "//loom/py/loom/tools:loom-lint",
            "exec_compatible_with": list(platform_constraints),
            "target_compatible_with": list(platform_constraints),
            "visibility": ["//visibility:public"],
        }
        sections.extend(["", _render_rule("loom_tools_toolchains", attributes)])
    package_directory.joinpath("BUILD.bazel").write_text(
        "\n".join(sections) + "\n", encoding="utf-8"
    )


def stage_native_artifact(
    *,
    build_events_paths: Sequence[Path],
    staging_root: Path,
    package_path: Path,
    profile: str,
    revision: str,
    llvm_dwp: Path,
    llvm_objcopy: Path,
    tool_labels: Iterable[str] = (),
    loom_toolchain: bool = False,
    platform_constraints: Sequence[str] = (),
) -> dict[str, object]:
    """Stages a complete artifact tree from typed Bazel outputs."""
    if staging_root.exists():
        raise FileExistsError(staging_root)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"revision must be a full lowercase Git commit: {revision}")
    if not profile:
        raise ValueError("native artifacts require a build profile")
    if not _is_safe_relative_path(package_path):
        raise ValueError(f"package path must be relative and contained: {package_path}")
    if not platform_constraints:
        raise ValueError("native artifacts require platform constraints")
    staging_root.mkdir(parents=True)
    package_directory = staging_root / package_path
    package_directory.mkdir(parents=True)

    build_exports = [load_build_export(path) for path in build_events_paths]
    roots, target_metadata = _load_target_metadata(build_exports)
    normalized_tool_labels = {_normalize_label(label) for label in tool_labels}
    tree = ArtifactTree(
        destination=package_directory,
        roots=roots,
        llvm_objcopy=llvm_objcopy,
    )
    packaged_targets = []
    used_names: dict[str, str] = {}
    all_debug_files: set[Path] = set()

    for target_export, metadata in target_metadata:
        label = target_export.label
        is_tool = label in normalized_tool_labels
        name = _target_name(label, is_tool=is_tool)
        previous_label = used_names.get(name)
        if previous_label is not None and previous_label != label:
            suffix = hashlib.sha256(label.encode()).hexdigest()[:8]
            name = f"{name}_{suffix}"
        used_names[name] = label

        runfiles_manifest_value = metadata.get("runfiles_manifest")
        if runfiles_manifest_value is None:
            raise ValueError(f"{label}: native artifact has no runfiles manifest")
        manifest_path = _producer_path(runfiles_manifest_value, roots.execution_root)
        executable = _producer_path(metadata.get("executable"), roots.execution_root)
        declared_files = frozenset(
            Path(os.path.abspath(path)) for path in target_export.payload_files
        )
        for required_path in (manifest_path, executable):
            if required_path not in declared_files:
                raise ValueError(
                    f"{label}: producer metadata references an undeclared file: "
                    f"{required_path}"
                )

        payload_files: set[Path] = set()
        target_elf_files: set[Path] = set()
        runfiles: dict[str, str] = {}
        executable_path = None
        for logical, physical_value in parse_runfiles_manifest(manifest_path):
            logical_path = Path(logical)
            if not _is_safe_relative_path(logical_path):
                raise ValueError(f"{label}: invalid runfile path: {logical}")
            runfile_output = package_directory / "runfiles" / name / logical_path
            if physical_value is None:
                runfile_output.parent.mkdir(parents=True, exist_ok=True)
                runfile_output.touch(exist_ok=False)
                payload_files.add(runfile_output)
            else:
                physical = _producer_path(physical_value, roots.execution_root)
                copied = tree.copy_file(
                    physical,
                    declared_files=declared_files,
                )
                tree.link_alias(runfile_output, copied.primary)
                payload_files.update(copied.payload_files)
                payload_files.add(runfile_output)
                target_elf_files.update(copied.elf_files)
                if physical.resolve(strict=True) == executable.resolve(strict=True):
                    executable_path = logical
            runfiles[logical] = runfile_output.relative_to(package_directory).as_posix()
        if executable_path is None:
            raise ValueError(
                f"{label}: executable is absent from its runfiles manifest"
            )

        all_debug_files.update(target_export.debug_files)
        packaged_targets.append(
            PackagedTarget(
                label=label,
                configuration=target_export.configuration,
                name=name,
                is_tool=is_tool,
                metadata=metadata,
                executable_path=executable_path,
                payload_files=payload_files,
                elf_files=target_elf_files,
                runfiles=runfiles,
            )
        )

    unknown_tools = normalized_tool_labels - {
        target.label for target in packaged_targets
    }
    if unknown_tools:
        raise ValueError(
            "tool labels were not exported as native executables: "
            + ", ".join(sorted(unknown_tools))
        )
    if not all_debug_files:
        raise ValueError("artifact has no split-DWARF files")
    missing_debug_files = [path for path in all_debug_files if not path.is_file()]
    if missing_debug_files:
        raise FileNotFoundError(missing_debug_files[0])

    debug_file = package_directory / ".debug" / "all.dwp"
    debug_file.parent.mkdir()
    subprocess.run(
        [llvm_dwp, *sorted(all_debug_files), "-o", debug_file],
        check=True,
    )
    debug_aliases = {}
    for elf_file in sorted(tree.elf_files):
        alias = elf_file.with_name(elf_file.name + ".dwp")
        os.link(debug_file, alias)
        debug_aliases[elf_file] = alias
    for target in packaged_targets:
        target.payload_files.update(
            debug_aliases[elf_file] for elf_file in target.elf_files
        )

    _write_build_file(
        package_directory=package_directory,
        targets=packaged_targets,
        loom_toolchain=loom_toolchain,
        platform_constraints=platform_constraints,
    )
    result: dict[str, object] = {
        "debug_alias_count": len(debug_aliases),
        "debug_file_bytes": debug_file.stat().st_size,
        "debug_file_count": len(all_debug_files),
        "package_path": package_path.as_posix(),
        "packaged_runtime_bytes": tree.packaged_unique_bytes,
        "physical_runtime_paths": len(tree.copies),
        "profile": profile,
        "revision": revision,
        "runtime_blob_count": len(tree.blobs),
        "source_runtime_bytes": tree.original_unique_bytes,
        "target_count": len(packaged_targets),
        "test_count": sum(not target.is_tool for target in packaged_targets),
        "tool_count": sum(target.is_tool for target in packaged_targets),
        "targets": [
            {
                "arguments": target.metadata.get("arguments", []),
                "configuration": target.configuration,
                "environment": target.metadata.get("environment", {}),
                "label": target.label,
                "rule_kind": target.metadata.get("rule_kind"),
                "runfiles": target.runfiles,
                "target": target.name,
                "tool": target.is_tool,
            }
            for target in packaged_targets
        ],
    }
    package_directory.joinpath("manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compress_artifact(
    *,
    staging_root: Path,
    output: Path,
    tar: str,
    zstd: str,
    compression_level: int,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tar_command = [
        tar,
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-cf",
        "-",
        "-C",
        os.fspath(staging_root),
        ".",
    ]
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary_directory:
        temporary_output = Path(temporary_directory) / output.name
        zstd_command = [
            zstd,
            f"-{compression_level}",
            "-T4",
            # Preserve cross-binary matching without exceeding the default decoder
            # window accepted by `tar --zstd`.
            "--long=27",
            "-q",
            "-o",
            os.fspath(temporary_output),
        ]
        with subprocess.Popen(tar_command, stdout=subprocess.PIPE) as tar_process:
            if tar_process.stdout is None:
                raise RuntimeError("tar did not expose its output pipe")
            try:
                subprocess.run(zstd_command, stdin=tar_process.stdout, check=True)
            finally:
                tar_process.stdout.close()
            tar_return_code = tar_process.wait()
        if tar_return_code != 0:
            raise subprocess.CalledProcessError(tar_return_code, tar_command)
        os.replace(temporary_output, output)
    return {
        "archive": os.fspath(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": _sha256(output),
        "compression_seconds": time.monotonic() - started,
    }


def verify_artifact(
    *,
    package_directory: Path,
    expected_profile: str,
    expected_revision: str,
    checkout_revision: str,
) -> dict[str, object]:
    """Verifies that a downloaded package belongs to the current checkout."""
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError(
            f"expected revision must be a full lowercase Git commit: "
            f"{expected_revision}"
        )
    if checkout_revision != expected_revision:
        raise ValueError(
            f"checkout revision {checkout_revision} does not match expected "
            f"revision {expected_revision}"
        )

    manifest_path = package_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read native artifact manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"native artifact manifest is not an object: {manifest_path}")
    artifact_revision = manifest.get("revision")
    if artifact_revision != expected_revision:
        raise ValueError(
            f"artifact revision {artifact_revision!r} does not match expected "
            f"revision {expected_revision}"
        )
    artifact_profile = manifest.get("profile")
    if artifact_profile != expected_profile:
        raise ValueError(
            f"artifact profile {artifact_profile!r} does not match expected "
            f"profile {expected_profile!r}"
        )
    return {
        "package_directory": os.fspath(package_directory),
        "profile": expected_profile,
        "revision": expected_revision,
        "target_count": manifest.get("target_count"),
        "test_count": manifest.get("test_count"),
        "tool_count": manifest.get("tool_count"),
    }


def _add_package_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build-events", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-path", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--llvm-dwp", required=True, type=Path)
    parser.add_argument("--llvm-objcopy", required=True, type=Path)
    parser.add_argument("--tool-label", action="append", default=[])
    parser.add_argument("--loom-toolchain", action="store_true")
    parser.add_argument("--platform-constraint", action="append", default=[])
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--tar", default="tar")
    parser.add_argument("--zstd", default="zstd")
    parser.add_argument("--compression-level", default=12, type=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package_parser = subparsers.add_parser(
        "package", help="Stage and compress one native artifact package."
    )
    _add_package_arguments(package_parser)
    verify_parser = subparsers.add_parser(
        "verify", help="Verify a downloaded package against its source checkout."
    )
    verify_parser.add_argument("--package", required=True, type=Path)
    verify_parser.add_argument("--profile", required=True)
    verify_parser.add_argument("--revision", required=True)
    verify_parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "verify":
        checkout_revision = subprocess.check_output(
            ["git", "-C", args.workspace, "rev-parse", "HEAD"], text=True
        ).strip()
        result = verify_artifact(
            package_directory=args.package,
            expected_profile=args.profile,
            expected_revision=args.revision,
            checkout_revision=checkout_revision,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    def stage_and_compress(staging_root: Path) -> dict[str, object]:
        started = time.monotonic()
        result = stage_native_artifact(
            build_events_paths=args.build_events,
            staging_root=staging_root,
            package_path=args.package_path,
            profile=args.profile,
            revision=args.revision,
            llvm_dwp=args.llvm_dwp,
            llvm_objcopy=args.llvm_objcopy,
            tool_labels=args.tool_label,
            loom_toolchain=args.loom_toolchain,
            platform_constraints=args.platform_constraint,
        )
        result["staging_seconds"] = time.monotonic() - started
        result.update(
            compress_artifact(
                staging_root=staging_root,
                output=args.output,
                tar=args.tar,
                zstd=args.zstd,
                compression_level=args.compression_level,
            )
        )
        return result

    if args.staging is not None:
        result = stage_and_compress(args.staging)
    else:
        with tempfile.TemporaryDirectory(prefix="iree-native-artifact-") as directory:
            result = stage_and_compress(Path(directory) / "staging")
    summary = {key: value for key, value in result.items() if key != "targets"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
