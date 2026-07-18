# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake File API helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FileApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class CMakeExecutableTarget:
    name: str
    output_name: str
    path: Path


def codemodel_query_path(build_dir: Path) -> Path:
    return build_dir / ".cmake/api/v1/query/codemodel-v2"


def target_aliases_path(build_dir: Path) -> Path:
    return build_dir / ".iree/target_aliases.json"


def resolve_target_name(build_dir: Path, target_name: str) -> str:
    aliases = load_target_aliases(build_dir)
    resolved_target_name = aliases.get(target_name)
    if resolved_target_name is not None:
        return resolved_target_name

    # Preserve exact concrete targets, including libraries and custom targets,
    # before considering executable output-name shorthand.
    if not _has_file_api_reply(build_dir):
        return target_name
    target_refs = _codemodel_target_refs(build_dir)
    if any(target_ref["name"] == target_name for target_ref in target_refs):
        return target_name

    matching_target_names = sorted(
        {
            target.name
            for target in _executable_targets_from_refs(build_dir, target_refs)
            if target_name in _executable_output_aliases(target)
        }
    )
    if len(matching_target_names) == 1:
        return matching_target_names[0]
    if len(matching_target_names) > 1:
        candidate_list = ", ".join(repr(name) for name in matching_target_names)
        raise FileApiError(
            f"CMake executable name {target_name!r} is ambiguous in {build_dir}; "
            f"use a scoped alias or concrete target name from: {candidate_list}"
        )
    return target_name


def resolve_executable(build_dir: Path, target_name: str) -> CMakeExecutableTarget:
    resolved_target_name = resolve_target_name(build_dir, target_name)
    for target in executable_targets(build_dir):
        if target.name == resolved_target_name:
            return target
    raise FileApiError(
        f"CMake executable target {target_name!r} was not found in {build_dir}"
    )


def load_target_aliases(build_dir: Path) -> dict[str, str]:
    alias_path = target_aliases_path(build_dir)
    if not alias_path.is_file():
        return {}
    try:
        with alias_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as exc:
        raise FileApiError(f"CMake target alias map is invalid: {alias_path}") from exc
    if not isinstance(loaded, dict):
        raise FileApiError(f"CMake target alias map is not an object: {alias_path}")
    aliases = {}
    for alias_name, target_name in loaded.items():
        if not isinstance(alias_name, str) or not isinstance(target_name, str):
            raise FileApiError(
                f"CMake target alias map contains a non-string entry: {alias_path}"
            )
        aliases[alias_name] = target_name
    return aliases


def executable_targets(build_dir: Path) -> list[CMakeExecutableTarget]:
    return _executable_targets_from_refs(
        build_dir,
        _codemodel_target_refs(build_dir),
    )


def _codemodel_target_refs(build_dir: Path) -> list[dict[str, Any]]:
    codemodel = _load_reply_object(build_dir, _codemodel_json_file(build_dir))
    target_refs = []
    for configuration in codemodel.get("configurations", []):
        target_refs.extend(configuration.get("targets", []))
    return target_refs


def _executable_targets_from_refs(
    build_dir: Path,
    target_refs: list[dict[str, Any]],
) -> list[CMakeExecutableTarget]:
    targets = []
    for target_ref in target_refs:
        target = _load_reply_object(build_dir, target_ref["jsonFile"])
        if target.get("type") != "EXECUTABLE":
            continue
        artifact_path = _target_artifact_path(build_dir, target)
        targets.append(
            CMakeExecutableTarget(
                name=target_ref["name"],
                output_name=target.get("nameOnDisk", artifact_path.name),
                path=artifact_path,
            )
        )
    return targets


def _executable_output_aliases(target: CMakeExecutableTarget) -> set[str]:
    output_names = {target.output_name, target.path.name}
    for output_name in tuple(output_names):
        if output_name.lower().endswith(".exe"):
            output_names.add(output_name[:-4])
    return output_names


def _codemodel_json_file(build_dir: Path) -> str:
    index = _load_reply_object(build_dir, _latest_index_path(build_dir).name)
    for obj in index.get("objects", []):
        if obj.get("kind") == "codemodel":
            return obj["jsonFile"]
    raise FileApiError(f"CMake File API reply in {build_dir} has no codemodel")


def _latest_index_path(build_dir: Path) -> Path:
    reply_dir = build_dir / ".cmake/api/v1/reply"
    index_paths = sorted(reply_dir.glob("index-*.json"), key=lambda path: path.name)
    if not index_paths:
        raise FileApiError(
            "CMake File API reply is missing; run iree-cmake-configure first"
        )
    return index_paths[-1]


def _has_file_api_reply(build_dir: Path) -> bool:
    reply_dir = build_dir / ".cmake/api/v1/reply"
    return any(reply_dir.glob("index-*.json"))


def _load_reply_object(build_dir: Path, json_file: str) -> dict[str, Any]:
    reply_path = build_dir / ".cmake/api/v1/reply" / json_file
    try:
        with reply_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except FileNotFoundError as exc:
        raise FileApiError(
            f"CMake File API reply file is missing: {reply_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FileApiError(
            f"CMake File API reply file is invalid: {reply_path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise FileApiError(f"CMake File API reply file is not an object: {reply_path}")
    return loaded


def _target_artifact_path(build_dir: Path, target: dict[str, Any]) -> Path:
    artifacts = target.get("artifacts", [])
    if not artifacts:
        target_name = target.get("name", "<unknown>")
        raise FileApiError(f"CMake target {target_name} has no artifact")
    artifact_path = Path(artifacts[0]["path"])
    if artifact_path.is_absolute():
        return artifact_path
    return build_dir / artifact_path
