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
from typing import Any, cast


class FileApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class CMakeExecutableTarget:
    name: str
    path: Path


def codemodel_query_path(build_dir: Path) -> Path:
    return build_dir / ".cmake/api/v1/query/codemodel-v2"


def target_aliases_path(build_dir: Path) -> Path:
    return build_dir / ".iree/target_aliases.json"


def resolve_target_name(build_dir: Path, target_name: str) -> str:
    return resolve_target_names(build_dir, [target_name])[0]


def resolve_target_names(build_dir: Path, target_names: list[str]) -> list[str]:
    return _resolve_target_names(build_dir, target_names)


def _resolve_target_names(
    build_dir: Path,
    target_names: list[str],
    *,
    targets: list[CMakeExecutableTarget] | None = None,
) -> list[str]:
    aliases = load_target_aliases(build_dir)
    resolved_target_names = [aliases.get(target_name) for target_name in target_names]
    unresolved_indices = [
        index
        for index, resolved_target_name in enumerate(resolved_target_names)
        if resolved_target_name is None
    ]
    if not unresolved_indices:
        return cast(list[str], resolved_target_names)

    # Preserve raw CMake target names before configuration has produced a File
    # API reply. This keeps `cmake build` useful for unmanaged build trees while
    # letting configured trees resolve the user-facing executable filename.
    if targets is None and not _latest_index_paths(build_dir):
        return [
            resolved_target_name or target_names[index]
            for index, resolved_target_name in enumerate(resolved_target_names)
        ]

    exact_target_names = (
        configured_target_names(build_dir)
        if targets is None
        else {target.name for target in targets}
    )
    artifact_indices = []
    for index in unresolved_indices:
        target_name = target_names[index]
        if target_name in exact_target_names:
            resolved_target_names[index] = target_name
        else:
            artifact_indices.append(index)

    if not artifact_indices:
        return cast(list[str], resolved_target_names)
    if targets is None:
        targets = executable_targets(build_dir)

    for index in artifact_indices:
        target_name = target_names[index]
        matching_target_names = {
            target.name
            for target in targets
            if _artifact_has_executable_name(target.path, target_name)
        }
        if len(matching_target_names) == 1:
            resolved_target_names[index] = matching_target_names.pop()
        elif len(matching_target_names) > 1:
            formatted_names = ", ".join(sorted(matching_target_names))
            raise FileApiError(
                f"CMake executable name {target_name!r} is ambiguous; matching "
                f"targets: {formatted_names}"
            )
        else:
            resolved_target_names[index] = target_name

    return cast(list[str], resolved_target_names)


def resolve_executable(build_dir: Path, target_name: str) -> CMakeExecutableTarget:
    targets = executable_targets(build_dir)
    resolved_target_name = _resolve_target_names(
        build_dir, [target_name], targets=targets
    )[0]
    for target in targets:
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
    codemodel = _load_reply_object(build_dir, _codemodel_json_file(build_dir))
    targets = []
    for configuration in codemodel.get("configurations", []):
        for target_ref in configuration.get("targets", []):
            target = _load_reply_object(build_dir, target_ref["jsonFile"])
            if target.get("type") != "EXECUTABLE":
                continue
            artifact_path = _target_artifact_path(build_dir, target)
            targets.append(
                CMakeExecutableTarget(
                    name=target_ref["name"],
                    path=artifact_path,
                )
            )
    return targets


def configured_target_names(build_dir: Path) -> set[str]:
    codemodel = _load_reply_object(build_dir, _codemodel_json_file(build_dir))
    return {
        target_ref["name"]
        for configuration in codemodel.get("configurations", [])
        for target_ref in configuration.get("targets", [])
    }


def _codemodel_json_file(build_dir: Path) -> str:
    index = _load_reply_object(build_dir, _latest_index_path(build_dir).name)
    for obj in index.get("objects", []):
        if obj.get("kind") == "codemodel":
            return obj["jsonFile"]
    raise FileApiError(f"CMake File API reply in {build_dir} has no codemodel")


def _latest_index_path(build_dir: Path) -> Path:
    index_paths = _latest_index_paths(build_dir)
    if not index_paths:
        raise FileApiError(
            "CMake File API reply is missing; run iree-cmake-configure first"
        )
    return index_paths[-1]


def _latest_index_paths(build_dir: Path) -> list[Path]:
    reply_dir = build_dir / ".cmake/api/v1/reply"
    return sorted(reply_dir.glob("index-*.json"), key=lambda path: path.name)


def _load_reply_object(build_dir: Path, json_file: str) -> dict[str, Any]:
    reply_path = build_dir / ".cmake/api/v1/reply" / json_file
    try:
        with reply_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except FileNotFoundError as exc:
        raise FileApiError(
            f"CMake File API reply file is missing: {reply_path}"
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


def _artifact_has_executable_name(artifact_path: Path, name: str) -> bool:
    if artifact_path.name == name:
        return True
    return artifact_path.suffix.casefold() == ".exe" and artifact_path.stem == name
