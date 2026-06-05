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
    path: Path


def codemodel_query_path(build_dir: Path) -> Path:
    return build_dir / ".cmake/api/v1/query/codemodel-v2"


def resolve_executable(build_dir: Path, target_name: str) -> CMakeExecutableTarget:
    for target in executable_targets(build_dir):
        if target.name == target_name:
            return target
    raise FileApiError(
        f"CMake executable target {target_name!r} was not found in {build_dir}"
    )


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
