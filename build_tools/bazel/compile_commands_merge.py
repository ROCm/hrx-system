#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merges Bazel compile command fragments into one compile_commands.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

COMPILE_COMMANDS_OUTPUT_GROUP = "iree_compile_commands_fragments"


def load_fragment(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: expected JSON object entries")
    return data


def sort_key(entry: dict[str, object]) -> tuple[str, str, str]:
    directory = entry.get("directory", "")
    file_path = entry.get("file", "")
    arguments = entry.get("arguments", "")
    return (str(directory), str(file_path), str(arguments))


def write_merged_compile_commands(
    *,
    output_path: Path,
    fragment_paths: list[Path],
    command_directory: Path | None = None,
) -> None:
    entries: list[dict[str, object]] = []
    for fragment_path in fragment_paths:
        for entry in load_fragment(fragment_path):
            if command_directory is not None:
                entry["directory"] = str(command_directory)
            entries.append(entry)
    entries.sort(key=sort_key)

    if output_path.exists():
        output_path.chmod(output_path.stat().st_mode | 0o200)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)
        output_file.write("\n")


def fragment_paths_from_bep(
    build_events_path: Path,
    *,
    output_group: str = COMPILE_COMMANDS_OUTPUT_GROUP,
) -> list[Path]:
    named_sets: dict[str, dict[str, object]] = {}
    root_set_ids: list[str] = []

    with build_events_path.open(encoding="utf-8") as build_events_file:
        for line in build_events_file:
            event = json.loads(line)
            named_set_id = event.get("id", {}).get("namedSet", {}).get("id")
            if named_set_id is not None:
                named_sets[named_set_id] = event.get("namedSetOfFiles", {})
            for group in event.get("completed", {}).get("outputGroup", []):
                if group.get("name") == output_group:
                    root_set_ids.extend(
                        file_set["id"] for file_set in group.get("fileSets", [])
                    )

    seen_set_ids: set[str] = set()
    seen_paths: set[Path] = set()
    fragment_paths: list[Path] = []

    def visit_set(set_id: str) -> None:
        if set_id in seen_set_ids:
            return
        seen_set_ids.add(set_id)
        named_set = named_sets.get(set_id)
        if named_set is None:
            raise ValueError(f"{build_events_path}: missing named set {set_id}")
        for file_set in named_set.get("fileSets", []):
            visit_set(file_set["id"])
        for file_entry in named_set.get("files", []):
            name = file_entry.get("name", "")
            if not name.endswith(".compile_commands.json"):
                continue
            fragment_path = file_uri_to_path(file_entry["uri"])
            if fragment_path in seen_paths:
                continue
            seen_paths.add(fragment_path)
            fragment_paths.append(fragment_path)

    for root_set_id in root_set_ids:
        visit_set(root_set_id)

    fragment_paths.sort()
    return fragment_paths


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected file URI for compile command fragment, got {uri}")
    if parsed.netloc:
        raise ValueError(f"file URI has unsupported host component: {uri}")
    return Path(unquote(parsed.path))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: compile_commands_merge.py OUT [FRAGMENT ...]",
            file=sys.stderr,
        )
        return 2

    output_path = Path(argv[1])
    write_merged_compile_commands(
        output_path=output_path,
        fragment_paths=[Path(fragment_arg) for fragment_arg in argv[2:]],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
