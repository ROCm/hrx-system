# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for generated Loom language reference documentation."""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

from loom.gen.docs.reference import (
    DIALECT_REFERENCE_SPECS,
    generate_reference_files,
    write_reference_files,
)
from loom.gen.ops.model import load_generation_model

_MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)#]+)(?:#[^)]+)?\)")


def _prose_without_fenced_code(contents: str) -> str:
    prose_lines = []
    in_fence = False
    for line in contents.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose_lines.append(line)
    assert not in_fence
    return "\n".join(prose_lines)


def test_reference_generation_is_deterministic_and_complete() -> None:
    model = load_generation_model()
    first = generate_reference_files(model)
    second = generate_reference_files(model)

    assert first == second
    assert list(first) == sorted(first)
    for path, contents in first.items():
        assert contents.endswith("\n"), path

    specs = {spec.name: spec for spec in DIALECT_REFERENCE_SPECS}
    for generation in model.dialects:
        dialect_name = generation.dialect.name
        if not specs[dialect_name].is_published:
            assert f"dialects/{dialect_name}/index.md" not in first
            continue
        assert f"dialects/{dialect_name}/index.md" in first
        prefix = f"{dialect_name}."
        for op in generation.ops:
            local_name = op.name.removeprefix(prefix).replace(".", "-")
            assert f"dialects/{dialect_name}/ops/{local_name}.md" in first


def test_dialect_index_sorts_each_user_facing_section() -> None:
    index = generate_reference_files()["dialects/index.md"]
    sections = re.split(r"^## ", index, flags=re.MULTILINE)[1:]

    assert sections
    for section in sections:
        dialect_names = re.findall(
            r"^\| \[`([^`]+)`\]\([^)]*\) \|",
            section,
            flags=re.MULTILINE,
        )
        assert dialect_names
        assert dialect_names == sorted(dialect_names)


def test_generated_reference_links_resolve() -> None:
    files = generate_reference_files()
    for source_path, contents in files.items():
        if not source_path.endswith(".md"):
            continue
        source_dir = posixpath.dirname(source_path)
        prose = _prose_without_fenced_code(contents)
        for target in _MARKDOWN_LINK_PATTERN.findall(prose):
            assert "://" not in target
            resolved = posixpath.normpath(posixpath.join(source_dir, target))
            assert resolved in files, f"{source_path}: missing {target}"


def test_reference_renders_semantics_and_canonical_examples() -> None:
    files = generate_reference_files()

    vector_load = files["dialects/vector/ops/load.md"]
    assert "Load a vector footprint" in vector_load
    assert "`MemoryAccess`" in vector_load
    assert "`SameElementType(view, result)`" in vector_load
    assert "%v = vector.load" in vector_load

    func_template = files["dialects/func/ops/template.md"]
    assert "requires [#target.subgroup.size<64>]" in func_template

    subgroup_size = files["attributes/families/target-subgroup-size.md"]
    assert "static target-applicability requirement" in subgroup_size
    assert "`size` is the compact positional parameter" in subgroup_size


def test_reference_coverage_ratchets_documentation() -> None:
    files = generate_reference_files()
    coverage = json.loads(files["coverage.json"])

    assert coverage["schema_version"] == 1
    for dialect_name, dialect in coverage["dialects"].items():
        assert dialect["documented_operation_count"] == dialect["operation_count"]
        missing_examples = dialect["operation_count"] - dialect["operations_with_examples"]
        if dialect_name == "vector":
            assert missing_examples <= 89
        else:
            assert missing_examples == 0
    assert not coverage["dialects"]["test"]["published"]


def test_write_reference_files_preserves_the_generated_file_set(
    tmp_path: Path,
) -> None:
    files = generate_reference_files()
    write_reference_files(tmp_path, files)

    written_paths = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert written_paths == list(files)
    for relative_path, expected_contents in files.items():
        assert (tmp_path / relative_path).read_text(encoding="utf-8") == expected_contents
