# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.gen.support.generated_file import GENERATED_FILE_MARKER
from loom.tools.source_surface import (
    build_source_surface_report,
    format_text_report,
    is_production_source_path,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_source_tree(repo_root: Path) -> None:
    source_root = repo_root / "loom/src/loom"
    _write(source_root / "ir/module.c", "int module;\n\n")
    _write(source_root / "ir/module.h", "int module_h;\n")
    _write(
        source_root / "ops/vector/tables.c", f"// {GENERATED_FILE_MARKER}\nint table;\n"
    )
    _write(source_root / "target/compile_report.c", "int target_root;\n")
    _write(source_root / "target/arch/amdgpu/lower.c", "int amdgpu_lower;\n")
    _write(source_root / "target/arch/x86/lower.c", "int x86_lower;\n")
    _write(source_root / "target/emit/native/object.c", "int native_object;\n")
    _write(source_root / "target/emit/native/amdgpu/hsaco.c", "int hsaco;\n")
    _write(source_root / "target/emit/native/x86/assembly.c", "int native_x86;\n")
    _write(source_root / "target/emit/llvmir/module_emitter.c", "int llvmir_emit;\n")

    _write(source_root / "tools/loom-opt/main.c", "int tool;\n")
    _write(source_root / "tooling/testbench/harness.c", "int harness;\n")
    _write(source_root / "testing/context.c", "int testing;\n")
    _write(source_root / "ops/test/ops.h", "int test_dialect;\n")
    _write(source_root / "target/tool/process.c", "int target_tool;\n")
    _write(source_root / "target/test/lower.c", "int target_test;\n")
    _write(source_root / "target/test_modules.c", "int target_test_module;\n")
    _write(
        source_root / "target/low_descriptor_registry_core_test.c",
        "int target_test_suffix;\n",
    )


def test_production_source_path_exclusions() -> None:
    assert is_production_source_path(Path("ir/module.c"))
    assert not is_production_source_path(Path("tools/loom-opt/main.c"))
    assert not is_production_source_path(Path("tooling/testbench/harness.c"))
    assert not is_production_source_path(Path("testing/context.c"))
    assert not is_production_source_path(Path("ops/test/ops.h"))
    assert not is_production_source_path(Path("target/tool/process.c"))
    assert not is_production_source_path(Path("target/test_modules.c"))
    assert not is_production_source_path(
        Path("target/low_descriptor_registry_core_test.c")
    )
    assert not is_production_source_path(Path("ir/module.txt"))


def test_report_counts_generated_inclusive_and_hand_authored_slices(
    tmp_path: Path,
) -> None:
    _write_source_tree(tmp_path)

    report = build_source_surface_report(tmp_path)
    all_targets = report["slices"]["all_targets"]
    amdgpu_hsaco = report["slices"]["native_amdgpu_hsaco"]

    assert all_targets["generated_inclusive"] == {
        "files": 10,
        "generated_files": 1,
        "generated_nonblank_lines": 2,
        "generated_physical_lines": 2,
        "nonblank_lines": 11,
        "physical_lines": 12,
    }
    assert all_targets["hand_authored_only"] == {
        "files": 9,
        "generated_files": 0,
        "generated_nonblank_lines": 0,
        "generated_physical_lines": 0,
        "nonblank_lines": 9,
        "physical_lines": 10,
    }
    assert amdgpu_hsaco["generated_inclusive"] == {
        "files": 7,
        "generated_files": 1,
        "generated_nonblank_lines": 2,
        "generated_physical_lines": 2,
        "nonblank_lines": 8,
        "physical_lines": 9,
    }
    assert amdgpu_hsaco["hand_authored_only"] == {
        "files": 6,
        "generated_files": 0,
        "generated_nonblank_lines": 0,
        "generated_physical_lines": 0,
        "nonblank_lines": 6,
        "physical_lines": 7,
    }


def test_json_report_is_stable_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_source_tree(tmp_path)

    assert main(["--repo-root", str(tmp_path), "--format", "json"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)

    assert report["schema"] == "loom.source_surface"
    assert report["schema_version"] == 1
    assert report["source_root"] == "loom/src/loom"
    assert tuple(report["source_suffixes"]) == (".c", ".h")
    assert tuple(report["slices"]) == ("all_targets", "native_amdgpu_hsaco")


def test_text_report_names_metrics_and_exclusions(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)

    report = build_source_surface_report(tmp_path)
    text_report = format_text_report(report)

    assert "Loom source-surface audit" in text_report
    assert "native_amdgpu_hsaco" in text_report
    assert "generated_inclusive" in text_report
    assert "hand_authored_only" in text_report
    assert "physical_lines: Source line records" in text_report
    expected_rule = (
        "path_segment: .notes, cmake, editor, gen, generator, generators, "
        "test, testing, tool, tooling, tools"
    )
    assert expected_rule in text_report
