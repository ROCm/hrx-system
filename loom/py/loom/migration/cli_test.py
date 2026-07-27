# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.format.bytecode.writer import FORMAT_VERSION, MAGIC
from loom.migration.cli import main


def test_single_source_defaults_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "input.loom"
    source_path.write_text("func.def @f() {\n  func.return\n}\n", encoding="utf-8")

    assert main([str(source_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "func.def @f() {\n  func.return\n}\n"
    assert captured.err == ""


def test_output_writes_single_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "input.loom"
    output_path = tmp_path / "output.loom"
    source_path.write_text("func.def @f() {\n  func.return\n}\n", encoding="utf-8")

    assert main([str(source_path), "--output", str(output_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert output_path.read_text(encoding="utf-8") == source_path.read_text(
        encoding="utf-8"
    )


def test_in_place_writes_loom_test_input_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "input.loom-test"
    source_path.write_text(
        "\n".join(
            (
                "// RUN: roundtrip",
                "",
                "func.def @f(%input: buffer) {",
                "  %global = buffer.assume.memory_space %input {memory_space = global} : buffer",
                "  func.return",
                "}",
                "",
                "// ----",
                "func.def @f(%input: buffer) {",
                "  %global = buffer.assume.memory_space %input {memory_space = global} : buffer",
                "  func.return",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert main([str(source_path), "--in-place"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    migrated_text = source_path.read_text(encoding="utf-8")
    assert (
        "  %global = buffer.assume.memory_space<global> %input : buffer\n"
        in migrated_text
    )
    assert (
        "  %global = buffer.assume.memory_space %input {memory_space = global} : buffer\n"
        in migrated_text
    )


def test_check_root_reports_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "input.loom").write_text(
        "func.def @f() {\n  func.return\n}\n",
        encoding="utf-8",
    )

    assert main(["--root", str(tmp_path), "--check", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["changed"] is False
    assert report["files"][0]["kind"] == "loom"


def test_old_bytecode_json_reports_versions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bytecode_path = tmp_path / "old.loombc"
    bytecode_path.write_bytes(MAGIC + bytes([FORMAT_VERSION - 1]))

    assert main([str(bytecode_path), "--check", "--json"]) == 1

    report = json.loads(capsys.readouterr().out)
    diagnostic = report["diagnostics"][0]
    assert diagnostic["rule_id"] == "loom.migrate.bytecode_version"
    assert diagnostic["actual_version"] == FORMAT_VERSION - 1
    assert diagnostic["current_version"] == FORMAT_VERSION


def test_root_requires_check_or_in_place(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path)])
