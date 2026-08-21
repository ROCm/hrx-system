# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from loom.gen.support.generated_file import (
    GENERATED_FILE_MARKER,
    GeneratedFile,
    GeneratedFileFamily,
    GeneratedFileSet,
    generated_comment,
    inspect_generated_file_set,
    line_comment_header,
    maintain_generated_file_families,
    maintain_generated_file_set,
    update_generated_file_set,
)


def test_line_comment_header() -> None:
    assert line_comment_header(
        "//",
        generator="loom.gen.example",
        regenerate="python3 run.py",
    ) == [
        f"// {GENERATED_FILE_MARKER}",
        "// Generator: loom.gen.example.",
        "// Regenerate: python3 run.py",
    ]


def test_generated_comment() -> None:
    assert generated_comment(generator="loom.gen.example") == (f"{GENERATED_FILE_MARKER} Generator: loom.gen.example.")


def test_generated_file_set_normalizes_ownership() -> None:
    generated_file_set = GeneratedFileSet.from_mapping(
        {
            "generated/z.txt": "z\n",
            "generated/a.txt": "a\n",
        },
        obsolete_paths=("generated/old.txt", "generated/old.txt"),
    )

    assert generated_file_set.output_paths == (
        "generated/a.txt",
        "generated/z.txt",
    )
    assert generated_file_set.obsolete_paths == ("generated/old.txt",)


@pytest.mark.parametrize(
    "path",
    ["", ".", "/absolute", "../outside", "generated/../outside", "generated\\file"],
)
def test_generated_file_set_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="generated file path"):
        GeneratedFileSet.from_mapping({path: "contents"})


def test_generated_file_set_rejects_overlapping_ownership() -> None:
    with pytest.raises(ValueError, match="overlap"):
        GeneratedFileSet.from_mapping(
            {"generated/file.txt": "contents"},
            obsolete_paths=("generated/file.txt",),
        )


def test_generated_file_set_rejects_noncanonical_direct_construction() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        GeneratedFileSet(
            (
                GeneratedFile("generated/z.txt", "z\n"),
                GeneratedFile("generated/a.txt", "a\n"),
            )
        )


def test_inspect_and_update_generated_file_set(tmp_path: Path) -> None:
    stale_path = tmp_path / "generated/stale.txt"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("old\n", encoding="utf-8")
    obsolete_path = tmp_path / "generated/obsolete.txt"
    obsolete_path.write_text("obsolete\n", encoding="utf-8")
    generated_file_set = GeneratedFileSet.from_mapping(
        {
            "generated/missing.txt": "created\n",
            "generated/stale.txt": "updated\n",
        },
        obsolete_paths=("generated/obsolete.txt",),
    )

    assert [(issue.path, issue.reason) for issue in inspect_generated_file_set(tmp_path, generated_file_set)] == [
        ("generated/missing.txt", "missing generated file"),
        ("generated/stale.txt", "stale generated file"),
        ("generated/obsolete.txt", "obsolete generated file"),
    ]
    assert update_generated_file_set(tmp_path, generated_file_set) == (
        "generated/missing.txt",
        "generated/obsolete.txt",
        "generated/stale.txt",
    )
    assert inspect_generated_file_set(tmp_path, generated_file_set) == ()
    assert update_generated_file_set(tmp_path, generated_file_set) == ()


def test_update_generated_file_set_replaces_output_link(tmp_path: Path) -> None:
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_path = outside_directory / "outside.txt"
    outside_path.write_text("outside\n", encoding="utf-8")
    generated_path = tmp_path / "generated/file.txt"
    generated_path.parent.mkdir(parents=True)
    if os.name == "nt":
        completed = subprocess.run(
            [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/d",
                "/c",
                "mklink",
                "/J",
                str(generated_path),
                str(outside_directory),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout
        assert generated_path.is_junction()
    else:
        generated_path.symlink_to(outside_path)
    generated_file_set = GeneratedFileSet.from_mapping({"generated/file.txt": "generated\n"})

    assert inspect_generated_file_set(tmp_path, generated_file_set)[0].reason == ("generated file must not be a filesystem link")
    assert update_generated_file_set(tmp_path, generated_file_set) == ("generated/file.txt",)
    assert not generated_path.is_symlink()
    assert not generated_path.is_junction()
    assert generated_path.read_text(encoding="utf-8") == "generated\n"
    assert outside_path.read_text(encoding="utf-8") == "outside\n"


def test_maintain_generated_file_set_reports_check_and_update(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generated_file_set = GeneratedFileSet.from_mapping({"generated/file.txt": "contents\n"})

    check_result = maintain_generated_file_set(
        tmp_path,
        generated_file_set,
        mode="check",
        description="example artifacts",
        regenerate_command="python generate.py --in-place",
    )
    assert not check_result.ok
    assert check_result.changed_paths == ()
    assert "generated/file.txt: missing generated file" in capsys.readouterr().err

    update_result = maintain_generated_file_set(
        tmp_path,
        generated_file_set,
        mode="update",
        description="example artifacts",
        regenerate_command="python generate.py --in-place",
    )
    assert update_result.ok
    assert update_result.changed_paths == ("generated/file.txt",)
    assert "updated 1 of 1 generated files" in capsys.readouterr().out


def test_maintain_generated_file_families_reports_exact_changed_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    families = (
        GeneratedFileFamily(
            description="first artifacts",
            regenerate_command="python first.py --in-place",
            file_set=GeneratedFileSet.from_mapping({"generated/first.txt": "first\n"}),
        ),
        GeneratedFileFamily(
            description="second artifacts",
            regenerate_command="python second.py --in-place",
            file_set=GeneratedFileSet.from_mapping({"generated/second.txt": "second\n"}),
        ),
    )

    result = maintain_generated_file_families(tmp_path, families, mode="update")

    assert result.ok
    assert result.changed_paths == (
        "generated/first.txt",
        "generated/second.txt",
    )
    output = capsys.readouterr().out
    assert "first artifacts: updated 1 of 1 generated files" in output
    assert "second artifacts: updated 1 of 1 generated files" in output


def test_maintain_generated_file_families_rejects_shared_ownership(
    tmp_path: Path,
) -> None:
    shared_path = "generated/shared.txt"
    families = (
        GeneratedFileFamily(
            description="current artifacts",
            regenerate_command="python current.py --in-place",
            file_set=GeneratedFileSet.from_mapping({shared_path: "current\n"}),
        ),
        GeneratedFileFamily(
            description="legacy artifacts",
            regenerate_command="python legacy.py --in-place",
            file_set=GeneratedFileSet.from_mapping(
                {},
                obsolete_paths=(shared_path,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="owned by both"):
        maintain_generated_file_families(tmp_path, families, mode="update")

    assert not (tmp_path / shared_path).exists()
