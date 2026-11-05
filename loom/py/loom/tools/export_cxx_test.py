# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public command behavior for readable C++ export."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

from loom.format.cxx.test_support import LOOP_MODULE, SOURCE_MODULE
from loom.tools.export_cxx import run


def test_cli_emits_utf8_with_legacy_stdout_encoding(tmp_path: Path) -> None:
    source = SOURCE_MODULE.replace(
        "  kernel.return",
        '  %text = buffer.constant utf8("café 🧵") : buffer\n  kernel.return',
    )
    process = subprocess.run(
        [sys.executable, "-m", "loom.tools.export_cxx", "-"],
        input=source.encode("utf-8"),
        capture_output=True,
        env={
            **os.environ,
            "PYTHONIOENCODING": "cp1252",
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stderr == b""
    assert 'u8"café 🧵"'.encode() in process.stdout

    output_path = tmp_path / "output.cxx"
    assert (
        run(
            "-",
            str(output_path),
            stdin=io.BytesIO(source.encode("utf-8")),
            stdout=io.BytesIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert output_path.read_bytes() == process.stdout


def test_reads_text_from_stdin_and_writes_cxx_to_stdout() -> None:
    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = run(
        "-",
        "-",
        stdin=io.BytesIO(SOURCE_MODULE.encode()),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert b"void copy_one(" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_writes_an_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.loom"
    output_path = tmp_path / "output.cxx"
    input_path.write_text(SOURCE_MODULE)

    exit_code = run(
        str(input_path),
        str(output_path),
        stdin=io.BytesIO(),
        stdout=io.BytesIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert "void copy_one(" in output_path.read_text()


def test_selects_repeated_callable_roots_from_a_mixed_module() -> None:
    source = (
        "func.def @helper() {\n  func.return\n}\n\n"
        + SOURCE_MODULE
        + "\n"
        + LOOP_MODULE
    )
    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = run(
        "-",
        "-",
        roots=("@accumulate", "@copy_one"),
        stdin=io.BytesIO(source.encode()),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue().index(b"void copy_one(") < stdout.getvalue().index(
        b"void accumulate()"
    )
    assert b"helper" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_reports_missing_callable_root_without_a_traceback() -> None:
    stderr = io.StringIO()

    exit_code = run(
        "-",
        "-",
        roots=("@missing",),
        stdin=io.BytesIO(SOURCE_MODULE.encode()),
        stdout=io.BytesIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert stderr.getvalue() == (
        "loom-export-cxx: error: cannot project module: "
        "callable root '@missing' was not found\n"
    )


def test_reports_invalid_input_without_a_traceback() -> None:
    stderr = io.StringIO()

    exit_code = run(
        "-",
        "-",
        stdin=io.BytesIO(b"not loom"),
        stdout=io.BytesIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert stderr.getvalue().startswith("loom-export-cxx: error: ")


def test_rejects_opaque_signature_without_emitting_partial_cpp() -> None:
    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = run(
        "-",
        "-",
        stdin=io.BytesIO(b"func.decl @create() -> (hal.buffer)\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert stdout.getvalue() == b""
    assert stderr.getvalue() == (
        "loom-export-cxx: error: cannot project module: "
        "value type hal.buffer has no C++ value spelling\n"
    )
