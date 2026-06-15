# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
from pathlib import Path

from loom.diagnostics import DiagnosticSeverity, SourceProvenance
from loom.tools.loom_opt import (
    LOOM_BIN_DIR_ENV_VAR,
    LoomOpt,
    parse_loom_diagnostic_stream,
)


def test_resolve_prefers_explicit_path() -> None:
    resolved = LoomOpt.resolve(Path("/tmp/loom/bin/loom-opt"))

    assert resolved is not None
    assert resolved.executable == Path("/tmp/loom/bin/loom-opt")


def test_resolve_uses_loom_bin_dir_override() -> None:
    old_value = os.environ.get(LOOM_BIN_DIR_ENV_VAR)
    os.environ[LOOM_BIN_DIR_ENV_VAR] = "/opt/loom/bin"
    try:
        resolved = LoomOpt.resolve()
    finally:
        if old_value is None:
            os.environ.pop(LOOM_BIN_DIR_ENV_VAR, None)
        else:
            os.environ[LOOM_BIN_DIR_ENV_VAR] = old_value

    assert resolved is not None
    assert resolved.executable == Path("/opt/loom/bin/loom-opt")


def test_parse_diagnostic_jsonl_preserves_structured_fields() -> None:
    payload = {
        "severity": "error",
        "error_id": "ERR_TYPE_001",
        "domain": "TYPE",
        "code": 1,
        "summary": "SameType constraint violated.",
        "emitter": "verifier",
        "source_location": {
            "provenance": "exact_source",
            "filename": "kernel.loom",
            "start_line": 3,
            "start_column": 7,
            "end_line": 3,
            "end_column": 9,
            "start_byte": 42,
            "end_byte": 44,
            "excerpt": {"text": "  %x = test.addi %a, %b : i32"},
        },
        "message": "'lhs' type i32 does not match 'rhs' type f32",
        "params": {
            "field_a": "lhs",
            "type_a": "i32",
            "field_b": "rhs",
            "type_b": "f32",
        },
        "param_fields": {
            "field_a": {"kind": "operand", "index": 0, "occurrence": 0},
        },
        "fix_hint": "Ensure 'lhs' and 'rhs' have the same type",
    }

    diagnostics, residual = parse_loom_diagnostic_stream(
        f"{json.dumps(payload)}\nsummary status line\n"
    )

    assert residual == "summary status line\n"
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.severity == DiagnosticSeverity.ERROR
    assert diagnostic.domain == "TYPE"
    assert diagnostic.code == "1"
    assert diagnostic.message == "'lhs' type i32 does not match 'rhs' type f32"
    assert diagnostic.rendered_params() == {
        "field_a": "lhs",
        "type_a": "i32",
        "field_b": "rhs",
        "type_b": "f32",
    }
    assert diagnostic.primary_location is not None
    assert diagnostic.primary_location.filename == Path("kernel.loom")
    assert diagnostic.primary_location.provenance == SourceProvenance.EXACT_SOURCE
    assert diagnostic.primary_location.start_line == 3
    assert diagnostic.params[0].field_ref is not None
    assert diagnostic.params[0].field_ref.kind == "operand"
