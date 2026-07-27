# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.mlir.runner import parse_check_cases, parse_mlir_run


def test_parse_check_cases_inherits_first_run_directive() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
// RUN: mlir --kernel first
module @first {}
// ----
first expected
// ====
module @second {}
// ----
second expected
""",
    )

    assert len(cases) == 2
    assert cases[0].run == "mlir --kernel first"
    assert cases[1].run == "mlir --kernel first"
    assert "// RUN:" not in cases[0].input
    assert cases[0].expected == "first expected\n"


def test_parse_check_cases_defaults_to_mlir_without_run_directives() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
hal.executable @first {}
// ----
first expected
// ====
hal.executable @second {}
// ----
second expected
""",
    )

    assert cases[0].run == "mlir"
    assert cases[1].run == "mlir"


def test_parse_check_cases_keeps_case_local_run_directive() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
// RUN: mlir --kernel first
module @first {}
// ----
first expected
// ====
// RUN: mlir --kernel second
module @second {}
// ----
second expected
""",
    )

    assert cases[1].run == "mlir --kernel second"


def test_parse_mlir_run_accepts_kernel_option() -> None:
    assert parse_mlir_run("mlir --kernel kernel").kernel == "kernel"
