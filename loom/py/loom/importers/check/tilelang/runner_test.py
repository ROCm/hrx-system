# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest

from loom.importers.check.tilelang.backend import TileLangBackend
from loom.importers.check.tilelang.runner import (
    TileLangCheckOptions,
    run_tilelang_check,
)


def test_run_tilelang_check_updates_and_verifies_python_fixture_with_tilelang() -> None:
    _require_tilelang()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tilelang_cases.py"
        path.write_text(
            """
from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


# ====
@tilelang_case(name="copy")
def copy(T: Any) -> TileLangImportInput:
    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float32),
        B: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            B[0] = A[0]

    return TileLangImportInput(source=main, target="hip", name="copy")


# ----
# old
"""
        )

        update_results = run_tilelang_check(
            path,
            options=TileLangCheckOptions(update=True),
        )
        updated_source = path.read_text()
        verify_results = run_tilelang_check(
            path,
            options=TileLangCheckOptions(),
        )

    assert [result.status for result in update_results] == ["updated"]
    assert [result.status for result in verify_results] == ["passed"]
    assert 'kernel.def target(@hip) export("copy")' in updated_source
    assert "view.load" in updated_source
    assert "view.store" in updated_source


def _require_tilelang() -> None:
    backend = TileLangBackend()
    availability = backend.probe()
    if availability.available:
        availability = backend.prepare(Namespace())
    if not availability.available:
        raise SkipTest(availability.message())
