# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.check.tilelang import TileLangHarness


def test_harness_input_builds_structured_import_input_without_tilelang() -> None:
    harness = object.__new__(TileLangHarness)

    import_input = harness.input(
        "prim_func",
        args=(1, 2),
        kwargs={"N": 64},
        target="hip",
        name="case0",
        metadata={"kind": "direct"},
    )

    assert import_input.source == "prim_func"
    assert import_input.args == (1, 2)
    assert import_input.kwargs["N"] == 64
    assert import_input.target == "hip"
    assert import_input.name == "case0"
    assert import_input.metadata["kind"] == "direct"
