# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, cast

from loom.importers.check.tilelang import (
    TileLangCaseMetadata,
    TileLangImportInput,
    get_tilelang_case_metadata,
    is_tilelang_case,
    tilelang_case,
)


def test_tilelang_case_decorator_attaches_metadata() -> None:
    @tilelang_case(name="vector_add", category="op", tags=("elementwise",))
    def vector_add() -> object:
        return object()

    assert is_tilelang_case(vector_add)
    assert get_tilelang_case_metadata(vector_add) == TileLangCaseMetadata(
        name="vector_add",
        category="op",
        tags=("elementwise",),
    )


def test_tilelang_import_input_freezes_mappings() -> None:
    kwargs = {"M": 128}
    metadata = {"source": "direct"}
    import_input = TileLangImportInput(
        source="prim_func",
        kwargs=kwargs,
        metadata=metadata,
    )
    kwargs["M"] = 256
    metadata["source"] = "jit"

    assert import_input.kwargs["M"] == 128
    assert import_input.metadata["source"] == "direct"
    try:
        cast(Any, import_input.kwargs)["M"] = 512
    except TypeError:
        pass
    else:
        raise AssertionError("expected TileLangImportInput kwargs to be read-only")
