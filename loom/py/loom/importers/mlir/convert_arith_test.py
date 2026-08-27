# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import loom
from loom.importers.mlir.convert_arith import convert_index_cast
from loom.importers.mlir.model import MlirConversionContext, SourceOp


class _MlirOperation:
    name = "arith.index_castui"

    def get_asm(self, *, enable_debug_info: bool, use_local_scope: bool) -> str:
        return "%result = arith.index_castui %input : i32 to index"


def test_unsigned_index_cast_is_not_silently_reinterpreted_as_signed() -> None:
    _, builder = loom.module_builder()
    context = MlirConversionContext.with_prelude(builder, {})
    source_op = SourceOp(
        source=_MlirOperation(),
        depth=1,
    )

    assert convert_index_cast(source_op, context)

    blocked = context.finish().blocked
    assert len(blocked) == 1
    assert blocked[0].source == ("%result = arith.index_castui %input : i32 to index")
    assert blocked[0].target == (
        "arith.index_castui cannot map to signed Loom index.cast semantics",
    )
