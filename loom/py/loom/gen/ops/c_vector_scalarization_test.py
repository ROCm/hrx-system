# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Family-completeness tests for vector scalarization metadata."""

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector.defs import VECTOR_CAST_OPS
from loom.gen.ops.c_traits import (
    _is_shape_preserving_elementwise_vector_decomposable,
)
from loom.gen.ops.c_vector_scalarization import (
    _EXPLICIT_VECTOR_LANE_PROGRAMS,
    collect_vector_scalarization_rows,
)


def test_every_decomposable_vector_op_has_a_reference_lane_program() -> None:
    decomposable_ops = {op.name for op in ALL_VECTOR_OPS if _is_shape_preserving_elementwise_vector_decomposable(op) or any(trait.name == "Decomposable" for trait in op.traits)}
    generated_scalarizations = {row.vector_op.name for row in collect_vector_scalarization_rows()}

    assert decomposable_ops == generated_scalarizations | _EXPLICIT_VECTOR_LANE_PROGRAMS


def test_vector_index_cast_uses_index_dialect_scalar_semantics() -> None:
    rows = {row.vector_op.name: row for row in collect_vector_scalarization_rows()}

    assert rows["vector.index_cast"].scalar_op.name == "index.cast"


def test_vector_casts_have_reference_lane_programs() -> None:
    generated_scalarizations = {row.vector_op.name for row in collect_vector_scalarization_rows()}
    cast_ops = {op.name for op in VECTOR_CAST_OPS}

    assert cast_ops - generated_scalarizations == {"vector.bitcast"}
