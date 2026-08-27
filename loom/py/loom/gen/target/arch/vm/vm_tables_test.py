# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the Loom-facing Core VM table projection."""

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.gen.target.arch.vm.vm_tables import generate_lowering_rows
from loom.scalar_type import ScalarTypeKind
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_INSTRUCTION_PROJECTIONS,
    VM_SOURCE_LOWERINGS,
)


def test_descriptors_preserve_instruction_identity() -> None:
    assert len(VM_CORE_DESCRIPTOR_SET.descriptors) == len(VM_INSTRUCTION_PROJECTIONS)
    for descriptor, projection in zip(
        VM_CORE_DESCRIPTOR_SET.descriptors,
        VM_INSTRUCTION_PROJECTIONS,
        strict=True,
    ):
        assert descriptor.encoding_id == projection.instruction.opcode
        assert descriptor.key == projection.key


def test_source_rows_reference_projected_descriptors() -> None:
    descriptor_keys = {descriptor.key for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    assert VM_SOURCE_LOWERINGS
    assert all(row.descriptor_key in descriptor_keys for row in VM_SOURCE_LOWERINGS)


def test_same_type_projection_derives_source_signatures() -> None:
    add_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is scalar_arithmetic.scalar_addi)
    assert tuple((row.operand_types, row.result_types, row.descriptor_key) for row in add_rows) == (
        (
            (ScalarTypeKind.I32, ScalarTypeKind.I32),
            (ScalarTypeKind.I32,),
            "vm.integer.add.i32",
        ),
        (
            (ScalarTypeKind.I64, ScalarTypeKind.I64),
            (ScalarTypeKind.I64,),
            "vm.integer.add.i64",
        ),
    )


def test_lowering_rows_are_data_only() -> None:
    rows = generate_lowering_rows()
    assert "LOOM_VM_SOURCE_LOWERING_LIMITS(\n    2, 1)" in rows
    assert "LOOM_OP_INDEX_MUL" in rows
    assert "VM_CORE_DESCRIPTOR_REF_INTEGER_MUL_I64" in rows
