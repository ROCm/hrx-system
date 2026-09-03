# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.vm.bytecode.spec.isa import IntegerBinarySemantics

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.scalar_type import ScalarTypeKind
from loom.target.arch.vm.projection import SOURCE_LOWERINGS


def test_source_lowerings_reference_real_source_and_vm_declarations() -> None:
    assert len(SOURCE_LOWERINGS) == 6
    assert {lowering.source_op for lowering in SOURCE_LOWERINGS} == {
        scalar_arithmetic.scalar_addi,
        scalar_arithmetic.scalar_subi,
        scalar_arithmetic.scalar_muli,
    }
    assert {lowering.scalar_type for lowering in SOURCE_LOWERINGS} == {
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }
    source_keys = tuple(
        (lowering.source_op_kind, lowering.scalar_type.value)
        for lowering in SOURCE_LOWERINGS
    )
    assert source_keys == tuple(sorted(source_keys))
    assert all(
        isinstance(lowering.instruction.semantics, IntegerBinarySemantics)
        for lowering in SOURCE_LOWERINGS
    )
