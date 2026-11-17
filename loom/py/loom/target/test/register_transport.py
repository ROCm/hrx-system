# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register transports between banks with different allocation-unit widths."""

from loom.target.low_descriptors import (
    AsmForm,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    InstructionClass,
    Operand,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    SpillSlotSpace,
)

TEST_REGISTER_TRANSPORT_CLASSES = (
    RegClass(
        "test.index20",
        alloc_unit_bits=20,
        spill_unit_bits=32,
        spill_slot_space=SpillSlotSpace.PRIVATE,
        flags=(RegClassFlag.VIRTUAL_ONLY,),
    ),
)

TEST_REGISTER_TRANSPORT_DESCRIPTORS = tuple(
    Descriptor(
        key=f"test.move.{name}",
        mnemonic=f"test.move.{name}",
        semantic_tag=f"register.move.{name}",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                reg_alts=(RegClassAlt(result_class),),
                unit_count=result_count,
                ready_stage=1,
            ),
            Operand(
                "src",
                OperandRole.OPERAND,
                reg_alts=(RegClassAlt(source_class),),
                unit_count=source_count,
            ),
        ),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        asm_forms=(
            AsmForm(mnemonic=f"test.move.{name}", results=("dst",), operands=("src",)),
        ),
        schedule_class="test.scalar.alu",
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        instruction_classes=(InstructionClass.REGISTER_MOVE,),
    )
    for name, source_class, source_count, result_class, result_count in (
        ("pair.to.wide", "test.i32", 2, "test.i64", 1),
        ("wide.to.pair", "test.i64", 1, "test.i32", 2),
        ("word.to.wide", "test.i32", 1, "test.i64", 1),
        ("wide.to.word", "test.i64", 1, "test.i32", 1),
        ("word.to.index", "test.i32", 1, "test.index20", 1),
        ("index.to.word", "test.index20", 1, "test.i32", 1),
    )
)
