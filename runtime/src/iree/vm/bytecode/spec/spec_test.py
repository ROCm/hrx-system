# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the contracts consumed by specification projections."""

from __future__ import annotations

import unittest

from iree.vm.bytecode.spec.isa import (
    BranchCondition,
    ControlFlow,
    FieldRole,
    InstructionField,
    Suspension,
    SwitchTargetsRule,
)
from iree.vm.bytecode.spec.isa import (
    FieldRule as InstructionFieldRule,
)
from iree.vm.bytecode.spec.isa.core import INTEGER_INSTRUCTIONS
from iree.vm.bytecode.spec.module import ExactBytesRule, WireField, WireRecord
from iree.vm.bytecode.spec.module.records import (
    IMAGE_HEADER,
    SIGNATURE_DESCRIPTOR_ROW,
    SIGNATURE_ROW,
    SIGNATURES_HEADER,
)
from iree.vm.bytecode.spec.schema import U8, U16, U64, Field
from iree.vm.bytecode.spec.specification import SPECIFICATION, Specification
from iree.vm.bytecode.spec.version import CORE_0, Version


def _instruction(mnemonic: str):
    return next(
        instruction
        for instruction in SPECIFICATION.instructions
        if instruction.mnemonic == mnemonic
    )


class SpecificationTest(unittest.TestCase):
    def test_integer_family_has_dense_four_byte_records(self) -> None:
        instruction = _instruction("integer.add.i32")
        self.assertEqual(instruction.opcode, 0x40)
        self.assertEqual(instruction.byte_length, 4)
        self.assertEqual(instruction.field_offsets, (1, 2, 3))

    def test_control_records_preserve_natural_layout(self) -> None:
        narrow = _instruction("control.branch.if.s16")
        wide = _instruction("control.branch.if.s32")
        switch = _instruction("control.switch")
        self.assertEqual(narrow.byte_length, 4)
        self.assertEqual(wide.byte_length, 8)
        self.assertEqual(wide.field_offsets, (1, 2, 4))
        self.assertEqual(switch.field_offsets, (1, 2, 4))

    def test_module_record_layouts_are_abi_edges(self) -> None:
        self.assertEqual((IMAGE_HEADER.byte_length, IMAGE_HEADER.alignment), (16, 2))
        self.assertEqual(
            (SIGNATURES_HEADER.byte_length, SIGNATURES_HEADER.alignment), (4, 4)
        )
        self.assertEqual((SIGNATURE_ROW.byte_length, SIGNATURE_ROW.alignment), (16, 4))
        self.assertEqual(
            (SIGNATURE_DESCRIPTOR_ROW.byte_length, SIGNATURE_DESCRIPTOR_ROW.alignment),
            (4, 2),
        )

    def test_layout_rejects_implicit_alignment_padding(self) -> None:
        record = WireRecord(
            name="misaligned",
            c_type="iree_vm_bytecode_misaligned_t",
            since=CORE_0,
            summary="Synthetic misaligned record.",
            contract="The test record must require explicit padding.",
            fields=(
                WireField(Field("prefix_u16", U16, "Prefix."), ExactBytesRule(b"xx")),
                WireField(
                    Field("payload_u64", U64, "Payload."),
                    ExactBytesRule(b"yyyyyyyy"),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "add explicit padding"):
            Specification("test", CORE_0, (), (), (record,))

    def test_layout_accepts_explicit_alignment_padding(self) -> None:
        record = WireRecord(
            name="aligned",
            c_type="iree_vm_bytecode_aligned_t",
            since=CORE_0,
            summary="Synthetic aligned record.",
            contract="The test record carries explicit alignment padding.",
            fields=(
                WireField(Field("prefix_u16", U16, "Prefix."), ExactBytesRule(b"xx")),
                WireField(
                    Field("zero_padding_u8", U8, "Padding.", 6),
                    ExactBytesRule(bytes(6)),
                ),
                WireField(
                    Field("payload_u64", U64, "Payload."),
                    ExactBytesRule(b"yyyyyyyy"),
                ),
            ),
        )
        specification = Specification("test", CORE_0, (), (), (record,))
        self.assertEqual(specification.records[0].byte_length, 16)
        self.assertEqual(specification.records[0].alignment, 8)

    def test_duplicate_opcode_is_rejected(self) -> None:
        duplicate = INTEGER_INSTRUCTIONS[1]._replace(
            opcode=INTEGER_INSTRUCTIONS[0].opcode,
        )
        with self.assertRaisesRegex(ValueError, "duplicate opcode"):
            Specification(
                "test",
                CORE_0,
                SPECIFICATION.families,
                (INTEGER_INSTRUCTIONS[0], duplicate),
                (),
            )

    def test_control_target_requires_matching_flow(self) -> None:
        branch = _instruction("control.branch.s16")
        malformed = branch._replace(
            control_flow=ControlFlow.SEQUENTIAL,
            semantics=BranchCondition.ALWAYS,
        )
        with self.assertRaisesRegex(ValueError, "invalid direct-target count"):
            Specification("test", CORE_0, SPECIFICATION.families, (malformed,), ())

    def test_yield_is_the_only_always_suspending_control(self) -> None:
        branch = _instruction("control.branch.s16")
        malformed = branch._replace(suspension=Suspension.ALWAYS)
        with self.assertRaisesRegex(ValueError, "inconsistent suspension"):
            Specification("test", CORE_0, SPECIFICATION.families, (malformed,), ())

    def test_switch_rule_must_name_encoded_fields(self) -> None:
        switch = _instruction("control.switch")
        malformed = switch._replace(
            rules=(SwitchTargetsRule("missing_u16", "target_base_u32"),),
        )
        with self.assertRaisesRegex(ValueError, "names missing field"):
            Specification("test", CORE_0, SPECIFICATION.families, (malformed,), ())

    def test_padding_must_be_canonical_zero(self) -> None:
        block = _instruction("control.block")
        malformed_field = InstructionField(
            block.fields[0].field,
            FieldRole.PADDING,
            InstructionFieldRule.ANY_BITS,
        )
        malformed = block._replace(fields=(malformed_field,))
        with self.assertRaisesRegex(ValueError, "padding is not canonical zero"):
            Specification("test", CORE_0, SPECIFICATION.families, (malformed,), ())

    def test_minor_projection_filters_concrete_declarations(self) -> None:
        core_1 = Version("core", 0, 1)
        future_record = SIGNATURES_HEADER._replace(
            name="future_header",
            c_type="iree_vm_bytecode_future_header_t",
            since=core_1,
        )
        specification = Specification(
            "test",
            core_1,
            SPECIFICATION.families,
            SPECIFICATION.instructions,
            (*SPECIFICATION.records, future_record),
        )
        self.assertNotIn(future_record, specification.project(CORE_0).records)
        self.assertIn(future_record, specification.project(core_1).records)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            SPECIFICATION.project(core_1)


if __name__ == "__main__":
    unittest.main()
