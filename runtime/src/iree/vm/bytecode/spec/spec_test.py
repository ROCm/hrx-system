# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the contracts consumed by specification projections."""

from __future__ import annotations

import unittest

from iree.vm.bytecode.spec import isa, module, schema
from iree.vm.bytecode.spec.isa.core import INTEGER_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core import rules as isa_rules
from iree.vm.bytecode.spec.isa.core.control import BranchCondition
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.module.container import MODULE_FORMAT
from iree.vm.bytecode.spec.module.numeric import NUMERIC_TABLES
from iree.vm.bytecode.spec.module.records import (
    IMAGE_HEADER,
    SECTION_DIRECTORY_ROW,
    SIGNATURE_DESCRIPTOR_ROW,
    SIGNATURE_ROW,
    SIGNATURES_HEADER,
)
from iree.vm.bytecode.spec.module.validation import project_module_format
from iree.vm.bytecode.spec.specification import SPECIFICATION, Specification
from iree.vm.bytecode.spec.version import CORE_0, Version


def _instruction(mnemonic: str):
    return next(
        instruction
        for instruction in SPECIFICATION.instructions
        if instruction.mnemonic == mnemonic
    )


def _exact_field(name, encoding, data: bytes) -> module.WireField:
    return module.WireField(
        schema.Field(
            name, encoding, "Synthetic field.", len(data) // encoding.byte_length
        ),
        module.FieldRuleUse(module_rules.FieldRule.EXACT_BYTES, data=data),
    )


def _record(name: str, *fields: module.WireField) -> module.WireRecord:
    return SIGNATURES_HEADER._replace(
        name=name,
        c_type=f"iree_vm_bytecode_{name}_t",
        summary="Synthetic record.",
        contract="The record exercises explicit natural layout.",
        fields=fields,
    )


def _section(record, section_type: int, since: Version = CORE_0) -> module.Section:
    field_name = record.fields[0].field.name
    constraint = module.StructuralConstraint(
        "section_shape",
        since,
        (module.RecordFieldReference(record, field_name),),
        "The section owns its record and referenced field.",
    )
    return MODULE_FORMAT.sections[0]._replace(
        name=f"test_section_{section_type}",
        section_type=section_type,
        since=since,
        records=(record,),
        constraints=(constraint,),
    )


def _module_format(*sections: module.Section, numeric_tables=()) -> module.ModuleFormat:
    return MODULE_FORMAT._replace(
        numeric_tables=numeric_tables,
        envelope=(IMAGE_HEADER,),
        sections=sections,
        constraints=(),
    )


def _specification(
    *,
    version: Version = CORE_0,
    families=SPECIFICATION.families,
    selectors=SPECIFICATION.selectors,
    instructions=(),
    module_format: module.ModuleFormat | None = None,
) -> Specification:
    return Specification(
        "test",
        version,
        families,
        selectors,
        instructions,
        module_format or _module_format(),
    )


class SpecificationTest(unittest.TestCase):
    def test_instruction_wire_layouts(self) -> None:
        cases = (
            ("integer.bitstream.pack", 0x76, 8, (1, 2, 3, 4, 5, 6, 7)),
            ("control.branch.if.s16", 0x06, 4, (1, 2)),
            ("control.branch.if.s32", 0x07, 8, (1, 2, 4)),
            ("control.switch", 0x0A, 8, (1, 2, 4)),
            ("control.call", 0x0B, 8, (1, 2, 4, 6)),
        )
        for mnemonic, opcode, byte_length, field_offsets in cases:
            with self.subTest(mnemonic=mnemonic):
                instruction = _instruction(mnemonic)
                self.assertEqual(
                    (
                        instruction.opcode,
                        instruction.byte_length,
                        instruction.field_offsets,
                    ),
                    (opcode, byte_length, field_offsets),
                )

    def test_module_record_layouts_are_abi_edges(self) -> None:
        records = (
            IMAGE_HEADER,
            SIGNATURES_HEADER,
            SIGNATURE_ROW,
            SIGNATURE_DESCRIPTOR_ROW,
        )
        self.assertEqual(
            tuple((record.byte_length, record.alignment) for record in records),
            ((16, 2), (4, 4), (16, 4), (4, 2)),
        )

    def test_complete_transcription_inventory(self) -> None:
        self.assertEqual(MODULE_FORMAT.envelope, (IMAGE_HEADER, SECTION_DIRECTORY_ROW))
        self.assertEqual(len(SPECIFICATION.module_format.records), 34)
        self.assertEqual(
            (
                len(SPECIFICATION.families),
                len(SPECIFICATION.instructions),
                len(SPECIFICATION.selectors),
                sum(len(table.values) for table in SPECIFICATION.selectors),
                tuple(section.section_type for section in MODULE_FORMAT.sections),
                len(MODULE_FORMAT.constraints)
                + sum(len(section.constraints) for section in MODULE_FORMAT.sections),
                len(NUMERIC_TABLES),
                sum(len(table.values) for table in NUMERIC_TABLES),
            ),
            (10, 152, 16, 171, tuple(range(1, 14)), 58, 9, 30),
        )

    def test_layout_rejects_implicit_alignment_padding(self) -> None:
        record = _record(
            "misaligned",
            _exact_field("prefix_u16", schema.U16, b"xx"),
            _exact_field("payload_u64", schema.U64, b"yyyyyyyy"),
        )
        with self.assertRaisesRegex(ValueError, "add explicit padding"):
            _specification(module_format=_module_format(_section(record, 1)))

    def test_layout_accepts_explicit_alignment_padding(self) -> None:
        record = _record(
            "aligned",
            _exact_field("prefix_u16", schema.U16, b"xx"),
            _exact_field("zero_padding_u8", schema.U8, bytes(6)),
            _exact_field("payload_u64", schema.U64, b"yyyyyyyy"),
        )
        specification = _specification(
            module_format=_module_format(_section(record, 1))
        )
        selected_record = specification.module_format.sections[0].records[0]
        self.assertEqual(selected_record.byte_length, 16)
        self.assertEqual(selected_record.alignment, 8)

    def test_duplicate_opcode_is_rejected(self) -> None:
        self.assertEqual(
            tuple(item.opcode for item in INTEGER_INSTRUCTIONS),
            tuple(range(0x40, 0x79)),
        )
        duplicate = INTEGER_INSTRUCTIONS[1]._replace(
            opcode=INTEGER_INSTRUCTIONS[0].opcode,
        )
        with self.assertRaisesRegex(ValueError, "duplicate opcode"):
            _specification(instructions=(INTEGER_INSTRUCTIONS[0], duplicate))

    def test_control_contracts_reject_inconsistent_declarations(self) -> None:
        branch = _instruction("control.branch.s16")
        switch = _instruction("control.switch")
        cases = (
            branch._replace(
                control_flow=isa.ControlFlow.SEQUENTIAL,
                semantics=BranchCondition.ALWAYS,
            ),
            branch._replace(suspension=isa.Suspension.ALWAYS),
            switch._replace(
                rules=(
                    switch.rules[0]._replace(fields=("missing_u16", "target_base_u32")),
                ),
            ),
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    _specification(instructions=(malformed,))

    def test_padding_must_be_canonical_zero(self) -> None:
        block = _instruction("control.block")
        malformed_field = isa.InstructionField(
            block.fields[0].field,
            isa.FieldRole.PADDING,
            isa.FieldRuleUse(isa_rules.FieldRule.ANY_BITS),
        )
        malformed = block._replace(fields=(malformed_field,))
        with self.assertRaisesRegex(ValueError, "padding is not canonical zero"):
            _specification(instructions=(malformed,))

    def test_minor_projection_filters_concrete_declarations(self) -> None:
        core_1 = Version("core", 0, 1)
        future_record = SIGNATURES_HEADER._replace(
            name="future_header",
            c_type="iree_vm_bytecode_future_header_t",
            since=core_1,
        )
        current = _section(SIGNATURES_HEADER, 4)
        future = _section(future_record, 5, core_1)
        specification = _specification(
            version=core_1,
            instructions=SPECIFICATION.instructions,
            module_format=_module_format(current, future),
        )
        self.assertNotIn(
            future_record, specification.project(CORE_0).module_format.records
        )
        self.assertIn(
            future_record, specification.project(core_1).module_format.records
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            SPECIFICATION.project(core_1)

    def test_invalid_specification_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid specification version"):
            _specification(version=Version("Core", 0, 0))

    def test_invalid_versions_are_never_available(self) -> None:
        self.assertFalse(Version("core", 0, -1).is_available_in(CORE_0))
        self.assertFalse(CORE_0.is_available_in(Version("core", 0, -1)))

    def test_numeric_tables_reject_aliases_and_project_values(self) -> None:
        core_1 = Version("core", 0, 1)
        table = schema.NumericTable(
            "test.selector",
            schema.U8,
            schema.NumericKind.SELECTOR,
            (
                schema.NumericValue("zero", 0, CORE_0, "Initial value."),
                schema.NumericValue("one", 1, core_1, "Added value."),
            ),
            CORE_0,
            "Synthetic selector table.",
        )
        schema.validate_numeric_table(table, core_1)
        with self.assertRaisesRegex(ValueError, "invalid numeric table"):
            schema.validate_numeric_table(table, CORE_0)
        projected = project_module_format(
            _module_format(numeric_tables=(table,)), CORE_0
        )
        self.assertEqual(len(projected.numeric_tables[0].values), 1)
        projected = _specification(version=core_1, selectors=(table,)).project(CORE_0)
        self.assertEqual(len(projected.selectors[0].values), 1)
        alias = table.values[1]._replace(value=0)
        with self.assertRaisesRegex(ValueError, "invalid numeric table"):
            schema.validate_numeric_table(
                table._replace(values=(table.values[0], alias)), core_1
            )

    def test_instruction_extension_contracts_reject_malformed_uses(self) -> None:
        instruction = _instruction("integer.add.i32")
        result = instruction.fields[0]
        malformed_rule = result._replace(
            rule=isa.FieldRuleUse(
                isa_rules.FieldRule.REGISTER_VALUE,
                values=(0,),
            )
        )
        misplaced_policy = result._replace(
            ref_policy=isa.RuntimeRefPolicy(
                "any",
                isa_rules.RefNullPolicy.NULLABLE,
                isa_rules.RefOwnership.BORROW,
            )
        )
        malformed_effect = isa.StateEffect(
            isa_rules.StateAccess.UNKNOWN, isa_rules.StateResource.BUFFER
        )
        cases = (
            instruction._replace(fields=(malformed_rule, *instruction.fields[1:])),
            instruction._replace(fields=(misplaced_policy, *instruction.fields[1:])),
            instruction._replace(state_effects=(malformed_effect,)),
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    _specification(instructions=(malformed,))
        with self.assertRaisesRegex(ValueError, "malformed field rule"):
            _specification(selectors=(), instructions=(_instruction("control.fail"),))

    def test_module_contracts_reject_malformed_declarations(self) -> None:
        duplicate = SIGNATURES_HEADER._replace(
            c_type="iree_vm_bytecode_duplicate_header_t"
        )
        fields = list(SIGNATURE_DESCRIPTOR_ROW.fields)
        fields[-1] = fields[-1]._replace(
            rule=module.FieldRuleUse(module_rules.FieldRule.SIGNATURE_DESCRIPTOR)
        )
        malformed_rule = SIGNATURE_DESCRIPTOR_ROW._replace(fields=tuple(fields))
        cases = (
            (_module_format(_section(SIGNATURES_HEADER, 0)), "invalid module section"),
            (
                _module_format(_section(SIGNATURES_HEADER, 1), _section(duplicate, 2)),
                "invalid module format",
            ),
            (_module_format(_section(malformed_rule, 1)), "invalid field rule"),
        )
        for module_format, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _specification(module_format=module_format)


if __name__ == "__main__":
    unittest.main()
