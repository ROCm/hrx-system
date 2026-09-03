# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests the contracts consumed by specification projections."""

from __future__ import annotations

import unittest

from iree.vm.bytecode.spec.isa import (
    ControlFlow,
    FieldRole,
    InstructionField,
    RecordRule,
    RuntimeRefPolicy,
    StateEffect,
    Suspension,
)
from iree.vm.bytecode.spec.isa import (
    FieldRuleUse as InstructionFieldRuleUse,
)
from iree.vm.bytecode.spec.isa.core import INTEGER_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core.control import BranchCondition
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule as InstructionFieldRule,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    RecordRuleKind,
    RefNullPolicy,
    RefOwnership,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.module import (
    FieldRuleUse,
    ModuleFormat,
    RecordFieldReference,
    Section,
    StructuralConstraint,
    WireField,
    WireRecord,
)
from iree.vm.bytecode.spec.module.container import MODULE_FORMAT
from iree.vm.bytecode.spec.module.numeric import NUMERIC_TABLES
from iree.vm.bytecode.spec.module.records import (
    IMAGE_HEADER,
    SECTION_DIRECTORY_ROW,
    SIGNATURE_DESCRIPTOR_ROW,
    SIGNATURE_ROW,
    SIGNATURES_HEADER,
)
from iree.vm.bytecode.spec.module.rules import FieldRule
from iree.vm.bytecode.spec.module.validation import (
    project_module_format,
    validate_module_format,
)
from iree.vm.bytecode.spec.schema import (
    U8,
    U16,
    U64,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
    validate_numeric_table,
)
from iree.vm.bytecode.spec.specification import SPECIFICATION, Specification
from iree.vm.bytecode.spec.version import CORE_0, Version


def _instruction(mnemonic: str):
    return next(
        instruction
        for instruction in SPECIFICATION.instructions
        if instruction.mnemonic == mnemonic
    )


def _exact_field(name, encoding, data: bytes) -> WireField:
    return WireField(
        Field(name, encoding, "Synthetic field.", len(data) // encoding.byte_length),
        FieldRuleUse(FieldRule.EXACT_BYTES, data=data),
    )


def _record(name: str, *fields: WireField) -> WireRecord:
    return SIGNATURES_HEADER._replace(
        name=name,
        c_type=f"iree_vm_bytecode_{name}_t",
        summary="Synthetic record.",
        contract="The record exercises explicit natural layout.",
        fields=fields,
    )


def _section(record, section_type: int, since: Version = CORE_0) -> Section:
    field_name = record.fields[0].field.name
    constraint = StructuralConstraint(
        "section_shape",
        since,
        (RecordFieldReference(record, field_name),),
        "The section owns its record and referenced field.",
    )
    return MODULE_FORMAT.sections[0]._replace(
        name=f"test_section_{section_type}",
        section_type=section_type,
        since=since,
        records=(record,),
        constraints=(constraint,),
    )


def _module_format(*sections: Section, numeric_tables=()) -> ModuleFormat:
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
    instructions=(),
    module_format: ModuleFormat | None = None,
) -> Specification:
    module_format = module_format or _module_format()
    return Specification("test", version, families, instructions, module_format)


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

    def test_complete_module_inventory(self) -> None:
        self.assertEqual(MODULE_FORMAT.envelope, (IMAGE_HEADER, SECTION_DIRECTORY_ROW))
        self.assertEqual(len(SPECIFICATION.module_format.records), 34)
        self.assertEqual(
            (
                tuple(section.section_type for section in MODULE_FORMAT.sections),
                len(MODULE_FORMAT.constraints)
                + sum(len(section.constraints) for section in MODULE_FORMAT.sections),
                len(NUMERIC_TABLES),
                sum(len(table.values) for table in NUMERIC_TABLES),
            ),
            (tuple(range(1, 14)), 58, 9, 30),
        )

    def test_layout_rejects_implicit_alignment_padding(self) -> None:
        record = _record(
            "misaligned",
            _exact_field("prefix_u16", U16, b"xx"),
            _exact_field("payload_u64", U64, b"yyyyyyyy"),
        )
        with self.assertRaisesRegex(ValueError, "add explicit padding"):
            _specification(module_format=_module_format(_section(record, 1)))

    def test_layout_accepts_explicit_alignment_padding(self) -> None:
        record = _record(
            "aligned",
            _exact_field("prefix_u16", U16, b"xx"),
            _exact_field("zero_padding_u8", U8, bytes(6)),
            _exact_field("payload_u64", U64, b"yyyyyyyy"),
        )
        specification = _specification(
            module_format=_module_format(_section(record, 1))
        )
        selected_record = specification.module_format.sections[0].records[0]
        self.assertEqual(selected_record.byte_length, 16)
        self.assertEqual(selected_record.alignment, 8)

    def test_duplicate_opcode_is_rejected(self) -> None:
        duplicate = INTEGER_INSTRUCTIONS[1]._replace(
            opcode=INTEGER_INSTRUCTIONS[0].opcode,
        )
        with self.assertRaisesRegex(ValueError, "duplicate opcode"):
            _specification(
                instructions=(INTEGER_INSTRUCTIONS[0], duplicate),
            )

    def test_control_target_requires_matching_flow(self) -> None:
        branch = _instruction("control.branch.s16")
        malformed = branch._replace(
            control_flow=ControlFlow.SEQUENTIAL,
            semantics=BranchCondition.ALWAYS,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent control rule"):
            _specification(instructions=(malformed,))

    def test_yield_is_the_only_always_suspending_control(self) -> None:
        branch = _instruction("control.branch.s16")
        malformed = branch._replace(suspension=Suspension.ALWAYS)
        with self.assertRaisesRegex(ValueError, "inconsistent control rule"):
            _specification(instructions=(malformed,))

    def test_switch_rule_must_name_encoded_fields(self) -> None:
        switch = _instruction("control.switch")
        malformed = switch._replace(
            rules=(
                RecordRule(
                    RecordRuleKind.SWITCH_TARGETS,
                    ("missing_u16", "target_base_u32"),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "names missing field"):
            _specification(instructions=(malformed,))

    def test_padding_must_be_canonical_zero(self) -> None:
        block = _instruction("control.block")
        malformed_field = InstructionField(
            block.fields[0].field,
            FieldRole.PADDING,
            InstructionFieldRule.ANY_BITS,
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
        table = NumericTable(
            "test.selector",
            U8,
            NumericKind.SELECTOR,
            (
                NumericValue("zero", 0, CORE_0, "Initial value."),
                NumericValue("one", 1, core_1, "Added value."),
            ),
            CORE_0,
            "Synthetic selector table.",
        )
        validate_numeric_table(table, core_1)
        projected = project_module_format(
            _module_format(numeric_tables=(table,)), CORE_0
        )
        self.assertEqual(len(projected.numeric_tables[0].values), 1)
        alias = table.values[1]._replace(value=0)
        with self.assertRaisesRegex(ValueError, "invalid numeric table"):
            validate_numeric_table(
                table._replace(values=(table.values[0], alias)), core_1
            )

    def test_instruction_extension_contracts_reject_malformed_uses(self) -> None:
        instruction = _instruction("integer.add.i32")
        result = instruction.fields[0]
        malformed_rule = result._replace(
            rule=InstructionFieldRuleUse(
                InstructionFieldRule.REGISTER_VALUE,
                values=(0,),
            )
        )
        misplaced_policy = result._replace(
            ref_policy=RuntimeRefPolicy(
                "any",
                RefNullPolicy.NULLABLE,
                RefOwnership.BORROW,
            )
        )
        malformed_effect = StateEffect(StateAccess.UNKNOWN, StateResource.BUFFER)
        cases = (
            instruction._replace(fields=(malformed_rule, *instruction.fields[1:])),
            instruction._replace(fields=(misplaced_policy, *instruction.fields[1:])),
            instruction._replace(state_effects=(malformed_effect,)),
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    _specification(
                        instructions=(malformed,),
                    )

    def test_module_sections_reject_invalid_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid module section"):
            validate_module_format(
                _module_format(_section(SIGNATURES_HEADER, 0)), CORE_0
            )

    def test_module_declaration_identities_are_unique(self) -> None:
        duplicate = SIGNATURES_HEADER._replace(
            c_type="iree_vm_bytecode_duplicate_header_t"
        )
        with self.assertRaisesRegex(ValueError, "invalid module format"):
            _specification(
                module_format=_module_format(
                    _section(SIGNATURES_HEADER, 1), _section(duplicate, 2)
                )
            )

    def test_module_related_field_rules_require_their_inputs(self) -> None:
        fields = list(SIGNATURE_DESCRIPTOR_ROW.fields)
        fields[-1] = fields[-1]._replace(
            rule=FieldRuleUse(FieldRule.SIGNATURE_DESCRIPTOR)
        )
        malformed = SIGNATURE_DESCRIPTOR_ROW._replace(fields=tuple(fields))
        with self.assertRaisesRegex(ValueError, "invalid field rule"):
            _specification(module_format=_module_format(_section(malformed, 1)))


if __name__ == "__main__":
    unittest.main()
