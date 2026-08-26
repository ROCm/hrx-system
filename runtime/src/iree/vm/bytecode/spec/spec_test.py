#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises historical projection and dependency closure independently."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import unittest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from model import (  # noqa: E402
    U8,
    Entity,
    NumericTable,
    NumericTableKind,
    NumericValue,
    RuleUse,
    ScalarEncoding,
    Specification,
    UnknownNumericValuePolicy,
    ValidationRule,
    ValidationScope,
    Version,
    VersionDomain,
    WireField,
    WireRecord,
    WireRecordLayout,
    compare_projections,
    selected_numeric_values,
    selected_record_layouts,
)
from model.isa import (  # noqa: E402
    ControlFlow,
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionField,
    InstructionFieldRole,
    InstructionSemantics,
    StateAccess,
    StateEffect,
    StateResource,
    Suspension,
)


def entity(
    entity_id: str,
    minor: int,
    summary: str,
    *dependencies: str,
    domain: str = "core",
) -> Entity:
    return Entity(
        entity_id=entity_id,
        since=Version(domain, 0, minor),
        summary=summary,
        dependencies=tuple(sorted(dependencies)),
    )


CORE_0 = VersionDomain("core", 0x00, 0, 0, "Core format and ISA.")
CORE_1 = dataclasses.replace(CORE_0, latest_minor=1)
HAL_0 = VersionDomain("hal", 0xF0, 0, 0, "Optional HAL instruction page.")

ANY_FIELD_RULE = ValidationRule(
    entity_id="core.validation.field.any",
    since=Version("core", 0, 0),
    summary="Accepts every encoded field value.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text="The complete encoded range is valid.",
)
ANY_FIELD_USE = (RuleUse(ANY_FIELD_RULE.entity_id),)

BASE_ENTITIES = (
    entity("core.machine.state", 0, "Core machine state."),
    entity(
        "core.selector.status",
        0,
        "Status selector table.",
        "core.machine.state",
    ),
    entity(
        "core.operation.fail",
        0,
        "Terminates the invocation.",
        "core.machine.state",
        "core.selector.status",
    ),
)

MINOR_1_ENTITIES = (
    entity(
        "core.section.trace",
        1,
        "Optional trace declarations.",
        "core.machine.state",
    ),
    entity(
        "core.selector.trace_format",
        1,
        "Trace payload format selector.",
        "core.machine.state",
    ),
    entity(
        "core.validation.trace_record",
        1,
        "Trace record structural validation.",
        "core.section.trace",
        "core.selector.trace_format",
    ),
    entity(
        "core.operation.trace",
        1,
        "Emits one trace record.",
        "core.machine.state",
        "core.selector.trace_format",
        "core.validation.trace_record",
    ),
)


class SpecificationModelTest(unittest.TestCase):
    def test_core_opcode_cannot_alias_a_page_prefix(self) -> None:
        instruction = Instruction(
            entity_id="core.instruction.ambiguous",
            since=Version("core", 0, 0),
            summary="Ambiguous core instruction.",
            opcode=0xF1,
            mnemonic="ambiguous",
            byte_length=4,
            family_id="core.family.test",
            fields=(
                InstructionField(
                    name="padding_u8",
                    offset=1,
                    encoding_id="core.encoding.u8",
                    role=InstructionFieldRole.PADDING,
                    description="Padding.",
                    validation=(RuleUse(ANY_FIELD_RULE.entity_id),),
                    array_length=3,
                ),
            ),
            range_groups=(),
            constraints=(),
            control_flow=ControlFlow.RETURN,
            suspension=Suspension.NEVER,
            state_effects=(),
            semantics=InstructionSemantics(
                description="Returns.",
                verification=("Padding is valid.",),
                preconditions=(),
                success=("Returns.",),
                failures=(),
                ownership=(),
                assembly=("ambiguous",),
                pseudocode="return;",
            ),
        )
        family = InstructionFamily(
            entity_id="core.family.test",
            since=Version("core", 0, 0),
            summary="Test family.",
            document_order=0,
            normative_text="Test contract.",
        )
        with self.assertRaisesRegex(ValueError, "reserved page prefix"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0, HAL_0),
                (U8, ANY_FIELD_RULE, family, instruction),
            )

        page_family = dataclasses.replace(
            family,
            entity_id="hal.family.test",
            since=Version("hal", 0, 0),
            minimum_consumer_version=Version("hal", 0, 0),
        )
        page_instruction = dataclasses.replace(
            instruction,
            entity_id="hal.instruction.last",
            since=Version("hal", 0, 0),
            minimum_consumer_version=Version("hal", 0, 0),
            opcode=0xFF,
            mnemonic="hal.last",
            family_id=page_family.entity_id,
            fields=(
                dataclasses.replace(
                    instruction.fields[0],
                    offset=2,
                    array_length=2,
                ),
            ),
        )
        Specification(
            "iree.vm.bytecode",
            (CORE_0, HAL_0),
            (U8, ANY_FIELD_RULE, page_family, page_instruction),
        )

    def test_version_domains_use_architectural_page_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "page ID 0xef is unavailable"):
            VersionDomain("bad", 0xEF, 0, 0, "Invalid low extension page.")
        with self.assertRaisesRegex(ValueError, "page ID 0xfe is unavailable"):
            VersionDomain("bad", 0xFE, 0, 0, "Reserved experiment page.")
        with self.assertRaisesRegex(ValueError, "only the core domain"):
            VersionDomain("bad", 0x00, 0, 0, "Invalid core alias.")

    def test_historical_projection_is_unchanged_after_additive_minor(self) -> None:
        base = Specification("iree.vm.bytecode", (CORE_0,), BASE_ENTITIES)
        evolved = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (*BASE_ENTITIES, *MINOR_1_ENTITIES),
        )

        base_view = base.project((Version("core", 0, 0),))
        historical_view = evolved.project((Version("core", 0, 0),))
        self.assertEqual(base_view.entities, historical_view.entities)
        self.assertEqual(
            compare_projections(base_view, historical_view).added,
            (),
        )
        self.assertEqual(historical_view.domains, (CORE_1,))

    def test_release_diff_names_only_added_minor_features(self) -> None:
        evolved = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (*BASE_ENTITIES, *MINOR_1_ENTITIES),
        )
        release_diff = compare_projections(
            evolved.project((Version("core", 0, 0),)),
            evolved.project((Version("core", 0, 1),)),
        )

        self.assertEqual(
            release_diff.added,
            tuple(sorted(item.entity_id for item in MINOR_1_ENTITIES)),
        )
        self.assertTrue(release_diff.is_additive)
        release_diff.require_additive()

    def test_required_minor_is_derived_from_transitive_features(self) -> None:
        evolved = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (*BASE_ENTITIES, *MINOR_1_ENTITIES),
        )

        self.assertEqual(
            evolved.derive_requirements(("core.operation.trace",)),
            (Version("core", 0, 1),),
        )
        with self.assertRaisesRegex(KeyError, "unavailable in projection"):
            evolved.project((Version("core", 0, 0),)).require_entity(
                "core.operation.trace"
            )

    def test_old_runtime_rejects_new_minor_while_new_runtime_accepts_it(
        self,
    ) -> None:
        old_runtime = Specification("iree.vm.bytecode", (CORE_0,), BASE_ENTITIES)
        new_runtime = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (*BASE_ENTITIES, *MINOR_1_ENTITIES),
        )
        requirement = (Version("core", 0, 1),)

        with self.assertRaisesRegex(ValueError, "latest minor 0"):
            old_runtime.project(requirement)
        new_runtime.project(requirement).require_entity("core.operation.trace")

    def test_older_loom_target_view_refuses_newer_feature(self) -> None:
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (*BASE_ENTITIES, *MINOR_1_ENTITIES),
        )
        older_target = specification.project((Version("core", 0, 0),))
        current_target = specification.project((Version("core", 0, 1),))

        with self.assertRaisesRegex(KeyError, "unavailable in projection"):
            older_target.require_entity("core.operation.trace")
        current_target.require_entity("core.operation.trace")

    def test_cross_domain_dependency_requires_both_selected_domains(self) -> None:
        hal_operation = entity(
            "hal.operation.dispatch",
            0,
            "Dispatches through HAL.",
            "core.machine.state",
            domain="hal",
        )
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_0, HAL_0),
            (*BASE_ENTITIES, hal_operation),
        )

        self.assertEqual(
            specification.derive_requirements((hal_operation.entity_id,)),
            (Version("core", 0, 0), Version("hal", 0, 0)),
        )
        with self.assertRaisesRegex(ValueError, "projection omits dependencies"):
            specification.project((Version("hal", 0, 0),))
        specification.project(
            (Version("core", 0, 0), Version("hal", 0, 0))
        ).require_entity(hal_operation.entity_id)

    def test_same_domain_forward_dependency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "introduced before"):
            Specification(
                "iree.vm.bytecode",
                (CORE_1,),
                (
                    entity(
                        "core.operation.early",
                        0,
                        "Invalid early operation.",
                        "core.rule.later",
                    ),
                    entity("core.rule.later", 1, "Later rule."),
                ),
            )

    def test_dependency_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    entity(
                        "core.rule.first",
                        0,
                        "First cyclic rule.",
                        "core.rule.second",
                    ),
                    entity(
                        "core.rule.second",
                        0,
                        "Second cyclic rule.",
                        "core.rule.first",
                    ),
                ),
            )

    def test_release_diff_rejects_reinterpretation(self) -> None:
        before = Specification("iree.vm.bytecode", (CORE_0,), BASE_ENTITIES).project(
            (Version("core", 0, 0),)
        )
        changed_entities = tuple(
            dataclasses.replace(item, summary="Changed behavior.")
            if item.entity_id == "core.operation.fail"
            else item
            for item in BASE_ENTITIES
        )
        after = Specification("iree.vm.bytecode", (CORE_0,), changed_entities).project(
            (Version("core", 0, 0),)
        )

        release_diff = compare_projections(before, after)
        self.assertEqual(release_diff.changed, ("core.operation.fail",))
        with self.assertRaisesRegex(ValueError, "not additive"):
            release_diff.require_additive()

    def test_record_layout_revision_preserves_historical_reserved_bytes(
        self,
    ) -> None:
        u16 = ScalarEncoding(
            entity_id="core.encoding.u16",
            since=Version("core", 0, 0),
            summary="Little-endian unsigned 16-bit integer.",
            c_type="uint16_t",
            byte_length=2,
            alignment=2,
        )
        record = WireRecord(
            entity_id="core.record.example",
            since=Version("core", 0, 0),
            summary="Synthetic four-byte record.",
            c_type="iree_vm_bytecode_example_t",
        )
        layout_0 = WireRecordLayout(
            entity_id="core.record.example.layout_0",
            since=Version("core", 0, 0),
            summary="Initial layout with reserved tail bytes.",
            record_id=record.entity_id,
            byte_length=4,
            alignment=2,
            fields=(
                WireField(
                    "value_u16",
                    0,
                    u16.entity_id,
                    "Example value.",
                    ANY_FIELD_USE,
                ),
                WireField(
                    "reserved_u16",
                    2,
                    u16.entity_id,
                    "Reserved and zero in core 0.0.",
                    ANY_FIELD_USE,
                ),
            ),
        )
        layout_1 = WireRecordLayout(
            entity_id="core.record.example.layout_1",
            since=Version("core", 0, 1),
            summary="Minor-one layout assigning the tail bytes.",
            record_id=record.entity_id,
            previous_layout_id=layout_0.entity_id,
            byte_length=4,
            alignment=2,
            fields=(
                WireField(
                    "value_u16",
                    0,
                    u16.entity_id,
                    "Example value.",
                    ANY_FIELD_USE,
                ),
                WireField(
                    "flags_u16",
                    2,
                    u16.entity_id,
                    "Minor-one flag bits.",
                    ANY_FIELD_USE,
                ),
            ),
        )
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (u16, ANY_FIELD_RULE, record, layout_0, layout_1),
        )

        historical = selected_record_layouts(
            specification.project((Version("core", 0, 0),))
        )[record.entity_id]
        current = selected_record_layouts(
            specification.project((Version("core", 0, 1),))
        )[record.entity_id]
        self.assertEqual(historical.fields[1].name, "reserved_u16")
        self.assertEqual(current.fields[1].name, "flags_u16")

    def test_record_layout_revision_must_form_one_chain(self) -> None:
        u16 = ScalarEncoding(
            entity_id="core.encoding.u16",
            since=Version("core", 0, 0),
            summary="Little-endian unsigned 16-bit integer.",
            c_type="uint16_t",
            byte_length=2,
            alignment=2,
        )
        record = WireRecord(
            entity_id="core.record.example",
            since=Version("core", 0, 0),
            summary="Synthetic scalar record.",
            c_type="iree_vm_bytecode_example_t",
        )
        layout = WireRecordLayout(
            entity_id="core.record.example.layout_0",
            since=Version("core", 0, 0),
            summary="Invalid initial layout chain.",
            record_id=record.entity_id,
            previous_layout_id="core.record.example.missing",
            byte_length=2,
            alignment=2,
            fields=(
                WireField(
                    "value_u16",
                    0,
                    u16.entity_id,
                    "Value.",
                    ANY_FIELD_USE,
                ),
            ),
            scalar_alias=True,
        )

        with self.assertRaisesRegex(ValueError, "missing dependency"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (u16, ANY_FIELD_RULE, record, layout),
            )

    def test_numeric_table_values_extend_without_redeclaring_table(self) -> None:
        u8 = ScalarEncoding(
            entity_id="core.encoding.u8",
            since=Version("core", 0, 0),
            summary="Unsigned eight-bit integer.",
            c_type="uint8_t",
            byte_length=1,
            alignment=1,
        )
        table = NumericTable(
            entity_id="core.selector.example",
            since=Version("core", 0, 0),
            summary="Synthetic selector table.",
            encoding_id=u8.entity_id,
            table_kind=NumericTableKind.SELECTOR,
            unknown_value_policy=UnknownNumericValuePolicy.REJECT,
        )
        value_0 = NumericValue(
            entity_id="core.selector.example.none",
            since=Version("core", 0, 0),
            summary="No behavior.",
            table_id=table.entity_id,
            name="none",
            value=0,
        )
        value_1 = NumericValue(
            entity_id="core.selector.example.trace",
            since=Version("core", 0, 1),
            summary="Trace behavior.",
            table_id=table.entity_id,
            name="trace",
            value=1,
        )
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (u8, table, value_0, value_1),
        )

        historical_values = selected_numeric_values(
            specification.project((Version("core", 0, 0),))
        )[table.entity_id]
        current_values = selected_numeric_values(
            specification.project((Version("core", 0, 1),))
        )[table.entity_id]
        self.assertEqual(tuple(value.value for value in historical_values), (0,))
        self.assertEqual(
            tuple(value.value for value in current_values),
            (0, 1),
        )
        self.assertEqual(
            specification.derive_requirements((value_1.entity_id,)),
            (Version("core", 0, 1),),
        )

    def test_open_numeric_value_does_not_raise_consumer_requirement(self) -> None:
        u8 = ScalarEncoding(
            entity_id="core.encoding.u8",
            since=Version("core", 0, 0),
            summary="Unsigned eight-bit integer.",
            c_type="uint8_t",
            byte_length=1,
            alignment=1,
        )
        table = NumericTable(
            entity_id="core.numeric.metadata_type",
            since=Version("core", 0, 0),
            summary="Synthetic open metadata type table.",
            encoding_id=u8.entity_id,
            table_kind=NumericTableKind.ENUM,
            unknown_value_policy=UnknownNumericValuePolicy.PRESERVE_NONZERO,
        )
        invalid = NumericValue(
            entity_id="core.numeric.metadata_type.invalid",
            since=Version("core", 0, 0),
            summary="Invalid metadata type.",
            minimum_consumer_version=Version("core", 0, 0),
            table_id=table.entity_id,
            name="invalid",
            value=0,
        )
        typed_blob = NumericValue(
            entity_id="core.numeric.metadata_type.typed_blob",
            since=Version("core", 0, 1),
            summary="A newly interpreted but previously opaque metadata blob.",
            minimum_consumer_version=Version("core", 0, 0),
            table_id=table.entity_id,
            name="typed_blob",
            value=1,
        )
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_1,),
            (u8, table, invalid, typed_blob),
        )

        self.assertEqual(
            specification.derive_projection_versions((typed_blob.entity_id,)),
            (Version("core", 0, 1),),
        )
        self.assertEqual(
            specification.derive_requirements((typed_blob.entity_id,)),
            (Version("core", 0, 0),),
        )
        with self.assertRaisesRegex(ValueError, "open numeric value"):
            Specification(
                "iree.vm.bytecode",
                (CORE_1,),
                (
                    u8,
                    table,
                    invalid,
                    dataclasses.replace(
                        typed_blob,
                        minimum_consumer_version=Version("core", 0, 1),
                    ),
                ),
            )

    def test_instruction_requires_complete_encoding_and_semantics(self) -> None:
        u8 = ScalarEncoding(
            entity_id="core.encoding.u8",
            since=Version("core", 0, 0),
            summary="Unsigned eight-bit integer.",
            c_type="uint8_t",
            byte_length=1,
            alignment=1,
        )
        register_rule = ValidationRule(
            entity_id="core.validation.field.register_value",
            since=Version("core", 0, 0),
            summary="Requires a valid value register.",
            scope=ValidationScope.FIELD,
            parameters=(),
            normative_text="The encoded ordinal names a value register.",
        )
        zero_rule = ValidationRule(
            entity_id="core.validation.field.zero",
            since=Version("core", 0, 0),
            summary="Requires canonical zero bytes.",
            scope=ValidationScope.FIELD,
            parameters=(),
            normative_text="Every encoded bit is zero.",
        )
        family = InstructionFamily(
            entity_id="core.family.value",
            since=Version("core", 0, 0),
            summary="Value movement instructions.",
            document_order=0,
            normative_text="Value operations read inputs before results.",
        )
        instruction = Instruction(
            entity_id="core.instruction.value.copy",
            since=Version("core", 0, 0),
            summary="Copies one complete value cell.",
            opcode=0x20,
            mnemonic="value.copy",
            byte_length=4,
            family_id=family.entity_id,
            fields=(
                InstructionField(
                    "dst_v8",
                    1,
                    u8.entity_id,
                    InstructionFieldRole.RESULT,
                    "Destination value register.",
                    (RuleUse(register_rule.entity_id),),
                ),
                InstructionField(
                    "src_v8",
                    2,
                    u8.entity_id,
                    InstructionFieldRole.OPERAND,
                    "Source value register.",
                    (RuleUse(register_rule.entity_id),),
                ),
                InstructionField(
                    "zero_padding_u8",
                    3,
                    u8.entity_id,
                    InstructionFieldRole.PADDING,
                    "Canonical zero padding.",
                    (RuleUse(zero_rule.entity_id),),
                ),
            ),
            range_groups=(),
            constraints=(),
            control_flow=ControlFlow.SEQUENTIAL,
            suspension=Suspension.NEVER,
            state_effects=(),
            semantics=InstructionSemantics(
                description="Copies all 64 bits without interpretation.",
                verification=("Validate both register ordinals.",),
                preconditions=(),
                success=("The destination contains the source bits.",),
                failures=(),
                ownership=(),
                assembly=("%v0 = value.copy %v1",),
                pseudocode="values[dst_v8] = values[src_v8]; pc += 4;",
            ),
        )
        specification = Specification(
            "iree.vm.bytecode",
            (CORE_0,),
            (u8, register_rule, zero_rule, family, instruction),
        )

        self.assertEqual(
            specification.project((Version("core", 0, 0),))
            .require_entity(instruction.entity_id)
            .normative_anchor,
            "spec-core-instruction-value-copy",
        )

        terminal_failure = dataclasses.replace(
            instruction,
            entity_id="core.instruction.control.fail",
            opcode=0x21,
            mnemonic="control.fail",
            control_flow=ControlFlow.FAIL,
            semantics=dataclasses.replace(
                instruction.semantics,
                success=(),
                failures=(
                    FailureCase(
                        "invalid_argument",
                        "The instruction explicitly selects failure.",
                        "No result state is published.",
                    ),
                ),
            ),
        )
        Specification(
            "iree.vm.bytecode",
            (CORE_0,),
            (u8, register_rule, zero_rule, family, terminal_failure),
        )
        with self.assertRaisesRegex(ValueError, "has success effects"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        terminal_failure,
                        semantics=dataclasses.replace(
                            terminal_failure.semantics,
                            success=("Impossible continuation.",),
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "has no failure case"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        terminal_failure,
                        semantics=dataclasses.replace(
                            terminal_failure.semantics,
                            failures=(),
                        ),
                    ),
                ),
            )

        read_effect = StateEffect(
            StateAccess.READ,
            StateResource.FRAME_LOCALS,
            ("src_v8",),
        )
        Specification(
            "iree.vm.bytecode",
            (CORE_0,),
            (
                u8,
                register_rule,
                zero_rule,
                family,
                dataclasses.replace(instruction, state_effects=(read_effect,)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate state effect"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        instruction,
                        state_effects=(read_effect, read_effect),
                    ),
                ),
            )
        with self.assertRaisesRegex(
            ValueError,
            "unknown state effect must be the only effect",
        ):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        instruction,
                        state_effects=(
                            StateEffect(StateAccess.UNKNOWN, StateResource.ANY),
                            read_effect,
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(
            ValueError,
            "unknown state access and any resource must be paired",
        ):
            StateEffect(StateAccess.UNKNOWN, StateResource.FRAME_LOCALS)
        with self.assertRaisesRegex(
            ValueError,
            "unknown state access and any resource must be paired",
        ):
            StateEffect(StateAccess.READ, StateResource.ANY)
        with self.assertRaisesRegex(
            ValueError,
            "any-resource state effect cannot name resource fields",
        ):
            StateEffect(
                StateAccess.UNKNOWN,
                StateResource.ANY,
                ("src_v8",),
            )
        with self.assertRaisesRegex(ValueError, "unknown field missing_v8"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        instruction,
                        state_effects=(
                            StateEffect(
                                StateAccess.WRITE,
                                StateResource.FRAME_LOCALS,
                                ("missing_v8",),
                            ),
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "references padding field"):
            Specification(
                "iree.vm.bytecode",
                (CORE_0,),
                (
                    u8,
                    register_rule,
                    zero_rule,
                    family,
                    dataclasses.replace(
                        instruction,
                        state_effects=(
                            StateEffect(
                                StateAccess.READ,
                                StateResource.FRAME_LOCALS,
                                ("zero_padding_u8",),
                            ),
                        ),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
