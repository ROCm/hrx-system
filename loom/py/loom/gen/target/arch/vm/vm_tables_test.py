# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the Loom-facing Core VM table projection."""

from loom.dialect.cfg import defs as cfg_defs
from loom.dialect.func import defs as func_defs
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.gen.target.arch.vm.vm_tables import (
    generate_encoding_rows,
    generate_lowering_rows,
    generate_verification_rows,
)
from loom.ir import BUFFER_TYPE, I1, I32, I64
from loom.target.arch.vm.projection import (
    VM_ABI_INSTRUCTIONS,
    VM_CORE_DESCRIPTOR_SET,
    VM_CORE_INSTRUCTIONS,
    VM_INSTRUCTION_PROJECTIONS,
    VM_MODULE_RESOURCES,
    VM_PACKET_DESCRIPTORS,
    VM_SOURCE_LOWERINGS,
    VM_STRUCTURAL_INSTRUCTIONS,
)
from loom.target.arch.vm.verification import (
    VM_MEMORY_FORMAT_UNIT_COUNTS,
    VM_MODULE_DEPENDENT_CONSTRAINTS,
    VM_PACKED_IMMEDIATE_MASKS,
    VM_PACKET_CONSTRAINT_RANGES,
    VM_PACKET_CONSTRAINTS,
    VM_TARGET_DEPENDENT_CONSTRAINTS,
    VmPacketConstraintKind,
)
from loom.target.low_descriptors import (
    ConstraintKind,
    DescriptorCarrier,
    DescriptorFlag,
    Effect,
    EffectFlag,
    EffectKind,
    ImmediateKind,
    MemorySpace,
)


def test_descriptors_preserve_instruction_identity() -> None:
    assert len(VM_INSTRUCTION_PROJECTIONS) == 171
    assert len(VM_CORE_DESCRIPTOR_SET.descriptors) == len(VM_PACKET_DESCRIPTORS) + 2
    for descriptor, projection in zip(
        VM_PACKET_DESCRIPTORS,
        VM_INSTRUCTION_PROJECTIONS,
        strict=True,
    ):
        assert descriptor.encoding_id == projection.instruction.opcode
        assert descriptor.key == projection.key
        assert descriptor.mnemonic == projection.mnemonic
        assert descriptor.asm_forms

    yield_s32 = VM_CORE_DESCRIPTOR_SET.descriptors[-2]
    assert yield_s32.key == "vm.control.yield.s32"
    assert yield_s32.carrier is DescriptorCarrier.BRANCH
    assert yield_s32.encoding_id == 0x03
    assert yield_s32.operands == ()
    assert yield_s32.immediates == ()
    assert tuple(effect.kind for effect in yield_s32.effects) == (EffectKind.CONTROL,)
    assert yield_s32.flags == (
        DescriptorFlag.SIDE_EFFECTING,
        DescriptorFlag.TERMINATOR,
        DescriptorFlag.MAY_YIELD,
    )

    switch = VM_CORE_DESCRIPTOR_SET.descriptors[-1]
    assert switch.key == "vm.control.switch"
    assert switch.carrier is DescriptorCarrier.SWITCH
    assert switch.encoding_id == 0x0A
    assert tuple(operand.field_name for operand in switch.operands) == ("selector",)
    assert switch.immediates == ()
    assert tuple(effect.kind for effect in switch.effects) == (EffectKind.CONTROL,)


def test_projection_partitions_every_core_instruction() -> None:
    assert len(VM_CORE_INSTRUCTIONS) == 183
    assert len(VM_INSTRUCTION_PROJECTIONS) == 171
    assert len(VM_STRUCTURAL_INSTRUCTIONS) == 12
    assert len(VM_ABI_INSTRUCTIONS) == 7
    partitioned_ids = {
        *(projection.instruction.entity_id for projection in VM_INSTRUCTION_PROJECTIONS),
        *(instruction.entity_id for instruction in VM_STRUCTURAL_INSTRUCTIONS),
    }
    assert partitioned_ids == {instruction.entity_id for instruction in VM_CORE_INSTRUCTIONS}
    projected_ids = {projection.instruction.entity_id for projection in VM_INSTRUCTION_PROJECTIONS}
    assert all(instruction.entity_id in projected_ids for instruction in VM_ABI_INSTRUCTIONS)


def test_abi_records_are_ordinary_sequential_descriptors() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    value_load = descriptors["vm.value.abi.argument.load"]
    assert value_load.immediates[0].kind is ImmediateKind.ORDINAL
    assert value_load.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.STACK,
            flags=(EffectFlag.DEPENDENCY,),
        ),
    )
    assert value_load.flags == (DescriptorFlag.DEAD_REMOVABLE,)

    ref_store = descriptors["vm.ref.abi.result.store.move"]
    assert ref_store.effects == (
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.STACK,
            flags=(EffectFlag.DEPENDENCY,),
        ),
        Effect(EffectKind.BARRIER, flags=(EffectFlag.ORDERED,)),
    )
    assert ref_store.flags == (DescriptorFlag.SIDE_EFFECTING,)


def test_ref_move_requires_distinct_registers() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    ref_move = descriptors["vm.ref.move"]
    assert tuple((constraint.kind, constraint.lhs_operand_index, constraint.rhs_operand_index) for constraint in ref_move.constraints) == ((ConstraintKind.EARLY_CLOBBER, 0, None),)


def test_partial_arithmetic_remains_memory_pure() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    integer_division = descriptors["vm.integer.div.u64"]
    assert integer_division.effects == ()
    assert integer_division.flags == (DescriptorFlag.DEAD_REMOVABLE,)


def test_control_assert_reads_diagnostic_but_is_not_dead_removable() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    control_assert = descriptors["vm.control.assert"]
    assert control_assert.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
        ),
    )
    assert control_assert.flags == ()


def test_control_fail_is_a_terminal_packet() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    control_fail = descriptors["vm.control.fail"]
    assert control_fail.carrier is DescriptorCarrier.PACKET
    assert control_fail.effects == (
        Effect(EffectKind.CONTROL, flags=(EffectFlag.ORDERED,)),
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
        ),
    )
    assert control_fail.flags == (
        DescriptorFlag.SIDE_EFFECTING,
        DescriptorFlag.TERMINATOR,
        DescriptorFlag.NO_RETURN,
    )


def test_state_effects_preserve_memory_and_ownership_ordering() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    buffer_load = descriptors["vm.buffer.load"]
    assert buffer_load.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
        ),
    )
    assert buffer_load.flags == (DescriptorFlag.DEAD_REMOVABLE,)

    buffer_store = descriptors["vm.buffer.store"]
    assert buffer_store.effects == (
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
        ),
    )
    assert buffer_store.flags == (DescriptorFlag.SIDE_EFFECTING,)

    ref_move = descriptors["vm.ref.move"]
    assert ref_move.effects == (Effect(EffectKind.BARRIER, flags=(EffectFlag.ORDERED,)),)
    assert ref_move.flags == (DescriptorFlag.SIDE_EFFECTING,)


def test_variable_register_ranges_preserve_isa_limits() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    bitstream_pack = descriptors["vm.integer.bitstream.pack"]
    assert tuple(operand.unit_count for operand in bitstream_pack.operands) == (
        64,
        64,
    )
    buffer_load = descriptors["vm.buffer.load"]
    assert buffer_load.operands[0].unit_count == 8


def test_source_rows_reference_projected_descriptors() -> None:
    descriptor_keys = {descriptor.key for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    assert VM_SOURCE_LOWERINGS
    assert all(row.descriptor_key in descriptor_keys for row in VM_SOURCE_LOWERINGS)


def test_module_resources_reference_projected_descriptors() -> None:
    descriptor_keys = {descriptor.key for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors}
    assert len(VM_MODULE_RESOURCES) == 7
    assert len({resource.kind for resource in VM_MODULE_RESOURCES}) == len(VM_MODULE_RESOURCES)
    for resource in VM_MODULE_RESOURCES:
        assert resource.load_descriptor_key in descriptor_keys
        if resource.store_descriptor_key is not None:
            assert resource.store_descriptor_key in descriptor_keys
        if resource.store_preserve_descriptor_key is not None:
            assert resource.store_preserve_descriptor_key in descriptor_keys
    preserving_resources = (resource for resource in VM_MODULE_RESOURCES if resource.store_preserve_descriptor_key is not None)
    assert {resource.kind.value for resource in preserving_resources} == {
        "REF_IMMUTABLE",
        "REF_MUTABLE",
    }


def test_same_type_projection_derives_source_signatures() -> None:
    add_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is scalar_arithmetic.scalar_addi)
    assert tuple((row.operand_types, row.result_types, row.descriptor_key) for row in add_rows) == (
        (
            (I32, I32),
            (I32,),
            "vm.integer.add.i32",
        ),
        (
            (I64, I64),
            (I64,),
            "vm.integer.add.i64",
        ),
    )


def test_comparison_projection_copies_the_verified_predicate() -> None:
    compare_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is scalar_comparison.scalar_cmpi)
    assert tuple(
        (
            row.operand_types,
            row.result_types,
            row.descriptor_key,
            row.selector_immediate_ordinal,
            row.selector_source_attr_ordinal,
        )
        for row in compare_rows
    ) == (
        (
            (I32, I32),
            (I1,),
            "vm.integer.compare.i32",
            0,
            0,
        ),
        (
            (I64, I64),
            (I1,),
            "vm.integer.compare.i64",
            0,
            0,
        ),
    )


def test_lowering_rows_are_data_only() -> None:
    rows = generate_lowering_rows()
    assert "LOOM_VM_MODULE_RESOURCE_ROW(VALUE_IMMUTABLE" in rows
    assert "LOOM_VM_MODULE_RESOURCE_ROW(RODATA" in rows
    assert "VM_CORE_DESCRIPTOR_REF_BUFFER_RODATA_LOAD, UINT16_MAX, UINT16_MAX" in rows
    assert "VM_CORE_DESCRIPTOR_REF_REF_RETAIN" in rows
    assert "LOOM_VM_SOURCE_LOWERING_LIMITS(\n    2, 1)" in rows
    assert "LOOM_OP_INDEX_MUL" in rows
    assert "VM_CORE_DESCRIPTOR_REF_INTEGER_MUL_I64" in rows
    assert ("LOOM_OP_SCALAR_CMPI, VM_CORE_DESCRIPTOR_REF_INTEGER_COMPARE_I32, 0, 0, 0") in rows
    assert "LOOM_OP_SCALAR_ADDI, VM_CORE_DESCRIPTOR_REF_INTEGER_ADD_I32, 2, 1" not in rows
    assert "LOOM_OP_CFG_ASSERT, VM_CORE_DESCRIPTOR_REF_CONTROL_ASSERT" in rows
    assert "LOOM_VM_SOURCE_TYPE_KEY(LOOM_TYPE_BUFFER, LOOM_SCALAR_TYPE_NONE)" in rows
    assert "LOOM_OP_FUNC_FAIL, VM_CORE_DESCRIPTOR_REF_CONTROL_FAIL, 0, 0, 0" in rows


def test_program_failure_source_rows_preserve_public_types_and_status() -> None:
    assert_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is cfg_defs.cfg_assert)
    assert tuple((row.operand_types, row.result_types) for row in assert_rows) == (((I1, BUFFER_TYPE), ()),)
    fail_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is func_defs.func_fail)
    assert tuple(
        (
            row.operand_types,
            row.result_types,
            row.selector_immediate_ordinal,
            row.selector_source_attr_ordinal,
        )
        for row in fail_rows
    ) == (((BUFFER_TYPE,), (), 0, 0),)


def test_encoding_rows_are_data_only() -> None:
    rows = generate_encoding_rows()
    assert "LOOM_VM_INSTRUCTION_ENCODING_LIMITS(36)" in rows
    expected_rows = "\n".join(f"LOOM_VM_INSTRUCTION_ENCODING_ROW({projection.instruction.byte_length})" for projection in VM_INSTRUCTION_PROJECTIONS)
    assert rows.endswith(expected_rows + "\n")
    assert rows.count("LOOM_VM_INSTRUCTION_ENCODING_ROW(") == len(VM_INSTRUCTION_PROJECTIONS)


def test_packet_constraints_preserve_isa_relationships() -> None:
    assert len(VM_PACKET_CONSTRAINT_RANGES) == len(VM_INSTRUCTION_PROJECTIONS)
    assert VM_MEMORY_FORMAT_UNIT_COUNTS == (1, 2, 4, 8) * 4
    assert len(VM_PACKED_IMMEDIATE_MASKS) == 5
    assert {constraint.kind for constraint in VM_PACKET_CONSTRAINTS} == set(VmPacketConstraintKind)
    assert len(VM_TARGET_DEPENDENT_CONSTRAINTS) == 3
    assert len(VM_MODULE_DEPENDENT_CONSTRAINTS) == 8


def test_verification_rows_are_data_only() -> None:
    rows = generate_verification_rows()
    assert "LOOM_VM_PACKET_CONSTRAINT_LIMITS(\n    171, 26,\n    5, 16)" in rows
    assert rows.count("LOOM_VM_PACKET_CONSTRAINT_RANGE_ROW(") == 171
    assert rows.count("LOOM_VM_PACKET_CONSTRAINT_ROW(") == 26
