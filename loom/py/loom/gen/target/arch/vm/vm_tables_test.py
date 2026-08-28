# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the Loom-facing Core VM table projection."""

from loom.dialect.cfg import defs as cfg_defs
from loom.dialect.func import defs as func_defs
from loom.dialect.index import defs as index_defs
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.gen.target.arch.vm.vm_tables import (
    generate_encoding_rows,
    generate_lowering_rows,
    generate_verification_rows,
)
from loom.ir import BUFFER_TYPE, I1, I32, I64, INDEX, OFFSET
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
    signatures = tuple((row.source_op.name, row.operand_types, row.result_types) for row in VM_SOURCE_LOWERINGS)
    assert len(signatures) == len(set(signatures))


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


def test_native_integer_projection_preserves_operations_and_widths() -> None:
    expected_descriptors = (
        (scalar_arithmetic.scalar_addi, "vm.integer.add.i32", "vm.integer.add.i64"),
        (scalar_arithmetic.scalar_subi, "vm.integer.sub.i32", "vm.integer.sub.i64"),
        (scalar_arithmetic.scalar_muli, "vm.integer.mul.i32", "vm.integer.mul.i64"),
        (scalar_arithmetic.scalar_divsi, "vm.integer.div.s32", "vm.integer.div.s64"),
        (scalar_arithmetic.scalar_divui, "vm.integer.div.u32", "vm.integer.div.u64"),
        (scalar_arithmetic.scalar_remsi, "vm.integer.rem.s32", "vm.integer.rem.s64"),
        (scalar_arithmetic.scalar_remui, "vm.integer.rem.u32", "vm.integer.rem.u64"),
        (scalar_arithmetic.scalar_negi, "vm.integer.neg.i32", "vm.integer.neg.i64"),
        (scalar_arithmetic.scalar_absi, "vm.integer.abs.s32", "vm.integer.abs.s64"),
        (scalar_arithmetic.scalar_minsi, "vm.integer.min.s32", "vm.integer.min.s64"),
        (scalar_arithmetic.scalar_maxsi, "vm.integer.max.s32", "vm.integer.max.s64"),
        (scalar_arithmetic.scalar_minui, "vm.integer.min.u32", "vm.integer.min.u64"),
        (scalar_arithmetic.scalar_maxui, "vm.integer.max.u32", "vm.integer.max.u64"),
        (scalar_bitwise.scalar_andi, "vm.integer.and.i32", "vm.integer.and.i64"),
        (scalar_bitwise.scalar_ori, "vm.integer.or.i32", "vm.integer.or.i64"),
        (scalar_bitwise.scalar_xori, "vm.integer.xor.i32", "vm.integer.xor.i64"),
        (scalar_bitwise.scalar_shli, "vm.integer.shift.left.i32", "vm.integer.shift.left.i64"),
        (scalar_bitwise.scalar_shrsi, "vm.integer.shift.right.s32", "vm.integer.shift.right.s64"),
        (scalar_bitwise.scalar_shrui, "vm.integer.shift.right.u32", "vm.integer.shift.right.u64"),
        (scalar_bitwise.scalar_rotli, "vm.integer.rotate.left.i32", "vm.integer.rotate.left.i64"),
        (scalar_bitwise.scalar_rotri, "vm.integer.rotate.right.i32", "vm.integer.rotate.right.i64"),
        (scalar_bitwise.scalar_ctlzi, "vm.integer.count.leading.zeros.i32", "vm.integer.count.leading.zeros.i64"),
        (scalar_bitwise.scalar_cttzi, "vm.integer.count.trailing.zeros.i32", "vm.integer.count.trailing.zeros.i64"),
        (scalar_bitwise.scalar_ctpopi, "vm.integer.popcount.i32", "vm.integer.popcount.i64"),
    )
    for source_op, i32_descriptor, i64_descriptor in expected_descriptors:
        rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is source_op)
        expected_rows = (
            ((I32,) * len(source_op.operands), (I32,), i32_descriptor),
            ((I64,) * len(source_op.operands), (I64,), i64_descriptor),
        )
        assert tuple((row.operand_types, row.result_types, row.descriptor_key) for row in rows if row.operand_types != (I1, I1)) == expected_rows

    for source_op, descriptor in (
        (scalar_bitwise.scalar_andi, "vm.integer.and.i32"),
        (scalar_bitwise.scalar_ori, "vm.integer.or.i32"),
        (scalar_bitwise.scalar_xori, "vm.integer.xor.i32"),
    ):
        assert any(row.source_op is source_op and row.operand_types == (I1, I1) and row.result_types == (I1,) and row.descriptor_key == descriptor for row in VM_SOURCE_LOWERINGS)


def test_address_domain_projection_preserves_index_semantics() -> None:
    expected_rows = (
        (index_defs.index_add, (INDEX, INDEX), INDEX, "vm.integer.add.i64"),
        (index_defs.index_add, (OFFSET, OFFSET), OFFSET, "vm.integer.add.i64"),
        (index_defs.index_sub, (INDEX, INDEX), INDEX, "vm.integer.sub.i64"),
        (index_defs.index_sub, (OFFSET, OFFSET), OFFSET, "vm.integer.sub.i64"),
        (index_defs.index_mul, (INDEX, INDEX), INDEX, "vm.integer.mul.i64"),
        (index_defs.index_scale, (INDEX, OFFSET), OFFSET, "vm.integer.mul.i64"),
        (index_defs.index_div, (INDEX, INDEX), INDEX, "vm.integer.div.u64"),
        (index_defs.index_rem, (INDEX, INDEX), INDEX, "vm.integer.rem.u64"),
        (index_defs.index_min, (INDEX, INDEX), INDEX, "vm.integer.min.s64"),
        (index_defs.index_max, (INDEX, INDEX), INDEX, "vm.integer.max.s64"),
        (index_defs.index_andi, (INDEX, INDEX), INDEX, "vm.integer.and.i64"),
        (index_defs.index_ori, (INDEX, INDEX), INDEX, "vm.integer.or.i64"),
        (index_defs.index_xori, (INDEX, INDEX), INDEX, "vm.integer.xor.i64"),
        (index_defs.index_shli, (INDEX, INDEX), INDEX, "vm.integer.shift.left.i64"),
        (index_defs.index_shrsi, (INDEX, INDEX), INDEX, "vm.integer.shift.right.s64"),
        (index_defs.index_shrui, (INDEX, INDEX), INDEX, "vm.integer.shift.right.u64"),
        (index_defs.index_rotli, (INDEX, INDEX), INDEX, "vm.integer.rotate.left.i64"),
        (index_defs.index_rotri, (INDEX, INDEX), INDEX, "vm.integer.rotate.right.i64"),
        (index_defs.index_ctlzi, (INDEX,), INDEX, "vm.integer.count.leading.zeros.i64"),
        (index_defs.index_cttzi, (INDEX,), INDEX, "vm.integer.count.trailing.zeros.i64"),
        (index_defs.index_ctpopi, (INDEX,), INDEX, "vm.integer.popcount.i64"),
    )
    for source_op, operands, result, descriptor in expected_rows:
        assert any(row.source_op is source_op and row.operand_types == operands and row.result_types == (result,) and row.descriptor_key == descriptor for row in VM_SOURCE_LOWERINGS)

    compare_rows = tuple(row for row in VM_SOURCE_LOWERINGS if row.source_op is index_defs.index_cmp)
    assert tuple(
        (
            row.operand_types,
            row.result_types,
            row.descriptor_key,
            row.selector_source_attr_ordinal,
        )
        for row in compare_rows
    ) == (
        ((INDEX, INDEX), (I1,), "vm.integer.compare.i64", 0),
        ((OFFSET, OFFSET), (I1,), "vm.integer.compare.i64", 0),
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
    assert "kVmSourceLoweringIndexRanges[LOOM_OP_INDEX_COUNT_]" in rows
    index_compare_start = next(index for index, row in enumerate(VM_SOURCE_LOWERINGS) if row.source_op is index_defs.index_cmp)
    assert f"[LOOM_OP_INDEX_CMP & 0xFF] = {{{index_compare_start}, 2}}" in rows
    assert "kVmSourceLoweringScalarRanges[LOOM_OP_SCALAR_COUNT_]" in rows
    scalar_and_start = next(index for index, row in enumerate(VM_SOURCE_LOWERINGS) if row.source_op is scalar_bitwise.scalar_andi)
    assert f"[LOOM_OP_SCALAR_ANDI & 0xFF] = {{{scalar_and_start}, 3}}" in rows
    assert "kVmSourceLoweringDialectRanges[LOOM_DIALECT_BUILTIN_COUNT_]" in rows


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
