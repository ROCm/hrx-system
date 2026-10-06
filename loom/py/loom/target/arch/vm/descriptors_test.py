# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from itertools import product

from iree.vm.bytecode.spec.isa import ControlFlow, FieldRole, Suspension
from iree.vm.bytecode.spec.isa.core.buffer import BUFFER_RODATA_LOAD
from iree.vm.bytecode.spec.isa.core.constant import CONSTANT_I32, CONSTANT_I64
from iree.vm.bytecode.spec.isa.core.float import (
    FloatBinarySemantics,
    FloatClampSemantics,
    FloatClassifySemantics,
    FloatCompareSemantics,
    FloatFmaSemantics,
    FloatMathSemantics,
    FloatMinmaxSemantics,
    FloatUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.integer import (
    IntegerBinarySemantics,
    IntegerCompareSemantics,
    IntegerDivisionSemantics,
    IntegerUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule, RecordRuleKind, StateAccess
from iree.vm.bytecode.spec.isa.core.stack import MEMORY_FORMAT_SELECTOR
from iree.vm.bytecode.spec.specification import SPECIFICATION

from loom.dialect.scalar import conversion
from loom.ir import ScalarType, ScalarTypeKind
from loom.target.arch.vm.contracts import (
    VM_CORE_CONTRACT_DIALECT_OPS,
    VM_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.vm.descriptors import VM_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    LOWER_EMIT_FLAG_RECORD_SOURCE_MEMORY,
    DescriptorRule,
    compile_lower_rule_set,
)
from loom.target.low_descriptors import (
    DescriptorFlag,
    DescriptorOpKind,
    EffectKind,
    ImmediateFlag,
    ImmediateKind,
    OperandRole,
)


def test_scalar_packets_preserve_spec_encoding_and_semantic_types():
    descriptors = {
        descriptor.encoding_id: descriptor
        for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
    }
    instructions = [
        instruction
        for instruction in SPECIFICATION.instructions
        if isinstance(
            instruction.semantics,
            (
                IntegerBinarySemantics,
                IntegerUnarySemantics,
                IntegerCompareSemantics,
                IntegerDivisionSemantics,
                FloatBinarySemantics,
                FloatUnarySemantics,
                FloatMinmaxSemantics,
                FloatCompareSemantics,
                FloatClassifySemantics,
                FloatClampSemantics,
                FloatFmaSemantics,
                FloatMathSemantics,
            ),
        )
    ]
    assert instructions
    for instruction in instructions:
        descriptor = descriptors[instruction.opcode]
        assert descriptor.mnemonic == instruction.mnemonic
        assert descriptor.encoding_format_id == instruction.byte_length
        assert instruction.control_flow is ControlFlow.SEQUENTIAL
        assert instruction.suspension is Suspension.NEVER
        assert not instruction.state_effects
        result_type = descriptor.asm_forms[0].result_value_types[0]
        scalar_types = (
            {32: ScalarTypeKind.F32, 64: ScalarTypeKind.F64}
            if isinstance(
                instruction.semantics,
                (
                    FloatBinarySemantics,
                    FloatUnarySemantics,
                    FloatMinmaxSemantics,
                    FloatClampSemantics,
                    FloatFmaSemantics,
                    FloatMathSemantics,
                ),
            )
            else {32: ScalarTypeKind.I32, 64: ScalarTypeKind.I64}
        )
        expected_type = (
            ScalarTypeKind.I1
            if isinstance(
                instruction.semantics,
                (
                    IntegerCompareSemantics,
                    FloatCompareSemantics,
                    FloatClassifySemantics,
                ),
            )
            else scalar_types[instruction.semantics.bit_width]
        )
        assert result_type.element_type is expected_type


def test_register_fields_preserve_the_wire_bank_and_position():
    instructions = {
        instruction.opcode: instruction for instruction in SPECIFICATION.instructions
    }
    banks = {FieldRule.REGISTER_VALUE: "vm.value", FieldRule.REGISTER_REF: "vm.ref"}
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        instruction = instructions[descriptor.encoding_id]
        fields = (
            (field, offset)
            for field, offset in zip(
                instruction.fields, instruction.field_offsets, strict=True
            )
            if field.role in (FieldRole.RESULT, FieldRole.OPERAND)
        )
        for (field, offset), operand in zip(fields, descriptor.operands, strict=True):
            assert operand.encoding_field_id == offset
            assert field.field.name == operand.field_name
            assert (field.role is FieldRole.RESULT) == (
                operand.role is OperandRole.RESULT
            )
            (alternative,) = operand.reg_alts
            assert alternative.reg_class == banks[field.rule.kind]


def test_effects_preserve_state_access_not_runtime_aborts():
    instructions = {
        instruction.opcode: instruction for instruction in SPECIFICATION.instructions
    }
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        instruction = instructions[descriptor.encoding_id]
        expected = tuple(
            dict.fromkeys(
                {
                    StateAccess.READ: EffectKind.READ,
                    StateAccess.WRITE: EffectKind.WRITE,
                    StateAccess.ALLOCATE: EffectKind.WRITE,
                }[effect.access]
                for effect in instruction.state_effects
            )
        )
        if expected:
            assert DescriptorFlag.SIDE_EFFECTING in descriptor.flags
            assert DescriptorFlag.DEAD_REMOVABLE not in descriptor.flags
            assert tuple(effect.kind for effect in descriptor.effects) == expected
        else:
            assert not descriptor.effects
            assert DescriptorFlag.DEAD_REMOVABLE in descriptor.flags


def test_lowering_uses_the_projected_descriptors():
    compiled = compile_lower_rule_set(
        VM_CORE_CONTRACT_FRAGMENT, dialect_ops=VM_CORE_CONTRACT_DIALECT_OPS
    )
    for emit in compiled.emits:
        records_access = bool(emit.flags & LOWER_EMIT_FLAG_RECORD_SOURCE_MEMORY)
        assert records_access == bool(
            emit.source_memory_ordinal
            and any(effect.is_memory_access for effect in emit.descriptor.effects)
        )
    descriptors = VM_CORE_DESCRIPTOR_SET.descriptors
    cases = (
        case
        for case in VM_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    emissions = tuple(emit for case in cases for emit in case.emit)
    assert {id(emit.descriptor) for emit in emissions} == {
        id(descriptor)
        for descriptor in descriptors
        # Symbol-kind selection uses the C recipe before final table ordinals
        # exist. Shared source and execution tests exercise that producer.
        if descriptor.encoding_id != BUFFER_RODATA_LOAD.opcode
    }
    for emit in emissions:
        assert set(emit.operands) == {
            operand.field_name
            for operand in emit.descriptor.operands
            if operand.role is OperandRole.OPERAND
        }
        assert set(emit.results) == {
            operand.field_name
            for operand in emit.descriptor.operands
            if operand.role is OperandRole.RESULT
        }


def test_immediate_positions_match_canonical_attributes():
    for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors:
        names = tuple(value.field_name for value in descriptor.immediates)
        assert names == tuple(sorted(names))
        assert all(
            ImmediateFlag.DEFAULT_VALUE not in value.flags
            for value in descriptor.immediates
        )


def test_selectors_preserve_the_spec_domain_and_encoding():
    descriptors = {
        descriptor.encoding_id: descriptor
        for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
    }
    domains = {domain.name: domain for domain in VM_CORE_DESCRIPTOR_SET.enum_domains}
    for instruction in SPECIFICATION.instructions:
        if instruction.opcode not in descriptors:
            continue
        descriptor = descriptors[instruction.opcode]
        if not descriptor.immediates or descriptor.op_kind is DescriptorOpKind.CONST:
            continue
        fields = {
            offset: field
            for field, offset in zip(
                instruction.fields, instruction.field_offsets, strict=True
            )
            if field.role is FieldRole.IMMEDIATE
        }
        for immediate in descriptor.immediates:
            field = fields[immediate.encoding_field_id]
            if field.rule.kind is FieldRule.PACKED_SELECTORS:
                # Joint domains are checked exhaustively below, including holes.
                continue
            assert immediate.field_name == field.field.name
            assert immediate.bit_width == field.field.byte_length * 8
            if field.rule.kind is FieldRule.SELECTOR:
                expected = {
                    entry.name: entry.value
                    for entry in field.rule.data.values
                    if field.rule.data is not MEMORY_FORMAT_SELECTOR
                    or entry.name.endswith(".x1")
                }
                assert {
                    entry.token: entry.value
                    for entry in domains[immediate.enum_domain].values
                } == expected
            elif field.rule.kind is FieldRule.ALLOWED_VALUES:
                assert (
                    tuple(
                        entry.value for entry in domains[immediate.enum_domain].values
                    )
                    == field.rule.values
                )
            elif field.rule.kind is FieldRule.ANY_BITS:
                assert immediate.unsigned_max == (1 << immediate.bit_width) - 1
            elif field.rule.kind is FieldRule.RODATA_ORDINAL:
                assert immediate.kind is ImmediateKind.ORDINAL
                assert immediate.flags == (ImmediateFlag.SYMBOLIC,)
                assert immediate.unsigned_max == (1 << immediate.bit_width) - 1
            else:
                assert field.rule.kind is FieldRule.ALLOWED_RANGE
                assert (
                    immediate.signed_min,
                    immediate.unsigned_max,
                ) == field.rule.values


def test_packed_selectors_encode_exactly_the_legal_wire_bytes():
    descriptors = {d.encoding_id: d for d in VM_CORE_DESCRIPTOR_SET.descriptors}
    domains = {d.name: d for d in VM_CORE_DESCRIPTOR_SET.enum_domains}
    for instruction in SPECIFICATION.instructions:
        if instruction.opcode not in descriptors:
            continue
        for field, offset in zip(
            instruction.fields, instruction.field_offsets, strict=True
        ):
            if field.rule.kind is not FieldRule.PACKED_SELECTORS:
                continue
            immediates = tuple(
                value
                for value in descriptors[instruction.opcode].immediates
                if value.encoding_field_id == offset
            )
            used_bits = 0
            for immediate in immediates:
                mask = ((1 << immediate.bit_width) - 1) << immediate.encoding_id
                assert not used_bits & mask
                assert mask < 256
                used_bits |= mask
            projected = {
                sum(
                    value.value << immediate.encoding_id
                    for value, immediate in zip(values, immediates, strict=True)
                )
                for values in product(
                    *(domains[i.enum_domain].values for i in immediates)
                )
            }
            expected = set()
            for bits in range(256):
                if bits & field.rule.values[0]:
                    continue
                decoded = {
                    part.name: (bits >> part.bit_offset) & ((1 << part.bit_length) - 1)
                    for part in field.rule.data
                }
                if any(
                    decoded[part.name]
                    not in (
                        part.allowed_values or tuple(v.value for v in part.table.values)
                    )
                    for part in field.rule.data
                ):
                    continue
                if any(
                    tuple(decoded[part.name] for part in rule.data)
                    not in tuple(zip(rule.values[::2], rule.values[1::2], strict=True))
                    for rule in instruction.rules
                    if rule.kind is RecordRuleKind.PACKED_SELECTOR_PAIRS
                    and field.field.name in rule.fields
                ):
                    continue
                expected.add(bits)
            assert projected == expected, instruction.mnemonic


def test_constant_immediates_preserve_the_wire_bits_and_alignment():
    descriptors = {
        descriptor.encoding_id: descriptor
        for descriptor in VM_CORE_DESCRIPTOR_SET.descriptors
    }
    for instruction in (CONSTANT_I32, CONSTANT_I64):
        descriptor = descriptors[instruction.opcode]
        assert descriptor.op_kind is DescriptorOpKind.CONST
        assert descriptor.encoding_format_id == instruction.byte_length
        (immediate,) = descriptor.immediates
        fields = tuple(
            (field.field, offset)
            for field, offset in zip(
                instruction.fields, instruction.field_offsets, strict=True
            )
            if field.role is FieldRole.IMMEDIATE
        )
        assert immediate.encoding_field_id == fields[0][1]
        assert immediate.bit_width == 8 * sum(field.byte_length for field, _ in fields)
        # One contiguous logical value must preserve each naturally aligned word,
        # including the sign bit and nonzero high half of a wide constant.
        for bits in (0, 0xFEDCBA98, 0x81234567FEDCBA98, 0xFFFFFFFFFFFFFFFF):
            value = bits & ((1 << immediate.bit_width) - 1)
            encoded = value.to_bytes(immediate.bit_width // 8, "little")
            for field, offset in fields:
                relative_offset = offset - immediate.encoding_field_id
                assert offset % field.encoding.alignment == 0
                assert (
                    int.from_bytes(
                        encoded[relative_offset : relative_offset + field.byte_length],
                        "little",
                    )
                    == (value >> (relative_offset * 8)) & 0xFFFFFFFF
                )


def test_numeric_conversions_cover_the_source_type_relations():
    operations = (
        conversion.scalar_extsi,
        conversion.scalar_extui,
        conversion.scalar_trunci,
        conversion.scalar_extf,
        conversion.scalar_fptrunc,
        conversion.scalar_sitofp,
        conversion.scalar_uitofp,
        conversion.scalar_fptosi,
        conversion.scalar_fptoui,
    )
    cases = {}
    for case in VM_CORE_CONTRACT_FRAGMENT.cases:
        if case.source_op not in operations:
            continue
        types = {guard.field: guard.type_pattern.element for guard in case.guards}
        key = (case.source_op, types["input"], types["result"])
        assert key not in cases
        cases[key] = case

    expected = set()
    types = [
        ScalarType(kind)
        for kind in ScalarTypeKind
        if kind not in (ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET)
    ]
    for source in types:
        for result in types:
            source_integer = str(source).startswith("i")
            result_integer = str(result).startswith("i")
            if source_integer != result_integer:
                ops = (
                    (conversion.scalar_sitofp, conversion.scalar_uitofp)
                    if source_integer
                    else (conversion.scalar_fptosi, conversion.scalar_fptoui)
                )
            elif source.bitwidth == result.bitwidth:
                continue
            elif source_integer:
                ops = (
                    (conversion.scalar_extsi, conversion.scalar_extui)
                    if source.bitwidth < result.bitwidth
                    else (conversion.scalar_trunci,)
                )
            else:
                ops = (
                    conversion.scalar_extf
                    if source.bitwidth < result.bitwidth
                    else conversion.scalar_fptrunc,
                )
            expected.update((op, str(source), str(result)) for op in ops)
    assert set(cases) == expected

    # Direct rounding must not be replaced with a staged conversion. Bfloat16
    # and f64 narrowing can distinguish even a one-bit intermediate error.
    for source in ("i32", "i64"):
        for op in (conversion.scalar_sitofp, conversion.scalar_uitofp):
            assert len(cases[(op, source, "bf16")].emit) == 1
    for result in ("f8E4M3", "f8E5M2", "f16", "bf16", "f32"):
        assert len(cases[(conversion.scalar_fptrunc, "f64", result)].emit) == 1
