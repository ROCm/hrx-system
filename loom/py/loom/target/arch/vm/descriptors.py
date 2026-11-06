# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Projects VM instruction records into Loom's shared Low descriptor schema.

The runtime spec owns opcodes, packet fields, and semantics. This projection
supplies the interpreter's register and scheduling model, not a second ISA.
Encoding IDs are byte opcodes, encoding format IDs are packet byte lengths,
and operand encoding field IDs are byte offsets from the instruction start.
Immediate encoding IDs are sub-byte shifts within their byte-offset fields.
"""

from pathlib import Path

from iree.vm.bytecode.spec.isa import FieldRole, Instruction
from iree.vm.bytecode.spec.isa.core.buffer import (
    BUFFER_ALLOCATE,
    BUFFER_ATOMIC_CMPXCHG,
    BUFFER_ATOMIC_REDUCE,
    BUFFER_ATOMIC_RMW,
    BUFFER_COMPARE,
    BUFFER_COPY,
    BUFFER_FILL,
    BUFFER_LENGTH,
    BUFFER_LOAD,
    BUFFER_RODATA_LOAD,
    BUFFER_STORE,
)
from iree.vm.bytecode.spec.isa.core.constant import CONSTANT_I32, CONSTANT_I64
from iree.vm.bytecode.spec.isa.core.conversion import (
    CONVERSION_INSTRUCTIONS,
    FLOAT_EXTEND_SELECTOR,
    FLOAT_TO_INTEGER_SELECTOR,
    FLOAT_TRUNCATE_SELECTOR,
    FLOAT_WIDTH_SELECTOR,
    INTEGER_CONVERT_SELECTOR,
    INTEGER_TO_FLOAT_SELECTOR,
)
from iree.vm.bytecode.spec.isa.core.float import (
    FloatBinarySemantics,
    FloatClampSemantics,
    FloatClassifySemantics,
    FloatCompareSemantics,
    FloatDecomposeSemantics,
    FloatFmaSemantics,
    FloatMathSemantics,
    FloatMinmaxSemantics,
    FloatScaleSemantics,
    FloatUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.integer import (
    IntegerBinarySemantics,
    IntegerCompareSemantics,
    IntegerDivisionSemantics,
    IntegerUnarySemantics,
)
from iree.vm.bytecode.spec.isa.core.ref import (
    REF_COMPARE_EQ,
    REF_COMPARE_NULL,
    REF_NULL,
    REF_SELECT,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RecordRuleKind,
    RefOwnership,
    StateAccess,
)
from iree.vm.bytecode.spec.isa.core.stack import MEMORY_FORMAT_SELECTOR
from iree.vm.bytecode.spec.isa.core.value import VALUE_COPY, VALUE_SELECT
from iree.vm.bytecode.spec.specification import SPECIFICATION

from loom.ir import ScalarTypeKind
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmResultValueType,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectKind,
    EnumDomain,
    EnumValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandRole,
    OwnershipEffect,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)

_OPERAND_ROLES = {
    FieldRole.RESULT: OperandRole.RESULT,
    FieldRole.OPERAND: OperandRole.OPERAND,
}
_REGISTER_CLASSES = {
    FieldRule.REGISTER_VALUE: "vm.value",
    FieldRule.REGISTER_REF: "vm.ref",
}
_ATOMIC_INSTRUCTIONS = (
    BUFFER_ATOMIC_REDUCE,
    BUFFER_ATOMIC_RMW,
    BUFFER_ATOMIC_CMPXCHG,
)
_BUFFER_INSTRUCTIONS = (
    BUFFER_ALLOCATE,
    BUFFER_LENGTH,
    BUFFER_FILL,
    BUFFER_COPY,
    BUFFER_COMPARE,
    BUFFER_LOAD,
    BUFFER_STORE,
    BUFFER_RODATA_LOAD,
    *_ATOMIC_INSTRUCTIONS,
)

_INTEGER_TYPES = {32: ScalarTypeKind.I32, 64: ScalarTypeKind.I64}
_FLOAT_TYPES = {32: ScalarTypeKind.F32, 64: ScalarTypeKind.F64}
_PREDICATE_TYPES = {32: ScalarTypeKind.I1, 64: ScalarTypeKind.I1}
_RESULT_TYPES = {
    IntegerBinarySemantics: _INTEGER_TYPES,
    IntegerUnarySemantics: _INTEGER_TYPES,
    IntegerCompareSemantics: _PREDICATE_TYPES,
    IntegerDivisionSemantics: _INTEGER_TYPES,
    FloatBinarySemantics: _FLOAT_TYPES,
    FloatUnarySemantics: _FLOAT_TYPES,
    FloatMinmaxSemantics: _FLOAT_TYPES,
    FloatCompareSemantics: _PREDICATE_TYPES,
    FloatClassifySemantics: _PREDICATE_TYPES,
    FloatClampSemantics: _FLOAT_TYPES,
    FloatFmaSemantics: _FLOAT_TYPES,
    FloatDecomposeSemantics: _FLOAT_TYPES,
    FloatScaleSemantics: _FLOAT_TYPES,
    FloatMathSemantics: _FLOAT_TYPES,
}


def scalar_result_types(instruction: Instruction) -> tuple[ScalarTypeKind, ...]:
    """Returns the result types fixed by scalar numeric semantics, in wire order."""
    types = _RESULT_TYPES.get(type(instruction.semantics))
    if types is None:
        return ()
    result_type = types[instruction.semantics.bit_width]
    if isinstance(instruction.semantics, FloatDecomposeSemantics):
        return (result_type, ScalarTypeKind.I32)
    return (result_type,)


_SCALAR_INSTRUCTIONS = tuple(
    instruction
    for instruction in SPECIFICATION.instructions
    if type(instruction.semantics) in _RESULT_TYPES
)
_SCALAR_CONVERSIONS = tuple(
    instruction
    for instruction in CONVERSION_INSTRUCTIONS
    if instruction.fields[-1].rule.data
    in (
        INTEGER_CONVERT_SELECTOR,
        FLOAT_EXTEND_SELECTOR,
        FLOAT_TRUNCATE_SELECTOR,
        FLOAT_WIDTH_SELECTOR,
        INTEGER_TO_FLOAT_SELECTOR,
        FLOAT_TO_INTEGER_SELECTOR,
    )
)
_SELECTORS = {
    field.rule.data.name: field.rule.data
    for instruction in (
        *_SCALAR_INSTRUCTIONS,
        *_SCALAR_CONVERSIONS,
        *_BUFFER_INSTRUCTIONS,
    )
    for field in instruction.fields
    if field.rule.kind is FieldRule.SELECTOR
}


def _packed_immediates(instruction: Instruction):
    for field, offset in zip(
        instruction.fields, instruction.field_offsets, strict=True
    ):
        if field.rule.kind is not FieldRule.PACKED_SELECTORS:
            continue
        paired = ()
        for rule in instruction.rules:
            if (
                rule.kind is RecordRuleKind.PACKED_SELECTOR_PAIRS
                and field.field.name in rule.fields
            ):
                assert not paired
                paired = rule.data
                left, right = paired
                assert right.bit_offset == left.bit_offset + left.bit_length
                assert right.bit_length == left.bit_length
                names = [
                    {value.value: value.name for value in part.table.values}
                    for part in paired
                ]
                domain = EnumDomain(
                    f"{instruction.mnemonic}.orderings",
                    tuple(
                        EnumValue(
                            f"{names[0][a]}.{names[1][b]}",
                            a | (b << left.bit_length),
                        )
                        for a, b in zip(
                            rule.values[::2], rule.values[1::2], strict=True
                        )
                    ),
                )
                yield (
                    Immediate(
                        "orderings",
                        ImmediateKind.ENUM,
                        bit_width=left.bit_length + right.bit_length,
                        encoding_field_id=offset,
                        encoding_id=left.bit_offset,
                        enum_domain=domain.name,
                    ),
                    domain,
                )
        for part in field.rule.data:
            if part in paired:
                continue
            domain = EnumDomain(
                f"{instruction.mnemonic}.{part.name}"
                if part.allowed_values
                else part.table.name,
                tuple(
                    EnumValue(value.name, value.value)
                    for value in part.table.values
                    if not part.allowed_values or value.value in part.allowed_values
                ),
            )
            yield (
                Immediate(
                    part.name,
                    ImmediateKind.ENUM,
                    bit_width=part.bit_length,
                    encoding_field_id=offset,
                    encoding_id=part.bit_offset,
                    enum_domain=domain.name,
                ),
                domain,
            )


_PACKED_IMMEDIATES = {
    instruction.opcode: tuple(_packed_immediates(instruction))
    for instruction in _BUFFER_INSTRUCTIONS
}


def _immediates(instruction: Instruction) -> tuple[Immediate, ...]:
    immediates = [row for row, _ in _PACKED_IMMEDIATES.get(instruction.opcode, ())]
    for field, offset in zip(
        instruction.fields, instruction.field_offsets, strict=True
    ):
        if (
            field.role is not FieldRole.IMMEDIATE
            or field.rule.kind is FieldRule.PACKED_SELECTORS
        ):
            continue
        bit_width = field.field.byte_length * 8
        minimum, maximum = 0, 0
        domain = None
        kind = ImmediateKind.UNSIGNED
        flags = ()
        if field.rule.kind is FieldRule.SELECTOR:
            kind = ImmediateKind.ENUM
            domain = field.rule.data.name
        elif field.rule.kind is FieldRule.ALLOWED_VALUES:
            kind = ImmediateKind.ENUM
            domain = f"{instruction.mnemonic}.{field.field.name}"
        elif field.rule.kind is FieldRule.ANY_BITS:
            maximum = (1 << bit_width) - 1
        elif field.rule.kind is FieldRule.RODATA_ORDINAL:
            kind = ImmediateKind.ORDINAL
            maximum = (1 << bit_width) - 1
            flags = (ImmediateFlag.SYMBOLIC,)
        else:
            assert field.rule.kind is FieldRule.ALLOWED_RANGE
            minimum, maximum = field.rule.values
        immediates.append(
            Immediate(
                field.field.name,
                kind,
                flags=flags,
                bit_width=bit_width,
                encoding_field_id=offset,
                enum_domain=domain,
                signed_min=minimum,
                unsigned_max=maximum,
            )
        )
    return tuple(immediates)


def _descriptor(
    instruction: Instruction,
    result_types: tuple[ScalarTypeKind, ...],
    *,
    op_kind: DescriptorOpKind = DescriptorOpKind.OP,
    immediates: tuple[Immediate, ...] | None = None,
) -> Descriptor:
    if immediates is None:
        immediates = _immediates(instruction)
    # Required immediates sort with canonical IR dictionaries. Assembly retains
    # wire order, while the emitter consumes attributes directly by position.
    assert instruction.byte_length <= CONSTANT_I64.byte_length
    assert all(ImmediateFlag.DEFAULT_VALUE not in value.flags for value in immediates)
    operands = tuple(
        Operand(
            field.field.name,
            _OPERAND_ROLES[field.role],
            (RegClassAlt(_REGISTER_CLASSES[field.rule.kind]),),
            encoding_field_id=offset,
            ownership=(
                {
                    RefOwnership.BORROW: OwnershipEffect.BORROW,
                    RefOwnership.INSPECT: OwnershipEffect.BORROW,
                    RefOwnership.RETAIN: OwnershipEffect.RETAIN,
                    RefOwnership.REPLACE_OWNER: OwnershipEffect.OWNED,
                    RefOwnership.REPLACE_RETAIN: OwnershipEffect.OWNED,
                    RefOwnership.REPLACE_BORROW: OwnershipEffect.BORROWED,
                    # A null result has no payload owner. Its authored type
                    # determines the source lifetime obligation.
                    RefOwnership.CLEAR: OwnershipEffect.NONE,
                }[field.ref_policy.ownership]
                if field.ref_policy
                else OwnershipEffect.NONE
            ),
        )
        for field, offset in zip(
            instruction.fields, instruction.field_offsets, strict=True
        )
        if field.role in _OPERAND_ROLES
    )
    # Invocation aborts are runtime behavior, not compiler-visible effects.
    # Owner acquisition must also precede later source consumption. A removal
    # flag alone does not establish scheduling dependencies with releases.
    creates_owner = any(
        operand.ownership is OwnershipEffect.OWNED for operand in operands
    )
    defines_null = any(
        field.role is FieldRole.RESULT
        and field.ref_policy
        and field.ref_policy.ownership is RefOwnership.CLEAR
        for field in instruction.fields
    )
    effects = tuple(
        dict.fromkeys(
            Effect(
                {
                    StateAccess.READ: EffectKind.READ,
                    StateAccess.WRITE: EffectKind.WRITE,
                    StateAccess.ALLOCATE: EffectKind.WRITE,
                }[effect.access],
                MemorySpace.GENERIC,
            )
            for effect in instruction.state_effects
        )
    ) + ((Effect(EffectKind.UNKNOWN, MemorySpace.GENERIC),) if creates_owner else ())
    # Null construction defines a source lifetime, not a memory access.
    if defines_null:
        effects += (Effect(EffectKind.WRITE, MemorySpace.NONE),)
    return Descriptor(
        key=f"vm.{instruction.mnemonic}",
        mnemonic=instruction.mnemonic,
        semantic_tag=None,
        operands=operands,
        schedule_class="vm.scalar",
        op_kind=op_kind,
        immediates=tuple(sorted(immediates, key=lambda value: value.field_name)),
        encoding_id=instruction.opcode,
        encoding_format_id=instruction.byte_length,
        effects=effects,
        flags=(
            DescriptorFlag.SIDE_EFFECTING if effects else DescriptorFlag.DEAD_REMOVABLE,
        )
        # Null owners have independent source lifetimes even without a payload.
        # Preserve them until mandatory ownership admission checks consumption.
        + ((DescriptorFlag.UNIQUE_IDENTITY,) if defines_null else ()),
        instruction_classes=(
            InstructionClass.GENERIC_MEMORY
            if instruction.state_effects
            else InstructionClass.SCALAR_ALU,
        )
        + ((InstructionClass.ATOMIC,) if instruction in _ATOMIC_INSTRUCTIONS else ()),
        asm_forms=(
            AsmForm(
                results=tuple(
                    operand.field_name
                    for operand in operands
                    if operand.role is OperandRole.RESULT
                ),
                operands=tuple(
                    operand.field_name
                    for operand in operands
                    if operand.role is OperandRole.OPERAND
                ),
                immediates=tuple(
                    AsmImmediate(value.field_name) for value in immediates
                ),
                result_value_types=tuple(
                    AsmResultValueType(result_type) for result_type in result_types
                ),
            ),
        ),
    )


def _constant_descriptor(instruction: Instruction) -> Descriptor:
    # Loom carries the exact scalar bits as one immediate. The wire stores wide
    # constants as consecutive u32 words to preserve instruction alignment.
    fields = tuple(
        (field.field, offset)
        for field, offset in zip(
            instruction.fields, instruction.field_offsets, strict=True
        )
        if field.role is FieldRole.IMMEDIATE
    )
    offset = fields[0][1]
    byte_length = sum(field.byte_length for field, _ in fields)
    assert tuple(position for _, position in fields) == tuple(
        range(offset, offset + byte_length, 4)
    )
    bit_width = byte_length * 8
    return _descriptor(
        instruction,
        ({32: ScalarTypeKind.I32, 64: ScalarTypeKind.I64}[bit_width],),
        op_kind=DescriptorOpKind.CONST,
        immediates=(
            Immediate(
                "bits",
                ImmediateKind.UNSIGNED if bit_width == 32 else ImmediateKind.SIGNED,
                bit_width=bit_width,
                encoding_field_id=offset,
                signed_min=-(1 << 63) if bit_width == 64 else 0,
                unsigned_max=(1 << bit_width) - 1,
            ),
        ),
    )


VM_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="vm.core",
    inline_schedule_model_key="vm.core",
    target_key="vm",
    feature_key=None,
    c_header_path=Path("loom/src/loom/target/arch/vm/descriptors/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/vm/descriptors/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_VM_DESCRIPTORS_DESCRIPTORS_H_",
    public_header="loom/target/arch/vm/descriptors/descriptors.h",
    function_name="loom_vm_core_descriptor_set",
    c_table_prefix="VmCore",
    c_enum_prefix="VM_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            "vm.value",
            alloc_unit_bits=64,
            spill_slot_space=SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
        ),
        RegClass(
            "vm.ref",
            alloc_unit_bits=128,
            spill_slot_space=SpillSlotSpace.PRIVATE,
            flags=(
                RegClassFlag.PHYSICAL,
                RegClassFlag.REFERENCE,
            ),
            allocatable_count=256,
        ),
    ),
    resources=(Resource("vm.issue", 1, ResourceKind.SCALAR_ALU),),
    schedule_classes=(
        ScheduleClass(
            "vm.scalar",
            latency_kind=LatencyKind.ESTIMATE,
            model_quality=ModelQuality.ESTIMATED,
            latency_cycles=1,
            issue_uses=(IssueUse("vm.issue", cycles=1, units=1),),
        ),
    ),
    requires_explicit_asm_surface=True,
    enum_domains=tuple(
        EnumDomain(
            selector.name,
            tuple(
                EnumValue(value.name, value.value)
                for value in selector.values
                # Scalar memory operands occupy one value cell. Wider groups
                # require matching register-group descriptors.
                if selector is not MEMORY_FORMAT_SELECTOR or value.name.endswith(".x1")
            ),
        )
        for selector in _SELECTORS.values()
    )
    + tuple(
        EnumDomain(
            f"{instruction.mnemonic}.{field.field.name}",
            tuple(EnumValue(str(value), value) for value in field.rule.values),
        )
        for instruction in _BUFFER_INSTRUCTIONS
        for field in instruction.fields
        if field.rule.kind is FieldRule.ALLOWED_VALUES
    )
    + tuple(
        {
            domain.name: domain
            for fields in _PACKED_IMMEDIATES.values()
            for _, domain in fields
        }.values()
    ),
    descriptors=(
        _descriptor(VALUE_COPY, (ScalarTypeKind.I64,)),
        _descriptor(VALUE_SELECT, (ScalarTypeKind.I64,)),
        _descriptor(REF_SELECT, ()),
        _descriptor(REF_NULL, ()),
        _descriptor(REF_COMPARE_NULL, (ScalarTypeKind.I1,)),
        _descriptor(REF_COMPARE_EQ, (ScalarTypeKind.I1,)),
        *(_constant_descriptor(op) for op in (CONSTANT_I32, CONSTANT_I64)),
        *(_descriptor(op, (ScalarTypeKind.I64,)) for op in _SCALAR_CONVERSIONS),
        *(
            _descriptor(instruction, scalar_result_types(instruction))
            for instruction in _SCALAR_INSTRUCTIONS
        ),
        *(
            _descriptor(
                op,
                (ScalarTypeKind.I64,)
                if op in (BUFFER_LENGTH, BUFFER_LOAD)
                else (ScalarTypeKind.I32,)
                if op is BUFFER_COMPARE
                else (),
            )
            for op in _BUFFER_INSTRUCTIONS
        ),
    ),
)
