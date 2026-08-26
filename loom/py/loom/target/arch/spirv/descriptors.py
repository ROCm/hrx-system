# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the SPIR-V logical target."""

from __future__ import annotations

from pathlib import Path

from loom.ir import parse_scalar_type_kind
from loom.target.arch.spirv.builtins import (
    BUILTIN_DIMENSIONS,
    BUILTIN_INDEX_QUERIES,
)
from loom.target.arch.spirv.cooperative_matrix import (
    COOPERATIVE_MATRIX_CASES,
    CooperativeMatrixCase,
    cooperative_matrix_descriptor_key,
)
from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorInstruction,
    OrdinaryVectorInstructionType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    OFFSET64_ALU_TYPE,
    OFFSET64_COMPARE_PREDICATES,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    BooleanConstant,
    IntegerAluTypePair,
    IntegerComparePredicate,
    ScalarAluType,
    ScalarBinaryOperation,
)
from loom.target.arch.spirv.scalar_constant import (
    BFLOAT16_CONSTANT_TYPE,
    FLOAT_CONSTANT_TYPES,
    FloatConstantType,
)
from loom.target.arch.spirv.scalar_conversion import (
    INTEGER_VALUE_VIEW_CONVERSIONS,
    LOW_SCALAR_CONVERSIONS,
    IntegerValueViewConversion,
    ScalarConversion,
)
from loom.target.arch.spirv.scalar_memory import (
    RAW_STORAGE_BUFFER_BYTE,
    STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
)
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmResultValueType,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

_REG_ID = "spirv.id"
_REG_OFFSET64 = "spirv.offset64"
_REG_PTR_FUNCTION = "spirv.ptr.function"
_REG_PTR_STORAGE_BUFFER = "spirv.ptr.storage_buffer"

_RESOURCE_ALU = "spirv.alu"
_RESOURCE_LOAD = "spirv.load"
_RESOURCE_MATRIX = "spirv.matrix"
_RESOURCE_STORE = "spirv.store"
_RESOURCE_VARIABLE = "spirv.variable"

_SCHEDULE_ALU = "spirv.alu"
_SCHEDULE_LOAD = "spirv.load"
_SCHEDULE_MATRIX = "spirv.matrix"
_SCHEDULE_STORE = "spirv.store"
_SCHEDULE_VARIABLE = "spirv.variable"

_ID_ALT = (RegClassAlt(_REG_ID),)
_OFFSET64_ALT = (RegClassAlt(_REG_OFFSET64),)
_PTR_FUNCTION_ALT = (RegClassAlt(_REG_PTR_FUNCTION),)
_PTR_STORAGE_BUFFER_ALT = (RegClassAlt(_REG_PTR_STORAGE_BUFFER),)

_OFFSET64_VALUE_IMMEDIATE = Immediate(
    "offset64_value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)


def _integer_constant_immediate(scalar_pair: IntegerAluTypePair) -> Immediate:
    return Immediate(
        f"{scalar_pair.source_type}_value",
        ImmediateKind.SIGNED,
        bit_width=scalar_pair.bit_width,
        signed_min=scalar_pair.signed_minimum,
        unsigned_max=scalar_pair.signed_maximum,
    )


def _float_constant_immediate(scalar: FloatConstantType) -> Immediate:
    return Immediate(
        f"{scalar.source_type}_bits",
        ImmediateKind.UNSIGNED,
        bit_width=scalar.bit_width,
        unsigned_max=(2**scalar.bit_width) - 1,
    )


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    result_value_types: tuple[AsmResultValueType | None, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(field_name) for field_name in immediates),
            result_value_types=result_value_types,
        ),
    )


def _scalar_result_value_type(source_type: str) -> AsmResultValueType:
    element_type = parse_scalar_type_kind(source_type)
    if element_type is None:
        raise ValueError(f"unknown Loom scalar result type '{source_type}'")
    return AsmResultValueType(element_type)


def _scalar_alu_result_value_type(
    scalar: ScalarAluType,
) -> AsmResultValueType | None:
    if scalar in UNSIGNED_INTEGER_SCALAR_ALU_TYPES:
        return None
    return _scalar_result_value_type(scalar.source_type)


def _single_result_value_types(
    result_value_type: AsmResultValueType | None,
) -> tuple[AsmResultValueType, ...]:
    return () if result_value_type is None else (result_value_type,)


def _id_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _ID_ALT)


def _id_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _ID_ALT)


def _offset64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _OFFSET64_ALT)


def _offset64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _OFFSET64_ALT)


def _ptr_function_result(field_name: str = "ptr") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PTR_FUNCTION_ALT)


def _ptr_workgroup_reg_class_name(scalar: StorageBufferScalar) -> str:
    return f"spirv.ptr.workgroup.{scalar.suffix}"


def _ptr_workgroup_array_reg_class_name(scalar: StorageBufferScalar) -> str:
    return f"spirv.ptr.workgroup.array.{scalar.suffix}"


def _ptr_workgroup_alt(scalar: StorageBufferScalar) -> tuple[RegClassAlt, ...]:
    return (RegClassAlt(_ptr_workgroup_reg_class_name(scalar)),)


def _ptr_workgroup_array_alt(
    scalar: StorageBufferScalar,
) -> tuple[RegClassAlt, ...]:
    return (RegClassAlt(_ptr_workgroup_array_reg_class_name(scalar)),)


def _ptr_workgroup_scalar_result(
    scalar: StorageBufferScalar,
    field_name: str = "ptr",
) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _ptr_workgroup_alt(scalar))


def _ptr_workgroup_scalar_operand(
    scalar: StorageBufferScalar,
    field_name: str,
) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _ptr_workgroup_alt(scalar))


def _ptr_workgroup_array_operand(
    scalar: StorageBufferScalar,
    field_name: str,
) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _ptr_workgroup_array_alt(scalar))


def _ptr_storage_buffer_result(field_name: str = "ptr") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PTR_STORAGE_BUFFER_ALT)


def _ptr_storage_buffer_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _PTR_STORAGE_BUFFER_ALT)


def _binary_same_type_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operands: tuple[Operand, Operand, Operand],
    result_value_type: AsmResultValueType | None,
    feature_bits: int = 0,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=operands,
        feature_mask_words=(feature_bits,) if feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("lhs", "rhs"),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _compare_descriptor(
    predicate: IntegerComparePredicate,
    *,
    scalar: ScalarAluType,
    operands: tuple[Operand, Operand, Operand],
) -> Descriptor:
    suffix = scalar.suffix
    key = f"spirv.op_{predicate.descriptor_suffix}.{suffix}"
    mnemonic = (
        predicate.mnemonic if suffix == "i32" else f"{predicate.mnemonic}.{suffix}"
    )
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=key,
        operands=operands,
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("lhs", "rhs"),
            result_value_types=(_scalar_result_value_type("i1"),),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _select_descriptor(
    *,
    key: str,
    mnemonic: str,
    operands: tuple[Operand, Operand, Operand, Operand],
    result_value_type: AsmResultValueType | None,
    feature_bits: int = 0,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=key,
        operands=operands,
        feature_mask_words=(feature_bits,) if feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("condition", "true_value", "false_value"),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _conversion_descriptor(row: ScalarConversion) -> Descriptor:
    result_value_type = _scalar_alu_result_value_type(row.result_type)
    return Descriptor(
        key=row.key,
        mnemonic=row.display_mnemonic,
        semantic_tag=row.key,
        operands=(_id_result(), _id_operand("input")),
        feature_mask_words=(row.feature_bits,) if row.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("input",),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _integer_value_view_descriptor(row: IntegerValueViewConversion) -> Descriptor:
    result_value_type = _scalar_alu_result_value_type(row.result_type)
    return Descriptor(
        key=row.key,
        mnemonic=row.display_mnemonic,
        semantic_tag=row.key,
        operands=(_id_result(), _id_operand("input")),
        feature_mask_words=(row.feature_bits,) if row.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("input",),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _builtin_index_descriptor(
    query_suffix: str,
    mnemonic_suffix: str,
    dimension: str,
) -> Descriptor:
    key = f"spirv.op_load_builtin.{query_suffix}.{dimension}"
    return Descriptor(
        key=key,
        mnemonic=f"OpLoadBuiltin.{mnemonic_suffix}.{dimension}",
        semantic_tag=key,
        operands=(_id_result(),),
        asm_forms=_asm(
            results=("dst",),
            result_value_types=(_scalar_result_value_type("index"),),
        ),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _builtin_index_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _builtin_index_descriptor(
            query.descriptor_suffix,
            query.mnemonic_suffix,
            dimension.source_keyword,
        )
        for query in BUILTIN_INDEX_QUERIES
        for dimension in BUILTIN_DIMENSIONS
    )


def _coordinate_copy_descriptor() -> Descriptor:
    key = "spirv.op_copy_object.i32"
    return _unary_typed_descriptor(
        key=key,
        mnemonic="OpCopyObject.i32",
        semantic_tag=key,
        operands=(_id_result(), _id_operand("input")),
        result_value_type=_scalar_result_value_type("i32"),
    )


def _conversion_descriptors() -> tuple[Descriptor, ...]:
    return (
        *(tuple(_conversion_descriptor(row) for row in LOW_SCALAR_CONVERSIONS)),
        *(
            tuple(
                _integer_value_view_descriptor(row)
                for row in INTEGER_VALUE_VIEW_CONVERSIONS
            )
        ),
    )


def _ternary_same_type_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operands: tuple[Operand, Operand, Operand, Operand],
    result_value_type: AsmResultValueType,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=operands,
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "c"),
            result_value_types=(result_value_type,),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _unary_typed_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operands: tuple[Operand, Operand],
    result_value_type: AsmResultValueType | None,
    feature_bits: int = 0,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=operands,
        feature_mask_words=(feature_bits,) if feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("input",),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _ordinary_vector_result(
    value_type: OrdinaryVectorInstructionType,
) -> Operand:
    if (
        isinstance(value_type, OrdinaryVectorComponentType)
        and value_type.kind == OrdinaryVectorComponentKind.OFFSET
    ):
        return _offset64_result()
    return _id_result()


def _ordinary_vector_operand(
    field_name: str,
    value_type: OrdinaryVectorInstructionType,
) -> Operand:
    if (
        isinstance(value_type, OrdinaryVectorComponentType)
        and value_type.kind == OrdinaryVectorComponentKind.OFFSET
    ):
        return _offset64_operand(field_name)
    return _id_operand(field_name)


def _ordinary_vector_result_value_type(
    value_type: OrdinaryVectorInstructionType,
) -> AsmResultValueType | None:
    component_type = (
        value_type.component_type
        if isinstance(value_type, OrdinaryVectorType)
        else value_type
    )
    if not component_type.source_types:
        return None
    if (
        isinstance(value_type, OrdinaryVectorComponentType)
        and component_type.kind == OrdinaryVectorComponentKind.OFFSET
    ):
        return None
    source_type = component_type.source_types[0]
    element_type = parse_scalar_type_kind(source_type)
    if element_type is None:
        raise ValueError(
            f"ordinary-vector component '{component_type.suffix}' has unknown "
            f"Loom source type '{source_type}'"
        )
    lane_count = (
        value_type.lane_count if isinstance(value_type, OrdinaryVectorType) else 0
    )
    return AsmResultValueType(element_type, vector_lane_count=lane_count)


def _ordinary_vector_descriptor(row: OrdinaryVectorInstruction) -> Descriptor:
    has_component_index = row.component_index_maximum is not None
    result_value_type = _ordinary_vector_result_value_type(row.result_type)
    immediates = (
        (
            Immediate(
                "component_index",
                ImmediateKind.UNSIGNED,
                bit_width=32,
                unsigned_max=row.component_index_maximum,
            ),
        )
        if has_component_index
        else ()
    )
    return Descriptor(
        key=row.key,
        mnemonic=row.mnemonic,
        semantic_tag=row.key,
        operands=(
            _ordinary_vector_result(row.result_type),
            *(
                _ordinary_vector_operand(name, operand_type)
                for name, operand_type in zip(
                    row.operand_names, row.operand_types, strict=True
                )
            ),
        ),
        immediates=immediates,
        feature_mask_words=(row.feature_bits,) if row.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=row.operand_names,
            immediates=("component_index",) if has_component_index else (),
            result_value_types=(
                (result_value_type,) if result_value_type is not None else ()
            ),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _address_conversion_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        scalar = (
            scalar_pair.signed if scalar_pair.bit_width == 64 else scalar_pair.unsigned
        )
        suffix = scalar.suffix
        feature_bits = scalar_pair.signed.feature_bits
        if scalar_pair.bit_width == 64:
            to_offset_opcode = "bitcast"
            to_offset_mnemonic = "OpBitcast"
        else:
            to_offset_opcode = "uconvert"
            to_offset_mnemonic = "OpUConvert"
        descriptors.append(
            _unary_typed_descriptor(
                key=f"spirv.op_{to_offset_opcode}.{suffix}.offset64",
                mnemonic=f"{to_offset_mnemonic}.{suffix}.offset64",
                semantic_tag=f"spirv.op_{to_offset_opcode}.{suffix}.offset64",
                operands=(_offset64_result(), _id_operand("input")),
                result_value_type=None,
                feature_bits=feature_bits,
            )
        )

        if scalar_pair.bit_width == 64:
            from_offset_opcode = "bitcast"
            from_offset_mnemonic = "OpBitcast"
        else:
            from_offset_opcode = "uconvert"
            from_offset_mnemonic = "OpUConvert"
        descriptors.append(
            _unary_typed_descriptor(
                key=f"spirv.op_{from_offset_opcode}.offset64.{suffix}",
                mnemonic=f"{from_offset_mnemonic}.offset64.{suffix}",
                semantic_tag=f"spirv.op_{from_offset_opcode}.offset64.{suffix}",
                operands=(_id_result(), _offset64_operand("input")),
                result_value_type=_scalar_alu_result_value_type(scalar),
                feature_bits=feature_bits,
            )
        )
    return tuple(descriptors)


def _storage_buffer_effect(
    kind: EffectKind,
    scalar: StorageBufferScalar,
) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.GLOBAL,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=scalar.byte_width * 8,
    )


def _workgroup_effect(
    kind: EffectKind,
    scalar: StorageBufferScalar,
) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.WORKGROUP,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=scalar.byte_width * 8,
    )


def _cooperative_matrix_effect(
    kind: EffectKind,
    *,
    byte_width: int,
    rows: int,
    columns: int,
) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.GLOBAL,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=byte_width * rows * columns * 8,
    )


def _ptr_access_chain_storage_buffer_descriptor(
    scalar: StorageBufferScalar,
) -> Descriptor:
    return Descriptor(
        key=f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset",
        mnemonic=f"OpPtrAccessChain.storage_buffer.{scalar.suffix}.byte_offset",
        semantic_tag=f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset",
        operands=(
            _ptr_storage_buffer_result(),
            _ptr_storage_buffer_operand("base"),
            _offset64_operand("byte_offset"),
        ),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(results=("ptr",), operands=("base", "byte_offset")),
        schedule_class=_SCHEDULE_VARIABLE,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        instruction_classes=(InstructionClass.OTHER,),
    )


def _load_storage_buffer_descriptor(scalar: StorageBufferScalar) -> Descriptor:
    result_value_type = (
        _scalar_result_value_type(scalar.source_type)
        if scalar.source_rule_enabled
        else None
    )
    return Descriptor(
        key=f"spirv.op_load.storage_buffer.{scalar.suffix}",
        mnemonic=f"OpLoad.storage_buffer.{scalar.suffix}",
        semantic_tag=f"spirv.op_load.storage_buffer.{scalar.suffix}",
        operands=(_id_result(), _ptr_storage_buffer_operand("ptr")),
        effects=(_storage_buffer_effect(EffectKind.READ, scalar),),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("ptr",),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _store_storage_buffer_descriptor(scalar: StorageBufferScalar) -> Descriptor:
    return Descriptor(
        key=f"spirv.op_store.storage_buffer.{scalar.suffix}",
        mnemonic=f"OpStore.storage_buffer.{scalar.suffix}",
        semantic_tag=f"spirv.op_store.storage_buffer.{scalar.suffix}",
        operands=(_ptr_storage_buffer_operand("ptr"), _id_operand("value")),
        effects=(_storage_buffer_effect(EffectKind.WRITE, scalar),),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(operands=("ptr", "value")),
        schedule_class=_SCHEDULE_STORE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _storage_buffer_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        descriptors.append(_ptr_access_chain_storage_buffer_descriptor(scalar))
        descriptors.append(_load_storage_buffer_descriptor(scalar))
        descriptors.append(_store_storage_buffer_descriptor(scalar))
    return tuple(descriptors)


def _raw_storage_buffer_byte_descriptors() -> tuple[Descriptor, ...]:
    scalar = RAW_STORAGE_BUFFER_BYTE
    convert_from_byte_key = f"spirv.op_uconvert.{scalar.suffix}.u32"
    convert_to_byte_key = f"spirv.op_uconvert.u32.{scalar.suffix}"
    return (
        _ptr_access_chain_storage_buffer_descriptor(scalar),
        _load_storage_buffer_descriptor(scalar),
        _store_storage_buffer_descriptor(scalar),
        _unary_typed_descriptor(
            key=convert_from_byte_key,
            mnemonic=f"OpUConvert.{scalar.suffix}.u32",
            semantic_tag=convert_from_byte_key,
            operands=(_id_result(), _id_operand("input")),
            result_value_type=None,
            feature_bits=scalar.feature_bits,
        ),
        _unary_typed_descriptor(
            key=convert_to_byte_key,
            mnemonic=f"OpUConvert.u32.{scalar.suffix}",
            semantic_tag=convert_to_byte_key,
            operands=(_id_result(), _id_operand("input")),
            result_value_type=None,
            feature_bits=scalar.feature_bits,
        ),
    )


def _access_chain_workgroup_descriptor(
    scalar: StorageBufferScalar,
) -> Descriptor:
    return Descriptor(
        key=f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index",
        mnemonic=f"OpAccessChain.workgroup.{scalar.suffix}.element_index",
        semantic_tag=f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index",
        operands=(
            _ptr_workgroup_scalar_result(scalar),
            _ptr_workgroup_array_operand(scalar, "base"),
            _id_operand("element_index"),
        ),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(results=("ptr",), operands=("base", "element_index")),
        schedule_class=_SCHEDULE_VARIABLE,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        instruction_classes=(InstructionClass.OTHER,),
    )


def _load_workgroup_descriptor(scalar: StorageBufferScalar) -> Descriptor:
    result_value_type = (
        _scalar_result_value_type(scalar.source_type)
        if scalar.source_rule_enabled
        else None
    )
    return Descriptor(
        key=f"spirv.op_load.workgroup.{scalar.suffix}",
        mnemonic=f"OpLoad.workgroup.{scalar.suffix}",
        semantic_tag=f"spirv.op_load.workgroup.{scalar.suffix}",
        operands=(_id_result(), _ptr_workgroup_scalar_operand(scalar, "ptr")),
        effects=(_workgroup_effect(EffectKind.READ, scalar),),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            operands=("ptr",),
            result_value_types=_single_result_value_types(result_value_type),
        ),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _store_workgroup_descriptor(scalar: StorageBufferScalar) -> Descriptor:
    return Descriptor(
        key=f"spirv.op_store.workgroup.{scalar.suffix}",
        mnemonic=f"OpStore.workgroup.{scalar.suffix}",
        semantic_tag=f"spirv.op_store.workgroup.{scalar.suffix}",
        operands=(
            _ptr_workgroup_scalar_operand(scalar, "ptr"),
            _id_operand("value"),
        ),
        effects=(_workgroup_effect(EffectKind.WRITE, scalar),),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(operands=("ptr", "value")),
        schedule_class=_SCHEDULE_STORE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _workgroup_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        descriptors.append(_access_chain_workgroup_descriptor(scalar))
        descriptors.append(_load_workgroup_descriptor(scalar))
        descriptors.append(_store_workgroup_descriptor(scalar))
    return tuple(descriptors)


def _control_barrier_descriptor(execution_scope: str) -> Descriptor:
    key = f"spirv.op_control_barrier.{execution_scope}.workgroup.acq_rel"
    return Descriptor(
        key=key,
        mnemonic=f"OpControlBarrier.{execution_scope}.workgroup.acq_rel",
        semantic_tag=key,
        operands=(),
        effects=(
            Effect(
                EffectKind.BARRIER,
                memory_space=MemorySpace.WORKGROUP,
                flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
            ),
            Effect(
                EffectKind.CONVERGENT,
                flags=(EffectFlag.ORDERED,),
            ),
        ),
        asm_forms=_asm(),
        schedule_class=_SCHEDULE_VARIABLE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _control_barrier_descriptors() -> tuple[Descriptor, ...]:
    return (
        _control_barrier_descriptor("subgroup"),
        _control_barrier_descriptor("workgroup"),
    )


def _cooperative_matrix_load_descriptor(
    *,
    role: str,
    case: CooperativeMatrixCase,
    scalar: StorageBufferScalar,
    rows: int,
    columns: int,
    feature_bits: int,
) -> Descriptor:
    key = cooperative_matrix_descriptor_key(
        "op_cooperative_matrix_load_khr",
        role=role,
        element=case.element,
        m_size=case.m_size,
        n_size=case.n_size,
        k_size=case.k_size,
        accumulator=case.accumulator,
        scope="subgroup",
        layout="row_major",
    )
    return Descriptor(
        key=key,
        mnemonic=(
            f"OpCooperativeMatrixLoadKHR.{role}.{case.element}."
            f"{case.m_size}x{case.n_size}x{case.k_size}."
            f"{case.accumulator}.subgroup.row_major"
        ),
        semantic_tag=key,
        operands=(_id_result(), _ptr_storage_buffer_operand("ptr")),
        effects=(
            _cooperative_matrix_effect(
                EffectKind.READ,
                byte_width=scalar.byte_width,
                rows=rows,
                columns=columns,
            ),
        ),
        feature_mask_words=(feature_bits,),
        asm_forms=_asm(results=("dst",), operands=("ptr",)),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _cooperative_matrix_store_descriptor(
    *,
    case: CooperativeMatrixCase,
    scalar: StorageBufferScalar,
    rows: int,
    columns: int,
    feature_bits: int,
) -> Descriptor:
    key = cooperative_matrix_descriptor_key(
        "op_cooperative_matrix_store_khr",
        role="result",
        element=case.element,
        m_size=case.m_size,
        n_size=case.n_size,
        k_size=case.k_size,
        accumulator=case.accumulator,
        scope="subgroup",
        layout="row_major",
    )
    return Descriptor(
        key=key,
        mnemonic=(
            f"OpCooperativeMatrixStoreKHR.result.{case.element}."
            f"{case.m_size}x{case.n_size}x{case.k_size}."
            f"{case.accumulator}.subgroup.row_major"
        ),
        semantic_tag=key,
        operands=(_ptr_storage_buffer_operand("ptr"), _id_operand("value")),
        effects=(
            _cooperative_matrix_effect(
                EffectKind.WRITE,
                byte_width=scalar.byte_width,
                rows=rows,
                columns=columns,
            ),
        ),
        feature_mask_words=(feature_bits,),
        asm_forms=_asm(operands=("ptr", "value")),
        schedule_class=_SCHEDULE_STORE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _cooperative_matrix_mul_add_descriptor(
    *,
    case: CooperativeMatrixCase,
) -> Descriptor:
    key = cooperative_matrix_descriptor_key(
        "op_cooperative_matrix_mul_add_khr",
        element=case.element,
        m_size=case.m_size,
        n_size=case.n_size,
        k_size=case.k_size,
        accumulator=case.accumulator,
        scope="subgroup",
        operand_mode=case.operand_mode,
    )
    operand_mode_part = f".{case.operand_mode}" if case.operand_mode else ""
    return Descriptor(
        key=key,
        mnemonic=(
            f"OpCooperativeMatrixMulAddKHR.{case.element}."
            f"{case.m_size}x{case.n_size}x{case.k_size}."
            f"{case.accumulator}.subgroup"
            f"{operand_mode_part}"
        ),
        semantic_tag=key,
        operands=(
            _id_result(),
            _id_operand("lhs"),
            _id_operand("rhs"),
            _id_operand("acc"),
        ),
        feature_mask_words=(case.feature_bits,),
        asm_forms=_asm(results=("dst",), operands=("lhs", "rhs", "acc")),
        schedule_class=_SCHEDULE_MATRIX,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _cooperative_matrix_descriptors_for_case(
    case: CooperativeMatrixCase,
) -> tuple[Descriptor, ...]:
    return (
        _cooperative_matrix_load_descriptor(
            role="lhs",
            case=case,
            scalar=case.lhs_scalar,
            rows=case.lhs_rows,
            columns=case.lhs_columns,
            feature_bits=case.memory_feature_bits(case.lhs_scalar),
        ),
        _cooperative_matrix_load_descriptor(
            role="rhs",
            case=case,
            scalar=case.rhs_scalar,
            rows=case.rhs_rows,
            columns=case.rhs_columns,
            feature_bits=case.memory_feature_bits(case.rhs_scalar),
        ),
        _cooperative_matrix_load_descriptor(
            role="init",
            case=case,
            scalar=case.accumulator_scalar,
            rows=case.accumulator_rows,
            columns=case.accumulator_columns,
            feature_bits=case.memory_feature_bits(case.accumulator_scalar),
        ),
        _cooperative_matrix_mul_add_descriptor(case=case),
        _cooperative_matrix_store_descriptor(
            case=case,
            scalar=case.result_scalar,
            rows=case.accumulator_rows,
            columns=case.accumulator_columns,
            feature_bits=case.memory_feature_bits(case.result_scalar),
        ),
    )


def _cooperative_matrix_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for case in COOPERATIVE_MATRIX_CASES:
        descriptors.extend(_cooperative_matrix_descriptors_for_case(case))
    return tuple(descriptors)


def _scalar_binary_descriptor(
    scalar: ScalarAluType,
    operation: ScalarBinaryOperation,
) -> Descriptor:
    key = f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}"
    mnemonic = (
        operation.mnemonic
        if scalar.suffix == "i32"
        else f"{operation.mnemonic}.{scalar.suffix}"
    )
    return _binary_same_type_descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=key,
        operands=(_id_result(), _id_operand("lhs"), _id_operand("rhs")),
        result_value_type=_scalar_alu_result_value_type(scalar),
        feature_bits=scalar.feature_bits,
    )


def _integer_constant_descriptor(scalar_pair: IntegerAluTypePair) -> Descriptor:
    scalar = scalar_pair.signed
    key = f"spirv.op_constant.{scalar.suffix}"
    immediate = _integer_constant_immediate(scalar_pair)
    return Descriptor(
        key=key,
        mnemonic=f"OpConstant.{scalar.suffix}",
        semantic_tag=key,
        operands=(_id_result(),),
        op_kind=DescriptorOpKind.CONST,
        immediates=(immediate,),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            immediates=(immediate.field_name,),
            result_value_types=(_scalar_result_value_type(scalar_pair.source_type),),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _float_constant_descriptor(scalar: FloatConstantType) -> Descriptor:
    key = f"spirv.op_constant.{scalar.suffix}"
    immediate = _float_constant_immediate(scalar)
    return Descriptor(
        key=key,
        mnemonic=f"OpConstant.{scalar.suffix}",
        semantic_tag=key,
        operands=(_id_result(),),
        op_kind=DescriptorOpKind.CONST,
        immediates=(immediate,),
        feature_mask_words=(scalar.feature_bits,) if scalar.feature_bits else (),
        asm_forms=_asm(
            results=("dst",),
            immediates=(immediate.field_name,),
            result_value_types=(_scalar_result_value_type(scalar.source_type),),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _boolean_constant_descriptor(row: BooleanConstant) -> Descriptor:
    key = f"spirv.op_constant_{row.descriptor_suffix}.bool"
    return Descriptor(
        key=key,
        mnemonic=f"{row.mnemonic}.bool",
        semantic_tag=key,
        operands=(_id_result(),),
        op_kind=DescriptorOpKind.CONST,
        asm_forms=_asm(
            results=("dst",),
            result_value_types=(_scalar_result_value_type("i1"),),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _boolean_binary_descriptor(operation: ScalarBinaryOperation) -> Descriptor:
    key = f"spirv.op_{operation.descriptor_suffix}.bool"
    return _binary_same_type_descriptor(
        key=key,
        mnemonic=f"{operation.mnemonic}.bool",
        semantic_tag=key,
        operands=(_id_result(), _id_operand("lhs"), _id_operand("rhs")),
        result_value_type=_scalar_result_value_type("i1"),
    )


def _scalar_binary_descriptors() -> tuple[Descriptor, ...]:
    descriptors = [
        _scalar_binary_descriptor(scalar, operation)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for operation in SIGNED_INTEGER_BINARY_OPERATIONS
    ]
    descriptors.extend(
        _scalar_binary_descriptor(scalar, operation)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for operation in INTEGER_BITWISE_BINARY_OPERATIONS
    )
    descriptors.extend(
        _scalar_binary_descriptor(scalar_pair.unsigned, operation)
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS
    )
    descriptors.extend(
        _scalar_binary_descriptor(scalar, operation)
        for scalar in FLOAT_SCALAR_ALU_TYPES
        for operation in FLOAT_BINARY_OPERATIONS
    )
    descriptors.extend(
        _boolean_binary_descriptor(operation) for operation in BOOLEAN_BINARY_OPERATIONS
    )
    return tuple(descriptors)


def _compare_descriptors() -> tuple[Descriptor, ...]:
    descriptors = [
        _compare_descriptor(
            predicate,
            scalar=scalar,
            operands=(
                _id_result(),
                _id_operand("lhs"),
                _id_operand("rhs"),
            ),
        )
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
    ]
    descriptors.extend(
        [
            _compare_descriptor(
                predicate,
                scalar=scalar_pair.unsigned,
                operands=(
                    _id_result(),
                    _id_operand("lhs"),
                    _id_operand("rhs"),
                ),
            )
            for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
            for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
        ]
    )
    descriptors.extend(
        [
            _compare_descriptor(
                predicate,
                scalar=OFFSET64_ALU_TYPE,
                operands=(
                    _id_result(),
                    _offset64_operand("lhs"),
                    _offset64_operand("rhs"),
                ),
            )
            for predicate in OFFSET64_COMPARE_PREDICATES
        ]
    )
    return tuple(descriptors)


def _select_descriptors() -> tuple[Descriptor, ...]:
    descriptors = [
        _select_descriptor(
            key=f"spirv.op_select.{scalar.suffix}",
            mnemonic="OpSelect"
            if scalar.suffix == "i32"
            else f"OpSelect.{scalar.suffix}",
            operands=(
                _id_result(),
                _id_operand("condition"),
                _id_operand("true_value"),
                _id_operand("false_value"),
            ),
            result_value_type=_scalar_alu_result_value_type(scalar),
            feature_bits=scalar.feature_bits,
        )
        for scalar in SCALAR_ALU_TYPES
    ]
    descriptors.append(
        _select_descriptor(
            key="spirv.op_select.bf16",
            mnemonic="OpSelect.bf16",
            operands=(
                _id_result(),
                _id_operand("condition"),
                _id_operand("true_value"),
                _id_operand("false_value"),
            ),
            result_value_type=_scalar_result_value_type("bf16"),
            feature_bits=BFLOAT16_CONSTANT_TYPE.feature_bits,
        )
    )
    descriptors.append(
        _select_descriptor(
            key="spirv.op_select.bool",
            mnemonic="OpSelect.bool",
            operands=(
                _id_result(),
                _id_operand("condition"),
                _id_operand("true_value"),
                _id_operand("false_value"),
            ),
            result_value_type=_scalar_result_value_type("i1"),
        )
    )
    descriptors.append(
        _select_descriptor(
            key="spirv.op_select.offset64",
            mnemonic="OpSelect.offset64",
            operands=(
                _offset64_result(),
                _id_operand("condition"),
                _offset64_operand("true_value"),
                _offset64_operand("false_value"),
            ),
            result_value_type=None,
        )
    )
    return tuple(descriptors)


def _ptr_workgroup_reg_classes() -> tuple[RegClass, ...]:
    classes: list[RegClass] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        classes.append(
            RegClass(
                _ptr_workgroup_array_reg_class_name(scalar),
                32,
                SpillSlotSpace.PRIVATE,
                flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.UNSPILLABLE),
            )
        )
        classes.append(
            RegClass(
                _ptr_workgroup_reg_class_name(scalar),
                32,
                SpillSlotSpace.PRIVATE,
                flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.UNSPILLABLE),
            )
        )
    return tuple(classes)


SPIRV_LOGICAL_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="spirv.logical.core",
    target_key="spirv",
    feature_key="spirv.logical.v1",
    c_header_path=Path("loom/src/loom/target/arch/spirv/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/spirv/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_SPIRV_DESCRIPTORS_H_",
    public_header="loom/target/arch/spirv/descriptors/descriptors.h",
    function_name="loom_spirv_logical_core_descriptor_set",
    c_table_prefix="SpirvLogicalCore",
    c_enum_prefix="SPIRV_LOGICAL_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_ID,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_OFFSET64,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_PTR_FUNCTION,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_PTR_STORAGE_BUFFER,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        *_ptr_workgroup_reg_classes(),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_MATRIX, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_VARIABLE, capacity_per_cycle=1, kind=ResourceKind.ADDRESS),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_LOAD,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MATRIX,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_MATRIX, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_STORE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VARIABLE,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VARIABLE, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
    ),
    descriptors=(
        *(_boolean_constant_descriptor(row) for row in BOOLEAN_CONSTANTS),
        *(
            _integer_constant_descriptor(scalar_pair)
            for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        ),
        *(_float_constant_descriptor(scalar) for scalar in FLOAT_CONSTANT_TYPES),
        Descriptor(
            key="spirv.op_constant.offset64",
            mnemonic="OpConstant.offset64",
            semantic_tag="spirv.op_constant.offset64",
            operands=(_offset64_result(),),
            op_kind=DescriptorOpKind.CONST,
            immediates=(_OFFSET64_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("offset64_value",)),
            schedule_class=_SCHEDULE_ALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_scalar_binary_descriptors(),
        *_conversion_descriptors(),
        *(_ordinary_vector_descriptor(row) for row in ORDINARY_VECTOR_INSTRUCTIONS),
        *(
            _ordinary_vector_descriptor(row)
            for row in ORDINARY_VECTOR_INTEGER_INSTRUCTIONS
        ),
        *(
            _ordinary_vector_descriptor(row)
            for row in ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS
        ),
        *(
            _ordinary_vector_descriptor(row)
            for row in ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS
        ),
        _coordinate_copy_descriptor(),
        _ternary_same_type_descriptor(
            key="spirv.op_imul_add.i32",
            mnemonic="OpIMulAdd",
            semantic_tag="spirv.op_imul_add.i32",
            operands=(
                _id_result(),
                _id_operand("a"),
                _id_operand("b"),
                _id_operand("c"),
            ),
            result_value_type=_scalar_result_value_type("i32"),
        ),
        _binary_same_type_descriptor(
            key="spirv.op_iadd.offset64",
            mnemonic="OpIAdd.offset64",
            semantic_tag="spirv.op_iadd.offset64",
            operands=(
                _offset64_result(),
                _offset64_operand("lhs"),
                _offset64_operand("rhs"),
            ),
            result_value_type=None,
        ),
        _binary_same_type_descriptor(
            key="spirv.op_isub.offset64",
            mnemonic="OpISub.offset64",
            semantic_tag="spirv.op_isub.offset64",
            operands=(
                _offset64_result(),
                _offset64_operand("lhs"),
                _offset64_operand("rhs"),
            ),
            result_value_type=None,
        ),
        _binary_same_type_descriptor(
            key="spirv.op_imul.offset64",
            mnemonic="OpIMul.offset64",
            semantic_tag="spirv.op_imul.offset64",
            operands=(
                _offset64_result(),
                _offset64_operand("lhs"),
                _offset64_operand("rhs"),
            ),
            result_value_type=None,
        ),
        _unary_typed_descriptor(
            key="spirv.op_bit_count.i32",
            mnemonic="OpBitCount",
            semantic_tag="spirv.op_bit_count.i32",
            operands=(_id_result(), _id_operand("input")),
            result_value_type=_scalar_result_value_type("i32"),
        ),
        *_address_conversion_descriptors(),
        *_builtin_index_descriptors(),
        *_compare_descriptors(),
        *_select_descriptors(),
        *_storage_buffer_descriptors(),
        *_raw_storage_buffer_byte_descriptors(),
        *_workgroup_descriptors(),
        *_control_barrier_descriptors(),
        *_cooperative_matrix_descriptors(),
        Descriptor(
            key="spirv.op_variable.function.ptr",
            mnemonic="OpVariable.function.ptr",
            semantic_tag="spirv.op_variable.function.ptr",
            operands=(_ptr_function_result(),),
            asm_forms=_asm(results=("ptr",)),
            schedule_class=_SCHEDULE_VARIABLE,
            instruction_classes=(InstructionClass.OTHER,),
        ),
    ),
)
