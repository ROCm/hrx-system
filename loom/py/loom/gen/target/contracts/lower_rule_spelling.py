# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C spelling helpers for generated lower-rule tables."""

from __future__ import annotations

from loom.dsl import Op
from loom.errors import ErrorDef, ErrorDomain
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.c import c_identifier_parts as _identifier_parts
from loom.target.contracts import (
    LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE,
    LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE,
    LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED,
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,
    ContractFragment,
    GuardKind,
    LowerAttrCopyKind,
    LowerEmitKind,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    SourceValueKind,
    TypePattern,
    contract_fragment_public_header,
)
from loom.target.contracts.diagnostics import DiagnosticParamKind

SCALAR_TYPE_C_NAMES = {
    "index": "LOOM_SCALAR_TYPE_INDEX",
    "offset": "LOOM_SCALAR_TYPE_OFFSET",
    "i1": "LOOM_SCALAR_TYPE_I1",
    "i8": "LOOM_SCALAR_TYPE_I8",
    "i16": "LOOM_SCALAR_TYPE_I16",
    "i32": "LOOM_SCALAR_TYPE_I32",
    "i64": "LOOM_SCALAR_TYPE_I64",
    "f8E4M3": "LOOM_SCALAR_TYPE_F8E4M3",
    "f8E5M2": "LOOM_SCALAR_TYPE_F8E5M2",
    "f16": "LOOM_SCALAR_TYPE_F16",
    "bf16": "LOOM_SCALAR_TYPE_BF16",
    "f32": "LOOM_SCALAR_TYPE_F32",
    "f64": "LOOM_SCALAR_TYPE_F64",
}

VALUE_REF_KIND_C_NAMES = {
    SourceValueKind.OPERAND: "LOOM_LOW_LOWER_VALUE_REF_OPERAND",
    SourceValueKind.RESULT: "LOOM_LOW_LOWER_VALUE_REF_RESULT",
    SourceValueKind.TEMPORARY: "LOOM_LOW_LOWER_VALUE_REF_TEMPORARY",
    SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM: "LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM",
    SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET: "LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET",
}

ATTR_KIND_C_NAMES = {
    "i64": "LOOM_ATTR_I64",
    "f64": "LOOM_ATTR_F64",
    "string": "LOOM_ATTR_STRING",
    "bool": "LOOM_ATTR_BOOL",
    "enum": "LOOM_ATTR_ENUM",
    "type": "LOOM_ATTR_TYPE",
    "i64_array": "LOOM_ATTR_I64_ARRAY",
    "encoding": "LOOM_ATTR_ENCODING",
    "symbol": "LOOM_ATTR_SYMBOL",
    "flags": "LOOM_ATTR_FLAGS",
    "predicate_list": "LOOM_ATTR_PREDICATE_LIST",
    "dict": "LOOM_ATTR_DICT",
}

GUARD_KIND_C_NAMES = {
    GuardKind.VALUE_TYPE: "LOOM_LOW_LOWER_GUARD_VALUE_TYPE",
    GuardKind.ATTR_KIND: "LOOM_LOW_LOWER_GUARD_ATTR_KIND",
    GuardKind.ENUM_ATTR_EQUALS: "LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ",
    GuardKind.I64_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE",
    GuardKind.DESCRIPTOR_AVAILABLE: "LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE",
    GuardKind.VALUE_MATERIALIZABLE: "LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE",
    GuardKind.LOW_VALUE_REGISTER_CLASS: "LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS",
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT: "LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT",
    GuardKind.VALUE_STATIC_DIM0_MULTIPLE: "LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE",
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ: "LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ",
    GuardKind.OPERAND_SEGMENT_COUNT: "LOOM_LOW_LOWER_GUARD_OPERAND_SEGMENT_COUNT_EQ",
    GuardKind.I64_ARRAY_COUNT: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ",
    GuardKind.I64_ARRAY_ELEMENT_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE",
    GuardKind.I64_ARRAY_ELEMENTS_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE",
    GuardKind.VALUE_SIGNED_BIT_COUNT: "LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT",
    GuardKind.VALUE_UNSIGNED_BIT_COUNT: "LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT",
    GuardKind.VALUE_EXACT_I64: "LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64",
    GuardKind.VALUE_EXACT_POWER_OF_TWO_I64: "LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64",
    GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD: "LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_IS_ADD",
    GuardKind.VALUE_EXACT_F64: "LOOM_LOW_LOWER_GUARD_VALUE_EXACT_F64",
    GuardKind.VALUE_I64_RANGE: "LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE",
    GuardKind.VALUE_I64_RANGE_LE: "LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_LE",
    GuardKind.VALUE_I64_RANGE_GE: "LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_GE",
    GuardKind.VALUE_F64_EQUALS: "LOOM_LOW_LOWER_GUARD_VALUE_F64_EQUALS",
    GuardKind.VALUE_STORAGE_ELEMENT_FORMAT: "LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT",
    GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES: "LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES",
    GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD: "LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD",
    GuardKind.VALUE_NO_USES: "LOOM_LOW_LOWER_GUARD_VALUE_NO_USES",
    GuardKind.INSTANCE_FLAGS_HAS_ALL: "LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL",
    GuardKind.VECTOR_EXTRACT_SHAPE: "LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE",
    GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ: "LOOM_LOW_LOWER_GUARD_VALUE_STATIC_ELEMENT_COUNT_EQ",
}

ATTR_COPY_KIND_C_NAMES = {
    LowerAttrCopyKind.DIRECT: "LOOM_LOW_LOWER_ATTR_COPY_DIRECT",
    LowerAttrCopyKind.ENUM_ORDINAL: "LOOM_LOW_LOWER_ATTR_COPY_ENUM_ORDINAL",
    LowerAttrCopyKind.I64_ARRAY_ELEMENT: "LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT",
    LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS: "LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS",
    LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE: "LOOM_LOW_LOWER_ATTR_COPY_I64_ATTRS_PACK_CONSECUTIVE",
    LowerAttrCopyKind.I64_LITERAL: "LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL",
    LowerAttrCopyKind.VALUE_EXACT_I64: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64",
    LowerAttrCopyKind.VALUE_EXACT_I64_NEGATE: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_NEGATE",
    LowerAttrCopyKind.VALUE_EXACT_I64_LOG2: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_LOG2",
    LowerAttrCopyKind.VALUE_EXACT_I64_MINUS_ONE: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE",
    LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_MULTIPLIER: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER",
    LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_SHIFT: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT",
    LowerAttrCopyKind.VALUE_I32_AS_U32_BITS: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_I32_AS_U32_BITS",
    LowerAttrCopyKind.VALUE_F64_AS_F16_BITS: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F16_BITS",
    LowerAttrCopyKind.VALUE_F64_AS_BF16_BITS: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_BF16_BITS",
    LowerAttrCopyKind.VALUE_F64_AS_F32_BITS: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F32_BITS",
    LowerAttrCopyKind.VALUE_F64_AS_F64_BITS: "LOOM_LOW_LOWER_ATTR_COPY_VALUE_F64_AS_F64_BITS",
    LowerAttrCopyKind.I64_ARRAY_LANE_BYTE: "LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_LANE_BYTE",
    LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET: "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET",
    LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT: "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT",
    LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER: "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER",
    LowerAttrCopyKind.SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE: "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE",
    LowerAttrCopyKind.SOURCE_OP_INSTANCE_FLAGS: "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_OP_INSTANCE_FLAGS",
    LowerAttrCopyKind.I64_LOW_BIT_MASK: "LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK",
    LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_MASK: "LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK",
    LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK: "LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK",
    LowerAttrCopyKind.I64_LITERAL_MINUS_ATTR: "LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR",
    LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS: "LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS",
}

EMIT_KIND_C_NAMES = {
    LowerEmitKind.DESCRIPTOR_OP: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP",
    LowerEmitKind.DESCRIPTOR_CONST: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST",
    LowerEmitKind.DESCRIPTOR_OP_FIRST_LANE: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE",
    LowerEmitKind.DESCRIPTOR_OP_PER_LANE: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE",
    LowerEmitKind.DESCRIPTOR_OP_PER_LANE_SEQUENCE: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE",
    LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES",
}

EMIT_FLAG_C_NAMES = {
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1: "LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1",
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS: "LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS",
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN: "LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN",
    LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE: "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE",
    LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED: "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED",
    LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE: "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE",
    LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE: "LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE",
}

SOURCE_MEMORY_OPERATION_C_NAMES = {
    SourceMemoryOperation.LOAD: "LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD",
    SourceMemoryOperation.STORE: "LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE",
    SourceMemoryOperation.PREFETCH: "LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH",
    SourceMemoryOperation.ATOMIC_REDUCE: "LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE",
    SourceMemoryOperation.ATOMIC_RMW: "LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW",
    SourceMemoryOperation.ATOMIC_CMPXCHG: "LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG",
}

SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_C_NAMES = {
    SourceMemoryDynamicIndexSource.NONE: "LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE",
    SourceMemoryDynamicIndexSource.VALUE: "LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE",
    SourceMemoryDynamicIndexSource.WORKITEM_ID: "LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID",
    SourceMemoryDynamicIndexSource.WORKGROUP_ID: "LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID",
}

SOURCE_MEMORY_ROOT_KIND_C_NAMES = {
    SourceMemoryRootKind.ANY: "LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY",
    SourceMemoryRootKind.BLOCK_ARGUMENT: "LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT",
}

SOURCE_MEMORY_SPACE_C_NAMES = {
    "unknown": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_UNKNOWN",
    "global": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GLOBAL",
    "workgroup": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_WORKGROUP",
    "private": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_PRIVATE",
    "constant": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_CONSTANT",
    "host": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_HOST",
    "descriptor": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_DESCRIPTOR",
    "generic": "LOOM_LOW_LOWER_SOURCE_MEMORY_SPACE_GENERIC",
}

DIAGNOSTIC_PARAM_KIND_C_NAMES = {
    DiagnosticParamKind.TARGET_KEY: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY",
    DiagnosticParamKind.EXPORT_NAME: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME",
    DiagnosticParamKind.CONFIG_KEY: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY",
    DiagnosticParamKind.FUNCTION_NAME: ("LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME"),
    DiagnosticParamKind.SOURCE_OP_NAME: ("LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME"),
    DiagnosticParamKind.STRING_LITERAL: ("LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL"),
    DiagnosticParamKind.VALUE_TYPE: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE",
    DiagnosticParamKind.I64_LITERAL: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL",
    DiagnosticParamKind.U32_LITERAL: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL",
    DiagnosticParamKind.U64_LITERAL: "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL",
    DiagnosticParamKind.BOOL_LITERAL: ("LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL"),
    DiagnosticParamKind.SOURCE_MEMORY_MINIMUM_ALIGNMENT: ("LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_MEMORY_MINIMUM_ALIGNMENT"),
}

ERROR_DOMAIN_C_NAMES = {
    ErrorDomain.TYPE: "LOOM_ERROR_DOMAIN_TYPE",
    ErrorDomain.SHAPE: "LOOM_ERROR_DOMAIN_SHAPE",
    ErrorDomain.SUBRANGE: "LOOM_ERROR_DOMAIN_SUBRANGE",
    ErrorDomain.ENCODING: "LOOM_ERROR_DOMAIN_ENCODING",
    ErrorDomain.STRUCTURE: "LOOM_ERROR_DOMAIN_STRUCTURE",
    ErrorDomain.DOMINANCE: "LOOM_ERROR_DOMAIN_DOMINANCE",
    ErrorDomain.SYMBOL: "LOOM_ERROR_DOMAIN_SYMBOL",
    ErrorDomain.PARSE: "LOOM_ERROR_DOMAIN_PARSE",
    ErrorDomain.BYTECODE: "LOOM_ERROR_DOMAIN_BYTECODE",
    ErrorDomain.FOLD: "LOOM_ERROR_DOMAIN_FOLD",
    ErrorDomain.LOWERING: "LOOM_ERROR_DOMAIN_LOWERING",
    ErrorDomain.BACKEND: "LOOM_ERROR_DOMAIN_BACKEND",
    ErrorDomain.TARGET: "LOOM_ERROR_DOMAIN_TARGET",
    ErrorDomain.AMDGPU: "LOOM_ERROR_DOMAIN_AMDGPU",
    ErrorDomain.X86: "LOOM_ERROR_DOMAIN_X86",
    ErrorDomain.WASM: "LOOM_ERROR_DOMAIN_WASM",
    ErrorDomain.SPIRV: "LOOM_ERROR_DOMAIN_SPIRV",
}


def u64_c_literal(value: int) -> str:
    return f"UINT64_C({value})"


def error_ref_c_expr(error: ErrorDef) -> str:
    return f"LOOM_ERROR_REF({ERROR_DOMAIN_C_NAMES[error.domain]}, {error.code})"


def type_kind_c_name(type_pattern: TypePattern) -> str:
    if type_pattern.kind == "scalar":
        return "LOOM_TYPE_SCALAR"
    if type_pattern.kind == "vector":
        return "LOOM_TYPE_VECTOR"
    if type_pattern.kind == "view":
        return "LOOM_TYPE_VIEW"
    raise ValueError(f"unknown type pattern kind '{type_pattern.kind}'")


def scalar_type_mask_c_expr(elements: tuple[str, ...]) -> str:
    return " | ".join(f"LOOM_LOW_LOWER_SCALAR_TYPE_BIT({scalar_type_c_name(element)})" for element in elements)


def scalar_type_c_name(element: str | None) -> str:
    if element is None:
        raise ValueError("type pattern element is required")
    c_name = SCALAR_TYPE_C_NAMES.get(element)
    if c_name is None:
        raise ValueError(f"unknown scalar type '{element}'")
    return c_name


def diagnostic_index(index: int) -> str:
    if index == 0xFFFF:
        return "LOOM_LOW_LOWER_DIAGNOSTIC_NONE"
    return str(index)


def source_memory_space_mask(memory_spaces: tuple[str, ...]) -> str:
    c_names: list[str] = []
    for memory_space in memory_spaces:
        c_name = SOURCE_MEMORY_SPACE_C_NAMES.get(memory_space)
        if c_name is None:
            raise ValueError(f"unknown source memory space '{memory_space}'")
        c_names.append(c_name)
    return " | ".join(c_names)


def attr_kind_c_name(attr_kind: str | None) -> str:
    if attr_kind is None:
        return "0"
    c_name = ATTR_KIND_C_NAMES.get(attr_kind)
    if c_name is None:
        raise ValueError(f"unknown attr kind '{attr_kind}'")
    return c_name


def emit_flags(flags: int) -> str:
    if flags == 0:
        return "0"
    names = [c_name for bit, c_name in sorted(EMIT_FLAG_C_NAMES.items()) if flags & bit]
    unknown = flags & ~sum(EMIT_FLAG_C_NAMES)
    if unknown:
        raise ValueError(f"unknown emit flags 0x{unknown:x}")
    return " | ".join(names)


def c_bool(value: bool) -> str:
    return "true" if value else "false"


def op_c_name(op: Op) -> str:
    return "LOOM_OP_" + _c_identifier(op.name).upper()


def generated_public_header(table: ContractFragment) -> str:
    public_header = contract_fragment_public_header(table)
    if not public_header.endswith(".h"):
        raise ValueError(f"contract fragment '{table.name}' public_header must end with .h")
    return f"{public_header[:-2]}_lower_rules.h"


def generated_symbol_name(table: ContractFragment) -> str:
    return f"loom_{_c_identifier(table.name).lower()}_lower_rule_set"


def generated_table_prefix(table: ContractFragment) -> str:
    return f"{pascal_identifier(table.name)}Lower"


def c_expression(value: int | str) -> str:
    return str(value)


def header_guard_from_public_header(public_header: str) -> str:
    return _c_identifier(public_header).upper() + "_"


def pascal_identifier(value: str) -> str:
    parts = _identifier_parts(value)
    if not parts:
        return "_"
    return "".join(part[:1].upper() + part[1:] for part in parts)
