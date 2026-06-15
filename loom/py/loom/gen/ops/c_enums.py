# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C enum spelling tables for generated op metadata."""

from __future__ import annotations

from typing import Any

from loom.dsl import (
    CallLikeKind,
    OperandOwnershipEffectKind,
    OwnershipCarrier,
    ResultOwnershipEffectKind,
    TypeConstraint,
)
from loom.fields import FieldKind

# Maps Python TypeConstraint enum to C constraint enum name.
TYPE_CONSTRAINT_MAP: dict[TypeConstraint, str] = {
    TypeConstraint.TILE: "LOOM_TYPE_CONSTRAINT_TILE",
    TypeConstraint.TENSOR: "LOOM_TYPE_CONSTRAINT_TENSOR",
    TypeConstraint.VECTOR: "LOOM_TYPE_CONSTRAINT_VECTOR",
    TypeConstraint.RANK_ONE_VECTOR: "LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR",
    TypeConstraint.ALL_STATIC_VECTOR: "LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR",
    TypeConstraint.ALL_STATIC_RANK_ONE_VECTOR: "LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR",
    TypeConstraint.VIEW: "LOOM_TYPE_CONSTRAINT_VIEW",
    TypeConstraint.BUFFER: "LOOM_TYPE_CONSTRAINT_BUFFER",
    TypeConstraint.INTEGER: "LOOM_TYPE_CONSTRAINT_INTEGER",
    TypeConstraint.FLOAT: "LOOM_TYPE_CONSTRAINT_FLOAT",
    TypeConstraint.INDEX_OR_NON_I1_INTEGER_SCALAR: "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR",
    TypeConstraint.INTEGER_ELEMENT: "LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT",
    TypeConstraint.FLOAT_ELEMENT: "LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT",
    TypeConstraint.INDEX_OR_NON_I1_INTEGER_ELEMENT: "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT",
    TypeConstraint.I1_ELEMENT: "LOOM_TYPE_CONSTRAINT_I1_ELEMENT",
    TypeConstraint.I8_ELEMENT: "LOOM_TYPE_CONSTRAINT_I8_ELEMENT",
    TypeConstraint.I32_ELEMENT: "LOOM_TYPE_CONSTRAINT_I32_ELEMENT",
    TypeConstraint.F16_OR_BF16_ELEMENT: "LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT",
    TypeConstraint.F32_ELEMENT: "LOOM_TYPE_CONSTRAINT_F32_ELEMENT",
    TypeConstraint.SCALAR: "LOOM_TYPE_CONSTRAINT_SCALAR",
    TypeConstraint.INDEX: "LOOM_TYPE_CONSTRAINT_INDEX",
    TypeConstraint.OFFSET: "LOOM_TYPE_CONSTRAINT_OFFSET",
    TypeConstraint.ADDRESS: "LOOM_TYPE_CONSTRAINT_ADDRESS",
    TypeConstraint.ANY: "LOOM_TYPE_CONSTRAINT_ANY",
    TypeConstraint.GROUP: "LOOM_TYPE_CONSTRAINT_GROUP",
    TypeConstraint.ANY_ENCODING: "LOOM_TYPE_CONSTRAINT_ANY_ENCODING",
    TypeConstraint.ENCODING_LAYOUT: "LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT",
    TypeConstraint.ENCODING_SCHEMA: "LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA",
    TypeConstraint.ENCODING_STORAGE: "LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE",
    TypeConstraint.ENCODING_TRANSFORM: "LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM",
    TypeConstraint.POOL: "LOOM_TYPE_CONSTRAINT_POOL",
    TypeConstraint.REGISTER: "LOOM_TYPE_CONSTRAINT_REGISTER",
    TypeConstraint.STORAGE: "LOOM_TYPE_CONSTRAINT_STORAGE",
    TypeConstraint.I1: "LOOM_TYPE_CONSTRAINT_I1",
}

CALL_LIKE_KIND_MAP: dict[CallLikeKind, str] = {
    CallLikeKind.SEMANTIC: "LOOM_CALL_LIKE_KIND_SEMANTIC",
    CallLikeKind.LOW_INTERNAL: "LOOM_CALL_LIKE_KIND_LOW_INTERNAL",
    CallLikeKind.LOW_INVOKE: "LOOM_CALL_LIKE_KIND_LOW_INVOKE",
}

OWNERSHIP_CARRIER_MAP: dict[OwnershipCarrier, str] = {
    OwnershipCarrier.BY_VALUE: "LOOM_OWNERSHIP_CARRIER_BY_VALUE",
    OwnershipCarrier.BY_REFERENCE: "LOOM_OWNERSHIP_CARRIER_BY_REFERENCE",
}

OPERAND_OWNERSHIP_EFFECT_MAP: dict[OperandOwnershipEffectKind, str] = {
    OperandOwnershipEffectKind.BORROW: "LOOM_OPERAND_OWNERSHIP_BORROW",
    OperandOwnershipEffectKind.CONSUME: "LOOM_OPERAND_OWNERSHIP_CONSUME",
    OperandOwnershipEffectKind.RETAIN: "LOOM_OPERAND_OWNERSHIP_RETAIN",
    OperandOwnershipEffectKind.RELEASE: "LOOM_OPERAND_OWNERSHIP_RELEASE",
    OperandOwnershipEffectKind.DISCARD: "LOOM_OPERAND_OWNERSHIP_DISCARD",
    OperandOwnershipEffectKind.ESCAPE: "LOOM_OPERAND_OWNERSHIP_ESCAPE",
}

RESULT_OWNERSHIP_EFFECT_MAP: dict[ResultOwnershipEffectKind, str] = {
    ResultOwnershipEffectKind.FRESH: "LOOM_RESULT_OWNERSHIP_FRESH",
    ResultOwnershipEffectKind.BORROWED: "LOOM_RESULT_OWNERSHIP_BORROWED",
    ResultOwnershipEffectKind.RETAINED: "LOOM_RESULT_OWNERSHIP_RETAINED",
    ResultOwnershipEffectKind.ALIAS: "LOOM_RESULT_OWNERSHIP_ALIAS",
}

_ERROR_REF_CODE_BITS = 10
_ERROR_REF_MAX_CODE = (1 << _ERROR_REF_CODE_BITS) - 1
_ERROR_REF_MAX_DOMAIN_VALUE = (1 << (16 - _ERROR_REF_CODE_BITS)) - 2


def error_ref_literal(error: Any) -> str:
    """Returns a LOOM_ERROR_REF literal for a DSL error definition."""
    if error.code > _ERROR_REF_MAX_CODE:
        raise ValueError(f"{error.error_id}: code {error.code} exceeds LOOM_ERROR_REF 10-bit max")
    if error.domain.value > _ERROR_REF_MAX_DOMAIN_VALUE:
        raise ValueError(f"{error.error_id}: domain {error.domain.name} exceeds LOOM_ERROR_REF 6-bit max")
    return f"LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_{error.domain.name}, {error.code})"


# Maps Python trait names to C trait bit names.
TRAIT_MAP: dict[str, str] = {
    "Pure": "LOOM_TRAIT_PURE",
    "Commutative": "LOOM_TRAIT_COMMUTATIVE",
    "Idempotent": "LOOM_TRAIT_IDEMPOTENT",
    "Involution": "LOOM_TRAIT_INVOLUTION",
    "Terminator": "LOOM_TRAIT_TERMINATOR",
    "ConstantLike": "LOOM_TRAIT_CONSTANT_LIKE",
    "Elementwise": "LOOM_TRAIT_ELEMENTWISE",
    "Decomposable": "LOOM_TRAIT_DECOMPOSABLE",
    "SymbolDefine": "LOOM_TRAIT_SYMBOL_DEFINE",
    "IsolatedFromAbove": "LOOM_TRAIT_ISOLATED_FROM_ABOVE",
    "NonDeterministic": "LOOM_TRAIT_NON_DETERMINISTIC",
    "UnknownEffects": "LOOM_TRAIT_UNKNOWN_EFFECTS",
    "Convergent": "LOOM_TRAIT_CONVERGENT",
    "UniqueIdentity": "LOOM_TRAIT_UNIQUE_IDENTITY",
    "Hint": "LOOM_TRAIT_HINT",
    "SafeToSpeculate": "LOOM_TRAIT_SAFE_TO_SPECULATE",
    "RefinableResultTypeRefs": "LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS",
    "PoisonBoundary": "LOOM_TRAIT_POISON_BOUNDARY",
    "FactIdentity": "LOOM_TRAIT_FACT_IDENTITY",
    "DistributionTransfer": "LOOM_TRAIT_DISTRIBUTION_TRANSFER",
    "ValueAlias": "LOOM_TRAIT_VALUE_ALIAS",
}

# Maps Python constraint names to (relation, property) C enum pairs.
CONSTRAINT_MAP: dict[str, tuple[str, str]] = {
    "SameType": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_TYPE"),
    "SameKind": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_KIND"),
    "SameRegisterClass": (
        "LOOM_RELATION_PAIRWISE_EQ",
        "LOOM_PROPERTY_REGISTER_CLASS",
    ),
    "SameElementType": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_ELEMENT_TYPE"),
    "SameEncoding": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_ENCODING"),
    "SameShape": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_SHAPE"),
    "RanksMatch": ("LOOM_RELATION_PAIRWISE_EQ", "LOOM_PROPERTY_RANK"),
    "HasIntegerElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT",
    ),
    "HasFloatElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT",
    ),
    "HasIndexOrNonI1IntegerScalar": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR",
    ),
    "HasIndexOrNonI1IntegerElement": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT",
    ),
    "HasI1Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I1_ELEMENT",
    ),
    "HasI8Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I8_ELEMENT",
    ),
    "HasI32Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_I32_ELEMENT",
    ),
    "HasF16OrBf16Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT",
    ),
    "HasF32Element": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_F32_ELEMENT",
    ),
    "HasRegister": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_REGISTER",
    ),
    "HasRankOneVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR",
    ),
    "HasAllStaticVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR",
    ),
    "HasAllStaticRankOneVector": (
        "LOOM_RELATION_FIELD_SATISFIES",
        "LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR",
    ),
    "ElementWidthGreaterThan": (
        "LOOM_RELATION_ELEMENT_WIDTH_ORDER",
        "LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN",
    ),
    "ElementWidthLessThan": (
        "LOOM_RELATION_ELEMENT_WIDTH_ORDER",
        "LOOM_PROPERTY_ELEMENT_WIDTH_LESS_THAN",
    ),
    "ElementWidthAtLeastAttr": (
        "LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR",
        "LOOM_PROPERTY_ELEMENT_WIDTH_AT_LEAST_ATTR",
    ),
    "BitRangeWithinElementWidth": (
        "LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH",
        "LOOM_PROPERTY_BIT_RANGE_WITHIN_ELEMENT_WIDTH",
    ),
    "PositiveBitWidthAttr": (
        "LOOM_RELATION_ATTR_I64_PREDICATE",
        "LOOM_PROPERTY_BIT_WIDTH_POSITIVE",
    ),
    "AttrMatchesElementType": (
        "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "LiteralMatchesElementType": (
        "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "RegisterUnitsSumTo": (
        "LOOM_RELATION_REGISTER_UNIT_COUNT_SUM",
        "LOOM_PROPERTY_REGISTER_UNIT_COUNT",
    ),
    "TotalBitCountEqual": (
        "LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL",
        "LOOM_PROPERTY_TOTAL_BIT_COUNT",
    ),
    "PackedPayloadBitCountMatchesStorage": (
        "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
        "LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
    ),
    "UnpackedPayloadBitCountMatchesStorage": (
        "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
        "LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE",
    ),
    "OffsetCountMatchesRank": (
        "LOOM_RELATION_COUNT_MATCHES_RANK",
        "LOOM_PROPERTY_RANK",
    ),
    "ValueCountMatchesStaticElementCount": (
        "LOOM_RELATION_COUNT_MATCHES_STATIC_ELEMENT_COUNT",
        "LOOM_PROPERTY_TYPE",
    ),
    "DimIndexInBounds": ("LOOM_RELATION_ATTR_IN_RANGE_RANK", "LOOM_PROPERTY_RANK"),
    "AllShapesMatch": ("LOOM_RELATION_ALL_SAME", "LOOM_PROPERTY_SHAPE"),
    "LastAxisGroupedBy": ("LOOM_RELATION_LAST_AXIS_GROUPED_BY", "$data"),
    "BlockArgCount": ("LOOM_RELATION_REGION_ARG_COUNT", "LOOM_PROPERTY_TYPE"),
    "BlockArgsSatisfy": (
        "LOOM_RELATION_REGION_ARGS_SATISFY",
        "$type_constraint_data",
    ),
    "BlockArgsMatchTypes": (
        "LOOM_RELATION_REGION_ARG_MATCH",
        "LOOM_PROPERTY_TYPE",
    ),
    "BlockArgsMatchElementTypes": (
        "LOOM_RELATION_REGION_ARG_MATCH",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "YieldCountMatchesResults": ("LOOM_RELATION_YIELD_COUNT", "LOOM_PROPERTY_TYPE"),
    "YieldTypesMatchResults": ("LOOM_RELATION_YIELD_MATCH", "LOOM_PROPERTY_TYPE"),
    "YieldElementTypesMatchResults": (
        "LOOM_RELATION_YIELD_MATCH",
        "LOOM_PROPERTY_ELEMENT_TYPE",
    ),
    "VariadicValuesMatch": (
        "LOOM_RELATION_VARIADIC_MATCH",
        "LOOM_PROPERTY_TYPE",
    ),
    "IterArgsMatchResults": (
        "LOOM_RELATION_VARIADIC_MATCH",
        "LOOM_PROPERTY_TYPE",
    ),
}

# Maps FieldKind to C field category value.
FIELD_CATEGORY_MAP: dict[int, int] = {
    FieldKind.OPERAND: 0,  # LOOM_FIELD_OPERAND
    FieldKind.RESULT: 1,  # LOOM_FIELD_RESULT
    FieldKind.ATTR: 2,  # LOOM_FIELD_ATTR
    FieldKind.REGION: 3,  # LOOM_FIELD_REGION
}

# LOOM_FIELD_REF stores a 6-bit index in the generated constraint-table
# encoding. Printer callbacks use a wider representation, but constraint args
# still need a hard generator-side guard against silent truncation.
LOOM_FIELD_REF_MAX_INDEX = 63

# Maps Python attr_type strings to C attr kind enum names.
ATTR_KIND_MAP: dict[str, str] = {
    "i64": "LOOM_ATTR_I64",
    "f64": "LOOM_ATTR_F64",
    "string": "LOOM_ATTR_STRING",
    "bool": "LOOM_ATTR_BOOL",
    "enum": "LOOM_ATTR_ENUM",
    "i64_array": "LOOM_ATTR_I64_ARRAY",
    "symbol": "LOOM_ATTR_SYMBOL",
    "type": "LOOM_ATTR_TYPE",
    "encoding": "LOOM_ATTR_ENCODING",
    "predicate_list": "LOOM_ATTR_PREDICATE_LIST",
    "dict": "LOOM_ATTR_DICT",
    "any": "LOOM_ATTR_ANY",
}
