# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom op declaration DSL.

This module defines the types used to declare loom operations in Python.
These declarations are the single source of truth: generators consume
them to produce C headers, typed builder APIs, JSON databases, and
documentation.

Op declarations are dataclass instances (not class hierarchies). Each
declaration is pure data describing an op's interface: what it takes,
what it produces, how it looks in textual assembly, what constraints
it enforces.

Dialect files import the specific types they need from this module and
from loom.assembly. ruff manages import lists automatically.

Quick reference — declaring an op:

    scalar_addi = Op(
        name="scalar.addi",
        group=scalar_ops,
        doc="Integer addition.",
        operands=[
            Operand("lhs", INTEGER),
            Operand("rhs", INTEGER),
        ],
        results=[Result("result", INTEGER)],
        constraints=[SameType("lhs", "rhs", "result")],
        traits=[PURE, COMMUTATIVE],
        format=[Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")],
    )
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from loom.assembly import FormatElement, OptionalGroup
from loom.errors import ErrorDef

if TYPE_CHECKING:
    from loom.assembly import FuncArgs

__all__ = [
    # Type constraints.
    "TypeConstraint",
    "TILE",
    "TENSOR",
    "VECTOR",
    "RANK_ONE_VECTOR",
    "ALL_STATIC_VECTOR",
    "ALL_STATIC_RANK_ONE_VECTOR",
    "VIEW",
    "BUFFER",
    "INTEGER",
    "INDEX_OR_NON_I1_INTEGER_SCALAR",
    "INTEGER_ELEMENT",
    "INDEX_OR_NON_I1_INTEGER_ELEMENT",
    "I8_ELEMENT",
    "I32_ELEMENT",
    "FLOAT",
    "FLOAT_ELEMENT",
    "F16_OR_BF16_ELEMENT",
    "F32_ELEMENT",
    "I1_ELEMENT",
    "SCALAR",
    "INDEX",
    "OFFSET",
    "ADDRESS",
    "ANY",
    "ANY_ENCODING",
    "ENCODING_LAYOUT",
    "ENCODING_SCHEMA",
    "ENCODING_STORAGE",
    "ENCODING_TRANSFORM",
    "POOL",
    "REGISTER",
    "STORAGE",
    "I1",
    # Type constraint helpers.
    "type_constraint_name",
    # Field descriptors.
    "OperandRole",
    "Operand",
    "Result",
    "TiedResult",
    "Successor",
    "AttrDef",
    "AttrType",
    "ATTR_TYPE_I64",
    "ATTR_TYPE_F64",
    "ATTR_TYPE_STRING",
    "ATTR_TYPE_BOOL",
    "ATTR_TYPE_ENUM",
    "ATTR_TYPE_ENUM_ARRAY",
    "ATTR_TYPE_SIGNED_ENUM_SET",
    "ATTR_TYPE_SCOPED_ENUM",
    "ATTR_TYPE_TYPE",
    "ATTR_TYPE_I64_ARRAY",
    "ATTR_TYPE_BYTES",
    "ATTR_TYPE_ENCODING",
    "ATTR_TYPE_ANY",
    "ATTR_TYPE_SYMBOL",
    "ATTR_TYPE_SYMBOL_ARRAY",
    "ATTR_TYPE_SYMBOL_SET",
    "ATTR_TYPE_FLAGS",
    "ATTR_TYPE_PREDICATE_LIST",
    "ATTR_TYPE_DICT",
    "ATTR_TYPE_PARAMETERIZED",
    "ATTR_TYPE_PARAMETERIZED_ARRAY",
    "RegionDef",
    # Symbol support.
    "SymbolDefinition",
    "SymbolDefinitionFlag",
    "SymbolReference",
    "SymbolValueContract",
    # Enum support.
    "EnumCase",
    "EnumDef",
    # Traits.
    "Trait",
    "PURE",
    "COMMUTATIVE",
    "IDEMPOTENT",
    "INVOLUTION",
    "TERMINATOR",
    "CONSTANT_LIKE",
    "POISON",
    "ELEMENTWISE",
    "DECOMPOSABLE",
    "SYMBOL_DEFINE",
    "MODULE_SCOPE",
    "KeyedModuleRecord",
    "ISOLATED_FROM_ABOVE",
    "NON_DETERMINISTIC",
    "UNKNOWN_EFFECTS",
    "MEMORY_FENCE",
    "CONVERGENT",
    "HINT",
    "COMPILE_TIME_ONLY",
    "SAFE_TO_SPECULATE",
    "REFINABLE_RESULT_TYPE_REFS",
    "POISON_BOUNDARY",
    "FACT_IDENTITY",
    "DISTRIBUTION_TRANSFER",
    "STORAGE_RELATION",
    "VALUE_ALIAS",
    # Semantic phase/contract metadata.
    "OpPhase",
    "ContractFamily",
    "ConditionRefinement",
    "ConditionRefinementTruth",
    "TypeSemantic",
    "OpCategory",
    # Trait constructors.
    "AllTypesMatch",
    "HasAncestor",
    "HasParent",
    "ImplicitTerminator",
    "NoAncestor",
    # Memory effects.
    "EffectKind",
    "Effect",
    "Reads",
    "Writes",
    "ReadWrites",
    # Ownership effects.
    "OwnershipCarrier",
    "BY_VALUE",
    "BY_REFERENCE",
    "OperandOwnershipEffectKind",
    "ResultOwnershipEffectKind",
    "OperandOwnershipEffect",
    "ResultOwnershipEffect",
    "Borrow",
    "Consume",
    "Retain",
    "Release",
    "Discard",
    "Escape",
    "FreshResult",
    "BorrowedResult",
    "RetainedResult",
    "AliasResult",
    "MovedResult",
    # Constraints.
    "Constraint",
    "SameType",
    "SameKind",
    "SameRegisterClass",
    "SameElementType",
    "SameEncoding",
    "SameShape",
    "RanksMatch",
    "HasIntegerElement",
    "HasFloatElement",
    "HasBitwiseScalar",
    "HasBitwiseElement",
    "HasIndexOrNonI1IntegerScalar",
    "HasIndexOrNonI1IntegerElement",
    "HasI1Element",
    "HasI8Element",
    "HasI32Element",
    "HasF16OrBf16Element",
    "HasF32Element",
    "HasRegister",
    "HasRankOneVector",
    "HasAllStaticVector",
    "HasAllStaticRankOneVector",
    "ElementWidthGreaterThan",
    "ElementWidthLessThan",
    "ElementWidthAtLeastAttr",
    "BitRangeWithinElementWidth",
    "PositiveBitWidthAttr",
    "TotalBitCountEqual",
    "PackedPayloadBitCountMatchesStorage",
    "UnpackedPayloadBitCountMatchesStorage",
    "OffsetCountMatchesRank",
    "ValueCountMatchesStaticElementCount",
    "DimIndexInBounds",
    "AllShapesMatch",
    "LastAxisGroupedBy",
    "BlockArgCount",
    "BlockArgsSatisfy",
    "BlockArgsMatchTypes",
    "BlockArgsMatchElementTypes",
    "YieldCountMatchesResults",
    "YieldTypesMatchResults",
    "YieldElementTypesMatchResults",
    "VariadicValuesMatch",
    "IterArgsMatchResults",
    "AttrMatchesElementType",
    "LiteralMatchesElementType",
    "RegisterUnitsSumTo",
    # Op group.
    "Dialect",
    "EncodingAliasDef",
    "EncodingOperandSummaryDef",
    "EncodingFamilyDef",
    "EncodingFamilyRole",
    "EncodingRecordDef",
    "ParameterizedAttrDef",
    # Legacy text-format migration declarations.
    "LegacyFieldDefault",
    "LegacyFieldMapping",
    "LegacyFormat",
    # Interfaces.
    "CallLikeInterface",
    "CallLikeKind",
    "FuncLikeInterfaceFlag",
    "FuncLikeInterface",
    "InlinePolicy",
    "LoopLikeInterface",
    "MemoryAccessInterface",
    "MemoryAccessOperationKind",
    "RegionBranchInterface",
    "TargetFactSpecialization",
    "TargetLikeInterface",
    # Op declaration.
    "Op",
    # Type declaration.
    "TypeDef",
    "TypeParam",
    "ShapeParam",
    "ScalarParam",
    "EncodingParam",
    # Helpers.
    "binary_op",
    "unary_op",
    "cast_op",
    "comparison_op",
]


# ============================================================================
# Type constraints
# ============================================================================


@unique
class TypeConstraint(Enum):
    """Constraint on the type of an operand or result.

    These are abstract categories, not concrete types. The verifier
    checks that the actual type of a value satisfies the constraint.
    The builder generator maps them to Python type hints.

    Constraint → ir.py types that satisfy it:
      TILE     → ShapedType with type_kind=TILE
      TENSOR   → ShapedType with type_kind=TENSOR
      VECTOR   → ShapedType with type_kind=VECTOR
      RANK_ONE_VECTOR → vector type with rank 1
      ALL_STATIC_VECTOR → vector type with all-static shape
      ALL_STATIC_RANK_ONE_VECTOR → vector type with all-static rank-1 shape
      VIEW     → ShapedType with type_kind=VIEW
      BUFFER   → BufferType
      INTEGER  → ScalarType with kind in {I1, I8, I16, I32, I64}
      FLOAT    → ScalarType with kind in {F8*, F16, BF16, F32, F64}
      BITWISE_SCALAR → ScalarType index, non-i1 integer, or floating-point
      INDEX_OR_NON_I1_INTEGER_SCALAR → ScalarType index or non-i1 integer
      INTEGER_ELEMENT → ShapedType with integer element type
      FLOAT_ELEMENT   → ShapedType with float element type
      BITWISE_ELEMENT → ShapedType with index, non-i1 integer, or float element
      INDEX_OR_NON_I1_INTEGER_ELEMENT → ShapedType index or non-i1 integer element
      I1_ELEMENT      → ShapedType with element type i1
      I8_ELEMENT      → ShapedType with element type i8
      I32_ELEMENT     → ShapedType with element type i32
      F16_OR_BF16_ELEMENT → ShapedType with element type f16 or bf16
      F32_ELEMENT     → ShapedType with element type f32
      SCALAR   → any ScalarType (INTEGER | FLOAT | INDEX | OFFSET)
      INDEX    → ScalarType with kind=INDEX
      OFFSET   → ScalarType with kind=OFFSET
      ADDRESS  → ScalarType with kind in {INDEX, OFFSET}
      ANY      → any type
      ANY_ENCODING → any EncodingType
      ENCODING_LAYOUT → EncodingType with role=layout
      ENCODING_SCHEMA → EncodingType with role=schema
      ENCODING_STORAGE → EncodingType with role=storage
      ENCODING_TRANSFORM → EncodingType with role=transform
      POOL     → PoolType
      REGISTER → RegisterType
      STORAGE  → StorageType

    Element-qualified constraints are shaped-only: tile, tensor, vector,
    and view types can satisfy them, while scalar values continue to use
    INTEGER/FLOAT/I1. Combine a shaped kind constraint with SameElementType
    when an op needs both a specific shaped kind and a shared element family.
    Vector shape constraints additionally require type_kind=VECTOR so ops can
    declare common register-shape invariants without custom verifier code.
    """

    TILE = "tile"
    TENSOR = "tensor"
    VECTOR = "vector"
    RANK_ONE_VECTOR = "rank_one_vector"
    ALL_STATIC_VECTOR = "all_static_vector"
    ALL_STATIC_RANK_ONE_VECTOR = "all_static_rank_one_vector"
    VIEW = "view"
    BUFFER = "buffer"
    INTEGER = "integer"
    FLOAT = "float"
    BITWISE_SCALAR = "bitwise_scalar"
    INDEX_OR_NON_I1_INTEGER_SCALAR = "index_or_non_i1_integer_scalar"
    INTEGER_ELEMENT = "integer_element"
    FLOAT_ELEMENT = "float_element"
    BITWISE_ELEMENT = "bitwise_element"
    INDEX_OR_NON_I1_INTEGER_ELEMENT = "index_or_non_i1_integer_element"
    I1_ELEMENT = "i1_element"
    I8_ELEMENT = "i8_element"
    I32_ELEMENT = "i32_element"
    F16_OR_BF16_ELEMENT = "f16_or_bf16_element"
    F32_ELEMENT = "f32_element"
    SCALAR = "scalar"
    INDEX = "index"
    OFFSET = "offset"
    ADDRESS = "address"
    ANY = "any"
    ANY_ENCODING = "encoding"
    ENCODING_LAYOUT = "encoding<layout>"
    ENCODING_SCHEMA = "encoding<schema>"
    ENCODING_STORAGE = "encoding<storage>"
    ENCODING_TRANSFORM = "encoding<transform>"
    POOL = "pool"
    REGISTER = "register"
    STORAGE = "storage"
    I1 = "i1"


# Singletons for use in op declarations.
TILE = TypeConstraint.TILE
TENSOR = TypeConstraint.TENSOR
VECTOR = TypeConstraint.VECTOR
RANK_ONE_VECTOR = TypeConstraint.RANK_ONE_VECTOR
ALL_STATIC_VECTOR = TypeConstraint.ALL_STATIC_VECTOR
ALL_STATIC_RANK_ONE_VECTOR = TypeConstraint.ALL_STATIC_RANK_ONE_VECTOR
VIEW = TypeConstraint.VIEW
BUFFER = TypeConstraint.BUFFER
INTEGER = TypeConstraint.INTEGER
FLOAT = TypeConstraint.FLOAT
BITWISE_SCALAR = TypeConstraint.BITWISE_SCALAR
INDEX_OR_NON_I1_INTEGER_SCALAR = TypeConstraint.INDEX_OR_NON_I1_INTEGER_SCALAR
INTEGER_ELEMENT = TypeConstraint.INTEGER_ELEMENT
FLOAT_ELEMENT = TypeConstraint.FLOAT_ELEMENT
BITWISE_ELEMENT = TypeConstraint.BITWISE_ELEMENT
INDEX_OR_NON_I1_INTEGER_ELEMENT = TypeConstraint.INDEX_OR_NON_I1_INTEGER_ELEMENT
I1_ELEMENT = TypeConstraint.I1_ELEMENT
I8_ELEMENT = TypeConstraint.I8_ELEMENT
I32_ELEMENT = TypeConstraint.I32_ELEMENT
F16_OR_BF16_ELEMENT = TypeConstraint.F16_OR_BF16_ELEMENT
F32_ELEMENT = TypeConstraint.F32_ELEMENT
SCALAR = TypeConstraint.SCALAR
INDEX = TypeConstraint.INDEX
OFFSET = TypeConstraint.OFFSET
ADDRESS = TypeConstraint.ADDRESS
ANY = TypeConstraint.ANY
ANY_ENCODING = TypeConstraint.ANY_ENCODING
ENCODING_LAYOUT = TypeConstraint.ENCODING_LAYOUT
ENCODING_SCHEMA = TypeConstraint.ENCODING_SCHEMA
ENCODING_STORAGE = TypeConstraint.ENCODING_STORAGE
ENCODING_TRANSFORM = TypeConstraint.ENCODING_TRANSFORM
POOL = TypeConstraint.POOL
REGISTER = TypeConstraint.REGISTER
STORAGE = TypeConstraint.STORAGE
I1 = TypeConstraint.I1


def type_constraint_name(constraint: TypeConstraint) -> str:
    """Return the display name for a type constraint."""
    return constraint.value


# ============================================================================
# Field descriptors
# ============================================================================


@unique
class OperandRole(Enum):
    """Semantic role of an operand field independent of its display name."""

    NONE = "none"
    CONTROL_CONDITION = "control_condition"
    SELECT_CONDITION = "select_condition"
    SELECT_PAYLOAD = "select_payload"
    BROADCAST_SOURCE = "broadcast_source"
    COMPOSITE_ELEMENT = "composite_element"
    FLOAT_EXTENSION_SOURCE = "float_extension_source"


@dataclass(frozen=True, slots=True)
class Operand:
    """An SSA value operand (input to the op at runtime).

    name: Field name, used in format specs and builders.
    type_constraint: What types this operand accepts.
    doc: Human-readable description.
    variadic: If True, this is zero-or-more values (list[Value]).
    optional: If True, this operand may be absent.
    role: Semantic role consumed by generic analyses.
    """

    name: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    optional: bool = False
    role: OperandRole = OperandRole.NONE


@dataclass(frozen=True, slots=True)
class Result:
    """An SSA value result (output of the op).

    name: Field name, used in format specs and builders.
    type_constraint: What types this result produces.
    doc: Human-readable description.
    variadic: If True, this is zero-or-more result values.
    allocates: If True, this result is a freshly allocated resource
        that cannot alias any pre-existing resource.
    """

    name: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    allocates: bool = False


@dataclass(frozen=True, slots=True)
class TiedResult:
    """A result that is always tied to a specific operand.

    The result reuses the operand's storage. The result type may differ
    from the operand type (e.g., reshape or encoding change over the
    same storage).

    name: Field name for the result.
    tied_to: Name of the operand this result is tied to.
    type_constraint: What types this result produces.
    doc: Human-readable description.
    variadic: For protocol compatibility with Result. Tied results
        cannot be variadic.
    allocates: Always False. Tied results reuse existing storage.
    """

    name: str
    tied_to: str
    type_constraint: TypeConstraint
    doc: str = ""
    variadic: bool = False
    allocates: bool = False


@dataclass(frozen=True, slots=True)
class Successor:
    """A CFG successor edge to another block in the enclosing region.

    name: Field name, used in format specs and builders.
    doc: Human-readable description.
    variadic: If True, this is a trailing zero-or-more successor field.

    Successors are semantic block references. Textual labels are parser/printer
    syntax and may be synthesized without changing the graph.
    """

    name: str
    doc: str = ""
    variadic: bool = False


# Allowed attribute type strings. Using Literal for static checking.
type AttrType = str  # One of the ATTR_TYPE_* constants below.

ATTR_TYPE_I64 = "i64"
ATTR_TYPE_F64 = "f64"
ATTR_TYPE_STRING = "string"
ATTR_TYPE_BOOL = "bool"
ATTR_TYPE_ENUM = "enum"
ATTR_TYPE_ENUM_ARRAY = "enum_array"
ATTR_TYPE_SIGNED_ENUM_SET = "signed_enum_set"
ATTR_TYPE_SCOPED_ENUM = "scoped_enum"
ATTR_TYPE_TYPE = "type"
ATTR_TYPE_I64_ARRAY = "i64_array"
ATTR_TYPE_BYTES = "bytes"
ATTR_TYPE_ENCODING = "encoding"
ATTR_TYPE_ANY = "any"
ATTR_TYPE_SYMBOL = "symbol"
ATTR_TYPE_SYMBOL_ARRAY = "symbol_array"
ATTR_TYPE_SYMBOL_SET = "symbol_set"
ATTR_TYPE_FLAGS = "flags"
ATTR_TYPE_PREDICATE_LIST = "predicate_list"
ATTR_TYPE_DICT = "dict"  # Named attribute dictionary.
ATTR_TYPE_PARAMETERIZED = "parameterized"
ATTR_TYPE_PARAMETERIZED_ARRAY = "parameterized_array"

_VALID_ATTR_TYPES = frozenset(
    {
        ATTR_TYPE_I64,
        ATTR_TYPE_F64,
        ATTR_TYPE_STRING,
        ATTR_TYPE_BOOL,
        ATTR_TYPE_ENUM,
        ATTR_TYPE_ENUM_ARRAY,
        ATTR_TYPE_SIGNED_ENUM_SET,
        ATTR_TYPE_SCOPED_ENUM,
        ATTR_TYPE_TYPE,
        ATTR_TYPE_I64_ARRAY,
        ATTR_TYPE_BYTES,
        ATTR_TYPE_ENCODING,
        ATTR_TYPE_ANY,
        ATTR_TYPE_SYMBOL,
        ATTR_TYPE_SYMBOL_ARRAY,
        ATTR_TYPE_SYMBOL_SET,
        ATTR_TYPE_FLAGS,
        ATTR_TYPE_PREDICATE_LIST,
        ATTR_TYPE_DICT,
        ATTR_TYPE_PARAMETERIZED,
        ATTR_TYPE_PARAMETERIZED_ARRAY,
    }
)


_VALID_SYMBOL_INTERFACES = frozenset(
    {
        "func_like",
        "callable",
        "template_family",
        "template_provider",
        "global",
        "executable",
        "record",
        "rodata",
        "target",
        "config",
        "kernel",
        "command_program",
    }
)


@unique
class SymbolReferenceRole(Enum):
    """Compile-time graph role of a symbol reference occurrence."""

    DEPENDENCY = "dependency"
    AVAILABILITY = "availability"


@dataclass(frozen=True, slots=True)
class SymbolReference:
    """Declares the expected target contract of a symbol attr.

    name: Human-readable expected symbol class used in diagnostics.
    interfaces: Generated symbol-definition interfaces accepted by the attr.
        These are structural contracts, not op names or bytecode wire kinds.
        An empty tuple accepts any symbol without imposing an additional
        interface constraint.
    role: Whether the reference contributes a dependency edge or only records
        where an otherwise non-live symbol may be found during compilation.
    """

    name: str
    interfaces: tuple[str, ...]
    role: SymbolReferenceRole

    def __init__(
        self,
        name: str,
        interfaces: list[str] | tuple[str, ...],
        role: SymbolReferenceRole = SymbolReferenceRole.DEPENDENCY,
    ) -> None:
        frozen_interfaces = tuple(interfaces)
        if not name:
            raise ValueError("SymbolReference: name must be non-empty")
        for interface in frozen_interfaces:
            if interface not in _VALID_SYMBOL_INTERFACES:
                raise ValueError(
                    f"SymbolReference '{name}': invalid interface "
                    f"'{interface}', must be one of "
                    f"{sorted(_VALID_SYMBOL_INTERFACES)}"
                )
        if not isinstance(role, SymbolReferenceRole):
            raise ValueError(
                f"SymbolReference: role must be a SymbolReferenceRole, got {role!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "interfaces", frozen_interfaces)
        object.__setattr__(self, "role", role)


@unique
class SymbolDefinitionFlag(Enum):
    """Generic roles carried by a symbol-defining operation."""

    DECLARATION = "declaration"
    TEST_ONLY = "test_only"


@dataclass(frozen=True, slots=True)
class SymbolValueContract:
    """Declares the typed-value contract carried by a symbol definition.

    result: Result field whose type is part of the symbol contract.
    value: Optional attribute containing an exact definition value.
    predicates: Optional predicate-list attribute constraining a partial value.

    The generated indices let generic link code compare and merge typed symbol
    contracts without switching on the concrete defining operation.
    """

    result: str
    value: str | None = None
    predicates: str | None = None

    def __post_init__(self) -> None:
        if not self.result:
            raise ValueError("SymbolValueContract: result must be non-empty")
        if self.value is not None and not self.value:
            raise ValueError("SymbolValueContract: value must be non-empty")
        if self.predicates is not None and not self.predicates:
            raise ValueError("SymbolValueContract: predicates must be non-empty")


@dataclass(frozen=True, slots=True)
class SymbolKernelContract:
    """Declares where a kernel symbol stores its workload signature.

    Definitions carry the signature as entry arguments of a configuration
    region. Declarations carry it as one segmented operand field. Exactly one
    representation is present on every symbol implementing the structural
    kernel interface. The launch ABI remains the ordinary FuncLike signature.
    """

    workload_region: str | None = None
    workload_operands: str | None = None

    def __post_init__(self) -> None:
        if (self.workload_region is None) == (self.workload_operands is None):
            raise ValueError(
                "SymbolKernelContract: exactly one of workload_region or "
                "workload_operands must be present"
            )
        if self.workload_region == "":
            raise ValueError("SymbolKernelContract: workload_region must be non-empty")
        if self.workload_operands == "":
            raise ValueError(
                "SymbolKernelContract: workload_operands must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class SymbolDefinition:
    """Declares how an op defines a module symbol.

    field: Symbol attr on the op that carries the symbol's identity.
    name: Human-readable symbol class used in diagnostics.
    interfaces: Structural interfaces implemented by the symbol.
    bytecode_kind: Existing bytecode payload kind for symbols that still
        serialize through the current symbol section. New symbol families can
        leave this as LOOM_SYMBOL_NONE while still participating in IR symbol
        lookup and verification.
    fact_domain: Optional C symbol for the dialect-owned symbol fact domain.
    retain: Optional enum attribute that marks symbols which ordinary symbol
        DCE must preserve even when unreachable.
    flags: Generic roles that affect symbol processing independent of the
        defining dialect.
    value_contract: Optional typed-value contract described by generated field
        indices.
    kernel_contract: Optional kernel workload-signature storage contract.
    """

    field: str
    name: str
    interfaces: tuple[str, ...]
    bytecode_kind: str = "LOOM_SYMBOL_NONE"
    fact_domain: str | None = None
    retain: str | None = None
    flags: tuple[SymbolDefinitionFlag, ...] = ()
    value_contract: SymbolValueContract | None = None
    kernel_contract: SymbolKernelContract | None = None

    def __init__(
        self,
        *,
        field: str,
        name: str,
        interfaces: list[str] | tuple[str, ...],
        bytecode_kind: str = "LOOM_SYMBOL_NONE",
        fact_domain: str | None = None,
        retain: str | None = None,
        flags: list[SymbolDefinitionFlag] | tuple[SymbolDefinitionFlag, ...] = (),
        value_contract: SymbolValueContract | None = None,
        kernel_contract: SymbolKernelContract | None = None,
    ) -> None:
        frozen_interfaces = tuple(interfaces)
        frozen_flags = tuple(flags)
        if not field:
            raise ValueError("SymbolDefinition: field must be non-empty")
        if not name:
            raise ValueError("SymbolDefinition: name must be non-empty")
        if retain is not None and not retain:
            raise ValueError("SymbolDefinition: retain must be non-empty")
        if not frozen_interfaces:
            raise ValueError("SymbolDefinition: interfaces must be non-empty")
        for interface in frozen_interfaces:
            if interface not in _VALID_SYMBOL_INTERFACES:
                raise ValueError(
                    f"SymbolDefinition '{name}': invalid interface "
                    f"'{interface}', must be one of "
                    f"{sorted(_VALID_SYMBOL_INTERFACES)}"
                )
        if len(frozen_flags) != len(set(frozen_flags)):
            raise ValueError(f"SymbolDefinition '{name}': flags must be unique")
        for flag in frozen_flags:
            if not isinstance(flag, SymbolDefinitionFlag):
                raise ValueError(
                    f"SymbolDefinition '{name}': invalid flag {flag!r}, "
                    "must be a SymbolDefinitionFlag"
                )
        implements_kernel = "kernel" in frozen_interfaces
        if implements_kernel != (kernel_contract is not None):
            raise ValueError(
                f"SymbolDefinition '{name}': the kernel interface and "
                "kernel_contract must be declared together"
            )
        if implements_kernel and "func_like" not in frozen_interfaces:
            raise ValueError(
                f"SymbolDefinition '{name}': the kernel interface requires "
                "the func_like interface for its launch ABI"
            )
        if "callable" in frozen_interfaces and "func_like" not in frozen_interfaces:
            raise ValueError(
                f"SymbolDefinition '{name}': the callable interface requires "
                "the func_like interface for its call ABI"
            )
        if (
            "command_program" in frozen_interfaces
            and "func_like" not in frozen_interfaces
        ):
            raise ValueError(
                f"SymbolDefinition '{name}': the command_program interface "
                "requires the func_like interface for its launch ABI"
            )
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "interfaces", frozen_interfaces)
        object.__setattr__(self, "bytecode_kind", bytecode_kind)
        object.__setattr__(self, "fact_domain", fact_domain)
        object.__setattr__(self, "retain", retain)
        object.__setattr__(self, "flags", frozen_flags)
        object.__setattr__(self, "value_contract", value_contract)
        object.__setattr__(self, "kernel_contract", kernel_contract)

    @property
    def is_declaration(self) -> bool:
        """Returns whether another definition may satisfy this declaration."""

        return SymbolDefinitionFlag.DECLARATION in self.flags

    @property
    def is_test_only(self) -> bool:
        """Returns whether the symbol exists only for tests or benchmarks."""

        return SymbolDefinitionFlag.TEST_ONLY in self.flags


@dataclass(frozen=True, slots=True)
class AttrDef:
    """A compile-time constant attribute on an op.

    name: Attribute name (key in the attribute dictionary or
          positional in the format spec).
    attr_type: The kind of attribute value. Must be one of the ATTR_TYPE_*
        constants. Scoped enums use a stable key at format boundaries and a
        dense ordinal interpreted by the enclosing representation contract in
        C IR. Parameterized attribute arrays and ordinary symbol arrays
        preserve order and repeated elements. Symbol sets are ordered by exact
        symbol-name bytes and contain no duplicates. All three collection kinds
        are valid only in descriptor-backed fields.
    doc: Human-readable description.
    default: Default value (None = required, not optional).
    enum_def: For enum and enum-array attrs, the EnumDef describing valid
        values. Required for both kinds.
    optional: If True, this attribute may be absent.
    elide_default: If True, inline AttrDict text printing omits this
        zero/false scalar attribute when its value equals default, and parsing
        restores the default when the attribute is omitted from that inline
        dictionary.
    open_enum: If True, generic verification preserves future raw enum
        ordinals and leaves selected/supported-case policy to the op verifier.
    bare_identifier: If True, string values use bare identifier spelling in
        descriptor-aware text formats instead of quoted string spelling.
    parameterized_attr: Optional exact family constraint for parameterized
        attributes or every element of a parameterized attribute array. None
        leaves the family open.
    symbol_ref: Target interface and graph-role contract for a symbol or every
        element of a symbol array or symbol set.
    """

    name: str
    attr_type: AttrType
    doc: str = ""
    default: Any = None
    enum_def: EnumDef | None = None
    optional: bool = False
    elide_default: bool = False
    symbol_ref: SymbolReference | None = None
    open_enum: bool = False
    parameterized_attr: ParameterizedAttrDef | None = None
    bare_identifier: bool = False

    def __post_init__(self) -> None:
        if self.attr_type not in _VALID_ATTR_TYPES:
            raise ValueError(
                f"AttrDef '{self.name}': invalid attr_type "
                f"'{self.attr_type}', must be one of {sorted(_VALID_ATTR_TYPES)}"
            )
        if (
            self.attr_type
            in (ATTR_TYPE_ENUM, ATTR_TYPE_ENUM_ARRAY, ATTR_TYPE_SIGNED_ENUM_SET)
            and self.enum_def is None
        ):
            raise ValueError(
                f"AttrDef '{self.name}': {self.attr_type!r} requires enum_def"
            )
        if self.attr_type == ATTR_TYPE_FLAGS and self.enum_def is None:
            raise ValueError(
                f"AttrDef '{self.name}': attr_type='flags' requires enum_def"
            )
        if self.open_enum and self.attr_type not in (
            ATTR_TYPE_ENUM,
            ATTR_TYPE_ENUM_ARRAY,
        ):
            raise ValueError(
                f"AttrDef '{self.name}': open_enum requires an enum attribute"
            )
        if self.attr_type == ATTR_TYPE_SCOPED_ENUM:
            if self.optional:
                raise ValueError(
                    f"AttrDef '{self.name}': scoped_enum attributes are required"
                )
            if self.default is not None:
                raise ValueError(
                    f"AttrDef '{self.name}': scoped_enum attributes cannot "
                    "have defaults"
                )
        if self.symbol_ref is not None and self.attr_type not in (
            ATTR_TYPE_SYMBOL,
            ATTR_TYPE_SYMBOL_ARRAY,
            ATTR_TYPE_SYMBOL_SET,
        ):
            raise ValueError(
                f"AttrDef '{self.name}': symbol_ref requires "
                "attr_type='symbol', 'symbol_array', or 'symbol_set'"
            )
        if (
            self.attr_type
            in (
                ATTR_TYPE_SYMBOL_ARRAY,
                ATTR_TYPE_SYMBOL_SET,
            )
            and self.symbol_ref is None
        ):
            raise ValueError(
                f"AttrDef '{self.name}': attr_type='{self.attr_type}' requires "
                "symbol_ref"
            )
        if self.parameterized_attr is not None and self.attr_type not in (
            ATTR_TYPE_PARAMETERIZED,
            ATTR_TYPE_PARAMETERIZED_ARRAY,
        ):
            raise ValueError(
                f"AttrDef '{self.name}': parameterized_attr requires "
                "attr_type='parameterized' or 'parameterized_array'"
            )
        if self.elide_default:
            if self.default is None:
                raise ValueError(
                    f"AttrDef '{self.name}': elide_default requires default"
                )
            if self.optional:
                raise ValueError(
                    f"AttrDef '{self.name}': elide_default attrs must be required"
                )
            if self.attr_type == ATTR_TYPE_I64:
                if isinstance(self.default, bool) or not isinstance(self.default, int):
                    raise ValueError(
                        f"AttrDef '{self.name}': i64 elide_default requires int default"
                    )
                if self.default != 0:
                    raise ValueError(
                        f"AttrDef '{self.name}': i64 elide_default currently "
                        "supports only zero defaults"
                    )
            elif self.attr_type == ATTR_TYPE_BOOL:
                if not isinstance(self.default, bool):
                    raise ValueError(
                        f"AttrDef '{self.name}': bool elide_default "
                        "requires bool default"
                    )
                if self.default:
                    raise ValueError(
                        f"AttrDef '{self.name}': bool elide_default currently "
                        "supports only false defaults"
                    )
            else:
                raise ValueError(
                    f"AttrDef '{self.name}': elide_default currently supports "
                    "only i64 and bool attrs"
                )
        if self.bare_identifier and self.attr_type != ATTR_TYPE_STRING:
            raise ValueError(
                f"AttrDef '{self.name}': bare_identifier requires attr_type='string'"
            )


@dataclass(frozen=True, slots=True)
class RegionDef:
    """A nested region on an op.

    name: Field name, used in format specs.
    doc: Human-readable description.
    single_block: If True, the region must have exactly one block.
    variadic: If True, this is a trailing zero-or-more region field.
    optional: If True, this trailing region may be absent.
    terminator: If set, explicit region terminators must have this op name,
        except that an explicit implicit-terminator op is also valid.
    implicit_args: Implicit block arguments created by the builder
        but not derived from operands. Each entry is (name, type_keyword)
        where type_keyword is a scalar type name (e.g., "index").
        These are prepended before any BindingList-derived args.
        Example: loop IV is ("iv", "index").
    arg_source: Optional variadic value field or FuncArgs field whose
        per-position types seed entry block arguments in generated builders.
        Text parsing gets concrete names and types from BindingList or
        BlockArgs format elements for ordinary value fields, or clones the
        FuncArgs signature names and types for projected FuncArgs regions.
    buffer_arg_memory_space: Optional target-independent memory-space fact to
        seed for buffer entry block arguments in this region. This refines
        region boundary facts without parameterizing the buffer type itself.
    arg_uniform_scope: Optional execution scope (currently "workgroup" or
        "cluster") over which scalar entry block arguments are identical.
        This is a region boundary contract rather than a property inferred
        from argument types.
    command_effects_only: If True, every directly observable effect executed
        in this region or a nested region must be carried by an operation with
        the CommandEffect trait. Pure computation and identity-producing
        operations remain valid.
    """

    name: str
    doc: str = ""
    single_block: bool = False
    variadic: bool = False
    optional: bool = False
    terminator: str | None = None
    implicit_args: tuple[tuple[str, str], ...] = ()
    arg_source: str | None = None
    buffer_arg_memory_space: str | None = None
    arg_uniform_scope: str | None = None
    command_effects_only: bool = False


# ============================================================================
# Enum support
# ============================================================================


@dataclass(frozen=True, slots=True)
class EnumCase:
    """A single case in an enum attribute.

    keyword: The textual keyword in assembly (e.g., "slt", "left").
    value: Integer value for binary encoding.
    doc: Human-readable description.
    """

    keyword: str
    value: int
    doc: str = ""


@dataclass(frozen=True, slots=True)
class EnumDef:
    """An enum attribute type with named cases.

    name: Enum type name (e.g., "CmpIPredicate", "BroadcastDir").
    cases: The valid cases.
    doc: Human-readable description.
    c_type: Optional externally defined C enum typedef used by generated
        accessors and builders instead of emitting a dialect-local enum.
    c_const_prefix: Optional prefix for cases in the external C enum.
    c_include: Optional header containing c_type.
    """

    name: str
    cases: tuple[EnumCase, ...]
    doc: str = ""
    c_type: str | None = None
    c_const_prefix: str | None = None
    c_include: str | None = None

    def __init__(
        self,
        name: str,
        cases: list[EnumCase] | tuple[EnumCase, ...],
        doc: str = "",
        *,
        c_type: str | None = None,
        c_const_prefix: str | None = None,
        c_include: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("EnumDef: name must be non-empty")
        object.__setattr__(self, "name", name)
        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise ValueError(f"EnumDef '{name}': cases must be non-empty")
        object.__setattr__(self, "cases", frozen_cases)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "c_type", c_type)
        object.__setattr__(self, "c_const_prefix", c_const_prefix)
        object.__setattr__(self, "c_include", c_include)
        if (c_type is None) != (c_const_prefix is None):
            raise ValueError(
                f"EnumDef '{name}': c_type and c_const_prefix must be provided together"
            )
        if c_include is not None and c_type is None:
            raise ValueError(f"EnumDef '{name}': c_include requires c_type")
        # Validate no duplicate keywords or values.
        seen_keywords: set[str] = set()
        seen_values: set[int] = set()
        for case in frozen_cases:
            if not case.keyword:
                raise ValueError(f"EnumDef '{name}': case keyword must be non-empty")
            if type(case.value) is not int or not 0 <= case.value <= 0xFF:
                raise ValueError(
                    f"EnumDef '{name}': case '{case.keyword}' value must be "
                    f"an integer in [0, 255], got {case.value!r}"
                )
            if case.keyword in seen_keywords:
                raise ValueError(
                    f"EnumDef '{name}': duplicate keyword '{case.keyword}'"
                )
            if case.value in seen_values:
                raise ValueError(f"EnumDef '{name}': duplicate value {case.value}")
            seen_keywords.add(case.keyword)
            seen_values.add(case.value)

    @property
    def keywords(self) -> tuple[str, ...]:
        """All valid keyword strings."""
        return tuple(c.keyword for c in self.cases)

    def case(self, keyword: str) -> EnumCase:
        """Returns the case with the given textual keyword."""
        for case in self.cases:
            if case.keyword == keyword:
                return case
        raise ValueError(f"EnumDef '{self.name}': unknown keyword '{keyword}'")


# ============================================================================
# Traits
# ============================================================================


@dataclass(frozen=True, slots=True)
class Trait:
    """A structural property of an operation.

    Traits describe invariants that the verifier checks and that
    optimizations may exploit. Simple traits are singletons (PURE,
    COMMUTATIVE). Parameterized traits are constructed with helper
    functions (AllTypesMatch, HasParent).

    name: Trait identifier.
    args: Optional arguments (field names, op names, etc.).
    """

    name: str
    args: tuple[str, ...] = ()

    def __init__(self, name: str, *args: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", args)

    def __repr__(self) -> str:
        if self.args:
            return f"{self.name}({', '.join(self.args)})"
        return self.name


# Standard singleton traits.
PURE = Trait("Pure")
COMMUTATIVE = Trait("Commutative")
IDEMPOTENT = Trait("Idempotent")
INVOLUTION = Trait("Involution")
TERMINATOR = Trait("Terminator")
CONSTANT_LIKE = Trait("ConstantLike")
POISON = Trait("Poison")
ELEMENTWISE = Trait("Elementwise")
DECOMPOSABLE = Trait("Decomposable")
SYMBOL_DEFINE = Trait("SymbolDefine")
# Op is valid only as a direct child of the module body. Module-owned operations
# use this independently from SymbolDefine so they do not enter the symbol table
# or pretend to own a symbol identity.
MODULE_SCOPE = Trait("ModuleScope")
# Op's regions cannot reference values from the enclosing scope.
# Values enter the region only through block arguments. Passes must
# not substitute inner values with outer definitions.
ISOLATED_FROM_ABOVE = Trait("IsolatedFromAbove")
# Same inputs may produce different outputs (RNG, timestamps). Prevents
# CSE and LICM but not DCE — unused non-deterministic results are dead
# as long as the op has no write effects.
NON_DETERMINISTIC = Trait("NonDeterministic")
# Effects depend on runtime state (e.g., func.call depends on the callee).
# Passes treat this conservatively as both READS_MEMORY and WRITES_MEMORY.
UNKNOWN_EFFECTS = Trait("UnknownEffects")
# Op represents an explicit command-program effect such as dispatch or schedule
# composition. The op must independently declare its exact or unknown effects;
# this trait classifies those effects instead of replacing them.
COMMAND_EFFECT = Trait("CommandEffect")
# Op orders memory accesses without directly reading or writing a resource.
# Fences are observable side effects but do not alias every memory operand;
# analyses preserve their ordering contract independently from footprint
# interference.
MEMORY_FENCE = Trait("MemoryFence")
# Execution depends on the dynamic set of participating invocations. This is
# independent of memory effects: a convergent op may still be pure, read/write
# memory, or have unknown effects, but generic optimizers must not erase,
# duplicate, speculate, or move it across convergence-changing boundaries.
CONVERGENT = Trait("Convergent")
# Each execution produces a result with a distinct identity, even when
# operands and attributes are identical. Prevents CSE but allows DCE
# (unused identity with no write effects is dead) and LICM. Derived
# automatically when any result has allocates=True, but can also be
# declared explicitly.
UNIQUE_IDENTITY = Trait("UniqueIdentity")
# Compiler hint with no semantic memory effects. Hint ops are preserved by
# canonicalization/DCE and removed only by an explicit hint-stripping pass.
HINT = Trait("Hint")
# Op contributes compiler facts or scheduling metadata but has no runtime
# representation once it reaches target-low emission. This trait does not
# itself preserve the op or impose source ordering. HINT already carries this
# property; this trait permits value-carrying fact identities to declare it.
COMPILE_TIME_ONLY = Trait("CompileTimeOnly")
# Op execution may be moved to a program point where it executes more often
# than in the source IR. This is stronger than Pure: the op must not trap,
# allocate a distinct identity, read runtime state, write memory, or rely on
# being control-dependent. It is used for branch reduction and predication
# rewrites, not for ordinary common-tail motion where every path already
# executes an equivalent op exactly once.
SAFE_TO_SPECULATE = Trait("SafeToSpeculate")
# Result types carry op-owned SSA references. Local canonicalization may retarget
# those references after proving the result has no type-sensitive users.
REFINABLE_RESULT_TYPE_REFS = Trait("RefinableResultTypeRefs")
# Op observes poison operands at a semantic boundary where poison can no longer
# propagate as an ordinary SSA value.
POISON_BOUNDARY = Trait("PoisonBoundary")
# Op refines facts or static type information while preserving operand/result
# value identity. Lowering can alias each result to the lowered operand at the
# same ordinal.
FACT_IDENTITY = Trait("FactIdentity")
# Op fact inference defines target-independent uniform/lane-varying distribution
# for its result when enough input distribution facts are available. Targets may
# use those distribution facts for placement instead of rediscovering the same
# property from producer structure.
DISTRIBUTION_TRANSFER = Trait("DistributionTransfer")
# Op creates storage relations between SSA values. Passes use this as a hot-path
# guard before asking dialect-owned helpers for exact relation rows.
STORAGE_RELATION = Trait("StorageRelation")
# Op attaches metadata or facts to operand 0 and produces one result that
# aliases the same physical value. Extra operands are interpretation data and do
# not force storage materialization.
VALUE_ALIAS = Trait("ValueAlias")


# ============================================================================
# Semantic phase and target-contract metadata
# ============================================================================


@unique
class OpPhase(Enum):
    """Semantic placement phase for an operation kind.

    This is source IR meaning, not a specific target's legality policy. Target
    stages derive their accepted/rejected classes from this placement plus
    declared target-contract families.
    """

    EXECUTABLE = "LOOM_OP_PHASE_EXECUTABLE"
    SOURCE_STRUCTURE = "LOOM_OP_PHASE_SOURCE_STRUCTURE"
    MODULE_METADATA = "LOOM_OP_PHASE_MODULE_METADATA"
    COMPILE_TIME_QUERY = "LOOM_OP_PHASE_COMPILE_TIME_QUERY"

    @property
    def c_name(self) -> str:
        return str(self.value)


@unique
class ContractFamily(Enum):
    """Semantic contract family that target packages may need to implement."""

    VECTOR_COORDINATE = (
        "vector.coordinate",
        "LOOM_CONTRACT_VECTOR_COORDINATE",
        "vector coordinate materialization",
    )
    REGISTER_PERMUTATION = (
        "register.permutation",
        "LOOM_CONTRACT_REGISTER_PERMUTATION",
        "register permutation",
    )
    VECTOR_TABLE_LOOKUP = (
        "vector.table_lookup",
        "LOOM_CONTRACT_VECTOR_TABLE_LOOKUP",
        "vector table lookup",
    )
    VECTOR_CONTRACTION = (
        "vector.contraction",
        "LOOM_CONTRACT_VECTOR_CONTRACTION",
        "vector contraction",
    )
    MEMORY_ATOMIC = (
        "memory.atomic",
        "LOOM_CONTRACT_MEMORY_ATOMIC",
        "atomic memory access",
    )
    KERNEL_ASYNC = (
        "kernel.async",
        "LOOM_CONTRACT_KERNEL_ASYNC",
        "kernel async pipeline",
    )
    KERNEL_SYNCHRONIZATION = (
        "kernel.synchronization",
        "LOOM_CONTRACT_KERNEL_SYNCHRONIZATION",
        "kernel synchronization",
    )
    TENSOR_MEMORY = (
        "kernel.tensor_memory",
        "LOOM_CONTRACT_TENSOR_MEMORY",
        "tensor memory transfer",
    )
    SANITIZER_RACE = (
        "sanitizer.race",
        "LOOM_CONTRACT_SANITIZER_RACE",
        "sanitizer race observation",
    )

    def __init__(self, key: str, c_name: str, diagnostic_name: str) -> None:
        self.key = key
        self.c_name = c_name
        self.diagnostic_name = diagnostic_name


@unique
class ConditionRefinementTruth(Enum):
    """Control-flow edges on which an op can refine one of its operands."""

    TRUE = "true"
    FALSE = "false"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ConditionRefinement:
    """Dialect-owned operand refinement implied by a boolean result.

    Generated metadata names the source operand by ordinal and points at a
    dialect callback that materializes its edge-local fact identity. Generic
    control-flow transforms invoke the callback only when the source is used on
    an edge whose truth value is listed here.
    """

    source: str
    truth: ConditionRefinementTruth
    materialize: str


def _validate_metadata_key(kind: str, key: str) -> None:
    if not key:
        raise ValueError(f"{kind} key must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(char not in allowed for char in key):
        raise ValueError(
            f"{kind} key {key!r} must contain only lowercase letters, "
            "digits, '.', '_', or '-'"
        )


@dataclass(frozen=True, slots=True)
class OpCategory:
    """Stable category for grouping related ops inside a dialect.

    Categories are generator-facing metadata. They give large dialects a durable
    semantic shard boundary so generated tables can be split without deriving
    file layout from incidental source order.
    """

    key: str
    doc: str = ""

    def __init__(self, key: str, *, doc: str = "") -> None:
        _validate_metadata_key("op category", key)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "doc", doc)


@unique
class TypeSemantic(Enum):
    """Semantic role for non-scalar type declarations."""

    ORDINARY = "LOOM_TYPE_SEMANTIC_ORDINARY"
    CONTROL_TOKEN = "LOOM_TYPE_SEMANTIC_CONTROL_TOKEN"
    TARGET_CONTRACT_VALUE = "LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE"
    MANAGED_REFERENCE = "LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE"

    @property
    def c_name(self) -> str:
        return str(self.value)


# ============================================================================
# Memory effects
# ============================================================================


@unique
class EffectKind(Enum):
    """Kind of memory effect on a resource-typed operand."""

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


@dataclass(frozen=True, slots=True)
class Effect:
    """Declares a memory effect on a specific operand.

    operand: Name of the resource-typed operand being accessed.
    kind: What the op does to the resource (read, write, or both).
    """

    operand: str
    kind: EffectKind


def Reads(operand: str) -> Effect:
    """Op reads from the named resource operand."""
    return Effect(operand, EffectKind.READ)


def Writes(operand: str) -> Effect:
    """Op writes to the named resource operand."""
    return Effect(operand, EffectKind.WRITE)


def ReadWrites(operand: str) -> Effect:
    """Op performs an atomic read-modify-write on the named resource operand."""
    return Effect(operand, EffectKind.READWRITE)


# ============================================================================
# Ownership effects
# ============================================================================


@unique
class OwnershipCarrier(Enum):
    """How an operand carries a managed resource into an operation."""

    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


BY_VALUE = OwnershipCarrier.BY_VALUE
BY_REFERENCE = OwnershipCarrier.BY_REFERENCE


@unique
class OperandOwnershipEffectKind(Enum):
    """Ownership action applied to an operand field."""

    BORROW = "borrow"
    CONSUME = "consume"
    RETAIN = "retain"
    RELEASE = "release"
    DISCARD = "discard"
    ESCAPE = "escape"


@unique
class ResultOwnershipEffectKind(Enum):
    """Ownership action applied to a result field."""

    FRESH = "fresh"
    BORROWED = "borrowed"
    RETAINED = "retained"
    ALIAS = "alias"
    MOVED = "moved"


@dataclass(frozen=True, slots=True)
class OperandOwnershipEffect:
    """Declares an ownership action on an operand field.

    operand: Name of the operand field.
    kind: Ownership action applied to each value in the field.
    carrier: Whether the field carries the resource by value or by reference.
    """

    operand: str
    kind: OperandOwnershipEffectKind
    carrier: OwnershipCarrier = BY_VALUE


@dataclass(frozen=True, slots=True)
class ResultOwnershipEffect:
    """Declares an ownership action on a result field.

    result: Name of the result field.
    kind: Ownership action applied to each value in the field.
    source: Operand field used by aliasing and moved result effects.
    """

    result: str
    kind: ResultOwnershipEffectKind
    source: str | None = None


def Borrow(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operand is observed without transferring ownership."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.BORROW, carrier)


def Consume(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operand ownership transfers into the operation."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.CONSUME, carrier)


def Retain(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operation retains the operand resource."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.RETAIN, carrier)


def Release(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operation releases one owned reference to the operand resource."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.RELEASE, carrier)


def Discard(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operation drops compiler ownership without emitting a release."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.DISCARD, carrier)


def Escape(
    operand: str, carrier: OwnershipCarrier = BY_VALUE
) -> OperandOwnershipEffect:
    """Operand resource escapes to an untracked owner."""
    return OperandOwnershipEffect(operand, OperandOwnershipEffectKind.ESCAPE, carrier)


def FreshResult(result: str) -> ResultOwnershipEffect:
    """Result creates a fresh owned resource."""
    return ResultOwnershipEffect(result, ResultOwnershipEffectKind.FRESH)


def BorrowedResult(result: str) -> ResultOwnershipEffect:
    """Result borrows a resource owned elsewhere."""
    return ResultOwnershipEffect(result, ResultOwnershipEffectKind.BORROWED)


def RetainedResult(result: str) -> ResultOwnershipEffect:
    """Result is an owned retained reference to an existing resource."""
    return ResultOwnershipEffect(result, ResultOwnershipEffectKind.RETAINED)


def AliasResult(result: str, source: str) -> ResultOwnershipEffect:
    """Result aliases an operand resource without consuming it."""
    return ResultOwnershipEffect(result, ResultOwnershipEffectKind.ALIAS, source)


def MovedResult(result: str, source: str) -> ResultOwnershipEffect:
    """Result receives the exact ownership state of a consumed operand."""
    return ResultOwnershipEffect(result, ResultOwnershipEffectKind.MOVED, source)


# Type constraints that represent mutable resources (as opposed to
# pure SSA values). Effects may only reference operands with one of
# these constraints.
_RESOURCE_TYPE_CONSTRAINTS = frozenset(
    {
        TypeConstraint.POOL,
        TypeConstraint.BUFFER,
        TypeConstraint.VIEW,
        TypeConstraint.TENSOR,
        TypeConstraint.ANY,
    }
)


# Parameterized trait constructors.
def AllTypesMatch(*fields: str) -> Trait:
    """All named fields must have identical types."""
    return Trait("AllTypesMatch", *fields)


def KeyedModuleRecord(key_attr: str) -> Trait:
    """Declares attr-only module metadata canonically ordered by one key."""

    return Trait("KeyedModuleRecord", key_attr)


def HasAncestor(op_name: str) -> Trait:
    """This op must be nested under the named op at any depth."""
    return Trait("HasAncestor", op_name)


def HasParent(op_name: str) -> Trait:
    """This op must be directly nested inside the named op."""
    return Trait("HasParent", op_name)


def NoAncestor(op_name: str) -> Trait:
    """This op must not be nested under the named op at any depth."""
    return Trait("NoAncestor", op_name)


def ImplicitTerminator(op_name: str) -> Trait:
    """Regions of this op have an implicit terminator of the given type."""
    return Trait("ImplicitTerminator", op_name)


# ============================================================================
# Constraints
# ============================================================================


# Validate signature: receives a dict mapping field names to values,
# returns (ok, message). Used by the Python validator/oracle.
type ValidateFn = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class Constraint:
    """A verification constraint on an op's fields.

    Constraints express relationships between operands, results, and
    attributes that the verifier checks. Each constraint has:

      name: Identifier for code generation and diagnostics.
      args: Field names this constraint references.
      error: Structured error definition emitted on failure.
      validate: Optional Python predicate for the oracle/validator.
      data: Optional small payload interpreted by the named constraint.

    Constraints are defined as module-level constructor functions
    (SameType, RanksMatch, etc.) that return Constraint instances
    with appropriate validation closures.
    """

    name: str
    args: tuple[str, ...]
    error: ErrorDef | None = None
    validate: ValidateFn | None = None
    data: int | TypeConstraint | None = None

    def __init__(
        self,
        name: str,
        args: tuple[str, ...],
        error: ErrorDef | None = None,
        validate: ValidateFn | None = None,
        data: int | TypeConstraint | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "validate", validate)
        object.__setattr__(self, "data", data)

    def check(self, fields: dict[str, Any]) -> tuple[bool, str]:
        """Run the validation predicate. Returns (ok, message)."""
        if self.validate is None:
            return (True, "")
        return self.validate(fields)

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


# --- Type and shape constraints ---


def _flatten_field(name: str, value: Any) -> list[tuple[str, Any]]:
    """Expand a field value to (display_name, item) pairs.

    For scalar fields (a single Value), returns [(name, value)].
    For variadic fields (a list of Values), returns
    [(name[0], items[0]), (name[1], items[1]), ...].

    None values and empty lists produce an empty result.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [(f"{name}[{i}]", item) for i, item in enumerate(value)]
    return [(name, value)]


def _field_value_type(item: Any) -> Any:
    """Returns the type carried by a field item, or the item if it is a type."""
    return item.type if hasattr(item, "type") else item


def _field_value_kind(item: Any) -> Any:
    """Returns a type-kind-like discriminator for a field item."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "type_kind"):
        return value_type.type_kind
    if hasattr(value_type, "kind"):
        return value_type.kind
    return None


def _field_element_type(item: Any) -> Any:
    """Returns the scalar element type carried by a scalar or shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "element_type"):
        return value_type.element_type
    if hasattr(value_type, "dtype"):
        return value_type.dtype
    if hasattr(value_type, "type_kind") and hasattr(value_type, "kind"):
        return value_type
    return None


def _field_register_class(item: Any) -> Any:
    """Returns the register class carried by a register field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "descriptor_set_stable_id") and hasattr(
        value_type, "register_class_id"
    ):
        return (value_type.descriptor_set_stable_id, value_type.register_class_id)
    return None


def _field_register_unit_count(item: Any) -> Any:
    """Returns the register unit count carried by a register field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "unit_count"):
        return value_type.unit_count
    return None


def _field_shape(item: Any) -> Any:
    """Returns the shape carried by a shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "dims"):
        return value_type.dims
    if hasattr(value_type, "shape"):
        return value_type.shape
    return None


def _field_rank(item: Any) -> Any:
    """Returns the rank carried by a shaped field."""
    value_type = _field_value_type(item)
    if hasattr(value_type, "rank"):
        return value_type.rank
    if hasattr(value_type, "ndim"):
        return value_type.ndim
    return None


def SameType(*fields: str) -> Constraint:
    """All named fields must have identical types.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        types = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                if hasattr(item, "type"):
                    types.append((display_name, item.type))
        if len(types) < 2:
            return (True, "")
        first_name, first_type = types[0]
        for entry_name, value_type in types[1:]:
            if value_type != first_type:
                return (
                    False,
                    f"'{entry_name}' type {value_type}"
                    f" != '{first_name}' type {first_type}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_001

    return Constraint(
        "SameType",
        fields,
        error=ERR_TYPE_001,
        validate=_validate,
    )


def SameKind(*fields: str) -> Constraint:
    """All named fields must have the same type kind."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        kinds = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                kind = _field_value_kind(item)
                if kind is not None:
                    kinds.append((display_name, kind))
        if len(kinds) < 2:
            return (True, "")
        first_name, first_kind = kinds[0]
        for entry_name, kind in kinds[1:]:
            if kind != first_kind:
                return (
                    False,
                    f"'{entry_name}' kind {kind} != '{first_name}' kind {first_kind}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_001

    return Constraint(
        "SameKind",
        fields,
        error=ERR_TYPE_001,
        validate=_validate,
    )


def SameRegisterClass(*fields: str) -> Constraint:
    """All named register fields must have the same register class.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        classes = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                reg_class = _field_register_class(item)
                if reg_class is not None:
                    classes.append((display_name, reg_class))
        if len(classes) < 2:
            return (True, "")
        first_name, first_class = classes[0]
        for entry_name, reg_class in classes[1:]:
            if reg_class != first_class:
                return (
                    False,
                    f"'{entry_name}' register class {reg_class}"
                    f" != '{first_name}' register class {first_class}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_001

    return Constraint(
        "SameRegisterClass",
        fields,
        error=ERR_TYPE_001,
        validate=_validate,
    )


def SameElementType(*fields: str) -> Constraint:
    """All named shaped fields must have the same element type.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        element_types: list[tuple[str, Any]] = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                element_type = _field_element_type(item)
                if element_type is not None:
                    element_types.append((display_name, element_type))
        if len(element_types) < 2:
            return (True, "")
        first_name, first_dtype = element_types[0]
        for entry_name, dtype in element_types[1:]:
            if dtype != first_dtype:
                return (
                    False,
                    f"'{entry_name}' element type {dtype} != "
                    f"'{first_name}' element type {first_dtype}",
                )
        return (True, "")

    from loom.error.type import ERR_TYPE_002

    return Constraint(
        "SameElementType",
        fields,
        error=ERR_TYPE_002,
        validate=_validate,
    )


def SameEncoding(*fields: str) -> Constraint:
    """All named fields must have the same encoding.

    Encodings are a type-system concept not present in numpy arrays.
    The Python validator is a no-op; the C verifier checks encoding
    attributes on the actual types.
    """
    from loom.error.encoding import ERR_ENCODING_001

    return Constraint(
        "SameEncoding",
        fields,
        error=ERR_ENCODING_001,
    )


def SameShape(*fields: str) -> Constraint:
    """All named shaped fields must have identical shapes.

    Handles variadic fields: if a field's value is a list, each
    element is checked individually against the others.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        shapes: list[tuple[str, Any]] = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                shape = _field_shape(item)
                if shape is not None:
                    shapes.append((display_name, shape))
        if len(shapes) < 2:
            return (True, "")
        first_name, first_shape = shapes[0]
        for entry_name, shape in shapes[1:]:
            if shape != first_shape:
                return (
                    False,
                    f"'{entry_name}' shape {shape}"
                    f" != '{first_name}' shape {first_shape}",
                )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_002

    return Constraint(
        "SameShape",
        fields,
        error=ERR_SHAPE_002,
        validate=_validate,
    )


def RanksMatch(a: str, b: str) -> Constraint:
    """Two shaped fields must have the same rank.

    Handles variadic fields: if either field's value is a list, each
    element is checked pairwise against all elements of the other field.
    """

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        items_a = [
            (display_name, rank)
            for display_name, item in _flatten_field(a, values.get(a))
            if (rank := _field_rank(item)) is not None
        ]
        items_b = [
            (display_name, rank)
            for display_name, item in _flatten_field(b, values.get(b))
            if (rank := _field_rank(item)) is not None
        ]
        if not items_a or not items_b:
            return (True, "")
        # Check all pairs: every item in a against every item in b.
        for name_a, rank_a in items_a:
            for name_b, rank_b in items_b:
                if rank_a != rank_b:
                    return (
                        False,
                        f"'{name_a}' rank {rank_a} != '{name_b}' rank {rank_b}",
                    )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_001

    return Constraint(
        "RanksMatch",
        (a, b),
        error=ERR_SHAPE_001,
        validate=_validate,
    )


def _type_satisfies_field_constraint(
    value_type: Any, constraint: TypeConstraint
) -> bool:
    from loom.ir import (
        RegisterType,
        ScalarType,
        ScalarTypeKind,
        ShapedType,
        StorageType,
        TypeKind,
    )

    if constraint == REGISTER:
        return isinstance(value_type, RegisterType)
    if constraint == STORAGE:
        return isinstance(value_type, StorageType)

    if constraint == RANK_ONE_VECTOR:
        return (
            isinstance(value_type, ShapedType)
            and value_type.type_kind == TypeKind.VECTOR
            and value_type.rank == 1
        )
    if constraint == ALL_STATIC_VECTOR:
        return (
            isinstance(value_type, ShapedType)
            and value_type.type_kind == TypeKind.VECTOR
            and value_type.is_all_static
        )
    if constraint == ALL_STATIC_RANK_ONE_VECTOR:
        return (
            isinstance(value_type, ShapedType)
            and value_type.type_kind == TypeKind.VECTOR
            and value_type.rank == 1
            and value_type.is_all_static
        )
    if constraint == INDEX_OR_NON_I1_INTEGER_SCALAR:
        if not isinstance(value_type, ScalarType):
            return False
        scalar_kind = value_type.kind
        return scalar_kind in {
            ScalarTypeKind.INDEX,
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
        }
    if constraint == BITWISE_SCALAR:
        if not isinstance(value_type, ScalarType):
            return False
        scalar_kind = value_type.kind
        return scalar_kind == ScalarTypeKind.INDEX or scalar_kind in {
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
            ScalarTypeKind.F8E4M3,
            ScalarTypeKind.F8E5M2,
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
            ScalarTypeKind.F32,
            ScalarTypeKind.F64,
        }
    if not isinstance(value_type, ShapedType):
        return False
    element_kind = value_type.element_type.kind
    if constraint == INDEX_OR_NON_I1_INTEGER_ELEMENT:
        return element_kind in {
            ScalarTypeKind.INDEX,
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
        }
    if constraint == BITWISE_ELEMENT:
        return element_kind == ScalarTypeKind.INDEX or element_kind in {
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
            ScalarTypeKind.F8E4M3,
            ScalarTypeKind.F8E5M2,
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
            ScalarTypeKind.F32,
            ScalarTypeKind.F64,
        }
    if constraint == INTEGER_ELEMENT:
        return element_kind in {
            ScalarTypeKind.I1,
            ScalarTypeKind.I8,
            ScalarTypeKind.I16,
            ScalarTypeKind.I32,
            ScalarTypeKind.I64,
        }
    if constraint == FLOAT_ELEMENT:
        return element_kind in {
            ScalarTypeKind.F8E4M3,
            ScalarTypeKind.F8E5M2,
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
            ScalarTypeKind.F32,
            ScalarTypeKind.F64,
        }
    if constraint == I1_ELEMENT:
        return element_kind == ScalarTypeKind.I1
    if constraint == I8_ELEMENT:
        return element_kind == ScalarTypeKind.I8
    if constraint == I32_ELEMENT:
        return element_kind == ScalarTypeKind.I32
    if constraint == F16_OR_BF16_ELEMENT:
        return element_kind in {
            ScalarTypeKind.F16,
            ScalarTypeKind.BF16,
        }
    if constraint == F32_ELEMENT:
        return element_kind == ScalarTypeKind.F32
    return False


def _has_field_constraint(name: str, constraint: TypeConstraint) -> Constraint:
    """A field's type must satisfy an abstract type constraint."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        for display_name, item in _flatten_field(name, values.get(name)):
            value_type = _field_value_type(item)
            if _type_satisfies_field_constraint(value_type, constraint):
                continue
            return (
                False,
                f"'{display_name}' type {value_type} does not satisfy "
                f"{constraint.value}",
            )
        return (True, "")

    from loom.error.type import ERR_TYPE_003

    return Constraint(
        f"Has{constraint.name.title().replace('_', '')}",
        (name,),
        error=ERR_TYPE_003,
        validate=_validate,
    )


def _has_element_constraint(name: str, constraint: TypeConstraint) -> Constraint:
    """A shaped field's element type must satisfy a family constraint."""

    return _has_field_constraint(name, constraint)


def HasIntegerElement(field: str) -> Constraint:
    """A shaped field must have an integer element type."""

    return _has_element_constraint(field, INTEGER_ELEMENT)


def HasFloatElement(field: str) -> Constraint:
    """A shaped field must have a floating-point element type."""

    return _has_element_constraint(field, FLOAT_ELEMENT)


def HasBitwiseScalar(field: str) -> Constraint:
    """A field must have a scalar type with a non-i1 bitwise payload."""

    return _has_field_constraint(field, BITWISE_SCALAR)


def HasBitwiseElement(field: str) -> Constraint:
    """A shaped field must have an element type with a non-i1 bitwise payload."""

    return _has_element_constraint(field, BITWISE_ELEMENT)


def HasIndexOrNonI1IntegerScalar(field: str) -> Constraint:
    """A field must have an index or non-i1 integer scalar type."""

    return _has_field_constraint(field, INDEX_OR_NON_I1_INTEGER_SCALAR)


def HasIndexOrNonI1IntegerElement(field: str) -> Constraint:
    """A shaped field must have an index or non-i1 integer element type."""

    return _has_element_constraint(field, INDEX_OR_NON_I1_INTEGER_ELEMENT)


def HasI1Element(field: str) -> Constraint:
    """A shaped field must have an i1 element type."""

    return _has_element_constraint(field, I1_ELEMENT)


def HasI8Element(field: str) -> Constraint:
    """A shaped field must have an i8 element type."""

    return _has_element_constraint(field, I8_ELEMENT)


def HasI32Element(field: str) -> Constraint:
    """A shaped field must have an i32 element type."""

    return _has_element_constraint(field, I32_ELEMENT)


def HasF16OrBf16Element(field: str) -> Constraint:
    """A shaped field must have an f16 or bf16 element type."""

    return _has_element_constraint(field, F16_OR_BF16_ELEMENT)


def HasF32Element(field: str) -> Constraint:
    """A shaped field must have an f32 element type."""

    return _has_element_constraint(field, F32_ELEMENT)


def HasRegister(field: str) -> Constraint:
    """A field must have a target-low register type."""

    return _has_field_constraint(field, REGISTER)


def HasRankOneVector(field: str) -> Constraint:
    """A field must have a vector type with rank 1."""

    return _has_field_constraint(field, RANK_ONE_VECTOR)


def HasAllStaticVector(field: str) -> Constraint:
    """A field must have a vector type with an all-static shape."""

    return _has_field_constraint(field, ALL_STATIC_VECTOR)


def HasAllStaticRankOneVector(field: str) -> Constraint:
    """A field must have a vector type with an all-static rank-1 shape."""

    return _has_field_constraint(field, ALL_STATIC_RANK_ONE_VECTOR)


def _field_element_bitwidth(item: Any) -> int | None:
    """Returns a scalar or shaped element bit width when statically known."""
    element_type = _field_element_type(item)
    if element_type is None:
        return None
    bitwidth = getattr(element_type, "bitwidth", None)
    return bitwidth if isinstance(bitwidth, int) and bitwidth > 0 else None


def _element_width_order(
    name: str,
    field: str,
    reference: str,
    *,
    greater_than: bool,
) -> Constraint:
    """A field's scalar/shaped element width must be ordered against another."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        field_value = values.get(field)
        reference_value = values.get(reference)
        if field_value is None or reference_value is None:
            return (True, "")
        field_width = _field_element_bitwidth(field_value)
        reference_width = _field_element_bitwidth(reference_value)
        if field_width is None or reference_width is None:
            return (True, "")
        if greater_than and field_width > reference_width:
            return (True, "")
        if not greater_than and field_width < reference_width:
            return (True, "")
        relation = ">" if greater_than else "<"
        return (
            False,
            f"'{field}' element bit width {field_width} is not {relation} "
            f"'{reference}' element bit width {reference_width}",
        )

    return Constraint(
        name,
        (field, reference),
        validate=_validate,
    )


def ElementWidthGreaterThan(field: str, reference: str) -> Constraint:
    """A field's scalar/shaped element bit width must be greater than another."""

    return _element_width_order(
        "ElementWidthGreaterThan",
        field,
        reference,
        greater_than=True,
    )


def ElementWidthLessThan(field: str, reference: str) -> Constraint:
    """A field's scalar/shaped element bit width must be less than another."""

    return _element_width_order(
        "ElementWidthLessThan",
        field,
        reference,
        greater_than=False,
    )


def _field_i64_attr_value(item: Any) -> int | None:
    """Returns an integer attribute-like value when statically known."""
    if isinstance(item, bool):
        return None
    return item if isinstance(item, int) else None


def ElementWidthAtLeastAttr(field: str, width_attr: str) -> Constraint:
    """A field's scalar/shaped element bit width must be at least an i64 attr."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        width = _field_i64_attr_value(values.get(width_attr))
        if width is None or width <= 0:
            return (True, "")
        element_width = _field_element_bitwidth(values.get(field))
        if element_width is None:
            return (True, "")
        if element_width >= width:
            return (True, "")
        return (
            False,
            f"'{field}' element bit width {element_width} is less than "
            f"'{width_attr}' value {width}",
        )

    return Constraint(
        "ElementWidthAtLeastAttr",
        (field, width_attr),
        validate=_validate,
    )


def BitRangeWithinElementWidth(
    field: str,
    offset_attr: str,
    width_attr: str,
) -> Constraint:
    """A bit offset/width attr pair must fit inside a field's element width."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        offset = _field_i64_attr_value(values.get(offset_attr))
        width = _field_i64_attr_value(values.get(width_attr))
        if offset is None or width is None:
            return (True, "")
        if offset < 0:
            return (False, f"'{offset_attr}' value {offset} is negative")
        if width <= 0:
            return (False, f"'{width_attr}' value {width} is not positive")
        element_width = _field_element_bitwidth(values.get(field))
        if element_width is None:
            return (True, "")
        if offset <= element_width and width <= element_width - offset:
            return (True, "")
        return (
            False,
            f"bit range [{offset}, {offset + width}) exceeds "
            f"'{field}' element bit width {element_width}",
        )

    return Constraint(
        "BitRangeWithinElementWidth",
        (field, offset_attr, width_attr),
        validate=_validate,
    )


def PositiveBitWidthAttr(attr: str) -> Constraint:
    """An i64 attribute spelling a bit width must be positive."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        value = _field_i64_attr_value(values.get(attr))
        if value is None or value > 0:
            return (True, "")
        return (False, f"'{attr}' value {value} is not a positive bit width")

    return Constraint(
        "PositiveBitWidthAttr",
        (attr,),
        validate=_validate,
    )


def _attr_matches_scalar_type(value: Any, element_type: Any) -> bool:
    """Returns true if a Python literal can represent a scalar element payload."""
    from loom.ir import ScalarType, ScalarTypeKind

    if not isinstance(element_type, ScalarType):
        return True
    element_kind = element_type.kind
    if element_kind == ScalarTypeKind.I1:
        return isinstance(value, bool) or (type(value) is int and value in (0, 1))
    if element_kind in {
        ScalarTypeKind.INDEX,
        ScalarTypeKind.OFFSET,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }:
        if type(value) is not int:
            return False
        if element_kind == ScalarTypeKind.OFFSET:
            return value >= 0
        if element_kind in {ScalarTypeKind.INDEX, ScalarTypeKind.I64}:
            return -(1 << 63) <= value <= (1 << 63) - 1
        bitwidth = element_type.bitwidth
        return -(1 << (bitwidth - 1)) <= value <= (1 << (bitwidth - 1)) - 1
    if element_kind in {
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    }:
        return type(value) is float
    return False


def AttrMatchesElementType(attr: str, field: str) -> Constraint:
    """An attribute literal must fit a field's scalar element type."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        value = values.get(attr)
        if value is None:
            return (True, "")
        element_type = _field_element_type(values.get(field))
        if element_type is None:
            return (True, "")
        if _attr_matches_scalar_type(value, element_type):
            return (True, "")
        return (
            False,
            f"'{attr}' literal {value!r} does not match "
            f"'{field}' element type {element_type}",
        )

    from loom.error.type import ERR_TYPE_005

    return Constraint(
        "AttrMatchesElementType",
        (attr, field),
        error=ERR_TYPE_005,
        validate=_validate,
    )


def LiteralMatchesElementType(literal: str, field: str) -> Constraint:
    """A literal payload must fit a field's scalar element type."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        value = values.get(literal)
        if value is None:
            return (True, "")
        element_type = _field_element_type(values.get(field))
        if element_type is None:
            return (True, "")
        if _attr_matches_scalar_type(value, element_type):
            return (True, "")
        return (
            False,
            f"'{literal}' literal {value!r} does not match "
            f"'{field}' element type {element_type}",
        )

    from loom.error.type import ERR_TYPE_005

    return Constraint(
        "LiteralMatchesElementType",
        (literal, field),
        error=ERR_TYPE_005,
        validate=_validate,
    )


def _static_element_count(item: Any) -> int | None:
    """Returns the static scalar/shaped element count, or None for dynamic."""
    from loom.ir import ScalarType, StaticDim

    value_type = _field_value_type(item)
    if isinstance(value_type, ScalarType):
        return 1
    dims = getattr(value_type, "dims", None)
    if dims is None:
        return None
    count = 1
    for dim in dims:
        if not isinstance(dim, StaticDim):
            return None
        count *= dim.size
    return count


def _static_total_bit_count(item: Any, bit_width_per_element: int) -> int | None:
    """Returns static element count times element width when both are known."""
    element_count = _static_element_count(item)
    if element_count is None or bit_width_per_element < 0:
        return None
    return element_count * bit_width_per_element


def _static_type_total_bit_count(item: Any) -> int | None:
    """Returns static shaped total bit count using the type element width."""
    element_width = _field_element_bitwidth(item)
    if element_width is None:
        return None
    return _static_total_bit_count(item, element_width)


def TotalBitCountEqual(lhs: str, rhs: str) -> Constraint:
    """Two shaped fields must have the same total bit count when provable."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        lhs_bit_count = _static_type_total_bit_count(values.get(lhs))
        rhs_bit_count = _static_type_total_bit_count(values.get(rhs))
        if lhs_bit_count is None or rhs_bit_count is None:
            return (True, "")
        if lhs_bit_count == rhs_bit_count:
            return (True, "")
        return (
            False,
            f"'{lhs}' total bit count {lhs_bit_count} != "
            f"'{rhs}' total bit count {rhs_bit_count}",
        )

    return Constraint(
        "TotalBitCountEqual",
        (lhs, rhs),
        validate=_validate,
    )


def _payload_bit_count_matches_storage(
    name: str,
    payload: str,
    width_attr: str,
    storage: str,
    diagnostic: str,
) -> Constraint:
    """Builds a static payload-width-times-count equals storage-count check."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        width = _field_i64_attr_value(values.get(width_attr))
        if width is None or width <= 0:
            return (True, "")
        payload_bit_count = _static_total_bit_count(values.get(payload), width)
        storage_bit_count = _static_type_total_bit_count(values.get(storage))
        if payload_bit_count is None or storage_bit_count is None:
            return (True, "")
        if payload_bit_count == storage_bit_count:
            return (True, "")
        return (
            False,
            f"'{payload}' payload bit count {payload_bit_count} != "
            f"'{storage}' storage bit count {storage_bit_count}",
        )

    return Constraint(
        name,
        (payload, width_attr, storage, diagnostic),
        validate=_validate,
    )


def PackedPayloadBitCountMatchesStorage(
    payload: str,
    width_attr: str,
    storage: str,
    diagnostic: str,
) -> Constraint:
    """Packed payload bit count must equal the static storage bit count."""

    return _payload_bit_count_matches_storage(
        "PackedPayloadBitCountMatchesStorage",
        payload,
        width_attr,
        storage,
        diagnostic,
    )


def UnpackedPayloadBitCountMatchesStorage(
    payload: str,
    width_attr: str,
    storage: str,
    diagnostic: str,
) -> Constraint:
    """Unpacked payload bit count must equal the static storage bit count."""

    return _payload_bit_count_matches_storage(
        "UnpackedPayloadBitCountMatchesStorage",
        payload,
        width_attr,
        storage,
        diagnostic,
    )


def OffsetCountMatchesRank(shaped: str, offsets: str) -> Constraint:
    """Offset count must equal the shaped field's rank."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        shaped_val = values.get(shaped)
        offsets_val = values.get(offsets, [])
        if shaped_val is None or not hasattr(shaped_val, "ndim"):
            return (True, "")
        if len(offsets_val) != shaped_val.ndim:
            return (
                False,
                f"'{shaped}' rank {shaped_val.ndim} != offset count {len(offsets_val)}",
            )
        return (True, "")

    from loom.error.subrange import ERR_SUBRANGE_001

    return Constraint(
        "OffsetCountMatchesRank",
        (shaped, offsets),
        error=ERR_SUBRANGE_001,
        validate=_validate,
    )


def ValueCountMatchesStaticElementCount(shaped: str, values: str) -> Constraint:
    """Value count must equal a shaped field's static element count."""

    def _validate(fields: dict[str, Any]) -> tuple[bool, str]:
        from loom.ir import DynamicDim, StaticDim

        value_type = _field_value_type(fields.get(shaped))
        items = fields.get(values, [])
        if value_type is None or not hasattr(value_type, "dims"):
            return (True, "")
        element_count = 1
        for dim in value_type.dims:
            if isinstance(dim, DynamicDim):
                return (True, "")
            if not isinstance(dim, StaticDim):
                return (True, "")
            element_count *= dim.size
        if len(items) == element_count:
            return (True, "")
        return (
            False,
            f"'{values}' count {len(items)} != "
            f"'{shaped}' static element count {element_count}",
        )

    from loom.error.structure import ERR_STRUCTURE_013

    return Constraint(
        "ValueCountMatchesStaticElementCount",
        (shaped, values),
        error=ERR_STRUCTURE_013,
        validate=_validate,
    )


def DimIndexInBounds(source: str, index: str) -> Constraint:
    """Dimension index must be in [0, rank) when statically known."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        source_val = values.get(source)
        index_val = values.get(index)
        if source_val is None or index_val is None:
            return (True, "")
        if not isinstance(index_val, int):
            return (True, "")
        if not hasattr(source_val, "ndim"):
            return (True, "")
        if index_val < 0 or index_val >= source_val.ndim:
            return (
                False,
                f"dimension index {index_val} out of bounds for rank {source_val.ndim}",
            )
        return (True, "")

    from loom.error.subrange import ERR_SUBRANGE_002

    return Constraint(
        "DimIndexInBounds",
        (source, index),
        error=ERR_SUBRANGE_002,
        validate=_validate,
    )


def AllShapesMatch(field: str) -> Constraint:
    """All values in a variadic field must have identical shapes."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        items = values.get(field, [])
        if len(items) < 2:
            return (True, "")
        if not hasattr(items[0], "shape"):
            return (True, "")
        first_shape = items[0].shape
        for i, item in enumerate(items[1:], 1):
            if item.shape != first_shape:
                return (
                    False,
                    f"input {i} shape {item.shape} != input 0 shape {first_shape}",
                )
        return (True, "")

    from loom.error.shape import ERR_SHAPE_003

    return Constraint(
        "AllShapesMatch",
        (field,),
        error=ERR_SHAPE_003,
        validate=_validate,
    )


def LastAxisGroupedBy(source: str, result: str, group_size: int) -> Constraint:
    """Result shape equals source shape with the last axis divided by group_size."""

    if group_size <= 0 or group_size > 255:
        raise ValueError("group_size must be in [1, 255]")

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        from loom.ir import DynamicDim, ShapedType, StaticDim, TypeKind

        source_value_type = _field_value_type(values.get(source))
        result_value_type = _field_value_type(values.get(result))
        if not isinstance(source_value_type, ShapedType) or not isinstance(
            result_value_type, ShapedType
        ):
            return (True, "")
        if (
            source_value_type.type_kind != TypeKind.VECTOR
            or result_value_type.type_kind != TypeKind.VECTOR
        ):
            return (True, "")
        if source_value_type.rank != result_value_type.rank:
            return (
                False,
                f"'{result}' rank {result_value_type.rank} != "
                f"'{source}' rank {source_value_type.rank}",
            )
        if source_value_type.rank == 0:
            return (True, "")
        leading_source_dims = source_value_type.dims[:-1]
        leading_result_dims = result_value_type.dims[:-1]
        if leading_source_dims != leading_result_dims:
            return (
                False,
                f"'{result}' leading shape {leading_result_dims} != "
                f"'{source}' leading shape {leading_source_dims}",
            )
        source_last_dim = source_value_type.dims[-1]
        result_last_dim = result_value_type.dims[-1]
        if isinstance(source_last_dim, DynamicDim):
            return (True, "")
        if not isinstance(source_last_dim, StaticDim):
            return (True, "")
        if source_last_dim.size % group_size != 0:
            return (
                False,
                f"'{source}' last axis {source_last_dim.size} is not divisible "
                f"by {group_size}",
            )
        if isinstance(result_last_dim, DynamicDim):
            return (True, "")
        if not isinstance(result_last_dim, StaticDim):
            return (True, "")
        expected_last_dim = source_last_dim.size // group_size
        if result_last_dim.size == expected_last_dim:
            return (True, "")
        return (
            False,
            f"'{result}' last axis {result_last_dim.size} != "
            f"'{source}' last axis {source_last_dim.size} divided by {group_size}",
        )

    return Constraint(
        "LastAxisGroupedBy",
        (source, result),
        data=group_size,
        validate=_validate,
    )


# --- Region constraints (structural, no-op in Python validator) ---


def BlockArgCount(region: str, inputs: str) -> Constraint:
    """Region block must have one argument per input or reference-region arg."""
    from loom.error.structure import ERR_STRUCTURE_007

    return Constraint(
        "BlockArgCount",
        (region, inputs),
        error=ERR_STRUCTURE_007,
    )


def BlockArgsSatisfy(region: str, constraint: TypeConstraint) -> Constraint:
    """Each region entry block argument must satisfy a type constraint."""
    from loom.error.type import ERR_TYPE_014

    return Constraint(
        "BlockArgsSatisfy",
        (region,),
        error=ERR_TYPE_014,
        data=constraint,
    )


def BlockArgsMatchTypes(region: str, inputs: str) -> Constraint:
    """Each block argument type must match its input or reference-region type."""
    from loom.error.type import ERR_TYPE_013

    return Constraint(
        "BlockArgsMatchTypes",
        (region, inputs),
        error=ERR_TYPE_013,
    )


def BlockArgsMatchElementTypes(region: str, inputs: str) -> Constraint:
    """Each block argument type must match its input element type."""
    from loom.error.type import ERR_TYPE_008

    return Constraint(
        "BlockArgsMatchElementTypes",
        (region, inputs),
        error=ERR_TYPE_008,
    )


def YieldCountMatchesResults(region: str, results: str) -> Constraint:
    """Region terminator must yield the same number of values as results."""
    from loom.error.structure import ERR_STRUCTURE_008

    return Constraint(
        "YieldCountMatchesResults",
        (region, results),
        error=ERR_STRUCTURE_008,
    )


def YieldTypesMatchResults(region: str, results: str) -> Constraint:
    """Yielded value types must match result types."""
    from loom.error.type import ERR_TYPE_009

    return Constraint(
        "YieldTypesMatchResults",
        (region, results),
        error=ERR_TYPE_009,
    )


def YieldElementTypesMatchResults(region: str, results: str) -> Constraint:
    """Yielded value element types must match result element types.

    For elementwise ops where the body operates at scalar granularity:
    the yield produces scalar values and the result is a shaped type.
    This constraint checks that the element types match, not the full
    types.
    """
    from loom.error.type import ERR_TYPE_009

    return Constraint(
        "YieldElementTypesMatchResults",
        (region, results),
        error=ERR_TYPE_009,
    )


def VariadicValuesMatch(lhs: str, rhs: str) -> Constraint:
    """Two variadic value fields must agree on count and per-position type."""
    from loom.error.structure import ERR_STRUCTURE_013

    return Constraint(
        "VariadicValuesMatch",
        (lhs, rhs),
        error=ERR_STRUCTURE_013,
    )


def IterArgsMatchResults(iter_args: str, results: str) -> Constraint:
    """Two variadic value fields must agree on count and per-position type.

    Used by ops that thread loop-carried state through a region (e.g.,
    scf.for): the iter_args operands provide initial values, the body
    yields the next iteration's values, and the results expose the
    final values. The yield-to-results match is enforced by
    YieldTypesMatchResults; this constraint enforces that iter_args
    and results agree directly so a count or type mismatch is reported
    on the loop op itself, not just the terminator.

    Both fields must be variadic value fields (operands or results).
    A count mismatch produces a single ERR_STRUCTURE_013; per-position
    type mismatches each produce an ERR_TYPE_001.
    """
    from loom.error.structure import ERR_STRUCTURE_013

    return Constraint(
        "IterArgsMatchResults",
        (iter_args, results),
        error=ERR_STRUCTURE_013,
    )


def RegisterUnitsSumTo(sources: str, result: str) -> Constraint:
    """Register unit counts in a variadic source field must sum to a result."""

    def _validate(values: dict[str, Any]) -> tuple[bool, str]:
        source_items = _flatten_field(sources, values.get(sources))
        result_items = _flatten_field(result, values.get(result))
        if len(result_items) != 1:
            return (True, "")
        unit_counts = []
        for _display_name, item in source_items:
            unit_count = _field_register_unit_count(item)
            if unit_count is not None:
                unit_counts.append(unit_count)
        result_unit_count = _field_register_unit_count(result_items[0][1])
        if result_unit_count is None:
            return (True, "")
        source_unit_count = sum(unit_counts)
        if source_unit_count == result_unit_count:
            return (True, "")
        return (
            False,
            f"'{sources}' register units sum to {source_unit_count}, "
            f"but '{result}' has {result_unit_count}",
        )

    from loom.error.type import ERR_TYPE_004

    return Constraint(
        "RegisterUnitsSumTo",
        (sources, result),
        error=ERR_TYPE_004,
        validate=_validate,
    )


# ============================================================================
# Op group
# ============================================================================


@dataclass(frozen=True, slots=True)
class Dialect:
    """A logical grouping of related operations.

    Corresponds to a namespace prefix in op names (scalar, tile, scf,
    func, etc.) and a subdirectory in the dialect file layout. Groups
    carry documentation, an optional dialect ID for bytecode encoding,
    and optionally shared enum definitions.

    Op names within a group share the group prefix: if the group is
    "scalar" and the op is "addi", the full op name is "scalar.addi".

    The dialect_id is a stable integer used in bytecode format to
    identify which dialect an op belongs to. Once assigned, it must
    never change. Unassigned groups use dialect_id=0.
    """

    name: str
    dialect_id: int
    doc: str = ""
    enums: tuple[EnumDef, ...] = ()
    default_phase: OpPhase | None = None
    categories: tuple[OpCategory, ...] = ()
    default_category: OpCategory | None = None
    c_path: str | None = None
    register_by_default: bool = True

    def __init__(
        self,
        name: str,
        *,
        dialect_id: int = 0,
        doc: str = "",
        enums: list[EnumDef] | tuple[EnumDef, ...] = (),
        default_phase: OpPhase | None = None,
        categories: list[OpCategory] | tuple[OpCategory, ...] = (),
        default_category: OpCategory | None = None,
        c_path: str | None = None,
        register_by_default: bool = True,
    ) -> None:
        frozen_categories = tuple(categories)
        if default_category is not None and default_category not in frozen_categories:
            raise ValueError(
                f"Dialect '{name}': default_category '{default_category.key}' "
                "is not declared in categories"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dialect_id", dialect_id)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "enums", tuple(enums))
        object.__setattr__(self, "default_phase", default_phase)
        object.__setattr__(self, "categories", frozen_categories)
        object.__setattr__(self, "default_category", default_category)
        object.__setattr__(self, "c_path", c_path)
        object.__setattr__(self, "register_by_default", register_by_default)


_DESCRIPTOR_PARAMETER_TYPES = frozenset(
    {
        ATTR_TYPE_I64,
        ATTR_TYPE_F64,
        ATTR_TYPE_STRING,
        ATTR_TYPE_BOOL,
        ATTR_TYPE_ENUM,
        ATTR_TYPE_ENUM_ARRAY,
        ATTR_TYPE_SIGNED_ENUM_SET,
        ATTR_TYPE_TYPE,
        ATTR_TYPE_I64_ARRAY,
        ATTR_TYPE_BYTES,
        ATTR_TYPE_ENCODING,
        ATTR_TYPE_SYMBOL,
        ATTR_TYPE_SYMBOL_ARRAY,
        ATTR_TYPE_DICT,
        ATTR_TYPE_PARAMETERIZED,
        ATTR_TYPE_PARAMETERIZED_ARRAY,
    }
)


def _is_ascii_identifier(value: str) -> bool:
    """Returns whether |value| is a bare Loom identifier."""

    if not value:
        return False
    first_character = value[0]
    if not (
        first_character == "_"
        or (first_character.isascii() and first_character.isalpha())
    ):
        return False
    return all(
        character.isascii() and (character.isalnum() or character == "_")
        for character in value[1:]
    )


def _is_ascii_qualified_identifier(value: str) -> bool:
    """Returns whether |value| is a dot-qualified Loom identifier."""

    return all(_is_ascii_identifier(part) for part in value.split("."))


def _validate_descriptor_parameters(
    owner: str, parameters: tuple[AttrDef, ...]
) -> None:
    """Validates one descriptor-backed parameter schema."""

    if len(parameters) > 0xFF:
        raise ValueError(
            f"{owner}: {len(parameters)} parameters exceed the uint8_t slot limit"
        )
    optional_parameter_count = sum(parameter.optional for parameter in parameters)
    if optional_parameter_count > 64:
        raise ValueError(
            f"{owner}: {optional_parameter_count} optional parameters exceed "
            "the uint64_t C builder flag limit"
        )

    seen_names: set[str] = set()
    for parameter in parameters:
        if not _is_ascii_identifier(parameter.name):
            raise ValueError(
                f"{owner}: parameter '{parameter.name}' must be a bare ASCII identifier"
            )
        if parameter.name in seen_names:
            raise ValueError(f"{owner}: duplicate parameter '{parameter.name}'")
        seen_names.add(parameter.name)
        if parameter.attr_type not in _DESCRIPTOR_PARAMETER_TYPES:
            raise ValueError(
                f"{owner}: parameter '{parameter.name}' has unsupported kind "
                f"'{parameter.attr_type}'"
            )


@dataclass(frozen=True, slots=True)
class ParameterizedAttrDef:
    """Declares one namespaced family of descriptor-backed attributes.

    The family name is stable public IR. Dense family ordinals and parameter
    slot positions are generated implementation details and never serialize.
    Parameters use AttrDef so operation fields, parameterized attributes, and
    generic parameterized types share one value schema. A family may designate
    one required parameter as its compact primary value; the parameter remains
    named everywhere except canonical text assembly, where it prints
    positionally. A family used as a static target-applicability clause may
    name its typed C condition descriptor.
    """

    name: str
    group: Dialect
    parameters: tuple[AttrDef, ...] = ()
    primary_parameter_index: int | None = None
    target_condition: str | None = None
    doc: str = ""

    def __init__(
        self,
        name: str,
        *,
        group: Dialect,
        parameters: list[AttrDef] | tuple[AttrDef, ...] = (),
        primary_parameter: str | None = None,
        target_condition: str | None = None,
        doc: str = "",
    ) -> None:
        name_parts = name.split(".")
        if len(name_parts) < 2 or name_parts[0] != group.name:
            raise ValueError(
                f"ParameterizedAttrDef '{name}': family name must begin with "
                f"the owning dialect namespace '{group.name}.'"
            )
        if not _is_ascii_qualified_identifier(name):
            raise ValueError(
                f"ParameterizedAttrDef '{name}': family name must be a dotted "
                "ASCII identifier"
            )

        frozen_parameters = tuple(parameters)
        _validate_descriptor_parameters(
            f"ParameterizedAttrDef '{name}'", frozen_parameters
        )
        primary_parameter_index = None
        if primary_parameter is not None:
            primary_parameter_index = next(
                (
                    index
                    for index, parameter in enumerate(frozen_parameters)
                    if parameter.name == primary_parameter
                ),
                None,
            )
            if primary_parameter_index is None:
                raise ValueError(
                    f"ParameterizedAttrDef '{name}': primary parameter "
                    f"'{primary_parameter}' is not declared"
                )
            if frozen_parameters[primary_parameter_index].optional:
                raise ValueError(
                    f"ParameterizedAttrDef '{name}': primary parameter "
                    f"'{primary_parameter}' must be required"
                )
        if target_condition is not None and not _is_ascii_identifier(target_condition):
            raise ValueError(
                f"ParameterizedAttrDef '{name}': target condition "
                f"'{target_condition}' must be a C symbol name"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "parameters", frozen_parameters)
        object.__setattr__(self, "primary_parameter_index", primary_parameter_index)
        object.__setattr__(self, "target_condition", target_condition)
        object.__setattr__(self, "doc", doc)

    @property
    def primary_parameter(self) -> AttrDef | None:
        """Returns the descriptor-declared compact positional parameter."""

        if self.primary_parameter_index is None:
            return None
        return self.parameters[self.primary_parameter_index]

    def __call__(self, **parameters: Any) -> Any:
        """Constructs one immutable value of this family."""

        from loom.ir import ParameterizedAttr

        return ParameterizedAttr(self, parameters)


@unique
class EncodingFamilyRole(Enum):
    """Semantic role of a registered static encoding family."""

    UNKNOWN = "LOOM_ENCODING_ROLE_UNKNOWN"
    ADDRESS_LAYOUT = "LOOM_ENCODING_ROLE_ADDRESS_LAYOUT"
    STORAGE_SCHEMA = "LOOM_ENCODING_ROLE_STORAGE_SCHEMA"
    PHYSICAL_STORAGE = "LOOM_ENCODING_ROLE_PHYSICAL_STORAGE"
    NUMERIC_TRANSFORM = "LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM"

    @property
    def c_name(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class EncodingRecordDef:
    """Exact physical geometry for one fixed encoding record."""

    logical_element_count: int
    storage_byte_count: int
    required_alignment: int = 1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("logical_element_count", self.logical_element_count),
            ("storage_byte_count", self.storage_byte_count),
            ("required_alignment", self.required_alignment),
        ):
            if type(value) is not int or not 1 <= value <= 0xFFFF:
                raise ValueError(
                    f"EncodingRecordDef: {field_name} must be an integer in "
                    f"[1, 65535], got {value!r}"
                )
        if self.required_alignment & (self.required_alignment - 1):
            raise ValueError(
                "EncodingRecordDef: required_alignment must be a power of two"
            )


@dataclass(frozen=True, slots=True)
class EncodingOperandSummaryDef:
    """Constant target-independent facts for one encoded operand schema."""

    element_format: int = 0
    scale_format: int = 0
    secondary_scale_format: int = 0
    payload_packing: int = 0
    scale_topology: int = 0
    affine_policy: int = 0
    rounding_policy: int = 0
    codebook_policy: int = 0
    sparsity_policy: int = 0
    zero_scale_fallback: bool = False
    sparsity_group_nonzero_element_count: int = 0
    sparsity_group_element_count: int = 0
    payload_register_count: int = 0
    payload_element_count: int = 0
    scale_group_element_count: int = 0
    scale_group_shape: tuple[int, ...] = ()
    scale_operand_count: int = 0

    def __post_init__(self) -> None:
        scale_group_shape = tuple(self.scale_group_shape)
        object.__setattr__(self, "scale_group_shape", scale_group_shape)
        for field_name in (
            "element_format",
            "scale_format",
            "secondary_scale_format",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(
                    f"EncodingOperandSummaryDef: {field_name} must be an "
                    f"unsigned 64-bit integer, got {value!r}"
                )
        for field_name in (
            "payload_packing",
            "scale_topology",
            "affine_policy",
            "rounding_policy",
            "codebook_policy",
            "sparsity_policy",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(
                    f"EncodingOperandSummaryDef: {field_name} must be an "
                    f"unsigned 32-bit integer, got {value!r}"
                )
        if type(self.zero_scale_fallback) is not bool:
            raise ValueError(
                "EncodingOperandSummaryDef: zero_scale_fallback must be a bool"
            )
        for field_name in (
            "sparsity_group_nonzero_element_count",
            "sparsity_group_element_count",
            "payload_register_count",
            "payload_element_count",
            "scale_group_element_count",
            "scale_operand_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= 0xFFFF:
                raise ValueError(
                    f"EncodingOperandSummaryDef: {field_name} must be an "
                    f"integer in [0, 65535], got {value!r}"
                )
        if len(scale_group_shape) > 4:
            raise ValueError(
                "EncodingOperandSummaryDef: scale_group_shape rank exceeds 4"
            )
        for extent in scale_group_shape:
            if type(extent) is not int or not 1 <= extent <= 0xFFFF:
                raise ValueError(
                    "EncodingOperandSummaryDef: scale_group_shape extents "
                    f"must be integers in [1, 65535], got {extent!r}"
                )
        if scale_group_shape:
            shape_element_count = math.prod(scale_group_shape)
            if shape_element_count > 0xFFFF:
                raise ValueError(
                    "EncodingOperandSummaryDef: scale_group_shape product exceeds 65535"
                )
            if self.scale_group_element_count not in (0, shape_element_count):
                raise ValueError(
                    "EncodingOperandSummaryDef: scale_group_element_count "
                    "must equal the scale_group_shape product"
                )
            object.__setattr__(self, "scale_group_element_count", shape_element_count)


@dataclass(frozen=True, slots=True)
class EncodingAliasDef:
    """Declares one canonical name for a parameterized encoding schema.

    Aliases are source spellings for a structural family instance, not
    independent encoding families. Fixed parameters establish identity and
    cannot be restated. Default parameters are expanded before module interning
    and may be overridden explicitly. Other family parameters remain available
    normally.
    """

    name: str
    fixed_parameters: tuple[tuple[str, Any], ...]
    default_parameters: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        name: str,
        *,
        fixed_parameters: Mapping[str, Any],
        default_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if not _is_ascii_qualified_identifier(name):
            raise ValueError(
                f"EncodingAliasDef '{name}': alias name must be a dotted "
                "ASCII identifier"
            )
        lexical_fixed_parameters = tuple(
            sorted(
                (parameter_name, _freeze_encoding_alias_parameter(value))
                for parameter_name, value in fixed_parameters.items()
            )
        )
        lexical_default_parameters = tuple(
            sorted(
                (parameter_name, _freeze_encoding_alias_parameter(value))
                for parameter_name, value in (default_parameters or {}).items()
            )
        )
        parameter_count = len(lexical_fixed_parameters) + len(
            lexical_default_parameters
        )
        if parameter_count > 0xFF:
            raise ValueError(
                f"EncodingAliasDef '{name}': {parameter_count} "
                "parameters exceed the uint8_t slot limit"
            )
        fixed_names = {name for name, _ in lexical_fixed_parameters}
        default_names = {name for name, _ in lexical_default_parameters}
        duplicate_names = fixed_names & default_names
        if duplicate_names:
            duplicate_name = min(duplicate_names)
            raise ValueError(
                f"EncodingAliasDef '{name}': parameter '{duplicate_name}' is "
                "both fixed and defaulted"
            )
        for parameter_name in fixed_names | default_names:
            if not _is_ascii_identifier(parameter_name):
                raise ValueError(
                    f"EncodingAliasDef '{name}': parameter '{parameter_name}' "
                    "must be a bare ASCII identifier"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fixed_parameters", lexical_fixed_parameters)
        object.__setattr__(self, "default_parameters", lexical_default_parameters)

    @property
    def parameters(self) -> tuple[tuple[str, Any], ...]:
        """Returns every contributed parameter in lexical order."""

        return tuple(sorted((*self.fixed_parameters, *self.default_parameters)))


def _freeze_encoding_alias_parameter(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze_encoding_alias_parameter(item) for item in value)
    return value


def _validate_encoding_alias_parameter(
    family_name: str,
    alias_name: str,
    descriptor: AttrDef,
    value: Any,
) -> None:
    """Validates one process-lifetime alias literal against its descriptor."""

    valid = False
    match descriptor.attr_type:
        case "i64":
            valid = type(value) is int and -(1 << 63) <= value < (1 << 63)
        case "f64":
            valid = type(value) in (int, float) and math.isfinite(float(value))
        case "bool":
            valid = type(value) is bool
        case "enum":
            valid = (
                type(value) is str
                and descriptor.enum_def is not None
                and value in descriptor.enum_def.keywords
            )
    if not valid:
        raise ValueError(
            f"EncodingFamilyDef '{family_name}': alias '{alias_name}' value "
            f"{value!r} does not satisfy {descriptor.attr_type} parameter "
            f"'{descriptor.name}'"
        )


@dataclass(frozen=True, slots=True)
class EncodingFamilyDef:
    """Declares one registered encoding family.

    Family names are stable public IR. Static parameters remain sparse named
    attributes in the IR and wire format. Dynamic parameters name ordinary SSA
    operands accepted by encoding.define. Lexical descriptor ordinals are
    generated implementation metadata used after parsing.
    """

    name: str
    group: Dialect
    role: EncodingFamilyRole
    parameters: tuple[AttrDef, ...] = ()
    aliases: tuple[EncodingAliasDef, ...] = ()
    alias_discriminator: AttrDef | None = None
    dynamic_parameters: tuple[Operand, ...] = ()
    fixed_record: EncodingRecordDef | None = None
    fixed_operand_summary: EncodingOperandSummaryDef | None = None
    auxiliary_key_enum: EnumDef | None = None
    required_auxiliary_keys: tuple[EnumCase, ...] = ()
    implicit_shaped_attachment: bool = False
    doc: str = ""

    def __init__(
        self,
        name: str,
        *,
        group: Dialect,
        role: EncodingFamilyRole,
        parameters: list[AttrDef] | tuple[AttrDef, ...] = (),
        aliases: list[EncodingAliasDef] | tuple[EncodingAliasDef, ...] = (),
        dynamic_parameters: list[Operand] | tuple[Operand, ...] = (),
        fixed_record: EncodingRecordDef | None = None,
        fixed_operand_summary: EncodingOperandSummaryDef | None = None,
        auxiliary_key_enum: EnumDef | None = None,
        required_auxiliary_keys: list[EnumCase] | tuple[EnumCase, ...] = (),
        implicit_shaped_attachment: bool = False,
        doc: str = "",
    ) -> None:
        if not _is_ascii_qualified_identifier(name):
            raise ValueError(
                f"EncodingFamilyDef '{name}': family name must be a dotted "
                "ASCII identifier"
            )
        if not isinstance(role, EncodingFamilyRole):
            raise ValueError(
                f"EncodingFamilyDef '{name}': role must be an EncodingFamilyRole"
            )
        if type(implicit_shaped_attachment) is not bool:
            raise ValueError(
                f"EncodingFamilyDef '{name}': implicit_shaped_attachment must be a bool"
            )
        if implicit_shaped_attachment:
            if role is not EncodingFamilyRole.ADDRESS_LAYOUT:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': an implicit shaped attachment "
                    "must have the ADDRESS_LAYOUT role"
                )
            if parameters or dynamic_parameters:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': an implicit shaped attachment "
                    "cannot have static or dynamic parameters"
                )
        if fixed_record is not None and not isinstance(fixed_record, EncodingRecordDef):
            raise ValueError(
                f"EncodingFamilyDef '{name}': fixed_record must be an EncodingRecordDef"
            )
        if fixed_operand_summary is not None and not isinstance(
            fixed_operand_summary, EncodingOperandSummaryDef
        ):
            raise ValueError(
                f"EncodingFamilyDef '{name}': fixed_operand_summary must be an "
                "EncodingOperandSummaryDef"
            )
        frozen_required_auxiliary_keys = tuple(required_auxiliary_keys)
        if (
            fixed_record is not None
            or fixed_operand_summary is not None
            or auxiliary_key_enum is not None
            or frozen_required_auxiliary_keys
        ) and role is not EncodingFamilyRole.STORAGE_SCHEMA:
            raise ValueError(
                f"EncodingFamilyDef '{name}': fixed storage metadata requires "
                "the STORAGE_SCHEMA role"
            )

        if auxiliary_key_enum is not None and not isinstance(
            auxiliary_key_enum, EnumDef
        ):
            raise ValueError(
                f"EncodingFamilyDef '{name}': auxiliary_key_enum must be an EnumDef"
            )
        if frozen_required_auxiliary_keys and auxiliary_key_enum is None:
            raise ValueError(
                f"EncodingFamilyDef '{name}': required auxiliary keys need an "
                "auxiliary key enum"
            )
        if auxiliary_key_enum is not None:
            if max(case.value for case in auxiliary_key_enum.cases) >= 64:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': auxiliary key ordinals must "
                    "fit in one uint64_t bitset"
                )
            for key in frozen_required_auxiliary_keys:
                if not any(key is case for case in auxiliary_key_enum.cases):
                    raise ValueError(
                        f"EncodingFamilyDef '{name}': required auxiliary key "
                        f"'{key.keyword}' is not in {auxiliary_key_enum.name}"
                    )
        if len(set(frozen_required_auxiliary_keys)) != len(
            frozen_required_auxiliary_keys
        ):
            raise ValueError(
                f"EncodingFamilyDef '{name}': duplicate required auxiliary key"
            )

        lexical_parameters = tuple(sorted(parameters, key=lambda value: value.name))
        _validate_descriptor_parameters(
            f"EncodingFamilyDef '{name}'", lexical_parameters
        )
        descriptors_by_name = {
            parameter.name: parameter for parameter in lexical_parameters
        }
        frozen_aliases = tuple(aliases)
        if len(frozen_aliases) > 0xFF:
            raise ValueError(
                f"EncodingFamilyDef '{name}': {len(frozen_aliases)} aliases "
                "exceed the uint8_t ordinal limit"
            )
        seen_alias_names: set[str] = set()
        for alias in frozen_aliases:
            if not isinstance(alias, EncodingAliasDef):
                raise ValueError(
                    f"EncodingFamilyDef '{name}': alias {alias!r} must be an "
                    "EncodingAliasDef"
                )
            if alias.name == name:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': alias repeats the family name"
                )
            if alias.name in seen_alias_names:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': duplicate alias '{alias.name}'"
                )
            seen_alias_names.add(alias.name)
            for parameter_name, parameter_value in alias.parameters:
                descriptor = descriptors_by_name.get(parameter_name)
                if descriptor is None:
                    raise ValueError(
                        f"EncodingFamilyDef '{name}': alias '{alias.name}' "
                        f"references unknown parameter '{parameter_name}'"
                    )
                _validate_encoding_alias_parameter(
                    name, alias.name, descriptor, parameter_value
                )
        frozen_aliases = tuple(sorted(frozen_aliases, key=lambda alias: alias.name))
        alias_discriminator = None
        if frozen_aliases:
            common_fixed_names = {
                parameter_name
                for parameter_name, _ in frozen_aliases[0].fixed_parameters
            }
            for alias in frozen_aliases[1:]:
                common_fixed_names &= {
                    parameter_name for parameter_name, _ in alias.fixed_parameters
                }
            for descriptor in lexical_parameters:
                if (
                    descriptor.name not in common_fixed_names
                    or descriptor.attr_type != ATTR_TYPE_ENUM
                ):
                    continue
                discriminator_values = {
                    dict(alias.fixed_parameters)[descriptor.name]
                    for alias in frozen_aliases
                }
                if len(discriminator_values) == len(frozen_aliases):
                    alias_discriminator = descriptor
                    break
            if alias_discriminator is None:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': canonical aliases require "
                    "one shared enum fixed parameter with a unique value per alias"
                )
        lexical_dynamic_parameters = tuple(
            sorted(dynamic_parameters, key=lambda value: value.name)
        )
        if len(lexical_dynamic_parameters) > 0xFF:
            raise ValueError(
                f"EncodingFamilyDef '{name}': {len(lexical_dynamic_parameters)} "
                "dynamic parameters exceed the uint8_t slot limit"
            )
        seen_dynamic_names: set[str] = set()
        for parameter in lexical_dynamic_parameters:
            if not _is_ascii_identifier(parameter.name):
                raise ValueError(
                    f"EncodingFamilyDef '{name}': dynamic parameter "
                    f"'{parameter.name}' must be a bare ASCII identifier"
                )
            if parameter.name in seen_dynamic_names:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': duplicate dynamic parameter "
                    f"'{parameter.name}'"
                )
            seen_dynamic_names.add(parameter.name)
            if parameter.variadic or parameter.optional:
                raise ValueError(
                    f"EncodingFamilyDef '{name}': dynamic parameter "
                    f"'{parameter.name}' cannot be variadic or optional; "
                    "family semantics determine presence"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "parameters", lexical_parameters)
        object.__setattr__(self, "aliases", frozen_aliases)
        object.__setattr__(self, "alias_discriminator", alias_discriminator)
        object.__setattr__(self, "dynamic_parameters", lexical_dynamic_parameters)
        object.__setattr__(self, "fixed_record", fixed_record)
        object.__setattr__(self, "fixed_operand_summary", fixed_operand_summary)
        object.__setattr__(self, "auxiliary_key_enum", auxiliary_key_enum)
        object.__setattr__(
            self, "required_auxiliary_keys", frozen_required_auxiliary_keys
        )
        object.__setattr__(
            self, "implicit_shaped_attachment", implicit_shaped_attachment
        )
        object.__setattr__(self, "doc", doc)


# ============================================================================
# Type declarations
# ============================================================================


@dataclass(frozen=True, slots=True)
class TypeParam:
    """A type parameter — another type nested inside this type.

    Used for parameterized types like vm.ref<hal.buffer> where the
    inner type (hal.buffer) is a type parameter.
    """

    name: str
    constraint: TypeConstraint = TypeConstraint.ANY
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ShapeParam:
    """Shape dimensions parameter for shaped types.

    Represents the dimension list in tile<4x4xf32>. Each dim is
    either a static integer or a dynamic dim reference [%name].
    """

    name: str
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ScalarParam:
    """Scalar element type parameter for shaped types.

    Represents the element type in tile<4x4xf32> — the f32 part.
    """

    name: str
    doc: str = ""


@dataclass(frozen=True, slots=True)
class EncodingParam:
    """Encoding parameter for shaped types.

    Represents the optional encoding suffix in
    tile<256xi8, #encoding.operand<element_format=i8, payload_elements=32,
    payload_packing=dense_lanes>>.
    """

    name: str
    optional: bool = True
    doc: str = ""


# Union of type parameter kinds. AttrDef parameters use the shared tagged-value
# schema and are consumed by Param format elements.
type TypeParamDef = TypeParam | ShapeParam | ScalarParam | EncodingParam | AttrDef


_COMPACT_SHAPE_IR_KINDS = frozenset(("pool", "tile", "tensor", "vector", "view"))


def _validate_compact_shape_format(
    name: str,
    ir_kind: str,
    params: tuple[TypeParamDef, ...],
    format_elements: tuple[FormatElement, ...],
) -> None:
    """Validates the direct compact shape grammars implemented by the IR."""
    compact_params = (ShapeParam, ScalarParam, EncodingParam)
    has_compact_params = any(isinstance(param, compact_params) for param in params)
    if ir_kind not in _COMPACT_SHAPE_IR_KINDS and not has_compact_params:
        return

    from loom.assembly import COMMA, EncodingOf, ScalarOf, ShapeOf, kw

    if ir_kind == "pool":
        if len(params) != 1 or not isinstance(params[0], ShapeParam):
            raise ValueError(
                f"TypeDef '{name}': pool representation requires one shape parameter"
            )
        expected_format: tuple[FormatElement, ...] = (ShapeOf(params[0].name),)
    elif ir_kind == "vector":
        if (
            len(params) == 3
            and isinstance(params[0], ShapeParam)
            and isinstance(params[1], ScalarParam)
            and isinstance(params[2], EncodingParam)
        ):
            raise ValueError(
                f"TypeDef '{name}': vector representation cannot carry an encoding"
            )
        if (
            len(params) != 2
            or not isinstance(params[0], ShapeParam)
            or not isinstance(params[1], ScalarParam)
        ):
            raise ValueError(
                f"TypeDef '{name}': vector representation requires shape and scalar "
                "parameters"
            )
        expected_format = (
            ShapeOf(params[0].name),
            kw("x"),
            ScalarOf(params[1].name),
        )
    elif ir_kind in ("tile", "tensor", "view"):
        if (
            len(params) != 3
            or not isinstance(params[0], ShapeParam)
            or not isinstance(params[1], ScalarParam)
            or not isinstance(params[2], EncodingParam)
        ):
            raise ValueError(
                f"TypeDef '{name}': shaped representation requires shape, scalar, "
                "and encoding parameters"
            )
        encoding_param = cast(EncodingParam, params[2])
        if not encoding_param.optional:
            raise ValueError(
                f"TypeDef '{name}': compact shape encodings must be optional"
            )
        expected_format = (
            ShapeOf(params[0].name),
            kw("x"),
            ScalarOf(params[1].name),
            OptionalGroup(
                [COMMA, EncodingOf(encoding_param.name)],
                anchor=encoding_param.name,
            ),
        )
    else:
        raise ValueError(
            f"TypeDef '{name}': compact shape parameters are unsupported for "
            f"IR kind '{ir_kind}'"
        )

    if format_elements != expected_format:
        raise ValueError(
            f"TypeDef '{name}': assembly format does not match its compact shape "
            "representation"
        )


@dataclass(frozen=True, slots=True)
class TypeDef:
    """Declarative type definition.

    Types are declared with the same pattern as ops: a name, optional
    parameters, and a format spec describing the textual syntax.

    The format describes the interior of name<...>:
      - Empty format = opaque type (no angle brackets): hal.buffer
      - Non-empty format = parameterized: vm.ref<hal.buffer>, tile<4x4xf32>
    The optional fact_domain names a C ``loom_value_fact_domain_t`` symbol
    attached to the generated type descriptor; typed fact extensions use the
    value's type to find this domain instead of a global schema registry.

    Scalar types (f32, i32, index) are NOT TypeDefs — they are
    keywords handled by a fixed name table. TypeDefs are for
    structured types with angle-bracket syntax.

    Examples:
        # Opaque type:
        TypeDef(name="hal.buffer")
        # Prints: hal.buffer

        # Single type parameter:
        TypeDef(name="vm.ref",
                params=[TypeParam("object", ANY)],
                format=[TypeOf("object")])
        # Prints: vm.ref<hal.buffer>

        # Descriptor-backed parameters with positional and keyed text:
        TypeDef(name="test.matrix",
                params=[AttrDef("element_type", ATTR_TYPE_TYPE),
                        AttrDef("rows", ATTR_TYPE_I64)],
                format=[Param("element_type"), COMMA,
                        kw("rows"), EQUALS, Param("rows")])
        # Prints: test.matrix<bf16, rows = 16>

        # Shaped type with dims, element, encoding:
        TypeDef(name="tile",
                params=[ShapeParam("dims"), ScalarParam("element_type"),
                        EncodingParam("encoding")],
                format=[ShapeOf("dims"), kw("x"), ScalarOf("element_type"),
                        OptionalGroup([COMMA, EncodingOf("encoding")],
                                      anchor="encoding")])
        # Prints: tile<4x4xf32, #test.schema>
    """

    name: str
    doc: str = ""
    params: tuple[TypeParamDef, ...] = ()
    format: tuple[FormatElement, ...] = ()
    ir_kind: str = "dialect"  # "tile", "tensor", "vector", "view", etc.
    python_type: type[Any] | None = None
    fact_domain: str | None = None
    semantic: TypeSemantic = TypeSemantic.ORDINARY
    contracts: tuple[ContractFamily, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        doc: str = "",
        params: list[TypeParamDef] | tuple[TypeParamDef, ...] = (),
        format: list[FormatElement] | tuple[FormatElement, ...] = (),
        ir_kind: str = "dialect",
        python_type: type[Any] | None = None,
        fact_domain: str | None = None,
        semantic: TypeSemantic = TypeSemantic.ORDINARY,
        contracts: list[ContractFamily] | tuple[ContractFamily, ...] = (),
    ) -> None:
        frozen_params = tuple(params)
        frozen_format = tuple(format)
        attribute_parameters = tuple(
            parameter for parameter in frozen_params if isinstance(parameter, AttrDef)
        )
        if attribute_parameters and len(attribute_parameters) != len(frozen_params):
            raise ValueError(
                f"TypeDef '{name}': descriptor-backed AttrDef parameters cannot "
                "be mixed with representation-specific type parameters"
            )
        _validate_compact_shape_format(name, ir_kind, frozen_params, frozen_format)
        if attribute_parameters:
            _validate_descriptor_parameters(f"TypeDef '{name}'", attribute_parameters)
        uses_inline_enum_parameter = bool(attribute_parameters) and ir_kind != "dialect"
        if uses_inline_enum_parameter:
            if len(attribute_parameters) != 1:
                raise ValueError(
                    f"TypeDef '{name}': compact descriptor-backed types require "
                    "exactly one parameter"
                )
            parameter = attribute_parameters[0]
            if parameter.attr_type != ATTR_TYPE_ENUM:
                raise ValueError(
                    f"TypeDef '{name}': compact descriptor-backed parameter "
                    "must be an enum"
                )
            if python_type is None:
                raise ValueError(
                    f"TypeDef '{name}': compact descriptor-backed types require "
                    "a Python value type"
                )
            if parameter.optional:
                assert parameter.enum_def is not None
                if parameter.open_enum:
                    raise ValueError(
                        f"TypeDef '{name}': optional compact enum parameter "
                        "cannot be open"
                    )
                if any(case.value == 0 for case in parameter.enum_def.cases):
                    raise ValueError(
                        f"TypeDef '{name}': optional compact enum parameter "
                        "reserves value zero for absence"
                    )
        elif python_type is not None:
            raise ValueError(
                f"TypeDef '{name}': Python value types are only supported for "
                "compact descriptor-backed types"
            )
        parameter_names = [parameter.name for parameter in frozen_params]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(f"TypeDef '{name}': duplicate parameter name")
        unknown_format_fields = _collect_format_fields(frozen_format) - set(
            parameter_names
        )
        if unknown_format_fields:
            raise ValueError(
                f"TypeDef '{name}': format references unknown parameter(s): "
                f"{', '.join(sorted(unknown_format_fields))}"
            )
        parameter_by_name = {parameter.name: parameter for parameter in frozen_params}
        parameter_format_fields = _collect_parameter_format_field_sequence(
            frozen_format
        )
        for field in parameter_format_fields:
            if not isinstance(parameter_by_name[field], AttrDef):
                raise ValueError(
                    f"TypeDef '{name}': Param('{field}') requires an AttrDef parameter"
                )
        if attribute_parameters:
            if len(set(parameter_format_fields)) != len(parameter_format_fields):
                raise ValueError(
                    f"TypeDef '{name}': each descriptor-backed parameter must "
                    "appear exactly once in the assembly format"
                )
            omitted_parameters = set(parameter_names) - set(parameter_format_fields)
            if omitted_parameters:
                raise ValueError(
                    f"TypeDef '{name}': assembly format omits descriptor-backed "
                    f"parameter(s): {', '.join(sorted(omitted_parameters))}"
                )
            omits_empty_parameter_list = bool(frozen_format) and all(
                isinstance(element, OptionalGroup) for element in frozen_format
            )
            if omits_empty_parameter_list and any(
                not parameter.optional for parameter in attribute_parameters
            ):
                raise ValueError(
                    f"TypeDef '{name}': an omitted empty parameter list requires "
                    "every parameter to be optional"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "params", frozen_params)
        object.__setattr__(self, "format", frozen_format)
        object.__setattr__(self, "ir_kind", ir_kind)
        object.__setattr__(self, "python_type", python_type)
        object.__setattr__(self, "fact_domain", fact_domain)
        object.__setattr__(self, "semantic", semantic)
        object.__setattr__(self, "contracts", tuple(contracts))

    def __repr__(self) -> str:
        return f"TypeDef({self.name!r})"

    @property
    def is_opaque(self) -> bool:
        """True if this type has no parameters (no angle brackets)."""
        return len(self.format) == 0

    @property
    def is_parameterized(self) -> bool:
        """True if this type has angle-bracket syntax."""
        return len(self.format) > 0

    @property
    def uses_attribute_parameters(self) -> bool:
        """True when this type uses descriptor-backed parameter metadata."""

        return bool(self.params) and isinstance(self.params[0], AttrDef)

    @property
    def uses_inline_enum_parameter(self) -> bool:
        """True when the sole enum parameter is stored in the type header."""

        return self.uses_attribute_parameters and self.ir_kind != "dialect"

    @property
    def uses_compact_shape_format(self) -> bool:
        """True when the type uses the direct compact shape representation."""

        return self.ir_kind in _COMPACT_SHAPE_IR_KINDS

    @property
    def omits_empty_parameter_list(self) -> bool:
        """True when absent optional parameters print without angle brackets."""

        return (
            self.uses_attribute_parameters
            and bool(self.format)
            and all(isinstance(element, OptionalGroup) for element in self.format)
        )

    def __call__(self, **parameters: Any) -> Any:
        """Constructs one immutable descriptor-backed value of this type."""

        if not self.uses_attribute_parameters:
            raise TypeError(
                f"TypeDef '{self.name}' does not use descriptor-backed parameters"
            )
        if self.uses_inline_enum_parameter:
            assert self.python_type is not None
            parameter = self.params[0]
            assert isinstance(parameter, AttrDef)
            unknown_names = sorted(set(parameters) - {parameter.name})
            if unknown_names:
                raise TypeError(
                    f"{self.name}: unknown parameter(s): {', '.join(unknown_names)}"
                )
            if parameter.name not in parameters:
                if not parameter.optional:
                    raise TypeError(
                        f"{self.name}: missing required parameter '{parameter.name}'"
                    )
                return self.python_type()

            assert parameter.enum_def is not None
            value = parameters[parameter.name]
            value_by_keyword = {
                case.keyword: case.value for case in parameter.enum_def.cases
            }
            if isinstance(value, str):
                if value not in value_by_keyword:
                    raise ValueError(
                        f"{self.name}: parameter '{parameter.name}' has unknown "
                        f"enum keyword {value!r}"
                    )
                value = value_by_keyword[value]
            elif isinstance(value, int) and not isinstance(value, bool):
                value = int(value)
                if not 0 <= value <= 0xFF:
                    raise ValueError(
                        f"{self.name}: parameter '{parameter.name}' enum value "
                        f"{value} is outside the byte domain"
                    )
                if (
                    not parameter.open_enum
                    and value not in value_by_keyword.values()
                    and not (parameter.optional and value == 0)
                ):
                    raise ValueError(
                        f"{self.name}: parameter '{parameter.name}' has "
                        f"undeclared enum value {value}"
                    )
            else:
                raise TypeError(
                    f"{self.name}: parameter '{parameter.name}' must be an enum "
                    f"keyword or byte value, got {value!r}"
                )
            return self.python_type(**{parameter.name: value})

        from loom.ir import ParameterizedType

        return ParameterizedType(self, parameters)

    def param(self, name: str) -> TypeParamDef | None:
        """Find a parameter by name."""
        for p in self.params:
            if p.name == name:
                return p
        return None


# ============================================================================
# Format field validation
# ============================================================================

# Fields that are implicit (not declared as operands/results/attrs/regions
# but valid in format specs). These are created by the printer/parser from
# context: "iv" is the induction variable for loops, "args" are function
# parameters, "predicates" are where-clause items.
_IMPLICIT_FORMAT_FIELDS = frozenset({"iv", "args", "predicates"})


def _collect_format_fields(elements: tuple[FormatElement, ...]) -> set[str]:
    """Recursively collect all field names referenced by format elements."""
    from loom.assembly import (
        AlignedRefs,
        Attr,
        AttrDict,
        AttrParams,
        AttrTable,
        BindingList,
        BlockArgs,
        BlockRef,
        Clause,
        EncodingOf,
        Flags,
        FuncArgs,
        Glue,
        IndexList,
        KeyRef,
        Keyword,
        OperandDict,
        OptionalGroup,
        Param,
        PredicateList,
        Ref,
        Refs,
        Region,
        ResultType,
        ResultTypeList,
        ScalarOf,
        Scope,
        ScopedEnumRef,
        ShapeOf,
        StableKeyRef,
        SymbolRef,
        TemplateParam,
        TemplateParamFlags,
        TypedRefs,
        TypeOf,
        TypesOf,
    )

    fields: set[str] = set()
    for elem in elements:
        match elem:
            case Ref(field=f) | Refs(field=f) | TypedRefs(field=f) | BlockRef(field=f):
                fields.add(f)
            case AlignedRefs(refs=refs, alignments=alignments):
                fields.add(refs)
                fields.add(alignments)
            case (
                Attr(field=f)
                | SymbolRef(field=f)
                | Flags(field=f)
                | KeyRef(field=f)
                | ScopedEnumRef(field=f)
                | TemplateParam(field=f)
                | AttrParams(field=f)
            ):
                fields.add(f)
            case TemplateParamFlags(param=param, flags=flags):
                fields.add(param)
                fields.add(flags)
            case StableKeyRef(key=key, stable_id=stable_id):
                fields.add(key)
                fields.add(stable_id)
            case TypeOf(field=f) | TypesOf(field=f):
                fields.add(f)
            case ResultType(field=f) | ResultTypeList(field=f):
                fields.add(f)
            case Region(field=f):
                fields.add(f)
            case BindingList(field=f) | BlockArgs(region=f):
                fields.add(f)
            case FuncArgs(field=f, start_attr=start_attr, end_attr=end_attr):
                fields.add(f)
                if start_attr is not None:
                    fields.add(start_attr)
                if end_attr is not None:
                    fields.add(end_attr)
            case PredicateList(field=f):
                fields.add(f)
            case IndexList(dynamic=d, static=s):
                fields.add(d)
                fields.add(s)
            case OperandDict(operands=operands, names=names):
                fields.add(operands)
                fields.add(names)
            case AttrTable(keys=keys, values=values):
                fields.add(keys)
                fields.add(values)
            case Clause(elements=inner):
                fields |= _collect_format_fields(inner)
            case OptionalGroup(elements=inner):
                fields |= _collect_format_fields(inner)
            case Scope(elements=inner):
                fields |= _collect_format_fields(inner)
            case (
                ShapeOf(field=f)
                | ScalarOf(field=f)
                | EncodingOf(field=f)
                | Param(field=f)
            ):
                fields.add(f)
            case Keyword() | AttrDict() | Glue():
                pass
    return fields


def _collect_parameter_format_field_sequence(
    elements: tuple[FormatElement, ...],
) -> tuple[str, ...]:
    """Collects Param fields in assembly order, retaining duplicates."""

    from loom.assembly import Clause, OptionalGroup, Param, Scope

    fields: list[str] = []
    for element in elements:
        match element:
            case Param(field=field):
                fields.append(field)
            case (
                Clause(elements=nested)
                | OptionalGroup(elements=nested)
                | Scope(elements=nested)
            ):
                fields.extend(_collect_parameter_format_field_sequence(nested))
            case _:
                pass
    return tuple(fields)


def _collect_func_args_fields(elements: tuple[FormatElement, ...]) -> set[str]:
    """Recursively collect FuncArgs field names referenced by format elements."""
    from loom.assembly import Clause, FuncArgs, OptionalGroup, Scope

    fields: set[str] = set()
    for elem in elements:
        match elem:
            case FuncArgs(field=f):
                fields.add(f)
            case Clause(elements=inner):
                fields |= _collect_func_args_fields(inner)
            case OptionalGroup(elements=inner):
                fields |= _collect_func_args_fields(inner)
            case Scope(elements=inner):
                fields |= _collect_func_args_fields(inner)
            case _:
                pass
    return fields


def _collect_func_args_elements(
    elements: tuple[FormatElement, ...],
) -> tuple[FuncArgs, ...]:
    """Collects FuncArgs elements in assembly order."""
    from loom.assembly import Clause, FuncArgs, OptionalGroup, Scope

    func_args: list[FuncArgs] = []
    for element in elements:
        match element:
            case FuncArgs():
                func_args.append(element)
            case (
                Clause(elements=nested)
                | OptionalGroup(elements=nested)
                | Scope(elements=nested)
            ):
                func_args.extend(_collect_func_args_elements(nested))
            case _:
                pass
    return tuple(func_args)


def _validate_func_args_partitions(
    op_name: str,
    format_elements: tuple[FormatElement, ...],
    attrs: tuple[AttrDef, ...],
) -> None:
    """Validates contiguous function argument partitions."""
    func_args = _collect_func_args_elements(format_elements)
    if not any(
        element.start_attr is not None or element.end_attr is not None
        for element in func_args
    ):
        return

    effective_groups = [element.group or element.field for element in func_args]
    if any(not group for group in effective_groups) or len(
        set(effective_groups)
    ) != len(effective_groups):
        raise ValueError(
            f"Op '{op_name}': projected FuncArgs groups require distinct "
            "non-empty group names"
        )

    groups_by_field: dict[str, list[FuncArgs]] = {}
    for element in func_args:
        groups_by_field.setdefault(element.field, []).append(element)
    for field, field_groups in groups_by_field.items():
        if not any(
            element.start_attr is not None or element.end_attr is not None
            for element in field_groups
        ):
            continue
        previous_end: str | None = None
        for index, element in enumerate(field_groups):
            group = element.group or field
            if element.start_attr != previous_end:
                expected = (
                    repr(previous_end) if previous_end is not None else "no start attr"
                )
                raise ValueError(
                    f"Op '{op_name}': FuncArgs group '{group}' must use "
                    f"{expected} as its start boundary"
                )
            if index + 1 < len(field_groups) and element.end_attr is None:
                raise ValueError(
                    f"Op '{op_name}': FuncArgs group '{group}' requires an "
                    "end boundary before the next group"
                )
            previous_end = element.end_attr
        if previous_end is not None:
            raise ValueError(
                f"Op '{op_name}': the final projected FuncArgs group for "
                f"field '{field}' must extend to the end of the function "
                "signature"
            )

    attrs_by_name = {attr.name: attr for attr in attrs}
    for element in func_args:
        if element.end_attr is None:
            continue
        attr = attrs_by_name.get(element.end_attr)
        if attr is None or attr.attr_type != ATTR_TYPE_I64 or attr.optional:
            raise ValueError(
                f"Op '{op_name}': FuncArgs boundary '{element.end_attr}' must "
                "name a required i64 attribute"
            )


def _validate_no_nested_scope(
    op_name: str,
    elements: tuple[FormatElement, ...],
    depth: int = 0,
) -> None:
    """Validate that Scope elements are not nested."""
    from loom.assembly import Clause, OptionalGroup, Scope

    for elem in elements:
        if isinstance(elem, Scope):
            if depth > 0:
                raise ValueError(
                    f"Op '{op_name}': nested Scope is not supported. "
                    f"Each op may have at most one Scope level."
                )
            _validate_no_nested_scope(op_name, elem.elements, depth + 1)
        elif isinstance(elem, Clause | OptionalGroup):
            _validate_no_nested_scope(op_name, elem.elements, depth)


def _declared_format_fields(
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    attrs: tuple[AttrDef, ...],
    successors: tuple[Successor, ...],
    regions: tuple[RegionDef, ...],
) -> set[str]:
    """Returns all fields a format element may reference."""
    return (
        {o.name for o in operands}
        | {r.name for r in results}
        | {a.name for a in attrs}
        | {s.name for s in successors}
        | {r.name for r in regions}
        | _IMPLICIT_FORMAT_FIELDS
    )


def _validate_format_fields(
    op_name: str,
    format_elements: tuple[FormatElement, ...],
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    attrs: tuple[AttrDef, ...],
    successors: tuple[Successor, ...],
    regions: tuple[RegionDef, ...],
) -> None:
    """Validate that all fields in format elements are declared on the op."""
    _validate_no_nested_scope(op_name, format_elements)
    declared = _declared_format_fields(operands, results, attrs, successors, regions)
    referenced = _collect_format_fields(format_elements)
    unknown = referenced - declared
    if unknown:
        raise ValueError(
            f"Op '{op_name}': format references undeclared fields: "
            f"{sorted(unknown)}. Declared: {sorted(declared)}"
        )


def _validate_scoped_enum_fields(
    op_name: str,
    format_elements: tuple[FormatElement, ...],
    attrs: tuple[AttrDef, ...],
) -> None:
    """Validates required representation-scoped enum syntax."""
    from loom.assembly import ScopedEnumRef

    scoped_attrs = [attr for attr in attrs if attr.attr_type == ATTR_TYPE_SCOPED_ENUM]
    if len(scoped_attrs) > 1:
        raise ValueError(
            f"Op '{op_name}': at most one scoped_enum attr may define its "
            "representation-scoped identity"
        )
    direct_refs = [
        element.field
        for element in format_elements
        if isinstance(element, ScopedEnumRef)
    ]
    attrs_by_name = {attr.name: attr for attr in attrs}
    for field in direct_refs:
        attr = attrs_by_name.get(field)
        if attr is None or attr.attr_type != ATTR_TYPE_SCOPED_ENUM:
            raise ValueError(
                f"Op '{op_name}': ScopedEnumRef field '{field}' must name a "
                "scoped_enum attr"
            )
    for attr in scoped_attrs:
        ref_count = direct_refs.count(attr.name)
        if ref_count != 1:
            raise ValueError(
                f"Op '{op_name}': scoped_enum attr '{attr.name}' requires "
                "exactly one top-level ScopedEnumRef"
            )


def _validate_attr_params_fields(
    op_name: str,
    format_elements: tuple[FormatElement, ...],
    attrs: tuple[AttrDef, ...],
) -> None:
    """Validates known-family parameterized attribute payload syntax."""
    from loom.assembly import AttrParams, Clause, OptionalGroup, Scope

    attrs_by_name = {attr.name: attr for attr in attrs}
    for element in format_elements:
        if isinstance(element, AttrParams):
            attr = attrs_by_name.get(element.field)
            if (
                attr is None
                or attr.attr_type != ATTR_TYPE_PARAMETERIZED
                or attr.parameterized_attr is None
            ):
                raise ValueError(
                    f"Op '{op_name}': AttrParams field '{element.field}' must "
                    "name a parameterized attr constrained to one exact family"
                )
        elif isinstance(element, Clause | OptionalGroup | Scope):
            _validate_attr_params_fields(op_name, element.elements, attrs)


def _validate_legacy_formats(
    op_name: str,
    legacy_formats: tuple[LegacyFormat, ...],
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    attrs: tuple[AttrDef, ...],
    successors: tuple[Successor, ...],
    regions: tuple[RegionDef, ...],
) -> None:
    """Validate legacy format declarations against the current op surface."""
    declared = _declared_format_fields(operands, results, attrs, successors, regions)
    seen_rule_ids: set[str] = set()
    for legacy_format in legacy_formats:
        _validate_no_nested_scope(op_name, legacy_format.format)
        if not legacy_format.rule_id:
            raise ValueError(f"Op '{op_name}': legacy format rule_id is empty.")
        if legacy_format.rule_id in seen_rule_ids:
            raise ValueError(
                f"Op '{op_name}': duplicate legacy format rule_id "
                f"'{legacy_format.rule_id}'."
            )
        seen_rule_ids.add(legacy_format.rule_id)
        if not legacy_format.introduced:
            raise ValueError(
                f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                "introduced baseline is empty."
            )
        if not legacy_format.replaced_by:
            raise ValueError(
                f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                "replaced_by baseline is empty."
            )
        field_mapping: dict[str, str] = {}
        mapped_current_fields: set[str] = set()
        for mapping in legacy_format.field_mappings:
            if not mapping.legacy:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    "has an empty legacy field mapping source."
                )
            if not mapping.current:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    "has an empty legacy field mapping target."
                )
            if mapping.legacy in field_mapping:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    f"maps legacy field '{mapping.legacy}' more than once."
                )
            if mapping.current in mapped_current_fields:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    f"maps more than one legacy field to current field "
                    f"'{mapping.current}'."
                )
            field_mapping[mapping.legacy] = mapping.current
            mapped_current_fields.add(mapping.current)

        referenced = _collect_format_fields(legacy_format.format)
        unused_mappings = set(field_mapping) - referenced
        if unused_mappings:
            raise ValueError(
                f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                "maps fields that are not referenced by its format: "
                f"{sorted(unused_mappings)}."
            )
        mapped_referenced = {field_mapping.get(field, field) for field in referenced}
        unknown = mapped_referenced - declared
        if unknown:
            raise ValueError(
                f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                "references undeclared current fields after mapping: "
                f"{sorted(unknown)}. Declared: {sorted(declared)}"
            )

        default_fields: set[str] = set()
        for field_default in legacy_format.field_defaults:
            if not field_default.field:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    "has an empty field default target."
                )
            if field_default.field in default_fields:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    f"defaults field '{field_default.field}' more than once."
                )
            if field_default.field in mapped_referenced:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    f"defaults field '{field_default.field}' that is already "
                    "parsed by its format."
                )
            default_fields.add(field_default.field)
            if field_default.field not in declared:
                raise ValueError(
                    f"Op '{op_name}': legacy format '{legacy_format.rule_id}' "
                    f"defaults undeclared field '{field_default.field}'. "
                    f"Declared: {sorted(declared)}"
                )


def _validate_region_arg_sources(
    op_name: str,
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    regions: tuple[RegionDef, ...],
    format_elements: tuple[FormatElement, ...],
) -> None:
    """Validate RegionDef.arg_source contracts."""
    value_fields: dict[str, bool] = {o.name: o.variadic for o in operands} | {
        r.name: r.variadic for r in results
    }
    func_args_fields = _collect_func_args_fields(format_elements)
    for region in regions:
        if region.arg_source is None:
            continue
        if region.arg_source in func_args_fields:
            continue
        variadic = value_fields.get(region.arg_source)
        if variadic is None:
            raise ValueError(
                f"Op '{op_name}': region '{region.name}' arg_source "
                f"references non-value/non-FuncArgs field '{region.arg_source}'."
            )
        if not variadic:
            raise ValueError(
                f"Op '{op_name}': region '{region.name}' arg_source "
                f"'{region.arg_source}' must reference a variadic value field."
            )


# ============================================================================
# Effect validation
# ============================================================================


def _validate_effects(
    op_name: str,
    effects: tuple[Effect, ...],
    operands: tuple[Operand, ...],
    traits: tuple[Trait, ...],
) -> None:
    """Validate memory effect declarations against operands and traits."""
    trait_names = {t.name for t in traits}
    operand_map = {o.name: o for o in operands}

    # PURE and effects are mutually exclusive.
    if "Pure" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both traits=[PURE] and effects=["
            f"{', '.join(repr(e) for e in effects)}]. "
            f"An op with memory effects cannot be pure."
        )

    # UNKNOWN_EFFECTS and explicit effects are mutually exclusive.
    if "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both UNKNOWN_EFFECTS and explicit "
            f"effects. If the effects are known, declare them; if unknown, "
            f"use UNKNOWN_EFFECTS alone."
        )
    if "Hint" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and explicit effects. "
            f"A hint is not a semantic memory effect; attach policy to the "
            f"real memory op instead."
        )
    if "CompileTimeOnly" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both COMPILE_TIME_ONLY and explicit "
            f"effects. A compile-time-only op has no runtime memory effect."
        )
    if "SafeToSpeculate" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and explicit "
            f"effects. Speculation must not introduce additional memory "
            f"accesses or runtime observations."
        )

    for effect in effects:
        # Effect must reference an existing operand.
        if effect.operand not in operand_map:
            raise ValueError(
                f"Op '{op_name}': effect references operand "
                f"'{effect.operand}' which is not declared. "
                f"Declared operands: {sorted(operand_map.keys())}"
            )

        # Referenced operand must have a resource-like type constraint.
        operand = operand_map[effect.operand]
        if operand.type_constraint not in _RESOURCE_TYPE_CONSTRAINTS:
            allowed = sorted(c.value for c in _RESOURCE_TYPE_CONSTRAINTS)
            raise ValueError(
                f"Op '{op_name}': effect on operand "
                f"'{effect.operand}' with type constraint "
                f"'{operand.type_constraint.value}' is not "
                f"allowed. Effects may only reference "
                f"resource-typed operands "
                f"({', '.join(allowed)})."
            )


def _validate_ownership_effects(
    op_name: str,
    ownership_effects: tuple[OperandOwnershipEffect | ResultOwnershipEffect, ...],
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    traits: tuple[Trait, ...],
) -> None:
    """Validate ownership effect declarations against operands and results."""
    trait_names = {t.name for t in traits}
    if "Pure" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both traits=[PURE] and ownership "
            f"effects. Ownership transfer is semantic and cannot be pure."
        )

    operand_map = {o.name: o for o in operands}
    result_map = {r.name: r for r in results}
    operand_effects: set[str] = set()
    result_effects: set[str] = set()
    moved_sources: set[str] = set()

    for effect in ownership_effects:
        if isinstance(effect, OperandOwnershipEffect):
            if effect.operand in operand_effects:
                raise ValueError(
                    f"Op '{op_name}': duplicate ownership effect for operand "
                    f"'{effect.operand}'."
                )
            operand_effects.add(effect.operand)
            if effect.operand not in operand_map:
                raise ValueError(
                    f"Op '{op_name}': ownership effect references operand "
                    f"'{effect.operand}' which is not declared. "
                    f"Declared operands: {sorted(operand_map.keys())}"
                )
            continue

        if effect.result in result_effects:
            raise ValueError(
                f"Op '{op_name}': duplicate ownership effect for result "
                f"'{effect.result}'."
            )
        result_effects.add(effect.result)
        if effect.result not in result_map:
            raise ValueError(
                f"Op '{op_name}': ownership effect references result "
                f"'{effect.result}' which is not declared. "
                f"Declared results: {sorted(result_map.keys())}"
            )
        if effect.kind in (
            ResultOwnershipEffectKind.ALIAS,
            ResultOwnershipEffectKind.MOVED,
        ):
            if effect.source is None:
                raise ValueError(
                    f"Op '{op_name}': {effect.kind.value} result ownership effect for "
                    f"'{effect.result}' must name a source operand."
                )
            if effect.source not in operand_map:
                raise ValueError(
                    f"Op '{op_name}': {effect.kind.value} result ownership effect for "
                    f"'{effect.result}' references operand '{effect.source}' "
                    f"which is not declared. Declared operands: "
                    f"{sorted(operand_map.keys())}"
                )
            source_operand = operand_map[effect.source]
            result = result_map[effect.result]
            if source_operand.variadic or result.variadic:
                raise ValueError(
                    f"Op '{op_name}': {effect.kind.value} result ownership effect for "
                    f"'{effect.result}' must use fixed operand/result fields."
                )
            if effect.kind == ResultOwnershipEffectKind.MOVED:
                if effect.source in moved_sources:
                    raise ValueError(
                        f"Op '{op_name}': operand '{effect.source}' may source "
                        f"only one moved result ownership effect."
                    )
                moved_sources.add(effect.source)
        elif effect.source is not None:
            raise ValueError(
                f"Op '{op_name}': ownership effect for result "
                f"'{effect.result}' may not name a source operand unless it "
                f"is an alias or moved effect."
            )

    conflicting_sources = moved_sources & operand_effects
    if conflicting_sources:
        source = sorted(conflicting_sources)[0]
        raise ValueError(
            f"Op '{op_name}': moved result source operand '{source}' may not "
            f"also declare an operand ownership effect."
        )


def _validate_no_effect_conflicts(
    op_name: str,
    traits: tuple[Trait, ...],
) -> None:
    """Validate trait-only ops for effect-related conflicts."""
    trait_names = {t.name for t in traits}
    if "CommandEffect" in trait_names and not trait_names.intersection(
        {"UnknownEffects", "MemoryFence", "NonDeterministic", "Convergent"}
    ):
        raise ValueError(
            f"Op '{op_name}': COMMAND_EFFECT requires explicit or unknown "
            "observable effects. The trait classifies effects but does not "
            "declare them."
        )
    if "CompileTimeOnly" in trait_names:
        if "Hint" in trait_names:
            raise ValueError(
                f"Op '{op_name}': COMPILE_TIME_ONLY is redundant with HINT. "
                "Hints are always omitted from runtime emission."
            )
        runtime_traits = {
            "UnknownEffects",
            "MemoryFence",
            "NonDeterministic",
            "Convergent",
            "UniqueIdentity",
        }
        conflicting_traits = sorted(trait_names & runtime_traits)
        if conflicting_traits:
            raise ValueError(
                f"Op '{op_name}': COMPILE_TIME_ONLY conflicts with runtime "
                f"trait {conflicting_traits[0]}."
            )
    if "Pure" in trait_names and "NonDeterministic" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and NON_DETERMINISTIC. "
            f"A pure op is deterministic by definition."
        )
    if "Pure" in trait_names and "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and UNKNOWN_EFFECTS. "
            f"A pure op has no effects."
        )
    if "Pure" in trait_names and "MemoryFence" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and MEMORY_FENCE. "
            f"A memory fence is an observable ordering effect."
        )
    if "Pure" in trait_names and "UniqueIdentity" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both PURE and UNIQUE_IDENTITY. "
            f"A pure op can be CSE'd; a unique identity op cannot."
        )
    if "Hint" in trait_names and "Pure" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and PURE. "
            f"Hints are semantically discardable but intentionally preserved "
            f"until an explicit strip pass."
        )
    if "Hint" in trait_names and "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and UNKNOWN_EFFECTS. "
            f"Hints are not semantic effects."
        )
    if "Hint" in trait_names and "MemoryFence" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and MEMORY_FENCE. "
            f"A memory fence is semantic, not a compiler hint."
        )
    if "Hint" in trait_names and "NonDeterministic" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and NON_DETERMINISTIC. "
            f"Hints do not produce observable values."
        )
    if "Hint" in trait_names and "Convergent" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both HINT and CONVERGENT. "
            f"Convergent execution is semantic, not a compiler hint."
        )
    if "SafeToSpeculate" in trait_names and "Hint" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and HINT. "
            f"Hints are preserved until explicitly stripped and must not be "
            f"introduced on additional control paths."
        )
    if "SafeToSpeculate" in trait_names and "UnknownEffects" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and UNKNOWN_EFFECTS. "
            f"Unknown effects cannot be executed on additional control paths."
        )
    if "SafeToSpeculate" in trait_names and "MemoryFence" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and MEMORY_FENCE. "
            f"Speculation must not add memory-ordering effects."
        )
    if "SafeToSpeculate" in trait_names and "NonDeterministic" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and NON_DETERMINISTIC. "
            f"Speculation must not add extra runtime observations."
        )
    if "SafeToSpeculate" in trait_names and "UniqueIdentity" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and UNIQUE_IDENTITY. "
            f"Speculation must not create extra identities."
        )
    if "SafeToSpeculate" in trait_names and "Convergent" in trait_names:
        raise ValueError(
            f"Op '{op_name}': declares both SAFE_TO_SPECULATE and CONVERGENT. "
            f"Speculation must not change the dynamic participant set."
        )


def _validate_trait_field_contracts(
    op_name: str,
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    regions: tuple[RegionDef, ...],
    traits: tuple[Trait, ...],
) -> None:
    """Validate trait contracts that depend on declared operand/result shape."""
    trait_names = {t.name for t in traits}
    if "ValueAlias" in trait_names and (len(operands) < 1 or len(results) != 1):
        raise ValueError(
            f"Op '{op_name}': VALUE_ALIAS requires at least one operand "
            "and exactly one result"
        )
    if "ConstantLike" in trait_names:
        if "Pure" not in trait_names:
            raise ValueError(f"Op '{op_name}': CONSTANT_LIKE requires PURE semantics")
        if operands or regions or len(results) != 1 or results[0].variadic:
            raise ValueError(
                f"Op '{op_name}': CONSTANT_LIKE requires no operands or regions "
                "and exactly one result"
            )
    if "Poison" in trait_names:
        if "Pure" not in trait_names:
            raise ValueError(f"Op '{op_name}': POISON requires PURE semantics")
        if "ConstantLike" in trait_names:
            raise ValueError(f"Op '{op_name}': POISON cannot also be CONSTANT_LIKE")
        if operands or regions or len(results) != 1 or results[0].variadic:
            raise ValueError(
                f"Op '{op_name}': POISON requires no operands or regions "
                "and exactly one result"
            )
    if (
        "CompileTimeOnly" in trait_names
        and results
        and ("FactIdentity" not in trait_names or "StorageRelation" not in trait_names)
    ):
        raise ValueError(
            f"Op '{op_name}': result-producing COMPILE_TIME_ONLY ops "
            "require FACT_IDENTITY and STORAGE_RELATION"
        )


# ============================================================================
# Interfaces
# ============================================================================


class CallLikeKind(Enum):
    """Semantic class of a direct call-like op."""

    # Ordinary runtime/semantic dispatch.
    SEMANTIC = "semantic"
    # Direct target-low function body to target-low function body call.
    LOW_INTERNAL = "low_internal"
    # Explicit semantic-to-target-low invocation of an already selected low function.
    LOW_INVOKE = "low_invoke"
    # Command-program invocation edge. Its operands are staged specialization
    # values followed by issue-time buffer bindings.
    COMMAND_PROGRAM = "command_program"
    # Exact compile-time template implementation call.
    TEMPLATE = "template"


class InlinePolicy(Enum):
    """Required callable-boundary policy."""

    UNSPECIFIED = 0
    INLINE = 1
    NOINLINE = 2


class CallLikeInterface(NamedTuple):
    """Interface for direct symbol call-like ops.

    The named operand/result fields are trailing slices containing call
    arguments and call results. A result field of None denotes an operation
    with no results. The generator resolves the slice starting offsets and
    emits a loom_call_like_vtable_t in .rodata.
    """

    # Symbol ref attr naming the direct callee.
    callee: str
    # Variadic operand field holding call arguments.
    operands: str
    # Variadic result field holding call results, or None for no-result calls.
    results: str | None
    # Optional purity enum attr. None if not applicable.
    purity: str | None = None
    # Optional temperature enum attr. None if not applicable.
    temperature: str | None = None
    # Optional inline policy enum attr. None if not applicable.
    inline_policy: str | None = None
    # Semantic class used by analyses to opt into only the call shapes they own.
    kind: CallLikeKind = CallLikeKind.SEMANTIC


@unique
class FuncLikeInterfaceFlag(Enum):
    """Semantic roles carried by a function-like operation kind."""

    # The operation defines an executable kernel entry. Kernel entries have an
    # implicit artifact export and a kernel ABI even without explicit attrs.
    KERNEL_ENTRY = "kernel_entry"


class FuncLikeInterface(NamedTuple):
    """Interface for function-like ops (def, decl, template, ukernel).

    Each field is the name of an attr or region on the op, or None if
    the op doesn't have that field. The generator resolves names to
    attr/region indices and emits a loom_func_like_vtable_t in .rodata.
    """

    # Symbol ref attr that names this function (required).
    callee: str
    # Optional import module string attr for external declarations.
    import_module: str | None = None
    # Optional import symbol string attr for external declarations.
    import_symbol: str | None = None
    # Optional symbol ref attr naming the resolved target record.
    target: str | None = None
    # Optional string attr naming the intrinsic contract under which the
    # function signature and body are represented. None means the op does not
    # expose an intrinsic representation contract.
    repr_contract: str | None = None
    # Optional ABI enum attr for concrete target-bound functions.
    abi: str | None = None
    # Optional ABI payload dictionary attr.
    abi_attrs: str | None = None
    # Optional exported symbol string attr.
    export_symbol: str | None = None
    # Optional export payload dictionary attr.
    export_attrs: str | None = None
    # Optional export linkage attr for entry-style exports.
    export_linkage: str | None = None
    # Visibility enum attr (e.g., public). None if not applicable.
    visibility: str | None = None
    # Calling convention enum attr. None if not applicable.
    cc: str | None = None
    # Purity enum attr. None if not applicable.
    purity: str | None = None
    # Temperature enum attr. None if not applicable.
    temperature: str | None = None
    # Inline policy enum attr. None if not applicable.
    inline_policy: str | None = None
    # Predicate list attr for where-clause constraints. None if absent.
    predicates: str | None = None
    # Parameterized attribute array naming provider proof requirements.
    # None for function kinds that are not implementation providers.
    requires: str | None = None
    # Optional i64 attr containing the number of leading arguments that
    # participate in staged specialization. Remaining arguments form the
    # issue-time binding ABI. None means there is no leading staged group.
    specialization_count: str | None = None
    # Region name for the function body. None for bodyless ops that
    # only declare a signature without providing an implementation.
    body: str | None = None
    # Symbol attr naming the template family implemented by this function.
    # None for function kinds that are not template providers.
    template_family: str | None = None
    # I64 attr for dispatch priority among competing implementations.
    # None for def/decl.
    priority: str | None = None
    # Variadic operand field storing signature arguments for bodyless
    # declarations. None means arguments are entry block values in body. An
    # operand-backed signature owns every op operand; kernel declarations may
    # divide those definitions between this ABI field and their workload field.
    args: str | None = None
    # Semantic roles intrinsic to the operation kind.
    flags: tuple[FuncLikeInterfaceFlag, ...] = ()


@unique
class TargetFactSpecialization(Enum):
    """Specialization relation for generated common target facts."""

    # Only the same projected fact object satisfies specialization.
    EXACT = "exact"
    # Distinct facts satisfy compatible common snapshots and configuration.
    STRUCTURAL = "structural"


class TargetLikeInterface(NamedTuple):
    """Interface for ops that define target environment records.

    The symbol field names the defining symbol attr. The selector field names
    the typed attr selecting the generated target row, such as a processor or
    generic target kind. The extensions field names an optional dict attr
    carrying target-specific extension data. Generated descriptor metadata
    binds the target op to its fact type and optional family fact projector.
    """

    # Symbol attr that names the target record.
    symbol: str
    # Typed attr selecting the target row used as the projection base.
    selector: str
    # Optional target-specific extension dictionary attr.
    extensions: str | None = None
    # Optional C symbol for the target-family projection descriptor.
    descriptor: str | None = None
    # Optional C symbol for the selector-indexed target bundle table. When
    # present, the C generator emits the TargetLike descriptor and projection
    # table instead of requiring hand-authored descriptor metadata.
    bundle_table: str | None = None
    # Optional C symbol for a family-owned target fact type. When absent, the C
    # generator emits a common fact type private to this op whose identity is
    # defined by selector equality.
    fact_type: str | None = None
    # Optional C symbol for the target-op adapter that projects family-owned
    # attributes into the typed fact extension.
    fact_projector: str | None = None
    # Specialization relation used by a generated common fact type.
    # Family-owned fact types carry their relation in the named descriptor.
    fact_specialization: TargetFactSpecialization = TargetFactSpecialization.EXACT


class LoopLikeInterface(NamedTuple):
    """Interface for loop-like ops that iterate a body region.

    The variadic iter_args operand is the complete loop-carried state domain.
    Implementing ops have one matching variadic result field and a required
    single-block body whose carried block arguments follow any induction
    variable. IterArgsMatchResults, YieldCountMatchesResults, and
    YieldTypesMatchResults constraints verify the complete state cycle.

    Each field is the name of a region, an implicit block argument, or an
    operand on the op (or None where applicable). Exactly one control form is
    present: either lower_bound, upper_bound, and step with an induction
    variable, or a separate condition_region without an induction variable.
    The generator resolves names to indices, validates the contract, and emits
    a loom_loop_like_vtable_t in .rodata. Used by LICM, loop-invariant sinking,
    trip count analysis, fact propagation, and loop transformation passes.
    """

    # Region name for the primary loop body. For scf.for this is the
    # one body region; for scf.while this is the "after" region that
    # contains the iteration body.
    body: str
    # Operand name where loop-carried state begins. For scf.for this
    # is the iter_args variadic operand following lower_bound,
    # upper_bound, step; for scf.while iter_args are the only
    # operands and this names that variadic.
    iter_args: str
    # Implicit block argument name for the induction variable on the
    # body region's entry block. None for loops without an IV (while
    # loops). For scf.for this is "iv" — the implicit_args entry on
    # the body region.
    iv: str | None = None
    # Region name for the condition region, if the loop has one
    # separate from the body. For scf.for this is None; for scf.while
    # this is the "before" region that computes the loop condition.
    condition_region: str | None = None
    # Operand name for the inclusive lower bound of a counted loop.
    # None for non-counted loop forms.
    lower_bound: str | None = None
    # Operand name for the exclusive upper bound of a counted loop.
    # None for non-counted loop forms.
    upper_bound: str | None = None
    # Operand name for the positive step of a counted loop. None for
    # non-counted loop forms.
    step: str | None = None


class RegionBranchInterface(NamedTuple):
    """Interface for ops whose regions are mutually-exclusive
    conditional branches of a single decision.

    Implementing this interface declares that every region of the op
    is an alternative of the same decision (then/else for scf.if,
    case/default for scf.switch). Ops with iterating body regions do
    NOT implement this interface — they implement LoopLikeInterface
    instead.

    The generator resolves the selector operand name to its index
    and emits a loom_region_branch_vtable_t in .rodata.
    """

    # Operand name that drives the branch decision. For scf.if this
    # is the i1 condition; for scf.switch this is the index selector.
    selector: str


_DEFAULT_INTERFACE_FIELD = object()


@unique
class MemoryAccessOperationKind(Enum):
    """Operation family represented by a MemoryAccess op shape."""

    LOAD = "load"
    STORE = "store"
    PREFETCH = "prefetch"
    ATOMIC_REDUCE = "atomic_reduce"
    ATOMIC_RMW = "atomic_rmw"
    ATOMIC_CMPXCHG = "atomic_cmpxchg"


@dataclass(frozen=True, slots=True, init=False)
class MemoryAccessInterface:
    """Interface for ops that access memory through a view-like operand.

    Each field names an operand or attr role on the op, or None when the
    role is not part of that op shape. The generator resolves names to
    compact indices in a loom_memory_access_vtable_t so passes can query
    load/store/gather/scatter/atomic shapes without switching on every
    concrete op kind. Common role names default to the same field names on
    the op; omitted optional defaults are soft and become ``none`` if the op
    does not declare the default field.
    """

    # Operand naming the accessed view or memory object.
    view: str | None
    # Operand naming the value written or atomic update contribution.
    value: str | None = None
    # Operand naming the expected compare-exchange value.
    expected: str | None = None
    # Operand naming the replacement compare-exchange value.
    replacement: str | None = None
    # Operand naming the lane/activity mask.
    mask: str | None = None
    # Operand naming the value used for inactive result lanes.
    passthrough: str | None = None
    # Operand naming per-lane offsets from the logical origin.
    offsets: str | None = None
    # Variadic operand field naming dynamic logical origin indices.
    indices: str | None = None
    # Attr naming the full-rank static logical origin indices.
    static_indices: str | None = None
    # Optional cache/coherency scope attr.
    cache_scope: str | None = None
    # Optional temporal cache-policy attr.
    cache_temporal: str | None = None
    # Atomic update kind attr.
    atomic_kind: str | None = None
    # Single atomic memory-ordering attr.
    atomic_ordering: str | None = None
    # Compare-exchange success memory-ordering attr.
    atomic_success_ordering: str | None = None
    # Compare-exchange failure memory-ordering attr.
    atomic_failure_ordering: str | None = None
    # Atomic synchronization scope attr.
    atomic_scope: str | None = None
    # Operation family represented by this op shape.
    operation_kind: MemoryAccessOperationKind | None = None
    # Fields explicitly supplied by the op declaration author.
    _explicit_fields: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        view: str | None | object = _DEFAULT_INTERFACE_FIELD,
        value: str | None | object = _DEFAULT_INTERFACE_FIELD,
        expected: str | None | object = _DEFAULT_INTERFACE_FIELD,
        replacement: str | None | object = _DEFAULT_INTERFACE_FIELD,
        mask: str | None | object = _DEFAULT_INTERFACE_FIELD,
        passthrough: str | None | object = _DEFAULT_INTERFACE_FIELD,
        offsets: str | None | object = _DEFAULT_INTERFACE_FIELD,
        indices: str | None | object = _DEFAULT_INTERFACE_FIELD,
        static_indices: str | None | object = _DEFAULT_INTERFACE_FIELD,
        cache_scope: str | None | object = _DEFAULT_INTERFACE_FIELD,
        cache_temporal: str | None | object = _DEFAULT_INTERFACE_FIELD,
        atomic_kind: str | None | object = _DEFAULT_INTERFACE_FIELD,
        atomic_ordering: str | None | object = _DEFAULT_INTERFACE_FIELD,
        atomic_success_ordering: str | None | object = _DEFAULT_INTERFACE_FIELD,
        atomic_failure_ordering: str | None | object = _DEFAULT_INTERFACE_FIELD,
        atomic_scope: str | None | object = _DEFAULT_INTERFACE_FIELD,
        operation_kind: MemoryAccessOperationKind | None = None,
    ) -> None:
        explicit_fields: set[str] = set()

        def _resolve(
            field_name: str,
            value: str | None | object,
            default: str,
        ) -> str | None:
            if value is _DEFAULT_INTERFACE_FIELD:
                return default
            explicit_fields.add(field_name)
            if value is None or isinstance(value, str):
                return value
            raise TypeError(
                f"MemoryAccessInterface field {field_name!r}: "
                f"expected str or None, got {value!r}"
            )

        object.__setattr__(self, "view", _resolve("view", view, "view"))
        object.__setattr__(self, "value", _resolve("value", value, "value"))
        object.__setattr__(self, "expected", _resolve("expected", expected, "expected"))
        object.__setattr__(
            self,
            "replacement",
            _resolve("replacement", replacement, "replacement"),
        )
        object.__setattr__(self, "mask", _resolve("mask", mask, "mask"))
        object.__setattr__(
            self,
            "passthrough",
            _resolve("passthrough", passthrough, "passthrough"),
        )
        object.__setattr__(self, "offsets", _resolve("offsets", offsets, "offsets"))
        object.__setattr__(self, "indices", _resolve("indices", indices, "indices"))
        object.__setattr__(
            self,
            "static_indices",
            _resolve("static_indices", static_indices, "static_indices"),
        )
        object.__setattr__(
            self,
            "cache_scope",
            _resolve("cache_scope", cache_scope, "cache_scope"),
        )
        object.__setattr__(
            self,
            "cache_temporal",
            _resolve("cache_temporal", cache_temporal, "cache_temporal"),
        )
        object.__setattr__(
            self,
            "atomic_kind",
            _resolve("atomic_kind", atomic_kind, "kind"),
        )
        object.__setattr__(
            self,
            "atomic_ordering",
            _resolve("atomic_ordering", atomic_ordering, "ordering"),
        )
        object.__setattr__(
            self,
            "atomic_success_ordering",
            _resolve(
                "atomic_success_ordering",
                atomic_success_ordering,
                "success_ordering",
            ),
        )
        object.__setattr__(
            self,
            "atomic_failure_ordering",
            _resolve(
                "atomic_failure_ordering",
                atomic_failure_ordering,
                "failure_ordering",
            ),
        )
        object.__setattr__(
            self,
            "atomic_scope",
            _resolve("atomic_scope", atomic_scope, "scope"),
        )
        if operation_kind is not None and not isinstance(
            operation_kind, MemoryAccessOperationKind
        ):
            raise TypeError(
                "MemoryAccessInterface field 'operation_kind': expected "
                f"MemoryAccessOperationKind or None, got {operation_kind!r}"
            )
        object.__setattr__(self, "operation_kind", operation_kind)
        object.__setattr__(self, "_explicit_fields", frozenset(explicit_fields))


# ============================================================================
# Op declaration
# ============================================================================


@dataclass(frozen=True, slots=True)
class LegacyFieldMapping:
    """Maps one parsed legacy format field onto the current op field."""

    legacy: str
    current: str


@dataclass(frozen=True, slots=True)
class LegacyFieldDefault:
    """Supplies a current field value omitted by a legacy format."""

    field: str
    value: Any


@dataclass(frozen=True, slots=True, init=False)
class LegacyFormat:
    """One legacy textual format accepted by migration tooling.

    Legacy formats are declaration metadata only. The strict current parser and
    printer continue to use Op.format; migration tooling may opt into these
    rules to parse older source and construct the current op shape.
    """

    rule_id: str
    format: tuple[FormatElement, ...]
    replaced_by: str
    expires_after: str = ""
    introduced: str = "pre-release"
    field_mappings: tuple[LegacyFieldMapping, ...] = ()
    field_defaults: tuple[LegacyFieldDefault, ...] = ()
    rewrite_hook: str = ""

    def __init__(
        self,
        rule_id: str,
        *,
        format: list[FormatElement] | tuple[FormatElement, ...],
        replaced_by: str,
        expires_after: str = "",
        introduced: str = "pre-release",
        field_mappings: list[LegacyFieldMapping] | tuple[LegacyFieldMapping, ...] = (),
        field_defaults: list[LegacyFieldDefault] | tuple[LegacyFieldDefault, ...] = (),
        rewrite_hook: str = "",
    ) -> None:
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "format", tuple(format))
        object.__setattr__(self, "replaced_by", replaced_by)
        object.__setattr__(self, "expires_after", expires_after)
        object.__setattr__(self, "introduced", introduced)
        object.__setattr__(self, "field_mappings", tuple(field_mappings))
        object.__setattr__(self, "field_defaults", tuple(field_defaults))
        object.__setattr__(self, "rewrite_hook", rewrite_hook)


def _validate_condition_refinement(
    op_name: str,
    refinement: ConditionRefinement | None,
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
) -> None:
    if refinement is None:
        return
    source_operand = next(
        (operand for operand in operands if operand.name == refinement.source),
        None,
    )
    if source_operand is None:
        raise ValueError(
            f"Op '{op_name}': condition refinement source "
            f"'{refinement.source}' does not name an operand"
        )
    if source_operand.variadic or source_operand.optional:
        raise ValueError(
            f"Op '{op_name}': condition refinement source "
            f"'{refinement.source}' must be a required non-variadic operand"
        )
    if (
        len(results) != 1
        or not isinstance(results[0], Result)
        or results[0].type_constraint is not I1
        or results[0].variadic
    ):
        raise ValueError(
            f"Op '{op_name}': condition refinement requires exactly one "
            "non-variadic i1 result"
        )
    if not refinement.materialize:
        raise ValueError(
            f"Op '{op_name}': condition refinement materializer must be non-empty"
        )


def _validate_keyed_module_record(
    op_name: str,
    operands: tuple[Operand, ...],
    results: tuple[Result | TiedResult, ...],
    attrs: tuple[AttrDef, ...],
    successors: tuple[Successor, ...],
    regions: tuple[RegionDef, ...],
    traits: tuple[Trait, ...],
    effects: tuple[Effect, ...],
    ownership_effects: tuple[OperandOwnershipEffect | ResultOwnershipEffect, ...],
) -> None:
    record_traits = [trait for trait in traits if trait.name == "KeyedModuleRecord"]
    if not record_traits:
        return
    if len(record_traits) != 1 or len(record_traits[0].args) != 1:
        raise ValueError(
            f"Op '{op_name}': KeyedModuleRecord requires exactly one key attr"
        )
    if not any(trait.name == "ModuleScope" for trait in traits):
        raise ValueError(
            f"Op '{op_name}': KeyedModuleRecord requires the ModuleScope trait"
        )
    unsupported_traits = sorted(
        trait.name
        for trait in traits
        if trait.name not in ("KeyedModuleRecord", "ModuleScope", "Pure")
    )
    if unsupported_traits:
        raise ValueError(
            f"Op '{op_name}': KeyedModuleRecord cannot carry semantic traits "
            f"{unsupported_traits}"
        )
    if operands or results or successors or regions or effects or ownership_effects:
        raise ValueError(f"Op '{op_name}': KeyedModuleRecord must be attr-only")
    key_attr_name = record_traits[0].args[0]
    key_attr = next((attr for attr in attrs if attr.name == key_attr_name), None)
    if key_attr is None:
        raise ValueError(
            f"Op '{op_name}': KeyedModuleRecord key attr "
            f"'{key_attr_name}' is not declared"
        )
    if (
        key_attr.attr_type != ATTR_TYPE_STRING
        or key_attr.optional
        or key_attr.default is not None
    ):
        raise ValueError(
            f"Op '{op_name}': KeyedModuleRecord key attr "
            f"'{key_attr_name}' must be a required string"
        )


@dataclass(frozen=True, slots=True)
class Op:
    """A complete operation declaration.

    This is the core type of the DSL. Each Op instance fully describes
    an operation's interface: what it takes, what it produces, how it
    looks in textual assembly, what constraints it enforces, and its
    structural properties.

    name: Full dotted name as it appears in text format.
          Examples: "scalar.addi", "tile.contract", "func.def"
    group: Dialect this op belongs to (for organization/docs).
    doc: Human-readable documentation string.
    operands: List of Operand descriptors.
    results: List of Result descriptors.
    attrs: List of AttrDef descriptors.
    successors: List of Successor descriptors.
    successor_selector: Operand name whose value selects between successors,
        or None for unconditional/non-selected successor edges.
    regions: List of RegionDef descriptors.
    constraints: List of Constraint instances.
    traits: List of Trait instances.
    ownership_effects: Ownership actions on operand/result fields.
    symbol_def: Symbol definition descriptor for SYMBOL_DEFINE ops.
    format: Format element list describing textual assembly.
    legacy_formats: Legacy textual formats accepted by migration tooling.
    examples: List of example IR strings for documentation.

    The format field is the critical connection point. It drives:
      - The text printer (walks elements, emits tokens).
      - The text parser (walks elements, consumes tokens).
      - Builder parameter ordering (format order = parameter order).
      - C code generation (same elements drive C printer/parser).
    """

    name: str
    group: Dialect | None = None
    doc: str = ""
    operands: tuple[Operand, ...] = ()
    results: tuple[Result | TiedResult, ...] = ()
    attrs: tuple[AttrDef, ...] = ()
    successors: tuple[Successor, ...] = ()
    successor_selector: str | None = None
    regions: tuple[RegionDef, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    traits: tuple[Trait, ...] = ()
    effects: tuple[Effect, ...] = ()
    ownership_effects: tuple[OperandOwnershipEffect | ResultOwnershipEffect, ...] = ()
    canonicalize: str = ""  # C function name for canonicalization, or "".
    effective_traits: str = (
        ""  # C function name for per-instance trait computation, or "".
    )
    facts: str = ""  # C function name for fact inference, or "".
    type_transfer: str = ""  # C function name for semantic type transfer, or "".
    condition_refinement: ConditionRefinement | None = None
    verify: str = ""  # C function name for op-specific verification, or "".
    builder_name: str | None = None  # Python builder method override, or None.
    generate_c_builder: bool = True  # False when construction requires domain context.
    phase: OpPhase | None = None
    contracts: tuple[ContractFamily, ...] = ()
    category: OpCategory | None = None
    symbol_def: SymbolDefinition | None = None
    interfaces: tuple[
        Any, ...
    ] = ()  # Interface implementations (FuncLikeInterface, etc.).
    format: tuple[FormatElement, ...] = ()
    legacy_formats: tuple[LegacyFormat, ...] = ()
    examples: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        group: Dialect | None = None,
        doc: str = "",
        operands: list[Operand] | tuple[Operand, ...] = (),
        results: list[Result | TiedResult] | tuple[Result | TiedResult, ...] = (),
        attrs: list[AttrDef] | tuple[AttrDef, ...] = (),
        successors: list[Successor] | tuple[Successor, ...] = (),
        successor_selector: str | None = None,
        regions: list[RegionDef] | tuple[RegionDef, ...] = (),
        constraints: list[Constraint] | tuple[Constraint, ...] = (),
        traits: list[Trait] | tuple[Trait, ...] = (),
        effects: list[Effect] | tuple[Effect, ...] = (),
        ownership_effects: list[OperandOwnershipEffect | ResultOwnershipEffect]
        | tuple[OperandOwnershipEffect | ResultOwnershipEffect, ...] = (),
        canonicalize: str = "",
        effective_traits: str = "",
        facts: str = "",
        type_transfer: str = "",
        condition_refinement: ConditionRefinement | None = None,
        verify: str = "",
        builder_name: str | None = None,
        generate_c_builder: bool = True,
        phase: OpPhase | None = None,
        contracts: list[ContractFamily] | tuple[ContractFamily, ...] = (),
        category: OpCategory | None = None,
        symbol_def: SymbolDefinition | None = None,
        interfaces: list[Any] | tuple[Any, ...] = (),
        format: list[FormatElement] | tuple[FormatElement, ...] = (),
        legacy_formats: list[LegacyFormat] | tuple[LegacyFormat, ...] = (),
        examples: list[str] | tuple[str, ...] = (),
    ) -> None:
        # Accept lists for ergonomics, store as tuples for immutability.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "doc", doc)
        frozen_operands = tuple(operands)
        frozen_results = tuple(results)
        frozen_attrs = tuple(attrs)
        frozen_successors = tuple(successors)
        frozen_regions = tuple(regions)
        frozen_effects = tuple(effects)
        frozen_ownership_effects = tuple(ownership_effects)
        frozen_format = tuple(format)
        frozen_legacy_formats = tuple(legacy_formats)
        object.__setattr__(self, "operands", frozen_operands)
        object.__setattr__(self, "results", frozen_results)
        object.__setattr__(self, "attrs", frozen_attrs)
        object.__setattr__(self, "successors", frozen_successors)
        object.__setattr__(self, "successor_selector", successor_selector)
        object.__setattr__(self, "regions", frozen_regions)
        object.__setattr__(self, "constraints", tuple(constraints))
        object.__setattr__(self, "traits", tuple(traits))
        object.__setattr__(self, "effects", frozen_effects)
        object.__setattr__(self, "ownership_effects", frozen_ownership_effects)
        object.__setattr__(self, "canonicalize", canonicalize)
        object.__setattr__(self, "effective_traits", effective_traits)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "type_transfer", type_transfer)
        object.__setattr__(self, "condition_refinement", condition_refinement)
        object.__setattr__(self, "verify", verify)
        object.__setattr__(self, "builder_name", builder_name)
        object.__setattr__(self, "generate_c_builder", generate_c_builder)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "contracts", tuple(contracts))
        _validate_condition_refinement(
            name, condition_refinement, frozen_operands, frozen_results
        )
        if (
            category is not None
            and group is not None
            and group.categories
            and category not in group.categories
        ):
            raise ValueError(
                f"Op '{name}': category '{category.key}' is not declared "
                f"by dialect '{group.name}'"
            )
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "symbol_def", symbol_def)
        object.__setattr__(self, "interfaces", tuple(interfaces))
        object.__setattr__(self, "format", frozen_format)
        object.__setattr__(self, "legacy_formats", frozen_legacy_formats)
        object.__setattr__(self, "examples", tuple(examples))
        has_symbol_define = any(trait.name == "SymbolDefine" for trait in traits)
        if has_symbol_define and symbol_def is None:
            raise ValueError(f"Op '{name}': SYMBOL_DEFINE requires symbol_def")
        if symbol_def is not None and not has_symbol_define:
            raise ValueError(f"Op '{name}': symbol_def requires SYMBOL_DEFINE trait")
        if symbol_def is not None:
            defining_attr = next(
                (attr for attr in frozen_attrs if attr.name == symbol_def.field),
                None,
            )
            if defining_attr is None:
                raise ValueError(
                    f"Op '{name}': symbol_def field '{symbol_def.field}' "
                    "does not name an attr"
                )
            if defining_attr.attr_type != ATTR_TYPE_SYMBOL:
                raise ValueError(
                    f"Op '{name}': symbol_def field '{symbol_def.field}' "
                    "must be a symbol attr"
                )
            if symbol_def.retain is not None:
                retain_attr = next(
                    (attr for attr in frozen_attrs if attr.name == symbol_def.retain),
                    None,
                )
                if retain_attr is None:
                    raise ValueError(
                        f"Op '{name}': symbol_def retain '{symbol_def.retain}' "
                        "does not name an attr"
                    )
                if retain_attr.attr_type != ATTR_TYPE_ENUM:
                    raise ValueError(
                        f"Op '{name}': symbol_def retain '{symbol_def.retain}' "
                        "must be an enum attr"
                    )
        func_like = next(
            (
                interface
                for interface in interfaces
                if isinstance(interface, FuncLikeInterface)
            ),
            None,
        )
        func_args_fields = _collect_func_args_fields(frozen_format)
        if func_args_fields and func_like is None:
            raise ValueError(
                f"Op '{name}': FuncArgs format elements require a "
                "FuncLikeInterface implementation"
            )
        kernel_contract = symbol_def.kernel_contract if symbol_def is not None else None
        if func_like is not None and func_like.args is not None and func_args_fields:
            expected_func_args_fields = {func_like.args}
            if kernel_contract is not None and kernel_contract.workload_operands:
                expected_func_args_fields.add(kernel_contract.workload_operands)
            if func_args_fields != expected_func_args_fields:
                raise ValueError(
                    f"Op '{name}': FuncArgs format fields must match the "
                    "declared function and kernel signature operands"
                )
        if kernel_contract is not None and func_like is None:
            raise ValueError(
                f"Op '{name}': a kernel contract requires a "
                "FuncLikeInterface implementation"
            )
        if func_like is not None and (func_like.body is None) == (
            func_like.args is None
        ):
            raise ValueError(
                f"Op '{name}': FuncLikeInterface must declare exactly one "
                "signature representation with body or args"
            )
        if func_like is not None and not func_args_fields:
            raise ValueError(
                f"Op '{name}': FuncLikeInterface requires an explicit "
                "FuncArgs format field"
            )
        if (
            func_like is not None
            and func_like.body is not None
            and len(func_args_fields) != 1
        ):
            raise ValueError(
                f"Op '{name}': body-backed FuncLikeInterface requires exactly "
                "one FuncArgs format field"
            )
        if kernel_contract is not None and func_like is not None:
            if (
                kernel_contract.workload_operands is not None
                and kernel_contract.workload_operands == func_like.args
            ):
                raise ValueError(
                    f"Op '{name}': kernel workload and launch ABI signatures "
                    "must use distinct operand fields"
                )
            if (
                kernel_contract.workload_region is not None
                and kernel_contract.workload_region == func_like.body
            ):
                raise ValueError(
                    f"Op '{name}': kernel workload and launch ABI signatures "
                    "must use distinct regions"
                )
        if func_like is not None and func_like.args is not None:
            signature_operand_names = {func_like.args}
            if kernel_contract is not None and kernel_contract.workload_operands:
                signature_operand_names.add(kernel_contract.workload_operands)
            operand_names = {operand.name for operand in frozen_operands}
            if signature_operand_names != operand_names:
                raise ValueError(
                    f"Op '{name}': operand-backed FuncLike signatures must "
                    "own every operand field"
                )
        if successor_selector is not None:
            if len(frozen_successors) < 2:
                raise ValueError(
                    f"Op '{name}': successor_selector requires at least two successors"
                )
            selector_operand = next(
                (
                    operand
                    for operand in frozen_operands
                    if operand.name == successor_selector
                ),
                None,
            )
            if selector_operand is None:
                raise ValueError(
                    f"Op '{name}': successor_selector field "
                    f"'{successor_selector}' does not name an operand"
                )
            if selector_operand.variadic:
                raise ValueError(
                    f"Op '{name}': successor_selector field "
                    f"'{successor_selector}' must not be variadic"
                )
        # Validate memory effect declarations.
        if frozen_effects:
            _validate_effects(name, frozen_effects, frozen_operands, tuple(traits))
        else:
            _validate_no_effect_conflicts(name, tuple(traits))
        if frozen_ownership_effects:
            _validate_ownership_effects(
                name,
                frozen_ownership_effects,
                frozen_operands,
                frozen_results,
                tuple(traits),
            )
        _validate_trait_field_contracts(
            name,
            frozen_operands,
            frozen_results,
            frozen_regions,
            tuple(traits),
        )
        _validate_keyed_module_record(
            name,
            frozen_operands,
            frozen_results,
            frozen_attrs,
            frozen_successors,
            frozen_regions,
            tuple(traits),
            frozen_effects,
            frozen_ownership_effects,
        )
        _validate_scoped_enum_fields(name, frozen_format, frozen_attrs)
        _validate_attr_params_fields(name, frozen_format, frozen_attrs)
        _validate_func_args_partitions(name, frozen_format, frozen_attrs)
        # Validate that format elements reference declared fields.
        if frozen_format:
            _validate_format_fields(
                name,
                frozen_format,
                frozen_operands,
                frozen_results,
                frozen_attrs,
                frozen_successors,
                frozen_regions,
            )
        if frozen_legacy_formats:
            _validate_legacy_formats(
                name,
                frozen_legacy_formats,
                frozen_operands,
                frozen_results,
                frozen_attrs,
                frozen_successors,
                frozen_regions,
            )
        _validate_region_arg_sources(
            name,
            frozen_operands,
            frozen_results,
            frozen_regions,
            frozen_format,
        )

    def __repr__(self) -> str:
        return f"Op({self.name!r})"

    # --- Lookup helpers ---

    def operand(self, name: str) -> Operand | None:
        """Find an operand by name."""
        for op in self.operands:
            if op.name == name:
                return op
        return None

    def result(self, name: str) -> Result | TiedResult | None:
        """Find a result by name."""
        for r in self.results:
            if r.name == name:
                return r
        return None

    def attr(self, name: str) -> AttrDef | None:
        """Find an attribute by name."""
        for a in self.attrs:
            if a.name == name:
                return a
        return None

    def successor(self, name: str) -> Successor | None:
        """Find a successor by name."""
        for s in self.successors:
            if s.name == name:
                return s
        return None

    def region(self, name: str) -> RegionDef | None:
        """Find a region by name."""
        for r in self.regions:
            if r.name == name:
                return r
        return None

    def has_trait(self, trait_name: str) -> bool:
        """Check if this op has a trait by name."""
        return any(t.name == trait_name for t in self.traits)

    @property
    def keyed_module_record_attr(self) -> str | None:
        """Returns the canonical module-record key attr, when declared."""

        for trait in self.traits:
            if trait.name == "KeyedModuleRecord":
                return trait.args[0]
        return None

    @property
    def effective_phase(self) -> OpPhase | None:
        """Returns the op semantic phase after applying its dialect default."""
        if self.phase is not None:
            return self.phase
        if self.group is None:
            return None
        return self.group.default_phase

    @property
    def effective_category(self) -> OpCategory | None:
        """Returns the op category after applying its dialect default."""
        if self.category is not None:
            return self.category
        if self.group is None:
            return None
        return self.group.default_category

    @property
    def is_pure(self) -> bool:
        """True if the op has no memory effects and is deterministic.

        An op is pure if it explicitly declares traits=[PURE], or if it
        has no effects, no ownership effects, no ALLOCATES results, and no HINT,
        NON_DETERMINISTIC, UNKNOWN_EFFECTS, MEMORY_FENCE, or UNIQUE_IDENTITY
        traits.
        """
        if self.has_trait("Pure"):
            return True
        if self.effects:
            return False
        if self.ownership_effects:
            return False
        if any(r.allocates for r in self.results):
            return False
        if self.has_trait("UniqueIdentity"):
            return False
        if self.has_trait("Hint"):
            return False
        if (
            self.has_trait("NonDeterministic")
            or self.has_trait("UnknownEffects")
            or self.has_trait("MemoryFence")
        ):
            return False
        return True

    @property
    def is_terminator(self) -> bool:
        return self.has_trait("Terminator")

    @property
    def is_commutative(self) -> bool:
        return self.has_trait("Commutative")

    @property
    def namespace(self) -> str:
        """The namespace prefix (everything before the last dot)."""
        dot = self.name.rfind(".")
        return self.name[:dot] if dot >= 0 else ""

    @property
    def short_name(self) -> str:
        """The short name (everything after the last dot)."""
        dot = self.name.rfind(".")
        return self.name[dot + 1 :] if dot >= 0 else self.name


# ============================================================================
# Op declaration helpers
# ============================================================================
# These reduce boilerplate for common op patterns. They produce Op
# instances — no class hierarchies, no metaclasses.


def binary_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    doc: str,
    commutative: bool = False,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a binary op: (lhs, rhs) -> result, all same type.

    Format: %r = name %a, %b : type
    Or with flags: %r = name<flag1|flag2> %a, %b : type
    """
    from loom.assembly import COLON, COMMA, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_traits = [PURE]
    if commutative:
        op_traits.append(COMMUTATIVE)
    if traits:
        op_traits.extend(traits)

    op_attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend([Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")])

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[
            Operand("lhs", type_constraint),
            Operand("rhs", type_constraint),
        ],
        results=[Result("result", type_constraint)],
        attrs=op_attrs,
        constraints=[SameType("lhs", "rhs", "result")],
        traits=op_traits,
        format=fmt,
        **kwargs,
    )


def unary_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    doc: str,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a unary op: (input) -> result, same type.

    Format: %r = name %x : type
    Or with flags: %r = name<flag1|flag2> %x : type
    """
    from loom.assembly import COLON, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_traits = [PURE]
    if traits:
        op_traits.extend(traits)

    op_attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend([Ref("input"), COLON, TypeOf("result")])

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[Operand("input", type_constraint)],
        results=[Result("result", type_constraint)],
        attrs=op_attrs,
        constraints=[SameType("input", "result")],
        traits=op_traits,
        format=fmt,
        **kwargs,
    )


def cast_op(
    name: str,
    *,
    group: Dialect,
    from_constraint: TypeConstraint,
    to_constraint: TypeConstraint,
    doc: str,
    input_role: OperandRole = OperandRole.NONE,
    traits: list[Trait] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a type-casting op: (input) -> result, different types.

    Format: %r = name %x : input_type to result_type
    """
    from loom.assembly import COLON, Ref, TypeOf, kw

    op_traits = [PURE]
    if traits:
        op_traits.extend(traits)

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[Operand("input", from_constraint, role=input_role)],
        results=[Result("result", to_constraint)],
        traits=op_traits,
        format=[
            Ref("input"),
            COLON,
            TypeOf("input"),
            kw("to"),
            TypeOf("result"),
        ],
        **kwargs,
    )


def comparison_op(
    name: str,
    *,
    group: Dialect,
    type_constraint: TypeConstraint,
    predicates: EnumDef,
    doc: str,
    flags: tuple[str, EnumDef] | None = None,
    **kwargs: Any,
) -> Op:
    """Declare a comparison op: (predicate, lhs, rhs) -> i1.

    Format: %r = name pred, %a, %b : operand_type
    Or with flags: %r = name<flag1|flag2> pred, %a, %b : operand_type

    Operands must have matching types (both satisfy type_constraint).
    The result is always i1 (integer predicate). The format prints the
    operand type after ':', and the result type (i1) is inferred.
    The builder takes both operand_type and result_type so the caller
    can construct the i1 result explicitly.
    """
    from loom.assembly import COLON, COMMA, Attr, Ref, TypeOf
    from loom.assembly import Flags as FlagsFmt

    op_attrs: list[AttrDef] = [AttrDef("predicate", "enum", enum_def=predicates)]
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        op_attrs.append(
            AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def)
        )
        fmt.append(FlagsFmt(attr_name))
    fmt.extend(
        [
            Attr("predicate"),
            COMMA,
            Ref("lhs"),
            COMMA,
            Ref("rhs"),
            COLON,
            TypeOf("lhs"),
        ]
    )

    return Op(
        name=name,
        group=group,
        doc=doc,
        operands=[
            Operand("lhs", type_constraint),
            Operand("rhs", type_constraint),
        ],
        results=[Result("result", I1)],
        attrs=op_attrs,
        constraints=[SameType("lhs", "rhs")],
        traits=[PURE, SAFE_TO_SPECULATE],
        format=fmt,
        **kwargs,
    )
