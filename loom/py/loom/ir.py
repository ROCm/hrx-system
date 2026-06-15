# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom in-memory IR representation.

The Python IR mirrors the C IR concepts and serialization tables, but uses
dataclasses and Python containers instead of arenas and packed structs.

Hierarchy: Context -> Module -> Symbol/Region/Block/Operation/Value.

Table-owned entities such as strings, types, locations, and values are
referenced by integer IDs. Container ownership uses ordinary Python object
references: symbols own their defining op, ops own regions, regions own
blocks, and CFG successor edges reference their target blocks directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import Any

__all__ = [
    # Scalar types.
    "ScalarTypeKind",
    "ScalarType",
    "scalar_type_name",
    "parse_scalar_type_kind",
    "INDEX",
    "OFFSET",
    "I1",
    "I8",
    "I16",
    "I32",
    "I64",
    "F16",
    "BF16",
    "F32",
    "F64",
    # Dimensions.
    "StaticDim",
    "DynamicDim",
    "Dim",
    # Type kinds and types.
    "TypeKind",
    "ShapedType",
    "BufferType",
    "BUFFER_TYPE",
    "GroupScope",
    "GROUP_SCOPE_BY_NAME",
    "GroupType",
    "StorageSpace",
    "STORAGE_SPACE_BY_NAME",
    "StorageType",
    "PoolType",
    "FunctionType",
    "NoneType",
    "RegisterType",
    "DialectType",
    "EncodingRole",
    "ENCODING_ROLE_BY_NAME",
    "EncodingType",
    "ENCODING_TYPE",
    "ENCODING_LAYOUT_TYPE",
    "ENCODING_SCHEMA_TYPE",
    "ENCODING_STORAGE_TYPE",
    "ENCODING_TRANSFORM_TYPE",
    "NONE_TYPE",
    "Type",
    "binding_element_type",
    # Locations.
    "LocationKind",
    "FileLocation",
    "FusedLocation",
    "OpaqueLocation",
    "LocationData",
    "LOCATION_UNKNOWN",
    "LOCATION_FLAG_SYNTHETIC",
    # Encoding instances.
    "DynamicEncoding",
    "EncodingInstance",
    # Predicates.
    "CanonicalAttrDict",
    "canonicalize_attr_dict",
    "replace_canonical_attr_dict",
    "PredicateArg",
    "Predicate",
    "PREDICATE_KINDS",
    "evaluate_predicate",
    "evaluate_predicates",
    # Tied results.
    "TiedResult",
    # Use-def.
    "Use",
    "VALUE_DEF_BLOCK_NONE",
    "VALUE_DEF_OP_NONE",
    "VALUE_FLAG_BLOCK_ARG",
    "VALUE_FLAG_CONSUMED",
    # IR graph.
    "Value",
    "Operation",
    "Block",
    "Region",
    # Symbols.
    "SymbolKind",
    "Symbol",
    "symbol_from_operation",
    "SymbolRef",
    "SymbolName",
    "SYMBOL_FLAG_IMPORT",
    "SYMBOL_FLAG_PUBLIC",
    # Tables.
    "StringTable",
    "TypeTable",
    "LocationTable",
    # Module and context.
    "Module",
    "Context",
    # Value metadata helpers.
    "record_block_value_metadata",
    "record_operation_value_metadata",
    "rebuild_value_metadata",
]


# ============================================================================
# Scalar types
# ============================================================================


@unique
class ScalarTypeKind(IntEnum):
    """Scalar element type kind.

    Values match loom_scalar_type_e in ir/types.h. These are internal
    compiler values, NOT bytecode-stable.

    Ordered: address types, integers by width, floats by width.
    """

    INDEX = 0
    OFFSET = 1
    I1 = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    F8E4M3 = 7
    F8E5M2 = 8
    F16 = 9
    BF16 = 10
    F32 = 11
    F64 = 12


# Scalar type name -> enum mapping.
_SCALAR_NAMES: dict[str, ScalarTypeKind] = {
    "index": ScalarTypeKind.INDEX,
    "offset": ScalarTypeKind.OFFSET,
    "i1": ScalarTypeKind.I1,
    "i8": ScalarTypeKind.I8,
    "i16": ScalarTypeKind.I16,
    "i32": ScalarTypeKind.I32,
    "i64": ScalarTypeKind.I64,
    "f8E4M3": ScalarTypeKind.F8E4M3,
    "f8E5M2": ScalarTypeKind.F8E5M2,
    "f16": ScalarTypeKind.F16,
    "bf16": ScalarTypeKind.BF16,
    "f32": ScalarTypeKind.F32,
    "f64": ScalarTypeKind.F64,
}

# Reverse mapping.
_SCALAR_KIND_NAMES: dict[ScalarTypeKind, str] = {v: k for k, v in _SCALAR_NAMES.items()}


def scalar_type_name(kind: ScalarTypeKind) -> str:
    """Return the textual name for a scalar type kind."""
    return _SCALAR_KIND_NAMES[kind]


def parse_scalar_type_kind(name: str) -> ScalarTypeKind | None:
    """Parse a scalar type name, returning None if not recognized."""
    return _SCALAR_NAMES.get(name)


# ============================================================================
# Dimensions
# ============================================================================


@dataclass(frozen=True, slots=True)
class StaticDim:
    """A static (compile-time known) dimension size."""

    size: int

    def __repr__(self) -> str:
        return str(self.size)


@dataclass(frozen=True, slots=True)
class DynamicDim:
    """A dynamic dimension. The actual size is bound at the use site,
    not in the type. Types are structural — DynamicDim carries no
    value reference."""

    def __repr__(self) -> str:
        return "?"


# A dimension is either static or dynamic.
type Dim = StaticDim | DynamicDim


# ============================================================================
# Type kinds
# ============================================================================


@unique
class TypeKind(IntEnum):
    """Type kind tag. Matches loom_type_kind_e in ir/types.h."""

    NONE = 0
    SCALAR = 1
    TILE = 2
    TENSOR = 3
    GROUP = 4
    FUNCTION = 5
    DIALECT = 6
    ENCODING = 7
    POOL = 8
    VECTOR = 9
    VIEW = 10
    BUFFER = 11
    REGISTER = 12
    STORAGE = 13
    PLACEHOLDER = 14


# ============================================================================
# Types — frozen, internable, structurally compared
# ============================================================================


@dataclass(frozen=True, slots=True)
class NoneType:
    """Absence of a type (for ops with no results)."""

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.NONE

    def __repr__(self) -> str:
        return "none"


# Singleton for no-type.
NONE_TYPE = NoneType()


@dataclass(frozen=True, slots=True)
class ScalarType:
    """A scalar type: f16, f32, i8, index, etc."""

    kind: ScalarTypeKind

    def __repr__(self) -> str:
        return scalar_type_name(self.kind)

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.SCALAR

    @property
    def bitwidth(self) -> int:
        widths = {
            ScalarTypeKind.INDEX: 64,
            ScalarTypeKind.OFFSET: 64,
            ScalarTypeKind.I1: 1,
            ScalarTypeKind.I8: 8,
            ScalarTypeKind.I16: 16,
            ScalarTypeKind.I32: 32,
            ScalarTypeKind.I64: 64,
            ScalarTypeKind.F8E4M3: 8,
            ScalarTypeKind.F8E5M2: 8,
            ScalarTypeKind.F16: 16,
            ScalarTypeKind.BF16: 16,
            ScalarTypeKind.F32: 32,
            ScalarTypeKind.F64: 64,
        }
        return widths[self.kind]


# Common scalar type singletons.
INDEX = ScalarType(ScalarTypeKind.INDEX)
OFFSET = ScalarType(ScalarTypeKind.OFFSET)
I1 = ScalarType(ScalarTypeKind.I1)
I8 = ScalarType(ScalarTypeKind.I8)
I16 = ScalarType(ScalarTypeKind.I16)
I32 = ScalarType(ScalarTypeKind.I32)
I64 = ScalarType(ScalarTypeKind.I64)
F16 = ScalarType(ScalarTypeKind.F16)
BF16 = ScalarType(ScalarTypeKind.BF16)
F32 = ScalarType(ScalarTypeKind.F32)
F64 = ScalarType(ScalarTypeKind.F64)


@dataclass(frozen=True, slots=True)
class ShapedType:
    """A shaped type with shape, element type, and optional encoding/layout.

    Types are structural: dynamic dims are DynamicDim() flags, not
    references to specific SSA values. The binding of dynamic dims to
    values happens on the Value that carries this type.

    Tensor, tile, and vector shaped types describe SSA values at different
    lowering levels. View shaped types describe typed, non-owning logical
    coordinate spaces over buffer storage; the view-producing op still carries
    the storage identity and offset/dynamic binding information needed for
    alias reasoning.

    Tile/tensor/view types may carry an encoding/layout attachment. The
    attachment is None for the default representation, an EncodingInstance
    for a statically-known attachment, or DynamicEncoding for an attachment
    bound via SSA value. Since all variants are frozen and hashable,
    ShapedType remains internable. Vector types are pure register lane grids
    and do not carry encoding/layout attachments.
    """

    type_kind: TypeKind  # TILE, TENSOR, VECTOR, or VIEW
    element_type: ScalarType
    dims: tuple[Dim, ...]
    encoding: EncodingInstance | DynamicEncoding | None = None

    def __post_init__(self) -> None:
        if self.type_kind not in (
            TypeKind.TILE,
            TypeKind.TENSOR,
            TypeKind.VECTOR,
            TypeKind.VIEW,
        ):
            raise ValueError(
                "ShapedType type_kind must be TILE, TENSOR, VECTOR, or VIEW, "
                f"got {self.type_kind}"
            )
        if self.type_kind == TypeKind.VECTOR:
            if not self.dims:
                raise ValueError("vector types must have rank >= 1")
            if self.encoding is not None:
                raise ValueError("vector types must not carry encodings/layouts")

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def has_encoding(self) -> bool:
        return self.encoding is not None

    @property
    def has_dynamic_encoding(self) -> bool:
        return isinstance(self.encoding, DynamicEncoding)

    @property
    def has_static_encoding(self) -> bool:
        return isinstance(self.encoding, EncodingInstance)

    @property
    def is_all_static(self) -> bool:
        return all(isinstance(d, StaticDim) for d in self.dims)

    def __repr__(self) -> str:
        kind_name = _SHAPED_TYPE_KIND_NAMES[self.type_kind]
        if self.dims:
            dim_strs = []
            for d in self.dims:
                match d:
                    case StaticDim(size=s):
                        dim_strs.append(str(s))
                    case DynamicDim():
                        dim_strs.append("?")
            shape = "x".join(dim_strs)
            return f"{kind_name}<{shape}x{self.element_type}>"
        return f"{kind_name}<{self.element_type}>"


_SHAPED_TYPE_KIND_NAMES: dict[TypeKind, str] = {
    TypeKind.TILE: "tile",
    TypeKind.TENSOR: "tensor",
    TypeKind.VECTOR: "vector",
    TypeKind.VIEW: "view",
}


@dataclass(frozen=True, slots=True)
class BufferType:
    """An opaque untyped storage identity used as the root for typed views."""

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.BUFFER

    def __repr__(self) -> str:
        return "buffer"


# Singleton for the first-class buffer type.
BUFFER_TYPE = BufferType()


class GroupScope(IntEnum):
    """Group scope kind. Must match loom_group_scope_t in types.h."""

    WORKGROUP = 0
    SUBGROUP = 1

    @property
    def text(self) -> str:
        """The canonical text representation for printing."""
        return _GROUP_SCOPE_NAMES[self]


_GROUP_SCOPE_NAMES: dict[GroupScope, str] = {
    GroupScope.WORKGROUP: "workgroup",
    GroupScope.SUBGROUP: "subgroup",
}

# Reverse mapping for parsing: "workgroup" -> GroupScope.WORKGROUP.
GROUP_SCOPE_BY_NAME: dict[str, GroupScope] = {
    name: scope for scope, name in _GROUP_SCOPE_NAMES.items()
}


@dataclass(frozen=True, slots=True)
class GroupType:
    """A group type: group<scope>."""

    scope: GroupScope

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.GROUP

    def __repr__(self) -> str:
        return f"group<{self.scope.text}>"


class StorageSpace(IntEnum):
    """Function-local byte storage space. Must match loom_storage_space_t."""

    STACK = 0
    SCRATCH = 1
    PRIVATE = 2
    WORKGROUP = 3

    @property
    def text(self) -> str:
        """The canonical text representation for printing."""
        return _STORAGE_SPACE_NAMES[self]


_STORAGE_SPACE_NAMES: dict[StorageSpace, str] = {
    StorageSpace.STACK: "stack",
    StorageSpace.SCRATCH: "scratch",
    StorageSpace.PRIVATE: "private",
    StorageSpace.WORKGROUP: "workgroup",
}

STORAGE_SPACE_BY_NAME: dict[str, StorageSpace] = {
    name: space for space, name in _STORAGE_SPACE_NAMES.items()
}


@dataclass(frozen=True, slots=True)
class StorageType:
    """A function-local byte storage handle: low.storage<space>."""

    space: StorageSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "space", StorageSpace(self.space))

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.STORAGE

    def __repr__(self) -> str:
        return f"low.storage<{self.space.text}>"


@dataclass(frozen=True, slots=True)
class FunctionType:
    """A function type: (arg_types) -> (result_types)."""

    arg_types: tuple[Type, ...]
    result_types: tuple[Type, ...]

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.FUNCTION

    def __repr__(self) -> str:
        args = ", ".join(repr(t) for t in self.arg_types)
        results = ", ".join(repr(t) for t in self.result_types)
        return f"({args}) -> ({results})"


@dataclass(frozen=True, slots=True)
class RegisterType:
    """A target-low register value with compact descriptor identity.

    Register types carry only physical allocation shape, not value semantics.
    The op descriptor determines whether a register is interpreted as i32,
    f32, v128, an address, or an instruction-specific packed payload.
    """

    descriptor_set_stable_id: int
    register_class_id: int
    unit_count: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        if self.descriptor_set_stable_id < 1:
            raise ValueError("register descriptor set stable ID must be non-zero")
        if self.register_class_id < 0 or self.register_class_id >= 2**16:
            raise ValueError("register class ID must fit uint16_t")
        if self.unit_count < 1:
            raise ValueError("register unit count must be >= 1")
        if self.unit_count >= 2**32:
            raise ValueError("register unit count must fit uint32_t")
        if self.name is not None:
            parts = self.name.split(".")
            if len(parts) < 2 or any(not part for part in parts):
                raise ValueError(
                    f"register class must be namespace-qualified: {self.name!r}"
                )

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.REGISTER

    def __repr__(self) -> str:
        reg_class = (
            self.name
            if self.name is not None
            else f"0x{self.descriptor_set_stable_id:x}:{self.register_class_id}"
        )
        if self.unit_count == 1:
            return f"reg<{reg_class}>"
        return f"reg<{reg_class} x{self.unit_count}>"


@dataclass(frozen=True, slots=True)
class DialectType:
    """A dialect-defined type (hal.buffer, vm.ref<T>, etc.).

    Catch-all for types defined by dialects outside the built-in set.
    Built-in types (ScalarType, ShapedType, GroupType, FunctionType)
    keep their specific dataclasses for performance and ergonomics.
    Dialect types use DialectType with the TypeDef driving print/parse.

    name: Full type name as it appears in text ("hal.buffer", "vm.ref").
    params: Type parameters (e.g., the hal.buffer in vm.ref<hal.buffer>).
    """

    name: str
    params: tuple[Type, ...] = ()

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.DIALECT

    def __repr__(self) -> str:
        if self.params:
            param_strs = ", ".join(repr(p) for p in self.params)
            return f"{self.name}<{param_strs}>"
        return self.name


@unique
class EncodingRole(IntEnum):
    """Semantic role carried by an encoding SSA value type."""

    UNKNOWN = 0
    LAYOUT = 1
    SCHEMA = 2
    STORAGE = 3
    TRANSFORM = 4

    @property
    def text(self) -> str:
        match self:
            case EncodingRole.UNKNOWN:
                return ""
            case EncodingRole.LAYOUT:
                return "layout"
            case EncodingRole.SCHEMA:
                return "schema"
            case EncodingRole.STORAGE:
                return "storage"
            case EncodingRole.TRANSFORM:
                return "transform"
        raise ValueError(f"unknown encoding role: {self!r}")


ENCODING_ROLE_BY_NAME: dict[str, EncodingRole] = {
    role.text: role for role in EncodingRole if role != EncodingRole.UNKNOWN
}


@dataclass(frozen=True, slots=True)
class EncodingType:
    """The type of an encoding SSA value.

    Encoding values may be role-qualified as encoding<layout>,
    encoding<schema>, encoding<storage>, or encoding<transform>. Values of
    EncodingType are referenced in encoding/layout attachment positions such as
    tile<4xf32, %enc> and view<[%N]xf32, %layout>.
    """

    role: EncodingRole = EncodingRole.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", EncodingRole(self.role))

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.ENCODING

    def __repr__(self) -> str:
        if self.role != EncodingRole.UNKNOWN:
            return f"encoding<{self.role.text}>"
        return "encoding"


# Singletons for encoding type variants.
ENCODING_TYPE = EncodingType()
ENCODING_LAYOUT_TYPE = EncodingType(EncodingRole.LAYOUT)
ENCODING_SCHEMA_TYPE = EncodingType(EncodingRole.SCHEMA)
ENCODING_STORAGE_TYPE = EncodingType(EncodingRole.STORAGE)
ENCODING_TRANSFORM_TYPE = EncodingType(EncodingRole.TRANSFORM)


@dataclass(frozen=True, slots=True)
class PoolType:
    """A block-managed device memory pool: pool<[%block_size]>.

    One parameter: the block size in bytes, which may be static or
    dynamic. The pool carries no capacity, no element type, no
    encoding — it's untyped bytes. Element type and encoding are
    imposed by pool ops at access time.

    Dynamic block_size uses the same DynamicDim/dim_bindings mechanism
    as shaped types: the Value carrying this type binds the dim at
    position 0 to an index-typed SSA value.
    """

    block_size: Dim

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.POOL

    @property
    def has_dynamic_block_size(self) -> bool:
        return isinstance(self.block_size, DynamicDim)

    def __repr__(self) -> str:
        match self.block_size:
            case StaticDim(size=size):
                return f"pool<{size}>"
            case DynamicDim():
                return "pool<?>"


@dataclass(frozen=True, slots=True)
class PlaceholderType:
    """A placeholder type for forward references in signatures.

    Used only during assembly parsing when a name is used in a type dim
    before it has been defined in the argument or result list. All
    placeholders must be resolved by the end of the signature.
    """

    @property
    def type_kind(self) -> TypeKind:
        return TypeKind.PLACEHOLDER

    def __repr__(self) -> str:
        return "<<placeholder>>"


# Union of all type kinds.
type Type = (
    ScalarType
    | ShapedType
    | BufferType
    | GroupType
    | StorageType
    | FunctionType
    | RegisterType
    | DialectType
    | EncodingType
    | PoolType
    | PlaceholderType
    | NoneType
)


def binding_element_type(operand_type: Type) -> Type:
    """Extract the element type for an 'element' binding.

    For shaped types (tile, tensor, vector, view): returns the scalar element type.
    For scalar types: returns the type itself (identity).
    For dialect types: returns the first type parameter if available,
      otherwise the type itself. Custom types can override this by
      implementing element extraction on their TypeDef.

    Used by the parser when BindingList kind='element' to determine
    the block arg type from the operand type.
    """
    match operand_type:
        case ShapedType(element_type=elem):
            return elem
        case ScalarType():
            return operand_type
        case DialectType(params=params) if params:
            return params[0]
        case _:
            return operand_type


# ============================================================================
# Source locations
# ============================================================================


@unique
class LocationKind(IntEnum):
    """Location kind tag. Matches loom_location_kind_e."""

    NONE = 0
    FILE = 1
    FUSED = 2
    OPAQUE = 3


# Location flag bits (matches loom_location_flag_bits_e in ir.h).
LOCATION_FLAG_SYNTHETIC = 1 << 0


@dataclass(frozen=True, slots=True)
class FileLocation:
    """Source range location: source:start_line:start_col to end_line:end_col."""

    source_id: int
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    flags: int = 0


@dataclass(frozen=True, slots=True)
class FusedLocation:
    """Derived from multiple source locations."""

    children: tuple[int, ...]
    flags: int = 0


@dataclass(frozen=True, slots=True)
class OpaqueLocation:
    """External system identifier (torch node, JAX trace)."""

    source_id: int
    data: bytes
    flags: int = 0


# Union of location data.
type LocationData = FileLocation | FusedLocation | OpaqueLocation | None

LOCATION_UNKNOWN = 0  # Location ID 0 is always unknown.


# ============================================================================
# Encoding instances
# ============================================================================


@dataclass(frozen=True, slots=True)
class DynamicEncoding:
    """A dynamic encoding bound at the use site via an SSA value of
    type EncodingType. The Value carrying this type holds the binding
    in its encoding_binding field.

    Analogous to DynamicDim for shape dimensions: the type is
    structural and carries no value reference. The binding lives on
    the Value, not the Type.
    """


@dataclass(frozen=True, slots=True, eq=False)
class EncodingInstance:
    """A parameterized encoding instance in the module's encoding table.

    Each unique encoding usage in a program gets one canonical module-table
    entry. `ShapedType.encoding` holds the encoding object directly in Python;
    bytecode writers assign table indices during numbering.

    name: Encoding name ("q8_0", "q6_k", "dense"). Always present.
        This is the name used in textual format: #q8_0, #q6_k.
    alias: Attribute alias for pretty-printing ("enc"), or empty.
        When set, the printer uses the alias instead of the full
        #name<params> form. Defined at file level: #enc = #q8_0<block=32>
        Aliases are display-only and do not participate in equality/hash.
    params: Canonical static attributes as sorted key-value pairs. Values use
        the same Python attribute domain as op attrs: ints, floats, bools,
        strings, EncodingInstance, CanonicalAttrDict, and i64 arrays. Dynamic
        encoding parameters are named SSA operands on encoding.define, not
        nested values in this static attribute payload.
    """

    name: str
    alias: str = ""
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"EncodingInstance.name must be a non-empty string, got {self.name!r}"
            )
        if self.alias and self.alias.startswith("#"):
            raise ValueError(
                "EncodingInstance.alias stores the bare alias name without '#', "
                f"got {self.alias!r}"
            )

        canonical_params: list[tuple[str, Any]] = []
        seen_names: set[str] = set()
        for param_name, param_value in self.params:
            if not isinstance(param_name, str) or not param_name:
                raise ValueError(
                    "encoding parameter names must be non-empty strings, "
                    f"got {param_name!r}"
                )
            if param_name in seen_names:
                raise ValueError(f"duplicate encoding parameter {param_name!r}")
            seen_names.add(param_name)
            canonical_params.append(
                (param_name, _canonicalize_encoding_param_value(param_value))
            )
        canonical_params.sort(key=lambda entry: entry[0])
        object.__setattr__(self, "params", tuple(canonical_params))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EncodingInstance):
            return NotImplemented
        return self.name == other.name and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.name, self.params))


def _canonicalize_encoding_param_value(value: Any) -> Any:
    """Canonicalize encoding parameter values into hashable attr values."""
    if isinstance(value, CanonicalAttrDict):
        return value
    if isinstance(value, Mapping):
        return CanonicalAttrDict(value.items())
    if isinstance(value, list | tuple):
        return tuple(_canonicalize_encoding_param_value(item) for item in value)
    return value


# ============================================================================
# Predicates
# ============================================================================

# Valid predicate kinds. Maps kind name to expected argument count.
PREDICATE_KINDS: dict[str, int] = {
    "eq": 2,
    "ne": 2,
    "lt": 2,
    "le": 2,
    "gt": 2,
    "ge": 2,
    "mul": 2,  # mul(a, n) — a is a multiple of n.
    "min": 2,  # min(a, n) — a >= n.
    "max": 2,  # max(a, n) — a <= n.
    "pow2": 1,  # pow2(a) — a is a power of 2.
    "range": 3,  # range(a, lo, hi) — lo <= a <= hi.
}


@dataclass(frozen=True, slots=True)
class PredicateArg:
    """A single argument to a predicate.

    tag: "value" for SSA value references, "const" for integer constants.
    value: For "value" tag, the bare SSA name as a string (no % prefix).
           For "const" tag, the integer constant value.
    """

    tag: str  # "value", "const"
    value: int | str


@dataclass(frozen=True, slots=True)
class Predicate:
    """A predicate constraint on dynamic dimension values.

    Used in function where clauses and assume ops to express
    constraints like "M is a multiple of 16" or "K is between 32 and 512".

    kind: Predicate kind name ("eq", "lt", "mul", "pow2", "range", etc.).
    args: Tuple of predicate arguments.
    """

    kind: str
    args: tuple[PredicateArg, ...]


def _resolve_predicate_arg(arg: PredicateArg, values: dict[str, int]) -> int | None:
    """Resolve a predicate argument to a concrete integer.

    Returns None if the value name is not in the values dict.
    """
    match arg.tag:
        case "const":
            assert isinstance(arg.value, int)
            return arg.value
        case "value":
            return values.get(str(arg.value))
        case _:
            raise ValueError(f"unknown predicate arg tag: {arg.tag!r}")


def evaluate_predicate(predicate: Predicate, values: dict[str, int]) -> bool:
    """Evaluate a predicate against concrete dimension values.

    values maps bare SSA names ("M", "K") to their integer values.
    Returns True if the predicate is satisfied or if any argument
    cannot be resolved (value name not in the dict).
    """
    resolved = [_resolve_predicate_arg(a, values) for a in predicate.args]
    if any(v is None for v in resolved):
        return True  # Unresolvable args — defer judgment.
    args = [v for v in resolved if v is not None]

    match predicate.kind:
        case "eq":
            return args[0] == args[1]
        case "ne":
            return args[0] != args[1]
        case "lt":
            return args[0] < args[1]
        case "le":
            return args[0] <= args[1]
        case "gt":
            return args[0] > args[1]
        case "ge":
            return args[0] >= args[1]
        case "mul":
            return args[1] != 0 and args[0] % args[1] == 0
        case "min":
            return args[0] >= args[1]
        case "max":
            return args[0] <= args[1]
        case "pow2":
            return args[0] > 0 and (args[0] & (args[0] - 1)) == 0
        case "range":
            return args[1] <= args[0] <= args[2]
        case _:
            raise ValueError(f"unknown predicate kind: {predicate.kind!r}")


def evaluate_predicates(predicates: list[Predicate], values: dict[str, int]) -> bool:
    """Evaluate all predicates. Returns True iff all are satisfied."""
    return all(evaluate_predicate(p, values) for p in predicates)


# ============================================================================
# Tied results
# ============================================================================


@dataclass(frozen=True, slots=True)
class TiedResult:
    """A result that reuses an operand's storage.

    result_index: which result is tied.
    operand_index: which operand is consumed.
    has_type_change: True if the result type differs from the operand type.
    """

    result_index: int
    operand_index: int
    has_type_change: bool = False


# ============================================================================
# Canonical attribute dictionaries
# ============================================================================


class CanonicalAttrDict(Mapping[str, Any]):
    """An immutable dict attribute with canonical key order.

    Keys are stored in ascending string order, and duplicate keys are rejected at
    construction time. Nested dict values are recursively canonicalized so text
    parsing, bytecode reading, and programmatic construction all converge to the
    same representation.
    """

    __slots__ = ("_items", "_index")

    def __init__(self, entries: Iterable[tuple[str, Any]] = ()) -> None:
        items: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for key, value in entries:
            if not isinstance(key, str):
                raise TypeError(
                    f"attribute dict keys must be strings, got {type(key).__name__}"
                )
            if key in seen:
                raise ValueError(f"duplicate attribute dict key {key!r}")
            seen.add(key)
            items.append((key, _canonicalize_attr_value(value)))
        items.sort(key=lambda entry: entry[0])
        self._items = tuple(items)
        self._index = dict(self._items)

    @classmethod
    def from_sorted_items(cls, entries: Iterable[tuple[str, Any]]) -> CanonicalAttrDict:
        """Build a canonical dict from already sorted and deduped entries."""
        items: list[tuple[str, Any]] = []
        previous_key: str | None = None
        for key, value in entries:
            if not isinstance(key, str):
                raise TypeError(
                    f"attribute dict keys must be strings, got {type(key).__name__}"
                )
            if previous_key is not None and key <= previous_key:
                if key == previous_key:
                    raise ValueError(f"duplicate attribute dict key {key!r}")
                raise ValueError(
                    "attribute dict keys must be sorted in canonical order: "
                    f"{previous_key!r} appears before {key!r}"
                )
            items.append((key, _canonicalize_attr_value(value)))
            previous_key = key
        attr_dict = cls.__new__(cls)
        attr_dict._items = tuple(items)
        attr_dict._index = dict(attr_dict._items)
        return attr_dict

    def __getitem__(self, key: str) -> Any:
        return self._index[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            "{" + ", ".join(f"{key!r}: {value!r}" for key, value in self._items) + "}"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            tuple(
                (key, _canonicalize_encoding_param_value(value))
                for key, value in self._items
            )
        )


def canonicalize_attr_dict(attributes: Mapping[str, Any] | None) -> CanonicalAttrDict:
    """Canonicalize an operation or nested dict attribute."""
    if attributes is None:
        return CanonicalAttrDict()
    if isinstance(attributes, CanonicalAttrDict):
        return attributes
    return CanonicalAttrDict(attributes.items())


def replace_canonical_attr_dict(
    attributes: Mapping[str, Any] | None,
    replacements: Mapping[str, Any | None],
) -> CanonicalAttrDict:
    """Build a new canonical dict attr from key updates/removals.

    replacements maps key -> new value, with None meaning remove the key.
    """
    merged = dict(canonicalize_attr_dict(attributes).items())
    for key, value in replacements.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = _canonicalize_attr_value(value)
    return CanonicalAttrDict(merged.items())


def _canonicalize_attr_value(value: Any) -> Any:
    """Recursively canonicalize nested dict-valued attributes."""
    if isinstance(value, CanonicalAttrDict):
        return value
    if isinstance(value, Mapping):
        return canonicalize_attr_dict(value)
    if isinstance(value, list):
        return [_canonicalize_attr_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_attr_value(item) for item in value)
    return value


# ============================================================================
# Use-def tracking
# ============================================================================


@dataclass(frozen=True, slots=True)
class Use:
    """An operand use of a value.

    user_op_index: index of the consuming op in its immediate parent block.
      For module-level symbol ops, this is the symbol index in module.symbols.
    operand_index: which operand slot is using this value.
    block_index: block index in the op's immediate parent region.
      VALUE_DEF_BLOCK_NONE means the user is a module-level symbol op or an op
      in a detached builder block with no parent region yet.
    """

    user_op_index: int
    operand_index: int
    block_index: int


# ============================================================================
# Values
# ============================================================================

# Value def-site sentinels and flags.
#
# VALUE_DEF_BLOCK_NONE means a value definition or use is not owned by a block
# in an attached region. Module-level symbol ops and detached builder blocks use
# this sentinel for block_index.
VALUE_DEF_BLOCK_NONE = -1

# VALUE_DEF_OP_NONE means a value is not defined by an operation result slot.
# Block arguments use this sentinel for def_op_index.
VALUE_DEF_OP_NONE = -1

VALUE_FLAG_BLOCK_ARG = 1 << 0
VALUE_FLAG_CONSUMED = 1 << 1


@dataclass(slots=True)
class Value:
    """An SSA value (operation result or block argument).

    Values live in the module's value table, accessed by integer ID.
    The type is structural (no SSA value references in dims).
    dim_bindings maps dynamic dim positions to the value IDs that
    provide the runtime sizes.

    name: Bare name without the '%' sigil. A value named "x" prints
    as %x; a value named "" is unnamed and gets an auto-name from
    its value ID (%0, %1, ...).

    encoding_binding: when the value's type has DynamicEncoding, this
    holds the value_id of the encoding-typed SSA value that provides
    the encoding at runtime. -1 means no binding.

    def_op_index/def_block_index/def_result_index locate the definition:
      - Operation result values:
          def_op_index = result op index in the defining block, or the symbol
            index in module.symbols for module-level symbol results
          def_block_index = defining block index in the immediate parent region,
            or VALUE_DEF_BLOCK_NONE for module-level symbol results
          def_result_index = result slot index on the defining op
      - Block argument values:
          flags has VALUE_FLAG_BLOCK_ARG
          def_op_index = VALUE_DEF_OP_NONE
          def_block_index = owning block index in the immediate parent region
          def_result_index = block argument index
      - Bodyless symbol signature arguments stored as op operands:
          flags does not have VALUE_FLAG_BLOCK_ARG
          def_op_index = symbol index in module.symbols
          def_block_index = VALUE_DEF_BLOCK_NONE
          def_result_index = signature argument index in op.operands

    Freshly allocated values may leave the def-site fields at the *_NONE
    sentinels until the parser/builder/bytecode reader attaches them to an IR
    container.
    """

    name: str
    type: Type
    flags: int = 0
    def_op_index: int = VALUE_DEF_OP_NONE
    def_block_index: int = VALUE_DEF_BLOCK_NONE
    def_result_index: int = 0
    dim_bindings: dict[int, int] = field(default_factory=dict)
    encoding_binding: int = -1
    location_id: int = LOCATION_UNKNOWN
    uses: list[Use] = field(default_factory=list)

    @property
    def is_block_arg(self) -> bool:
        return (self.flags & VALUE_FLAG_BLOCK_ARG) != 0

    @property
    def is_consumed(self) -> bool:
        return (self.flags & VALUE_FLAG_CONSUMED) != 0


# ============================================================================
# Operations
# ============================================================================


@dataclass(slots=True)
class Operation:
    """A single IR operation.

    Operands and results are value IDs (indices into module's value table).
    Successors are direct references to blocks in the enclosing region.
    The op kind is an integer ID indexing into the context's op vtable.
    """

    kind: int = 0
    operands: list[int] = field(default_factory=list)
    operand_segment_counts: tuple[int, ...] = ()
    results: list[int] = field(default_factory=list)
    tied_results: list[TiedResult] = field(default_factory=list)
    successors: list[Block] = field(default_factory=list)
    attributes: Mapping[str, Any] = field(default_factory=CanonicalAttrDict)
    regions: list[Region] = field(default_factory=list)
    location_id: int = LOCATION_UNKNOWN
    comments: tuple[str, ...] = ()
    name: str = ""
    is_dead: bool = False
    _attributes_frozen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", canonicalize_attr_dict(self.attributes))
        object.__setattr__(self, "_attributes_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "attributes" and getattr(self, "_attributes_frozen", False):
            raise AttributeError(
                "Operation.attributes is immutable after construction; build a "
                "replacement Operation or use a dedicated attr replacement helper"
            )
        object.__setattr__(self, name, value)


# ============================================================================
# Blocks and regions
# ============================================================================


@dataclass(slots=True)
class Block:
    """A basic block: a sequence of operations with optional arguments."""

    label: str = ""
    arg_ids: list[int] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)
    comments: tuple[str, ...] = ()


@dataclass(slots=True)
class Region:
    """An ordered list of blocks."""

    blocks: list[Block] = field(default_factory=list)


# ============================================================================
# Symbols
# ============================================================================


@unique
class SymbolKind(IntEnum):
    """Symbol kind encoded in bytecode SYMBOLS entries."""

    NONE = -1
    FUNC_DEF = 0
    FUNC_DECL = 1
    FUNC_TEMPLATE = 2
    FUNC_UKERNEL = 3
    GLOBAL = 4
    EXECUTABLE = 5
    RECORD = 6


# Symbol flags.
SYMBOL_FLAG_PUBLIC = 1 << 0
SYMBOL_FLAG_IMPORT = 1 << 1


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """Cross-module symbol reference."""

    module_id: int
    symbol_id: int


class SymbolName(str):
    """Module-local symbol-name attribute payload."""

    __slots__ = ()


@dataclass(slots=True)
class Symbol:
    """A module-level named entity.

    Import symbols (SYMBOL_FLAG_IMPORT set) are forward declarations
    of symbols defined in another module. source_module names the
    module; source_symbol names the symbol in that module (defaults
    to name if empty, supporting import aliasing).

    op is the defining Operation (func.def, func.decl, etc.). All
    function properties (args, results, body, attributes) are read
    directly from op rather than from any intermediate struct.
    """

    name: str = ""
    kind: SymbolKind = SymbolKind.FUNC_DEF
    flags: int = 0
    op: Operation | None = None
    source_module: str = ""
    source_symbol: str = ""
    uses: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def is_import(self) -> bool:
        return (self.flags & SYMBOL_FLAG_IMPORT) != 0

    @property
    def is_public(self) -> bool:
        return (self.flags & SYMBOL_FLAG_PUBLIC) != 0


_BYTECODE_SYMBOL_KIND_BY_NAME: dict[str, SymbolKind] = {
    "LOOM_SYMBOL_FUNC_DEF": SymbolKind.FUNC_DEF,
    "LOOM_SYMBOL_FUNC_DECL": SymbolKind.FUNC_DECL,
    "LOOM_SYMBOL_FUNC_TEMPLATE": SymbolKind.FUNC_TEMPLATE,
    "LOOM_SYMBOL_FUNC_UKERNEL": SymbolKind.FUNC_UKERNEL,
    "LOOM_SYMBOL_GLOBAL": SymbolKind.GLOBAL,
    "LOOM_SYMBOL_EXECUTABLE": SymbolKind.EXECUTABLE,
    "LOOM_SYMBOL_RECORD": SymbolKind.RECORD,
}


def symbol_from_operation(operation: Operation, op_decl: Any | None = None) -> Symbol:
    """Build a module Symbol entry for a symbol-defining operation.

    The op declaration is the source of truth for which attribute defines the
    symbol and which legacy bytecode payload kind, if any, should be used. This
    keeps Python symbol materialization aligned with the generated C metadata
    instead of maintaining a second op-name table.
    """
    symbol_def = getattr(op_decl, "symbol_def", None)
    if symbol_def is None:
        raise ValueError(
            f"op '{operation.name}' does not declare a generated symbol_def"
        )
    kind = _BYTECODE_SYMBOL_KIND_BY_NAME.get(symbol_def.bytecode_kind, SymbolKind.NONE)
    name_attr = symbol_def.field
    symbol_flags = 0
    if operation.attributes.get("visibility") == "public":
        symbol_flags |= SYMBOL_FLAG_PUBLIC

    source_module = operation.attributes.get("import_module", "")
    if source_module:
        symbol_flags |= SYMBOL_FLAG_IMPORT

    return Symbol(
        name=operation.attributes.get(name_attr, ""),
        kind=kind,
        flags=symbol_flags,
        op=operation,
        source_module=source_module,
        source_symbol=operation.attributes.get("import_symbol", ""),
    )


# ============================================================================
# Tables
# ============================================================================


class StringTable:
    """Interned string table. Deduplicates strings by content."""

    __slots__ = ("_strings", "_index")

    def __init__(self) -> None:
        self._strings: list[str] = []
        self._index: dict[str, int] = {}

    def intern(self, s: str) -> int:
        """Intern a string, returning its ID."""
        if s in self._index:
            return self._index[s]
        idx = len(self._strings)
        self._strings.append(s)
        self._index[s] = idx
        return idx

    def get(self, string_id: int) -> str:
        """Look up a string by ID."""
        return self._strings[string_id]

    def __len__(self) -> int:
        return len(self._strings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._strings)


class TypeTable:
    """Interned type table. Deduplicates types by structural equality."""

    __slots__ = ("_types", "_index")

    def __init__(self) -> None:
        self._types: list[Type] = []
        self._index: dict[Type, int] = {}

    def intern(self, t: Type) -> int:
        """Intern a type, returning its ID."""
        if t in self._index:
            return self._index[t]
        idx = len(self._types)
        self._types.append(t)
        self._index[t] = idx
        return idx

    def get(self, type_id: int) -> Type:
        """Look up a type by ID."""
        return self._types[type_id]

    def __len__(self) -> int:
        return len(self._types)

    def __iter__(self) -> Iterator[Type]:
        return iter(self._types)


class LocationTable:
    """Interned location table. Entry 0 is always None (unknown).
    O(1) dedup via hash map on frozen LocationData."""

    __slots__ = ("_locations", "_index")

    def __init__(self) -> None:
        self._locations: list[LocationData] = [None]  # Index 0 = unknown.
        self._index: dict[LocationData, int] = {None: 0}

    def add(self, loc: LocationData) -> int:
        """Add a location, returning its ID. Deduplicates."""
        if loc in self._index:
            return self._index[loc]
        loc_id = len(self._locations)
        self._locations.append(loc)
        self._index[loc] = loc_id
        return loc_id

    def get(self, location_id: int) -> LocationData:
        """Look up a location by ID."""
        return self._locations[location_id]

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self) -> Iterator[LocationData]:
        return iter(self._locations)


# ============================================================================
# Module
# ============================================================================


@dataclass
class Module:
    """Top-level IR container.

    Owns all IR through its tables. The module is the unit of
    serialization, linking, and compilation.
    """

    name: str = ""
    flags: int = 0

    # Tables.
    strings: StringTable = field(default_factory=StringTable)
    types: TypeTable = field(default_factory=TypeTable)
    locations: LocationTable = field(default_factory=LocationTable)
    values: list[Value] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    encodings: list[EncodingInstance] = field(default_factory=list)

    # Source table is on the context, but for standalone modules
    # we keep a local source list.
    sources: list[str] = field(default_factory=list)

    def add_value(self, value: Value) -> int:
        """Add a value to the module, returning its ID."""
        value_id = len(self.values)
        self.values.append(value)
        return value_id

    def add_location(self, loc: LocationData) -> int:
        """Add a location, returning its ID. O(1) dedup."""
        return self.locations.add(loc)

    def add_symbol(self, symbol: Symbol) -> int:
        """Add a symbol, returning its ID."""
        sym_id = len(self.symbols)
        self.symbols.append(symbol)
        return sym_id

    def add_encoding(self, instance: EncodingInstance) -> int:
        """Add an encoding instance, returning its index. Deduplicates."""
        for _param_name, param_value in instance.params:
            self._add_nested_encodings(param_value)
        if instance.alias:
            for existing in self.encodings:
                if existing.alias == instance.alias and existing != instance:
                    raise ValueError(
                        f"encoding alias {instance.alias!r} already names a "
                        "different encoding"
                    )
        for i, existing in enumerate(self.encodings):
            if existing == instance:
                if instance.alias and not existing.alias:
                    self.encodings[i] = EncodingInstance(
                        name=existing.name,
                        alias=instance.alias,
                        params=existing.params,
                    )
                return i
        idx = len(self.encodings)
        self.encodings.append(instance)
        return idx

    def _add_nested_encodings(self, value: Any) -> None:
        """Register child encoding attrs before their parents."""
        if isinstance(value, EncodingInstance):
            self.add_encoding(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self._add_nested_encodings(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                self._add_nested_encodings(item)


# ============================================================================
# Value metadata helpers
# ============================================================================


def record_block_value_metadata(
    module: Module,
    block: Block,
    *,
    block_index: int,
) -> None:
    """Record block-arg definitions and nested-op metadata for one block."""
    for arg_index, value_id in enumerate(block.arg_ids):
        value = module.values[value_id]
        value.flags |= VALUE_FLAG_BLOCK_ARG
        value.def_op_index = VALUE_DEF_OP_NONE
        value.def_block_index = block_index
        value.def_result_index = arg_index

    for op_index, operation in enumerate(block.ops):
        record_operation_value_metadata(
            module,
            operation,
            block_index=block_index,
            op_index=op_index,
        )


def record_operation_value_metadata(
    module: Module,
    operation: Operation,
    *,
    block_index: int,
    op_index: int,
    operand_def_count: int = 0,
) -> None:
    """Record result definitions, operand uses, and nested-region metadata.

    operand_def_count marks leading operands that are definition slots rather
    than ordinary uses. Bodyless func-like symbol ops store signature arguments
    in op.operands, so parser/builder/bytecode reader callers pass
    operand_def_count=len(op.operands) for those symbols.
    """
    for result_index, value_id in enumerate(operation.results):
        value = module.values[value_id]
        value.def_op_index = op_index
        value.def_block_index = block_index
        value.def_result_index = result_index

    for operand_index, value_id in enumerate(operation.operands):
        value = module.values[value_id]
        if operand_index < operand_def_count:
            value.flags &= ~VALUE_FLAG_BLOCK_ARG
            value.def_op_index = op_index
            value.def_block_index = block_index
            value.def_result_index = operand_index
            continue
        value.uses.append(
            Use(
                user_op_index=op_index,
                operand_index=operand_index,
                block_index=block_index,
            )
        )

    for region in operation.regions:
        for child_block_index, child_block in enumerate(region.blocks):
            record_block_value_metadata(
                module,
                child_block,
                block_index=child_block_index,
            )


def rebuild_value_metadata(module: Module) -> None:
    """Rebuild all def/use metadata from module.symbols and nested regions."""
    for value in module.values:
        value.flags &= ~VALUE_FLAG_BLOCK_ARG
        value.def_op_index = VALUE_DEF_OP_NONE
        value.def_block_index = VALUE_DEF_BLOCK_NONE
        value.def_result_index = 0
        value.uses.clear()

    for symbol_index, symbol in enumerate(module.symbols):
        if symbol.op is None:
            continue
        operand_def_count = len(symbol.op.operands) if not symbol.op.regions else 0
        record_operation_value_metadata(
            module,
            symbol.op,
            block_index=VALUE_DEF_BLOCK_NONE,
            op_index=symbol_index,
            operand_def_count=operand_def_count,
        )


# ============================================================================
# Context
# ============================================================================


@dataclass
class Context:
    """Global state: vtables, source table, module registry.

    Immutable during compilation. Created once at startup.
    """

    sources: list[str] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)

    def add_source(self, name: str) -> int:
        """Register a source identifier, returning its ID."""
        for i, existing in enumerate(self.sources):
            if existing == name:
                return i
        source_id = len(self.sources)
        self.sources.append(name)
        return source_id
