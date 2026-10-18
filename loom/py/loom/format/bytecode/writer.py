# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bytecode writer: serializes ir.py Module to .loombc format.

Two-pass design:
  Pass 1 (numbering): Walk the module, assign integer IDs to all
    strings, types, op names, sources, encodings, locations.
  Pass 2 (writing): Emit sections using ByteBuffer, referencing
    IDs from the numbering pass.

The writer produces deterministic output: identical modules produce
identical bytes. This is required for caching and CAS storage.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, cast

from loom.dsl import FuncLikeInterface
from loom.fields import compute_layout, resolve_fields
from loom.format.block_order import ordered_blocks
from loom.format.bytecode.encoding import ByteBuffer
from loom.format.bytecode.op_decls import (
    attr_def_for_op,
    build_op_decl_map,
    func_like_interface_for_op,
    symbol_def_for_op,
)
from loom.format.bytecode.symbol_references import SymbolReferenceProjectionBuilder
from loom.ir import (
    ATTR_AGGREGATE_MAX_NESTING_DEPTH,
    PREDICATE_KINDS,
    REGION_SOURCE_FLAG_MASK,
    Block,
    BufferType,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    EnumArrayAttr,
    FileLocation,
    FunctionType,
    FusedLocation,
    GroupType,
    Module,
    NoneType,
    OpaqueLocation,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    PoolType,
    Predicate,
    PredicateArg,
    PredicateListAttr,
    Region,
    RegisterType,
    ScalarType,
    ShapedType,
    SignedEnumSetAttr,
    StaticDim,
    StorageType,
    StringArrayAttr,
    Symbol,
    SymbolKind,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    TaggedLocation,
    Type,
    TypeKind,
    Value,
)
from loom.type_binding import binding_children, iter_value_bindings
from loom.type_identity import TypeIdentity

_IR_TYPE_CLASSES = (
    ScalarType,
    ShapedType,
    BufferType,
    FunctionType,
    RegisterType,
    StorageType,
    DialectType,
    ParameterizedType,
    EncodingType,
    PoolType,
    GroupType,
    NoneType,
)

__all__ = [
    "BytecodeWriter",
    "write_module",
]

# Section kind constants (must match loom_bytecode_section_kind_e).
SECTION_STRINGS = 0
SECTION_SOURCES = 1
SECTION_TYPES = 2
SECTION_ENCODINGS = 3
SECTION_OPS = 4
SECTION_LOCATIONS = 5
SECTION_SYMBOLS = 6
SECTION_IR = 7
SECTION_RESOURCES = 8
SECTION_SOURCE_TRIVIA = 9
SECTION_SYMBOL_REFERENCES = 10

SECTION_WRITE_ORDER = (
    SECTION_IR,
    SECTION_SYMBOLS,
    SECTION_SYMBOL_REFERENCES,
    SECTION_STRINGS,
    SECTION_SOURCES,
    SECTION_TYPES,
    SECTION_ENCODINGS,
    SECTION_OPS,
    SECTION_LOCATIONS,
    SECTION_SOURCE_TRIVIA,
)

FUNCTION_SYMBOL_KINDS = frozenset(
    {
        SymbolKind.FUNC_DEF,
        SymbolKind.FUNC_DECL,
        SymbolKind.TEMPLATE_DECL,
        SymbolKind.TEMPLATE_DEF,
        SymbolKind.TEMPLATE_UKERNEL,
    }
)

LOCATION_MODE_SOURCE_LOCATIONS = 0
LOCATION_MODE_NO_LOCATIONS = 1
LOCATION_MODE_FULL_LOCATIONS = 2

# Attribute value kind bytes.
ATTR_KIND_I64 = 0
ATTR_KIND_F64 = 1
ATTR_KIND_STRING = 2
ATTR_KIND_BOOL = 3
ATTR_KIND_ENUM = 4
ATTR_KIND_I64_ARRAY = 5
ATTR_KIND_SYMBOL = 6
ATTR_KIND_TYPE = 7
ATTR_KIND_PREDICATE_LIST = 8
ATTR_KIND_DICT = 9
ATTR_KIND_ENCODING = 10
ATTR_KIND_BYTES = 11
ATTR_KIND_SCOPED_ENUM = 12
ATTR_KIND_ENUM_ARRAY = 13
ATTR_KIND_PARAMETERIZED = 14
ATTR_KIND_PARAMETERIZED_ARRAY = 15
ATTR_KIND_SIGNED_ENUM_SET = 16
ATTR_KIND_SYMBOL_ARRAY = 17
ATTR_KIND_SYMBOL_SET = 18
ATTR_KIND_STRING_ARRAY = 19

# Type kind bytes. These must match loom_bytecode_type_kind_e, not just the
# current Python enum spelling.
BYTECODE_TYPE_KIND_BY_IR_KIND: dict[TypeKind, int] = {
    TypeKind.NONE: 0,
    TypeKind.SCALAR: 1,
    TypeKind.TILE: 2,
    TypeKind.TENSOR: 3,
    TypeKind.FUNCTION: 5,
    TypeKind.DIALECT: 6,
    TypeKind.ENCODING: 7,
    TypeKind.POOL: 8,
    TypeKind.VECTOR: 9,
    TypeKind.VIEW: 10,
    TypeKind.BUFFER: 11,
    TypeKind.REGISTER: 12,
    TypeKind.STORAGE: 13,
    TypeKind.PARAMETERIZED: 14,
    TypeKind.GROUP: 15,
}

BYTECODE_IR_KIND_BY_TYPE_KIND: dict[int, TypeKind] = {
    type_kind: ir_kind for ir_kind, type_kind in BYTECODE_TYPE_KIND_BY_IR_KIND.items()
}

# File magic and version.
MAGIC = b"LOOM"
FORMAT_VERSION = 42
PRODUCER = "loom-py"

SYMBOL_INTERFACE_FLAG_MASK = (1 << 14) - 1

SOURCE_TRIVIA_LEADING_BLANK_LINE = 1
SOURCE_TRIVIA_COMMENT_COUNT_SHIFT = 1
MAX_SOURCE_COMMENT_COUNT = 0xFFFF

SYMBOL_FLAG_PUBLIC = 0x0001
SYMBOL_FLAG_IMPORT = 0x0002
SYMBOL_FLAG_IMPORT_SYMBOL = 0x0004
SYMBOL_FLAG_RETAIN = 0x0008
SYMBOL_FLAG_DECLARATION = 0x0010
SYMBOL_FLAG_TEST_ONLY = 0x0020
_SYMBOL_FLAG_PREDICATES = 0x0040
SYMBOL_FLAG_EXPORT = 0x0080


# ============================================================================
# Numbering context
# ============================================================================


class NumberingContext:
    """Assigns integer IDs to all entities in a module.

    Built during Pass 1 (numbering). Used by all section writers
    during Pass 2 to resolve cross-references.
    """

    def __init__(self) -> None:
        self.strings: dict[str, int] = {}
        self.ops: dict[str, int] = {}
        self.sources: list[str] = []
        self._type_list: list[Type] = []
        self._type_lookup: dict[int, int] = {}
        # Invocation-local structural keys never recursively hash type DAGs.
        self._type_identity = TypeIdentity()
        # Static encodings are invocation-owned and complete after their types.
        self.encodings: list[EncodingInstance] = []
        # Completed type prefix immediately before each encoding was published.
        self.encoding_type_counts: list[int] = []
        # Physical encoding identities mapped to their canonical wire ordinals.
        self.encoding_indices: dict[int, int] = {}
        # Structural identities shared with the invocation's type catalog.
        self._encoding_lookup: dict[int, int] = {}
        # Authored aliases mapped to identities to reject conflicting names.
        self._encoding_aliases: dict[str, int] = {}
        # Value definitions use string id 0 as "no SSA name". Keep the empty
        # string at bytecode string-table slot 0 so anonymous values do not
        # accidentally pick up the first real symbol name during reading.
        self.intern_string("")

    def intern_string(self, text: str) -> int:
        """Intern a string, returning its ID."""
        if text in self.strings:
            return self.strings[text]
        string_id = len(self.strings)
        self.strings[text] = string_id
        return string_id

    def intern_type(self, ir_type: Type) -> int:
        """Intern a type, returning its ID."""
        identity = self._type_identity.intern(ir_type)
        if identity in self._type_lookup:
            return self._type_lookup[identity]
        type_id = len(self._type_list)
        self._type_list.append(ir_type)
        self._type_lookup[identity] = type_id
        return type_id

    def intern_op(self, op_name: str) -> int:
        """Intern an op name, returning its kind ID."""
        if op_name in self.ops:
            return self.ops[op_name]
        op_id = len(self.ops)
        self.ops[op_name] = op_id
        return op_id

    def intern_encoding(self, instance: EncodingInstance) -> None:
        """Publish an encoding after all of its parameter dependencies complete."""
        identity = self._type_identity.intern(instance)
        if instance.alias:
            previous = self._encoding_aliases.setdefault(instance.alias, identity)
            if previous != identity:
                raise ValueError(
                    f"encoding alias {instance.alias!r} already names a "
                    "different encoding"
                )
        index = self._encoding_lookup.get(identity)
        if index is None:
            index = len(self.encodings)
            self._encoding_lookup[identity] = index
            self.encodings.append(instance)
            self.encoding_type_counts.append(len(self._type_list))
        elif instance.alias and not self.encodings[index].alias:
            self.encodings[index] = instance
        self.encoding_indices[id(instance)] = index

    @property
    def string_list(self) -> list[str]:
        """Strings in ID order."""
        result = [""] * len(self.strings)
        for text, idx in self.strings.items():
            result[idx] = text
        return result

    @property
    def type_list(self) -> list[Type]:
        """Types in ID order."""
        return self._type_list

    @property
    def op_list(self) -> list[str]:
        """Op names in ID order."""
        result = [""] * len(self.ops)
        for name, idx in self.ops.items():
            result[idx] = name
        return result


# ============================================================================
# Bytecode writer
# ============================================================================


class _ValueNumbering(dict[int, int]):
    """One independently decoded SSA namespace and its completed type records."""

    def __init__(self, values: Mapping[int, int] | None = None) -> None:
        super().__init__(values or {})
        # Scope-local completed ordinals, keyed by source type identity.
        self.types: dict[int, int] = {}
        # Completed type/attribute nodes in this scope's dependency traversal.
        self.completed: set[int] = set()


class BytecodeWriter:
    """Serializes an ir.py Module to .loombc bytes.

    Usage:
        writer = BytecodeWriter(module)
        data = writer.write()
        with open("output.loombc", "wb") as f:
            f.write(data)
    """

    def __init__(
        self,
        module: Module,
        *,
        location_mode: int = LOCATION_MODE_SOURCE_LOCATIONS,
        op_decls: Iterable[Any] | None = None,
    ) -> None:
        if location_mode not in (
            LOCATION_MODE_SOURCE_LOCATIONS,
            LOCATION_MODE_NO_LOCATIONS,
            LOCATION_MODE_FULL_LOCATIONS,
        ):
            raise ValueError(f"unsupported location mode: {location_mode}")
        if location_mode == LOCATION_MODE_FULL_LOCATIONS:
            raise NotImplementedError(
                "FULL_LOCATIONS bytecode mode requires field span emission"
            )
        self._module = module
        self._location_mode = location_mode
        self._op_decls_by_name = build_op_decl_map(op_decls)
        self._ctx = NumberingContext()
        # Indexed binding facts for each reachable type/attribute graph node.
        self._binding_facts: dict[int, bool] = {}
        # Physical type nodes whose strings and immediate children are numbered.
        self._numbered_types: set[int] = set()
        # Dominance order retained once for value numbering and block emission.
        self._region_blocks: dict[int, list[Block]] = {}
        self._wire_symbols, self._wire_symbol_indices = (
            self._build_wire_symbol_projection()
        )
        self._number_module()
        (
            self._module_dependencies,
            self._symbol_dependencies,
            self._symbol_template_demands,
        ) = SymbolReferenceProjectionBuilder(
            self._module,
            self._ctx.encodings,
            self._wire_symbol_indices,
            self._op_decls_by_name,
        ).build()

    # --- Pass 1: Numbering ---

    def _build_wire_symbol_projection(self) -> tuple[list[Symbol], dict[str, int]]:
        """Build the complete wire symbol projection."""
        wire_symbols = list(self._module.symbols)
        symbol_indices: dict[str, int] = {}
        for symbol_index, symbol in enumerate(wire_symbols):
            if symbol.name in symbol_indices:
                raise ValueError(f"duplicate symbol name {symbol.name!r}")
            symbol_indices[symbol.name] = symbol_index
            if symbol.kind == SymbolKind.NONE:
                raise ValueError(
                    f"symbol {symbol.name!r} has no serializable symbol kind"
                )
        return wire_symbols, symbol_indices

    def _number_module(self) -> None:
        """Walk the module and assign IDs to all entities."""
        module = self._module

        # Module name.
        if module.name:
            self._ctx.intern_string(module.name)

        # Sources.
        self._ctx.sources = list(module.sources)

        # Preserve authored catalog order, completing mixed dependencies first.
        # Reachable encodings absent from that catalog are numbered below.
        for encoding in module.encodings:
            self._number_encoding_instance(encoding)

        # Walk symbols.
        for symbol in self._wire_symbols:
            self._ctx.intern_string(symbol.name)
            if symbol.source_module:
                self._ctx.intern_string(symbol.source_module)
            if symbol.source_symbol:
                self._ctx.intern_string(symbol.source_symbol)
            if symbol.op is not None:
                if symbol.kind == SymbolKind.GLOBAL:
                    self._number_global_op(symbol.op)
                elif symbol.kind == SymbolKind.RECORD:
                    self._number_record_op(symbol.op)
                elif symbol.kind in FUNCTION_SYMBOL_KINDS:
                    self._number_func_op(symbol.op)
                else:
                    raise ValueError(
                        f"symbol {symbol.name!r} of kind {symbol.kind.name} "
                        "has no supported defining op"
                    )

    def _number_func_op(self, op: Operation) -> None:
        """Number all entities in a func-like op (func.def, func.decl, etc.)."""
        module = self._module
        shared_attr_keys = self._shared_func_metadata_attr_keys(op)

        # Defining func-like op name.
        self._ctx.intern_op(op.name)
        self._ctx.intern_string(op.name)

        # Kernel workload and ordinary FuncLike signature names/types.
        workload_arg_ids = self._kernel_workload_arg_ids(op)
        arg_ids = self._func_arg_ids(op)
        for arg_id in [*workload_arg_ids, *arg_ids]:
            value = module.values[arg_id]
            self._ctx.intern_string(value.name)
            self._number_type(value.type)

        # Result types.
        for result_id in op.results:
            value = module.values[result_id]
            if value.name:
                self._ctx.intern_string(value.name)
            self._number_type(value.type)

        for key, value in op.attributes.items():
            if key in shared_attr_keys:
                continue
            self._ctx.intern_string(key)
            self._number_attr_value(value, self._attr_def_for_op_attr(op.name, key))

        for region in op.regions:
            self._number_region(region)

    def _func_body_region_index(self, op: Operation) -> int | None:
        """Return op's FuncLike body region index, if it has one."""
        func_like = func_like_interface_for_op(self._op_decls_by_name, op.name)
        if func_like is None or func_like.body is None:
            return None
        op_decl = self._op_decls_by_name.get(op.name)
        if op_decl is None:
            return None
        for region_index, region_def in enumerate(getattr(op_decl, "regions", ())):
            if region_def.name == func_like.body:
                return region_index
        return None

    def _func_arg_ids(self, op: Operation) -> list[int]:
        """Return the logical FuncLike signature argument value ids."""
        func_like = func_like_interface_for_op(self._op_decls_by_name, op.name)
        op_decl = self._op_decls_by_name.get(op.name)
        if func_like is not None and func_like.args is not None and op_decl is not None:
            return resolve_fields(compute_layout(op_decl), op, self._module).value_ids(
                func_like.args
            )
        body_region_index = self._func_body_region_index(op)
        if body_region_index is None:
            return []
        if body_region_index >= len(op.regions):
            return []
        body = op.regions[body_region_index]
        if not body.blocks:
            return []
        return list(body.blocks[0].arg_ids)

    def _kernel_workload_arg_ids(self, op: Operation) -> list[int]:
        """Return a kernel symbol's logical workload signature value ids."""
        op_decl = self._op_decls_by_name.get(op.name)
        symbol_def = symbol_def_for_op(self._op_decls_by_name, op.name)
        if op_decl is None or symbol_def is None or symbol_def.kernel_contract is None:
            return []
        contract = symbol_def.kernel_contract
        layout = compute_layout(op_decl)
        if contract.workload_operands is not None:
            return resolve_fields(layout, op, self._module).value_ids(
                contract.workload_operands
            )
        assert contract.workload_region is not None
        region_index = layout.fields[contract.workload_region].index
        if region_index >= len(op.regions):
            return []
        region = op.regions[region_index]
        if not region.blocks:
            return []
        return list(region.blocks[0].arg_ids)

    def _shared_func_metadata_attr_keys(self, op: Operation) -> frozenset[str]:
        """Return func-like attrs encoded by fixed symbol metadata fields."""
        symbol_def = symbol_def_for_op(self._op_decls_by_name, op.name)
        keys = {
            cast(str, symbol_def.field),
            "import_module",
            "import_symbol",
        }
        func_like = func_like_interface_for_op(self._op_decls_by_name, op.name)
        if func_like is not None:
            for field_name in ("visibility", "cc", "effect_contract", "predicates"):
                attr_name = getattr(func_like, field_name, None)
                if attr_name is not None:
                    keys.add(attr_name)
            for field_name in ("template_family", "priority"):
                attr_name = getattr(func_like, field_name, None)
                if attr_name is not None:
                    keys.add(attr_name)
        return frozenset(keys)

    def _symbol_definition_flags(self, op: Operation) -> int:
        """Return bytecode roles declared by the symbol-defining op."""

        symbol_def = symbol_def_for_op(self._op_decls_by_name, op.name)
        flags = 0
        if symbol_def.is_declaration:
            flags |= SYMBOL_FLAG_DECLARATION
        if symbol_def.is_test_only:
            flags |= SYMBOL_FLAG_TEST_ONLY
        return flags

    def _symbol_is_exported(self, symbol: Symbol) -> bool:
        """Return whether a symbol is available for static linkage."""
        if symbol.flags & SYMBOL_FLAG_IMPORT:
            return False
        if symbol.flags & SYMBOL_FLAG_PUBLIC:
            return True
        if symbol.op is None:
            return False
        symbol_def = symbol_def_for_op(self._op_decls_by_name, symbol.op.name)
        return bool(
            {"kernel", "kernel_entry"}.intersection(symbol_def.interfaces)
        ) or bool(
            symbol_def.visibility
            and symbol.op.attributes.get(symbol_def.visibility) == "public"
        )

    def _number_global_op(self, op: Operation) -> None:
        """Number all entities in a global symbol-defining op."""
        symbol_field = self._symbol_field_for_op(op)
        self._ctx.intern_op(op.name)
        self._ctx.intern_string(op.name)

        for result_id in self._collect_global_local_values(op):
            value = self._module.values[result_id]
            if value.name:
                self._ctx.intern_string(value.name)
            self._number_type(value.type)

        for key, value in op.attributes.items():
            if key == symbol_field:
                continue
            self._ctx.intern_string(key)
            self._number_attr_value(value, self._attr_def_for_op_attr(op.name, key))

    def _number_record_op(self, op: Operation) -> None:
        """Number all entities in a record symbol-defining op."""
        symbol_field = self._symbol_field_for_op(op)
        self._ctx.intern_op(op.name)
        self._ctx.intern_string(op.name)

        for key, value in op.attributes.items():
            if key == symbol_field:
                continue
            self._ctx.intern_string(key)
            self._number_attr_value(value, self._attr_def_for_op_attr(op.name, key))
        if len(op.regions) > 1:
            raise ValueError(
                f"record symbol op {op.name!r} has {len(op.regions)} regions; "
                "bytecode record symbols support at most one body region"
            )
        if op.regions:
            self._number_region(op.regions[0])

    def _symbol_field_for_op(self, op: Operation) -> str:
        """Return the generated symbol identity attr field for ``op``."""
        return cast(str, symbol_def_for_op(self._op_decls_by_name, op.name).field)

    def _collect_global_local_values(self, op: Operation) -> list[int]:
        """Return declaration-local values required to materialize one global."""
        module = self._module
        local_values: list[int] = []
        seen: set[int] = set()

        def add_value(value_id: int) -> None:
            if value_id < 0 or value_id >= len(module.values):
                raise ValueError(
                    f"global {op.name} references value {value_id} outside "
                    f"the module value table"
                )
            if value_id in seen:
                return
            seen.add(value_id)
            local_values.append(value_id)

        def collect_value_bindings(scan_start: int) -> int:
            scan_index = scan_start
            while scan_index < len(local_values):
                value = module.values[local_values[scan_index]]
                scan_index += 1
                collect_attr_value(value.type)
            return scan_index

        def collect_attr_value(value: Any) -> None:
            for binding in iter_value_bindings(value):
                value_id = (
                    binding.value
                    if isinstance(binding, PredicateArg)
                    else binding.value_id
                )
                if value_id is not None:
                    add_value(value_id)

        for result_id in op.results:
            add_value(result_id)
        scan_index = collect_value_bindings(0)

        for key, value in op.attributes.items():
            if key == "symbol":
                continue
            collect_attr_value(value)
        collect_value_bindings(scan_index)
        return local_values

    def _attr_def_for_op_attr(self, op_name: str, attr_name: str) -> Any | None:
        """Return generated attr metadata when the writer knows the op."""
        return attr_def_for_op(self._op_decls_by_name, op_name, attr_name)

    def _number_region(self, region: Region) -> None:
        """Number all entities in a region (recursive)."""
        blocks = ordered_blocks(region)
        self._region_blocks[id(region)] = blocks
        for block in blocks:
            if block.label:
                self._ctx.intern_string(block.label)
            for arg_id in block.arg_ids:
                value = self._module.values[arg_id]
                if value.name:
                    self._ctx.intern_string(value.name)
                self._number_type(value.type)
            for op in block.ops:
                self._number_operation(op)

    def _number_operation(self, op: Operation) -> None:
        """Number all entities in an operation."""
        # Op name.
        self._ctx.intern_op(op.name)
        self._ctx.intern_string(op.name)

        # Results.
        for result_id in op.results:
            value = self._module.values[result_id]
            if value.name:
                self._ctx.intern_string(value.name)
            self._number_type(value.type)

        # Attributes.
        for key, value in op.attributes.items():
            self._ctx.intern_string(key)
            self._number_attr_value(value, self._attr_def_for_op_attr(op.name, key))

        # Regions.
        for region in op.regions:
            self._number_region(region)

    def _has_type_bindings(self, root: Any) -> bool:
        """Compute binding facts once across the invocation's shared graph."""
        known = self._binding_facts.get(id(root))
        if known is not None:
            return known
        pending = [(root, False)]
        while pending:
            value, expanded = pending.pop()
            identity = id(value)
            if identity in self._binding_facts:
                continue
            children = tuple(binding_children(value))
            if children and not expanded:
                pending.append((value, True))
                pending.extend((child, False) for child in children)
                continue
            bound = any(self._binding_facts[id(child)] for child in children)
            if isinstance(value, DynamicDim):
                if value.value_id is None:
                    raise ValueError("dynamic dimension has no SSA binding")
                bound = True
            elif isinstance(value, DynamicEncoding):
                bound = True
            elif isinstance(value, PredicateArg):
                bound |= value.tag == "value"
            self._binding_facts[identity] = bound
        return self._binding_facts[id(root)]

    @staticmethod
    def _number(steps: Iterator[Any] | None) -> None:
        """Complete postorder numbering without growing the Python call stack."""
        if steps is None:
            return
        pending = [steps]
        while pending:
            try:
                child = next(pending[-1])
            except StopIteration:
                pending.pop()
            else:
                if child is not None:
                    pending.append(child)

    def _number_type(self, ir_type: Type) -> None:
        self._number(self._number_type_steps(ir_type))

    def _number_attr_value(self, value: Any, attr_def: Any | None = None) -> None:
        self._number(self._number_attr_steps(value, attr_def))

    def _number_encoding_instance(self, value: EncodingInstance) -> None:
        self._number(self._number_encoding_steps(value))

    def _number_type_steps(self, ir_type: Type) -> Iterator[Any] | None:
        """Finish repeated and scalar types without allocating a continuation."""
        if id(ir_type) in self._numbered_types:
            return None
        self._numbered_types.add(id(ir_type))
        if isinstance(ir_type, ScalarType):
            self._binding_facts[id(ir_type)] = False
            self._ctx.intern_type(ir_type)
            return None
        return self._number_type_children(ir_type)

    def _number_type_children(self, ir_type: Type) -> Iterator[Any]:
        """Ensure a type and all its sub-types are interned.

        Sub-types are interned before their parent, so the reader resolves each
        child from an earlier entry in the topologically ordered type table.
        """
        # Complete immediate children before numbering their parent.
        match ir_type:
            case ShapedType(element_type=element, encoding=encoding):
                yield self._number_type_steps(element)
                if isinstance(encoding, EncodingInstance):
                    yield self._number_encoding_steps(encoding)
            case FunctionType(arg_types=args, result_types=results):
                for child in args:
                    yield self._number_type_steps(child)
                for child in results:
                    yield self._number_type_steps(child)
            case DialectType(name=name, params=parameters):
                self._ctx.intern_string(name)
                for parameter in parameters:
                    yield self._number_type_steps(parameter)
            case RegisterType(value_type=value_type):
                if value_type is not None:
                    yield self._number_type_steps(value_type)
            case ParameterizedType(
                definition=definition,
                slots=slots,
            ):
                self._ctx.intern_string(definition.name)
                for parameter, value in zip(definition.params, slots, strict=True):
                    if value is None:
                        continue
                    self._ctx.intern_string(parameter.name)
                    yield self._number_attr_steps(value, parameter)
            case _:
                pass
        # Intern the parent AFTER sub-types (ensures sub-types have lower IDs).
        if not self._has_type_bindings(ir_type):
            self._ctx.intern_type(ir_type)

    def _number_attr_steps(
        self,
        value: Any,
        attr_def: Any | None = None,
        aggregate_nesting_depth: int = 0,
    ) -> Iterator[Any] | None:
        """Number leaves immediately; only structural children need suspension."""
        if getattr(attr_def, "attr_type", None) == "enum":
            return None
        if isinstance(value, str):
            self._ctx.intern_string(value)
            return None
        if isinstance(value, (int, float, bytes, bytearray)):
            return None
        return self._number_attr_children(value, attr_def, aggregate_nesting_depth)

    def _number_attr_children(
        self,
        value: Any,
        attr_def: Any | None,
        aggregate_nesting_depth: int,
    ) -> Iterator[Any]:
        """Intern strings referenced by attribute values."""
        attr_type = getattr(attr_def, "attr_type", None)
        if isinstance(value, StringArrayAttr):
            if attr_type not in (None, "any", "string_array"):
                raise ValueError(
                    "string arrays require a string-array or generic field"
                )
            for string in value:
                self._ctx.intern_string(string)
        elif isinstance(value, SymbolNameArray):
            if attr_type != "symbol_array":
                raise ValueError("symbol arrays require a descriptor-backed field")
            for name in value:
                self._ctx.intern_string(str(name))
        elif isinstance(value, SymbolNameSet):
            if attr_type != "symbol_set":
                raise ValueError("symbol sets require a descriptor-backed field")
            for name in value:
                self._ctx.intern_string(str(name))
        elif isinstance(value, ParameterizedAttr):
            if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
                raise ValueError(
                    "aggregate attribute nesting exceeds maximum depth "
                    f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
                )
            self._ctx.intern_string(value.family_name)
            for parameter, slot in zip(
                value.definition.parameters, value.slots, strict=True
            ):
                if slot is None:
                    continue
                self._ctx.intern_string(parameter.name)
                yield self._number_attr_steps(
                    slot, parameter, aggregate_nesting_depth + 1
                )
        elif isinstance(value, ParameterizedAttrArray):
            if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
                raise ValueError(
                    "aggregate attribute nesting exceeds maximum depth "
                    f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
                )
            if getattr(attr_def, "attr_type", None) != "parameterized_array":
                raise ValueError(
                    "parameterized attribute arrays require a descriptor-backed field"
                )
            for element in value:
                yield self._number_attr_steps(
                    element, attr_def, aggregate_nesting_depth + 1
                )
        elif isinstance(value, _IR_TYPE_CLASSES):
            yield self._number_type_steps(cast(Type, value))
        elif isinstance(value, EncodingInstance):
            yield self._number_encoding_steps(value)
        elif isinstance(value, Mapping):
            if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
                raise ValueError(
                    "aggregate attribute nesting exceeds maximum depth "
                    f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
                )
            for k, v in value.items():
                self._ctx.intern_string(k)
                yield self._number_attr_steps(
                    v, aggregate_nesting_depth=aggregate_nesting_depth + 1
                )
        elif isinstance(value, list | tuple):
            for item in value:
                yield self._number_attr_steps(item)

    def _number_encoding_steps(self, value: EncodingInstance) -> Iterator[Any]:
        """Complete one encoding's mixed type/attribute dependencies once."""
        if id(value) in self._ctx.encoding_indices:
            return
        for param_name, param_value in value.params:
            self._ctx.intern_string(param_name)
            yield self._number_attr_steps(param_value)
            if self._has_type_bindings(param_value):
                raise ValueError("static encoding parameters cannot capture SSA values")
        self._ctx.intern_string(value.name)
        if value.alias:
            self._ctx.intern_string(value.alias)
        self._ctx.intern_encoding(value)

    # --- Pass 2: Section writers ---

    def write(self) -> bytes:
        """Write complete .loombc file."""
        sections: dict[int, bytes] = {}
        sections[SECTION_STRINGS] = self._write_strings()
        sections[SECTION_SOURCES] = self._write_sources()
        sections[SECTION_ENCODINGS] = self._write_encodings()
        sections[SECTION_TYPES] = self._write_types()
        sections[SECTION_OPS] = self._write_ops()
        if self._location_mode != LOCATION_MODE_NO_LOCATIONS:
            sections[SECTION_LOCATIONS] = self._write_locations()
        if self._module.file_header:
            sections[SECTION_SOURCE_TRIVIA] = self._write_source_trivia_section()
        ir_bytes, ir_regions = self._write_ir()
        sections[SECTION_IR] = ir_bytes
        sections[SECTION_SYMBOLS] = self._write_symbols(ir_regions)
        sections[SECTION_SYMBOL_REFERENCES] = self._write_symbol_references()
        return self._assemble(sections, self._module_allocation_counts())

    def _write_strings(self) -> bytes:
        """Write the STRINGS section."""
        buf = ByteBuffer()
        strings = self._ctx.string_list
        buf.write_varint(len(strings))
        for text in strings:
            buf.write_string(text)
        return buf.get_bytes()

    def _write_sources(self) -> bytes:
        """Write the SOURCES section."""
        buf = ByteBuffer()
        sources = self._ctx.sources
        buf.write_varint(len(sources))
        for source in sources:
            buf.write_string(source)
        return buf.get_bytes()

    def _write_source_trivia_section(self) -> bytes:
        """Write module-owned source presentation."""
        buf = ByteBuffer()
        self._write_source_trivia(
            buf, leading_blank_line=False, comments=self._module.file_header
        )
        return buf.get_bytes()

    def _write_encodings(self) -> bytes:
        """Write the ENCODINGS section."""
        buf = ByteBuffer()
        encodings = self._ctx.encodings

        # Encoding family registry from unique encoding names.
        family_names: list[str] = []
        family_map: dict[str, int] = {}
        for enc in encodings:
            if enc.name not in family_map:
                family_map[enc.name] = len(family_names)
                family_names.append(enc.name)

        buf.write_varint(len(family_names))
        for name in family_names:
            buf.write_varint(self._ctx.strings[name])

        # Encoding instances.
        buf.write_varint(len(encodings))
        for enc, type_count in zip(
            encodings, self._ctx.encoding_type_counts, strict=True
        ):
            buf.write_varint(type_count)
            buf.write_varint(family_map[enc.name])
            alias_string_id_plus1 = self._ctx.strings[enc.alias] + 1 if enc.alias else 0
            buf.write_varint(alias_string_id_plus1)
            # Structured parameters: same attribute serialization as IR.
            buf.write_varint(len(enc.params))
            for param_name, param_value in enc.params:
                buf.write_varint(self._ctx.intern_string(param_name))
                self._write_attr_value(buf, param_value)

        return buf.get_bytes()

    def _write_types(self) -> bytes:
        """Write the TYPES section."""
        buf = ByteBuffer()
        types = self._ctx.type_list
        buf.write_varint(len(types))
        for ir_type in types:
            self._write_one_type(buf, ir_type)
        return buf.get_bytes()

    def _type_reference(self, ir_type: Type, values: _ValueNumbering | None) -> int:
        if values is None:
            return self._ctx.intern_type(ir_type)
        if self._binding_facts[id(ir_type)]:
            return (values.types[id(ir_type)] << 1) | 1
        return self._ctx.intern_type(ir_type) << 1

    def _write_one_type(
        self, buf: ByteBuffer, ir_type: Type, values: _ValueNumbering | None = None
    ) -> None:
        """Serialize a static table entry or a complete scoped record."""
        match ir_type:
            case NoneType():
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.NONE])
            case ScalarType(kind=kind):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.SCALAR])
                buf.write_u8(kind.value)
            case ShapedType():
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[ir_type.type_kind])
                buf.write_u8(ir_type.element_type.kind.value)
                buf.write_u8(ir_type.rank)
                # Encoding attachment: 0 = none, 1 = static, 2 = scoped SSA.
                if isinstance(ir_type.encoding, DynamicEncoding):
                    buf.write_u8(2)  # dynamic SSA encoding
                    buf.write_varint(
                        self._value_number_or_error(
                            values, ir_type.encoding.value_id, "dynamic encoding"
                        )
                    )
                elif isinstance(ir_type.encoding, EncodingInstance):
                    buf.write_u8(1)  # static encoding
                    buf.write_varint(
                        self._ctx.encoding_indices[id(ir_type.encoding)] + 1
                    )
                else:
                    buf.write_u8(0)  # no encoding
                    buf.write_varint(0)
                if ir_type.type_kind == TypeKind.VIEW:
                    buf.write_u8(ir_type.alignment or 0)
                for dim in ir_type.dims:
                    match dim:
                        case StaticDim(size=size):
                            buf.write_u8(0)  # is_dynamic = false
                            buf.write_varint(size)
                        case DynamicDim(value_id=value_id):
                            buf.write_u8(1)  # is_dynamic = true
                            if values is not None:
                                buf.write_varint(
                                    0
                                    if value_id is None
                                    else 1
                                    + self._value_number_or_error(
                                        values, value_id, "dynamic dimension"
                                    )
                                )
            case FunctionType(arg_types=args, result_types=results):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.FUNCTION])
                buf.write_varint(len(args))
                buf.write_varint(len(results))
                for arg in args:
                    buf.write_varint(self._type_reference(arg, values))
                for result in results:
                    buf.write_varint(self._type_reference(result, values))
            case DialectType(name=name, params=params):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.DIALECT])
                buf.write_varint(self._ctx.strings[name])
                buf.write_varint(len(params))
                for param in params:
                    buf.write_varint(self._type_reference(param, values))
            case ParameterizedType(
                definition=definition,
                slots=slots,
            ):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED])
                buf.write_varint(self._ctx.strings[definition.name])
                present_parameters = [
                    (parameter, value)
                    for parameter, value in zip(definition.params, slots, strict=True)
                    if value is not None
                ]
                buf.write_varint(len(present_parameters))
                for parameter, value in present_parameters:
                    buf.write_varint(self._ctx.strings[parameter.name])
                    self._write_attr_value(
                        buf,
                        value,
                        values,
                        attr_def=parameter,
                        completed_types=values is not None,
                    )
            case RegisterType(
                descriptor_set_stable_id=descriptor_set_stable_id,
                register_class_id=register_class_id,
                unit_count=unit_count,
                value_type=value_type,
            ):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER])
                buf.write_varint(descriptor_set_stable_id)
                buf.write_varint(register_class_id | (unit_count << 16))
                if values is None:
                    buf.write_u8(1 if value_type is not None else 0)
                if value_type is not None:
                    buf.write_varint(self._type_reference(value_type, values))
            case StorageType(space=space):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.STORAGE])
                buf.write_u8(space.value)
            case EncodingType(role=role):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.ENCODING])
                buf.write_u8(role.value)
            case PoolType():
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.POOL])
            case GroupType(dims=dims):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.GROUP])
                buf.write_u8(len(dims))
                for dimension in dims:
                    match dimension:
                        case StaticDim(size=size):
                            buf.write_u8(0)
                            buf.write_varint(size)
                        case DynamicDim(value_id=value_id):
                            buf.write_u8(1)
                            if values is not None:
                                buf.write_varint(
                                    0
                                    if value_id is None
                                    else 1
                                    + self._value_number_or_error(
                                        values,
                                        value_id,
                                        "group dimension",
                                    )
                                )
            case BufferType():
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.BUFFER])
            case _:
                raise ValueError(f"unsupported bytecode type: {ir_type!r}")

    def _write_ops(self) -> bytes:
        """Write the OPS section."""
        buf = ByteBuffer()
        ops = self._ctx.op_list
        buf.write_varint(len(ops))
        for op_name in ops:
            buf.write_varint(self._ctx.strings[op_name])
        return buf.get_bytes()

    def _write_locations(self) -> bytes:
        """Write the LOCATIONS section."""
        buf = ByteBuffer()
        locations = list(self._module.locations)
        buf.write_varint(len(locations))
        for loc in locations:
            if loc is None:
                buf.write_u8(0)  # NONE
                buf.write_u8(0)  # flags
            elif isinstance(loc, FileLocation):
                buf.write_u8(1)  # FILE
                buf.write_u8(loc.flags)
                buf.write_varint(loc.source_id)
                buf.write_varint(loc.start_line)
                buf.write_varint(loc.start_col)
                buf.write_varint(loc.end_line)
                buf.write_varint(loc.end_col)
            elif isinstance(loc, FusedLocation):
                buf.write_u8(2)  # FUSED
                buf.write_u8(loc.flags)
                buf.write_varint(len(loc.children))
                for child in loc.children:
                    buf.write_varint(child)
            elif isinstance(loc, OpaqueLocation):
                buf.write_u8(3)  # OPAQUE
                buf.write_u8(loc.flags)
                buf.write_varint(loc.source_id)
                buf.write_varint(len(loc.data))
                buf.write_bytes(loc.data)
            elif isinstance(loc, TaggedLocation):
                if loc.tag <= 0 or loc.tag > 0xFFFF:
                    raise ValueError("tagged location tag must be in [1, 65535]")
                buf.write_u8(4)  # TAGGED
                buf.write_u8(loc.flags)
                buf.write_varint(loc.tag)
                buf.write_varint(loc.child)
                buf.write_varint(len(loc.data))
                buf.write_bytes(loc.data)
        return buf.get_bytes()

    def _write_ir(self) -> tuple[bytes, dict[int, list[tuple[int, int, int]]]]:
        """Write independently bounded root-region payloads.

        Returns the IR bytes and symbol-indexed lists of
        (region_index, offset, length) records.
        """
        buf = ByteBuffer()
        ir_regions: dict[int, list[tuple[int, int, int]]] = {}

        for symbol_index, symbol in enumerate(self._module.symbols):
            if symbol.op is not None and symbol.op.regions:
                payloads: list[tuple[int, int, int]] = []
                for region_index, region in enumerate(symbol.op.regions):
                    start = buf.position
                    self._write_root_region_payload(buf, region)
                    payloads.append((region_index, start, buf.position - start))
                ir_regions[symbol_index] = payloads

        return buf.get_bytes(), ir_regions

    def _write_root_region_payload(self, buf: ByteBuffer, region: Region) -> None:
        """Write one root region with an independent SSA namespace."""
        value_numbers = _ValueNumbering()
        self._assign_value_numbers(region, value_numbers)
        value_count, region_count, block_count, op_count = self._count_region_tree(
            region
        )
        buf.write_varint(value_count)
        buf.write_varint(region_count)
        buf.write_varint(block_count)
        buf.write_varint(op_count)
        self._write_region(buf, region, value_numbers)

    def _assign_value_numbers(self, region: Region, numbers: dict[int, int]) -> None:
        """Assign sequential value numbers within one root region."""
        for block in self._region_blocks[id(region)]:
            for arg_id in block.arg_ids:
                if arg_id not in numbers:
                    numbers[arg_id] = len(numbers)
            for op in block.ops:
                if op.is_dead:
                    continue
                for result_id in op.results:
                    if result_id not in numbers:
                        numbers[result_id] = len(numbers)
                for nested_region in op.regions:
                    self._assign_value_numbers(nested_region, numbers)

    def _count_region_tree(self, region: Region) -> tuple[int, int, int, int]:
        """Return value, region, block, and op counts for a live region tree."""
        value_count = 0
        region_count = 1
        block_count = len(region.blocks)
        op_count = 0
        for block in region.blocks:
            value_count += len(block.arg_ids)
            for op in block.ops:
                if op.is_dead:
                    continue
                value_count += len(op.results)
                op_count += 1
                for nested_region in op.regions:
                    nested_counts = self._count_region_tree(nested_region)
                    value_count += nested_counts[0]
                    region_count += nested_counts[1]
                    block_count += nested_counts[2]
                    op_count += nested_counts[3]
        return value_count, region_count, block_count, op_count

    def _count_region_forest(
        self, op: Operation, region_indices: list[int] | None = None
    ) -> tuple[int, int, int, int]:
        """Return value, region, block, and op counts for root regions."""
        value_count = 0
        region_count = 0
        block_count = 0
        op_count = 0
        for region_index in (
            region_indices if region_indices is not None else range(len(op.regions))
        ):
            counts = self._count_region_tree(op.regions[region_index])
            value_count += counts[0]
            region_count += counts[1]
            block_count += counts[2]
            op_count += counts[3]
        return value_count, region_count, block_count, op_count

    def _module_allocation_counts(self) -> tuple[int, int, int, int]:
        """Return module value, serialized region, block, and op counts."""
        value_count = 0
        region_count = 0
        block_count = 0
        op_count = 0
        for symbol in self._module.symbols:
            if symbol.op is None or not symbol.op.regions:
                if symbol.op is not None and symbol.kind == SymbolKind.GLOBAL:
                    value_count += len(self._collect_global_local_values(symbol.op))
                elif symbol.op is not None:
                    value_count += len(symbol.op.operands) + len(symbol.op.results)
                continue
            value_count += len(symbol.op.results)
            counts = self._count_region_forest(symbol.op)
            value_count += counts[0]
            region_count += counts[1]
            block_count += counts[2]
            op_count += counts[3]
        return value_count, region_count, block_count, op_count

    def _value_number_or_error(
        self, value_numbers: _ValueNumbering, value_id: int, context: str
    ) -> int:
        try:
            return value_numbers[value_id]
        except KeyError as exc:
            raise ValueError(
                f"{context} references value {value_id} with no local bytecode number"
            ) from exc

    def _write_region(
        self, buf: ByteBuffer, region: Region, value_numbers: _ValueNumbering
    ) -> None:
        """Write a region (source_flags + block_count + blocks)."""
        if region.source_flags < 0 or (region.source_flags & ~REGION_SOURCE_FLAG_MASK):
            raise ValueError(
                f"region has unsupported source flag bits: {region.source_flags:#x}"
            )
        buf.write_varint(region.source_flags)
        buf.write_varint(len(region.blocks))
        blocks = self._region_blocks[id(region)]
        block_indices = {id(block): index for index, block in enumerate(blocks)}
        for block in blocks:
            self._write_block(buf, block, value_numbers, block_indices)

    def _write_block(
        self,
        buf: ByteBuffer,
        block: Block,
        value_numbers: _ValueNumbering,
        block_indices: dict[int, int],
    ) -> None:
        """Write a block (label, args, ops)."""
        has_label = bool(block.label)
        buf.write_u8(1 if has_label else 0)
        if has_label:
            buf.write_varint(self._ctx.strings[block.label])
        self._write_source_trivia(buf, block.leading_blank_line, block.comments)

        # Block args.
        buf.write_varint(len(block.arg_ids))
        for arg_id in block.arg_ids:
            self._write_value_def(buf, self._module.values[arg_id], value_numbers)

        # Operations.
        live_ops = [op for op in block.ops if not op.is_dead]
        buf.write_varint(len(live_ops))
        for op in live_ops:
            self._write_operation(buf, op, value_numbers, block_indices)

    def _write_type_use(
        self, buf: ByteBuffer, ir_type: Type, values: _ValueNumbering | None
    ) -> None:
        """Extend the current scope with final records and select its root."""
        bound = self._binding_facts[id(ir_type)]
        if values is None:
            if bound:
                raise ValueError("SSA-bound type requires a value scope")
            buf.write_varint(self._ctx.intern_type(ir_type))
            return
        buf.write_varint(1 if bound else self._ctx.intern_type(ir_type) << 1)
        if not bound:
            buf.write_varint(0)
            return
        records: list[Type] = []
        pending = [(ir_type, False)]
        while pending:
            value, expanded = pending.pop()
            identity = id(value)
            if identity in values.completed or not self._binding_facts[identity]:
                continue
            children = tuple(binding_children(value))
            if children and not expanded:
                pending.append((value, True))
                pending.extend((child, False) for child in reversed(children))
                continue
            values.completed.add(identity)
            if isinstance(value, _IR_TYPE_CLASSES):
                values.types[identity] = len(values.types)
                records.append(value)
        payload = ByteBuffer()
        payload.write_varint(len(records))
        for record in records:
            self._write_one_type(payload, record, values)
        payload.write_varint(values.types[id(ir_type)] + 1)
        data = payload.get_bytes()
        buf.write_varint(len(data))
        buf.write_bytes(data)

    def _write_source_trivia(
        self,
        buf: ByteBuffer,
        leading_blank_line: bool,
        comments: tuple[str, ...],
    ) -> None:
        if len(comments) > MAX_SOURCE_COMMENT_COUNT:
            raise ValueError(
                f"comment count {len(comments)} exceeds maximum "
                f"{MAX_SOURCE_COMMENT_COUNT}"
            )
        source_trivia = len(comments) << SOURCE_TRIVIA_COMMENT_COUNT_SHIFT
        if leading_blank_line:
            source_trivia |= SOURCE_TRIVIA_LEADING_BLANK_LINE
        buf.write_varint(source_trivia)
        for comment in comments:
            encoded = ((" " + comment) if comment else "").encode("utf-8")
            buf.write_varint(len(encoded))
            buf.write_bytes(encoded)

    def _write_value_def(
        self, buf: ByteBuffer, value: Value, value_numbers: _ValueNumbering
    ) -> None:
        buf.write_varint(self._ctx.strings.get(value.name, 0))
        self._write_type_use(buf, value.type, value_numbers)

    def _write_operation(
        self,
        buf: ByteBuffer,
        op: Operation,
        value_numbers: _ValueNumbering,
        block_indices: dict[int, int],
    ) -> None:
        """Write a single operation."""
        buf.write_varint(self._ctx.ops[op.name] + 1)
        buf.write_u8(0)  # flags
        location_id = (
            0 if self._location_mode == LOCATION_MODE_NO_LOCATIONS else op.location_id
        )
        buf.write_varint(location_id)
        self._write_source_trivia(buf, op.leading_blank_line, op.comments)

        # Operands.
        buf.write_varint(len(op.operands))
        for operand_id in op.operands:
            value_number = self._value_number_or_error(
                value_numbers, operand_id, "operation operand"
            )
            buf.write_varint(value_number)
        op_decl = self._op_decls_by_name.get(op.name)
        if op_decl is not None:
            layout = compute_layout(op_decl)
            if layout.segmented_operands:
                if len(op.operand_segment_counts) != len(op_decl.operands):
                    raise ValueError(
                        f"operation {op.name} has "
                        f"{len(op.operand_segment_counts)} operand segment counts, "
                        f"expected {len(op_decl.operands)}"
                    )
                if sum(op.operand_segment_counts) != len(op.operands):
                    raise ValueError(
                        f"operation {op.name} operand segment counts do not "
                        "sum to operand count"
                    )
                for count in op.operand_segment_counts:
                    buf.write_varint(count)

        # Successors are encoded as region-local block ordinals instead of
        # labels. Labels are optional text metadata; the edge is semantic.
        buf.write_varint(len(op.successors))
        for successor in op.successors:
            try:
                block_index = block_indices[id(successor)]
            except KeyError as exc:
                raise ValueError(
                    f"operation {op.name} successor targets a block outside "
                    "its enclosing region"
                ) from exc
            buf.write_varint(block_index)

        # Results.
        buf.write_varint(len(op.results))
        for result_id in op.results:
            value = self._module.values[result_id]
            self._write_value_def(buf, value, value_numbers)

        # Tied results.
        buf.write_varint(len(op.tied_results))
        for tied in op.tied_results:
            buf.write_varint(tied.result_index)
            buf.write_varint(tied.operand_index)

        # Attributes.
        buf.write_varint(len(op.attributes))
        # Operation attributes are canonicalized at IR construction time, so
        # emit the stored order directly.
        for key, value in op.attributes.items():
            buf.write_varint(self._ctx.strings[key])
            self._write_attr_value(
                buf,
                value,
                value_numbers,
                self._attr_def_for_op_attr(op.name, key),
            )

        # Regions.
        buf.write_varint(len(op.regions))
        for region in op.regions:
            self._write_region(buf, region, value_numbers)

    def _write_parameterized_attr_payload(
        self,
        buf: ByteBuffer,
        value: ParameterizedAttr,
        value_numbers: _ValueNumbering | None,
        attr_def: Any | None,
        aggregate_nesting_depth: int,
        completed_types: bool,
    ) -> None:
        """Write a parameterized attribute payload without its kind byte."""
        if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
            raise ValueError(
                "aggregate attribute nesting exceeds maximum depth "
                f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
            )
        expected_definition = getattr(attr_def, "parameterized_attr", None)
        if (
            expected_definition is not None
            and expected_definition.name != value.family_name
        ):
            raise TypeError(
                "parameterized attribute family "
                f"{value.family_name!r} does not match the field contract"
            )
        buf.write_varint(self._ctx.strings[value.family_name])
        present_items = value.present_items()
        buf.write_varint(len(present_items))
        parameter_by_name = {
            parameter.name: parameter for parameter in value.definition.parameters
        }
        for parameter_name, parameter_value in present_items:
            buf.write_varint(self._ctx.strings[parameter_name])
            self._write_attr_value(
                buf,
                parameter_value,
                value_numbers,
                parameter_by_name[parameter_name],
                aggregate_nesting_depth + 1,
                completed_types=completed_types,
            )

    def _write_parameterized_attr_value(
        self,
        buf: ByteBuffer,
        value: ParameterizedAttr,
        value_numbers: _ValueNumbering | None,
        attr_def: Any | None,
        aggregate_nesting_depth: int,
        completed_types: bool,
    ) -> None:
        """Write a complete descriptor-backed parameterized attribute."""
        if attr_def is not None and getattr(attr_def, "attr_type", None) != (
            "parameterized"
        ):
            raise TypeError(
                "parameterized attribute value does not match the field contract"
            )
        buf.write_u8(ATTR_KIND_PARAMETERIZED)
        self._write_parameterized_attr_payload(
            buf,
            value,
            value_numbers,
            attr_def,
            aggregate_nesting_depth,
            completed_types,
        )

    def _write_parameterized_attr_array_value(
        self,
        buf: ByteBuffer,
        value: ParameterizedAttrArray,
        value_numbers: _ValueNumbering | None,
        attr_def: Any | None,
        aggregate_nesting_depth: int,
        completed_types: bool,
    ) -> None:
        """Write a descriptor-backed ordered parameterized attribute array."""
        if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
            raise ValueError(
                "aggregate attribute nesting exceeds maximum depth "
                f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
            )
        if getattr(attr_def, "attr_type", None) != "parameterized_array":
            raise TypeError(
                "parameterized attribute arrays require a descriptor-backed field"
            )
        buf.write_u8(ATTR_KIND_PARAMETERIZED_ARRAY)
        buf.write_varint(len(value))
        for element in value:
            self._write_parameterized_attr_payload(
                buf,
                element,
                value_numbers,
                attr_def,
                aggregate_nesting_depth + 1,
                completed_types,
            )

    def _write_dict_attr_value(
        self,
        buf: ByteBuffer,
        value: Mapping[str, Any],
        value_numbers: _ValueNumbering | None,
        aggregate_nesting_depth: int,
        completed_types: bool,
    ) -> None:
        """Write a canonical generic attribute dictionary."""
        if aggregate_nesting_depth >= ATTR_AGGREGATE_MAX_NESTING_DEPTH:
            raise ValueError(
                "aggregate attribute nesting exceeds maximum depth "
                f"{ATTR_AGGREGATE_MAX_NESTING_DEPTH}"
            )
        buf.write_u8(ATTR_KIND_DICT)
        buf.write_varint(len(value))
        for key, item in value.items():
            buf.write_varint(self._ctx.intern_string(key))
            self._write_attr_value(
                buf,
                item,
                value_numbers,
                aggregate_nesting_depth=aggregate_nesting_depth + 1,
                completed_types=completed_types,
            )

    def _dispatch_parameterized_attr_value(
        self,
        buf: ByteBuffer,
        value: Any,
        value_numbers: _ValueNumbering | None,
        attr_def: Any | None,
        aggregate_nesting_depth: int,
        completed_types: bool,
    ) -> bool:
        """Writes or rejects a parameterized field, returning whether handled."""
        attr_type = getattr(attr_def, "attr_type", None)
        if isinstance(value, ParameterizedAttr):
            self._write_parameterized_attr_value(
                buf,
                value,
                value_numbers,
                attr_def,
                aggregate_nesting_depth,
                completed_types,
            )
            return True
        if isinstance(value, ParameterizedAttrArray):
            self._write_parameterized_attr_array_value(
                buf,
                value,
                value_numbers,
                attr_def,
                aggregate_nesting_depth,
                completed_types,
            )
            return True
        if attr_type == "parameterized":
            raise TypeError(
                "parameterized attribute field requires ParameterizedAttr, "
                f"got {value!r}"
            )
        if attr_type == "parameterized_array":
            raise TypeError(
                "parameterized attribute array field requires "
                f"ParameterizedAttrArray, got {value!r}"
            )
        return False

    def _write_attr_value(
        self,
        buf: ByteBuffer,
        value: Any,
        value_numbers: _ValueNumbering | None = None,
        attr_def: Any | None = None,
        aggregate_nesting_depth: int = 0,
        *,
        completed_types: bool = False,
    ) -> None:
        """Write an attribute value with its kind byte."""
        attr_type = getattr(attr_def, "attr_type", None)
        if self._dispatch_parameterized_attr_value(
            buf,
            value,
            value_numbers,
            attr_def,
            aggregate_nesting_depth,
            completed_types,
        ):
            return
        if self._dispatch_symbol_attr_value(buf, value, attr_def):
            return
        if attr_type == "predicate_list" or isinstance(value, PredicateListAttr):
            if not isinstance(value, list | PredicateListAttr) or not all(
                isinstance(predicate, Predicate) for predicate in value
            ):
                raise TypeError(
                    "predicate-list attribute value must contain Predicate "
                    f"objects, got {value!r}"
                )
            buf.write_u8(ATTR_KIND_PREDICATE_LIST)
            self._write_predicate_list(buf, value, value_numbers)
            return
        if self._dispatch_enum_attr_value(buf, value, attr_def):
            return
        if attr_type == "scoped_enum":
            if not isinstance(value, str):
                raise TypeError(
                    "scoped enum attribute value must be a stable string key, "
                    f"got {value!r}"
                )
            buf.write_u8(ATTR_KIND_SCOPED_ENUM)
            buf.write_varint(self._ctx.strings[value])
            return
        if attr_type == "type" or isinstance(value, _IR_TYPE_CLASSES):
            if not isinstance(value, _IR_TYPE_CLASSES):
                raise TypeError(f"type attribute value must be a Type, got {value!r}")
            buf.write_u8(ATTR_KIND_TYPE)
            if completed_types:
                buf.write_varint(self._type_reference(value, value_numbers))
            else:
                self._write_type_use(buf, value, value_numbers)
            return
        if attr_type == "bytes":
            if not isinstance(value, bytes | bytearray):
                raise TypeError(f"bytes attribute value must be bytes, got {value!r}")
            data = bytes(value)
            buf.write_u8(ATTR_KIND_BYTES)
            buf.write_varint(len(data))
            buf.write_bytes(data)
            return
        if attr_type == "string_array" or isinstance(value, StringArrayAttr):
            if not isinstance(value, StringArrayAttr):
                raise TypeError(
                    "string-array attribute value must be StringArrayAttr, "
                    f"got {value!r}"
                )
            if attr_type not in (None, "any", "string_array"):
                raise TypeError(
                    f"string-array attribute does not match field kind {attr_type!r}"
                )
            buf.write_u8(ATTR_KIND_STRING_ARRAY)
            buf.write_varint(len(value))
            for string in value:
                buf.write_varint(self._ctx.strings[string])
            return
        if isinstance(value, SymbolName):
            buf.write_u8(ATTR_KIND_SYMBOL)
            buf.write_varint(self._ctx.strings[str(value)])
            return
        if isinstance(value, bool):
            buf.write_u8(ATTR_KIND_BOOL)
            buf.write_u8(1 if value else 0)
        elif isinstance(value, int):
            buf.write_u8(ATTR_KIND_I64)
            buf.write_signed_varint(value)
        elif isinstance(value, float):
            buf.write_u8(ATTR_KIND_F64)
            buf.write_bytes(struct.pack("<d", value))
        elif isinstance(value, str):
            buf.write_u8(ATTR_KIND_STRING)
            buf.write_varint(self._ctx.strings[value])
        elif isinstance(value, bytes | bytearray):
            data = bytes(value)
            buf.write_u8(ATTR_KIND_BYTES)
            buf.write_varint(len(data))
            buf.write_bytes(data)
        elif isinstance(value, EnumArrayAttr):
            raise ValueError("enum arrays require a descriptor-backed field")
        elif isinstance(value, SignedEnumSetAttr):
            raise ValueError("signed enum sets require a descriptor-backed field")
        elif isinstance(value, SymbolNameArray):
            raise ValueError("symbol arrays require a descriptor-backed field")
        elif isinstance(value, SymbolNameSet):
            raise ValueError("symbol sets require a descriptor-backed field")
        elif isinstance(value, Mapping):
            self._write_dict_attr_value(
                buf, value, value_numbers, aggregate_nesting_depth, completed_types
            )
        elif isinstance(value, EncodingInstance):
            buf.write_u8(ATTR_KIND_ENCODING)
            buf.write_varint(self._ctx.encoding_indices[id(value)] + 1)
        elif isinstance(value, list | tuple):
            # Check if all ints → i64_array.
            if all(isinstance(v, int) for v in value):
                buf.write_u8(ATTR_KIND_I64_ARRAY)
                buf.write_varint(len(value))
                for v in value:
                    buf.write_signed_varint(v)
            else:
                # Mixed array — serialize as string.
                buf.write_u8(ATTR_KIND_STRING)
                buf.write_varint(self._ctx.strings[str(value)])
        else:
            buf.write_u8(ATTR_KIND_STRING)
            buf.write_varint(self._ctx.strings[str(value)])

    def _dispatch_enum_attr_value(
        self, buf: ByteBuffer, value: Any, attr_def: Any | None
    ) -> bool:
        """Writes a descriptor-backed enum-shaped field when applicable."""
        attr_type = getattr(attr_def, "attr_type", None)
        if attr_type == "enum":
            enum_def = getattr(attr_def, "enum_def", None)
            if isinstance(value, str):
                for enum_case in getattr(enum_def, "cases", ()):
                    if enum_case.keyword == value:
                        value = enum_case.value
                        break
                else:
                    raise ValueError(
                        f"enum attribute value {value!r} is not declared by "
                        f"{getattr(enum_def, 'name', 'enum')}"
                    )
            if type(value) is not int or value < 0 or value > 0xFF:
                raise TypeError(
                    "enum attribute value must be a keyword or uint8 ordinal, "
                    f"got {value!r}"
                )
            if not getattr(attr_def, "open_enum", False):
                known_values = {
                    enum_case.value for enum_case in getattr(enum_def, "cases", ())
                }
                if value not in known_values:
                    raise ValueError(
                        f"enum attribute ordinal {value} is not declared by "
                        f"{getattr(enum_def, 'name', 'enum')}"
                    )
            buf.write_u8(ATTR_KIND_ENUM)
            buf.write_u8(value)
            return True
        if attr_type == "enum_array":
            if not isinstance(value, EnumArrayAttr):
                raise TypeError(
                    f"enum-array attribute value must be EnumArrayAttr, got {value!r}"
                )
            buf.write_u8(ATTR_KIND_ENUM_ARRAY)
            buf.write_varint(len(value))
            buf.write_bytes(bytes(value.values))
            return True
        if attr_type == "signed_enum_set":
            self._write_signed_enum_set_attr_value(buf, value, attr_def)
            return True
        return False

    def _write_signed_enum_set_attr_value(
        self, buf: ByteBuffer, value: Any, attr_def: Any
    ) -> None:
        if not isinstance(value, SignedEnumSetAttr):
            raise TypeError(
                "signed enum-set attribute value must be SignedEnumSetAttr, "
                f"got {value!r}"
            )
        if getattr(attr_def, "open_enum", False):
            raise ValueError("signed enum sets require a closed enum descriptor")
        enum_def = getattr(attr_def, "enum_def", None)
        if enum_def is None:
            raise ValueError("signed enum sets require an enum descriptor")
        declared_values = {enum_case.value for enum_case in enum_def.cases}
        asserted_values = value.positive_values + value.negative_values
        undeclared_values = sorted(set(asserted_values) - declared_values)
        if undeclared_values:
            raise ValueError(
                "signed enum set contains undeclared stable value(s) "
                f"{undeclared_values}"
            )

        word_count = max(asserted_values, default=-1) // 64 + 1
        positive_words = [0] * word_count
        negative_words = [0] * word_count
        for stable_value in value.positive_values:
            positive_words[stable_value // 64] |= 1 << (stable_value % 64)
        for stable_value in value.negative_values:
            negative_words[stable_value // 64] |= 1 << (stable_value % 64)

        buf.write_u8(ATTR_KIND_SIGNED_ENUM_SET)
        buf.write_u8(word_count)
        for word in positive_words + negative_words:
            buf.write_u64_le(word)

    def _dispatch_symbol_attr_value(
        self,
        buf: ByteBuffer,
        value: Any,
        attr_def: Any | None,
    ) -> bool:
        """Write a descriptor-backed symbol payload when applicable."""
        attr_type = getattr(attr_def, "attr_type", None)
        if attr_type == "symbol":
            if not isinstance(value, str):
                raise TypeError(
                    f"symbol attribute value must be a string, got {value!r}"
                )
            buf.write_u8(ATTR_KIND_SYMBOL)
            buf.write_varint(self._ctx.strings[value])
            return True
        if attr_type == "symbol_array":
            if not isinstance(value, SymbolNameArray):
                raise TypeError(
                    "symbol-array attribute value must be SymbolNameArray, "
                    f"got {value!r}"
                )
            buf.write_u8(ATTR_KIND_SYMBOL_ARRAY)
            buf.write_varint(len(value))
            for name in value:
                buf.write_varint(self._ctx.strings[str(name)])
            return True
        if attr_type == "symbol_set":
            if not isinstance(value, SymbolNameSet):
                raise TypeError(
                    f"symbol-set attribute value must be SymbolNameSet, got {value!r}"
                )
            buf.write_u8(ATTR_KIND_SYMBOL_SET)
            buf.write_varint(len(value))
            for name in value:
                buf.write_varint(self._ctx.strings[str(name)])
            return True
        return False

    # Predicate arg tag bytes.
    _PRED_ARG_TAG_VALUE = 1
    _PRED_ARG_TAG_CONST = 2

    # Predicate kind name → byte mapping.
    _PRED_KIND_BYTES: ClassVar[dict[str, int]] = {
        name: kind for kind, name in enumerate(PREDICATE_KINDS)
    }

    def _write_predicate_list(
        self,
        buf: ByteBuffer,
        predicates: Sequence[Predicate],
        value_numbers: _ValueNumbering | None = None,
    ) -> None:
        """Write a predicate list: count + per-predicate data."""
        buf.write_varint(len(predicates))
        for predicate in predicates:
            kind_byte = self._PRED_KIND_BYTES.get(predicate.kind)
            if kind_byte is None:
                raise ValueError(f"unknown predicate kind: {predicate.kind!r}")
            buf.write_u8(kind_byte)
            buf.write_u8(len(predicate.args))
            for arg in predicate.args:
                self._write_predicate_arg(buf, arg, value_numbers)

    def _write_predicate_arg(
        self,
        buf: ByteBuffer,
        arg: PredicateArg,
        value_numbers: _ValueNumbering | None = None,
    ) -> None:
        """Write a single predicate argument: tag + value."""
        match arg.tag:
            case "value":
                buf.write_u8(self._PRED_ARG_TAG_VALUE)
                if value_numbers is None:
                    raise ValueError("predicate value reference requires a value scope")
                buf.write_varint(
                    self._value_number_or_error(
                        value_numbers, arg.value, "predicate argument"
                    )
                )
            case "const":
                buf.write_u8(self._PRED_ARG_TAG_CONST)
                buf.write_signed_varint(arg.value)
            case _:
                raise ValueError(f"unknown predicate arg tag: {arg.tag!r}")

    def _write_function_implementation_metadata(
        self,
        buf: ByteBuffer,
        op: Operation,
        symbol_name: str,
        func_like: FuncLikeInterface,
    ) -> None:
        """Write the fixed implementation metadata declared by a FuncLike op."""
        if func_like.template_family is None:
            return
        template_family = op.attributes.get(func_like.template_family)
        if not isinstance(template_family, str):
            raise ValueError(
                f"{op.name} symbol {symbol_name!r} must have a template family symbol"
            )
        try:
            template_family_symbol_ordinal = self._wire_symbol_indices[template_family]
        except KeyError as exc:
            raise ValueError(
                f"{op.name} symbol {symbol_name!r} references unknown template "
                f"family {template_family!r}"
            ) from exc
        priority = (
            op.attributes.get(func_like.priority, 0)
            if func_like.priority is not None
            else 0
        )
        if not isinstance(priority, int) or priority < 0:
            raise ValueError(
                f"{op.name} symbol {symbol_name!r} priority "
                "must be a non-negative integer"
            )
        buf.write_varint(template_family_symbol_ordinal)
        buf.write_varint(priority)

    def _write_symbols(
        self, ir_regions: dict[int, list[tuple[int, int, int]]]
    ) -> bytes:
        """Write the SYMBOLS section.

        Section layout: symbol_count, import_count, export_count,
        import offset table, export offset table, symbol entries.
        Import/export offset tables are uint64 byte offsets from the
        start of the symbol entries to each import/export entry.
        """
        buf = ByteBuffer()
        symbols = self._wire_symbols

        # Classify symbols into imports and exports.
        import_indices: list[int] = []
        export_indices: list[int] = []
        exported_symbols = [self._symbol_is_exported(symbol) for symbol in symbols]
        for i, symbol in enumerate(symbols):
            is_import = (symbol.flags & SYMBOL_FLAG_IMPORT) != 0
            if is_import:
                import_indices.append(i)
            elif exported_symbols[i]:
                export_indices.append(i)

        buf.write_varint(len(symbols))
        buf.write_varint(len(import_indices))
        buf.write_varint(len(export_indices))
        buf.write_varint(sum(len(payloads) for payloads in ir_regions.values()))

        # Reserve space for offset tables (patched after writing entries).
        import_table_offset = buf.position
        for _ in import_indices:
            buf.write_u64_le(0)
        export_table_offset = buf.position
        for _ in export_indices:
            buf.write_u64_le(0)

        # Track the start of symbol entries for offset computation.
        entries_start = buf.position

        # Maps symbol index → byte offset from entries_start.
        entry_offsets: dict[int, int] = {}

        for symbol_index, symbol in enumerate(symbols):
            entry_offsets[symbol_index] = buf.position - entries_start
            buf.write_varint(self._ctx.strings[symbol.name])
            buf.write_u8(symbol.kind.value)
            buf.write_u8(0 if (symbol.flags & SYMBOL_FLAG_PUBLIC) else 1)
            if symbol.op is None:
                raise ValueError(f"symbol {symbol.name!r} has no defining op")
            bytecode_flags = (
                symbol.flags & ~(SYMBOL_FLAG_DECLARATION | SYMBOL_FLAG_TEST_ONLY)
            ) | self._symbol_definition_flags(symbol.op)
            if exported_symbols[symbol_index]:
                bytecode_flags |= SYMBOL_FLAG_EXPORT
            if symbol.source_symbol and symbol.source_symbol != symbol.name:
                bytecode_flags |= SYMBOL_FLAG_IMPORT_SYMBOL
            predicates_attr_name: str | None = None
            if symbol.kind in FUNCTION_SYMBOL_KINDS:
                func_like = func_like_interface_for_op(
                    self._op_decls_by_name, symbol.op.name
                )
                if func_like is not None:
                    predicates_attr_name = getattr(func_like, "predicates", None)
            if (
                predicates_attr_name is not None
                and predicates_attr_name in symbol.op.attributes
            ):
                bytecode_flags |= _SYMBOL_FLAG_PREDICATES
            buf.write_u16_le(bytecode_flags)
            buf.write_varint(
                0
                if self._location_mode == LOCATION_MODE_NO_LOCATIONS
                else symbol.op.location_id
            )

            # Import metadata: source module and symbol for cross-module refs.
            if symbol.flags & SYMBOL_FLAG_IMPORT:
                buf.write_varint(self._ctx.strings[symbol.source_module])
                # source_symbol defaults to the symbol's own name.
                source_sym = symbol.source_symbol or symbol.name
                buf.write_varint(self._ctx.strings[source_sym])

            if symbol.kind in FUNCTION_SYMBOL_KINDS and symbol.op is not None:
                op = symbol.op
                module = self._module
                func_like = func_like_interface_for_op(self._op_decls_by_name, op.name)
                if func_like is None:
                    raise ValueError(f"function symbol {op.name!r} is not FuncLike")

                buf.write_varint(self._ctx.ops[op.name] + 1)
                self._write_source_trivia(buf, op.leading_blank_line, op.comments)

                _cc_to_byte: dict[str, int] = {
                    "host": 1,
                    "device": 2,
                    "initializer": 3,
                    "deinitializer": 4,
                }
                cc_byte = _cc_to_byte.get(op.attributes.get("cc", ""), 0)
                buf.write_u8(cc_byte)
                _effect_contract_to_byte: dict[str, int] = {
                    "pure": 1,
                    "readonly": 2,
                }
                effect_contract_byte = _effect_contract_to_byte.get(
                    op.attributes.get("effect_contract", ""), 0
                )
                buf.write_u8(effect_contract_byte)

                workload_arg_ids = self._kernel_workload_arg_ids(op)
                arg_ids = self._func_arg_ids(op)

                # Result types and tied results.
                result_ids = op.results
                tied_results = op.tied_results
                signature_value_numbers = _ValueNumbering(
                    {
                        value_id: value_number
                        for value_number, value_id in enumerate(
                            [*workload_arg_ids, *arg_ids, *result_ids]
                        )
                    }
                )

                buf.write_varint(len(workload_arg_ids))
                buf.write_varint(len(arg_ids))
                buf.write_varint(len(result_ids))

                for workload_arg_id in workload_arg_ids:
                    self._write_value_def(
                        buf,
                        module.values[workload_arg_id],
                        signature_value_numbers,
                    )
                for arg_id in arg_ids:
                    self._write_value_def(
                        buf, module.values[arg_id], signature_value_numbers
                    )
                for i, result_id in enumerate(result_ids):
                    is_tied = any(t.result_index == i for t in tied_results)
                    buf.write_u8(1 if is_tied else 0)
                    self._write_value_def(
                        buf, module.values[result_id], signature_value_numbers
                    )
                    if is_tied:
                        tied = next(t for t in tied_results if t.result_index == i)
                        buf.write_varint(tied.operand_index)

                buf.write_varint(len(tied_results))
                predicates = (
                    op.attributes.get(predicates_attr_name, [])
                    if predicates_attr_name is not None
                    else []
                )
                self._write_predicate_list(
                    buf,
                    predicates,
                    signature_value_numbers,
                )

                self._write_function_implementation_metadata(
                    buf, op, symbol.name, func_like
                )

                shared_attr_keys = self._shared_func_metadata_attr_keys(op)
                payload_attrs = [
                    (key, value)
                    for key, value in op.attributes.items()
                    if key not in shared_attr_keys
                ]
                buf.write_varint(len(payload_attrs))
                for key, value in payload_attrs:
                    buf.write_varint(self._ctx.strings[key])
                    self._write_attr_value(
                        buf,
                        value,
                        signature_value_numbers,
                        self._attr_def_for_op_attr(op.name, key),
                    )

                payloads = ir_regions.get(symbol_index, ())
                buf.write_varint(len(payloads))
                for region_index, offset, length in payloads:
                    buf.write_u8(region_index)
                    buf.write_u64_le(offset)
                    buf.write_u32_le(length)
            elif symbol.kind == SymbolKind.GLOBAL and symbol.op is not None:
                op = symbol.op
                module = self._module
                symbol_field = self._symbol_field_for_op(op)
                if op.operands or op.regions or not op.results:
                    raise ValueError(
                        f"global symbol {symbol.name!r} must be defined by a "
                        "result-only top-level op"
                    )

                buf.write_varint(self._ctx.ops[op.name] + 1)
                self._write_source_trivia(buf, op.leading_blank_line, op.comments)

                result_ids = list(op.results)
                local_value_ids = self._collect_global_local_values(op)
                local_value_numbers = _ValueNumbering(
                    {
                        value_id: value_number
                        for value_number, value_id in enumerate(local_value_ids)
                    }
                )
                buf.write_varint(len(result_ids))
                buf.write_varint(len(local_value_ids))
                for value_id in local_value_ids:
                    self._write_value_def(
                        buf, module.values[value_id], local_value_numbers
                    )

                payload_attrs = [
                    (key, value)
                    for key, value in op.attributes.items()
                    if key != symbol_field
                ]
                buf.write_varint(len(payload_attrs))
                for key, value in payload_attrs:
                    buf.write_varint(self._ctx.strings[key])
                    self._write_attr_value(
                        buf,
                        value,
                        local_value_numbers,
                        self._attr_def_for_op_attr(op.name, key),
                    )
            elif symbol.kind == SymbolKind.RECORD and symbol.op is not None:
                op = symbol.op
                symbol_field = self._symbol_field_for_op(op)
                if op.operands or op.results or len(op.regions) > 1:
                    raise ValueError(
                        f"record symbol {symbol.name!r} must be defined by an "
                        "operandless/resultless top-level op with at most one "
                        "body region"
                    )

                buf.write_varint(self._ctx.ops[op.name] + 1)
                self._write_source_trivia(buf, op.leading_blank_line, op.comments)

                payload_attrs = [
                    (key, value)
                    for key, value in op.attributes.items()
                    if key != symbol_field
                ]
                buf.write_varint(len(payload_attrs))
                for key, value in payload_attrs:
                    buf.write_varint(self._ctx.strings[key])
                    self._write_attr_value(
                        buf,
                        value,
                        attr_def=self._attr_def_for_op_attr(op.name, key),
                    )
                payloads = ir_regions.get(symbol_index, ())
                buf.write_varint(len(payloads))
                for region_index, offset, length in payloads:
                    buf.write_u8(region_index)
                    buf.write_u64_le(offset)
                    buf.write_u32_le(length)
            else:
                raise ValueError(
                    f"symbol {symbol.name!r} of kind {symbol.kind.name} "
                    "has no supported defining op"
                )

        # Patch import/export offset tables.
        for table_idx, symbol_idx in enumerate(import_indices):
            buf.patch_u64_le(
                import_table_offset + table_idx * 8,
                entry_offsets[symbol_idx],
            )
        for table_idx, symbol_idx in enumerate(export_indices):
            buf.patch_u64_le(
                export_table_offset + table_idx * 8,
                entry_offsets[symbol_idx],
            )

        return buf.get_bytes()

    def _write_symbol_references(self) -> bytes:
        """Write direct dependency and abstract-provider demand rows."""
        buf = ByteBuffer()
        total_dependency_count = len(self._module_dependencies) + sum(
            len(row) for row in self._symbol_dependencies
        )
        total_template_demand_count = sum(
            len(row) for row in self._symbol_template_demands
        )
        buf.write_varint(len(self._wire_symbols))
        buf.write_varint(total_dependency_count)
        buf.write_varint(total_template_demand_count)
        buf.write_varint(len(self._module_dependencies))
        for dependency in self._module_dependencies:
            buf.write_varint(dependency.source_root_region_index_plus_one)
            buf.write_varint(dependency.target_symbol_index)
            buf.write_varint(dependency.contract)
        for dependencies, template_demands in zip(
            self._symbol_dependencies,
            self._symbol_template_demands,
            strict=True,
        ):
            buf.write_varint(len(dependencies))
            for dependency in dependencies:
                buf.write_varint(dependency.source_root_region_index_plus_one)
                buf.write_varint(dependency.target_symbol_index)
                buf.write_varint(dependency.contract)
            buf.write_varint(len(template_demands))
            for demand in template_demands:
                buf.write_varint(demand.source_root_region_index_plus_one)
                buf.write_varint(demand.target_symbol_index)
        return buf.get_bytes()

    # --- File assembly ---

    def _assemble(
        self, sections: dict[int, bytes], allocation_counts: tuple[int, int, int, int]
    ) -> bytes:
        """Assemble the complete .loombc file."""
        buf = ByteBuffer()

        # File header.
        buf.write_bytes(MAGIC)
        buf.write_u8(FORMAT_VERSION)
        buf.write_u8(self._location_mode)
        buf.write_u16_le(1)  # module_count = 1
        # File string pool: just the module name for now.
        module_name = self._module.name.encode("utf-8")
        buf.write_u32_le(len(module_name))
        buf.write_u32_le(0)  # reserved
        buf.write_null_terminated_string(PRODUCER)
        buf.pad_to_alignment(8)

        # Module directory (1 entry).
        buf.write_u32_le(0)  # name_offset (into string pool)
        buf.write_u16_le(len(module_name))
        buf.write_u16_le(0)  # module flags
        module_offset_patch = buf.position
        buf.write_u64_le(0)  # module_offset (patched later)
        buf.write_u64_le(0)  # module_length (patched later)

        # File string pool.
        buf.write_bytes(module_name)
        buf.pad_to_alignment(8)

        # Module data starts here.
        module_start = buf.position
        buf.patch_u64_le(module_offset_patch, module_start)

        # Module allocation summary and section directory.
        section_kinds = [kind for kind in SECTION_WRITE_ORDER if kind in sections]
        buf.write_varint(len(section_kinds))
        for count in allocation_counts:
            buf.write_varint(count)

        # Reserve space for section directory (32 bytes per entry).
        section_dir_entries: list[int] = []
        for _ in section_kinds:
            entry_offset = buf.position
            section_dir_entries.append(entry_offset)
            buf.write_u16_le(0)  # section_kind (patched)
            buf.write_u16_le(0)  # flags
            buf.write_u32_le(0)  # reserved
            buf.write_u64_le(0)  # offset (patched)
            buf.write_u64_le(0)  # length (patched)
            buf.write_u64_le(0)  # uncompressed_length

        # Write sections and patch directory.
        for i, kind in enumerate(section_kinds):
            section_data = sections[kind]
            section_offset = buf.position - module_start
            section_length = len(section_data)

            # Patch directory entry.
            entry_offset = section_dir_entries[i]
            struct.pack_into("<H", buf._data, entry_offset, kind)
            struct.pack_into("<Q", buf._data, entry_offset + 8, section_offset)
            struct.pack_into("<Q", buf._data, entry_offset + 16, section_length)

            buf.write_bytes(section_data)

        # Patch module length.
        module_length = buf.position - module_start
        buf.patch_u64_le(module_offset_patch + 8, module_length)

        return buf.get_bytes()


# ============================================================================
# Convenience function
# ============================================================================


def write_module(
    module: Module,
    *,
    location_mode: int = LOCATION_MODE_SOURCE_LOCATIONS,
    op_decls: Iterable[Any] | None = None,
) -> bytes:
    """Write a module to .loombc bytes."""
    return BytecodeWriter(
        module, location_mode=location_mode, op_decls=op_decls
    ).write()
