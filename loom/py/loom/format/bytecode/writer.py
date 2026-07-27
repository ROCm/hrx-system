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
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, cast

from loom.fields import compute_layout
from loom.format.bytecode.encoding import ByteBuffer
from loom.format.bytecode.op_decls import (
    attr_def_for_op,
    build_op_decl_map,
    func_like_interface_for_op,
    symbol_def_for_op,
)
from loom.ir import (
    Block,
    BufferType,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    FileLocation,
    FunctionType,
    FusedLocation,
    GroupType,
    Module,
    NoneType,
    OpaqueLocation,
    Operation,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    RegisterType,
    ScalarType,
    ShapedType,
    StaticDim,
    StorageType,
    SymbolKind,
    SymbolName,
    TaggedLocation,
    Type,
    TypeKind,
    Value,
)

_IR_TYPE_CLASSES = (
    ScalarType,
    ShapedType,
    BufferType,
    GroupType,
    FunctionType,
    RegisterType,
    StorageType,
    DialectType,
    EncodingType,
    PoolType,
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

SECTION_WRITE_ORDER = (
    SECTION_IR,
    SECTION_SYMBOLS,
    SECTION_STRINGS,
    SECTION_SOURCES,
    SECTION_TYPES,
    SECTION_ENCODINGS,
    SECTION_OPS,
    SECTION_LOCATIONS,
)

FUNCTION_SYMBOL_KINDS = frozenset(
    {
        SymbolKind.FUNC_DEF,
        SymbolKind.FUNC_DECL,
        SymbolKind.FUNC_TEMPLATE,
        SymbolKind.FUNC_UKERNEL,
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

# Type kind bytes. These must match loom_bytecode_type_kind_e, not just the
# current Python enum spelling.
BYTECODE_TYPE_KIND_BY_IR_KIND: dict[TypeKind, int] = {
    TypeKind.NONE: 0,
    TypeKind.SCALAR: 1,
    TypeKind.TILE: 2,
    TypeKind.TENSOR: 3,
    TypeKind.GROUP: 4,
    TypeKind.FUNCTION: 5,
    TypeKind.DIALECT: 6,
    TypeKind.ENCODING: 7,
    TypeKind.POOL: 8,
    TypeKind.VECTOR: 9,
    TypeKind.VIEW: 10,
    TypeKind.BUFFER: 11,
    TypeKind.REGISTER: 12,
    TypeKind.STORAGE: 13,
}

BYTECODE_IR_KIND_BY_TYPE_KIND: dict[int, TypeKind] = {
    type_kind: ir_kind for ir_kind, type_kind in BYTECODE_TYPE_KIND_BY_IR_KIND.items()
}

# File magic and version.
MAGIC = b"LOOM"
FORMAT_VERSION = 16
PRODUCER = "loom-py"

SYMBOL_FLAG_PUBLIC = 0x0001
SYMBOL_FLAG_IMPORT = 0x0002
SYMBOL_FLAG_IMPORT_SYMBOL = 0x0004
SYMBOL_FLAG_RETAIN = 0x0008


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
        self._type_lookup: dict[Type, int] = {}
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
        if ir_type in self._type_lookup:
            return self._type_lookup[ir_type]
        type_id = len(self._type_list)
        self._type_list.append(ir_type)
        self._type_lookup[ir_type] = type_id
        return type_id

    def intern_op(self, op_name: str) -> int:
        """Intern an op name, returning its kind ID."""
        if op_name in self.ops:
            return self.ops[op_name]
        op_id = len(self.ops)
        self.ops[op_name] = op_id
        return op_id

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
        self._number_module()

    # --- Pass 1: Numbering ---

    def _number_module(self) -> None:
        """Walk the module and assign IDs to all entities."""
        module = self._module

        # Module name.
        if module.name:
            self._ctx.intern_string(module.name)

        # Sources.
        self._ctx.sources = list(module.sources)

        # Walk symbols.
        for symbol in module.symbols:
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

        # Encodings: recursively number child encoding params before parents so
        # the ENCODINGS section has no forward references.
        for enc in module.encodings:
            self._number_encoding_instance(enc)

    def _number_func_op(self, op: Operation) -> None:
        """Number all entities in a func-like op (func.def, func.decl, etc.)."""
        module = self._module
        shared_attr_keys = self._shared_func_metadata_attr_keys(op)

        # Defining func-like op name.
        self._ctx.intern_op(op.name)
        self._ctx.intern_string(op.name)

        # Arg names/types: entry block args for defs, operands for decls.
        arg_ids = self._func_arg_ids(op)
        for arg_id in arg_ids:
            value = module.values[arg_id]
            self._ctx.intern_string(value.name)
            self._number_type(value.type)

        # Result types.
        for result_id in op.results:
            value = module.values[result_id]
            if value.name:
                self._ctx.intern_string(value.name)
            self._number_type(value.type)

        # Predicate value name strings.
        for predicate in op.attributes.get("predicates", []):
            for arg in predicate.args:
                if arg.tag == "value" and isinstance(arg.value, str):
                    self._ctx.intern_string(arg.value)

        implements = op.attributes.get("implements")
        if isinstance(implements, str):
            self._ctx.intern_string(implements)

        for key, value in op.attributes.items():
            if key in shared_attr_keys:
                continue
            self._ctx.intern_string(key)
            self._number_attr_value(value, self._attr_def_for_op_attr(op.name, key))

        for region_index in self._root_region_write_order(op):
            self._number_region(op.regions[region_index])

    def _func_body_region_index(self, op: Operation) -> int | None:
        """Return op's FuncLike body region index, if it has one."""
        func_like = func_like_interface_for_op(self._op_decls_by_name, op.name)
        if func_like is None or func_like.args_as_operands or func_like.body is None:
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
        body_region_index = self._func_body_region_index(op)
        if body_region_index is None:
            return list(op.operands)
        if body_region_index >= len(op.regions):
            return []
        body = op.regions[body_region_index]
        if not body.blocks:
            return []
        return list(body.blocks[0].arg_ids)

    def _root_region_write_order(self, op: Operation) -> list[int]:
        """Return root region indices in deterministic bytecode order.

        The FuncLike body region is serialized first so its entry arguments
        occupy the first function-local value numbers. The payload still writes
        each root region's declared index, so the reader reconstructs the
        operation's region slots instead of relying on serialized order.
        """
        if not op.regions:
            return []
        body_region_index = self._func_body_region_index(op)
        ordered: list[int] = []
        if body_region_index is not None:
            if body_region_index >= len(op.regions):
                raise ValueError(
                    f"func-like op {op.name!r} has root regions but is missing "
                    "its FuncLike body region"
                )
            ordered.append(body_region_index)
        seen = set(ordered)
        ordered.extend(i for i in range(len(op.regions)) if i not in seen)
        return ordered

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
            for field_name in ("visibility", "cc", "purity", "predicates"):
                attr_name = getattr(func_like, field_name, None)
                if attr_name is not None:
                    keys.add(attr_name)
            if symbol_def.bytecode_kind in (
                "LOOM_SYMBOL_FUNC_TEMPLATE",
                "LOOM_SYMBOL_FUNC_UKERNEL",
            ):
                for field_name in ("implements", "priority"):
                    attr_name = getattr(func_like, field_name, None)
                    if attr_name is not None:
                        keys.add(attr_name)
        return frozenset(keys)

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
                for binding_id in value.dim_bindings.values():
                    add_value(binding_id)
                if value.encoding_binding >= 0:
                    add_value(value.encoding_binding)
            return scan_index

        for result_id in op.results:
            add_value(result_id)
        scan_index = collect_value_bindings(0)

        name_to_value_id = {
            module.values[value_id].name: value_id
            for value_id in local_values
            if module.values[value_id].name
        }

        def collect_attr_value(value: Any) -> None:
            if isinstance(value, list) and value and isinstance(value[0], Predicate):
                for predicate in value:
                    for arg in predicate.args:
                        if arg.tag != "value":
                            continue
                        if isinstance(arg.value, int):
                            add_value(arg.value)
                            continue
                        try:
                            add_value(name_to_value_id[str(arg.value)])
                        except KeyError as exc:
                            raise ValueError(
                                f"global {op.name} predicate references "
                                f"unknown value {arg.value!r}"
                            ) from exc
                return
            if isinstance(value, Mapping):
                for nested_value in value.values():
                    collect_attr_value(nested_value)

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
        for block in region.blocks:
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

    def _number_type(self, ir_type: Type) -> None:
        """Ensure a type and all its sub-types are interned.

        Sub-types are interned BEFORE their parent so that the type
        table is in topological order (the reader can resolve forward
        references by index).
        """
        # Recurse into sub-types first (topological order).
        match ir_type:
            case ShapedType(element_type=elem, encoding=enc):
                self._number_type(elem)
                if isinstance(enc, EncodingInstance):
                    self._number_encoding_instance(enc)
            case FunctionType(arg_types=args, result_types=results):
                for t in args:
                    self._number_type(t)
                for t in results:
                    self._number_type(t)
            case GroupType():
                pass  # Group scope is a byte, not a string.
            case DialectType(name=name, params=params):
                self._ctx.intern_string(name)
                for p in params:
                    self._number_type(p)
            case RegisterType():
                pass
            case _:
                pass
        # Intern the parent AFTER sub-types (ensures sub-types have lower IDs).
        self._ctx.intern_type(ir_type)

    def _number_attr_value(self, value: Any, attr_def: Any | None = None) -> None:
        """Intern strings referenced by attribute values."""
        if getattr(attr_def, "attr_type", None) == "enum":
            return
        if isinstance(value, str):
            self._ctx.intern_string(value)
        elif isinstance(value, bytes | bytearray):
            pass
        elif isinstance(value, _IR_TYPE_CLASSES):
            self._ctx.intern_type(cast(Type, value))
        elif isinstance(value, EncodingInstance):
            self._number_encoding_instance(value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                self._ctx.intern_string(k)
                self._number_attr_value(v)
        elif isinstance(value, list | tuple):
            for item in value:
                self._number_attr_value(item)

    def _number_encoding_instance(self, value: EncodingInstance) -> None:
        """Intern one static encoding and any nested encoding-valued params."""
        for param_name, param_value in value.params:
            self._ctx.intern_string(param_name)
            self._number_attr_value(param_value)
        self._ctx.intern_string(value.name)
        if value.alias:
            self._ctx.intern_string(value.alias)
        self._module.add_encoding(value)

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
        ir_bytes, ir_offsets = self._write_ir()
        sections[SECTION_IR] = ir_bytes
        sections[SECTION_SYMBOLS] = self._write_symbols(ir_offsets)
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

    def _write_encodings(self) -> bytes:
        """Write the ENCODINGS section."""
        buf = ByteBuffer()
        encodings = self._module.encodings

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
        for enc in encodings:
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

    def _write_one_type(self, buf: ByteBuffer, ir_type: Type) -> None:
        """Serialize one type entry."""
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
                # Encoding attachment: 0 = none, 1 = static (table index
                # follows), 2 = dynamic SSA (value_id on the Value, not the
                # type).
                if isinstance(ir_type.encoding, DynamicEncoding):
                    buf.write_u8(2)  # dynamic SSA encoding
                    buf.write_varint(0)
                elif isinstance(ir_type.encoding, EncodingInstance):
                    # Find the encoding in the module's table.
                    enc_index = 0
                    for i, enc in enumerate(self._module.encodings):
                        if enc == ir_type.encoding:
                            enc_index = i + 1  # 1-based
                            break
                    if enc_index == 0:
                        raise ValueError(
                            f"encoding {ir_type.encoding!r} was not numbered"
                        )
                    buf.write_u8(1)  # static encoding
                    buf.write_varint(enc_index)
                else:
                    buf.write_u8(0)  # no encoding
                    buf.write_varint(0)
                for dim in ir_type.dims:
                    match dim:
                        case StaticDim(size=size):
                            buf.write_u8(0)  # is_dynamic = false
                            buf.write_varint(size)
                        case DynamicDim():
                            buf.write_u8(1)  # is_dynamic = true
            case GroupType(scope=scope):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.GROUP])
                buf.write_u8(scope.value)
            case FunctionType(arg_types=args, result_types=results):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.FUNCTION])
                buf.write_varint(len(args))
                buf.write_varint(len(results))
                for arg in args:
                    buf.write_varint(self._ctx.intern_type(arg))
                for result in results:
                    buf.write_varint(self._ctx.intern_type(result))
            case DialectType(name=name, params=params):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.DIALECT])
                buf.write_varint(self._ctx.strings[name])
                buf.write_varint(len(params))
                for param in params:
                    buf.write_varint(self._ctx.intern_type(param))
            case RegisterType(
                descriptor_set_stable_id=descriptor_set_stable_id,
                register_class_id=register_class_id,
                unit_count=unit_count,
            ):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER])
                buf.write_varint(descriptor_set_stable_id)
                buf.write_varint(register_class_id | (unit_count << 16))
            case StorageType(space=space):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.STORAGE])
                buf.write_u8(space.value)
            case EncodingType(role=role):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.ENCODING])
                buf.write_u8(role.value)
            case PoolType(block_size=block_size):
                buf.write_u8(BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.POOL])
                match block_size:
                    case StaticDim(size=size):
                        buf.write_u8(0)  # static
                        buf.write_varint(size)
                    case DynamicDim():
                        buf.write_u8(1)  # dynamic
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

    def _write_ir(self) -> tuple[bytes, dict[int, tuple[int, int]]]:
        """Write the IR section (function bodies).

        Returns (ir_bytes, ir_offsets) where ir_offsets maps symbol
        index to (offset, length) within the IR section.
        """
        buf = ByteBuffer()
        ir_offsets: dict[int, tuple[int, int]] = {}

        for symbol_index, symbol in enumerate(self._module.symbols):
            if symbol.op is not None and symbol.op.regions:
                start = buf.position
                self._write_op_region_payload(buf, symbol.op)
                length = buf.position - start
                ir_offsets[symbol_index] = (start, length)

        return buf.get_bytes(), ir_offsets

    def _write_op_region_payload(self, buf: ByteBuffer, op: Operation) -> None:
        """Write a symbol op's root regions with value numbering."""
        root_region_indices = self._root_region_write_order(op)
        assert root_region_indices, "cannot write region payload for bodyless op"

        value_numbers: dict[int, int] = {}
        for region_index in root_region_indices:
            self._assign_value_numbers(op.regions[region_index], value_numbers)

        value_count, region_count, block_count, op_count = self._count_region_forest(
            op, root_region_indices
        )
        buf.write_varint(value_count)
        buf.write_varint(region_count)
        buf.write_varint(block_count)
        buf.write_varint(op_count)
        buf.write_varint(len(root_region_indices))
        for region_index in root_region_indices:
            buf.write_varint(region_index)
            self._write_region(buf, op.regions[region_index], value_numbers)

    def _assign_value_numbers(self, region: Region, numbers: dict[int, int]) -> None:
        """Assign sequential value numbers within a function."""
        for block in region.blocks:
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
        self, value_numbers: dict[int, int], value_id: int, context: str
    ) -> int:
        try:
            return value_numbers[value_id]
        except KeyError as exc:
            raise ValueError(
                f"{context} references value {value_id} with no local bytecode number"
            ) from exc

    def _write_region(
        self, buf: ByteBuffer, region: Region, value_numbers: dict[int, int]
    ) -> None:
        """Write a region (block_count + blocks)."""
        buf.write_varint(len(region.blocks))
        block_indices = {id(block): index for index, block in enumerate(region.blocks)}
        for block in region.blocks:
            self._write_block(buf, block, value_numbers, block_indices)

    def _write_block(
        self,
        buf: ByteBuffer,
        block: Block,
        value_numbers: dict[int, int],
        block_indices: dict[int, int],
    ) -> None:
        """Write a block (label, args, ops)."""
        has_label = bool(block.label)
        buf.write_u8(1 if has_label else 0)
        if has_label:
            buf.write_varint(self._ctx.strings[block.label])
        self._write_comment_list(buf, block.comments)

        # Block args.
        buf.write_varint(len(block.arg_ids))
        for arg_id in block.arg_ids:
            self._write_value_def(buf, self._module.values[arg_id], value_numbers)

        # Operations.
        live_ops = [op for op in block.ops if not op.is_dead]
        buf.write_varint(len(live_ops))
        for op in live_ops:
            self._write_operation(buf, op, value_numbers, block_indices)

    def _write_dim_bindings(
        self, buf: ByteBuffer, value: Value, value_numbers: dict[int, int]
    ) -> None:
        """Write dim bindings and encoding binding for a value.

        Every dynamic dim in the value's type must have a corresponding
        entry in dim_bindings referencing an SSA value. Missing bindings
        indicate invalid IR (anonymous dynamic dims are not permitted).
        """
        dims = value.type.dims if hasattr(value.type, "dims") else ()
        dynamic_count = sum(1 for d in dims if isinstance(d, DynamicDim))
        if dynamic_count > 0 and len(value.dim_bindings) != dynamic_count:
            raise ValueError(
                f"value '{value.name}' has {dynamic_count} dynamic dim(s) "
                f"but {len(value.dim_bindings)} dim binding(s) — every "
                f"dynamic dim must reference an SSA value"
            )
        buf.write_varint(dynamic_count)
        for _position, value_id in sorted(value.dim_bindings.items()):
            value_number = self._value_number_or_error(
                value_numbers, value_id, "dynamic dimension binding"
            )
            buf.write_signed_varint(value_number)
        # Encoding binding: 0 = none, else 1 + value_number.
        if value.encoding_binding >= 0:
            value_number = self._value_number_or_error(
                value_numbers, value.encoding_binding, "dynamic encoding binding"
            )
            buf.write_varint(1 + value_number)
        else:
            buf.write_varint(0)

    def _write_comment_list(self, buf: ByteBuffer, comments: tuple[str, ...]) -> None:
        buf.write_varint(len(comments))
        for comment in comments:
            encoded = comment.encode("utf-8")
            buf.write_varint(len(encoded))
            buf.write_bytes(encoded)

    def _write_value_def(
        self, buf: ByteBuffer, value: Value, value_numbers: dict[int, int]
    ) -> None:
        buf.write_varint(self._ctx.strings.get(value.name, 0))
        buf.write_varint(self._ctx.intern_type(value.type))
        self._write_dim_bindings(buf, value, value_numbers)

    def _value_numbers_by_name(self, value_numbers: dict[int, int]) -> dict[str, int]:
        return {
            self._module.values[value_id].name: value_number
            for value_id, value_number in value_numbers.items()
            if self._module.values[value_id].name
        }

    def _write_operation(
        self,
        buf: ByteBuffer,
        op: Operation,
        value_numbers: dict[int, int],
        block_indices: dict[int, int],
    ) -> None:
        """Write a single operation."""
        buf.write_varint(self._ctx.ops[op.name] + 1)
        buf.write_u8(0)  # flags
        location_id = (
            0 if self._location_mode == LOCATION_MODE_NO_LOCATIONS else op.location_id
        )
        buf.write_varint(location_id)
        self._write_comment_list(buf, op.comments)

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
        value_numbers_by_name = self._value_numbers_by_name(value_numbers)
        for key, value in op.attributes.items():
            buf.write_varint(self._ctx.strings[key])
            self._write_attr_value(
                buf,
                value,
                value_numbers_by_name,
                self._attr_def_for_op_attr(op.name, key),
            )

        # Regions.
        buf.write_varint(len(op.regions))
        for region in op.regions:
            self._write_region(buf, region, value_numbers)

    def _write_attr_value(
        self,
        buf: ByteBuffer,
        value: Any,
        value_numbers_by_name: dict[str, int] | None = None,
        attr_def: Any | None = None,
    ) -> None:
        """Write an attribute value with its kind byte."""
        # Check for predicate list attribute (list of Predicate objects).
        if isinstance(value, list) and value and isinstance(value[0], Predicate):
            buf.write_u8(ATTR_KIND_PREDICATE_LIST)
            self._write_predicate_list(buf, value, value_numbers_by_name)
            return
        attr_type = getattr(attr_def, "attr_type", None)
        if attr_type == "symbol":
            if not isinstance(value, str):
                raise TypeError(
                    f"symbol attribute value must be a string, got {value!r}"
                )
            buf.write_u8(ATTR_KIND_SYMBOL)
            buf.write_varint(self._ctx.strings[value])
            return
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
            return
        if attr_type == "type":
            if not isinstance(value, _IR_TYPE_CLASSES):
                raise TypeError(f"type attribute value must be a Type, got {value!r}")
            buf.write_u8(ATTR_KIND_TYPE)
            buf.write_varint(self._ctx.intern_type(cast(Type, value)))
            return
        if attr_type == "bytes":
            if not isinstance(value, bytes | bytearray):
                raise TypeError(f"bytes attribute value must be bytes, got {value!r}")
            data = bytes(value)
            buf.write_u8(ATTR_KIND_BYTES)
            buf.write_varint(len(data))
            buf.write_bytes(data)
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
        elif isinstance(value, Mapping):
            buf.write_u8(ATTR_KIND_DICT)
            buf.write_varint(len(value))
            # Nested dict attrs are also canonicalized up front.
            for k, v in value.items():
                buf.write_varint(self._ctx.intern_string(k))
                self._write_attr_value(buf, v, value_numbers_by_name)
        elif isinstance(value, EncodingInstance):
            buf.write_u8(ATTR_KIND_ENCODING)
            buf.write_varint(self._module.add_encoding(value) + 1)
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

    # Predicate arg tag bytes.
    _PRED_ARG_TAG_VALUE = 1
    _PRED_ARG_TAG_CONST = 2

    # Predicate kind name → byte mapping.
    _PRED_KIND_BYTES: ClassVar[dict[str, int]] = {
        "eq": 0,
        "ne": 1,
        "lt": 2,
        "le": 3,
        "gt": 4,
        "ge": 5,
        "mul": 6,
        "min": 7,
        "max": 8,
        "pow2": 9,
        "range": 10,
        "not_nan": 11,
        "not_inf": 12,
        "finite": 13,
    }

    def _write_predicate_list(
        self,
        buf: ByteBuffer,
        predicates: list[Predicate],
        value_numbers_by_name: dict[str, int] | None = None,
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
                self._write_predicate_arg(buf, arg, value_numbers_by_name)

    def _write_predicate_arg(
        self,
        buf: ByteBuffer,
        arg: PredicateArg,
        value_numbers_by_name: dict[str, int] | None = None,
    ) -> None:
        """Write a single predicate argument: tag + value."""
        match arg.tag:
            case "value":
                buf.write_u8(self._PRED_ARG_TAG_VALUE)
                name = arg.value if isinstance(arg.value, str) else str(arg.value)
                if value_numbers_by_name is not None:
                    buf.write_varint(value_numbers_by_name[name])
                else:
                    buf.write_varint(self._ctx.intern_string(name))
            case "const":
                buf.write_u8(self._PRED_ARG_TAG_CONST)
                assert isinstance(arg.value, int)
                buf.write_signed_varint(arg.value)
            case _:
                raise ValueError(f"unknown predicate arg tag: {arg.tag!r}")

    def _write_symbols(self, ir_offsets: dict[int, tuple[int, int]]) -> bytes:
        """Write the SYMBOLS section.

        Section layout: symbol_count, import_count, export_count,
        import offset table, export offset table, symbol entries.
        Import/export offset tables are uint64 byte offsets from the
        start of the symbol entries to each import/export entry.
        """
        buf = ByteBuffer()
        symbols = self._module.symbols

        # Classify symbols into imports and exports.
        import_indices: list[int] = []
        export_indices: list[int] = []
        for i, symbol in enumerate(symbols):
            is_import = (symbol.flags & SYMBOL_FLAG_IMPORT) != 0
            is_public = (symbol.flags & SYMBOL_FLAG_PUBLIC) != 0
            if is_import:
                import_indices.append(i)
            elif is_public:
                export_indices.append(i)

        buf.write_varint(len(symbols))
        buf.write_varint(len(import_indices))
        buf.write_varint(len(export_indices))

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
            if symbol.kind == SymbolKind.NONE:
                raise ValueError(
                    f"symbol {symbol.name!r} has no bytecode symbol payload kind"
                )
            buf.write_u8(symbol.kind.value)
            buf.write_u8(0 if (symbol.flags & SYMBOL_FLAG_PUBLIC) else 1)
            bytecode_flags = symbol.flags
            if symbol.source_symbol and symbol.source_symbol != symbol.name:
                bytecode_flags |= SYMBOL_FLAG_IMPORT_SYMBOL
            buf.write_u16_le(bytecode_flags)

            # Import metadata: source module and symbol for cross-module refs.
            if symbol.flags & SYMBOL_FLAG_IMPORT:
                buf.write_varint(self._ctx.strings[symbol.source_module])
                # source_symbol defaults to the symbol's own name.
                source_sym = symbol.source_symbol or symbol.name
                buf.write_varint(self._ctx.strings[source_sym])

            if symbol.kind in FUNCTION_SYMBOL_KINDS and symbol.op is not None:
                op = symbol.op
                module = self._module

                buf.write_varint(self._ctx.ops[op.name] + 1)
                self._write_comment_list(buf, op.comments)

                _cc_to_byte: dict[str, int] = {
                    "host": 1,
                    "device": 2,
                    "initializer": 3,
                    "deinitializer": 4,
                }
                cc_byte = _cc_to_byte.get(op.attributes.get("cc", ""), 0)
                buf.write_u8(cc_byte)
                _purity_to_byte: dict[str, int] = {"pure": 1}
                purity_byte = _purity_to_byte.get(op.attributes.get("purity", ""), 0)
                buf.write_u8(purity_byte)

                # Arg types: entry block args for defs, operands for decls.
                if op.regions and op.regions[0].blocks:
                    arg_ids = op.regions[0].blocks[0].arg_ids
                else:
                    arg_ids = op.operands

                # Result types and tied results.
                result_ids = op.results
                tied_results = op.tied_results
                signature_value_numbers = {
                    value_id: value_number
                    for value_number, value_id in enumerate([*arg_ids, *result_ids])
                }

                buf.write_varint(len(arg_ids))
                buf.write_varint(len(result_ids))

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
                predicates = op.attributes.get("predicates", [])
                self._write_predicate_list(
                    buf,
                    predicates,
                    self._value_numbers_by_name(signature_value_numbers),
                )

                if symbol.kind in (
                    SymbolKind.FUNC_TEMPLATE,
                    SymbolKind.FUNC_UKERNEL,
                ):
                    implements = op.attributes.get("implements")
                    if not isinstance(implements, str):
                        raise ValueError(
                            f"{op.name} symbol {symbol.name!r} must have "
                            "a string implements attr"
                        )
                    priority = op.attributes.get("priority", 0)
                    if not isinstance(priority, int) or priority < 0:
                        raise ValueError(
                            f"{op.name} symbol {symbol.name!r} priority "
                            "must be a non-negative integer"
                        )
                    buf.write_varint(self._ctx.strings[implements])
                    buf.write_varint(priority)

                shared_attr_keys = self._shared_func_metadata_attr_keys(op)
                payload_attrs = [
                    (key, value)
                    for key, value in op.attributes.items()
                    if key not in shared_attr_keys
                ]
                buf.write_varint(len(payload_attrs))
                value_numbers_by_name = self._value_numbers_by_name(
                    signature_value_numbers
                )
                for key, value in payload_attrs:
                    buf.write_varint(self._ctx.strings[key])
                    self._write_attr_value(
                        buf,
                        value,
                        value_numbers_by_name,
                        self._attr_def_for_op_attr(op.name, key),
                    )

                # Body reference.
                has_body = symbol_index in ir_offsets
                buf.write_u8(1 if has_body else 0)
                if has_body:
                    offset, length = ir_offsets[symbol_index]
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
                self._write_comment_list(buf, op.comments)

                result_ids = list(op.results)
                local_value_ids = self._collect_global_local_values(op)
                local_value_numbers = {
                    value_id: value_number
                    for value_number, value_id in enumerate(local_value_ids)
                }
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
                value_numbers_by_name = self._value_numbers_by_name(local_value_numbers)
                for key, value in payload_attrs:
                    buf.write_varint(self._ctx.strings[key])
                    self._write_attr_value(
                        buf,
                        value,
                        value_numbers_by_name,
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
                self._write_comment_list(buf, op.comments)

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
                has_body = bool(op.regions)
                buf.write_u8(1 if has_body else 0)
                if has_body:
                    try:
                        ir_offset, ir_length = ir_offsets[symbol_index]
                    except KeyError as exc:
                        raise ValueError(
                            f"record symbol {symbol.name!r} has a body region "
                            "but no serialized IR offset"
                        ) from exc
                    buf.write_u64_le(ir_offset)
                    buf.write_u32_le(ir_length)
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
