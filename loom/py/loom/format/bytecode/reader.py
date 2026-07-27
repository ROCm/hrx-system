# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bytecode reader: deserializes .loombc bytes to ir.py Module.

Mirror of the writer. Reads the file header, section directory,
and each section in order, constructing the in-memory IR.

All data is validated: offsets, lengths, counts, varints. A
malformed .loombc file produces a clear error, never a crash.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import Any, ClassVar, cast

from loom.fields import compute_layout
from loom.format.bytecode.encoding import decode_signed_varint, decode_varint
from loom.format.bytecode.op_decls import (
    attr_def_for_op,
    build_op_decl_map,
    func_like_interface_for_op,
    symbol_def_for_op,
)
from loom.format.bytecode.writer import (
    BYTECODE_IR_KIND_BY_TYPE_KIND,
    FORMAT_VERSION,
    LOCATION_MODE_FULL_LOCATIONS,
    LOCATION_MODE_NO_LOCATIONS,
    LOCATION_MODE_SOURCE_LOCATIONS,
    MAGIC,
    SECTION_ENCODINGS,
    SECTION_IR,
    SECTION_LOCATIONS,
    SECTION_OPS,
    SECTION_SOURCES,
    SECTION_STRINGS,
    SECTION_SYMBOLS,
    SECTION_TYPES,
)
from loom.ir import (
    BUFFER_TYPE,
    ENCODING_TYPE,
    NONE_TYPE,
    SYMBOL_FLAG_IMPORT,
    SYMBOL_FLAG_PUBLIC,
    SYMBOL_FLAG_RETAIN,
    Block,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingRole,
    EncodingType,
    FileLocation,
    FunctionType,
    FusedLocation,
    GroupScope,
    GroupType,
    Module,
    OpaqueLocation,
    Operation,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    RegisterType,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    StorageSpace,
    StorageType,
    Symbol,
    SymbolKind,
    SymbolName,
    TaggedLocation,
    TiedResult,
    Type,
    TypeKind,
    Value,
    rebuild_value_metadata,
)
from loom.stable_id import stable_id_from_string
from loom.target.descriptor_sets import DESCRIPTOR_SET_REGISTRATIONS

__all__ = [
    "BytecodeReader",
    "read_module",
]

# Maps the cc byte to a cc attribute string.
_FUNC_CC_BYTES: list[str | None] = [
    None,
    "host",
    "device",
    "initializer",
    "deinitializer",
]

# Maps the purity byte to a purity attribute string.
_FUNC_PURITY_BYTES: list[str | None] = [None, "pure"]

# Bytecode-only flag bit recording whether an import explicitly wrote its
# source symbol, even when the source name matches the local symbol name.
_SYMBOL_FLAG_IMPORT_SYMBOL = 0x0004
_SYMBOL_SUPPORTED_FLAGS = (
    SYMBOL_FLAG_PUBLIC
    | SYMBOL_FLAG_IMPORT
    | _SYMBOL_FLAG_IMPORT_SYMBOL
    | SYMBOL_FLAG_RETAIN
)


class BytecodeError(Exception):
    """Error reading bytecode."""


def _register_name_for_payload(
    descriptor_set_stable_id: int, register_class_id: int
) -> str | None:
    for registration in DESCRIPTOR_SET_REGISTRATIONS:
        if stable_id_from_string(registration.key) != descriptor_set_stable_id:
            continue
        descriptor_set = registration.load()
        if register_class_id >= len(descriptor_set.reg_classes):
            return None
        return descriptor_set.reg_classes[register_class_id].name
    return None


class BytecodeReader:
    """Reads .loombc bytes and constructs an ir.py Module."""

    def __init__(self, data: bytes, *, op_decls: Iterable[Any] | None = None) -> None:
        self._data = data
        self._offset = 0
        self._op_decls_by_name = build_op_decl_map(op_decls)
        self._strings: list[str] = []
        self._sources: list[str] = []
        self._types: list[Type] = []
        self._ops: list[str] = []
        self._encodings: list[EncodingInstance] = []
        self._encoding_families: list[str] = []
        self._location_mode = LOCATION_MODE_SOURCE_LOCATIONS
        self._module_value_count = 0
        self._module_region_count = 0
        self._module_block_count = 0
        self._module_op_count = 0

    def read(self) -> Module:
        """Read and return the module."""
        self._read_file_header()
        module_offset, module_length = self._read_module_directory()
        self._read_file_string_pool()

        # Read module sections.
        self._offset = module_offset
        sections = self._read_section_directory(module_length)

        # Read sections in dependency order.
        if SECTION_STRINGS in sections:
            self._read_strings_section(sections[SECTION_STRINGS])
        if SECTION_SOURCES in sections:
            self._read_sources_section(sections[SECTION_SOURCES])
        if SECTION_ENCODINGS in sections:
            self._read_encodings_section(sections[SECTION_ENCODINGS])
        if SECTION_TYPES in sections:
            self._read_types_section(sections[SECTION_TYPES])
        if SECTION_OPS in sections:
            self._read_ops_section(sections[SECTION_OPS])

        # Build module.
        module = Module()
        module.encodings = list(self._encodings)
        module.sources = list(self._sources)

        if self._location_mode == LOCATION_MODE_NO_LOCATIONS:
            if SECTION_LOCATIONS in sections:
                raise BytecodeError(
                    "NO_LOCATIONS bytecode must not contain a LOCATIONS section"
                )
        elif SECTION_LOCATIONS not in sections:
            raise BytecodeError("bytecode with source locations must have LOCATIONS")

        # Read locations.
        if SECTION_LOCATIONS in sections:
            self._read_locations_section(sections[SECTION_LOCATIONS], module)

        # Read IR and symbols.
        ir_data = sections.get(SECTION_IR, (0, b""))
        symbols_data = sections.get(SECTION_SYMBOLS, (0, b""))
        if isinstance(symbols_data, tuple):
            self._read_symbols_section(symbols_data, ir_data, module)

        allocation_counts = self._module_allocation_counts(module)
        if allocation_counts != (
            self._module_value_count,
            self._module_region_count,
            self._module_block_count,
            self._module_op_count,
        ):
            raise BytecodeError("module allocation summary does not match IR")

        rebuild_value_metadata(module)
        return module

    def _module_allocation_counts(self, module: Module) -> tuple[int, int, int, int]:
        """Return module value, serialized region, block, and op counts."""
        value_count = len(module.values)
        region_count = 0
        block_count = 0
        op_count = 0
        for symbol in module.symbols:
            if symbol.op is None or not symbol.op.regions:
                continue
            counts = self._count_region_forest(symbol.op.regions)
            region_count += counts[1]
            block_count += counts[2]
            op_count += counts[3]
        return value_count, region_count, block_count, op_count

    # --- Low-level reads ---

    def _read_u8(self) -> int:
        if self._offset >= len(self._data):
            raise BytecodeError("unexpected end of data")
        value = self._data[self._offset]
        self._offset += 1
        return value

    def _read_u16_le(self) -> int:
        if self._offset + 2 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<H", self._data, self._offset)[0]
        self._offset += 2
        return value

    def _read_u32_le(self) -> int:
        if self._offset + 4 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<I", self._data, self._offset)[0]
        self._offset += 4
        return value

    def _read_u64_le(self) -> int:
        if self._offset + 8 > len(self._data):
            raise BytecodeError("unexpected end of data")
        value: int = struct.unpack_from("<Q", self._data, self._offset)[0]
        self._offset += 8
        return value

    def _read_varint(self) -> int:
        value, self._offset = decode_varint(self._data, self._offset)
        return value

    def _read_signed_varint(self) -> int:
        value, self._offset = decode_signed_varint(self._data, self._offset)
        return value

    def _read_bytes(self, length: int) -> bytes:
        if self._offset + length > len(self._data):
            raise BytecodeError("unexpected end of data")
        data = self._data[self._offset : self._offset + length]
        self._offset += length
        return data

    def _read_string(self) -> str:
        length = self._read_varint()
        data = self._read_bytes(length)
        return data.decode("utf-8")

    def _skip_to_alignment(self, alignment: int) -> None:
        remainder = self._offset % alignment
        if remainder != 0:
            self._offset += alignment - remainder

    # --- File structure ---

    def _read_file_header(self) -> None:
        magic = self._read_bytes(4)
        if magic != MAGIC:
            raise BytecodeError(f"invalid magic: expected {MAGIC!r}, got {magic!r}")
        version = self._read_u8()
        if version != FORMAT_VERSION:
            raise BytecodeError(f"unsupported format version: {version}")
        self._location_mode = self._read_u8()
        if self._location_mode not in (
            LOCATION_MODE_SOURCE_LOCATIONS,
            LOCATION_MODE_NO_LOCATIONS,
            LOCATION_MODE_FULL_LOCATIONS,
        ):
            raise BytecodeError(f"unsupported location mode: {self._location_mode}")
        if self._location_mode == LOCATION_MODE_FULL_LOCATIONS:
            raise BytecodeError("FULL_LOCATIONS bytecode is not implemented")
        self._module_count = self._read_u16_le()
        self._string_pool_length = self._read_u32_le()
        self._reserved = self._read_u32_le()
        # Producer string (null-terminated).
        producer_start = self._offset
        while self._offset < len(self._data) and self._data[self._offset] != 0:
            self._offset += 1
        self._producer = self._data[producer_start : self._offset].decode("utf-8")
        self._offset += 1  # skip null
        self._skip_to_alignment(8)

    def _read_module_directory(self) -> tuple[int, int]:
        """Read module directory, return (offset, length) of first module."""
        self._read_u32_le()
        self._read_u16_le()
        self._read_u16_le()
        module_offset = self._read_u64_le()
        module_length = self._read_u64_le()
        return module_offset, module_length

    def _read_file_string_pool(self) -> None:
        """Read the file string pool (module names)."""
        if self._string_pool_length > 0:
            pool_data = self._read_bytes(self._string_pool_length)
            self._module_name = pool_data.decode("utf-8")
        else:
            self._module_name = ""
        self._skip_to_alignment(8)

    def _read_section_directory(
        self, module_length: int
    ) -> dict[int, tuple[int, bytes]]:
        """Read section directory, return {kind: (offset, data)}."""
        sections: dict[int, tuple[int, bytes]] = {}
        module_start = self._offset
        module_end = module_start + module_length
        if module_end > len(self._data):
            raise BytecodeError("module extends past end of file")

        section_count = self._read_varint()
        self._module_value_count = self._read_varint()
        self._module_region_count = self._read_varint()
        self._module_block_count = self._read_varint()
        self._module_op_count = self._read_varint()

        entries: list[tuple[int, int, int]] = []
        seen_kinds: set[int] = set()
        for _ in range(section_count):
            if self._offset + 32 > module_end:
                raise BytecodeError("section directory extends past module")
            kind = struct.unpack_from("<H", self._data, self._offset)[0]
            flags = struct.unpack_from("<H", self._data, self._offset + 2)[0]
            reserved = struct.unpack_from("<I", self._data, self._offset + 4)[0]
            offset = struct.unpack_from("<Q", self._data, self._offset + 8)[0]
            length = struct.unpack_from("<Q", self._data, self._offset + 16)[0]
            uncompressed_length = struct.unpack_from(
                "<Q", self._data, self._offset + 24
            )[0]
            if kind in seen_kinds:
                raise BytecodeError(f"duplicate section kind: {kind}")
            if reserved != 0:
                raise BytecodeError(f"section {kind} reserved field must be zero")
            if flags != 0:
                raise BytecodeError(f"section {kind} has unsupported flags {flags}")
            if uncompressed_length != 0:
                raise BytecodeError(f"section {kind} uncompressed length must be zero")
            seen_kinds.add(kind)
            entries.append((kind, offset, length))
            self._offset += 32

        minimum_section_offset = self._offset - module_start
        previous_end = minimum_section_offset
        for kind, offset, length in entries:
            if offset < previous_end:
                raise BytecodeError("section directory is not sorted by offset")
            section_end = offset + length
            if section_end > module_length:
                raise BytecodeError(f"section {kind} extends past module")
            abs_offset = module_start + offset
            section_data = self._data[abs_offset : abs_offset + length]
            sections[kind] = (abs_offset, section_data)
            previous_end = section_end

        return sections

    # --- Section readers ---

    def _read_strings_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._strings = []
        for _ in range(count):
            length, offset = decode_varint(data, offset)
            text = data[offset : offset + length].decode("utf-8")
            offset += length
            self._strings.append(text)

    def _read_sources_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._sources = []
        for _ in range(count):
            length, offset = decode_varint(data, offset)
            text = data[offset : offset + length].decode("utf-8")
            offset += length
            self._sources.append(text)

    def _read_encodings_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0

        # Encoding family registry.
        family_count, offset = decode_varint(data, offset)
        self._encoding_families = []
        for _ in range(family_count):
            name_id, offset = decode_varint(data, offset)
            self._encoding_families.append(self._strings[name_id])

        # Encoding instances with structured attribute parameters.
        instance_count, offset = decode_varint(data, offset)
        self._encodings = []
        for _ in range(instance_count):
            family_index, offset = decode_varint(data, offset)
            if family_index >= len(self._encoding_families):
                raise BytecodeError(
                    f"encoding family index {family_index} out of range "
                    f"(family table has {len(self._encoding_families)} entries)"
                )
            alias_string_id_plus1, offset = decode_varint(data, offset)
            if alias_string_id_plus1 > len(self._strings):
                alias_string_id = alias_string_id_plus1 - 1
                raise BytecodeError(
                    f"encoding alias string_id {alias_string_id} out of range "
                    f"(string table has {len(self._strings)} entries)"
                )
            param_count, offset = decode_varint(data, offset)

            name = self._encoding_families[family_index]
            alias = (
                self._strings[alias_string_id_plus1 - 1]
                if alias_string_id_plus1 > 0
                else ""
            )
            param_list: list[tuple[str, Any]] = []
            for _ in range(param_count):
                param_name_id, offset = decode_varint(data, offset)
                param_name = self._strings[param_name_id]
                param_value, offset = self._read_attr_value(data, offset)
                param_list.append((param_name, param_value))

            self._encodings.append(
                EncodingInstance(name=name, alias=alias, params=tuple(param_list))
            )

    def _read_types_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._types = []
        for _ in range(count):
            kind = data[offset]
            offset += 1
            try:
                type_kind = BYTECODE_IR_KIND_BY_TYPE_KIND[kind]
            except KeyError as err:
                raise BytecodeError(f"unknown type kind: {kind}") from err
            ir_type: Type
            match type_kind:
                case TypeKind.NONE:
                    ir_type = NONE_TYPE
                case TypeKind.SCALAR:
                    scalar_kind = ScalarTypeKind(data[offset])
                    offset += 1
                    ir_type = ScalarType(scalar_kind)
                case TypeKind.TILE | TypeKind.TENSOR | TypeKind.VECTOR | TypeKind.VIEW:
                    elem_kind = ScalarTypeKind(data[offset])
                    offset += 1
                    rank = data[offset]
                    offset += 1
                    encoding_attachment = data[offset]
                    offset += 1
                    enc_instance, offset = decode_varint(data, offset)

                    dims: list[StaticDim | DynamicDim] = []
                    for _ in range(rank):
                        is_dynamic = data[offset]
                        offset += 1
                        if is_dynamic:
                            dims.append(DynamicDim())
                        else:
                            size, offset = decode_varint(data, offset)
                            dims.append(StaticDim(size))

                    encoding: EncodingInstance | DynamicEncoding | None = None
                    match encoding_attachment:
                        case 0:
                            if enc_instance != 0:
                                raise BytecodeError(
                                    "none encoding attachment must have id 0"
                                )
                        case 1:
                            if enc_instance == 0 or enc_instance > len(self._encodings):
                                raise BytecodeError(
                                    f"static encoding id out of range: {enc_instance}"
                                )
                            encoding = self._encodings[enc_instance - 1]
                        case 2:
                            if enc_instance != 0:
                                raise BytecodeError(
                                    "dynamic encoding attachment must have id 0"
                                )
                            encoding = DynamicEncoding()
                        case _:
                            raise BytecodeError(
                                f"unknown encoding attachment: {encoding_attachment}"
                            )
                    if type_kind == TypeKind.VECTOR and encoding is not None:
                        raise BytecodeError(
                            "vector types must not carry encoding/layout suffixes"
                        )

                    try:
                        ir_type = ShapedType(
                            type_kind=type_kind,
                            element_type=ScalarType(elem_kind),
                            dims=tuple(dims),
                            encoding=encoding,
                        )
                    except ValueError as err:
                        raise BytecodeError(str(err)) from err
                case TypeKind.GROUP:
                    scope_byte = data[offset]
                    offset += 1
                    try:
                        scope = GroupScope(scope_byte)
                    except ValueError as err:
                        raise BytecodeError(
                            f"unknown group scope byte: {scope_byte}"
                        ) from err
                    ir_type = GroupType(scope)
                case TypeKind.FUNCTION:
                    arg_count, offset = decode_varint(data, offset)
                    result_count, offset = decode_varint(data, offset)
                    arg_types = []
                    for _ in range(arg_count):
                        type_idx, offset = decode_varint(data, offset)
                        arg_types.append(self._types[type_idx])
                    result_types = []
                    for _ in range(result_count):
                        type_idx, offset = decode_varint(data, offset)
                        result_types.append(self._types[type_idx])
                    ir_type = FunctionType(tuple(arg_types), tuple(result_types))
                case TypeKind.DIALECT:
                    name_id, offset = decode_varint(data, offset)
                    param_count, offset = decode_varint(data, offset)
                    params = []
                    for _ in range(param_count):
                        type_idx, offset = decode_varint(data, offset)
                        params.append(self._types[type_idx])
                    ir_type = DialectType(self._strings[name_id], tuple(params))
                case TypeKind.REGISTER:
                    descriptor_set_stable_id, offset = decode_varint(data, offset)
                    payload1, offset = decode_varint(data, offset)
                    register_class_id = payload1 & 0xFFFF
                    unit_count = (payload1 >> 16) & 0xFFFFFFFF
                    if descriptor_set_stable_id == 0:
                        raise BytecodeError(
                            "register descriptor set stable ID must be non-zero"
                        )
                    if unit_count == 0:
                        raise BytecodeError("register unit count must be non-zero")
                    if payload1 >> 48:
                        raise BytecodeError(
                            "register payload reserved bits must be zero"
                        )
                    name = _register_name_for_payload(
                        descriptor_set_stable_id, register_class_id
                    )
                    try:
                        ir_type = RegisterType(
                            descriptor_set_stable_id,
                            register_class_id,
                            unit_count,
                            name,
                        )
                    except ValueError as err:
                        raise BytecodeError(str(err)) from err
                case TypeKind.STORAGE:
                    space_byte = data[offset]
                    offset += 1
                    try:
                        ir_type = StorageType(StorageSpace(space_byte))
                    except ValueError as err:
                        raise BytecodeError(
                            f"unknown storage space byte: {space_byte}"
                        ) from err
                case TypeKind.ENCODING:
                    role_byte = data[offset]
                    offset += 1
                    try:
                        role = EncodingRole(role_byte)
                    except ValueError as e:
                        raise BytecodeError(
                            f"unsupported encoding role byte: {role_byte}"
                        ) from e
                    ir_type = (
                        ENCODING_TYPE
                        if role == EncodingRole.UNKNOWN
                        else EncodingType(role)
                    )
                case TypeKind.POOL:
                    is_dynamic = data[offset]
                    offset += 1
                    if is_dynamic:
                        ir_type = PoolType(block_size=DynamicDim())
                    else:
                        size, offset = decode_varint(data, offset)
                        ir_type = PoolType(block_size=StaticDim(size))
                case TypeKind.BUFFER:
                    ir_type = BUFFER_TYPE
                case _:
                    raise BytecodeError(f"unsupported type kind: {type_kind.name}")
            self._types.append(ir_type)

    def _read_ops_section(self, section: tuple[int, bytes]) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        self._ops = []
        for _ in range(count):
            name_id, offset = decode_varint(data, offset)
            self._ops.append(self._strings[name_id])

    def _read_locations_section(
        self, section: tuple[int, bytes], module: Module
    ) -> None:
        _, data = section
        offset = 0
        count, offset = decode_varint(data, offset)
        for _ in range(count):
            kind = data[offset]
            offset += 1
            flags = data[offset]
            offset += 1
            match kind:
                case 0:  # NONE
                    module.locations.add(None)
                case 1:  # FILE
                    source_id, offset = decode_varint(data, offset)
                    start_line, offset = decode_varint(data, offset)
                    start_col, offset = decode_varint(data, offset)
                    end_line, offset = decode_varint(data, offset)
                    end_col, offset = decode_varint(data, offset)
                    module.locations.add(
                        FileLocation(
                            source_id, start_line, start_col, end_line, end_col, flags
                        )
                    )
                case 2:  # FUSED
                    child_count, offset = decode_varint(data, offset)
                    children = []
                    for _ in range(child_count):
                        child, offset = decode_varint(data, offset)
                        children.append(child)
                    module.locations.add(FusedLocation(tuple(children), flags))
                case 3:  # OPAQUE
                    source_id, offset = decode_varint(data, offset)
                    data_length, offset = decode_varint(data, offset)
                    opaque_data = data[offset : offset + data_length]
                    offset += data_length
                    module.locations.add(OpaqueLocation(source_id, opaque_data, flags))
                case 4:  # TAGGED
                    tag, offset = decode_varint(data, offset)
                    child, offset = decode_varint(data, offset)
                    data_length, offset = decode_varint(data, offset)
                    tagged_data = data[offset : offset + data_length]
                    offset += data_length
                    module.locations.add(
                        TaggedLocation(
                            tag=tag, child=child, data=tagged_data, flags=flags
                        )
                    )

    def _read_comment_list(
        self, data: bytes, offset: int
    ) -> tuple[tuple[str, ...], int]:
        """Read a bytecode comment list encoded as raw UTF-8 strings."""
        count, offset = decode_varint(data, offset)
        comments: list[str] = []
        for _ in range(count):
            length, offset = decode_varint(data, offset)
            comment_bytes = data[offset : offset + length]
            if len(comment_bytes) != length:
                raise BytecodeError("comment extends past end of data")
            offset += length
            comments.append(comment_bytes.decode("utf-8"))
        return tuple(comments), offset

    def _map_value_ref(self, value_ref: int, value_map: list[int]) -> int:
        """Translate an available function-local value number to a module value id."""
        if 0 <= value_ref < len(value_map):
            return value_map[value_ref]
        raise BytecodeError(
            "value reference must target an available value: "
            f"{value_ref} with {len(value_map)} value(s) available"
        )

    def _reserve_value_defs(
        self,
        module: Module,
        value_map: list[int],
        count: int,
        predefined_values: list[int] | None = None,
    ) -> list[int]:
        """Reserve a definition group before decoding its value definitions.

        Value definitions carry shape and encoding bindings that may reference
        co-results in the same group. Reserving the group first gives those
        bindings stable module value IDs while the definitions are decoded.
        """
        reserved: list[int] = []
        predefined_values = predefined_values or []
        for _ in range(count):
            local_number = len(value_map)
            if local_number < len(predefined_values):
                value_id = predefined_values[local_number]
            else:
                value_id = module.add_value(Value(name="", type=NONE_TYPE))
            value_map.append(value_id)
            reserved.append(value_id)
        return reserved

    def _read_value_def(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
        target_value_id: int | None = None,
    ) -> tuple[int, int]:
        """Read one SSA value definition and return its module value id."""
        name_id, offset = decode_varint(data, offset)
        if name_id != 0 and name_id >= len(self._strings):
            raise BytecodeError(
                f"value name string_id {name_id} out of range "
                f"(string table has {len(self._strings)} entries)"
            )
        type_idx, offset = decode_varint(data, offset)
        if type_idx >= len(self._types):
            raise BytecodeError(
                f"value type index {type_idx} out of range "
                f"(type table has {len(self._types)} entries)"
            )

        dim_count, offset = decode_varint(data, offset)
        dim_bindings: dict[int, int] = {}
        for dimension_index in range(dim_count):
            value_ref, offset = decode_signed_varint(data, offset)
            dim_bindings[dimension_index] = self._map_value_ref(value_ref, value_map)

        enc_binding_raw, offset = decode_varint(data, offset)
        encoding_binding = enc_binding_raw - 1 if enc_binding_raw > 0 else -1
        if encoding_binding >= 0:
            encoding_binding = self._map_value_ref(encoding_binding, value_map)

        value = Value(
            name="" if name_id == 0 else self._strings[name_id],
            type=self._types[type_idx],
            dim_bindings=dim_bindings,
            encoding_binding=encoding_binding,
        )
        if target_value_id is not None:
            existing = module.values[target_value_id]
            is_placeholder = (
                existing.name == ""
                and existing.type == NONE_TYPE
                and not existing.dim_bindings
                and existing.encoding_binding == -1
            )
            if is_placeholder:
                existing.name = value.name
                existing.type = value.type
                existing.dim_bindings = value.dim_bindings
                existing.encoding_binding = value.encoding_binding
            elif (
                existing.name != value.name
                or existing.type != value.type
                or existing.dim_bindings != value.dim_bindings
                or existing.encoding_binding != value.encoding_binding
            ):
                raise BytecodeError(
                    "predefined function signature value does not match body value"
                )
            return target_value_id, offset
        return module.add_value(value), offset

    def _read_symbols_section(
        self,
        symbols_section: tuple[int, bytes],
        ir_section: tuple[int, bytes],
        module: Module,
    ) -> None:
        _, sym_data = symbols_section
        _, ir_data = ir_section
        offset = 0
        count, offset = decode_varint(sym_data, offset)

        # Import/export offset tables.
        import_count, offset = decode_varint(sym_data, offset)
        export_count, offset = decode_varint(sym_data, offset)
        # Skip the offset tables (uint64 each).
        offset += import_count * 8
        offset += export_count * 8

        for _ in range(count):
            name_id, offset = decode_varint(sym_data, offset)
            kind = sym_data[offset]
            offset += 1
            visibility = sym_data[offset]
            offset += 1
            flags = struct.unpack_from("<H", sym_data, offset)[0]
            offset += 2

            name = self._strings[name_id]
            if visibility not in (0, 1):
                raise BytecodeError(f"unsupported symbol visibility byte: {visibility}")
            if flags & ~_SYMBOL_SUPPORTED_FLAGS:
                raise BytecodeError("symbol has unsupported flag bits")
            if (visibility == 0) != ((flags & SYMBOL_FLAG_PUBLIC) != 0):
                raise BytecodeError("symbol visibility byte does not match flags")

            # Import metadata: source module and symbol for cross-module refs.
            is_import = (flags & SYMBOL_FLAG_IMPORT) != 0
            source_module = ""
            source_symbol = ""
            if is_import:
                source_module_id, offset = decode_varint(sym_data, offset)
                source_symbol_id, offset = decode_varint(sym_data, offset)
                source_module = self._strings[source_module_id]
                source_symbol = self._strings[source_symbol_id]

            if kind <= 3:  # FUNC_DEF, FUNC_DECL, FUNC_TEMPLATE, FUNC_UKERNEL
                op_table_index_plus1, offset = decode_varint(sym_data, offset)
                if op_table_index_plus1 == 0:
                    raise BytecodeError(
                        "function symbol op_table_index_plus1 must be nonzero"
                    )
                op_table_index = op_table_index_plus1 - 1
                if op_table_index >= len(self._ops):
                    raise BytecodeError(
                        "function symbol op_table_index_plus1 references OPS "
                        f"entry {op_table_index} but only {len(self._ops)} exist"
                    )
                op_name = self._ops[op_table_index]
                op_comments, offset = self._read_comment_list(sym_data, offset)

                cc_byte = sym_data[offset]
                offset += 1
                purity_byte = sym_data[offset]
                offset += 1
                arg_count, offset = decode_varint(sym_data, offset)
                result_count, offset = decode_varint(sym_data, offset)

                signature_value_map: list[int] = []
                signature_value_ids = self._reserve_value_defs(
                    module, signature_value_map, arg_count + result_count
                )
                arg_ids: list[int] = []
                for i in range(arg_count):
                    value_id, offset = self._read_value_def(
                        sym_data,
                        offset,
                        module,
                        signature_value_map,
                        signature_value_ids[i],
                    )
                    arg_ids.append(value_id)

                result_ids: list[int] = []
                tied_results: list[TiedResult] = []
                for i in range(result_count):
                    is_tied = sym_data[offset]
                    offset += 1
                    value_id, offset = self._read_value_def(
                        sym_data,
                        offset,
                        module,
                        signature_value_map,
                        signature_value_ids[arg_count + i],
                    )
                    result_ids.append(value_id)
                    if is_tied:
                        tied_op_idx, offset = decode_varint(sym_data, offset)
                        tied_results.append(
                            TiedResult(result_index=i, operand_index=tied_op_idx)
                        )

                _tied_count, offset = decode_varint(sym_data, offset)
                predicates, offset = self._read_predicate_list(
                    sym_data, offset, module, signature_value_map
                )

                implements: str | None = None
                priority = 0
                if kind in (
                    SymbolKind.FUNC_TEMPLATE.value,
                    SymbolKind.FUNC_UKERNEL.value,
                ):
                    implements_id, offset = decode_varint(sym_data, offset)
                    if implements_id >= len(self._strings):
                        raise BytecodeError(
                            "function implements string_id "
                            f"{implements_id} out of range "
                            f"(string table has {len(self._strings)} entries)"
                        )
                    priority, offset = decode_varint(sym_data, offset)
                    implements = self._strings[implements_id]

                payload_attr_count, offset = decode_varint(sym_data, offset)
                payload_attrs, offset = self._read_op_attr_entries(
                    sym_data,
                    offset,
                    payload_attr_count,
                    module,
                    signature_value_map,
                    op_name,
                )

                has_body = sym_data[offset]
                offset += 1
                regions: list[Region] = []
                if has_body:
                    ir_offset = struct.unpack_from("<Q", sym_data, offset)[0]
                    offset += 8
                    ir_length = struct.unpack_from("<I", sym_data, offset)[0]
                    offset += 4
                    body_region_index = self._func_body_region_index(op_name)
                    regions = self._read_symbol_regions(
                        ir_data[ir_offset : ir_offset + ir_length],
                        module,
                        root_region_count=self._op_region_count(op_name),
                        predefined_values=arg_ids,
                        predefined_region_index=body_region_index,
                    )

                # Build the attributes dict for this func-like op.
                symbol_field = self._symbol_field_for_op(op_name)
                op_attrs: dict[str, Any] = {symbol_field: name}
                if flags & SYMBOL_FLAG_PUBLIC:
                    op_attrs["visibility"] = "public"
                if flags & SYMBOL_FLAG_RETAIN:
                    op_attrs["retain"] = "retain"
                cc_str = (
                    _FUNC_CC_BYTES[cc_byte] if cc_byte < len(_FUNC_CC_BYTES) else None
                )
                if cc_str is not None:
                    op_attrs["cc"] = cc_str
                purity_str = (
                    _FUNC_PURITY_BYTES[purity_byte]
                    if purity_byte < len(_FUNC_PURITY_BYTES)
                    else None
                )
                if purity_str is not None:
                    op_attrs["purity"] = purity_str
                if predicates:
                    op_attrs["predicates"] = predicates
                if source_module:
                    op_attrs["import_module"] = source_module
                    if flags & _SYMBOL_FLAG_IMPORT_SYMBOL:
                        op_attrs["import_symbol"] = source_symbol
                if implements is not None:
                    op_attrs["implements"] = implements
                    if priority != 0:
                        op_attrs["priority"] = priority
                shared_attr_keys = self._shared_func_metadata_attr_keys(op_name)
                for key, value in payload_attrs.items():
                    if key in shared_attr_keys:
                        raise BytecodeError(
                            f"function payload attr duplicates shared metadata: {key!r}"
                        )
                    op_attrs[key] = value

                # For declaration-style ops (no body), args are operands.
                # For definition-style ops (with body), args are entry block args.
                operand_ids: list[int] = []
                if not regions:
                    operand_ids = arg_ids

                op = Operation(
                    name=op_name,
                    operands=operand_ids,
                    results=result_ids,
                    tied_results=tied_results,
                    attributes=op_attrs,
                    regions=regions,
                    comments=op_comments,
                )
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind(kind),
                    flags=flags,
                    op=op,
                    source_module=source_module,
                    source_symbol=source_symbol,
                )
                module.add_symbol(symbol)
            elif kind == SymbolKind.GLOBAL.value:
                op_table_index_plus1, offset = decode_varint(sym_data, offset)
                if op_table_index_plus1 == 0:
                    raise BytecodeError(
                        "global symbol op_table_index_plus1 must be nonzero"
                    )
                op_table_index = op_table_index_plus1 - 1
                if op_table_index >= len(self._ops):
                    raise BytecodeError(
                        "global symbol op_table_index_plus1 references OPS "
                        f"entry {op_table_index} but only {len(self._ops)} exist"
                    )
                op_name = self._ops[op_table_index]
                op_comments, offset = self._read_comment_list(sym_data, offset)

                result_count, offset = decode_varint(sym_data, offset)
                if result_count == 0:
                    raise BytecodeError("global symbol result_count must be nonzero")
                local_value_count, offset = decode_varint(sym_data, offset)
                if local_value_count < result_count:
                    raise BytecodeError(
                        "global symbol local_value_count must cover all results"
                    )
                local_value_map: list[int] = []
                local_value_ids = self._reserve_value_defs(
                    module, local_value_map, local_value_count
                )
                for i in range(local_value_count):
                    value_id, offset = self._read_value_def(
                        sym_data,
                        offset,
                        module,
                        local_value_map,
                        local_value_ids[i],
                    )
                    local_value_ids[i] = value_id
                result_ids = local_value_ids[:result_count]

                attr_count, offset = decode_varint(sym_data, offset)
                op_attrs, offset = self._read_op_attr_entries(
                    sym_data, offset, attr_count, module, local_value_map, op_name
                )
                symbol_field = self._symbol_field_for_op(op_name)
                op_attrs = {symbol_field: name, **op_attrs}

                op = Operation(
                    name=op_name,
                    operands=[],
                    results=result_ids,
                    attributes=op_attrs,
                    comments=op_comments,
                )
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind.GLOBAL,
                    flags=flags,
                    op=op,
                    source_module=source_module,
                    source_symbol=source_symbol,
                )
                module.add_symbol(symbol)
            elif kind == SymbolKind.RECORD.value:
                offset = self._read_record_symbol_payload(
                    sym_data,
                    offset,
                    ir_data,
                    module,
                    name,
                    flags,
                    source_module,
                    source_symbol,
                )
            else:
                raise BytecodeError(f"unsupported symbol kind: {kind}")

    def _read_record_symbol_payload(
        self,
        sym_data: bytes,
        offset: int,
        ir_data: bytes,
        module: Module,
        name: str,
        flags: int,
        source_module: str,
        source_symbol: str,
    ) -> int:
        """Read one RECORD symbol payload and append it to ``module``."""
        op_table_index_plus1, offset = decode_varint(sym_data, offset)
        if op_table_index_plus1 == 0:
            raise BytecodeError("record symbol op_table_index_plus1 must be nonzero")
        op_table_index = op_table_index_plus1 - 1
        if op_table_index >= len(self._ops):
            raise BytecodeError(
                "record symbol op_table_index_plus1 references OPS "
                f"entry {op_table_index} but only {len(self._ops)} exist"
            )
        op_name = self._ops[op_table_index]
        op_comments, offset = self._read_comment_list(sym_data, offset)

        attr_count, offset = decode_varint(sym_data, offset)
        op_attrs, offset = self._read_op_attr_entries(
            sym_data, offset, attr_count, module, None, op_name
        )
        symbol_field = self._symbol_field_for_op(op_name)
        op_attrs = {symbol_field: name, **op_attrs}

        has_body = sym_data[offset]
        offset += 1
        if has_body > 1:
            raise BytecodeError(f"invalid record has_body value: {has_body}")
        record_regions: list[Region] = []
        if has_body:
            if offset + 12 > len(sym_data):
                raise BytecodeError("record body reference is truncated")
            ir_offset = struct.unpack_from("<Q", sym_data, offset)[0]
            offset += 8
            ir_length = struct.unpack_from("<I", sym_data, offset)[0]
            offset += 4
            if ir_offset + ir_length > len(ir_data):
                raise BytecodeError("record body range extends past IR section")
            record_regions = self._read_symbol_regions(
                ir_data[ir_offset : ir_offset + ir_length],
                module,
                root_region_count=self._op_region_count(op_name),
            )

        op = Operation(
            name=op_name,
            operands=[],
            results=[],
            attributes=op_attrs,
            regions=record_regions,
            comments=op_comments,
        )
        symbol = Symbol(
            name=name,
            kind=SymbolKind.RECORD,
            flags=flags,
            op=op,
            source_module=source_module,
            source_symbol=source_symbol,
        )
        module.add_symbol(symbol)
        return offset

    def _shared_func_metadata_attr_keys(self, op_name: str) -> frozenset[str]:
        """Return func-like attrs encoded by fixed symbol metadata fields."""
        symbol_def = symbol_def_for_op(self._op_decls_by_name, op_name)
        keys = {
            cast(str, symbol_def.field),
            "import_module",
            "import_symbol",
        }
        func_like = func_like_interface_for_op(self._op_decls_by_name, op_name)
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

    def _func_body_region_index(self, op_name: str) -> int | None:
        """Return op_name's FuncLike body region index, if it has one."""
        func_like = func_like_interface_for_op(self._op_decls_by_name, op_name)
        if func_like is None or func_like.args_as_operands or func_like.body is None:
            return None
        op_decl = self._op_decls_by_name.get(op_name)
        if op_decl is None:
            return None
        for region_index, region_def in enumerate(getattr(op_decl, "regions", ())):
            if region_def.name == func_like.body:
                return region_index
        return None

    def _op_region_count(self, op_name: str) -> int:
        """Return the declared root region slot count for op_name."""
        op_decl = self._op_decls_by_name.get(op_name)
        if op_decl is None:
            return 0
        return len(getattr(op_decl, "regions", ()))

    def _read_symbol_regions(
        self,
        data: bytes,
        module: Module,
        root_region_count: int,
        predefined_values: list[int] | None = None,
        predefined_region_index: int | None = None,
    ) -> list[Region]:
        """Read a symbol root-region payload from IR section data.

        The bytecode uses function-local sequential value numbers (0, 1, 2, ...)
        for all SSA references within the symbol regions. The value_map
        translates these numbers to module-level value IDs as block args and op
        results are created during reading. When predefined_values are supplied,
        they are bound only while reading predefined_region_index.
        """
        offset = 0
        value_map: list[int] = []
        _value_count, offset = decode_varint(data, offset)
        _region_count, offset = decode_varint(data, offset)
        _block_count, offset = decode_varint(data, offset)
        _op_count, offset = decode_varint(data, offset)
        encoded_root_region_count, offset = decode_varint(data, offset)
        regions_by_index: dict[int, Region] = {}
        if encoded_root_region_count == 0:
            raise BytecodeError("symbol root region count must be nonzero")
        if encoded_root_region_count > root_region_count:
            raise BytecodeError("symbol root region count exceeds op region slots")
        for root_ordinal in range(encoded_root_region_count):
            region_index, offset = decode_varint(data, offset)
            if region_index >= root_region_count:
                raise BytecodeError("symbol region index is out of range")
            if region_index in regions_by_index:
                raise BytecodeError("symbol region index appears more than once")
            if (
                predefined_region_index is not None
                and root_ordinal == 0
                and region_index != predefined_region_index
            ):
                raise BytecodeError("FuncLike body region must be first")
            region_predefined_values = (
                predefined_values or []
                if predefined_region_index is not None
                and region_index == predefined_region_index
                else []
            )
            region, offset = self._read_region(
                data,
                offset,
                module,
                value_map,
                predefined_values=region_predefined_values,
            )
            regions_by_index[region_index] = region
        if offset != len(data):
            raise BytecodeError("symbol region payload has trailing bytes")
        if (
            predefined_region_index is not None
            and predefined_region_index not in regions_by_index
        ):
            raise BytecodeError("symbol region payload is missing the FuncLike body")

        regions: list[Region] = []
        if regions_by_index:
            for region_index in range(max(regions_by_index.keys()) + 1):
                if region_index not in regions_by_index:
                    raise BytecodeError("symbol region payload has index gaps")
                regions.append(regions_by_index[region_index])
        parsed_counts = self._count_region_forest(regions)
        if parsed_counts != (_value_count, _region_count, _block_count, _op_count):
            raise BytecodeError("symbol region allocation summary does not match IR")
        return regions

    def _count_region_tree(self, region: Region) -> tuple[int, int, int, int]:
        """Return value, region, block, and op counts for a parsed region tree."""
        value_count = 0
        region_count = 1
        block_count = len(region.blocks)
        op_count = 0
        for block in region.blocks:
            value_count += len(block.arg_ids)
            for op in block.ops:
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
        self, regions: list[Region] | tuple[Region, ...]
    ) -> tuple[int, int, int, int]:
        """Return value, region, block, and op counts for root regions."""
        value_count = 0
        region_count = 0
        block_count = 0
        op_count = 0
        for region in regions:
            counts = self._count_region_tree(region)
            value_count += counts[0]
            region_count += counts[1]
            block_count += counts[2]
            op_count += counts[3]
        return value_count, region_count, block_count, op_count

    def _read_region(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
        predefined_values: list[int],
    ) -> tuple[Region, int]:
        block_count, offset = decode_varint(data, offset)
        blocks = [Block() for _ in range(block_count)]
        for block in blocks:
            offset = self._read_block(
                data, offset, module, value_map, predefined_values, block, blocks
            )
        return Region(blocks=blocks), offset

    def _read_block(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
        predefined_values: list[int],
        block: Block,
        region_blocks: list[Block],
    ) -> int:
        has_label = data[offset]
        offset += 1
        label = ""
        if has_label:
            label_id, offset = decode_varint(data, offset)
            label = self._strings[label_id]
        comments, offset = self._read_comment_list(data, offset)

        # Block args.
        arg_count, offset = decode_varint(data, offset)
        arg_ids = self._reserve_value_defs(
            module, value_map, arg_count, predefined_values
        )
        for i, arg_id in enumerate(arg_ids):
            vid, offset = self._read_value_def(data, offset, module, value_map, arg_id)
            arg_ids[i] = vid
        if len(value_map) < len(predefined_values):
            raise BytecodeError("function body entry block is missing signature args")

        # Operations.
        op_count, offset = decode_varint(data, offset)
        ops = []
        for _ in range(op_count):
            op, offset = self._read_operation(
                data, offset, module, value_map, region_blocks
            )
            ops.append(op)

        block.label = label
        block.arg_ids = arg_ids
        block.ops = ops
        block.comments = comments
        return offset

    def _read_operation(
        self,
        data: bytes,
        offset: int,
        module: Module,
        value_map: list[int],
        region_blocks: list[Block],
    ) -> tuple[Operation, int]:
        op_table_index_plus1, offset = decode_varint(data, offset)
        if op_table_index_plus1 == 0:
            raise BytecodeError("operation op_table_index_plus1 must be nonzero")
        _instance_flags = data[offset]
        offset += 1
        location_id, offset = decode_varint(data, offset)
        if self._location_mode == LOCATION_MODE_NO_LOCATIONS and location_id != 0:
            raise BytecodeError("NO_LOCATIONS bytecode op location must be 0")
        comments, offset = self._read_comment_list(data, offset)

        kind_id = op_table_index_plus1 - 1
        op_name = self._ops[kind_id] if kind_id < len(self._ops) else ""

        # Operands: map function-local value numbers to module value IDs.
        operand_count, offset = decode_varint(data, offset)
        operands = []
        for _ in range(operand_count):
            value_ref, offset = decode_varint(data, offset)
            operands.append(self._map_value_ref(value_ref, value_map))
        operand_segment_counts: tuple[int, ...] = ()
        op_decl = self._op_decls_by_name.get(op_name)
        if op_decl is not None:
            layout = compute_layout(op_decl)
            if layout.segmented_operands:
                counts: list[int] = []
                for operand in op_decl.operands:
                    count, offset = decode_varint(data, offset)
                    if not operand.variadic:
                        if operand.optional:
                            if count > 1:
                                raise BytecodeError(
                                    "optional operand segment count exceeds one"
                                )
                        elif count != 1:
                            raise BytecodeError(
                                "required operand segment count is not one"
                            )
                    counts.append(count)
                if sum(counts) != operand_count:
                    raise BytecodeError(
                        "operand segment counts do not sum to operand count"
                    )
                operand_segment_counts = tuple(counts)

        # Successors are region-local block ordinals. Region blocks are
        # preallocated before block bodies are decoded so forward edges can
        # resolve directly to their target Block objects.
        successor_count, offset = decode_varint(data, offset)
        successors = []
        for _ in range(successor_count):
            block_index, offset = decode_varint(data, offset)
            try:
                successors.append(region_blocks[block_index])
            except IndexError as exc:
                raise BytecodeError("successor block index is out of range") from exc

        # Results.
        result_count, offset = decode_varint(data, offset)
        result_ids = self._reserve_value_defs(module, value_map, result_count)
        for i, result_id in enumerate(result_ids):
            vid, offset = self._read_value_def(
                data, offset, module, value_map, result_id
            )
            result_ids[i] = vid

        # Tied results.
        tied_count, offset = decode_varint(data, offset)
        tied_results = []
        for _ in range(tied_count):
            result_idx, offset = decode_varint(data, offset)
            operand_idx, offset = decode_varint(data, offset)
            tied_results.append(
                TiedResult(result_index=result_idx, operand_index=operand_idx)
            )

        # Attributes.
        attr_count, offset = decode_varint(data, offset)
        attributes, offset = self._read_op_attr_entries(
            data, offset, attr_count, module, value_map, op_name
        )

        # Regions.
        region_count, offset = decode_varint(data, offset)
        regions = []
        for _ in range(region_count):
            region, offset = self._read_region(
                data,
                offset,
                module,
                value_map,
                predefined_values=[],
            )
            regions.append(region)

        op = Operation(
            name=op_name,
            operands=operands,
            operand_segment_counts=operand_segment_counts,
            results=result_ids,
            tied_results=tied_results,
            successors=successors,
            attributes=attributes,
            regions=regions,
            location_id=location_id,
            comments=comments,
        )
        return op, offset

    def _read_attr_value(
        self,
        data: bytes,
        offset: int,
        module: Module | None = None,
        value_map: list[int] | None = None,
    ) -> tuple[Any, int]:
        kind = data[offset]
        offset += 1
        match kind:
            case 0:  # I64
                value, offset = decode_signed_varint(data, offset)
                return value, offset
            case 1:  # F64
                value = struct.unpack_from("<d", data, offset)[0]
                return value, offset + 8
            case 2:  # STRING
                string_id, offset = decode_varint(data, offset)
                return self._strings[string_id], offset
            case 3:  # BOOL
                return bool(data[offset]), offset + 1
            case 4:  # ENUM
                return data[offset], offset + 1
            case 5:  # I64_ARRAY
                count, offset = decode_varint(data, offset)
                values = []
                for _ in range(count):
                    v, offset = decode_signed_varint(data, offset)
                    values.append(v)
                return values, offset
            case 6:  # SYMBOL
                string_id, offset = decode_varint(data, offset)
                return SymbolName(self._strings[string_id]), offset
            case 7:  # TYPE
                type_idx, offset = decode_varint(data, offset)
                return self._types[type_idx], offset
            case 8:  # PREDICATE_LIST
                predicates, offset = self._read_predicate_list(
                    data, offset, module, value_map
                )
                return predicates, offset
            case 9:  # DICT
                count, offset = decode_varint(data, offset)
                return self._read_attr_dict_entries(
                    data, offset, count, module, value_map
                )
            case 10:  # ENCODING
                encoding_id, offset = decode_varint(data, offset)
                if encoding_id == 0 or encoding_id > len(self._encodings):
                    raise BytecodeError(
                        f"encoding attr id {encoding_id} out of range "
                        f"(encoding table has {len(self._encodings)} entries)"
                    )
                return self._encodings[encoding_id - 1], offset
            case 11:  # BYTES
                byte_length, offset = decode_varint(data, offset)
                end_offset = offset + byte_length
                if end_offset > len(data):
                    raise BytecodeError(
                        f"bytes attr length {byte_length} exceeds payload size"
                    )
                return data[offset:end_offset], end_offset
            case _:
                raise BytecodeError(f"unknown attr value kind: {kind}")

    def _read_attr_dict_entries(
        self,
        data: bytes,
        offset: int,
        count: int,
        module: Module | None = None,
        value_map: list[int] | None = None,
    ) -> tuple[CanonicalAttrDict, int]:
        """Read canonical dict attr entries and verify sorted/deduped order."""
        entries: list[tuple[str, Any]] = []
        previous_key: str | None = None
        for _ in range(count):
            key_id, offset = decode_varint(data, offset)
            if key_id >= len(self._strings):
                raise BytecodeError(
                    f"dict attr key string_id {key_id} out of range "
                    f"(string table has {len(self._strings)} entries)"
                )
            key = self._strings[key_id]
            if previous_key is not None and key <= previous_key:
                if key == previous_key:
                    raise BytecodeError(f"duplicate dict attr key: {key!r}")
                raise BytecodeError(
                    "dict attr keys are not in canonical order: "
                    f"{previous_key!r} appears before {key!r}"
                )
            value, offset = self._read_attr_value(data, offset, module, value_map)
            entries.append((key, value))
            previous_key = key
        return CanonicalAttrDict.from_sorted_items(entries), offset

    def _read_op_attr_entries(
        self,
        data: bytes,
        offset: int,
        count: int,
        module: Module | None = None,
        value_map: list[int] | None = None,
        op_name: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Read op attribute entries, preserving payload order."""
        attributes: dict[str, Any] = {}
        for _ in range(count):
            key_id, offset = decode_varint(data, offset)
            if key_id >= len(self._strings):
                raise BytecodeError(
                    f"op attr key string_id {key_id} out of range "
                    f"(string table has {len(self._strings)} entries)"
                )
            key = self._strings[key_id]
            if key in attributes:
                raise BytecodeError(f"duplicate op attr key: {key!r}")
            value, offset = self._read_attr_value(data, offset, module, value_map)
            attr_def = self._attr_def_for_op_attr(op_name, key) if op_name else None
            if getattr(attr_def, "attr_type", None) == "enum":
                value = self._enum_value_for_attr(attr_def, value)
            attributes[key] = value
        return attributes, offset

    def _attr_def_for_op_attr(self, op_name: str, attr_name: str) -> Any | None:
        """Return generated attr metadata when the reader knows the op."""
        return attr_def_for_op(self._op_decls_by_name, op_name, attr_name)

    def _enum_value_for_attr(self, attr_def: Any, value: int) -> str | int:
        """Map known enum ordinals back to keywords and preserve future ordinals."""
        enum_def = getattr(attr_def, "enum_def", None)
        for enum_case in getattr(enum_def, "cases", ()):
            if enum_case.value == value:
                return cast(str, enum_case.keyword)
        return value

    # Predicate kind byte → name mapping (inverse of writer).
    _PRED_KIND_NAMES: ClassVar[list[str]] = [
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "mul",
        "min",
        "max",
        "pow2",
        "range",
        "not_nan",
        "not_inf",
        "finite",
    ]

    def _read_predicate_list(
        self,
        data: bytes,
        offset: int,
        module: Module | None = None,
        value_map: list[int] | None = None,
    ) -> tuple[list[Predicate], int]:
        """Read a predicate list: count + per-predicate data."""
        count, offset = decode_varint(data, offset)
        predicates: list[Predicate] = []
        for _ in range(count):
            kind_byte = data[offset]
            offset += 1
            arg_count = data[offset]
            offset += 1
            if kind_byte >= len(self._PRED_KIND_NAMES):
                raise BytecodeError(f"unknown predicate kind byte: {kind_byte}")
            kind = self._PRED_KIND_NAMES[kind_byte]
            args: list[PredicateArg] = []
            for _ in range(arg_count):
                arg, offset = self._read_predicate_arg(data, offset, module, value_map)
                args.append(arg)
            predicates.append(Predicate(kind=kind, args=tuple(args)))
        return predicates, offset

    def _read_predicate_arg(
        self,
        data: bytes,
        offset: int,
        module: Module | None = None,
        value_map: list[int] | None = None,
    ) -> tuple[PredicateArg, int]:
        """Read a single predicate argument: tag + value."""
        tag_byte = data[offset]
        offset += 1
        match tag_byte:
            case 1:  # VALUE
                encoded_value, offset = decode_varint(data, offset)
                if module is not None and value_map is not None:
                    if encoded_value >= len(value_map):
                        raise BytecodeError(
                            "predicate value number "
                            f"{encoded_value} out of range "
                            f"(function has {len(value_map)} values so far)"
                        )
                    value_id = value_map[encoded_value]
                    return (
                        PredicateArg(tag="value", value=module.values[value_id].name),
                        offset,
                    )
                if encoded_value >= len(self._strings):
                    raise BytecodeError(
                        f"predicate value string_id {encoded_value} out of range "
                        f"(string table has {len(self._strings)} entries)"
                    )
                return (
                    PredicateArg(tag="value", value=self._strings[encoded_value]),
                    offset,
                )
            case 2:  # CONST
                value, offset = decode_signed_varint(data, offset)
                return PredicateArg(tag="const", value=value), offset
            case _:
                raise BytecodeError(f"unknown predicate arg tag: {tag_byte}")

    def _symbol_field_for_op(self, op_name: str) -> str:
        """Return the generated symbol identity attr field for ``op_name``."""
        try:
            return cast(str, symbol_def_for_op(self._op_decls_by_name, op_name).field)
        except ValueError as err:
            raise BytecodeError(str(err)) from err


def read_module(
    data: bytes,
    *,
    op_decls: Iterable[Any] | None = None,
    verify: bool = False,
) -> Module:
    """Read a module from .loombc bytes."""
    op_decl_tuple = tuple(op_decls) if op_decls is not None else None
    try:
        module = BytecodeReader(data, op_decls=op_decl_tuple).read()
    except ValueError as err:
        raise BytecodeError(str(err)) from err
    if verify:
        from loom.verify import verify_module

        diagnostics = verify_module(module, ops=op_decl_tuple)
        diagnostics.raise_if_errors()
    return module
