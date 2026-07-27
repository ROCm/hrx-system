# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Comprehensive tests for the bytecode reader.

Every construct that the writer can produce is verified here through
write → read round-trips. Validation tests exercise error handling
for malformed input. The reader is the first consumer of the writer's
output — these tests prove both are correct.
"""

import struct
from typing import Any

import pytest

from loom.format.bytecode.encoding import decode_varint
from loom.format.bytecode.reader import BytecodeError, BytecodeReader, read_module
from loom.format.bytecode.writer import (
    BYTECODE_TYPE_KIND_BY_IR_KIND,
    LOCATION_MODE_FULL_LOCATIONS,
    LOCATION_MODE_NO_LOCATIONS,
    SECTION_ENCODINGS,
    SECTION_IR,
    SECTION_LOCATIONS,
    write_module,
)
from loom.ir import (
    BF16,
    BUFFER_TYPE,
    F32,
    I8,
    I32,
    I64,
    INDEX,
    LOCATION_TAG_SANITIZER_SITE,
    OFFSET,
    SYMBOL_FLAG_IMPORT,
    SYMBOL_FLAG_PUBLIC,
    VALUE_DEF_OP_NONE,
    Block,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    FileLocation,
    FunctionType,
    GroupScope,
    GroupType,
    Module,
    Operation,
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
    TaggedLocation,
    TiedResult,
    Type,
    TypeKind,
    Use,
    Value,
)
from loom.stable_id import stable_id_from_string
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET

_TEST_LOW_CORE_STABLE_ID = stable_id_from_string(TEST_LOW_CORE_DESCRIPTOR_SET.key)
_TEST_PTR_REGISTER_CLASS_ID = next(
    i
    for i, register_class in enumerate(TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    if register_class.name == "test.ptr"
)

# ============================================================================
# Helpers
# ============================================================================


def _roundtrip(module: Module) -> Module:
    """Write → read round-trip."""
    return read_module(write_module(module))


def _test_ptr_register_type(unit_count: int = 1) -> RegisterType:
    return RegisterType(
        _TEST_LOW_CORE_STABLE_ID,
        _TEST_PTR_REGISTER_CLASS_ID,
        unit_count,
        "test.ptr",
    )


def _make_value_with_bindings(
    module: Module, name: str, value_type: Type
) -> tuple[int, list[int]]:
    """Create a value with proper dim_bindings for any dynamic dims.

    Returns (value_id, list of dim value_ids that must be block args).
    """
    dim_bindings: dict[int, int] = {}
    dim_value_ids: list[int] = []
    if hasattr(value_type, "dims"):
        for i, dim in enumerate(value_type.dims):
            if isinstance(dim, DynamicDim):
                dim_id = module.add_value(Value(name=f"{name}_d{i}", type=INDEX))
                dim_bindings[i] = dim_id
                dim_value_ids.append(dim_id)
    value_id = module.add_value(
        Value(name=name, type=value_type, dim_bindings=dim_bindings)
    )
    return value_id, dim_value_ids


def _make_func(
    module: Module,
    name: str,
    arg_types: list[Type],
    result_types: list[Type] | None = None,
    ops: list[Operation] | None = None,
    is_public: bool = False,
    is_declaration: bool = False,
) -> None:
    """Add a func-like op to the module as a symbol."""
    result_types = result_types or []

    attrs: dict[str, Any] = {"callee": name}
    if is_public:
        attrs["visibility"] = "public"

    # Create anonymous result value IDs at module level.
    result_ids = []
    for rt in result_types:
        rid, _ = _make_value_with_bindings(module, "", rt)
        result_ids.append(rid)

    if is_declaration:
        # func.decl: args are operands.
        operand_ids = []
        for i, at in enumerate(arg_types):
            vid, dim_ids = _make_value_with_bindings(module, f"{name}_arg{i}", at)
            operand_ids.extend(dim_ids)
            operand_ids.append(vid)
        op = Operation(
            name="func.decl",
            operands=operand_ids,
            results=result_ids,
            attributes=attrs,
        )
        sym_kind = SymbolKind.FUNC_DECL
    else:
        # func.def: args are entry block arguments.
        arg_ids = []
        for i, at in enumerate(arg_types):
            vid, dim_ids = _make_value_with_bindings(module, f"{name}_arg{i}", at)
            arg_ids.extend(dim_ids)
            arg_ids.append(vid)
        body_ops = ops or [
            Operation(name="test.yield", operands=arg_ids[-1:] if arg_ids else [])
        ]
        block = Block(arg_ids=arg_ids, ops=body_ops)
        body = Region(blocks=[block])
        op = Operation(
            name="func.def",
            results=result_ids,
            attributes=attrs,
            regions=[body],
        )
        sym_kind = SymbolKind.FUNC_DEF

    flags = SYMBOL_FLAG_PUBLIC if is_public else 0
    module.add_symbol(Symbol(name=name, kind=sym_kind, flags=flags, op=op))


def _module_range(data: bytes | bytearray) -> tuple[int, int]:
    """Return the absolute offset and length of the first module."""
    producer_end = data.index(0, 16)
    directory_offset = (producer_end + 1 + 7) & ~7
    module_offset = struct.unpack_from("<Q", data, directory_offset + 8)[0]
    module_length = struct.unpack_from("<Q", data, directory_offset + 16)[0]
    return module_offset, module_length


def _section_entries(data: bytes | bytearray) -> list[tuple[int, int, int, int]]:
    """Return (entry_offset, kind, section_offset, length) directory entries."""
    module_offset, _module_length = _module_range(data)
    offset = module_offset
    section_count, offset = decode_varint(data, offset)
    for _ in range(4):
        _count, offset = decode_varint(data, offset)

    entries = []
    for _ in range(section_count):
        kind = struct.unpack_from("<H", data, offset)[0]
        section_offset = struct.unpack_from("<Q", data, offset + 8)[0]
        length = struct.unpack_from("<Q", data, offset + 16)[0]
        entries.append((offset, kind, section_offset, length))
        offset += 32
    return entries


def _find_section_entry(
    data: bytes | bytearray, section_kind: int
) -> tuple[int, int, int]:
    """Return (entry_offset, section_offset, length) for a section kind."""
    for entry_offset, kind, section_offset, length in _section_entries(data):
        if kind == section_kind:
            return entry_offset, section_offset, length
    raise AssertionError(f"missing section kind {section_kind}")


def _make_single_op_body_module(op_name: str = "test.yield") -> Module:
    """Return a module with one function body containing one zero-operand op."""
    module = Module(name="test")
    block = Block(ops=[Operation(name=op_name)])
    body = Region(blocks=[block])
    module.add_symbol(
        Symbol(
            name="f",
            kind=SymbolKind.FUNC_DEF,
            op=Operation(
                name="func.def",
                attributes={"callee": "f"},
                regions=[body],
            ),
        )
    )
    return module


def _single_op_offset(data: bytes | bytearray) -> int:
    """Return the absolute offset of the first op in _make_single_op_body_module."""
    module_offset, _module_length = _module_range(data)
    _entry_offset, section_offset, _section_length = _find_section_entry(
        data, SECTION_IR
    )
    body_offset = module_offset + section_offset
    offset = body_offset
    for _ in range(4):
        _count, offset = decode_varint(data, offset)
    _root_region_count, offset = decode_varint(data, offset)
    _root_region_index, offset = decode_varint(data, offset)
    _block_count, offset = decode_varint(data, offset)
    offset += 1  # block has_label byte.
    comment_count, offset = decode_varint(data, offset)
    for _ in range(comment_count):
        comment_length, offset = decode_varint(data, offset)
        offset += comment_length
    _arg_count, offset = decode_varint(data, offset)
    _op_count, offset = decode_varint(data, offset)
    return offset


def _make_encoding_alias_module() -> Module:
    """Return a module with one aliased encoding instance."""
    enc = EncodingInstance(name="q8_0", alias="enc", params=(("block", 32),))
    module = Module(name="test")
    module.add_encoding(enc)
    tile_type = ShapedType(TypeKind.TILE, I8, (StaticDim(128),), encoding=enc)
    value_id = module.add_value(Value(name="v", type=tile_type))
    block = Block(arg_ids=[value_id], ops=[Operation(name="test.yield")])
    body = Region(blocks=[block])
    module.add_symbol(
        Symbol(
            name="f",
            kind=SymbolKind.FUNC_DEF,
            op=Operation(
                name="func.def",
                attributes={"callee": "f"},
                regions=[body],
            ),
        )
    )
    return module


# ============================================================================
# Validation — malformed input
# ============================================================================


class TestBadMagic:
    def test_wrong_magic(self) -> None:
        with pytest.raises(BytecodeError, match="invalid magic"):
            read_module(b"BADM" + b"\x00" * 20)

    def test_empty_input(self) -> None:
        with pytest.raises(BytecodeError):
            read_module(b"")

    def test_truncated_magic(self) -> None:
        with pytest.raises(BytecodeError):
            read_module(b"LOO")

    def test_only_magic(self) -> None:
        with pytest.raises(BytecodeError):
            read_module(b"LOOM")


class TestBadVersion:
    def test_future_version(self) -> None:
        data = bytearray(b"LOOM")
        data.append(0xFF)
        data.extend(b"\x00" * 20)
        with pytest.raises(BytecodeError, match="unsupported format version"):
            read_module(bytes(data))


class TestTruncatedInput:
    def test_truncated_header(self) -> None:
        with pytest.raises(BytecodeError):
            read_module(b"LOOM\x00\x00")

    def test_truncated_after_header(self) -> None:
        data = write_module(Module(name="test"))
        with pytest.raises((BytecodeError, Exception)):
            read_module(data[:24])


class TestMalformedSectionDirectory:
    def test_non_canonical_section_count_varint_is_rejected(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        module_offset, _module_length = _module_range(data)

        data[module_offset : module_offset + 1] = b"\x88\x00"

        with pytest.raises(BytecodeError, match="non-canonical"):
            read_module(bytes(data))

    def test_duplicate_section_kind_is_rejected(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        entries = _section_entries(data)
        assert len(entries) >= 2

        struct.pack_into("<H", data, entries[1][0], entries[0][1])

        with pytest.raises(BytecodeError, match="duplicate section kind"):
            read_module(bytes(data))

    def test_overlapping_section_range_is_rejected(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        entries = _section_entries(data)
        assert len(entries) >= 2
        target_index = next(
            index for index in range(1, len(entries)) if entries[index - 1][3] > 0
        )

        struct.pack_into(
            "<Q", data, entries[target_index][0] + 8, entries[target_index - 1][2]
        )

        with pytest.raises(BytecodeError, match="section directory is not sorted"):
            read_module(bytes(data))

    def test_unsupported_section_flags_are_rejected(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        entry_offset, _kind, _section_offset, _length = _section_entries(data)[0]

        struct.pack_into("<H", data, entry_offset + 2, 1)

        with pytest.raises(BytecodeError, match="unsupported flags"):
            read_module(bytes(data))


class TestMalformedLocationMode:
    def test_full_locations_mode_is_rejected_until_implemented(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        data[5] = LOCATION_MODE_FULL_LOCATIONS

        with pytest.raises(BytecodeError, match="FULL_LOCATIONS"):
            read_module(bytes(data))

    def test_no_locations_mode_rejects_locations_section(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        data[5] = LOCATION_MODE_NO_LOCATIONS

        with pytest.raises(BytecodeError, match="NO_LOCATIONS bytecode"):
            read_module(bytes(data))

    def test_source_locations_mode_requires_locations_section(self) -> None:
        data = bytearray(write_module(Module(name="test")))
        locations_entry_offset, _section_offset, _length = _find_section_entry(
            data, SECTION_LOCATIONS
        )

        struct.pack_into("<H", data, locations_entry_offset, 0xFF)

        with pytest.raises(BytecodeError, match="source locations"):
            read_module(bytes(data))


class TestMalformedIrSection:
    def test_op_table_index_plus1_zero_is_rejected(self) -> None:
        data = bytearray(write_module(_make_single_op_body_module()))
        op_offset = _single_op_offset(data)

        data[op_offset] = 0

        with pytest.raises(BytecodeError, match="op_table_index_plus1"):
            read_module(bytes(data))

    def test_function_body_summary_mismatch_is_rejected(self) -> None:
        data = bytearray(write_module(_make_single_op_body_module()))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _section_length = _find_section_entry(
            data, SECTION_IR
        )

        # The valid fixture has one operation. Claim the symbol region payload
        # has zero operations so its local summary fails before the module
        # summary is checked.
        body_offset = module_offset + section_offset
        op_count_offset = body_offset + 3
        data[op_count_offset] = 0

        with pytest.raises(BytecodeError, match="symbol region allocation summary"):
            read_module(bytes(data))


class TestMalformedEncodingSection:
    def test_alias_string_id_plus1_out_of_range_is_rejected(self) -> None:
        data = bytearray(write_module(_make_encoding_alias_module()))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _section_length = _find_section_entry(
            data, SECTION_ENCODINGS
        )
        offset = module_offset + section_offset

        family_count, offset = decode_varint(data, offset)
        for _ in range(family_count):
            _name_id, offset = decode_varint(data, offset)
        instance_count, offset = decode_varint(data, offset)
        assert instance_count == 1
        _family_index, offset = decode_varint(data, offset)

        data[offset] = 0x7F

        with pytest.raises(BytecodeError, match="encoding alias string_id"):
            read_module(bytes(data))


class TestMalformedTypeSection:
    def _read_types(
        self,
        data: bytes,
        encodings: list[EncodingInstance] | None = None,
    ) -> list[Type]:
        reader = BytecodeReader(b"")
        reader._encodings = encodings or []
        reader._read_types_section((0, data))
        return reader._types

    def test_vector_rank_zero_is_rejected(self) -> None:
        data = bytes(
            [
                1,  # type count
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.VECTOR],
                F32.kind.value,
                0,  # rank
                0,  # no encoding/layout attachment
                0,  # attachment id
            ]
        )
        with pytest.raises(BytecodeError, match="rank >= 1"):
            self._read_types(data)

    def test_vector_encoding_attachment_is_rejected(self) -> None:
        data = bytes(
            [
                1,  # type count
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.VECTOR],
                F32.kind.value,
                1,  # rank
                1,  # static encoding/layout attachment
                1,  # encoding table id
                0,  # static dim
                4,  # dim size
            ]
        )
        with pytest.raises(BytecodeError, match="must not carry"):
            self._read_types(data, [EncodingInstance(name="dense")])

    def test_unknown_encoding_attachment_is_rejected(self) -> None:
        data = bytes(
            [
                1,  # type count
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.TILE],
                F32.kind.value,
                0,  # rank
                9,  # unknown encoding attachment
                0,  # attachment id
            ]
        )
        with pytest.raises(BytecodeError, match="unknown encoding attachment"):
            self._read_types(data)

    def test_static_encoding_id_zero_is_rejected(self) -> None:
        data = bytes(
            [
                1,  # type count
                2,  # tile
                F32.kind.value,
                0,  # rank
                1,  # static encoding attachment
                0,  # invalid zero table id
            ]
        )
        with pytest.raises(BytecodeError, match="static encoding id out of range"):
            self._read_types(data)


# ============================================================================
# Module structure
# ============================================================================


class TestModuleStructure:
    def test_empty_module(self) -> None:
        loaded = _roundtrip(Module(name="empty"))
        assert len(loaded.symbols) == 0

    def test_module_with_one_function(self) -> None:
        module = Module(name="test")
        _make_func(module, "f", [F32])
        loaded = _roundtrip(module)
        assert len(loaded.symbols) == 1
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.attributes.get("callee") == "f"

    def test_module_with_multiple_functions(self) -> None:
        module = Module(name="test")
        _make_func(module, "f1", [F32])
        _make_func(module, "f2", [I32])
        _make_func(module, "f3", [INDEX])
        loaded = _roundtrip(module)
        assert len(loaded.symbols) == 3
        names = {s.name for s in loaded.symbols}
        assert names == {"f1", "f2", "f3"}

    def test_declaration(self) -> None:
        module = Module(name="test")
        _make_func(module, "extern", [I32], result_types=[I32], is_declaration=True)
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert not loaded_op.regions  # Declarations have no body region.

    def test_public_function(self) -> None:
        module = Module(name="test")
        _make_func(module, "exported", [F32], is_public=True)
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.attributes.get("visibility") == "public"


# ============================================================================
# Type round-trips (every variant)
# ============================================================================


class TestTypeRoundTrips:
    def _roundtrip_type(self, ir_type: Type) -> Type:
        module = Module(name="test")
        _make_func(module, "f", [ir_type])
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        # func.def: args are in the entry block. The value under test
        # is the last arg (dim values for dynamic dims come first).
        arg_id = loaded_op.regions[0].blocks[0].arg_ids[-1]
        return loaded.values[arg_id].type

    # Scalars.
    def test_f32(self) -> None:
        assert self._roundtrip_type(F32) == F32

    def test_i32(self) -> None:
        assert self._roundtrip_type(I32) == I32

    def test_i8(self) -> None:
        assert self._roundtrip_type(I8) == I8

    def test_i64(self) -> None:
        assert self._roundtrip_type(I64) == I64

    def test_index(self) -> None:
        assert self._roundtrip_type(INDEX) == INDEX

    def test_offset(self) -> None:
        assert self._roundtrip_type(OFFSET) == OFFSET

    def test_bf16(self) -> None:
        assert self._roundtrip_type(BF16) == BF16

    def test_f8e4m3(self) -> None:
        assert self._roundtrip_type(ScalarType(ScalarTypeKind.F8E4M3)) == ScalarType(
            ScalarTypeKind.F8E4M3
        )

    def test_f8e5m2(self) -> None:
        assert self._roundtrip_type(ScalarType(ScalarTypeKind.F8E5M2)) == ScalarType(
            ScalarTypeKind.F8E5M2
        )

    # Shaped types.
    def test_tile_0d(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, ())
        assert self._roundtrip_type(t) == t

    def test_tile_1d(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        assert self._roundtrip_type(t) == t

    def test_tile_2d(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(8)))
        assert self._roundtrip_type(t) == t

    def test_tile_3d(self) -> None:
        t = ShapedType(TypeKind.TILE, I32, (StaticDim(2), StaticDim(3), StaticDim(4)))
        assert self._roundtrip_type(t) == t

    def test_tensor_1d(self) -> None:
        t = ShapedType(TypeKind.TENSOR, I8, (StaticDim(256),))
        assert self._roundtrip_type(t) == t

    def test_vector_1d(self) -> None:
        t = ShapedType(TypeKind.VECTOR, F32, (StaticDim(16),))
        assert self._roundtrip_type(t) == t

    def test_vector_zero_extent(self) -> None:
        t = ShapedType(TypeKind.VECTOR, F32, (StaticDim(0),))
        assert self._roundtrip_type(t) == t

    def test_vector_dynamic(self) -> None:
        t = ShapedType(TypeKind.VECTOR, I32, (DynamicDim(),))
        assert self._roundtrip_type(t) == t

    def test_view_1d(self) -> None:
        t = ShapedType(TypeKind.VIEW, I8, (StaticDim(256),))
        assert self._roundtrip_type(t) == t

    def test_view_with_layout(self) -> None:
        layout = EncodingInstance(name="strided", params=(("stride", 64),))
        t = ShapedType(TypeKind.VIEW, F32, (StaticDim(256),), encoding=layout)
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, ShapedType)
        assert loaded.type_kind == TypeKind.VIEW
        assert isinstance(loaded.encoding, EncodingInstance)
        assert loaded.encoding.name == "strided"
        assert loaded.encoding.params == (("stride", 64),)

    def test_storage_workgroup(self) -> None:
        t = StorageType(StorageSpace.WORKGROUP)
        assert self._roundtrip_type(t) == t

    def test_view_with_dynamic_layout(self) -> None:
        t = ShapedType(
            TypeKind.VIEW,
            F32,
            (StaticDim(256),),
            encoding=DynamicEncoding(),
        )
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, ShapedType)
        assert loaded.type_kind == TypeKind.VIEW
        assert isinstance(loaded.encoding, DynamicEncoding)

    def test_tile_dynamic(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        assert self._roundtrip_type(t) == t

    def test_tile_all_dynamic(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (DynamicDim(), DynamicDim()))
        assert self._roundtrip_type(t) == t

    def test_tile_large_dim(self) -> None:
        t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(1048576),))
        assert self._roundtrip_type(t) == t

    def test_tile_with_encoding(self) -> None:
        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, ShapedType)
        assert loaded.has_encoding
        assert isinstance(loaded.encoding, EncodingInstance)
        assert loaded.encoding.name == "q8_0"
        assert loaded.encoding.params == (("block", 32),)

    def test_tile_encoding_no_params(self) -> None:
        enc = EncodingInstance(name="dense")
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, ShapedType)
        assert loaded.has_encoding
        assert isinstance(loaded.encoding, EncodingInstance)
        assert loaded.encoding.name == "dense"

    def test_tile_encoding_multiple_params(self) -> None:
        enc = EncodingInstance(name="q8_0", params=(("block", 32), ("group", 128)))
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, ShapedType)
        assert isinstance(loaded.encoding, EncodingInstance)
        assert loaded.encoding.params == (("block", 32), ("group", 128))

    # Group type.
    def test_group_workgroup(self) -> None:
        assert self._roundtrip_type(GroupType(GroupScope.WORKGROUP)) == GroupType(
            GroupScope.WORKGROUP
        )

    def test_group_subgroup(self) -> None:
        assert self._roundtrip_type(GroupType(GroupScope.SUBGROUP)) == GroupType(
            GroupScope.SUBGROUP
        )

    # Function type.
    def test_function_type(self) -> None:
        ft = FunctionType((F32, I32), (F32,))
        assert self._roundtrip_type(ft) == ft

    def test_function_type_empty(self) -> None:
        ft = FunctionType((), ())
        assert self._roundtrip_type(ft) == ft

    def test_function_type_many_args(self) -> None:
        ft = FunctionType((F32, I32, INDEX, BF16), (F32, I32))
        assert self._roundtrip_type(ft) == ft

    # Dialect types.
    def test_dialect_opaque(self) -> None:
        t = DialectType("hal.buffer")
        assert self._roundtrip_type(t) == t

    def test_dialect_parameterized(self) -> None:
        t = DialectType("vm.ref", (DialectType("hal.buffer"),))
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, DialectType)
        assert loaded.name == "vm.ref"
        assert isinstance(loaded.params[0], DialectType)
        assert loaded.params[0].name == "hal.buffer"

    def test_dialect_nested(self) -> None:
        inner = DialectType("vm.list", (I32,))
        t = DialectType("vm.ref", (inner,))
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, DialectType)
        assert loaded.name == "vm.ref"
        assert isinstance(loaded.params[0], DialectType)
        assert loaded.params[0].name == "vm.list"
        assert loaded.params[0].params[0] == I32

    def test_dialect_multiple_params(self) -> None:
        t = DialectType(
            "hal.pair", (DialectType("hal.buffer"), DialectType("hal.fence"))
        )
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, DialectType)
        assert len(loaded.params) == 2
        assert isinstance(loaded.params[0], DialectType)
        assert isinstance(loaded.params[1], DialectType)
        assert loaded.params[0].name == "hal.buffer"
        assert loaded.params[1].name == "hal.fence"

    def test_register_type(self) -> None:
        t = _test_ptr_register_type(4)
        assert self._roundtrip_type(t) == t

    def test_buffer_type(self) -> None:
        assert self._roundtrip_type(BUFFER_TYPE) is BUFFER_TYPE


# ============================================================================
# Attribute round-trips (every kind)
# ============================================================================


class TestAttributeRoundTrips:
    def _roundtrip_attr(self, key: str, value: object) -> object:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        op = Operation(name="test.op", results=[], attributes={key: value})
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        return loaded_op.regions[0].blocks[0].ops[0].attributes.get(key)

    # Integers.
    def test_i64_zero(self) -> None:
        assert self._roundtrip_attr("v", 0) == 0

    def test_i64_positive(self) -> None:
        assert self._roundtrip_attr("v", 42) == 42

    def test_i64_negative(self) -> None:
        assert self._roundtrip_attr("v", -7) == -7

    def test_i64_large_positive(self) -> None:
        assert self._roundtrip_attr("v", 2**40) == 2**40

    def test_i64_large_negative(self) -> None:
        assert self._roundtrip_attr("v", -(2**40)) == -(2**40)

    def test_i64_max(self) -> None:
        assert self._roundtrip_attr("v", 2**62) == 2**62

    # Floats.
    def test_f64_positive(self) -> None:
        result = self._roundtrip_attr("v", 3.14)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-15

    def test_f64_negative(self) -> None:
        result = self._roundtrip_attr("v", -2.718)
        assert isinstance(result, float)
        assert abs(result - (-2.718)) < 1e-15

    def test_f64_zero(self) -> None:
        assert self._roundtrip_attr("v", 0.0) == 0.0

    def test_f64_small(self) -> None:
        result = self._roundtrip_attr("v", 1e-30)
        assert isinstance(result, float)
        assert abs(result - 1e-30) < 1e-45

    # Strings.
    def test_string_ascii(self) -> None:
        assert self._roundtrip_attr("v", "hello") == "hello"

    def test_string_empty(self) -> None:
        assert self._roundtrip_attr("v", "") == ""

    def test_string_with_spaces(self) -> None:
        assert self._roundtrip_attr("v", "hello world") == "hello world"

    # Bytes.
    def test_bytes_empty(self) -> None:
        assert self._roundtrip_attr("v", b"") == b""

    def test_bytes_payload(self) -> None:
        assert self._roundtrip_attr("v", b"\x00\x11\xfe\xff") == b"\x00\x11\xfe\xff"

    # Booleans.
    def test_bool_true(self) -> None:
        assert self._roundtrip_attr("v", True) is True

    def test_bool_false(self) -> None:
        assert self._roundtrip_attr("v", False) is False

    # Integer arrays.
    def test_i64_array_empty(self) -> None:
        assert self._roundtrip_attr("v", []) == []

    def test_i64_array_single(self) -> None:
        assert self._roundtrip_attr("v", [42]) == [42]

    def test_i64_array_multiple(self) -> None:
        assert self._roundtrip_attr("v", [1, 2, 3]) == [1, 2, 3]

    def test_i64_array_negative(self) -> None:
        assert self._roundtrip_attr("v", [-1, -2, -3]) == [-1, -2, -3]

    def test_i64_array_with_sentinel(self) -> None:
        sentinel = -(2**63)
        assert self._roundtrip_attr("v", [0, sentinel, 4]) == [0, sentinel, 4]

    # Multiple attributes on one op.
    def test_multiple_attrs(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        op = Operation(
            name="test.op",
            results=[],
            attributes={"axis": 0, "label": "foo", "flag": True, "scale": 0.5},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        attrs = loaded_op.regions[0].blocks[0].ops[0].attributes
        assert attrs["axis"] == 0
        assert attrs["label"] == "foo"
        assert attrs["flag"] is True
        scale = attrs["scale"]
        assert isinstance(scale, float)
        assert abs(scale - 0.5) < 1e-15


# ============================================================================
# IR structure round-trips
# ============================================================================


class TestIRStructure:
    def test_op_with_results(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        neg = Operation(name="test.neg", operands=[x], results=[r])
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[x], ops=[neg, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        neg_loaded = loaded_op.regions[0].blocks[0].ops[0]
        assert neg_loaded.name == "test.neg"
        assert len(neg_loaded.operands) == 1
        assert len(neg_loaded.results) == 1

    def test_value_metadata_is_rebuilt_after_read(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        neg = Operation(name="test.neg", operands=[x], results=[r])
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[x], ops=[neg, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(
            name="func.def",
            attributes={"callee": "f"},
            regions=[body],
        )
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        entry_block = loaded_op.regions[0].blocks[0]

        arg = loaded.values[entry_block.arg_ids[0]]
        assert arg.is_block_arg
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 0
        assert arg.def_result_index == 0
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=0)]

        result = loaded.values[entry_block.ops[0].results[0]]
        assert result.def_op_index == 0
        assert result.def_block_index == 0
        assert result.def_result_index == 0
        assert result.uses == [Use(user_op_index=1, operand_index=0, block_index=0)]

    def test_tied_result(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        module = Module(name="test")
        tile = module.add_value(Value(name="tile", type=tile_t))
        tensor = module.add_value(Value(name="tensor", type=tensor_t))
        result = module.add_value(Value(name="r", type=tensor_t))
        update = Operation(
            name="test.update",
            operands=[tile, tensor],
            results=[result],
            tied_results=[TiedResult(result_index=0, operand_index=1)],
        )
        yield_op = Operation(name="test.yield", operands=[result])
        block = Block(arg_ids=[tile, tensor], ops=[update, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        update_op = loaded_op.regions[0].blocks[0].ops[0]
        assert len(update_op.tied_results) == 1
        assert update_op.tied_results[0].result_index == 0
        assert update_op.tied_results[0].operand_index == 1

    def test_nested_region(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        e = module.add_value(Value(name="e", type=F32))
        inner_neg = Operation(name="test.neg", operands=[e], results=[e])
        inner_yield = Operation(name="test.yield", operands=[e])
        inner_block = Block(arg_ids=[e], ops=[inner_neg, inner_yield])
        inner_region = Region(blocks=[inner_block])
        r = module.add_value(Value(name="r", type=F32))
        map_op = Operation(
            name="test.map", operands=[x], results=[r], regions=[inner_region]
        )
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[x], ops=[map_op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        map_loaded = loaded_op.regions[0].blocks[0].ops[0]
        assert len(map_loaded.regions) == 1
        assert len(map_loaded.regions[0].blocks) == 1
        assert len(map_loaded.regions[0].blocks[0].ops) == 2

    def test_variadic_operands_no_results(self) -> None:
        module = Module(name="test")
        a = module.add_value(Value(name="a", type=F32))
        b = module.add_value(Value(name="b", type=I32))
        c = module.add_value(Value(name="c", type=INDEX))
        yield_op = Operation(name="test.yield", operands=[a, b, c])
        block = Block(arg_ids=[a, b, c], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        yield_loaded = loaded_op.regions[0].blocks[0].ops[0]
        assert len(yield_loaded.operands) == 3
        assert len(yield_loaded.results) == 0

    def test_multiple_blocks(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        yield1 = Operation(name="test.yield", operands=[x])
        block1 = Block(arg_ids=[x], ops=[yield1])
        y = module.add_value(Value(name="y", type=F32))
        yield2 = Operation(name="test.yield", operands=[y])
        block2 = Block(label="bb1", arg_ids=[y], ops=[yield2])
        body = Region(blocks=[block1, block2])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        assert len(loaded_op.regions[0].blocks) == 2

    def test_value_names_preserved(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="weights", type=F32))
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        arg_id = loaded_op.regions[0].blocks[0].arg_ids[0]
        assert loaded.values[arg_id].name == "weights"

    def test_result_names_preserved(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="negated", type=F32))
        neg = Operation(name="test.neg", operands=[x], results=[r])
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[x], ops=[neg, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        neg_loaded = loaded_op.regions[0].blocks[0].ops[0]
        result_id = neg_loaded.results[0]
        assert loaded.values[result_id].name == "negated"


# ============================================================================
# Encoding round-trips
# ============================================================================


class TestEncodingRoundTrips:
    def test_single_encoding(self) -> None:
        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        _make_func(module, "f", [t])
        loaded = _roundtrip(module)
        assert len(loaded.encodings) >= 1
        assert loaded.encodings[0].name == "q8_0"
        assert loaded.encodings[0].params == (("block", 32),)

    def test_multiple_encodings(self) -> None:
        enc1 = EncodingInstance(name="q8_0")
        enc2 = EncodingInstance(name="q6_k")
        module = Module(name="test")
        module.add_encoding(enc1)
        module.add_encoding(enc2)
        t1 = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc1)
        t2 = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc2)
        _make_func(module, "f", [t1, t2])
        loaded = _roundtrip(module)
        names = {e.name for e in loaded.encodings}
        assert "q8_0" in names
        assert "q6_k" in names


# ============================================================================
# Cross-format round-trip
# ============================================================================


class TestCrossFormat:
    def test_text_to_bytecode_and_back(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.text.parser import Parser

        text = (
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )
        from loom.dialect.func import ALL_FUNC_OPS

        parser = Parser()
        parser.register_ops(list(ALL_FUNC_OPS) + list(ALL_TEST_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(text)

        bc_data = write_module(module)
        loaded = read_module(bc_data)

        assert len(loaded.symbols) == 1
        op = loaded.symbols[0].op
        assert op is not None
        assert op.attributes.get("callee") == "negate"
        assert op.regions
        assert len(op.regions[0].blocks[0].ops) == 2


# ============================================================================
# Import/export round-trips (reader-focused)
# ============================================================================


class TestImportRoundTrips:
    """Reader correctly reconstructs import metadata from bytecode."""

    def test_import_source_module_preserved(self) -> None:
        module = Module(name="test")
        arg_vid = module.add_value(Value(name="", type=I32))
        result_vid = module.add_value(Value(name="", type=I32))
        op = Operation(
            name="func.decl",
            operands=[arg_vid],
            results=[result_vid],
            attributes={"callee": "f"},
        )
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=op,
                source_module="other_module",
            )
        )
        loaded = _roundtrip(module)
        assert loaded.symbols[0].source_module == "other_module"

    def test_import_source_symbol_preserved(self) -> None:
        module = Module(name="test")
        arg_vid = module.add_value(Value(name="", type=F32))
        op = Operation(
            name="func.decl",
            operands=[arg_vid],
            attributes={"callee": "local_name"},
        )
        module.add_symbol(
            Symbol(
                name="local_name",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=op,
                source_module="lib",
                source_symbol="original_name",
            )
        )
        loaded = _roundtrip(module)
        sym = loaded.symbols[0]
        assert sym.name == "local_name"
        assert sym.source_module == "lib"
        assert sym.source_symbol == "original_name"

    def test_non_import_has_empty_source(self) -> None:
        module = Module(name="test")
        _make_func(module, "f", [F32])
        loaded = _roundtrip(module)
        sym = loaded.symbols[0]
        assert sym.source_module == ""
        assert sym.source_symbol == ""
        assert not sym.is_import

    def test_import_with_full_signature(self) -> None:
        """Import carries full type information for linker verification."""
        tile_t = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        module = Module(name="test")
        dim_vid = module.add_value(Value(name="M", type=INDEX))
        arg0_vid = module.add_value(
            Value(name="", type=tile_t, dim_bindings={0: dim_vid})
        )
        arg1_vid = module.add_value(Value(name="", type=I32))
        result_vid = module.add_value(
            Value(name="", type=tile_t, dim_bindings={0: dim_vid})
        )
        op = Operation(
            name="func.decl",
            operands=[dim_vid, arg0_vid, arg1_vid],
            results=[result_vid],
            attributes={"callee": "transform"},
        )
        module.add_symbol(
            Symbol(
                name="transform",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=op,
                source_module="transforms",
            )
        )
        loaded = _roundtrip(module)
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.source_module == "transforms"
        loaded_op = sym.op
        assert loaded_op is not None
        # func.decl: args as operands.
        assert len(loaded_op.operands) == 3
        assert loaded.values[loaded_op.operands[1]].type == tile_t

    def test_public_import_flags_both_survive(self) -> None:
        """Both PUBLIC and IMPORT flags survive round-trip."""
        module = Module(name="test")
        arg_vid = module.add_value(Value(name="", type=F32))
        op = Operation(
            name="func.decl",
            operands=[arg_vid],
            attributes={"callee": "reexported", "visibility": "public"},
        )
        module.add_symbol(
            Symbol(
                name="reexported",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_PUBLIC | SYMBOL_FLAG_IMPORT,
                op=op,
                source_module="upstream",
            )
        )
        loaded = _roundtrip(module)
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.is_public
        assert sym.source_module == "upstream"


# ============================================================================
# Dict attribute round-trips
# ============================================================================


class TestDictAttributeRoundTrips:
    def test_dict_with_int_values(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"axis": 0, "count": 42}},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        d = loaded_op.attributes.get("dict")
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("count", 42)]
        assert d["axis"] == 0
        assert d["count"] == 42

    def test_dict_with_string_values(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"label": "hello", "tag": "world"}},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        d = loaded_op.attributes.get("dict")
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("label", "hello"), ("tag", "world")]
        assert d["label"] == "hello"
        assert d["tag"] == "world"

    def test_dict_with_mixed_values(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"axis": 0, "label": "foo", "enabled": True}},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        d = loaded_op.attributes.get("dict")
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("enabled", True), ("label", "foo")]
        assert d["axis"] == 0
        assert d["label"] == "foo"
        assert d["enabled"] is True

    def test_empty_dict(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {}},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        d = loaded_op.attributes.get("dict")
        assert d is None or d == CanonicalAttrDict()

    def test_nested_dict_round_trip_is_canonical(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"meta": {"phase": "link", "opt": 3}, "axis": 0}},
        )
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        d = loaded_op.attributes["dict"]
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [
            ("axis", 0),
            ("meta", {"opt": 3, "phase": "link"}),
        ]
        assert isinstance(d["meta"], CanonicalAttrDict)
        assert list(d["meta"].items()) == [("opt", 3), ("phase", "link")]


class TestMalformedDictAttributeWireOrder:
    def _read_dict_value(
        self, strings: list[str], data: bytes
    ) -> tuple[CanonicalAttrDict, int]:
        reader = BytecodeReader(b"")
        reader._strings = strings
        value, offset = reader._read_attr_value(data, 0)
        assert isinstance(value, CanonicalAttrDict)
        return value, offset

    def test_unsorted_dict_keys_are_rejected(self) -> None:
        # dict<z = 0, a = 1>
        data = bytes([9, 2, 1, 0, 0, 2, 0, 2])
        with pytest.raises(BytecodeError, match="not in canonical order"):
            self._read_dict_value(["", "z", "a"], data)

    def test_duplicate_dict_keys_are_rejected(self) -> None:
        # dict<axis = 0, axis = 1>
        data = bytes([9, 2, 1, 0, 0, 1, 0, 2])
        with pytest.raises(BytecodeError, match="duplicate dict attr key"):
            self._read_dict_value(["", "axis"], data)

    def test_unsorted_nested_dict_keys_are_rejected(self) -> None:
        # dict<meta = dict<z = 0, a = 1>>
        data = bytes([9, 1, 1, 9, 2, 2, 0, 0, 3, 0, 2])
        with pytest.raises(BytecodeError, match="not in canonical order"):
            self._read_dict_value(["", "meta", "z", "a"], data)

    def test_duplicate_nested_dict_keys_are_rejected(self) -> None:
        # dict<meta = dict<axis = 0, axis = 1>>
        data = bytes([9, 1, 1, 9, 2, 2, 0, 0, 2, 0, 2])
        with pytest.raises(BytecodeError, match="duplicate dict attr key"):
            self._read_dict_value(["", "meta", "axis"], data)


# ============================================================================
# Location bytecode round-trips
# ============================================================================


class TestLocationRoundTrips:
    def test_file_location_survives(self) -> None:
        from loom.ir import FileLocation

        module = Module(name="test")
        loc_id = module.locations.add(
            FileLocation(
                source_id=0, start_line=42, start_col=3, end_line=42, end_col=58
            )
        )
        module.sources.append("model.loom")
        x = module.add_value(Value(name="x", type=F32))
        yield_op = Operation(
            name="test.yield",
            operands=[x],
            location_id=loc_id,
        )
        block = Block(arg_ids=[x], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        assert loaded_op.location_id != 0
        loc = loaded.locations.get(loaded_op.location_id)
        assert isinstance(loc, FileLocation)
        assert loc.start_line == 42
        assert loc.start_col == 3

    def test_opaque_location_survives(self) -> None:
        from loom.ir import OpaqueLocation

        module = Module(name="test")
        loc_id = module.locations.add(OpaqueLocation(source_id=0, data=b"node_id=42"))
        module.sources.append("torch")
        x = module.add_value(Value(name="x", type=F32))
        yield_op = Operation(
            name="test.yield",
            operands=[x],
            location_id=loc_id,
        )
        block = Block(arg_ids=[x], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        assert loaded_op.location_id != 0
        loc = loaded.locations.get(loaded_op.location_id)
        assert isinstance(loc, OpaqueLocation)
        assert loc.data == b"node_id=42"

    def test_tagged_location_survives(self) -> None:
        module = Module(name="test")
        module.sources.append("model.loom")
        child_id = module.locations.add(
            FileLocation(source_id=0, start_line=5, start_col=6, end_line=5, end_col=6)
        )
        loc_id = module.locations.add(
            TaggedLocation(
                tag=LOCATION_TAG_SANITIZER_SITE,
                child=child_id,
                data=b"\x01\x2a\xff",
            )
        )
        x = module.add_value(Value(name="x", type=F32))
        yield_op = Operation(
            name="test.yield",
            operands=[x],
            location_id=loc_id,
        )
        block = Block(arg_ids=[x], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = _roundtrip(module)
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        assert loaded_sym_op.regions
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        assert loaded_op.location_id != 0
        loc = loaded.locations.get(loaded_op.location_id)
        assert isinstance(loc, TaggedLocation)
        assert loc.tag == LOCATION_TAG_SANITIZER_SITE
        assert loc.child == child_id
        assert loc.data == b"\x01\x2a\xff"
