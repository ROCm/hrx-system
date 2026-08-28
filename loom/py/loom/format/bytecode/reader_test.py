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

from loom.dialect.test.defs import (
    ALL_TEST_TYPES,
    test_enum_array_attrs,
    test_options_attr,
    test_parameterized_attr,
    test_parameterized_attr_array,
    test_signed_enum_set_attrs,
    test_symbol_array_attrs,
    test_symbol_set_attrs,
)
from loom.format.bytecode.encoding import (
    decode_varint,
    encode_signed_varint,
    encode_varint,
)
from loom.format.bytecode.reader import BytecodeError, BytecodeReader, read_module
from loom.format.bytecode.writer import (
    BYTECODE_TYPE_KIND_BY_IR_KIND,
    LOCATION_MODE_FULL_LOCATIONS,
    LOCATION_MODE_NO_LOCATIONS,
    SECTION_ENCODINGS,
    SECTION_IR,
    SECTION_LOCATIONS,
    SECTION_SOURCE_TRIVIA,
    SECTION_SYMBOL_REFERENCES,
    SECTION_SYMBOLS,
    SOURCE_TRIVIA_LEADING_BLANK_LINE,
    SYMBOL_FLAG_EXPORT,
    write_module,
)
from loom.ir import (
    ATTR_AGGREGATE_MAX_NESTING_DEPTH,
    BF16,
    BUFFER_TYPE,
    F32,
    I8,
    I32,
    I64,
    INDEX,
    LOCATION_TAG_SANITIZER_SITE,
    OFFSET,
    REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM,
    SYMBOL_FLAG_IMPORT,
    SYMBOL_FLAG_PUBLIC,
    VALUE_DEF_OP_NONE,
    Block,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EnumArrayAttr,
    FileLocation,
    FunctionType,
    Module,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    Predicate,
    PredicateArg,
    Region,
    RegisterType,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    SignedEnumSetAttr,
    StaticDim,
    StorageSpace,
    StorageType,
    Symbol,
    SymbolKind,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    TaggedLocation,
    TiedResult,
    Type,
    TypeKind,
    U64Attr,
    Use,
    Value,
)
from loom.stable_id import stable_id_from_string
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET

_SYMBOL_FLAG_PREDICATES = 1 << 6

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


def _test_ptr_register_type(
    unit_count: int = 1, value_type: Type | None = None
) -> RegisterType:
    return RegisterType(
        _TEST_LOW_CORE_STABLE_ID,
        _TEST_PTR_REGISTER_CLASS_ID,
        unit_count,
        "test.ptr",
        value_type,
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


def _root_region_source_flags_offset(data: bytes | bytearray) -> int:
    """Return the absolute offset of the first root region's source flags."""
    module_offset, _module_length = _module_range(data)
    _entry_offset, section_offset, _section_length = _find_section_entry(
        data, SECTION_IR
    )
    body_offset = module_offset + section_offset
    offset = body_offset
    for _ in range(4):
        _count, offset = decode_varint(data, offset)
    return offset


def _single_op_offset(data: bytes | bytearray) -> int:
    """Return the absolute offset of the first op in _make_single_op_body_module."""
    offset = _root_region_source_flags_offset(data)
    _source_flags, offset = decode_varint(data, offset)
    _block_count, offset = decode_varint(data, offset)
    offset += 1  # block has_label byte.
    source_trivia, offset = decode_varint(data, offset)
    comment_count = source_trivia >> 1
    for _ in range(comment_count):
        comment_length, offset = decode_varint(data, offset)
        offset += comment_length
    _arg_count, offset = decode_varint(data, offset)
    _op_count, offset = decode_varint(data, offset)
    return offset


def _root_block_source_trivia_offset(data: bytes | bytearray) -> int:
    """Return the first root block's source-trivia scalar offset."""
    offset = _root_region_source_flags_offset(data)
    _source_flags, offset = decode_varint(data, offset)
    _block_count, offset = decode_varint(data, offset)
    has_label = data[offset]
    offset += 1
    if has_label:
        _label_id, offset = decode_varint(data, offset)
    return offset


def _first_symbol_flags_offset(data: bytes | bytearray) -> int:
    """Return the absolute flags offset of the first symbol entry."""
    module_offset, _module_length = _module_range(data)
    _entry_offset, section_offset, _section_length = _find_section_entry(
        data, SECTION_SYMBOLS
    )
    offset = module_offset + section_offset
    symbol_count, offset = decode_varint(data, offset)
    assert symbol_count > 0
    import_count, offset = decode_varint(data, offset)
    export_count, offset = decode_varint(data, offset)
    _root_region_payload_count, offset = decode_varint(data, offset)
    offset += (import_count + export_count) * 8
    _name_id, offset = decode_varint(data, offset)
    return offset + 2


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


def _make_predicate_function_module() -> Module:
    """Return a declaration carrying one function predicate."""
    module = Module(name="test")
    argument_id = module.add_value(Value(name="M", type=INDEX))
    predicate = Predicate(
        kind="mul",
        args=(
            PredicateArg(tag="value", value="M"),
            PredicateArg(tag="const", value=16),
        ),
    )
    operation = Operation(
        name="func.decl",
        operands=[argument_id],
        attributes={"callee": "f", "predicates": [predicate]},
    )
    module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DECL, op=operation))
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

    def test_file_header_leading_blank_line_is_rejected(self) -> None:
        module = Module(name="test", file_header=("File overview.",))
        data = bytearray(write_module(module))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _length = _find_section_entry(
            data, SECTION_SOURCE_TRIVIA
        )
        payload_offset = module_offset + section_offset
        assert data[payload_offset] == 2
        data[payload_offset] |= SOURCE_TRIVIA_LEADING_BLANK_LINE

        with pytest.raises(BytecodeError, match="leading blank line"):
            read_module(bytes(data))

    def test_invalid_utf8_file_header_is_rejected(self) -> None:
        module = Module(name="test", file_header=("File overview.",))
        data = bytearray(write_module(module))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _length = _find_section_entry(
            data, SECTION_SOURCE_TRIVIA
        )
        payload_offset = module_offset + section_offset
        _source_trivia, payload_offset = decode_varint(data, payload_offset)
        _comment_length, payload_offset = decode_varint(data, payload_offset)
        assert data[payload_offset] == ord(" ")
        data[payload_offset + 1] = 0xFF

        with pytest.raises(BytecodeError, match="not valid UTF-8"):
            read_module(bytes(data))

    def test_file_header_trailing_bytes_are_rejected(self) -> None:
        module = Module(name="test", file_header=("File overview.",))
        data = bytearray(write_module(module))
        module_offset, module_length = _module_range(data)
        entry_offset, section_offset, section_length = _find_section_entry(
            data, SECTION_SOURCE_TRIVIA
        )
        assert module_offset + section_offset + section_length == len(data)
        data.append(0)
        struct.pack_into("<Q", data, entry_offset + 16, section_length + 1)
        producer_end = data.index(0, 16)
        directory_offset = (producer_end + 1 + 7) & ~7
        struct.pack_into("<Q", data, directory_offset + 16, module_length + 1)

        with pytest.raises(BytecodeError, match="trailing bytes"):
            read_module(bytes(data))


class TestMalformedSymbolReferences:
    def test_declared_records_must_fit_section_payload(self) -> None:
        module = Module(name="test")
        _make_func(module, "f", [])
        data = bytearray(write_module(module))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _section_length = _find_section_entry(
            data, SECTION_SYMBOL_REFERENCES
        )
        offset = module_offset + section_offset
        symbol_count, offset = decode_varint(data, offset)
        assert symbol_count == 1
        assert data[offset] == 0
        data[offset] = 1

        with pytest.raises(BytecodeError, match="declared records exceed"):
            read_module(bytes(data))

    def test_dependency_source_root_must_belong_to_owning_symbol(self) -> None:
        module = Module(name="test")
        _make_func(module, "target", [], is_declaration=True)
        _make_func(
            module,
            "caller",
            [],
            ops=[
                Operation(
                    name="func.call",
                    attributes={"callee": SymbolName("target")},
                ),
                Operation(name="test.yield"),
            ],
        )
        data = bytearray(write_module(module))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _section_length = _find_section_entry(
            data, SECTION_SYMBOL_REFERENCES
        )
        offset = module_offset + section_offset
        symbol_count, offset = decode_varint(data, offset)
        dependency_count, offset = decode_varint(data, offset)
        template_demand_count, offset = decode_varint(data, offset)
        module_dependency_count, offset = decode_varint(data, offset)
        assert (symbol_count, dependency_count, template_demand_count) == (2, 1, 0)
        assert module_dependency_count == 0
        target_dependency_count, offset = decode_varint(data, offset)
        target_demand_count, offset = decode_varint(data, offset)
        caller_dependency_count, offset = decode_varint(data, offset)
        assert (target_dependency_count, target_demand_count) == (0, 0)
        assert caller_dependency_count == 1
        assert data[offset] == 1
        data[offset] = 2

        with pytest.raises(BytecodeError, match="source root region index"):
            read_module(bytes(data))

    def test_dependency_target_interfaces_must_be_known(self) -> None:
        module = Module(name="test")
        _make_func(module, "target", [], is_declaration=True)
        _make_func(
            module,
            "caller",
            [],
            ops=[
                Operation(
                    name="func.call",
                    attributes={"callee": SymbolName("target")},
                ),
                Operation(name="test.yield"),
            ],
        )
        data = bytearray(write_module(module))
        module_offset, _module_length = _module_range(data)
        _entry_offset, section_offset, _section_length = _find_section_entry(
            data, SECTION_SYMBOL_REFERENCES
        )
        offset = module_offset + section_offset
        symbol_count, offset = decode_varint(data, offset)
        dependency_count, offset = decode_varint(data, offset)
        template_demand_count, offset = decode_varint(data, offset)
        module_dependency_count, offset = decode_varint(data, offset)
        assert (symbol_count, dependency_count, template_demand_count) == (2, 1, 0)
        assert module_dependency_count == 0
        target_dependency_count, offset = decode_varint(data, offset)
        target_demand_count, offset = decode_varint(data, offset)
        caller_dependency_count, offset = decode_varint(data, offset)
        assert (target_dependency_count, target_demand_count) == (0, 0)
        assert caller_dependency_count == 1
        _source_root, offset = decode_varint(data, offset)
        _target_symbol, offset = decode_varint(data, offset)
        assert data[offset : offset + 2] == b"\x80\x02"
        data[offset + 1] = 0x40

        with pytest.raises(BytecodeError, match="target interfaces"):
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


class TestMalformedSymbolSection:
    def test_predicates_flag_on_nonfunction_symbol_is_rejected(self) -> None:
        data = bytearray(write_module(_make_single_op_body_module()))
        flags_offset = _first_symbol_flags_offset(data)
        data[flags_offset - 2] = SymbolKind.GLOBAL.value
        flags = struct.unpack_from("<H", data, flags_offset)[0]
        struct.pack_into("<H", data, flags_offset, flags | _SYMBOL_FLAG_PREDICATES)

        with pytest.raises(BytecodeError, match="requires a function symbol"):
            read_module(bytes(data))

    def test_public_definition_without_export_flag_is_rejected(self) -> None:
        module = Module(name="test")
        _make_func(module, "f", [F32], is_public=True)
        data = bytearray(write_module(module))
        flags_offset = _first_symbol_flags_offset(data)
        flags = struct.unpack_from("<H", data, flags_offset)[0]
        assert flags & SYMBOL_FLAG_EXPORT
        struct.pack_into("<H", data, flags_offset, flags & ~SYMBOL_FLAG_EXPORT)

        with pytest.raises(BytecodeError, match="requires export flag"):
            read_module(bytes(data))

    def test_imported_symbol_with_export_flag_is_rejected(self) -> None:
        module = Module(name="test")
        argument_id = module.add_value(Value(name="", type=F32))
        function_op = Operation(
            name="func.decl",
            operands=[argument_id],
            attributes={"callee": "f"},
        )
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=function_op,
                source_module="upstream",
            )
        )
        data = bytearray(write_module(module))
        flags_offset = _first_symbol_flags_offset(data)
        flags = struct.unpack_from("<H", data, flags_offset)[0]
        assert not flags & SYMBOL_FLAG_EXPORT
        struct.pack_into("<H", data, flags_offset, flags | SYMBOL_FLAG_EXPORT)

        with pytest.raises(BytecodeError, match="both imported and exported"):
            read_module(bytes(data))

    def test_nonempty_predicates_without_flag_are_rejected(self) -> None:
        data = bytearray(write_module(_make_predicate_function_module()))
        flags_offset = _first_symbol_flags_offset(data)
        flags = struct.unpack_from("<H", data, flags_offset)[0]
        assert flags & _SYMBOL_FLAG_PREDICATES
        struct.pack_into("<H", data, flags_offset, flags & ~_SYMBOL_FLAG_PREDICATES)

        with pytest.raises(BytecodeError, match="nonempty function predicate"):
            read_module(bytes(data))


class TestMalformedIrSection:
    def test_unsupported_region_source_flags_are_rejected(self) -> None:
        data = bytearray(write_module(_make_single_op_body_module()))
        source_flags_offset = _root_region_source_flags_offset(data)
        assert data[source_flags_offset] == 0
        data[source_flags_offset] = 1 << 1

        with pytest.raises(BytecodeError, match="unsupported source flag bits"):
            read_module(bytes(data))

    def test_source_trivia_comment_count_beyond_field_width_is_rejected(
        self,
    ) -> None:
        module = _make_single_op_body_module()
        func_op = module.symbols[0].op
        assert func_op is not None
        func_op.regions[0].blocks[0].comments = ("x",)
        data = bytearray(write_module(module))
        source_trivia_offset = _root_block_source_trivia_offset(data)
        assert data[source_trivia_offset : source_trivia_offset + 3] == b"\x02\x02 "
        data[source_trivia_offset : source_trivia_offset + 3] = b"\x80\x80\x08"

        with pytest.raises(BytecodeError, match="comment count exceeds UINT16_MAX"):
            read_module(bytes(data))

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

        with pytest.raises(BytecodeError, match="root region allocation summary"):
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
        *,
        strings: list[str] | None = None,
        type_defs: tuple[object, ...] | None = None,
    ) -> list[Type]:
        reader = BytecodeReader(b"", type_defs=type_defs)
        reader._encodings = encodings or []
        reader._strings = strings or []
        reader._read_types_section((0, data))
        return reader._types

    def test_unassigned_kind_is_rejected(self) -> None:
        with pytest.raises(BytecodeError, match="unknown type kind: 4"):
            self._read_types(bytes([1, 4]))

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

    def test_invalid_register_value_type_presence_is_rejected(self) -> None:
        data = b"".join(
            (
                bytes(
                    [
                        2,  # type count
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.SCALAR],
                        I32.kind.value,
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER],
                        1,  # descriptor-set stable ID
                    ]
                ),
                encode_varint(1 << 16),  # class 0, one unit
                bytes([2]),  # invalid has_value_type
            )
        )
        with pytest.raises(BytecodeError, match="has_value_type must be 0 or 1"):
            self._read_types(data)

    def test_truncated_register_value_type_presence_is_rejected(self) -> None:
        data = b"".join(
            (
                bytes(
                    [
                        1,  # type count
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER],
                        1,  # descriptor-set stable ID
                    ]
                ),
                encode_varint(1 << 16),  # class 0, one unit
            )
        )
        with pytest.raises(BytecodeError, match="presence is truncated"):
            self._read_types(data)

    def test_truncated_register_value_type_reference_is_rejected(self) -> None:
        data = b"".join(
            (
                bytes(
                    [
                        1,  # type count
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER],
                        1,  # descriptor-set stable ID
                    ]
                ),
                encode_varint(1 << 16),  # class 0, one unit
                bytes([1]),  # value type present, but reference omitted
            )
        )
        with pytest.raises(BytecodeError, match="value type reference"):
            self._read_types(data)

    def test_forward_register_value_type_reference_is_rejected(self) -> None:
        data = b"".join(
            (
                bytes(
                    [
                        2,  # type count
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.SCALAR],
                        I32.kind.value,
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER],
                        1,  # descriptor-set stable ID
                    ]
                ),
                encode_varint(1 << 16),  # class 0, one unit
                bytes([1, 1]),  # present, self-reference instead of prior type
            )
        )
        with pytest.raises(BytecodeError, match="must refer to a prior type"):
            self._read_types(data)

    def test_register_payload_reserved_bits_are_rejected(self) -> None:
        data = b"".join(
            (
                bytes(
                    [
                        1,  # type count
                        BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.REGISTER],
                        1,  # descriptor-set stable ID
                    ]
                ),
                encode_varint((1 << 48) | (1 << 16)),
                bytes([0]),  # no value type
            )
        )
        with pytest.raises(BytecodeError, match="reserved bits"):
            self._read_types(data)

    def test_unknown_parameterized_type_family_is_rejected(self) -> None:
        data = bytes(
            [
                1,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED],
                0,  # family string id
                0,  # present parameter count
            ]
        )
        with pytest.raises(BytecodeError, match="is not registered"):
            self._read_types(
                data,
                strings=["test.unknown"],
                type_defs=ALL_TEST_TYPES,
            )

    def test_unknown_parameterized_type_parameter_is_rejected(self) -> None:
        data = bytes(
            [
                1,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED],
                0,  # test.scope family string id
                1,  # present parameter count
                1,  # unknown parameter string id
                4,  # enum attribute kind
                2,  # subgroup
            ]
        )
        with pytest.raises(BytecodeError, match="unknown parameter 'bogus'"):
            self._read_types(
                data,
                strings=["test.scope", "bogus"],
                type_defs=ALL_TEST_TYPES,
            )

    def test_parameterized_type_parameters_must_be_in_order(self) -> None:
        data = bytes(
            [
                2,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.SCALAR],
                BF16.kind.value,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED],
                0,  # test.matrix family string id
                2,  # present parameter count
                2,  # rows parameter string id
                0,  # i64 attribute kind
                32,  # zigzag-encoded 16
                1,  # scope parameter string id, out of order
                4,  # enum attribute kind
                2,  # subgroup
            ]
        )
        with pytest.raises(BytecodeError, match="not in declaration order"):
            self._read_types(
                data,
                strings=["test.matrix", "scope", "rows"],
                type_defs=ALL_TEST_TYPES,
            )

    def test_parameterized_type_requires_mandatory_parameters(self) -> None:
        data = bytes(
            [
                1,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED],
                0,  # test.scope family string id
                0,  # present parameter count
            ]
        )
        with pytest.raises(BytecodeError, match="required parameter 'scope'"):
            self._read_types(
                data,
                strings=["test.scope"],
                type_defs=ALL_TEST_TYPES,
            )

    def test_parameterized_type_rejects_forward_type_parameter(self) -> None:
        data = bytes(
            [
                1,
                BYTECODE_TYPE_KIND_BY_IR_KIND[TypeKind.PARAMETERIZED],
                0,  # test.array family string id
                1,  # present parameter count
                1,  # element_type parameter string id
                7,  # type attribute kind
                0,  # self-reference instead of a prior type
            ]
        )
        with pytest.raises(BytecodeError, match="must refer to a prior type"):
            self._read_types(
                data,
                strings=["test.array", "element_type"],
                type_defs=ALL_TEST_TYPES,
            )


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
        t = DialectType("test.ref", (DialectType("hal.buffer"),))
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, DialectType)
        assert loaded.name == "test.ref"
        assert isinstance(loaded.params[0], DialectType)
        assert loaded.params[0].name == "hal.buffer"

    def test_dialect_nested(self) -> None:
        inner = DialectType("test.list", (I32,))
        t = DialectType("test.ref", (inner,))
        loaded = self._roundtrip_type(t)
        assert isinstance(loaded, DialectType)
        assert loaded.name == "test.ref"
        assert isinstance(loaded.params[0], DialectType)
        assert loaded.params[0].name == "test.list"
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

    def test_register_scalar_value_type(self) -> None:
        t = _test_ptr_register_type(value_type=I32)
        assert self._roundtrip_type(t) == t

    def test_register_vector_value_type(self) -> None:
        vector_type = ShapedType(TypeKind.VECTOR, I32, (StaticDim(4),))
        t = _test_ptr_register_type(4, vector_type)
        assert self._roundtrip_type(t) == t

    def test_register_dialect_value_type(self) -> None:
        dialect_type = DialectType("test.ref", (I32,))
        t = _test_ptr_register_type(value_type=dialect_type)
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

    def test_u64_max(self) -> None:
        value = U64Attr(2**64 - 1)
        assert self._roundtrip_attr("v", value) == value

    def test_plain_integer_outside_i64_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="use U64Attr"):
            self._roundtrip_attr("v", 2**63)

    def test_mixed_generic_array_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="signed 64-bit integer elements"):
            self._roundtrip_attr("v", [1, "two"])

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
    def test_region_source_flags_roundtrip(self) -> None:
        module = _make_single_op_body_module()
        func_op = module.symbols[0].op
        assert func_op is not None
        func_op.regions[0].source_flags = REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM

        loaded = _roundtrip(module)
        loaded_func_op = loaded.symbols[0].op
        assert loaded_func_op is not None
        assert (
            loaded_func_op.regions[0].source_flags
            == REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM
        )

    def test_source_trivia_roundtrip(self) -> None:
        module = _make_single_op_body_module()
        module.file_header = ("File-level overview.", "Second header line.")
        func_op = module.symbols[0].op
        assert func_op is not None
        block = func_op.regions[0].blocks[0]
        body_op = block.ops[0]
        func_op.leading_blank_line = True
        func_op.comments = ("symbol",)
        block.leading_blank_line = True
        block.comments = ("block",)
        body_op.leading_blank_line = True
        body_op.comments = ("operation",)

        loaded = _roundtrip(module)
        assert loaded.file_header == module.file_header
        loaded_func_op = loaded.symbols[0].op
        assert loaded_func_op is not None
        loaded_block = loaded_func_op.regions[0].blocks[0]
        loaded_body_op = loaded_block.ops[0]
        assert loaded_func_op.leading_blank_line
        assert loaded_func_op.comments == ("symbol",)
        assert loaded_block.leading_blank_line
        assert loaded_block.comments == ("block",)
        assert loaded_body_op.leading_blank_line
        assert loaded_body_op.comments == ("operation",)

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

    def test_qualified_encoding_name(self) -> None:
        encoding = EncodingInstance(name="ggml.q4_k")
        module = Module(name="test")
        module.add_encoding(encoding)
        shaped_type = ShapedType(
            TypeKind.TILE,
            I8,
            (StaticDim(256),),
            encoding=encoding,
        )
        _make_func(module, "f", [shaped_type])

        loaded = _roundtrip(module)

        assert loaded.encodings[0].name == "ggml.q4_k"


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

    def test_kernel_workload_and_launch_signatures_roundtrip(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.kernel import ALL_KERNEL_OPS
        from loom.format.text.parser import Parser

        text = (
            "kernel.def @dynamic(%grid: index) "
            "where [range(%grid, 1, 65535)] {\n"
            "  kernel.launch.config workgroups(%grid, %grid, %grid) "
            "workgroup_size(%grid, %grid, %grid) : index\n"
            "} launch(%count: index, %output: buffer) "
            "where [mul(%count, 4)] {\n"
            "  kernel.return\n"
            "}\n"
            "kernel.decl @external(%workload: index) "
            "where [range(%workload, 1, 65535)] "
            "launch(%abi_count: index, %output: buffer) "
            "where [mul(%abi_count, 4)]\n"
        )
        parser = Parser()
        parser.register_ops(ALL_KERNEL_OPS)
        parser.register_types(ALL_BUILTIN_TYPES)
        module = parser.parse(text)

        bytecode = write_module(module)
        loaded = read_module(bytecode)
        assert write_module(loaded) == bytecode

        definition = loaded.symbols[0].op
        assert definition is not None
        assert "workload_predicates" in definition.attributes
        assert "predicates" in definition.attributes
        assert [len(region.blocks[0].arg_ids) for region in definition.regions] == [
            1,
            2,
        ]
        declaration = loaded.symbols[1].op
        assert declaration is not None
        assert "workload_predicates" in declaration.attributes
        assert "predicates" in declaration.attributes
        assert declaration.operand_segment_counts == (1, 2)
        assert len(declaration.operands) == 3


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


class TestEnumArrayAttributeWireFormat:
    def _read_enum_array(self, data: bytes) -> tuple[EnumArrayAttr, int]:
        reader = BytecodeReader(b"", op_decls=[test_enum_array_attrs])
        attr_def = reader._attr_def_for_op_attr(
            "test.enum_array_attrs", "required_values"
        )
        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)
        assert isinstance(value, EnumArrayAttr)
        return value, offset

    def test_reads_ordered_duplicate_values(self) -> None:
        value, offset = self._read_enum_array(bytes([13, 3, 1, 255, 1]))

        assert value == EnumArrayAttr([1, 255, 1])
        assert offset == 5

    def test_rejects_oversized_array(self) -> None:
        data = bytes([13]) + encode_varint(0x10000)

        with pytest.raises(BytecodeError, match="exceeds UINT16_MAX"):
            self._read_enum_array(data)

    def test_rejects_truncated_array(self) -> None:
        with pytest.raises(BytecodeError, match="exceeds payload size"):
            self._read_enum_array(bytes([13, 3, 1]))

    def test_rejects_array_without_field_descriptor(self) -> None:
        reader = BytecodeReader(b"")

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([13, 0]), 0)

    def test_rejects_array_nested_in_generic_dict(self) -> None:
        reader = BytecodeReader(b"")
        reader._strings = ["", "values"]
        data = bytes([9, 1, 1, 13, 0])

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(data, 0)


class TestSignedEnumSetAttributeWireFormat:
    def _read_signed_enum_set(self, data: bytes) -> tuple[SignedEnumSetAttr, int]:
        reader = BytecodeReader(b"", op_decls=[test_signed_enum_set_attrs])
        attr_def = reader._attr_def_for_op_attr(
            "test.signed_enum_set_attrs", "required_features"
        )
        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)
        assert isinstance(value, SignedEnumSetAttr)
        return value, offset

    def test_reads_canonical_positive_and_negative_words(self) -> None:
        positive_words = (1 << 1, 0, 0, 1 << 63)
        negative_words = (1 << 7, 0, 0, 0)
        data = bytes([16, 4]) + struct.pack("<8Q", *positive_words, *negative_words)

        value, offset = self._read_signed_enum_set(data)

        assert value == SignedEnumSetAttr([1, 255], [7])
        assert offset == len(data)

    def test_reads_present_empty_set(self) -> None:
        value, offset = self._read_signed_enum_set(bytes([16, 0]))

        assert value == SignedEnumSetAttr()
        assert offset == 2

    def test_rejects_excess_word_count(self) -> None:
        with pytest.raises(BytecodeError, match="word count 5 exceeds"):
            self._read_signed_enum_set(bytes([16, 5]))

    def test_rejects_truncated_words(self) -> None:
        with pytest.raises(BytecodeError, match="words exceed payload size"):
            self._read_signed_enum_set(bytes([16, 1]) + struct.pack("<Q", 2))

    def test_rejects_contradictory_assertions(self) -> None:
        data = bytes([16, 1]) + struct.pack("<2Q", 1 << 1, 1 << 1)

        with pytest.raises(BytecodeError, match="contradictory assertions"):
            self._read_signed_enum_set(data)

    def test_rejects_noncanonical_trailing_zero_pair(self) -> None:
        data = bytes([16, 2]) + struct.pack("<4Q", 1 << 1, 0, 0, 0)

        with pytest.raises(BytecodeError, match="not canonically trimmed"):
            self._read_signed_enum_set(data)

    def test_rejects_undeclared_stable_value(self) -> None:
        data = bytes([16, 1]) + struct.pack("<2Q", 1 << 2, 0)

        with pytest.raises(BytecodeError, match=r"undeclared.*\[2\]"):
            self._read_signed_enum_set(data)

    def test_rejects_set_without_field_descriptor(self) -> None:
        reader = BytecodeReader(b"")

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([16, 0]), 0)

    def test_rejects_set_nested_in_generic_dict(self) -> None:
        reader = BytecodeReader(b"")
        reader._strings = ["", "features"]
        data = bytes([9, 1, 1, 16, 0])

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(data, 0)


class TestSymbolArrayAttributeWireFormat:
    def _read_symbol_array(self, data: bytes) -> tuple[SymbolNameArray, int]:
        reader = BytecodeReader(b"", op_decls=[test_symbol_array_attrs])
        reader._strings = ["", "a", "b"]
        attr_def = reader._attr_def_for_op_attr(
            "test.symbol_array_attrs", "dependencies"
        )
        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)
        assert isinstance(value, SymbolNameArray)
        return value, offset

    def test_reads_ordered_duplicate_names(self) -> None:
        value, offset = self._read_symbol_array(bytes([17, 3, 2, 1, 2]))

        assert value == SymbolNameArray(
            [SymbolName("b"), SymbolName("a"), SymbolName("b")]
        )
        assert offset == 5

    def test_rejects_oversized_array(self) -> None:
        data = bytes([17]) + encode_varint(0x10000)

        with pytest.raises(BytecodeError, match="exceeds UINT16_MAX"):
            self._read_symbol_array(data)

    def test_rejects_out_of_range_symbol_name(self) -> None:
        with pytest.raises(BytecodeError, match="string_id 3 out of range"):
            self._read_symbol_array(bytes([17, 1, 3]))

    def test_rejects_array_without_field_descriptor(self) -> None:
        reader = BytecodeReader(b"")

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([17, 0]), 0)

    def test_rejects_array_nested_in_generic_dict(self) -> None:
        reader = BytecodeReader(b"")
        reader._strings = ["", "values"]
        data = bytes([9, 1, 1, 17, 0])

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(data, 0)


class TestSymbolSetAttributeWireFormat:
    def _read_symbol_set(self, data: bytes) -> tuple[SymbolNameSet, int]:
        reader = BytecodeReader(b"", op_decls=[test_symbol_set_attrs])
        reader._strings = ["", "a", "b"]
        attr_def = reader._attr_def_for_op_attr("test.symbol_set_attrs", "symbols")
        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)
        assert isinstance(value, SymbolNameSet)
        return value, offset

    def test_reads_sorted_unique_names(self) -> None:
        value, offset = self._read_symbol_set(bytes([18, 2, 1, 2]))

        assert value == SymbolNameSet([SymbolName("a"), SymbolName("b")])
        assert offset == 4

    @pytest.mark.parametrize(
        "name_ids",
        [
            (2, 1),
            (1, 1),
        ],
    )
    def test_rejects_noncanonical_names(self, name_ids: tuple[int, int]) -> None:
        with pytest.raises(BytecodeError, match="not sorted and unique"):
            self._read_symbol_set(bytes([18, 2, *name_ids]))

    def test_rejects_out_of_range_symbol_name(self) -> None:
        with pytest.raises(BytecodeError, match="string_id 3 out of range"):
            self._read_symbol_set(bytes([18, 1, 3]))

    def test_rejects_set_without_field_descriptor(self) -> None:
        reader = BytecodeReader(b"")

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([18, 0]), 0)

    def test_rejects_set_for_symbol_array_field(self) -> None:
        reader = BytecodeReader(b"", op_decls=[test_symbol_array_attrs])
        attr_def = reader._attr_def_for_op_attr(
            "test.symbol_array_attrs", "dependencies"
        )

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([18, 0]), 0, attr_def=attr_def)


class TestParameterizedAttributeWireFormat:
    def _reader(self, strings: list[str]) -> tuple[BytecodeReader, Any]:
        reader = BytecodeReader(
            b"", op_decls=[test_parameterized_attr, test_enum_array_attrs]
        )
        reader._strings = strings
        attr_def = reader._attr_def_for_op_attr("test.parameterized_attr", "options")
        return reader, attr_def

    def test_reads_named_slots_in_declaration_order(self) -> None:
        reader, attr_def = self._reader(["", "test.options", "mode"])
        data = bytes([14, 1, 1, 2, 4, 1])

        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)

        assert isinstance(value, ParameterizedAttr)
        assert value.definition is test_options_attr
        assert value.get("mode") == 1
        assert offset == len(data)

    def test_rejects_unknown_family(self) -> None:
        reader, attr_def = self._reader(["", "test.unknown"])

        with pytest.raises(BytecodeError, match="is not registered"):
            reader._read_attr_value(bytes([14, 1, 0]), 0, attr_def=attr_def)

    def test_rejects_wrong_family_for_field(self) -> None:
        reader, attr_def = self._reader(["", "test.tile"])

        with pytest.raises(BytecodeError, match="does not match field contract"):
            reader._read_attr_value(bytes([14, 1, 0]), 0, attr_def=attr_def)

    def test_rejects_parameterized_value_for_other_field_kind(self) -> None:
        reader, _ = self._reader(["", "test.options"])
        enum_array_def = reader._attr_def_for_op_attr(
            "test.enum_array_attrs", "required_values"
        )

        with pytest.raises(BytecodeError, match="does not match the field contract"):
            reader._read_attr_value(bytes([14, 1, 0]), 0, attr_def=enum_array_def)

    def test_rejects_unknown_parameter(self) -> None:
        reader, attr_def = self._reader(["", "test.options", "unknown"])

        with pytest.raises(BytecodeError, match="unknown parameter"):
            reader._read_attr_value(bytes([14, 1, 1, 2]), 0, attr_def=attr_def)

    @pytest.mark.parametrize(
        "data",
        [
            bytes([14, 1, 2, 2, 4, 1, 2, 4, 1]),
            bytes([14, 1, 2, 3, 13, 0, 2, 4, 1]),
        ],
        ids=["duplicate", "out_of_order"],
    )
    def test_rejects_noncanonical_parameter_order(self, data: bytes) -> None:
        reader, attr_def = self._reader(["", "test.options", "mode", "scopes"])

        with pytest.raises(BytecodeError, match="not in declaration order"):
            reader._read_attr_value(data, 0, attr_def=attr_def)

    def test_rejects_missing_required_parameter(self) -> None:
        reader, attr_def = self._reader(["", "test.options"])

        with pytest.raises(BytecodeError, match="missing required parameter 'mode'"):
            reader._read_attr_value(bytes([14, 1, 0]), 0, attr_def=attr_def)


class TestParameterizedAttributeArrayWireFormat:
    def _reader(
        self, strings: list[str], field_name: str = "values"
    ) -> tuple[BytecodeReader, Any]:
        reader = BytecodeReader(
            b"",
            op_decls=[test_parameterized_attr_array],
            parameterized_attrs=[test_options_attr],
        )
        reader._strings = strings
        attr_def = reader._attr_def_for_op_attr(
            "test.parameterized_attr_array", field_name
        )
        return reader, attr_def

    def test_reads_ordered_repeated_mixed_families(self) -> None:
        reader, attr_def = self._reader(
            ["", "test.tile", "width", "test.options", "mode"]
        )
        tile_payload = bytes([1, 1, 2, 0]) + encode_signed_varint(8)
        options_payload = bytes([3, 1, 4, 4, 1])
        data = bytes([15, 3]) + tile_payload + options_payload + tile_payload

        value, offset = reader._read_attr_value(data, 0, attr_def=attr_def)

        assert isinstance(value, ParameterizedAttrArray)
        assert tuple(element.family_name for element in value) == (
            "test.tile",
            "test.options",
            "test.tile",
        )
        assert value.values[0] == value.values[2]
        assert offset == len(data)

    def test_rejects_array_without_field_descriptor(self) -> None:
        reader = BytecodeReader(b"")

        with pytest.raises(BytecodeError, match="descriptor-backed"):
            reader._read_attr_value(bytes([15, 0]), 0)

    def test_rejects_oversized_array(self) -> None:
        reader, attr_def = self._reader([""])
        data = bytes([15]) + encode_varint(0x10000)

        with pytest.raises(BytecodeError, match="exceeds UINT16_MAX"):
            reader._read_attr_value(data, 0, attr_def=attr_def)

    def test_rejects_array_beyond_aggregate_nesting_limit(self) -> None:
        reader, attr_def = self._reader([""])

        with pytest.raises(BytecodeError, match="maximum depth"):
            reader._read_attr_value(
                bytes([15, 0]),
                0,
                attr_def=attr_def,
                aggregate_nesting_depth=ATTR_AGGREGATE_MAX_NESTING_DEPTH,
            )

    def test_rejects_wrong_family_for_exact_array(self) -> None:
        reader, attr_def = self._reader(
            ["", "test.options", "mode"], field_name="tiles"
        )
        data = bytes([15, 1, 1, 1, 2, 4, 1])

        with pytest.raises(BytecodeError, match="does not match field contract"):
            reader._read_attr_value(data, 0, attr_def=attr_def)


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
