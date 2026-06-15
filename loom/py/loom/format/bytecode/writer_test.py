# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Comprehensive tests for the bytecode writer.

Every construct that can produce bytes in a .loombc file is tested:
file structure, every section type, every type variant, every
attribute kind, value numbering, and tied results. The Python writer
is the oracle for the C reader — these tests must be exhaustive.
"""

import struct
from dataclasses import replace

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.cfg import ALL_CFG_OPS
from loom.dialect.encoding import ALL_ENCODING_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.kernel import ALL_KERNEL_OPS, ALL_KERNEL_TYPES
from loom.dialect.low import ALL_LOW_OPS
from loom.dialect.pass_ import ALL_PASS_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.target import ALL_TARGET_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dsl import (
    ANY,
    SYMBOL_DEFINE,
    AttrDef,
    FuncLikeInterface,
    Op,
    Result,
    SymbolDefinition,
    SymbolReference,
)
from loom.format.bytecode.encoding import decode_varint
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import (
    FORMAT_VERSION,
    LOCATION_MODE_FULL_LOCATIONS,
    LOCATION_MODE_NO_LOCATIONS,
    LOCATION_MODE_SOURCE_LOCATIONS,
    SECTION_LOCATIONS,
    write_module,
)
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.ir import (
    BF16,
    BUFFER_TYPE,
    ENCODING_LAYOUT_TYPE,
    ENCODING_SCHEMA_TYPE,
    ENCODING_STORAGE_TYPE,
    ENCODING_TRANSFORM_TYPE,
    F32,
    I8,
    I32,
    I64,
    INDEX,
    LOCATION_UNKNOWN,
    SYMBOL_FLAG_IMPORT,
    SYMBOL_FLAG_PUBLIC,
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
    LocationTable,
    Module,
    Operation,
    PlaceholderType,
    PoolType,
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
    TiedResult,
    Type,
    TypeKind,
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


def _append_unique[T](items: list[T], additions: tuple[T, ...]) -> None:
    seen_ids = {id(item) for item in items}
    for item in additions:
        item_id = id(item)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(item)


def _module_range(data: bytes | bytearray) -> tuple[int, int]:
    producer_end = data.index(0, 16)
    directory_offset = (producer_end + 1 + 7) & ~7
    module_offset = struct.unpack_from("<Q", data, directory_offset + 8)[0]
    module_length = struct.unpack_from("<Q", data, directory_offset + 16)[0]
    return module_offset, module_length


def _section_kinds(data: bytes | bytearray) -> list[int]:
    module_offset, _module_length = _module_range(data)
    offset = module_offset
    section_count, offset = decode_varint(data, offset)
    for _ in range(4):
        _count, offset = decode_varint(data, offset)
    section_kinds = []
    for _ in range(section_count):
        section_kinds.append(struct.unpack_from("<H", data, offset)[0])
        offset += 32
    return section_kinds


def _test_ptr_register_type(unit_count: int = 1) -> RegisterType:
    return RegisterType(
        _TEST_LOW_CORE_STABLE_ID,
        _TEST_PTR_REGISTER_CLASS_ID,
        unit_count,
        "test.ptr",
    )


def _text_parser(
    *,
    include_encoding: bool = False,
    include_global: bool = False,
    include_vector: bool = False,
    include_kernel: bool = False,
    include_low: bool = False,
    include_pass: bool = False,
) -> Parser:
    parser = Parser()
    ops = list(ALL_FUNC_OPS) + list(ALL_CFG_OPS) + list(ALL_TEST_OPS)
    if include_encoding:
        _append_unique(ops, ALL_ENCODING_OPS)
    if include_global:
        _append_unique(ops, ALL_GLOBAL_OPS)
    if include_vector:
        _append_unique(ops, ALL_VECTOR_OPS)
    if include_kernel:
        _append_unique(
            ops,
            ALL_SCALAR_OPS
            + ALL_INDEX_OPS
            + ALL_ENCODING_OPS
            + ALL_BUFFER_OPS
            + ALL_VECTOR_OPS
            + ALL_KERNEL_OPS,
        )
    if include_low:
        _append_unique(ops, ALL_LOW_OPS)
    if include_pass:
        _append_unique(ops, ALL_PASS_OPS)
    parser.register_ops(ops)
    types = list(ALL_BUILTIN_TYPES)
    if include_kernel:
        _append_unique(types, ALL_KERNEL_TYPES)
    parser.register_types(types)
    return parser


def _text_printer(
    *,
    include_encoding: bool = False,
    include_global: bool = False,
    include_vector: bool = False,
    include_kernel: bool = False,
    include_low: bool = False,
    include_pass: bool = False,
) -> Printer:
    printer = Printer()
    ops = list(ALL_FUNC_OPS) + list(ALL_CFG_OPS) + list(ALL_TEST_OPS)
    if include_encoding:
        _append_unique(ops, ALL_ENCODING_OPS)
    if include_global:
        _append_unique(ops, ALL_GLOBAL_OPS)
    if include_vector:
        _append_unique(ops, ALL_VECTOR_OPS)
    if include_kernel:
        _append_unique(
            ops,
            ALL_SCALAR_OPS
            + ALL_INDEX_OPS
            + ALL_ENCODING_OPS
            + ALL_BUFFER_OPS
            + ALL_VECTOR_OPS
            + ALL_KERNEL_OPS,
        )
    if include_low:
        _append_unique(ops, ALL_LOW_OPS)
    if include_pass:
        _append_unique(ops, ALL_PASS_OPS)
    printer.register_ops(ops)
    types = list(ALL_BUILTIN_TYPES)
    if include_kernel:
        _append_unique(types, ALL_KERNEL_TYPES)
    printer.register_types(types)
    return printer


def _parse_write_read(
    text: str,
    *,
    include_encoding: bool = False,
    include_global: bool = False,
    include_vector: bool = False,
    include_kernel: bool = False,
    include_low: bool = False,
) -> Module:
    module = _text_parser(
        include_encoding=include_encoding,
        include_global=include_global,
        include_vector=include_vector,
        include_kernel=include_kernel,
        include_low=include_low,
    ).parse(text)
    return read_module(write_module(module))


def _roundtrip_text_through_bytecode(
    text: str,
    *,
    include_encoding: bool = False,
    include_global: bool = False,
    include_vector: bool = False,
    include_kernel: bool = False,
    include_low: bool = False,
) -> str:
    loaded = _parse_write_read(
        text,
        include_encoding=include_encoding,
        include_global=include_global,
        include_vector=include_vector,
        include_kernel=include_kernel,
        include_low=include_low,
    )
    return _text_printer(
        include_encoding=include_encoding,
        include_global=include_global,
        include_vector=include_vector,
        include_kernel=include_kernel,
        include_low=include_low,
    ).print_module(loaded)


def _make_func_op(
    module: Module,
    func_name: str,
    arg_types: list[Type],
    result_types: list[Type],
    body_ops: list[Operation] | None = None,
    is_public: bool = False,
    is_declaration: bool = False,
) -> tuple[Operation, int]:
    """Build a func.def or func.decl Operation and add it to the module.

    Returns (func_op, sym_flags) for use with Symbol construction.
    """
    op_attrs: dict[str, object] = {"callee": func_name}
    if is_public:
        op_attrs["visibility"] = "public"

    result_ids = [module.add_value(Value(name="", type=rt)) for rt in result_types]
    sym_flags = SYMBOL_FLAG_PUBLIC if is_public else 0

    if is_declaration:
        operand_ids = [
            module.add_value(Value(name=f"arg{i}", type=at))
            for i, at in enumerate(arg_types)
        ]
        func_op = Operation(
            name="func.decl",
            operands=operand_ids,
            results=result_ids,
            attributes=op_attrs,
        )
    else:
        arg_ids = [
            module.add_value(Value(name=f"arg{i}", type=at))
            for i, at in enumerate(arg_types)
        ]
        ops = body_ops or []
        if not ops:
            # Default: yield the first arg.
            ops = [
                Operation(name="test.yield", operands=[arg_ids[0]] if arg_ids else [])
            ]
        block = Block(arg_ids=arg_ids, ops=ops)
        func_op = Operation(
            name="func.def",
            results=result_ids,
            attributes=op_attrs,
            regions=[Region(blocks=[block])],
        )

    return func_op, sym_flags


def _make_func_module(
    name: str = "test",
    func_name: str = "f",
    arg_types: list[Type] | None = None,
    result_types: list[Type] | None = None,
    body_ops: list[Operation] | None = None,
    attributes: dict[str, object] | None = None,
    is_public: bool = False,
    is_declaration: bool = False,
) -> Module:
    """Build a module with one function. Convenience for tests."""
    module = Module(name=name)
    arg_types = arg_types or [F32]
    result_types = result_types or []
    func_op, sym_flags = _make_func_op(
        module,
        func_name,
        arg_types,
        result_types,
        body_ops=body_ops,
        is_public=is_public,
        is_declaration=is_declaration,
    )
    sym_kind = SymbolKind.FUNC_DECL if is_declaration else SymbolKind.FUNC_DEF
    module.add_symbol(
        Symbol(name=func_name, kind=sym_kind, flags=sym_flags, op=func_op)
    )
    return module


def _make_attrs_module(dict_attr: dict[str, object]) -> Module:
    """Build a func.def containing one test.attrs op with dict_attr."""
    module = Module()
    x = module.add_value(Value(name="x", type=F32))
    r = module.add_value(Value(name="r", type=F32))
    attrs_op = Operation(
        name="test.attrs",
        operands=[x],
        results=[r],
        attributes={"dict": dict_attr},
    )
    yield_op = Operation(name="test.yield", operands=[r])
    block = Block(arg_ids=[x], ops=[attrs_op, yield_op])
    body = Region(blocks=[block])
    func_result = module.add_value(Value(name="", type=F32))
    func_op = Operation(
        name="func.def",
        results=[func_result],
        attributes={"callee": "f"},
        regions=[body],
    )
    module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
    return module


def _strip_locations(module: Module) -> None:
    """Drop parser-produced locations so byte comparisons only test IR shape."""
    module.sources = []
    module.locations = LocationTable()

    def clear_op(op: Operation) -> None:
        op.location_id = LOCATION_UNKNOWN
        for region in op.regions:
            for block in region.blocks:
                for child_op in block.ops:
                    clear_op(child_op)

    for symbol in module.symbols:
        if symbol.op is not None:
            clear_op(symbol.op)


def _first_attrs_dict(module: Module) -> CanonicalAttrDict:
    func_op = module.symbols[0].op
    assert func_op is not None
    attrs_op = func_op.regions[0].blocks[0].ops[0]
    dict_attr = attrs_op.attributes["dict"]
    assert isinstance(dict_attr, CanonicalAttrDict)
    return dict_attr


# ============================================================================
# File structure
# ============================================================================


class TestFileHeader:
    def test_magic_bytes(self) -> None:
        data = write_module(Module(name="test"))
        assert data[0:4] == b"LOOM"

    def test_format_version(self) -> None:
        data = write_module(Module(name="test"))
        assert data[4] == FORMAT_VERSION

    def test_location_mode_source(self) -> None:
        data = write_module(Module(name="test"))
        assert data[5] == LOCATION_MODE_SOURCE_LOCATIONS

    def test_location_mode_no_locations(self) -> None:
        data = write_module(
            Module(name="test"), location_mode=LOCATION_MODE_NO_LOCATIONS
        )
        assert data[5] == LOCATION_MODE_NO_LOCATIONS
        assert SECTION_LOCATIONS not in _section_kinds(data)
        read_module(data)

    def test_location_mode_no_locations_strips_op_locations(self) -> None:
        module = Module(name="test")
        module.sources.append("model.loom")
        loc_id = module.locations.add(
            FileLocation(
                source_id=0, start_line=42, start_col=3, end_line=42, end_col=58
            )
        )
        value_id = module.add_value(Value(name="x", type=F32))
        block = Block(
            arg_ids=[value_id],
            ops=[Operation(name="test.yield", operands=[value_id], location_id=loc_id)],
        )
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

        loaded = read_module(
            write_module(module, location_mode=LOCATION_MODE_NO_LOCATIONS)
        )
        loaded_sym_op = loaded.symbols[0].op
        assert loaded_sym_op is not None
        loaded_op = loaded_sym_op.regions[0].blocks[0].ops[0]
        assert loaded_op.location_id == 0

    def test_location_mode_full_rejected_until_field_spans_exist(self) -> None:
        with pytest.raises(NotImplementedError, match="FULL_LOCATIONS"):
            write_module(
                Module(name="test"), location_mode=LOCATION_MODE_FULL_LOCATIONS
            )

    def test_module_count_one(self) -> None:
        data = write_module(Module(name="test"))
        assert struct.unpack_from("<H", data, 6)[0] == 1

    def test_producer_string(self) -> None:
        data = write_module(Module(name="test"))
        # Producer starts at offset 16, null-terminated.
        producer_start = 16
        producer_end = data.index(0, producer_start)
        producer = data[producer_start:producer_end].decode("utf-8")
        assert "loom" in producer.lower()

    def test_header_aligned_to_8(self) -> None:
        data = write_module(Module(name="test"))
        # After the header (including padding), the next structure starts
        # at an 8-byte aligned offset.
        # Header: 16 fixed + producer + null + padding.
        producer_start = 16
        producer_end = data.index(0, producer_start)
        header_end = producer_end + 1
        aligned = (header_end + 7) & ~7
        # Check padding is zeros.
        for i in range(header_end, aligned):
            assert data[i] == 0, f"non-zero padding at offset {i}"

    def test_deterministic_output(self) -> None:
        m1 = _make_func_module()
        m2 = _make_func_module()
        assert write_module(m1) == write_module(m2)


class TestModuleDirectory:
    def test_module_offset_valid(self) -> None:
        data = write_module(Module(name="test"))
        # Module directory is after the aligned header.
        # Read module_offset from directory entry.
        # Find the directory: it's right after the aligned header.
        producer_start = 16
        producer_end = data.index(0, producer_start)
        dir_start = (producer_end + 1 + 7) & ~7
        module_offset = struct.unpack_from("<Q", data, dir_start + 8)[0]
        assert module_offset > 0
        assert module_offset < len(data)

    def test_module_length_valid(self) -> None:
        data = write_module(Module(name="test"))
        producer_start = 16
        producer_end = data.index(0, producer_start)
        dir_start = (producer_end + 1 + 7) & ~7
        module_offset = struct.unpack_from("<Q", data, dir_start + 8)[0]
        module_length = struct.unpack_from("<Q", data, dir_start + 16)[0]
        assert module_offset + module_length == len(data)


# ============================================================================
# STRINGS section
# ============================================================================


class TestStringsSection:
    def test_empty_module_has_strings(self) -> None:
        """Even an empty module interns at least the module name."""
        data = write_module(Module(name="mymodule"))
        assert b"mymodule" in data

    def test_function_names_interned(self) -> None:
        module = _make_func_module(func_name="compute")
        data = write_module(module)
        assert b"compute" in data

    def test_op_names_interned(self) -> None:
        module = _make_func_module()
        data = write_module(module)
        assert b"test.yield" in data

    def test_value_names_interned(self) -> None:
        module = _make_func_module()
        data = write_module(module)
        assert b"arg0" in data


# ============================================================================
# TYPES section — every type variant
# ============================================================================


class TestTypesSection:
    def _roundtrip_type(self, ir_type: Type) -> None:
        """Write a module with a value of this type, read back, verify.

        For types with dynamic dims, creates index-typed SSA values
        for each dynamic dim and populates dim_bindings accordingly.
        Every dynamic dim must have a binding.
        """
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        # Create dim values for any dynamic dims in the type.
        dim_bindings: dict[int, int] = {}
        if hasattr(ir_type, "dims"):
            for i, dim in enumerate(ir_type.dims):
                if isinstance(dim, DynamicDim):
                    dim_id = module.add_value(Value(name=f"d{i}", type=INDEX))
                    dim_bindings[i] = dim_id
        vid = module.add_value(Value(name="v", type=ir_type, dim_bindings=dim_bindings))
        all_arg_ids = [*dim_bindings.values(), vid]
        yield_op = Operation(name="test.yield", operands=[vid])
        block = Block(arg_ids=all_arg_ids, ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        data = write_module(module)
        loaded = read(data)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        # The value under test is the last block arg.
        arg_id = loaded_op.regions[0].blocks[0].arg_ids[-1]
        loaded_type = loaded.values[arg_id].type
        assert loaded_type == ir_type, f"Type mismatch: {loaded_type} != {ir_type}"

    def test_scalar_f32(self) -> None:
        self._roundtrip_type(F32)

    def test_scalar_i32(self) -> None:
        self._roundtrip_type(I32)

    def test_scalar_i64(self) -> None:
        self._roundtrip_type(I64)

    def test_scalar_i8(self) -> None:
        self._roundtrip_type(I8)

    def test_scalar_index(self) -> None:
        self._roundtrip_type(INDEX)

    def test_scalar_bf16(self) -> None:
        self._roundtrip_type(BF16)

    def test_scalar_f8e4m3(self) -> None:
        self._roundtrip_type(ScalarType(ScalarTypeKind.F8E4M3))

    def test_scalar_f8e5m2(self) -> None:
        self._roundtrip_type(ScalarType(ScalarTypeKind.F8E5M2))

    def test_tile_0d(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.TILE, F32, ()))

    def test_tile_1d(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.TILE, F32, (StaticDim(4),)))

    def test_tile_2d(self) -> None:
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(8)))
        )

    def test_tile_3d(self) -> None:
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, I32, (StaticDim(2), StaticDim(3), StaticDim(4)))
        )

    def test_tile_dynamic(self) -> None:
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        )

    def test_tile_all_dynamic(self) -> None:
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, F32, (DynamicDim(), DynamicDim()))
        )

    def test_dynamic_dim_missing_binding_raises(self) -> None:
        """Dynamic dims without dim_bindings are invalid IR."""
        module = Module(name="test")
        vid = module.add_value(
            Value(name="v", type=ShapedType(TypeKind.TILE, F32, (DynamicDim(),)))
        )
        yield_op = Operation(name="test.yield", operands=[vid])
        block = Block(arg_ids=[vid], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        with pytest.raises(ValueError, match=r"1 dynamic dim.*0 dim binding"):
            write_module(module)

    def test_placeholder_type_fails_loudly(self) -> None:
        """Placeholder types are parser-internal and must not reach bytecode."""
        module = Module(name="test")
        vid = module.add_value(Value(name="v", type=PlaceholderType()))
        yield_op = Operation(name="test.yield", operands=[vid])
        block = Block(arg_ids=[vid], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        with pytest.raises(ValueError, match="unsupported bytecode type"):
            write_module(module)

    def test_unavailable_operand_fails_loudly(self) -> None:
        """Function-local operands must reference values defined in scope."""
        module = Module(name="test")
        stray_id = module.add_value(Value(name="stray", type=F32))
        yield_op = Operation(name="test.yield", operands=[stray_id])
        block = Block(ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        with pytest.raises(ValueError, match="operation operand references value"):
            write_module(module)

    def test_tensor_1d(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.TENSOR, I8, (StaticDim(256),)))

    def test_vector_1d(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.VECTOR, F32, (StaticDim(16),)))

    def test_vector_zero_extent(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.VECTOR, F32, (StaticDim(0),)))

    def test_vector_dynamic(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.VECTOR, I32, (DynamicDim(),)))

    def test_view_1d(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.VIEW, I8, (StaticDim(256),)))

    def test_view_with_layout(self) -> None:
        layout = EncodingInstance(name="strided", params=(("stride", 64),))
        self._roundtrip_type(
            ShapedType(TypeKind.VIEW, F32, (StaticDim(256),), encoding=layout)
        )

    def test_view_with_dynamic_layout(self) -> None:
        self._roundtrip_type(
            ShapedType(
                TypeKind.VIEW,
                F32,
                (StaticDim(256),),
                encoding=DynamicEncoding(),
            )
        )

    def test_storage_workgroup(self) -> None:
        self._roundtrip_type(StorageType(StorageSpace.WORKGROUP))

    def test_tensor_large_dim(self) -> None:
        self._roundtrip_type(ShapedType(TypeKind.TENSOR, F32, (StaticDim(1048576),)))

    def test_tile_with_encoding(self) -> None:
        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        )

    def test_tile_with_encoding_no_params(self) -> None:
        enc = EncodingInstance(name="dense")
        self._roundtrip_type(
            ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        )

    def test_group_type(self) -> None:
        self._roundtrip_type(GroupType(GroupScope.WORKGROUP))

    def test_group_subgroup(self) -> None:
        self._roundtrip_type(GroupType(GroupScope.SUBGROUP))

    def test_function_type(self) -> None:
        self._roundtrip_type(FunctionType((F32, I32), (F32,)))

    def test_function_type_empty(self) -> None:
        self._roundtrip_type(FunctionType((), ()))

    def test_dialect_type_opaque(self) -> None:
        self._roundtrip_type(DialectType("hal.buffer"))

    def test_dialect_type_parameterized(self) -> None:
        self._roundtrip_type(DialectType("vm.ref", (DialectType("hal.buffer"),)))

    def test_dialect_type_nested(self) -> None:
        inner = DialectType("vm.list", (I32,))
        self._roundtrip_type(DialectType("vm.ref", (inner,)))

    def test_register_type(self) -> None:
        self._roundtrip_type(_test_ptr_register_type(4))

    def test_pool_static(self) -> None:
        self._roundtrip_type(PoolType(StaticDim(65536)))

    def test_pool_static_small(self) -> None:
        self._roundtrip_type(PoolType(StaticDim(4096)))

    def test_pool_dynamic(self) -> None:
        self._roundtrip_type(PoolType(DynamicDim()))

    def test_buffer_type(self) -> None:
        self._roundtrip_type(BUFFER_TYPE)

    def test_encoding_type(self) -> None:
        from loom.ir import ENCODING_TYPE

        self._roundtrip_type(ENCODING_TYPE)

    def test_encoding_role_types(self) -> None:
        self._roundtrip_type(ENCODING_LAYOUT_TYPE)
        self._roundtrip_type(ENCODING_SCHEMA_TYPE)
        self._roundtrip_type(ENCODING_STORAGE_TYPE)
        self._roundtrip_type(ENCODING_TRANSFORM_TYPE)


# ============================================================================
# ENCODINGS section
# ============================================================================


class TestEncodingsSection:
    def test_empty_encodings(self) -> None:
        from loom.format.bytecode.reader import read_module as read

        module = _make_func_module()
        data = write_module(module)
        loaded = read(data)
        assert len(loaded.encodings) == 0

    def test_single_encoding(self) -> None:
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        module = Module(name="test")
        module.add_encoding(enc)
        enc_type = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        vid = module.add_value(Value(name="v", type=enc_type))
        yield_op = Operation(name="test.yield", operands=[vid])
        block = Block(arg_ids=[vid], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        data = write_module(module)
        loaded = read(data)
        assert len(loaded.encodings) >= 1
        assert loaded.encodings[0].name == "q8_0"
        assert loaded.encodings[0].params == (("block", 32),)

    def test_multiple_encoding_families(self) -> None:
        from loom.format.bytecode.reader import read_module as read

        enc1 = EncodingInstance(name="q8_0")
        enc2 = EncodingInstance(name="q6_k")
        module = Module(name="test")
        module.add_encoding(enc1)
        module.add_encoding(enc2)
        t1 = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc1)
        t2 = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc2)
        v1 = module.add_value(Value(name="a", type=t1))
        v2 = module.add_value(Value(name="b", type=t2))
        yield_op = Operation(name="test.yield", operands=[v1, v2])
        block = Block(arg_ids=[v1, v2], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        data = write_module(module)
        loaded = read(data)
        names = {e.name for e in loaded.encodings}
        assert "q8_0" in names
        assert "q6_k" in names

    def test_encoding_integer_param_roundtrip(self) -> None:
        """Integer params (block=32) round-trip through structured I64 encoding."""
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        v = module.add_value(Value(name="v", type=t))
        block = Block(arg_ids=[v], ops=[Operation(name="test.yield", operands=[v])])
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DEF,
                op=Operation(
                    name="func.def",
                    attributes={"callee": "f"},
                    regions=[Region(blocks=[block])],
                ),
            )
        )
        data = write_module(module)
        loaded = read(data)
        assert loaded.encodings[0].params == (("block", 32),)

    def test_encoding_string_param_roundtrip(self) -> None:
        """String params (layout=nchw) round-trip through structured STRING encoding."""
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(name="dense", params=(("layout", "nchw"),))
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(4)), encoding=enc)
        v = module.add_value(Value(name="v", type=t))
        block = Block(arg_ids=[v], ops=[Operation(name="test.yield", operands=[v])])
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DEF,
                op=Operation(
                    name="func.def",
                    attributes={"callee": "f"},
                    regions=[Region(blocks=[block])],
                ),
            )
        )
        data = write_module(module)
        loaded = read(data)
        assert loaded.encodings[0].params == (("layout", "nchw"),)

    def test_encoding_multiple_params_roundtrip(self) -> None:
        """Multiple params round-trip with correct names and values."""
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(
            name="q4_k",
            params=(("block", 32), ("group_size", 128)),
        )
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(512),), encoding=enc)
        v = module.add_value(Value(name="v", type=t))
        block = Block(arg_ids=[v], ops=[Operation(name="test.yield", operands=[v])])
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DEF,
                op=Operation(
                    name="func.def",
                    attributes={"callee": "f"},
                    regions=[Region(blocks=[block])],
                ),
            )
        )
        data = write_module(module)
        loaded = read(data)
        assert loaded.encodings[0].params == (("block", 32), ("group_size", 128))

    def test_encoding_with_alias_roundtrip(self) -> None:
        """Encoding alias round-trips correctly."""
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(
            name="q8_0",
            alias="enc",
            params=(("block", 32),),
        )
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(128),), encoding=enc)
        v = module.add_value(Value(name="v", type=t))
        block = Block(arg_ids=[v], ops=[Operation(name="test.yield", operands=[v])])
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DEF,
                op=Operation(
                    name="func.def",
                    attributes={"callee": "f"},
                    regions=[Region(blocks=[block])],
                ),
            )
        )
        data = write_module(module)
        loaded = read(data)
        assert loaded.encodings[0].alias == "enc"
        assert loaded.encodings[0].name == "q8_0"
        assert loaded.encodings[0].params == (("block", 32),)

    def test_encoding_no_params_roundtrip(self) -> None:
        """Encoding with zero params round-trips correctly."""
        from loom.format.bytecode.reader import read_module as read

        enc = EncodingInstance(name="dense")
        module = Module(name="test")
        module.add_encoding(enc)
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        v = module.add_value(Value(name="v", type=t))
        block = Block(arg_ids=[v], ops=[Operation(name="test.yield", operands=[v])])
        module.add_symbol(
            Symbol(
                name="f",
                kind=SymbolKind.FUNC_DEF,
                op=Operation(
                    name="func.def",
                    attributes={"callee": "f"},
                    regions=[Region(blocks=[block])],
                ),
            )
        )
        data = write_module(module)
        loaded = read(data)
        assert loaded.encodings[0].name == "dense"
        assert loaded.encodings[0].params == ()


# ============================================================================
# OPS section
# ============================================================================


class TestOpsSection:
    def test_op_names_in_bytecode(self) -> None:
        module = _make_func_module()
        data = write_module(module)
        assert b"test.yield" in data

    def test_multiple_op_kinds(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        r = module.add_value(Value(name="r", type=F32))
        neg = Operation(name="test.neg", operands=[x], results=[r])
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[x], ops=[neg, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        data = write_module(module)
        assert b"test.neg" in data
        assert b"test.yield" in data


# ============================================================================
# IR section — attribute values
# ============================================================================


class TestAttributeValues:
    def _roundtrip_attr(self, key: str, value: object) -> object:
        """Write an op with an attribute, read back, verify."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        op = Operation(name="test.constant", results=[], attributes={key: value})
        yield_op = Operation(name="test.yield", operands=[x])
        block = Block(arg_ids=[x], ops=[op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        data = write_module(module)
        loaded = read(data)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        target_op = loaded_op.regions[0].blocks[0].ops[0]
        return target_op.attributes.get(key)

    def test_i64_positive(self) -> None:
        assert self._roundtrip_attr("val", 42) == 42

    def test_i64_negative(self) -> None:
        assert self._roundtrip_attr("val", -7) == -7

    def test_i64_zero(self) -> None:
        assert self._roundtrip_attr("val", 0) == 0

    def test_i64_large(self) -> None:
        assert self._roundtrip_attr("val", 2**40) == 2**40

    def test_f64(self) -> None:
        result = self._roundtrip_attr("val", 3.14)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-15

    def test_f64_negative(self) -> None:
        result = self._roundtrip_attr("val", -2.718)
        assert isinstance(result, float)
        assert abs(result - (-2.718)) < 1e-15

    def test_f64_zero(self) -> None:
        assert self._roundtrip_attr("val", 0.0) == 0.0

    def test_string_ascii(self) -> None:
        assert self._roundtrip_attr("label", "hello") == "hello"

    def test_string_empty(self) -> None:
        assert self._roundtrip_attr("label", "") == ""

    def test_bool_true(self) -> None:
        assert self._roundtrip_attr("flag", True) is True

    def test_bool_false(self) -> None:
        assert self._roundtrip_attr("flag", False) is False

    def test_i64_array(self) -> None:
        assert self._roundtrip_attr("offsets", [0, 1, 2]) == [0, 1, 2]

    def test_i64_array_empty(self) -> None:
        assert self._roundtrip_attr("offsets", []) == []

    def test_i64_array_with_sentinel(self) -> None:
        sentinel = -(2**63)
        assert self._roundtrip_attr("offsets", [0, sentinel]) == [0, sentinel]

    def test_i64_array_negative(self) -> None:
        assert self._roundtrip_attr("vals", [-1, -2, -3]) == [-1, -2, -3]


# ============================================================================
# IR section — op patterns (round-trip)
# ============================================================================


class TestOpPatterns:
    """Round-trip every op pattern through bytecode."""

    def _roundtrip_ops(self, module: Module) -> Module:
        from loom.format.bytecode.reader import read_module as read

        data = write_module(module)
        return read(data)

    def test_binary_op(self) -> None:
        module = Module(name="test")
        a = module.add_value(Value(name="a", type=I32))
        b = module.add_value(Value(name="b", type=I32))
        r = module.add_value(Value(name="r", type=I32))
        add = Operation(name="test.addi", operands=[a, b], results=[r])
        yield_op = Operation(name="test.yield", operands=[r])
        block = Block(arg_ids=[a, b], ops=[add, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        add_op = loaded_op.regions[0].blocks[0].ops[0]
        assert add_op.name == "test.addi"
        assert len(add_op.operands) == 2
        assert len(add_op.results) == 1

    def test_segmented_operands(self) -> None:
        module = Module(name="test")
        root = module.add_value(Value(name="root", type=I32))
        guard = module.add_value(Value(name="guard", type=I32))
        lhs = module.add_value(Value(name="lhs", type=I32))
        rhs = module.add_value(Value(name="rhs", type=I32))
        result = module.add_value(Value(name="result", type=I32))
        segmented = Operation(
            name="test.segmented",
            operands=[root, guard, lhs, rhs],
            operand_segment_counts=(1, 1, 1, 1),
            results=[result],
        )
        yield_op = Operation(name="test.yield", operands=[result])
        block = Block(arg_ids=[root, guard, lhs, rhs], ops=[segmented, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        segmented_op = loaded_op.regions[0].blocks[0].ops[0]
        assert segmented_op.name == "test.segmented"
        assert segmented_op.operand_segment_counts == (1, 1, 1, 1)
        assert len(segmented_op.operands) == 4

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
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        update_op = loaded_op.regions[0].blocks[0].ops[0]
        assert len(update_op.tied_results) == 1
        assert update_op.tied_results[0].result_index == 0
        assert update_op.tied_results[0].operand_index == 1

    def test_result_dim_reference_roundtrip(self) -> None:
        """Result dim referencing another result survives bytecode round-trip."""
        tensor_dyn = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(),))
        module = Module(name="test")
        input_id = module.add_value(
            Value(name="input", type=tensor_dyn, dim_bindings={0: 0})
        )
        # Create length first so we can reference it in output's dim.
        length_id = module.add_value(Value(name="length", type=INDEX))
        # Result 0: tensor<[%length]xf32> — dim references length directly.
        output_id = module.add_value(
            Value(name="output", type=tensor_dyn, dim_bindings={0: length_id})
        )
        deflate_op = Operation(
            name="test.deflate",
            operands=[input_id],
            results=[output_id, length_id],
        )
        yield_op = Operation(name="test.yield", operands=[output_id])
        block = Block(arg_ids=[input_id], ops=[deflate_op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        deflate = loaded_op.regions[0].blocks[0].ops[0]
        assert deflate.name == "test.deflate"
        assert len(deflate.results) == 2
        # Check the output value's dim binding references the loaded length.
        output_value = loaded.values[deflate.results[0]]
        length_value_id = deflate.results[1]
        assert output_value.dim_bindings[0] == length_value_id

    def test_nested_region(self) -> None:
        module = Module(name="test")
        x = module.add_value(Value(name="x", type=F32))
        e = module.add_value(Value(name="e", type=F32))
        neg = Operation(name="test.neg", operands=[e], results=[e])
        inner_yield = Operation(name="test.yield", operands=[e])
        inner_block = Block(arg_ids=[e], ops=[neg, inner_yield])
        inner_region = Region(blocks=[inner_block])
        map_result = module.add_value(Value(name="r", type=F32))
        map_op = Operation(
            name="test.map", operands=[x], results=[map_result], regions=[inner_region]
        )
        yield_op = Operation(name="test.yield", operands=[map_result])
        block = Block(arg_ids=[x], ops=[map_op, yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        map_loaded = loaded_op.regions[0].blocks[0].ops[0]
        assert map_loaded.name == "test.map"
        assert len(map_loaded.regions) == 1
        assert len(map_loaded.regions[0].blocks) == 1
        assert len(map_loaded.regions[0].blocks[0].ops) == 2

    def test_successor_reference(self) -> None:
        module = Module(name="test")
        exit_block = Block(label="exit", ops=[Operation(name="test.yield")])
        entry_block = Block(
            label="entry",
            ops=[Operation(name="test.br", successors=[exit_block])],
        )
        body = Region(blocks=[entry_block, exit_block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        loaded_body = loaded_op.regions[0]
        assert len(loaded_body.blocks) == 2
        branch = loaded_body.blocks[0].ops[0]
        assert branch.name == "test.br"
        assert branch.successors == [loaded_body.blocks[1]]
        assert branch.successors[0].label == "exit"

    def test_no_results_op(self) -> None:
        module = Module(name="test")
        a = module.add_value(Value(name="a", type=F32))
        b = module.add_value(Value(name="b", type=I32))
        yield_op = Operation(name="test.yield", operands=[a, b])
        block = Block(arg_ids=[a, b], ops=[yield_op])
        body = Region(blocks=[block])
        func_op = Operation(name="func.def", attributes={"callee": "f"}, regions=[body])
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))
        loaded = self._roundtrip_ops(module)
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.regions
        yield_loaded = loaded_op.regions[0].blocks[0].ops[0]
        assert yield_loaded.name == "test.yield"
        assert len(yield_loaded.operands) == 2
        assert len(yield_loaded.results) == 0

    def test_declaration_no_body(self) -> None:
        module = _make_func_module(is_declaration=True)
        from loom.format.bytecode.reader import read_module as read

        loaded = read(write_module(module))
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert not loaded_op.regions

    def test_public_function(self) -> None:
        module = _make_func_module(is_public=True)
        from loom.format.bytecode.reader import read_module as read

        loaded = read(write_module(module))
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.attributes.get("visibility") == "public"

    def test_multiple_functions(self) -> None:
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        for fn_name in ["f1", "f2", "f3"]:
            vid = module.add_value(Value(name=f"x_{fn_name}", type=F32))
            yield_op = Operation(name="test.yield", operands=[vid])
            block = Block(arg_ids=[vid], ops=[yield_op])
            body = Region(blocks=[block])
            func_op = Operation(
                name="func.def",
                attributes={"callee": fn_name},
                regions=[body],
            )
            module.add_symbol(
                Symbol(name=fn_name, kind=SymbolKind.FUNC_DEF, op=func_op)
            )
        loaded = read(write_module(module))
        assert len(loaded.symbols) == 3
        names = {s.name for s in loaded.symbols}
        assert names == {"f1", "f2", "f3"}


# ============================================================================
# Cross-format round-trip
# ============================================================================


class TestCrossFormatRoundTrip:
    """text → parse → IR → write bytecode → read bytecode → IR → print text."""

    def test_simple_function(self) -> None:
        text = (
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

        loaded = _parse_write_read(text)
        assert len(loaded.symbols) == 1
        func_op = loaded.symbols[0].op
        assert func_op is not None
        assert func_op.attributes.get("callee") == "negate"
        assert func_op.regions
        assert len(func_op.regions[0].blocks[0].ops) == 2

    def test_optional_region_count_survives_bytecode(self) -> None:
        text = (
            "func.def @optional_region(%cond: i1) {\n"
            "  test.optional_region %cond {\n"
            "    test.yield\n"
            "  }\n"
            "  test.optional_region %cond {\n"
            "    test.yield\n"
            "  } else {\n"
            "    test.yield\n"
            "  }\n"
            "  func.return\n"
            "}\n"
        )
        assert _roundtrip_text_through_bytecode(text) == text

    def test_projected_func_regions_survive_bytecode(self) -> None:
        text = (
            "test.split_func @projected(%arg: i32, %other: index) {\n"
            "  test.use %arg : i32\n"
            "  test.use %other : index\n"
            "  test.yield\n"
            "} launch {\n"
            "  test.use %arg : i32\n"
            "  test.use %other : index\n"
            "  test.yield\n"
            "}\n"
        )
        loaded = _parse_write_read(text)
        op = loaded.symbols[0].op
        assert op is not None
        assert len(op.regions) == 2
        config_arg_ids = op.regions[0].blocks[0].arg_ids
        body_arg_ids = op.regions[1].blocks[0].arg_ids
        assert len(body_arg_ids) == len(config_arg_ids) == 2
        assert body_arg_ids[0] != config_arg_ids[0]
        assert loaded.values[body_arg_ids[0]].name == "arg"
        assert loaded.values[config_arg_ids[0]].name == "arg"
        assert _roundtrip_text_through_bytecode(text) == text

    def test_record_symbol_survives_bytecode(self) -> None:
        text = 'test.record @target {arch = "gfx1100", lanes = 64}\n'
        loaded = _parse_write_read(text)
        assert len(loaded.symbols) == 1
        symbol = loaded.symbols[0]
        assert symbol.kind == SymbolKind.RECORD
        assert symbol.op is not None
        assert symbol.op.attributes["symbol"] == "target"
        assert dict(symbol.op.attributes["dict"]) == {
            "arch": "gfx1100",
            "lanes": 64,
        }
        assert _roundtrip_text_through_bytecode(text) == text

    def test_record_symbol_uses_generated_symbol_field(self) -> None:
        custom_record = Op(
            "bytecode.record",
            traits=[SYMBOL_DEFINE],
            symbol_def=SymbolDefinition(
                field="name",
                name="record",
                interfaces=["record"],
                bytecode_kind="LOOM_SYMBOL_RECORD",
            ),
            attrs=[
                AttrDef("name", "symbol"),
                AttrDef(
                    "parent",
                    "symbol",
                    optional=True,
                    symbol_ref=SymbolReference("record", ["record"]),
                ),
                AttrDef("payload", "dict", optional=True),
            ],
        )
        module = Module()
        parent_op = Operation(
            name="bytecode.record",
            attributes={
                "name": "target",
                "payload": {"arch": "gfx1100", "lanes": 64},
            },
        )
        child_op = Operation(
            name="bytecode.record",
            attributes={
                "name": "child",
                "parent": "target",
                "payload": {"arch": "gfx1200", "lanes": 32},
            },
        )
        module.add_symbol(Symbol(name="target", kind=SymbolKind.RECORD, op=parent_op))
        module.add_symbol(Symbol(name="child", kind=SymbolKind.RECORD, op=child_op))

        loaded = read_module(
            write_module(module, op_decls=[custom_record]),
            op_decls=[custom_record],
        )

        assert len(loaded.symbols) == 2
        symbol = loaded.symbols[0]
        assert symbol.op is not None
        assert symbol.op.attributes["name"] == "target"
        assert "symbol" not in symbol.op.attributes
        assert dict(symbol.op.attributes["payload"]) == {
            "arch": "gfx1100",
            "lanes": 64,
        }
        child = loaded.symbols[1]
        assert child.op is not None
        assert child.op.attributes["name"] == "child"
        assert child.op.attributes["parent"] == "target"
        assert dict(child.op.attributes["payload"]) == {
            "arch": "gfx1200",
            "lanes": 32,
        }

    def test_func_like_payload_attrs_survive_bytecode(self) -> None:
        custom_func = Op(
            "bytecode.func",
            traits=[SYMBOL_DEFINE],
            symbol_def=SymbolDefinition(
                field="callee",
                name="function",
                interfaces=["func_like"],
                bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
            ),
            interfaces=[FuncLikeInterface(callee="callee", args_as_operands=True)],
            attrs=[
                AttrDef("callee", "symbol"),
                AttrDef(
                    "target",
                    "symbol",
                    optional=True,
                    symbol_ref=SymbolReference("record", ["record"]),
                ),
                AttrDef("tag", "string", optional=True),
                AttrDef("priority", "i64", optional=True),
            ],
            results=[Result("results", ANY, variadic=True)],
        )
        module = Module()
        input_id = module.add_value(Value(name="input", type=F32))
        result_id = module.add_value(Value(name="result", type=F32))
        func_op = Operation(
            name="bytecode.func",
            operands=[input_id],
            results=[result_id],
            attributes={
                "callee": "entry",
                "target": "gfx1100",
                "tag": "amdgpu",
                "priority": 3,
            },
        )
        module.add_symbol(Symbol(name="entry", kind=SymbolKind.FUNC_DECL, op=func_op))

        loaded = read_module(
            write_module(module, op_decls=[custom_func]),
            op_decls=[custom_func],
        )

        assert len(loaded.symbols) == 1
        symbol = loaded.symbols[0]
        assert symbol.op is not None
        assert symbol.op.attributes == {
            "callee": "entry",
            "target": "gfx1100",
            "tag": "amdgpu",
            "priority": 3,
        }

    def test_func_template_metadata_survives_bytecode(self) -> None:
        text = (
            "func.template<tile.contract> device priority(7) "
            "@kernel(%input: f32) -> (f32) {\n"
            "  func.return %input : f32\n"
            "}\n"
        )

        loaded = _parse_write_read(text)
        assert len(loaded.symbols) == 1
        symbol = loaded.symbols[0]
        assert symbol.kind == SymbolKind.FUNC_TEMPLATE
        assert symbol.op is not None
        assert symbol.op.attributes["implements"] == "tile.contract"
        assert symbol.op.attributes["priority"] == 7
        assert symbol.op.attributes["cc"] == "device"
        assert _roundtrip_text_through_bytecode(text) == text

    def test_low_func_target_and_register_body_survive_bytecode(self) -> None:
        text = (
            "test.record @test_target\n\n"
            "low.func.def target(@test_target) "
            "@add(%lhs: reg<test.ptr>, %rhs: reg<test.ptr>) "
            "-> (reg<test.ptr>) {\n"
            "  %sum = low.op<test.add.i32>(%lhs, %rhs) : "
            "(reg<test.ptr>, reg<test.ptr>) -> reg<test.ptr>\n"
            "  low.return %sum : reg<test.ptr>\n"
            "}\n"
        )

        loaded = _parse_write_read(text, include_low=True)
        assert len(loaded.symbols) == 2
        symbol = loaded.symbols[1]
        assert symbol.kind == SymbolKind.FUNC_DEF
        assert symbol.op is not None
        assert symbol.op.attributes["callee"] == "add"
        assert symbol.op.attributes["target"] == "test_target"
        assert _roundtrip_text_through_bytecode(text, include_low=True) == text

    def test_low_invoke_and_low_function_survive_bytecode(self) -> None:
        text = (
            "test.record @test_target\n\n"
            "func.def @caller(%lhs: i32, %rhs: i32) -> (i32) {\n"
            "  %sum = low.invoke @test_add(%lhs, %rhs) : "
            "(i32, i32) -> (i32)\n"
            "  func.return %sum : i32\n"
            "}\n\n"
            "low.func.decl target(@test_target) @test_add(%lhs: reg<test.ptr>, "
            "%rhs: reg<test.ptr>) -> (reg<test.ptr>)\n"
        )

        loaded = _parse_write_read(text, include_low=True)
        assert len(loaded.symbols) == 3
        low_symbol = loaded.symbols[2]
        assert low_symbol.op is not None
        assert "source" not in low_symbol.op.attributes
        assert _roundtrip_text_through_bytecode(text, include_low=True) == text

    def test_enum_future_ordinal_survives_bytecode(self) -> None:
        text = (
            "target.generic<reference> @future_target "
            "{artifact_format = elf, codegen_format = vm}\n"
        )
        parser = Parser()
        parser.register_ops(ALL_TARGET_OPS)
        module = parser.parse(text)
        target_op = module.symbols[0].op
        assert target_op is not None
        module.symbols[0].op = replace(
            target_op,
            attributes={
                **target_op.attributes,
                "artifact_format": 250,
                "codegen_format": 251,
            },
        )

        loaded = read_module(write_module(module))
        loaded_target = loaded.symbols[0].op
        assert loaded_target is not None
        assert loaded_target.attributes["artifact_format"] == 250
        assert loaded_target.attributes["codegen_format"] == 251

    def test_closed_enum_future_ordinal_fails_loudly(self) -> None:
        module = Module(name="closed_enum")
        lhs = module.add_value(Value(name="lhs", type=I32))
        rhs = module.add_value(Value(name="rhs", type=I32))
        cmp_result = module.add_value(Value(name="cmp", type=I32))
        cmp_op = Operation(
            name="test.cmp",
            operands=[lhs, rhs],
            results=[cmp_result],
            attributes={"predicate": 250},
        )
        yield_op = Operation(name="test.yield", operands=[cmp_result])
        func_op = Operation(
            name="test.func",
            attributes={"callee": "f"},
            regions=[
                Region(blocks=[Block(arg_ids=[lhs, rhs], ops=[cmp_op, yield_op])])
            ],
        )
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        with pytest.raises(ValueError, match="enum attribute ordinal 250"):
            write_module(module)

    def test_enum_bool_ordinal_fails_loudly(self) -> None:
        module = Module(name="enum_bool")
        lhs = module.add_value(Value(name="lhs", type=I32))
        rhs = module.add_value(Value(name="rhs", type=I32))
        cmp_result = module.add_value(Value(name="cmp", type=I32))
        cmp_op = Operation(
            name="test.cmp",
            operands=[lhs, rhs],
            results=[cmp_result],
            attributes={"predicate": True},
        )
        yield_op = Operation(name="test.yield", operands=[cmp_result])
        func_op = Operation(
            name="test.func",
            attributes={"callee": "f"},
            regions=[
                Region(blocks=[Block(arg_ids=[lhs, rhs], ops=[cmp_op, yield_op])])
            ],
        )
        module.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        with pytest.raises(TypeError, match="uint8 ordinal"):
            write_module(module)

    def test_symbol_without_bytecode_payload_kind_fails_loud(self) -> None:
        module = Module()
        module.add_symbol(Symbol(name="opaque", kind=SymbolKind.NONE))

        with pytest.raises(ValueError, match="no bytecode symbol payload kind"):
            write_module(module)

    def test_operand_dict_survives_bytecode(self) -> None:
        text = (
            "func.def @operand_dict(%input: f32, %beta: f32, %alpha: i32) "
            "-> (f32) {\n"
            "  %result = test.operand_dict %input "
            "{beta = %beta : f32, alpha = %alpha : i32} : f32\n"
            "  test.yield %result : f32\n"
            "}\n"
        )
        expected = (
            "func.def @operand_dict(%input: f32, %beta: f32, %alpha: i32) "
            "-> (f32) {\n"
            "  %result = test.operand_dict %input "
            "{alpha = %alpha : i32, beta = %beta : f32} : f32\n"
            "  test.yield %result : f32\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text) == expected

    def test_encoding_define_dynamic_params_survive_bytecode(self) -> None:
        text = (
            "func.def @encoding_params(%group_size: index, %scale: f32) -> () {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{scale = %scale : f32, group_size = %group_size : index} : "
            "encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        expected = (
            "func.def @encoding_params(%group_size: index, %scale: f32) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{group_size = %group_size : index, scale = %scale : f32} : "
            "encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_encoding=True) == expected

    def test_vector_encode_decode_auxiliary_operands_survive_bytecode(self) -> None:
        text = (
            "func.def @vector_schema_aux(%payload: vector<4xi32>, "
            "%scale: vector<1xf16>, %values: vector<32xf32>, "
            "%amax: vector<1xf32>) -> (vector<32xf32>, vector<4xi32>) {\n"
            "  %schema = encoding.define #ggml_q4_0<block_elems=32, "
            "storage_bytes=18> : encoding<schema>\n"
            "  %decoded = vector.decode %payload using %schema "
            "{scale = %scale : vector<1xf16>} : vector<4xi32>, "
            "encoding<schema> -> vector<32xf32>\n"
            "  %encoded = vector.encode %values using %schema "
            "{scale = %scale : vector<1xf16>, amax = %amax : vector<1xf32>} "
            ": vector<32xf32>, encoding<schema> -> vector<4xi32>\n"
            "  func.return %decoded, %encoded : vector<32xf32>, vector<4xi32>\n"
            "}\n"
        )
        expected = (
            "func.def @vector_schema_aux(%payload: vector<4xi32>, "
            "%scale: vector<1xf16>, %values: vector<32xf32>, "
            "%amax: vector<1xf32>) -> (vector<32xf32>, vector<4xi32>) {\n"
            "  %schema = encoding.define #ggml_q4_0<block_elems=32, "
            "storage_bytes=18> : encoding<schema>\n"
            "  %decoded = vector.decode %payload using %schema "
            "{scale = %scale : vector<1xf16>} : vector<4xi32>, "
            "encoding<schema> -> vector<32xf32>\n"
            "  %encoded = vector.encode %values using %schema "
            "{amax = %amax : vector<1xf32>, scale = %scale : vector<1xf16>} "
            ": vector<32xf32>, encoding<schema> -> vector<4xi32>\n"
            "  func.return %decoded, %encoded : vector<32xf32>, vector<4xi32>\n"
            "}\n"
        )

        assert (
            _roundtrip_text_through_bytecode(
                text,
                include_encoding=True,
                include_vector=True,
            )
            == expected
        )

    def test_global_symbols_survive_bytecode(self) -> None:
        text = (
            "global.constant @answer : index = 42\n"
            "\n"
            "func.def @load_answer() -> (index) {\n"
            "  %value = global.load @answer : index\n"
            "  func.return %value : index\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_global=True) == text

    def test_dynamic_global_symbol_values_survive_bytecode(self) -> None:
        text = (
            "global.constant @weights : tile<[%n]xf32> where [mul(%n, 16)]\n"
            "\n"
            "func.def @load_weights() -> (index) {\n"
            "  %tile, %m = global.load @weights : tile<[%m]xf32>\n"
            "  test.use %tile : tile<[%m]xf32>\n"
            "  func.return %m : index\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_global=True) == text

    def test_static_encoding_global_symbols_survive_bytecode(self) -> None:
        text = (
            "global.constant @weights : tile<[%m]x[%k]xi8, #q8_0<block=32>> "
            "where [mul(%m, 16)]\n"
            "\n"
            "func.def @load_weights() {\n"
            "  %tile, %m, %k = global.load @weights : "
            "tile<[%m]x[%k]xi8, #q8_0<block=32>>\n"
            "  test.use %tile, %m, %k : "
            "tile<[%m]x[%k]xi8, #q8_0<block=32>>, index, index\n"
            "  func.return\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_global=True) == text

    def test_global_variable_store_survives_bytecode(self) -> None:
        text = (
            "global.variable @state : f32 = 0.0\n"
            "\n"
            "func.def @set_state(%value: f32) {\n"
            "  global.store %value, @state : f32\n"
            "  func.return\n"
            "}\n"
            "\n"
            "func.def @load_state() -> (f32) {\n"
            "  %value = global.load @state : f32\n"
            "  func.return %value : f32\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_global=True) == text

    def test_ssa_encoding_global_symbols_survive_bytecode(self) -> None:
        text = (
            "global.variable @scratch_view : view<[%n]xf32, %layout> "
            "where [mul(%n, 16)]\n"
            "\n"
            "func.def @roundtrip_scratch_view() {\n"
            "  %view, %n, %layout = global.load @scratch_view : "
            "view<[%n]xf32, %layout>\n"
            "  test.use %view, %n, %layout : view<[%n]xf32, %layout>, index, "
            "encoding\n"
            "  global.store %view, @scratch_view : view<[%n]xf32, %layout>\n"
            "  func.return\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_global=True) == text

    def test_vector_iota_and_range_mask_survive_bytecode(self) -> None:
        text = (
            "func.def @vector_coords(%lo: index, %hi: index, %step: index, "
            "%base: i32, %stride: i32) {\n"
            "  %lanes = vector.iota %lo step %step : vector<16xindex>\n"
            "  %mask = vector.mask.range [%lo to %hi step %step] : index -> "
            "vector<16xi1>\n"
            "  %ints = vector.iota %base step %stride : vector<[%hi]xi32>\n"
            "  test.use %lanes, %mask, %ints : vector<16xindex>, "
            "vector<16xi1>, vector<[%hi]xi32>\n"
            "  func.return\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_vector=True) == text

    def test_kernel_cluster_gather_ops_survive_bytecode(self) -> None:
        text = (
            "func.def @async_cluster_gather_bytecode(%source_buffer: buffer, "
            "%source_offset: offset, %in_bounds: i1) {\n"
            "  %zero = index.constant 0 : offset\n"
            "  %bytes = index.constant 1024 : offset\n"
            "  %layout = encoding.layout.dense : encoding<layout>\n"
            "  %cluster_mask = scalar.constant 15 : i32\n"
            "  %source_global = buffer.assume.memory_space<global> "
            "%source_buffer : buffer\n"
            "  %source = buffer.view %source_global[%source_offset] : buffer -> "
            "view<16xi8, %layout>\n"
            "  %scratch = buffer.alloca %bytes {base_alignment = 64, "
            "memory_space = workgroup} : buffer\n"
            "  %dest = buffer.view %scratch[%zero] : buffer -> view<16xi8, %layout>\n"
            "  %copy = kernel.async.cluster.gather %source to %dest using "
            "%cluster_mask {cache_scope = se, cache_temporal = high_temporal} : "
            "view<16xi8, %layout> to view<16xi8, %layout>, i32 -> "
            "kernel.async.token\n"
            "  %masked = kernel.async.cluster.gather.mask %source to %dest using "
            "%cluster_mask, %in_bounds {cache_scope = cu, cache_temporal = regular} "
            ": view<16xi8, %layout> to view<16xi8, %layout>, i32, i1 -> "
            "kernel.async.token\n"
            "  %group = kernel.async.group %copy, %masked : kernel.async.token, "
            "kernel.async.token -> kernel.async.group\n"
            "  kernel.async.wait %group {newer_groups = 0} : kernel.async.group\n"
            "  func.return\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_kernel=True) == text

    def test_kernel_tensor_lds_ops_survive_bytecode(self) -> None:
        text = (
            "func.def @async_tensor_lds_bytecode(%source_buffer: buffer, "
            "%dest_buffer: buffer, %source_offset: offset, %dest_offset: offset) {\n"
            "  %zero = index.constant 0 : offset\n"
            "  %bytes = index.constant 16384 : offset\n"
            "  %layout = encoding.layout.dense : encoding<layout>\n"
            "  %i32_zero = scalar.constant 0 : i32\n"
            "  %d0 = vector.splat %i32_zero : vector<4xi32>\n"
            "  %d1 = vector.splat %i32_zero : vector<8xi32>\n"
            "  %d2 = vector.splat %i32_zero : vector<4xi32>\n"
            "  %d3 = vector.splat %i32_zero : vector<4xi32>\n"
            "  %desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1, %d2, %d3) "
            ": vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> -> "
            "kernel.tensor.lds.descriptor\n"
            "  %source_global = buffer.assume.memory_space<global> "
            "%source_buffer : buffer\n"
            "  %source = buffer.view %source_global[%source_offset] : buffer -> "
            "view<64x64xf32, %layout>\n"
            "  %scratch = buffer.alloca %bytes {base_alignment = 256, "
            "memory_space = workgroup} : buffer\n"
            "  %lds = buffer.view %scratch[%zero] : buffer -> "
            "view<64x64xf32, %layout>\n"
            "  %load = kernel.async.tensor.load.to.lds %source to %lds using "
            "%desc {cache_scope = cu, cache_temporal = regular} : "
            "view<64x64xf32, %layout> to view<64x64xf32, %layout>, "
            "kernel.tensor.lds.descriptor -> kernel.async.token\n"
            "  %load_group = kernel.async.group %load : kernel.async.token -> "
            "kernel.async.group\n"
            "  kernel.async.wait %load_group {newer_groups = 0} : "
            "kernel.async.group\n"
            "  %dest_global = buffer.assume.memory_space<global> "
            "%dest_buffer : buffer\n"
            "  %dest = buffer.view %dest_global[%dest_offset] : buffer -> "
            "view<64x64xf32, %layout>\n"
            "  %store = kernel.async.tensor.store.from.lds %lds to %dest using "
            "%desc {cache_scope = device, cache_temporal = non_temporal_writeback} "
            ": view<64x64xf32, %layout> to view<64x64xf32, %layout>, "
            "kernel.tensor.lds.descriptor -> kernel.async.token\n"
            "  %store_group = kernel.async.group %store : kernel.async.token -> "
            "kernel.async.group\n"
            "  kernel.async.wait %store_group {newer_groups = 0} : "
            "kernel.async.group\n"
            "  func.return\n"
            "}\n"
        )

        assert _roundtrip_text_through_bytecode(text, include_kernel=True) == text

    def test_attr_dict_parser_programmatic_and_readback_converge(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.func import ALL_FUNC_OPS
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.bytecode.reader import read_module as read
        from loom.format.text.parser import Parser

        text = (
            "func.def @f(%x: f32) -> (f32) {\n"
            '  %r = test.attrs %x {meta = {phase = "link", opt = 3}, axis = 0}'
            " : f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )

        parser = Parser()
        parser.register_ops(list(ALL_FUNC_OPS) + list(ALL_TEST_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        parsed_module = parser.parse(text)
        _strip_locations(parsed_module)

        programmatic_module = _make_attrs_module(
            {"meta": {"phase": "link", "opt": 3}, "axis": 0}
        )

        parsed_bytes = write_module(parsed_module)
        programmatic_bytes = write_module(programmatic_module)
        assert parsed_bytes == programmatic_bytes

        parsed_attrs = _first_attrs_dict(parsed_module)
        programmatic_attrs = _first_attrs_dict(programmatic_module)
        loaded_module = read(parsed_bytes)
        loaded_attrs = _first_attrs_dict(loaded_module)

        assert parsed_attrs == programmatic_attrs == loaded_attrs
        assert hash(parsed_attrs) == hash(programmatic_attrs) == hash(loaded_attrs)
        assert list(loaded_attrs.items()) == [
            ("axis", 0),
            ("meta", CanonicalAttrDict([("opt", 3), ("phase", "link")])),
        ]

    def test_attr_dict_symbol_ref_survives_bytecode(self) -> None:
        from loom.builtin_types import ALL_BUILTIN_TYPES
        from loom.dialect.func import ALL_FUNC_OPS
        from loom.dialect.test import ALL_TEST_OPS
        from loom.format.bytecode.reader import read_module as read
        from loom.format.text.parser import Parser

        text = (
            "func.def @f(%x: f32) -> (f32) {\n"
            "  %r = test.attrs %x {target = @target} : f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )

        parser = Parser()
        parser.register_ops(list(ALL_FUNC_OPS) + list(ALL_TEST_OPS))
        parser.register_types(ALL_BUILTIN_TYPES)
        parsed_module = parser.parse(text)
        loaded_module = read(write_module(parsed_module))
        loaded_attrs = _first_attrs_dict(loaded_module)

        assert loaded_attrs["target"] == SymbolName("target")
        assert isinstance(loaded_attrs["target"], SymbolName)


class TestPredicateBytecodeRoundTrip:
    """Predicate serialization: write → read → verify."""

    def test_function_predicates_roundtrip(self) -> None:
        """Function predicates survive bytecode round-trip."""
        from loom.format.bytecode.reader import read_module as read
        from loom.ir import (
            Module,
            Operation,
            Predicate,
            PredicateArg,
            Symbol,
            SymbolKind,
            Value,
        )

        module = Module(name="test")
        predicates = [
            Predicate(
                kind="mul",
                args=(
                    PredicateArg(tag="value", value="M"),
                    PredicateArg(tag="const", value=16),
                ),
            ),
            Predicate(
                kind="lt",
                args=(
                    PredicateArg(tag="value", value="K"),
                    PredicateArg(tag="const", value=1024),
                ),
            ),
            Predicate(
                kind="ne",
                args=(
                    PredicateArg(tag="value", value="N"),
                    PredicateArg(tag="const", value=0),
                ),
            ),
            Predicate(
                kind="pow2",
                args=(PredicateArg(tag="value", value="N"),),
            ),
            Predicate(
                kind="range",
                args=(
                    PredicateArg(tag="value", value="M"),
                    PredicateArg(tag="const", value=32),
                    PredicateArg(tag="const", value=512),
                ),
            ),
        ]
        m_id = module.add_value(Value(name="M", type=INDEX))
        k_id = module.add_value(Value(name="K", type=INDEX))
        n_id = module.add_value(Value(name="N", type=INDEX))
        arg_id = module.add_value(Value(name="", type=F32))
        result_id = module.add_value(Value(name="", type=F32))
        func_op = Operation(
            name="func.decl",
            operands=[m_id, k_id, n_id, arg_id],
            results=[result_id],
            attributes={"callee": "test", "predicates": predicates},
        )
        module.add_symbol(Symbol(name="test", kind=SymbolKind.FUNC_DECL, op=func_op))

        bc_data = write_module(module)
        loaded = read(bc_data)

        assert len(loaded.symbols) == 1
        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        loaded_preds = loaded_op.attributes.get("predicates", [])
        assert len(loaded_preds) == 5

        # Verify each predicate survived.
        assert loaded_preds[0].kind == "mul"
        assert len(loaded_preds[0].args) == 2
        assert loaded_preds[0].args[0].tag == "value"
        assert loaded_preds[0].args[0].value == "M"
        assert loaded_preds[0].args[1].tag == "const"
        assert loaded_preds[0].args[1].value == 16

        assert loaded_preds[1].kind == "lt"
        assert loaded_preds[1].args[1].value == 1024

        assert loaded_preds[2].kind == "ne"
        assert loaded_preds[2].args[1].value == 0

        assert loaded_preds[3].kind == "pow2"
        assert len(loaded_preds[3].args) == 1

        assert loaded_preds[4].kind == "range"
        assert len(loaded_preds[4].args) == 3

    def test_empty_predicates_roundtrip(self) -> None:
        """Function with no predicates survives bytecode round-trip."""
        from loom.format.bytecode.reader import read_module as read
        from loom.ir import (
            Module,
            Operation,
            Symbol,
            SymbolKind,
            Value,
        )

        module = Module(name="test")
        arg_id = module.add_value(Value(name="", type=F32))
        result_id = module.add_value(Value(name="", type=F32))
        func_op = Operation(
            name="func.decl",
            operands=[arg_id],
            results=[result_id],
            attributes={"callee": "no_preds"},
        )
        module.add_symbol(
            Symbol(name="no_preds", kind=SymbolKind.FUNC_DECL, op=func_op)
        )

        bc_data = write_module(module)
        loaded = read(bc_data)

        loaded_op = loaded.symbols[0].op
        assert loaded_op is not None
        assert loaded_op.attributes.get("predicates", []) == []


# ============================================================================
# Import/export bytecode round-trips
# ============================================================================


class TestImportExportBytecodeRoundTrip:
    """Import/export metadata survives bytecode serialization."""

    def test_import_declaration_roundtrip(self) -> None:
        """Import declaration with source module survives round-trip."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        operand_ids = [module.add_value(Value(name="", type=at)) for at in [F32, F32]]
        result_id = module.add_value(Value(name="", type=F32))
        func_op = Operation(
            name="func.decl",
            operands=operand_ids,
            results=[result_id],
            attributes={"callee": "matmul"},
        )
        module.add_symbol(
            Symbol(
                name="matmul",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=func_op,
                source_module="linalg_lib",
            )
        )
        loaded = read(write_module(module))
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.source_module == "linalg_lib"
        # source_symbol defaults to name when empty.
        assert sym.source_symbol == "matmul"

    def test_import_with_alias_roundtrip(self) -> None:
        """Import with aliasing: local name differs from source symbol."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        arg_id = module.add_value(Value(name="", type=F32))
        result_id = module.add_value(Value(name="", type=F32))
        func_op = Operation(
            name="func.decl",
            operands=[arg_id],
            results=[result_id],
            attributes={"callee": "my_matmul"},
        )
        module.add_symbol(
            Symbol(
                name="my_matmul",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=func_op,
                source_module="linalg_lib",
                source_symbol="matmul",
            )
        )
        loaded = read(write_module(module))
        sym = loaded.symbols[0]
        assert sym.name == "my_matmul"
        assert sym.is_import
        assert sym.source_module == "linalg_lib"
        assert sym.source_symbol == "matmul"

    def test_public_import_roundtrip(self) -> None:
        """Import can also be public (re-export)."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        arg_id = module.add_value(Value(name="", type=F32))
        result_id = module.add_value(Value(name="", type=F32))
        func_op = Operation(
            name="func.decl",
            operands=[arg_id],
            results=[result_id],
            attributes={"callee": "relu", "visibility": "public"},
        )
        module.add_symbol(
            Symbol(
                name="relu",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_PUBLIC | SYMBOL_FLAG_IMPORT,
                op=func_op,
                source_module="activations",
            )
        )
        loaded = read(write_module(module))
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.is_public
        assert sym.source_module == "activations"

    def test_export_offset_table(self) -> None:
        """Public symbols appear in the export offset table."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        # Private function.
        vid = module.add_value(Value(name="x", type=F32))
        body = Region(
            blocks=[
                Block(arg_ids=[vid], ops=[Operation(name="test.yield", operands=[vid])])
            ]
        )
        helper_op = Operation(
            name="func.def", attributes={"callee": "helper"}, regions=[body]
        )
        module.add_symbol(
            Symbol(name="helper", kind=SymbolKind.FUNC_DEF, flags=0, op=helper_op)
        )
        # Public function.
        vid2 = module.add_value(Value(name="y", type=F32))
        body2 = Region(
            blocks=[
                Block(
                    arg_ids=[vid2], ops=[Operation(name="test.yield", operands=[vid2])]
                )
            ]
        )
        entry_op = Operation(
            name="func.def",
            attributes={"callee": "entry", "visibility": "public"},
            regions=[body2],
        )
        module.add_symbol(
            Symbol(
                name="entry",
                kind=SymbolKind.FUNC_DEF,
                flags=SYMBOL_FLAG_PUBLIC,
                op=entry_op,
            )
        )
        # Round-trip: both survive, visibility preserved.
        loaded = read(write_module(module))
        assert len(loaded.symbols) == 2
        names = {s.name: s for s in loaded.symbols}
        assert not names["helper"].is_public
        assert names["entry"].is_public

    def test_mixed_import_export_private(self) -> None:
        """Module with imports, exports, and private symbols."""
        from loom.format.bytecode.reader import read_module as read

        module = Module(name="test")
        # Import.
        import_operands = [
            module.add_value(Value(name="", type=at)) for at in [F32, F32]
        ]
        import_result = module.add_value(Value(name="", type=F32))
        import_op = Operation(
            name="func.decl",
            operands=import_operands,
            results=[import_result],
            attributes={"callee": "extern_add"},
        )
        module.add_symbol(
            Symbol(
                name="extern_add",
                kind=SymbolKind.FUNC_DECL,
                flags=SYMBOL_FLAG_IMPORT,
                op=import_op,
                source_module="math_lib",
            )
        )
        # Private.
        vid = module.add_value(Value(name="x", type=F32))
        helper_op = Operation(
            name="func.def",
            attributes={"callee": "helper"},
            regions=[
                Region(
                    blocks=[
                        Block(
                            arg_ids=[vid],
                            ops=[Operation(name="test.yield", operands=[vid])],
                        )
                    ]
                )
            ],
        )
        module.add_symbol(
            Symbol(name="helper", kind=SymbolKind.FUNC_DEF, flags=0, op=helper_op)
        )
        # Export.
        vid2 = module.add_value(Value(name="y", type=F32))
        entry_op = Operation(
            name="func.def",
            attributes={"callee": "entry", "visibility": "public"},
            regions=[
                Region(
                    blocks=[
                        Block(
                            arg_ids=[vid2],
                            ops=[Operation(name="test.yield", operands=[vid2])],
                        )
                    ]
                )
            ],
        )
        module.add_symbol(
            Symbol(
                name="entry",
                kind=SymbolKind.FUNC_DEF,
                flags=SYMBOL_FLAG_PUBLIC,
                op=entry_op,
            )
        )
        loaded = read(write_module(module))
        assert len(loaded.symbols) == 3
        syms = {s.name: s for s in loaded.symbols}
        assert syms["extern_add"].is_import
        assert syms["extern_add"].source_module == "math_lib"
        assert not syms["helper"].is_import
        assert not syms["helper"].is_public
        assert syms["entry"].is_public
        assert not syms["entry"].is_import

    def test_no_imports_no_exports(self) -> None:
        """Module with only private symbols has empty offset tables."""
        from loom.format.bytecode.reader import read_module as read

        module = _make_func_module()
        loaded = read(write_module(module))
        for sym in loaded.symbols:
            assert not sym.is_import
            assert not sym.is_public
