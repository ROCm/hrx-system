# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the format-driven parser — name scope, type parsing, round-trip."""

import math
from typing import Any

import pytest

from loom.assembly import TypeOf
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import ANY, TypeDef, TypeParam
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import (
    NameScope,
    Parser,
    parse_type_string,
)
from loom.format.text.printer import Printer, print_type
from loom.format.text.tokenizer import ParseError
from loom.ir import (
    BF16,
    BUFFER_TYPE,
    ENCODING_LAYOUT_TYPE,
    ENCODING_SCHEMA_TYPE,
    ENCODING_STORAGE_TYPE,
    ENCODING_TRANSFORM_TYPE,
    ENCODING_TYPE,
    F32,
    I8,
    I32,
    I64,
    INDEX,
    OFFSET,
    VALUE_DEF_BLOCK_NONE,
    VALUE_DEF_OP_NONE,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingRole,
    EncodingType,
    FunctionType,
    GroupScope,
    GroupType,
    Module,
    Operation,
    RegisterType,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    StorageSpace,
    StorageType,
    SymbolName,
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


def _parse(text: str, **kwargs: Any) -> tuple[Type, dict[int, int]]:
    """Parse a type string, returning (type, dim_bindings)."""
    result: tuple[Type, dict[int, int]] = parse_type_string(text, **kwargs)
    return result


def _parse_type(text: str, **kwargs: Any) -> Type:
    """Parse a type string, returning just the type."""
    return _parse(text, **kwargs)[0]


def _test_ptr_register_type(unit_count: int = 1) -> RegisterType:
    return RegisterType(
        _TEST_LOW_CORE_STABLE_ID,
        _TEST_PTR_REGISTER_CLASS_ID,
        unit_count,
        "test.ptr",
    )


# ============================================================================
# Name scope
# ============================================================================


class TestNameScope:
    def test_define_and_lookup(self) -> None:
        scope = NameScope()
        scope.define("x", 0)
        assert scope.lookup("x") == 0

    def test_multiple_definitions(self) -> None:
        scope = NameScope()
        scope.define("a", 0)
        scope.define("b", 1)
        scope.define("c", 2)
        assert scope.lookup("a") == 0
        assert scope.lookup("b") == 1
        assert scope.lookup("c") == 2

    def test_redefinition_fails(self) -> None:
        scope = NameScope()
        scope.define("x", 0)
        with pytest.raises(ValueError, match="already defined"):
            scope.define("x", 1)

    def test_undefined_fails(self) -> None:
        scope = NameScope()
        with pytest.raises(KeyError, match="undefined"):
            scope.lookup("missing")

    def test_child_sees_parent(self) -> None:
        parent = NameScope()
        parent.define("x", 0)
        child = parent.push()
        assert child.lookup("x") == 0

    def test_parent_does_not_see_child(self) -> None:
        parent = NameScope()
        child = parent.push()
        child.define("inner", 42)
        with pytest.raises(KeyError, match="undefined"):
            parent.lookup("inner")

    def test_child_shadows_parent(self) -> None:
        parent = NameScope()
        parent.define("x", 0)
        child = parent.push()
        child.define("x", 99)
        assert child.lookup("x") == 99
        assert parent.lookup("x") == 0

    def test_grandchild_sees_grandparent(self) -> None:
        root = NameScope()
        root.define("global", 0)
        child = root.push()
        grandchild = child.push()
        assert grandchild.lookup("global") == 0

    def test_sibling_scopes_isolated(self) -> None:
        parent = NameScope()
        child_a = parent.push()
        child_b = parent.push()
        child_a.define("only_in_a", 1)
        child_b.define("only_in_b", 2)
        with pytest.raises(KeyError):
            child_b.lookup("only_in_a")
        with pytest.raises(KeyError):
            child_a.lookup("only_in_b")


# ============================================================================
# Scalar types
# ============================================================================


class TestParseScalarTypes:
    def test_f32(self) -> None:
        assert _parse_type("f32") == F32

    def test_i32(self) -> None:
        assert _parse_type("i32") == I32

    def test_index(self) -> None:
        assert _parse_type("index") == INDEX

    def test_offset(self) -> None:
        assert _parse_type("offset") == OFFSET

    def test_bf16(self) -> None:
        assert _parse_type("bf16") == BF16

    def test_i8(self) -> None:
        assert _parse_type("i8") == I8

    def test_i1(self) -> None:
        assert _parse_type("i1") == ScalarType(ScalarTypeKind.I1)

    def test_i64(self) -> None:
        assert _parse_type("i64") == I64

    def test_f16(self) -> None:
        assert _parse_type("f16") == ScalarType(ScalarTypeKind.F16)

    def test_f64(self) -> None:
        assert _parse_type("f64") == ScalarType(ScalarTypeKind.F64)

    def test_f8E4M3(self) -> None:
        assert _parse_type("f8E4M3") == ScalarType(ScalarTypeKind.F8E4M3)

    def test_f8E5M2(self) -> None:
        assert _parse_type("f8E5M2") == ScalarType(ScalarTypeKind.F8E5M2)

    def test_unknown_scalar_fails(self) -> None:
        with pytest.raises(ParseError, match="expected type"):
            _parse_type("f128")


# ============================================================================
# Shaped types — static dims
# ============================================================================


class TestParseShapedStatic:
    def test_tile_1d(self) -> None:
        result = _parse_type("tile<4xf32>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.TILE
        assert result.element_type == F32
        assert result.rank == 1
        assert result.dims == (StaticDim(4),)

    def test_tile_2d(self) -> None:
        result = _parse_type("tile<4x4xf32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 2
        assert result.dims == (StaticDim(4), StaticDim(4))

    def test_tensor_1d(self) -> None:
        result = _parse_type("tensor<256xi8>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.TENSOR
        assert result.element_type == I8
        assert result.dims == (StaticDim(256),)

    def test_vector_1d(self) -> None:
        result = _parse_type("vector<16xf32>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VECTOR
        assert result.element_type == F32
        assert result.dims == (StaticDim(16),)

    def test_vector_zero_extent(self) -> None:
        result = _parse_type("vector<0xf32>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VECTOR
        assert result.element_type == F32
        assert result.dims == (StaticDim(0),)

    def test_vector_2d(self) -> None:
        result = _parse_type("vector<4x16xf32>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VECTOR
        assert result.dims == (StaticDim(4), StaticDim(16))

    def test_view_1d(self) -> None:
        result = _parse_type("view<256xi8>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VIEW
        assert result.element_type == I8
        assert result.dims == (StaticDim(256),)

    def test_tile_0d(self) -> None:
        result = _parse_type("tile<f32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 0
        assert result.element_type == F32

    def test_vector_0d_fails(self) -> None:
        with pytest.raises(ParseError, match="rank >= 1"):
            _parse_type("vector<f32>")

    def test_large_dims(self) -> None:
        result = _parse_type("tile<1024x2048xf16>")
        assert isinstance(result, ShapedType)
        assert result.dims == (StaticDim(1024), StaticDim(2048))

    def test_3d(self) -> None:
        result = _parse_type("tensor<2x3x4xi32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 3
        assert result.dims == (StaticDim(2), StaticDim(3), StaticDim(4))


# ============================================================================
# Shaped types — dynamic dims
# ============================================================================


class TestParseDynamicDims:
    def test_single_dynamic(self) -> None:
        scope = NameScope()
        module = Module()
        dim_id = module.add_value(Value(name="M", type=INDEX))
        scope.define("M", dim_id)

        result, bindings = _parse("tile<[%M]x4xf32>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.dims == (DynamicDim(), StaticDim(4))
        assert bindings[0] == dim_id

    def test_all_dynamic(self) -> None:
        scope = NameScope()
        module = Module()
        m_id = module.add_value(Value(name="M", type=INDEX))
        k_id = module.add_value(Value(name="K", type=INDEX))
        scope.define("M", m_id)
        scope.define("K", k_id)

        result, bindings = _parse("tensor<[%M]x[%K]xf32>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.dims == (DynamicDim(), DynamicDim())
        assert bindings[0] == m_id
        assert bindings[1] == k_id

    def test_dynamic_vector(self) -> None:
        scope = NameScope()
        module = Module()
        n_id = module.add_value(Value(name="N", type=INDEX))
        scope.define("N", n_id)

        result, bindings = _parse("vector<[%N]xi32>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VECTOR
        assert result.dims == (DynamicDim(),)
        assert bindings[0] == n_id

    def test_undefined_dim_fails(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value"):
            _parse("tile<[%M]xf32>")


# ============================================================================
# Shaped types — with encoding
# ============================================================================


class TestParseEncoding:
    def test_simple_encoding(self) -> None:
        result = _parse_type("tile<256xi8, #q8_0>")
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        assert isinstance(result.encoding, EncodingInstance)
        assert result.encoding.name == "q8_0"

    def test_encoding_with_params(self) -> None:
        module = Module()
        result, _ = _parse("tile<256xi8, #q8_0<block=32>>", module=module)
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        assert len(module.encodings) == 1
        assert module.encodings[0].name == "q8_0"
        assert module.encodings[0].params == (("block", 32),)

    def test_encoding_dedup(self) -> None:
        module = Module()
        _parse("tile<128xi8, #q8_0<block=32>>", module=module)
        _parse("tile<256xi8, #q8_0<block=32>>", module=module)
        assert len(module.encodings) == 1  # Same encoding, deduplicated.

    def test_different_encodings(self) -> None:
        module = Module()
        _parse("tile<128xi8, #q8_0>", module=module)
        _parse("tile<256xi8, #q6_k>", module=module)
        assert len(module.encodings) == 2

    def test_malformed_encoding_params_gives_parse_error(self) -> None:
        """Malformed params raise ParseError, not bare ValueError."""
        from loom.format.text.tokenizer import ParseError as TokenError

        with pytest.raises((ParseError, TokenError), match="expected EQUALS"):
            _parse("tile<256xi8, #q8_0<block32>>")

    def test_encoding_multiple_params(self) -> None:
        module = Module()
        result, _ = _parse("tile<256xi8, #q8_0<block=32, group=128>>", module=module)
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        enc = module.encodings[0]
        assert enc.params == (("block", 32), ("group", 128))

    def test_view_static_layout(self) -> None:
        module = Module()
        result, _ = _parse("view<256xf32, #strided<stride=64>>", module=module)
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VIEW
        assert result.has_encoding
        assert len(module.encodings) == 1
        assert module.encodings[0].name == "strided"
        assert module.encodings[0].params == (("stride", 64),)

    def test_view_dynamic_layout(self) -> None:
        scope = NameScope()
        module = Module()
        n_id = module.add_value(Value(name="N", type=INDEX))
        layout_id = module.add_value(Value(name="layout", type=ENCODING_TYPE))
        scope.define("N", n_id)
        scope.define("layout", layout_id)

        result, bindings = _parse("view<[%N]xf32, %layout>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.VIEW
        assert result.encoding == DynamicEncoding()
        assert bindings[0] == n_id
        assert bindings[-1] == layout_id

    def test_vector_encoding_fails(self) -> None:
        with pytest.raises(ParseError, match="must not carry"):
            _parse("vector<4xf32, #dense>")


# ============================================================================
# Group types
# ============================================================================


class TestParseGroupType:
    def test_basic(self) -> None:
        result = _parse_type("group<workgroup>")
        assert isinstance(result, GroupType)
        assert result.scope == GroupScope.WORKGROUP

    def test_subgroup(self) -> None:
        result = _parse_type("group<subgroup>")
        assert isinstance(result, GroupType)
        assert result.scope == GroupScope.SUBGROUP


# ============================================================================
# Buffer type
# ============================================================================


class TestParseBufferType:
    def test_basic(self) -> None:
        result = _parse_type("buffer")
        assert result is BUFFER_TYPE


# ============================================================================
# Storage type
# ============================================================================


class TestParseStorageType:
    def test_workgroup(self) -> None:
        assert _parse_type("low.storage<workgroup>") == StorageType(
            StorageSpace.WORKGROUP
        )

    def test_private(self) -> None:
        assert _parse_type("low.storage<private>") == StorageType(StorageSpace.PRIVATE)

    def test_unknown_space_fails(self) -> None:
        with pytest.raises(ParseError, match="storage space"):
            _parse_type("low.storage<lds>")


# ============================================================================
# Register type
# ============================================================================


class TestParseRegisterType:
    def test_single_unit(self) -> None:
        assert _parse_type("reg<test.ptr>") == _test_ptr_register_type()

    def test_multiple_units(self) -> None:
        assert _parse_type("reg<test.ptr x4>") == _test_ptr_register_type(4)

    def test_multiple_units_with_spaced_suffix(self) -> None:
        assert _parse_type("reg<test.ptr x 4>") == _test_ptr_register_type(4)

    def test_requires_namespace(self) -> None:
        with pytest.raises(ParseError, match="expected OP_NAME"):
            _parse_type("reg<vgpr>")

    def test_rejects_zero_units(self) -> None:
        with pytest.raises(ParseError, match="unit count"):
            _parse_type("reg<test.ptr x0>")


# ============================================================================
# Function types
# ============================================================================


class TestParseFunctionType:
    def test_basic(self) -> None:
        result = _parse_type("(f32, i32) -> (f32)")
        assert isinstance(result, FunctionType)
        assert result.arg_types == (F32, I32)
        assert result.result_types == (F32,)

    def test_empty_args(self) -> None:
        result = _parse_type("() -> (f32)")
        assert isinstance(result, FunctionType)
        assert result.arg_types == ()

    def test_empty_results(self) -> None:
        result = _parse_type("(f32) -> ()")
        assert isinstance(result, FunctionType)
        assert result.result_types == ()

    def test_multiple_results(self) -> None:
        result = _parse_type("(f32) -> (i32, f32)")
        assert isinstance(result, FunctionType)
        assert result.result_types == (I32, F32)


# ============================================================================
# Custom dialect types (via type registry)
# ============================================================================


class TestParseDialectTypes:
    def _registry(self) -> dict[str, TypeDef]:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        # Add custom types.
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        registry["hal.fence"] = TypeDef(name="hal.fence")
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        return registry

    def test_opaque_type(self) -> None:
        result = _parse_type("hal.buffer", type_registry=self._registry())
        assert isinstance(result, DialectType)
        assert result.name == "hal.buffer"
        assert result.params == ()

    def test_parameterized_type(self) -> None:
        result = _parse_type("vm.ref<hal.buffer>", type_registry=self._registry())
        assert isinstance(result, DialectType)
        assert result.name == "vm.ref"
        assert len(result.params) == 1
        assert isinstance(result.params[0], DialectType)
        assert result.params[0].name == "hal.buffer"

    def test_nested_parameterized(self) -> None:
        from loom.assembly import TypeOf
        from loom.dsl import ANY, TypeDef, TypeParam

        registry = self._registry()
        registry["vm.list"] = TypeDef(
            name="vm.list",
            params=[TypeParam("element", ANY)],
            format=[TypeOf("element")],
        )
        result = _parse_type("vm.ref<vm.list<i32>>", type_registry=registry)
        assert isinstance(result, DialectType)
        assert result.name == "vm.ref"
        inner = result.params[0]
        assert isinstance(inner, DialectType)
        assert inner.name == "vm.list"
        assert inner.params[0] == I32

    def test_unknown_dotted_type_is_opaque_dialect_type(self) -> None:
        result = _parse_type("hal.unknown", type_registry=self._registry())
        assert isinstance(result, DialectType)
        assert result.name == "hal.unknown"
        assert result.params == ()

    def test_unknown_bare_type_fails(self) -> None:
        with pytest.raises(ParseError, match="expected type"):
            _parse_type("unknown", type_registry=self._registry())


# ============================================================================
# Type round-trip: print(parse(text)) == text
# ============================================================================


class TestTypeRoundTrip:
    def _roundtrip(self, text: str, **kwargs: Any) -> None:
        """Parse then print, assert identical."""
        parsed_type, _ = _parse(text, **kwargs)
        printed = print_type(parsed_type)
        assert printed == text, f"Round-trip failed: {text!r} -> {printed!r}"

    def test_scalar_roundtrip(self) -> None:
        for name in ["f32", "i32", "index", "bf16", "i8", "i64", "f16", "f64"]:
            self._roundtrip(name)

    def test_shaped_roundtrip(self) -> None:
        self._roundtrip("tile<4xf32>")
        self._roundtrip("tile<4x4xf32>")
        self._roundtrip("tensor<256xi8>")
        self._roundtrip("vector<16xf32>")
        self._roundtrip("vector<4x16xf32>")
        self._roundtrip("view<256xi8>")
        self._roundtrip("tile<f32>")

    def test_group_roundtrip(self) -> None:
        self._roundtrip("group<workgroup>")

    def test_buffer_roundtrip(self) -> None:
        self._roundtrip("buffer")

    def test_dialect_opaque_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        self._roundtrip("hal.buffer", type_registry=registry)

    def test_dialect_parameterized_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        self._roundtrip("vm.ref<hal.buffer>", type_registry=registry)

    def test_dialect_nested_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["vm.list"] = TypeDef(
            name="vm.list",
            params=[TypeParam("element", ANY)],
            format=[TypeOf("element")],
        )
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        self._roundtrip("vm.ref<vm.list<i32>>", type_registry=registry)


# ============================================================================
# Op parsing — simple patterns
# ============================================================================


def _op_parser() -> Parser:
    """Create a parser with test ops and built-in types."""
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _op_printer(**kwargs: Any) -> Printer:
    """Create a printer with test ops and built-in types."""
    printer = Printer(**kwargs)
    printer.register_ops(ALL_TEST_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer


def _parse_op(
    text: str, module: Module | None = None, scope: NameScope | None = None
) -> Operation:
    """Parse a single op from text."""

    parser = _op_parser()
    return parser.parse_operation_from_text(text, module=module, scope=scope)


def _setup_scope(*names_and_types: tuple[str, Type]) -> tuple[Module, NameScope]:
    """Create module + scope with pre-defined values."""
    module = Module()
    scope = NameScope()
    for name, value_type in names_and_types:
        vid = module.add_value(Value(name=name, type=value_type))
        scope.define(name, vid)
    return module, scope


class TestParseBinaryOp:
    def test_addi(self) -> None:
        module, scope = _setup_scope(("a", I32), ("b", I32))
        op = _parse_op("%r = test.addi %a, %b : i32", module=module, scope=scope)
        assert op.name == "test.addi"
        assert len(op.operands) == 2
        assert len(op.results) == 1
        assert module.values[op.results[0]].name == "r"
        assert module.values[op.results[0]].type == I32


class TestParseUnaryOp:
    def test_neg(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op("%r = test.neg %x : f32", module=module, scope=scope)
        assert op.name == "test.neg"
        assert len(op.operands) == 1
        assert module.values[op.results[0]].type == F32


class TestParseCastOp:
    def test_cast(self) -> None:
        module, scope = _setup_scope(("x", I32))
        op = _parse_op("%r = test.cast %x : i32 to f32", module=module, scope=scope)
        assert op.name == "test.cast"
        assert module.values[op.results[0]].type == F32


class TestParseConvertOp:
    def test_convert_scalar(self) -> None:
        module, scope = _setup_scope(("x", I32))
        op = _parse_op("%r = test.convert %x : i32 -> f32", module=module, scope=scope)
        assert op.name == "test.convert"
        assert module.values[op.results[0]].type == F32

    def test_convert_tile(self) -> None:
        tile_i8 = ShapedType(
            TypeKind.TILE, ScalarType(ScalarTypeKind.I8), (StaticDim(4),)
        )
        module, scope = _setup_scope(("x", tile_i8))
        op = _parse_op(
            "%r = test.convert %x : tile<4xi8> -> tile<4xf32>",
            module=module,
            scope=scope,
        )
        assert op.name == "test.convert"
        result_type = module.values[op.results[0]].type
        assert isinstance(result_type, ShapedType)
        assert result_type.element_type == F32


class TestParseConstantOp:
    def test_integer(self) -> None:
        op = _parse_op("%c42 = test.constant 42 : i32")
        assert op.name == "test.constant"
        assert op.attributes["value"] == 42

    def test_float(self) -> None:
        op = _parse_op("%pi = test.constant 3.14 : f32")
        assert abs(op.attributes["value"] - 3.14) < 1e-10

    def test_special_float_values(self) -> None:
        nan_op = _parse_op("%nan = test.constant nan : f32")
        assert math.isnan(nan_op.attributes["value"])
        inf_op = _parse_op("%inf = test.constant inf : f32")
        assert inf_op.attributes["value"] == math.inf
        negative_inf_op = _parse_op("%ninf = test.constant -inf : f32")
        assert negative_inf_op.attributes["value"] == -math.inf

    def test_clause_constant_roundtrip(self) -> None:
        source = "%c42 = test.clause_constant value(42) : i32"
        module = Module()
        op = _parse_op(source, module=module)
        assert op.name == "test.clause_constant"
        assert op.attributes["value"] == 42
        assert _op_printer().print_operation(op, module) == source


class TestParseClauseCopyOp:
    def test_dynamic_operand_clauses_roundtrip(self) -> None:
        source = "test.clause_copy source(%src) target(%dst) : i32"
        module, scope = _setup_scope(("src", I32), ("dst", I32))
        op = _parse_op(source, module=module, scope=scope)
        assert op.name == "test.clause_copy"
        assert len(op.operands) == 2
        assert _op_printer().print_operation(op, module) == source


class TestParseComparisonOp:
    def test_cmp(self) -> None:
        module, scope = _setup_scope(("a", I32), ("b", I32))
        op = _parse_op("%r = test.cmp lt, %a, %b : i32", module=module, scope=scope)
        assert op.attributes["predicate"] == "lt"


class TestParseYieldOp:
    def test_single(self) -> None:
        module, scope = _setup_scope(("a", F32))
        op = _parse_op("test.yield %a : f32", module=module, scope=scope)
        assert op.name == "test.yield"
        assert len(op.operands) == 1
        assert len(op.results) == 0

    def test_multiple(self) -> None:
        module, scope = _setup_scope(("a", F32), ("b", I32))
        op = _parse_op("test.yield %a, %b : f32, i32", module=module, scope=scope)
        assert len(op.operands) == 2


class TestParseSegmentedOp:
    def test_present_optional_and_two_variadic_spans(self) -> None:
        module, scope = _setup_scope(
            ("root", I32),
            ("guard", I32),
            ("lhs0", I32),
            ("lhs1", I32),
            ("rhs", I32),
        )
        op = _parse_op(
            "%result = test.segmented %root base %guard "
            "values %lhs0, %lhs1 expected %rhs : i32 -> i32",
            module=module,
            scope=scope,
        )
        assert op.name == "test.segmented"
        assert [module.values[value_id].name for value_id in op.operands] == [
            "root",
            "guard",
            "lhs0",
            "lhs1",
            "rhs",
        ]
        assert op.operand_segment_counts == (1, 1, 2, 1)

    def test_absent_optional_and_empty_variadic_span(self) -> None:
        module, scope = _setup_scope(("root", I32), ("rhs0", I32), ("rhs1", I32))
        op = _parse_op(
            "%result = test.segmented %root values expected %rhs0, %rhs1 : i32 -> i32",
            module=module,
            scope=scope,
        )
        assert [module.values[value_id].name for value_id in op.operands] == [
            "root",
            "rhs0",
            "rhs1",
        ]
        assert op.operand_segment_counts == (1, 0, 0, 2)


class TestParseAttrDictOp:
    def test_with_attrs(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {axis = 0, label = "foo"} : f32',
            module=module,
            scope=scope,
        )
        assert "dict" in op.attributes
        d = op.attributes["dict"]
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("label", "foo")]
        assert d["axis"] == 0
        assert d["label"] == "foo"

    def test_empty_dict(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x : f32",
            module=module,
            scope=scope,
        )
        assert "dict" not in op.attributes or op.attributes.get("dict") in (
            None,
            {},
        )

    def test_single_entry(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {count = 42} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert d["count"] == 42

    def test_bool_values(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {enabled = true, debug = false} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert d["enabled"] is True
        assert d["debug"] is False

    def test_string_value_escapes_are_decoded(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            r'%r = test.attrs %x {msg = "quote=\" slash=\\ \b \f \n \r \t \u0001 \u03BB \uD83D\uDD25"} : f32',
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert d["msg"] == 'quote=" slash=\\ \b \f \n \r \t \x01 λ 🔥'

    def test_symbol_ref_value_roundtrip(self) -> None:
        source = "%r = test.attrs %x {target = @target} : f32"
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(source, module=module, scope=scope)
        d = op.attributes["dict"]
        assert d["target"] == SymbolName("target")
        assert isinstance(d["target"], SymbolName)
        assert _op_printer().print_operation(op, module) == source

    def test_symbol_template_param_roundtrip(self) -> None:
        source = "test.template_param_symbol<@target>"
        module = Module()
        op = _parse_op(source, module=module, scope=NameScope())
        assert op.attributes["target"] == "target"
        assert _op_printer().print_operation(op, module) == source

    def test_symbol_template_param_flags_roundtrip(self) -> None:
        source = "test.template_param_symbol_flags<@target, debug|trace>"
        module = Module()
        op = _parse_op(source, module=module, scope=NameScope())
        assert op.attributes["target"] == "target"
        assert op.attributes["flags"] == "debug|trace"
        assert _op_printer().print_operation(op, module) == source

    def test_string_that_looks_like_symbol_stays_string(self) -> None:
        source = '%r = test.attrs %x {target = "@target"} : f32'
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(source, module=module, scope=scope)
        d = op.attributes["dict"]
        assert d["target"] == "@target"
        assert not isinstance(d["target"], SymbolName)
        assert _op_printer().print_operation(op, module) == source

    def test_round_trip(self) -> None:
        """Parse → print → parse produces the same dict entries."""
        import loom.dialect.test.defs as td
        from loom.dsl import Op as DslOp

        ops = [v for v in vars(td).values() if isinstance(v, DslOp)]
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {axis = 0, label = "hello"} : f32',
            module=module,
            scope=scope,
        )
        p = Printer()
        p.register_ops(ops)
        text = p.print_operation(op, module)
        assert "axis = 0" in text
        assert 'label = "hello"' in text

    def test_unsorted_input_is_canonicalized(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {label = "foo", axis = 0} : f32',
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("label", "foo")]

    def test_duplicate_key_is_rejected(self) -> None:
        module, scope = _setup_scope(("x", F32))
        with pytest.raises(ParseError, match="duplicate attribute dict key 'axis'"):
            _parse_op(
                "%r = test.attrs %x {axis = 0, axis = 1} : f32",
                module=module,
                scope=scope,
            )


class TestParseSliceOp:
    def test_static_offsets(self) -> None:
        tile_type = ShapedType(
            TypeKind.TILE,
            ScalarType(ScalarTypeKind.F16),
            (StaticDim(64), StaticDim(64)),
        )
        module, scope = _setup_scope(("src", tile_type))
        op = _parse_op(
            "%r = test.slice %src[0, 0] : tile<64x64xf16> -> (tile<16x16xf16>)",
            module=module,
            scope=scope,
        )
        assert op.attributes["static_offsets"] == [0, 0]

    def test_dynamic_offsets(self) -> None:
        tile_type = ShapedType(
            TypeKind.TILE,
            ScalarType(ScalarTypeKind.F16),
            (StaticDim(64), StaticDim(64)),
        )
        module, scope = _setup_scope(("src", tile_type), ("off", INDEX))
        sentinel = -(2**63)
        op = _parse_op(
            "%r = test.slice %src[0, %off] : tile<64x64xf16> -> (tile<16x16xf16>)",
            module=module,
            scope=scope,
        )
        assert op.attributes["static_offsets"] == [0, sentinel]


class TestParseShapeOp:
    def test_unglued_index_list(self) -> None:
        tile_type = ShapedType(
            TypeKind.TILE,
            ScalarType(ScalarTypeKind.F16),
            (StaticDim(64), StaticDim(64)),
        )
        module, scope = _setup_scope(("src", tile_type), ("dim", INDEX))
        sentinel = -(2**63)
        op = _parse_op(
            "test.shape %src shape [%dim, 4] : tile<64x64xf16>",
            module=module,
            scope=scope,
        )
        assert op.name == "test.shape"
        assert op.attributes["static_dims"] == [sentinel, 4]


class TestParseTiedResult:
    def test_update(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        module, scope = _setup_scope(
            ("tile", tile_t), ("tensor", tensor_t), ("off", INDEX)
        )
        op = _parse_op(
            "%r = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)",
            module=module,
            scope=scope,
        )
        assert len(op.tied_results) == 1
        assert op.tied_results[0].operand_index == 1  # %tensor


class TestParseInvokeOp:
    def test_with_tie(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module, scope = _setup_scope(("weights", tile_t), ("input", INDEX))
        op = _parse_op(
            "%output, %count = test.invoke @compute(%weights, %input)"
            " : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)",
            module=module,
            scope=scope,
        )
        assert op.name == "test.invoke"
        assert op.attributes["callee"] == "compute"
        assert len(op.results) == 2
        assert len(op.tied_results) == 1
        assert op.tied_results[0].operand_index == 0


# ============================================================================
# Module-level parsing
# ============================================================================


class TestParseModule:
    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_empty_module(self) -> None:
        module = self._parse_module("")
        assert len(module.symbols) == 0

    def test_simple_function(self) -> None:
        module = self._parse_module(
            "test.func @negate(%input: f32) -> (f32) {\n"
            "  %r = test.neg %input : f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )
        assert len(module.symbols) == 1
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("callee") == "negate"
        assert len(op.regions[0].blocks[0].arg_ids) == 1
        assert len(op.results) == 1
        assert op.regions
        assert len(op.regions[0].blocks) == 1
        assert len(op.regions[0].blocks[0].ops) == 2

    def test_simple_function_records_value_metadata(self) -> None:
        module = self._parse_module(
            "test.func @negate(%input: f32) -> (f32) {\n"
            "  %r = test.neg %input : f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        entry_block = func_op.regions[0].blocks[0]

        arg_id = entry_block.arg_ids[0]
        arg = module.values[arg_id]
        assert arg.is_block_arg
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 0
        assert arg.def_result_index == 0
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=0)]

        neg_op = entry_block.ops[0]
        result_id = neg_op.results[0]
        result = module.values[result_id]
        assert not result.is_block_arg
        assert result.def_op_index == 0
        assert result.def_block_index == 0
        assert result.def_result_index == 0
        assert result.uses == [Use(user_op_index=1, operand_index=0, block_index=0)]

    def test_public_declaration(self) -> None:
        module = self._parse_module("test.decl public @exported(%a: i32) -> (i32)\n")
        assert len(module.symbols) == 1
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("visibility") == "public"
        assert not op.regions  # Declarations have no body region.

    def test_declaration_signature_args_are_operand_defs(self) -> None:
        module = self._parse_module("test.decl @identity(%a: f32) -> (%a as f32)\n")
        op = module.symbols[0].op
        assert op is not None

        arg_id = op.operands[0]
        arg = module.values[arg_id]
        assert not arg.is_block_arg
        assert arg.def_op_index == 0
        assert arg.def_block_index == VALUE_DEF_BLOCK_NONE
        assert arg.def_result_index == 0
        assert arg.uses == []

        result_id = op.results[0]
        result = module.values[result_id]
        assert result.def_op_index == 0
        assert result.def_block_index == VALUE_DEF_BLOCK_NONE
        assert result.def_result_index == 0

    def test_multiple_functions(self) -> None:
        module = self._parse_module(
            "test.func @f1(%a: f32) -> (f32) {\n"
            "  test.yield %a : f32\n"
            "}\n"
            "\n"
            "test.func @f2(%b: i32) -> (i32) {\n"
            "  test.yield %b : i32\n"
            "}\n"
        )
        assert len(module.symbols) == 2
        op0 = module.symbols[0].op
        op1 = module.symbols[1].op
        assert op0 is not None
        assert op1 is not None
        assert op0.attributes.get("callee") == "f1"
        assert op1.attributes.get("callee") == "f2"


class TestParseEncodingAlias:
    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_alias_definition(self) -> None:
        module = self._parse_module(
            "#enc = #q8_0<block=32>\n"
            "test.func @f(%a: tile<256xi8, #enc>) -> (tile<256xi8, #enc>) {\n"
            "  test.yield %a : tile<256xi8, #enc>\n"
            "}\n"
        )
        assert len(module.symbols) == 1
        assert len(module.encodings) >= 1
        enc = module.encodings[0]
        assert enc.alias == "enc"
        assert enc.name == "q8_0"
        assert enc.params == (("block", 32),)

    def test_alias_without_params(self) -> None:
        module = self._parse_module(
            "#w = #q6_k\n"
            "test.func @f(%a: tile<256xi8, #w>) -> (tile<256xi8, #w>) {\n"
            "  test.yield %a : tile<256xi8, #w>\n"
            "}\n"
        )
        enc = module.encodings[0]
        assert enc.alias == "w"
        assert enc.name == "q6_k"
        assert enc.params == ()

    def test_alias_cannot_shadow_known_family(self) -> None:
        parser = _op_parser()
        parser.register_encodings(["q8_0", "dense"])
        with pytest.raises(ParseError, match="alias name shadows"):
            parser.parse("#q8_0 = #dense\n")

    def test_duplicate_alias_definition_fails(self) -> None:
        with pytest.raises(ParseError, match="duplicate encoding alias name"):
            self._parse_module("#enc = #dense\n#enc = #q8_0<block=32>\n")


class TestEncodingValidation:
    def test_known_encoding_accepted(self) -> None:
        parser = _op_parser()
        parser.register_encodings(["q8_0", "q6_k"])
        # Should not raise.
        parser.parse(
            "test.func @f(%a: tile<256xi8, #q8_0>) -> (tile<256xi8, #q8_0>) {\n"
            "  test.yield %a : tile<256xi8, #q8_0>\n"
            "}\n"
        )

    def test_unknown_encoding_rejected(self) -> None:
        parser = _op_parser()
        parser.register_encodings(["q8_0"])
        with pytest.raises(ParseError, match="unknown encoding"):
            parser.parse(
                "test.func @f(%a: tile<256xi8, #q999>) -> (tile<256xi8, #q999>) {\n"
                "  test.yield %a : tile<256xi8, #q999>\n"
                "}\n"
            )


# ============================================================================
# Round-trip tests: print → parse → print
# ============================================================================


class TestRoundTrip:
    """The gold standard: print(parse(print(construct()))) == print(construct())."""

    def _roundtrip_text(self, text: str) -> None:
        """Parse text, print, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_simple_function(self) -> None:
        self._roundtrip_text(
            "test.func @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

    def test_operation_and_block_comments(self) -> None:
        self._roundtrip_text(
            "// top-level function\n"
            "test.func @comments() {\n"
            "// explicit entry block\n"
            "^entry:\n"
            "  // body terminator\n"
            "  test.yield\n"
            "}\n"
        )

    def test_convert_bare_result_type(self) -> None:
        """test.convert uses ResultType (bare, no parens) and round-trips."""
        self._roundtrip_text(
            "test.func @convert(%x: i32) -> (f32) {\n"
            "  %r = test.convert %x : i32 -> f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )

    def test_convert_tile_result_type(self) -> None:
        """test.convert with tile types round-trips with bare result."""
        self._roundtrip_text(
            "test.func @convert_tile(%x: tile<4xi8>) -> (tile<4xf32>) {\n"
            "  %r = test.convert %x : tile<4xi8> -> tile<4xf32>\n"
            "  test.yield %r : tile<4xf32>\n"
            "}\n"
        )

    def test_empty_operand_dict(self) -> None:
        self._roundtrip_text(
            "test.func @operand_dict_empty(%input: f32) -> (f32) {\n"
            "  %result = test.operand_dict %input : f32\n"
            "  test.yield %result : f32\n"
            "}\n"
        )


class TestOperandDict:
    def test_unsorted_keys_print_in_canonical_order(self) -> None:
        module = _op_parser().parse(
            "test.func @operand_dict(%input: f32, %beta: f32, %alpha: i32) "
            "-> (f32) {\n"
            "  %result = test.operand_dict %input "
            "{beta = %beta : f32, alpha = %alpha : i32} : f32\n"
            "  test.yield %result : f32\n"
            "}\n"
        )
        printed = _op_printer().print_module(module)
        assert printed == (
            "test.func @operand_dict(%input: f32, %beta: f32, %alpha: i32) "
            "-> (f32) {\n"
            "  %result = test.operand_dict %input "
            "{alpha = %alpha : i32, beta = %beta : f32} : f32\n"
            "  test.yield %result : f32\n"
            "}\n"
        )

    def test_duplicate_key_rejected(self) -> None:
        with pytest.raises(ParseError, match="duplicate operand dictionary key"):
            _op_parser().parse(
                "test.func @operand_dict(%input: f32, %alpha: f32) -> (f32) {\n"
                "  %result = test.operand_dict %input "
                "{alpha = %alpha : f32, alpha = %input : f32} : f32\n"
                "  test.yield %result : f32\n"
                "}\n"
            )

    def test_type_annotation_mismatch_rejected(self) -> None:
        with pytest.raises(ParseError, match="operand dictionary entry 'alpha'"):
            _op_parser().parse(
                "test.func @operand_dict(%input: f32, %alpha: i32) -> (f32) {\n"
                "  %result = test.operand_dict %input {alpha = %alpha : f32} : f32\n"
                "  test.yield %result : f32\n"
                "}\n"
            )


class TestAttrTable:
    def test_grouped_rows_roundtrip(self) -> None:
        module = _op_parser().parse(
            "test.func @attr_table(%selector: index, %a0: i32, %b0: f32, "
            "%a1: i32, %b1: f32, %ad: i32, %bd: f32) -> (i32, f32) {\n"
            "  %x, %y = test.attr_table %selector "
            "{0 = (%a0, %b0), 1 = (%a1, %b1)} default(%ad, %bd) : i32, f32\n"
            "  test.yield %x, %y : i32, f32\n"
            "}\n"
        )
        printed = _op_printer().print_module(module)
        assert printed == (
            "test.func @attr_table(%selector: index, %a0: i32, %b0: f32, "
            "%a1: i32, %b1: f32, %ad: i32, %bd: f32) -> (i32, f32) {\n"
            "  %x, %y = test.attr_table %selector {\n"
            "    0 = (%a0, %b0),\n"
            "    1 = (%a1, %b1)\n"
            "  } default(%ad, %bd) : i32, f32\n"
            "  test.yield %x, %y : i32, f32\n"
            "}\n"
        )

    def test_empty_cases_roundtrip(self) -> None:
        module = _op_parser().parse(
            "test.func @attr_table_empty(%selector: index, %fallback: i32) -> (i32) {\n"
            "  %x = test.attr_table %selector {} default(%fallback) : i32\n"
            "  test.yield %x : i32\n"
            "}\n"
        )
        printed = _op_printer().print_module(module)
        assert printed == (
            "test.func @attr_table_empty(%selector: index, %fallback: i32) -> (i32) {\n"
            "  %x = test.attr_table %selector {} default(%fallback) : i32\n"
            "  test.yield %x : i32\n"
            "}\n"
        )

    def test_case_row_width_mismatch_rejected(self) -> None:
        with pytest.raises(ParseError, match="attribute table rows"):
            _op_parser().parse(
                "test.func @attr_table_bad(%selector: index, %a: i32, %b: i32) "
                "-> (i32, i32) {\n"
                "  %x, %y = test.attr_table %selector "
                "{0 = (%a, %b), 1 = (%a)} default(%a, %b) : i32, i32\n"
                "  test.yield %x, %y : i32, i32\n"
                "}\n"
            )

    def test_default_row_width_mismatch_rejected(self) -> None:
        with pytest.raises(ParseError, match="attribute table default row"):
            _op_parser().parse(
                "test.func @attr_table_bad(%selector: index, %a: i32, %b: i32) "
                "-> (i32, i32) {\n"
                "  %x, %y = test.attr_table %selector "
                "{0 = (%a, %b)} default(%a) : i32, i32\n"
                "  test.yield %x, %y : i32, i32\n"
                "}\n"
            )


class TestRegionTable:
    def test_grouped_regions_roundtrip(self) -> None:
        module = _op_parser().parse(
            "test.func @region_table(%selector: index) {\n"
            "  test.region_table %selector {\n"
            "    case 0 {\n"
            "      test.yield\n"
            "    }\n"
            "    case 1 {\n"
            "      test.yield\n"
            "    }\n"
            "    default {\n"
            "      test.yield\n"
            "    }\n"
            "  }\n"
            "  test.yield\n"
            "}\n"
        )
        printed = _op_printer().print_module(module)
        assert printed == (
            "test.func @region_table(%selector: index) {\n"
            "  test.region_table %selector {\n"
            "    case 0 {\n"
            "      test.yield\n"
            "    }\n"
            "    case 1 {\n"
            "      test.yield\n"
            "    }\n"
            "    default {\n"
            "      test.yield\n"
            "    }\n"
            "  }\n"
            "  test.yield\n"
            "}\n"
        )

    def test_empty_cases_roundtrip(self) -> None:
        module = _op_parser().parse(
            "test.func @region_table_empty(%selector: index) {\n"
            "  test.region_table %selector {\n"
            "    default {\n"
            "      test.yield\n"
            "    }\n"
            "  }\n"
            "  test.yield\n"
            "}\n"
        )
        printed = _op_printer().print_module(module)
        assert printed == (
            "test.func @region_table_empty(%selector: index) {\n"
            "  test.region_table %selector {\n"
            "    default {\n"
            "      test.yield\n"
            "    }\n"
            "  }\n"
            "  test.yield\n"
            "}\n"
        )


# ============================================================================
# Op parsing — region-containing ops
# ============================================================================


class TestParseMapOp:
    def test_elementwise(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module, scope = _setup_scope(("x", tile_t))
        op = _parse_op(
            "%r = test.map(%e = %x : tile<4xf32>) {\n"
            "  %v = test.neg %e : f32\n"
            "  test.yield %v : f32\n"
            "} -> (tile<4xf32>)",
            module=module,
            scope=scope,
        )
        assert op.name == "test.map"
        assert len(op.operands) == 1  # %x
        assert len(op.regions) == 1
        body = op.regions[0]
        assert len(body.blocks) == 1
        assert len(body.blocks[0].ops) == 2  # neg + yield

    def test_element_binding_block_arg_records_metadata(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module, scope = _setup_scope(("x", tile_t))
        op = _parse_op(
            "%r = test.map(%e = %x : tile<4xf32>) {\n"
            "  test.yield %e : f32\n"
            "} -> (tile<4xf32>)",
            module=module,
            scope=scope,
        )
        entry_block = op.regions[0].blocks[0]
        arg_id = entry_block.arg_ids[0]
        arg = module.values[arg_id]
        assert arg.is_block_arg
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 0
        assert arg.def_result_index == 0
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=0)]

    def test_binding_name_can_shadow_outer_scope_name(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module = _op_parser().parse(
            "test.func @shadow(%x: f32, %tile: tile<4xf32>) -> (f32) {\n"
            "  %mapped = test.map(%x = %tile : tile<4xf32>) {\n"
            "    test.yield %x : f32\n"
            "  } -> (f32)\n"
            "  %negated = test.neg %x : f32\n"
            "  test.yield %negated : f32\n"
            "}\n"
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        func_entry = func_op.regions[0].blocks[0]
        map_op = func_entry.ops[0]
        map_entry = map_op.regions[0].blocks[0]
        assert map_entry.arg_ids[0] != func_entry.arg_ids[0]
        assert map_entry.ops[0].operands[0] == map_entry.arg_ids[0]
        assert func_entry.ops[1].operands[0] == func_entry.arg_ids[0]
        assert module.values[func_entry.arg_ids[1]].type == tile_t

    def test_binding_name_is_region_local(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value '%element'"):
            _op_parser().parse(
                "test.func @leak(%tile: tile<4xf32>) -> (f32) {\n"
                "  %mapped = test.map(%element = %tile : tile<4xf32>) {\n"
                "    test.yield %element : f32\n"
                "  } -> (f32)\n"
                "  %bad = test.neg %element : f32\n"
                "  test.yield %bad : f32\n"
                "}\n"
            )


class TestParseLoopOp:
    def test_without_iter_args_synthesizes_implicit_terminator(self) -> None:
        module, scope = _setup_scope(("c0", INDEX), ("n", INDEX), ("c1", INDEX))
        op = _parse_op(
            "test.loop %i = %c0 to %n step %c1 {\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.loop"
        assert len(op.regions) == 1
        body = op.regions[0]
        assert len(body.blocks) == 1
        assert len(body.blocks[0].arg_ids) == 1
        assert len(body.blocks[0].ops) == 1
        assert body.blocks[0].ops[0].name == "test.implicit_yield"
        assert body.blocks[0].ops[0].operands == []
        assert (
            _op_printer().print_operation(op, module)
            == "test.loop %i = %c0 to %n step %c1 {\n}"
        )

    def test_explicit_implicit_yield_is_canonicalized(self) -> None:
        module, scope = _setup_scope(("c0", INDEX), ("n", INDEX), ("c1", INDEX))
        op = _parse_op(
            "test.loop %i = %c0 to %n step %c1 {\n  test.implicit_yield\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.loop"
        assert len(op.regions) == 1
        body = op.regions[0]
        assert len(body.blocks) == 1
        assert len(body.blocks[0].ops) == 1
        assert body.blocks[0].ops[0].name == "test.implicit_yield"
        assert body.blocks[0].ops[0].operands == []
        assert (
            _op_printer().print_operation(op, module)
            == "test.loop %i = %c0 to %n step %c1 {\n}"
        )

    def test_explicit_empty_yield_is_preserved(self) -> None:
        module, scope = _setup_scope(("c0", INDEX), ("n", INDEX), ("c1", INDEX))
        op = _parse_op(
            "test.loop %i = %c0 to %n step %c1 {\n  test.yield\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.loop"
        assert len(op.regions) == 1
        body = op.regions[0]
        assert len(body.blocks) == 1
        assert len(body.blocks[0].ops) == 1
        assert body.blocks[0].ops[0].name == "test.yield"
        assert body.blocks[0].ops[0].operands == []
        assert (
            _op_printer().print_operation(op, module)
            == "test.loop %i = %c0 to %n step %c1 {\n"
            "  test.yield\n"
            "}"
        )

    def test_with_iter_args(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        op = _parse_op(
            "%r = test.loop %i = %c0 to %n step %c1"
            " iter_args(%acc = %init : f32) -> (%init as f32) {\n"
            "  test.yield %acc : f32\n"
            "}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.loop"
        assert len(op.operands) == 4  # lb, ub, step + %init iter_arg
        assert len(op.regions) == 1
        assert len(op.results) == 1
        assert len(op.tied_results) == 1
        assert op.tied_results[0].result_index == 0
        assert op.tied_results[0].operand_index == 3
        assert len(op.regions[0].blocks[0].arg_ids) == 2
        assert (
            op.regions[0].blocks[0].ops[0].operands[0]
            == (op.regions[0].blocks[0].arg_ids[1])
        )
        assert (
            _op_printer().print_operation(op, module)
            == "%r = test.loop %i = %c0 to %n step %c1"
            " iter_args(%acc = %init : f32) -> (%init as f32) {\n"
            "  test.yield %acc : f32\n"
            "}"
        )

    def test_iter_arg_name_is_not_tied_result_target(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        with pytest.raises(
            ParseError, match="tied result 'acc' not found in args or operands"
        ):
            _parse_op(
                "%r = test.loop %i = %c0 to %n step %c1"
                " iter_args(%acc = %init : f32) -> (%acc as f32) {\n"
                "  test.yield %acc : f32\n"
                "}",
                module=module,
                scope=scope,
            )

    def test_iv_name_is_not_tied_result_target(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        with pytest.raises(
            ParseError, match="tied result 'i' not found in args or operands"
        ):
            _parse_op(
                "%r = test.loop %i = %c0 to %n step %c1"
                " iter_args(%acc = %init : f32) -> (%i as f32) {\n"
                "  test.yield %acc : f32\n"
                "}",
                module=module,
                scope=scope,
            )

    def test_iv_name_is_region_local(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value '%i'"):
            _op_parser().parse(
                "test.func @loop(%lo: index, %hi: index, %step: index) {\n"
                "  test.loop %i = %lo to %hi step %step {\n"
                "  }\n"
                "  test.loop %again = %i to %hi step %step {\n"
                "  }\n"
                "}\n"
            )


class TestParseBranchOp:
    def test_empty_regions_synthesize_implicit_terminators(self) -> None:
        module, scope = _setup_scope(("cond", I32))
        op = _parse_op(
            "test.branch %cond {\n} else {\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.branch"
        assert len(op.regions) == 2
        assert len(op.regions[0].blocks) == 1
        assert len(op.regions[1].blocks) == 1
        assert len(op.regions[0].blocks[0].ops) == 1
        assert len(op.regions[1].blocks[0].ops) == 1
        assert op.regions[0].blocks[0].ops[0].name == "test.implicit_yield"
        assert op.regions[1].blocks[0].ops[0].name == "test.implicit_yield"
        assert op.regions[0].blocks[0].ops[0].operands == []
        assert op.regions[1].blocks[0].ops[0].operands == []
        assert (
            _op_printer().print_operation(op, module)
            == "test.branch %cond {\n} else {\n}"
        )

    def test_if_else(self) -> None:
        module, scope = _setup_scope(("cond", I32), ("a", F32), ("b", F32))
        op = _parse_op(
            "%r = test.branch %cond -> (f32) {\n"
            "  test.yield %a : f32\n"
            "} else {\n"
            "  test.yield %b : f32\n"
            "}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.branch"
        assert len(op.regions) == 2  # then + else
        assert len(op.regions[0].blocks[0].ops) == 1  # yield in then
        assert len(op.regions[1].blocks[0].ops) == 1  # yield in else

    def test_optional_region_absent(self) -> None:
        module, scope = _setup_scope(("cond", I32))
        op = _parse_op(
            "test.optional_region %cond {\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.optional_region"
        assert len(op.regions) == 1
        assert _op_printer().print_operation(op, module) == (
            "test.optional_region %cond {\n}"
        )

    def test_optional_region_present(self) -> None:
        module, scope = _setup_scope(("cond", I32))
        op = _parse_op(
            "test.optional_region %cond {\n} else {\n}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.optional_region"
        assert len(op.regions) == 2
        assert _op_printer().print_operation(op, module) == (
            "test.optional_region %cond {\n} else {\n}"
        )


# ============================================================================
# Encoding type and dynamic encoding parsing
# ============================================================================


class TestParseEncodingType:
    def test_encoding_keyword(self) -> None:
        """'encoding' parses to EncodingType."""
        result_type = _parse_type("encoding")
        assert isinstance(result_type, EncodingType)
        assert result_type == ENCODING_TYPE

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("encoding<layout>", ENCODING_LAYOUT_TYPE),
            ("encoding<schema>", ENCODING_SCHEMA_TYPE),
            ("encoding<storage>", ENCODING_STORAGE_TYPE),
            ("encoding<transform>", ENCODING_TRANSFORM_TYPE),
        ],
    )
    def test_encoding_role(self, text: str, expected: EncodingType) -> None:
        result_type = _parse_type(text)
        assert result_type == expected

    def test_encoding_role_records_structural_role(self) -> None:
        result_type = _parse_type("encoding<layout>")
        assert isinstance(result_type, EncodingType)
        assert result_type.role == EncodingRole.LAYOUT

    def test_unknown_encoding_role_raises(self) -> None:
        with pytest.raises(ParseError, match="unknown encoding role"):
            _parse_type("encoding<address>")


class TestParseDynamicEncoding:
    def test_tile_with_ssa_encoding(self) -> None:
        """tile<4xf32, %enc> parses to ShapedType with DynamicEncoding."""
        module = Module()
        scope = NameScope()
        enc_id = module.add_value(Value(name="enc", type=ENCODING_TYPE))
        scope.define("enc", enc_id)
        shaped_type, bindings = _parse("tile<4xf32, %enc>", scope=scope, module=module)
        assert isinstance(shaped_type, ShapedType)
        assert shaped_type.type_kind == TypeKind.TILE
        assert shaped_type.element_type == F32
        assert shaped_type.dims == (StaticDim(4),)
        assert isinstance(shaped_type.encoding, DynamicEncoding)
        # Encoding binding uses sentinel key -1.
        assert bindings[-1] == enc_id

    def test_static_encoding_still_works(self) -> None:
        """tile<4xf32, #q8_0> still parses as static encoding."""
        shaped_type = _parse_type("tile<4xf32, #q8_0>")
        assert isinstance(shaped_type, ShapedType)
        assert isinstance(shaped_type.encoding, EncodingInstance)
        assert shaped_type.encoding.name == "q8_0"

    def test_tile_without_encoding_still_works(self) -> None:
        """tile<4xf32> still parses with no encoding."""
        shaped_type = _parse_type("tile<4xf32>")
        assert isinstance(shaped_type, ShapedType)
        assert shaped_type.encoding is None

    def test_undefined_encoding_value_errors(self) -> None:
        """Referencing undefined %enc in encoding position errors."""
        from loom.format.text.tokenizer import ParseError as TokParseError

        with pytest.raises((ParseError, TokParseError, KeyError)):
            _parse("tile<4xf32, %undefined>")

    def test_non_encoding_typed_value_errors(self) -> None:
        """Referencing a non-encoding-typed value in encoding position errors."""
        module = Module()
        scope = NameScope()
        # %x is i32, not encoding.
        x_id = module.add_value(Value(name="x", type=I32))
        scope.define("x", x_id)
        with pytest.raises((ParseError, ValueError)):
            _parse("tile<4xf32, %x>", scope=scope, module=module)


# ============================================================================
# Round-trip: encoding type in function signatures
# ============================================================================


# ============================================================================
# Pool type parsing
# ============================================================================


class TestParsePoolType:
    def test_static_pool(self) -> None:
        """pool<65536> parses to PoolType with static block size."""
        from loom.ir import PoolType

        pool_type = _parse_type("pool<65536>")
        assert isinstance(pool_type, PoolType)
        assert pool_type.block_size == StaticDim(65536)

    def test_dynamic_pool(self) -> None:
        """pool<[%BS]> parses to PoolType with dynamic block size."""
        from loom.ir import PoolType

        module = Module()
        scope = NameScope()
        bs_id = module.add_value(Value(name="BS", type=INDEX))
        scope.define("BS", bs_id)
        pool_type, bindings = _parse("pool<[%BS]>", scope=scope, module=module)
        assert isinstance(pool_type, PoolType)
        assert pool_type.has_dynamic_block_size
        assert bindings[0] == bs_id

    def test_pool_roundtrip(self) -> None:
        """pool<[%BS]> round-trips through parse → print in a function."""
        text = (
            "test.func @use_pool(%BS: index, %pool: pool<[%BS]>) -> (pool<[%BS]>) {\n"
            "  test.yield %pool : pool<[%BS]>\n"
            "}\n"
        )
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )


# ============================================================================
# Predicate parsing and round-trip
# ============================================================================


class TestParsePredicates:
    """Tests for parsing where-clause predicates on functions."""

    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_single_mul_predicate(self) -> None:
        """Parse a function with where [mul(%M, 16)]."""

        module = self._parse_module(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>)"
            " where [mul(%M, 16)] {\n"
            "  test.yield\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        pred = predicates[0]
        assert pred.kind == "mul"
        assert len(pred.args) == 2
        assert pred.args[0].tag == "value"
        assert pred.args[0].value == "M"
        assert pred.args[1].tag == "const"
        assert pred.args[1].value == 16

    def test_multiple_predicates(self) -> None:
        """Parse a function with multiple predicates."""

        module = self._parse_module(
            "test.func @f(%M: index, %K: index, %a: tensor<[%M]x[%K]xf32>)"
            " where [mul(%M, 16), lt(%K, 1024), ne(%K, 0), range(%M, 32, 512)] {\n"
            "  test.yield\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 4
        assert predicates[0].kind == "mul"
        assert predicates[1].kind == "lt"
        assert predicates[2].kind == "ne"
        assert predicates[3].kind == "range"
        # range has 3 args.
        range_pred = predicates[3]
        assert len(range_pred.args) == 3
        assert range_pred.args[0].value == "M"
        assert range_pred.args[1].value == 32
        assert range_pred.args[2].value == 512

    def test_pow2_single_arg(self) -> None:
        """Parse pow2(%N) — single-argument predicate."""

        module = self._parse_module(
            "test.func @f(%N: index, %a: tensor<[%N]xf32>)"
            " where [pow2(%N)] {\n"
            "  test.yield\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        assert predicates[0].kind == "pow2"
        assert len(predicates[0].args) == 1

    def test_predicate_arity_mismatch_rejected(self) -> None:
        """Predicate kind names have fixed arity."""

        with pytest.raises(
            ParseError, match="predicate 'pow2' expects 1 arguments, got 2"
        ):
            self._parse_module(
                "test.func @f(%N: index, %a: tensor<[%N]xf32>)"
                " where [pow2(%N, 16)] {\n"
                "  test.yield\n"
                "}\n"
            )

    def test_named_result_predicate(self) -> None:
        """Parse eq(%idx, %M) — result name argument."""

        module = self._parse_module(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>) -> (%idx: index)"
            " where [eq(%idx, %M)] {\n"
            "  test.yield %M : index\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        pred = predicates[0]
        assert pred.kind == "eq"
        assert pred.args[0].tag == "value"
        assert pred.args[0].value == "idx"
        assert pred.args[1].tag == "value"
        assert pred.args[1].value == "M"

    def test_no_predicates(self) -> None:
        """Functions without where clause have empty predicates list."""
        module = self._parse_module(
            "test.func @f(%a: f32) -> (f32) {\n  test.yield %a : f32\n}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("predicates", []) == []

    def test_predicates_with_body(self) -> None:
        """Where clause followed by function body."""
        module = self._parse_module(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)] {\n"
            "  test.yield %a : tensor<[%M]xf32>\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        assert op.regions


class TestPredicateRoundTrip:
    """Round-trip tests: print(parse(text)) == text."""

    def _roundtrip_text(self, text: str) -> None:
        """Parse text, print, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)

        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_single_predicate_function(self) -> None:
        """Function-like ops must round-trip dim names and predicates."""
        self._roundtrip_text(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>) where [mul(%M, 16)] {\n"
            "  test.yield\n"
            "}\n"
        )

    def test_multiple_predicates_function(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index, %K: index, %a: tensor<[%M]x[%K]xf32>) "
            "where [mul(%M, 16), lt(%K, 1024), range(%M, 32, 512)] {\n"
            "  test.yield\n"
            "}\n"
        )

    def test_pow2_predicate(self) -> None:
        self._roundtrip_text(
            "test.func @f(%N: index, %a: tensor<[%N]xf32>) where [pow2(%N)] {\n"
            "  test.yield\n"
            "}\n"
        )

    def test_result_name_predicate(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>) -> (%idx: index) "
            "where [eq(%idx, %M)] {\n"
            "  test.yield %M : index\n"
            "}\n"
        )

    def test_predicates_with_body(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)] {\n"
            "  test.yield %a : tensor<[%M]xf32>\n"
            "}\n"
        )

    def test_public_device_with_predicates(self) -> None:
        self._roundtrip_text(
            "test.func public device @vnni(%M: index, %K: index, %w: tensor<[%M]x[%K]xi8>) "
            "where [mul(%M, 16), mul(%K, 32)] {\n"
            "  test.yield\n"
            "}\n"
        )


class TestAssumeOpRoundTrip:
    """Round-trip tests for test.assume (predicate-constrained identity)."""

    def _roundtrip_text(self, text: str) -> None:
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_single_predicate(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index) -> (index) {\n"
            "  %M2 = test.assume %M [mul(%M, 16)] : index\n"
            "  test.yield %M2 : index\n"
            "}\n"
        )

    def test_multiple_predicates(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index, %K: index) -> (index, index) {\n"
            "  %M2, %K2 = test.assume %M, %K [mul(%M, 16), lt(%K, 1024)] : index, index\n"
            "  test.yield %M2, %K2 : index, index\n"
            "}\n"
        )

    def test_pow2_predicate(self) -> None:
        self._roundtrip_text(
            "test.func @f(%N: index) -> (index) {\n"
            "  %N2 = test.assume %N [pow2(%N)] : index\n"
            "  test.yield %N2 : index\n"
            "}\n"
        )

    def test_range_predicate(self) -> None:
        self._roundtrip_text(
            "test.func @f(%M: index) -> (index) {\n"
            "  %M2 = test.assume %M [range(%M, 32, 512)] : index\n"
            "  test.yield %M2 : index\n"
            "}\n"
        )


# ============================================================================
# Dynamic dim and encoding scoping in function-like signatures
# ============================================================================


class TestDynamicDimScoping:
    """Dynamic dim names in types reference earlier args, not define new ones.

    This is critical for C parser parity: %M in tile<[%M]xf32> must resolve
    to the %M: index arg, not create a duplicate definition.
    """

    def _roundtrip_text(self, text: str) -> None:
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip mismatch:\n  expected: {text!r}\n  got:      {printed!r}"
        )

    def test_dim_references_earlier_arg(self) -> None:
        """Dynamic dim %M in second arg references %M: index first arg."""
        self._roundtrip_text(
            "test.func @f(%M: index, %tile: tile<[%M]xf32>) -> (tile<[%M]xf32>) {\n"
            "  test.yield %tile : tile<[%M]xf32>\n"
            "}\n"
        )

    def test_multiple_dynamic_dims(self) -> None:
        """Two dynamic dims from separate earlier args."""
        self._roundtrip_text(
            "test.decl @g(%M: index, %N: index, %t: tensor<[%M]x[%N]xf32>)"
            " -> (tensor<[%M]x[%N]xf32>)\n"
        )

    def test_mixed_static_dynamic_dims(self) -> None:
        self._roundtrip_text(
            "test.decl @h(%K: index, %t: tile<[%K]x4xf32>) -> (tile<[%K]x4xf32>)\n"
        )

    def test_dim_shared_across_args(self) -> None:
        """Same dim name %M used by two different arg types."""
        self._roundtrip_text(
            "test.decl @shared(%M: index, %a: tile<[%M]xf32>, %b: tile<[%M]xi8>)"
            " -> (tile<[%M]xf32>)\n"
        )

    def test_dim_explicitly_defined(self) -> None:
        """Explicitly defined %M in a type."""
        self._roundtrip_text(
            "test.decl @explicit(%M: index, %a: tile<[%M]xf32>, %b: tile<[%M]xi8>)"
            " -> (tile<[%M]xf32>)\n"
        )

    def test_dynamic_pool_references_earlier_arg(self) -> None:
        """Dynamic pool block size references earlier index arg."""
        self._roundtrip_text(
            "test.decl @p(%BS: index, %pool: pool<[%BS]>) -> (pool<[%BS]>)\n"
        )

    def test_dynamic_encoding_references_earlier_arg(self) -> None:
        """Dynamic encoding %enc in type references earlier encoding arg."""
        self._roundtrip_text(
            "test.decl @enc(%enc: encoding, %t: tile<4xf32, %enc>)"
            " -> (tile<4xf32, %enc>)\n"
        )

    def test_all_dynamic_in_one_signature(self) -> None:
        """Dynamic dims, encoding, and pool all in one function."""
        self._roundtrip_text(
            "test.decl @kitchen_sink(%M: index, %N: index, %enc: encoding,"
            " %t: tile<[%M]x[%N]xf32, %enc>, %p: pool<[%M]>)"
            " -> (tile<[%M]x[%N]xf32, %enc>)\n"
        )

    def test_def_with_dynamic_dims_in_body(self) -> None:
        """Dynamic dims flow into body ops."""
        self._roundtrip_text(
            "test.func @f(%M: index, %t: tile<[%M]xf32>) -> (tile<[%M]xf32>) {\n"
            "  %neg = test.neg %t : tile<[%M]xf32>\n"
            "  test.yield %neg : tile<[%M]xf32>\n"
            "}\n"
        )


# ============================================================================
# Location parsing
# ============================================================================


class TestLocationParsing:
    """Tests for parsing loc(...) annotations on ops."""

    def test_file_location(self) -> None:
        """Parse a FILE location annotation."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc("model.loom":42:3 to 42:58)',
            module=module,
            scope=scope,
        )
        from loom.ir import FileLocation

        loc = module.locations.get(op.location_id)
        assert isinstance(loc, FileLocation)
        assert loc.start_line == 42
        assert loc.start_col == 3
        assert loc.end_line == 42
        assert loc.end_col == 58
        # Source name should be registered in the module.
        assert module.sources[loc.source_id] == "model.loom"

    def test_file_location_escapes_are_decoded(self) -> None:
        """Parse escaped source names in FILE locations."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            r'%r = test.neg %x : f32 loc("model \"main\"\\v2\n.loom":42:3 to 42:58)',
            module=module,
            scope=scope,
        )
        from loom.ir import FileLocation

        loc = module.locations.get(op.location_id)
        assert isinstance(loc, FileLocation)
        assert module.sources[loc.source_id] == 'model "main"\\v2\n.loom'

    def test_fused_location(self) -> None:
        """Parse a FUSED location annotation."""
        from loom.ir import FileLocation, FusedLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)',
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        assert isinstance(loc, FusedLocation)
        assert len(loc.children) == 2
        # Each child should be a FileLocation.
        child_a = module.locations.get(loc.children[0])
        child_b = module.locations.get(loc.children[1])
        assert isinstance(child_a, FileLocation)
        assert isinstance(child_b, FileLocation)
        assert child_a.start_line == 1
        assert child_a.start_col == 1
        assert child_b.start_line == 2
        assert child_b.start_col == 2
        assert module.sources[child_a.source_id] == "a.loom"
        assert module.sources[child_b.source_id] == "b.loom"

    def test_opaque_location(self) -> None:
        """Parse an OPAQUE location annotation."""
        from loom.ir import OpaqueLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc(opaque<"torch", "node_id=42">)',
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        assert isinstance(loc, OpaqueLocation)
        assert module.sources[loc.source_id] == "torch"
        assert loc.data == b"node_id=42"

    def test_opaque_location_escapes_are_decoded(self) -> None:
        """Parse escaped tag and payload bytes in OPAQUE locations."""
        from loom.ir import OpaqueLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            r'%r = test.neg %x : f32 loc(opaque<"torch \"aten\"", "node\\id\n\u0001\u03BB">)',
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        assert isinstance(loc, OpaqueLocation)
        assert module.sources[loc.source_id] == 'torch "aten"'
        assert loc.data == b"node\\id\n\x01\xce\xbb"

    def test_no_location(self) -> None:
        """Ops without explicit location use implicit source position."""
        from loom.ir import FileLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.neg %x : f32",
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        # Implicit locations are FileLocations derived from the source text.
        assert isinstance(loc, FileLocation)
        # source_id is 0 (the default for implicit locations).
        assert loc.source_id == 0


# ============================================================================
# Location round-trip: parse with loc() → print with print_locations → match
# ============================================================================


class TestLocationRoundTrip:
    """Parse text with locations, print with locations, verify identical output."""

    def _roundtrip_with_locations(self, text: str) -> None:
        """Parse text, print with print_locations=True, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer(print_locations=True).print_module(module)

        assert printed == text, (
            f"Location round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_file_location_roundtrip(self) -> None:
        self._roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":42:3 to 42:58)\n'
            '  test.yield %neg0 : f32 loc("model.loom":43:3 to 43:28)\n'
            '} loc("model.loom":41:1 to 44:2)\n'
        )

    def test_top_level_declaration_location_roundtrip(self) -> None:
        self._roundtrip_with_locations(
            'test.decl @located() loc("model.loom":1:1 to 1:20)\n'
        )

    def test_comments_survive_explicit_location_roundtrip(self) -> None:
        self._roundtrip_with_locations(
            "test.func @located() -> (i32) {\n"
            "  // located constant\n"
            '  %c = test.constant 42 : i32 loc("model.loom":42:3 to 42:58)\n'
            '  test.yield %c : i32 loc("model.loom":43:3 to 43:28)\n'
            '} loc("model.loom":41:1 to 44:2)\n'
        )

    def test_fused_location_roundtrip(self) -> None:
        """All ops have explicit locations (fused + file)."""
        self._roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)\n'
            '  test.yield %neg0 : f32 loc("a.loom":3:1 to 3:28)\n'
            '} loc(fused<"a.loom":1:1, "b.loom":3:2>)\n'
        )

    def test_opaque_location_roundtrip(self) -> None:
        """All ops have explicit locations (opaque + file)."""
        self._roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(opaque<"torch", "node_id=42">)\n'
            '  test.yield %neg0 : f32 loc("model.loom":2:3 to 2:28)\n'
            '} loc(opaque<"torch", "node_id=func">)\n'
        )

    def test_escaped_location_roundtrip_is_canonical(self) -> None:
        """Escaped source/tag/data strings print in one canonical form."""
        self._roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(opaque<"torch \\"aten\\"", "node\\\\id\\n\\u0001">)\n'
            '  test.yield %neg0 : f32 loc("model \\"main\\"\\\\v2\\n.loom":2:3 to 2:28)\n'
            '} loc(opaque<"torch \\"aten\\"", "fn\\\\id\\n\\u0001">)\n'
        )

    def test_stable_after_two_rounds(self) -> None:
        """Parse → print → parse → print is stable (mixed implicit/explicit)."""
        text = (
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":10:3 to 10:40)\n'
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

        def parse_and_print(source: str) -> str:
            module = _op_parser().parse(source)
            return _op_printer(print_locations=True).print_module(module)

        round1 = parse_and_print(text)
        round2 = parse_and_print(round1)
        assert round1 == round2, (
            f"Not stable after two rounds.\nRound 1:\n{round1}\nRound 2:\n{round2}"
        )


# ============================================================================
# Cross-format round-trip: text → bytecode → text with locations
# ============================================================================


class TestLocationCrossFormatRoundTrip:
    """Test that locations survive text → bytecode → text."""

    def _cross_roundtrip_with_locations(
        self, text: str, expected: str | None = None
    ) -> None:
        module = _op_parser().parse(text)
        loaded = read_module(write_module(module))
        printed = _op_printer(print_locations=True).print_module(loaded)
        target = expected if expected is not None else text
        assert printed == target, (
            f"Cross-format location round-trip mismatch:\n"
            f"  expected: {target!r}\n"
            f"  got:      {printed!r}"
        )

    def test_file_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":42:3 to 42:58)\n'
            '  test.yield %neg0 : f32 loc("model.loom":43:3 to 43:28)\n'
            "}\n"
        )

    def test_opaque_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(opaque<"torch", "node_id=42">)\n'
            '  test.yield %neg0 : f32 loc("model.loom":3:3 to 3:28)\n'
            "}\n"
        )

    def test_fused_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "test.func @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)\n'
            '  test.yield %neg0 : f32 loc("model.loom":3:3 to 3:28)\n'
            "}\n"
        )


# ============================================================================
# Nested dict attribute values
# ============================================================================


class TestNestedDictAttr:
    """Test parsing of nested dict values in attribute dicts."""

    @staticmethod
    def _nested_dict_text(nested_dict_depth: int) -> str:
        """Returns a nested dict value with nested_dict_depth dict wrappers."""
        value = "7"
        for index in range(nested_dict_depth - 1, -1, -1):
            value = f"{{k{index} = {value}}}"
        return value

    def test_nested_dict(self) -> None:
        """Parse {outer = {inner = 42}} as nested dict."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {meta = {depth = 3, name = "layer0"}} : f32',
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d["meta"], CanonicalAttrDict)
        assert list(d["meta"].items()) == [("depth", 3), ("name", "layer0")]
        assert d["meta"]["depth"] == 3
        assert d["meta"]["name"] == "layer0"

    def test_deeply_nested_dict(self) -> None:
        """Parse three levels of nesting."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {a = {b = {c = 99}}} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d["a"], CanonicalAttrDict)
        assert isinstance(d["a"]["b"], CanonicalAttrDict)
        assert d["a"]["b"]["c"] == 99

    def test_nested_duplicate_key_is_rejected(self) -> None:
        module, scope = _setup_scope(("x", F32))
        with pytest.raises(ParseError, match="duplicate attribute dict key 'depth'"):
            _parse_op(
                "%r = test.attrs %x {meta = {depth = 3, depth = 4}} : f32",
                module=module,
                scope=scope,
            )

    def test_unsorted_nested_dict_cross_format_round_trip(self) -> None:
        text = (
            "test.func @f(%x: f32) -> (f32) {\n"
            '  %r = test.attrs %x {meta = {phase = "link", opt = 3}, axis = 0} : f32\n'
            "  test.yield %r : f32\n"
            "}\n"
        )
        module = _op_parser().parse(text)
        loaded = read_module(write_module(module))
        printed = _op_printer().print_module(loaded)
        assert printed == (
            "test.func @f(%x: f32) -> (f32) {\n"
            '  %r = test.attrs %x {axis = 0, meta = {opt = 3, phase = "link"}} : f32\n'
            "  test.yield %r : f32\n"
            "}\n"
        )

    def test_max_nested_dict_depth_is_accepted(self) -> None:
        # The top-level AttrDict field itself is parser depth 0, so 15 nested
        # dict values below it gives 16 total dict wrappers and is the deepest
        # valid shape accepted by the C parser/verifier contract.
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            f"%r = test.attrs %x {{meta = {self._nested_dict_text(15)}}} : f32",
            module=module,
            scope=scope,
        )
        value = op.attributes["dict"]["meta"]
        for index in range(14):
            assert isinstance(value, CanonicalAttrDict)
            value = value[f"k{index}"]
        assert value == CanonicalAttrDict([("k14", 7)])

    def test_over_max_nested_dict_depth_is_rejected(self) -> None:
        module, scope = _setup_scope(("x", F32))
        with pytest.raises(ParseError, match="maximum depth 16"):
            _parse_op(
                f"%r = test.attrs %x {{meta = {self._nested_dict_text(16)}}} : f32",
                module=module,
                scope=scope,
            )
