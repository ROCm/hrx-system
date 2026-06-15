# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the IR builder."""

import pytest

from loom.builder import IndexedValue, IRBuilder, TiedResultSpec, ValueRef, tied
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.ir import (
    F32,
    I32,
    INDEX,
    VALUE_DEF_BLOCK_NONE,
    VALUE_DEF_OP_NONE,
    Block,
    Region,
    ShapedType,
    StaticDim,
    SymbolKind,
    TypeKind,
    Use,
)

# ============================================================================
# Helpers
# ============================================================================


def _builder() -> IRBuilder:
    b = IRBuilder(insertion_block=Block())
    b.register_ops(ALL_TEST_OPS)
    b.register_types(ALL_BUILTIN_TYPES)
    return b


# ============================================================================
# ValueRef
# ============================================================================


class TestValueRef:
    def test_create(self) -> None:
        b = _builder()
        v = b.value("x", F32)
        assert isinstance(v, ValueRef)
        assert v.id == 0
        assert v.name == "x"
        assert v.type == F32

    def test_repr(self) -> None:
        b = _builder()
        v = b.value("x", F32)
        assert "x" in repr(v)

    def test_int_conversion(self) -> None:
        b = _builder()
        v = b.value("x", F32)
        assert int(v) == 0


class TestIndexedValue:
    def test_single_static(self) -> None:
        b = _builder()
        tensor = b.value("t", ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),)))
        indexed = tensor[0]
        assert isinstance(indexed, IndexedValue)
        base, static, dynamic = indexed.decompose()
        assert base == tensor.id
        assert static == [0]
        assert dynamic == []

    def test_single_dynamic(self) -> None:
        b = _builder()
        tensor = b.value("t", ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),)))
        off = b.value("off", INDEX)
        indexed = tensor[off]
        base, static, dynamic = indexed.decompose()
        assert base == tensor.id
        sentinel = -(2**63)
        assert static == [sentinel]
        assert dynamic == [off.id]

    def test_mixed(self) -> None:
        b = _builder()
        tile = b.value(
            "t", ShapedType(TypeKind.TILE, F32, (StaticDim(64), StaticDim(64)))
        )
        off = b.value("off", INDEX)
        indexed = tile[0, off]
        _base, static, dynamic = indexed.decompose()
        sentinel = -(2**63)
        assert static == [0, sentinel]
        assert dynamic == [off.id]

    def test_invalid_index_type(self) -> None:
        b = _builder()
        t = b.value("t", F32)
        indexed = t["bad"]
        with pytest.raises(TypeError, match=r"int.*or ValueRef"):
            indexed.decompose()


class TestTiedResultSpec:
    def test_tied_function(self) -> None:
        b = _builder()
        v = b.value("x", F32)
        spec = tied(v, I32)
        assert isinstance(spec, TiedResultSpec)
        assert spec.operand_id == v.id
        assert spec.result_type == I32

    def test_as_type_method(self) -> None:
        b = _builder()
        v = b.value("x", F32)
        spec = v.as_type(I32)
        assert isinstance(spec, TiedResultSpec)
        assert spec.operand_id == v.id
        assert spec.result_type == I32


# ============================================================================
# Generic build()
# ============================================================================


class TestBuild:
    def test_binary_op(self) -> None:
        b = _builder()
        a = b.value("a", I32)
        c = b.value("b", I32)
        result = b.build("test.addi", [a, c], results=[I32], result_names=["r"])
        assert isinstance(result, ValueRef)
        assert result.name == "r"
        assert result.type == I32

    def test_unary_op(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        result = b.build("test.neg", [x], results=[F32])
        assert isinstance(result, ValueRef)
        assert result.type == F32

    def test_constant(self) -> None:
        b = _builder()
        result = b.build(
            "test.constant",
            [],
            results=[I32],
            result_names=["c42"],
            attributes={"value": 42},
        )
        assert isinstance(result, ValueRef)
        assert result.name == "c42"

    def test_no_results(self) -> None:
        b = _builder()
        a = b.value("a", F32)
        result = b.build("test.yield", [a])
        assert result is None

    def test_multi_result(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = _builder()
        w = b.value("weights", tile_t)
        x = b.value("input", INDEX)
        results = b.build(
            "test.invoke",
            [w, x],
            results=[tile_t, INDEX],
            result_names=["out", "count"],
            attributes={"callee": "compute"},
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].name == "out"
        assert results[1].name == "count"

    def test_tied_result_with_spec(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        b = _builder()
        tile = b.value("tile", tile_t)
        tensor = b.value("tensor", tensor_t)
        off = b.value("off", INDEX)
        sentinel = -(2**63)
        result = b.build(
            "test.update",
            [tile, tensor, off],
            results=[tied(tensor, tensor_t)],
            attributes={"static_offsets": [sentinel]},
        )
        assert isinstance(result, ValueRef)

    def test_tied_with_as_type(self) -> None:
        """Use .as_type() method for tied result."""
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        b = _builder()
        tile = b.value("tile", tile_t)
        tensor = b.value("tensor", tensor_t)
        off = b.value("off", INDEX)
        sentinel = -(2**63)
        result = b.build(
            "test.update",
            [tile, tensor, off],
            results=[tensor.as_type(tensor_t)],
            attributes={"static_offsets": [sentinel]},
        )
        assert isinstance(result, ValueRef)

    def test_call_with_mixed_results(self) -> None:
        """func.call-like: some results tied, some fresh."""
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = _builder()
        weights = b.value("weights", tile_t)
        input_val = b.value("input", INDEX)
        results = b.build(
            "test.invoke",
            [weights, input_val],
            results=[tied(weights, tile_t), INDEX],
            result_names=["out", "count"],
            attributes={"callee": "compute"},
        )
        assert isinstance(results, list)
        assert len(results) == 2
        # First result tied to weights, second is fresh.
        assert results[0].type == tile_t
        assert results[1].type == INDEX

    def test_unknown_op_fails(self) -> None:
        b = _builder()
        with pytest.raises(ValueError, match="Unknown op"):
            b.build("nonexistent.op", [])

    def test_auto_named_results(self) -> None:
        b = _builder()
        a = b.value("a", I32)
        c = b.value("b", I32)
        result = b.build("test.addi", [a, c], results=[I32])
        assert isinstance(result, ValueRef)
        assert result.name == ""  # Unnamed, printer assigns at print time.

    def test_body_op_is_inserted_into_current_block(self) -> None:
        b = _builder()
        a = b.value("a", I32)
        c = b.value("b", I32)
        result = b.build("test.addi", [a, c], results=[I32], result_names=["sum"])
        assert isinstance(result, ValueRef)
        assert b.insertion_block is not None
        assert len(b.insertion_block.ops) == 1
        op = b.insertion_block.ops[0]
        assert op.name == "test.addi"
        assert op.operands == [a.id, c.id]
        assert op.results == [result.id]

    def test_symbol_op_is_inserted_into_module_symbols(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        body = Region(blocks=[Block(arg_ids=[x.id])])
        result = b.build(
            "test.func",
            func_args=[x],
            results=[F32],
            attributes={"callee": "identity", "visibility": "public"},
            regions=[body],
        )
        assert isinstance(result, list)
        assert len(b.module.symbols) == 1
        symbol = b.module.symbols[0]
        assert symbol.name == "identity"
        assert symbol.kind == SymbolKind.FUNC_DEF
        assert symbol.is_public
        assert symbol.op is not None
        assert symbol.op.name == "test.func"
        assert symbol.op.regions[0].blocks[0].arg_ids == [x.id]

    def test_non_symbol_without_insertion_block_fails(self) -> None:
        b = IRBuilder()
        b.register_ops(ALL_TEST_OPS)
        a = b.value("a", I32)
        c = b.value("b", I32)
        with pytest.raises(ValueError, match="no insertion block is set"):
            b.build("test.addi", [a, c], results=[I32])

    def test_decl_signature_args_resolve_tied_results(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        result = b.build(
            "test.decl",
            func_args=[x],
            results=[x.as_type(F32)],
            attributes={"callee": "identity"},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].type == F32
        assert len(b.module.symbols) == 1
        assert b.module.symbols[0].op is not None
        assert b.module.symbols[0].op.operands == [x.id]
        assert b.module.symbols[0].op.tied_results[0].operand_index == 0

        arg = b.module.values[x.id]
        assert not arg.is_block_arg
        assert arg.def_op_index == 0
        assert arg.def_block_index == VALUE_DEF_BLOCK_NONE
        assert arg.def_result_index == 0
        assert arg.uses == []

        result_value = b.module.values[result[0].id]
        assert result_value.def_op_index == 0
        assert result_value.def_block_index == VALUE_DEF_BLOCK_NONE
        assert result_value.def_result_index == 0

    def test_func_signature_args_seed_body_entry_block(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        body = Region(blocks=[Block()])
        result = b.build(
            "test.func",
            func_args=[x],
            results=[x.as_type(F32)],
            attributes={"callee": "identity"},
            regions=[body],
        )
        assert isinstance(result, list)
        assert body.blocks[0].arg_ids == [x.id]
        assert len(b.module.symbols) == 1
        assert b.module.symbols[0].op is not None
        assert b.module.symbols[0].op.operands == []
        assert b.module.symbols[0].op.tied_results[0].operand_index == 0

        arg = b.module.values[x.id]
        assert arg.is_block_arg
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 0
        assert arg.def_result_index == 0

    def test_func_args_on_non_func_like_op_fails(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        with pytest.raises(ValueError, match="does not implement FuncLikeInterface"):
            b.build("test.neg", [x], func_args=[x], results=[F32])

    def test_attached_body_op_records_value_metadata_and_uses(self) -> None:
        b = _builder()
        x = b.value("x", F32)
        body = Region(blocks=[Block()])
        b.build(
            "test.func",
            func_args=[x],
            results=[x.as_type(F32)],
            attributes={"callee": "identity"},
            regions=[body],
        )
        b.set_insertion_block(body.blocks[0])

        result = b.build("test.neg", [x], results=[F32], result_names=["neg"])
        assert isinstance(result, ValueRef)

        arg = b.module.values[x.id]
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=0)]

        result_value = b.module.values[result.id]
        assert not result_value.is_block_arg
        assert result_value.def_op_index == 0
        assert result_value.def_block_index == 0
        assert result_value.def_result_index == 0
