# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom.ir — in-memory IR representation."""

import math
from itertools import repeat

import pytest

from loom.dialect.test import (
    test_array_type,
    test_compact_attr,
    test_feature_set_attr,
    test_matrix_type,
    test_node_attr,
    test_options_attr,
    test_scope_type,
    test_tile_attr,
    test_variant_set_type,
)
from loom.dsl import AttrDef, Dialect, ParameterizedAttrDef, SymbolReference
from loom.ir import (
    BF16,
    BUFFER_TYPE,
    ENCODING_LAYOUT_TYPE,
    ENCODING_SCHEMA_TYPE,
    ENCODING_STORAGE_TYPE,
    ENCODING_TRANSFORM_TYPE,
    ENCODING_TYPE,
    F16,
    F32,
    F64,
    I1,
    I8,
    I32,
    I64,
    INDEX,
    LOCATION_FLAG_SYNTHETIC,
    LOCATION_UNKNOWN,
    NONE_TYPE,
    SYMBOL_FLAG_PUBLIC,
    VALUE_DEF_BLOCK_NONE,
    VALUE_DEF_OP_NONE,
    VALUE_FLAG_BLOCK_ARG,
    VALUE_FLAG_CONSUMED,
    Block,
    BufferType,
    CanonicalAttrDict,
    Context,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingRole,
    EncodingType,
    EnumArrayAttr,
    FileLocation,
    FunctionType,
    FusedLocation,
    LocationTable,
    Module,
    OpaqueLocation,
    Operation,
    ParameterizedAttr,
    ParameterizedAttrArray,
    ParameterizedType,
    PoolType,
    Predicate,
    PredicateArg,
    Region,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    SignedEnumSetAttr,
    StaticDim,
    StorageSpace,
    StorageType,
    StringTable,
    Symbol,
    SymbolKind,
    SymbolName,
    SymbolNameArray,
    SymbolNameSet,
    SymbolRef,
    TaggedLocation,
    TiedResult,
    TypeKind,
    TypeTable,
    Use,
    Value,
    evaluate_predicate,
    evaluate_predicates,
    parse_scalar_type_kind,
    rebuild_value_metadata,
    record_block_value_metadata,
    record_operation_value_metadata,
    replace_canonical_attr_dict,
    scalar_type_name,
)

# ============================================================================
# Scalar types
# ============================================================================


class TestScalarTypes:
    def test_all_kinds_have_names(self) -> None:
        for kind in ScalarTypeKind:
            name = scalar_type_name(kind)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_parse_round_trip(self) -> None:
        for kind in ScalarTypeKind:
            name = scalar_type_name(kind)
            parsed = parse_scalar_type_kind(name)
            assert parsed == kind

    def test_parse_unknown_returns_none(self) -> None:
        assert parse_scalar_type_kind("float128") is None
        assert parse_scalar_type_kind("") is None
        assert parse_scalar_type_kind("i") is None

    def test_scalar_type_repr(self) -> None:
        assert repr(F32) == "f32"
        assert repr(INDEX) == "index"
        assert repr(I8) == "i8"
        assert repr(BF16) == "bf16"

    def test_scalar_type_equality(self) -> None:
        assert F32 == ScalarType(ScalarTypeKind.F32)
        assert F32 != I32
        assert INDEX != I64

    def test_scalar_type_hash(self) -> None:
        s = {F32, I32, F32}
        assert len(s) == 2

    def test_bitwidth(self) -> None:
        assert I1.bitwidth == 1
        assert I8.bitwidth == 8
        assert I32.bitwidth == 32
        assert F16.bitwidth == 16
        assert F32.bitwidth == 32
        assert F64.bitwidth == 64
        assert INDEX.bitwidth == 64
        assert BF16.bitwidth == 16

    def test_singleton_equality(self) -> None:
        """Module-level singletons are equal to freshly constructed types."""
        assert F32 == ScalarType(ScalarTypeKind.F32)
        assert I32 == ScalarType(ScalarTypeKind.I32)


# ============================================================================
# Dimensions
# ============================================================================


class TestDimensions:
    def test_static_dim(self) -> None:
        d = StaticDim(4)
        assert d.size == 4
        assert repr(d) == "4"

    def test_dynamic_dim(self) -> None:
        d = DynamicDim()
        assert repr(d) == "?"

    def test_dim_equality(self) -> None:
        assert StaticDim(4) == StaticDim(4)
        assert StaticDim(4) != StaticDim(8)
        assert DynamicDim() == DynamicDim()
        assert StaticDim(4) != DynamicDim()  # type: ignore[comparison-overlap]

    def test_dim_hashable(self) -> None:
        s = {StaticDim(4), StaticDim(4), DynamicDim(), DynamicDim()}
        assert len(s) == 2


# ============================================================================
# Shaped types
# ============================================================================


class TestShapedTypes:
    def test_tile_static(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(4)))
        assert t.type_kind == TypeKind.TILE
        assert t.rank == 2
        assert t.is_all_static
        assert not t.has_encoding
        assert repr(t) == "tile<4x4xf32>"

    def test_tensor_dynamic(self) -> None:
        t = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(), StaticDim(4)))
        assert t.type_kind == TypeKind.TENSOR
        assert t.rank == 2
        assert not t.is_all_static
        assert repr(t) == "tensor<?x4xf32>"

    def test_tile_0d(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, ())
        assert t.rank == 0
        assert t.is_all_static
        assert repr(t) == "tile<f32>"

    def test_tile_1d(self) -> None:
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),))
        assert t.rank == 1
        assert repr(t) == "tile<256xi8>"

    def test_tile_with_encoding(self) -> None:
        enc = EncodingInstance(name="q8_0", params=(("block", 32),))
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(256),), encoding=enc)
        assert t.has_encoding
        assert isinstance(t.encoding, EncodingInstance)
        assert t.encoding.name == "q8_0"
        assert t.encoding.params == (("block", 32),)

    def test_view_with_layout(self) -> None:
        layout = EncodingInstance(name="strided", params=(("stride", 64),))
        t = ShapedType(TypeKind.VIEW, F32, (DynamicDim(),), encoding=layout)
        assert t.type_kind == TypeKind.VIEW
        assert t.rank == 1
        assert t.has_encoding
        assert repr(t) == "view<?xf32>"

    def test_vector_static(self) -> None:
        t = ShapedType(TypeKind.VECTOR, F32, (StaticDim(16),))
        assert t.type_kind == TypeKind.VECTOR
        assert t.rank == 1
        assert t.is_all_static
        assert repr(t) == "vector<16xf32>"

    def test_vector_zero_extent_is_empty_not_rank_zero(self) -> None:
        t = ShapedType(TypeKind.VECTOR, F32, (StaticDim(0),))
        assert t.type_kind == TypeKind.VECTOR
        assert t.rank == 1
        assert t.is_all_static
        assert repr(t) == "vector<0xf32>"

    def test_vector_dynamic(self) -> None:
        t = ShapedType(TypeKind.VECTOR, I32, (DynamicDim(),))
        assert t.type_kind == TypeKind.VECTOR
        assert not t.is_all_static
        assert repr(t) == "vector<?xi32>"

    def test_vector_0d_rejected(self) -> None:
        with pytest.raises(ValueError, match="rank >= 1"):
            ShapedType(TypeKind.VECTOR, F32, ())

    def test_vector_encoding_rejected(self) -> None:
        enc = EncodingInstance(name="dense")
        with pytest.raises(ValueError, match="must not carry"):
            ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),), encoding=enc)

    def test_shaped_type_equality(self) -> None:
        a = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        c = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        assert a == b
        assert a != c  # Different kind (tile vs tensor).

    def test_shaped_type_hashable(self) -> None:
        a = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be TILE, TENSOR, VECTOR, or VIEW"):
            ShapedType(TypeKind.SCALAR, F32, ())


# ============================================================================
# Other types
# ============================================================================


def test_unassigned_type_kind_is_invalid() -> None:
    with pytest.raises(ValueError, match="4 is not a valid TypeKind"):
        TypeKind(4)


class TestFunctionType:
    def test_basic(self) -> None:
        t = FunctionType((F32, I32), (F32,))
        assert t.type_kind == TypeKind.FUNCTION
        assert len(t.arg_types) == 2
        assert len(t.result_types) == 1

    def test_repr(self) -> None:
        t = FunctionType((F32,), (I32,))
        assert repr(t) == "(f32) -> (i32)"


class TestNoneType:
    def test_singleton(self) -> None:
        assert NONE_TYPE.type_kind == TypeKind.NONE
        assert repr(NONE_TYPE) == "none"


class TestEncodingType:
    def test_singleton(self) -> None:
        assert ENCODING_TYPE.type_kind == TypeKind.ENCODING
        assert repr(ENCODING_TYPE) == "encoding"
        assert ENCODING_LAYOUT_TYPE.type_kind == TypeKind.ENCODING
        assert repr(ENCODING_LAYOUT_TYPE) == "encoding<layout>"
        assert repr(ENCODING_SCHEMA_TYPE) == "encoding<schema>"
        assert repr(ENCODING_STORAGE_TYPE) == "encoding<storage>"
        assert repr(ENCODING_TRANSFORM_TYPE) == "encoding<transform>"

    def test_equality(self) -> None:
        assert EncodingType() == EncodingType()
        assert EncodingType() == ENCODING_TYPE
        assert EncodingType(EncodingRole.LAYOUT) == ENCODING_LAYOUT_TYPE
        assert ENCODING_LAYOUT_TYPE != ENCODING_SCHEMA_TYPE

    def test_hashable(self) -> None:
        s = {EncodingType(), ENCODING_TYPE, EncodingType()}
        assert len(s) == 1
        role_set = {ENCODING_LAYOUT_TYPE, ENCODING_SCHEMA_TYPE, EncodingType()}
        assert len(role_set) == 3


class TestBufferType:
    def test_singleton(self) -> None:
        assert BUFFER_TYPE.type_kind == TypeKind.BUFFER
        assert repr(BUFFER_TYPE) == "buffer"

    def test_equality(self) -> None:
        assert BufferType() == BufferType()
        assert BufferType() == BUFFER_TYPE

    def test_hashable(self) -> None:
        s = {BufferType(), BUFFER_TYPE, BufferType()}
        assert len(s) == 1


class TestStorageType:
    def test_basic(self) -> None:
        t = StorageType(StorageSpace.WORKGROUP)
        assert t.space == StorageSpace.WORKGROUP
        assert t.type_kind == TypeKind.STORAGE
        assert repr(t) == "low.storage<workgroup>"

    def test_equality(self) -> None:
        assert StorageType(StorageSpace.WORKGROUP) == StorageType(
            StorageSpace.WORKGROUP
        )
        assert StorageType(StorageSpace.WORKGROUP) != StorageType(StorageSpace.PRIVATE)

    def test_hashable(self) -> None:
        a = StorageType(StorageSpace.WORKGROUP)
        b = StorageType(StorageSpace.WORKGROUP)
        assert len({a, b}) == 1


class TestDynamicEncoding:
    def test_hashable(self) -> None:
        a = DynamicEncoding()
        b = DynamicEncoding()
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestShapedTypeWithDynamicEncoding:
    def test_dynamic_encoding_creates(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding())
        assert t.has_encoding
        assert t.has_dynamic_encoding
        assert not t.has_static_encoding

    def test_static_encoding_properties(self) -> None:
        enc = EncodingInstance(name="q8_0")
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        assert t.has_encoding
        assert t.has_static_encoding
        assert not t.has_dynamic_encoding

    def test_no_encoding_properties(self) -> None:
        t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        assert not t.has_encoding
        assert not t.has_static_encoding
        assert not t.has_dynamic_encoding

    def test_distinct_from_no_encoding(self) -> None:
        a = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding())
        b = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        assert a != b

    def test_distinct_from_static_encoding(self) -> None:
        a = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding())
        enc = EncodingInstance(name="q8_0")
        b = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        assert a != b


# ============================================================================
# Pool type
# ============================================================================


class TestPoolType:
    def test_static_block_size(self) -> None:
        t = PoolType(block_size=StaticDim(65536))
        assert t.type_kind == TypeKind.POOL
        assert not t.has_dynamic_block_size
        assert repr(t) == "pool<65536>"

    def test_dynamic_block_size(self) -> None:
        t = PoolType(block_size=DynamicDim())
        assert t.type_kind == TypeKind.POOL
        assert t.has_dynamic_block_size
        assert repr(t) == "pool<?>"

    def test_equality(self) -> None:
        a = PoolType(block_size=StaticDim(4096))
        b = PoolType(block_size=StaticDim(4096))
        c = PoolType(block_size=StaticDim(65536))
        assert a == b
        assert a != c

    def test_hashable(self) -> None:
        a = PoolType(block_size=StaticDim(4096))
        b = PoolType(block_size=StaticDim(4096))
        assert len({a, b}) == 1

    def test_dynamic_vs_static(self) -> None:
        a = PoolType(block_size=DynamicDim())
        b = PoolType(block_size=StaticDim(4096))
        assert a != b


# ============================================================================
# Locations
# ============================================================================


class TestLocations:
    def test_file_location(self) -> None:
        loc = FileLocation(
            source_id=0, start_line=42, start_col=3, end_line=42, end_col=58
        )
        assert loc.start_line == 42
        assert loc.end_col == 58
        assert loc.flags == 0

    def test_file_location_synthetic(self) -> None:
        loc = FileLocation(
            source_id=0,
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=1,
            flags=LOCATION_FLAG_SYNTHETIC,
        )
        assert loc.flags & LOCATION_FLAG_SYNTHETIC

    def test_fused_location(self) -> None:
        loc = FusedLocation(children=(1, 2, 3))
        assert len(loc.children) == 3
        assert loc.flags == 0

    def test_opaque_location(self) -> None:
        loc = OpaqueLocation(source_id=1, data=b"node_id=42")
        assert loc.data == b"node_id=42"
        assert loc.flags == 0

    def test_tagged_location(self) -> None:
        loc = TaggedLocation(tag=1, child=2, data=b"\x01\x2a\xff")
        assert loc.tag == 1
        assert loc.child == 2
        assert loc.data == b"\x01\x2a\xff"
        assert loc.flags == 0

    def test_unknown_is_zero(self) -> None:
        assert LOCATION_UNKNOWN == 0

    def test_file_location_hashable(self) -> None:
        a = FileLocation(0, 1, 1, 1, 10)
        b = FileLocation(0, 1, 1, 1, 10)
        assert hash(a) == hash(b)
        assert a == b
        s = {a, b}
        assert len(s) == 1

    def test_fused_location_hashable(self) -> None:
        a = FusedLocation(children=(1, 2))
        b = FusedLocation(children=(1, 2))
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_tagged_location_hashable(self) -> None:
        a = TaggedLocation(tag=1, child=2, data=b"\x01")
        b = TaggedLocation(tag=1, child=2, data=b"\x01")
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1


# ============================================================================
# Tied results
# ============================================================================


class TestTiedResult:
    def test_basic(self) -> None:
        tr = TiedResult(result_index=0, operand_index=1)
        assert tr.result_index == 0
        assert tr.operand_index == 1
        assert not tr.has_type_change

    def test_with_type_change(self) -> None:
        tr = TiedResult(result_index=0, operand_index=1, has_type_change=True)
        assert tr.has_type_change


# ============================================================================
# Values
# ============================================================================


class TestValues:
    def test_basic_value(self) -> None:
        v = Value(name="x", type=F32)
        assert v.name == "x"
        assert v.type == F32
        assert not v.is_block_arg
        assert not v.is_consumed

    def test_block_arg(self) -> None:
        v = Value(name="arg0", type=F32, flags=VALUE_FLAG_BLOCK_ARG)
        assert v.is_block_arg

    def test_consumed(self) -> None:
        v = Value(name="t", type=F32, flags=VALUE_FLAG_CONSUMED)
        assert v.is_consumed

    def test_dim_bindings(self) -> None:
        ty = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        v = Value(name="r", type=ty, dim_bindings={0: 42})
        assert v.dim_bindings[0] == 42

    def test_encoding_binding_default(self) -> None:
        v = Value(name="x", type=F32)
        assert v.encoding_binding == -1

    def test_encoding_binding(self) -> None:
        ty = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding())
        v = Value(name="t", type=ty, encoding_binding=42)
        assert v.encoding_binding == 42

    def test_definition_defaults_are_explicitly_unattached(self) -> None:
        v = Value(name="x", type=F32)
        assert v.def_op_index == VALUE_DEF_OP_NONE
        assert v.def_block_index == VALUE_DEF_BLOCK_NONE
        assert v.def_result_index == 0

    def test_uses(self) -> None:
        v = Value(name="x", type=F32)
        v.uses.append(Use(user_op_index=0, operand_index=0, block_index=0))
        assert len(v.uses) == 1

    def test_record_block_and_operation_value_metadata(self) -> None:
        module = Module()
        arg_id = module.add_value(Value(name="arg", type=F32))
        result_id = module.add_value(Value(name="r", type=F32))
        block = Block(
            arg_ids=[arg_id],
            ops=[Operation(name="test.neg", operands=[arg_id], results=[result_id])],
        )
        record_block_value_metadata(module, block, block_index=2)

        arg = module.values[arg_id]
        assert arg.is_block_arg
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 2
        assert arg.def_result_index == 0
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=2)]

        result = module.values[result_id]
        assert not result.is_block_arg
        assert result.def_op_index == 0
        assert result.def_block_index == 2
        assert result.def_result_index == 0

    def test_record_operation_value_metadata_supports_operand_defs(self) -> None:
        module = Module()
        arg_id = module.add_value(Value(name="x", type=F32))
        result_id = module.add_value(Value(name="r", type=F32))
        op = Operation(name="test.decl", operands=[arg_id], results=[result_id])
        record_operation_value_metadata(
            module,
            op,
            block_index=VALUE_DEF_BLOCK_NONE,
            op_index=5,
            operand_def_count=1,
        )

        arg = module.values[arg_id]
        assert not arg.is_block_arg
        assert arg.def_op_index == 5
        assert arg.def_block_index == VALUE_DEF_BLOCK_NONE
        assert arg.def_result_index == 0
        assert arg.uses == []

        result = module.values[result_id]
        assert result.def_op_index == 5
        assert result.def_block_index == VALUE_DEF_BLOCK_NONE
        assert result.def_result_index == 0


# ============================================================================
# Operations
# ============================================================================


class TestCanonicalAttrDict:
    def test_entries_are_sorted_and_nested_dicts_are_canonicalized(self) -> None:
        attrs = CanonicalAttrDict(
            [("z", 1), ("meta", {"phase": "link", "opt": 3}), ("axis", 0)]
        )
        assert list(attrs.items()) == [
            ("axis", 0),
            ("meta", {"opt": 3, "phase": "link"}),
            ("z", 1),
        ]
        assert isinstance(attrs["meta"], CanonicalAttrDict)
        assert list(attrs["meta"].items()) == [("opt", 3), ("phase", "link")]

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate attribute dict key 'axis'"):
            CanonicalAttrDict([("axis", 0), ("axis", 1)])

    def test_from_sorted_items_rejects_unsorted_input(self) -> None:
        with pytest.raises(ValueError, match="must be sorted in canonical order"):
            CanonicalAttrDict.from_sorted_items([("z", 1), ("a", 2)])

    def test_replace_canonical_attr_dict_updates_and_removes_keys(self) -> None:
        attrs = CanonicalAttrDict(
            [("phase", "compile"), ("config", {"threads": 4, "debug": True})]
        )
        updated = replace_canonical_attr_dict(
            attrs,
            {
                "config": {"threads": 8, "debug": None},
                "phase": None,
                "target": "llvm-cpu",
            },
        )
        assert isinstance(updated, CanonicalAttrDict)
        assert list(updated.items()) == [
            ("config", {"debug": None, "threads": 8}),
            ("target", "llvm-cpu"),
        ]
        assert isinstance(updated["config"], CanonicalAttrDict)


class TestParameterizedAttr:
    def test_canonicalizes_symbol_array_parameters(self) -> None:
        family = ParameterizedAttrDef(
            "test.providers",
            group=Dialect("test"),
            parameters=[
                AttrDef(
                    "values",
                    "symbol_array",
                    symbol_ref=SymbolReference("record", ["record"]),
                )
            ],
        )

        value = ParameterizedAttr(
            family,
            {
                "values": [
                    SymbolName("b"),
                    SymbolName("a"),
                    SymbolName("b"),
                ]
            },
        )

        assert value.get("values") == SymbolNameArray(
            [SymbolName("b"), SymbolName("a"), SymbolName("b")]
        )

    def test_rejects_untyped_symbol_array_parameters(self) -> None:
        family = ParameterizedAttrDef(
            "test.providers",
            group=Dialect("test"),
            parameters=[
                AttrDef(
                    "values",
                    "symbol_array",
                    symbol_ref=SymbolReference("record", ["record"]),
                )
            ],
        )

        with pytest.raises(TypeError, match="symbol name array element 0"):
            ParameterizedAttr(family, {"values": ["untyped"]})

    def test_compact_primary_preserves_named_declaration_order_storage(self) -> None:
        compact = test_compact_attr(label="wave", value=64)

        assert compact.slots == ("wave", 64)
        assert compact.present_items() == (("label", "wave"), ("value", 64))
        assert compact.definition.primary_parameter_index == 1

    def test_constructs_declaration_order_slots_and_nested_values(self) -> None:
        tile = test_tile_attr(width=16)
        options = test_options_attr(
            mode="fast",
            scopes=["subgroup", 254],
            element_type=BF16,
            tile=tile,
            tiles=[tile, test_tile_attr(width=8)],
            target=SymbolName("target"),
        )

        assert isinstance(options, ParameterizedAttr)
        assert options.family_name == "test.options"
        assert options.slots == (
            1,
            EnumArrayAttr([2, 254]),
            BF16,
            tile,
            SymbolName("target"),
            ParameterizedAttrArray([tile, test_tile_attr(width=8)]),
        )
        assert options.present_items() == (
            ("mode", 1),
            ("scopes", EnumArrayAttr([2, 254])),
            ("element_type", BF16),
            ("tile", tile),
            ("target", SymbolName("target")),
            ("tiles", ParameterizedAttrArray([tile, test_tile_attr(width=8)])),
        )

    def test_preserves_absent_and_present_empty_optional_values(self) -> None:
        absent = test_options_attr(mode="precise")
        present_empty = test_options_attr(mode="precise", scopes=[])

        assert not absent.has("scopes")
        assert absent.get("scopes") is None
        assert absent.slots == (2, None, None, None, None, None)
        assert present_empty.has("scopes")
        assert present_empty.get("scopes") == EnumArrayAttr()
        assert absent != present_empty

    def test_rejects_missing_unknown_and_wrong_family_parameters(self) -> None:
        with pytest.raises(TypeError, match="missing required parameter 'mode'"):
            test_options_attr()
        with pytest.raises(TypeError, match=r"unknown parameter.*mood"):
            test_options_attr(mode="fast", mood="precise")
        with pytest.raises(ValueError, match=r"has family 'test\.options'"):
            test_options_attr(mode="fast", tile=test_options_attr(mode="precise"))
        with pytest.raises(ValueError, match=r"element 0 has family 'test\.options'"):
            test_options_attr(
                mode="fast",
                tiles=[test_options_attr(mode="precise")],
            )

    def test_rejects_closed_enum_values_and_bad_scalar_types(self) -> None:
        with pytest.raises(ValueError, match="undeclared enum value 255"):
            test_options_attr(mode=255)
        with pytest.raises(TypeError, match="must be an enum keyword or byte"):
            test_options_attr(mode=True)
        with pytest.raises(TypeError, match="signed 64-bit integer"):
            test_tile_attr(width=True)

    def test_equality_and_hash_use_stable_family_and_frozen_values(self) -> None:
        lhs = test_options_attr(mode="fast", scopes=["subgroup"])
        rhs = test_options_attr(mode=1, scopes=EnumArrayAttr([2]))

        assert lhs == rhs
        assert hash(lhs) == hash(rhs)
        assert {lhs, rhs} == {lhs}

    def test_signed_enum_set_parameter_resolves_sparse_domain(self) -> None:
        features = test_feature_set_attr(features={"high": True, "low": False, 7: True})

        assert features.slots == (SignedEnumSetAttr([7, 255], [1]),)

    def test_signed_enum_set_parameter_rejects_invalid_assertions(self) -> None:
        with pytest.raises(ValueError, match="undeclared enum value 2"):
            test_feature_set_attr(features={2: True})
        with pytest.raises(TypeError, match="exact Boolean assertions"):
            test_feature_set_attr(features={"low": 1})


class TestParameterizedAttrArray:
    def test_preserves_order_repetition_and_mixed_families(self) -> None:
        tile = test_tile_attr(width=8)
        options = test_options_attr(mode="fast")

        values = ParameterizedAttrArray([tile, options, tile])

        assert values.values == (tile, options, tile)
        assert tuple(values) == (tile, options, tile)
        assert len(values) == 3

    def test_empty_array_is_immutable_and_hashable(self) -> None:
        assert ParameterizedAttrArray() == ParameterizedAttrArray([])
        assert hash(ParameterizedAttrArray()) == hash(ParameterizedAttrArray([]))

    def test_rejects_non_parameterized_elements_and_excessive_length(self) -> None:
        with pytest.raises(TypeError, match="element 0 must be a ParameterizedAttr"):
            ParameterizedAttrArray([42])  # type: ignore[list-item]

        tile = test_tile_attr(width=8)
        with pytest.raises(ValueError, match="exceeds UINT16_MAX"):
            ParameterizedAttrArray(repeat(tile, 0x10000))

    def test_recursively_nests_open_family_arrays(self) -> None:
        leaf = test_node_attr(value=0)
        child = test_node_attr(value=1, children=[leaf])
        root = test_node_attr(
            value=2,
            children=[child, test_tile_attr(width=8), child],
        )

        children = root.get("children")
        assert isinstance(children, ParameterizedAttrArray)
        assert children.values == (child, test_tile_attr(width=8), child)


class TestParameterizedType:
    def test_constructs_positional_and_mixed_format_families(self) -> None:
        scope = test_scope_type(scope="subgroup")
        matrix = test_matrix_type(
            element_type=BF16,
            scope="workgroup",
            rows=16,
            target=SymbolName("target"),
        )

        assert isinstance(scope, ParameterizedType)
        assert scope.type_kind == TypeKind.PARAMETERIZED
        assert scope.slots == (2,)
        assert matrix.family_name == "test.matrix"
        assert matrix.present_items() == (
            ("element_type", BF16),
            ("scope", 1),
            ("rows", 16),
            ("target", SymbolName("target")),
        )

    def test_equality_hash_and_validation_use_named_schema(self) -> None:
        lhs = test_matrix_type(element_type=BF16, scope="subgroup", rows=16)
        rhs = test_matrix_type(element_type=BF16, scope=2, rows=16)

        assert lhs == rhs
        assert hash(lhs) == hash(rhs)
        with pytest.raises(TypeError, match="missing required parameter 'rows'"):
            test_matrix_type(element_type=BF16, scope="subgroup")
        with pytest.raises(TypeError, match=r"unknown parameter.*columns"):
            test_matrix_type(
                element_type=BF16,
                scope="subgroup",
                rows=16,
                columns=16,
            )

    def test_optional_parameter_has_explicit_presence(self) -> None:
        packed = test_array_type(element_type=BF16)
        aligned = test_array_type(element_type=BF16, alignment=16)

        assert packed.slots == (BF16, None, None)
        assert not packed.has("alignment")
        assert packed.get("alignment") is None
        assert aligned.has("alignment")
        assert aligned.get("alignment") == 16

    def test_dictionary_parameter_preserves_nested_family_values(self) -> None:
        metadata = CanonicalAttrDict(
            (("tile", test_tile_attr(width=8)), ("purpose", "scratch"))
        )
        array = test_array_type(element_type=BF16, metadata=metadata)

        assert array.has("metadata")
        assert array.get("metadata") == metadata

    def test_parameterized_array_parameters_preserve_family_contracts(self) -> None:
        tile = test_tile_attr(width=8)
        options = test_options_attr(mode="fast")
        variants = test_variant_set_type(
            values=[tile, options, tile],
            alternatives=[],
        )

        assert variants.get("values") == ParameterizedAttrArray([tile, options, tile])
        assert variants.has("alternatives")
        assert variants.get("alternatives") == ParameterizedAttrArray()


class TestOperations:
    def test_basic_op(self) -> None:
        op = Operation(kind=1, name="test.unary", operands=[0], results=[1])
        assert op.kind == 1
        assert op.name == "test.unary"
        assert len(op.operands) == 1
        assert len(op.results) == 1
        assert not op.is_dead

    def test_op_with_tied(self) -> None:
        op = Operation(
            kind=2,
            name="test.tied",
            operands=[0, 1],
            results=[2],
            tied_results=[TiedResult(0, 1)],
        )
        assert len(op.tied_results) == 1
        assert op.tied_results[0].operand_index == 1

    def test_op_with_regions(self) -> None:
        block = Block(ops=[Operation(kind=0, name="test.terminator")])
        region = Region(blocks=[block])
        op = Operation(kind=3, name="test.region", regions=[region])
        assert len(op.regions) == 1
        assert len(op.regions[0].blocks) == 1

    def test_op_with_attributes(self) -> None:
        op = Operation(
            kind=4, name="test.attrs", attributes={"combine": "add", "axis": 0}
        )
        assert isinstance(op.attributes, CanonicalAttrDict)
        assert list(op.attributes.items()) == [("axis", 0), ("combine", "add")]
        assert op.attributes["axis"] == 0
        assert op.attributes["combine"] == "add"

    def test_enum_array_attribute_preserves_stable_values(self) -> None:
        values = EnumArrayAttr([1, 255, 1])
        op = Operation(
            kind=4,
            name="test.enum_array_attrs",
            attributes={"required_values": values},
        )

        assert op.attributes["required_values"] is values
        assert values.values == (1, 255, 1)
        assert len(EnumArrayAttr()) == 0

    def test_enum_array_attribute_rejects_invalid_values(self) -> None:
        for values in ([False], [-1], [256], ["low"]):
            with pytest.raises(ValueError, match="enum array element"):
                EnumArrayAttr(values)  # type: ignore[arg-type]

    def test_signed_enum_set_is_canonical_and_distinguishes_polarity(self) -> None:
        value = SignedEnumSetAttr([255, 1], [7])

        assert value.positive_values == (1, 255)
        assert value.negative_values == (7,)
        assert value == SignedEnumSetAttr([1, 255], [7])
        assert hash(value) == hash(SignedEnumSetAttr([1, 255], [7]))
        assert SignedEnumSetAttr() != value

    def test_signed_enum_set_rejects_invalid_and_ambiguous_values(self) -> None:
        with pytest.raises(ValueError, match="duplicate positive"):
            SignedEnumSetAttr([1, 1])
        with pytest.raises(ValueError, match="contradictory assertions"):
            SignedEnumSetAttr([1], [1])
        for value in (False, -1, 256, "low"):
            with pytest.raises(ValueError, match="signed enum set positive"):
                SignedEnumSetAttr([value])  # type: ignore[list-item]

    def test_symbol_name_array_preserves_order_and_duplicates(self) -> None:
        values = SymbolNameArray(
            [
                SymbolName("provider_b"),
                SymbolName("provider_a"),
                SymbolName("provider_b"),
            ]
        )
        op = Operation(
            kind=4,
            name="test.symbol_array_attrs",
            attributes={"providers": values},
        )

        assert op.attributes["providers"] is values
        assert values.values == (
            SymbolName("provider_b"),
            SymbolName("provider_a"),
            SymbolName("provider_b"),
        )
        assert len(SymbolNameArray()) == 0

    def test_symbol_name_array_rejects_untyped_strings(self) -> None:
        with pytest.raises(TypeError, match="symbol name array element 1"):
            SymbolNameArray([SymbolName("typed"), "untyped"])  # type: ignore[list-item]

    def test_symbol_name_set_sorts_and_rejects_duplicates(self) -> None:
        values = SymbolNameSet(
            [
                SymbolName("provider_b"),
                SymbolName("provider_a"),
            ]
        )

        assert values.values == (
            SymbolName("provider_a"),
            SymbolName("provider_b"),
        )
        assert len(SymbolNameSet()) == 0
        with pytest.raises(ValueError, match="duplicate symbol name '@provider_a'"):
            SymbolNameSet(
                [
                    SymbolName("provider_a"),
                    SymbolName("provider_a"),
                ]
            )

    def test_symbol_name_set_rejects_untyped_strings(self) -> None:
        with pytest.raises(TypeError, match="symbol name set element 1"):
            SymbolNameSet([SymbolName("typed"), "untyped"])  # type: ignore[list-item]

    def test_op_with_nested_canonical_attr_dict(self) -> None:
        op = Operation(
            kind=4,
            name="test.attrs",
            attributes={"dict": {"meta": {"phase": "link", "opt": 3}, "axis": 0}},
        )
        assert list(op.attributes.items()) == [
            ("dict", {"axis": 0, "meta": {"opt": 3, "phase": "link"}})
        ]
        dict_attr = op.attributes["dict"]
        assert isinstance(dict_attr, CanonicalAttrDict)
        assert list(dict_attr.items()) == [
            ("axis", 0),
            ("meta", {"opt": 3, "phase": "link"}),
        ]
        assert isinstance(dict_attr["meta"], CanonicalAttrDict)
        assert list(dict_attr["meta"].items()) == [("opt", 3), ("phase", "link")]

    def test_op_attributes_are_immutable(self) -> None:
        op = Operation(kind=4, name="test.attrs", attributes={"axis": 0})
        with pytest.raises(TypeError):
            op.attributes["axis"] = 1  # type: ignore[index]
        with pytest.raises(AttributeError, match="immutable"):
            op.attributes = {"axis": 1}

    def test_op_dead_flag(self) -> None:
        op = Operation(kind=1, name="test.unary")
        assert not op.is_dead
        op.is_dead = True
        assert op.is_dead


# ============================================================================
# Blocks and regions
# ============================================================================


class TestBlocksAndRegions:
    def test_empty_block(self) -> None:
        b = Block()
        assert b.label == ""
        assert len(b.arg_ids) == 0
        assert len(b.ops) == 0

    def test_block_with_args(self) -> None:
        b = Block(label="bb0", arg_ids=[0, 1])
        assert b.label == "bb0"
        assert b.arg_ids == [0, 1]

    def test_region_with_blocks(self) -> None:
        r = Region(blocks=[Block(), Block(label="bb1")])
        assert len(r.blocks) == 2


# ============================================================================
# Functions
# ============================================================================


# ============================================================================
# Symbols
# ============================================================================


class TestSymbols:
    def test_func_symbol(self) -> None:
        func_op = Operation(name="func.def", attributes={"callee": "@foo"})
        sym = Symbol(name="foo", kind=SymbolKind.FUNC_DEF, op=func_op)
        assert sym.kind == SymbolKind.FUNC_DEF

    def test_public_symbol(self) -> None:
        sym = Symbol(name="foo", kind=SymbolKind.FUNC_DEF, flags=SYMBOL_FLAG_PUBLIC)
        assert sym.flags & SYMBOL_FLAG_PUBLIC

    def test_symbol_ref(self) -> None:
        ref = SymbolRef(module_id=0, symbol_id=3)
        assert ref.module_id == 0
        assert ref.symbol_id == 3


# ============================================================================
# String table
# ============================================================================


class TestStringTable:
    def test_intern_new(self) -> None:
        st = StringTable()
        assert st.intern("hello") == 0
        assert st.intern("world") == 1
        assert len(st) == 2

    def test_intern_dedup(self) -> None:
        st = StringTable()
        id1 = st.intern("hello")
        id2 = st.intern("hello")
        assert id1 == id2
        assert len(st) == 1

    def test_get(self) -> None:
        st = StringTable()
        idx = st.intern("test")
        assert st.get(idx) == "test"

    def test_iteration(self) -> None:
        st = StringTable()
        st.intern("a")
        st.intern("b")
        assert list(st) == ["a", "b"]

    def test_empty_string(self) -> None:
        st = StringTable()
        idx = st.intern("")
        assert st.get(idx) == ""


# ============================================================================
# Type table
# ============================================================================


class TestTypeTable:
    def test_intern_new(self) -> None:
        tt = TypeTable()
        assert tt.intern(F32) == 0
        assert tt.intern(I32) == 1
        assert len(tt) == 2

    def test_intern_dedup(self) -> None:
        tt = TypeTable()
        id1 = tt.intern(F32)
        id2 = tt.intern(F32)
        assert id1 == id2
        assert len(tt) == 1

    def test_intern_shaped_dedup(self) -> None:
        tt = TypeTable()
        t1 = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(4)))
        t2 = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(4)))
        id1 = tt.intern(t1)
        id2 = tt.intern(t2)
        assert id1 == id2
        assert len(tt) == 1

    def test_dynamic_dims_are_structural(self) -> None:
        """Two types with DynamicDim at the same position are equal."""
        tt = TypeTable()
        t1 = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        t2 = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        id1 = tt.intern(t1)
        id2 = tt.intern(t2)
        assert id1 == id2

    def test_get(self) -> None:
        tt = TypeTable()
        idx = tt.intern(F32)
        assert tt.get(idx) == F32


# ============================================================================
# Module
# ============================================================================


class TestModule:
    def test_empty_module(self) -> None:
        m = Module(name="test")
        assert m.name == "test"
        assert len(m.values) == 0
        assert len(m.symbols) == 0
        assert len(m.locations) == 1  # Index 0 is unknown.

    def test_add_value(self) -> None:
        m = Module()
        vid = m.add_value(Value(name="x", type=F32))
        assert vid == 0
        assert m.values[vid].name == "x"

    def test_add_location_dedup(self) -> None:
        m = Module()
        loc = FileLocation(0, 1, 1, 1, 10)
        id1 = m.add_location(loc)
        id2 = m.add_location(loc)
        assert id1 == id2
        assert id1 > 0  # 0 is unknown.

    def test_add_location_o1_dedup(self) -> None:
        """Verify location dedup is O(1), not O(n²)."""
        m = Module()
        # Add many unique locations.
        for i in range(1000):
            m.add_location(FileLocation(0, i, 0, i, 80))
        # Dedup should be fast (hash lookup, not linear scan).
        loc = FileLocation(0, 500, 0, 500, 80)
        existing_id = m.add_location(loc)
        assert existing_id == 501  # 0=unknown, 1-1000=the 1000 locs

    def test_add_encoding(self) -> None:
        m = Module()
        enc = EncodingInstance(
            name="q8_0",
            alias="enc",
            params=(("block", 32),),
        )
        idx = m.add_encoding(enc)
        assert idx == 0
        assert m.encodings[idx].alias == "enc"
        assert m.encodings[idx].name == "q8_0"

    def test_add_encoding_deduplicates_alias_independently_of_params(self) -> None:
        m = Module()
        plain = EncodingInstance(name="q8_0", params=(("block", 32),))
        aliased = EncodingInstance(name="q8_0", alias="enc", params=(("block", 32),))
        plain_idx = m.add_encoding(plain)
        aliased_idx = m.add_encoding(aliased)

        assert plain_idx == 0
        assert aliased_idx == 0
        assert m.encodings[0].alias == "enc"
        assert m.encodings[0].params == (("block", 32),)

    def test_add_encoding_rejects_duplicate_alias_for_different_encodings(
        self,
    ) -> None:
        m = Module()
        assert m.add_encoding(EncodingInstance(name="q8_0", alias="enc")) == 0
        with pytest.raises(ValueError, match="already names a different encoding"):
            m.add_encoding(EncodingInstance(name="dense", alias="enc"))

    def test_add_symbol(self) -> None:
        m = Module()
        func_op = Operation(name="func.def", attributes={"callee": "@foo"})
        sym = Symbol(name="foo", kind=SymbolKind.FUNC_DEF, op=func_op)
        sid = m.add_symbol(sym)
        assert m.symbols[sid].name == "foo"
        assert m.body.ops == [func_op]

    def test_add_top_level_operation(self) -> None:
        m = Module()
        metadata_op = Operation(name="test.module_metadata")
        operation_index = m.add_top_level_operation(metadata_op)
        assert operation_index == 0
        assert m.body.ops == [metadata_op]
        assert not m.symbols

    def test_rebuild_value_metadata(self) -> None:
        m = Module()
        arg_id = m.add_value(Value(name="x", type=F32))
        result_id = m.add_value(Value(name="r", type=F32))
        neg_op = Operation(name="test.neg", operands=[arg_id], results=[result_id])
        block = Block(arg_ids=[arg_id], ops=[neg_op])
        func_op = Operation(
            name="func.def",
            attributes={"callee": "f"},
            regions=[Region(blocks=[block])],
        )
        m.add_symbol(Symbol(name="f", kind=SymbolKind.FUNC_DEF, op=func_op))

        # Seed stale metadata so the test verifies a real rebuild.
        m.values[arg_id].flags |= VALUE_FLAG_CONSUMED
        m.values[arg_id].def_block_index = 99
        m.values[arg_id].uses.append(
            Use(user_op_index=42, operand_index=7, block_index=3)
        )
        rebuild_value_metadata(m)

        arg = m.values[arg_id]
        assert arg.is_block_arg
        assert arg.is_consumed
        assert arg.def_op_index == VALUE_DEF_OP_NONE
        assert arg.def_block_index == 0
        assert arg.def_result_index == 0
        assert arg.uses == [Use(user_op_index=0, operand_index=0, block_index=0)]

        result = m.values[result_id]
        assert result.def_op_index == 0
        assert result.def_block_index == 0
        assert result.def_result_index == 0

    def test_string_interning(self) -> None:
        m = Module()
        id1 = m.strings.intern("hello")
        id2 = m.strings.intern("hello")
        assert id1 == id2

    def test_type_interning(self) -> None:
        m = Module()
        id1 = m.types.intern(F32)
        id2 = m.types.intern(F32)
        assert id1 == id2


# ============================================================================
# Encoding instances
# ============================================================================


class TestEncodingInstance:
    def test_basic(self) -> None:
        enc = EncodingInstance(name="q8_0")
        assert enc.name == "q8_0"
        assert enc.alias == ""
        assert enc.params == ()

    def test_with_params(self) -> None:
        enc = EncodingInstance(
            name="q8_0",
            alias="enc",
            params=(("block", 32),),
        )
        assert enc.alias == "enc"
        assert len(enc.params) == 1
        assert enc.params[0] == ("block", 32)

    def test_hashable(self) -> None:
        a = EncodingInstance(name="q8_0", params=(("block", 32),))
        b = EncodingInstance(name="q8_0", alias="enc", params=(("block", 32),))
        assert a == b
        assert hash(a) == hash(b)

    def test_params_are_sorted_canonically(self) -> None:
        enc = EncodingInstance(
            name="q8_0",
            params=(("group_size", 128), ("block", 32)),
        )
        assert enc.params == (("block", 32), ("group_size", 128))

    def test_duplicate_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate encoding parameter"):
            EncodingInstance(
                name="q8_0",
                params=(("block", 32), ("block", 64)),
            )


# ============================================================================
# Location table
# ============================================================================


class TestLocationTable:
    def test_unknown_at_zero(self) -> None:
        lt = LocationTable()
        assert lt.get(0) is None
        assert len(lt) == 1

    def test_add_and_dedup(self) -> None:
        lt = LocationTable()
        loc = FileLocation(0, 10, 1, 10, 50)
        id1 = lt.add(loc)
        id2 = lt.add(loc)
        assert id1 == id2
        assert id1 > 0

    def test_different_locations_different_ids(self) -> None:
        lt = LocationTable()
        a = FileLocation(0, 1, 1, 1, 10)
        b = FileLocation(0, 2, 1, 2, 10)
        id_a = lt.add(a)
        id_b = lt.add(b)
        assert id_a != id_b

    def test_fused_dedup(self) -> None:
        lt = LocationTable()
        a = FusedLocation(children=(1, 2))
        b = FusedLocation(children=(1, 2))
        assert lt.add(a) == lt.add(b)

    def test_tagged_dedup(self) -> None:
        lt = LocationTable()
        a = TaggedLocation(tag=1, child=2, data=b"\x01")
        b = TaggedLocation(tag=1, child=2, data=b"\x01")
        assert lt.add(a) == lt.add(b)

    def test_iteration(self) -> None:
        lt = LocationTable()
        lt.add(FileLocation(0, 1, 1, 1, 10))
        # Iteration yields both the unknown slot and the added location.
        assert len(list(lt)) == 2


# ============================================================================
# Context
# ============================================================================


class TestContext:
    def test_empty_context(self) -> None:
        ctx = Context()
        assert len(ctx.sources) == 0
        assert len(ctx.modules) == 0

    def test_add_source(self) -> None:
        ctx = Context()
        id1 = ctx.add_source("model.loom")
        id2 = ctx.add_source("model.loom")
        assert id1 == id2
        assert ctx.sources[id1] == "model.loom"

    def test_add_source_different(self) -> None:
        ctx = Context()
        id1 = ctx.add_source("a.loom")
        id2 = ctx.add_source("b.loom")
        assert id1 != id2


# ============================================================================
# Predicate evaluation
# ============================================================================


class TestPredicateEvaluation:
    """Tests for evaluate_predicate and evaluate_predicates."""

    def _pred(self, kind: str, *args: tuple[str, int | str]) -> Predicate:
        """Build a predicate from (tag, value) pairs."""
        return Predicate(
            kind=kind,
            args=tuple(PredicateArg(tag=t, value=v) for t, v in args),
        )

    def _vals(self, **kwargs: int | float) -> dict[str, int | float]:
        """Build a values dict with bare name keys."""
        return dict(kwargs)

    def test_eq_true(self) -> None:
        assert evaluate_predicate(
            self._pred("eq", ("value", "M"), ("const", 16)),
            self._vals(M=16),
        )

    def test_eq_false(self) -> None:
        assert not evaluate_predicate(
            self._pred("eq", ("value", "M"), ("const", 16)),
            self._vals(M=17),
        )

    def test_ne(self) -> None:
        assert evaluate_predicate(
            self._pred("ne", ("value", "M"), ("const", 16)),
            self._vals(M=17),
        )
        assert not evaluate_predicate(
            self._pred("ne", ("value", "M"), ("const", 16)),
            self._vals(M=16),
        )

    def test_lt_true(self) -> None:
        assert evaluate_predicate(
            self._pred("lt", ("value", "K"), ("const", 1024)),
            self._vals(K=512),
        )

    def test_lt_false(self) -> None:
        assert not evaluate_predicate(
            self._pred("lt", ("value", "K"), ("const", 1024)),
            self._vals(K=1024),
        )

    def test_le_boundary(self) -> None:
        assert evaluate_predicate(
            self._pred("le", ("value", "K"), ("const", 1024)),
            self._vals(K=1024),
        )

    def test_gt(self) -> None:
        assert evaluate_predicate(
            self._pred("gt", ("value", "M"), ("const", 0)),
            self._vals(M=1),
        )

    def test_ge(self) -> None:
        assert evaluate_predicate(
            self._pred("ge", ("value", "M"), ("const", 16)),
            self._vals(M=16),
        )

    def test_mul_true(self) -> None:
        assert evaluate_predicate(
            self._pred("mul", ("value", "M"), ("const", 16)),
            self._vals(M=64),
        )

    def test_mul_false(self) -> None:
        assert not evaluate_predicate(
            self._pred("mul", ("value", "M"), ("const", 16)),
            self._vals(M=17),
        )

    def test_mul_zero_modulus(self) -> None:
        assert not evaluate_predicate(
            self._pred("mul", ("value", "M"), ("const", 0)),
            self._vals(M=42),
        )

    def test_min(self) -> None:
        assert evaluate_predicate(
            self._pred("min", ("value", "M"), ("const", 32)),
            self._vals(M=32),
        )
        assert not evaluate_predicate(
            self._pred("min", ("value", "M"), ("const", 32)),
            self._vals(M=31),
        )

    def test_max(self) -> None:
        assert evaluate_predicate(
            self._pred("max", ("value", "M"), ("const", 512)),
            self._vals(M=512),
        )
        assert not evaluate_predicate(
            self._pred("max", ("value", "M"), ("const", 512)),
            self._vals(M=513),
        )

    def test_pow2_true(self) -> None:
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
            assert evaluate_predicate(
                self._pred("pow2", ("value", "N")),
                self._vals(N=n),
            ), f"pow2({n}) should be true"

    def test_pow2_false(self) -> None:
        for n in [0, 3, 5, 6, 7, 9, 10, 15, 17, 100]:
            assert not evaluate_predicate(
                self._pred("pow2", ("value", "N")),
                self._vals(N=n),
            ), f"pow2({n}) should be false"

    def test_pow2_rejects_float(self) -> None:
        assert not evaluate_predicate(
            self._pred("pow2", ("value", "N")),
            self._vals(N=8.0),
        )

    def test_range_inside(self) -> None:
        assert evaluate_predicate(
            self._pred("range", ("value", "M"), ("const", 32), ("const", 512)),
            self._vals(M=128),
        )

    def test_range_boundaries(self) -> None:
        pred = self._pred("range", ("value", "M"), ("const", 32), ("const", 512))
        assert evaluate_predicate(pred, self._vals(M=32))
        assert evaluate_predicate(pred, self._vals(M=512))

    def test_range_outside(self) -> None:
        pred = self._pred("range", ("value", "M"), ("const", 32), ("const", 512))
        assert not evaluate_predicate(pred, self._vals(M=31))
        assert not evaluate_predicate(pred, self._vals(M=513))

    def test_not_nan(self) -> None:
        assert evaluate_predicate(
            self._pred("not_nan", ("value", "X")),
            self._vals(X=math.inf),
        )
        assert not evaluate_predicate(
            self._pred("not_nan", ("value", "X")),
            self._vals(X=math.nan),
        )

    def test_not_inf(self) -> None:
        assert evaluate_predicate(
            self._pred("not_inf", ("value", "X")),
            self._vals(X=math.nan),
        )
        assert not evaluate_predicate(
            self._pred("not_inf", ("value", "X")),
            self._vals(X=math.inf),
        )

    def test_finite(self) -> None:
        assert evaluate_predicate(
            self._pred("finite", ("value", "X")),
            self._vals(X=1.0),
        )
        assert not evaluate_predicate(
            self._pred("finite", ("value", "X")),
            self._vals(X=math.inf),
        )
        assert not evaluate_predicate(
            self._pred("finite", ("value", "X")),
            self._vals(X=math.nan),
        )

    def test_missing_value_always_true(self) -> None:
        """Missing values can't be evaluated — defer judgment."""
        assert evaluate_predicate(
            self._pred("mul", ("value", "%UNKNOWN"), ("const", 16)),
            self._vals(M=42),
        )

    def test_evaluate_predicates_all_true(self) -> None:
        preds = [
            self._pred("mul", ("value", "M"), ("const", 16)),
            self._pred("lt", ("value", "K"), ("const", 1024)),
        ]
        assert evaluate_predicates(preds, self._vals(M=64, K=512))

    def test_evaluate_predicates_one_false(self) -> None:
        preds = [
            self._pred("mul", ("value", "M"), ("const", 16)),
            self._pred("lt", ("value", "K"), ("const", 1024)),
        ]
        assert not evaluate_predicates(preds, self._vals(M=64, K=2048))

    def test_evaluate_predicates_empty(self) -> None:
        assert evaluate_predicates([], self._vals(M=42))

    def test_eq_two_values(self) -> None:
        """eq(%M, %K) — compare two SSA values."""
        assert evaluate_predicate(
            self._pred("eq", ("value", "M"), ("value", "K")),
            self._vals(M=64, K=64),
        )
        assert not evaluate_predicate(
            self._pred("eq", ("value", "M"), ("value", "K")),
            self._vals(M=64, K=32),
        )
