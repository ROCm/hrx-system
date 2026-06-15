# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom.dsl — op declaration DSL."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import loom.ir as ir
from loom.assembly import (
    COLON,
    EQUALS,
    Attr,
    BlockRef,
    Clause,
    IndexList,
    Keyword,
    Ref,
    Region,
    ResultType,
    Scope,
    TypeOf,
    kw,
)
from loom.dsl import (
    ADDRESS,
    ANY,
    ANY_ENCODING,
    BUFFER,
    BY_REFERENCE,
    COMMUTATIVE,
    CONSTANT_LIKE,
    CONVERGENT,
    DECOMPOSABLE,
    ELEMENTWISE,
    ENCODING_LAYOUT,
    ENCODING_SCHEMA,
    ENCODING_STORAGE,
    ENCODING_TRANSFORM,
    FLOAT,
    FLOAT_ELEMENT,
    HINT,
    I1,
    I1_ELEMENT,
    IDEMPOTENT,
    INDEX,
    INTEGER,
    INTEGER_ELEMENT,
    INVOLUTION,
    NON_DETERMINISTIC,
    OFFSET,
    POOL,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    REGISTER,
    SAFE_TO_SPECULATE,
    STORAGE,
    TENSOR,
    TERMINATOR,
    TILE,
    UNIQUE_IDENTITY,
    UNKNOWN_EFFECTS,
    AliasResult,
    AllShapesMatch,
    AllTypesMatch,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    BlockArgCount,
    BlockArgsMatchElementTypes,
    BlockArgsMatchTypes,
    BlockArgsSatisfy,
    Borrow,
    CallLikeInterface,
    CallLikeKind,
    Consume,
    ContractFamily,
    Dialect,
    DimIndexInBounds,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    ElementWidthLessThan,
    EnumCase,
    EnumDef,
    FreshResult,
    HasAllStaticRankOneVector,
    HasAllStaticVector,
    HasAncestor,
    HasF16OrBf16Element,
    HasF32Element,
    HasFloatElement,
    HasI1Element,
    HasI8Element,
    HasI32Element,
    HasIndexOrNonI1IntegerElement,
    HasIndexOrNonI1IntegerScalar,
    HasIntegerElement,
    HasParent,
    HasRankOneVector,
    HasRegister,
    ImplicitTerminator,
    LastAxisGroupedBy,
    LegacyFieldDefault,
    LegacyFieldMapping,
    LegacyFormat,
    LiteralMatchesElementType,
    MemoryAccessInterface,
    NoAncestor,
    OffsetCountMatchesRank,
    Op,
    OpCategory,
    Operand,
    OperandOwnershipEffect,
    OpPhase,
    PackedPayloadBitCountMatchesStorage,
    PositiveBitWidthAttr,
    RanksMatch,
    Reads,
    RegionDef,
    Result,
    ResultOwnershipEffect,
    SameElementType,
    SameEncoding,
    SameKind,
    SameShape,
    SameType,
    Successor,
    TargetLikeInterface,
    TotalBitCountEqual,
    TypeConstraint,
    TypeDef,
    TypeSemantic,
    UnpackedPayloadBitCountMatchesStorage,
    ValueCountMatchesStaticElementCount,
    Writes,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
    binary_op,
    cast_op,
    comparison_op,
    type_constraint_name,
    unary_op,
)
from loom.ir import (
    BF16,
    F16,
    F32,
    I8,
    I16,
    I32,
    DynamicDim,
    RegisterType,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
)


@contextmanager
def _raises(
    exception_type: type[BaseException], match: str | None = None
) -> Iterator[None]:
    try:
        yield
    except exception_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"exception message {str(exc)!r} did not match {match!r}"
            ) from exc
        return
    raise AssertionError(f"{exception_type.__name__} was not raised")


# ============================================================================
# Test fixtures
# ============================================================================

_scalar_ops = Dialect("scalar", doc="Scalar ops.")
_tile_ops = Dialect("tile", doc="Tile ops.")
_func_ops = Dialect("func", doc="Function ops.")
_scf_ops = Dialect("scf", doc="Structured control flow.")

_cmpi_preds = EnumDef(
    "CmpIPredicate",
    [
        EnumCase("eq", 0),
        EnumCase("ne", 1),
        EnumCase("slt", 2),
        EnumCase("sle", 3),
        EnumCase("sgt", 4),
        EnumCase("sge", 5),
    ],
)


# ============================================================================
# Type constraints
# ============================================================================


class TestTypeConstraints:
    def test_all_constraints_have_names(self) -> None:
        for tc in TypeConstraint:
            name = type_constraint_name(tc)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_singleton_equality(self) -> None:
        assert TILE == TypeConstraint.TILE
        assert INTEGER == TypeConstraint.INTEGER
        assert OFFSET == TypeConstraint.OFFSET
        assert ADDRESS == TypeConstraint.ADDRESS
        assert BUFFER == TypeConstraint.BUFFER
        assert STORAGE == TypeConstraint.STORAGE
        assert ANY_ENCODING == TypeConstraint.ANY_ENCODING
        assert ENCODING_LAYOUT == TypeConstraint.ENCODING_LAYOUT
        assert INTEGER_ELEMENT == TypeConstraint.INTEGER_ELEMENT
        assert TILE != TENSOR

    def test_values(self) -> None:
        assert TILE.value == "tile"
        assert FLOAT.value == "float"
        assert ANY_ENCODING.value == "encoding"
        assert ENCODING_LAYOUT.value == "encoding<layout>"
        assert ENCODING_SCHEMA.value == "encoding<schema>"
        assert ENCODING_STORAGE.value == "encoding<storage>"
        assert ENCODING_TRANSFORM.value == "encoding<transform>"
        assert FLOAT_ELEMENT.value == "float_element"
        assert I1_ELEMENT.value == "i1_element"
        assert INDEX.value == "index"
        assert OFFSET.value == "offset"
        assert ADDRESS.value == "address"
        assert STORAGE.value == "storage"

    def test_element_family_constraints_are_shaped_specific(self) -> None:
        element_family_constraints = {
            constraint
            for constraint in TypeConstraint
            if constraint.value.startswith(("integer_", "float_", "i1_"))
        }
        assert element_family_constraints == {
            TypeConstraint.INTEGER_ELEMENT,
            TypeConstraint.FLOAT_ELEMENT,
            TypeConstraint.I1_ELEMENT,
        }


# ============================================================================
# Operands and results
# ============================================================================


class TestOperand:
    def test_basic(self) -> None:
        o = Operand("lhs", INTEGER)
        assert o.name == "lhs"
        assert o.type_constraint == INTEGER
        assert not o.variadic
        assert not o.optional

    def test_variadic(self) -> None:
        o = Operand("inputs", TILE, variadic=True)
        assert o.variadic

    def test_optional(self) -> None:
        o = Operand("acc", TILE, optional=True)
        assert o.optional

    def test_with_doc(self) -> None:
        o = Operand("lhs", INTEGER, doc="Left operand.")
        assert o.doc == "Left operand."


class TestResult:
    def test_basic(self) -> None:
        r = Result("result", FLOAT)
        assert r.name == "result"
        assert r.type_constraint == FLOAT
        assert not r.variadic

    def test_variadic(self) -> None:
        r = Result("results", ANY, variadic=True)
        assert r.variadic


# ============================================================================
# Attributes and enums
# ============================================================================


class TestAttrDef:
    def test_basic(self) -> None:
        a = AttrDef("axis", "i64")
        assert a.name == "axis"
        assert a.attr_type == "i64"
        assert a.default is None
        assert not a.optional

    def test_with_default(self) -> None:
        a = AttrDef("transpose", "bool", default="false")
        assert a.default == "false"

    def test_elide_default(self) -> None:
        a = AttrDef("offset", "i64", default=0, elide_default=True)
        assert a.default == 0
        assert a.elide_default

    def test_elide_default_requires_default(self) -> None:
        with _raises(ValueError, match="elide_default requires default"):
            AttrDef("offset", "i64", elide_default=True)

    def test_enum_attr(self) -> None:
        a = AttrDef("predicate", "enum", enum_def=_cmpi_preds)
        assert a.enum_def is not None
        assert a.enum_def.name == "CmpIPredicate"

    def test_invalid_attr_type_rejected(self) -> None:
        with _raises(ValueError, match="invalid attr_type"):
            AttrDef("axis", "i64array")  # Invalid attr_type.

    def test_enum_without_enum_def_rejected(self) -> None:
        with _raises(ValueError, match="requires enum_def"):
            AttrDef("pred", "enum")  # Missing enum_def!

    def test_all_valid_attr_types(self) -> None:
        """All documented attr_type values are accepted."""
        for attr_type in ["i64", "f64", "string", "bool", "type", "i64_array", "any"]:
            AttrDef("test", attr_type)  # Should not raise.
        AttrDef("test", "enum", enum_def=_cmpi_preds)  # enum needs enum_def.


class TestEnumDef:
    def test_basic(self) -> None:
        e = _cmpi_preds
        assert e.name == "CmpIPredicate"
        assert len(e.cases) == 6

    def test_keywords(self) -> None:
        assert _cmpi_preds.keywords == ("eq", "ne", "slt", "sle", "sgt", "sge")

    def test_accepts_list(self) -> None:
        e = EnumDef("Test", [EnumCase("a", 0)])
        assert isinstance(e.cases, tuple)

    def test_duplicate_keyword_rejected(self) -> None:
        with _raises(ValueError, match="duplicate keyword 'eq'"):
            EnumDef("Bad", [EnumCase("eq", 0), EnumCase("eq", 1)])

    def test_duplicate_value_rejected(self) -> None:
        with _raises(ValueError, match="duplicate value 0"):
            EnumDef("Bad", [EnumCase("a", 0), EnumCase("b", 0)])

    def test_external_c_enum_alias_metadata(self) -> None:
        e = EnumDef(
            "Mode",
            [EnumCase("fast", 0)],
            c_type="loom_shared_mode_t",
            c_const_prefix="LOOM_SHARED_MODE",
            c_include="loom/shared/mode.h",
        )

        assert e.c_type == "loom_shared_mode_t"
        assert e.c_const_prefix == "LOOM_SHARED_MODE"
        assert e.c_include == "loom/shared/mode.h"

    def test_external_c_enum_alias_requires_type_and_prefix(self) -> None:
        with _raises(
            ValueError,
            match="c_type and c_const_prefix must be provided together",
        ):
            EnumDef(
                "Mode",
                [EnumCase("fast", 0)],
                c_type="loom_shared_mode_t",
            )

        with _raises(ValueError, match="c_include requires c_type"):
            EnumDef(
                "Mode",
                [EnumCase("fast", 0)],
                c_include="loom/shared/mode.h",
            )


class TestEnumCase:
    def test_basic(self) -> None:
        c = EnumCase("slt", 2, doc="Signed less than.")
        assert c.keyword == "slt"
        assert c.value == 2
        assert c.doc == "Signed less than."


# ============================================================================
# Regions
# ============================================================================


class TestRegionDef:
    def test_basic(self) -> None:
        r = RegionDef("body")
        assert r.name == "body"
        assert not r.single_block

    def test_single_block(self) -> None:
        r = RegionDef("body", single_block=True)
        assert r.single_block

    def test_arg_source(self) -> None:
        r = RegionDef("body", arg_source="inputs")
        assert r.arg_source == "inputs"


# ============================================================================
# Traits
# ============================================================================


class TestTraits:
    def test_simple_trait(self) -> None:
        assert PURE.name == "Pure"
        assert PURE.args == ()

    def test_trait_repr(self) -> None:
        assert repr(PURE) == "Pure"
        assert repr(COMMUTATIVE) == "Commutative"

    def test_parameterized_trait(self) -> None:
        t = AllTypesMatch("lhs", "rhs", "result")
        assert t.name == "AllTypesMatch"
        assert t.args == ("lhs", "rhs", "result")
        assert repr(t) == "AllTypesMatch(lhs, rhs, result)"

    def test_has_parent(self) -> None:
        t = HasParent("scf.for")
        assert t.args == ("scf.for",)

    def test_ancestor_placement_traits(self) -> None:
        required = HasAncestor("low.func.def")
        forbidden = NoAncestor("low.func.def")
        assert required.name == "HasAncestor"
        assert required.args == ("low.func.def",)
        assert forbidden.name == "NoAncestor"
        assert forbidden.args == ("low.func.def",)

    def test_implicit_terminator(self) -> None:
        t = ImplicitTerminator("scf.yield")
        assert t.args == ("scf.yield",)

    def test_all_standard_traits(self) -> None:
        standard = [
            PURE,
            COMMUTATIVE,
            IDEMPOTENT,
            INVOLUTION,
            TERMINATOR,
            CONSTANT_LIKE,
            ELEMENTWISE,
            DECOMPOSABLE,
            CONVERGENT,
            SAFE_TO_SPECULATE,
            REFINABLE_RESULT_TYPE_REFS,
        ]
        names = [t.name for t in standard]
        assert len(set(names)) == len(names), "Duplicate trait names"


# ============================================================================
# Interfaces
# ============================================================================


class TestInterfaces:
    def test_call_like_interface_defaults_to_semantic(self) -> None:
        interface = CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
        )
        assert interface.callee == "callee"
        assert interface.operands == "operands"
        assert interface.results == "results"
        assert interface.purity is None
        assert interface.kind == CallLikeKind.SEMANTIC

    def test_call_like_kind_values(self) -> None:
        assert CallLikeKind.SEMANTIC.value == "semantic"
        assert CallLikeKind.LOW_INTERNAL.value == "low_internal"
        assert CallLikeKind.LOW_INVOKE.value == "low_invoke"

    def test_target_like_interface_defaults_to_no_extensions(self) -> None:
        interface = TargetLikeInterface(symbol="symbol", selector="kind")

        assert interface.symbol == "symbol"
        assert interface.selector == "kind"
        assert interface.extensions is None
        assert interface.descriptor is None

    def test_memory_access_interface_uses_soft_field_defaults(self) -> None:
        interface = MemoryAccessInterface()

        assert interface.view == "view"
        assert interface.value == "value"
        assert interface.indices == "indices"
        assert interface.static_indices == "static_indices"
        assert interface.cache_scope == "cache_scope"
        assert interface.cache_temporal == "cache_temporal"
        assert interface.atomic_kind == "kind"
        assert interface.atomic_ordering == "ordering"
        assert interface.atomic_success_ordering == "success_ordering"
        assert interface.atomic_failure_ordering == "failure_ordering"
        assert interface.atomic_scope == "scope"
        assert interface._explicit_fields == frozenset()

    def test_memory_access_interface_tracks_explicit_overrides(self) -> None:
        interface = MemoryAccessInterface(value="stored", cache_scope=None)

        assert interface.value == "stored"
        assert interface.cache_scope is None
        assert interface._explicit_fields == frozenset({"value", "cache_scope"})


# ============================================================================
# Constraints
# ============================================================================


class TestConstraints:
    def test_same_type_repr(self) -> None:
        c = SameType("lhs", "rhs")
        assert repr(c) == "SameType(lhs, rhs)"
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_001"

    def test_same_type_validation_pass(self) -> None:
        c = SameType("a", "b")

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        ok, msg = c.check({"a": FakeValue("f32"), "b": FakeValue("f32")})
        assert ok
        assert msg == ""

    def test_same_type_validation_fail(self) -> None:
        c = SameType("a", "b")

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        ok, msg = c.check({"a": FakeValue("f32"), "b": FakeValue("i32")})
        assert not ok
        assert "'b'" in msg
        assert "'a'" in msg

    def test_same_type_missing_values(self) -> None:
        c = SameType("a", "b")
        ok, _msg = c.check({"a": None})
        assert ok, "Missing values should pass (can't check)"

    def test_same_kind(self) -> None:
        c = SameKind("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_001"

    def test_same_element_type(self) -> None:
        c = SameElementType("x", "y")
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_002"

    def test_same_encoding(self) -> None:
        c = SameEncoding("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_ENCODING_001"
        # No validate function — C-only check.
        ok, _msg = c.check({})
        assert ok

    def test_same_shape(self) -> None:
        c = SameShape("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_002"

    def test_ranks_match(self) -> None:
        c = RanksMatch("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_001"

    def test_element_family_constraints(self) -> None:
        assert HasIntegerElement("x").name == "HasIntegerElement"
        assert HasFloatElement("x").name == "HasFloatElement"
        assert HasIndexOrNonI1IntegerScalar("x").name == "HasIndexOrNonI1IntegerScalar"
        assert (
            HasIndexOrNonI1IntegerElement("x").name == "HasIndexOrNonI1IntegerElement"
        )
        assert HasI1Element("x").name == "HasI1Element"
        assert HasI8Element("x").name == "HasI8Element"
        assert HasI32Element("x").name == "HasI32Element"
        assert HasF16OrBf16Element("x").name == "HasF16OrBf16Element"
        assert HasF32Element("x").name == "HasF32Element"
        assert HasRegister("x").name == "HasRegister"
        assert HasRankOneVector("x").name == "HasRankOneVector"
        assert HasAllStaticVector("x").name == "HasAllStaticVector"
        assert HasAllStaticRankOneVector("x").name == "HasAllStaticRankOneVector"

    def test_register_constraint_validates_register_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        assert HasRegister("x").check({"x": FakeValue(RegisterType(1, 0))})[0]
        assert not HasRegister("x").check({"x": FakeValue(I32)})[0]

    def test_exact_element_constraints_validate_shaped_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        assert HasI8Element("x").check({"x": FakeValue(vector_type(I8))})[0]
        assert not HasI8Element("x").check({"x": FakeValue(vector_type(I16))})[0]
        assert HasI32Element("x").check({"x": FakeValue(vector_type(I32))})[0]
        assert not HasI32Element("x").check({"x": FakeValue(vector_type(I8))})[0]
        assert HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(F16))})[0]
        assert HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(BF16))})[0]
        assert not HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(F32))})[0]
        assert HasF32Element("x").check({"x": FakeValue(vector_type(F32))})[0]
        assert not HasF32Element("x").check({"x": FakeValue(I32)})[0]

    def test_index_or_non_i1_integer_constraints_validate_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        assert HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.INDEX)})[0]
        assert HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(I32)})[0]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.I1)})[0]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.OFFSET)})[
            0
        ]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(F32)})[0]
        assert HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.INDEX))}
        )[0]
        assert HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(I8))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.I1))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.OFFSET))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(F32))}
        )[0]

    def test_vector_shape_constraints_validate_vector_shape(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        vector_1d = ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))
        vector_2d = ShapedType(TypeKind.VECTOR, F32, (StaticDim(2), StaticDim(2)))
        vector_dynamic = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(),))
        tile_1d = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))

        assert HasRankOneVector("x").check({"x": FakeValue(vector_1d)})[0]
        assert not HasRankOneVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasRankOneVector("x").check({"x": FakeValue(tile_1d)})[0]
        assert HasAllStaticVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasAllStaticVector("x").check({"x": FakeValue(vector_dynamic)})[0]
        assert HasAllStaticRankOneVector("x").check({"x": FakeValue(vector_1d)})[0]
        assert not HasAllStaticRankOneVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasAllStaticRankOneVector("x").check(
            {"x": FakeValue(vector_dynamic)}
        )[0]

    def test_element_width_order_constraints(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        greater = ElementWidthGreaterThan("result", "input")
        assert greater.name == "ElementWidthGreaterThan"
        assert greater.args == ("result", "input")
        assert greater.check(
            {
                "input": FakeValue(vector_type(F16)),
                "result": FakeValue(vector_type(F32)),
            }
        )[0]
        ok, message = greater.check(
            {
                "input": FakeValue(vector_type(F32)),
                "result": FakeValue(vector_type(F16)),
            }
        )
        assert not ok
        assert "result" in message
        assert "input" in message

        less = ElementWidthLessThan("result", "input")
        assert less.name == "ElementWidthLessThan"
        assert less.args == ("result", "input")
        assert less.check(
            {
                "input": FakeValue(vector_type(I32)),
                "result": FakeValue(vector_type(I8)),
            }
        )[0]
        assert not less.check(
            {
                "input": FakeValue(vector_type(I8)),
                "result": FakeValue(vector_type(I8)),
            }
        )[0]

    def test_element_width_at_least_attr(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = ElementWidthAtLeastAttr("result", "width")
        assert constraint.name == "ElementWidthAtLeastAttr"
        assert constraint.args == ("result", "width")
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 4,
            }
        )[0]
        assert not constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 16,
            }
        )[0]
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 0,
            }
        )[0]

    def test_bit_range_within_element_width(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = BitRangeWithinElementWidth("source", "offset", "width")
        assert constraint.name == "BitRangeWithinElementWidth"
        assert constraint.args == ("source", "offset", "width")
        source = FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),)))
        assert constraint.check({"source": source, "offset": 2, "width": 4})[0]
        assert not constraint.check({"source": source, "offset": -1, "width": 4})[0]
        assert not constraint.check({"source": source, "offset": 2, "width": 0})[0]
        assert not constraint.check({"source": source, "offset": 6, "width": 4})[0]

    def test_positive_bit_width_attr(self) -> None:
        constraint = PositiveBitWidthAttr("width")
        assert constraint.name == "PositiveBitWidthAttr"
        assert constraint.args == ("width",)
        assert constraint.check({"width": 1})[0]
        assert not constraint.check({"width": 0})[0]
        assert not constraint.check({"width": -1})[0]

    def test_attr_matches_element_type(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = AttrMatchesElementType("value", "result")
        assert constraint.name == "AttrMatchesElementType"
        assert constraint.args == ("value", "result")
        assert constraint.check(
            {
                "value": 0.0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check(
            {
                "value": 7,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 128,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": -129,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": True,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check(
            {
                "value": True,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert constraint.check(
            {
                "value": 1,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 2,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert constraint.check({"value": 4, "result": FakeValue(ir.INDEX)})[0]
        assert constraint.check({"value": 4, "result": FakeValue(ir.OFFSET)})[0]
        assert not constraint.check({"value": True, "result": FakeValue(ir.INDEX)})[0]
        assert not constraint.check({"value": -1, "result": FakeValue(ir.OFFSET)})[0]

    def test_literal_matches_element_type(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = LiteralMatchesElementType("value", "result")
        assert constraint.name == "LiteralMatchesElementType"
        assert constraint.args == ("value", "result")
        assert constraint.check(
            {
                "value": 1.0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 1,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check({"value": 127, "result": FakeValue(I8)})[0]
        assert not constraint.check({"value": 128, "result": FakeValue(I8)})[0]
        assert constraint.check({"value": 0, "result": FakeValue(ir.OFFSET)})[0]
        assert not constraint.check({"value": -1, "result": FakeValue(ir.OFFSET)})[0]

    def test_total_bit_count_equal(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = TotalBitCountEqual("input", "result")
        assert constraint.name == "TotalBitCountEqual"
        assert constraint.args == ("input", "result")
        assert constraint.check(
            {
                "input": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I32, (StaticDim(1),))),
            }
        )[0]
        assert constraint.check(
            {
                "input": FakeValue(I32),
                "result": FakeValue(F32),
            }
        )[0]
        assert not constraint.check(
            {
                "input": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(3),))),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I16, (StaticDim(1),))),
            }
        )[0]
        assert not constraint.check(
            {
                "input": FakeValue(I16),
                "result": FakeValue(F32),
            }
        )[0]
        assert constraint.check(
            {
                "input": FakeValue(
                    ShapedType(TypeKind.VECTOR, I8, (DynamicDim(), StaticDim(2)))
                ),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I16, (DynamicDim(),))),
            }
        )[0]

    def test_payload_bit_count_matches_storage_constraints(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType, *sizes: int | None) -> ShapedType:
            return ShapedType(
                TypeKind.VECTOR,
                element_type,
                tuple(
                    DynamicDim() if size is None else StaticDim(size) for size in sizes
                ),
            )

        pack = PackedPayloadBitCountMatchesStorage(
            "source", "width", "result", "result"
        )
        assert pack.name == "PackedPayloadBitCountMatchesStorage"
        assert pack.args == ("source", "width", "result", "result")
        assert pack.check(
            {
                "source": FakeValue(vector_type(I8, 32)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 16)),
            }
        )[0]
        assert not pack.check(
            {
                "source": FakeValue(vector_type(I8, 32)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 15)),
            }
        )[0]
        assert pack.check(
            {
                "source": FakeValue(vector_type(I8, None)),
                "width": 4,
                "result": FakeValue(vector_type(I8, None)),
            }
        )[0]

        unpack = UnpackedPayloadBitCountMatchesStorage(
            "result", "width", "source", "result"
        )
        assert unpack.name == "UnpackedPayloadBitCountMatchesStorage"
        assert unpack.args == ("result", "width", "source", "result")
        assert unpack.check(
            {
                "source": FakeValue(vector_type(I8, 16)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 32)),
            }
        )[0]
        assert not unpack.check(
            {
                "source": FakeValue(vector_type(I8, 16)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 31)),
            }
        )[0]

    def test_offset_count_matches_rank(self) -> None:
        c = OffsetCountMatchesRank("src", "offsets")
        assert c.error is not None
        assert c.error.error_id == "ERR_SUBRANGE_001"

    def test_value_count_matches_static_element_count(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = ValueCountMatchesStaticElementCount("result", "elements")
        assert constraint.error is not None
        assert constraint.error.error_id == "ERR_STRUCTURE_013"
        assert constraint.check(
            {
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, F32, (StaticDim(2), StaticDim(2)))
                ),
                "elements": [object(), object(), object(), object()],
            }
        )[0]
        assert not constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
                "elements": [object(), object(), object()],
            }
        )[0]
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (DynamicDim(),))),
                "elements": [object()],
            }
        )[0]

    def test_dim_index_in_bounds(self) -> None:
        c = DimIndexInBounds("src", "dim")
        assert c.error is not None
        assert c.error.error_id == "ERR_SUBRANGE_002"

    def test_all_shapes_match(self) -> None:
        c = AllShapesMatch("inputs")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_003"

    def test_last_axis_grouped_by(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(*sizes: int) -> ShapedType:
            return ShapedType(
                TypeKind.VECTOR, F32, tuple(StaticDim(size) for size in sizes)
            )

        constraint = LastAxisGroupedBy("lhs", "result", 4)

        assert constraint.name == "LastAxisGroupedBy"
        assert constraint.args == ("lhs", "result")
        assert constraint.data == 4
        assert constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(2, 4)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 10)),
                "result": FakeValue(vector_type(2, 2)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(3, 4)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(2, 5)),
            }
        )[0]

    def test_region_constraints(self) -> None:
        block_arg_count = BlockArgCount("body", "inputs")
        assert block_arg_count.error is not None
        assert block_arg_count.error.error_id == "ERR_STRUCTURE_007"

        block_args_satisfy = BlockArgsSatisfy("body", REGISTER)
        assert block_args_satisfy.error is not None
        assert block_args_satisfy.error.error_id == "ERR_TYPE_014"
        assert block_args_satisfy.data == REGISTER

        block_args_match = BlockArgsMatchElementTypes("body", "inputs")
        assert block_args_match.error is not None
        assert block_args_match.error.error_id == "ERR_TYPE_008"

        block_args_type_match = BlockArgsMatchTypes("body", "inputs")
        assert block_args_type_match.error is not None
        assert block_args_type_match.error.error_id == "ERR_TYPE_013"

        yield_count = YieldCountMatchesResults("body", "results")
        assert yield_count.error is not None
        assert yield_count.error.error_id == "ERR_STRUCTURE_008"

        yield_types = YieldTypesMatchResults("body", "results")
        assert yield_types.error is not None
        assert yield_types.error.error_id == "ERR_TYPE_009"

    # --- Variadic field handling ---

    def test_same_type_variadic_pass(self) -> None:
        """SameType across a variadic list and a scalar field."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32"), FakeValue("f32"), FakeValue("f32")],
                "result": FakeValue("f32"),
            }
        )
        assert ok
        assert msg == ""

    def test_same_type_variadic_mismatch_within_list(self) -> None:
        """SameType detects mismatch within a variadic list."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32"), FakeValue("i32")],
                "result": FakeValue("f32"),
            }
        )
        assert not ok
        assert "inputs[1]" in msg
        assert "inputs[0]" in msg

    def test_same_type_variadic_mismatch_against_scalar(self) -> None:
        """SameType detects mismatch between variadic element and scalar."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32")],
                "result": FakeValue("i32"),
            }
        )
        assert not ok
        assert "result" in msg
        assert "inputs[0]" in msg

    def test_same_type_empty_variadic(self) -> None:
        """SameType with empty variadic list passes (nothing to compare)."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, _msg = c.check({"inputs": [], "result": FakeValue("f32")})
        assert ok, "Single value with empty list should pass"

    def test_same_element_type_variadic_pass(self) -> None:
        """SameElementType across variadic and scalar."""

        class FakeTile:
            def __init__(self, dt: str):
                self.dtype = dt

        c = SameElementType("inputs", "result")
        ok, _msg = c.check(
            {
                "inputs": [FakeTile("f32"), FakeTile("f32")],
                "result": FakeTile("f32"),
            }
        )
        assert ok

    def test_same_element_type_variadic_fail(self) -> None:
        """SameElementType detects mismatch in variadic list."""

        class FakeTile:
            def __init__(self, dt: str):
                self.dtype = dt

        c = SameElementType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeTile("f32"), FakeTile("i32")],
                "result": FakeTile("f32"),
            }
        )
        assert not ok
        assert "inputs[1]" in msg

    def test_same_shape_variadic_pass(self) -> None:
        """SameShape across variadic and scalar."""

        class FakeTile:
            def __init__(self, s: tuple[int, ...]):
                self.shape = s

        c = SameShape("inputs", "result")
        ok, _msg = c.check(
            {
                "inputs": [FakeTile((4, 8)), FakeTile((4, 8))],
                "result": FakeTile((4, 8)),
            }
        )
        assert ok

    def test_same_shape_variadic_fail(self) -> None:
        """SameShape detects mismatch in variadic list."""

        class FakeTile:
            def __init__(self, s: tuple[int, ...]):
                self.shape = s

        c = SameShape("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeTile((4, 8)), FakeTile((4, 16))],
                "result": FakeTile((4, 8)),
            }
        )
        assert not ok
        assert "inputs[1]" in msg
        assert "inputs[0]" in msg

    def test_ranks_match_variadic_pass(self) -> None:
        """RanksMatch with variadic field."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, _msg = c.check(
            {
                "a": [FakeTile(2), FakeTile(2)],
                "b": FakeTile(2),
            }
        )
        assert ok

    def test_ranks_match_variadic_fail(self) -> None:
        """RanksMatch detects rank mismatch with variadic field."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, msg = c.check(
            {
                "a": [FakeTile(2), FakeTile(3)],
                "b": FakeTile(2),
            }
        )
        assert not ok
        assert "a[1]" in msg

    def test_ranks_match_both_variadic(self) -> None:
        """RanksMatch with both fields variadic."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, msg = c.check(
            {
                "a": [FakeTile(2)],
                "b": [FakeTile(3)],
            }
        )
        assert not ok
        assert "a[0]" in msg
        assert "b[0]" in msg


# ============================================================================
# Op group
# ============================================================================


class TestDialect:
    def test_basic(self) -> None:
        g = Dialect("scalar", doc="Scalar ops.")
        assert g.name == "scalar"
        assert g.doc == "Scalar ops."

    def test_with_enums(self) -> None:
        g = Dialect("scalar", enums=[_cmpi_preds])
        assert len(g.enums) == 1
        assert isinstance(g.enums, tuple)

    def test_default_phase(self) -> None:
        g = Dialect("target", default_phase=OpPhase.MODULE_METADATA)
        assert g.default_phase == OpPhase.MODULE_METADATA

    def test_categories(self) -> None:
        structural = OpCategory("structure", doc="Structural ops.")
        compute = OpCategory("compute", doc="Compute ops.")

        g = Dialect(
            "vector",
            categories=[structural, compute],
            default_category=compute,
        )

        assert g.categories == (structural, compute)
        assert g.default_category == compute

    def test_default_category_must_be_declared(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"Dialect 'vector': default_category 'compute' "
                r"is not declared in categories"
            ),
        ):
            Dialect(
                "vector",
                categories=[OpCategory("structure")],
                default_category=OpCategory("compute"),
            )


class TestOpCategory:
    def test_basic(self) -> None:
        category = OpCategory("memory.atomic", doc="Atomic memory ops.")

        assert category.key == "memory.atomic"
        assert category.doc == "Atomic memory ops."

    def test_rejects_empty_key(self) -> None:
        with _raises(ValueError, match="op category key must not be empty"):
            OpCategory("")

    def test_rejects_unstable_key_spelling(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"op category key 'Memory/Atomic' must contain only lowercase "
                r"letters, digits, '.', '_', or '-'"
            ),
        ):
            OpCategory("Memory/Atomic")


# ============================================================================
# Type declaration
# ============================================================================


class TestTypeDef:
    def test_semantic_metadata_defaults_to_ordinary(self) -> None:
        type_def = TypeDef("test.handle")

        assert type_def.semantic == TypeSemantic.ORDINARY
        assert type_def.contracts == ()

    def test_semantic_metadata_is_stored_as_tuples(self) -> None:
        type_def = TypeDef(
            "kernel.async.token",
            semantic=TypeSemantic.CONTROL_TOKEN,
            contracts=[ContractFamily.KERNEL_ASYNC],
        )

        assert type_def.semantic == TypeSemantic.CONTROL_TOKEN
        assert type_def.contracts == (ContractFamily.KERNEL_ASYNC,)


# ============================================================================
# Op declaration
# ============================================================================


class TestOp:
    def test_basic(self) -> None:
        op = Op(
            "scalar.addi",
            group=_scalar_ops,
            doc="Integer addition.",
            operands=[Operand("lhs", INTEGER), Operand("rhs", INTEGER)],
            results=[Result("result", INTEGER)],
        )
        assert op.name == "scalar.addi"
        assert op.namespace == "scalar"
        assert op.short_name == "addi"
        assert len(op.operands) == 2
        assert len(op.results) == 1

    def test_semantic_metadata(self) -> None:
        op = Op(
            "vector.iota",
            phase=OpPhase.EXECUTABLE,
            contracts=[ContractFamily.VECTOR_COORDINATE],
        )

        assert op.phase == OpPhase.EXECUTABLE
        assert op.contracts == (ContractFamily.VECTOR_COORDINATE,)

    def test_effective_phase_uses_dialect_default(self) -> None:
        op = Op(
            "target.generic",
            group=Dialect("target", default_phase=OpPhase.MODULE_METADATA),
        )

        assert op.effective_phase == OpPhase.MODULE_METADATA

    def test_category_metadata(self) -> None:
        coordinate = OpCategory("coordinate")
        memory = OpCategory("memory")
        dialect = Dialect("vector", categories=[coordinate, memory])
        op = Op("vector.iota", group=dialect, category=coordinate)

        assert op.category == coordinate
        assert op.effective_category == coordinate

    def test_effective_category_uses_dialect_default(self) -> None:
        compute = OpCategory("compute")
        op = Op(
            "vector.addf",
            group=Dialect(
                "vector",
                categories=[compute],
                default_category=compute,
            ),
        )

        assert op.effective_category == compute

    def test_category_must_be_declared_by_dialect(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"Op 'vector\.load': category 'memory' is not declared "
                r"by dialect 'vector'"
            ),
        ):
            Op(
                "vector.load",
                group=Dialect("vector", categories=[OpCategory("compute")]),
                category=OpCategory("memory"),
            )

    def test_repr(self) -> None:
        op = Op("test.op")
        assert repr(op) == "Op('test.op')"

    def test_tuples_stored(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("x", ANY)],
            results=[Result("y", ANY)],
            successors=[Successor("dest")],
            traits=[PURE],
            constraints=[SameType("x", "y")],
            format=[Ref("x"), BlockRef("dest"), COLON, TypeOf("y")],
            legacy_formats=[
                LegacyFormat(
                    "test.op.old",
                    format=[Ref("x"), COLON, TypeOf("y")],
                    replaced_by="loom-source-format-2026-06-09",
                    expires_after="loom-source-format-2026-07-01",
                )
            ],
            examples=["example"],
        )
        assert isinstance(op.operands, tuple)
        assert isinstance(op.results, tuple)
        assert isinstance(op.successors, tuple)
        assert isinstance(op.traits, tuple)
        assert isinstance(op.constraints, tuple)
        assert isinstance(op.format, tuple)
        assert isinstance(op.legacy_formats, tuple)
        assert isinstance(op.examples, tuple)

    def test_legacy_format_metadata(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("input", ANY)],
            results=[Result("result", ANY)],
            attrs=[AttrDef("mode", "string"), AttrDef("policy", "string")],
            format=[Ref("input"), COLON, TypeOf("result")],
            legacy_formats=[
                LegacyFormat(
                    "test.op.mode-attr",
                    format=[
                        Ref("legacy_input"),
                        Attr("legacy_mode"),
                        COLON,
                        TypeOf("result"),
                    ],
                    field_mappings=[
                        LegacyFieldMapping("legacy_input", "input"),
                        LegacyFieldMapping("legacy_mode", "mode"),
                    ],
                    field_defaults=[LegacyFieldDefault("policy", "default")],
                    replaced_by="loom-source-format-2026-06-09",
                    expires_after="loom-source-format-2026-07-01",
                    rewrite_hook="migrate_test_op_mode_attr",
                )
            ],
        )

        legacy_format = op.legacy_formats[0]
        assert legacy_format.rule_id == "test.op.mode-attr"
        assert legacy_format.format[0] == Ref("legacy_input")
        assert legacy_format.field_mappings == (
            LegacyFieldMapping("legacy_input", "input"),
            LegacyFieldMapping("legacy_mode", "mode"),
        )
        assert legacy_format.field_defaults == (
            LegacyFieldDefault("policy", "default"),
        )
        assert legacy_format.expires_after == "loom-source-format-2026-07-01"
        assert legacy_format.rewrite_hook == "migrate_test_op_mode_attr"

    def test_legacy_format_expiration_is_optional(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("input", ANY)],
            legacy_formats=[
                LegacyFormat(
                    "test.op.old",
                    format=[Ref("input")],
                    replaced_by="loom-source-format-2026-06-09",
                )
            ],
        )

        legacy_format = op.legacy_formats[0]
        assert legacy_format.replaced_by == "loom-source-format-2026-06-09"
        assert legacy_format.expires_after == ""

    def test_duplicate_legacy_format_rule_id_is_rejected(self) -> None:
        with _raises(ValueError, match="duplicate legacy format rule_id"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.old",
                        format=[Ref("input")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    ),
                    LegacyFormat(
                        "test.op.old",
                        format=[Ref("input")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    ),
                ],
            )

    def test_legacy_format_field_validation_catches_typo(self) -> None:
        with _raises(ValueError, match="undeclared current fields"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("missing")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_unused_field_mapping_is_rejected(self) -> None:
        with _raises(ValueError, match="not referenced"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_mappings=[
                            LegacyFieldMapping("old_input", "input"),
                        ],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_duplicate_current_mapping_is_rejected(self) -> None:
        with _raises(ValueError, match="more than one legacy field"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("old_input"), Ref("older_input")],
                        field_mappings=[
                            LegacyFieldMapping("old_input", "input"),
                            LegacyFieldMapping("older_input", "input"),
                        ],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_default_field_must_exist(self) -> None:
        with _raises(ValueError, match="defaults undeclared field"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_defaults=[LegacyFieldDefault("missing", "value")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_default_cannot_replace_parsed_field(self) -> None:
        with _raises(ValueError, match="already parsed"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_defaults=[LegacyFieldDefault("input", "value")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_lookup_operand(self) -> None:
        op = Op(
            "test.op",
            operands=[
                Operand("lhs", INTEGER),
                Operand("rhs", INTEGER),
            ],
        )
        lhs_operand = op.operand("lhs")
        assert lhs_operand is not None
        assert lhs_operand.name == "lhs"
        assert op.operand("missing") is None

    def test_lookup_result(self) -> None:
        op = Op("test.op", results=[Result("out", FLOAT)])
        assert op.result("out") is not None
        assert op.result("missing") is None

    def test_lookup_attr(self) -> None:
        op = Op("test.op", attrs=[AttrDef("axis", "i64")])
        assert op.attr("axis") is not None
        assert op.attr("missing") is None

    def test_lookup_successor(self) -> None:
        op = Op("test.op", successors=[Successor("dest")])
        assert op.successor("dest") is not None
        assert op.successor("missing") is None

    def test_lookup_region(self) -> None:
        op = Op("test.op", regions=[RegionDef("body")])
        assert op.region("body") is not None
        assert op.region("missing") is None

    def test_trait_queries(self) -> None:
        op = Op("test.op", traits=[PURE, COMMUTATIVE, TERMINATOR])
        assert op.is_pure
        assert op.is_commutative
        assert op.is_terminator
        assert op.has_trait("Pure")
        assert not op.has_trait("Idempotent")

    def test_no_traits(self) -> None:
        op = Op("test.op")
        # An op with no effects, no traits, and no allocating results
        # is derived pure — it does nothing observable.
        assert op.is_pure
        assert not op.is_terminator

    def test_namespace_no_dot(self) -> None:
        op = Op("nodot")
        assert op.namespace == ""
        assert op.short_name == "nodot"

    def test_keyword_only_args(self) -> None:
        """Op constructor requires keyword arguments after name."""
        # This should work:
        Op("test.op", doc="hello")
        # Positional args after name should fail:
        with _raises(TypeError):
            Op("test.op", "hello")  # type: ignore[arg-type, misc]

    def test_format_field_validation_catches_typo(self) -> None:
        """Format referencing undeclared field is caught at declaration time."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("input", ANY)],
                results=[Result("result", ANY)],
                format=[Ref("source"), COLON, TypeOf("result")],  # Wrong name.
            )

    def test_format_field_validation_accepts_successor(self) -> None:
        """BlockRef fields are validated against declared successors."""
        op = Op(
            "test.br",
            successors=[Successor("dest")],
            format=[BlockRef("dest")],
        )
        assert op.successor("dest") is not None

    def test_format_field_validation_descends_into_clause(self) -> None:
        """Clause payload fields are validated like top-level format fields."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("input", ANY)],
                format=[Clause("value", Ref("missing"))],
            )

    def test_format_field_validation_allows_implicit(self) -> None:
        """Implicit fields (iv, args, predicates) are allowed."""
        # Should not raise — "iv" and "args" are implicit.
        Op(
            "test.loop",
            operands=[Operand("lower_bound", INDEX)],
            results=[Result("results", ANY, variadic=True)],
            regions=[RegionDef("body")],
            format=[Ref("iv"), EQUALS, Ref("lower_bound"), Region("body")],
        )

    def test_format_field_validation_index_list(self) -> None:
        """IndexList fields (both dynamic and static) are validated."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("source", TILE)],
                results=[Result("result", TILE)],
                format=[
                    Ref("source"),
                    IndexList("offsets", "static_offsets"),  # Neither declared!
                ],
            )

    def test_region_arg_source_must_be_value_field(self) -> None:
        """Region arg_source must name a variadic value or FuncArgs field."""
        with _raises(ValueError, match="arg_source references non-value/non-FuncArgs"):
            Op(
                "test.bad",
                attrs=[AttrDef("types", "string")],
                regions=[RegionDef("body", arg_source="types")],
                format=[Region("body")],
            )

    def test_region_arg_source_validates_without_format(self) -> None:
        """Region arg_source is an op contract, not an assembly-format detail."""
        with _raises(ValueError, match="arg_source references non-value/non-FuncArgs"):
            Op(
                "test.bad",
                regions=[RegionDef("body", arg_source="missing")],
            )

    def test_region_arg_source_must_be_variadic(self) -> None:
        """Region arg_source maps one region arg per source value."""
        with _raises(ValueError, match="must reference a variadic"):
            Op(
                "test.bad",
                operands=[Operand("input", INTEGER)],
                regions=[RegionDef("body", arg_source="input")],
                format=[Ref("input"), Region("body")],
            )

    def test_nested_scope_rejected(self) -> None:
        """Scope(...) is one-level only; nested Scope is a declaration error."""
        with _raises(ValueError, match="nested Scope is not supported"):
            Op(
                "test.bad",
                results=[Result("result", ANY)],
                format=[Scope([Scope([ResultType("result")])])],
            )


# ============================================================================
# Memory effects
# ============================================================================


class TestEffects:
    def test_valid_read_effect(self) -> None:
        op = Op(
            "test.load",
            operands=[Operand("source", POOL)],
            results=[Result("result", ANY)],
            effects=[Reads("source")],
        )
        assert not op.is_pure
        assert op.effects[0].operand == "source"

    def test_valid_write_effect(self) -> None:
        op = Op(
            "test.store",
            operands=[Operand("target", POOL), Operand("data", TILE)],
            effects=[Writes("target")],
        )
        assert not op.is_pure

    def test_pure_with_effects_raises(self) -> None:
        with _raises(ValueError, match="PURE.*effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[PURE],
                effects=[Reads("pool")],
            )

    def test_effect_on_nonexistent_operand_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                operands=[Operand("input", POOL)],
                effects=[Reads("nonexistent")],
            )

    def test_effect_on_non_resource_operand_raises(self) -> None:
        with _raises(ValueError, match="not allowed"):
            Op(
                "test.bad",
                operands=[Operand("value", INTEGER)],
                effects=[Reads("value")],
            )

    def test_unknown_effects_with_pure_raises(self) -> None:
        with _raises(ValueError, match="PURE.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[PURE, UNKNOWN_EFFECTS])

    def test_unknown_effects_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="UNKNOWN_EFFECTS.*explicit"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[UNKNOWN_EFFECTS],
                effects=[Reads("pool")],
            )

    def test_pure_with_non_deterministic_raises(self) -> None:
        with _raises(ValueError, match="PURE.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[PURE, NON_DETERMINISTIC])

    def test_non_deterministic_not_pure(self) -> None:
        op = Op("test.rng", traits=[NON_DETERMINISTIC])
        assert not op.is_pure

    def test_unknown_effects_not_pure(self) -> None:
        op = Op("test.call", traits=[UNKNOWN_EFFECTS])
        assert not op.is_pure

    def test_allocating_result(self) -> None:
        op = Op(
            "test.alloc",
            results=[Result("pool", POOL, allocates=True)],
        )
        assert not op.is_pure
        assert op.results[0].allocates

    def test_unique_identity_not_pure(self) -> None:
        op = Op("test.handle", traits=[UNIQUE_IDENTITY])
        assert not op.is_pure

    def test_pure_with_unique_identity_raises(self) -> None:
        with _raises(ValueError, match="PURE.*UNIQUE_IDENTITY"):
            Op("test.bad", traits=[PURE, UNIQUE_IDENTITY])

    def test_hint_not_pure(self) -> None:
        op = Op("test.hint", traits=[HINT])
        assert not op.is_pure

    def test_hint_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="HINT.*explicit effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[HINT],
                effects=[Reads("pool")],
            )

    def test_hint_with_pure_raises(self) -> None:
        with _raises(ValueError, match="HINT.*PURE"):
            Op("test.bad", traits=[HINT, PURE])

    def test_hint_with_unknown_effects_raises(self) -> None:
        with _raises(ValueError, match="HINT.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[HINT, UNKNOWN_EFFECTS])

    def test_hint_with_non_deterministic_raises(self) -> None:
        with _raises(ValueError, match="HINT.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[HINT, NON_DETERMINISTIC])

    def test_hint_with_convergent_raises(self) -> None:
        with _raises(ValueError, match="HINT.*CONVERGENT"):
            Op("test.bad", traits=[HINT, CONVERGENT])

    def test_convergent_can_be_pure(self) -> None:
        op = Op("test.convergent", traits=[PURE, CONVERGENT])
        assert op.is_pure

    def test_safe_to_speculate_conflicts_with_hint(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*HINT"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, HINT])


# ============================================================================
# Ownership effects
# ============================================================================


class TestOwnershipEffects:
    def test_valid_operand_and_result_ownership_effects(self) -> None:
        op = Op(
            "test.resource.retain",
            operands=[Operand("resource", POOL)],
            results=[Result("result", POOL)],
            ownership_effects=[
                Borrow("resource", BY_REFERENCE),
                AliasResult("result", "resource"),
            ],
        )
        assert not op.is_pure
        operand_effect = op.ownership_effects[0]
        result_effect = op.ownership_effects[1]
        assert isinstance(operand_effect, OperandOwnershipEffect)
        assert isinstance(result_effect, ResultOwnershipEffect)
        assert operand_effect.operand == "resource"
        assert operand_effect.carrier == BY_REFERENCE
        assert result_effect.source == "resource"

    def test_fresh_result_ownership_effect(self) -> None:
        op = Op(
            "test.resource.alloc",
            results=[Result("result", POOL, allocates=True)],
            ownership_effects=[FreshResult("result")],
        )
        assert not op.is_pure
        result_effect = op.ownership_effects[0]
        assert isinstance(result_effect, ResultOwnershipEffect)
        assert result_effect.result == "result"

    def test_alias_result_ownership_effect_requires_fixed_fields(self) -> None:
        with _raises(ValueError, match="fixed operand/result"):
            Op(
                "test.resource.alias_many",
                operands=[Operand("resources", POOL, variadic=True)],
                results=[Result("results", POOL, variadic=True)],
                ownership_effects=[AliasResult("results", "resources")],
            )

    def test_operand_ownership_effect_on_missing_operand_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                operands=[Operand("resource", POOL)],
                ownership_effects=[Consume("missing")],
            )

    def test_pure_with_ownership_effects_raises(self) -> None:
        with _raises(ValueError, match="PURE.*ownership"):
            Op(
                "test.bad",
                operands=[Operand("resource", POOL)],
                traits=[PURE],
                ownership_effects=[Consume("resource")],
            )

    def test_result_ownership_effect_on_missing_result_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                results=[Result("result", POOL)],
                ownership_effects=[FreshResult("missing")],
            )

    def test_alias_result_requires_source_operand(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                results=[Result("result", POOL)],
                ownership_effects=[AliasResult("result", "missing")],
            )

    def test_safe_to_speculate_conflicts_with_unknown_effects(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, UNKNOWN_EFFECTS])

    def test_safe_to_speculate_conflicts_with_non_deterministic(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, NON_DETERMINISTIC])

    def test_safe_to_speculate_conflicts_with_unique_identity(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*UNIQUE_IDENTITY"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, UNIQUE_IDENTITY])

    def test_safe_to_speculate_conflicts_with_convergent(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*CONVERGENT"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, CONVERGENT])

    def test_safe_to_speculate_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*explicit effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[SAFE_TO_SPECULATE],
                effects=[Reads("pool")],
            )


# ============================================================================
# Helper functions
# ============================================================================


class TestBinaryOp:
    def test_basic(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Integer addition.",
        )
        assert op.name == "scalar.addi"
        assert len(op.operands) == 2
        assert op.operands[0].name == "lhs"
        assert op.operands[1].name == "rhs"
        assert len(op.results) == 1
        assert op.is_pure
        assert not op.is_commutative

    def test_commutative(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Integer addition.",
            commutative=True,
        )
        assert op.is_commutative

    def test_format(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Add.",
        )
        assert len(op.format) == 5
        assert isinstance(op.format[0], Ref)
        assert isinstance(op.format[1], Keyword)
        assert isinstance(op.format[2], Ref)
        assert isinstance(op.format[3], Keyword)
        assert isinstance(op.format[4], TypeOf)


class TestUnaryOp:
    def test_basic(self) -> None:
        op = unary_op(
            "scalar.negf",
            group=_scalar_ops,
            type_constraint=FLOAT,
            doc="Negate.",
        )
        assert len(op.operands) == 1
        assert op.operands[0].name == "input"
        assert op.is_pure

    def test_format(self) -> None:
        op = unary_op(
            "scalar.negf",
            group=_scalar_ops,
            type_constraint=FLOAT,
            doc="Negate.",
        )
        assert len(op.format) == 3
        assert isinstance(op.format[0], Ref)
        assert isinstance(op.format[1], Keyword)
        assert isinstance(op.format[2], TypeOf)


class TestCastOp:
    def test_basic(self) -> None:
        op = cast_op(
            "scalar.sitofp",
            group=_scalar_ops,
            from_constraint=INTEGER,
            to_constraint=FLOAT,
            doc="Signed int to float.",
        )
        assert op.operands[0].type_constraint == INTEGER
        assert op.results[0].type_constraint == FLOAT
        assert op.is_pure

    def test_format(self) -> None:
        op = cast_op(
            "scalar.sitofp",
            group=_scalar_ops,
            from_constraint=INTEGER,
            to_constraint=FLOAT,
            doc="Cast.",
        )
        # Expected format: Ref(input) COLON TypeOf(input) kw(to) TypeOf(result)
        assert len(op.format) == 5
        assert op.format[3] == kw("to")


class TestComparisonOp:
    def test_basic(self) -> None:
        op = comparison_op(
            "scalar.cmpi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            predicates=_cmpi_preds,
            doc="Integer comparison.",
        )
        assert len(op.operands) == 2
        assert len(op.attrs) == 1
        assert op.attrs[0].enum_def is _cmpi_preds
        assert op.results[0].type_constraint == I1

    def test_format(self) -> None:
        op = comparison_op(
            "scalar.cmpi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            predicates=_cmpi_preds,
            doc="Compare.",
        )
        # Expected format: Attr(predicate) COMMA Ref(lhs) COMMA Ref(rhs)
        # COLON TypeOf(lhs)
        assert len(op.format) == 7
        assert isinstance(op.format[0], Attr)
