# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom.assembly — assembly format elements."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    FOR,
    LBRACE,
    LBRACKET,
    LPAREN,
    RBRACE,
    RBRACKET,
    RPAREN,
    TO,
    Attr,
    AttrDict,
    AttrTable,
    BindingList,
    BlockArgs,
    Clause,
    FuncArgs,
    IndexList,
    Keyword,
    OperandDict,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    RegionTable,
    ResultTypeList,
    SymbolRef,
    TypeOf,
    TypesOf,
    kw,
)


class TestRef:
    def test_construct(self) -> None:
        r = Ref("lhs")
        assert r.field == "lhs"

    def test_equality(self) -> None:
        assert Ref("lhs") == Ref("lhs")
        assert Ref("lhs") != Ref("rhs")

    def test_hashable(self) -> None:
        s = {Ref("a"), Ref("a"), Ref("b")}
        assert len(s) == 2


class TestRefs:
    def test_construct(self) -> None:
        r = Refs("operands")
        assert r.field == "operands"

    def test_distinct_from_ref(self) -> None:
        assert Ref("x") != Refs("x")  # type: ignore[comparison-overlap]


class TestAttr:
    def test_construct(self) -> None:
        a = Attr("value")
        assert a.field == "value"


class TestSymbolRef:
    def test_construct(self) -> None:
        s = SymbolRef("callee")
        assert s.field == "callee"


class TestTypeOf:
    def test_construct(self) -> None:
        t = TypeOf("result")
        assert t.field == "result"


class TestTypesOf:
    def test_construct(self) -> None:
        t = TypesOf("operands")
        assert t.field == "operands"

    def test_distinct_from_typeof(self) -> None:
        assert TypeOf("x") != TypesOf("x")  # type: ignore[comparison-overlap]


class TestResultTypeList:
    def test_construct(self) -> None:
        r = ResultTypeList("results")
        assert r.field == "results"


class TestKeyword:
    def test_construct(self) -> None:
        k = Keyword("->")
        assert k.text == "->"

    def test_repr(self) -> None:
        assert repr(Keyword("->")) == "kw('->')"

    def test_kw_shorthand(self) -> None:
        assert kw("step") == Keyword("step")

    def test_common_keywords(self) -> None:
        assert COMMA.text == ","
        assert COLON.text == ":"
        assert ARROW.text == "->"
        assert LPAREN.text == "("
        assert RPAREN.text == ")"
        assert LBRACKET.text == "["
        assert RBRACKET.text == "]"
        assert LBRACE.text == "{"
        assert RBRACE.text == "}"
        assert EQUALS.text == "="
        assert TO.text == "to"
        assert FOR.text == "for"

    def test_equality(self) -> None:
        assert kw("->") == ARROW
        assert kw(",") == COMMA


class TestAttrDict:
    def test_construct(self) -> None:
        a = AttrDict()
        assert isinstance(a, AttrDict)

    def test_singleton_equality(self) -> None:
        assert AttrDict() == AttrDict()


class TestClause:
    def test_construct(self) -> None:
        clause = Clause("range", Attr("minimum"), TO, Attr("maximum"))
        assert clause.name == "range"
        assert clause.elements == (Attr("minimum"), TO, Attr("maximum"))


class TestOperandDict:
    def test_construct(self) -> None:
        operand_dict = OperandDict("params", "param_names")
        assert operand_dict.operands == "params"
        assert operand_dict.names == "param_names"


class TestAttrTable:
    def test_construct(self) -> None:
        attr_table = AttrTable("case_keys", "values")
        assert attr_table.keys == "case_keys"
        assert attr_table.values == "values"


class TestRegionTable:
    def test_construct(self) -> None:
        region_table = RegionTable("case_keys", "case_regions", "default_region")
        assert region_table.keys == "case_keys"
        assert region_table.case_regions == "case_regions"
        assert region_table.default_region == "default_region"


class TestRegion:
    def test_construct(self) -> None:
        r = Region("body")
        assert r.field == "body"
        assert r.syntax == ""

    def test_construct_with_syntax(self) -> None:
        r = Region("body", syntax="test.do")
        assert r.field == "body"
        assert r.syntax == "test.do"


class TestIndexList:
    def test_construct(self) -> None:
        il = IndexList("offsets", "static_offsets")
        assert il.dynamic == "offsets"
        assert il.static == "static_offsets"
        assert il.glue is True

    def test_construct_without_leading_glue(self) -> None:
        il = IndexList("dims", "static_dims", glue=False)
        assert il.dynamic == "dims"
        assert il.static == "static_dims"
        assert il.glue is False


class TestBindingList:
    def test_construct(self) -> None:
        bl = BindingList("inputs")
        assert bl.field == "inputs"


class TestBlockArgs:
    def test_construct(self) -> None:
        args = BlockArgs("body")
        assert args.region == "body"


class TestFuncArgs:
    def test_construct(self) -> None:
        fa = FuncArgs("args")
        assert fa.field == "args"


class TestPredicateList:
    def test_construct(self) -> None:
        pl = PredicateList("predicates")
        assert pl.field == "predicates"


class TestOptionalGroup:
    def test_construct_from_list(self) -> None:
        opt = OptionalGroup([kw("else"), Region("else_region")], anchor="else_region")
        assert opt.anchor == "else_region"
        assert isinstance(opt.elements, tuple)
        assert len(opt.elements) == 2

    def test_construct_from_tuple(self) -> None:
        opt = OptionalGroup((COMMA, COLON), anchor="x")
        assert isinstance(opt.elements, tuple)

    def test_nested_optional(self) -> None:
        inner = OptionalGroup([kw("step"), Ref("step")], anchor="step")
        outer = OptionalGroup([inner, Region("body")], anchor="body")
        assert len(outer.elements) == 2
        assert isinstance(outer.elements[0], OptionalGroup)


class TestFormatSpecs:
    """Test that real-world format specs are constructible and readable."""

    def test_binary_scalar(self) -> None:
        # %r = scalar.addi %a, %b : f32
        fmt = [Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")]
        assert len(fmt) == 5
        assert isinstance(fmt[0], Ref)
        assert isinstance(fmt[1], Keyword)

    def test_cast(self) -> None:
        # %r = scalar.sitofp %x : i32 to f32
        fmt = [Ref("input"), COLON, TypeOf("input"), kw("to"), TypeOf("result")]
        assert len(fmt) == 5

    def test_constant(self) -> None:
        # %c = index.constant 42 : index
        fmt = [Attr("value"), COLON, TypeOf("result")]
        assert len(fmt) == 3

    def test_slice_with_indices(self) -> None:
        # %t = tile.slice %src[0, %off] : tile<...> -> tile<...>
        fmt = [
            Ref("source"),
            IndexList("offsets", "static_offsets"),
            COLON,
            TypeOf("source"),
            ARROW,
            TypeOf("result"),
        ]
        assert len(fmt) == 6

    def test_elementwise_with_binding(self) -> None:
        # %r = tile.elementwise(%a = %x : type) { ... } -> (type)
        fmt = [BindingList("inputs"), Region("body"), ARROW, ResultTypeList("result")]
        assert len(fmt) == 4

    def test_func_call(self) -> None:
        # %r = func.call @f(%a, %b) : (t0, t1) -> (t0, t1)
        fmt = [
            SymbolRef("callee"),
            LPAREN,
            Refs("operands"),
            RPAREN,
            COLON,
            LPAREN,
            TypesOf("operands"),
            RPAREN,
            ARROW,
            ResultTypeList("results"),
        ]
        assert len(fmt) == 10

    def test_scf_for(self) -> None:
        # %r = scf.for %i = %lb to %ub step %s
        #     iter_args(%out = %init : type) -> (%init as type) { ... }
        fmt = [
            Ref("iv"),
            EQUALS,
            Ref("lower_bound"),
            kw("to"),
            Ref("upper_bound"),
            kw("step"),
            Ref("step"),
            OptionalGroup(
                [kw("iter_args"), BindingList("iter_args")], anchor="iter_args"
            ),
            ARROW,
            ResultTypeList("results"),
            Region("body"),
        ]
        assert len(fmt) == 11

    def test_scf_if(self) -> None:
        # %r = scf.if %cond -> (type) { ... } else { ... }
        fmt = [
            Ref("condition"),
            ARROW,
            ResultTypeList("results"),
            Region("then_region"),
            OptionalGroup([kw("else"), Region("else_region")], anchor="else_region"),
        ]
        assert len(fmt) == 5

    def test_func_def(self) -> None:
        # func.def public host @name(%a: type) -> (type)
        #     where [...] { ... }
        fmt = [
            OptionalGroup([Attr("visibility")], anchor="visibility"),
            OptionalGroup([Attr("cc")], anchor="cc"),
            SymbolRef("name"),
            FuncArgs("args"),
            OptionalGroup([ARROW, ResultTypeList("results")], anchor="results"),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")], anchor="predicates"
            ),
            OptionalGroup([Region("body")], anchor="body"),
        ]
        assert len(fmt) == 7

    def test_dispatch_region(self) -> None:
        # %out = dispatch.region(%w = %weights : type)
        #     -> (%weights as type) { ... }
        fmt = [
            BindingList("captures"),
            ARROW,
            ResultTypeList("results"),
            Region("body"),
        ]
        assert len(fmt) == 4

    def test_yield(self) -> None:
        # scf.yield %a, %b : type, type
        fmt = [Refs("values"), COLON, TypesOf("values")]
        assert len(fmt) == 3
