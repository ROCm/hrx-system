# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Template dialect operation definitions.

Template families are compile-time callable contracts. A family declaration
owns the stable signature, definitions provide independently named candidate
implementations, apply selects a candidate from the active library universe,
and call names one exact implementation.
"""

from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Refs,
    Region,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TypesOf,
    kw,
)
from loom.dialect.func.defs import CallingConv, Purity, Retain, Temperature, Visibility
from loom.dsl import (
    ANY,
    ATTR_TYPE_PARAMETERIZED_ARRAY,
    COMMAND_EFFECT,
    ISOLATED_FROM_ABOVE,
    POISON_BOUNDARY,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    CallLikeInterface,
    CallLikeKind,
    Dialect,
    FuncLikeInterface,
    Op,
    Operand,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolReference,
)

# ============================================================================
# Op group and shared format fragments
# ============================================================================

template_ops = Dialect(
    "template",
    dialect_id=0x20,
    doc="Compile-time callable families and implementations.",
)

_RETAIN_ATTR = AttrDef("retain", "enum", enum_def=Retain, optional=True)

_MODIFIER_ATTRS = [
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_MODIFIER_FORMAT = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("retain")], anchor="retain"),
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
    OptionalGroup([Attr("temperature")], anchor="temperature"),
]

_APPLICABILITY_ATTRS = [
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef(
        "requires",
        ATTR_TYPE_PARAMETERIZED_ARRAY,
        optional=True,
        doc=("Typed proof requirements over application-site target facts. Requirements filter eligibility and never add facts."),
    ),
]

_TARGET_FORMAT = [
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_REQUIRES_FORMAT = [
    OptionalGroup(
        [kw("requires"), Attr("requires")],
        anchor="requires",
    ),
]


def _signature_format(symbol_field: str) -> list[Any]:
    return [
        SymbolRef(symbol_field),
        Scope(
            [
                FuncArgs("args"),
                OptionalGroup(
                    [ARROW, ResultTypeList("results")],
                    anchor="results",
                ),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
    ]


_FUNC_LIKE_COMMON: dict[str, Any] = dict(
    visibility="visibility",
    cc="cc",
    purity="purity",
    temperature="temperature",
    predicates="predicates",
    target="target",
    requires="requires",
)

# ============================================================================
# template.decl - abstract callable-family declaration
# ============================================================================

template_decl = Op(
    "template.decl",
    group=template_ops,
    doc=("Abstract compile-time callable-family declaration. The declaration owns the stable signature and coarse applicability contract."),
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=[
        AttrDef("family", "symbol"),
        *_MODIFIER_ATTRS,
        *_APPLICABILITY_ATTRS,
        _RETAIN_ATTR,
    ],
    symbol_def=SymbolDefinition(
        field="family",
        name="template family",
        interfaces=["func_like", "template_family"],
        bytecode_kind="LOOM_SYMBOL_TEMPLATE_DECL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
        flags=[SymbolDefinitionFlag.DECLARATION],
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[
        FuncLikeInterface(
            callee="family",
            **_FUNC_LIKE_COMMON,
            args="args",
        )
    ],
    verify="loom_template_decl_verify",
    format=[
        *_MODIFIER_FORMAT,
        *_TARGET_FORMAT,
        *_REQUIRES_FORMAT,
        *_signature_format("family"),
    ],
    examples=[
        "template.decl public @vector_transform(%value: vector<32xf32>) -> (vector<32xf32>)",
        "template.decl requires [#target.subgroup.size<32>] @wave32_transform(%value: vector<32xf32>) -> (vector<32xf32>)",
    ],
)

# ============================================================================
# template.def - bodyful candidate implementation
# ============================================================================

_PROVIDER_ATTRS = [
    AttrDef(
        "family",
        "symbol",
        symbol_ref=SymbolReference("template family", ["template_family"]),
    ),
    AttrDef("implementation", "symbol"),
    *_MODIFIER_ATTRS,
    *_APPLICABILITY_ATTRS,
    AttrDef("priority", "i64", optional=True),
    _RETAIN_ATTR,
]

_PROVIDER_FORMAT = [
    TemplateParam("family"),
    *_MODIFIER_FORMAT,
    *_TARGET_FORMAT,
    *_REQUIRES_FORMAT,
    OptionalGroup(
        [kw("priority"), GLUE, LPAREN, GLUE, Attr("priority"), GLUE, RPAREN],
        anchor="priority",
    ),
]

template_def = Op(
    "template.def",
    group=template_ops,
    doc="Constraint-matched bodyful implementation of a template family.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_PROVIDER_ATTRS),
    symbol_def=SymbolDefinition(
        field="implementation",
        name="template implementation",
        interfaces=["func_like", "template_provider"],
        bytecode_kind="LOOM_SYMBOL_TEMPLATE_DEF",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Template implementation body.",
            terminator="template.return",
        )
    ],
    interfaces=[
        FuncLikeInterface(
            callee="implementation",
            **_FUNC_LIKE_COMMON,
            template_family="family",
            priority="priority",
            body="body",
        )
    ],
    verify="loom_template_def_verify",
    format=[
        *_PROVIDER_FORMAT,
        *_signature_format("implementation"),
        Region("body"),
    ],
    examples=[
        "template.def<@vector_transform> device priority(20) @vector_transform_fast(%value: vector<32xf32>) -> (vector<32xf32>) {\n  template.return %value : vector<32xf32>\n}",
        "template.def<@wave32_transform> requires [#target.subgroup.size<32>] @wave32_transform_impl(%value: vector<32xf32>) -> (vector<32xf32>) {\n  template.return %value : vector<32xf32>\n}",
    ],
)

# ============================================================================
# template.ukernel - opaque candidate implementation
# ============================================================================

template_ukernel = Op(
    "template.ukernel",
    group=template_ops,
    doc="Constraint-matched opaque implementation of a template family.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_PROVIDER_ATTRS),
    symbol_def=SymbolDefinition(
        field="implementation",
        name="template implementation",
        interfaces=["func_like", "template_provider"],
        bytecode_kind="LOOM_SYMBOL_TEMPLATE_UKERNEL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[
        FuncLikeInterface(
            callee="implementation",
            **_FUNC_LIKE_COMMON,
            template_family="family",
            priority="priority",
            args="args",
        )
    ],
    verify="loom_template_ukernel_verify",
    format=[
        *_PROVIDER_FORMAT,
        *_signature_format("implementation"),
    ],
    examples=[
        "template.ukernel<@vector_transform> device priority(10) @vector_transform_asm(%value: vector<32xf32>) -> (vector<32xf32>)",
    ],
)

# ============================================================================
# template.apply - fact-driven family application
# ============================================================================

template_apply = Op(
    "template.apply",
    group=template_ops,
    doc=(
        "Compile-time family application. A specializing link selects one "
        "applicable implementation before executable lowering. In a command "
        "program this is source composition whose selected implementation is "
        "inlined before portable command preparation."
    ),
    operands=[Operand("operands", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "family",
            "symbol",
            symbol_ref=SymbolReference("template family", ["template_family"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
        AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS, COMMAND_EFFECT],
    verify="loom_template_apply_verify",
    canonicalize="loom_template_apply_canonicalize",
    effective_traits="loom_template_apply_effective_traits",
    format=[
        TemplateParam("family"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%result = template.apply<@vector_transform>(%value) : (vector<32xf32>) -> (vector<32xf32>)",
    ],
)

# ============================================================================
# template.call - exact implementation application
# ============================================================================

template_call = Op(
    "template.call",
    group=template_ops,
    doc=(
        "Exact compile-time implementation call. This bypasses candidate "
        "ranking but still checks the implementation contract. In a command "
        "program the implementation is inlined before portable command "
        "preparation."
    ),
    operands=[Operand("operands", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("template implementation", ["template_provider"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
        AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS, COMMAND_EFFECT],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            purity="purity",
            temperature="temperature",
            kind=CallLikeKind.TEMPLATE,
        )
    ],
    verify="loom_template_call_verify",
    canonicalize="loom_template_call_canonicalize",
    effective_traits="loom_template_call_effective_traits",
    format=[
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%result = template.call @vector_transform_fast(%value) : (vector<32xf32>) -> (vector<32xf32>)",
    ],
)

# ============================================================================
# template.return - template body terminator
# ============================================================================

template_return = Op(
    "template.return",
    group=template_ops,
    doc="Return values from a template implementation body.",
    operands=[Operand("operands", ANY, variadic=True)],
    traits=[TERMINATOR, POISON_BOUNDARY],
    format=[
        OptionalGroup(
            [Refs("operands"), COLON, TypesOf("operands")],
            anchor="operands",
        ),
    ],
    examples=[
        "template.return",
        "template.return %result : f32",
    ],
)

ALL_TEMPLATE_OPS: tuple[Op, ...] = (
    template_decl,
    template_def,
    template_ukernel,
    template_apply,
    template_call,
    template_return,
)
