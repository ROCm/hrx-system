# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM IR target dialect op definitions."""

from __future__ import annotations

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    Flags,
    OpRef,
    OptionalGroup,
    Refs,
    ResultTypeList,
    SymbolRef,
    TemplateParam,
    TypesOf,
)
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ANY,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_STRING,
    SYMBOL_DEFINE,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SymbolDefinition,
    TargetLikeInterface,
)

llvmir_ops = Dialect(
    "llvmir",
    dialect_id=0x11,
    doc=("LLVM IR target punch-through operations. These ops preserve target specific intent in Loom IR while still lowering through structured LLVMIR builders instead of raw textual emission."),
)

AsmFlags = EnumDef(
    "AsmFlags",
    [
        EnumCase("sideeffect", 1, doc="Inline asm has side effects."),
        EnumCase("alignstack", 2, doc="Inline asm may require stack realignment."),
        EnumCase("inteldialect", 4, doc="Inline asm uses Intel assembly syntax."),
    ],
    doc="LLVM inline asm call flags.",
)

LlvmirTargetKind = EnumDef(
    "LlvmirTargetKind",
    [
        EnumCase("object", 1, doc="Generic LLVM object-function target row."),
    ],
    doc="LLVMIR target row selected by llvmir.target.",
)

llvmir_target = Op(
    "llvmir.target",
    group=llvmir_ops,
    doc=("LLVMIR target-family record. The selector chooses an LLVMIR bundle row while optional LLVM-specific attributes own the triple, data layout, CPU, and feature-string vocabulary."),
    phase=OpPhase.MODULE_METADATA,
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_llvmir_target_bundles",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=[
        *target_record_attrs(LlvmirTargetKind),
        AttrDef(
            "triple",
            ATTR_TYPE_STRING,
            doc="LLVM target triple emitted into the module and passed to LLVM tools.",
        ),
        AttrDef(
            "data_layout",
            ATTR_TYPE_STRING,
            optional=True,
            doc="LLVM data layout emitted into the module when present.",
        ),
        AttrDef(
            "cpu",
            ATTR_TYPE_STRING,
            optional=True,
            doc="LLVM target CPU passed to LLVM tools when present.",
        ),
        AttrDef(
            "features",
            ATTR_TYPE_STRING,
            optional=True,
            doc="LLVM target feature string passed to LLVM tools when present.",
        ),
    ],
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'llvmir.target<object> @llvm_host {triple = "x86_64-unknown-linux-gnu"}',
    ],
)

llvmir_inline_asm = Op(
    "llvmir.inline_asm",
    group=llvmir_ops,
    doc=("Structured LLVM inline assembly call. The asm template and constraint strings use LLVM inline asm syntax; operands/results remain ordinary typed Loom SSA values."),
    operands=[Operand("operands", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef("flags", ATTR_TYPE_FLAGS, optional=True, enum_def=AsmFlags),
        AttrDef(
            "asm_template",
            ATTR_TYPE_STRING,
            doc="LLVM inline asm template string.",
        ),
        AttrDef(
            "constraints",
            ATTR_TYPE_STRING,
            doc="LLVM inline asm constraint string.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    format=[
        Flags("flags"),
        Attr("asm_template"),
        COMMA,
        Attr("constraints"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup([ARROW, ResultTypeList("results", parens=False)], anchor="results"),
    ],
    examples=[
        '%sum = llvmir.inline_asm<sideeffect> "addl $2, $0", "=r,r,r"(%lhs, %rhs) : (i32, i32) -> i32',
    ],
)

llvmir_intrinsic = Op(
    "llvmir.intrinsic",
    group=llvmir_ops,
    doc=("Structured call to a supported LLVM intrinsic. The intrinsic spelling is a string so target-family providers can recognize their own intrinsics without extending a central enum."),
    operands=[Operand("operands", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "kind",
            ATTR_TYPE_STRING,
            doc=("LLVM intrinsic spelling, such as llvm.memcpy or llvm.amdgcn.workitem.id.x."),
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    format=[
        OpRef("kind"),
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup([ARROW, ResultTypeList("results", parens=False)], anchor="results"),
    ],
    examples=[
        "%ticks = llvmir.intrinsic<llvm.x86.rdtsc> () : () -> i64",
        "llvmir.intrinsic<llvm.x86.sse2.pause> () : ()",
        "llvmir.intrinsic<llvm.memcpy> (%target, %source, %length, %is_volatile) : (!buffer, !buffer, offset, i1)",
    ],
)

ALL_LLVMIR_OPS = (
    llvmir_target,
    llvmir_inline_asm,
    llvmir_intrinsic,
)
