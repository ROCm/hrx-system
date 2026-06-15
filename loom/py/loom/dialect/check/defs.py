# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check dialect op definitions.

The check dialect is a production testbench dialect. A check.case is an
ordinary SSA program that materializes inputs, calls functions, obtains
expected outputs, and checks values.
"""

from loom.assembly import (
    ARROW,
    COLON,
    LPAREN,
    RPAREN,
    TO,
    Attr,
    AttrDict,
    Clause,
    FormatElement,
    IndexList,
    OpRef,
    OptionalGroup,
    Ref,
    Refs,
    Region,
    ResultTypeList,
    SymbolRef,
    TemplateParam,
    TypeOf,
    TypesOf,
)
from loom.dsl import (
    ANY,
    CONSTANT_LIKE,
    INDEX,
    INDEX_OR_NON_I1_INTEGER_SCALAR,
    INTEGER,
    ISOLATED_FROM_ABOVE,
    PURE,
    SCALAR,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    HasAncestor,
    LiteralMatchesElementType,
    OffsetCountMatchesRank,
    Op,
    Operand,
    RegionDef,
    Result,
    SameType,
    SymbolDefinition,
    SymbolReference,
)

check_ops = Dialect("check", dialect_id=0x16, doc="Production testbench operations.")


Visibility = EnumDef(
    "Visibility",
    [EnumCase("public", 1, doc="Visible to testbench discovery.")],
    doc="Check symbol visibility. Absent (0) means private.",
)

RangePolicy = EnumDef(
    "RangePolicy",
    [
        EnumCase("linear", 1, doc="Linearly samples the closed interval."),
        EnumCase("po2", 2, doc="Samples powers of two in the closed interval."),
    ],
    doc="Deterministic scalar range sampling policy.",
)

NanPolicy = EnumDef(
    "NanPolicy",
    [
        EnumCase("same", 1, doc="NaNs compare equal only when both sides are NaN."),
        EnumCase("different", 2, doc="NaNs compare unequal."),
    ],
    doc="NaN comparison policy for approximate expectations.",
)

FileWriteMode = EnumDef(
    "FileWriteMode",
    [
        EnumCase("always", 1, doc="Always writes the value."),
        EnumCase("on_failure", 2, doc="Writes the value only when the case fails."),
    ],
    doc="Fixture output policy for file writes.",
)


_CASE_SYMBOL_ATTRS = [
    AttrDef("case_symbol", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
]

_CASE_BODY_TRAITS = [UNKNOWN_EFFECTS, HasAncestor("check.case")]

_CASE_SYMBOL_DEF = SymbolDefinition(
    field="case_symbol",
    name="check case",
    interfaces=["record"],
    bytecode_kind="LOOM_SYMBOL_RECORD",
)


# ============================================================================
# Harness structure
# ============================================================================

check_case = Op(
    "check.case",
    group=check_ops,
    doc="Named correctness harness.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=_CASE_SYMBOL_ATTRS,
    symbol_def=_CASE_SYMBOL_DEF,
    regions=[RegionDef("body", doc="Case body.", terminator="check.return")],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        SymbolRef("case_symbol"),
        Region("body"),
    ],
    examples=[
        "check.case @empty {\n  check.return\n}",
        "check.case public @smoke {\n  check.return\n}",
    ],
)

check_return = Op(
    "check.return",
    group=check_ops,
    doc="Terminates a check.case body.",
    traits=[TERMINATOR, HasAncestor("check.case")],
    format=[],
    examples=["check.return"],
)


# ============================================================================
# Requirements and skips
# ============================================================================

check_requires = Op(
    "check.requires",
    group=check_ops,
    doc="Declares a provider requirement; unmet requirements skip the case.",
    attrs=[
        AttrDef("provider", "string"),
        AttrDef("attrs", "dict"),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        OpRef("provider"),
        AttrDict("attrs"),
    ],
    examples=['check.requires<target.feature> {feature = "amdgpu.gfx11"}'],
)

check_skip_if = Op(
    "check.skip_if",
    group=check_ops,
    doc="Declares a provider skip predicate for exceptional environments.",
    attrs=[
        AttrDef("provider", "string"),
        AttrDef("attrs", "dict"),
        AttrDef("reason", "string", optional=True),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        OpRef("provider"),
        AttrDict("attrs"),
        OptionalGroup([Clause("reason", Attr("reason"))], anchor="reason"),
    ],
    examples=[
        'check.skip_if<device.memory> {max_bytes = 1073741824} reason("fixture too large")',
    ],
)


# ============================================================================
# Parameter sweeps
# ============================================================================

check_param_range = Op(
    "check.param.range",
    group=check_ops,
    doc="Produces one sampled scalar parameter from a static interval.",
    results=[Result("result", SCALAR)],
    attrs=[
        AttrDef("policy", "enum", enum_def=RangePolicy),
        AttrDef("lower", "any"),
        AttrDef("upper", "any"),
        AttrDef("step", "any", optional=True),
        AttrDef("param_name", "string", optional=True),
    ],
    constraints=[
        LiteralMatchesElementType("lower", "result"),
        LiteralMatchesElementType("upper", "result"),
        LiteralMatchesElementType("step", "result"),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        Attr("policy"),
        Clause("bounds", Attr("lower"), TO, Attr("upper")),
        OptionalGroup([Clause("step", Attr("step"))], anchor="step"),
        OptionalGroup([Clause("name", Attr("param_name"))], anchor="param_name"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%m = check.param.range po2 bounds(1 to 64) : index",
        "%alpha = check.param.range linear bounds(0.0 to 1.0) step(0.25) : f32",
    ],
)

check_param_choice = Op(
    "check.param.choice",
    group=check_ops,
    doc="Produces one sampled integer/index parameter from an explicit choice set.",
    results=[Result("result", INDEX_OR_NON_I1_INTEGER_SCALAR)],
    attrs=[
        AttrDef("values", "i64_array"),
        AttrDef("param_name", "string", optional=True),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        Clause("values", Attr("values")),
        OptionalGroup([Clause("name", Attr("param_name"))], anchor="param_name"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%k = check.param.choice values([16, 24, 32, 64]) : index",
    ],
)

check_param_seed = Op(
    "check.param.seed",
    group=check_ops,
    doc="Produces deterministic i64 seeds for randomized generators.",
    results=[Result("result", INTEGER)],
    attrs=[
        AttrDef("base", "i64"),
        AttrDef("count", "i64"),
        AttrDef("param_name", "string", optional=True),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        Clause("base", Attr("base")),
        Clause("count", Attr("count")),
        OptionalGroup([Clause("name", Attr("param_name"))], anchor="param_name"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%seed = check.param.seed base(0x4c6f6f6d) count(32) : i64",
    ],
)


# ============================================================================
# Value sources
# ============================================================================

check_literal = Op(
    "check.literal",
    group=check_ops,
    doc="Materializes a typed scalar literal for reproducer inputs or expected values.",
    results=[Result("result", SCALAR)],
    attrs=[AttrDef("value", "any")],
    constraints=[LiteralMatchesElementType("value", "result")],
    traits=[PURE, CONSTANT_LIKE, HasAncestor("check.case")],
    format=[
        Clause("value", Attr("value")),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%scalar = check.literal value(42) : i32",
    ],
)

check_generate_iota = Op(
    "check.generate.iota",
    group=check_ops,
    doc="Generates a deterministic iota-shaped value.",
    results=[Result("result", ANY)],
    attrs=[
        AttrDef("offset", "any"),
        AttrDef("step", "any"),
    ],
    constraints=[
        LiteralMatchesElementType("offset", "result"),
        LiteralMatchesElementType("step", "result"),
    ],
    traits=[PURE, HasAncestor("check.case")],
    format=[
        Clause("offset", Attr("offset")),
        Clause("step", Attr("step")),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%lhs = check.generate.iota offset(0) step(1) : tensor<[%m]x[%n]xi32>",
    ],
)

check_generate_fill = Op(
    "check.generate.fill",
    group=check_ops,
    doc="Generates a value filled with one static scalar payload.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("value", "any")],
    constraints=[LiteralMatchesElementType("value", "result")],
    traits=[PURE, HasAncestor("check.case")],
    format=[
        Clause("value", Attr("value")),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%rhs = check.generate.fill value(17) : tensor<[%m]x[%n]xi32>",
    ],
)

check_generate_random_uniform = Op(
    "check.generate.random.uniform",
    group=check_ops,
    doc="Generates a deterministic random-uniform value from a sampled seed.",
    operands=[Operand("seed", INTEGER)],
    results=[Result("result", ANY)],
    attrs=[
        AttrDef("lower", "any"),
        AttrDef("upper", "any"),
    ],
    constraints=[
        LiteralMatchesElementType("lower", "result"),
        LiteralMatchesElementType("upper", "result"),
    ],
    traits=[PURE, HasAncestor("check.case")],
    format=[
        Clause("seed", Ref("seed")),
        Clause("range", Attr("lower"), TO, Attr("upper")),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%input = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<[%m]xf32>",
    ],
)

check_file_read_npy = Op(
    "check.file.read.npy",
    group=check_ops,
    doc="Reads a typed value from an NPY fixture file.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("path", "string")],
    traits=_CASE_BODY_TRAITS,
    format=[
        Clause("path", Attr("path")),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        '%input = check.file.read.npy path("fixtures/layer_norm/input.npy") : tensor<1024xf32>',
    ],
)

check_file_write_npy = Op(
    "check.file.write.npy",
    group=check_ops,
    doc="Writes a typed value to an NPY file according to a static output policy.",
    operands=[Operand("value", ANY)],
    attrs=[
        AttrDef("path", "string"),
        AttrDef("mode", "enum", enum_def=FileWriteMode, optional=True),
    ],
    traits=_CASE_BODY_TRAITS,
    format=[
        Clause("value", Ref("value")),
        Clause("path", Attr("path")),
        OptionalGroup([Clause("mode", Attr("mode"))], anchor="mode"),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        'check.file.write.npy value(%actual) path("outputs/layer_norm.actual.npy") mode(on_failure) : tensor<1024xf32>',
    ],
)


# ============================================================================
# Oracles
# ============================================================================

check_oracle_call = Op(
    "check.oracle.call",
    group=check_ops,
    doc="Calls a pluggable oracle provider for expected values.",
    operands=[Operand("inputs", ANY, variadic=True)],
    attrs=[
        AttrDef("provider", "string"),
        AttrDef("attrs", "dict", optional=True),
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=_CASE_BODY_TRAITS,
    format=[
        OpRef("provider"),
        OptionalGroup([AttrDict("attrs")], anchor="attrs"),
        Clause("callee", SymbolRef("callee")),
        Clause("inputs", Refs("inputs")),
        COLON,
        LPAREN,
        TypesOf("inputs"),
        RPAREN,
        ARROW,
        ResultTypeList("results"),
    ],
    examples=[
        "%expected = check.oracle.call<reference.scalar> callee(@gemv_f32) inputs(%lhs, %rhs) : (tensor<[%m]x[%n]xf32>, tensor<[%n]xf32>) -> (tensor<[%m]xf32>)",
    ],
)


# ============================================================================
# Expectations
# ============================================================================

_EXPECT_OPERANDS = [
    Operand("actual", ANY),
    Operand("expected", ANY),
]

_EXPECT_VALUE_CLAUSES: list[FormatElement] = [
    Clause("actual", Ref("actual")),
    Clause("expected", Ref("expected")),
]

_EXPECT_FORMAT: list[FormatElement] = [
    *_EXPECT_VALUE_CLAUSES,
    COLON,
    TypeOf("actual"),
]

check_expect_equal = Op(
    "check.expect.equal",
    group=check_ops,
    doc="Requires actual and expected values to compare equal.",
    operands=_EXPECT_OPERANDS,
    constraints=[SameType("actual", "expected")],
    traits=_CASE_BODY_TRAITS,
    format=_EXPECT_FORMAT,
    examples=[
        "check.expect.equal actual(%actual) expected(%expected) : tensor<[%m]xi32>",
    ],
)

check_expect_bitwise = Op(
    "check.expect.bitwise",
    group=check_ops,
    doc="Requires actual and expected values to match bit-for-bit.",
    operands=_EXPECT_OPERANDS,
    constraints=[SameType("actual", "expected")],
    traits=_CASE_BODY_TRAITS,
    format=_EXPECT_FORMAT,
    examples=[
        "check.expect.bitwise actual(%actual) expected(%expected) : tensor<1024xf32>",
    ],
)

check_expect_close = Op(
    "check.expect.close",
    group=check_ops,
    doc="Requires actual and expected floating-point values to be approximately equal.",
    operands=_EXPECT_OPERANDS,
    attrs=[
        AttrDef("atol", "f64"),
        AttrDef("rtol", "f64"),
        AttrDef("nan", "enum", enum_def=NanPolicy),
    ],
    constraints=[SameType("actual", "expected")],
    traits=_CASE_BODY_TRAITS,
    format=[
        *_EXPECT_VALUE_CLAUSES,
        Clause("atol", Attr("atol")),
        Clause("rtol", Attr("rtol")),
        Clause("nan", Attr("nan")),
        COLON,
        TypeOf("actual"),
    ],
    examples=[
        "check.expect.close actual(%actual) expected(%expected) atol(0.0001) rtol(0.0001) nan(same) : tensor<[%m]xf32>",
    ],
)

check_expect_shape = Op(
    "check.expect.shape",
    group=check_ops,
    doc="Requires a shaped value to have the expected dynamic/static shape.",
    operands=[
        Operand("value", ANY),
        Operand("dims", INDEX, variadic=True),
    ],
    attrs=[AttrDef("static_dims", "i64_array")],
    constraints=[OffsetCountMatchesRank("value", "static_dims")],
    traits=_CASE_BODY_TRAITS,
    format=[
        Clause("value", Ref("value")),
        Clause("shape", IndexList("dims", "static_dims")),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        "check.expect.shape value(%actual) shape([%m, %n, 4]) : tensor<[%m]x[%n]x4xf32>",
    ],
)

check_expect = Op(
    "check.expect",
    group=check_ops,
    doc="Runs a pluggable custom validator over actual and expected values.",
    operands=_EXPECT_OPERANDS,
    attrs=[
        AttrDef("provider", "string"),
        AttrDef("attrs", "dict", optional=True),
    ],
    constraints=[SameType("actual", "expected")],
    traits=_CASE_BODY_TRAITS,
    format=[
        OpRef("provider"),
        *_EXPECT_VALUE_CLAUSES,
        OptionalGroup([AttrDict("attrs")], anchor="attrs"),
        COLON,
        TypeOf("actual"),
    ],
    examples=[
        "check.expect<topk.equal> actual(%actual) expected(%expected) {k = 5} : tensor<1000xf32>",
    ],
)


# ============================================================================
# Benchmark records
# ============================================================================

check_benchmark = Op(
    "check.benchmark",
    group=check_ops,
    doc="Declares a benchmark slice over a check.case.",
    traits=[SYMBOL_DEFINE],
    attrs=[
        AttrDef("benchmark", "symbol", optional=True),
        AttrDef(
            "case_ref",
            "symbol",
            symbol_ref=SymbolReference("check case", ["record"]),
        ),
        AttrDef("attrs", "dict", optional=True),
    ],
    symbol_def=SymbolDefinition(
        field="benchmark",
        name="check benchmark",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    format=[
        TemplateParam("case_ref"),
        OptionalGroup([SymbolRef("benchmark")], anchor="benchmark"),
        AttrDict("attrs"),
    ],
    examples=[
        "check.benchmark<@gemv_sweep> @gemv_latency {m = 8, n = 96}",
    ],
)


ALL_CHECK_OPS = (
    check_case,
    check_return,
    check_requires,
    check_skip_if,
    check_param_range,
    check_param_choice,
    check_param_seed,
    check_literal,
    check_generate_iota,
    check_generate_fill,
    check_generate_random_uniform,
    check_file_read_npy,
    check_file_write_npy,
    check_oracle_call,
    check_expect_equal,
    check_expect_bitwise,
    check_expect_close,
    check_expect_shape,
    check_expect,
    check_benchmark,
)
