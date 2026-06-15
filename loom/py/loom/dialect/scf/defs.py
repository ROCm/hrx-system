# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SCF dialect op definitions.

Core ops:

  scf.for     — Bounded counted loop with bracketed range syntax.
  scf.if      — Conditional execution with optional else for resultless ops.
  scf.while   — Unbounded loop with explicit before/after regions.
  scf.lookup  — Keyed whole-value table selection.
  scf.select  — Scalar-condition whole-value selection.
  scf.condition — Boolean forwarding terminator for scf.while before regions.
  scf.switch  — Multi-way index branch with explicit case regions.
  scf.yield   — Variadic terminator for scf value-forwarding regions.

Every scf op region still contains a concrete terminator in memory. Text forms
may elide an empty `scf.yield` for ops whose regions all use it as their
terminator; the parser materializes the terminator again on read.
"""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    LBRACKET,
    RBRACKET,
    Attr,
    AttrTable,
    BindingList,
    BlockArgs,
    Clause,
    OptionalGroup,
    Ref,
    Refs,
    Region,
    RegionTable,
    ResultType,
    ResultTypeList,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64_ARRAY,
    DISTRIBUTION_TRANSFER,
    I1,
    INDEX,
    PURE,
    SAFE_TO_SPECULATE,
    TERMINATOR,
    AttrDef,
    BlockArgCount,
    BlockArgsMatchTypes,
    Dialect,
    EnumCase,
    EnumDef,
    ImplicitTerminator,
    IterArgsMatchResults,
    LoopLikeInterface,
    Op,
    Operand,
    OpPhase,
    RegionBranchInterface,
    RegionDef,
    Result,
    SameType,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
)

# ============================================================================
# Dialect
# ============================================================================

scf_ops = Dialect(
    "scf",
    dialect_id=0x05,
    doc="Structured control flow operations.",
    default_phase=OpPhase.SOURCE_STRUCTURE,
)

ScfForUnrollPolicy = EnumDef(
    "ScfForUnrollPolicy",
    [
        EnumCase(
            "unroll",
            1,
            doc="Require full unrolling when the consuming transform runs.",
        ),
    ],
    doc="Local scf.for unroll policy.",
)

# ============================================================================
# scf.yield — variadic region terminator
# ============================================================================
#
# Shared terminator for scf regions that forward values to a parent op. Empty
# `scf.yield` may be implicit in text for parent ops that opt in, but the
# terminator is always present in memory.

scf_yield = Op(
    "scf.yield",
    group=scf_ops,
    doc="Region terminator forwarding values to the parent scf op.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=[
        "scf.yield",
        "scf.yield %first, %second : f32, i32",
    ],
)

# ============================================================================
# scf.condition — while condition terminator
# ============================================================================
#
# Terminates the `before` region of scf.while. The first operand is the loop
# continuation condition. The forwarded operands become the entry block
# arguments of the `after` region on the next iteration.

scf_condition = Op(
    "scf.condition",
    group=scf_ops,
    doc=("Terminates the before region of scf.while with a scalar i1 continuation condition and the values forwarded to the after region."),
    operands=[
        Operand("condition", I1),
        Operand("forwarded", ANY, variadic=True),
    ],
    traits=[TERMINATOR],
    format=[
        Ref("condition"),
        OptionalGroup(
            [COMMA, Refs("forwarded")],
            anchor="forwarded",
        ),
        COLON,
        TypeOf("condition"),
        OptionalGroup(
            [COMMA, TypesOf("forwarded")],
            anchor="forwarded",
        ),
    ],
    examples=[
        "scf.condition %keep_going : i1",
        "scf.condition %keep_going, %next : i1, index",
    ],
)

# ============================================================================
# scf.select — scalar-condition whole-value select
# ============================================================================
#
# Chooses one of two already-computed SSA values using a scalar i1 condition.
# This is deliberately different from vector.select, which is a lanewise
# vector-mask operation. scf.select is the generic result of reducing
# yield-only scf.if regions and can select scalars, address-domain values,
# vectors, tiles, views, encodings, or any other same-typed SSA value without
# making the scf dialect depend on every value dialect.

scf_select = Op(
    "scf.select",
    group=scf_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Select between two same-typed SSA values using a scalar i1 condition. "
        "This is whole-value selection: if the selected values are vectors or "
        "tiles, the entire aggregate is chosen as one value. Lanewise vector "
        "masking remains vector.select."
    ),
    operands=[
        Operand("condition", I1),
        Operand("true_value", ANY),
        Operand("false_value", ANY),
    ],
    results=[Result("result", ANY)],
    constraints=[SameType("true_value", "false_value", "result")],
    traits=[PURE, SAFE_TO_SPECULATE, DISTRIBUTION_TRANSFER],
    canonicalize="loom_scf_select_canonicalize",
    facts="loom_scf_select_facts",
    format=[
        Ref("condition"),
        COMMA,
        Ref("true_value"),
        COMMA,
        Ref("false_value"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%r = scf.select %cond, %a, %b : f32",
        "%v = scf.select %cond, %then_vec, %else_vec : vector<16xf32>",
    ],
)

# ============================================================================
# scf.lookup — keyed whole-value table selection
# ============================================================================
#
# Total table lookup over already-computed SSA values. The case table names
# every explicit selector value, and every row lists one complete result tuple.
# The final `default(...)` row is selected when the selector does not equal any
# explicit key. The grouped assembly is backed by generic i64-array attrs and
# variadic operands, preserving a strict verifier contract without a custom
# op-specific parser.

scf_lookup = Op(
    "scf.lookup",
    group=scf_ops,
    doc=(
        "Total table lookup over already-computed SSA values. The selector is "
        "an index value. The table gives sorted unique explicit case rows. "
        "default(...) gives the total fallback row. The result count is the "
        "row width, and every payload value in a column must match that "
        "column's result type."
    ),
    operands=[
        Operand("selector", INDEX),
        Operand("values", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Sorted unique selector values for explicit lookup cases.",
        ),
    ],
    traits=[PURE, SAFE_TO_SPECULATE],
    verify="loom_scf_lookup_verify",
    canonicalize="loom_scf_lookup_canonicalize",
    facts="loom_scf_lookup_facts",
    format=[
        Ref("selector"),
        AttrTable("case_keys", "values"),
        COLON,
        ResultTypeList("results", parens=False),
    ],
    examples=[
        "%ordinal, %wgx = scf.lookup %variant {\n  0 = (%gemm0, %x0),\n  1 = (%gemm1, %x1)\n} default(%fallback, %xf) : index, index",
    ],
)

# ============================================================================
# scf.for — bounded counted loop
# ============================================================================
#
# Iterates an induction variable from lower_bound (inclusive) to
# upper_bound (exclusive) by step. Optionally carries loop state via
# a binding list that captures values into the body's entry block.
# Body terminated by scf.yield.
#
# Surface syntax pattern:
#
#   scf.for %iv = [%lo to %hi step %step](%a = %init : T) -> (T) { body }
#
# The bracketed range groups the iteration bounds. The parenthesized
# binding list (when present) captures values that the loop carries
# across iterations — these become block args on the body region's
# entry block alongside the induction variable. There is no
# `iter_args` keyword: the parens themselves are the syntactic marker
# and reading "for iv = range(captures) -> results" mirrors the
# function-call shape rather than tagging the captures with a name
# that only describes their role.

scf_for = Op(
    "scf.for",
    group=scf_ops,
    doc="Bounded counted loop with optional loop-carried state.",
    canonicalize="loom_scf_for_canonicalize",
    verify="loom_scf_for_verify",
    operands=[
        Operand("lower_bound", INDEX),
        Operand("upper_bound", INDEX),
        Operand("step", INDEX),
        Operand("iter_args", ANY, variadic=True),
        Operand(
            "unroll_factor",
            INDEX,
            optional=True,
            doc="Optional SSA unroll factor policy consumed by unroll transforms.",
        ),
    ],
    attrs=[
        AttrDef(
            "unroll_policy",
            ATTR_TYPE_ENUM,
            enum_def=ScfForUnrollPolicy,
            optional=True,
            doc="Optional bare unroll policy for required full unroll.",
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Loop body. Terminated by scf.yield.",
            single_block=True,
            terminator="scf.yield",
            implicit_args=(("iv", "index"),),
        ),
    ],
    interfaces=[
        LoopLikeInterface(
            body="body",
            iter_args="iter_args",
            iv="iv",
            lower_bound="lower_bound",
            upper_bound="upper_bound",
            step="step",
        ),
    ],
    constraints=[
        IterArgsMatchResults("iter_args", "results"),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    traits=[ImplicitTerminator("scf.yield")],
    format=[
        Ref("iv"),
        EQUALS,
        LBRACKET,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        RBRACKET,
        OptionalGroup(
            [BindingList("iter_args")],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        OptionalGroup(
            [Clause("unroll", Ref("unroll_factor"))],
            anchor="unroll_factor",
        ),
        OptionalGroup(
            [Attr("unroll_policy")],
            anchor="unroll_policy",
        ),
        Region("body"),
    ],
    examples=[
        "scf.for %iv = [%c0 to %n step %c1] {\n  scf.yield\n}",
        "%result = scf.for %iv = [%c0 to %n step %c1](%acc = %init : f32) -> (f32) {\n  %next = scalar.addf %acc, %acc : f32\n  scf.yield %next : f32\n}",
    ],
)

# ============================================================================
# scf.while — unbounded loop with explicit before/after regions
# ============================================================================
#
# Runs the `before` region, branches on its scf.condition result, then runs the
# `after` region when the condition is true. Loop-carried values are named in
# both regions explicitly. The before region uses BindingList because each
# block argument is bound to an initial/current operand. The after region uses
# BlockArgs because the values arrive from scf.condition, not directly from the
# while op surface syntax.

scf_while = Op(
    "scf.while",
    group=scf_ops,
    doc=("Unbounded loop with explicit before and after regions. The before region terminates with scf.condition, and the after region terminates with scf.yield."),
    verify="loom_scf_while_verify",
    operands=[Operand("iter_args", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "before",
            doc="Runs before each condition check. Terminated by scf.condition.",
            single_block=True,
            terminator="scf.condition",
        ),
        RegionDef(
            "after",
            doc="Runs when the condition is true. Terminated by scf.yield.",
            single_block=True,
            terminator="scf.yield",
            arg_source="iter_args",
        ),
    ],
    interfaces=[
        LoopLikeInterface(
            body="after",
            condition_region="before",
            iter_args="iter_args",
        ),
    ],
    constraints=[
        IterArgsMatchResults("iter_args", "results"),
        BlockArgCount("before", "iter_args"),
        BlockArgsMatchTypes("before", "iter_args"),
        BlockArgCount("after", "iter_args"),
        BlockArgsMatchTypes("after", "iter_args"),
        YieldCountMatchesResults("after", "results"),
        YieldTypesMatchResults("after", "results"),
    ],
    format=[
        OptionalGroup(
            [BindingList("iter_args")],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("before"),
        kw("do"),
        OptionalGroup(
            [BlockArgs("after")],
            anchor="iter_args",
        ),
        Region("after"),
    ],
    examples=[
        "scf.while {\n  scf.condition %cond : i1\n} do {\n  scf.yield\n}",
        "%result = scf.while(%before = %init : index) -> (index) {\n  scf.condition %keep_going, %before : i1, index\n} do(%body: index) {\n  %next = index.add %body, %one : index\n  scf.yield %next : index\n}",
    ],
)

# ============================================================================
# scf.if — conditional with optional else for resultless ops
# ============================================================================
#
# Executes one of two regions based on a boolean condition. The else region may
# be omitted only for resultless conditionals; result-producing scf.if needs a
# total false path to define its result tuple.

scf_if = Op(
    "scf.if",
    group=scf_ops,
    doc="Conditional execution with optional else region for resultless conditionals.",
    verify="loom_scf_if_verify",
    canonicalize="loom_scf_if_canonicalize",
    type_transfer="loom_scf_region_branch_type_transfer",
    operands=[Operand("condition", I1)],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "then_region",
            doc="Executed when condition is true. Terminated by scf.yield.",
            single_block=True,
            terminator="scf.yield",
        ),
        RegionDef(
            "else_region",
            doc="Executed when condition is false. Terminated by scf.yield.",
            single_block=True,
            optional=True,
            terminator="scf.yield",
        ),
    ],
    interfaces=[
        RegionBranchInterface(selector="condition"),
    ],
    constraints=[
        YieldCountMatchesResults("then_region", "results"),
        YieldTypesMatchResults("then_region", "results"),
        YieldCountMatchesResults("else_region", "results"),
        YieldTypesMatchResults("else_region", "results"),
    ],
    traits=[ImplicitTerminator("scf.yield")],
    format=[
        Ref("condition"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("then_region"),
        OptionalGroup(
            [kw("else"), Region("else_region")],
            anchor="else_region",
        ),
    ],
    examples=[
        "scf.if %cond {\n  scf.yield\n}",
        "scf.if %cond {\n  scf.yield\n} else {\n  scf.yield\n}",
        "%result = scf.if %cond -> (f32) {\n  scf.yield %a : f32\n} else {\n  scf.yield %b : f32\n}",
    ],
)

# ============================================================================
# scf.switch — multi-way branch with explicit default
# ============================================================================
#
# Branches on an index selector. Each explicit case owns a full region, and the
# default region is mandatory so the op is total. The printed case table is
# block-shaped instead of row-shaped because each entry may contain arbitrary
# side-effecting IR before yielding the parent results.

scf_switch = Op(
    "scf.switch",
    group=scf_ops,
    doc=(
        "Multi-way branch over an index selector. Case keys are sorted unique "
        "i64 literals. The default region is mandatory and is selected when "
        "the selector does not equal any explicit case key. Every region must "
        "terminate with scf.yield matching the switch result tuple."
    ),
    canonicalize="loom_scf_switch_canonicalize",
    verify="loom_scf_switch_verify",
    type_transfer="loom_scf_region_branch_type_transfer",
    operands=[Operand("selector", INDEX)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Sorted unique selector values for explicit switch cases.",
        ),
    ],
    regions=[
        RegionDef(
            "default_region",
            doc="Executed when no explicit case key matches. Terminated by scf.yield.",
            single_block=True,
            terminator="scf.yield",
        ),
        RegionDef(
            "case_regions",
            doc="Case regions in the same order as case_keys. Terminated by scf.yield.",
            single_block=True,
            variadic=True,
            terminator="scf.yield",
        ),
    ],
    interfaces=[
        RegionBranchInterface(selector="selector"),
    ],
    traits=[ImplicitTerminator("scf.yield")],
    format=[
        Ref("selector"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        RegionTable("case_keys", "case_regions", "default_region"),
    ],
    examples=[
        "scf.switch %selector {\n  case 0 {\n    scf.yield\n  }\n  default {\n    scf.yield\n  }\n}",
        "%result = scf.switch %selector -> (f32) {\n  case 0 {\n    scf.yield %a : f32\n  }\n  case 1 {\n    scf.yield %b : f32\n  }\n  default {\n    scf.yield %fallback : f32\n  }\n}",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_SCF_OPS: tuple[Op, ...] = (
    scf_for,
    scf_if,
    scf_switch,
    scf_yield,
    scf_select,
    scf_lookup,
    scf_condition,
    scf_while,
)
