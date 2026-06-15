# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pass pipeline control dialect op definitions."""

from loom.assembly import (
    Attr,
    AttrDict,
    OpRef,
    Region,
    SymbolRef,
    TemplateParam,
)
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_STRING,
    ISOLATED_FROM_ABOVE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    ImplicitTerminator,
    Op,
    RegionDef,
    SymbolDefinition,
    SymbolReference,
)

# ============================================================================
# Dialect
# ============================================================================

pass_ops = Dialect(
    "pass",
    dialect_id=0x15,
    doc="Descriptor-backed pass pipeline control operations.",
)

# ============================================================================
# Shared enums
# ============================================================================

PassAnchor = EnumDef(
    "PassAnchor",
    [
        EnumCase("module", 0, doc="The active anchor is the whole module."),
        EnumCase("func", 1, doc="The active anchor is one function-like symbol."),
    ],
    doc="Pass pipeline execution anchor.",
)

PassRepeatMode = EnumDef(
    "PassRepeatMode",
    [
        EnumCase("fixed", 0, doc="Run the body a fixed number of times."),
        EnumCase(
            "until_converged",
            1,
            doc="Run the body until no pass reports a mutation.",
        ),
    ],
    doc="Pass pipeline repeat mode.",
)

# ============================================================================
# pass.pipeline<anchor>
# ============================================================================

pass_pipeline = Op(
    "pass.pipeline",
    group=pass_ops,
    doc="Named pass pipeline entry point.",
    traits=[
        SYMBOL_DEFINE,
        ISOLATED_FROM_ABOVE,
        ImplicitTerminator("pass.yield"),
    ],
    attrs=[
        AttrDef("anchor", ATTR_TYPE_ENUM, enum_def=PassAnchor),
        AttrDef("symbol", "symbol"),
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="pass pipeline",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    regions=[
        RegionDef(
            "body",
            doc="Pass pipeline body.",
            terminator="pass.yield",
        )
    ],
    format=[
        TemplateParam("anchor"),
        SymbolRef("symbol"),
        Region("body", syntax="pipeline"),
    ],
    examples=[
        "pass.pipeline<module> @cleanup pipeline {\n  canonicalize\n}",
        ("pass.pipeline<func> @function_cleanup pipeline {\n  for func {\n    cse\n  }\n}"),
    ],
)

# ============================================================================
# pass.for<anchor>
# ============================================================================

pass_for = Op(
    "pass.for",
    group=pass_ops,
    doc="Deterministically iterate symbols of the selected anchor kind.",
    traits=[UNKNOWN_EFFECTS, ImplicitTerminator("pass.yield")],
    attrs=[
        AttrDef("anchor", ATTR_TYPE_ENUM, enum_def=PassAnchor),
    ],
    regions=[
        RegionDef(
            "body",
            doc="Body executed once for each selected symbol.",
            terminator="pass.yield",
        )
    ],
    format=[
        TemplateParam("anchor"),
        Region("body", syntax="pipeline"),
    ],
    examples=[
        "pass.for<func> pipeline {\n  cse\n}",
    ],
)

# ============================================================================
# pass.where<predicate>
# ============================================================================

pass_where = Op(
    "pass.where",
    group=pass_ops,
    doc="Guard a nested pipeline body with a descriptor-backed pass predicate.",
    traits=[UNKNOWN_EFFECTS, ImplicitTerminator("pass.yield")],
    attrs=[
        AttrDef("predicate", ATTR_TYPE_STRING),
        AttrDef("attrs", "dict", optional=True),
    ],
    regions=[
        RegionDef(
            "body",
            doc="Body executed when the predicate matches the current context.",
            terminator="pass.yield",
        )
    ],
    format=[
        OpRef("predicate"),
        AttrDict("attrs"),
        Region("body", syntax="pipeline"),
    ],
    examples=[
        'pass.where<name> {value = "matmul"} pipeline {\n  canonicalize\n}',
        'pass.where<has_attr> {name = "kernel.entry"} pipeline {\n  dce\n}',
    ],
)

# ============================================================================
# pass.repeat<mode>
# ============================================================================

pass_repeat = Op(
    "pass.repeat",
    group=pass_ops,
    doc="Repeat a nested pipeline body.",
    traits=[UNKNOWN_EFFECTS, ImplicitTerminator("pass.yield")],
    attrs=[
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=PassRepeatMode),
        AttrDef("count", ATTR_TYPE_I64, optional=True),
        AttrDef("max_iterations", ATTR_TYPE_I64, optional=True),
    ],
    regions=[
        RegionDef(
            "body",
            doc="Body executed according to the repeat mode.",
            terminator="pass.yield",
        )
    ],
    format=[
        TemplateParam("mode"),
        AttrDict(),
        Region("body", syntax="pipeline"),
    ],
    examples=[
        "pass.repeat<fixed> {count = 2} pipeline {\n  canonicalize\n}",
        ("pass.repeat<until_converged> {max_iterations = 10} pipeline {\n  cse\n}"),
    ],
)

# ============================================================================
# pass.call
# ============================================================================

pass_call = Op(
    "pass.call",
    group=pass_ops,
    doc="Statically call another named pass pipeline.",
    traits=[UNKNOWN_EFFECTS],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("pass pipeline", ["record"]),
        ),
    ],
    format=[SymbolRef("callee")],
    examples=[
        "pass.call @cleanup",
    ],
)

# ============================================================================
# pass.run<key>
# ============================================================================

pass_run = Op(
    "pass.run",
    group=pass_ops,
    doc="Descriptor-backed leaf pass invocation.",
    traits=[UNKNOWN_EFFECTS],
    attrs=[
        AttrDef("key", ATTR_TYPE_STRING),
        AttrDef("options", "dict", optional=True),
    ],
    format=[
        OpRef("key"),
        AttrDict("options"),
    ],
    examples=[
        "pass.run<canonicalize>",
        "pass.run<canonicalize> {max_iterations = 10}",
        "pass.run<vector-memory-footprint> {budget_bytes = 4096}",
    ],
)

# ============================================================================
# pass.fail
# ============================================================================

pass_fail = Op(
    "pass.fail",
    group=pass_ops,
    doc="Emit a structured pipeline assertion failure.",
    traits=[UNKNOWN_EFFECTS],
    attrs=[
        AttrDef("message", ATTR_TYPE_STRING),
    ],
    format=[Attr("message")],
    examples=[
        'pass.fail "expected canonical form"',
    ],
)

# ============================================================================
# pass.halt
# ============================================================================

pass_halt = Op(
    "pass.halt",
    group=pass_ops,
    doc="Deliberately stop pipeline execution with a diagnostic message.",
    traits=[UNKNOWN_EFFECTS],
    attrs=[
        AttrDef("message", ATTR_TYPE_STRING),
    ],
    format=[Attr("message")],
    examples=[
        'pass.halt "inspect lowered IR"',
    ],
)

# ============================================================================
# pass.yield
# ============================================================================

pass_yield = Op(
    "pass.yield",
    group=pass_ops,
    doc="Terminate a pass pipeline control region.",
    traits=[TERMINATOR],
    examples=[
        "pass.yield",
    ],
)

ALL_PASS_OPS: tuple[Op, ...] = (
    pass_pipeline,
    pass_for,
    pass_where,
    pass_repeat,
    pass_call,
    pass_run,
    pass_fail,
    pass_halt,
    pass_yield,
)
