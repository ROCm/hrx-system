# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Global dialect op definitions.

Four ops for module-level state:

Top-level (module-level symbols):
  global.constant  — Immutable global (weights, parameters, constants).
  global.variable  — Mutable global (KV cache, running state).
  global.rodata    — Read-only executable data payload.

Body ops (inside function/template bodies):
  global.load      — Load value + dynamic dims/encoding from global.
  global.store     — Store value + dynamic dims/encoding to global.

Global definitions are module-private today. `global.rodata` is executable
data, not a value global: it gives target emitters named bytes they can place
in read-only artifact sections without making the payload loadable through
`global.load`.
"""

from loom.assembly import (
    COLON,
    COMMA,
    EQUALS,
    Attr,
    OptionalGroup,
    PredicateList,
    Ref,
    ResultType,
    Scope,
    SymbolRef,
    TypeOf,
    kw,
)
from loom.dsl import (
    ANY,
    SYMBOL_DEFINE,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    Op,
    Operand,
    Result,
    SymbolDefinition,
    SymbolReference,
)

# ============================================================================
# Op group
# ============================================================================

global_ops = Dialect(
    "global",
    dialect_id=0x0B,
    doc="Module-level state: constants, parameters, mutable state.",
)

# ============================================================================
# global.constant — immutable global value
# ============================================================================

global_constant = Op(
    "global.constant",
    group=global_ops,
    doc=(
        "Immutable global value with an optional inline scalar initializer. "
        "Declaration-local dim/encoding names in the type annotation express "
        "structural constraints. Predicates constrain dynamic dimensions and "
        "are propagated to every load site as value facts. Non-scalar or "
        "computed initialization is modeled by global.store in initializer "
        "functions; resource-backed artifact payloads belong in global.rodata "
        "instead of overloading inline attrs."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="global",
        interfaces=["global"],
        bytecode_kind="LOOM_SYMBOL_GLOBAL",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("predicates", "predicate_list", optional=True),
        AttrDef("initializer", "any", optional=True),
    ],
    # Single result carries the global's type. SYMBOL_DEFINE suppresses
    # the %name = prefix, so no SSA value is printed — just the type
    # after the colon.
    results=[Result("type", ANY)],
    format=[
        SymbolRef("symbol"),
        COLON,
        Scope(
            [
                ResultType("type"),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
        OptionalGroup(
            [EQUALS, Attr("initializer")],
            anchor="initializer",
        ),
    ],
    verify="loom_global_constant_verify",
    examples=[
        "global.constant @pi : f32 = 3.14159265358979",
        "global.constant @weights : tile<[%m]x[%k]xf32> where [mul(%m, 16)]",
    ],
)

# ============================================================================
# global.variable — mutable global value
# ============================================================================

global_variable = Op(
    "global.variable",
    group=global_ops,
    doc=(
        "Mutable global value with an optional inline scalar default "
        "initializer. Can be stored from any function at any time. "
        "Declaration-local dim/encoding names and predicates work the same as "
        "global.constant."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="global",
        interfaces=["global"],
        bytecode_kind="LOOM_SYMBOL_GLOBAL",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("predicates", "predicate_list", optional=True),
        AttrDef("initializer", "any", optional=True),
    ],
    results=[Result("type", ANY)],
    format=[
        SymbolRef("symbol"),
        COLON,
        Scope(
            [
                ResultType("type"),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
        OptionalGroup(
            [EQUALS, Attr("initializer")],
            anchor="initializer",
        ),
    ],
    verify="loom_global_variable_verify",
    examples=[
        "global.variable @kv_cache : tile<[%s]x[%d]xf32> where [mul(%s, 64)]",
        "global.variable @step_count : index = 0",
    ],
)

# ============================================================================
# global.rodata — read-only executable data payload
# ============================================================================

global_rodata = Op(
    "global.rodata",
    group=global_ops,
    doc=(
        "Read-only executable data payload. This defines a named artifact "
        "symbol containing uninterpreted bytes, optionally with a stronger "
        "power-of-two byte alignment requirement. It is used for compiler-owned "
        "tables and metadata such as sanitizer site records; user-visible value "
        "globals remain global.constant/global.variable."
    ),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="rodata",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("contents", "bytes"),
        AttrDef("alignment", "i64", optional=True),
    ],
    format=[
        SymbolRef("symbol"),
        EQUALS,
        Attr("contents"),
        OptionalGroup([COMMA, kw("align"), Attr("alignment")], anchor="alignment"),
    ],
    verify="loom_global_rodata_verify",
    examples=[
        'global.rodata @loom_sanitizer_sites = bytes("4c53495401000000"), align 8',
    ],
)

# ============================================================================
# global.load — load value + dynamic dims/encoding from global
# ============================================================================

global_load = Op(
    "global.load",
    group=global_ops,
    doc=("Load a value from a global. Dynamic dims and encodings in the type annotation reference co-results by name. Predicates on the global definition are propagated as value facts."),
    attrs=[
        AttrDef(
            "global",
            "symbol",
            symbol_ref=SymbolReference("global", ["global"]),
        ),
    ],
    # Variadic results: the loaded value + any dim/encoding co-results.
    # The result count is determined by the LHS result names.
    results=[Result("result", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    canonicalize="loom_global_load_canonicalize",
    facts="loom_global_load_facts",
    verify="loom_global_load_verify",
    format=[
        SymbolRef("global"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%tile, %m, %k = global.load @weights : tile<[%m]x[%k]xf32>",
        "%tile = global.load @bias : tile<[%m]xf32>",
        "%cache, %s, %d = global.load @kv_cache : tile<[%s]x[%s]x[%d]xf32>",
    ],
)

# ============================================================================
# global.store — store value + dynamic dims/encoding to global
# ============================================================================

global_store = Op(
    "global.store",
    group=global_ops,
    doc=(
        "Store a value to a global. Dynamic dims and encodings are captured "
        "implicitly from the type annotation, which references existing SSA "
        "values. The verifier checks that stores to global.constant only "
        "happen from initializer-reachable code paths."
    ),
    operands=[
        Operand("value", ANY, doc="The value to store."),
    ],
    attrs=[
        AttrDef(
            "global",
            "symbol",
            symbol_ref=SymbolReference("global", ["global"]),
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_global_store_verify",
    format=[
        Ref("value"),
        COMMA,
        SymbolRef("global"),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        "global.store %tile, @kv_cache : tile<[%m]xf32>",
        "global.store %table, @lut : tile<16x32xf32>",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_GLOBAL_OPS: tuple[Op, ...] = (
    global_constant,
    global_variable,
    global_rodata,
    global_load,
    global_store,
)
