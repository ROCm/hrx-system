# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Config dialect op definitions.

Config symbols represent compile/link-time constants that are expected to fold
out of executable IR. Libraries can declare required config values with
config.decl, config files or driver inputs provide config.def, and bodies read
values through config.get so dependency analysis can see the symbol edge.
"""

from loom.assembly import (
    COLON,
    EQUALS,
    Attr,
    OptionalGroup,
    PredicateList,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    kw,
)
from loom.dsl import (
    ANY,
    PURE,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    Op,
    Result,
    SymbolDefinition,
    SymbolReference,
)

# ============================================================================
# Op group
# ============================================================================

config_ops = Dialect(
    "config",
    dialect_id=0x1C,
    doc="Compile/link-time configuration symbols and reads.",
)

_CONFIG_SYMBOL_DEF = SymbolDefinition(
    field="symbol",
    name="config",
    interfaces=["config"],
    bytecode_kind="LOOM_SYMBOL_GLOBAL",
)

_CONFIG_SYMBOL_REF = SymbolReference("config", ["config"])

# ============================================================================
# config.decl - required external config value
# ============================================================================

config_decl = Op(
    "config.decl",
    group=config_ops,
    doc=(
        "Declare a required compile/link-time configuration value. The op "
        "defines a symbol and a required type but intentionally carries no "
        "initializer. Predicates constrain the unresolved value until a "
        "config.def supplies an exact value; final executable compilation "
        "must resolve reachable config.get users to exactly one config.def."
    ),
    traits=[SYMBOL_DEFINE, PURE],
    symbol_def=_CONFIG_SYMBOL_DEF,
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("predicates", "predicate_list", optional=True),
    ],
    results=[Result("type", ANY)],
    verify="loom_config_decl_verify",
    format=[
        SymbolRef("symbol"),
        COLON,
        Scope(
            [
                ResultTypeList("type", parens=False),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
    ],
    examples=[
        "config.decl @model36.model.hidden_size : index",
        "config.decl @model36.model.max_context : %value: index where [range(%value, 1, 131072)]",
        "config.decl @model36.features.enable_mtp : i1",
        "config.decl @model36.quant.dense_encoding : encoding<schema>",
    ],
)

# ============================================================================
# config.def - provided config value
# ============================================================================

config_def = Op(
    "config.def",
    group=config_ops,
    doc=(
        "Define a compile/link-time configuration value. The initializer is "
        "required and must match the declared result type. Scalar values seed "
        "ordinary value facts so config.get can fold through canonicalization."
    ),
    traits=[SYMBOL_DEFINE, PURE],
    symbol_def=_CONFIG_SYMBOL_DEF,
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("value", "any"),
    ],
    results=[Result("type", ANY)],
    verify="loom_config_def_verify",
    facts="loom_config_def_facts",
    format=[
        SymbolRef("symbol"),
        EQUALS,
        Attr("value"),
        COLON,
        Scope([ResultType("type")]),
    ],
    examples=[
        "config.def @model36.model.hidden_size = 2048 : index",
        "config.def @model36.features.enable_mtp = true : i1",
        "config.def @model36.quant.dense_encoding = #q8_0<block=32> : encoding<schema>",
    ],
)

# ============================================================================
# config.get - read a config value
# ============================================================================

config_get = Op(
    "config.get",
    group=config_ops,
    doc=(
        "Read a compile/link-time configuration value. The symbol reference is "
        "a generated symbol-ref attr so dependency analysis and bytecode index "
        "metadata can see config sensitivity through the normal symbol path."
    ),
    attrs=[
        AttrDef(
            "config",
            "symbol",
            symbol_ref=_CONFIG_SYMBOL_REF,
        ),
    ],
    results=[Result("result", ANY)],
    traits=[PURE],
    facts="loom_config_get_facts",
    verify="loom_config_get_verify",
    format=[
        SymbolRef("config"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%hidden = config.get @model36.model.hidden_size : index",
        "%enabled = config.get @model36.features.enable_mtp : i1",
        "%dense_encoding = config.get @model36.quant.dense_encoding : encoding<schema>",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_CONFIG_OPS: tuple[Op, ...] = (
    config_decl,
    config_def,
    config_get,
)
