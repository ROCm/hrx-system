# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declarations of opaque runtime-provided payload identities."""

from loom.assembly import COMMA, Attr, Clause, SymbolRef
from loom.dsl import (
    MODULE_SCOPE,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    Op,
    OpPhase,
    SymbolDefinition,
    SymbolTypeContract,
)

type_ops = Dialect(
    "type",
    dialect_id=0x25,
    doc="Opaque payload identity declarations.",
    default_phase=OpPhase.MODULE_METADATA,
)

type_decl = Op(
    "type.decl",
    group=type_ops,
    doc=(
        "Declare an external managed payload identity under a local name. "
        "The namespace and payload name identify the type independently of "
        "its local spelling. Providers supply payload behavior and ownership; "
        "the declaration exposes no layout, address, or implementation."
    ),
    traits=[SYMBOL_DEFINE, MODULE_SCOPE],
    attrs=[
        AttrDef("name", "symbol", doc="Module-local payload type name."),
        AttrDef("namespace", "string", doc="External provider namespace."),
        AttrDef("type_name", "string", doc="Exact payload name in the provider namespace."),
    ],
    symbol_def=SymbolDefinition(
        field="name",
        name="managed type",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        type_contract=SymbolTypeContract(namespace="namespace", name="type_name"),
    ),
    verify="loom_type_decl_verify",
    format=[
        Clause("import", Attr("namespace"), COMMA, Attr("type_name")),
        SymbolRef("name"),
    ],
    examples=['type.decl import("input", "decoder") @Decoder'],
)

ALL_TYPE_OPS: tuple[Op, ...] = (type_decl,)
