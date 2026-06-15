# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wasm target-family record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    SYMBOL_DEFINE,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    TargetLikeInterface,
)

wasm_ops = Dialect(
    "wasm",
    dialect_id=0x19,
    doc="Wasm target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/wasm/ops",
    register_by_default=False,
)

WasmTargetKind = EnumDef(
    "WasmTargetKind",
    [
        EnumCase("simd128", 1, doc="Wasm32 target row with SIMD128."),
    ],
    doc="Wasm target row selected by wasm.target.",
)

wasm_target = Op(
    "wasm.target",
    group=wasm_ops,
    doc=(
        "Wasm target-family record. The selector chooses an authored Wasm row; "
        "optional attrs structurally override common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_wasm_target_bundles",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(WasmTargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "wasm.target<simd128> @wasm",
    ],
)

ALL_WASM_OPS: tuple[Op, ...] = (wasm_target,)
