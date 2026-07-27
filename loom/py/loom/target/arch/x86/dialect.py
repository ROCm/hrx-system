# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 target-family record dialect."""

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

x86_ops = Dialect(
    "x86",
    dialect_id=0x18,
    doc="x86 target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/x86/ops",
    register_by_default=False,
)

X86TargetKind = EnumDef(
    "X86TargetKind",
    [
        EnumCase("avx512", 1, doc="x86-64 AVX-512 target row."),
        EnumCase("packed_dot", 2, doc="x86-64 packed-dot target row."),
        EnumCase(
            "avx512_packed_dot",
            3,
            doc="x86-64 AVX-512 target row with packed-dot features.",
        ),
        EnumCase("scalar", 4, doc="Baseline x86-64 scalar target row."),
        EnumCase("simd128", 5, doc="x86-64 SIMD128 target row."),
        EnumCase("avx2", 6, doc="x86-64 AVX2 target row."),
    ],
    doc="x86 target row selected by x86.target.",
)

x86_target = Op(
    "x86.target",
    group=x86_ops,
    doc=(
        "x86 target-family record. The selector chooses an authored CPU row; "
        "optional attrs structurally override common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_x86_target_bundles",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(X86TargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "x86.target<scalar> @host",
        "x86.target<simd128> @host",
        "x86.target<avx2> @host",
        "x86.target<avx512> @host",
        "x86.target<packed_dot> @host",
    ],
)

ALL_X86_OPS: tuple[Op, ...] = (x86_target,)
