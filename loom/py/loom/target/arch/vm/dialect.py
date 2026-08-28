# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""VM target-family record dialect."""

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

vm_ops = Dialect(
    "vm",
    dialect_id=0x1A,
    doc="VM target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/vm/ops",
    register_by_default=False,
)

VmTargetKind = EnumDef(
    "VmTargetKind",
    [EnumCase("core", 1, doc="Core VM target row.")],
    doc="VM target row selected by vm.target.",
)

vm_target = Op(
    "vm.target",
    group=vm_ops,
    doc=(
        "VM target-family record. The selector chooses an authored VM row; "
        "optional attrs structurally override common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_vm_target_bundles",
            fact_type="loom_vm_target_fact_type",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(VmTargetKind),
    verify="loom_target_record_verify",
    format=[TemplateParam("kind"), SymbolRef("symbol"), AttrDict()],
    examples=["vm.target<core> @vm"],
)

ALL_VM_OPS: tuple[Op, ...] = (vm_target,)
