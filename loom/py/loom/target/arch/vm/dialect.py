# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""VM target-family records and physical module resources."""

from loom.assembly import Attr, AttrDict, Clause, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    MODULE_SCOPE,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    SymbolReference,
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

vm_global = Op(
    "vm.global",
    group=vm_ops,
    doc=(
        "Assign a generic value global its physical Core VM bank ordinal. "
        "The referenced definition determines the value, ref, or function "
        "bank and whether the ordinal belongs to its immutable prefix or "
        "mutable suffix."
    ),
    traits=[MODULE_SCOPE],
    attrs=[
        AttrDef(
            "source",
            "symbol",
            symbol_ref=SymbolReference("global", ["global"]),
        ),
        AttrDef("ordinal", "i64"),
    ],
    verify="loom_vm_global_verify",
    format=[SymbolRef("source"), Clause("ordinal", Attr("ordinal"))],
    examples=["vm.global @state ordinal(3)"],
)

vm_rodata = Op(
    "vm.rodata",
    group=vm_ops,
    doc=(
        "Assign a generic read-only data definition its physical Core VM "
        "rodata ordinal. The payload remains owned by the referenced source "
        "symbol and is not copied into this record."
    ),
    traits=[MODULE_SCOPE],
    attrs=[
        AttrDef(
            "source",
            "symbol",
            symbol_ref=SymbolReference("read-only data", ["rodata"]),
        ),
        AttrDef("ordinal", "i64"),
    ],
    verify="loom_vm_rodata_verify",
    format=[SymbolRef("source"), Clause("ordinal", Attr("ordinal"))],
    examples=["vm.rodata @lookup_table ordinal(0)"],
)

ALL_VM_OPS: tuple[Op, ...] = (vm_target, vm_global, vm_rodata)
