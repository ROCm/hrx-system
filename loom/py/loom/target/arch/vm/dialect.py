# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""VM target selection; instruction records remain in the runtime ISA spec."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam, TypeOf, TypesOf
from loom.dsl import (
    ATTR_TYPE_ENUM,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    TargetFactSpecialization,
    TargetLikeInterface,
    TypeDef,
    TypeParam,
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
    [EnumCase("core", 1, doc="Portable Core VM instruction set.")],
    doc="Instruction set selected by vm.target.",
)

vm_target = Op(
    "vm.target",
    group=vm_ops,
    doc="Selects the portable VM instruction set and host function ABI.",
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_vm_target_bundles",
            fact_specialization=TargetFactSpecialization.STRUCTURAL,
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=VmTargetKind),
    ],
    verify="loom_target_record_verify",
    format=[TemplateParam("kind"), SymbolRef("symbol"), AttrDict()],
    examples=["vm.target<core> @vm"],
)

ALL_VM_OPS: tuple[Op, ...] = (vm_target,)

vm_record = TypeDef(
    "vm.record",
    params=[TypeParam("fields", variadic=True)],
    format=[TypesOf("fields")],
    doc=(
        "Ordered native ABI record fields. Scalars and managed references are "
        "leaves; vm.range contains repeated records and vm.out marks result leaves."
    ),
)
vm_range = TypeDef(
    "vm.range",
    params=[TypeParam("fields", variadic=True)],
    format=[TypesOf("fields")],
    doc=(
        "Statically grouped native ABI range with the given ordered element fields. "
        "Its layout is independent of callsite cardinality, including empty ranges."
    ),
)
vm_out = TypeDef(
    "vm.out",
    params=[TypeParam("value")],
    format=[TypeOf("value")],
    doc=(
        "A native ABI result scalar or managed-reference leaf, published after "
        "all input use."
    ),
)
ALL_VM_TYPES: tuple[TypeDef, ...] = (vm_record, vm_range, vm_out)
