# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Baseline x86-64 scalar/control descriptor rows."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    EnumDomain,
    EnumValue,
    Immediate,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    RegClass,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

from .common import (
    _ADDRESS_SCALE_ENUM,
    _ADDRESS_SCALE_IMMEDIATE,
    _CONTROL_EFFECT,
    _DISP32_IMMEDIATE,
    _GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
    _IMM32_IMMEDIATE,
    _IMM64_IMMEDIATE,
    _REG_GPR32,
    _REG_GPR64,
    _RESOURCE_ADDRESS,
    _RESOURCE_CONTROL,
    _RESOURCE_LOAD,
    _RESOURCE_SCALAR,
    _RESOURCE_STORE,
    _SCHEDULE_ADDRESS,
    _SCHEDULE_CONTROL,
    _SCHEDULE_MEMORY_LOAD_GPR32,
    _SCHEDULE_MEMORY_LOAD_GPR64,
    _SCHEDULE_MEMORY_STORE_GPR32,
    _SCHEDULE_MEMORY_STORE_GPR64,
    _SCHEDULE_SCALAR,
    _SHIFT32_IMMEDIATE,
    _SHIFT64_IMMEDIATE,
    _TARGET_BLOCK_IMMEDIATE,
    _asm,
    _gpr32_operand,
    _gpr32_result,
    _gpr64_operand,
    _gpr64_resource,
    _gpr64_result,
    _load_effect,
    _store_effect,
)


def _gpr32_destructive_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_destructive_binary_descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr32_result(),
        lhs=_gpr32_operand("lhs"),
        rhs=_gpr32_operand("rhs"),
        asm_suffix="gpr32",
    )


def _gpr64_destructive_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_destructive_binary_descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr64_result(),
        lhs=_gpr64_operand("lhs"),
        rhs=_gpr64_operand("rhs"),
        asm_suffix="gpr64",
    )


def _gpr_destructive_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    result: Operand,
    lhs: Operand,
    rhs: Operand,
    asm_suffix: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(result, lhs, rhs),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.{asm_suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr32_destructive_shift_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_destructive_shift_descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr32_result(),
        source=_gpr32_operand("lhs"),
        immediate=_SHIFT32_IMMEDIATE,
        asm_suffix="gpr32",
    )


def _gpr64_destructive_shift_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_destructive_shift_descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr64_result(),
        source=_gpr64_operand("lhs"),
        immediate=_SHIFT64_IMMEDIATE,
        asm_suffix="gpr64",
    )


def _gpr_destructive_shift_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    result: Operand,
    source: Operand,
    immediate: Immediate,
    asm_suffix: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(result, source),
        immediates=(immediate,),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.imm.{asm_suffix}",
            results=("dst",),
            operands=("lhs",),
            immediates=("shift",),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr32_to_gpr64_extend_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    asm_mnemonic: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(_gpr64_result(), _gpr32_operand("src")),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            results=("dst",),
            operands=("src",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr32_compare_descriptor(
    *,
    predicate: str,
    setcc: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_compare_descriptor(
        predicate=predicate,
        setcc=setcc,
        semantic_tag=semantic_tag,
        lhs=_gpr32_operand("lhs"),
        rhs=_gpr32_operand("rhs"),
        asm_suffix="gpr32",
    )


def _gpr64_compare_descriptor(
    *,
    predicate: str,
    setcc: str,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_compare_descriptor(
        predicate=predicate,
        setcc=setcc,
        semantic_tag=semantic_tag,
        lhs=_gpr64_operand("lhs"),
        rhs=_gpr64_operand("rhs"),
        asm_suffix="gpr64",
    )


def _gpr_compare_descriptor(
    *,
    predicate: str,
    setcc: str,
    semantic_tag: str,
    lhs: Operand,
    rhs: Operand,
    asm_suffix: str,
) -> Descriptor:
    return Descriptor(
        key=f"x86.scalar.cmp.{predicate}.{asm_suffix}",
        mnemonic=f"cmp.{setcc}",
        semantic_tag=semantic_tag,
        operands=(
            _gpr32_result(),
            lhs,
            rhs,
        ),
        asm_forms=_asm(
            mnemonic=f"cmp.{predicate}.{asm_suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_CMP_PREDICATE_SETCC = (
    ("eq", "sete"),
    ("ne", "setne"),
    ("slt", "setl"),
    ("sle", "setle"),
    ("sgt", "setg"),
    ("sge", "setge"),
    ("ult", "setb"),
    ("ule", "setbe"),
    ("ugt", "seta"),
    ("uge", "setae"),
)


X86_SCALAR_PREFIX_DESCRIPTORS = (
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.add.gpr32",
        mnemonic="add",
        semantic_tag="integer.add.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.add.gpr64",
        mnemonic="add",
        semantic_tag="integer.add.i64",
    ),
)

X86_SCALAR_SUFFIX_DESCRIPTORS = (
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.sub.gpr32",
        mnemonic="sub",
        semantic_tag="integer.sub.i32",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.imul.gpr32",
        mnemonic="imul",
        semantic_tag="integer.mul.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.sub.gpr64",
        mnemonic="sub",
        semantic_tag="integer.sub.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.imul.gpr64",
        mnemonic="imul",
        semantic_tag="integer.mul.i64",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.and.gpr32",
        mnemonic="and",
        semantic_tag="integer.and.i32",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.or.gpr32",
        mnemonic="or",
        semantic_tag="integer.or.i32",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.xor.gpr32",
        mnemonic="xor",
        semantic_tag="integer.xor.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.and.gpr64",
        mnemonic="and",
        semantic_tag="integer.and.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.or.gpr64",
        mnemonic="or",
        semantic_tag="integer.or.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.xor.gpr64",
        mnemonic="xor",
        semantic_tag="integer.xor.i64",
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.shl.imm.gpr32",
        mnemonic="shl",
        semantic_tag="integer.shl.i32",
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.sar.imm.gpr32",
        mnemonic="sar",
        semantic_tag="integer.shrs.i32",
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.shr.imm.gpr32",
        mnemonic="shr",
        semantic_tag="integer.shru.i32",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.shl.imm.gpr64",
        mnemonic="shl",
        semantic_tag="integer.shl.i64",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.sar.imm.gpr64",
        mnemonic="sar",
        semantic_tag="integer.shrs.i64",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.shr.imm.gpr64",
        mnemonic="shr",
        semantic_tag="integer.shru.i64",
    ),
    *(
        _gpr32_compare_descriptor(
            predicate=predicate,
            setcc=setcc,
            semantic_tag=f"integer.cmp.{predicate}.i32",
        )
        for predicate, setcc in _CMP_PREDICATE_SETCC
    ),
    *(
        _gpr64_compare_descriptor(
            predicate=predicate,
            setcc=setcc,
            semantic_tag=f"integer.cmp.{predicate}.i64",
        )
        for predicate, setcc in _CMP_PREDICATE_SETCC
    ),
    Descriptor(
        key="x86.scalar.movimm.gpr32",
        mnemonic="mov",
        semantic_tag="integer.const.i32",
        operands=(_gpr32_result(),),
        immediates=(_IMM32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.imm32",
            results=("dst",),
            immediates=("imm32",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.mov.load.gpr32",
        mnemonic="mov",
        semantic_tag="memory.load.i32",
        operands=(_gpr32_result(), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.load.gpr32",
            results=("dst",),
            operands=("base",),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_load_effect(32),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_GPR32,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.load.indexed.gpr32",
        mnemonic="mov",
        semantic_tag="memory.load.indexed.i32",
        operands=(
            _gpr32_result(),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="mov.load.indexed.gpr32",
            results=("dst",),
            operands=("base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_load_effect(32),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_GPR32,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.store.gpr32",
        mnemonic="mov",
        semantic_tag="memory.store.i32",
        operands=(_gpr32_operand("value"), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.store.gpr32",
            operands=("value", "base"),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_store_effect(32),),
        schedule_class=_SCHEDULE_MEMORY_STORE_GPR32,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.store.indexed.gpr32",
        mnemonic="mov",
        semantic_tag="memory.store.indexed.i32",
        operands=(
            _gpr32_operand("value"),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="mov.store.indexed.gpr32",
            operands=("value", "base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_store_effect(32),),
        schedule_class=_SCHEDULE_MEMORY_STORE_GPR32,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.load.gpr64",
        mnemonic="mov",
        semantic_tag="memory.load.i64",
        operands=(_gpr64_result(), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.load.gpr64",
            results=("dst",),
            operands=("base",),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_load_effect(64),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_GPR64,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.load.indexed.gpr64",
        mnemonic="mov",
        semantic_tag="memory.load.indexed.i64",
        operands=(
            _gpr64_result(),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="mov.load.indexed.gpr64",
            results=("dst",),
            operands=("base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_load_effect(64),),
        schedule_class=_SCHEDULE_MEMORY_LOAD_GPR64,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.store.gpr64",
        mnemonic="mov",
        semantic_tag="memory.store.i64",
        operands=(_gpr64_operand("value"), _gpr64_resource("base")),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.store.gpr64",
            operands=("value", "base"),
            immediates=("disp32",),
            named_immediates=True,
        ),
        effects=(_store_effect(64),),
        schedule_class=_SCHEDULE_MEMORY_STORE_GPR64,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.store.indexed.gpr64",
        mnemonic="mov",
        semantic_tag="memory.store.indexed.i64",
        operands=(
            _gpr64_operand("value"),
            _gpr64_resource("base"),
            _gpr64_resource("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="mov.store.indexed.gpr64",
            operands=("value", "base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        effects=(_store_effect(64),),
        schedule_class=_SCHEDULE_MEMORY_STORE_GPR64,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key="x86.scalar.mov.gpr64",
        mnemonic="mov",
        semantic_tag="integer.move.i64",
        operands=(_gpr64_result(), _gpr64_operand("src")),
        asm_forms=_asm(results=("dst",), operands=("src",)),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    _gpr32_to_gpr64_extend_descriptor(
        key="x86.scalar.movsxd.gpr64.gpr32",
        mnemonic="movsxd",
        semantic_tag="integer.extsi.i32.i64",
        asm_mnemonic="movsxd.gpr64.gpr32",
    ),
    _gpr32_to_gpr64_extend_descriptor(
        key="x86.scalar.movzx.gpr64.gpr32",
        mnemonic="movzx",
        semantic_tag="integer.extui.i32.i64",
        asm_mnemonic="movzx.gpr64.gpr32",
    ),
    Descriptor(
        key="x86.scalar.movimm.gpr64",
        mnemonic="mov",
        semantic_tag="integer.const.i64",
        operands=(_gpr64_result(),),
        immediates=(_IMM64_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="mov.imm64",
            results=("dst",),
            immediates=("imm64",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.lea.add.gpr64",
        mnemonic="lea",
        semantic_tag="integer.add.i64",
        operands=(
            _gpr64_result(),
            _gpr64_operand("lhs"),
            _gpr64_operand("rhs"),
        ),
        asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.lea.disp.gpr64",
        mnemonic="lea",
        semantic_tag="integer.add.disp.i64",
        operands=(
            _gpr64_result(),
            _gpr64_operand("base"),
        ),
        immediates=(_DISP32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="lea.disp",
            results=("dst",),
            operands=("base",),
            immediates=("disp32",),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.lea.scale.gpr64",
        mnemonic="lea",
        semantic_tag="integer.scale.disp.i64",
        operands=(
            _gpr64_result(),
            _gpr64_operand("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="lea.scale",
            results=("dst",),
            operands=("index",),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.lea.add_scale.gpr64",
        mnemonic="lea",
        semantic_tag="integer.add.scale.disp.i64",
        operands=(
            _gpr64_result(),
            _gpr64_operand("base"),
            _gpr64_operand("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="lea.add_scale",
            results=("dst",),
            operands=("base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="x86.scalar.jmp",
        mnemonic="jmp",
        semantic_tag="control.branch",
        operands=(),
        immediates=(_TARGET_BLOCK_IMMEDIATE,),
        asm_forms=_asm(immediates=("target_block",)),
        effects=(_CONTROL_EFFECT,),
        schedule_class=_SCHEDULE_CONTROL,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
    ),
)

X86_SCALAR_DESCRIPTORS = (
    *X86_SCALAR_PREFIX_DESCRIPTORS,
    *X86_SCALAR_SUFFIX_DESCRIPTORS,
)

X86_SCALAR_DESCRIPTOR_SET = DescriptorSet(
    key="x86.scalar.core",
    target_key="x86",
    feature_key="x86.scalar.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/scalar_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/scalar_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_SCALAR_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/descriptors/scalar_descriptors.h",
    function_name="loom_x86_scalar_core_descriptor_set",
    c_table_prefix="X86ScalarCore",
    c_enum_prefix="X86_SCALAR_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_GPR32,
            32,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=1,
        ),
        RegClass(
            _REG_GPR64,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=1,
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(
            _RESOURCE_ADDRESS,
            capacity_per_cycle=1,
            kind=ResourceKind.ADDRESS,
            contention_group_id=1,
        ),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_SCALAR,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_ADDRESS,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_GPR32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_GPR64,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_GPR32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_GPR64,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_CONTROL,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    enum_domains=(
        EnumDomain(
            _ADDRESS_SCALE_ENUM,
            (
                EnumValue("1", 1),
                EnumValue("2", 2),
                EnumValue("4", 4),
                EnumValue("8", 8),
            ),
        ),
    ),
    descriptors=X86_SCALAR_DESCRIPTORS,
)
