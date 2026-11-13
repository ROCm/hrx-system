# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical AIE configuration and invocation commands for complete workers."""

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)

_KEY = "amd.xdna.aie2p.configuration"
_SCALAR = "aie2p.config.scalar"
_BINDING = "aie2p.config.binding"
_SPAN = "aie2p.config.span"
_SCHEDULE = f"{_KEY}.schedule"
_CONTROL = f"{_KEY}.control"


def _result(kind: str) -> Operand:
    return Operand("result", OperandRole.RESULT, (RegClassAlt(kind),))


def _operand(name: str, kind: str = _SCALAR, *, variadic: bool = False) -> Operand:
    return Operand(
        name,
        OperandRole.OPERAND,
        (RegClassAlt(kind),),
        flags=(OperandFlag.VARIADIC,) if variadic else (),
    )


def _symbol(name: str) -> Immediate:
    return Immediate(
        name,
        ImmediateKind.ORDINAL,
        bit_width=32,
        unsigned_max=2**32 - 1,
        flags=(ImmediateFlag.SYMBOLIC,),
    )


def _instruction(
    name: str,
    operands: tuple[Operand, ...] = (),
    immediates: tuple[Immediate, ...] = (),
    *,
    pure: bool = False,
    form: AsmForm | None = None,
) -> Descriptor:
    return Descriptor(
        key=f"{_KEY}.{name}",
        mnemonic=name,
        semantic_tag=f"config.{name}",
        operands=operands,
        immediates=immediates,
        asm_forms=(
            form
            or AsmForm(
                mnemonic=name,
                results=tuple(
                    value.field_name
                    for value in operands
                    if value.role == OperandRole.RESULT
                ),
                operands=tuple(
                    value.field_name
                    for value in operands
                    if value.role == OperandRole.OPERAND
                ),
                immediates=tuple(
                    AsmImmediate(value.field_name) for value in immediates
                ),
            ),
        ),
        effects=() if pure else (Effect(EffectKind.CALL),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,)
        if pure
        else (DescriptorFlag.SIDE_EFFECTING,),
        schedule_class=_SCHEDULE,
        op_kind=DescriptorOpKind.CONST if name == "constant" else DescriptorOpKind.OP,
    )


_DESCRIPTORS = (
    _instruction("measure.begin"),
    _instruction("measure.end"),
    _instruction(
        "constant",
        (_result(_SCALAR),),
        (
            Immediate(
                "value", ImmediateKind.UNSIGNED, bit_width=64, unsigned_max=2**63 - 1
            ),
        ),
        pure=True,
    ),
    # The binding declaration covers all payload and auxiliary service access.
    # A command's touched addresses cannot establish this external contract.
    _instruction(
        "binding",
        (
            _result(_BINDING),
            *(_operand(name) for name in ("ordinal", "access", "length", "alignment")),
        ),
    ),
    # Preserve an unused public argument ordinal without a memory requirement
    # or a binding value that could be used to form a relocated address.
    _instruction("binding.unused", (_operand("ordinal"),)),
    _instruction(
        "range",
        (
            _result(_SPAN),
            _operand("binding", _BINDING),
            _operand("offset"),
            _operand("length"),
        ),
        pure=True,
    ),
    _instruction(
        "entry", (_operand("columns"),), (_symbol("initialize"), _symbol("invoke"))
    ),
    _instruction("write32", (_operand("address"), _operand("value"))),
    _instruction(
        "write.mask32", (_operand("address"), _operand("mask"), _operand("value"))
    ),
    _instruction(
        "wait.mask32", (_operand("address"), _operand("mask"), _operand("value"))
    ),
    _instruction(
        "write.block32",
        (_operand("address"), _operand("words", variadic=True)),
        form=AsmForm(
            mnemonic="write.block32",
            operand_segments=(
                AsmOperandSegment(AsmOperandSegmentDelimiter.PAREN, ("address",)),
                AsmOperandSegment(AsmOperandSegmentDelimiter.SQUARE, ("words",)),
            ),
        ),
    ),
    _instruction(
        "write.address",
        (_operand("address"), _operand("buffer", _SPAN), _operand("flags")),
    ),
    _instruction(
        "shim.descriptor",
        (
            _operand("address"),
            _operand("buffer", _SPAN),
            *(
                _operand(name)
                for name in (
                    "length",
                    "flags",
                    "dimension0",
                    "dimension1",
                    "dimension2",
                    "iteration",
                    "control",
                )
            ),
        ),
    ),
    # The referenced core already contains its full control and channel protocol.
    # Loading it introduces no record loop or implicit acquire/release operations.
    _instruction(
        "program.load", (_operand("column"), _operand("row")), (_symbol("program"),)
    ),
    # These bytes belong to resident data and communication, independently of
    # any loaded program's private storage. Repeated ranges name their union.
    _instruction(
        "data.reserve",
        tuple(_operand(name) for name in ("column", "row", "offset", "length")),
    ),
    _instruction(
        "dma.wait",
        tuple(
            _operand(name)
            for name in ("column", "row", "direction", "channel", "columns", "rows")
        ),
    ),
)

AIE2P_CONFIGURATION_DESCRIPTOR_SET = DescriptorSet(
    key=_KEY,
    inline_schedule_model_key=_KEY,
    target_key="amd.xdna.aie2p",
    feature_key=f"{_KEY}.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.h"
    ),
    c_source_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.c"
    ),
    header_guard="LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_CONFIGURATION_DESCRIPTORS_H_",
    public_header="loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.h",
    function_name="loom_aie2p_configuration_descriptor_set",
    c_table_prefix="Aie2pConfiguration",
    c_enum_prefix="AIE2P_CONFIGURATION",
    generator_version=1,
    reg_classes=tuple(
        RegClass(
            name,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(
                RegClassFlag.VIRTUAL_ONLY,
                RegClassFlag.REFERENCE,
                RegClassFlag.UNSPILLABLE,
            ),
            target_bank_id=1,
            alias_set_id=1,
        )
        for name in (_SCALAR, _BINDING, _SPAN)
    ),
    resources=(Resource(_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_CONTROL, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    descriptors=_DESCRIPTORS,
    requires_explicit_asm_surface=True,
)
