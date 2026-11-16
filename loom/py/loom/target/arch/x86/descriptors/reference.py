# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native typed-cell operations shared by every x86 register profile."""

from dataclasses import replace

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateKind,
    MemorySpace,
    Operand,
    OperandRole,
    RegClassAlt,
)

from .common import _REG_GPR32, _REG_GPR64, _SCHEDULE_SCALAR, _asm
from .encoding import Form
from .memory import memory_descriptors

X86_REFERENCE_DESCRIPTORS = tuple(
    Descriptor(
        key=f"x86.ref.{operation}",
        encoding_format_id=form,
        encoding_id=1,
        mnemonic=f"ref.{operation}",
        semantic_tag=f"reference.{operation}",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt(result_class),)),
            *(
                Operand(name, OperandRole.OPERAND, (RegClassAlt(register_class),))
                for name, register_class in inputs
            ),
        ),
        asm_forms=_asm(results=("dst",), operands=tuple(name for name, _ in inputs)),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    for operation, form, result_class, inputs in (
        ("null", Form.REFERENCE_NULL, "x86.ref", ()),
        (
            "compare.null",
            Form.REFERENCE_COMPARE_NULL,
            _REG_GPR32,
            (("value", "x86.ref"),),
        ),
        (
            "compare.eq",
            Form.REFERENCE_COMPARE_EQ,
            _REG_GPR32,
            (("left", "x86.ref"), ("right", "x86.ref")),
        ),
        (
            "select",
            Form.REFERENCE_SELECT,
            "x86.ref",
            (
                ("condition", _REG_GPR32),
                ("true_value", "x86.ref"),
                ("false_value", "x86.ref"),
            ),
        ),
    )
)

X86_REFERENCE_MEMORY_DESCRIPTORS = (
    *(
        replace(
            descriptor,
            key=descriptor.key.removesuffix(".ref"),
            encoding_format_id=(
                Form.REFERENCE_LOAD
                if descriptor.encoding_format_id is Form.LOAD
                else Form.REFERENCE_STORE
            ),
        )
        for descriptor in memory_descriptors(
            key_prefix="x86",
            load_mnemonic="ref",
            store_mnemonic="ref",
            register_class="x86.ref",
            register_suffix="ref",
            semantic_type="ref",
            width_bits=128,
            load_schedule_class=_SCHEDULE_SCALAR,
            store_schedule_class=_SCHEDULE_SCALAR,
            assembly_suffix="",
            encoding_ids=(1, 1),
        )
    ),
    Descriptor(
        key="x86.ref.discard.range",
        mnemonic="ref.discard.range",
        encoding_format_id=Form.REFERENCE_DISCARD_RANGE,
        encoding_id=1,
        semantic_tag="reference.discard.range",
        operands=(Operand("base", OperandRole.RESOURCE, (RegClassAlt(_REG_GPR64),)),),
        immediates=(
            Immediate(
                "count", ImmediateKind.UNSIGNED, bit_width=32, unsigned_max=2**32 - 1
            ),
        ),
        asm_forms=_asm(
            operands=("base",), immediates=("count",), named_immediates=True
        ),
        effects=(
            Effect(
                EffectKind.WRITE,
                memory_space=MemorySpace.GENERIC,
                flags=(EffectFlag.DEPENDENCY,),
            ),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
)
