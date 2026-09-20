# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Memory effects and full-width x86 load/store descriptor families."""

from __future__ import annotations

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    Effect,
    EffectFlag,
    EffectKind,
    MemorySpace,
    Operand,
    OperandRole,
    RegClassAlt,
)

from .common import (
    _ADDRESS_SCALE_IMMEDIATE,
    _DISP32_IMMEDIATE,
    _asm,
    _gpr64_resource,
)


def _load_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _store_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.WRITE,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def memory_descriptors(
    *,
    key_prefix: str,
    mnemonic: str,
    register_class: str,
    register_suffix: str,
    semantic_type: str,
    width_bits: int,
    load_schedule_class: str,
    store_schedule_class: str,
    assembly_suffix: str,
    native_assembly_mnemonic: str | None = None,
) -> tuple[Descriptor, ...]:
    """Declares static/indexed loads followed by static/indexed stores."""
    register_alternatives = (RegClassAlt(register_class),)
    descriptors = []
    for operation, value, effect, schedule_class in (
        (
            "load",
            Operand("dst", OperandRole.RESULT, register_alternatives),
            _load_effect(width_bits),
            load_schedule_class,
        ),
        (
            "store",
            Operand("value", OperandRole.OPERAND, register_alternatives),
            _store_effect(width_bits),
            store_schedule_class,
        ),
    ):
        results = (value.field_name,) if value.role == OperandRole.RESULT else ()
        value_fields = () if results else (value.field_name,)
        for address_suffix, address_fields, immediates in (
            ("", ("base",), (_DISP32_IMMEDIATE,)),
            (
                ".indexed",
                ("base", "index"),
                (_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            ),
        ):
            operation_name = f"{operation}{address_suffix}"
            descriptors.append(
                Descriptor(
                    key=f"{key_prefix}.{mnemonic}.{operation_name}.{register_suffix}",
                    mnemonic=mnemonic,
                    semantic_tag=f"memory.{operation_name}.{semantic_type}",
                    operands=(
                        value,
                        *(_gpr64_resource(field_name) for field_name in address_fields),
                    ),
                    immediates=immediates,
                    asm_forms=_asm(
                        mnemonic=f"{mnemonic}.{operation_name}{assembly_suffix}",
                        native_assembly_mnemonic=native_assembly_mnemonic,
                        results=results,
                        operands=(*value_fields, *address_fields),
                        immediates=tuple(item.field_name for item in immediates),
                        named_immediates=True,
                    ),
                    effects=(effect,),
                    schedule_class=schedule_class,
                    flags=(DescriptorFlag.SIDE_EFFECTING,),
                )
            )
    return tuple(descriptors)
