# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Memory effects and exact-width x86 load/store descriptor families."""

from __future__ import annotations

from collections.abc import Callable

from loom.target.arch.x86.vector_encoding import VectorMachineInstruction
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
    _GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
    _asm,
    _gpr64_resource,
)
from .encoding import Flag, Form, encoding

_MEMORY_ADDRESS_FORMS = (
    ("", ("base",), (_DISP32_IMMEDIATE,)),
    (
        ".indexed",
        ("base", "index"),
        (_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
    ),
)
_NO_ENCODING_FLAGS = Flag(0)


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
    load_mnemonic: str,
    store_mnemonic: str,
    register_class: str,
    register_suffix: str,
    semantic_type: str,
    width_bits: int,
    load_schedule_class: str,
    store_schedule_class: str,
    assembly_suffix: str,
    encoding_ids: tuple[int, int] | None = None,
    vector_instructions: tuple[VectorMachineInstruction, ...] | None = None,
    vector_binder: Callable[[Descriptor, VectorMachineInstruction], Descriptor]
    | None = None,
) -> tuple[Descriptor, ...]:
    """Declares static/indexed loads followed by static/indexed stores."""
    register_alternatives = (RegClassAlt(register_class),)
    descriptors = []
    for operation, mnemonic, value, effect, schedule_class in (
        (
            "load",
            load_mnemonic,
            Operand("dst", OperandRole.RESULT, register_alternatives),
            _load_effect(width_bits),
            load_schedule_class,
        ),
        (
            "store",
            store_mnemonic,
            Operand("value", OperandRole.OPERAND, register_alternatives),
            _store_effect(width_bits),
            store_schedule_class,
        ),
    ):
        results = (value.field_name,) if value.role == OperandRole.RESULT else ()
        value_fields = () if results else (value.field_name,)
        for address_suffix, address_fields, immediates in _MEMORY_ADDRESS_FORMS:
            operation_name = f"{operation}{address_suffix}"
            descriptors.append(
                Descriptor(
                    key=f"{key_prefix}.{mnemonic}.{operation_name}.{register_suffix}",
                    mnemonic=mnemonic,
                    encoding_format_id=(
                        Form.LOAD if operation == "load" else Form.STORE
                    )
                    if encoding_ids
                    else 0,
                    encoding_id=(
                        encoding_ids[0 if operation == "load" else 1]
                        | (Flag.INDEXED if address_suffix else 0)
                    )
                    if encoding_ids
                    else 0,
                    semantic_tag=f"memory.{operation_name}.{semantic_type}",
                    operands=(
                        value,
                        *(_gpr64_resource(field_name) for field_name in address_fields),
                    ),
                    immediates=immediates,
                    asm_forms=_asm(
                        mnemonic=f"{mnemonic}.{operation_name}{assembly_suffix}",
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
    if vector_instructions is None:
        if vector_binder is not None:
            raise ValueError("vector binder requires native instruction facts")
        return tuple(descriptors)
    if vector_binder is None or len(vector_instructions) != len(descriptors):
        raise ValueError("vector memory descriptors require complete native facts")
    return tuple(
        vector_binder(descriptor, instruction)
        for descriptor, instruction in zip(
            descriptors, vector_instructions, strict=True
        )
    )


def read_memory_binary_descriptors(
    *,
    key_prefix: str,
    mnemonic: str,
    register_class: str,
    register_suffix: str,
    semantic_tag: str,
    width_bits: int,
    schedule_class: str,
    opcode: int,
    encoding_flags: Flag = _NO_ENCODING_FLAGS,
) -> tuple[Descriptor, ...]:
    """Declares destructive register/read-memory scalar binary forms."""

    register_alternatives = (RegClassAlt(register_class),)
    descriptors = []
    for address_suffix, address_fields, immediates in _MEMORY_ADDRESS_FORMS:
        descriptors.append(
            Descriptor(
                key=f"{key_prefix}.load{address_suffix}.{register_suffix}",
                mnemonic=mnemonic,
                encoding_format_id=Form.BINARY_R_MEMORY,
                encoding_id=encoding(
                    opcode,
                    flags=encoding_flags | (Flag.INDEXED if address_suffix else 0),
                ),
                semantic_tag=semantic_tag,
                operands=(
                    Operand("dst", OperandRole.RESULT, register_alternatives),
                    Operand("lhs", OperandRole.OPERAND, register_alternatives),
                    *(_gpr64_resource(field_name) for field_name in address_fields),
                ),
                immediates=immediates,
                constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
                asm_forms=_asm(
                    mnemonic=f"{mnemonic}.load{address_suffix}.{register_suffix}",
                    results=("dst",),
                    operands=("lhs", *address_fields),
                    immediates=tuple(item.field_name for item in immediates),
                    named_immediates=True,
                ),
                effects=(_load_effect(width_bits),),
                schedule_class=schedule_class,
                flags=(DescriptorFlag.SIDE_EFFECTING,),
            )
        )
    return tuple(descriptors)
