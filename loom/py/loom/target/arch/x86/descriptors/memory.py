# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Memory effects and exact-width x86 load/store descriptor families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.vector_encoding import VectorMachineInstruction
from loom.target.arch.x86.vector_families import X86_VECTOR_STORAGE_FAMILIES
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
    _SCHEDULE_MEMORY_LOAD_XMM,
    _SCHEDULE_MEMORY_LOAD_YMM,
    _SCHEDULE_MEMORY_LOAD_ZMM,
    _SCHEDULE_MEMORY_STORE_XMM,
    _SCHEDULE_MEMORY_STORE_YMM,
    _SCHEDULE_MEMORY_STORE_ZMM,
    _asm,
    _avx512_descriptor,
    _gpr64_resource,
    _vex_descriptor,
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


def _memory_transfer_descriptors(
    *,
    key_prefix: str,
    operation: str,
    mnemonic: str,
    register_class: str,
    register_suffix: str,
    semantic_type: str,
    width_bits: int,
    schedule_class: str,
    assembly_suffix: str,
    encoding_id: int | None,
) -> tuple[Descriptor, ...]:
    """Declares static and indexed forms for one memory transfer direction."""
    register_alternatives = (RegClassAlt(register_class),)
    if operation == "load":
        value = Operand("dst", OperandRole.RESULT, register_alternatives)
        effect = _load_effect(width_bits)
        encoding_format_id = Form.LOAD
    elif operation == "store":
        value = Operand("value", OperandRole.OPERAND, register_alternatives)
        effect = _store_effect(width_bits)
        encoding_format_id = Form.STORE
    else:
        raise ValueError(f"unsupported memory transfer operation '{operation}'")

    results = (value.field_name,) if value.role == OperandRole.RESULT else ()
    value_fields = () if results else (value.field_name,)
    return tuple(
        Descriptor(
            key=(
                f"{key_prefix}.{mnemonic}.{operation}{address_suffix}.{register_suffix}"
            ),
            mnemonic=mnemonic,
            encoding_format_id=encoding_format_id if encoding_id is not None else 0,
            encoding_id=(
                encoding_id | (Flag.INDEXED if address_suffix else 0)
                if encoding_id is not None
                else 0
            ),
            semantic_tag=f"memory.{operation}{address_suffix}.{semantic_type}",
            operands=(
                value,
                *(_gpr64_resource(field_name) for field_name in address_fields),
            ),
            immediates=immediates,
            asm_forms=_asm(
                mnemonic=(f"{mnemonic}.{operation}{address_suffix}{assembly_suffix}"),
                results=results,
                operands=(*value_fields, *address_fields),
                immediates=tuple(item.field_name for item in immediates),
                named_immediates=True,
            ),
            effects=(effect,),
            schedule_class=schedule_class,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        )
        for address_suffix, address_fields, immediates in _MEMORY_ADDRESS_FORMS
    )


def memory_load_descriptors(
    *,
    key_prefix: str,
    load_mnemonic: str,
    register_class: str,
    register_suffix: str,
    semantic_type: str,
    width_bits: int,
    load_schedule_class: str,
    assembly_suffix: str,
    encoding_id: int,
) -> tuple[Descriptor, ...]:
    """Declares static and indexed loads without an artificial store family."""
    return _memory_transfer_descriptors(
        key_prefix=key_prefix,
        operation="load",
        mnemonic=load_mnemonic,
        register_class=register_class,
        register_suffix=register_suffix,
        semantic_type=semantic_type,
        width_bits=width_bits,
        schedule_class=load_schedule_class,
        assembly_suffix=assembly_suffix,
        encoding_id=encoding_id,
    )


def broadcast_memory_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    """Bit-preserving scalar loads replicated into complete SIMD carriers."""
    descriptors = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        instructions = native.broadcast_memory_instructions(family.element_bit_width)
        mnemonic = instructions[0].descriptor_mnemonic
        for width, suffix, schedule in (
            (128, "xmm", _SCHEDULE_MEMORY_LOAD_XMM),
            (256, "ymm", _SCHEDULE_MEMORY_LOAD_YMM),
            *(((512, "zmm", _SCHEDULE_MEMORY_LOAD_ZMM),) if evex else ()),
        ):
            rows = _memory_transfer_descriptors(
                key_prefix="x86.avx512" if evex else "x86.avx2",
                operation="load",
                mnemonic=mnemonic,
                register_class=f"x86.{suffix}",
                register_suffix=suffix,
                semantic_type=f"broadcast.i{family.element_bit_width}.v{width}",
                width_bits=family.element_bit_width,
                schedule_class=schedule,
                assembly_suffix=f".{suffix}",
                encoding_id=None,
            )
            if evex and width < 512:
                rows = tuple(
                    replace(
                        row,
                        asm_forms=tuple(
                            replace(form, mnemonic=f"avx512.{form.mnemonic}")
                            for form in row.asm_forms
                        ),
                    )
                    for row in rows
                )
            descriptors.extend(
                _avx512_descriptor(row, instruction, scalar=False)
                if evex
                else _vex_descriptor(row, instruction)
                for row, instruction in zip(rows, instructions, strict=True)
            )
    return tuple(descriptors)


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
    assembly_prefix: str = "",
    encoding_ids: tuple[int, int] | None = None,
    vector_instructions: tuple[VectorMachineInstruction, ...] | None = None,
    vector_binder: Callable[[Descriptor, VectorMachineInstruction], Descriptor]
    | None = None,
) -> tuple[Descriptor, ...]:
    """Declares static/indexed loads followed by static/indexed stores."""
    descriptors = [
        *_memory_transfer_descriptors(
            key_prefix=key_prefix,
            operation="load",
            mnemonic=load_mnemonic,
            register_class=register_class,
            register_suffix=register_suffix,
            semantic_type=semantic_type,
            width_bits=width_bits,
            schedule_class=load_schedule_class,
            assembly_suffix=assembly_suffix,
            encoding_id=encoding_ids[0] if encoding_ids is not None else None,
        ),
        *_memory_transfer_descriptors(
            key_prefix=key_prefix,
            operation="store",
            mnemonic=store_mnemonic,
            register_class=register_class,
            register_suffix=register_suffix,
            semantic_type=semantic_type,
            width_bits=width_bits,
            schedule_class=store_schedule_class,
            assembly_suffix=assembly_suffix,
            encoding_id=encoding_ids[1] if encoding_ids is not None else None,
        ),
    ]
    if assembly_prefix:
        descriptors = [
            replace(
                row,
                asm_forms=tuple(
                    replace(form, mnemonic=assembly_prefix + form.mnemonic)
                    for form in row.asm_forms
                ),
            )
            for row in descriptors
        ]
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


def simd_memory_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    """Exact-width scalar and vector transfers for one native SIMD domain."""
    descriptors = []
    for width in (128, 32, 64, 256, 512) if evex else (128, 32, 64, 256):
        suffix, load_schedule, store_schedule = {
            128: ("xmm", _SCHEDULE_MEMORY_LOAD_XMM, _SCHEDULE_MEMORY_STORE_XMM),
            256: ("ymm", _SCHEDULE_MEMORY_LOAD_YMM, _SCHEDULE_MEMORY_STORE_YMM),
            512: ("zmm", _SCHEDULE_MEMORY_LOAD_ZMM, _SCHEDULE_MEMORY_STORE_ZMM),
        }[max(128, width)]
        mnemonic, instructions = {
            32: ("vmovss", native.XMM_DWORD_MEMORY),
            64: ("vmovsd", native.XMM_QWORD_MEMORY),
        }.get(width, ("vmovdqu32", native.VECTOR_MEMORY))
        descriptors.extend(
            memory_descriptors(
                key_prefix="x86.avx512" if evex else "x86.avx2",
                load_mnemonic=mnemonic,
                store_mnemonic=mnemonic,
                register_class=f"x86.{suffix}",
                register_suffix=suffix,
                semantic_type=f"v{width}",
                width_bits=width,
                load_schedule_class=load_schedule,
                store_schedule_class=store_schedule,
                assembly_suffix="" if width == 512 else f".{suffix}",
                assembly_prefix="avx512." if evex and width < 512 else "",
                vector_instructions=instructions,
                vector_binder=partial(_avx512_descriptor, scalar=width < 128)
                if evex
                else _vex_descriptor,
            )
        )
    return tuple(descriptors)


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
