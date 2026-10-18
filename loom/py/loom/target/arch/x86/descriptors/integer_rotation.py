# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete native scalar and AVX-512 integer rotation descriptor families."""

from __future__ import annotations

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.scalar_families import X86_SCALAR_ROTATION_FAMILIES
from loom.target.arch.x86.vector_families import AVX512_ROTATION_FAMILIES
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    Immediate,
    ImmediateKind,
    Operand,
    OperandRole,
    RegClassAlt,
)

from .common import (
    _GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
    _SCHEDULE_SCALAR,
    _asm,
    _avx512_descriptor,
    _gpr32_result,
    _gpr64_result,
    _gpr_width_flags,
    _gpr_width_operand,
    _vector_i32_schedule_class,
    _vector_operand,
    _vector_result,
)
from .encoding import Form, encoding

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_ROTATION_IMMEDIATE = Immediate(
    "amount",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)


def _scalar_rotation_descriptor(
    *,
    bit_count: int,
    mnemonic: str,
    semantic: str,
    opcode_extension: int,
    immediate: bool,
) -> Descriptor:
    register_suffix = f"gpr{bit_count}"
    result = _gpr64_result() if bit_count == 64 else _gpr32_result()
    operands = [result, _gpr_width_operand(bit_count, "lhs")]
    if not immediate:
        operands.append(
            Operand(
                "rhs",
                OperandRole.OPERAND,
                (RegClassAlt("x86.rcx" if bit_count == 64 else "x86.ecx"),),
            )
        )
    return Descriptor(
        key=(
            f"x86.scalar.{mnemonic}.imm.{register_suffix}"
            if immediate
            else f"x86.scalar.{mnemonic}.cl.{register_suffix}"
        ),
        encoding_format_id=(Form.SHIFT_IMMEDIATE if immediate else Form.SHIFT_COUNT),
        encoding_id=encoding(
            (0xC0 if bit_count == 8 else 0xC1)
            if immediate
            else (0xD2 if bit_count == 8 else 0xD3),
            extension=opcode_extension,
            flags=_gpr_width_flags(bit_count),
        ),
        mnemonic=mnemonic,
        semantic_tag=f"integer.{semantic}.i{bit_count}",
        operands=tuple(operands),
        immediates=(_ROTATION_IMMEDIATE,) if immediate else (),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=(
                f"{mnemonic}.imm.{register_suffix}"
                if immediate
                else f"{mnemonic}.cl.{register_suffix}"
            ),
            results=("dst",),
            operands=("lhs",) if immediate else ("lhs", "rhs"),
            immediates=("amount",) if immediate else (),
            named_immediates=immediate,
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


X86_SCALAR_ROTATION_DESCRIPTORS = tuple(
    _scalar_rotation_descriptor(
        bit_count=bit_count,
        mnemonic=family.mnemonic,
        semantic=family.semantic,
        opcode_extension=family.opcode_extension,
        immediate=immediate,
    )
    for family in X86_SCALAR_ROTATION_FAMILIES
    for bit_count in (8, 16, 32, 64)
    for immediate in (True, False)
)


def _vector_rotation_descriptor(
    *,
    vector_bit_width: int,
    element_bit_width: int,
    mnemonic: str,
    semantic: str,
    instruction: native.VectorMachineInstruction,
    immediate: bool,
) -> Descriptor:
    register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
    lane_count = vector_bit_width // element_bit_width
    descriptor = Descriptor(
        key=f"x86.avx512.{mnemonic}.{register_suffix}",
        mnemonic=mnemonic,
        semantic_tag=f"{semantic}.i{element_bit_width}x{lane_count}",
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "lhs"),
            *(() if immediate else (_vector_operand(vector_bit_width, "rhs"),)),
        ),
        immediates=(_ROTATION_IMMEDIATE,) if immediate else (),
        asm_forms=_asm(
            mnemonic=f"avx512.{mnemonic}.{register_suffix}",
            results=("dst",),
            operands=("lhs",) if immediate else ("lhs", "rhs"),
            immediates=("amount",) if immediate else (),
            named_immediates=immediate,
        ),
        schedule_class=_vector_i32_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return _avx512_descriptor(descriptor, instruction, scalar=False)


X86_AVX512_ROTATION_DESCRIPTORS = tuple(
    descriptor
    for family, immediate_instruction, variable_instruction in zip(
        AVX512_ROTATION_FAMILIES,
        native.AVX512_ROTATE_IMMEDIATE,
        native.AVX512_ROTATE_VARIABLE,
        strict=True,
    )
    for vector_bit_width in (128, 256, 512)
    for descriptor in (
        _vector_rotation_descriptor(
            vector_bit_width=vector_bit_width,
            element_bit_width=family.element.bit_width,
            mnemonic=family.immediate_mnemonic,
            semantic=family.semantic,
            instruction=immediate_instruction,
            immediate=True,
        ),
        _vector_rotation_descriptor(
            vector_bit_width=vector_bit_width,
            element_bit_width=family.element.bit_width,
            mnemonic=family.variable_mnemonic,
            semantic=family.semantic,
            instruction=variable_instruction,
            immediate=False,
        ),
    )
)
