# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""X86 integer bit-scan and native count descriptor rows."""

from loom.target.arch.x86.feature_bits import (
    FEATURE_BMI,
    FEATURE_LZCNT,
    FEATURE_POPCNT,
)
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _SCHEDULE_SCALAR,
    _asm,
    _gpr32_result,
    _gpr64_result,
    _gpr_width_flags,
    _gpr_width_operand,
)
from .encoding import Flag, Form, encoding


def _bit_scan_descriptor(operation: str, opcode: int, bit_count: int) -> Descriptor:
    result = _gpr32_result if bit_count == 32 else _gpr64_result
    return Descriptor(
        key=f"x86.scalar.{operation}.gpr{bit_count}",
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding(opcode, flags=Flag.REX_W if bit_count == 64 else 0),
        mnemonic=operation,
        semantic_tag=f"integer.bit_scan.{operation}.i{bit_count}",
        operands=(result(), _gpr_width_operand(bit_count, "input")),
        asm_forms=_asm(
            mnemonic=f"{operation}.gpr{bit_count}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


X86_SCALAR_INTEGER_BIT_DESCRIPTORS = tuple(
    _bit_scan_descriptor(operation, opcode, bit_count)
    for operation, opcode in (("bsf", 0x0FBC), ("bsr", 0x0FBD))
    for bit_count in (32, 64)
)


def _native_count_descriptor(
    operation: str,
    semantic: str,
    opcode: int,
    feature: int,
    bit_count: int,
) -> Descriptor:
    return Descriptor(
        key=f"x86.scalar.{operation}.gpr{bit_count}",
        encoding_format_id=Form.UNARY_F3_R_RM,
        encoding_id=encoding(opcode, flags=_gpr_width_flags(bit_count)),
        mnemonic=operation,
        semantic_tag=f"integer.bit_count.{semantic}.i{bit_count}",
        operands=(
            _gpr64_result() if bit_count == 64 else _gpr32_result(),
            _gpr_width_operand(bit_count, "input"),
        ),
        asm_forms=_asm(
            mnemonic=f"{operation}.gpr{bit_count}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.SAFE_TO_SPECULATE),
        feature_mask_words=(feature,),
    )


X86_SCALAR_NATIVE_COUNT_DESCRIPTORS = tuple(
    _native_count_descriptor(operation, semantic, opcode, feature, bit_count)
    for operation, semantic, opcode, feature in (
        ("lzcnt", "ctlz", 0x0FBD, FEATURE_LZCNT),
        ("tzcnt", "cttz", 0x0FBC, FEATURE_BMI),
        ("popcnt", "ctpop", 0x0FB8, FEATURE_POPCNT),
    )
    for bit_count in (16, 32, 64)
)
