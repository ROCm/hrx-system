# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Baseline x86-64 scalar/control descriptor rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.target.arch.x86 import native_vector as native
from loom.target.arch.x86.feature_bits import FEATURE_BMI2
from loom.target.arch.x86.scalar_families import (
    X86_SCALAR_READ_MEMORY_FAMILIES,
    X86_SCALAR_VARIABLE_SHIFT_FAMILIES,
)
from loom.target.arch.x86.vector_encoding import (
    VectorEncodingPrefix,
    VectorMachineInstruction,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EnumDomain,
    EnumValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    PhysicalRegister,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    RegisterPackingResource,
    RegisterPackingResourceMember,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

from .common import (
    _ADDRESS_SCALE_ENUM,
    _ADDRESS_SCALE_IMMEDIATE,
    _DISP32_IMMEDIATE,
    _GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
    _IMM8_IMMEDIATE,
    _IMM16_IMMEDIATE,
    _IMM32_IMMEDIATE,
    _IMM64_IMMEDIATE,
    _REG_GPR32,
    _REG_GPR64,
    _RESOURCE_ADDRESS,
    _RESOURCE_CONTROL,
    _RESOURCE_DIVISION,
    _RESOURCE_LOAD,
    _RESOURCE_SCALAR,
    _RESOURCE_STORE,
    _SCHEDULE_ADDRESS,
    _SCHEDULE_CONTROL,
    _SCHEDULE_DIVISION_GPR32,
    _SCHEDULE_DIVISION_GPR64,
    _SCHEDULE_MEMORY_ALU_GPR32,
    _SCHEDULE_MEMORY_ALU_GPR64,
    _SCHEDULE_MEMORY_LOAD_GPR32,
    _SCHEDULE_MEMORY_LOAD_GPR64,
    _SCHEDULE_MEMORY_STORE_GPR32,
    _SCHEDULE_MEMORY_STORE_GPR64,
    _SCHEDULE_SCALAR,
    _SCHEDULE_SCALAR_CONTROL,
    _SHIFT32_IMMEDIATE,
    _SHIFT64_IMMEDIATE,
    _asm,
    _gpr32_operand,
    _gpr32_result,
    _gpr64_operand,
    _gpr64_result,
)
from .encoding import Flag, Form, encoding
from .integer_bits import X86_SCALAR_INTEGER_BIT_DESCRIPTORS
from .memory import memory_descriptors, read_memory_binary_descriptors

_NON_TRAPPING_SCALAR_FORMS = frozenset(
    (
        Form.BINARY_RM_R,
        Form.BINARY_R_RM,
        Form.MULTIPLY_IMMEDIATE,
        Form.BINARY_IMMEDIATE,
        Form.SHIFT_IMMEDIATE,
        Form.SHIFT_COUNT,
        Form.UNARY_R_RM,
        Form.SELECT,
        Form.SUBTRACT_IF_UGE,
        Form.COMPARE,
        Form.COMPARE_IMMEDIATE,
        Form.COMPARE_PREDICATE,
        Form.COMPARE_IMMEDIATE_PREDICATE,
        Form.COMPARE_SELECT,
        Form.COMPARE_IMMEDIATE_SELECT,
        Form.CONSTANT,
        Form.ADDRESS_ADD,
        Form.ADDRESS_DISPLACEMENT,
        Form.ADDRESS_SCALE,
        Form.ADDRESS_ADD_SCALE,
        Form.ADDRESS_PC_RELATIVE,
        Form.UNARY_RM,
    )
)

_SPECULATION_FORBIDDEN_FLAGS = frozenset(
    (
        DescriptorFlag.SIDE_EFFECTING,
        DescriptorFlag.TERMINATOR,
        DescriptorFlag.BARRIER,
        DescriptorFlag.UNIQUE_IDENTITY,
        DescriptorFlag.VARIADIC_OPERANDS,
        DescriptorFlag.STATE_ASSIGNMENT,
    )
)


def _with_scalar_speculation_facts(descriptor: Descriptor) -> Descriptor:
    """Marks total scalar instructions after excluding observable state."""
    if (
        Form(descriptor.encoding_format_id) not in _NON_TRAPPING_SCALAR_FORMS
        or _SPECULATION_FORBIDDEN_FLAGS.intersection(descriptor.flags)
        or descriptor.effects
        or descriptor.storage_leases
        or any(
            OperandFlag.STATE_WRITE in operand.flags for operand in descriptor.operands
        )
    ):
        return descriptor
    return replace(
        descriptor,
        flags=(*descriptor.flags, DescriptorFlag.SAFE_TO_SPECULATE),
    )


# Physical IDs follow the native GPR encoding order, including width aliases.
_GPR_NAMES = (
    "rax",
    "rcx",
    "rdx",
    "rbx",
    "rsp",
    "rbp",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
)
_REG_RAX = "x86.rax"
_REG_RDX = "x86.rdx"
_REG_EAX = "x86.eax"
_REG_EDX = "x86.edx"
_REG_ECX = "x86.ecx"
_REG_RCX = "x86.rcx"

_FIXED_GPR_REGISTER_CLASSES = (
    (_REG_EAX, 32, "rax"),
    (_REG_RAX, 64, "rax"),
    (_REG_EDX, 32, "rdx"),
    (_REG_RDX, 64, "rdx"),
    # Count carriers preserve the source width while both classes occupy RCX.
    # The shift instruction reads CL; ordinary copies use ECX/RCX.
    (_REG_ECX, 32, "rcx"),
    (_REG_RCX, 64, "rcx"),
)
_GPR_PACKING_REGISTER_CLASSES = (
    _REG_GPR32,
    _REG_GPR64,
    *(row[0] for row in _FIXED_GPR_REGISTER_CLASSES),
)


def _gpr32_destructive_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    opcode: int,
    form: Form = Form.BINARY_RM_R,
) -> Descriptor:
    return _gpr_destructive_binary_descriptor(
        key=key,
        encoding_form=form,
        encoding_id=encoding(opcode),
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
    opcode: int,
    form: Form = Form.BINARY_RM_R,
) -> Descriptor:
    return _gpr_destructive_binary_descriptor(
        key=key,
        encoding_form=form,
        encoding_id=encoding(opcode, flags=Flag.REX_W),
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr64_result(),
        lhs=_gpr64_operand("lhs"),
        rhs=_gpr64_operand("rhs"),
        asm_suffix="gpr64",
    )


def _gpr_destructive_unary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    bit_count: int,
    extension: int,
) -> Descriptor:
    result = _gpr32_result() if bit_count == 32 else _gpr64_result()
    input_operand = (
        _gpr32_operand("input") if bit_count == 32 else _gpr64_operand("input")
    )
    return Descriptor(
        key=key,
        encoding_format_id=Form.UNARY_RM,
        encoding_id=encoding(
            0xF7, extension=extension, flags=Flag.REX_W if bit_count == 64 else 0
        ),
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(result, input_operand),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.gpr{bit_count}",
            results=("dst",),
            operands=("input",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _fixed_gpr_operand(field_name: str, register_class: str) -> Operand:
    return Operand(
        field_name,
        OperandRole.OPERAND,
        (RegClassAlt(register_class),),
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
    )


def _fixed_gpr_result(field_name: str, register_class: str) -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        (RegClassAlt(register_class),),
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
    )


def _division_descriptors(bit_count: int) -> tuple[Descriptor, ...]:
    """Returns the explicit fixed-register preparation and divide family."""
    accumulator_class = _REG_EAX if bit_count == 32 else _REG_RAX
    high_class = _REG_EDX if bit_count == 32 else _REG_RDX
    operand = _gpr32_operand if bit_count == 32 else _gpr64_operand
    width_flags = Flag.REX_W if bit_count == 64 else 0
    division_schedule = (
        _SCHEDULE_DIVISION_GPR32 if bit_count == 32 else _SCHEDULE_DIVISION_GPR64
    )
    descriptors = [
        Descriptor(
            key=f"x86.scalar.dividend.low.gpr{bit_count}",
            encoding_format_id=Form.UNARY_R_RM,
            encoding_id=encoding(0x8B, flags=width_flags),
            mnemonic="mov.dividend.low",
            semantic_tag=f"integer.dividend.low.i{bit_count}",
            operands=(
                _fixed_gpr_result("dst", accumulator_class),
                operand("src"),
            ),
            asm_forms=_asm(
                mnemonic=f"mov.dividend.low.gpr{bit_count}",
                results=("dst",),
                operands=("src",),
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(
                DescriptorFlag.DEAD_REMOVABLE,
                DescriptorFlag.STATE_ASSIGNMENT,
            ),
        ),
        Descriptor(
            key=f"x86.scalar.dividend.zero_high.gpr{bit_count}",
            encoding_format_id=Form.ZERO_REGISTER,
            encoding_id=encoding(0x31, flags=width_flags),
            mnemonic="zero.dividend.high",
            semantic_tag=f"integer.dividend.high.zero.i{bit_count}",
            operands=(_fixed_gpr_result("dst", high_class),),
            asm_forms=_asm(
                mnemonic=f"zero.dividend.high.gpr{bit_count}",
                results=("dst",),
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(
                DescriptorFlag.DEAD_REMOVABLE,
                DescriptorFlag.STATE_ASSIGNMENT,
            ),
        ),
        Descriptor(
            key=f"x86.scalar.dividend.sign_high.gpr{bit_count}",
            encoding_format_id=Form.SIGN_EXTEND_DIVIDEND,
            encoding_id=encoding(0x99, flags=width_flags),
            mnemonic="sign.extend.dividend",
            semantic_tag=f"integer.dividend.high.sign.i{bit_count}",
            operands=(
                _fixed_gpr_result("dst", high_class),
                _fixed_gpr_operand("low", accumulator_class),
            ),
            asm_forms=_asm(
                mnemonic=f"sign.extend.dividend.gpr{bit_count}",
                results=("dst",),
                operands=("low",),
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    ]
    for signed, mnemonic, extension in (
        (False, "divu", 6),
        (True, "divs", 7),
    ):
        descriptors.append(
            Descriptor(
                key=f"x86.scalar.{mnemonic}.divrem.gpr{bit_count}",
                encoding_format_id=Form.DIVIDE,
                encoding_id=encoding(0xF7, extension=extension, flags=width_flags),
                mnemonic=mnemonic,
                semantic_tag=(
                    f"integer.div.{'signed' if signed else 'unsigned'}."
                    f"divrem.i{bit_count}"
                ),
                operands=(
                    _fixed_gpr_result("quotient", accumulator_class),
                    _fixed_gpr_result("remainder", high_class),
                    _fixed_gpr_operand("low", accumulator_class),
                    _fixed_gpr_operand("high", high_class),
                    operand("divisor"),
                ),
                constraints=(
                    Constraint(ConstraintKind.TIED, 0, 2),
                    Constraint(ConstraintKind.DESTRUCTIVE, 0, 2),
                    Constraint(ConstraintKind.TIED, 1, 3),
                    Constraint(ConstraintKind.DESTRUCTIVE, 1, 3),
                ),
                asm_forms=_asm(
                    mnemonic=f"{mnemonic}.divrem.gpr{bit_count}",
                    results=("quotient", "remainder"),
                    operands=("low", "high", "divisor"),
                ),
                schedule_class=division_schedule,
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
        )
        for remainder in (False, True):
            result_name = "remainder" if remainder else "quotient"
            result_class = high_class if remainder else accumulator_class
            overwritten_class = accumulator_class if remainder else high_class
            tied_operand_index = 2 if remainder else 1
            descriptors.append(
                Descriptor(
                    key=(f"x86.scalar.{mnemonic}.{result_name}.gpr{bit_count}"),
                    encoding_format_id=Form.DIVIDE,
                    encoding_id=encoding(0xF7, extension=extension, flags=width_flags),
                    mnemonic=mnemonic,
                    semantic_tag=(
                        f"integer.div.{'signed' if signed else 'unsigned'}."
                        f"{result_name}.i{bit_count}"
                    ),
                    operands=(
                        _fixed_gpr_result("dst", result_class),
                        _fixed_gpr_operand("low", accumulator_class),
                        _fixed_gpr_operand("high", high_class),
                        operand("divisor"),
                        Operand(
                            "other_result",
                            OperandRole.IMPLICIT,
                            (RegClassAlt(overwritten_class),),
                            flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
                        ),
                    ),
                    constraints=(
                        Constraint(ConstraintKind.TIED, 0, tied_operand_index),
                        Constraint(ConstraintKind.DESTRUCTIVE, 0, tied_operand_index),
                    ),
                    asm_forms=_asm(
                        mnemonic=f"{mnemonic}.{result_name}.gpr{bit_count}",
                        results=("dst",),
                        operands=("low", "high", "divisor"),
                    ),
                    schedule_class=division_schedule,
                    flags=(DescriptorFlag.DEAD_REMOVABLE,),
                )
            )
    return tuple(descriptors)


def _gpr_destructive_binary_descriptor(
    *,
    key: str,
    encoding_form: Form,
    encoding_id: int,
    mnemonic: str,
    semantic_tag: str,
    result: Operand,
    lhs: Operand,
    rhs: Operand,
    asm_suffix: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        encoding_format_id=encoding_form,
        encoding_id=encoding_id,
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
    extension: int,
) -> Descriptor:
    return _gpr_destructive_immediate_descriptor(
        key=key,
        encoding_form=Form.SHIFT_IMMEDIATE,
        encoding_id=encoding(0xC1, extension=extension),
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
    extension: int,
) -> Descriptor:
    return _gpr_destructive_immediate_descriptor(
        key=key,
        encoding_form=Form.SHIFT_IMMEDIATE,
        encoding_id=encoding(0xC1, extension=extension, flags=Flag.REX_W),
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        result=_gpr64_result(),
        source=_gpr64_operand("lhs"),
        immediate=_SHIFT64_IMMEDIATE,
        asm_suffix="gpr64",
    )


def _gpr_count_shift_descriptor(
    *,
    mnemonic: str,
    semantic: str,
    bit_count: int,
    extension: int,
) -> Descriptor:
    result = _gpr32_result() if bit_count == 32 else _gpr64_result()
    lhs = _gpr32_operand("lhs") if bit_count == 32 else _gpr64_operand("lhs")
    return Descriptor(
        encoding_format_id=Form.SHIFT_COUNT,
        encoding_id=encoding(
            0xD3, extension=extension, flags=Flag.REX_W if bit_count == 64 else 0
        ),
        key=f"x86.scalar.{mnemonic}.cl.gpr{bit_count}",
        mnemonic=mnemonic,
        semantic_tag=f"integer.{semantic}.i{bit_count}",
        operands=(
            result,
            lhs,
            Operand(
                "rhs",
                OperandRole.OPERAND,
                (RegClassAlt(_REG_ECX if bit_count == 32 else _REG_RCX),),
            ),
        ),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.cl.gpr{bit_count}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr_variable_shift_descriptor(
    *,
    descriptor_operation: str,
    semantic: str,
    bit_count: int,
    native_instruction: VectorMachineInstruction,
) -> Descriptor:
    result = _gpr32_result() if bit_count == 32 else _gpr64_result()
    operand = _gpr32_operand if bit_count == 32 else _gpr64_operand
    descriptor = Descriptor(
        key=f"x86.bmi2.{descriptor_operation}.gpr{bit_count}",
        mnemonic=descriptor_operation,
        semantic_tag=f"integer.{semantic}.i{bit_count}",
        operands=(result, operand("lhs"), operand("rhs")),
        asm_forms=_asm(
            mnemonic=f"{descriptor_operation}.gpr{bit_count}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.SAFE_TO_SPECULATE),
        feature_mask_words=(FEATURE_BMI2,),
    )
    return native_instruction.bind(descriptor, VectorEncodingPrefix.VEX)


def _gpr_destructive_immediate_descriptor(
    *,
    key: str,
    encoding_form: Form,
    encoding_id: int,
    mnemonic: str,
    semantic_tag: str,
    result: Operand,
    source: Operand,
    immediate: Immediate,
    asm_suffix: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        encoding_format_id=encoding_form,
        encoding_id=encoding_id,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(result, source),
        immediates=(immediate,),
        constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.imm.{asm_suffix}",
            results=("dst",),
            operands=("lhs",),
            immediates=(immediate.field_name,),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr32_to_gpr64_extend_descriptor(
    *,
    key: str,
    encoding_id: int,
    mnemonic: str,
    semantic_tag: str,
    asm_mnemonic: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding_id,
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


def _gpr64_to_gpr32_truncate_descriptor() -> Descriptor:
    return Descriptor(
        key="x86.scalar.mov.trunc.gpr32.gpr64",
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding(0x8B),
        mnemonic="mov.trunc",
        semantic_tag="integer.trunc.i64.i32",
        operands=(_gpr32_result(), _gpr64_operand("src")),
        asm_forms=_asm(
            mnemonic="mov.trunc.gpr32.gpr64",
            results=("dst",),
            operands=("src",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr_narrow_zero_extend_descriptor(
    source_bit_count: int, result_bit_count: int
) -> Descriptor:
    # The source owns a full GPR32 allocation unit, but MOVZX reads only its
    # low byte or word. A GPR32 destination also defines the complete GPR64, so
    # unsigned i64 results deliberately retain the shorter non-REX.W form.
    return Descriptor(
        key=f"x86.scalar.movzx.u{source_bit_count}.gpr{result_bit_count}",
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding(
            0x0FB6 if source_bit_count == 8 else 0x0FB7,
            flags=Flag.BYTE if source_bit_count == 8 else 0,
        ),
        mnemonic="movzx",
        semantic_tag=f"integer.extui.i{source_bit_count}.i{result_bit_count}",
        operands=(
            _gpr32_result() if result_bit_count == 32 else _gpr64_result(),
            _gpr32_operand("src"),
        ),
        asm_forms=_asm(
            mnemonic=f"movzx.u{source_bit_count}.gpr{result_bit_count}",
            results=("dst",),
            operands=("src",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr_narrow_sign_extend_descriptor(
    source_bit_count: int, result_bit_count: int
) -> Descriptor:
    # The source owns a full GPR32 allocation unit, but MOVSX reads only its
    # low byte or word and defines the complete signed result carrier.
    return Descriptor(
        key=f"x86.scalar.movsx.i{source_bit_count}.gpr{result_bit_count}",
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding(
            0x0FBE if source_bit_count == 8 else 0x0FBF,
            flags=(Flag.BYTE if source_bit_count == 8 else 0)
            | (Flag.REX_W if result_bit_count == 64 else 0),
        ),
        mnemonic="movsx",
        semantic_tag=f"integer.extsi.i{source_bit_count}.i{result_bit_count}",
        operands=(
            _gpr32_result() if result_bit_count == 32 else _gpr64_result(),
            _gpr32_operand("src"),
        ),
        asm_forms=_asm(
            mnemonic=f"movsx.i{source_bit_count}.gpr{result_bit_count}",
            results=("dst",),
            operands=("src",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr_select_descriptor(bit_count: int) -> Descriptor:
    result = _gpr64_result() if bit_count == 64 else _gpr32_result()
    operand = _gpr64_operand if bit_count == 64 else _gpr32_operand
    return Descriptor(
        key=f"x86.scalar.select.gpr{bit_count}",
        encoding_format_id=Form.SELECT,
        encoding_id=encoding(0x0F45, flags=Flag.REX_W if bit_count == 64 else 0),
        mnemonic="select.cmovne",
        semantic_tag=f"integer.select.i{bit_count}",
        operands=(
            result,
            _gpr32_operand("condition"),
            operand("true_value"),
            operand("false_value"),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 3),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 3),
        ),
        asm_forms=_asm(
            mnemonic=f"select.gpr{bit_count}",
            results=("dst",),
            operands=("condition", "true_value", "false_value"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_CMP_PREDICATE_SETCC = (
    ("eq", "sete", 0x94),
    ("ne", "setne", 0x95),
    ("slt", "setl", 0x9C),
    ("sle", "setle", 0x9E),
    ("sgt", "setg", 0x9F),
    ("sge", "setge", 0x9D),
    ("ult", "setb", 0x92),
    ("ule", "setbe", 0x96),
    ("ugt", "seta", 0x97),
    ("uge", "setae", 0x93),
)

X86_INTEGER_CONDITION_CODES = tuple(
    (predicate, opcode & 0x0F) for predicate, _, opcode in _CMP_PREDICATE_SETCC
)

_INTEGER_PREDICATE_ENUM = "x86.integer.predicate"
_INTEGER_PREDICATE_IMMEDIATE = Immediate(
    "predicate",
    ImmediateKind.ENUM,
    enum_domain=_INTEGER_PREDICATE_ENUM,
    bit_width=8,
    unsigned_max=0x0F,
)


def _gpr_width_flags(bit_count: int) -> Flag:
    if bit_count == 8:
        return Flag.BYTE
    if bit_count == 16:
        return Flag.OPERAND_16
    if bit_count == 64:
        return Flag.REX_W
    return Flag(0)


def _gpr_width_operand(bit_count: int, field_name: str) -> Operand:
    return _gpr64_operand(field_name) if bit_count == 64 else _gpr32_operand(field_name)


def _gpr_compare_immediate(bit_count: int) -> Immediate:
    if bit_count == 8:
        return _IMM8_IMMEDIATE
    if bit_count == 16:
        return _IMM16_IMMEDIATE
    return _IMM32_IMMEDIATE


def _gpr_compare_opcode(bit_count: int, *, immediate: bool) -> int:
    if immediate:
        return 0x80 if bit_count == 8 else 0x81
    return 0x38 if bit_count == 8 else 0x39


def _gpr_predicate_compare_descriptor(bit_count: int, *, immediate: bool) -> Descriptor:
    value_immediate = _gpr_compare_immediate(bit_count)
    form_name = "imm." if immediate else ""
    return Descriptor(
        key=f"x86.scalar.cmp.{form_name}gpr{bit_count}",
        encoding_format_id=(
            Form.COMPARE_IMMEDIATE_PREDICATE if immediate else Form.COMPARE_PREDICATE
        ),
        encoding_id=encoding(
            _gpr_compare_opcode(bit_count, immediate=immediate),
            extension=7 if immediate else 0,
            flags=_gpr_width_flags(bit_count),
        ),
        mnemonic="cmp",
        semantic_tag=f"integer.compare.i{bit_count}",
        operands=(
            _gpr32_result(),
            _gpr_width_operand(bit_count, "lhs"),
            *((_gpr_width_operand(bit_count, "rhs"),) if not immediate else ()),
        ),
        immediates=(
            (value_immediate, _INTEGER_PREDICATE_IMMEDIATE)
            if immediate
            else (_INTEGER_PREDICATE_IMMEDIATE,)
        ),
        asm_forms=_asm(
            mnemonic=f"cmp.{form_name}gpr{bit_count}",
            results=("dst",),
            operands=("lhs",) if immediate else ("lhs", "rhs"),
            immediates=(value_immediate.field_name, "predicate")
            if immediate
            else ("predicate",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr_compare_branch_descriptor(bit_count: int, *, immediate: bool) -> Descriptor:
    value_immediate = _gpr_compare_immediate(bit_count)
    form_name = "imm." if immediate else ""
    return Descriptor(
        key=f"x86.scalar.cmp.branch.{form_name}gpr{bit_count}",
        encoding_format_id=(
            Form.COMPARE_IMMEDIATE_BRANCH if immediate else Form.COMPARE_BRANCH
        ),
        encoding_id=encoding(
            _gpr_compare_opcode(bit_count, immediate=immediate),
            extension=7 if immediate else 0,
            flags=_gpr_width_flags(bit_count),
        ),
        mnemonic="cmp.branch",
        semantic_tag=f"control.integer_compare.i{bit_count}",
        operands=(
            _gpr_width_operand(bit_count, "lhs"),
            *((_gpr_width_operand(bit_count, "rhs"),) if not immediate else ()),
        ),
        op_kind=DescriptorOpKind.COND_BR,
        immediates=(
            (value_immediate, _INTEGER_PREDICATE_IMMEDIATE)
            if immediate
            else (_INTEGER_PREDICATE_IMMEDIATE,)
        ),
        asm_surface=DescriptorAsmSurface.STRUCTURAL,
        asm_surface_reason="CFG successors use the canonical low.op.cond_br syntax.",
        effects=(
            Effect(
                EffectKind.CONTROL,
                flags=(EffectFlag.ORDERED,),
            ),
        ),
        schedule_class=_SCHEDULE_SCALAR_CONTROL,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        instruction_classes=(InstructionClass.BRANCH,),
    )


def _gpr_compare_select_descriptor(
    compare_bit_count: int, result_bit_count: int, *, immediate: bool
) -> Descriptor:
    result = _gpr64_result() if result_bit_count == 64 else _gpr32_result()
    result_operand = _gpr64_operand if result_bit_count == 64 else _gpr32_operand
    false_operand_index = 3 if immediate else 4
    form = Form.COMPARE_IMMEDIATE_SELECT if immediate else Form.COMPARE_SELECT
    form_name = "imm." if immediate else ""
    value_immediate = _gpr_compare_immediate(compare_bit_count)
    immediate_fields = (
        (value_immediate, _INTEGER_PREDICATE_IMMEDIATE)
        if immediate
        else (_INTEGER_PREDICATE_IMMEDIATE,)
    )
    return Descriptor(
        key=(
            f"x86.scalar.cmp.select.{form_name}gpr{compare_bit_count}."
            f"gpr{result_bit_count}"
        ),
        encoding_format_id=form,
        encoding_id=encoding(
            _gpr_compare_opcode(compare_bit_count, immediate=immediate),
            extension=7 if immediate else 0,
            flags=_gpr_width_flags(compare_bit_count)
            | (Flag.RESULT_REX_W if result_bit_count == 64 else 0),
        ),
        mnemonic="cmp.select",
        semantic_tag=(
            f"integer.compare.select.i{compare_bit_count}.i{result_bit_count}"
        ),
        operands=(
            result,
            _gpr_width_operand(compare_bit_count, "lhs"),
            *((_gpr_width_operand(compare_bit_count, "rhs"),) if not immediate else ()),
            result_operand("true_value"),
            result_operand("false_value"),
        ),
        immediates=immediate_fields,
        constraints=(
            Constraint(ConstraintKind.TIED, 0, false_operand_index),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, false_operand_index),
        ),
        asm_forms=_asm(
            mnemonic=(
                f"cmp.select.{form_name}gpr{compare_bit_count}.gpr{result_bit_count}"
            ),
            results=("dst",),
            operands=("lhs", "true_value", "false_value")
            if immediate
            else ("lhs", "rhs", "true_value", "false_value"),
            immediates=(value_immediate.field_name, "predicate")
            if immediate
            else ("predicate",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _gpr32_compare_descriptor(
    *,
    predicate: str,
    setcc: str,
    opcode: int,
    semantic_tag: str,
) -> Descriptor:
    return _gpr_compare_descriptor(
        predicate=predicate,
        setcc=setcc,
        encoding_id=encoding(opcode),
        semantic_tag=semantic_tag,
        lhs=_gpr32_operand("lhs"),
        rhs=_gpr32_operand("rhs"),
        asm_suffix="gpr32",
    )


def _gpr_compare_descriptor(
    *,
    predicate: str,
    setcc: str,
    encoding_id: int,
    semantic_tag: str,
    lhs: Operand,
    rhs: Operand | Immediate,
    asm_suffix: str,
) -> Descriptor:
    immediate = isinstance(rhs, Immediate)
    return Descriptor(
        encoding_format_id=Form.COMPARE_IMMEDIATE if immediate else Form.COMPARE,
        encoding_id=encoding_id,
        key=f"x86.scalar.cmp.{predicate}.{asm_suffix}",
        mnemonic=f"cmp.{setcc}",
        semantic_tag=semantic_tag,
        operands=(
            _gpr32_result(),
            lhs,
            *((rhs,) if not immediate else ()),
        ),
        immediates=(rhs,) if immediate else (),
        asm_forms=_asm(
            mnemonic=f"cmp.{predicate}.{asm_suffix}",
            results=("dst",),
            operands=("lhs",) if immediate else ("lhs", "rhs"),
            immediates=("imm32",) if immediate else (),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_X86_SCALAR_PREFIX_DESCRIPTORS = (
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.add.gpr32",
        opcode=0x01,
        mnemonic="add",
        semantic_tag="integer.add.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.add.gpr64",
        opcode=0x01,
        mnemonic="add",
        semantic_tag="integer.add.i64",
    ),
)

_X86_SCALAR_SUFFIX_DESCRIPTORS = (
    *X86_SCALAR_INTEGER_BIT_DESCRIPTORS,
    Descriptor(
        key="x86.scalar.lea.symbol.gpr64",
        encoding_format_id=Form.ADDRESS_PC_RELATIVE,
        encoding_id=encoding(0x8D, flags=Flag.REX_W),
        mnemonic="lea",
        semantic_tag="address.symbol",
        operands=(_gpr64_result(),),
        immediates=(
            Immediate(
                "symbol",
                ImmediateKind.ORDINAL,
                flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
                bit_width=32,
                unsigned_max=(2**31) - 1,
            ),
        ),
        asm_forms=_asm(
            mnemonic="lea.symbol.gpr64",
            results=("dst",),
            immediates=("symbol",),
        ),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.sub.gpr32",
        opcode=0x29,
        mnemonic="sub",
        semantic_tag="integer.sub.i32",
    ),
    *(
        _gpr_destructive_unary_descriptor(
            key=f"x86.scalar.neg.gpr{bit_count}",
            mnemonic="neg",
            semantic_tag=f"integer.neg.i{bit_count}",
            bit_count=bit_count,
            extension=3,
        )
        for bit_count in (32, 64)
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.imul.gpr32",
        opcode=0x0FAF,
        form=Form.BINARY_R_RM,
        mnemonic="imul",
        semantic_tag="integer.mul.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.sub.gpr64",
        opcode=0x29,
        mnemonic="sub",
        semantic_tag="integer.sub.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.imul.gpr64",
        opcode=0x0FAF,
        form=Form.BINARY_R_RM,
        mnemonic="imul",
        semantic_tag="integer.mul.i64",
    ),
    *(
        descriptor
        for width, register_class, register_suffix, schedule_class in (
            (
                32,
                _REG_GPR32,
                "gpr32",
                _SCHEDULE_MEMORY_ALU_GPR32,
            ),
            (
                64,
                _REG_GPR64,
                "gpr64",
                _SCHEDULE_MEMORY_ALU_GPR64,
            ),
        )
        for family in X86_SCALAR_READ_MEMORY_FAMILIES
        for descriptor in read_memory_binary_descriptors(
            key_prefix=f"x86.scalar.{family.descriptor_operation}",
            mnemonic=family.mnemonic,
            register_class=register_class,
            register_suffix=register_suffix,
            semantic_tag=f"integer.{family.semantic}.i{width}",
            width_bits=width,
            schedule_class=schedule_class,
            opcode=family.opcode,
            encoding_flags=Flag.REX_W if width == 64 else 0,
        )
    ),
    *(
        descriptor
        for bit_count in (32, 64)
        for descriptor in _division_descriptors(bit_count)
    ),
    Descriptor(
        key="x86.scalar.mul.high.gpr64",
        encoding_format_id=Form.MULTIPLY_HIGH,
        encoding_id=encoding(0xF7, extension=4, flags=Flag.REX_W),
        mnemonic="mul",
        semantic_tag="integer.mul.high.u64",
        operands=(
            _fixed_gpr_result("dst", _REG_RDX),
            _fixed_gpr_operand("lhs", _REG_RAX),
            _gpr64_operand("rhs"),
            Operand(
                "low",
                OperandRole.IMPLICIT,
                (RegClassAlt(_REG_RAX),),
                flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
            ),
        ),
        asm_forms=_asm(
            mnemonic="mul.high.gpr64",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    *(
        Descriptor(
            key=f"x86.scalar.imul.imm.gpr{width}",
            encoding_format_id=Form.MULTIPLY_IMMEDIATE,
            encoding_id=encoding(0x69, flags=Flag.REX_W if width == 64 else 0),
            mnemonic="imul",
            semantic_tag=f"integer.mul.signed_imm32.i{width}",
            operands=(result(), operand("lhs")),
            immediates=(_IMM32_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic=f"imul.imm.gpr{width}",
                results=("dst",),
                operands=("lhs",),
                immediates=("imm32",),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        for width, result, operand in (
            (32, _gpr32_result, _gpr32_operand),
            (64, _gpr64_result, _gpr64_operand),
        )
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.and.gpr32",
        opcode=0x21,
        mnemonic="and",
        semantic_tag="integer.and.i32",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.or.gpr32",
        opcode=0x09,
        mnemonic="or",
        semantic_tag="integer.or.i32",
    ),
    _gpr32_destructive_binary_descriptor(
        key="x86.scalar.xor.gpr32",
        opcode=0x31,
        mnemonic="xor",
        semantic_tag="integer.xor.i32",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.and.gpr64",
        opcode=0x21,
        mnemonic="and",
        semantic_tag="integer.and.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.or.gpr64",
        opcode=0x09,
        mnemonic="or",
        semantic_tag="integer.or.i64",
    ),
    _gpr64_destructive_binary_descriptor(
        key="x86.scalar.xor.gpr64",
        opcode=0x31,
        mnemonic="xor",
        semantic_tag="integer.xor.i64",
    ),
    *(
        _gpr_destructive_immediate_descriptor(
            key=f"x86.scalar.{operation}.imm.gpr{width}",
            encoding_form=Form.BINARY_IMMEDIATE,
            encoding_id=encoding(
                0x81, extension=extension, flags=Flag.REX_W if width == 64 else 0
            ),
            mnemonic=operation,
            semantic_tag=f"integer.{operation}.signed_imm32.i{width}",
            result=result(),
            source=operand("lhs"),
            immediate=_IMM32_IMMEDIATE,
            asm_suffix=f"gpr{width}",
        )
        for width, result, operand in (
            (32, _gpr32_result, _gpr32_operand),
            (64, _gpr64_result, _gpr64_operand),
        )
        for operation, extension in (
            ("add", 0),
            ("sub", 5),
            ("and", 4),
            ("or", 1),
            ("xor", 6),
        )
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.shl.imm.gpr32",
        extension=4,
        mnemonic="shl",
        semantic_tag="integer.shl.i32",
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.sar.imm.gpr32",
        extension=7,
        mnemonic="sar",
        semantic_tag="integer.shrs.i32",
    ),
    _gpr32_destructive_shift_descriptor(
        key="x86.scalar.shr.imm.gpr32",
        extension=5,
        mnemonic="shr",
        semantic_tag="integer.shru.i32",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.shl.imm.gpr64",
        extension=4,
        mnemonic="shl",
        semantic_tag="integer.shl.i64",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.sar.imm.gpr64",
        extension=7,
        mnemonic="sar",
        semantic_tag="integer.shrs.i64",
    ),
    _gpr64_destructive_shift_descriptor(
        key="x86.scalar.shr.imm.gpr64",
        extension=5,
        mnemonic="shr",
        semantic_tag="integer.shru.i64",
    ),
    *(
        _gpr_count_shift_descriptor(
            mnemonic=mnemonic,
            semantic=semantic,
            bit_count=bit_count,
            extension=extension,
        )
        for bit_count in (32, 64)
        for mnemonic, semantic, extension in (
            ("shl", "shl", 4),
            ("sar", "shrs", 7),
            ("shr", "shru", 5),
        )
    ),
    _gpr64_to_gpr32_truncate_descriptor(),
    _gpr_select_descriptor(32),
    _gpr_select_descriptor(64),
    *(
        _gpr_predicate_compare_descriptor(bit_count, immediate=immediate)
        for bit_count in (8, 16, 32, 64)
        for immediate in (False, True)
    ),
    *(
        _gpr_compare_branch_descriptor(bit_count, immediate=immediate)
        for bit_count in (8, 16, 32, 64)
        for immediate in (False, True)
    ),
    *(
        _gpr_compare_select_descriptor(
            compare_bit_count, result_bit_count, immediate=immediate
        )
        for compare_bit_count in (8, 16, 32, 64)
        for result_bit_count in (32, 64)
        for immediate in (False, True)
    ),
    # Restore the original lhs when the subtraction borrows. The original is
    # still read after dst is written, so allocation keeps the two disjoint.
    Descriptor(
        key="x86.scalar.sub.if_uge.imm.gpr32",
        encoding_format_id=Form.SUBTRACT_IF_UGE,
        encoding_id=encoding(0x0F42),
        mnemonic="sub.cmovb",
        semantic_tag="integer.subtract_if_unsigned_ge.i32",
        operands=(
            _gpr32_result(),
            Operand(
                "lhs",
                OperandRole.OPERAND,
                (RegClassAlt(_REG_GPR32, late_read_subgroup_size=0),),
            ),
        ),
        immediates=(_IMM32_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic="sub.if_uge.imm.gpr32",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    # Closed descriptor recipes have no source predicate attribute to project.
    # Keep only the fixed identities those recipes bind; ordinary source
    # comparisons use the width-parametric predicate descriptors above.
    *(
        _gpr_compare_descriptor(
            predicate=predicate,
            setcc=setcc,
            encoding_id=encoding(opcode, flags=Flag.REX_W if width == 64 else 0),
            semantic_tag=f"integer.cmp.{predicate}.i{width}",
            lhs=operand("lhs"),
            rhs=_IMM32_IMMEDIATE,
            asm_suffix=f"imm.gpr{width}",
        )
        for width, operand in ((32, _gpr32_operand), (64, _gpr64_operand))
        for predicate, setcc, opcode in _CMP_PREDICATE_SETCC
        if (width, predicate)
        in {
            (32, "eq"),
            (32, "slt"),
            (32, "sge"),
            (32, "sgt"),
            (32, "uge"),
            (64, "eq"),
        }
    ),
    *(
        _gpr32_compare_descriptor(
            predicate=predicate,
            setcc=setcc,
            opcode=opcode,
            semantic_tag=f"integer.cmp.{predicate}.i32",
        )
        for predicate, setcc, opcode in _CMP_PREDICATE_SETCC
        if predicate in {"slt", "sge", "sgt"}
    ),
    Descriptor(
        key="x86.scalar.movimm.gpr32",
        encoding_format_id=Form.CONSTANT,
        encoding_id=encoding(0xB8),
        mnemonic="mov",
        semantic_tag="integer.const.i32",
        operands=(_gpr32_result(),),
        op_kind=DescriptorOpKind.CONST,
        immediates=(_IMM32_IMMEDIATE,),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        asm_forms=_asm(
            mnemonic="mov.imm32",
            results=("dst",),
            immediates=("imm32",),
        ),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    *(
        descriptor
        for width_bits in (8, 16)
        for descriptor in memory_descriptors(
            key_prefix="x86.scalar",
            load_mnemonic="movzx",
            store_mnemonic="mov",
            register_class=_REG_GPR32,
            register_suffix=f"u{width_bits}.gpr32",
            semantic_type=f"i{width_bits}",
            width_bits=width_bits,
            load_schedule_class=_SCHEDULE_MEMORY_LOAD_GPR32,
            store_schedule_class=_SCHEDULE_MEMORY_STORE_GPR32,
            assembly_suffix=f".u{width_bits}.gpr32",
            encoding_ids=(
                encoding(0x0FB6 if width_bits == 8 else 0x0FB7),
                encoding(0x88, flags=Flag.BYTE)
                if width_bits == 8
                else encoding(0x89, flags=Flag.OPERAND_16),
            ),
        )
    ),
    *memory_descriptors(
        key_prefix="x86.scalar",
        load_mnemonic="mov",
        store_mnemonic="mov",
        register_class=_REG_GPR32,
        register_suffix="gpr32",
        semantic_type="i32",
        width_bits=32,
        load_schedule_class=_SCHEDULE_MEMORY_LOAD_GPR32,
        store_schedule_class=_SCHEDULE_MEMORY_STORE_GPR32,
        assembly_suffix=".gpr32",
        encoding_ids=(encoding(0x8B), encoding(0x89)),
    ),
    *memory_descriptors(
        key_prefix="x86.scalar",
        load_mnemonic="mov",
        store_mnemonic="mov",
        register_class=_REG_GPR64,
        register_suffix="gpr64",
        semantic_type="i64",
        width_bits=64,
        load_schedule_class=_SCHEDULE_MEMORY_LOAD_GPR64,
        store_schedule_class=_SCHEDULE_MEMORY_STORE_GPR64,
        assembly_suffix=".gpr64",
        encoding_ids=(
            encoding(0x8B, flags=Flag.REX_W),
            encoding(0x89, flags=Flag.REX_W),
        ),
    ),
    Descriptor(
        key="x86.scalar.mov.gpr64",
        encoding_format_id=Form.UNARY_R_RM,
        encoding_id=encoding(0x8B, flags=Flag.REX_W),
        mnemonic="mov",
        semantic_tag="integer.move.i64",
        operands=(_gpr64_result(), _gpr64_operand("src")),
        asm_forms=_asm(results=("dst",), operands=("src",)),
        schedule_class=_SCHEDULE_SCALAR,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    _gpr32_to_gpr64_extend_descriptor(
        key="x86.scalar.movsxd.gpr64.gpr32",
        encoding_id=encoding(0x63, flags=Flag.REX_W),
        mnemonic="movsxd",
        semantic_tag="integer.extsi.i32.i64",
        asm_mnemonic="movsxd.gpr64.gpr32",
    ),
    _gpr32_to_gpr64_extend_descriptor(
        key="x86.scalar.movzx.gpr64.gpr32",
        encoding_id=encoding(0x8B),
        mnemonic="movzx",
        semantic_tag="integer.extui.i32.i64",
        asm_mnemonic="movzx.gpr64.gpr32",
    ),
    *(
        _gpr_narrow_zero_extend_descriptor(source_width, result_width)
        for source_width in (8, 16)
        for result_width in (32, 64)
    ),
    *(
        _gpr_narrow_sign_extend_descriptor(source_width, result_width)
        for source_width in (8, 16)
        for result_width in (32, 64)
    ),
    Descriptor(
        key="x86.scalar.movimm.gpr64",
        encoding_format_id=Form.CONSTANT,
        encoding_id=encoding(0xB8, flags=Flag.REX_W),
        mnemonic="mov",
        semantic_tag="integer.const.i64",
        operands=(_gpr64_result(),),
        op_kind=DescriptorOpKind.CONST,
        immediates=(_IMM64_IMMEDIATE,),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
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
        encoding_format_id=Form.ADDRESS_ADD,
        encoding_id=encoding(0x8D, flags=Flag.REX_W),
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
        encoding_format_id=Form.ADDRESS_DISPLACEMENT,
        encoding_id=encoding(0x8D, flags=Flag.REX_W),
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
        encoding_format_id=Form.ADDRESS_SCALE,
        encoding_id=encoding(0x8D, flags=Flag.REX_W),
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
        encoding_format_id=Form.ADDRESS_ADD_SCALE,
        encoding_id=encoding(0x8D, flags=Flag.REX_W),
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
        key="x86.scalar.lea.add_scale.gpr32",
        encoding_format_id=Form.ADDRESS_ADD_SCALE,
        encoding_id=encoding(0x8D),
        mnemonic="lea",
        semantic_tag="integer.add.scale.disp.i64.trunc.i32",
        operands=(
            _gpr32_result(),
            _gpr64_operand("base"),
            _gpr64_operand("index"),
        ),
        immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
        asm_forms=_asm(
            mnemonic="lea.add_scale.gpr32",
            results=("dst",),
            operands=("base", "index"),
            immediates=("disp32", "scale"),
            named_immediates=True,
        ),
        schedule_class=_SCHEDULE_ADDRESS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
)

X86_SCALAR_PREFIX_DESCRIPTORS = tuple(
    _with_scalar_speculation_facts(descriptor)
    for descriptor in _X86_SCALAR_PREFIX_DESCRIPTORS
)
X86_SCALAR_SUFFIX_DESCRIPTORS = tuple(
    _with_scalar_speculation_facts(descriptor)
    for descriptor in _X86_SCALAR_SUFFIX_DESCRIPTORS
)
X86_SCALAR_BASE_DESCRIPTORS = (
    *X86_SCALAR_PREFIX_DESCRIPTORS,
    *X86_SCALAR_SUFFIX_DESCRIPTORS,
)
X86_SCALAR_BMI2_DESCRIPTORS = tuple(
    _gpr_variable_shift_descriptor(
        descriptor_operation=family.descriptor_operation,
        semantic=family.semantic,
        bit_count=bit_count,
        native_instruction=native_instruction,
    )
    for (bit_count, family), native_instruction in zip(
        (
            (bit_count, family)
            for bit_count in (32, 64)
            for family in X86_SCALAR_VARIABLE_SHIFT_FAMILIES
        ),
        native.BMI2_VARIABLE_SHIFT,
        strict=True,
    )
)
X86_SCALAR_DESCRIPTORS = (
    *X86_SCALAR_BASE_DESCRIPTORS,
    *X86_SCALAR_BMI2_DESCRIPTORS,
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
    supports_native_scheduling=True,
    physical_registers=tuple(
        PhysicalRegister(name, (index,)) for index, name in enumerate(_GPR_NAMES)
    ),
    reg_classes=(
        RegClass(
            _REG_GPR32,
            32,
            SpillSlotSpace.STACK,
            flags=(
                RegClassFlag.PHYSICAL,
                RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                RegClassFlag.CONTIGUOUS_PHYSICAL_REGISTER_CANDIDATES,
            ),
            physical_registers=_GPR_NAMES,
        ),
        RegClass(
            _REG_GPR64,
            64,
            SpillSlotSpace.STACK,
            flags=(
                RegClassFlag.PHYSICAL,
                RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                RegClassFlag.CONTIGUOUS_PHYSICAL_REGISTER_CANDIDATES,
            ),
            physical_registers=_GPR_NAMES,
        ),
        *(
            RegClass(
                register_class,
                bit_count,
                SpillSlotSpace.STACK,
                flags=(
                    RegClassFlag.PHYSICAL,
                    RegClassFlag.UNSPILLABLE,
                    RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                ),
                physical_registers=(register,),
            )
            for register_class, bit_count, register in _FIXED_GPR_REGISTER_CLASSES
        ),
    ),
    register_packing_resources=(
        RegisterPackingResource(
            name="x86.gpr.units",
            capacity_register_class=_REG_GPR64,
            members=tuple(
                RegisterPackingResourceMember(register_class)
                for register_class in _GPR_PACKING_REGISTER_CLASSES
            ),
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(
            _RESOURCE_DIVISION,
            capacity_per_cycle=1,
            kind=ResourceKind.PIPELINE,
        ),
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
            instruction_classes=(InstructionClass.SCALAR_ALU,),
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_GPR32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD_GPR64,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_ALU_GPR32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_ALU_GPR64,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            minimum_issue_separation_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),
            ),
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
            _SCHEDULE_SCALAR_CONTROL,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),
                IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),
            ),
            flags=(ScheduleClassFlag.CONTROL,),
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
        ScheduleClass(
            _SCHEDULE_DIVISION_GPR32,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=20,
            minimum_issue_separation_cycles=6,
            issue_uses=(IssueUse(_RESOURCE_DIVISION, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            instruction_classes=(InstructionClass.SCALAR_ALU,),
        ),
        ScheduleClass(
            _SCHEDULE_DIVISION_GPR64,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=40,
            minimum_issue_separation_cycles=10,
            issue_uses=(IssueUse(_RESOURCE_DIVISION, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            instruction_classes=(InstructionClass.SCALAR_ALU,),
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
        EnumDomain(
            _INTEGER_PREDICATE_ENUM,
            tuple(
                EnumValue(predicate, condition_code)
                for predicate, condition_code in X86_INTEGER_CONDITION_CODES
            ),
        ),
    ),
    descriptors=X86_SCALAR_DESCRIPTORS,
)

X86_SCALAR_BASE_DESCRIPTOR_SET = replace(
    X86_SCALAR_DESCRIPTOR_SET,
    descriptors=X86_SCALAR_BASE_DESCRIPTORS,
)
