# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P native approximate nonlinear math contracts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, unique

from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    emit_round_nearest_f32_to_i32,
    emit_signed_integer_to_f32,
)
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.contracts.floating import (
    emit_f32_scalar_accumulator_binary,
)
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    ResultTypeBinding,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)

_F32 = Scalar("f32")
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_BF16X32_VECTOR = Vector("bf16", lanes=32)

# AIE2P VEXP2 accepts binary32 accumulator lanes but returns bfloat16. EXPF is
# therefore necessarily approximate and, under afn, narrows its input and
# log2(e) scale to bfloat16 so the scale can use the native vector datapath.
_LOG2_E_BF16_BITS = 0x3FB9
# Radian inputs are narrowed under afn before scaling to turns. The remaining
# polynomial evaluates sin(2*pi*t) for t in [0, 0.25] after quadrant folding.
_INVERSE_TWO_PI_BF16_BITS = 0x3E23
# Coefficients are ordered from t through t^7 and tuned for the bfloat16
# rounding points between AIE2P VMAC stages. The maximum absolute error of the
# polynomial over the folded interval is below 0.005.
_SINE_POLYNOMIAL_BF16_BITS = (0x40C9, 0xC225, 0x42A1, 0xC290)
_BF16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=2,
    multiplication_mode=3,
    compute_mode=1,
)


@unique
class _TrigOperation(Enum):
    SIN_RADIANS = "sin_radians"
    COS_RADIANS = "cos_radians"
    SIN_TURNS = "sin_turns"
    COS_TURNS = "cos_turns"


def _emit(
    descriptor_key: str,
    *,
    operands: Mapping[str, ValueRef] | None = None,
    results: Mapping[str, ValueRef] | None = None,
    result_types: Mapping[str, ResultTypeBinding] | None = None,
    immediates: Mapping[str, int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
    )


def _inferred_emit(
    descriptor_key: str,
    *,
    operands: Mapping[str, ValueRef],
    result: ValueRef,
) -> EmitDescriptorOp:
    return _emit(
        descriptor_key,
        operands=operands,
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
    )


def _constant_emit(
    descriptor_key: str,
    result: ValueRef,
    value: int,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _bf16_splat_emits(
    prefix: str,
    bits: int,
) -> tuple[list[ContractEmit], ValueRef]:
    bits_value = ValueRef.temporary(f"{prefix}_bits")
    result = ValueRef.temporary(prefix)
    return [
        _constant_emit("amd.xdna.aie2p.constant.i32", bits_value, bits),
        _inferred_emit(
            "amd.xdna.aie2p.splat.i16x32",
            operands={"src": bits_value},
            result=result,
        ),
    ], result


def _scalar_f32_to_bf16x32_emits(
    input_value: ValueRef,
    prefix: str,
) -> tuple[list[ContractEmit], ValueRef]:
    input_vector = ValueRef.temporary(f"{prefix}_f32x16")
    input_accumulator = ValueRef.temporary(f"{prefix}_accumulator")
    narrow = ValueRef.temporary(f"{prefix}_bf16x16")
    result = ValueRef.temporary(f"{prefix}_bf16x32")
    return [
        _inferred_emit(
            "amd.xdna.aie2p.splat.i32x16",
            operands={"src": input_value},
            result=input_vector,
        ),
        _inferred_emit(
            "amd.xdna.aie2p.move.vector512.to.accumulator512",
            operands={"src": input_vector},
            result=input_accumulator,
        ),
        _emit(
            "amd.xdna.aie2p.state.rounding.immediate",
            immediates={"i": BF16_CONVERSION_ROUNDING},
        ),
        _inferred_emit(
            "amd.xdna.aie2p.convert.f32x16.to.bf16x16",
            operands={"src": input_accumulator},
            result=narrow,
        ),
        EmitRegisterConcat(
            sources=(narrow, narrow),
            result=result,
            result_type=_BF16X32_VECTOR,
        ),
    ], result


def _radians_to_turns_emits() -> tuple[list[ContractEmit], ValueRef]:
    emits, radians = _scalar_f32_to_bf16x32_emits(ValueRef.operand("input"), "radians")
    coefficient_emits, inverse_two_pi = _bf16_splat_emits(
        "inverse_two_pi", _INVERSE_TWO_PI_BF16_BITS
    )
    emits.extend(coefficient_emits)
    multiply_control = ValueRef.temporary("radian_scale_control")
    scaled_accumulator = ValueRef.temporary("scaled_turns_accumulator")
    scaled_accumulator_unit = ValueRef.temporary("scaled_turns_accumulator_unit")
    scaled_vector = ValueRef.temporary("scaled_turns_vector")
    turns = ValueRef.temporary("turns")
    emits.extend(
        (
            _constant_emit(
                "amd.xdna.aie2p.constant.i32.mova",
                multiply_control,
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.multiply.bf16x32.configured",
                operands={
                    "s1": radians,
                    "s2": inverse_two_pi,
                    "acc": multiply_control,
                },
                result=scaled_accumulator,
            ),
            EmitRegisterSlice(
                source=scaled_accumulator,
                result=scaled_accumulator_unit,
                unit_count=1,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.move.accumulator512.to.vector512",
                operands={"src": scaled_accumulator_unit},
                result=scaled_vector,
            ),
            _emit(
                "amd.xdna.aie2p.extract.i32.immediate",
                operands={"s1": scaled_vector},
                results={"dst": turns},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
            ),
        )
    )
    return emits, turns


def _range_reduce_turns_emits(
    turns: ValueRef,
    operation: _TrigOperation,
) -> tuple[list[ContractEmit], ValueRef, ValueRef]:
    range_program = ScalarProgram("trig_range_")
    rounded = emit_round_nearest_f32_to_i32(range_program, turns, 0, "nearest_integer")
    rounded_float = emit_signed_integer_to_f32(
        range_program, rounded, None, "nearest_integer_float"
    )
    reduced = ValueRef.temporary("trig_reduced")
    emits: list[ContractEmit] = [
        *range_program.emits,
        *emit_f32_scalar_accumulator_binary(
            turns,
            rounded_float,
            reduced,
            "amd.xdna.aie2p.sub.f32x64.configured",
            temporary_prefix="trig_reduce_",
        ),
    ]

    fold_program = ScalarProgram("trig_fold_")
    absolute_mask = fold_program.constant("absolute_mask", 0x7FFFFFFF)
    sign_mask = fold_program.constant("sign_mask", -(2**31))
    absolute = fold_program.binary("absolute", "and.i32", reduced, absolute_mask)
    if operation in (_TrigOperation.SIN_RADIANS, _TrigOperation.SIN_TURNS):
        half = fold_program.constant("half", 0x3F000000)
        quarter = fold_program.constant("quarter", 0x3E800000)
        result_sign = fold_program.binary("sign", "and.i32", reduced, sign_mask)
        emits.extend(fold_program.emits)
        mirrored = ValueRef.temporary("trig_fold_mirrored")
        emits.extend(
            emit_f32_scalar_accumulator_binary(
                half,
                absolute,
                mirrored,
                "amd.xdna.aie2p.sub.f32x64.configured",
                temporary_prefix="trig_mirror_",
            )
        )
        select_program = ScalarProgram("trig_fold_select_")
        use_mirror = select_program.binary(
            "use_mirror", "cmp.ult.i32", quarter, absolute
        )
        argument = select_program.select("argument", mirrored, absolute, use_mirror)
        emits.extend(select_program.emits)
        return emits, argument, result_sign

    quarter = fold_program.constant("quarter", 0x3E800000)
    emits.extend(fold_program.emits)
    signed_argument = ValueRef.temporary("trig_fold_signed_argument")
    emits.extend(
        emit_f32_scalar_accumulator_binary(
            quarter,
            absolute,
            signed_argument,
            "amd.xdna.aie2p.sub.f32x64.configured",
            temporary_prefix="trig_quarter_",
        )
    )
    cosine_program = ScalarProgram("trig_cosine_")
    argument = cosine_program.binary(
        "argument", "and.i32", signed_argument, absolute_mask
    )
    result_sign = cosine_program.binary("sign", "and.i32", signed_argument, sign_mask)
    emits.extend(cosine_program.emits)
    return emits, argument, result_sign


def _sine_polynomial_emits(
    argument: ValueRef,
    result: ValueRef,
) -> tuple[ContractEmit, ...]:
    emits, argument_vector = _scalar_f32_to_bf16x32_emits(
        argument, "trig_polynomial_argument"
    )
    multiply_control = ValueRef.temporary("trig_polynomial_control")
    emits.append(
        _constant_emit(
            "amd.xdna.aie2p.constant.i32.mova",
            multiply_control,
            _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
        )
    )

    argument_squared_accumulator = ValueRef.temporary(
        "trig_polynomial_argument_squared_accumulator"
    )
    argument_squared_narrow = ValueRef.temporary(
        "trig_polynomial_argument_squared_narrow"
    )
    argument_squared = ValueRef.temporary("trig_polynomial_argument_squared")
    emits.extend(
        (
            _inferred_emit(
                "amd.xdna.aie2p.multiply.bf16x32.configured",
                operands={
                    "s1": argument_vector,
                    "s2": argument_vector,
                    "acc": multiply_control,
                },
                result=argument_squared_accumulator,
            ),
            EmitRegisterSlice(
                source=argument_squared_accumulator,
                result=argument_squared_narrow,
                unit_count=2,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.convert.f32x32.to.bf16x32",
                operands={"src": argument_squared_narrow},
                result=argument_squared,
            ),
        )
    )

    one_emits, one = _bf16_splat_emits("trig_polynomial_one", 0x3F80)
    emits.extend(one_emits)
    coefficient_vectors: dict[int, ValueRef] = {}

    def coefficient(index: int) -> ValueRef:
        coefficient_vector = coefficient_vectors.get(index)
        if coefficient_vector is None:
            coefficient_emits, coefficient_vector = _bf16_splat_emits(
                f"trig_polynomial_coefficient_{index}",
                _SINE_POLYNOMIAL_BF16_BITS[index],
            )
            emits.extend(coefficient_emits)
            coefficient_vectors[index] = coefficient_vector
        return coefficient_vector

    polynomial = coefficient(len(_SINE_POLYNOMIAL_BF16_BITS) - 1)
    for coefficient_index in range(len(_SINE_POLYNOMIAL_BF16_BITS) - 2, -1, -1):
        coefficient_accumulator = ValueRef.temporary(
            f"trig_polynomial_coefficient_{coefficient_index}_accumulator"
        )
        polynomial_accumulator = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}_accumulator"
        )
        polynomial_narrow = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}_narrow"
        )
        next_polynomial = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}"
        )
        emits.extend(
            (
                _inferred_emit(
                    "amd.xdna.aie2p.multiply.bf16x32.configured",
                    operands={
                        "s1": coefficient(coefficient_index),
                        "s2": one,
                        "acc": multiply_control,
                    },
                    result=coefficient_accumulator,
                ),
                _inferred_emit(
                    "amd.xdna.aie2p.accumulate.bf16x32.configured",
                    operands={
                        "acc1": coefficient_accumulator,
                        "s1": argument_squared,
                        "s2": polynomial,
                        "acc": multiply_control,
                    },
                    result=polynomial_accumulator,
                ),
                EmitRegisterSlice(
                    source=polynomial_accumulator,
                    result=polynomial_narrow,
                    unit_count=2,
                ),
                _inferred_emit(
                    "amd.xdna.aie2p.convert.f32x32.to.bf16x32",
                    operands={"src": polynomial_narrow},
                    result=next_polynomial,
                ),
            )
        )
        polynomial = next_polynomial

    result_accumulator = ValueRef.temporary("trig_polynomial_result_accumulator")
    result_accumulator_unit = ValueRef.temporary(
        "trig_polynomial_result_accumulator_unit"
    )
    result_vector = ValueRef.temporary("trig_polynomial_result_vector")
    emits.extend(
        (
            _inferred_emit(
                "amd.xdna.aie2p.multiply.bf16x32.configured",
                operands={
                    "s1": argument_vector,
                    "s2": polynomial,
                    "acc": multiply_control,
                },
                result=result_accumulator,
            ),
            EmitRegisterSlice(
                source=result_accumulator,
                result=result_accumulator_unit,
                unit_count=1,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.move.accumulator512.to.vector512",
                operands={"src": result_accumulator_unit},
                result=result_vector,
            ),
            _emit(
                "amd.xdna.aie2p.extract.i32.immediate",
                operands={"s1": result_vector},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
            ),
        )
    )
    return tuple(emits)


def _trig_emits(operation: _TrigOperation) -> tuple[ContractEmit, ...]:
    if operation in (_TrigOperation.SIN_RADIANS, _TrigOperation.COS_RADIANS):
        emits, turns = _radians_to_turns_emits()
    else:
        emits = []
        turns = ValueRef.operand("input")

    reduction_emits, argument, result_sign = _range_reduce_turns_emits(turns, operation)
    emits.extend(reduction_emits)
    polynomial_result = ValueRef.temporary("trig_polynomial_result")
    emits.extend(_sine_polynomial_emits(argument, polynomial_result))

    result_program = ScalarProgram("trig_result_")
    absolute_mask = result_program.constant("absolute_mask", 0x7FFFFFFF)
    sign_mask = result_program.constant("sign_mask", -(2**31))
    large_threshold = result_program.constant("large_threshold", 0x4B000000)
    infinity = result_program.constant("infinity", 0x7F800000)
    canonical_nan = result_program.constant("canonical_nan", 0x7FC00000)
    one = result_program.constant("one", 0x3F800000)
    input_absolute = result_program.binary(
        "input_absolute", "and.i32", ValueRef.operand("input"), absolute_mask
    )
    turns_absolute = result_program.binary(
        "turns_absolute", "and.i32", turns, absolute_mask
    )
    input_sign = result_program.binary(
        "input_sign", "and.i32", ValueRef.operand("input"), sign_mask
    )
    signed_result = result_program.binary(
        "signed", "xor.i32", polynomial_result, result_sign
    )
    large = result_program.binary(
        "large", "cmp.uge.i32", turns_absolute, large_threshold
    )
    invalid = result_program.binary("invalid", "cmp.uge.i32", input_absolute, infinity)
    large_result = (
        one
        if operation in (_TrigOperation.COS_RADIANS, _TrigOperation.COS_TURNS)
        else input_sign
    )
    finite_result = result_program.select("finite", large_result, signed_result, large)
    result_program.select(None, canonical_nan, finite_result, invalid)
    emits.extend(result_program.emits)
    return tuple(emits)


def _trig_rule(
    source_op: Op,
    operation: _TrigOperation,
) -> DescriptorRule:
    descriptor_key = "amd.xdna.aie2p.multiply.bf16x32.configured"
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", _F32),
            Guard.instance_flags_has_all("fastmath", "afn"),
        ),
        emit=_trig_emits(operation),
        report_key=f"native_bf16_mac_{operation.value}",
    )


def _input_vector_emits(*, scalar: bool) -> tuple[list[ContractEmit], ValueRef]:
    if not scalar:
        return [], ValueRef.operand("input")
    input_vector = ValueRef.temporary("input_vector")
    return [
        _inferred_emit(
            "amd.xdna.aie2p.splat.i32x16",
            operands={"src": ValueRef.operand("input")},
            result=input_vector,
        )
    ], input_vector


def _result_emits(*, scalar: bool) -> tuple[ContractEmit, ...]:
    result_vector = (
        ValueRef.temporary("result_vector") if scalar else ValueRef.result("result")
    )
    emits: list[ContractEmit] = [
        _emit(
            "amd.xdna.aie2p.move.accumulator512.to.vector512",
            operands={"src": ValueRef.temporary("result_accumulator")},
            results={"dst": result_vector},
            result_types={"dst": DescriptorResultType()} if scalar else None,
        )
    ]
    if scalar:
        emits.append(
            _emit(
                "amd.xdna.aie2p.extract.i32.immediate",
                operands={"s1": result_vector},
                results={"dst": ValueRef.result("result")},
                immediates={"idx": 0},
            )
        )
    return tuple(emits)


def _native_nonlinear_emits(
    native_descriptor_key: str,
    *,
    scalar: bool,
) -> tuple[ContractEmit, ...]:
    emits, input_vector = _input_vector_emits(scalar=scalar)
    emits.extend(
        (
            _inferred_emit(
                "amd.xdna.aie2p.move.vector512.to.accumulator512",
                operands={"src": input_vector},
                result=ValueRef.temporary("input_accumulator"),
            ),
            _inferred_emit(
                native_descriptor_key,
                operands={"src": ValueRef.temporary("input_accumulator")},
                result=ValueRef.temporary("bf16_result"),
            ),
            _inferred_emit(
                "amd.xdna.aie2p.convert.bf16x16.to.f32x16",
                operands={"src": ValueRef.temporary("bf16_result")},
                result=ValueRef.temporary("result_accumulator"),
            ),
            *_result_emits(scalar=scalar),
        )
    )
    return tuple(emits)


def _native_exp_emits(*, scalar: bool) -> tuple[ContractEmit, ...]:
    emits, input_vector = _input_vector_emits(scalar=scalar)
    bf16_input = ValueRef.temporary("bf16_input")
    emits.extend(
        (
            _inferred_emit(
                "amd.xdna.aie2p.move.vector512.to.accumulator512",
                operands={"src": input_vector},
                result=ValueRef.temporary("input_accumulator"),
            ),
            _emit(
                "amd.xdna.aie2p.state.rounding.immediate",
                immediates={"i": BF16_CONVERSION_ROUNDING},
            ),
            _inferred_emit(
                "amd.xdna.aie2p.convert.f32x16.to.bf16x16",
                operands={"src": ValueRef.temporary("input_accumulator")},
                result=bf16_input,
            ),
            EmitRegisterConcat(
                sources=(bf16_input, bf16_input),
                result=ValueRef.temporary("bf16_input_x32"),
                result_type=_BF16X32_VECTOR,
            ),
            _constant_emit(
                "amd.xdna.aie2p.constant.i32",
                ValueRef.temporary("log2_e_bits"),
                _LOG2_E_BF16_BITS,
            ),
        )
    )
    emits.extend(
        (
            _inferred_emit(
                "amd.xdna.aie2p.splat.i16x32",
                operands={"src": ValueRef.temporary("log2_e_bits")},
                result=ValueRef.temporary("log2_e"),
            ),
            _constant_emit(
                "amd.xdna.aie2p.constant.i32.mova",
                ValueRef.temporary("multiply_control"),
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.multiply.bf16x32.configured",
                operands={
                    "s1": ValueRef.temporary("bf16_input_x32"),
                    "s2": ValueRef.temporary("log2_e"),
                    "acc": ValueRef.temporary("multiply_control"),
                },
                result=ValueRef.temporary("scaled_accumulator_wide"),
            ),
            EmitRegisterSlice(
                source=ValueRef.temporary("scaled_accumulator_wide"),
                result=ValueRef.temporary("scaled_accumulator"),
                unit_count=1,
            ),
            _inferred_emit(
                "amd.xdna.aie2p.exp2.f32x16.to.bf16x16",
                operands={"src": ValueRef.temporary("scaled_accumulator")},
                result=ValueRef.temporary("bf16_result"),
            ),
            _inferred_emit(
                "amd.xdna.aie2p.convert.bf16x16.to.f32x16",
                operands={"src": ValueRef.temporary("bf16_result")},
                result=ValueRef.temporary("result_accumulator"),
            ),
            *_result_emits(scalar=scalar),
        )
    )
    return tuple(emits)


def _native_nonlinear_rule(
    source_op: Op,
    type_pattern: TypePattern,
    native_descriptor_key: str,
    report_key: str,
    *,
    scalar: bool,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, native_descriptor_key),
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", type_pattern),
            Guard.instance_flags_has_all("fastmath", "afn"),
        ),
        emit=_native_nonlinear_emits(native_descriptor_key, scalar=scalar),
        report_key=report_key,
    )


def _native_exp_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    scalar: bool,
) -> DescriptorRule:
    descriptor_key = "amd.xdna.aie2p.exp2.f32x16.to.bf16x16"
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", type_pattern),
            Guard.instance_flags_has_all("fastmath", "afn"),
        ),
        emit=_native_exp_emits(scalar=scalar),
        report_key=(
            "native_bf16_exp_scalar_f32" if scalar else "native_bf16_exp_vector_f32"
        ),
    )


_VECTOR_NONLINEAR_RULES = tuple(
    _native_nonlinear_rule(
        source_op,
        type_pattern,
        native_descriptor_key,
        report_key,
        scalar=is_scalar,
    )
    for source_op, type_pattern, native_descriptor_key, report_key, is_scalar in (
        (
            scalar_math.scalar_exp2f,
            _F32,
            "amd.xdna.aie2p.exp2.f32x16.to.bf16x16",
            "native_bf16_exp2_scalar_f32",
            True,
        ),
        (
            vector.vector_exp2f,
            _F32_VECTOR,
            "amd.xdna.aie2p.exp2.f32x16.to.bf16x16",
            "native_bf16_exp2_vector_f32",
            False,
        ),
        (
            scalar_math.scalar_tanhf,
            _F32,
            "amd.xdna.aie2p.tanh.f32x16.to.bf16x16",
            "native_bf16_tanh_scalar_f32",
            True,
        ),
        (
            vector.vector_tanhf,
            _F32_VECTOR,
            "amd.xdna.aie2p.tanh.f32x16.to.bf16x16",
            "native_bf16_tanh_vector_f32",
            False,
        ),
    )
)

_SCALAR_NONLINEAR_RULES = tuple(
    DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", _F32),
            Guard.instance_flags_has_all("fastmath", "afn"),
        ),
        emit=(
            _emit(
                descriptor_key,
                operands={"s0": ValueRef.operand("input")},
                results={"d0": ValueRef.result("result")},
            ),
        ),
        report_key=report_key,
    )
    for source_op, descriptor_key, report_key in (
        (
            scalar_math.scalar_sqrtf,
            "amd.xdna.aie2p.sqrt.f32",
            "native_approximate_sqrt_f32",
        ),
        (
            scalar_math.scalar_rsqrtf,
            "amd.xdna.aie2p.reciprocal-sqrt.f32",
            "native_approximate_reciprocal_sqrt_f32",
        ),
    )
)

AIE2P_NONLINEAR_RULES = (
    _trig_rule(scalar_math.scalar_sinf, _TrigOperation.SIN_RADIANS),
    _trig_rule(scalar_math.scalar_cosf, _TrigOperation.COS_RADIANS),
    _trig_rule(scalar_math.scalar_sinturnsf, _TrigOperation.SIN_TURNS),
    _trig_rule(scalar_math.scalar_costurnsf, _TrigOperation.COS_TURNS),
    _native_exp_rule(scalar_math.scalar_expf, _F32, scalar=True),
    _native_exp_rule(vector.vector_expf, _F32_VECTOR, scalar=False),
    *_VECTOR_NONLINEAR_RULES,
    *_SCALAR_NONLINEAR_RULES,
)
