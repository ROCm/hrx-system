# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P native floating-point selection rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    emit_f16_to_f32,
    emit_f32_to_f16,
)
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.contracts.f32 import emit_f32_multiply
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
from loom.target.low_descriptors import Descriptor

_F32 = Scalar("f32")
_F16 = Scalar("f16")
_BF16X8_VECTOR = Vector("bf16", lanes=8)
_BF16_DOT2_VECTOR = Vector(
    "bf16", minimum_static_elements=2, maximum_static_elements=32
)
_BF16X32_VECTOR = Vector("bf16", lanes=32)
_BF16X64_VECTOR = Vector("bf16", lanes=64)
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_F32X4_VECTOR = Vector("f32", lanes=4)
_F32X16_VECTOR = Vector("f32", lanes=16)
_F32X64_ACCUMULATOR = Vector("f32", lanes=64)

_BF16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=2,
    multiplication_mode=3,
    compute_mode=1,
)
# AIE2P T16_32x2_lo/hi select the even and odd BF16 lanes from a 512-bit
# source. Two ordered VMACs over those streams implement vector.dot2f's two
# sequential fused accumulations without weakening its exact source contract.
_BF16_DOT2_DEINTERLEAVE_CONTROLS = (2, 3)

_BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS = (52, 53)
_BF16_OUTER_PRODUCT_MULTIPLY_CONTROL = _BF16_ELEMENTWISE_MULTIPLY_CONTROL
_F32_ACCUMULATOR_ADD_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=2,
    multiplication_mode=3,
    compute_mode=1,
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: Iterable[str], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: Mapping[str, ValueRef] | None = None,
    results: Mapping[str, ValueRef] | None = None,
    result_types: Mapping[str, ResultTypeBinding] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        form=DescriptorEmitForm.OP,
    )


def _constant_emit(
    descriptor: Descriptor,
    result: ValueRef,
    value: int,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _vector_multiply_bf16x32_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    multiply = _descriptor("amd.xdna.aie2p.multiply.bf16x32.configured")
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    convert = _descriptor("amd.xdna.aie2p.convert.f32x32.to.bf16x32")
    return DescriptorRule(
        source_op=vector.vector_mulf,
        descriptor=convert,
        guards=_typed_guards(("lhs", "rhs", "result"), _BF16X32_VECTOR),
        emit=(
            _constant_emit(
                config_constant,
                ValueRef.temporary("multiply_control"),
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _op_emit(
                multiply,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                    "acc": ValueRef.temporary("multiply_control"),
                },
                results={"dst": ValueRef.temporary("wide_product")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterSlice(
                source=ValueRef.temporary("wide_product"),
                result=ValueRef.temporary("product"),
                unit_count=2,
            ),
            EmitDescriptorOp(
                descriptor=set_rounding,
                immediates={"i": BF16_CONVERSION_ROUNDING},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                convert,
                operands={"src": ValueRef.temporary("product")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_multiply_f16_rule() -> DescriptorRule:
    """Multiplies binary16 exactly through the binary32 software datapath."""

    # A binary16 product has at most 22 significant bits and remains within
    # binary32's exponent range, so widening both operands and multiplying in
    # binary32 computes the exact finite product before the final narrowing.
    lhs_program = ScalarProgram("lhs_")
    lhs = emit_f16_to_f32(
        lhs_program,
        ValueRef.operand("lhs"),
        "wide",
    )
    rhs_program = ScalarProgram("rhs_")
    rhs = emit_f16_to_f32(
        rhs_program,
        ValueRef.operand("rhs"),
        "wide",
    )
    multiply_emits = emit_f32_multiply(
        lhs=lhs,
        rhs=rhs,
        result_name="result",
        temporary_prefix="multiply_",
    )
    product = ValueRef.temporary("multiply_result")
    narrow_program = ScalarProgram("narrow_")
    emit_f32_to_f16(narrow_program, product, None)
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_mulf,
        descriptor=narrow_program.emits[-1].descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _F16),
        emit=(
            *lhs_program.emits,
            *rhs_program.emits,
            *multiply_emits,
            *narrow_program.emits,
        ),
        report_key="exact_binary16",
    )


def _vector_dot2f_bf16_rule(
    input_type: TypePattern,
    result_type: TypePattern,
    *,
    broadcast_inputs: bool,
    report_key: str,
) -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    broadcast = _descriptor("amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    clear = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    move_to_accumulator = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
    accumulate = _descriptor("amd.xdna.aie2p.accumulate.bf16x32.configured")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )

    emits: list[ContractEmit] = []
    input_values = {
        operand_name: ValueRef.operand(operand_name) for operand_name in ("lhs", "rhs")
    }
    if broadcast_inputs:
        for operand_name in ("lhs", "rhs"):
            broadcast_value = ValueRef.temporary(f"{operand_name}_broadcast")
            emits.append(
                EmitDescriptorOp(
                    descriptor=broadcast,
                    operands={"s1": ValueRef.operand(operand_name)},
                    results={"dst": broadcast_value},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"idx": 0},
                    form=DescriptorEmitForm.OP,
                )
            )
            input_values[operand_name] = broadcast_value
    for lane_group, control in zip(
        ("even", "odd"), _BF16_DOT2_DEINTERLEAVE_CONTROLS, strict=True
    ):
        control_value = ValueRef.temporary(f"{lane_group}_control")
        emits.append(_constant_emit(config_constant, control_value, control))
        emits.extend(
            _op_emit(
                shuffle,
                operands={
                    "s1": input_values[operand_name],
                    "s2": input_values[operand_name],
                    "mod": control_value,
                },
                results={"dst": ValueRef.temporary(f"{operand_name}_{lane_group}")},
                result_types={"dst": DescriptorResultType()},
            )
            for operand_name in ("lhs", "rhs")
        )

    emits.extend(
        (
            _op_emit(
                clear,
                results={"dst": ValueRef.temporary("zero_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            *(
                EmitRegisterSlice(
                    source=ValueRef.temporary("zero_accumulator"),
                    result=ValueRef.temporary(f"zero_accumulator_unit_{unit}"),
                    unit_offset=unit,
                    unit_count=1,
                )
                for unit in range(1, 4)
            ),
            _op_emit(
                move_to_accumulator,
                operands={"src": ValueRef.operand("acc")},
                results={"dst": ValueRef.temporary("initial_accumulator_unit")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("initial_accumulator_unit"),
                    ValueRef.temporary("zero_accumulator_unit_1"),
                    ValueRef.temporary("zero_accumulator_unit_2"),
                    ValueRef.temporary("zero_accumulator_unit_3"),
                ),
                result=ValueRef.temporary("initial_accumulator"),
                result_type=_F32X64_ACCUMULATOR,
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("accumulate_control"),
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _op_emit(
                accumulate,
                operands={
                    "acc1": ValueRef.temporary("initial_accumulator"),
                    "s1": ValueRef.temporary("lhs_even"),
                    "s2": ValueRef.temporary("rhs_even"),
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("even_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                accumulate,
                operands={
                    "acc1": ValueRef.temporary("even_accumulator"),
                    "s1": ValueRef.temporary("lhs_odd"),
                    "s2": ValueRef.temporary("rhs_odd"),
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("result_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterSlice(
                source=ValueRef.temporary("result_accumulator"),
                result=ValueRef.temporary("result_accumulator_unit"),
                unit_count=1,
            ),
            _op_emit(
                move_from_accumulator,
                operands={"src": ValueRef.temporary("result_accumulator_unit")},
                results={"dst": ValueRef.result("result")},
            ),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=accumulate,
        guards=(
            Guard.value_type("lhs", input_type),
            Guard.value_type("rhs", input_type),
            Guard.value_type("acc", result_type),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(emits),
        report_key=report_key,
    )


def _matrix_multiply_bf16bf16_m8n8k1_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    broadcast = _descriptor("amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    move = _descriptor("amd.xdna.aie2p.move.vector512")
    multiply = _descriptor(
        "amd.xdna.aie2p.matrix.accumulate.bf16bf16.m8n8k1.configured"
    )
    return DescriptorRule(
        source_op=vector.vector_mma,
        descriptor=multiply,
        guards=(
            Guard.value_type("lhs", _BF16X8_VECTOR),
            Guard.value_type("rhs", _BF16X8_VECTOR),
            Guard.value_type("init", _F32X64_ACCUMULATOR),
            Guard.value_type("result", _F32X64_ACCUMULATOR),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=broadcast,
                operands={"s1": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_broadcast")},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("lhs_shuffle_even_control"),
                _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[0],
            ),
            _op_emit(
                shuffle,
                operands={
                    "s1": ValueRef.temporary("lhs_broadcast"),
                    "s2": ValueRef.temporary("lhs_broadcast"),
                    "mod": ValueRef.temporary("lhs_shuffle_even_control"),
                },
                results={"dst": ValueRef.temporary("lhs_rows_even")},
                result_types={"dst": DescriptorResultType()},
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("lhs_shuffle_odd_control"),
                _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[1],
            ),
            _op_emit(
                shuffle,
                operands={
                    "s1": ValueRef.temporary("lhs_broadcast"),
                    "s2": ValueRef.temporary("lhs_broadcast"),
                    "mod": ValueRef.temporary("lhs_shuffle_odd_control"),
                },
                results={"dst": ValueRef.temporary("lhs_rows_odd")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("lhs_rows_even"),
                    ValueRef.temporary("lhs_rows_odd"),
                ),
                result=ValueRef.temporary("lhs_rows"),
                result_type=_BF16X64_VECTOR,
            ),
            EmitDescriptorOp(
                descriptor=broadcast,
                operands={"s1": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_columns_low")},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                move,
                operands={"src": ValueRef.temporary("rhs_columns_low")},
                results={"dst": ValueRef.temporary("rhs_columns_high")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("rhs_columns_low"),
                    ValueRef.temporary("rhs_columns_high"),
                ),
                result=ValueRef.temporary("rhs_columns"),
                result_type=_BF16X64_VECTOR,
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("multiply_control"),
                _BF16_OUTER_PRODUCT_MULTIPLY_CONTROL,
            ),
            _op_emit(
                multiply,
                operands={
                    "acc1": ValueRef.operand("init"),
                    "s1": ValueRef.temporary("lhs_rows"),
                    "s2": ValueRef.temporary("rhs_columns"),
                    "acc": ValueRef.temporary("multiply_control"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
        report_key="bf16bf16_m8n8k1",
    )


def _float_matrix_accumulator_zero_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", _F32X64_ACCUMULATOR),
            Guard.value_float_equals("result", 0.0),
        ),
        emit=(
            _op_emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _float_matrix_accumulator_add_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    add = _descriptor("amd.xdna.aie2p.add.f32x64.configured")
    return DescriptorRule(
        source_op=vector.vector_addf,
        descriptor=add,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32X64_ACCUMULATOR),
        emit=(
            _constant_emit(
                config_constant,
                ValueRef.temporary("add_control"),
                _F32_ACCUMULATOR_ADD_CONTROL,
            ),
            _op_emit(
                add,
                operands={
                    "acc1": ValueRef.operand("lhs"),
                    "acc2": ValueRef.operand("rhs"),
                    "acc": ValueRef.temporary("add_control"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _float_accumulator_binary_emits(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation_descriptor_key: str,
    *,
    extract_scalar_result: bool,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    clear = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    move_to_accumulator = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    operation = _descriptor(operation_descriptor_key)
    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    final_vector = temporary("result_vector") if extract_scalar_result else result
    final_vector_result_types = (
        {"dst": DescriptorResultType()} if extract_scalar_result else None
    )
    emits: list[ContractEmit] = [
        _op_emit(
            clear,
            results={"dst": temporary("zero_accumulator")},
            result_types={"dst": DescriptorResultType()},
        ),
        EmitRegisterSlice(
            source=temporary("zero_accumulator"),
            result=temporary("zero_accumulator_unit"),
            unit_count=1,
        ),
    ]
    for operand_name, operand in (("lhs", lhs), ("rhs", rhs)):
        accumulator_unit = temporary(f"{operand_name}_accumulator_unit")
        emits.extend(
            (
                _op_emit(
                    move_to_accumulator,
                    operands={"src": operand},
                    results={"dst": accumulator_unit},
                    result_types={"dst": DescriptorResultType()},
                ),
                EmitRegisterConcat(
                    sources=(
                        accumulator_unit,
                        temporary("zero_accumulator_unit"),
                        temporary("zero_accumulator_unit"),
                        temporary("zero_accumulator_unit"),
                    ),
                    result=temporary(f"{operand_name}_accumulator"),
                    result_type=_F32X64_ACCUMULATOR,
                ),
            )
        )
    emits.extend(
        (
            _constant_emit(
                config_constant,
                temporary("arithmetic_control"),
                _F32_ACCUMULATOR_ADD_CONTROL,
            ),
            _op_emit(
                operation,
                operands={
                    "acc1": temporary("lhs_accumulator"),
                    "acc2": temporary("rhs_accumulator"),
                    "acc": temporary("arithmetic_control"),
                },
                results={"dst": temporary("result_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterSlice(
                source=temporary("result_accumulator"),
                result=temporary("result_accumulator_unit"),
                unit_count=1,
            ),
            _op_emit(
                move_from_accumulator,
                operands={"src": temporary("result_accumulator_unit")},
                results={"dst": final_vector},
                result_types=final_vector_result_types,
            ),
        )
    )
    if extract_scalar_result:
        emits.append(
            EmitDescriptorOp(
                descriptor=extract,
                operands={"s1": final_vector},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            )
        )
    return tuple(emits)


def emit_f32_scalar_accumulator_binary(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation_descriptor_key: str,
    *,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Builds one scalar binary32 add/sub through the native accumulator."""

    broadcast = _descriptor("amd.xdna.aie2p.splat.i32x16")

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    return (
        _op_emit(
            broadcast,
            operands={"src": lhs},
            results={"dst": temporary("lhs_vector")},
            result_types={"dst": DescriptorResultType()},
        ),
        _op_emit(
            broadcast,
            operands={"src": rhs},
            results={"dst": temporary("rhs_vector")},
            result_types={"dst": DescriptorResultType()},
        ),
        *_float_accumulator_binary_emits(
            temporary("lhs_vector"),
            temporary("rhs_vector"),
            result,
            operation_descriptor_key,
            extract_scalar_result=True,
            temporary_prefix=temporary_prefix,
        ),
    )


def _float_vector_accumulator_binary_rule(
    source_op: Op,
    operation_descriptor_key: str,
) -> DescriptorRule:
    operation = _descriptor(operation_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=operation,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32_VECTOR),
        emit=_float_accumulator_binary_emits(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation_descriptor_key,
            extract_scalar_result=False,
        ),
    )


def _float_scalar_accumulator_binary_rule(
    source_op: Op,
    operation_descriptor_key: str,
) -> DescriptorRule:
    operation = _descriptor(operation_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=operation,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32),
        emit=emit_f32_scalar_accumulator_binary(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation_descriptor_key,
        ),
    )


AIE2P_BF16_MATRIX_RULES = (_matrix_multiply_bf16bf16_m8n8k1_rule(),)

AIE2P_FLOATING_RULES = (
    _scalar_multiply_f16_rule(),
    _vector_multiply_bf16x32_rule(),
    _float_matrix_accumulator_zero_rule(),
    _float_matrix_accumulator_add_rule(),
    *(
        _float_vector_accumulator_binary_rule(source_op, descriptor_key)
        for source_op, descriptor_key in (
            (vector.vector_addf, "amd.xdna.aie2p.add.f32x64.configured"),
            (vector.vector_subf, "amd.xdna.aie2p.sub.f32x64.configured"),
        )
    ),
    *(
        _float_scalar_accumulator_binary_rule(source_op, descriptor_key)
        for source_op, descriptor_key in (
            (
                scalar_arithmetic.scalar_addf,
                "amd.xdna.aie2p.add.f32x64.configured",
            ),
            (
                scalar_arithmetic.scalar_subf,
                "amd.xdna.aie2p.sub.f32x64.configured",
            ),
        )
    ),
    # A vector<8xbf16> is the native outer-product operand type and therefore
    # uses the narrow EWL carrier. Broadcast it into the ordinary X carrier
    # before using the same VMAC realization. Specialized rules precede ranged
    # rules.
    _vector_dot2f_bf16_rule(
        _BF16X8_VECTOR,
        _F32X4_VECTOR,
        broadcast_inputs=True,
        report_key="bf16_dot2_x8_broadcast",
    ),
    _vector_dot2f_bf16_rule(
        _BF16_DOT2_VECTOR,
        _F32_VECTOR,
        broadcast_inputs=False,
        report_key="bf16_dot2",
    ),
)
