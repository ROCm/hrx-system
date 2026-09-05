# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P packed integer dot-product selection rules."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    vector_data_path_control,
)
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
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8X64_VECTOR = Vector("i8", lanes=64)
_I8X128_VECTOR = Vector("i8", lanes=128)
# Each 256-bit chunk feeds eight independent four-lane channels. The source op
# verifier requires complete groups of four, so physical padding can affect only
# discarded result channels when a logical vector occupies part of a chunk.
_I8_DOT4_LOW_VECTOR = Vector(
    "i8", minimum_static_elements=4, maximum_static_elements=32
)
_I8_DOT4_HIGH_VECTOR = Vector(
    "i8", minimum_static_elements=36, maximum_static_elements=64
)
_I32_DOT4_LOW_VECTOR = Vector(
    "i32", minimum_static_elements=1, maximum_static_elements=8
)
_I32_DOT4_HIGH_VECTOR = Vector(
    "i32", minimum_static_elements=9, maximum_static_elements=16
)
_I32X16_VECTOR = Vector("i32", lanes=16)

# Transposes eight consecutive four-byte rows into four eight-byte channel
# rows. The channel multiply consumes replicated data in a Y register and
# zero-padded coefficients in an X register to produce eight dot4 results.
_DOT4_TRANSPOSE_CONTROL = 46
_DOT4_GROW_CONTROL = 10


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _constant_emit(
    descriptor: Descriptor, result: ValueRef, value: int
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: dict[str, ValueRef],
    results: dict[str, ValueRef],
    descriptor_result_type: bool = False,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands=operands,
        results=results,
        result_types=(
            {name: DescriptorResultType() for name in results}
            if descriptor_result_type
            else None
        ),
        form=DescriptorEmitForm.OP,
    )


def _dot4_chunk_emits(
    chunk_index: int,
    *,
    zero_vector: ValueRef,
    zero_vector_unit: ValueRef,
    transpose_control: ValueRef,
    grow_control: ValueRef,
    multiply_control: ValueRef,
) -> tuple[ContractEmit, ...]:
    chunk_name = "low" if chunk_index == 0 else "high"
    emits: list[ContractEmit] = []

    for operand_name in ("lhs", "rhs"):
        chunk = ValueRef.temporary(f"{operand_name}_{chunk_name}_chunk")
        carrier = ValueRef.temporary(f"{operand_name}_{chunk_name}_carrier")
        transpose = ValueRef.temporary(f"{operand_name}_{chunk_name}_transpose")
        emits.extend(
            (
                EmitRegisterSlice(
                    source=ValueRef.operand(operand_name),
                    result=chunk,
                    unit_offset=chunk_index,
                    unit_count=1,
                ),
                EmitRegisterConcat(
                    sources=(chunk, zero_vector_unit),
                    result=carrier,
                    result_type=_I8X64_VECTOR,
                ),
                _op_emit(
                    _descriptor("amd.xdna.aie2p.shuffle.x.configured"),
                    operands={
                        "s1": carrier,
                        "s2": zero_vector,
                        "mod": transpose_control,
                    },
                    results={"dst": transpose},
                    descriptor_result_type=True,
                ),
            )
        )

    coefficient_transpose = ValueRef.temporary(f"lhs_{chunk_name}_transpose")
    coefficient_unit = ValueRef.temporary(f"lhs_{chunk_name}_transpose_unit")
    coefficients = ValueRef.temporary(f"lhs_{chunk_name}_coefficients")
    data_transpose = ValueRef.temporary(f"rhs_{chunk_name}_transpose")
    data_grown = ValueRef.temporary(f"rhs_{chunk_name}_data_grown")
    data_copy = ValueRef.temporary(f"rhs_{chunk_name}_data_copy")
    data = ValueRef.temporary(f"rhs_{chunk_name}_data")
    emits.extend(
        (
            EmitRegisterSlice(
                source=coefficient_transpose,
                result=coefficient_unit,
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(coefficient_unit, zero_vector_unit),
                result=coefficients,
                result_type=_I8X64_VECTOR,
            ),
            _op_emit(
                _descriptor("amd.xdna.aie2p.shuffle.x.configured"),
                operands={
                    "s1": data_transpose,
                    "s2": data_transpose,
                    "mod": grow_control,
                },
                results={"dst": data_grown},
                descriptor_result_type=True,
            ),
            _op_emit(
                _descriptor("amd.xdna.aie2p.move.vector512"),
                operands={"src": data_grown},
                results={"dst": data_copy},
                descriptor_result_type=True,
            ),
            EmitRegisterConcat(
                sources=(data_grown, data_copy),
                result=data,
                result_type=_I8X128_VECTOR,
            ),
        )
    )

    product_accumulator = ValueRef.temporary(f"product_{chunk_name}_accumulator")
    product_accumulator_unit = ValueRef.temporary(
        f"product_{chunk_name}_accumulator_unit"
    )
    product_vector = ValueRef.temporary(f"product_{chunk_name}_vector")
    product_chunk = ValueRef.temporary(f"product_{chunk_name}_chunk")
    emits.extend(
        (
            _op_emit(
                _descriptor("amd.xdna.aie2p.dot4i.i8x64.configured"),
                operands={
                    "s1": data,
                    "s2": coefficients,
                    "acc": multiply_control,
                },
                results={"dst": product_accumulator},
                descriptor_result_type=True,
            ),
            EmitRegisterSlice(
                source=product_accumulator,
                result=product_accumulator_unit,
                unit_count=1,
            ),
            _op_emit(
                _descriptor("amd.xdna.aie2p.move.accumulator512.to.vector512"),
                operands={"src": product_accumulator_unit},
                results={"dst": product_vector},
                descriptor_result_type=True,
            ),
            EmitRegisterSlice(
                source=product_vector,
                result=product_chunk,
                unit_count=1,
            ),
        )
    )
    return tuple(emits)


def _dot4i_i8_rule(
    kind: str,
    input_type: TypePattern,
    result_type: TypePattern,
    chunk_count: int,
    *,
    lhs_signed: bool,
    rhs_signed: bool,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    splat = _descriptor("amd.xdna.aie2p.splat.i8x64")
    add = _descriptor("amd.xdna.aie2p.add.i32x16")
    zero_scalar = ValueRef.temporary("zero_scalar")
    zero_vector = ValueRef.temporary("zero_vector")
    zero_vector_unit = ValueRef.temporary("zero_vector_unit")
    transpose_control = ValueRef.temporary("transpose_control")
    grow_control = ValueRef.temporary("grow_control")
    multiply_control = ValueRef.temporary("multiply_control")

    multiply_control_value = vector_data_path_control(
        sign_x=rhs_signed,
        sign_y=lhs_signed,
        accumulator_mode=0,
        multiplication_mode=1,
        compute_mode=2,
    )
    emits: list[ContractEmit] = [
        _constant_emit(constant, zero_scalar, 0),
        _op_emit(
            splat,
            operands={"src": zero_scalar},
            results={"dst": zero_vector},
            descriptor_result_type=True,
        ),
        EmitRegisterSlice(
            source=zero_vector,
            result=zero_vector_unit,
            unit_count=1,
        ),
        _constant_emit(constant, transpose_control, _DOT4_TRANSPOSE_CONTROL),
        _constant_emit(constant, grow_control, _DOT4_GROW_CONTROL),
        _constant_emit(constant, multiply_control, multiply_control_value),
    ]
    for chunk_index in range(chunk_count):
        emits.extend(
            _dot4_chunk_emits(
                chunk_index,
                zero_vector=zero_vector,
                zero_vector_unit=zero_vector_unit,
                transpose_control=transpose_control,
                grow_control=grow_control,
                multiply_control=multiply_control,
            )
        )
    product_chunks = tuple(
        ValueRef.temporary(f"product_{'low' if chunk_index == 0 else 'high'}_chunk")
        for chunk_index in range(chunk_count)
    )
    if chunk_count == 1:
        product_chunks += (zero_vector_unit,)
    emits.extend(
        (
            EmitRegisterConcat(
                sources=product_chunks,
                result=ValueRef.temporary("product"),
                result_type=_I32X16_VECTOR,
            ),
            _op_emit(
                add,
                operands={
                    "s1": ValueRef.temporary("product"),
                    "s2": ValueRef.operand("acc"),
                },
                results={"d": ValueRef.result("result")},
            ),
        )
    )

    return DescriptorRule(
        source_op=vector.vector_dot4i,
        descriptor=_descriptor("amd.xdna.aie2p.dot4i.i8x64.configured"),
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("lhs", input_type),
            Guard.value_type("rhs", input_type),
            Guard.value_type("acc", result_type),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(emits),
        report_key=f"dot4i_{kind}_{chunk_count}x256",
    )


AIE2P_PACKED_DOT_RULES = tuple(
    _dot4i_i8_rule(
        kind,
        input_type,
        result_type,
        chunk_count,
        lhs_signed=lhs_signed,
        rhs_signed=rhs_signed,
    )
    for kind, lhs_signed, rhs_signed in (
        ("u8u8", False, False),
        ("u8s8", False, True),
        ("s8u8", True, False),
        ("s8s8", True, True),
    )
    for input_type, result_type, chunk_count in (
        (_I8_DOT4_LOW_VECTOR, _I32_DOT4_LOW_VECTOR, 1),
        (_I8_DOT4_HIGH_VECTOR, _I32_DOT4_HIGH_VECTOR, 2),
    )
)
