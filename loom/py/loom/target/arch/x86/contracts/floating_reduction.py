# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 floating-point reduction emission."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    EmitDescriptorOp,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_F32 = Scalar("f32")
_V4F32 = Vector("f32", lanes=4)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: Mapping[str, AttrProject | int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
    )


def f32x4_reassociated_reduce_emit_chain(
    input_value: ValueRef,
    descriptor_lookup: _DescriptorLookup,
    *,
    temporary_prefix: str = "",
) -> tuple[EmitDescriptorOp, ...]:
    shuffle = descriptor_lookup("x86.avx2.vpermilps.xmm")
    addps = descriptor_lookup("x86.avx2.vaddps.xmm")
    addss = descriptor_lookup("x86.avx2.vaddss.xmm")

    def temp(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    return (
        _op_emit(
            descriptor=shuffle,
            operands={"source": input_value},
            results={"dst": temp("shuffle0")},
            result_types={"dst": _V4F32},
            immediates={"control": 78},
        ),
        _op_emit(
            descriptor=addps,
            operands={"lhs": input_value, "rhs": temp("shuffle0")},
            results={"dst": temp("pair_sum")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=shuffle,
            operands={"source": temp("pair_sum")},
            results={"dst": temp("shuffle1")},
            result_types={"dst": _V4F32},
            immediates={"control": 177},
        ),
        _op_emit(
            descriptor=addps,
            operands={"lhs": temp("pair_sum"), "rhs": temp("shuffle1")},
            results={"dst": temp("vector_sum")},
            result_types={"dst": _V4F32},
        ),
        _op_emit(
            descriptor=addss,
            operands={
                "lhs": ValueRef.operand("init"),
                "rhs": temp("vector_sum"),
            },
            results={"dst": ValueRef.result("result")},
            result_types={"dst": _F32},
        ),
    )


def ordered_f32_reduce_emit_chain(
    input_values: Sequence[ValueRef],
    descriptor_lookup: _DescriptorLookup,
    *,
    temporary_prefix: str = "",
) -> tuple[EmitDescriptorOp, ...]:
    shuffle = descriptor_lookup("x86.avx2.vpermilps.xmm")
    addss = descriptor_lookup("x86.avx2.vaddss.xmm")
    lane_count = len(input_values) * 4
    lane_ordinal = 0
    accumulator = ValueRef.operand("init")
    emit_ops: list[EmitDescriptorOp] = []
    for input_value in input_values:
        for lane in range(4):
            lane_value = input_value
            if lane != 0:
                lane_value = ValueRef.temporary(f"{temporary_prefix}lane{lane_ordinal}")
                emit_ops.append(
                    _op_emit(
                        descriptor=shuffle,
                        operands={"source": input_value},
                        results={"dst": lane_value},
                        result_types={"dst": _V4F32},
                        immediates={"control": lane},
                    )
                )
            lane_ordinal += 1
            next_accumulator = (
                ValueRef.result("result")
                if lane_ordinal == lane_count
                else ValueRef.temporary(f"{temporary_prefix}accumulator{lane_ordinal}")
            )
            emit_ops.append(
                _op_emit(
                    descriptor=addss,
                    operands={"lhs": accumulator, "rhs": lane_value},
                    results={"dst": next_accumulator},
                    result_types={"dst": _F32},
                )
            )
            accumulator = next_accumulator
    return tuple(emit_ops)
