# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P vector reduction selection rules."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")

# AIE2P's 512-bit shuffle network halves the active i32 lanes with each
# control. These sequences are independently witnessed against the AIE API
# reduce_add implementation for the hardware-native vector widths.
_I32_REDUCTION_CONTROLS = (
    (4, (7, 5)),
    (8, (9, 7, 5)),
    (16, (11, 9, 7, 5)),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _constant_emit(result: ValueRef, value: int) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=_descriptor("amd.xdna.aie2p.constant.i32.mova"),
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
    immediates: dict[str, int] | None = None,
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
        immediates=immediates or {},
        form=DescriptorEmitForm.OP,
    )


def _reduce_add_i32_rule(
    lane_count: int, controls: tuple[int, ...], *, zero_init: bool
) -> DescriptorRule:
    input_type = Vector("i32", lanes=lane_count)
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    vector_add = _descriptor("amd.xdna.aie2p.add.i32x16")
    extract = _descriptor("amd.xdna.aie2p.extract.i32.immediate")
    scalar_add = _descriptor("amd.xdna.aie2p.add.i32")

    emits: list[ContractEmit] = []
    current = ValueRef.operand("input")
    for stage, control in enumerate(controls):
        control_value = ValueRef.temporary(f"control_{stage}")
        shuffled = ValueRef.temporary(f"shuffled_{stage}")
        reduced = ValueRef.temporary(f"reduced_{stage}")
        emits.extend(
            (
                _constant_emit(control_value, control),
                _op_emit(
                    shuffle,
                    operands={"s1": current, "s2": current, "mod": control_value},
                    results={"dst": shuffled},
                    descriptor_result_type=True,
                ),
                _op_emit(
                    vector_add,
                    operands={"s1": current, "s2": shuffled},
                    results={"d": reduced},
                    descriptor_result_type=True,
                ),
            )
        )
        current = reduced

    extracted = (
        ValueRef.result("result") if zero_init else ValueRef.temporary("reduced_scalar")
    )
    emits.append(
        _op_emit(
            extract,
            operands={"s1": current},
            results={"dst": extracted},
            descriptor_result_type=not zero_init,
            immediates={"idx": 0},
        )
    )
    if not zero_init:
        emits.append(
            _op_emit(
                scalar_add,
                operands={"s0": ValueRef.operand("init"), "s1": extracted},
                results={"d0": ValueRef.result("result")},
            )
        )

    guards = [
        Guard.enum_attr_equals("kind", "addi"),
        Guard.value_type("input", input_type),
        Guard.value_type("init", _I32),
        Guard.value_type("result", _I32),
    ]
    if zero_init:
        guards.append(Guard.value_i64_range("init", 0, 0))
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=extract if zero_init else scalar_add,
        guards=tuple(guards),
        emit=tuple(emits),
    )


AIE2P_REDUCTION_RULES = tuple(
    _reduce_add_i32_rule(lane_count, controls, zero_init=zero_init)
    for lane_count, controls in _I32_REDUCTION_CONTROLS
    for zero_init in (True, False)
)
