# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P vector structural selection rules."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8X32_VECTOR = Vector("i8", lanes=32)
_I8X64_VECTOR = Vector("i8", lanes=64)

# AIE2P VSHUFFLE modes that select the even and odd byte lanes from the first
# 512-bit source. Each logical 32-byte result retains the target's 512-bit X
# carrier, with the remaining lanes outside the source vector's value domain.
_I8_DEINTERLEAVE_CONTROLS = (0, 1)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _vector_deinterleave_i8x64_rule() -> DescriptorRule:
    control_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")

    emits = []
    for result_index, result_name in enumerate(("even", "odd")):
        control_name = f"{result_name}_control"
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=control_constant,
                    results={"dst": ValueRef.temporary(control_name)},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": _I8_DEINTERLEAVE_CONTROLS[result_index]},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=shuffle,
                    operands={
                        "s1": ValueRef.operand("source"),
                        "s2": ValueRef.operand("source"),
                        "mod": ValueRef.temporary(control_name),
                    },
                    results={"dst": ValueRef.result("results", element=result_index)},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_deinterleave,
        descriptor=shuffle,
        guards=(
            Guard.value_type("source", _I8X64_VECTOR),
            Guard.value_type("results", _I8X32_VECTOR),
            Guard.attr_kind("axis", "i64"),
            Guard.i64_range("axis", 0, 0),
        ),
        emit=tuple(emits),
    )


AIE2P_STRUCTURAL_RULES = (_vector_deinterleave_i8x64_rule(),)
