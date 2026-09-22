# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P register-table lane broadcasts."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)


def _table_broadcast_rule(
    elements: tuple[str, ...], lane_count: int, descriptor_key: str
) -> DescriptorRule:
    descriptor = descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key)
    payload = Vector(
        elements, minimum_static_elements=1, maximum_static_elements=lane_count
    )
    return DescriptorRule(
        source_op=vector.vector_table_lookup,
        descriptor=descriptor,
        guards=(
            Guard.value_type("table", payload),
            Guard.value_type("result", payload),
            Guard.value_exact_i64("indices"),
            Guard.value_i64_range("indices", 0, lane_count - 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"s1": ValueRef.operand("table")},
                results={"dst": ValueRef.result("result")},
                immediates={"idx": ValueProject.exact_i64("indices")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


AIE2P_TABLE_RULES = tuple(
    _table_broadcast_rule(elements, lane_count, descriptor_key)
    for elements, lane_count, descriptor_key in (
        (("i8", "f8E4M3", "f8E5M2"), 64, "amd.xdna.aie2p.broadcast.i8x64.from-vector"),
        (("i16", "f16", "bf16"), 32, "amd.xdna.aie2p.broadcast.i16x32.from-vector"),
        (("i32", "f32"), 16, "amd.xdna.aie2p.broadcast.i32x16.from-vector"),
        (("i64", "f64"), 8, "amd.xdna.aie2p.broadcast.i64x8.from-vector"),
    )
)
