# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared AIE2P X-carrier shapes and assembly helpers."""

from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    ValueRef,
    ValueTypeProject,
    descriptor_by_key,
)

# Ordinary vectors retain one or two full X carriers independently of their
# logical extent. F32 excludes vector<32xf32>, whose accumulator contract uses
# a distinct physical representation.
BITCAST_VECTOR_CARRIER_SPECS = (
    (("i8", "f8E4M3", "f8E5M2"), 1, 128),
    (("i16", "f16", "bf16"), 2, 64),
    (("i32",), 4, 32),
    (("f32",), 4, 31),
    (("i64", "f64"), 8, 16),
)

# Index-shaped values use the same 32-bit X representation as i32 values but
# are excluded from numeric reinterpretation bitcasts.
VECTOR_CARRIER_SPECS = (
    *BITCAST_VECTOR_CARRIER_SPECS,
    (("index", "offset"), 4, 32),
)


def concat_x_carriers_with_controls_emits(
    left: ValueRef,
    right: ValueRef,
    result: ValueRef,
    *,
    left_bytes: ValueRef,
    remaining_bytes: ValueRef,
    temporary_prefix: str = "",
    result_type: DescriptorResultType | None = None,
) -> tuple[ContractEmit, ...]:
    """Joins the suffix-aligned left payload to the right carrier prefix."""

    shift = descriptor_by_key(
        AIE2P_CORE_DESCRIPTOR_SET,
        "amd.xdna.aie2p.shift.bytes.x.configured",
    )
    rotated_left = ValueRef.temporary(f"{temporary_prefix}rotated_left")
    return (
        EmitDescriptorOp(
            descriptor=shift,
            operands={"s1": left, "s2": left, "shift": left_bytes},
            results={"d": rotated_left},
            result_types={"d": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=shift,
            operands={
                "s1": rotated_left,
                "s2": right,
                "shift": remaining_bytes,
            },
            results={"d": result},
            result_types=({"d": result_type} if result_type is not None else None),
            form=DescriptorEmitForm.OP,
        ),
    )


def concat_x_carriers_emits(
    left: ValueRef,
    right: ValueRef,
    result: ValueRef,
    *,
    left_byte_count: ValueTypeProject,
    remaining_byte_count: ValueTypeProject,
    temporary_prefix: str = "",
    result_type: DescriptorResultType | None = None,
) -> tuple[ContractEmit, ...]:
    """Joins X carriers using byte counts projected from logical types."""

    constant = descriptor_by_key(
        AIE2P_CORE_DESCRIPTOR_SET,
        "amd.xdna.aie2p.constant.i32.mova",
    )
    left_bytes = ValueRef.temporary(f"{temporary_prefix}left_bytes")
    remaining_bytes = ValueRef.temporary(f"{temporary_prefix}remaining_bytes")
    merge_emits = concat_x_carriers_with_controls_emits(
        left,
        right,
        result,
        left_bytes=left_bytes,
        remaining_bytes=remaining_bytes,
        temporary_prefix=temporary_prefix,
        result_type=result_type,
    )
    return (
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": left_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": left_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        merge_emits[0],
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": remaining_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": remaining_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        merge_emits[1],
    )
