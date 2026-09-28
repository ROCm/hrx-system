# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V low-bit truncation from integer and address carriers to Boolean."""

from loom.dsl import Op
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
)


def integer_to_boolean_rule(source_op: Op, source_type: str) -> DescriptorRule:
    """Retains bit zero; this is distinct from nonzero truth conversion."""
    # Address casts share their index constants with other index operations.
    constant_type = "index" if source_type == "offset" else source_type
    suffix = "i32" if constant_type == "index" else constant_type
    compare = logical_core_descriptor(f"spirv.op_i_equal.{suffix}")
    prefix: tuple[EmitDescriptorOp, ...] = ()
    input_ref = ValueRef.operand("input")
    if source_type == "offset":
        # Offset has its own wide carrier without ordinary int64 arithmetic.
        # Only the low word is needed to retain bit zero.
        prefix = (
            emit_descriptor_op(
                descriptor=logical_core_descriptor("spirv.op_uconvert.offset64.u32"),
                operands={"input": input_ref},
                results={"dst": ValueRef.temporary("unsigned_low_bits")},
                result_types={"dst": DescriptorResultType()},
            ),
            emit_descriptor_op(
                descriptor=logical_core_descriptor("spirv.op_bitcast.u32.i32"),
                operands={"input": ValueRef.temporary("unsigned_low_bits")},
                results={"dst": ValueRef.temporary("low_bits")},
                result_types={"dst": Scalar("i32")},
            ),
        )
        input_ref = ValueRef.temporary("low_bits")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare,
        guards=(
            Guard.value_type("input", Scalar(source_type)),
            Guard.value_type("result", Scalar("i1")),
            *descriptor_feature_guards(compare),
        ),
        emit=(
            *prefix,
            EmitDescriptorOp(
                descriptor=logical_core_descriptor(f"spirv.op_constant.{suffix}"),
                results={"dst": ValueRef.temporary("one")},
                result_types={"dst": Scalar(constant_type)},
                immediates={f"{suffix}_value": 1},
                form=DescriptorEmitForm.CONST,
            ),
            emit_descriptor_op(
                descriptor=logical_core_descriptor(f"spirv.op_bitwise_and.{suffix}"),
                operands={
                    "lhs": input_ref,
                    "rhs": ValueRef.temporary("one"),
                },
                results={"dst": ValueRef.temporary("low_bit")},
                result_types={"dst": Scalar(suffix)},
            ),
            emit_descriptor_op(
                descriptor=compare,
                operands={
                    "lhs": ValueRef.temporary("low_bit"),
                    "rhs": ValueRef.temporary("one"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )
