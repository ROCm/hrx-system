# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native x86 scalar, index, and packed integer rotation rules."""

from __future__ import annotations

from collections.abc import Callable

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise as scalar
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.contracts.rule_builders import (
    full_vector_type as _full_vector_type,
)
from loom.target.arch.x86.contracts.rule_builders import (
    value_type_guards as _typed_guards,
)
from loom.target.arch.x86.scalar_families import X86_SCALAR_ROTATION_FAMILIES
from loom.target.arch.x86.vector_families import AVX512_ROTATION_FAMILIES
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueProject,
    ValueRef,
)
from loom.target.low_descriptors import Descriptor

_SCALAR_SOURCE_OPS = {
    "rotli": scalar.scalar_rotli,
    "rotri": scalar.scalar_rotri,
}
_INDEX_SOURCE_OPS = {
    "rotli": index.index_rotli,
    "rotri": index.index_rotri,
}
_VECTOR_SOURCE_OPS = {
    "rotli": vector.vector_rotli,
    "rotri": vector.vector_rotri,
}
_REGISTER_CLASSES = {
    128: "x86.xmm",
    256: "x86.ymm",
    512: "x86.zmm",
}
_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_IMMEDIATE_MIN = 0
_IMMEDIATE_MAX = (2**8) - 1


def scalar_integer_rotation_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[DescriptorRule, ...]:
    """Returns complete native scalar and index rotation rules."""
    rules: list[DescriptorRule] = []
    for family in X86_SCALAR_ROTATION_FAMILIES:
        source_cells = (
            *(
                (
                    Scalar(f"i{width}"),
                    width,
                    _SCALAR_SOURCE_OPS[family.source_operation],
                )
                for width in (8, 16, 32, 64)
            ),
            (Scalar("index"), 64, _INDEX_SOURCE_OPS[family.source_operation]),
        )
        for type_pattern, bit_count, source_op in source_cells:
            immediate_descriptor = descriptor_lookup(
                f"x86.scalar.{family.mnemonic}.imm.gpr{bit_count}"
            )
            rules.append(
                DescriptorRule(
                    source_op=source_op,
                    descriptor=immediate_descriptor,
                    priority=2,
                    guards=(
                        Guard.value_exact_i64("rhs"),
                        *_typed_guards(("lhs", "rhs", "result"), type_pattern),
                        Guard.value_i64_range("rhs", _IMMEDIATE_MIN, _IMMEDIATE_MAX),
                    ),
                    emit=(
                        _op_emit(
                            descriptor=immediate_descriptor,
                            operands={"lhs": ValueRef.operand("lhs")},
                            results={"dst": ValueRef.result("result")},
                            immediates={"amount": ValueProject.exact_i64("rhs")},
                        ),
                    ),
                )
            )
            variable_descriptor = descriptor_lookup(
                f"x86.scalar.{family.mnemonic}.cl.gpr{bit_count}"
            )
            rules.append(
                DescriptorRule(
                    source_op=source_op,
                    descriptor=variable_descriptor,
                    priority=1,
                    guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=variable_descriptor,
                            operands={
                                "lhs": ValueRef.operand("lhs"),
                                "rhs": ValueRef.operand("rhs"),
                            },
                            results={"dst": ValueRef.result("result")},
                            # CL is a use-local architectural constraint. The
                            # source value remains in its ordinary GPR class.
                            copy_operands=("rhs",),
                        ),
                    ),
                )
            )
    return tuple(rules)


def avx512_vector_rotation_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[DescriptorRule, ...]:
    """Returns complete AVX-512 i32/i64 immediate and per-lane rotations."""
    rules: list[DescriptorRule] = []
    for family in AVX512_ROTATION_FAMILIES:
        source_op = _VECTOR_SOURCE_OPS[family.source_operation]
        for vector_bit_width in (128, 256, 512):
            register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
            register_class = _REGISTER_CLASSES[vector_bit_width]
            value_type = _full_vector_type(family.element, vector_bit_width)
            immediate_descriptor = descriptor_lookup(
                f"x86.avx512.{family.immediate_mnemonic}.{register_suffix}"
            )
            variable_descriptor = descriptor_lookup(
                f"x86.avx512.{family.variable_mnemonic}.{register_suffix}"
            )
            rules.extend(
                (
                    DescriptorRule(
                        source_op=source_op,
                        descriptor=immediate_descriptor,
                        priority=2,
                        guards=(
                            Guard.value_exact_i64("rhs"),
                            *(
                                (Guard.descriptor_available(immediate_descriptor),)
                                if immediate_descriptor.feature_mask_words
                                else ()
                            ),
                            *_typed_guards(("lhs", "rhs", "result"), value_type),
                            Guard.value_i64_range(
                                "rhs", _IMMEDIATE_MIN, _IMMEDIATE_MAX
                            ),
                        ),
                        emit=(
                            _op_emit(
                                descriptor=immediate_descriptor,
                                operands={"lhs": ValueRef.operand("lhs")},
                                results={"dst": ValueRef.result("result")},
                                immediates={"amount": ValueProject.exact_i64("rhs")},
                            ),
                        ),
                    ),
                    DescriptorRule(
                        source_op=source_op,
                        descriptor=variable_descriptor,
                        priority=1,
                        guards=(
                            *(
                                (Guard.descriptor_available(variable_descriptor),)
                                if variable_descriptor.feature_mask_words
                                else ()
                            ),
                            *_typed_guards(("lhs", "rhs", "result"), value_type),
                            Guard.low_value_register_class("lhs", register_class),
                            Guard.low_value_register_class("rhs", register_class),
                            Guard.low_value_register_class("result", register_class),
                        ),
                        emit=(
                            _op_emit(
                                descriptor=variable_descriptor,
                                operands={
                                    "lhs": ValueRef.operand("lhs"),
                                    "rhs": ValueRef.operand("rhs"),
                                },
                                results={"dst": ValueRef.result("result")},
                            ),
                        ),
                    ),
                )
            )
    return tuple(rules)
