# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX512-FP16 source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dsl import Op
from loom.target.arch.x86.contracts.avx512_predicate import (
    avx512_fp16_compare_rules,
    avx512_fp16_scalar_compare_rule,
    avx512_fp16_select_rules,
)
from loom.target.arch.x86.contracts.constants import (
    floating_scalar_constant_bits_rule,
    floating_scalar_zero_rule,
)
from loom.target.arch.x86.contracts.float_roots import (
    scalar_float_root_rules,
    vector_float_root_rules,
)
from loom.target.arch.x86.contracts.float_rounding import float_rounding_rules
from loom.target.arch.x86.contracts.floating_extrema import (
    avx512_fp16_float_extrema_rules,
)
from loom.target.arch.x86.contracts.floating_reduction import (
    avx512_fp16_float_dot_rules,
    avx512_fp16_float_reduction_rules,
)
from loom.target.arch.x86.contracts.lane_movement import (
    avx512_fp16_lane_movement_rules,
)
from loom.target.arch.x86.contracts.memory import (
    x86_low_xmm_vector_memory_rules,
    x86_scalar_xmm_word_memory_rules,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.contracts.scalar_float import (
    scalar_register_conversion_rule,
    xmm_scalar_select_rule,
)
from loom.target.arch.x86.contracts.shuffle import x86_low_xmm_word_shuffle_rule
from loom.target.arch.x86.contracts.vector_arithmetic import (
    direct_vector_family_rules,
    vector_fma_family_rules,
)
from loom.target.arch.x86.contracts.vector_construction import (
    avx512_fp16_vector_construction_rules,
)
from loom.target.arch.x86.descriptors import X86_AVX512_FEATURES_DESCRIPTOR_SET
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_FP16
from loom.target.arch.x86.vector_families import (
    AVX512_CARRIER_VECTOR_BIT_WIDTHS,
    AVX512_FP16_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_FLOAT_FMA_MNEMONIC,
    AVX512_FP16_FLOAT_ROOT_FAMILIES,
    AVX512_FP16_FLOAT_ROUNDING_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC,
    AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES,
    AVX512_FP16_VECTOR_BIT_WIDTHS,
    FP16_ELEMENT,
)
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    DirectDescriptorCase,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
    binary_descriptor_rules,
    descriptor_by_key,
    ternary_descriptor_rules,
    unary_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_I16 = Scalar("i16")
_I32 = Scalar("i32")
_F16 = Scalar("f16")
_F32 = Scalar("f32")
_REGISTER_SUFFIXES = {
    32: "xmm",
    64: "xmm",
    128: "xmm",
    256: "ymm",
    512: "zmm",
}

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx512-fp16",
    constraint_key="x86.avx512_fp16.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX512_FEATURES_DESCRIPTOR_SET, key)


def _scalar_float_conversion_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "passthrough": ValueRef.operand("input"),
                    "input": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=1,
    )


def _scalar_rules() -> tuple[DescriptorRule, ...]:
    return (
        floating_scalar_zero_rule(
            _F16,
            _descriptor("x86.avx2.vxorps.zero.xmm"),
        ),
        floating_scalar_constant_bits_rule(
            _F16,
            _I32,
            "imm32",
            ValueProject.float_bits("result"),
            _descriptor("x86.scalar.movimm.gpr32"),
            _descriptor("x86.avx2.vmovd.xmm.gpr32"),
            priority=1,
        ),
        xmm_scalar_select_rule(_F16, _descriptor, priority=1),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _F16,
            _I16,
            "x86.avx2.vmovd.gpr32.xmm",
            _descriptor,
            priority=1,
        ),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _I16,
            _F16,
            "x86.avx2.vmovd.xmm.gpr32",
            _descriptor,
            priority=1,
        ),
        *binary_descriptor_rules(
            tuple(
                DirectDescriptorCase(
                    getattr(scalar_arithmetic, f"scalar_{row.source_operation}"),
                    _descriptor(f"x86.avx512_fp16.{row.mnemonic}.xmm"),
                    _F16,
                    priority=1,
                )
                for row in AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES
            ),
            form=DescriptorEmitForm.OP,
        ),
        *ternary_descriptor_rules(
            (
                DirectDescriptorCase(
                    scalar_math.scalar_fmaf,
                    _descriptor(
                        f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC}.xmm"
                    ),
                    _F16,
                    priority=1,
                ),
            ),
            form=DescriptorEmitForm.OP,
            descriptor_a="lhs",
            descriptor_b="rhs",
            descriptor_c="acc",
        ),
        *scalar_float_root_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx512_fp16",
            families=AVX512_FP16_SCALAR_FLOAT_ROOT_FAMILIES,
            priority=1,
        ),
        avx512_fp16_scalar_compare_rule(_descriptor),
        _scalar_float_conversion_rule(
            scalar_conversion.scalar_extf,
            _F16,
            _F32,
            "x86.avx512_fp16.vcvtsh2ss.xmm",
        ),
        _scalar_float_conversion_rule(
            scalar_conversion.scalar_fptrunc,
            _F32,
            _F16,
            "x86.avx512_fp16.vcvtss2sh.xmm",
        ),
    )


def _vector_conversion_rules() -> tuple[DescriptorRule, ...]:
    cases: list[DirectDescriptorCase] = []
    for lane_count, input_suffix, result_suffix in (
        (2, "xmm", "xmm"),
        (4, "xmm", "xmm"),
        (8, "xmm", "ymm"),
        (16, "ymm", "zmm"),
    ):
        cases.append(
            DirectDescriptorCase(
                vector.vector_extf,
                _descriptor(
                    f"x86.avx512_fp16.vcvtph2psx.{result_suffix}.{input_suffix}"
                ),
                {
                    "input": Vector("f16", lanes=lane_count),
                    "result": Vector("f32", lanes=lane_count),
                },
                priority=1,
            )
        )
        cases.append(
            DirectDescriptorCase(
                vector.vector_fptrunc,
                _descriptor(
                    f"x86.avx512_fp16.vcvtps2phx.{input_suffix}.{result_suffix}"
                ),
                {
                    "input": Vector("f32", lanes=lane_count),
                    "result": Vector("f16", lanes=lane_count),
                },
                priority=1,
            )
        )
    return tuple(
        replace(
            rule, guards=(*rule.guards, Guard.descriptor_available(rule.descriptor))
        )
        for rule in unary_descriptor_rules(cases, form=DescriptorEmitForm.OP)
    )


def _vector_rules() -> tuple[DescriptorRule, ...]:
    return (
        *direct_vector_family_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx512_fp16",
            vector_bit_widths=AVX512_FP16_VECTOR_BIT_WIDTHS,
            integer_families=(),
            float_families=AVX512_FP16_FLOAT_BINARY_FAMILIES,
            priority=1,
        ),
        *vector_fma_family_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx512_fp16",
            vector_bit_widths=AVX512_FP16_VECTOR_BIT_WIDTHS,
            fma_mnemonics={"f16": AVX512_FP16_FLOAT_FMA_MNEMONIC},
            elements=(FP16_ELEMENT,),
            priority=1,
        ),
        *vector_float_root_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx512_fp16",
            families=AVX512_FP16_FLOAT_ROOT_FAMILIES,
            vector_bit_widths=AVX512_FP16_VECTOR_BIT_WIDTHS,
            priority=1,
        ),
        *avx512_fp16_compare_rules(_descriptor),
        *avx512_fp16_select_rules(_descriptor),
        *avx512_fp16_vector_construction_rules(_descriptor),
        x86_low_xmm_word_shuffle_rule(_descriptor),
        *_vector_conversion_rules(),
    )


def _transport_rules() -> tuple[DescriptorRule, ...]:
    return (
        *x86_scalar_xmm_word_memory_rules(
            _descriptor,
            value_type=_F16,
            diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
            priority=1,
        ),
        *x86_low_xmm_vector_memory_rules(
            _descriptor,
            value_type=Vector("f16", lanes=4),
            element_byte_count=2,
            lane_count=4,
            diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
            priority=1,
        ),
        *avx512_fp16_lane_movement_rules(_descriptor),
    )


def _cases() -> Sequence[ContractCase]:
    cases = (
        *_scalar_rules(),
        *_vector_rules(),
        *float_rounding_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx512_fp16",
            families=AVX512_FP16_FLOAT_ROUNDING_FAMILIES,
            vector_bit_widths=AVX512_CARRIER_VECTOR_BIT_WIDTHS,
            priority=1,
        ),
        *avx512_fp16_float_extrema_rules(_descriptor),
        *avx512_fp16_float_reduction_rules(_descriptor),
        *avx512_fp16_float_dot_rules(_descriptor),
        *_transport_rules(),
    )
    # Primary descriptors remain queryable when unavailable so diagnostics can
    # explain missing features. Matching must explicitly require availability
    # to let the shared widening/decomposition paths handle those inputs.
    # Base transport instructions also depend on the native half carrier.
    native_half = _descriptor("x86.avx512_fp16.vcvtsh2ss.xmm")
    guarded_cases = []
    for case in cases:
        required_descriptor = (
            case.descriptor
            if isinstance(case, DescriptorRule)
            and case.descriptor is not None
            and case.descriptor.feature_mask_words
            and case.descriptor.feature_mask_words[0] & FEATURE_AVX512_FP16
            else native_half
        )
        availability = Guard.descriptor_available(required_descriptor)
        if availability not in case.guards:
            case = replace(case, guards=(*case.guards, availability))
        guarded_cases.append(case)
    return tuple(guarded_cases)


X86_AVX512_FP16_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX512_FP16_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx512_fp16",
    descriptor_set=X86_AVX512_FEATURES_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/avx512_fp16.h",
    cases=_cases(),
)
