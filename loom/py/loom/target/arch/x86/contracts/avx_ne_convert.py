# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX-NE-CONVERT fused source-memory contract fragment."""

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.memory import x86_fused_load_rules
from loom.target.arch.x86.descriptors import X86_AVX2_FEATURES_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractFragment,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceNode,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx-ne-convert",
    constraint_key="x86.avx_ne_convert.source_memory",
)

_BROADCAST_MNEMONICS = {
    "bf16": "vbcstnebf162ps",
    "f16": "vbcstnesh2ps",
}

_DEINTERLEAVE_MNEMONICS = {
    ("bf16", "even"): "vcvtneebf162ps",
    ("f16", "even"): "vcvtneeph2ps",
    ("bf16", "odd"): "vcvtneobf162ps",
    ("f16", "odd"): "vcvtneoph2ps",
}


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX2_FEATURES_DESCRIPTOR_SET, key)


def _register_suffix(lane_count: int) -> str:
    return {4: "xmm", 8: "ymm"}[lane_count]


def _input_subnormal_guards(source_element: str) -> tuple[Guard, ...]:
    if source_element != "bf16":
        return ()
    return (
        Guard.value_not_subnormal_or_instance_flags_has_all(
            "input", "subnormal", "daz"
        ),
    )


def _broadcast_rules(
    source_element: str, lane_count: int
) -> tuple[DescriptorRule, ...]:
    source_type = Scalar(source_element)
    extended_type = Scalar("f32")
    result_type = Vector("f32", lanes=lane_count)
    mnemonic = _BROADCAST_MNEMONICS[source_element]
    source_nodes = (
        SourceNode.adjacent_unique_user(
            "extend",
            source_op=scalar_conversion.scalar_extf,
            parent_result=ValueRef.result("result"),
            node_operand=ValueRef.operand("input"),
            guards=(
                Guard.value_type("input", source_type),
                Guard.value_type("result", extended_type),
                *_input_subnormal_guards(source_element),
            ),
        ),
        SourceNode.adjacent_unique_user(
            "splat",
            source_op=vector.vector_splat,
            parent="extend",
            parent_result=ValueRef.result("result"),
            node_operand=ValueRef.operand("scalar"),
            guards=(
                Guard.value_type("scalar", extended_type),
                Guard.value_type("result", result_type),
            ),
        ),
    )
    return x86_fused_load_rules(
        _descriptor,
        source_op=view.view_load,
        source_type=source_type,
        source_nodes=source_nodes,
        result=ValueRef.result("result", source_node="splat"),
        element_byte_count=2,
        lane_count=1,
        descriptor_key_prefix=f"x86.avx_ne_convert.{mnemonic}",
        register_suffix=_register_suffix(lane_count),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
        report_key=f"native_memory_broadcast_{source_element}_to_f32x{lane_count}",
    )


def _deinterleave_rules(
    source_element: str,
    selection: str,
    lane_count: int,
) -> tuple[DescriptorRule, ...]:
    source_type = Vector(source_element, lanes=lane_count * 2)
    selected_type = Vector(source_element, lanes=lane_count)
    result_type = Vector("f32", lanes=lane_count)
    sibling = "odd" if selection == "even" else "even"
    mnemonic = _DEINTERLEAVE_MNEMONICS[(source_element, selection)]
    source_nodes = (
        SourceNode.adjacent_unique_user(
            "deinterleave",
            source_op=vector.vector_deinterleave,
            parent_result=ValueRef.result("result"),
            node_operand=ValueRef.operand("source"),
            guards=(
                Guard.value_type("source", source_type),
                Guard.value_type("even", selected_type),
                Guard.value_type("odd", selected_type),
                Guard.i64_range("axis", 0, 0),
                Guard.value_no_uses(sibling),
            ),
        ),
        SourceNode.adjacent_unique_user(
            "extend",
            source_op=vector.vector_extf,
            parent="deinterleave",
            parent_result=ValueRef.result(selection),
            node_operand=ValueRef.operand("input"),
            guards=(
                Guard.value_type("input", selected_type),
                Guard.value_type("result", result_type),
                *_input_subnormal_guards(source_element),
            ),
        ),
    )
    return x86_fused_load_rules(
        _descriptor,
        source_op=vector.vector_load,
        source_type=source_type,
        source_nodes=source_nodes,
        result=ValueRef.result("result", source_node="extend"),
        element_byte_count=2,
        lane_count=lane_count * 2,
        descriptor_key_prefix=f"x86.avx_ne_convert.{mnemonic}",
        register_suffix=_register_suffix(lane_count),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
        report_key=(
            f"native_memory_{selection}_{source_element}x{lane_count * 2}_"
            f"to_f32x{lane_count}"
        ),
    )


def _narrow_rule(lane_count: int) -> DescriptorRule:
    input_type = Vector("f32", lanes=lane_count)
    result_type = Vector("bf16", lanes=lane_count)
    descriptor = _descriptor(
        f"x86.avx_ne_convert.vcvtneps2bf16.xmm.{_register_suffix(lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
            Guard.value_not_subnormal_or_instance_flags_has_all(
                "input", "subnormal", "daz"
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
        report_key=f"native_f32x{lane_count}_to_bf16x{lane_count}",
        priority=2,
    )


def _rules() -> tuple[DescriptorRule, ...]:
    return (
        *(_narrow_rule(lane_count) for lane_count in (4, 8)),
        *(
            rule
            for source_element in _BROADCAST_MNEMONICS
            for lane_count in (4, 8)
            for rule in _broadcast_rules(source_element, lane_count)
        ),
        *(
            rule
            for source_element, selection in _DEINTERLEAVE_MNEMONICS
            for lane_count in (4, 8)
            for rule in _deinterleave_rules(source_element, selection, lane_count)
        ),
    )


X86_AVX_NE_CONVERT_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX_NE_CONVERT_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx_ne_convert",
    descriptor_set=X86_AVX2_FEATURES_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/avx_ne_convert.h",
    cases=_rules(),
)
