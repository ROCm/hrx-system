# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX512-BF16 narrowing and fused source-memory contract rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx512_predicate import (
    avx512_predicate_carrier_register_class,
    avx512_predicate_carrier_to_mask_emits,
)
from loom.target.arch.x86.contracts.memory import x86_fused_load_rules
from loom.target.arch.x86.descriptors import X86_AVX512_FEATURES_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
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

_F32 = Scalar("f32")
_VECTOR_BIT_WIDTHS = (128, 256, 512)
_REGISTER_SUFFIXES = {64: "xmm", 128: "xmm", 256: "ymm", 512: "zmm"}

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx512-bf16",
    constraint_key="x86.avx512_bf16.source_memory",
)


@dataclass(frozen=True, slots=True)
class _ResultForm:
    source_nodes: tuple[SourceNode, ...]
    result: ValueRef
    descriptor_modifiers: tuple[str, ...]
    instruction_operands: Mapping[str, ValueRef]
    pre_instruction_emits: tuple[EmitDescriptorOp, ...]
    priority: int
    report_suffix: str


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX512_FEATURES_DESCRIPTOR_SET, key)


def _vector_types(lane_count: int) -> tuple[Vector, Vector]:
    return Vector("f32", lanes=lane_count), Vector("bf16", lanes=lane_count)


def _narrow_guards(lane_count: int) -> tuple[Guard, ...]:
    input_type, result_type = _vector_types(lane_count)
    return (
        Guard.value_type("input", input_type),
        Guard.value_type("result", result_type),
        Guard.value_not_subnormal_or_instance_flags_has_all(
            "input", "subnormal", "daz"
        ),
    )


def _scalar_narrow_guards() -> tuple[Guard, ...]:
    return (
        Guard.value_type("input", _F32),
        Guard.value_type("result", Scalar("bf16")),
        Guard.value_not_subnormal_or_instance_flags_has_all(
            "input", "subnormal", "daz"
        ),
    )


def _result_ref(source_node: str) -> ValueRef:
    return ValueRef.result("result", source_node=source_node)


def _result_forms(
    source_nodes: Sequence[SourceNode],
    *,
    parent: str,
    lane_count: int,
    base_priority: int,
) -> tuple[_ResultForm, ...]:
    result_type = Vector("bf16", lanes=lane_count)
    forms = [
        _ResultForm(
            source_nodes=tuple(source_nodes),
            result=_result_ref(parent),
            descriptor_modifiers=(),
            instruction_operands={},
            pre_instruction_emits=(),
            priority=base_priority,
            report_suffix="",
        )
    ]
    for masking in ("merge", "zero"):
        for condition_register_class in (
            "x86.k",
            avx512_predicate_carrier_register_class(lane_count),
        ):
            select_guards = [
                Guard.value_type("condition", Vector("i1", lanes=lane_count)),
                Guard.value_type("true_value", result_type),
                Guard.value_type("false_value", result_type),
                Guard.value_type("result", result_type),
                Guard.low_value_register_class("condition", condition_register_class),
            ]
            if masking == "zero":
                select_guards.append(Guard.value_float_equals("false_value", 0.0))
            select = SourceNode.exclusive_user(
                "select",
                source_op=vector.vector_select,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("true_value"),
                parent=parent,
                guards=select_guards,
            )
            condition = ValueRef.operand("condition", source_node="select")
            mask = condition
            pre_instruction_emits: tuple[EmitDescriptorOp, ...] = ()
            if condition_register_class != "x86.k":
                mask = ValueRef.temporary("mask")
                pre_instruction_emits = avx512_predicate_carrier_to_mask_emits(
                    _descriptor,
                    lane_count,
                    condition,
                    mask,
                )
            instruction_operands = {"mask": mask}
            if masking == "merge":
                instruction_operands["passthrough"] = ValueRef.operand(
                    "false_value", source_node="select"
                )
            representation = (
                "mask" if condition_register_class == "x86.k" else "carrier"
            )
            forms.append(
                _ResultForm(
                    source_nodes=(*source_nodes, select),
                    result=_result_ref("select"),
                    descriptor_modifiers=(masking,),
                    instruction_operands=instruction_operands,
                    pre_instruction_emits=pre_instruction_emits,
                    priority=base_priority + (2 if masking == "zero" else 1),
                    report_suffix=f"_{masking}_{representation}",
                )
            )
    return tuple(forms)


def _register_unary_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in _VECTOR_BIT_WIDTHS:
        lane_count = vector_bit_width // 32
        result_bit_width = vector_bit_width // 2
        for form in _result_forms(
            (), parent="", lane_count=lane_count, base_priority=0
        ):
            key_parts = (
                "x86.avx512_bf16.vcvtneps2bf16",
                *form.descriptor_modifiers,
                _REGISTER_SUFFIXES[result_bit_width],
                _REGISTER_SUFFIXES[vector_bit_width],
            )
            descriptor = _descriptor(".".join(key_parts))
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_fptrunc,
                    descriptor=descriptor,
                    source_nodes=form.source_nodes,
                    guards=_narrow_guards(lane_count),
                    emit=(
                        *form.pre_instruction_emits,
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                **form.instruction_operands,
                                "input": ValueRef.operand("input"),
                            },
                            results={"dst": form.result},
                            form=DescriptorEmitForm.OP,
                        ),
                    ),
                    priority=form.priority,
                    report_key=(
                        f"native_f32x{lane_count}_to_bf16x{lane_count}"
                        f"{form.report_suffix}"
                    ),
                )
            )
    return tuple(rules)


def _binary_source_nodes(lane_count: int) -> tuple[SourceNode, ...]:
    return (
        SourceNode.exclusive_definition(
            "low",
            source_op=vector.vector_fptrunc,
            parent_operand=ValueRef.operand("inputs", element=0),
            node_result=ValueRef.result("result"),
            guards=_narrow_guards(lane_count),
        ),
        SourceNode.exclusive_definition(
            "high",
            source_op=vector.vector_fptrunc,
            parent_operand=ValueRef.operand("inputs", element=1),
            node_result=ValueRef.result("result"),
            guards=_narrow_guards(lane_count),
        ),
    )


def _concat_guards(lane_count: int) -> tuple[Guard, ...]:
    _, narrow_type = _vector_types(lane_count)
    return (
        Guard.i64_range("axis", 0, 0),
        Guard.operand_segment_count("inputs", 2),
        Guard.value_type("inputs", narrow_type),
        Guard.value_type("inputs", narrow_type, element=1),
        Guard.value_type("result", Vector("bf16", lanes=lane_count * 2)),
    )


def _register_binary_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in _VECTOR_BIT_WIDTHS:
        lane_count = vector_bit_width // 32
        source_nodes = _binary_source_nodes(lane_count)
        for form in _result_forms(
            source_nodes,
            parent="",
            lane_count=lane_count * 2,
            base_priority=1,
        ):
            key_parts = (
                "x86.avx512_bf16.vcvtne2ps2bf16",
                *form.descriptor_modifiers,
                _REGISTER_SUFFIXES[vector_bit_width],
            )
            descriptor = _descriptor(".".join(key_parts))
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_concat,
                    descriptor=descriptor,
                    source_nodes=form.source_nodes,
                    guards=_concat_guards(lane_count),
                    emit=(
                        *form.pre_instruction_emits,
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                **form.instruction_operands,
                                "high": ValueRef.operand("input", source_node="high"),
                                "low": ValueRef.operand("input", source_node="low"),
                            },
                            results={"dst": form.result},
                            form=DescriptorEmitForm.OP,
                        ),
                    ),
                    priority=form.priority,
                    report_key=(
                        f"native_concat_f32x{lane_count}_to_bf16x"
                        f"{lane_count * 2}{form.report_suffix}"
                    ),
                )
            )
    return tuple(rules)


def _memory_source_nodes(
    lane_count: int,
    *,
    source_count: int,
    broadcast: bool,
) -> tuple[tuple[SourceNode, ...], str]:
    source_nodes: list[SourceNode] = []
    parent = ""
    if broadcast:
        source_nodes.append(
            SourceNode.exclusive_user(
                "scalar_narrow",
                source_op=scalar_conversion.scalar_fptrunc,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("input"),
                guards=_scalar_narrow_guards(),
            )
        )
        source_nodes.append(
            SourceNode.exclusive_user(
                "low",
                source_op=vector.vector_splat,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("scalar"),
                parent="scalar_narrow",
                guards=(
                    Guard.value_type("scalar", Scalar("bf16")),
                    Guard.value_type("result", Vector("bf16", lanes=lane_count)),
                ),
            )
        )
    else:
        source_nodes.append(
            SourceNode.exclusive_user(
                "low",
                source_op=vector.vector_fptrunc,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("input"),
                guards=_narrow_guards(lane_count),
            )
        )
    parent = "low"
    if source_count == 2:
        source_nodes.extend(
            (
                SourceNode.exclusive_user(
                    "concat",
                    source_op=vector.vector_concat,
                    parent_result=ValueRef.result("result"),
                    node_operand=ValueRef.operand("inputs", element=0),
                    parent=parent,
                    guards=_concat_guards(lane_count),
                ),
                SourceNode.exclusive_definition(
                    "high",
                    source_op=vector.vector_fptrunc,
                    parent_operand=ValueRef.operand("inputs", element=1),
                    node_result=ValueRef.result("result"),
                    parent="concat",
                    guards=_narrow_guards(lane_count),
                ),
            )
        )
        parent = "concat"
    return tuple(source_nodes), parent


def _memory_rules(
    *,
    source_count: int,
    broadcast: bool,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    mnemonic = "vcvtneps2bf16" if source_count == 1 else "vcvtne2ps2bf16"
    for vector_bit_width in _VECTOR_BIT_WIDTHS:
        lane_count = vector_bit_width // 32
        result_lane_count = lane_count * source_count
        result_bit_width = (
            vector_bit_width // 2 if source_count == 1 else vector_bit_width
        )
        source_nodes, parent = _memory_source_nodes(
            lane_count,
            source_count=source_count,
            broadcast=broadcast,
        )
        for form in _result_forms(
            source_nodes,
            parent=parent,
            lane_count=result_lane_count,
            base_priority=source_count,
        ):
            instruction_operands = dict(form.instruction_operands)
            if source_count == 2:
                instruction_operands["high"] = ValueRef.operand(
                    "input", source_node="high"
                )
            register_suffix = _REGISTER_SUFFIXES[result_bit_width]
            if source_count == 1:
                register_suffix += f".{_REGISTER_SUFFIXES[vector_bit_width]}"
            memory_kind = "broadcast" if broadcast else "load"
            source_kind = "scalar" if broadcast else f"f32x{lane_count}"
            rules.extend(
                x86_fused_load_rules(
                    _descriptor,
                    source_op=view.view_load if broadcast else vector.vector_load,
                    source_type=_F32 if broadcast else Vector("f32", lanes=lane_count),
                    source_nodes=form.source_nodes,
                    result=form.result,
                    element_byte_count=4,
                    lane_count=1 if broadcast else lane_count,
                    descriptor_key_prefix=f"x86.avx512_bf16.{mnemonic}",
                    descriptor_memory_form=memory_kind,
                    descriptor_key_modifiers=form.descriptor_modifiers,
                    register_suffix=register_suffix,
                    instruction_operands=instruction_operands,
                    pre_memory_emits=form.pre_instruction_emits,
                    diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
                    report_key=(
                        f"native_memory_{mnemonic}_{source_kind}_to_bf16x"
                        f"{result_lane_count}{form.report_suffix}"
                    ),
                    priority=form.priority,
                )
            )
    return tuple(rules)


def _rules() -> tuple[DescriptorRule, ...]:
    return (
        *_register_unary_rules(),
        *_register_binary_rules(),
        *_memory_rules(source_count=1, broadcast=False),
        *_memory_rules(source_count=2, broadcast=False),
        *_memory_rules(source_count=1, broadcast=True),
        *_memory_rules(source_count=2, broadcast=True),
    )


X86_AVX512_BF16_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX512_BF16_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx512_bf16",
    descriptor_set=X86_AVX512_FEATURES_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/avx512_bf16.h",
    cases=_rules(),
)
