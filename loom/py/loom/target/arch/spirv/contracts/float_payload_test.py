# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scalar import conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.bfloat import (
    bfloat_carrier_memory_rules,
    bfloat_to_f32_emits,
    bfloat_widen_rules,
)
from loom.target.arch.spirv.contracts.float8 import float8_narrow_rules
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import DescriptorRule, GuardKind, ValueProjectKind, ValueRef


def _payload_rules(source_op, field, *, kind="scalar", lanes=None):
    rules = {}
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op is not source_op:
            continue
        for guard in rule.guards:
            if guard.kind is not GuardKind.VALUE_TYPE or guard.field != field:
                continue
            pattern = guard.type_pattern
            if pattern.kind != kind or (kind == "vector" and pattern.lanes != lanes):
                continue
            for element in pattern.elements:
                rules.setdefault(element, []).append(rule)
    return rules


def test_float_constants_and_selection_cover_native_and_payload_types() -> None:
    constants = _payload_rules(conversion.scalar_constant, "result")
    selects = _payload_rules(scf.scf_select, "result")
    expected = {
        "f8E4M3": ("i8",),
        "f8E5M2": ("i8",),
        "f16": ("f16",),
        "bf16": ("bf16", "i32"),
        "f32": ("f32",),
        "f64": ("f64",),
    }
    for source_type, carriers in expected.items():
        # Native rows must precede software rows when both are applicable.
        assert tuple(rule.descriptor.key for rule in selects[source_type]) == tuple(
            f"spirv.op_select.{carrier}" for carrier in carriers
        )
        assert len(constants[source_type]) == len(carriers)
        for rule in constants[source_type]:
            immediate = next(iter(rule.emit[0].immediates.values()))
            assert immediate.kind is ValueProjectKind.FLOAT_BITS


def test_singleton_constants_and_selection_preserve_native_feature_priority() -> None:
    for source_op in (vector.vector_constant, scf.scf_select):
        rules = _payload_rules(source_op, "result", kind="vector", lanes=1)
        for native_bfloat in (False, True):
            features = feature_bits_value(
                ("vulkan_shader", "int8", "float16", "float64")
                + (("bfloat16_type_khr",) if native_bfloat else ())
            )
            for element, carrier in (
                ("f8E4M3", "i8"),
                ("f8E5M2", "i8"),
                ("bf16", "bf16" if native_bfloat else "i32"),
                ("f16", "f16"),
                ("f32", "f32"),
                ("f64", "f64"),
            ):
                applicable = tuple(
                    rule
                    for rule in rules[element]
                    if all(
                        not mask & ~features
                        for guard in rule.guards
                        if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
                        for mask in guard.descriptor.feature_mask_words
                    )
                )
                selected = applicable[0]
                selected.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
                assert selected.emit[-1].descriptor.key.endswith(f".{carrier}")
                if source_op is vector.vector_constant:
                    immediate = next(iter(selected.emit[0].immediates.values()))
                    assert immediate.kind is ValueProjectKind.FLOAT_BITS


def test_bfloat_memory_uses_existing_bit_preserving_storage_conversions() -> None:
    rules = bfloat_carrier_memory_rules()
    assert len(rules) == 6  # Global and both Workgroup coordinate forms, R/W.
    seen = set()
    for rule in rules:
        memory = next(step for step in rule.emit if step.source_memory is not None)
        constraint = memory.source_memory
        address = memory.source_memory_address_materializer
        seen.add(
            (rule.source_op.name, constraint.memory_spaces, address.coordinate_type)
        )
        assert constraint.element_byte_count == 2
        assert constraint.minimum_alignment == 2
        assert memory.descriptor.key.endswith(".i16")
        # No floating conversion can quiet a NaN or alter its payload.
        assert not any("f_convert" in step.descriptor.key for step in rule.emit)
        if rule.source_op.name == "view.load":
            assert tuple(step.descriptor.key for step in rule.emit[1:]) == (
                "spirv.op_bitcast.i16.u16",
                "spirv.op_u_convert.u16.u32",
                "spirv.op_bitcast.u32.i32",
            )
        else:
            assert rule.emit[0].descriptor.key == "spirv.op_s_convert.i32.i16"
    assert len(seen) == 6


def test_bfloat_widening_preserves_payloads_without_optional_float_controls() -> None:
    rules = bfloat_widen_rules()
    assert len(rules) == 2
    for native_input in (False, True):
        features = feature_bits_value(
            ("vulkan_shader",) + (("bfloat16_type_khr",) if native_input else ())
        )
        applicable = tuple(
            rule
            for rule in rules
            if all(
                not mask & ~features
                for guard in rule.guards
                if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
                for mask in guard.descriptor.feature_mask_words
            )
        )
        rule = applicable[0]
        assert rule is rules[0 if native_input else 1]
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        expected = (
            "spirv.op_constant.i32",
            "spirv.op_shift_left_logical.i32",
            "spirv.op_bitcast.i32.f32",
        )
        if native_input:
            expected = (
                "spirv.op_composite_construct.v2bf16",
                "spirv.op_bitcast.v2bf16.i32",
                *expected,
            )
        assert tuple(step.descriptor.key for step in rule.emit) == expected
        assert rule.emit[-3].immediates == {"i32_value": 16}
        assert rule.emit[-1].results == {"dst": ValueRef.result("result")}


def test_bfloat_to_float8_reuses_the_exact_widening_path() -> None:
    rules = tuple(
        rule
        for rule in float8_narrow_rules()
        if rule.report_key.startswith("exact_bfloat16_")
    )
    assert len(rules) == 4
    for rule, native_input in zip(rules, (True, True, False, False), strict=True):
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        prefix = bfloat_to_f32_emits(
            ValueRef.operand("input"),
            ValueRef.temporary("wide_input"),
            native_input=native_input,
        )
        assert rule.emit[: len(prefix)] == prefix
