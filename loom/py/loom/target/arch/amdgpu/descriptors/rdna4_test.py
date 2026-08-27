# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amdgpu.descriptors import (
    _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64,
    amdgpu_encoding_field_id,
)
from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from loom.target.low_descriptors import (
    ConstraintKind,
    EncodingFieldValue,
    ImmediateKind,
    LatencyKind,
    ModelQuality,
    NativeAsmValueKind,
)


def _matrix_descriptors(descriptor_set, family: str):
    prefix = f"matrix.{family}."
    return tuple(
        descriptor
        for descriptor in descriptor_set.descriptors
        if (descriptor.semantic_tag or "").startswith(prefix)
    )


def _immediate(descriptor, field_name: str):
    return next(
        immediate
        for immediate in descriptor.immediates
        if immediate.field_name == field_name
    )


def test_gfx125x_wmma_catalog_is_portable_across_exact_and_generic_targets() -> None:
    gfx1250_wmma = _matrix_descriptors(
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE, "wmma"
    )
    gfx1251_wmma = _matrix_descriptors(
        _AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE, "wmma"
    )
    generic_wmma = _matrix_descriptors(
        _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE, "wmma"
    )

    def without_schedule_class(descriptors):
        return tuple(
            replace(descriptor, schedule_class="") for descriptor in descriptors
        )

    assert without_schedule_class(gfx1251_wmma) == without_schedule_class(gfx1250_wmma)
    assert without_schedule_class(generic_wmma) == without_schedule_class(gfx1250_wmma)
    assert (
        _matrix_descriptors(_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE, "swmmac")
        == ()
    )


def test_gfx1251_wmma_schedule_classes_match_llvm_speed_model() -> None:
    exact_descriptors = {
        descriptor.key: descriptor
        for descriptor in _AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE.descriptors
    }
    generic_descriptors = {
        descriptor.key: descriptor
        for descriptor in _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE.descriptors
    }
    exact_schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in (
            _AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE.schedule_classes
        )
    }
    generic_schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in (
            _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE.schedule_classes
        )
    }
    schedule_cases = (
        ("amdgpu.v_wmma_f32_16x16x32_bf16", 16),
        ("amdgpu.v_wmma_f32_16x16x64_fp8_fp8", 16),
        ("amdgpu.v_wmma_f32_16x16x128_fp8_fp8", 32),
        ("amdgpu.v_wmma_i32_16x16x64_iu8", 32),
        ("amdgpu.v_wmma_f32_16x16x128_f8f6f4_f4_f4", 16),
        ("amdgpu.v_wmma_f32_16x16x128_f8f6f4_f8_f8", 32),
        ("amdgpu.v_wmma_f32_32x16x128_f4", 32),
    )

    for descriptor_key, latency_cycles in schedule_cases:
        exact_schedule = exact_schedule_classes[
            exact_descriptors[descriptor_key].schedule_class
        ]
        assert exact_schedule.latency_kind is LatencyKind.EXACT
        assert exact_schedule.latency_cycles == latency_cycles
        assert exact_schedule.model_quality is ModelQuality.EXACT

        generic_schedule = generic_schedule_classes[
            generic_descriptors[descriptor_key].schedule_class
        ]
        assert generic_schedule.latency_kind is LatencyKind.ESTIMATE
        assert generic_schedule.latency_cycles == latency_cycles
        assert generic_schedule.model_quality is ModelQuality.ESTIMATED


def test_gfx1250_stepping_wmma_schedules_match_llvm_speed_model() -> None:
    b0_descriptors = {
        descriptor.key: descriptor
        for descriptor in _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.descriptors
    }
    a0_descriptors = {
        descriptor.key: descriptor
        for descriptor in _AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE.descriptors
    }
    assert {
        key: replace(descriptor, schedule_class="")
        for key, descriptor in b0_descriptors.items()
    } == {
        key: replace(descriptor, schedule_class="")
        for key, descriptor in a0_descriptors.items()
    }

    b0_schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in (
            _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.schedule_classes
        )
    }
    a0_schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in (
            _AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE.schedule_classes
        )
    }
    schedule_cases = (
        ("amdgpu.v_wmma_f32_16x16x4_f32", 16, 16),
        ("amdgpu.v_wmma_f32_16x16x64_fp8_fp8", 4, 8),
        ("amdgpu.v_wmma_f32_16x16x128_f8f6f4_f4_f4", 4, 8),
        ("amdgpu.v_wmma_f32_16x16x128_f8f6f4_f6_f4", 8, 8),
        ("amdgpu.v_wmma_f32_16x16x128_f8f6f4_f8_f4", 8, 16),
        ("amdgpu.v_wmma_i32_16x16x64_iu8", 16, 16),
        ("amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f4_f4", 4, 8),
    )
    for descriptor_key, expected_b0_latency, expected_a0_latency in schedule_cases:
        for descriptor, schedule_classes, expected_latency in (
            (
                b0_descriptors[descriptor_key],
                b0_schedule_classes,
                expected_b0_latency,
            ),
            (
                a0_descriptors[descriptor_key],
                a0_schedule_classes,
                expected_a0_latency,
            ),
        ):
            schedule_class = schedule_classes[descriptor.schedule_class]
            assert schedule_class.latency_kind is LatencyKind.EXACT
            assert schedule_class.latency_cycles == expected_latency
            assert schedule_class.model_quality is ModelQuality.EXACT


def test_gfx125x_f8f6f4_wmma_descriptors_model_all_physical_abis() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in _matrix_descriptors(
            _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE, "wmma"
        )
    }
    enum_domains = {
        domain.name: {value.token: value.value for value in domain.values}
        for domain in _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.enum_domains
    }

    for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
            suffix = f"{lhs_format.token}_{rhs_format.token}"
            descriptor = descriptors[f"amdgpu.v_wmma_f32_16x16x128_f8f6f4_{suffix}"]
            assert tuple(operand.unit_count for operand in descriptor.operands) == (
                8,
                lhs_format.register_count_for(64),
                rhs_format.register_count_for(64),
                8,
            )

            format_immediates = (
                _immediate(descriptor, "matrix_a_fmt"),
                _immediate(descriptor, "matrix_b_fmt"),
            )
            assert tuple(immediate.kind for immediate in format_immediates) == (
                ImmediateKind.ENUM,
                ImmediateKind.ENUM,
            )
            assert enum_domains[format_immediates[0].enum_domain] == dict(
                lhs_format.selector_values
            )
            assert enum_domains[format_immediates[1].enum_domain] == dict(
                rhs_format.selector_values
            )

            form = descriptor.asm_forms[0]
            assert form.native_assembly_mnemonic == "v_wmma_f32_16x16x128_f8f6f4"
            format_values = form.native_assembly_values[-2:]
            assert tuple(value.kind for value in format_values) == (
                NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
            )
            assert tuple(value.target_format_id for value in format_values) == (
                AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64,
                AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64,
            )


def test_gfx125x_scaled_f8f6f4_wmma_descriptors_model_all_physical_abis() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in _matrix_descriptors(
            _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE, "wmma"
        )
    }

    expected_keys = set()
    for scale_kind in ("scale", "scale16"):
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
            for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
                key = (
                    f"amdgpu.v_wmma_{scale_kind}_f32_16x16x128_f8f6f4_"
                    f"{lhs_format.token}_{rhs_format.token}"
                )
                expected_keys.add(key)
                descriptor = descriptors[key]
                assert tuple(operand.unit_count for operand in descriptor.operands) == (
                    8,
                    lhs_format.register_count_for(64),
                    rhs_format.register_count_for(64),
                    8,
                    1 if scale_kind == "scale" else 2,
                    1 if scale_kind == "scale" else 2,
                )
                assert (
                    descriptor.asm_forms[0].native_assembly_mnemonic
                    == f"v_wmma_{scale_kind}_f32_16x16x128_f8f6f4"
                )

    actual_keys = {
        key
        for key in descriptors
        if key.startswith("amdgpu.v_wmma_scale") and "_f32_16x16x128_f8f6f4_" in key
    }
    assert actual_keys == expected_keys


def test_gfx125x_fixed_f4_wmma_descriptors_do_not_expose_format_selectors() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in _matrix_descriptors(
            _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE, "wmma"
        )
    }

    for scale_kind in ("scale", "scale16"):
        descriptor = descriptors[f"amdgpu.v_wmma_{scale_kind}_f32_32x16x128_f4"]
        immediate_names = tuple(
            immediate.field_name for immediate in descriptor.immediates
        )
        assert "matrix_a_fmt" not in immediate_names
        assert "matrix_b_fmt" not in immediate_names
        assert tuple(
            value.field_name
            for value in descriptor.asm_forms[0].native_assembly_values
            if value.kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT
        ) == (
            "matrix_a_scale",
            "matrix_b_scale",
            "matrix_a_scale_fmt",
            "matrix_b_scale_fmt",
            "matrix_a_reuse",
            "matrix_b_reuse",
        )
        assert descriptor.encoding_field_values[-1].encoding_field_id == (
            amdgpu_encoding_field_id("OPSEL_HI")
        )
        assert descriptor.encoding_field_values[-1].value == 7

    unscaled = descriptors["amdgpu.v_wmma_f32_32x16x128_f4"]
    assert unscaled.encoding_field_values == (
        EncodingFieldValue(amdgpu_encoding_field_id("OPSEL_HI"), 7),
    )


def test_gfx125x_bf16f32_wmma_keeps_result_distinct_from_accumulator() -> None:
    descriptor = next(
        descriptor
        for descriptor in _matrix_descriptors(
            _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE, "wmma"
        )
        if descriptor.key == "amdgpu.v_wmma_bf16f32_16x16x32_bf16"
    )

    assert descriptor.operands[0].unit_count == 4
    assert descriptor.operands[3].unit_count == 8
    assert tuple(constraint.kind for constraint in descriptor.constraints) == (
        ConstraintKind.EARLY_CLOBBER,
    )


def test_gfx125x_integer_matrix_native_modifiers_follow_llvm_order() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE.descriptors
    }

    for key in (
        "amdgpu.v_wmma_i32_16x16x64_iu8",
        "amdgpu.v_swmmac_i32_16x16x128_iu8",
    ):
        modifier_names = tuple(
            value.field_name
            for value in descriptors[key].asm_forms[0].native_assembly_values
            if value.kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT
        )
        assert modifier_names[-4:] == (
            "matrix_a_reuse",
            "matrix_b_reuse",
            "neg_lo",
            "clamp",
        )
