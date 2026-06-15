# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for SPIR-V NV cooperative vector support."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_expression, feature_bits_value


@dataclass(frozen=True, slots=True)
class CooperativeVectorCase:
    property_name: str
    m_size: int
    k_size: int
    input_type: str
    input_interpretation: str
    matrix_interpretation: str
    bias_interpretation: str
    result_type: str
    matrix_layout_flags: str
    storage_class_flags: str
    feature_atoms: tuple[str, ...]
    flags: str = "0"

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)

    @property
    def feature_bits_c_expression(self) -> str:
        return feature_bits_expression(self.feature_atoms)

    @property
    def shape_key(self) -> int:
        return (self.m_size << 16) | self.k_size


COOPERATIVE_VECTOR_CASES = (
    CooperativeVectorCase(
        property_name="nv.cooperative_vector.f16.16x16.f16",
        m_size=16,
        k_size=16,
        input_type="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        input_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        matrix_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        bias_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        result_type="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        matrix_layout_flags="VECTOR_LAYOUT_INFERENCE_ANY",
        storage_class_flags="STORAGE_BUFFER_OR_BDA",
        feature_atoms=("cooperative_vector_nv",),
        flags="LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRANSPOSE",
    ),
    CooperativeVectorCase(
        property_name="nv.cooperative_vector.f16.16x32.e4m3",
        m_size=16,
        k_size=32,
        input_type="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        input_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV",
        matrix_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV",
        bias_interpretation="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        result_type="LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV",
        matrix_layout_flags="VECTOR_LAYOUT_INFERENCE_ANY",
        storage_class_flags="STORAGE_BUFFER_OR_BDA",
        feature_atoms=("cooperative_vector_nv",),
        flags="LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRANSPOSE",
    ),
    CooperativeVectorCase(
        property_name="nv.cooperative_vector.u32.32x32.s8_packed",
        m_size=32,
        k_size=32,
        input_type="LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV",
        input_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV",
        matrix_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV",
        bias_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV",
        result_type="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV",
        matrix_layout_flags="VECTOR_LAYOUT_INFERENCE_ANY",
        storage_class_flags="STORAGE_BUFFER_OR_BDA",
        feature_atoms=("cooperative_vector_nv",),
    ),
    CooperativeVectorCase(
        property_name="nv.cooperative_vector.training.u32.32x32.s8_packed",
        m_size=32,
        k_size=32,
        input_type="LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV",
        input_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV",
        matrix_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV",
        bias_interpretation="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV",
        result_type="LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV",
        matrix_layout_flags=(
            "LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_BIT"
        ),
        storage_class_flags="STORAGE_BUFFER_OR_BDA",
        feature_atoms=("cooperative_vector_nv", "cooperative_vector_training_nv"),
        flags="LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING",
    ),
)
