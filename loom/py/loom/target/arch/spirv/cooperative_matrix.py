# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for SPIR-V KHR cooperative matrix support."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_expression, feature_bits_value
from loom.target.arch.spirv.scalar_memory import (
    StorageBufferScalar,
    storage_buffer_scalar_by_suffix,
)


def cooperative_matrix_descriptor_key(
    op_name: str,
    *,
    role: str | None = None,
    element: str,
    m_size: int,
    n_size: int,
    k_size: int,
    accumulator: str,
    scope: str,
    layout: str | None = None,
    operand_mode: str | None = None,
) -> str:
    role_part = f".{role}" if role else ""
    layout_part = f".{layout}" if layout else ""
    operand_mode_part = f".{operand_mode}" if operand_mode else ""
    return (
        f"spirv.{op_name}{role_part}.{element}."
        f"m{m_size}n{n_size}k{k_size}.{accumulator}.{scope}"
        f"{layout_part}{operand_mode_part}"
    )


@dataclass(frozen=True, slots=True)
class CooperativeMatrixCase:
    element: str
    lhs_source_type: str
    rhs_source_type: str
    accumulator_source_type: str
    result_source_type: str
    accumulator: str
    m_size: int
    n_size: int
    k_size: int
    lhs_rows: int
    lhs_columns: int
    lhs_vector_lanes: int
    rhs_rows: int
    rhs_columns: int
    rhs_vector_lanes: int
    accumulator_rows: int
    accumulator_columns: int
    accumulator_vector_lanes: int
    feature_atoms: tuple[str, ...]
    lhs_scalar_suffix: str | None = None
    rhs_scalar_suffix: str | None = None
    accumulator_scalar_suffix: str | None = None
    result_scalar_suffix: str | None = None
    property_storage_flags: str = "MATRIX_STORAGE_ANY"
    operand_mode: str | None = None
    property_operand_flags: str = "0"
    packet_operand_mask: str | None = None
    source_rule_enabled: bool = True

    @property
    def property_name(self) -> str:
        operand_part = f".{self.operand_mode}" if self.operand_mode else ""
        return (
            f"khr.cooperative_matrix.{self.element}."
            f"{self.m_size}x{self.n_size}x{self.k_size}."
            f"{self.accumulator}.subgroup{operand_part}"
        )

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)

    @property
    def feature_bits_c_expression(self) -> str:
        return feature_bits_expression(self.feature_atoms)

    @property
    def lhs_scalar(self) -> StorageBufferScalar:
        return _require_storage_buffer_scalar(
            self.lhs_scalar_suffix or self.lhs_source_type
        )

    @property
    def rhs_scalar(self) -> StorageBufferScalar:
        return _require_storage_buffer_scalar(
            self.rhs_scalar_suffix or self.rhs_source_type
        )

    @property
    def accumulator_scalar(self) -> StorageBufferScalar:
        return _require_storage_buffer_scalar(
            self.accumulator_scalar_suffix or self.accumulator_source_type
        )

    @property
    def result_scalar(self) -> StorageBufferScalar:
        return _require_storage_buffer_scalar(
            self.result_scalar_suffix or self.result_source_type
        )

    @property
    def shape_key(self) -> int:
        return (self.m_size << 32) | (self.n_size << 16) | self.k_size

    def descriptor_key(
        self,
        op_name: str,
        *,
        role: str | None = None,
        layout: str | None = None,
        include_operand_mode: bool = False,
    ) -> str:
        return cooperative_matrix_descriptor_key(
            op_name,
            role=role,
            element=self.element,
            m_size=self.m_size,
            n_size=self.n_size,
            k_size=self.k_size,
            accumulator=self.accumulator,
            scope="subgroup",
            layout=layout,
            operand_mode=self.operand_mode if include_operand_mode else None,
        )

    def memory_feature_bits(self, scalar: StorageBufferScalar) -> int:
        return self.feature_bits | scalar.feature_bits


def _require_storage_buffer_scalar(suffix: str) -> StorageBufferScalar:
    scalar = storage_buffer_scalar_by_suffix(suffix)
    if scalar is None:
        raise ValueError(f"missing SPIR-V storage-buffer scalar row for {suffix}")
    return scalar


_SIGNED_SATURATING_PROPERTY_OPERANDS = " | ".join(
    (
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION",
    )
)

_SIGNED_SATURATING_PACKET_OPERANDS = " | ".join(
    (
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_MATRIX_A_SIGNED_COMPONENTS_KHR_MASK",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_MATRIX_B_SIGNED_COMPONENTS_KHR_MASK",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_MATRIX_C_SIGNED_COMPONENTS_KHR_MASK",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_MATRIX_RESULT_SIGNED_COMPONENTS_KHR_MASK",
        "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_SATURATING_ACCUMULATION_KHR_MASK",
    )
)


COOPERATIVE_MATRIX_CASES = (
    CooperativeMatrixCase(
        element="f16",
        lhs_source_type="f16",
        rhs_source_type="f16",
        accumulator_source_type="f32",
        result_source_type="f32",
        accumulator="f32",
        m_size=16,
        n_size=16,
        k_size=16,
        lhs_rows=16,
        lhs_columns=16,
        lhs_vector_lanes=16,
        rhs_rows=16,
        rhs_columns=16,
        rhs_vector_lanes=16,
        accumulator_rows=16,
        accumulator_columns=16,
        accumulator_vector_lanes=8,
        feature_atoms=("cooperative_matrix_khr", "float16"),
    ),
    CooperativeMatrixCase(
        element="bf16",
        lhs_source_type="bf16",
        rhs_source_type="bf16",
        accumulator_source_type="f32",
        result_source_type="f32",
        accumulator="f32",
        m_size=16,
        n_size=16,
        k_size=16,
        lhs_rows=16,
        lhs_columns=16,
        lhs_vector_lanes=16,
        rhs_rows=16,
        rhs_columns=16,
        rhs_vector_lanes=16,
        accumulator_rows=16,
        accumulator_columns=16,
        accumulator_vector_lanes=8,
        feature_atoms=(
            "cooperative_matrix_khr",
            "bfloat16_type_khr",
            "bfloat16_cooperative_matrix_khr",
        ),
    ),
    CooperativeMatrixCase(
        element="s8",
        lhs_source_type="i8",
        rhs_source_type="i8",
        accumulator_source_type="i32",
        result_source_type="i32",
        accumulator="s32",
        m_size=16,
        n_size=16,
        k_size=32,
        lhs_rows=16,
        lhs_columns=32,
        lhs_vector_lanes=32,
        rhs_rows=32,
        rhs_columns=16,
        rhs_vector_lanes=32,
        accumulator_rows=16,
        accumulator_columns=16,
        accumulator_vector_lanes=8,
        feature_atoms=(
            "cooperative_matrix_khr",
            "int8",
            "storage_buffer_8bit_access",
        ),
        property_storage_flags="STORAGE_BUFFER_OR_BDA",
        operand_mode="signed_saturating",
        property_operand_flags=_SIGNED_SATURATING_PROPERTY_OPERANDS,
        packet_operand_mask=_SIGNED_SATURATING_PACKET_OPERANDS,
    ),
    CooperativeMatrixCase(
        element="u8",
        lhs_source_type="i8",
        rhs_source_type="i8",
        accumulator_source_type="i32",
        result_source_type="i32",
        accumulator="u32",
        m_size=16,
        n_size=16,
        k_size=32,
        lhs_rows=16,
        lhs_columns=32,
        lhs_vector_lanes=32,
        rhs_rows=32,
        rhs_columns=16,
        rhs_vector_lanes=32,
        accumulator_rows=16,
        accumulator_columns=16,
        accumulator_vector_lanes=8,
        feature_atoms=(
            "cooperative_matrix_khr",
            "int8",
            "storage_buffer_8bit_access",
        ),
        lhs_scalar_suffix="u8",
        rhs_scalar_suffix="u8",
        accumulator_scalar_suffix="u32",
        result_scalar_suffix="u32",
        property_storage_flags="STORAGE_BUFFER_OR_BDA",
    ),
)
