# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for SPIR-V KHR cooperative matrix support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loom.target.arch.spirv.features import feature_bits_expression, feature_bits_value
from loom.target.arch.spirv.scalar_memory import (
    StorageBufferScalar,
    storage_buffer_scalar_by_suffix,
)


def cooperative_matrix_descriptor_key(
    op_name: str,
    *,
    element: str,
    m_size: int,
    n_size: int,
    k_size: int,
    accumulator: str,
    scope: str,
    operand_mode: str | None = None,
) -> str:
    operand_mode_part = f".{operand_mode}" if operand_mode else ""
    return (
        f"spirv.{op_name}.{element}."
        f"m{m_size}n{n_size}k{k_size}.{accumulator}.{scope}"
        f"{operand_mode_part}"
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
    rhs_rows: int
    rhs_columns: int
    accumulator_rows: int
    accumulator_columns: int
    feature_atoms: tuple[str, ...]
    lhs_scalar_suffix: str | None = None
    rhs_scalar_suffix: str | None = None
    accumulator_scalar_suffix: str | None = None
    result_scalar_suffix: str | None = None
    # Saturates the accumulation into the result component's integer range.
    saturating_accumulation: bool = False
    source_rule_enabled: bool = True

    @property
    def operand_mode(self) -> str | None:
        return "saturating" if self.saturating_accumulation else None

    @property
    def signed_component_roles(self) -> tuple[str, ...]:
        return tuple(
            role
            for role, scalar in (
                ("A", self.lhs_scalar),
                ("B", self.rhs_scalar),
                ("C", self.accumulator_scalar),
                ("RESULT", self.result_scalar),
            )
            if scalar.suffix.startswith("i")
        )

    @property
    def property_operand_flags(self) -> str:
        operands = [
            f"LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_{role}_SIGNED_COMPONENTS"
            for role in self.signed_component_roles
        ]
        if self.saturating_accumulation:
            operands.append(
                "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION"
            )
        return " | ".join(operands) or "0"

    @property
    def packet_operand_mask(self) -> str | None:
        operands = [
            "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_"
            f"MATRIX_{role}_SIGNED_COMPONENTS_KHR_MASK"
            for role in self.signed_component_roles
        ]
        if self.saturating_accumulation:
            operands.append(
                "LOOM_SPIRV_COOPERATIVE_MATRIX_OPERANDS_"
                "SATURATING_ACCUMULATION_KHR_MASK"
            )
        return " | ".join(operands) or None

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
        include_operand_mode: bool = False,
    ) -> str:
        return cooperative_matrix_descriptor_key(
            op_name,
            element=self.element,
            m_size=self.m_size,
            n_size=self.n_size,
            k_size=self.k_size,
            accumulator=self.accumulator,
            scope="subgroup",
            operand_mode=self.operand_mode if include_operand_mode else None,
        )


def _require_storage_buffer_scalar(suffix: str) -> StorageBufferScalar:
    scalar = storage_buffer_scalar_by_suffix(suffix)
    if scalar is None:
        raise ValueError(f"missing SPIR-V storage-buffer scalar row for {suffix}")
    return scalar


@dataclass(frozen=True, slots=True)
class CooperativeMatrixMemory:
    """A matrix component and its independently typed memory pointer."""

    # Scalar interpreted by the cooperative matrix instruction.
    component: StorageBufferScalar
    # Pointee scalar selected by the allocation's storage layout.
    carrier: StorageBufferScalar
    # SPIR-V memory family used by the address materializer and effects.
    space: str

    @property
    def descriptor_suffix(self) -> str:
        if self.carrier == self.component:
            return self.space
        return f"{self.space}.{self.carrier.suffix}"

    @property
    def feature_bits(self) -> int:
        if self.space == "workgroup":
            return (
                self.component.register_feature_bits
                | self.carrier.register_feature_bits
            )
        return self.component.feature_bits


def cooperative_matrix_memory_variants(
    component: StorageBufferScalar,
) -> tuple[CooperativeMatrixMemory, ...]:
    memory = (
        CooperativeMatrixMemory(component, component, "storage_buffer"),
        CooperativeMatrixMemory(component, component, "workgroup"),
    )
    # Workgroup layout retains a signed integer carrier when same-width views
    # alias. KHR cooperative memory interprets those bits as matrix components;
    # equal widths preserve the stride's units without a conversion or copy.
    carrier = _require_storage_buffer_scalar(f"i{component.byte_width * 8}")
    if carrier != component:
        memory += (CooperativeMatrixMemory(component, carrier, "workgroup"),)
    return memory


COOPERATIVE_MATRIX_CASES = (
    *(
        CooperativeMatrixCase(
            element="f16",
            lhs_source_type="f16",
            rhs_source_type="f16",
            accumulator_source_type=accumulator,
            result_source_type=accumulator,
            accumulator=accumulator,
            m_size=16,
            n_size=16,
            k_size=16,
            lhs_rows=16,
            lhs_columns=16,
            rhs_rows=16,
            rhs_columns=16,
            accumulator_rows=16,
            accumulator_columns=16,
            feature_atoms=("cooperative_matrix_khr", "float16"),
        )
        for accumulator in ("f32", "f16")
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
        rhs_rows=16,
        rhs_columns=16,
        accumulator_rows=16,
        accumulator_columns=16,
        feature_atoms=(
            "cooperative_matrix_khr",
            "bfloat16_type_khr",
            "bfloat16_cooperative_matrix_khr",
        ),
    ),
    *(
        CooperativeMatrixCase(
            element=lhs_name if lhs == rhs else f"{lhs_name}_{rhs_name}",
            lhs_source_type="i8",
            rhs_source_type="i8",
            accumulator_source_type="i32",
            result_source_type="i32",
            accumulator=accumulator_name,
            m_size=16,
            n_size=16,
            k_size=reduction,
            lhs_rows=16,
            lhs_columns=reduction,
            rhs_rows=reduction,
            rhs_columns=16,
            accumulator_rows=16,
            accumulator_columns=16,
            feature_atoms=("cooperative_matrix_khr", "int8"),
            lhs_scalar_suffix=lhs,
            rhs_scalar_suffix=rhs,
            accumulator_scalar_suffix=accumulator,
            result_scalar_suffix=accumulator,
            saturating_accumulation=saturating,
        )
        for reduction in (16, 32)
        for lhs_name, lhs in (("s8", "i8"), ("u8", "u8"))
        for rhs_name, rhs in (("s8", "i8"), ("u8", "u8"))
        for accumulator_name, accumulator in (("s32", "i32"), ("u32", "u32"))
        for saturating in (False, True)
    ),
)


@dataclass(frozen=True, slots=True)
class CooperativeMatrixType:
    """One native value type shared by the declared matrix operations."""

    scalar: StorageBufferScalar
    rows: int
    columns: int
    role: str

    def arithmetic_key(self, operation: CooperativeMatrixArithmetic) -> str:
        return (
            f"spirv.op_{operation.suffix}.cooperative.{self.scalar.suffix}."
            f"{self.rows}x{self.columns}.{self.role}"
        )

    @property
    def constructor_key(self) -> str:
        return (
            f"spirv.op_composite_construct.cooperative.{self.scalar.suffix}."
            f"{self.rows}x{self.columns}.{self.role}"
        )

    def memory_key(
        self, operation: Literal["load", "store"], memory: CooperativeMatrixMemory
    ) -> str:
        return (
            f"spirv.op_cooperative_matrix_{operation}_khr.{self.role}."
            f"{self.scalar.suffix}.{self.rows}x{self.columns}.subgroup."
            f"{memory.descriptor_suffix}"
        )

    @property
    def feature_bits(self) -> int:
        atoms = ["cooperative_matrix_khr"]
        if self.scalar.source_type == "bf16":
            atoms.append("bfloat16_cooperative_matrix_khr")
        return feature_bits_value(atoms) | self.scalar.register_feature_bits

    @property
    def source_roles(self) -> tuple[str, ...]:
        return ("init", "result") if self.role == "accumulator" else (self.role,)

    @property
    def use_c_expression(self) -> str:
        return {
            "lhs": "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR",
            "rhs": "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR",
            "accumulator": "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR",
        }[self.role]


# Distinct contractions can share native matrix types. Constructors and memory
# operations belong to those types, independent of the contraction using them.
COOPERATIVE_MATRIX_TYPES = tuple(
    dict.fromkeys(
        matrix_type
        for case in COOPERATIVE_MATRIX_CASES
        if case.source_rule_enabled
        for matrix_type in (
            CooperativeMatrixType(
                case.lhs_scalar, case.lhs_rows, case.lhs_columns, "lhs"
            ),
            CooperativeMatrixType(
                case.rhs_scalar, case.rhs_rows, case.rhs_columns, "rhs"
            ),
            CooperativeMatrixType(
                case.accumulator_scalar,
                case.accumulator_rows,
                case.accumulator_columns,
                "accumulator",
            ),
            CooperativeMatrixType(
                case.result_scalar,
                case.accumulator_rows,
                case.accumulator_columns,
                "accumulator",
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class CooperativeMatrixArithmetic:
    """One component-wise operation admitted by SPV_KHR_cooperative_matrix."""

    # Vector dialect operation suffix.
    source: str
    # SPIR-V descriptor operation suffix.
    suffix: str
    # Assembly instruction spelling.
    mnemonic: str
    # SPIR-V wire opcode constant.
    opcode: str
    # Number of same-typed cooperative matrix operands.
    operand_count: int = 2

    @property
    def floating_point(self) -> bool:
        return self.source.endswith("f")

    @property
    def controls_contraction(self) -> bool:
        return self.floating_point and self.operand_count == 2

    def supports(self, matrix_type: CooperativeMatrixType) -> bool:
        # BF16 permits matrix multiply and conversion, but not these arithmetic
        # instructions. Signedness belongs to the opcode for integer arithmetic.
        if self.floating_point:
            return matrix_type.scalar.source_type in ("f16", "f32", "f64")
        return matrix_type.scalar.source_type.startswith("i")


COOPERATIVE_MATRIX_ARITHMETIC = (
    CooperativeMatrixArithmetic("addf", "fadd", "OpFAdd", "LOOM_SPIRV_OP_F_ADD"),
    CooperativeMatrixArithmetic("subf", "fsub", "OpFSub", "LOOM_SPIRV_OP_F_SUB"),
    CooperativeMatrixArithmetic("mulf", "fmul", "OpFMul", "LOOM_SPIRV_OP_F_MUL"),
    CooperativeMatrixArithmetic("divf", "fdiv", "OpFDiv", "LOOM_SPIRV_OP_F_DIV"),
    CooperativeMatrixArithmetic(
        "negf", "fnegate", "OpFNegate", "LOOM_SPIRV_OP_F_NEGATE", 1
    ),
    CooperativeMatrixArithmetic("addi", "iadd", "OpIAdd", "LOOM_SPIRV_OP_I_ADD"),
    CooperativeMatrixArithmetic("subi", "isub", "OpISub", "LOOM_SPIRV_OP_I_SUB"),
    CooperativeMatrixArithmetic("muli", "imul", "OpIMul", "LOOM_SPIRV_OP_I_MUL"),
    CooperativeMatrixArithmetic("divsi", "sdiv", "OpSDiv", "LOOM_SPIRV_OP_S_DIV"),
    CooperativeMatrixArithmetic("divui", "udiv", "OpUDiv", "LOOM_SPIRV_OP_U_DIV"),
    CooperativeMatrixArithmetic(
        "negi", "snegate", "OpSNegate", "LOOM_SPIRV_OP_S_NEGATE", 1
    ),
)
