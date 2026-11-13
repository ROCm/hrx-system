# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V storage-buffer memory carriers."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value


@dataclass(frozen=True, slots=True)
class StorageBufferScalar:
    source_type: str
    suffix: str
    scalar_enum: str
    byte_width: int
    feature_atoms: tuple[str, ...] = ()
    source_rule_enabled: bool = True

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)

    @property
    def register_feature_bits(self) -> int:
        return feature_bits_value(
            atom
            for atom in self.feature_atoms
            if not atom.startswith("storage_buffer_")
        )

    @property
    def numeric_format_c_expression(self) -> str | None:
        return _NUMERIC_FORMAT_C_EXPRESSIONS.get(self.suffix)


_NUMERIC_FORMAT_C_EXPRESSIONS = {
    "i8": "LOOM_VALUE_FACT_NUMERIC_FORMAT_I8",
    "u8": "LOOM_VALUE_FACT_NUMERIC_FORMAT_U8",
    "i16": "LOOM_VALUE_FACT_NUMERIC_FORMAT_I16",
    "u16": "LOOM_VALUE_FACT_NUMERIC_FORMAT_U16",
    "i32": "LOOM_VALUE_FACT_NUMERIC_FORMAT_I32",
    "u32": "LOOM_VALUE_FACT_NUMERIC_FORMAT_U32",
    "f16": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F16",
    "bf16": "LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16",
    "f32": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F32",
    "f64": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F64",
}


STORAGE_BUFFER_SCALARS = (
    StorageBufferScalar(
        source_type="i8",
        suffix="i8",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S8",
        byte_width=1,
        feature_atoms=("int8", "storage_buffer_8bit_access"),
    ),
    StorageBufferScalar(
        source_type="i8",
        suffix="u8",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U8",
        byte_width=1,
        feature_atoms=("int8", "storage_buffer_8bit_access"),
        source_rule_enabled=False,
    ),
    StorageBufferScalar(
        source_type="i16",
        suffix="i16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S16",
        byte_width=2,
        feature_atoms=("int16", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="i16",
        suffix="u16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U16",
        byte_width=2,
        feature_atoms=("int16", "storage_buffer_16bit_access"),
        source_rule_enabled=False,
    ),
    StorageBufferScalar(
        source_type="i32",
        suffix="i32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
        byte_width=4,
    ),
    StorageBufferScalar(
        source_type="i32",
        suffix="u32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U32",
        byte_width=4,
        source_rule_enabled=False,
    ),
    StorageBufferScalar(
        source_type="i64",
        suffix="i64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S64",
        byte_width=8,
        feature_atoms=("int64",),
    ),
    StorageBufferScalar(
        source_type="i64",
        suffix="u64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U64",
        byte_width=8,
        feature_atoms=("int64",),
        source_rule_enabled=False,
    ),
    StorageBufferScalar(
        source_type="f16",
        suffix="f16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F16",
        byte_width=2,
        feature_atoms=("float16", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="bf16",
        suffix="bf16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_BF16",
        byte_width=2,
        feature_atoms=("bfloat16_type_khr", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="f32",
        suffix="f32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F32",
        byte_width=4,
    ),
    StorageBufferScalar(
        source_type="f64",
        suffix="f64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F64",
        byte_width=8,
        feature_atoms=("float64",),
    ),
)

# Internal byte carrier used by raw buffer access. SPIR-V permits this
# restricted u8 value to participate in storage loads, stores, and integer
# conversions with StorageBuffer8BitAccess alone; general u8 arithmetic still
# requires Int8 and uses the ordinary u8 scalar rows above.
RAW_STORAGE_BUFFER_BYTE = StorageBufferScalar(
    source_type="i8",
    suffix="raw_u8",
    scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U8",
    byte_width=1,
    feature_atoms=("storage_buffer_8bit_access",),
    source_rule_enabled=False,
)

SOURCE_STORAGE_BUFFER_SCALARS = tuple(
    scalar for scalar in STORAGE_BUFFER_SCALARS if scalar.source_rule_enabled
)


@dataclass(frozen=True, slots=True)
class StorageBufferVector:
    """Integer memory packet with alignment determined by its component."""

    # Scalar component type, including its storage and arithmetic capabilities.
    component: StorageBufferScalar
    # Number of components in one native SPIR-V vector transfer.
    lane_count: int

    @property
    def suffix(self) -> str:
        return f"v{self.lane_count}{self.component.suffix}"

    @property
    def scalar_enum(self) -> str:
        return self.component.scalar_enum

    @property
    def byte_width(self) -> int:
        return self.component.byte_width * self.lane_count

    @property
    def feature_bits(self) -> int:
        return self.component.feature_bits


# Integer packets covering a 16-, 32-, or 64-bit payload with narrower
# components. A physical pointer's alignment follows its largest scalar
# component, so these carriers preserve exact under-aligned word accesses.
PACKED_STORAGE_BUFFER_VECTORS = tuple(
    StorageBufferVector(scalar, byte_width // scalar.byte_width)
    for scalar in SOURCE_STORAGE_BUFFER_SCALARS
    if scalar.source_type in ("i8", "i16", "i32")
    for byte_width in (2, 4, 8)
    if byte_width % scalar.byte_width == 0 and 2 <= byte_width // scalar.byte_width <= 4
)


def storage_buffer_scalar_by_source_type(
    source_type: str,
) -> StorageBufferScalar | None:
    for row in SOURCE_STORAGE_BUFFER_SCALARS:
        if row.source_type == source_type:
            return row
    return None


def storage_buffer_scalar_by_suffix(suffix: str) -> StorageBufferScalar | None:
    for row in STORAGE_BUFFER_SCALARS:
        if row.suffix == suffix:
            return row
    return None
