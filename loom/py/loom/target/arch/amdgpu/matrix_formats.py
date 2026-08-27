# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU selector-driven matrix physical format families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmdgpuF8F6F4MatrixPhysicalFormat:
    """One physical operand-width family in a selector-driven f8/f6/f4 ISA."""

    token: str
    contract_numeric_type: str
    element_bit_count: int
    selector_values: tuple[tuple[str, int], ...]

    def register_count_for(self, element_count: int) -> int:
        payload_bit_count = element_count * self.element_bit_count
        if payload_bit_count % 32 != 0:
            raise ValueError(
                f"AMDGPU matrix format '{self.token}' payload occupies "
                f"{payload_bit_count} bits, not whole 32-bit registers"
            )
        return payload_bit_count // 32


AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS = (
    AmdgpuF8F6F4MatrixPhysicalFormat(
        token="f8",
        contract_numeric_type="f8",
        element_bit_count=8,
        selector_values=(("fp8", 0), ("bf8", 1)),
    ),
    AmdgpuF8F6F4MatrixPhysicalFormat(
        token="f6",
        contract_numeric_type="f6",
        element_bit_count=6,
        selector_values=(("fp6", 2), ("bf6", 3)),
    ),
    AmdgpuF8F6F4MatrixPhysicalFormat(
        token="f4",
        contract_numeric_type="fp4",
        element_bit_count=4,
        selector_values=(("fp4", 4),),
    ),
)

AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES = {
    physical_format.token: f"amdgpu.cdna4.matrix_format.{physical_format.token}"
    for physical_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
}
