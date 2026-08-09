# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from loom.gen.target.arch.amdgpu.lower import amdgpu_narrow_float_matrix
from loom.gen.target.arch.amdgpu.lower.amdgpu_narrow_float_tables import _FP8_FORMAT_ROWS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[8]
_EXECUTION_TEST_ROOT = _REPOSITORY_ROOT / "loom/src/loom/target/arch/amdgpu/test/execution"


def _format_row(keyword: str):
    return next(row for row in _FP8_FORMAT_ROWS if row.keyword == keyword)


@pytest.mark.parametrize("row", _FP8_FORMAT_ROWS, ids=lambda row: row.keyword)
def test_fp8_reference_round_trips_every_finite_payload(row) -> None:
    for payload in range(256):
        value = amdgpu_narrow_float_matrix._fp8_decode_bits(row, payload)
        if value.kind != "finite":
            continue
        assert amdgpu_narrow_float_matrix._fp8_encode_bits(row, value) == payload


@pytest.mark.parametrize(
    ("keyword", "one", "maximum", "infinity", "nan", "negative_zero"),
    [
        ("f8e4m3", 0x38, 0x77, 0x78, 0x7F, 0x80),
        ("f8e5m2", 0x3C, 0x7B, 0x7C, 0x7F, 0x80),
        ("f8e4m3fn", 0x38, 0x7E, 0x7E, 0x7F, 0x80),
        ("f8e4m3fnuz", 0x40, 0x7F, 0x7F, 0x80, 0x00),
        ("f8e5m2fnuz", 0x40, 0x7F, 0x7F, 0x80, 0x00),
    ],
)
def test_fp8_reference_special_policy_boundaries(
    keyword: str,
    one: int,
    maximum: int,
    infinity: int,
    nan: int,
    negative_zero: int,
) -> None:
    row = _format_row(keyword)
    encode = amdgpu_narrow_float_matrix._fp8_encode_bits
    value = amdgpu_narrow_float_matrix._BinaryValue
    assert encode(row, value("finite", magnitude=Fraction(1))) == one
    assert amdgpu_narrow_float_matrix._fp8_maximum_finite_payload(row) == maximum
    assert encode(row, value("infinity")) == infinity
    assert encode(row, value("nan")) == nan
    assert encode(row, value("finite", sign=1)) == negative_zero


@pytest.mark.parametrize(
    ("path_name", "generated"),
    [
        (
            "fp8_bidirectional_encode_e4m3.loom",
            lambda: amdgpu_narrow_float_matrix.emit_fp8_encode_matrix("e4m3"),
        ),
        (
            "fp8_bidirectional_encode_e5m2.loom",
            lambda: amdgpu_narrow_float_matrix.emit_fp8_encode_matrix("e5m2"),
        ),
        (
            "fp8_bidirectional_decode_e4m3.loom",
            lambda: amdgpu_narrow_float_matrix.emit_fp8_decode_matrix("e4m3"),
        ),
        (
            "fp8_bidirectional_decode_e5m2.loom",
            lambda: amdgpu_narrow_float_matrix.emit_fp8_decode_matrix("e5m2"),
        ),
    ],
)
def test_fp8_matrix_checked_in_sources_are_current(path_name: str, generated) -> None:
    assert (_EXECUTION_TEST_ROOT / path_name).read_text(encoding="utf-8") == generated()
