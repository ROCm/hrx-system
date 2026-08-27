# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from loom.dialect.encoding.numeric_formats import FP8_FORMATS
from loom.gen.test import numeric_conversion_matrix

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_CORPUS_ROOT = _REPOSITORY_ROOT / "loom/src/loom/test/corpus/encoding"


def _format_row(keyword: str):
    return next(row for row in FP8_FORMATS if row.keyword == keyword)


@pytest.mark.parametrize("row", FP8_FORMATS, ids=lambda row: row.keyword)
def test_fp8_reference_round_trips_every_finite_payload(row) -> None:
    for payload in range(256):
        value = numeric_conversion_matrix._fp8_decode_bits(row, payload)
        if value.kind != "finite":
            continue
        assert numeric_conversion_matrix._fp8_encode_bits(row, value) == payload


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
    encode = numeric_conversion_matrix._fp8_encode_bits
    value = numeric_conversion_matrix._BinaryValue
    assert encode(row, value("finite", magnitude=Fraction(1))) == one
    assert numeric_conversion_matrix._fp8_maximum_finite_payload(row) == maximum
    assert encode(row, value("infinity")) == infinity
    assert encode(row, value("nan")) == nan
    assert encode(row, value("finite", sign=1)) == negative_zero


@pytest.mark.parametrize("row", FP8_FORMATS, ids=lambda row: row.keyword)
@pytest.mark.parametrize("bit_count", [16, 32])
def test_fp8_decode_comparison_ignores_only_nan_sign(row, bit_count: int) -> None:
    all_bits = (1 << bit_count) - 1
    nan_payload = numeric_conversion_matrix._fp8_nan_payload(row, sign=0)
    assert numeric_conversion_matrix._decode_comparison_mask(row, nan_payload, bit_count) == all_bits >> 1
    assert numeric_conversion_matrix._decode_comparison_mask(row, 0x00, bit_count) == all_bits
    assert (
        numeric_conversion_matrix._decode_comparison_mask(
            row,
            numeric_conversion_matrix._fp8_encode_bits(
                row,
                numeric_conversion_matrix._BinaryValue("finite", sign=1, magnitude=Fraction(1)),
            ),
            bit_count,
        )
        == all_bits
    )


@pytest.mark.parametrize(
    ("path_name", "generated"),
    [
        (
            "fp8_bidirectional_encode_e4m3.loom",
            lambda: numeric_conversion_matrix.emit_fp8_encode_matrix("e4m3"),
        ),
        (
            "fp8_bidirectional_encode_e5m2.loom",
            lambda: numeric_conversion_matrix.emit_fp8_encode_matrix("e5m2"),
        ),
        (
            "fp8_bidirectional_decode_e4m3.loom",
            lambda: numeric_conversion_matrix.emit_fp8_decode_matrix("e4m3"),
        ),
        (
            "fp8_bidirectional_decode_e5m2.loom",
            lambda: numeric_conversion_matrix.emit_fp8_decode_matrix("e5m2"),
        ),
    ],
)
def test_fp8_matrix_checked_in_sources_are_current(path_name: str, generated) -> None:
    assert (_CORPUS_ROOT / path_name).read_text(encoding="utf-8") == generated()


def test_checked_in_file_set_owns_only_corpus_witnesses() -> None:
    generated_file_set = numeric_conversion_matrix.checked_in_file_set()

    assert generated_file_set.output_paths == (
        "loom/src/loom/test/corpus/encoding/fp8_bidirectional_decode_e4m3.loom",
        "loom/src/loom/test/corpus/encoding/fp8_bidirectional_decode_e5m2.loom",
        "loom/src/loom/test/corpus/encoding/fp8_bidirectional_encode_e4m3.loom",
        "loom/src/loom/test/corpus/encoding/fp8_bidirectional_encode_e5m2.loom",
    )
    assert generated_file_set.obsolete_paths == ()


def test_main_requires_an_explicit_mode_or_build_output() -> None:
    with pytest.raises(SystemExit, match="2"):
        numeric_conversion_matrix.main([])


def test_main_selects_checked_in_maintenance_modes() -> None:
    with mock.patch.object(
        numeric_conversion_matrix,
        "maintain_checked_in_files",
        return_value=SimpleNamespace(ok=True),
    ) as maintain_checked_in_files:
        assert numeric_conversion_matrix.main(["--check"]) == 0
        assert numeric_conversion_matrix.main(["--in-place"]) == 0

    assert maintain_checked_in_files.call_args_list == [
        mock.call("check"),
        mock.call("update"),
    ]


def test_main_rejects_mixed_maintenance_and_build_outputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        numeric_conversion_matrix.main(["--in-place", "--encode-e4m3-output", str(tmp_path / "encode.loom")])


def test_main_writes_explicit_build_output(tmp_path: Path) -> None:
    output_path = tmp_path / "generated/fp8_decode_e5m2.loom"

    assert numeric_conversion_matrix.main(["--decode-e5m2-output", str(output_path)]) == 0

    assert output_path.read_text(encoding="utf-8") == (numeric_conversion_matrix.emit_fp8_decode_matrix("e5m2"))
