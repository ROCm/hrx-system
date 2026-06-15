# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

from loom.format.text.printer import print_type
from loom.importers.tilelang.types import TileLangTypeConverter


@dataclass
class Buffer:
    dtype: str
    shape: tuple[object, ...]


def test_maps_tilelang_float8_spelling_variants() -> None:
    converter = TileLangTypeConverter()

    assert str(converter.map_dtype("float8_e4m3")) == "f8E4M3"
    assert str(converter.map_dtype("float8_e5m2")) == "f8E5M2"

    for source_dtype in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    ):
        message = ""
        try:
            converter.map_dtype(source_dtype)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"expected {source_dtype} to require format metadata")
        assert "numeric-format semantics" in message


def test_counts_tilelang_float8_buffers_as_one_byte_per_element() -> None:
    converter = TileLangTypeConverter()

    assert converter.buffer_byte_length(Buffer("float8_e4m3fn", (16,))) == 16
    assert converter.buffer_base_alignment(Buffer("float8_e5m2fnuz", (16,))) == 1


def test_preserves_tilelang_float8_storage_format_on_views() -> None:
    converter = TileLangTypeConverter()

    assert (
        print_type(converter.view_type(Buffer("float8_e4m3fnuz", (16,))))
        == 'view<16xf8E4M3, #tilelang.fp8<format="e4m3fnuz">>'
    )
    assert (
        print_type(converter.view_type(Buffer("float8_e5m2fnuz", (16,))))
        == 'view<16xf8E5M2, #tilelang.fp8<format="e5m2fnuz">>'
    )
