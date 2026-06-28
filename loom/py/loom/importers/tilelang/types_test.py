# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

from loom.importers.tilelang.types import TileLangTypeConverter


@dataclass
class Buffer:
    dtype: str
    shape: tuple[object, ...]


def test_maps_tilelang_float8_spelling_variants() -> None:
    converter = TileLangTypeConverter()

    assert str(converter.map_dtype("float8_e4m3")) == "f8E4M3"
    assert str(converter.map_dtype("float8_e5m2")) == "f8E5M2"
    assert str(converter.map_dtype("float8_e4m3x4")) == "vector<4xf8E4M3>"
    assert str(converter.map_dtype("float8_e5m2x4")) == "vector<4xf8E5M2>"

    for source_dtype in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e4m3fnuzx4",
        "float8_e5m2fnuz",
        "float8_e5m2fnuzx4",
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


def test_preserves_tilelang_float8_storage_format_as_schema() -> None:
    converter = TileLangTypeConverter()

    assert (
        str(converter.view_type(Buffer("float8_e4m3fn", (16,))).element_type)
        == "f8E4M3"
    )
    e4m3fn_schema = converter.buffer_storage_schema(Buffer("float8_e4m3fn", (16,)))
    assert e4m3fn_schema is not None
    assert e4m3fn_schema.name == "fp8_e4m3fn"
    assert (
        str(converter.view_type(Buffer("float8_e4m3fnuz", (16,))).element_type)
        == "f8E4M3"
    )
    e4m3fnuz_schema = converter.buffer_storage_schema(Buffer("float8_e4m3fnuz", (16,)))
    assert e4m3fnuz_schema is not None
    assert e4m3fnuz_schema.name == "fp8_e4m3fnuz"
    assert (
        str(converter.view_type(Buffer("float8_e5m2fnuz", (16,))).element_type)
        == "f8E5M2"
    )
    e5m2fnuz_schema = converter.buffer_storage_schema(Buffer("float8_e5m2fnuz", (16,)))
    assert e5m2fnuz_schema is not None
    assert e5m2fnuz_schema.name == "fp8_e5m2fnuz"
