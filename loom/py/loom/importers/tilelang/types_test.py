# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import pytest

from loom.importers.tilelang.types import (
    TileLangTypeConversionError,
    TileLangTypeConverter,
    storage_schema_name_hint,
)
from loom.ir import EncodingInstance


@dataclass
class Buffer:
    dtype: str
    shape: tuple[object, ...]
    attrs: dict[str, object] | None = None
    annotations: dict[str, object] | None = None


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

    assert converter.buffer_storage_schema(Buffer("float8_e4m3", (16,))) is None
    assert (
        str(converter.view_type(Buffer("float8_e4m3fn", (16,))).element_type)
        == "f8E4M3"
    )
    e4m3fn_schema = converter.buffer_storage_schema(Buffer("float8_e4m3fn", (16,)))
    assert e4m3fn_schema is not None
    assert e4m3fn_schema == _scalar_schema("f8e4m3fn")
    assert (
        str(converter.view_type(Buffer("float8_e4m3fnuz", (16,))).element_type)
        == "f8E4M3"
    )
    e4m3fnuz_schema = converter.buffer_storage_schema(Buffer("float8_e4m3fnuz", (16,)))
    assert e4m3fnuz_schema is not None
    assert e4m3fnuz_schema == _scalar_schema("f8e4m3fnuz")
    assert (
        str(converter.view_type(Buffer("float8_e5m2fnuz", (16,))).element_type)
        == "f8E5M2"
    )
    e5m2fnuz_schema = converter.buffer_storage_schema(Buffer("float8_e5m2fnuz", (16,)))
    assert e5m2fnuz_schema is not None
    assert e5m2fnuz_schema == _scalar_schema("f8e5m2fnuz")


def test_preserves_explicit_tilelang_float8_storage_rounding() -> None:
    converter = TileLangTypeConverter()

    finite_fn_schema = converter.buffer_storage_schema(
        Buffer(
            "float8_e4m3fn",
            (16,),
            attrs={"loom.storage.rounding": "finite_only"},
        )
    )
    assert finite_fn_schema is not None
    assert finite_fn_schema.name == "encoding.operand"
    assert finite_fn_schema.params == (
        ("element_format", "f8e4m3fn"),
        ("payload_elements", 1),
        ("payload_packing", "dense_lanes"),
        ("rounding", "finite_only"),
    )

    raw_finite_schema = converter.buffer_storage_schema(
        Buffer(
            "float8_e4m3",
            (16,),
            annotations={"loom.storage.rounding": "finite_only"},
        )
    )
    assert raw_finite_schema is not None
    assert raw_finite_schema.name == "encoding.operand"
    assert raw_finite_schema.params == (
        ("element_format", "f8e4m3"),
        ("payload_elements", 1),
        ("payload_packing", "dense_lanes"),
        ("rounding", "finite_only"),
    )


def test_rejects_storage_rounding_on_non_float8_buffer() -> None:
    converter = TileLangTypeConverter()

    with pytest.raises(
        TileLangTypeConversionError,
        match="requires an FP8 buffer dtype",
    ):
        converter.buffer_storage_schema(
            Buffer(
                "float32",
                (16,),
                attrs={"loom.storage.rounding": "finite_only"},
            )
        )


def test_names_structural_operand_schemas_by_numeric_format() -> None:
    assert storage_schema_name_hint(_scalar_schema("f8e4m3fnuz")) == (
        "f8e4m3fnuz_schema"
    )
    assert storage_schema_name_hint(EncodingInstance(name="ggml.q4_k")) == (
        "ggml.q4_k_schema"
    )


def _scalar_schema(element_format: str) -> EncodingInstance:
    return EncodingInstance(
        name="encoding.operand",
        params=(
            ("element_format", element_format),
            ("payload_elements", 1),
            ("payload_packing", "dense_lanes"),
        ),
    )
