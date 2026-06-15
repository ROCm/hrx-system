# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR-to-Loom type conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loom.format.text.parser import parse_type_string
from loom.ir import (
    Dim,
    DynamicDim,
    ScalarType,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
    parse_scalar_type_kind,
)


@dataclass(slots=True)
class MlirTypeConverter:
    """Converts post-bufferization MLIR kernel types into Loom types."""

    _cache: dict[str, Type] = field(default_factory=dict)

    def map(self, value_or_type: object) -> Type:
        return self.map_text(str(value_or_type))

    def map_text(self, text: str) -> Type:
        normalized = text.strip()
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        mapped_text = (
            self.memref_to_view_type_text(normalized)
            if normalized.startswith("memref<")
            else normalized
        )
        try:
            result = parse_type_string(mapped_text)[0]
        except Exception:
            result = _parse_shaped_type(mapped_text)
        self._cache[normalized] = result
        return result

    def memref_to_view_type_text(self, source_type: str) -> str:
        prefix = "memref<"
        if not source_type.startswith(prefix) or not source_type.endswith(">"):
            raise ValueError(f"expected memref type, got `{source_type}`")
        body = source_type[len(prefix) : -1]
        shape_and_element = _split_top_level_comma(body)[0].strip()
        return f"view<{_flatten_vector_element(shape_and_element)}, #dense>"

    def coerce_constant_value(self, value: Any, value_type: str) -> Any:
        if not isinstance(value, str):
            return value
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return -float("inf")
        if value_type.startswith("f") or value_type in {"bf16", "f8E4M3", "f8E5M2"}:
            return float(value)
        return int(value)


def _parse_shaped_type(text: str) -> ShapedType:
    if text.startswith("view<") and text.endswith(">"):
        return _parse_shaped_body(TypeKind.VIEW, text[5:-1])
    if text.startswith("tensor<") and text.endswith(">"):
        return _parse_shaped_body(TypeKind.TENSOR, text[7:-1])
    if text.startswith("vector<") and text.endswith(">"):
        return _parse_shaped_body(TypeKind.VECTOR, text[7:-1])
    raise ValueError(f"unsupported Loom type text `{text}`")


def _parse_shaped_body(type_kind: TypeKind, body: str) -> ShapedType:
    shape_and_element = _split_top_level_comma(body)[0].strip()
    pieces = _split_top_level_x(shape_and_element)
    if len(pieces) < 2:
        raise ValueError(f"expected shaped type body with dimensions: {body}")
    element_kind = parse_scalar_type_kind(pieces[-1])
    if element_kind is None:
        raise ValueError(f"unsupported element type `{pieces[-1]}` in `{body}`")
    dims: list[Dim] = []
    for dim in pieces[:-1]:
        if dim == "?" or (dim.startswith("[") and dim.endswith("]")):
            dims.append(DynamicDim())
        else:
            dims.append(StaticDim(int(dim)))
    return ShapedType(type_kind, ScalarType(element_kind), tuple(dims))


def _flatten_vector_element(shape_and_element: str) -> str:
    pieces = _split_top_level_x(shape_and_element)
    if not pieces:
        return shape_and_element
    element = pieces[-1]
    if not element.startswith("vector<") or not element.endswith(">"):
        return shape_and_element
    nested = _split_top_level_x(element[7:-1])
    if not nested:
        return shape_and_element
    return "x".join((*pieces[:-1], *nested))


def _split_top_level_comma(text: str) -> tuple[str, ...]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return tuple(pieces)


def _split_top_level_x(text: str) -> tuple[str, ...]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "x" and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return tuple(pieces)
