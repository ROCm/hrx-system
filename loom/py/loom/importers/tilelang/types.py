# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR type conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loom.ir import (
    BF16,
    F16,
    F32,
    F64,
    I1,
    I8,
    I16,
    I32,
    I64,
    INDEX,
    OFFSET,
    DynamicDim,
    EncodingInstance,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
)


class TileLangTypeConversionError(ValueError):
    """Raised when a TileLang type cannot be represented faithfully."""


class TileLangTypeConverter:
    """Maps TileLang/TIR dtypes and buffers to Loom types."""

    def map_dtype(self, dtype: Any, *, index_like: bool = False) -> Type:
        if index_like:
            return INDEX
        text = str(dtype)
        return _map_scalar_or_vector_dtype(text)

    def vector_type(self, dtype: Any, lanes: int) -> ShapedType:
        element_type = dtype if isinstance(dtype, ScalarType) else self.map_dtype(dtype)
        if not isinstance(element_type, ScalarType):
            raise TileLangTypeConversionError(
                f"vector element dtype must be scalar, got {dtype!r}"
            )
        return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(lanes),))

    def view_type(self, buffer: object) -> ShapedType:
        dtype = _attribute(buffer, "dtype")
        element_type, encoding = _storage_element_type(dtype)
        if not isinstance(element_type, ScalarType):
            raise TileLangTypeConversionError(
                f"buffer element dtype must be scalar, got {dtype!r}"
            )
        shape = _attribute(buffer, "shape", ())
        if shape is None:
            shape = ()
        if not isinstance(shape, Iterable) or isinstance(shape, str | bytes):
            raise TileLangTypeConversionError(
                f"buffer shape must be iterable, got {shape!r}"
            )
        dims = tuple(_shape_dim(dim) for dim in shape)
        return ShapedType(TypeKind.VIEW, element_type, dims, encoding=encoding)

    def buffer_byte_length(self, buffer: object) -> int | None:
        view_type = self.view_type(buffer)
        if not view_type.is_all_static:
            return None
        element_size = _ELEMENT_BYTE_SIZES.get(str(view_type.element_type))
        if element_size is None:
            return None
        byte_length = element_size
        for dim in view_type.dims:
            if not isinstance(dim, StaticDim):
                return None
            byte_length *= dim.size
        return byte_length

    def buffer_base_alignment(self, buffer: object) -> int:
        element_type = self.view_type(buffer).element_type
        return _ELEMENT_BYTE_SIZES.get(str(element_type), 1)


def _shape_dim(value: object) -> StaticDim | DynamicDim:
    if isinstance(value, int):
        return StaticDim(value)
    payload = getattr(value, "value", None)
    if isinstance(payload, int):
        return StaticDim(payload)
    text = str(value)
    if text.isdecimal():
        return StaticDim(int(text))
    return DynamicDim()


def _attribute(value: object, name: str, default: object | None = None) -> object:
    return getattr(value, name, default)


def _parse_vector_dtype(text: str) -> ShapedType | None:
    head, separator, tail = text.rpartition("x")
    if not separator or not tail.isdecimal():
        return None
    element_type = _DTYPE_MAP.get(head)
    if not isinstance(element_type, ScalarType):
        return None
    return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(int(tail)),))


def _storage_element_type(
    dtype: object,
) -> tuple[Type, EncodingInstance | None]:
    text = str(dtype)
    fp8_format = _FP8_FORMATS.get(text)
    if fp8_format is None:
        return _map_scalar_or_vector_dtype(text), None
    element_type, format_name = fp8_format
    encoding = EncodingInstance(
        name="tilelang.fp8",
        params=(("format", format_name),),
    )
    return element_type, encoding


def _map_scalar_or_vector_dtype(text: str) -> Type:
    if text in _FORMAT_PRESERVING_FP8_DTYPES:
        raise TileLangTypeConversionError(
            f"TileLang dtype `{text}` carries numeric-format semantics that "
            "cannot be represented as a bare Loom scalar/register type"
        )
    mapped = _DTYPE_MAP.get(text)
    if mapped is None:
        mapped = _parse_vector_dtype(text)
    if mapped is None:
        raise TileLangTypeConversionError(f"unsupported TileLang dtype `{text}`")
    return mapped


F8E4M3 = ScalarType(ScalarTypeKind.F8E4M3)
F8E5M2 = ScalarType(ScalarTypeKind.F8E5M2)

_DTYPE_MAP: dict[str, Type] = {
    "bool": I1,
    "int8": I8,
    "i8": I8,
    "uint8": I8,
    "u8": I8,
    "int16": I16,
    "i16": I16,
    "uint16": I16,
    "u16": I16,
    "int32": I32,
    "i32": I32,
    "uint32": I32,
    "u32": I32,
    "int64": I64,
    "i64": I64,
    "uint64": I64,
    "u64": I64,
    "index": INDEX,
    "offset": OFFSET,
    "float16": F16,
    "float32": F32,
    "float64": F64,
    "bfloat16": BF16,
    "float8_e4m3": F8E4M3,
    "float8_e5m2": F8E5M2,
}

_FP8_FORMATS: dict[str, tuple[ScalarType, str]] = {
    "float8_e4m3": (F8E4M3, "e4m3"),
    "float8_e4m3fn": (F8E4M3, "e4m3fn"),
    "float8_e4m3fnuz": (F8E4M3, "e4m3fnuz"),
    "float8_e5m2": (F8E5M2, "e5m2"),
    "float8_e5m2fnuz": (F8E5M2, "e5m2fnuz"),
}

_FORMAT_PRESERVING_FP8_DTYPES = set(_FP8_FORMATS) - set(_DTYPE_MAP)

_ELEMENT_BYTE_SIZES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "index": 8,
    "offset": 8,
    "f8E4M3": 1,
    "f8E5M2": 1,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}
