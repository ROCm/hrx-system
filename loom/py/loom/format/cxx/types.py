# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ carrier types for projected Loom values."""

from __future__ import annotations

from loom.builtin_types import ChannelReadMode
from loom.format.cxx.errors import CxxRenderError
from loom.ir import (
    BufferType,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    Module,
    ParameterizedType,
    PoolType,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
)

SCALAR_CPP_TYPES = {
    "i1": "bool",
    "i8": "signed char",
    "i16": "short",
    "i32": "int",
    "i64": "long long",
    "f16": "std::float16_t",
    "bf16": "std::bfloat16_t",
    "f32": "float",
    "f64": "double",
    "f8E4M3": "loom::type::float8_e4m3fn_t",
    "f8E5M2": "loom::type::float8_e5m2_t",
}

_UNSIGNED_INTEGER_TYPES = {
    "i1": "bool",
    "i8": "unsigned char",
    "i16": "unsigned short",
    "i32": "unsigned",
    "i64": "unsigned long long",
}

# Auto-deduced kernel topology values retain the index domain while exposing a
# conventional unsigned conversion for ordinary C++ expressions.
KERNEL_COORDINATE_CPP_TYPE = "loom::kernel::coordinate"
_CHANNEL_READ_MUTABLE_MODE = ChannelReadMode.case("mutable").value


class CxxTypes:
    """Plans deterministic C++ carrier names for one verified module."""

    def __init__(self, module: Module, signed_values: set[int]):
        self.module = module
        self.signed_values = signed_values
        self.vector_aliases: dict[tuple[str, tuple[int, ...], bool], str] = {}

    def scalar(self, value_type: ScalarType, signed: bool = False) -> str:
        name = repr(value_type)
        if name == "index":
            return "loom::type::index"
        if name == "offset":
            return "loom::type::offset"
        if name.startswith("i") and not signed:
            result = _UNSIGNED_INTEGER_TYPES.get(name)
        else:
            result = SCALAR_CPP_TYPES.get(name)
        if result is None:
            raise CxxRenderError(None, f"scalar type {name} has no C++ spelling")
        return result

    def vector(self, value_type: ShapedType, signed: bool) -> str:
        if not value_type.is_all_static:
            raise CxxRenderError(
                None,
                f"dependent vector type {value_type!r} has no native C++ carrier",
            )
        dims = tuple(
            dimension.size
            for dimension in value_type.dims
            if isinstance(dimension, StaticDim)
        )
        element = repr(value_type.element_type)
        key = (element, dims, signed)
        alias = self.vector_aliases.get(key)
        if alias is not None:
            return alias
        prefix = {
            "i1": "Bool",
            "i8": "Int8" if signed else "UInt8",
            "i16": "Int16" if signed else "UInt16",
            "i32": "Int32" if signed else "UInt32",
            "i64": "Int64" if signed else "UInt64",
            "f16": "Float16",
            "bf16": "BFloat16",
            "f32": "Float32",
            "f64": "Float64",
            "f8E4M3": "Float8E4M3",
            "f8E5M2": "Float8E5M2",
        }.get(element)
        if prefix is None:
            raise CxxRenderError(
                None, f"vector element type {element} has no C++ spelling"
            )
        alias = prefix + "x" + "x".join(str(dimension) for dimension in dims)
        self.vector_aliases[key] = alias
        return alias

    def shaped_arguments(
        self, value_type: ShapedType, signed: bool, *, const: bool = False
    ) -> tuple[str, str, str | None]:
        """Returns the shape, element, and encoding template arguments."""

        dimensions = ", ".join(
            str(dimension.size)
            if isinstance(dimension, StaticDim)
            else "loom::type::dynamic"
            for dimension in value_type.dims
        )
        shape = f"loom::type::shape<{dimensions}>"
        element = self.scalar(value_type.element_type, signed)
        if const:
            element = f"const {element}"

        encoding = value_type.encoding
        if encoding is None:
            return shape, element, None
        if isinstance(encoding, EncodingInstance):
            raise CxxRenderError(
                None,
                f"statically encoded view type {value_type!r} has no C++ "
                "signature spelling",
            )
        if not isinstance(encoding, DynamicEncoding):
            raise CxxRenderError(
                None, f"view encoding {encoding!r} has no C++ signature spelling"
            )
        encoding_type = self.module.values[encoding.value_id].type
        if not isinstance(encoding_type, EncodingType):
            raise CxxRenderError(
                None, "dynamic view encoding does not reference an encoding value"
            )
        role = encoding_type.role.name.lower()
        return (
            shape,
            element,
            f"loom::type::encoding<loom::encoding::role::{role}, {value_type.rank}>",
        )

    def view(self, value_type: ShapedType, signed: bool, *, const: bool = False) -> str:
        """Returns the source carrier for a statically ranked Loom view."""

        shape, element, encoding = self.shaped_arguments(
            value_type, signed, const=const
        )
        arguments = [shape, element]
        if encoding is not None:
            arguments.append(encoding)
        return f"loom::type::view<{', '.join(arguments)}>"

    def tile(self, value_type: ShapedType, signed: bool) -> str:
        """Returns the source carrier for a tile-level aggregate value."""

        shape, element, encoding = self.shaped_arguments(value_type, signed)
        arguments = [shape, element]
        if encoding is not None:
            arguments.append(encoding)
        return f"loom::type::tile<{', '.join(arguments)}>"

    def parameterized(self, value_type: ParameterizedType, signed: bool) -> str:
        """Returns the source carrier for a builtin parameterized value."""

        family = value_type.family_name
        if family not in {"channel", "read", "write"}:
            raise CxxRenderError(
                None,
                f"parameterized type {value_type!r} has no C++ value spelling",
            )
        payload = value_type.get("payload")
        if not isinstance(payload, ShapedType) or payload.type_kind != TypeKind.TILE:
            raise CxxRenderError(
                None, f"{family} payload {payload!r} has no C++ type spelling"
            )
        arguments = [self.tile(payload, signed)]
        if family == "read":
            mode = value_type.get("mode")
            if mode is not None:
                if mode != _CHANNEL_READ_MUTABLE_MODE:
                    raise CxxRenderError(
                        None, f"channel read mode {mode!r} has no C++ spelling"
                    )
                arguments.append("loom::channel::mode::mutable_")
        return f"loom::type::{family}<{', '.join(arguments)}>"

    def type(self, value_type: object, signed: bool = False) -> str:
        """Returns a C++ type spelling independent of one SSA identity."""

        if isinstance(value_type, ScalarType):
            return self.scalar(value_type, signed)
        if isinstance(value_type, ShapedType):
            if value_type.type_kind == TypeKind.VECTOR:
                return self.vector(value_type, signed)
            if value_type.type_kind == TypeKind.VIEW:
                return self.view(value_type, signed)
            if value_type.type_kind == TypeKind.TILE:
                return self.tile(value_type, signed)
        if isinstance(value_type, BufferType):
            return "unsigned char*"
        if isinstance(value_type, EncodingType):
            role = value_type.role.name.lower()
            return f"loom::type::encoding<loom::encoding::role::{role}>"
        if isinstance(value_type, PoolType):
            return "loom::type::pool"
        if isinstance(value_type, ParameterizedType):
            return self.parameterized(value_type, signed)
        raise CxxRenderError(
            None, f"value type {value_type!r} has no C++ value spelling"
        )

    def value(self, value_id: int, *, signed: bool | None = None) -> str:
        value_type = self.module.values[value_id].type
        use_signed = value_id in self.signed_values if signed is None else signed
        return self.type(value_type, use_signed)
