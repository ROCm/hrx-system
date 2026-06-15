# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small structured-object helpers for TileLang/TVM TIR imports."""

from __future__ import annotations

from collections.abc import Mapping


def node_kind(node: object) -> str:
    """Returns the source class name used as the importer dispatch key."""

    return type(node).__name__


def node_text(node: object) -> str:
    """Returns a compact one-line diagnostic rendering for a source node."""

    return str(node).splitlines()[0]


def source_name(value: object, *, fallback: str) -> str:
    """Returns the stable human-authored name carried by a source object."""

    for attr in ("name_hint", "name", "__name__"):
        name = getattr(value, attr, None)
        if name:
            return str(name)
    return fallback


def dtype(value: object) -> str:
    """Returns a TIR-like dtype string for source nodes with scalar type."""

    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None:
        return "int32"
    return str(source_dtype)


def attrs(value: object) -> Mapping[str, object]:
    """Returns source attributes with string keys."""

    source_attrs = getattr(value, "attrs", {})
    if source_attrs is None:
        return {}
    return {str(key): item for key, item in mapping_items(source_attrs)}


def buffer_map(prim_func: object) -> dict[object, object]:
    """Returns the PrimFunc parameter-to-buffer map."""

    return dict(mapping_items(getattr(prim_func, "buffer_map", {})))


def lookup_buffer(
    source_buffer_map: Mapping[object, object],
    param: object,
) -> object | None:
    """Finds a buffer by object identity first and source name second."""

    if param in source_buffer_map:
        return source_buffer_map[param]
    param_name = source_name(param, fallback=str(param))
    for key, value in source_buffer_map.items():
        if source_name(key, fallback=str(key)) == param_name:
            return value
    return None


def mapping_items(value: object) -> tuple[tuple[object, object], ...]:
    """Returns mapping items without requiring a concrete Mapping instance."""

    items = getattr(value, "items", None)
    if items is None:
        return ()
    return tuple(items())
