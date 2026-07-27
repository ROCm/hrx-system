# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured TileLang/TIR pre-conversion analysis."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from loom.importers.core import sanitize_identifier, target_preset_amdgpu_kind
from loom.importers.tilelang.context import (
    TileLangAddressLayoutPreference,
    address_layout_keys,
)
from loom.importers.tilelang.nodes import mapping_items, node_kind, source_name


def collect_address_layout_preferences(
    prim_func: object,
    *,
    target_preset: str,
) -> dict[object, TileLangAddressLayoutPreference]:
    """Collect physical address-layout choices implied by target tile ops."""

    if target_preset_amdgpu_kind(target_preset) != "gfx1100":
        return {}
    preferences: dict[object, TileLangAddressLayoutPreference] = {}
    for call in _walk_calls(getattr(prim_func, "body", None)):
        if _call_op_name(call) != "tl.tileop.gemm":
            continue
        _record_gfx1100_dense_gemm_preferences(call, preferences)
    return preferences


def _record_gfx1100_dense_gemm_preferences(
    call: object,
    preferences: dict[object, TileLangAddressLayoutPreference],
) -> None:
    args = _args(call)
    if not _is_supported_gfx1100_dense_gemm(call, args):
        return
    rhs_buffer = _region_source_buffer(args[1])
    if rhs_buffer is None or _buffer_scope(rhs_buffer) not in ("shared", "shared.dyn"):
        return
    preference = TileLangAddressLayoutPreference(
        strides=(1, 16),
        name=f"{sanitize_identifier(source_name(rhs_buffer, fallback='rhs'))}_layout",
    )
    for key in address_layout_keys(rhs_buffer):
        preferences[key] = preference


def _is_supported_gfx1100_dense_gemm(call: object, args: tuple[object, ...]) -> bool:
    if len(args) < 19 or _call_annotations(call):
        return False
    transpose_lhs = _static_bool(args[3])
    transpose_rhs = _static_bool(args[4])
    if transpose_lhs is not False or transpose_rhs is not False:
        return False
    shape = tuple(_integer_value(arg) for arg in args[5:8])
    if shape != (16, 16, 16):
        return False
    if _integer_value(args[8]) != 0:
        return False
    if _static_bool(args[9]) is None:
        return False
    if _integer_value(args[14]) != 1:
        return False
    return _integer_value(args[15]) == 0


def _walk_calls(node: object) -> Iterator[object]:
    if node is None or isinstance(node, str | bytes | int | float | bool):
        return
    kind = node_kind(node)
    if kind == "Call":
        yield node
        for arg in _args(node):
            yield from _walk_calls(arg)
        return
    if kind == "SeqStmt":
        for child in tuple(getattr(node, "seq", ()) or ()):
            yield from _walk_calls(child)
        return
    if kind == "BlockRealize":
        yield from _walk_calls(getattr(node, "block", None))
        return
    if kind == "Block":
        yield from _walk_calls(getattr(node, "body", None))
        return
    if kind == "AttrStmt":
        yield from _walk_calls(getattr(node, "value", None))
        yield from _walk_calls(getattr(node, "body", None))
        return
    if kind == "Evaluate":
        yield from _walk_calls(getattr(node, "value", None))
        return
    if kind == "LetStmt":
        yield from _walk_calls(getattr(node, "value", None))
        yield from _walk_calls(getattr(node, "body", None))
        return
    if kind == "For":
        yield from _walk_calls(getattr(node, "min", None))
        yield from _walk_calls(getattr(node, "extent", None))
        yield from _walk_calls(getattr(node, "body", None))
        return
    if kind == "IfThenElse":
        yield from _walk_calls(getattr(node, "condition", None))
        yield from _walk_calls(getattr(node, "then_case", None))
        yield from _walk_calls(getattr(node, "else_case", None))
        return
    if kind == "BufferStore":
        yield from _walk_calls(getattr(node, "value", None))
        for index in tuple(getattr(node, "indices", ()) or ()):
            yield from _walk_calls(index)
        return
    if kind == "BufferLoad":
        for index in tuple(getattr(node, "indices", ()) or ()):
            yield from _walk_calls(index)
        return
    for field_name in _FALLBACK_CHILD_FIELDS:
        yield from _walk_calls(getattr(node, field_name, None))


def _region_source_buffer(region_like: object) -> object | None:
    if _call_op_name(region_like) == "tl.tileop.region":
        args = _args(region_like)
        if not args:
            return None
        return _region_source_buffer(args[0])
    if node_kind(region_like) in ("BufferLoad", "BufferRegion"):
        return getattr(region_like, "buffer", None)
    return None


def _call_op_name(call: object) -> str | None:
    op = getattr(call, "op", None)
    if op is None:
        return None
    name = getattr(op, "name", None)
    if name:
        return str(name)
    get_name = getattr(op, "get_name", None)
    if get_name is None:
        return None
    return str(get_name())


def _call_annotations(call: object) -> dict[str, object]:
    annotations = getattr(call, "annotations", None)
    return {str(key): value for key, value in mapping_items(annotations)}


def _args(call: object) -> tuple[object, ...]:
    return tuple(getattr(call, "args", ()))


def _static_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    integer = _integer_value(value)
    if integer in (0, 1):
        return bool(integer)
    return None


def _integer_value(value: object) -> int | None:
    payload: Any = getattr(value, "value", value)
    try:
        return int(payload)
    except (TypeError, ValueError):
        return None


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""


_FALLBACK_CHILD_FIELDS = (
    "body",
    "then_case",
    "else_case",
    "block",
    "value",
)
