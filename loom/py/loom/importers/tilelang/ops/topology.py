# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR launch topology helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.nodes import node_kind, source_name
from loom.ir import INDEX


@dataclass(frozen=True, slots=True)
class ThreadAxis:
    """One recognized GPU launch axis in TileLang/TIR spelling."""

    source_tag: str
    loom_scope: str
    dimension: str
    default_name: str


THREAD_AXES: Mapping[str, ThreadAxis] = {
    "blockIdx.x": ThreadAxis("blockIdx.x", "workgroup", "x", "wgx"),
    "blockIdx.y": ThreadAxis("blockIdx.y", "workgroup", "y", "wgy"),
    "blockIdx.z": ThreadAxis("blockIdx.z", "workgroup", "z", "wgz"),
    "threadIdx.x": ThreadAxis("threadIdx.x", "workitem", "x", "tidx"),
    "threadIdx.y": ThreadAxis("threadIdx.y", "workitem", "y", "tidy"),
    "threadIdx.z": ThreadAxis("threadIdx.z", "workitem", "z", "tidz"),
}


def thread_axis_from_binding(binding: object) -> ThreadAxis | None:
    """Returns the recognized Loom thread axis for a TIR binding object."""

    tag = thread_tag(binding)
    if tag is None:
        return None
    return THREAD_AXES.get(tag)


def thread_tag(binding: object) -> str | None:
    """Extracts a TVM/TileLang thread tag from IterVar or string-like nodes."""

    for attr in ("thread_tag", "value"):
        value = getattr(binding, attr, None)
        if value:
            return str(value)
    text = str(binding).strip("\"'")
    return text if text in THREAD_AXES else None


def thread_binding_var(binding: object) -> object | None:
    """Returns the TIR Var carried by a thread-binding IterVar."""

    return getattr(binding, "var", None)


def map_thread_axis(
    *,
    axis: ThreadAxis,
    sources: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    lower: object | None = None,
    name: str | None = None,
) -> ValueRef | None:
    """Map TIR thread-bound values to Loom kernel topology query ops."""

    result = context.topology_value(axis.source_tag)
    if result is None:
        result = build_thread_axis_id(axis, context, name=name)
        context.map_topology_value(axis.source_tag, result)
    lower_integer = integer_value(lower)
    if lower is not None and lower_integer != 0:
        lower_value = converter.convert_expr(lower, context, index_like=True)
        if lower_value is None:
            return None
        result = context.builder.index.add(
            lhs=lower_value,
            rhs=result,
            results=[INDEX],
            name=context.fresh_name(f"{result.name or axis.default_name}_offset"),
        )
    for source in sources:
        context.map_value(source, result, "index")
    return result


def build_thread_axis_id(
    axis: ThreadAxis,
    context: TileLangConversionContext,
    *,
    name: str | None = None,
) -> ValueRef:
    """Builds the Loom topology query for one TileLang thread axis."""

    result_name = name or context.fresh_name(axis.default_name)
    if axis.loom_scope == "workgroup":
        return context.builder.kernel.workgroup_id(
            dimension=axis.dimension,
            results=[INDEX],
            name=result_name,
        )
    if axis.loom_scope == "workitem":
        return context.builder.kernel.workitem_id(
            dimension=axis.dimension,
            results=[INDEX],
            name=result_name,
        )
    raise ValueError(f"unsupported TileLang thread axis scope `{axis.loom_scope}`")


def collect_thread_extents(stmt: object) -> Mapping[str, object]:
    """Collects thread extents from nested AttrStmt/For wrappers."""

    extents: dict[str, object] = {}
    _collect_thread_extents(stmt, extents)
    return extents


def integer_value(value: object) -> int | None:
    """Returns a Python integer for immediate-like source values."""

    payload: Any = getattr(value, "value", value)
    try:
        return int(payload)
    except (TypeError, ValueError):
        return None


def _collect_thread_extents(stmt: object, extents: dict[str, object]) -> None:
    kind = node_kind(stmt)
    if kind == "AttrStmt":
        if str(getattr(stmt, "attr_key", "")) == "thread_extent":
            tag = thread_tag(getattr(stmt, "node", None))
            if tag is not None:
                extents.setdefault(tag, getattr(stmt, "value", None))
        _collect_thread_extents(getattr(stmt, "body", None), extents)
        return
    if kind == "For":
        axis = thread_axis_from_binding(getattr(stmt, "thread_binding", None))
        if axis is not None:
            extents.setdefault(axis.source_tag, getattr(stmt, "extent", None))
        _collect_thread_extents(getattr(stmt, "body", None), extents)
        return
    if kind == "SeqStmt":
        for child in tuple(getattr(stmt, "seq", ())):
            _collect_thread_extents(child, extents)
        return
    for attr in ("body", "then_case", "else_case", "block"):
        child = getattr(stmt, attr, None)
        if child is not None:
            _collect_thread_extents(child, extents)


def mapped_thread_axis_sources(
    *,
    loop_var: object | None = None,
    binding: object | None = None,
) -> tuple[object, ...]:
    """Returns source objects that should resolve to the same topology value."""

    sources: list[object] = []
    for source in (loop_var, binding, thread_binding_var(binding)):
        if source is not None and all(source is not existing for existing in sources):
            sources.append(source)
    return tuple(sources)


def thread_axis_name(source: object | None, axis: ThreadAxis) -> str:
    """Chooses a stable SSA name for an imported thread-axis value."""

    if source is None:
        return axis.default_name
    binding_var = thread_binding_var(source)
    if binding_var is not None:
        return sanitize_identifier(
            source_name(binding_var, fallback=axis.default_name),
            fallback=axis.default_name,
        )
    return sanitize_identifier(
        source_name(source, fallback=axis.default_name),
        fallback=axis.default_name,
    )
