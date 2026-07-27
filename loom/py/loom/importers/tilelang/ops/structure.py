# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural TileLang/TIR wrapper node converters."""

from __future__ import annotations

from collections.abc import Iterable

from loom.importers.tilelang.context import (
    TileLangConversionContext,
    TileLangReducerInfo,
)
from loom.importers.tilelang.converter import (
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import attrs, mapping_items, node_text, source_name
from loom.importers.tilelang.ops.assumptions import (
    ASSUME_ATTR_KEYS,
    convert_assume_attr_stmt,
)
from loom.importers.tilelang.ops.memory import map_alloc_buffer
from loom.importers.tilelang.ops.topology import (
    integer_value,
    map_thread_axis,
    mapped_thread_axis_sources,
    thread_axis_from_binding,
    thread_axis_name,
    thread_tag,
)


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("Block", convert_block)
    registry.register_statement("BlockRealize", convert_block_realize)
    registry.register_statement("Bind", convert_bind)
    registry.register_statement("LetStmt", convert_let_stmt)
    registry.register_statement("AttrStmt", convert_attr_stmt)


def convert_block(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR block wrapper when it has no extra region semantics."""

    for buffer in tuple(getattr(stmt, "alloc_buffers", ()) or ()):
        if not map_alloc_buffer(buffer, context):
            return
    if not _import_reducer_infos(stmt, context):
        return
    if not _import_local_var_initializers(stmt, context, converter):
        return
    if _has_items(getattr(stmt, "match_buffers", ())):
        context.record_blocked(node_text(stmt), "block match buffers are not mapped")
        return
    init = getattr(stmt, "init", None)
    if init is not None:
        context.record_blocked(node_text(stmt), "block init region is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), "tir.Block normalized")


def _import_local_var_initializers(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    """Import TileLang alloc_var initializer annotations as explicit stores."""

    local_var_init = _metadata_attrs(stmt).get("tl.local_var_init")
    if local_var_init is None:
        return True
    zero = context.ensure_constant("0", "index", "c0")
    for data_var, init_expr in mapping_items(local_var_init):
        view = context.mapped_buffer_data(data_var)
        if view is None:
            context.record_blocked(
                node_text(stmt),
                "local variable initializer target is not mapped",
            )
            return False
        init_value = converter.convert_expr(init_expr, context)
        if init_value is None:
            context.record_blocked(
                node_text(stmt),
                "local variable initializer value is not mapped",
            )
            return False
        context.map_buffer_access(
            init_expr,
            view,
            (zero,),
            "local.var",
            init_value,
        )
        context.record_converted(node_text(init_expr), "local var initializer")
    return True


def _import_reducer_infos(
    stmt: object,
    context: TileLangConversionContext,
) -> bool:
    reducer_infos = _metadata_attrs(stmt).get("reducer_info")
    if reducer_infos is None:
        return True
    reducer_info_items = mapping_items(reducer_infos)
    if not reducer_info_items and bool(reducer_infos):
        context.record_blocked(node_text(stmt), "reducer_info map is not mapped")
        return False
    for data_var, source_info in reducer_info_items:
        reducer_info = _decode_reducer_info(source_info, stmt, context)
        if reducer_info is None:
            return False
        context.map_reducer_info(data_var, reducer_info)
        context.record_converted(
            node_text(data_var),
            f"reducer_info {reducer_info.operation}/{reducer_info.replication}",
        )
    return True


def _decode_reducer_info(
    source_info: object,
    owner: object,
    context: TileLangConversionContext,
) -> TileLangReducerInfo | None:
    fields = {str(key): value for key, value in mapping_items(source_info)}
    if fields:
        unknown_fields = sorted(set(fields) - {"op", "rep"})
        if unknown_fields:
            context.record_blocked(
                node_text(owner),
                ("reducer_info fields are not imported: " + ", ".join(unknown_fields)),
            )
            return None
        operation = _decode_reducer_operation(fields.get("op"))
        replication = _decode_reducer_replication(fields.get("rep"))
    else:
        operation = _decode_reducer_operation(getattr(source_info, "op", None))
        replication = _decode_reducer_replication(getattr(source_info, "rep", None))
    if operation is None:
        context.record_blocked(node_text(owner), "reducer_info op is not mapped")
        return None
    if replication is None:
        context.record_blocked(node_text(owner), "reducer_info rep is not mapped")
        return None
    return TileLangReducerInfo(operation=operation, replication=replication)


def _decode_reducer_operation(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value in _REDUCER_OPERATIONS else None
    if isinstance(value, int):
        return _REDUCER_OPERATION_CODES.get(value)
    payload = getattr(value, "value", None)
    if isinstance(payload, int):
        return _REDUCER_OPERATION_CODES.get(payload)
    text = str(value)
    if text in _REDUCER_OPERATIONS:
        return text
    return None


def _decode_reducer_replication(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value in _REDUCER_REPLICATIONS else None
    if isinstance(value, int):
        return _REDUCER_REPLICATION_CODES.get(value)
    payload = getattr(value, "value", None)
    if isinstance(payload, int):
        return _REDUCER_REPLICATION_CODES.get(payload)
    text = str(value)
    if text in _REDUCER_REPLICATIONS:
        return text
    return None


def _metadata_attrs(value: object) -> dict[str, object]:
    result = dict(attrs(value))
    for key, item in mapping_items(getattr(value, "annotations", {})):
        result[str(key)] = item
    return result


def convert_block_realize(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR BlockRealize wrapper with a true predicate."""

    predicate = getattr(stmt, "predicate", None)
    if predicate is not None and not _is_true_predicate(predicate):
        context.record_blocked(node_text(stmt), "block predicate is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "block", None), context)
    context.record_converted(node_text(stmt), "tir.BlockRealize normalized")


def convert_let_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR let statement to an SSA value binding."""

    value = converter.convert_expr(getattr(stmt, "value", None), context)
    if value is None:
        context.record_blocked(node_text(stmt), "let value is not mapped")
        return
    var = getattr(stmt, "var", None)
    name = source_name(var, fallback="let")
    context.map_value(var, value, str(value.type))
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), f"tir.LetStmt %{name} normalized")


def convert_bind(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TileLang eager Bind statement to an SSA value binding."""

    value = converter.convert_expr(getattr(stmt, "value", None), context)
    if value is None:
        context.record_blocked(node_text(stmt), "bind value is not mapped")
        return
    var = getattr(stmt, "var", None)
    name = source_name(var, fallback="bind")
    context.map_value(var, value, str(value.type))
    context.record_converted(node_text(stmt), f"tir.Bind %{name} normalized")


def convert_attr_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize known metadata-only AttrStmt wrappers."""

    attr_key = str(getattr(stmt, "attr_key", ""))
    if attr_key in ASSUME_ATTR_KEYS:
        convert_assume_attr_stmt(stmt, context, converter)
        return
    if attr_key == "thread_extent":
        node = getattr(stmt, "node", None)
        thread_axis = thread_axis_from_binding(node)
        extent_value = getattr(stmt, "value", None)
        if thread_axis is None:
            if thread_tag(node) is not None:
                context.record_blocked(
                    node_text(stmt),
                    "thread_extent axis is not mapped",
                )
                return
            converter.convert_stmt(getattr(stmt, "body", None), context)
            context.record_converted(
                node_text(stmt),
                "AttrStmt `thread_extent` normalized",
            )
            return
        context.map_topology_extent(
            thread_axis.source_tag,
            None,
            static_extent=integer_value(extent_value),
        )
        thread_value = map_thread_axis(
            axis=thread_axis,
            sources=mapped_thread_axis_sources(binding=node),
            context=context,
            converter=converter,
            name=thread_axis_name(node, thread_axis),
        )
        if thread_value is None:
            context.record_blocked(node_text(stmt), "thread_extent value is not mapped")
            return
        converter.convert_stmt(getattr(stmt, "body", None), context)
        operation_name = f"kernel.{thread_axis.loom_scope}.id<{thread_axis.dimension}>"
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(thread_value)} = {operation_name}",
        )
        return
    if attr_key not in _NORMALIZED_ATTR_KEYS:
        context.record_blocked(node_text(stmt), f"AttrStmt `{attr_key}` is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), f"AttrStmt `{attr_key}` normalized")


def _has_items(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return True
    return bool(tuple(value))


def _is_true_predicate(value: object) -> bool:
    if value is True:
        return True
    payload = getattr(value, "value", None)
    if payload is not None:
        return bool(payload)
    return bool(value)


_NORMALIZED_ATTR_KEYS = {
    "thread_extent",
    "threadblock_swizzle_pattern",
}

_REDUCER_OPERATIONS = frozenset(("sum", "max", "min"))
_REDUCER_OPERATION_CODES = {
    0: "sum",
    1: "max",
    2: "min",
}
_REDUCER_REPLICATIONS = frozenset(("all", "none"))
_REDUCER_REPLICATION_CODES = {
    0: "all",
    1: "none",
}
