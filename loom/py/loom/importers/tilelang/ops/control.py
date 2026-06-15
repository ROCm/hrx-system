# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured control-flow TileLang/TIR node converters."""

from __future__ import annotations

from dataclasses import dataclass

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier
from loom.importers.tilelang.context import (
    TileLangBufferAccessKey,
    TileLangConversionContext,
    TileLangFragmentVector,
)
from loom.importers.tilelang.converter import (
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import node_kind, node_text, source_name
from loom.importers.tilelang.ops.calls import is_thread_return_call
from loom.importers.tilelang.ops.conditions import coerce_condition
from loom.importers.tilelang.ops.memory import try_convert_parallel_vector_store
from loom.importers.tilelang.ops.topology import (
    integer_value,
    map_thread_axis,
    mapped_thread_axis_sources,
    thread_axis_from_binding,
    thread_axis_name,
)
from loom.ir import INDEX, Type

TIR_FOR_SERIAL = 0
TIR_FOR_PARALLEL = 1
TIR_FOR_UNROLLED = 3


@dataclass(frozen=True, slots=True)
class _FragmentVectorStateSlot:
    view_id: int
    view: ValueRef
    initial: TileLangFragmentVector
    name: str


@dataclass(frozen=True, slots=True)
class _BufferAccessStateSlot:
    key: TileLangBufferAccessKey
    initial: ValueRef
    name: str


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("SeqStmt", convert_seq_stmt)
    registry.register_statement("For", convert_for)
    registry.register_statement("While", convert_while)
    registry.register_statement("IfThenElse", convert_if_then_else)
    registry.register_statement("Evaluate", convert_evaluate)


def convert_seq_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR statement sequence by preserving source order."""

    for child in tuple(getattr(stmt, "seq", ())):
        if convert_thread_return_prefix_guard(child, context, converter):
            continue
        converter.convert_stmt(child, context)


def convert_for(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR for-loop as scf.for."""

    loop_var = getattr(stmt, "loop_var", None)
    thread_binding = getattr(stmt, "thread_binding", None)
    thread_axis = thread_axis_from_binding(thread_binding)
    if thread_binding is not None and thread_axis is None:
        context.record_blocked(node_text(stmt), "thread binding axis is not mapped")
        return
    if thread_axis is not None:
        context.map_topology_extent(
            thread_axis.source_tag,
            None,
            static_extent=integer_value(getattr(stmt, "extent", None)),
        )
        thread_value = map_thread_axis(
            axis=thread_axis,
            sources=mapped_thread_axis_sources(
                loop_var=loop_var,
                binding=thread_binding,
            ),
            context=context,
            converter=converter,
            lower=getattr(stmt, "min", None),
            name=thread_axis_name(loop_var, thread_axis),
        )
        if thread_value is None:
            context.record_blocked(
                node_text(stmt),
                "thread binding lower bound is not mapped",
            )
            return
        converter.convert_stmt(getattr(stmt, "body", None), context)
        operation_name = f"kernel.{thread_axis.loom_scope}.id<{thread_axis.dimension}>"
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(thread_value)} = {operation_name}",
        )
        return

    loop_kind = integer_value(getattr(stmt, "kind", None))
    if loop_kind == TIR_FOR_PARALLEL:
        _convert_parallel_for(stmt, context, converter)
        return
    record_name = (
        "tir.For unrolled<scf.for>" if loop_kind == TIR_FOR_UNROLLED else "scf.for"
    )
    unroll_policy = "unroll" if loop_kind == TIR_FOR_UNROLLED else None
    _convert_counted_for(
        stmt,
        context,
        converter,
        record_name=record_name,
        unroll_policy=unroll_policy,
    )


def _convert_counted_for(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    record_name: str,
    unroll_policy: str | None,
) -> None:
    """Import a TileLang/TIR counted loop as structured scf.for."""

    loop_var = getattr(stmt, "loop_var", None)
    loop_name = sanitize_identifier(source_name(loop_var, fallback="iv"))
    lower = converter.convert_expr(
        getattr(stmt, "min", None),
        context,
        index_like=True,
    )
    extent = converter.convert_expr(
        getattr(stmt, "extent", None),
        context,
        index_like=True,
    )
    if lower is None or extent is None:
        context.record_blocked(node_text(stmt), "loop bounds are not mapped")
        return
    if integer_value(getattr(stmt, "min", None)) == 0:
        upper = extent
    else:
        upper = context.builder.index.add(
            lhs=lower,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name(f"{loop_name}_ub"),
        )
    step = context.ensure_constant("1", "index", "c1")
    state_slots = _fragment_vector_state_slots_written_by(
        getattr(stmt, "body", None),
        context,
    )
    buffer_slots = _buffer_access_state_slots_written_by(
        getattr(stmt, "body", None),
        context,
    )
    body_args: list[tuple[str, Type]] = [(loop_name, INDEX)]
    body_args.extend(
        (f"{slot.name}_iter", slot.initial.value.type) for slot in state_slots
    )
    body_args.extend((f"{slot.name}_iter", slot.initial.type) for slot in buffer_slots)
    result_names = [
        *(context.fresh_name(f"{slot.name}_next") for slot in state_slots),
        *(context.fresh_name(f"{slot.name}_next") for slot in buffer_slots),
    ]
    body = context.builder.region(args=body_args)
    loop_var_ref = ValueRef(body.blocks[0].arg_ids[0], context.builder.ir)
    child = context.fork(preview_block=body.blocks[0])
    child.map_value(loop_var, loop_var_ref, "index")
    for slot, arg_id in zip(
        state_slots,
        body.blocks[0].arg_ids[1 : 1 + len(state_slots)],
        strict=True,
    ):
        child.map_fragment_vector(
            slot.view,
            TileLangFragmentVector(
                value=ValueRef(arg_id, context.builder.ir),
                lane_count=slot.initial.lane_count,
                base=slot.initial.base,
            ),
        )
    buffer_arg_offset = 1 + len(state_slots)
    for buffer_slot, arg_id in zip(
        buffer_slots,
        body.blocks[0].arg_ids[buffer_arg_offset:],
        strict=True,
    ):
        child.map_buffer_access_key(
            buffer_slot.key,
            ValueRef(arg_id, context.builder.ir),
        )
    with context.builder.insertion_block(body.blocks[0]):
        converter.convert_stmt(getattr(stmt, "body", None), child)
        context.builder.scf.yield_(
            values=[
                *_fragment_vector_state_values(state_slots, child),
                *_buffer_access_state_values(buffer_slots, child),
            ]
        )
    context.merge_child_records(child)
    loop_results = context.builder.scf.for_(
        lower_bound=lower,
        upper_bound=upper,
        step=step,
        unroll_policy=unroll_policy,
        iter_args=[
            *(slot.initial.value for slot in state_slots),
            *(slot.initial for slot in buffer_slots),
        ],
        results=[
            *(slot.initial.value.type for slot in state_slots),
            *(slot.initial.type for slot in buffer_slots),
        ],
        body=body,
        names=result_names or None,
    )
    _map_state_results(state_slots, buffer_slots, loop_results, context)
    context.record_converted(node_text(stmt), record_name)


def _convert_parallel_for(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import TileLang's parallel loop form conservatively."""

    loop_var = getattr(stmt, "loop_var", None)
    loop_name = sanitize_identifier(source_name(loop_var, fallback="i"))
    lower = getattr(stmt, "min", None)
    lower_integer = integer_value(lower)
    if lower_integer != 0:
        context.record_blocked(node_text(stmt), "parallel loop lower bound is not zero")
        return
    extent_source = getattr(stmt, "extent", None)
    extent_integer = integer_value(extent_source)
    if extent_integer is None:
        context.record_blocked(
            node_text(stmt),
            "parallel loop requires static extent",
        )
        return

    if try_convert_parallel_vector_store(
        stmt,
        context,
        converter,
        loop_var=loop_var,
        extent_integer=extent_integer,
        index_name=loop_name,
    ):
        context.record_converted(node_text(stmt), "tir.For parallel<vector.store>")
        return

    _convert_counted_for(
        stmt,
        context,
        converter,
        record_name="tir.For parallel<scf.for>",
        unroll_policy=None,
    )


def convert_while(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR while-loop as resultless scf.while."""

    before_region = context.builder.region()
    before_child = context.fork(preview_block=before_region.blocks[0])
    with context.builder.insertion_block(before_region.blocks[0]):
        condition = converter.convert_expr(
            getattr(stmt, "condition", None),
            before_child,
        )
        if condition is None:
            before_child.record_blocked(
                node_text(stmt), "while condition is not mapped"
            )
            context.merge_child_records(before_child)
            return
        condition = coerce_condition(condition, before_child, stmt)
        if condition is None:
            context.merge_child_records(before_child)
            return
        context.builder.scf.condition(condition=condition, forwarded=[])
    context.merge_child_records(before_child)

    after_region = context.builder.region()
    after_child = context.fork(preview_block=after_region.blocks[0])
    with context.builder.insertion_block(after_region.blocks[0]):
        converter.convert_stmt(getattr(stmt, "body", None), after_child)
        context.builder.scf.yield_()
    context.merge_child_records(after_child)
    context.builder.scf.while_(
        iter_args=[],
        results=[],
        before=before_region,
        after=after_region,
    )
    context.record_converted(node_text(stmt), "scf.while")


def convert_if_then_else(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR if-then-else as scf.if."""

    if convert_thread_return_prefix_guard(stmt, context, converter):
        return

    condition = converter.convert_expr(getattr(stmt, "condition", None), context)
    if condition is None:
        context.record_blocked(node_text(stmt), "if condition is not mapped")
        return
    condition = coerce_condition(condition, context, stmt)
    if condition is None:
        return
    then_region = context.builder.region()
    else_case = getattr(stmt, "else_case", None)
    else_region = context.builder.region() if else_case is not None else None
    then_child = context.fork(preview_block=then_region.blocks[0])
    with context.builder.insertion_block(then_region.blocks[0]):
        converter.convert_stmt(getattr(stmt, "then_case", None), then_child)
    else_child = None
    if else_region is not None:
        else_child = context.fork(preview_block=else_region.blocks[0])
        with context.builder.insertion_block(else_region.blocks[0]):
            converter.convert_stmt(else_case, else_child)
    state_slots = _changed_fragment_vector_state_slots(
        context,
        then_child,
        else_child,
    )
    buffer_slots = _changed_buffer_access_state_slots(
        context,
        then_child,
        else_child,
    )
    if else_region is None and (state_slots or buffer_slots):
        else_region = context.builder.region()
    with context.builder.insertion_block(then_region.blocks[0]):
        context.builder.scf.yield_(
            values=[
                *_fragment_vector_state_values(state_slots, then_child),
                *_buffer_access_state_values(buffer_slots, then_child),
            ]
        )
    context.merge_child_records(then_child)
    if else_region is not None:
        with context.builder.insertion_block(else_region.blocks[0]):
            if else_child is not None:
                context.builder.scf.yield_(
                    values=[
                        *_fragment_vector_state_values(state_slots, else_child),
                        *_buffer_access_state_values(buffer_slots, else_child),
                    ]
                )
                context.merge_child_records(else_child)
            else:
                context.builder.scf.yield_(
                    values=[
                        *(slot.initial.value for slot in state_slots),
                        *(slot.initial for slot in buffer_slots),
                    ]
                )
    results = context.builder.scf.if_(
        condition=condition,
        results=[
            *(slot.initial.value.type for slot in state_slots),
            *(slot.initial.type for slot in buffer_slots),
        ],
        then_region=then_region,
        else_region=else_region,
        names=[
            *(context.fresh_name(f"{slot.name}_if") for slot in state_slots),
            *(context.fresh_name(f"{slot.name}_if") for slot in buffer_slots),
        ]
        or None,
    )
    _map_state_results(state_slots, buffer_slots, results, context)
    context.record_converted(node_text(stmt), "scf.if")


def _changed_fragment_vector_state_slots(
    parent: TileLangConversionContext,
    *children: TileLangConversionContext | None,
) -> list[_FragmentVectorStateSlot]:
    slots: list[_FragmentVectorStateSlot] = []
    for view_id, initial in parent.fragment_vectors.items():
        changed = False
        for child in children:
            if child is None:
                continue
            current = child.fragment_vectors.get(view_id)
            if current is None:
                continue
            if current.value.id != initial.value.id:
                changed = True
                break
        if changed:
            slots.append(_fragment_vector_state_slot(parent, view_id, initial))
    return slots


def _changed_buffer_access_state_slots(
    parent: TileLangConversionContext,
    *children: TileLangConversionContext | None,
) -> list[_BufferAccessStateSlot]:
    slots: list[_BufferAccessStateSlot] = []
    for key, initial in parent.buffer_access_values.items():
        changed = False
        for child in children:
            if child is None:
                continue
            current = child.buffer_access_values.get(key)
            if current is None:
                continue
            if current.id != initial.id:
                changed = True
                break
        if changed:
            slots.append(_buffer_access_state_slot(parent, key, initial))
    return slots


def _fragment_vector_state_slots_written_by(
    stmt: object | None,
    context: TileLangConversionContext,
) -> list[_FragmentVectorStateSlot]:
    view_ids: set[int] = set()
    _collect_fragment_vector_store_view_ids(stmt, context, view_ids)
    slots: list[_FragmentVectorStateSlot] = []
    for view_id in sorted(view_ids):
        initial = context.fragment_vectors.get(view_id)
        if initial is None:
            continue
        slots.append(_fragment_vector_state_slot(context, view_id, initial))
    return slots


def _buffer_access_state_slots_written_by(
    stmt: object | None,
    context: TileLangConversionContext,
) -> list[_BufferAccessStateSlot]:
    keys: set[TileLangBufferAccessKey] = set()
    _collect_buffer_access_store_keys(stmt, context, keys)
    slots: list[_BufferAccessStateSlot] = []
    for key in sorted(keys, key=lambda item: (item.view_id, item.index_ids)):
        initial = context.buffer_access_values.get(key)
        if initial is None:
            continue
        slots.append(_buffer_access_state_slot(context, key, initial))
    return slots


def _fragment_vector_state_slot(
    context: TileLangConversionContext,
    view_id: int,
    initial: TileLangFragmentVector,
) -> _FragmentVectorStateSlot:
    view_value = context.builder.module.values[view_id]
    base_name = sanitize_identifier(view_value.name or "fragment")
    return _FragmentVectorStateSlot(
        view_id=view_id,
        view=ValueRef(view_id, context.builder.ir),
        initial=initial,
        name=f"{base_name}_state",
    )


def _buffer_access_state_slot(
    context: TileLangConversionContext,
    key: TileLangBufferAccessKey,
    initial: ValueRef,
) -> _BufferAccessStateSlot:
    view_value = context.builder.module.values[key.view_id]
    base_name = sanitize_identifier(view_value.name or "local")
    return _BufferAccessStateSlot(
        key=key,
        initial=initial,
        name=f"{base_name}_state",
    )


def _fragment_vector_state_values(
    slots: list[_FragmentVectorStateSlot],
    context: TileLangConversionContext,
) -> list[ValueRef]:
    values: list[ValueRef] = []
    for slot in slots:
        current = context.fragment_vectors.get(slot.view_id)
        values.append(current.value if current is not None else slot.initial.value)
    return values


def _buffer_access_state_values(
    slots: list[_BufferAccessStateSlot],
    context: TileLangConversionContext,
) -> list[ValueRef]:
    return [context.buffer_access_values.get(slot.key, slot.initial) for slot in slots]


def _map_state_results(
    slots: list[_FragmentVectorStateSlot],
    buffer_slots: list[_BufferAccessStateSlot],
    results: list[ValueRef],
    context: TileLangConversionContext,
) -> None:
    fragment_results = results[: len(slots)]
    buffer_results = results[len(slots) :]
    for slot, result in zip(slots, fragment_results, strict=True):
        context.map_fragment_vector(
            slot.view,
            TileLangFragmentVector(
                value=result,
                lane_count=slot.initial.lane_count,
                base=slot.initial.base,
            ),
        )
    for buffer_slot, result in zip(buffer_slots, buffer_results, strict=True):
        context.map_buffer_access_key(buffer_slot.key, result)


def _collect_fragment_vector_store_view_ids(
    stmt: object | None,
    context: TileLangConversionContext,
    out_view_ids: set[int],
) -> None:
    if stmt is None:
        return
    kind = node_kind(stmt)
    if kind == "BufferStore":
        view = _stored_local_fragment_view(stmt, context)
        if view is not None:
            out_view_ids.add(view.id)
        return
    if kind == "SeqStmt":
        for child in tuple(getattr(stmt, "seq", ()) or ()):
            _collect_fragment_vector_store_view_ids(child, context, out_view_ids)
        return
    if kind == "IfThenElse":
        _collect_fragment_vector_store_view_ids(
            getattr(stmt, "then_case", None),
            context,
            out_view_ids,
        )
        _collect_fragment_vector_store_view_ids(
            getattr(stmt, "else_case", None),
            context,
            out_view_ids,
        )
        return
    for field_name in ("body", "block", "then_case", "else_case"):
        child = getattr(stmt, field_name, None)
        if child is not None:
            _collect_fragment_vector_store_view_ids(child, context, out_view_ids)


def _collect_buffer_access_store_keys(
    stmt: object | None,
    context: TileLangConversionContext,
    out_keys: set[TileLangBufferAccessKey],
) -> None:
    if stmt is None:
        return
    kind = node_kind(stmt)
    if kind == "BufferStore":
        key = _stored_tracked_buffer_access_key(stmt, context)
        if key is not None:
            out_keys.add(key)
        return
    if kind == "SeqStmt":
        for child in tuple(getattr(stmt, "seq", ()) or ()):
            _collect_buffer_access_store_keys(child, context, out_keys)
        return
    if kind == "IfThenElse":
        _collect_buffer_access_store_keys(
            getattr(stmt, "then_case", None),
            context,
            out_keys,
        )
        _collect_buffer_access_store_keys(
            getattr(stmt, "else_case", None),
            context,
            out_keys,
        )
        return
    for field_name in ("body", "block", "then_case", "else_case"):
        child = getattr(stmt, field_name, None)
        if child is not None:
            _collect_buffer_access_store_keys(child, context, out_keys)


def _stored_local_fragment_view(
    stmt: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    buffer = getattr(stmt, "buffer", None)
    if _buffer_scope(buffer) != "local.fragment":
        return None
    view = context.mapped(buffer)
    if view is not None:
        return view if context.fragment_vector(view) is not None else None
    data = getattr(buffer, "data", None)
    view = context.mapped_buffer_data(data)
    if view is None:
        return None
    return view if context.fragment_vector(view) is not None else None


def _stored_tracked_buffer_access_key(
    stmt: object,
    context: TileLangConversionContext,
) -> TileLangBufferAccessKey | None:
    buffer = getattr(stmt, "buffer", None)
    memory_scope = _buffer_scope(buffer)
    if memory_scope != "local.var":
        return None
    view = context.mapped(buffer)
    if view is None:
        data = getattr(buffer, "data", None)
        view = context.mapped_buffer_data(data)
    if view is None:
        return None
    indices: list[ValueRef] = []
    for source_index in tuple(getattr(stmt, "indices", ()) or ()):
        index = _mapped_existing_index(source_index, context)
        if index is None:
            return None
        indices.append(index)
    key = context.buffer_access_key(view, tuple(indices), memory_scope)
    if key is None or key not in context.buffer_access_values:
        return None
    return key


def _mapped_existing_index(
    source: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if isinstance(source, ValueRef):
        return source
    mapped = context.mapped_index_value(source)
    if mapped is not None:
        return mapped
    value = integer_value(source)
    if value is None:
        return None
    return context.constants.get(("index", str(value)))


def _buffer_scope(buffer: object | None) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        scope = scope()
    return str(scope or "")


def convert_evaluate(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import an effect expression wrapper."""

    if convert_thread_return_prefix_guard(stmt, context, converter):
        return

    value = getattr(stmt, "value", None)
    if value is not None:
        converter.convert_expr(value, context, effect=True)


def convert_thread_return_prefix_guard(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    """Import a top-level `if cond: tir.thread_return()` as kernel.exit."""

    if not context.can_emit_kernel_exit():
        return False

    if node_kind(stmt) == "Evaluate" and _is_thread_return_statement(stmt):
        exit_condition = context.ensure_constant("1", "bool", "thread_return")
        context.builder.kernel.exit(condition=exit_condition)
        context.record_converted(node_text(stmt), "kernel.exit")
        return True

    if node_kind(stmt) != "IfThenElse":
        return False
    if getattr(stmt, "else_case", None) is not None:
        return False

    exit_path = _thread_return_exit_path(getattr(stmt, "then_case", None))
    if exit_path is None:
        return False

    condition = converter.convert_expr(getattr(stmt, "condition", None), context)
    if condition is None:
        context.record_blocked(
            node_text(stmt),
            "thread_return guard condition is not mapped",
        )
        return True
    condition = coerce_condition(condition, context, stmt)
    if condition is None:
        return True

    body = None
    if exit_path:
        body = context.builder.region()
        child = context.fork(preview_block=body.blocks[0])
        with context.builder.insertion_block(body.blocks[0]):
            for body_stmt in exit_path:
                converter.convert_stmt(body_stmt, child)
            context.builder.kernel.return_()
        context.merge_child_records(child)

    context.builder.kernel.exit(condition=condition, body=body)
    context.record_converted(node_text(stmt), "kernel.exit")
    return True


def _thread_return_exit_path(stmt: object | None) -> tuple[object, ...] | None:
    if stmt is None:
        return None
    if _is_thread_return_statement(stmt):
        return ()
    if node_kind(stmt) != "SeqStmt":
        return None
    statements = tuple(getattr(stmt, "seq", ()))
    if not statements or not _is_thread_return_statement(statements[-1]):
        return None
    return statements[:-1]


def _is_thread_return_statement(stmt: object) -> bool:
    if node_kind(stmt) != "Evaluate":
        return False
    return is_thread_return_call(getattr(stmt, "value", None))
