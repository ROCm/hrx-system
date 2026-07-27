# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer load/store TileLang/TIR node converters."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier
from loom.importers.tilelang.buffers import (
    TileLangBufferAccess,
    resolve_buffer_access,
    resolve_buffer_index_map,
)
from loom.importers.tilelang.context import (
    TileLangConversionContext,
    TileLangDistributedIndex,
    TileLangFragmentVector,
)
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_kind, node_text, source_name
from loom.importers.tilelang.ops.distribution import (
    MAX_DISTRIBUTED_UNROLL_LANES,
    materialize_distributed_1d_plan,
)
from loom.importers.tilelang.ops.topology import integer_value
from loom.importers.tilelang.ops.vector import vector_lanes
from loom.ir import (
    BUFFER_TYPE,
    INDEX,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
)


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("BufferStore", convert_buffer_store)
    registry.register_expression("BufferLoad", convert_buffer_load)


def try_convert_parallel_vector_store(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    loop_var: object,
    extent_integer: int,
    index_name: str,
) -> bool:
    """Import a 1D distributed parallel store as one vector.store per workitem."""

    body = _single_statement_body(getattr(stmt, "body", None))
    if node_kind(body) != "BufferStore":
        return False
    source_indices = tuple(getattr(body, "indices", ()))
    target_axis_position = _direct_loop_var_axis(source_indices, loop_var, context)
    if target_axis_position is None:
        return False
    target_index_map = resolve_buffer_index_map(
        getattr(body, "buffer", None),
        context,
        converter,
        diagnostic_owner=body,
    )
    if target_index_map is None:
        return False
    if (
        not target_index_map.is_identity
        or target_axis_position != target_index_map.logical_rank - 1
    ):
        return False
    if not _parallel_store_value_is_vectorizable(
        body,
        context,
        converter,
        loop_var=loop_var,
    ):
        return False

    thread_count = context.static_topology_extent("threadIdx.x")
    if thread_count is None or thread_count <= 0:
        return False
    if extent_integer <= thread_count or extent_integer % thread_count != 0:
        return False
    lane_count = extent_integer // thread_count
    if lane_count > MAX_DISTRIBUTED_UNROLL_LANES:
        return False

    plan = materialize_distributed_1d_plan(
        stmt,
        context,
        converter,
        extent_integer=extent_integer,
        index_name=index_name,
    )
    if plan is None or plan.lane_count != lane_count:
        return False

    child = context.fork(preview_block=context.builder.ir.insertion_block)
    child.map_value(loop_var, plan.base, "index")
    child.map_index_value(loop_var, plan.base)
    child.map_distributed_index(
        plan.base,
        TileLangDistributedIndex(
            base=plan.base,
            lane=0,
            lane_count=plan.lane_count,
        ),
    )
    access = resolve_buffer_access(
        getattr(body, "buffer", None),
        source_indices,
        child,
        converter,
        diagnostic_owner=body,
    )
    if access is None:
        return False
    if target_axis_position != len(access.indices) - 1:
        return False

    vector_value = _parallel_store_vector_value(
        body,
        child,
        converter,
        lane_count=plan.lane_count,
        target_view=access.view,
    )
    if vector_value is None:
        return False

    indices: list[int | ValueRef] = list(access.indices)
    indices[target_axis_position] = plan.base
    child.invalidate_buffer_accesses(access.view)
    child.builder.vector.store(
        value=vector_value,
        view=access.view,
        indices=indices,
    )
    child.record_converted(node_text(body), "vector.store distributed")
    context.merge_child_records(child)
    return True


def _single_statement_body(stmt: object) -> object:
    if node_kind(stmt) != "SeqStmt":
        return stmt
    children = tuple(getattr(stmt, "seq", ()) or ())
    if len(children) != 1:
        return stmt
    return children[0]


def _direct_loop_var_axis(
    source_indices: tuple[object, ...],
    loop_var: object,
    context: TileLangConversionContext,
) -> int | None:
    loop_key = context.source_key(loop_var)
    match = None
    for axis, source_index in enumerate(source_indices):
        if context.source_key(source_index) != loop_key and not _same_source_var(
            source_index, loop_var
        ):
            continue
        if match is not None:
            return None
        match = axis
    return match


def _same_source_var(lhs: object, rhs: object) -> bool:
    if node_kind(lhs) not in ("Var", "SizeVar") or node_kind(rhs) not in (
        "Var",
        "SizeVar",
    ):
        return False
    return dtype(lhs) == dtype(rhs) and source_name(
        lhs, fallback=str(lhs)
    ) == source_name(
        rhs,
        fallback=str(rhs),
    )


def _parallel_store_value_is_vectorizable(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    loop_var: object,
) -> bool:
    value_expr = getattr(stmt, "value", None)
    if node_kind(value_expr) in ("FloatImm", "IntImm"):
        return True
    if node_kind(value_expr) != "BufferLoad":
        return False
    source_indices = tuple(getattr(value_expr, "indices", ()))
    source_axis_position = _direct_loop_var_axis(source_indices, loop_var, context)
    if source_axis_position is None:
        return False
    index_map = resolve_buffer_index_map(
        getattr(value_expr, "buffer", None),
        context,
        converter,
        diagnostic_owner=value_expr,
    )
    if index_map is None:
        return False
    return (
        index_map.is_identity
        and source_axis_position == index_map.logical_rank - 1
        and context.fragment_vector(index_map.view) is not None
    )


def _parallel_store_vector_value(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    lane_count: int,
    target_view: ValueRef,
) -> ValueRef | None:
    value_expr = getattr(stmt, "value", None)
    if node_kind(value_expr) == "BufferLoad":
        fragment_value = _fragment_vector_value_for_load(value_expr, context, converter)
        if fragment_value is None:
            return None
        if fragment_value.lane_count != lane_count:
            context.record_blocked(
                node_text(stmt),
                "local.fragment vector lane count does not match store distribution",
            )
            return None
        return fragment_value.value

    if node_kind(value_expr) not in ("FloatImm", "IntImm"):
        return None
    scalar = converter.convert_expr(value_expr, context)
    if scalar is None:
        return None
    scalar = _coerce_store_value(scalar, target_view, stmt, context)
    if scalar is None:
        return None
    result_type = context.type_converter.vector_type(scalar.type, lane_count)
    return context.builder.vector.splat(
        scalar=scalar,
        results=[result_type],
        name=context.fresh_name("store_splat"),
    )


def map_alloc_buffer(
    buffer: object,
    context: TileLangConversionContext,
) -> bool:
    """Import a TIR block allocation as a Loom scratch buffer root."""

    byte_length = context.type_converter.buffer_byte_length(buffer)
    if byte_length is None:
        context.record_blocked(
            node_text(buffer),
            "allocated buffer byte length is not statically known",
        )
        return False
    memory_space = _buffer_memory_space(buffer)
    if memory_space is None:
        context.record_blocked(
            node_text(buffer),
            f"allocated buffer scope `{_buffer_scope(buffer)}` is not mapped",
        )
        return False
    buffer_name = sanitize_identifier(source_name(buffer, fallback="scratch"))
    root = context.builder.buffer.alloca(
        byte_length=context.ensure_constant(
            str(byte_length),
            "offset",
            f"{buffer_name}_bytes",
        ),
        base_alignment=context.type_converter.buffer_base_alignment(buffer),
        memory_space=memory_space,
        results=[BUFFER_TYPE],
        name=context.fresh_name(f"{buffer_name}_buffer"),
    )
    view_type = context.buffer_view_type(buffer)
    view = context.builder.buffer.view(
        buffer=root,
        byte_offset=context.ensure_constant("0", "offset", "c0_bytes"),
        results=[view_type],
        name=context.fresh_name(buffer_name),
    )
    context.bind_buffer_view_layout(view, buffer)
    context.map_value(buffer, view, str(view.type))
    context.map_named_buffer(buffer, view)
    data = getattr(buffer, "data", None)
    if data is not None:
        context.map_buffer_data(data, view, buffer=buffer)
    _map_initial_fragment_vector(buffer, view, view_type, buffer_name, context)
    context.record_converted(
        node_text(buffer),
        (
            f"{context.ssa(root)} = buffer.alloca",
            f"{context.ssa(view)} = buffer.view",
        ),
    )
    return True


def _map_initial_fragment_vector(
    buffer: object,
    view: ValueRef,
    view_type: Type,
    buffer_name: str,
    context: TileLangConversionContext,
) -> None:
    if not _should_track_fragment_vector(buffer):
        return
    if not isinstance(view_type, ShapedType) or view_type.type_kind != TypeKind.VIEW:
        return
    if len(view_type.dims) != 1 or not isinstance(view_type.dims[0], StaticDim):
        return
    lane_count = view_type.dims[0].size
    if lane_count <= 0:
        return
    scalar_zero = _build_zero_fragment_seed(view_type.element_type, context)
    if scalar_zero is None:
        return
    initial = context.builder.vector.splat(
        scalar=scalar_zero,
        results=[
            context.type_converter.vector_type(view_type.element_type, lane_count)
        ],
        name=context.fresh_name(f"{buffer_name}_state"),
    )
    context.map_fragment_vector(
        view,
        TileLangFragmentVector(value=initial, lane_count=lane_count),
    )


def _should_track_fragment_vector(buffer: object) -> bool:
    if _buffer_scope(buffer) == "local.fragment":
        return True
    return (
        _buffer_scope(buffer) == "local"
        and str(getattr(buffer, "dtype", "")) in _FORMAT_PRESERVING_FP8_DTYPES
    )


def _build_zero_fragment_seed(
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if not isinstance(element_type, ScalarType):
        return None
    value: int | float
    if element_type.kind in (
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    ):
        value = 0.0
    else:
        value = 0
    return context.ensure_typed_constant(
        value,
        element_type,
        f"{element_type}_zero",
    )


def convert_buffer_store(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR buffer store as view.store."""

    source_indices = tuple(getattr(stmt, "indices", ()))
    buffer = getattr(stmt, "buffer", None)
    if _try_convert_distributed_fragment_store(
        stmt,
        buffer,
        source_indices,
        context,
        converter,
    ):
        return
    if _try_convert_format_preserving_fragment_store(
        stmt,
        buffer,
        source_indices,
        context,
        converter,
    ):
        return
    value = converter.convert_expr(getattr(stmt, "value", None), context)
    if value is None:
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    has_ramp_index = _has_ramp_index(stmt)
    if has_ramp_index and not is_vector_type(value.type):
        context.record_blocked(
            node_text(stmt), "vector store value is not vector-typed"
        )
        return
    if is_vector_type(value.type) or has_ramp_index:
        vector_indices = _vector_memory_indices(
            source_indices,
            context,
            converter,
        )
        if vector_indices is None:
            context.record_blocked(
                node_text(stmt), "vector store indices are not mapped"
            )
            return
        view = _resolve_vector_view(buffer, context, diagnostic_owner=stmt)
        if view is None:
            return
        context.builder.vector.store(
            value=value,
            view=view,
            indices=vector_indices,
        )
        if _buffer_scope(buffer) in ("shared", "shared.dyn"):
            context.mark_pending_workgroup_memory_write()
        context.invalidate_buffer_accesses(view)
        context.record_converted(node_text(stmt), "vector.store")
        return
    access = resolve_buffer_access(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=stmt,
    )
    if access is None:
        return
    if context.fragment_vector(access.view) is not None:
        fragment_value = _try_convert_fragment_vector_store(
            stmt,
            value,
            access,
            context,
        )
        if fragment_value is None:
            return
        context.map_fragment_vector(access.view, fragment_value)
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(fragment_value.value)} = vector.insert",
        )
        return
    if access.memory_scope == "local.var":
        value = _coerce_store_value(value, access.view, stmt, context)
        if value is None:
            return
        context.invalidate_buffer_accesses(access.view)
        context.map_buffer_access(
            stmt,
            access.view,
            access.indices,
            access.memory_scope,
            value,
        )
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(value)} = local.var",
        )
        return
    if context.fragment_vector(access.view) is not None:
        context.clear_fragment_vector(access.view)
    value = _coerce_store_value(value, access.view, stmt, context)
    if value is None:
        return
    context.invalidate_buffer_accesses(access.view)
    context.builder.view.store(
        value=value,
        view=access.view,
        indices=list(access.indices),
    )
    if access.memory_scope in ("shared", "shared.dyn"):
        context.mark_pending_workgroup_memory_write()
    context.map_buffer_access(
        stmt,
        access.view,
        access.indices,
        access.memory_scope,
        value,
    )
    context.record_converted(node_text(stmt), "view.store")


def convert_buffer_load(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR buffer load as view.load."""

    source_indices = tuple(getattr(expr, "indices", ()))
    buffer = getattr(expr, "buffer", None)
    result_type = context.type_converter.map_dtype(dtype(expr))
    if is_vector_type(result_type) or _has_ramp_index(expr):
        vector_indices = _vector_memory_indices(source_indices, context, converter)
        if vector_indices is None:
            context.record_blocked(
                node_text(expr), "vector load indices are not mapped"
            )
            return None
        view = _resolve_vector_view(buffer, context, diagnostic_owner=expr)
        if view is None:
            return None
        if not is_vector_type(result_type):
            lanes = _single_ramp_lanes(source_indices)
            if lanes is None:
                context.record_blocked(
                    node_text(expr), "vector load lanes are not mapped"
                )
                return None
            result_type = context.type_converter.vector_type(dtype(expr), lanes)
        fragment_result = _try_convert_fragment_vector_memory_load(
            expr,
            buffer,
            source_indices,
            vector_indices,
            result_type,
            context,
            converter,
        )
        if fragment_result is not None:
            context.map_value(expr, fragment_result, str(result_type))
            context.record_converted(
                node_text(expr), f"{context.ssa(fragment_result)} = local.fragment"
            )
            return fragment_result
        if _buffer_scope(buffer) in ("shared", "shared.dyn"):
            context.flush_pending_workgroup_memory_barrier()
        result = context.builder.vector.load(
            view=view,
            indices=vector_indices,
            results=[result_type],
            name=context.fresh_name("load"),
        )
        context.map_value(expr, result, str(result_type))
        context.record_converted(
            node_text(expr), f"{context.ssa(result)} = vector.load"
        )
        return result
    access = resolve_buffer_access(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=expr,
    )
    if access is None:
        return None
    if context.fragment_vector(access.view) is not None:
        fragment_result = _try_convert_fragment_vector_load(
            expr,
            access,
            result_type,
            context,
        )
        if fragment_result is None:
            return None
        context.map_value(expr, fragment_result, str(result_type))
        context.record_converted(
            node_text(expr), f"{context.ssa(fragment_result)} = vector.extract"
        )
        return fragment_result
    mapped_result = context.mapped_buffer_access(
        access.view,
        access.indices,
        access.memory_scope,
    )
    if mapped_result is not None:
        context.map_value(expr, mapped_result, str(mapped_result.type))
        context.map_buffer_access(
            expr,
            access.view,
            access.indices,
            access.memory_scope,
            mapped_result,
        )
        context.record_converted(
            node_text(expr), f"{context.ssa(mapped_result)} = cached buffer load"
        )
        if options.index_like:
            index_result = _coerce_index_like_load(mapped_result, expr, context)
            if index_result is None:
                return None
            return index_result
        return mapped_result
    if access.memory_scope in ("shared", "shared.dyn"):
        context.flush_pending_workgroup_memory_barrier()
    result = context.builder.view.load(
        view=access.view,
        indices=list(access.indices),
        results=[result_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(result_type))
    context.map_buffer_access(
        expr,
        access.view,
        access.indices,
        access.memory_scope,
        result,
    )
    context.record_converted(node_text(expr), f"{context.ssa(result)} = view.load")
    if options.index_like:
        index_result = _coerce_index_like_load(result, expr, context)
        if index_result is None:
            return None
        return index_result
    return result


def _try_convert_distributed_fragment_store(
    stmt: object,
    buffer: object,
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    value_expr = getattr(stmt, "value", None)
    if node_kind(value_expr) != "BufferLoad":
        return False
    target_axis = _direct_distributed_source_axis(source_indices, context)
    if target_axis is None or target_axis[1].lane_count <= 1:
        return False
    target_axis_position, target_lane = target_axis
    fragment_value = _fragment_vector_value_for_load(value_expr, context, converter)
    if fragment_value is None:
        return False
    if fragment_value.lane_count != target_lane.lane_count:
        context.record_blocked(
            node_text(stmt),
            "local.fragment vector lane count does not match store distribution",
        )
        return True
    if target_lane.lane != 0:
        context.record_converted(node_text(stmt), "distributed vector.store lane")
        return True

    access = resolve_buffer_access(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=stmt,
    )
    if access is None:
        return True
    if target_axis_position != len(access.indices) - 1:
        context.record_blocked(
            node_text(stmt),
            "distributed local.fragment store must write the trailing view axis",
        )
        return True
    indices: list[int | ValueRef] = list(access.indices)
    indices[target_axis_position] = target_lane.base
    context.builder.vector.store(
        value=fragment_value.value,
        view=access.view,
        indices=indices,
    )
    context.record_converted(node_text(stmt), "distributed vector.store")
    return True


def _try_convert_format_preserving_fragment_store(
    stmt: object,
    buffer: object,
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    value_expr = getattr(stmt, "value", None)
    source_element_type = _format_preserving_fp8_element_type(dtype(value_expr))
    if source_element_type is None:
        return False
    access = resolve_buffer_access(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=stmt,
    )
    if access is None:
        return True
    if context.fragment_vector(access.view) is None:
        return False
    value = _try_convert_format_preserving_fp8_load(
        value_expr,
        source_element_type,
        context,
        converter,
    )
    if value is None:
        return False
    fragment_value = _try_convert_fragment_vector_store(stmt, value, access, context)
    if fragment_value is None:
        return True
    context.map_fragment_vector(access.view, fragment_value)
    context.record_converted(
        node_text(stmt),
        f"{context.ssa(fragment_value.value)} = vector.insert<fp8-format>",
    )
    return True


def _try_convert_format_preserving_fp8_load(
    expr: object,
    element_type: ScalarType,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    if node_kind(expr) != "BufferLoad":
        return None
    access = resolve_buffer_access(
        getattr(expr, "buffer", None),
        tuple(getattr(expr, "indices", ())),
        context,
        converter,
        diagnostic_owner=expr,
    )
    if access is None:
        return None
    if access.memory_scope in ("shared", "shared.dyn"):
        context.flush_pending_workgroup_memory_barrier()
    result = context.builder.view.load(
        view=access.view,
        indices=list(access.indices),
        results=[element_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(element_type))
    context.map_buffer_access(
        expr,
        access.view,
        access.indices,
        access.memory_scope,
        result,
    )
    context.record_converted(node_text(expr), f"{context.ssa(result)} = view.load")
    return result


def _try_convert_fragment_vector_load(
    expr: object,
    access: TileLangBufferAccess,
    result_type: Type,
    context: TileLangConversionContext,
) -> ValueRef | None:
    fragment_value = context.fragment_vector(access.view)
    if fragment_value is None:
        return None
    if fragment_value.base is None:
        if len(access.indices) != 1:
            context.record_blocked(
                node_text(expr),
                "local.fragment vector access is not rank-1",
            )
            return None
        return context.builder.vector.extract(
            source=fragment_value.value,
            indices=[access.indices[0]],
            results=[result_type],
            name=context.fresh_name("load"),
        )
    lane = _single_distributed_index(access.indices, context)
    if lane is None or lane.lane < 0 or lane.lane_count != fragment_value.lane_count:
        context.record_blocked(
            node_text(expr),
            "local.fragment vector access is not aligned with a static lane",
        )
        return None
    return context.builder.vector.extract(
        source=fragment_value.value,
        indices=[lane.lane],
        results=[result_type],
        name=context.fresh_name("load"),
    )


def _try_convert_fragment_vector_store(
    stmt: object,
    value: ValueRef,
    access: TileLangBufferAccess,
    context: TileLangConversionContext,
) -> TileLangFragmentVector | None:
    fragment_value = context.fragment_vector(access.view)
    if fragment_value is None:
        return None
    coerced_value = _coerce_store_value(value, access.view, stmt, context)
    if coerced_value is None:
        return None
    if fragment_value.base is None:
        if len(access.indices) != 1:
            context.record_blocked(
                node_text(stmt),
                "local.fragment vector store is not rank-1",
            )
            return None
        result = context.builder.vector.insert(
            value=coerced_value,
            dest=fragment_value.value,
            indices=[access.indices[0]],
            results=[fragment_value.value.type],
            name=context.fresh_name("store"),
        )
        return TileLangFragmentVector(
            value=result,
            lane_count=fragment_value.lane_count,
            base=None,
        )

    lane = _single_distributed_index(access.indices, context)
    if lane is None or lane.lane < 0 or lane.lane_count != fragment_value.lane_count:
        context.record_blocked(
            node_text(stmt),
            "local.fragment vector store is not aligned with a static lane",
        )
        return None
    result = context.builder.vector.insert(
        value=coerced_value,
        dest=fragment_value.value,
        indices=[lane.lane],
        results=[fragment_value.value.type],
        name=context.fresh_name("store"),
    )
    return TileLangFragmentVector(
        value=result,
        lane_count=fragment_value.lane_count,
        base=fragment_value.base,
    )


def _fragment_vector_value_for_load(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangFragmentVector | None:
    access = resolve_buffer_access(
        getattr(expr, "buffer", None),
        tuple(getattr(expr, "indices", ())),
        context,
        converter,
        diagnostic_owner=expr,
    )
    if access is None:
        return None
    lane = _single_distributed_index(access.indices, context)
    if lane is None or lane.lane < 0:
        return None
    fragment_value = context.fragment_vector(access.view)
    if fragment_value is None:
        return None
    if fragment_value.base is None:
        return None
    if fragment_value.lane_count != lane.lane_count:
        return None
    return fragment_value


def _try_convert_fragment_vector_memory_load(
    expr: object,
    buffer: object,
    source_indices: tuple[object, ...],
    vector_indices: list[int | ValueRef],
    result_type: Type,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    if not is_vector_type(result_type):
        return None
    index_map = resolve_buffer_index_map(
        buffer,
        context,
        converter,
        diagnostic_owner=expr,
    )
    if index_map is None:
        return None
    fragment_value = context.fragment_vector(index_map.view)
    if fragment_value is None:
        return None
    if str(fragment_value.value.type) != str(result_type):
        return None
    if len(source_indices) != 1 or len(vector_indices) != 1:
        return None
    ramp = source_indices[0]
    if node_kind(ramp) != "Ramp":
        return None
    stride = integer_value(getattr(ramp, "stride", None))
    if stride != 1:
        return None
    if fragment_value.base is None:
        base = vector_indices[0]
        if not isinstance(base, ValueRef) or not _is_zero_index(base, context):
            return None
        return fragment_value.value
    base = vector_indices[0]
    if not isinstance(base, ValueRef) or base.id != fragment_value.base.id:
        return None
    return fragment_value.value


def _direct_distributed_source_axis(
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
) -> tuple[int, TileLangDistributedIndex] | None:
    for axis, source_index in enumerate(source_indices):
        mapped = context.mapped_index_value(source_index)
        if mapped is None:
            continue
        lane = context.distributed_index(mapped)
        if lane is not None:
            return axis, lane
    return None


def _single_distributed_index(
    indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> TileLangDistributedIndex | None:
    match = None
    for index in indices:
        lane = context.distributed_index(index)
        if lane is None:
            continue
        if match is not None:
            return None
        match = lane
    return match


def _resolve_vector_view(
    buffer: object,
    context: TileLangConversionContext,
    *,
    diagnostic_owner: object,
) -> ValueRef | None:
    view = context.mapped(buffer)
    if view is not None:
        return view
    data = getattr(buffer, "data", None)
    view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target is not mapped",
        )
        return None
    return view


def _coerce_index_like_load(
    value: ValueRef,
    source: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(value.type) in ("index", "offset"):
        context.map_index_value(source, value)
        return value
    if _integer_bit_width(str(value.type)) is None:
        context.record_blocked(
            node_text(source),
            f"buffer load value conversion {value.type} to index is not imported",
        )
        return None
    result = context.builder.index.cast(
        input=value,
        results=[INDEX],
        name=context.fresh_name("load_idx"),
    )
    context.map_index_value(source, result)
    context.record_converted(node_text(source), f"{context.ssa(result)} = index.cast")
    return result


def _coerce_store_value(
    value: ValueRef,
    view: ValueRef,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    view_type = _view_type(view, context)
    if not isinstance(view_type, ShapedType):
        context.record_blocked(node_text(owner), "buffer store target is not shaped")
        return None
    return _cast_store_value(value, view_type.element_type, owner, context)


def _cast_store_value(
    value: ValueRef,
    target_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    source_type = value.type
    if str(source_type) == str(target_type):
        return value
    source_text = str(source_type)
    target_text = str(target_type)
    if (
        source_text in ("index", "offset")
        and _integer_bit_width(target_text) is not None
    ):
        return context.builder.index.cast(
            input=value,
            results=[target_type],
            name=context.fresh_name("store_cast"),
        )
    source_width = _integer_bit_width(source_text)
    target_width = _integer_bit_width(target_text)
    if source_width is not None and target_width is not None:
        if source_width < target_width:
            return context.builder.scalar.extsi(
                input=value,
                results=[target_type],
                name=context.fresh_name("store_ext"),
            )
        if source_width > target_width:
            return context.builder.scalar.trunci(
                input=value,
                results=[target_type],
                name=context.fresh_name("store_trunc"),
            )
    context.record_blocked(
        node_text(owner),
        f"buffer store value conversion {source_type} to {target_type} is not imported",
    )
    return None


def _integer_bit_width(type_text: str) -> int | None:
    if not type_text.startswith("i") or not type_text[1:].isdecimal():
        return None
    return int(type_text[1:])


def _format_preserving_fp8_element_type(dtype_text: str) -> ScalarType | None:
    return _FORMAT_PRESERVING_FP8_DTYPES.get(dtype_text)


def _view_type(view: ValueRef, context: TileLangConversionContext) -> object:
    return context.builder.module.values[view.id].type


def is_vector_type(value_type: object) -> bool:
    return (
        isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VECTOR
    )


def _has_ramp_index(expr: object) -> bool:
    return any(
        node_kind(index) == "Ramp" for index in tuple(getattr(expr, "indices", ()))
    )


def _vector_memory_indices(
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> list[int | ValueRef] | None:
    mapped_indices: list[int | ValueRef] = []
    ramp_count = 0
    for source_index in source_indices:
        if node_kind(source_index) == "Ramp":
            ramp_count += 1
            if ramp_count > 1:
                return None
            stride = integer_value(getattr(source_index, "stride", None))
            if stride != 1:
                return None
            base = converter.convert_expr(
                getattr(source_index, "base", None),
                context,
                index_like=True,
            )
            if base is None:
                return None
            mapped_indices.append(base)
            continue
        mapped = converter.convert_expr(source_index, context, index_like=True)
        if mapped is None:
            return None
        mapped_indices.append(mapped)
    return mapped_indices


def _single_ramp_lanes(source_indices: tuple[object, ...]) -> int | None:
    lanes: int | None = None
    for source_index in source_indices:
        if node_kind(source_index) != "Ramp":
            continue
        if lanes is not None:
            return None
        lanes = vector_lanes(source_index)
    return lanes


def _is_zero_index(value: ValueRef, context: TileLangConversionContext) -> bool:
    zero = context.constants.get(("index", "0"))
    return zero is not None and value.id == zero.id


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""


def _buffer_memory_space(buffer: object) -> str | None:
    scope = _buffer_scope(buffer)
    if scope in ("", "local", "local.dyn", "local.fragment", "local.var"):
        return "private"
    if scope in ("shared", "shared.dyn"):
        return "workgroup"
    return None


_FORMAT_PRESERVING_FP8_DTYPES = {
    "float8_e4m3fn": ScalarType(ScalarTypeKind.F8E4M3),
    "float8_e4m3fnuz": ScalarType(ScalarTypeKind.F8E4M3),
    "float8_e5m2fnuz": ScalarType(ScalarTypeKind.F8E5M2),
}
