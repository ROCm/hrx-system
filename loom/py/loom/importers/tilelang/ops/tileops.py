# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang tileop call converters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier, target_preset_amdgpu_kind
from loom.importers.tilelang.buffers import (
    TileLangBufferAccess,
    TileLangBufferRegion,
    add_index,
    map_region_indices,
    resolve_buffer_access,
    resolve_buffer_region,
)
from loom.importers.tilelang.context import (
    TileLangConversionContext,
    TileLangDistributedIndex,
    TileLangFragmentVector,
    TileLangMatrixFragment,
)
from loom.importers.tilelang.converter import ExpressionOptions, TileLangConverter
from loom.importers.tilelang.coverage import coverage_row
from loom.importers.tilelang.nodes import dtype, mapping_items, node_kind, node_text
from loom.importers.tilelang.ops.distribution import (
    MAX_DISTRIBUTED_UNROLL_LANES,
    emit_distributed_1d,
    materialize_distributed_1d_plan,
    static_index_value,
)
from loom.importers.tilelang.ops.topology import (
    THREAD_AXES,
    integer_value,
    map_thread_axis,
)
from loom.ir import (
    I32,
    INDEX,
    EncodingInstance,
    ScalarType,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
)


@dataclass(frozen=True, slots=True)
class TileReduceSpec:
    """Decoded TileLang reduction semantics over one element stream."""

    source_kind: str
    combiner: str
    absolute: bool = False


@dataclass(frozen=True, slots=True)
class _FragmentVectorLoopState:
    """A fragment vector carried through synthetic tile-operation loops."""

    view: ValueRef
    initial: TileLangFragmentVector
    name: str


@dataclass(frozen=True, slots=True)
class _DenseGemmSpec:
    """Decoded dense TileLang GEMM slice supported by the first bridge."""

    lhs: TileLangBufferRegion
    rhs: TileLangBufferRegion
    accumulator: TileLangBufferRegion
    lhs_type: ScalarType
    rhs_type: ScalarType
    accumulator_type: ScalarType
    clear_accumulator: bool


@dataclass(frozen=True, slots=True)
class _CdnaFp8GemmSpec:
    """Decoded CDNA FP8 TileLang GEMM tile for the staged MFMA bridge."""

    lhs: TileLangBufferRegion
    rhs: TileLangBufferRegion
    accumulator: TileLangBufferRegion
    lhs_type: ScalarType
    rhs_type: ScalarType
    accumulator_type: ScalarType
    lhs_format: str
    rhs_format: str


class _BlockedCdnaFp8Gemm:
    pass


_CDNA_FP8_TARGET_KINDS = frozenset(("gfx940", "gfx941", "gfx942"))
_FP8_ELEMENT_TYPES = frozenset(("f8E4M3", "f8E5M2"))
_FP8_SOURCE_FORMATS = {
    "float8_e4m3": "f8e4m3",
    "float8_e4m3fn": "f8e4m3fn",
    "float8_e4m3fnuz": "f8e4m3fnuz",
    "float8_e5m2": "f8e5m2",
    "float8_e5m2fnuz": "f8e5m2fnuz",
}


def is_tileop_call(op_name: str) -> bool:
    """Returns true for calls owned by TileLang's tileop namespace."""

    return op_name.startswith("tl.tileop.")


def convert_tileop_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    """Import a TileLang tileop call."""

    if op_name in _REGION_CALLS:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` must be consumed by a tile operation",
        )
        return None
    if op_name in _COPY_CALLS:
        return _convert_copy_call(expr, context, converter, options, op_name)
    if op_name in _FILL_CALLS:
        return _convert_fill_call(expr, context, converter, options, op_name)
    if op_name in _REDUCE_CALLS:
        return _convert_reduce_call(expr, context, converter, options, op_name)
    if op_name in _CUMSUM_CALLS:
        return _convert_cumsum_call(expr, context, converter, options, op_name)
    if op_name in _FINALIZE_REDUCER_CALLS:
        return _convert_finalize_reducer_call(
            expr, context, converter, options, op_name
        )
    if op_name in _GEMM_CALLS:
        return _convert_gemm_call(expr, context, converter, options, op_name)
    _record_unsupported_tileop(expr, context, op_name)
    return None


def _convert_copy_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 2 regions")
        return None
    if not _validate_copy_annotations(expr, context):
        return None
    source = _decode_region(args[0], expr, context, converter, expected_access=1)
    target = _decode_region(args[1], expr, context, converter, expected_access=2)
    if source is None or target is None:
        return None
    loop_extents = _copy_loop_extents(source, target, context)
    if loop_extents is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination extents differ",
        )
        return None
    source_element_type = _view_element_type(source.view, context)
    if source_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source region is not a shaped view",
        )
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination region is not a shaped view",
        )
        return None
    distribution = _copy_uses_distributed_index_space(
        source,
        target,
        loop_extents,
        context,
    )
    if _region_is_shared(source) and context.flush_pending_workgroup_memory_barrier():
        context.record_converted(
            node_text(expr), "kernel.barrier<workgroup> before shared copy"
        )
    if _try_emit_matrix_fragment_copy(
        expr,
        context,
        source=source,
        target=target,
        loop_extents=loop_extents,
        source_element_type=source_element_type,
        target_element_type=target_element_type,
    ):
        context.record_converted(node_text(expr), "tl.tileop.copy matrix fragment")
        return None
    if _try_emit_fragment_vector_copy(
        expr,
        context,
        converter,
        source=source,
        target=target,
        loop_extents=loop_extents,
        source_element_type=source_element_type,
        target_element_type=target_element_type,
    ):
        context.record_converted(node_text(expr), "tl.tileop.copy vector")
        return None
    if _try_emit_distributed_vector_copy(
        expr,
        context,
        converter,
        source=source,
        target=target,
        loop_extents=loop_extents,
        source_element_type=source_element_type,
        target_element_type=target_element_type,
    ):
        context.record_converted(node_text(expr), "tl.tileop.copy vector")
        return None
    if _try_emit_distributed_vector_load_copy(
        expr,
        context,
        converter,
        source=source,
        target=target,
        loop_extents=loop_extents,
        source_element_type=source_element_type,
        target_element_type=target_element_type,
    ):
        context.record_converted(node_text(expr), "tl.tileop.copy vector-load")
        return None

    def emit_copy(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        source_indices = _project_region_indices(source, indices, expr, body_context)
        target_indices = _project_region_indices(target, indices, expr, body_context)
        if source_indices is None or target_indices is None:
            return
        value = body_context.mapped_buffer_access(
            source.view,
            source_indices,
            source.index_map.memory_scope,
        )
        if value is None and source.index_map.memory_scope == "local.fragment":
            value = _load_tracked_local_value(
                source.view,
                source_indices,
                source.index_map.memory_scope,
                source_element_type,
                body_context,
                name="copy",
            )
        if value is None:
            value = body_context.builder.view.load(
                view=source.view,
                indices=list(source_indices),
                results=[source_element_type],
                name=body_context.fresh_name("copy"),
            )
        converted_value = _copy_value_cast(
            value,
            source_element_type,
            target_element_type,
            expr,
            body_context,
        )
        if converted_value is None:
            return
        _store_tracked_local_value(
            expr,
            target.view,
            target_indices,
            target.index_map.memory_scope,
            converted_value,
            body_context,
        )

    if not _emit_region_loops(
        loop_extents,
        context,
        emit_copy,
        distribution=distribution,
        converter=converter,
        owner=expr,
        fragment_vector_state=_fragment_vector_loop_state_views(context, target),
    ):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` region is not mapped"
        )
        return None
    if distribution and _region_is_shared(target):
        context.mark_pending_workgroup_memory_write()
    context.record_converted(node_text(expr), "tl.tileop.copy")
    return None


def _try_emit_distributed_vector_copy(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    loop_extents: tuple[ValueRef, ...],
    source_element_type: Type,
    target_element_type: Type,
) -> bool:
    if _region_is_fragment(source) or _region_is_fragment(target):
        return False
    if str(source_element_type) != str(target_element_type):
        return False
    if not source.index_map.is_identity or not target.index_map.is_identity:
        return False
    if len(source.extents) != len(loop_extents) or len(target.extents) != len(
        loop_extents
    ):
        return False
    if not _region_has_dense_layout(source, context) or not _region_has_dense_layout(
        target, context
    ):
        return False
    static_extents = _static_extent_values(loop_extents, context)
    if static_extents is None or not static_extents:
        return False
    thread_count = context.static_topology_extent("threadIdx.x")
    if thread_count is None or thread_count <= 1:
        return False
    element_count = _product(static_extents)
    lane_count = (element_count + thread_count - 1) // thread_count
    if lane_count <= 1 or lane_count > MAX_DISTRIBUTED_UNROLL_LANES:
        return False
    if lane_count * thread_count != element_count:
        return False
    trailing_extent = static_extents[-1]
    if trailing_extent % lane_count != 0:
        return False

    thread_index = map_thread_axis(
        axis=THREAD_AXES["threadIdx.x"],
        sources=(),
        context=context,
        converter=converter,
        name="tx",
    )
    if thread_index is None:
        return False
    loop_indices = _unflatten_vector_chunk_indices(
        thread_index,
        static_extents,
        lane_count,
        context,
    )
    source_indices = _project_region_indices(source, loop_indices, expr, context)
    target_indices = _project_region_indices(target, loop_indices, expr, context)
    if source_indices is None or target_indices is None:
        return False
    result_type = context.type_converter.vector_type(source_element_type, lane_count)
    vector = context.builder.vector.load(
        view=source.view,
        indices=list(source_indices),
        results=[result_type],
        name=context.fresh_name("copy"),
    )
    context.builder.vector.store(
        value=vector,
        view=target.view,
        indices=list(target_indices),
    )
    if _region_is_shared(target):
        context.mark_pending_workgroup_memory_write()
    return True


def _try_emit_distributed_vector_load_copy(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    loop_extents: tuple[ValueRef, ...],
    source_element_type: Type,
    target_element_type: Type,
) -> bool:
    if _region_is_fragment(source) or _region_is_fragment(target):
        return False
    if str(source_element_type) != str(target_element_type):
        return False
    if not source.index_map.is_identity or not target.index_map.is_identity:
        return False
    if _projected_region_rank(source, context) != len(
        loop_extents
    ) or _projected_region_rank(target, context) != len(loop_extents):
        return False
    if not _region_has_dense_layout(source, context):
        return False
    static_extents = _static_extent_values(loop_extents, context)
    if static_extents is None or not static_extents:
        return False
    thread_count = context.static_topology_extent("threadIdx.x")
    if thread_count is None or thread_count <= 1:
        return False
    element_count = _product(static_extents)
    lane_count = (element_count + thread_count - 1) // thread_count
    if lane_count <= 1 or lane_count > MAX_DISTRIBUTED_UNROLL_LANES:
        return False
    if lane_count * thread_count != element_count:
        return False
    trailing_extent = static_extents[-1]
    if trailing_extent % lane_count != 0:
        return False

    thread_index = map_thread_axis(
        axis=THREAD_AXES["threadIdx.x"],
        sources=(),
        context=context,
        converter=converter,
        name="tx",
    )
    if thread_index is None:
        return False
    loop_indices = _unflatten_vector_chunk_indices(
        thread_index,
        static_extents,
        lane_count,
        context,
    )
    source_indices = _project_region_indices(source, loop_indices, expr, context)
    if source_indices is None:
        return False
    result_type = context.type_converter.vector_type(source_element_type, lane_count)
    vector = context.builder.vector.load(
        view=source.view,
        indices=list(source_indices),
        results=[result_type],
        name=context.fresh_name("copy"),
    )

    for lane in range(lane_count):
        lane_indices = _offset_trailing_lane(loop_indices, lane, context)
        target_indices = cast(
            tuple[ValueRef, ...],
            _project_region_indices(target, lane_indices, expr, context),
        )
        value = context.builder.vector.extract(
            source=vector,
            indices=[lane],
            results=[source_element_type],
            name=context.fresh_name("copy"),
        )
        _store_tracked_local_value(
            expr,
            target.view,
            target_indices,
            target.index_map.memory_scope,
            value,
            context,
        )
    return True


def _try_emit_fragment_vector_copy(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    loop_extents: tuple[ValueRef, ...],
    source_element_type: Type,
    target_element_type: Type,
) -> bool:
    if not _region_is_fragment(target) or _region_is_fragment(source):
        return False
    if len(loop_extents) != 1:
        return False
    extent_integer = static_index_value(loop_extents[0], context)
    if extent_integer is None:
        return False
    if context.static_topology_extent("threadIdx.x") == 1:
        return _try_emit_full_fragment_vector_copy(
            expr,
            context,
            source=source,
            target=target,
            extent_integer=extent_integer,
            source_element_type=source_element_type,
            target_element_type=target_element_type,
        )
    plan = materialize_distributed_1d_plan(
        expr,
        context,
        converter,
        extent_integer=extent_integer,
        index_name="i0",
    )
    if plan is None:
        return False
    if plan.lane_count <= 1:
        return _try_emit_full_fragment_vector_copy(
            expr,
            context,
            source=source,
            target=target,
            extent_integer=extent_integer,
            source_element_type=source_element_type,
            target_element_type=target_element_type,
        )
    if str(source_element_type) != str(target_element_type):
        return False
    source_indices = _project_region_indices(source, (plan.base,), expr, context)
    target_indices = _project_region_indices(target, (plan.base,), expr, context)
    if source_indices is None or target_indices is None:
        return False
    result_type = context.type_converter.vector_type(
        source_element_type,
        plan.lane_count,
    )
    vector = context.builder.vector.load(
        view=source.view,
        indices=list(source_indices),
        results=[result_type],
        name=context.fresh_name("copy"),
    )
    context.clear_matrix_fragment(target.view)
    context.map_fragment_vector(
        target.view,
        TileLangFragmentVector(
            value=vector,
            lane_count=plan.lane_count,
            base=None if plan.thread_count == 1 else plan.base,
        ),
    )
    return True


def _try_emit_matrix_fragment_copy(
    expr: object,
    context: TileLangConversionContext,
    *,
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    loop_extents: tuple[ValueRef, ...],
    source_element_type: Type,
    target_element_type: Type,
) -> bool:
    if source.index_map.memory_scope != "local.fragment":
        return False
    if target.index_map.memory_scope == "local.fragment":
        return False
    fragment = context.matrix_fragment(source.view)
    if fragment is None:
        return False
    if source_element_type != target_element_type:
        return False
    if fragment.element_type != target_element_type:
        return False
    if len(loop_extents) != 2 or not _all_zero_indices(source.indices, context):
        return False
    if not _same_value_sequence(loop_extents, (fragment.rows, fragment.columns)):
        return False
    zero = context.ensure_constant("0", "index", "c0")
    target_indices = _region_indices(target, (zero, zero), context)
    context.builder.vector.fragment_store(
        role="result",
        value=fragment.value,
        view=target.view,
        indices=list(target_indices),
        rows=fragment.rows,
        columns=fragment.columns,
    )
    return True


def _try_emit_full_fragment_vector_copy(
    expr: object,
    context: TileLangConversionContext,
    *,
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    extent_integer: int,
    source_element_type: Type,
    target_element_type: Type,
) -> bool:
    if str(source_element_type) != str(target_element_type):
        return False
    zero = context.ensure_constant("0", "index", "c0")
    source_indices = _project_region_indices(source, (zero,), expr, context)
    target_indices = _project_region_indices(target, (zero,), expr, context)
    if source_indices is None or target_indices is None:
        return False
    if not _all_zero_indices(target_indices, context):
        return False
    result_type = context.type_converter.vector_type(
        source_element_type,
        extent_integer,
    )
    vector = context.builder.vector.load(
        view=source.view,
        indices=list(source_indices),
        results=[result_type],
        name=context.fresh_name("copy"),
    )
    context.clear_matrix_fragment(target.view)
    context.map_fragment_vector(
        target.view,
        TileLangFragmentVector(value=vector, lane_count=extent_integer),
    )
    return True


def _validate_copy_annotations(
    expr: object,
    context: TileLangConversionContext,
) -> bool:
    annotations = _call_annotations(expr)
    unknown_annotations = sorted(set(annotations) - _COPY_ANNOTATIONS)
    if unknown_annotations:
        context.record_blocked(
            node_text(expr),
            (
                "tl.tileop.copy annotations are not imported: "
                + ", ".join(unknown_annotations)
            ),
        )
        return False
    if "disable_tma" in annotations:
        disable_tma = _static_bool(annotations["disable_tma"])
        if disable_tma is None:
            context.record_blocked(
                node_text(expr),
                "tl.tileop.copy annotation `disable_tma` is not static",
            )
            return False
    for annotation in sorted(set(annotations) - {"disable_tma"}):
        context.record_blocked(
            node_text(expr),
            f"copy annotation `{annotation}` needs scheduling import",
        )
        return False
    return True


def _convert_fill_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects region and value",
        )
        return None
    target = _decode_region(args[0], expr, context, converter, expected_access=2)
    if target is None:
        return None
    element_type = _view_element_type(target.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` target region is not a shaped view",
        )
        return None
    value = _convert_fill_value(args[1], element_type, context, converter)
    if value is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` fill value is not mapped",
        )
        return None
    if _try_emit_matrix_fragment_fill(
        expr,
        context,
        target=target,
        value=value,
        fill_source=args[1],
        element_type=element_type,
    ):
        context.record_converted(
            node_text(expr), "tl.tileop.fill<local.fragment.matrix>"
        )
        return None
    if _try_emit_fragment_vector_fill(
        expr,
        context,
        target=target,
        value=value,
        element_type=element_type,
    ):
        context.record_converted(node_text(expr), "tl.tileop.fill<local.fragment>")
        return None

    def emit_fill(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        body_context.builder.view.store(
            value=value,
            view=target.view,
            indices=list(_region_indices(target, indices, body_context)),
        )

    if not _emit_region_loops(
        target.extents,
        context,
        emit_fill,
        distribution=_region_is_fragment(target),
        converter=converter,
        owner=expr,
    ):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` region is not mapped"
        )
        return None
    context.record_converted(node_text(expr), "tl.tileop.fill")
    return None


def _try_emit_fragment_vector_fill(
    expr: object,
    context: TileLangConversionContext,
    *,
    target: TileLangBufferRegion,
    value: ValueRef,
    element_type: ScalarType,
) -> bool:
    if target.index_map.memory_scope != "local.fragment":
        return False
    if len(target.extents) != 1 or not _all_zero_indices(target.indices, context):
        return False
    extent_integer = static_index_value(target.extents[0], context)
    if extent_integer is None:
        return False
    result_type = context.type_converter.vector_type(element_type, extent_integer)
    vector = context.builder.vector.splat(
        scalar=value,
        results=[result_type],
        name=context.fresh_name("fill"),
    )
    context.clear_matrix_fragment(target.view)
    context.map_fragment_vector(
        target.view,
        TileLangFragmentVector(value=vector, lane_count=extent_integer),
    )
    return True


def _try_emit_matrix_fragment_fill(
    expr: object,
    context: TileLangConversionContext,
    *,
    target: TileLangBufferRegion,
    value: ValueRef,
    fill_source: object,
    element_type: ScalarType,
) -> bool:
    if target.index_map.memory_scope != "local.fragment":
        return False
    if len(target.extents) != 2 or not _all_zero_indices(target.indices, context):
        return False
    if not _is_zero_fill_source(fill_source):
        return False
    if _static_extent_tuple(target.extents, context) != (16, 16):
        return False
    if str(element_type) != "f32":
        return False
    payload_type = context.type_converter.vector_type(element_type, 8)
    payload = context.builder.vector.splat(
        scalar=value,
        results=[payload_type],
        name=context.fresh_name("fill"),
    )
    fragment = context.builder.vector.fragment(
        role="init",
        data=payload,
        rows=target.extents[0],
        columns=target.extents[1],
        results=[payload_type],
        name=context.fresh_name("init"),
    )
    context.clear_fragment_vector(target.view)
    context.map_matrix_fragment(
        target.view,
        TileLangMatrixFragment(
            value=fragment,
            rows=target.extents[0],
            columns=target.extents[1],
            element_type=element_type,
        ),
    )
    return True


def _convert_reduce_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 5:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects source, destination, kind, dim, and clear",
        )
        return None
    dim = integer_value(args[3])
    if dim is None:
        context.record_blocked(node_text(expr), f"call `{op_name}` dim is not static")
        return None
    clear = integer_value(args[4])
    if clear is None:
        context.record_blocked(node_text(expr), f"call `{op_name}` clear is not static")
        return None
    source_axis = _single_ramp_axis(args[0])
    if source_axis is None:
        return _convert_region_reduce_call(
            expr,
            context,
            converter,
            op_name,
            dim=dim,
            clear=bool(clear),
            kind_expr=args[2],
        )
    if source_axis != dim:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` dim {dim} does not match source ramp axis {source_axis}",
        )
        return None
    source = converter.convert_expr(args[0], context)
    if source is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source vector is not mapped",
        )
        return None
    if not _is_rank_one_vector_type(source.type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source must map to a rank-1 vector",
        )
        return None
    source_type = cast(ShapedType, source.type)
    source_element_type = source_type.element_type
    target = _decode_buffer_access(args[1], expr, context, converter)
    if target is None:
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination is not a shaped view",
        )
        return None
    element_type = target_element_type
    source = _reduce_vector_value_cast(
        source,
        source_type,
        source_element_type,
        element_type,
        expr,
        context,
    )
    if source is None:
        return None
    reduce_spec = _reduce_spec(args[2], args[0], element_type, expr, context)
    if reduce_spec is None:
        return None
    init = _reduce_init(
        bool(clear),
        reduce_spec,
        element_type,
        target,
        expr,
        context,
    )
    if init is None:
        return None
    source = _build_vector_reduce_input(source, reduce_spec, element_type, context)
    reduce_name = context.fresh_name("reduce")
    reduce_fastmath = _additive_reduce_fastmath_flags(
        reduce_spec,
        element_type,
        context,
    )
    if reduce_fastmath is None:
        result = context.builder.vector.reduce(
            kind=reduce_spec.combiner,
            input=source,
            init=init,
            results=[element_type],
            name=reduce_name,
        )
    else:
        result = context.builder.vector.reduce(
            kind=reduce_spec.combiner,
            fastmath=reduce_fastmath,
            input=source,
            init=init,
            results=[element_type],
            name=reduce_name,
        )
    _store_tracked_local_value(
        expr,
        target.view,
        target.indices,
        target.memory_scope,
        result,
        context,
    )
    context.record_converted(
        node_text(expr),
        f"tl.tileop.reduce<{reduce_spec.source_kind}>",
    )
    return None


def _convert_region_reduce_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    dim: int,
    clear: bool,
    kind_expr: object,
) -> ValueRef | None:
    source = _decode_region_arg(
        args=_args(expr), position=0, expr=expr, context=context, converter=converter
    )
    target = _decode_region_arg(
        args=_args(expr), position=1, expr=expr, context=context, converter=converter
    )
    if source is None or target is None:
        return None
    if dim < 0 or dim >= len(source.extents):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` dim {dim} is outside source rank {len(source.extents)}",
        )
        return None
    element_type = _view_element_type(source.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source region is not a shaped view",
        )
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination region is not a shaped view",
        )
        return None
    source_element_type = element_type
    element_type = target_element_type
    if not _reduce_value_cast_is_supported(source_element_type, element_type):
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` source and destination element conversion "
                f"{source_element_type} to {element_type} is not imported"
            ),
        )
        return None
    reduce_spec = _reduce_spec(kind_expr, _args(expr)[0], element_type, expr, context)
    if reduce_spec is None:
        return None
    output_extents = tuple(
        extent for axis, extent in enumerate(source.extents) if axis != dim
    )

    def emit_reduce(
        output_indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        target_indices = _reduction_target_indices(
            target,
            len(source.extents),
            dim,
            output_indices,
            body_context,
        )
        if target_indices is None:
            body_context.record_blocked(
                node_text(expr),
                f"call `{op_name}` destination rank does not match reduced shape",
            )
            return
        init = _reduce_init(
            clear,
            reduce_spec,
            element_type,
            TileLangBufferAccess(
                target.view,
                target_indices,
                target.index_map.memory_scope,
            ),
            expr,
            body_context,
        )
        if init is None:
            return
        result = _try_emit_fragment_vector_reduce(
            expr,
            source,
            dim,
            output_indices,
            init,
            reduce_spec,
            source_element_type,
            element_type,
            body_context,
        )
        if result is None:
            result = _emit_scalar_reduce_loop(
                source,
                dim,
                output_indices,
                init,
                reduce_spec,
                source_element_type,
                element_type,
                body_context,
            )
        if result is None:
            return
        _store_tracked_local_value(
            expr,
            target.view,
            target_indices,
            target.index_map.memory_scope,
            result,
            body_context,
        )

    if not _emit_region_loops(output_extents, context, emit_reduce):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` output region is not mapped"
        )
        return None
    context.record_converted(
        node_text(expr),
        f"tl.tileop.reduce<{reduce_spec.source_kind}>",
    )
    return None


def _convert_cumsum_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 4:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects source, destination, dim, and reverse",
        )
        return None
    source = _decode_region_arg(
        args=args,
        position=0,
        expr=expr,
        context=context,
        converter=converter,
    )
    target = _decode_region_arg(
        args=args,
        position=1,
        expr=expr,
        context=context,
        converter=converter,
    )
    if source is None or target is None:
        return None
    dim = integer_value(args[2])
    if dim is None:
        context.record_blocked(node_text(expr), f"call `{op_name}` dim is not static")
        return None
    reverse = _bool_value(args[3])
    if reverse is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` reverse flag is not static"
        )
        return None
    if len(source.extents) != 1 or len(target.extents) != 1 or dim != 0:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` first slice supports rank-1 dim-0 regions",
        )
        return None
    if not _region_is_shared(source) or not _region_is_shared(target):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` first slice requires shared-memory regions",
        )
        return None
    loop_extents = _copy_loop_extents(source, target, context)
    if loop_extents is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination extents differ",
        )
        return None
    static_extents = _static_extent_values(loop_extents, context)
    if static_extents is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` first slice requires a static region extent",
        )
        return None
    thread_count = context.static_topology_extent("threadIdx.x")
    if thread_count is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` first slice requires a static threadIdx.x extent",
        )
        return None
    if static_extents != (thread_count,):
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` rank-1 extent {static_extents[0]} must match "
                f"threadIdx.x extent {thread_count}"
            ),
        )
        return None
    for axis in ("threadIdx.y", "threadIdx.z"):
        axis_count = context.static_topology_extent(axis)
        if axis_count != 1:
            context.record_blocked(
                node_text(expr),
                f"call `{op_name}` first slice requires {axis} extent 1",
            )
            return None
    source_element_type = _view_element_type(source.view, context)
    if source_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source region is not a shaped view",
        )
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination region is not a shaped view",
        )
        return None
    if str(source_element_type) != str(target_element_type):
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` source and destination element types differ: "
                f"{source_element_type} vs {target_element_type}"
            ),
        )
        return None
    combiner = _cumsum_combiner(source_element_type)
    if combiner is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` is not imported for {source_element_type}",
        )
        return None
    thread_index = map_thread_axis(
        axis=THREAD_AXES["threadIdx.x"],
        sources=(),
        context=context,
        converter=converter,
        name="tx",
    )
    if thread_index is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` threadIdx.x value is not mapped",
        )
        return None
    source_indices = _region_indices(source, (thread_index,), context)
    target_indices = _region_indices(target, (thread_index,), context)
    value = _load_tracked_local_value(
        view=source.view,
        indices=source_indices,
        memory_scope=source.index_map.memory_scope,
        result_type=source_element_type,
        context=context,
        name="cumsum_value",
    )
    if value is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source value is not mapped",
        )
        return None
    result = context.builder.kernel.workgroup_scan(
        kind=combiner,
        value=value,
        mode="inclusive",
        direction="reverse" if reverse else "forward",
        results=[source_element_type],
        name=context.fresh_name("cumsum"),
    )
    _store_tracked_local_value(
        expr,
        target.view,
        target_indices,
        target.index_map.memory_scope,
        result,
        context,
    )
    context.record_converted(
        node_text(expr),
        f"tl.tileop.cumsum<{'reverse' if reverse else 'forward'}>",
    )
    return None


def _convert_finalize_reducer_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) not in (1, 2):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects reducer region and optional op code",
        )
        return None
    data_source = _reducer_data_source(args[0], expr, context)
    if data_source is None:
        return None
    reducer_info = context.reducer_info(data_source)
    if reducer_info is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.finalize_reducer reducer_info metadata is missing",
        )
        return None
    if len(args) == 2:
        op_code = integer_value(args[1])
        expected_op_code = _REDUCER_OPERATION_CODES.get(reducer_info.operation)
        if op_code is None or op_code != expected_op_code:
            context.record_blocked(
                node_text(expr),
                "tl.tileop.finalize_reducer op code does not match reducer_info",
            )
            return None
    if reducer_info.replication == "none":
        context.record_converted(
            node_text(expr),
            f"tl.tileop.finalize_reducer<{reducer_info.operation}, none> normalized",
        )
        return None
    if reducer_info.replication == "all":
        return _convert_finalize_reducer_all(
            expr,
            context,
            converter,
            op_name,
            args=args,
            operation=reducer_info.operation,
            data_source=data_source,
        )
    context.record_blocked(
        node_text(expr),
        (
            "tl.tileop.finalize_reducer replication "
            f"`{reducer_info.replication}` is not imported"
        ),
    )
    return None


def _convert_finalize_reducer_all(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    args: tuple[object, ...],
    operation: str,
    data_source: object,
) -> ValueRef | None:
    target = _decode_region_arg(
        args=args,
        position=0,
        expr=expr,
        context=context,
        converter=converter,
    )
    if target is None:
        return None
    element_type = _view_element_type(target.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` reducer region is not a shaped view",
        )
        return None
    source_dtype = _reducer_source_dtype(data_source, context)
    reduce_spec = _reducer_reduce_spec(operation, source_dtype, element_type)
    if reduce_spec is None:
        context.record_blocked(
            node_text(expr),
            (
                f"tl.tileop.finalize_reducer operation `{operation}` "
                f"is not imported for {element_type}"
            ),
        )
        return None

    def emit_reduce(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        target_indices = _region_indices(target, indices, body_context)
        value = _load_tracked_local_value(
            view=target.view,
            indices=target_indices,
            memory_scope=target.index_map.memory_scope,
            result_type=element_type,
            context=body_context,
            name="reducer_value",
        )
        if value is None:
            body_context.record_blocked(
                node_text(expr),
                f"call `{op_name}` reducer value is not mapped",
            )
            return
        result = body_context.builder.kernel.workgroup_reduce(
            kind=reduce_spec.combiner,
            value=value,
            results=[element_type],
            name=body_context.fresh_name("reducer_all"),
        )
        _store_tracked_local_value(
            expr,
            target.view,
            target_indices,
            target.index_map.memory_scope,
            result,
            body_context,
        )

    if not _emit_region_loops(
        target.extents,
        context,
        emit_reduce,
        fragment_vector_state=_fragment_vector_loop_state_views(context, target),
    ):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` reducer region is not mapped"
        )
        return None
    context.record_converted(
        node_text(expr),
        f"tl.tileop.finalize_reducer<{operation}, all>",
    )
    return None


def _convert_gemm_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) < 19:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects at least 19 descriptor operands",
        )
        return None
    if _convert_cdna_fp8_gemm_call(expr, context, converter, args):
        return None
    spec = _decode_dense_gemm_spec(expr, context, converter, args)
    if spec is None:
        return None
    init = _gemm_init_fragment(expr, context, spec)
    if init is None:
        return None
    if context.flush_pending_workgroup_memory_barrier():
        context.record_converted(
            node_text(expr), "kernel.barrier<workgroup> before shared gemm"
        )
    lhs_payload_type = context.type_converter.vector_type(spec.lhs_type, 16)
    accumulator_payload_type = context.type_converter.vector_type(
        spec.accumulator_type, 8
    )
    lhs = context.builder.vector.fragment_load(
        role="lhs",
        view=spec.lhs.view,
        indices=list(spec.lhs.indices),
        rows=spec.lhs.extents[0],
        columns=spec.lhs.extents[1],
        results=[lhs_payload_type],
        name=context.fresh_name("lhs"),
    )
    rhs = context.builder.vector.fragment_load(
        role="rhs",
        view=spec.rhs.view,
        indices=list(spec.rhs.indices),
        rows=spec.rhs.extents[0],
        columns=spec.rhs.extents[1],
        results=[lhs_payload_type],
        name=context.fresh_name("rhs"),
    )
    result = context.builder.vector.mma(
        lhs=lhs,
        rhs=rhs,
        init=init,
        results=[accumulator_payload_type],
        name=context.fresh_name("gemm"),
    )
    context.clear_fragment_vector(spec.accumulator.view)
    context.map_matrix_fragment(
        spec.accumulator.view,
        TileLangMatrixFragment(
            value=result,
            rows=spec.accumulator.extents[0],
            columns=spec.accumulator.extents[1],
            element_type=spec.accumulator_type,
        ),
    )
    context.record_converted(node_text(expr), "tl.tileop.gemm dense f16/f32")
    return None


def _decode_dense_gemm_spec(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    args: tuple[object, ...],
) -> _DenseGemmSpec | None:
    annotations = _call_annotations(expr)
    if annotations:
        context.record_blocked(
            node_text(expr),
            (
                "tl.tileop.gemm vector.fragment bridge does not import "
                "pipeline or scheduling annotations"
            ),
        )
        return None
    if target_preset_amdgpu_kind(context.target_preset) != "gfx1100":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires AMDGPU gfx1100",
        )
        return None
    transpose_lhs = _static_bool(args[3])
    transpose_rhs = _static_bool(args[4])
    if transpose_lhs or transpose_rhs:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires non-transposed inputs",
        )
        return None
    if transpose_lhs is None or transpose_rhs is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm transpose flags must be static",
        )
        return None
    shape = tuple(integer_value(arg) for arg in args[5:8])
    if shape != (16, 16, 16):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires m/n/k = 16/16/16",
        )
        return None
    if integer_value(args[8]) != 0:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires default warp policy",
        )
        return None
    clear_accumulator = _static_bool(args[9])
    if clear_accumulator is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm clear_accum flag must be static",
        )
        return None
    if integer_value(args[14]) != 1:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires k_pack = 1",
        )
        return None
    if integer_value(args[15]) != 0:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge does not import async waits",
        )
        return None
    lhs = _decode_region(args[0], expr, context, converter, expected_access=1)
    rhs = _decode_region(args[1], expr, context, converter, expected_access=1)
    accumulator = _decode_region(args[2], expr, context, converter, expected_access=2)
    if lhs is None or rhs is None or accumulator is None:
        return None
    lhs_type = _view_element_type(lhs.view, context)
    rhs_type = _view_element_type(rhs.view, context)
    accumulator_type = _view_element_type(accumulator.view, context)
    if lhs_type is None or rhs_type is None or accumulator_type is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm regions must resolve to shaped views",
        )
        return None
    if str(lhs_type) != "f16" or str(rhs_type) != "f16":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires f16 inputs",
        )
        return None
    if str(accumulator_type) != "f32":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires f32 accumulator",
        )
        return None
    if not _region_is_shared(lhs) or not _region_is_shared(rhs):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires shared input tiles",
        )
        return None
    if accumulator.index_map.memory_scope != "local.fragment":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires local.fragment accumulator",
        )
        return None
    if not _validate_gemm_region_shapes(expr, context, lhs, rhs, accumulator):
        return None
    return _DenseGemmSpec(
        lhs=lhs,
        rhs=rhs,
        accumulator=accumulator,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        accumulator_type=accumulator_type,
        clear_accumulator=clear_accumulator,
    )


def _convert_cdna_fp8_gemm_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    args: tuple[object, ...],
) -> bool:
    decoded = _decode_cdna_fp8_gemm_spec(expr, context, converter, args)
    if decoded is None:
        return False
    if decoded is _CDNA_FP8_GEMM_BLOCKED:
        return True
    spec = decoded
    if context.flush_pending_workgroup_memory_barrier():
        context.record_converted(
            node_text(expr), "kernel.barrier<workgroup> before cdna fp8 gemm"
        )
    thread_index = map_thread_axis(
        axis=THREAD_AXES["threadIdx.x"],
        sources=(),
        context=context,
        converter=converter,
        name="tx",
    )
    if thread_index is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires mapped threadIdx.x",
        )
        return True

    lhs_state = _build_cdna_fp8_lhs_staging(expr, context, spec, thread_index)
    rhs_state = _build_cdna_fp8_rhs_staging(expr, context, spec, thread_index)
    _emit_cdna_fp8_mfma_updates(expr, context, spec, lhs_state, rhs_state)
    context.record_converted(node_text(expr), "tl.tileop.gemm cdna fp8")
    return True


def _decode_cdna_fp8_gemm_spec(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    args: tuple[object, ...],
) -> _CdnaFp8GemmSpec | _BlockedCdnaFp8Gemm | None:
    target_kind = target_preset_amdgpu_kind(context.target_preset)
    if target_kind not in _CDNA_FP8_TARGET_KINDS:
        return None
    lhs_format = _region_source_element_format(args[0])
    rhs_format = _region_source_element_format(args[1])
    if lhs_format is None or rhs_format is None:
        return None
    transpose_lhs = _static_bool(args[3])
    transpose_rhs = _static_bool(args[4])
    if transpose_lhs is None or transpose_rhs is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 transpose flags must be static",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if transpose_lhs is not False or transpose_rhs is not True:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires transpose_B=True only",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    shape = tuple(integer_value(arg) for arg in args[5:8])
    if shape != (64, 64, 64):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires m/n/k = 64/64/64",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if integer_value(args[8]) != 1:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires FullRow warp policy",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    clear_accumulator = _static_bool(args[9])
    if clear_accumulator is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 clear_accum flag must be static",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if integer_value(args[10]) != 64 or integer_value(args[11]) != 64:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires 64x64 warp tile extents",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if integer_value(args[14]) != 2:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires k_pack = 2",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if integer_value(args[15]) != 0:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge does not import async waits",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if context.static_topology_extent("threadIdx.x") != 256:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires 256 workitems",
        )
        return _CDNA_FP8_GEMM_BLOCKED

    lhs = _decode_region(args[0], expr, context, converter, expected_access=1)
    rhs = _decode_region(args[1], expr, context, converter, expected_access=1)
    accumulator = _decode_region(args[2], expr, context, converter, expected_access=2)
    if lhs is None or rhs is None or accumulator is None:
        return _CDNA_FP8_GEMM_BLOCKED
    lhs_type = _view_element_type(lhs.view, context)
    rhs_type = _view_element_type(rhs.view, context)
    accumulator_type = _view_element_type(accumulator.view, context)
    if lhs_type is None or rhs_type is None or accumulator_type is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge regions must resolve to shaped views",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if (
        str(lhs_type) not in _FP8_ELEMENT_TYPES
        or str(rhs_type) not in _FP8_ELEMENT_TYPES
    ):
        return None
    if str(accumulator_type) != "f32":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires f32 accumulator",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if not _region_is_shared(lhs) or not _region_is_shared(rhs):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires shared input tiles",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if accumulator.index_map.memory_scope != "local.fragment":
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires local.fragment accumulator",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    if (
        _static_extent_tuple(lhs.extents, context) != (64, 64)
        or _static_extent_tuple(rhs.extents, context) != (64, 64)
        or _static_extent_tuple(accumulator.extents, context) != (64, 64)
    ):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm CDNA FP8 bridge requires full 64x64 tile regions",
        )
        return _CDNA_FP8_GEMM_BLOCKED
    return _CdnaFp8GemmSpec(
        lhs=lhs,
        rhs=rhs,
        accumulator=accumulator,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        accumulator_type=accumulator_type,
        lhs_format=lhs_format,
        rhs_format=rhs_format,
    )


_CDNA_FP8_GEMM_BLOCKED = _BlockedCdnaFp8Gemm()


def _build_cdna_fp8_lhs_staging(
    expr: object,
    context: TileLangConversionContext,
    spec: _CdnaFp8GemmSpec,
    thread_index: ValueRef,
) -> ValueRef:
    state = _zero_cdna_fp8_staging_vector(context, spec.lhs_type, "lhs_state")
    lane_mod_16 = _index_rem_const(context, thread_index, 64, "lhs_lane_group")
    lane_div_16 = _index_div_const(context, lane_mod_16, 16, "lhs_lane_row")
    lane_mod_16 = _index_rem_const(context, lane_mod_16, 16, "lhs_lane_col")
    row_group = _index_div_const(context, thread_index, 64, "lhs_row_group")
    row_group = _index_rem_const(context, row_group, 2, "lhs_row_group")
    row_group = _index_mul_const(context, row_group, 32, "lhs_row_base")
    column_group = _index_mul_const(context, lane_div_16, 16, "lhs_col_base")
    for tile in range(2):
        tile_row = _index_add_const(context, row_group, tile * 16, "lhs_tile_row")
        row = context.builder.index.add(
            lhs=tile_row,
            rhs=lane_mod_16,
            results=[INDEX],
            name=context.fresh_name("lhs_row"),
        )
        for lane in range(16):
            column = _index_add_const(context, column_group, lane, "lhs_column")
            value = context.builder.view.load(
                view=spec.lhs.view,
                indices=[row, column],
                results=[spec.lhs_type],
                name=context.fresh_name("lhs_load"),
            )
            state = context.builder.vector.insert(
                value=value,
                dest=state,
                indices=[tile * 16 + lane],
                results=[state.type],
                name=context.fresh_name("lhs_store"),
            )
    return state


def _build_cdna_fp8_rhs_staging(
    expr: object,
    context: TileLangConversionContext,
    spec: _CdnaFp8GemmSpec,
    thread_index: ValueRef,
) -> ValueRef:
    state = _zero_cdna_fp8_staging_vector(context, spec.rhs_type, "rhs_state")
    lane_mod_64 = _index_rem_const(context, thread_index, 64, "rhs_lane_group")
    lane_div_16 = _index_div_const(context, lane_mod_64, 16, "rhs_lane_row")
    lane_mod_16 = _index_rem_const(context, lane_mod_64, 16, "rhs_lane_col")
    column_group = _index_mul_const(context, lane_div_16, 16, "rhs_col_base")
    row_group = _index_div_const(context, thread_index, 128, "rhs_row_group")
    row_group = _index_rem_const(context, row_group, 2, "rhs_row_group")
    row_group = _index_mul_const(context, row_group, 32, "rhs_row_base")
    for tile in range(2):
        tile_row = _index_add_const(context, row_group, tile * 16, "rhs_tile_row")
        row = context.builder.index.add(
            lhs=tile_row,
            rhs=lane_mod_16,
            results=[INDEX],
            name=context.fresh_name("rhs_row"),
        )
        for lane in range(16):
            column = _index_add_const(context, column_group, lane, "rhs_column")
            value = context.builder.view.load(
                view=spec.rhs.view,
                indices=[row, column],
                results=[spec.rhs_type],
                name=context.fresh_name("rhs_load"),
            )
            state = context.builder.vector.insert(
                value=value,
                dest=state,
                indices=[tile * 16 + lane],
                results=[state.type],
                name=context.fresh_name("rhs_store"),
            )
    return state


def _emit_cdna_fp8_mfma_updates(
    expr: object,
    context: TileLangConversionContext,
    spec: _CdnaFp8GemmSpec,
    lhs_state: ValueRef,
    rhs_state: ValueRef,
) -> None:
    source_lhs_type = context.type_converter.vector_type(spec.lhs_type, 8)
    source_rhs_type = context.type_converter.vector_type(spec.rhs_type, 8)
    payload_type = context.type_converter.vector_type(I32, 2)
    accumulator_type = context.type_converter.vector_type(spec.accumulator_type, 4)
    m = context.ensure_constant("16", "index", "c16")
    n = context.ensure_constant("16", "index", "c16")
    k = context.ensure_constant("32", "index", "c32")
    lhs_schema = _cdna_fp8_matrix_schema(context, spec.lhs_format)
    rhs_schema = _cdna_fp8_matrix_schema(context, spec.rhs_format)
    zero = context.ensure_constant("0", "index", "c0")
    for k_part in range(2):
        for row_tile in range(2):
            for column_tile in range(2):
                lhs_lanes = context.builder.vector.slice(
                    source=lhs_state,
                    offsets=[_index_const(context, (row_tile * 2 + k_part) * 8)],
                    results=[source_lhs_type],
                    name=context.fresh_name("lhs_lanes"),
                )
                lhs_payload = context.builder.vector.bitcast(
                    input=lhs_lanes,
                    results=[payload_type],
                    name=context.fresh_name("lhs_payload"),
                )
                lhs = context.builder.vector.fragment(
                    role="lhs",
                    data=lhs_payload,
                    rows=m,
                    columns=k,
                    params={"schema": lhs_schema},
                    results=[payload_type],
                    name=context.fresh_name("lhs"),
                )
                rhs_lanes = context.builder.vector.slice(
                    source=rhs_state,
                    offsets=[_index_const(context, (column_tile * 2 + k_part) * 8)],
                    results=[source_rhs_type],
                    name=context.fresh_name("rhs_lanes"),
                )
                rhs_payload = context.builder.vector.bitcast(
                    input=rhs_lanes,
                    results=[payload_type],
                    name=context.fresh_name("rhs_payload"),
                )
                rhs = context.builder.vector.fragment(
                    role="rhs",
                    data=rhs_payload,
                    rows=k,
                    columns=n,
                    params={"schema": rhs_schema},
                    results=[payload_type],
                    name=context.fresh_name("rhs"),
                )
                accumulator_column = _index_const(
                    context,
                    (row_tile * 2 + column_tile) * 4,
                )
                accumulator_payload = context.builder.vector.load(
                    view=spec.accumulator.view,
                    indices=[zero, accumulator_column],
                    results=[accumulator_type],
                    name=context.fresh_name("acc_payload"),
                )
                init = context.builder.vector.fragment(
                    role="init",
                    data=accumulator_payload,
                    rows=m,
                    columns=n,
                    results=[accumulator_type],
                    name=context.fresh_name("init"),
                )
                result = context.builder.vector.mma(
                    lhs=lhs,
                    rhs=rhs,
                    init=init,
                    results=[accumulator_type],
                    name=context.fresh_name("mfma"),
                )
                context.builder.vector.store(
                    value=result,
                    view=spec.accumulator.view,
                    indices=[zero, accumulator_column],
                )


def _zero_cdna_fp8_staging_vector(
    context: TileLangConversionContext,
    element_type: ScalarType,
    name: str,
) -> ValueRef:
    zero = context.ensure_typed_constant(0.0, element_type, base="zero")
    return context.builder.vector.splat(
        scalar=zero,
        results=[context.type_converter.vector_type(element_type, 32)],
        name=context.fresh_name(name),
    )


def _cdna_fp8_matrix_schema(
    context: TileLangConversionContext,
    element_format: str,
) -> ValueRef:
    return context.storage_schema_value(
        EncodingInstance(
            name="matrix_operand",
            params=(
                ("element_format", element_format),
                ("payload_elements", 8),
                ("payload_registers", 2),
            ),
        )
    )


def _region_source_element_format(expr: object) -> str | None:
    if _call_op_name(expr) != "tl.tileop.region":
        return None
    args = _args(expr)
    if not args or node_kind(args[0]) != "BufferLoad":
        return None
    source_dtype = dtype(getattr(args[0], "buffer", None))
    return _FP8_SOURCE_FORMATS.get(source_dtype)


def _index_const(context: TileLangConversionContext, value: int) -> ValueRef:
    return context.ensure_constant(str(value), "index", f"c{value}")


def _index_add_const(
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: int,
    name: str,
) -> ValueRef:
    if rhs == 0:
        return lhs
    return context.builder.index.add(
        lhs=lhs,
        rhs=_index_const(context, rhs),
        results=[INDEX],
        name=context.fresh_name(name),
    )


def _index_mul_const(
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: int,
    name: str,
) -> ValueRef:
    return context.builder.index.mul(
        lhs=lhs,
        rhs=_index_const(context, rhs),
        results=[INDEX],
        name=context.fresh_name(name),
    )


def _index_div_const(
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: int,
    name: str,
) -> ValueRef:
    return context.builder.index.div(
        lhs=lhs,
        rhs=_index_const(context, rhs),
        results=[INDEX],
        name=context.fresh_name(name),
    )


def _index_rem_const(
    context: TileLangConversionContext,
    lhs: ValueRef,
    rhs: int,
    name: str,
) -> ValueRef:
    return context.builder.index.rem(
        lhs=lhs,
        rhs=_index_const(context, rhs),
        results=[INDEX],
        name=context.fresh_name(name),
    )


def _validate_gemm_region_shapes(
    expr: object,
    context: TileLangConversionContext,
    lhs: TileLangBufferRegion,
    rhs: TileLangBufferRegion,
    accumulator: TileLangBufferRegion,
) -> bool:
    if _static_extent_tuple(lhs.extents, context) != (16, 16):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires full 16x16 lhs region",
        )
        return False
    if _static_extent_tuple(rhs.extents, context) != (16, 16):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm vector.fragment bridge requires full 16x16 rhs region",
        )
        return False
    if _static_extent_tuple(accumulator.extents, context) != (16, 16):
        context.record_blocked(
            node_text(expr),
            (
                "tl.tileop.gemm vector.fragment bridge requires full 16x16 "
                "accumulator region"
            ),
        )
        return False
    if not _all_zero_indices(accumulator.indices, context):
        context.record_blocked(
            node_text(expr),
            (
                "tl.tileop.gemm vector.fragment bridge requires full 16x16 "
                "accumulator region"
            ),
        )
        return False
    return True


def _gemm_init_fragment(
    expr: object,
    context: TileLangConversionContext,
    spec: _DenseGemmSpec,
) -> ValueRef | None:
    if spec.clear_accumulator:
        zero = context.ensure_typed_constant(
            0.0,
            spec.accumulator_type,
            base="zero",
        )
        return _build_zero_init_fragment(context, spec, zero)
    fragment = context.matrix_fragment(spec.accumulator.view)
    if fragment is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm accumulator has no imported matrix fragment state",
        )
        return None
    if not _same_value_sequence(
        (fragment.rows, fragment.columns),
        (spec.accumulator.extents[0], spec.accumulator.extents[1]),
    ):
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm accumulator fragment shape does not match region",
        )
        return None
    if fragment.element_type != spec.accumulator_type:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.gemm accumulator fragment element type does not match region",
        )
        return None
    return fragment.value


def _build_zero_init_fragment(
    context: TileLangConversionContext,
    spec: _DenseGemmSpec,
    zero: ValueRef,
) -> ValueRef:
    payload_type = context.type_converter.vector_type(spec.accumulator_type, 8)
    payload = context.builder.vector.splat(
        scalar=zero,
        results=[payload_type],
        name=context.fresh_name("zero"),
    )
    return context.builder.vector.fragment(
        role="init",
        data=payload,
        rows=spec.accumulator.extents[0],
        columns=spec.accumulator.extents[1],
        results=[payload_type],
        name=context.fresh_name("init"),
    )


def _require_effect(
    expr: object,
    context: TileLangConversionContext,
    options: ExpressionOptions,
    op_name: str,
) -> bool:
    if options.effect:
        return True
    context.record_blocked(
        node_text(expr),
        f"call `{op_name}` is effect-only and must appear under tir.Evaluate",
    )
    return False


def _record_unsupported_tileop(
    expr: object,
    context: TileLangConversionContext,
    op_name: str,
) -> None:
    row = coverage_row(op_name)
    if row is None:
        context.record_blocked(
            node_text(expr),
            f"TileLang tile operation `{op_name}` is not in the coverage manifest",
        )
        return
    context.record_blocked(
        node_text(expr),
        f"call `{op_name}` coverage state is {row.state.value}: {row.note}",
    )


def _decode_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    expected_access: int,
) -> TileLangBufferRegion | None:
    op_name = _call_op_name(expr)
    if op_name != "tl.tileop.region":
        context.record_blocked(
            node_text(owner),
            f"tile operation region must be tl.tileop.region, got `{op_name}`",
        )
        return None
    args = _args(expr)
    if len(args) < 2:
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region expects buffer load, access mask, and extents",
        )
        return None
    buffer_load = args[0]
    if node_kind(buffer_load) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region base is not a BufferLoad",
        )
        return None
    access_mask = integer_value(args[1])
    if access_mask is None:
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region access mask is not static",
        )
        return None
    if access_mask & expected_access == 0:
        context.record_blocked(
            node_text(owner),
            f"tl.tileop.region access mask `{access_mask}` does not permit "
            "the operation",
        )
        return None
    return resolve_buffer_region(
        getattr(buffer_load, "buffer", None),
        tuple(getattr(buffer_load, "indices", ())),
        args[2:],
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_region_arg(
    *,
    args: tuple[object, ...],
    position: int,
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferRegion | None:
    value = args[position]
    if _call_op_name(value) == "tl.tileop.region":
        return _decode_region(value, expr, context, converter, expected_access=3)
    if node_kind(value) == "BufferRegion":
        return _decode_buffer_region(value, expr, context, converter)
    if node_kind(value) == "BufferLoad":
        return _decode_buffer_load_region(value, expr, context, converter)
    context.record_blocked(
        node_text(expr),
        f"tile operation region argument must be region-like, got `{node_kind(value)}`",
    )
    return None


def _decode_buffer_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferRegion | None:
    ranges = tuple(getattr(expr, "region", ()))
    return resolve_buffer_region(
        getattr(expr, "buffer", None),
        tuple(getattr(item, "min", None) for item in ranges),
        tuple(getattr(item, "extent", None) for item in ranges),
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_buffer_load_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferRegion | None:
    one = context.ensure_constant("1", "index", "c1")
    indices: list[ValueRef] = []
    extents: list[ValueRef] = []
    for index in tuple(getattr(expr, "indices", ())):
        if node_kind(index) == "Ramp":
            stride = integer_value(getattr(index, "stride", None))
            lanes = integer_value(getattr(index, "lanes", None))
            if stride != 1 or lanes is None:
                context.record_blocked(
                    node_text(owner),
                    "tile operation ramp region requires unit stride and static lanes",
                )
                return None
            base = converter.convert_expr(
                getattr(index, "base", None),
                context,
                index_like=True,
            )
            if base is None:
                context.record_blocked(
                    node_text(owner),
                    "tile operation ramp region base is not mapped",
                )
                return None
            indices.append(base)
            extents.append(context.ensure_constant(str(lanes), "index", f"c{lanes}"))
            continue
        mapped = converter.convert_expr(index, context, index_like=True)
        if mapped is None:
            context.record_blocked(
                node_text(owner),
                "tile operation buffer load region index is not mapped",
            )
            return None
        indices.append(mapped)
        extents.append(one)
    return resolve_buffer_region(
        getattr(expr, "buffer", None),
        tuple(indices),
        tuple(extents),
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_buffer_access(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferAccess | None:
    if node_kind(expr) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            f"tile operation buffer access must be BufferLoad, got `{node_kind(expr)}`",
        )
        return None
    return resolve_buffer_access(
        getattr(expr, "buffer", None),
        tuple(getattr(expr, "indices", ())),
        context,
        converter,
        diagnostic_owner=owner,
    )


def _reducer_data_source(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
) -> object | None:
    region_like = expr
    if _call_op_name(region_like) == "tl.tileop.region":
        args = _args(region_like)
        if not args:
            context.record_blocked(
                node_text(owner),
                "tl.tileop.finalize_reducer region has no base",
            )
            return None
        region_like = args[0]
    if node_kind(region_like) == "BufferLoad":
        buffer = getattr(region_like, "buffer", None)
        data = getattr(buffer, "data", None)
        if data is not None:
            return cast(object, data)
    if node_kind(region_like) == "BufferRegion":
        buffer = getattr(region_like, "buffer", None)
        data = getattr(buffer, "data", None)
        if data is not None:
            return cast(object, data)
    context.record_blocked(
        node_text(owner),
        (
            "tl.tileop.finalize_reducer target must be a reducer BufferLoad, "
            f"BufferRegion, or tl.tileop.region, got `{node_kind(expr)}`"
        ),
    )
    return None


def _copy_loop_extents(
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    source_extents = _squeezed_singleton_extents(source.extents, context)
    target_extents = _squeezed_singleton_extents(target.extents, context)
    if len(source_extents) != len(target_extents):
        return None
    if all(
        lhs.id == rhs.id
        for lhs, rhs in zip(source_extents, target_extents, strict=False)
    ):
        return source_extents
    return None


def _squeezed_singleton_extents(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    return tuple(
        extent for extent in extents if not _is_static_one_extent(extent, context)
    )


def _projected_region_rank(
    region: TileLangBufferRegion,
    context: TileLangConversionContext,
) -> int:
    return len(_squeezed_singleton_extents(region.extents, context))


def _is_static_one_extent(
    extent: ValueRef,
    context: TileLangConversionContext,
) -> bool:
    one = context.constants.get(("index", "1"))
    return one is not None and extent.id == one.id


def _bool_value(value: object) -> bool | None:
    payload = getattr(value, "value", value)
    if isinstance(payload, bool):
        return payload
    integer = integer_value(value)
    if integer in (0, 1):
        return bool(integer)
    return None


def _all_zero_indices(
    indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> bool:
    zero = context.constants.get(("index", "0"))
    return zero is not None and all(index.id == zero.id for index in indices)


def _load_tracked_local_value(
    view: ValueRef,
    indices: tuple[ValueRef, ...],
    memory_scope: str,
    result_type: Type,
    context: TileLangConversionContext,
    *,
    name: str,
) -> ValueRef | None:
    if memory_scope == "local.fragment":
        fragment_value = context.fragment_vector(view)
        if fragment_value is not None:
            if fragment_value.base is None:
                if len(indices) != 1:
                    return None
                return context.builder.vector.extract(
                    source=fragment_value.value,
                    indices=[indices[0]],
                    results=[result_type],
                    name=context.fresh_name(name),
                )
            lane = _single_distributed_index(indices, context)
            if (
                lane is None
                or lane.lane < 0
                or lane.lane_count != fragment_value.lane_count
            ):
                return None
            return context.builder.vector.extract(
                source=fragment_value.value,
                indices=[lane.lane],
                results=[result_type],
                name=context.fresh_name(name),
            )
    value = context.mapped_buffer_access(view, indices, memory_scope)
    if value is not None:
        return value
    if _memory_scope_is_shared(memory_scope):
        context.flush_pending_workgroup_memory_barrier()
    return context.builder.view.load(
        view=view,
        indices=list(indices),
        results=[result_type],
        name=context.fresh_name(name),
    )


def _store_tracked_local_value(
    owner: object,
    view: ValueRef,
    indices: tuple[ValueRef, ...],
    memory_scope: str,
    value: ValueRef,
    context: TileLangConversionContext,
) -> None:
    if memory_scope == "local.fragment":
        context.clear_matrix_fragment(view)
        fragment_value = context.fragment_vector(view)
        if fragment_value is not None:
            updated = _insert_fragment_vector_value(
                owner,
                fragment_value,
                indices,
                value,
                context,
            )
            if updated is not None:
                context.map_fragment_vector(view, updated)
            return
    if memory_scope == "local.var":
        context.invalidate_buffer_accesses(view)
        context.map_buffer_access(owner, view, indices, memory_scope, value)
        return
    context.builder.view.store(value=value, view=view, indices=list(indices))
    if _memory_scope_is_shared(memory_scope):
        context.mark_pending_workgroup_memory_write()
    context.invalidate_buffer_accesses(view)
    context.map_buffer_access(owner, view, indices, memory_scope, value)


def _insert_fragment_vector_value(
    owner: object,
    fragment_value: TileLangFragmentVector,
    indices: tuple[ValueRef, ...],
    value: ValueRef,
    context: TileLangConversionContext,
) -> TileLangFragmentVector | None:
    if fragment_value.base is None:
        if len(indices) != 1:
            context.record_blocked(
                node_text(owner),
                "local.fragment vector store is not rank-1",
            )
            return None
        result = context.builder.vector.insert(
            value=value,
            dest=fragment_value.value,
            indices=[indices[0]],
            results=[fragment_value.value.type],
            name=context.fresh_name("store"),
        )
        return TileLangFragmentVector(
            value=result,
            lane_count=fragment_value.lane_count,
            base=None,
        )
    lane = _single_distributed_index(indices, context)
    if lane is None or lane.lane < 0 or lane.lane_count != fragment_value.lane_count:
        context.record_blocked(
            node_text(owner),
            "local.fragment vector store is not aligned with a static lane",
        )
        return None
    result = context.builder.vector.insert(
        value=value,
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


def _fragment_vector_loop_state_views(
    context: TileLangConversionContext,
    *regions: TileLangBufferRegion,
) -> tuple[ValueRef, ...]:
    views: list[ValueRef] = []
    seen_view_ids: set[int] = set()
    for region in regions:
        if region.index_map.memory_scope != "local.fragment":
            continue
        if region.view.id in seen_view_ids:
            continue
        if context.fragment_vector(region.view) is None:
            continue
        views.append(region.view)
        seen_view_ids.add(region.view.id)
    return tuple(views)


def _fragment_vector_loop_state_slots(
    context: TileLangConversionContext,
    views: tuple[ValueRef, ...],
) -> list[_FragmentVectorLoopState]:
    slots: list[_FragmentVectorLoopState] = []
    for view in views:
        initial = context.fragment_vector(view)
        if initial is None:
            continue
        view_value = context.builder.module.values[view.id]
        base_name = sanitize_identifier(view_value.name or "fragment")
        slots.append(
            _FragmentVectorLoopState(
                view=view,
                initial=initial,
                name=f"{base_name}_state",
            )
        )
    return slots


def _fragment_vector_loop_state_values(
    slots: list[_FragmentVectorLoopState],
    context: TileLangConversionContext,
) -> list[ValueRef]:
    values: list[ValueRef] = []
    for slot in slots:
        current = context.fragment_vector(slot.view)
        values.append(current.value if current is not None else slot.initial.value)
    return values


def _map_fragment_vector_loop_state_args(
    slots: list[_FragmentVectorLoopState],
    arg_ids: list[int],
    context: TileLangConversionContext,
) -> None:
    for slot, arg_id in zip(slots, arg_ids, strict=True):
        context.map_fragment_vector(
            slot.view,
            TileLangFragmentVector(
                value=ValueRef(arg_id, context.builder.ir),
                lane_count=slot.initial.lane_count,
                base=slot.initial.base,
            ),
        )


def _map_fragment_vector_loop_state_results(
    slots: list[_FragmentVectorLoopState],
    results: list[ValueRef],
    context: TileLangConversionContext,
) -> None:
    for slot, result in zip(slots, results, strict=True):
        context.map_fragment_vector(
            slot.view,
            TileLangFragmentVector(
                value=result,
                lane_count=slot.initial.lane_count,
                base=slot.initial.base,
            ),
        )


def _emit_region_loops(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
    emit_body: Callable[[tuple[ValueRef, ...], TileLangConversionContext], None],
    *,
    distribution: bool = False,
    converter: TileLangConverter | None = None,
    owner: object | None = None,
    fragment_vector_state: tuple[ValueRef, ...] = (),
) -> bool:
    state_slots = _fragment_vector_loop_state_slots(context, fragment_vector_state)
    if not extents:
        emit_body((), context)
        return True

    if distribution and converter is not None and owner is not None and not state_slots:
        static_extents = _static_extent_values(extents, context)
        if static_extents is not None:
            extent_integer = _product(static_extents)

            def emit_flat_body(
                flat_indices: tuple[ValueRef, ...],
                body_context: TileLangConversionContext,
            ) -> None:
                emit_body(
                    _unflatten_loop_indices(flat_indices[0], extents, body_context),
                    body_context,
                )

            return emit_distributed_1d(
                owner,
                context,
                converter,
                extent=extents[0] if len(extents) == 1 else None,
                extent_integer=extent_integer,
                index_name="i0" if len(extents) == 1 else "i",
                emit_body=emit_flat_body,
            )

    zero = context.ensure_constant("0", "index", "c0")
    one = context.ensure_constant("1", "index", "c1")

    def emit_at_depth(
        depth: int,
        indices: tuple[ValueRef, ...],
        loop_context: TileLangConversionContext,
    ) -> None:
        if depth == len(extents):
            emit_body(indices, loop_context)
            return
        result_names = [
            loop_context.fresh_name(f"{slot.name}_next") for slot in state_slots
        ]
        loop_body = loop_context.builder.region(
            args=[
                (f"i{depth}", INDEX),
                *(
                    (f"{slot.name}_iter", slot.initial.value.type)
                    for slot in state_slots
                ),
            ]
        )
        iv = ValueRef(loop_body.blocks[0].arg_ids[0], loop_context.builder.ir)
        child = loop_context.fork(preview_block=loop_body.blocks[0])
        _map_fragment_vector_loop_state_args(
            state_slots,
            loop_body.blocks[0].arg_ids[1:],
            child,
        )
        with loop_context.builder.insertion_block(loop_body.blocks[0]):
            emit_at_depth(depth + 1, (*indices, iv), child)
            loop_context.builder.scf.yield_(
                values=_fragment_vector_loop_state_values(state_slots, child)
            )
        loop_context.merge_child_records(child)
        loop_results = loop_context.builder.scf.for_(
            lower_bound=zero,
            upper_bound=extents[depth],
            step=one,
            iter_args=_fragment_vector_loop_state_values(state_slots, loop_context),
            results=[slot.initial.value.type for slot in state_slots],
            body=loop_body,
            names=result_names or None,
        )
        _map_fragment_vector_loop_state_results(
            state_slots,
            loop_results,
            loop_context,
        )

    emit_at_depth(0, (), context)
    return True


def _copy_uses_distributed_index_space(
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    loop_extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> bool:
    if len(loop_extents) == 1 and (
        _region_is_fragment(source) or _region_is_fragment(target)
    ):
        return True
    thread_count = context.static_topology_extent("threadIdx.x")
    return (
        thread_count is not None
        and thread_count > 1
        and (_region_is_shared(source) or _region_is_shared(target))
    )


def _region_is_fragment(region: TileLangBufferRegion) -> bool:
    return region.index_map.memory_scope == "local.fragment"


def _region_is_shared(region: TileLangBufferRegion) -> bool:
    return _memory_scope_is_shared(region.index_map.memory_scope)


def _memory_scope_is_shared(memory_scope: str) -> bool:
    return memory_scope.startswith("shared")


def _region_has_dense_layout(
    region: TileLangBufferRegion,
    context: TileLangConversionContext,
) -> bool:
    if context.dense_layout is None:
        return False
    view = context.builder.module.values[region.view.id]
    return view.encoding_binding == context.dense_layout.id


def _same_value_sequence(
    lhs: tuple[ValueRef, ...],
    rhs: tuple[ValueRef, ...],
) -> bool:
    return len(lhs) == len(rhs) and all(
        lhs_value.id == rhs_value.id
        for lhs_value, rhs_value in zip(lhs, rhs, strict=True)
    )


def _static_extent_tuple(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[int, ...] | None:
    return _static_extent_values(extents, context)


def _static_extent_values(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[int, ...] | None:
    values: list[int] = []
    for extent in extents:
        value = static_index_value(extent, context)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _product(values: tuple[int, ...]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


def _unflatten_loop_indices(
    flat_index: ValueRef,
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    if len(extents) == 1:
        return (flat_index,)
    indices: list[ValueRef | None] = [None] * len(extents)
    running = flat_index
    for position in range(len(extents) - 1, 0, -1):
        extent = extents[position]
        indices[position] = context.builder.index.rem(
            lhs=running,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name("i_rem"),
        )
        running = context.builder.index.div(
            lhs=running,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name("i_div"),
        )
    indices[0] = running
    return tuple(index for index in indices if index is not None)


def _unflatten_vector_chunk_indices(
    chunk_index: ValueRef,
    static_extents: tuple[int, ...],
    lane_count: int,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    chunk_extents = [
        context.ensure_constant(str(extent), "index", f"c{extent}")
        for extent in (*static_extents[:-1], static_extents[-1] // lane_count)
    ]
    indices = list(_unflatten_loop_indices(chunk_index, tuple(chunk_extents), context))
    lane_count_value = context.ensure_constant(
        str(lane_count), "index", f"c{lane_count}"
    )
    indices[-1] = context.builder.index.mul(
        lhs=indices[-1],
        rhs=lane_count_value,
        results=[INDEX],
        name=context.fresh_name("i_offset"),
    )
    return tuple(indices)


def _offset_trailing_lane(
    indices: tuple[ValueRef, ...],
    lane: int,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    if lane == 0:
        return indices
    lane_offset = context.ensure_constant(str(lane), "index", f"c{lane}")
    return (
        *indices[:-1],
        add_index(indices[-1], lane_offset, context),
    )


def _is_zero_fill_source(source: object) -> bool:
    payload = getattr(source, "value", source)
    if isinstance(payload, bool):
        return not payload
    if isinstance(payload, int | float):
        return payload == 0
    return False


def _region_indices(
    region: TileLangBufferRegion,
    loop_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    return map_region_indices(region, loop_indices, context)


def _project_region_indices(
    region: TileLangBufferRegion,
    loop_indices: tuple[ValueRef, ...],
    owner: object,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    zero = context.ensure_constant("0", "index", "c0")
    offsets: list[ValueRef] = []
    loop_index = 0
    for extent in region.extents:
        if _is_static_one_extent(extent, context):
            offsets.append(zero)
            continue
        if loop_index >= len(loop_indices):
            context.record_blocked(
                node_text(owner),
                "tile operation copy projected rank does not match loop rank",
            )
            return None
        offsets.append(loop_indices[loop_index])
        loop_index += 1
    if loop_index != len(loop_indices):
        context.record_blocked(
            node_text(owner),
            "tile operation copy loop rank does not match projected rank",
        )
        return None
    return _region_indices(region, tuple(offsets), context)


def _reduction_target_indices(
    target: TileLangBufferRegion,
    source_rank: int,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    if len(target.indices) == len(output_indices):
        return _region_indices(target, output_indices, context)
    if len(target.indices) == source_rank:
        expanded: list[ValueRef] = []
        output_axis = 0
        zero = context.ensure_constant("0", "index", "c0")
        for axis in range(source_rank):
            if axis == dim:
                expanded.append(zero)
                continue
            expanded.append(output_indices[output_axis])
            output_axis += 1
        return _region_indices(target, tuple(expanded), context)
    if not output_indices and len(target.indices) == 1:
        zero = context.ensure_constant("0", "index", "c0")
        return _region_indices(target, (zero,), context)
    return None


def _try_emit_fragment_vector_reduce(
    owner: object,
    source: TileLangBufferRegion,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    init: ValueRef,
    reduce_spec: TileReduceSpec,
    source_element_type: ScalarType,
    element_type: ScalarType,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if dim != 0 or output_indices:
        return None
    if not _region_is_fragment(source) or len(source.extents) != 1:
        return None
    fragment_value = context.fragment_vector(source.view)
    if fragment_value is None or fragment_value.base is not None:
        return None
    extent_value = static_index_value(source.extents[0], context)
    if extent_value is None or extent_value != fragment_value.lane_count:
        return None
    zero = context.ensure_constant("0", "index", "c0")
    source_indices = _region_indices(source, (zero,), context)
    if not _all_zero_indices(source_indices, context):
        return None
    source_type = context.builder.module.values[fragment_value.value.id].type
    if not _is_rank_one_vector_type(source_type):
        return None
    source_type = cast(ShapedType, source_type)
    if source_type.dims[0].size != fragment_value.lane_count:
        return None
    source_value = _reduce_vector_value_cast(
        fragment_value.value,
        source_type,
        source_element_type,
        element_type,
        owner,
        context,
    )
    if source_value is None:
        return None
    source_value = _build_vector_reduce_input(
        source_value,
        reduce_spec,
        element_type,
        context,
    )
    reduce_name = context.fresh_name("reduce")
    reduce_fastmath = _additive_reduce_fastmath_flags(
        reduce_spec,
        element_type,
        context,
    )
    if reduce_fastmath is None:
        return context.builder.vector.reduce(
            kind=reduce_spec.combiner,
            input=source_value,
            init=init,
            results=[element_type],
            name=reduce_name,
        )
    return context.builder.vector.reduce(
        kind=reduce_spec.combiner,
        fastmath=reduce_fastmath,
        input=source_value,
        init=init,
        results=[element_type],
        name=reduce_name,
    )


def _emit_scalar_reduce_loop(
    source: TileLangBufferRegion,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    init: ValueRef,
    reduce_spec: TileReduceSpec,
    source_element_type: ScalarType,
    element_type: ScalarType,
    context: TileLangConversionContext,
) -> ValueRef | None:
    zero = context.ensure_constant("0", "index", "c0")
    one = context.ensure_constant("1", "index", "c1")
    loop_body = context.builder.region(args=[("r", INDEX), ("acc", element_type)])
    reduce_index = ValueRef(loop_body.blocks[0].arg_ids[0], context.builder.ir)
    accumulator = ValueRef(loop_body.blocks[0].arg_ids[1], context.builder.ir)
    child = context.fork(preview_block=loop_body.blocks[0])
    with context.builder.insertion_block(loop_body.blocks[0]):
        source_indices = _reduction_source_indices(
            source,
            dim,
            output_indices,
            reduce_index,
            child,
        )
        value = child.builder.view.load(
            view=source.view,
            indices=list(source_indices),
            results=[source_element_type],
            name=child.fresh_name("reduce_value"),
        )
        cast_value = _reduce_scalar_value_cast(
            value,
            source_element_type,
            element_type,
            child,
        )
        if cast_value is None:
            child.builder.scf.yield_(values=[accumulator])
            context.merge_child_records(child)
            return None
        value = cast_value
        value = _build_scalar_reduce_input(value, reduce_spec, element_type, child)
        combined = _build_scalar_combiner(
            reduce_spec.combiner,
            accumulator,
            value,
            element_type,
            child,
        )
        child.builder.scf.yield_(values=[combined])
    context.merge_child_records(child)
    loop_results = context.builder.scf.for_(
        lower_bound=zero,
        upper_bound=source.extents[dim],
        step=one,
        iter_args=[init],
        results=[element_type],
        body=loop_body,
        names=[context.fresh_name("reduce")],
    )
    return loop_results[0]


def _reduce_vector_value_cast(
    value: ValueRef,
    value_type: ShapedType,
    source_element_type: ScalarType,
    target_element_type: ScalarType,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(source_element_type) == str(target_element_type):
        return value
    target_type = ShapedType(
        TypeKind.VECTOR,
        target_element_type,
        value_type.dims,
        encoding=value_type.encoding,
    )
    builder_name = _reduce_value_cast_builder_name(
        source_element_type,
        target_element_type,
    )
    if builder_name is None:
        context.record_blocked(
            node_text(owner),
            (
                "tl.tileop.reduce source vector conversion "
                f"{source_element_type} to {target_element_type} is not imported"
            ),
        )
        return None
    builder = getattr(context.builder.vector, builder_name)
    return cast(
        ValueRef,
        builder(
            input=value,
            results=[target_type],
            name=context.fresh_name("reduce_cast"),
        ),
    )


def _reduce_scalar_value_cast(
    value: ValueRef,
    source_element_type: ScalarType,
    target_element_type: ScalarType,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(source_element_type) == str(target_element_type):
        return value
    builder_name = _reduce_value_cast_builder_name(
        source_element_type,
        target_element_type,
    )
    if builder_name is None:
        return None
    builder = getattr(context.builder.scalar, builder_name)
    return cast(
        ValueRef,
        builder(
            input=value,
            results=[target_element_type],
            name=context.fresh_name("reduce_cast"),
        ),
    )


def _reduce_value_cast_is_supported(
    source_element_type: ScalarType,
    target_element_type: ScalarType,
) -> bool:
    return (
        str(source_element_type) == str(target_element_type)
        or _reduce_value_cast_builder_name(source_element_type, target_element_type)
        is not None
    )


def _reduce_value_cast_builder_name(
    source_element_type: ScalarType,
    target_element_type: ScalarType,
) -> str | None:
    source_text = str(source_element_type)
    target_text = str(target_element_type)
    if _is_float_type(source_element_type) and _is_float_type(target_element_type):
        source_width = _floating_point_bit_width(source_text)
        target_width = _floating_point_bit_width(target_text)
        if source_width < target_width:
            return "extf"
        if source_width > target_width:
            return "fptrunc"
        return None
    source_integer_width = _integer_bit_width(source_text)
    target_integer_width = _integer_bit_width(target_text)
    if source_integer_width is not None and target_integer_width is not None:
        if source_integer_width < target_integer_width:
            return "extsi"
        if source_integer_width > target_integer_width:
            return "trunci"
        return "bitcast"
    return None


def _reduction_source_indices(
    source: TileLangBufferRegion,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    reduce_index: ValueRef,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    source_offsets: list[ValueRef] = []
    output_axis = 0
    for axis in range(len(source.indices)):
        if axis == dim:
            source_offsets.append(reduce_index)
            continue
        source_offsets.append(output_indices[output_axis])
        output_axis += 1
    return _region_indices(source, tuple(source_offsets), context)


def _build_vector_reduce_input(
    source: ValueRef,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    if not reduce_spec.absolute:
        return source
    vector_type = context.builder.module.values[source.id].type
    builder_name = "absf" if _is_float_type(element_type) else "absi"
    builder = getattr(context.builder.vector, builder_name)
    return cast(
        ValueRef,
        builder(
            input=source,
            results=[vector_type],
            name=context.fresh_name("abs"),
        ),
    )


def _build_scalar_reduce_input(
    source: ValueRef,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    if not reduce_spec.absolute:
        return source
    builder_name = "absf" if _is_float_type(element_type) else "absi"
    builder = getattr(context.builder.scalar, builder_name)
    return cast(
        ValueRef,
        builder(
            input=source,
            results=[element_type],
            name=context.fresh_name("abs"),
        ),
    )


def _build_scalar_combiner(
    kind: str,
    lhs: ValueRef,
    rhs: ValueRef,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    builder = getattr(context.builder.scalar, kind)
    fastmath = (
        context.float_fastmath_flags
        if kind == "addf" and _is_float_type(element_type)
        else None
    )
    if fastmath is not None:
        return cast(
            ValueRef,
            builder(
                fastmath=fastmath,
                lhs=lhs,
                rhs=rhs,
                results=[element_type],
                name=context.fresh_name(kind),
            ),
        )
    return cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[element_type],
            name=context.fresh_name(kind),
        ),
    )


def _additive_reduce_fastmath_flags(
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> str | None:
    if reduce_spec.combiner != "addf" or not _is_float_type(element_type):
        return None
    return context.float_fastmath_flags


def _cumsum_combiner(element_type: Type) -> str | None:
    if _is_float_type(element_type):
        return "addf"
    bit_width = _integer_bit_width(str(element_type))
    if bit_width is not None and bit_width > 1:
        return "addi"
    return None


def _copy_value_cast(
    value: ValueRef,
    source_element_type: Type,
    target_element_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(source_element_type) == str(target_element_type):
        return value
    source_width = _integer_bit_width(str(source_element_type))
    target_width = _integer_bit_width(str(target_element_type))
    if source_width is not None and target_width is not None:
        if source_width < target_width:
            return context.builder.scalar.extsi(
                input=value,
                results=[target_element_type],
                name=context.fresh_name("copy_ext"),
            )
        if source_width > target_width:
            return context.builder.scalar.trunci(
                input=value,
                results=[target_element_type],
                name=context.fresh_name("copy_trunc"),
            )
    context.record_blocked(
        node_text(owner),
        (
            "tl.tileop.copy element conversion "
            f"{source_element_type} to {target_element_type} is not imported"
        ),
    )
    return None


def _view_element_type(
    view: ValueRef,
    context: TileLangConversionContext,
) -> ScalarType | None:
    view_type = context.builder.module.values[view.id].type
    if not isinstance(view_type, ShapedType):
        return None
    return view_type.element_type


def _is_rank_one_vector_type(value_type: Type) -> bool:
    return (
        isinstance(value_type, ShapedType)
        and value_type.type_kind == TypeKind.VECTOR
        and len(value_type.dims) == 1
        and isinstance(value_type.dims[0], StaticDim)
    )


def _single_ramp_axis(expr: object) -> int | None:
    if node_kind(expr) != "BufferLoad":
        return None
    ramp_axis: int | None = None
    for axis, index in enumerate(tuple(getattr(expr, "indices", ()))):
        if node_kind(index) != "Ramp":
            continue
        if ramp_axis is not None:
            return None
        ramp_axis = axis
    return ramp_axis


def _reduce_spec(
    kind_expr: object,
    source: object,
    element_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> TileReduceSpec | None:
    reduce_type = getattr(kind_expr, "value", None)
    if reduce_type is None:
        context.record_blocked(node_text(owner), "tl.tileop.reduce kind is not static")
        return None
    reduce_type = str(reduce_type)
    annotations = _call_annotations(owner)
    unknown_annotations = sorted(set(annotations) - {"nan_propagate"})
    if unknown_annotations:
        context.record_blocked(
            node_text(owner),
            (
                "tl.tileop.reduce annotations are not imported: "
                + ", ".join(unknown_annotations)
            ),
        )
        return None
    nan_propagate = bool(annotations.get("nan_propagate", False))
    element_text = str(element_type)
    source_dtype = _scalar_dtype_text(dtype(source))
    is_float = element_text in ("f16", "bf16", "f32", "f64")
    is_unsigned = source_dtype.startswith(("uint", "u"))
    if reduce_type == "sum":
        return TileReduceSpec(reduce_type, "addf" if is_float else "addi")
    if reduce_type == "abssum":
        return TileReduceSpec(
            reduce_type,
            "addf" if is_float else "addi",
            absolute=True,
        )
    if reduce_type == "max":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "maximumf" if nan_propagate else "maxnumf",
            )
        return TileReduceSpec(reduce_type, "maxui" if is_unsigned else "maxsi")
    if reduce_type == "absmax":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "maximumf" if nan_propagate else "maxnumf",
                absolute=True,
            )
        return TileReduceSpec(
            reduce_type,
            "maxui" if is_unsigned else "maxsi",
            absolute=True,
        )
    if reduce_type == "min":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "minimumf" if nan_propagate else "minnumf",
            )
        return TileReduceSpec(reduce_type, "minui" if is_unsigned else "minsi")
    if reduce_type == "bitand" and not is_float:
        return TileReduceSpec(reduce_type, "andi")
    if reduce_type == "bitor" and not is_float:
        return TileReduceSpec(reduce_type, "ori")
    if reduce_type == "bitxor" and not is_float:
        return TileReduceSpec(reduce_type, "xori")
    context.record_blocked(
        node_text(owner),
        f"tl.tileop.reduce kind `{reduce_type}` is not imported",
    )
    return None


def _reducer_reduce_spec(
    operation: str,
    source_dtype: str | None,
    element_type: Type,
) -> TileReduceSpec | None:
    element_text = str(element_type)
    is_float = _is_float_type(element_type)
    is_unsigned = source_dtype is not None and _scalar_dtype_text(
        source_dtype
    ).startswith(("uint", "u"))
    if operation == "sum":
        return TileReduceSpec(operation, "addf" if is_float else "addi")
    if operation == "max":
        if is_float:
            return TileReduceSpec(operation, "maxnumf")
        if _integer_bit_width(element_text) is not None:
            return TileReduceSpec(operation, "maxui" if is_unsigned else "maxsi")
    if operation == "min":
        if is_float:
            return TileReduceSpec(operation, "minnumf")
        if _integer_bit_width(element_text) is not None:
            return TileReduceSpec(operation, "minui" if is_unsigned else "minsi")
    return None


def _reducer_source_dtype(
    data_source: object,
    context: TileLangConversionContext,
) -> str | None:
    buffer = context.mapped_buffer_for_data(data_source)
    return None if buffer is None else dtype(buffer)


def _reduce_init(
    clear: bool,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    target: TileLangBufferAccess,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if not clear:
        return _load_tracked_local_value(
            target.view,
            target.indices,
            target.memory_scope,
            element_type,
            context,
            name="reduce_init",
        )
    identity = _reduce_identity(reduce_spec, element_type)
    if identity is None:
        context.record_blocked(
            node_text(owner),
            (
                f"tl.tileop.reduce<{reduce_spec.source_kind}> identity "
                f"is not imported for {element_type}"
            ),
        )
        return None
    return context.ensure_typed_constant(
        identity,
        element_type,
        "identity",
    )


def _reduce_identity(
    reduce_spec: TileReduceSpec,
    element_type: Type,
) -> int | float | None:
    is_float = _is_float_type(element_type)
    if reduce_spec.source_kind == "absmax":
        return 0.0 if is_float else 0
    kind = reduce_spec.combiner
    element_text = str(element_type)
    bit_width = _integer_bit_width(element_text)
    if kind == "addf":
        return 0.0
    if kind in ("addi", "ori", "xori"):
        return 0
    if kind == "mulf":
        return 1.0
    if kind == "muli":
        return 1
    if kind == "andi":
        return -1
    if kind in ("maxnumf", "maximumf"):
        return float("-inf")
    if kind in ("minnumf", "minimumf"):
        return float("inf")
    if kind == "maxui":
        return 0
    if kind == "minui":
        return -1
    if bit_width is None:
        return None
    if kind == "maxsi":
        return -(1 << (bit_width - 1))
    if kind == "minsi":
        return (1 << (bit_width - 1)) - 1
    return None


def _integer_bit_width(element_text: str) -> int | None:
    if not element_text.startswith("i") or not element_text[1:].isdecimal():
        return None
    return int(element_text[1:])


def _floating_point_bit_width(element_text: str) -> int:
    if element_text == "bf16":
        return 16
    if element_text.startswith("f") and element_text[1:].isdecimal():
        return int(element_text[1:])
    return 0


def _is_float_type(value_type: Type) -> bool:
    return str(value_type) in ("f16", "bf16", "f32", "f64")


def _scalar_dtype_text(source_dtype: str) -> str:
    head, separator, tail = source_dtype.rpartition("x")
    if separator and tail.isdecimal():
        return head
    return source_dtype


def _call_annotations(call: object) -> dict[str, object]:
    annotations = getattr(call, "annotations", None)
    return {str(key): value for key, value in mapping_items(annotations)}


def _static_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    integer = integer_value(value)
    if integer in (0, 1):
        return bool(integer)
    return None


def _field_bool(value: object) -> str:
    boolean = _static_bool(value)
    if boolean is None:
        return "dynamic"
    return "true" if boolean else "false"


def _field_integer(value: object) -> str:
    integer = integer_value(value)
    if integer is None:
        return "dynamic"
    return str(integer)


def _region_source_name(value: object) -> str:
    if _call_op_name(value) == "tl.tileop.region":
        args = _args(value)
        if args:
            return _region_source_name(args[0])
    if node_kind(value) == "BufferLoad":
        buffer = getattr(value, "buffer", None)
        name = getattr(buffer, "name", None)
        if name:
            return str(name)
        data = getattr(buffer, "data", None)
        data_name = getattr(data, "name", None)
        if data_name:
            return str(data_name)
    return node_kind(value)


def _convert_fill_value(
    source: object,
    element_type: Type,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    payload = getattr(source, "value", source)
    if isinstance(payload, int | float | bool):
        return context.ensure_typed_constant(
            payload,
            element_type,
            base="const",
        )
    return converter.convert_expr(source, context)


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


def _args(call: object) -> tuple[object, ...]:
    return tuple(getattr(call, "args", ()))


_REGION_CALLS = {
    "tl.tileop.region",
}

_COPY_CALLS = {
    "tl.tileop.copy",
}

_COPY_ANNOTATIONS = frozenset(
    (
        "barrier",
        "coalesced_width",
        "disable_tma",
        "emit_arrive",
        "eviction_policy",
        "force_cp_async",
        "is_async_copy",
        "is_tma_copy",
        "no_implicit_async_commit_wait",
        "parallel_loop_layout",
        "tl.pipeline_mbar_phase_expr",
        "use_2cta",
    )
)

_FILL_CALLS = {
    "tl.tileop.fill",
}

_REDUCE_CALLS = {
    "tl.tileop.reduce",
}

_CUMSUM_CALLS = {
    "tl.tileop.cumsum",
}

_FINALIZE_REDUCER_CALLS = {
    "tl.tileop.finalize_reducer",
}

_GEMM_CALLS = {
    "tl.tileop.gemm",
}

_REDUCER_OPERATION_CODES = {
    "sum": 0,
    "max": 1,
    "min": 2,
}
