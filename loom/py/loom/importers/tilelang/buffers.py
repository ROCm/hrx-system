# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang buffer/view access mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.nodes import node_text
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import INDEX, DynamicDim, ShapedType, StaticDim


@dataclass(frozen=True, slots=True)
class TileLangBufferIndexMap:
    """Maps logical TileLang buffer coordinates to a physical Loom view."""

    view: ValueRef
    memory_scope: str
    view_rank: int
    logical_rank_value: int
    logical_extents: tuple[ValueRef, ...] | None
    physical_extents: tuple[ValueRef, ...] | None
    identity: bool

    @property
    def logical_rank(self) -> int:
        return self.logical_rank_value

    @property
    def physical_rank(self) -> int:
        return self.view_rank

    @property
    def is_identity(self) -> bool:
        return self.identity


@dataclass(frozen=True, slots=True)
class TileLangBufferAccess:
    """One physical access into a Loom view."""

    view: ValueRef
    indices: tuple[ValueRef, ...]
    memory_scope: str = ""


@dataclass(frozen=True, slots=True)
class TileLangBufferRegion:
    """One logical TileLang buffer region over a physical Loom view."""

    view: ValueRef
    indices: tuple[ValueRef, ...]
    extents: tuple[ValueRef, ...]
    index_map: TileLangBufferIndexMap


def resolve_buffer_access(
    buffer: object,
    source_indices: Sequence[object | ValueRef],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> TileLangBufferAccess | None:
    """Resolve a scalar TileLang buffer access to physical Loom view indices."""

    index_map = resolve_buffer_index_map(
        buffer,
        context,
        converter,
        diagnostic_owner=diagnostic_owner,
    )
    if index_map is None:
        return None
    logical_indices = convert_index_sequence(
        source_indices,
        diagnostic_owner,
        context,
        converter,
        what="buffer indices",
    )
    if logical_indices is None:
        return None
    if len(logical_indices) != index_map.logical_rank:
        context.record_blocked(
            node_text(diagnostic_owner),
            (
                "buffer access logical rank "
                f"{len(logical_indices)} does not match buffer rank "
                f"{index_map.logical_rank}"
            ),
        )
        return None
    physical_indices = map_logical_indices(index_map, logical_indices, context)
    return TileLangBufferAccess(
        view=index_map.view,
        indices=physical_indices,
        memory_scope=index_map.memory_scope,
    )


def resolve_buffer_region(
    buffer: object,
    source_indices: Sequence[object | ValueRef],
    source_extents: Sequence[object | ValueRef],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> TileLangBufferRegion | None:
    """Resolve a TileLang buffer region while preserving logical region rank."""

    index_map = resolve_buffer_index_map(
        buffer,
        context,
        converter,
        diagnostic_owner=diagnostic_owner,
    )
    if index_map is None:
        return None
    indices = convert_index_sequence(
        source_indices,
        diagnostic_owner,
        context,
        converter,
        what="region indices",
    )
    extents = convert_index_sequence(
        source_extents,
        diagnostic_owner,
        context,
        converter,
        what="region extents",
    )
    if indices is None or extents is None:
        return None
    if len(indices) != len(extents):
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer region index and extent ranks differ",
        )
        return None
    if len(indices) != index_map.logical_rank:
        context.record_blocked(
            node_text(diagnostic_owner),
            (
                "buffer access logical rank "
                f"{len(indices)} does not match buffer rank {index_map.logical_rank}"
            ),
        )
        return None
    return TileLangBufferRegion(
        view=index_map.view,
        indices=tuple(indices),
        extents=tuple(extents),
        index_map=index_map,
    )


def resolve_buffer_index_map(
    buffer: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> TileLangBufferIndexMap | None:
    """Resolve the physical view and logical-to-physical map for `buffer`."""

    view = context.mapped(buffer)
    if view is not None:
        view_rank = _physical_view_rank(
            view,
            context,
            diagnostic_owner=diagnostic_owner,
        )
        if view_rank is None:
            return None
        return TileLangBufferIndexMap(
            view=view,
            memory_scope=_buffer_scope(buffer),
            view_rank=view_rank,
            logical_rank_value=view_rank,
            logical_extents=None,
            physical_extents=None,
            identity=True,
        )
    data = getattr(buffer, "data", None)
    view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target is not mapped",
        )
        return None
    view_rank = _physical_view_rank(
        view,
        context,
        diagnostic_owner=diagnostic_owner,
    )
    if view_rank is None:
        return None
    logical_shape = tuple(getattr(buffer, "shape", ()) or ())
    if not logical_shape:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer alias has no logical shape",
        )
        return None
    logical_rank = len(logical_shape)
    physical_extents = None
    if view_rank > 1:
        physical_extents = _physical_view_extents(
            view,
            context,
            diagnostic_owner=diagnostic_owner,
        )
        if physical_extents is None:
            return None
    logical_extents = None
    if logical_rank > 1:
        logical_extents = _logical_buffer_extents(
            logical_shape,
            context,
            converter,
            diagnostic_owner=diagnostic_owner,
        )
        if logical_extents is None:
            return None
    if not _validate_alias_shape(
        logical_shape,
        view_rank,
        view,
        context,
        diagnostic_owner=diagnostic_owner,
    ):
        return None
    return TileLangBufferIndexMap(
        view=view,
        memory_scope=_buffer_scope(buffer),
        view_rank=view_rank,
        logical_rank_value=logical_rank,
        logical_extents=logical_extents,
        physical_extents=physical_extents,
        identity=False,
    )


def convert_index_sequence(
    values: Sequence[object | ValueRef],
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    what: str,
) -> tuple[ValueRef, ...] | None:
    """Convert TileLang index expressions to Loom index values."""

    converted: list[ValueRef] = []
    for value in values:
        index: ValueRef | None
        if isinstance(value, ValueRef):
            index = value
        else:
            index = converter.convert_expr(value, context, index_like=True)
        if index is None:
            context.record_blocked(node_text(owner), f"{what} are not mapped")
            return None
        if str(index.type) != "index":
            context.record_blocked(node_text(owner), f"{what} must be index values")
            return None
        converted.append(index)
    return tuple(converted)


def map_region_indices(
    region: TileLangBufferRegion,
    offsets: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    """Map logical region offsets to physical Loom view coordinates."""

    logical_indices = tuple(
        add_index(base, offset, context)
        for base, offset in zip(region.indices, offsets, strict=True)
    )
    return map_logical_indices(region.index_map, logical_indices, context)


def map_logical_indices(
    index_map: TileLangBufferIndexMap,
    logical_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    """Map logical buffer coordinates to physical Loom view coordinates."""

    if index_map.is_identity:
        return logical_indices
    logical_extents = index_map.logical_extents
    physical_extents = index_map.physical_extents
    if (
        logical_extents is not None
        and physical_extents is not None
        and len(logical_indices) == index_map.physical_rank
        and all(
            logical.id == physical.id
            for logical, physical in zip(
                logical_extents, physical_extents, strict=False
            )
        )
    ):
        return logical_indices
    if logical_extents is None:
        flat_index = logical_indices[0]
    else:
        flat_index = _flatten_logical_indices(
            logical_indices,
            logical_extents,
            context,
        )
    if physical_extents is None:
        return (flat_index,)
    return _unflatten_physical_index(flat_index, physical_extents, context)


def add_index(
    base: ValueRef,
    offset: ValueRef,
    context: TileLangConversionContext,
) -> ValueRef:
    """Add two index values, preserving zero identities for readable IR."""

    zero = context.constants.get(("index", "0"))
    if zero is not None:
        if base.id == zero.id:
            return offset
        if offset.id == zero.id:
            return base
    return context.builder.index.add(
        lhs=base,
        rhs=offset,
        results=[INDEX],
        name=context.fresh_name("idx"),
    )


def _flatten_logical_indices(
    logical_indices: tuple[ValueRef, ...],
    logical_extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> ValueRef:
    if len(logical_indices) == 1:
        return logical_indices[0]
    running = logical_indices[0]
    for index, extent in zip(logical_indices[1:], logical_extents[1:], strict=True):
        product = context.builder.index.mul(
            lhs=running,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name("mul"),
        )
        running = context.builder.index.add(
            lhs=product,
            rhs=index,
            results=[INDEX],
            name=context.fresh_name("add"),
        )
    return running


def _unflatten_physical_index(
    flat_index: ValueRef,
    physical_extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    if len(physical_extents) == 1:
        return (flat_index,)
    remapped: list[ValueRef | None] = [None] * len(physical_extents)
    running = flat_index
    for position in range(len(physical_extents) - 1, 0, -1):
        extent = physical_extents[position]
        remapped[position] = context.builder.index.rem(
            lhs=running,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name("rem"),
        )
        running = context.builder.index.div(
            lhs=running,
            rhs=extent,
            results=[INDEX],
            name=context.fresh_name("div"),
        )
    remapped[0] = running
    return tuple(index for index in remapped if index is not None)


def _physical_view_extents(
    view: ValueRef,
    context: TileLangConversionContext,
    *,
    diagnostic_owner: object,
) -> tuple[ValueRef, ...] | None:
    view_value = context.builder.module.values[view.id]
    view_type = view_value.type
    if not isinstance(view_type, ShapedType):
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target is not a shaped view",
        )
        return None
    extents: list[ValueRef] = []
    for position, dim in enumerate(view_type.dims):
        if isinstance(dim, StaticDim):
            extents.append(
                context.ensure_constant(str(dim.size), "index", f"c{dim.size}")
            )
            continue
        if isinstance(dim, DynamicDim):
            value_id = view_value.dim_bindings.get(position)
            if value_id is None:
                context.record_blocked(
                    node_text(diagnostic_owner),
                    "buffer access target dimension is not bound",
                )
                return None
            extents.append(ValueRef(value_id, context.builder.ir))
            continue
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target dimension is not mapped",
        )
        return None
    return tuple(extents)


def _physical_view_rank(
    view: ValueRef,
    context: TileLangConversionContext,
    *,
    diagnostic_owner: object,
) -> int | None:
    view_type = context.builder.module.values[view.id].type
    if not isinstance(view_type, ShapedType):
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target is not a shaped view",
        )
        return None
    return len(view_type.dims)


def _logical_buffer_extents(
    shape: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> tuple[ValueRef, ...] | None:
    return convert_index_sequence(
        shape,
        diagnostic_owner,
        context,
        converter,
        what="buffer alias shape",
    )


def _validate_alias_shape(
    logical_shape: tuple[object, ...],
    view_rank: int,
    view: ValueRef,
    context: TileLangConversionContext,
    *,
    diagnostic_owner: object,
) -> bool:
    logical_product = _static_product(logical_shape)
    physical_product = _static_physical_product(view, context)
    if logical_product is not None and physical_product is not None:
        if logical_product == physical_product:
            return True
        context.record_blocked(
            node_text(diagnostic_owner),
            (
                "buffer alias element count "
                f"{logical_product} does not match source view element count "
                f"{physical_product}"
            ),
        )
        return False
    if len(logical_shape) == 1:
        return True
    if logical_product is not None and view_rank == 1:
        return True
    context.record_blocked(
        node_text(diagnostic_owner),
        "buffer alias reshape product is not statically provable",
    )
    return False


def _static_product(values: tuple[object, ...]) -> int | None:
    product = 1
    for value in values:
        integer = integer_value(value)
        if integer is None:
            return None
        product *= integer
    return product


def _static_physical_product(
    view: ValueRef,
    context: TileLangConversionContext,
) -> int | None:
    view_type = context.builder.module.values[view.id].type
    if not isinstance(view_type, ShapedType):
        return None
    product = 1
    for dim in view_type.dims:
        if not isinstance(dim, StaticDim):
            return None
        product *= dim.size
    return product


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""
