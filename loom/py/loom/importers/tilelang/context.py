# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang-specific import session state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loom.builder import ValueRef
from loom.importers.core import SourceImportSession, source_key
from loom.importers.tilelang.nodes import dtype, node_kind, source_name
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import (
    ENCODING_LAYOUT_TYPE,
    ENCODING_SCHEMA_TYPE,
    ENCODING_STORAGE_TYPE,
    INDEX,
    OFFSET,
    DynamicEncoding,
    EncodingInstance,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    Type,
)


@dataclass(frozen=True, slots=True)
class TileLangDistributedIndex:
    """A logical TileLang parallel index lane within one workitem chunk."""

    base: ValueRef
    lane: int
    lane_count: int


@dataclass(frozen=True, slots=True)
class TileLangFragmentVector:
    """Current SSA vector payload backing one TileLang local.fragment view."""

    value: ValueRef
    lane_count: int
    base: ValueRef | None = None


@dataclass(frozen=True, slots=True)
class TileLangMatrixFragment:
    """Current SSA matrix fragment payload backing one TileLang local.fragment view."""

    value: ValueRef
    rows: ValueRef
    columns: ValueRef
    element_type: ScalarType


@dataclass(frozen=True, slots=True)
class TileLangBufferAccessKey:
    """Current-value key for a TileLang local buffer access."""

    view_id: int
    index_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TileLangReducerInfo:
    """Reducer metadata captured from TileLang reducer_info annotations."""

    operation: str
    replication: str


@dataclass(frozen=True, slots=True)
class TileLangAddressLayoutPreference:
    """Preferred physical address layout for a TileLang buffer view."""

    strides: tuple[int, ...]
    name: str


@dataclass(slots=True)
class TileLangConversionContext(SourceImportSession):
    """TileLang-specialized import session using Loom dynamic builders."""

    type_converter: TileLangTypeConverter = field(default_factory=TileLangTypeConverter)
    target_preset: str = "tilelang.generic"
    float_fastmath_flags: str | None = None
    index_values: dict[object, ValueRef] = field(default_factory=dict)
    semantic_values: dict[tuple[object, ...], ValueRef] = field(default_factory=dict)
    semantic_value_types: dict[tuple[object, ...], str] = field(default_factory=dict)
    buffer_data_values: dict[tuple[object, ...], ValueRef] = field(default_factory=dict)
    buffer_data_buffers: dict[tuple[object, ...], object] = field(default_factory=dict)
    semantic_index_values: dict[tuple[object, ...], ValueRef] = field(
        default_factory=dict
    )
    reducer_infos: dict[object, TileLangReducerInfo] = field(default_factory=dict)
    topology_values: dict[str, ValueRef] = field(default_factory=dict)
    topology_extent_values: dict[str, ValueRef] = field(default_factory=dict)
    static_topology_extents: dict[str, int] = field(default_factory=dict)
    distributed_indices: dict[int, TileLangDistributedIndex] = field(
        default_factory=dict
    )
    fragment_vectors: dict[int, TileLangFragmentVector] = field(default_factory=dict)
    matrix_fragments: dict[int, TileLangMatrixFragment] = field(default_factory=dict)
    buffer_access_values: dict[TileLangBufferAccessKey, ValueRef] = field(
        default_factory=dict
    )
    buffer_access_source_keys: dict[object, TileLangBufferAccessKey] = field(
        default_factory=dict
    )
    address_layout_preferences: dict[object, TileLangAddressLayoutPreference] = field(
        default_factory=dict
    )
    address_layout_values: dict[object, ValueRef] = field(default_factory=dict)
    storage_schema_values: dict[EncodingInstance, ValueRef] = field(
        default_factory=dict
    )
    physical_storage_values: dict[tuple[int, EncodingInstance], ValueRef] = field(
        default_factory=dict
    )
    named_buffer_values: dict[str, ValueRef] = field(default_factory=dict)
    named_buffer_sources: dict[str, object] = field(default_factory=dict)
    kernel_body_block: object | None = None
    dense_layout: ValueRef | None = None
    pending_workgroup_memory_write: bool = False

    def type(self, value_type: str) -> Type:
        return self.type_converter.map_dtype(value_type)

    def default_address_layout(self) -> ValueRef:
        if self.dense_layout is None:
            self.dense_layout = self.builder.encoding.layout_dense(
                results=[ENCODING_LAYOUT_TYPE],
                name=self.reserve_name("layout"),
            )
        return self.dense_layout

    def buffer_view_type(self, buffer: object) -> ShapedType:
        view_type = self.type_converter.view_type(buffer)
        if view_type.encoding is not None:
            return view_type
        self.storage_encoding_for_buffer(buffer)
        return ShapedType(
            view_type.type_kind,
            view_type.element_type,
            view_type.dims,
            encoding=DynamicEncoding(),
        )

    def storage_encoding_for_buffer(self, buffer: object | None) -> ValueRef:
        schema = (
            self.type_converter.buffer_storage_schema(buffer)
            if buffer is not None
            else None
        )
        if schema is None:
            return self.address_layout_for_buffer(buffer)
        layout = self.address_layout_for_buffer(buffer)
        key = (layout.id, schema)
        storage = self.physical_storage_values.get(key)
        if storage is None:
            schema_value = self.storage_schema_value(schema)
            storage = self.builder.encoding.define(
                spec=EncodingInstance(name="physical_storage"),
                params={"layout": layout, "schema": schema_value},
                results=[ENCODING_STORAGE_TYPE],
                name=self.reserve_name(_storage_encoding_name(buffer)),
            )
            self.physical_storage_values[key] = storage
        return storage

    def storage_schema_value(self, schema: EncodingInstance) -> ValueRef:
        schema_value = self.storage_schema_values.get(schema)
        if schema_value is None:
            schema_value = self.builder.encoding.define(
                spec=schema,
                results=[ENCODING_SCHEMA_TYPE],
                name=self.reserve_name(f"{schema.name}_schema"),
            )
            self.storage_schema_values[schema] = schema_value
        return schema_value

    def address_layout_for_buffer(self, buffer: object | None) -> ValueRef:
        preference = self.address_layout_preference(buffer)
        if preference is None:
            return self.default_address_layout()
        if buffer is None:
            return self.default_address_layout()
        key = address_layout_keys(buffer)[0]
        layout = self.address_layout_values.get(key)
        if layout is None:
            layout = self.builder.encoding.layout_strided(
                strides=list(preference.strides),
                results=[ENCODING_LAYOUT_TYPE],
                name=self.reserve_name(preference.name),
            )
            self.address_layout_values[key] = layout
        return layout

    def address_layout_preference(
        self,
        buffer: object | None,
    ) -> TileLangAddressLayoutPreference | None:
        if buffer is None:
            return None
        for key in address_layout_keys(buffer):
            preference = self.address_layout_preferences.get(key)
            if preference is not None:
                return preference
        return None

    def bind_buffer_view_layout(
        self,
        view: ValueRef,
        buffer: object | None = None,
    ) -> None:
        view_value = self.builder.module.values[view.id]
        view_type = view_value.type
        if isinstance(view_type, ShapedType) and isinstance(
            view_type.encoding, DynamicEncoding
        ):
            view_value.encoding_binding = self.storage_encoding_for_buffer(buffer).id

    def float_operation_kwargs(self) -> dict[str, str]:
        if self.float_fastmath_flags is None:
            return {}
        return {"fastmath": self.float_fastmath_flags}

    def build_constant(self, value: Any, value_type: str, name: str) -> ValueRef:
        result_type = self.type_converter.map_dtype(
            value_type,
            index_like=value_type == "index",
        )
        return self.build_typed_constant(value, result_type, name)

    def build_typed_constant(
        self, value: Any, result_type: Type, name: str
    ) -> ValueRef:
        value = _typed_constant_value(value, result_type)
        if result_type in (INDEX, OFFSET):
            return self.builder.index.constant(
                value=value,
                results=[result_type],
                name=name,
            )
        return self.builder.scalar.constant(
            value=value,
            results=[result_type],
            name=name,
        )

    def ensure_typed_constant(
        self,
        value: Any,
        result_type: Type,
        base: str,
    ) -> ValueRef:
        value = _typed_constant_value(value, result_type)
        key = (str(result_type), str(value))
        existing = self.constants.get(key)
        if existing is not None:
            return existing
        result = self.build_typed_constant(value, result_type, self.reserve_name(base))
        self.constants[key] = result
        return result

    def map_value(
        self,
        source: object,
        ref: ValueRef,
        value_type: str | None = None,
    ) -> None:
        SourceImportSession.map_value(self, source, ref, value_type)
        semantic_key = _semantic_source_key(source)
        if semantic_key is not None:
            self.semantic_values[semantic_key] = ref
            self.semantic_value_types[semantic_key] = (
                value_type if value_type is not None else str(ref.type)
            )

    def mapped(self, source: object) -> ValueRef | None:
        mapped_value = SourceImportSession.mapped(self, source)
        if mapped_value is not None:
            return mapped_value
        semantic_key = _semantic_source_key(source)
        if semantic_key is None:
            return None
        return self.semantic_values.get(semantic_key)

    def mapped_value_type(self, source: object) -> str | None:
        mapped_type = SourceImportSession.mapped_value_type(self, source)
        if mapped_type is not None:
            return mapped_type
        semantic_key = _semantic_source_key(source)
        if semantic_key is None:
            return None
        return self.semantic_value_types.get(semantic_key)

    def map_buffer_data(
        self,
        source: object,
        ref: ValueRef,
        *,
        buffer: object | None = None,
    ) -> None:
        self.map_value(source, ref, str(ref.type))
        semantic_key = _semantic_buffer_data_key(source)
        if semantic_key is not None:
            self.buffer_data_values[semantic_key] = ref
            if buffer is not None:
                self.buffer_data_buffers[semantic_key] = buffer

    def mapped_buffer_data(self, source: object) -> ValueRef | None:
        semantic_key = _semantic_buffer_data_key(source)
        if semantic_key is not None:
            mapped_data = self.buffer_data_values.get(semantic_key)
            if mapped_data is not None:
                return mapped_data
        return self.mapped(source)

    def mapped_buffer_for_data(self, source: object) -> object | None:
        semantic_key = _semantic_buffer_data_key(source)
        if semantic_key is None:
            return None
        return self.buffer_data_buffers.get(semantic_key)

    def map_reducer_info(
        self,
        source: object,
        reducer_info: TileLangReducerInfo,
    ) -> None:
        self.reducer_infos[_reducer_info_key(source)] = reducer_info

    def reducer_info(self, source: object) -> TileLangReducerInfo | None:
        return self.reducer_infos.get(_reducer_info_key(source))

    def topology_value(self, source_tag: str) -> ValueRef | None:
        return self.topology_values.get(source_tag)

    def map_topology_value(self, source_tag: str, ref: ValueRef) -> None:
        self.topology_values[source_tag] = ref

    def map_topology_extent(
        self,
        source_tag: str,
        extent: ValueRef | None,
        *,
        static_extent: int | None = None,
    ) -> None:
        if extent is not None:
            self.topology_extent_values[source_tag] = extent
        if static_extent is not None:
            self.static_topology_extents[source_tag] = static_extent

    def topology_extent(self, source_tag: str) -> ValueRef | None:
        return self.topology_extent_values.get(source_tag)

    def static_topology_extent(self, source_tag: str) -> int | None:
        return self.static_topology_extents.get(source_tag)

    def map_distributed_index(
        self,
        ref: ValueRef,
        info: TileLangDistributedIndex,
    ) -> None:
        self.distributed_indices[ref.id] = info

    def distributed_index(self, ref: ValueRef) -> TileLangDistributedIndex | None:
        return self.distributed_indices.get(ref.id)

    def map_fragment_vector(
        self,
        view: ValueRef,
        vector: TileLangFragmentVector,
    ) -> None:
        self.fragment_vectors[view.id] = vector

    def fragment_vector(self, view: ValueRef) -> TileLangFragmentVector | None:
        return self.fragment_vectors.get(view.id)

    def clear_fragment_vector(self, view: ValueRef) -> None:
        self.fragment_vectors.pop(view.id, None)

    def map_matrix_fragment(
        self,
        view: ValueRef,
        fragment: TileLangMatrixFragment,
    ) -> None:
        self.matrix_fragments[view.id] = fragment

    def matrix_fragment(self, view: ValueRef) -> TileLangMatrixFragment | None:
        return self.matrix_fragments.get(view.id)

    def clear_matrix_fragment(self, view: ValueRef) -> None:
        self.matrix_fragments.pop(view.id, None)

    def buffer_access_key(
        self,
        view: ValueRef,
        indices: tuple[ValueRef, ...],
        memory_scope: str,
    ) -> TileLangBufferAccessKey | None:
        if memory_scope not in _TRACKED_LOCAL_BUFFER_SCOPES:
            return None
        return TileLangBufferAccessKey(
            view_id=view.id,
            index_ids=tuple(index.id for index in indices),
        )

    def mapped_buffer_access(
        self,
        view: ValueRef,
        indices: tuple[ValueRef, ...],
        memory_scope: str,
    ) -> ValueRef | None:
        key = self.buffer_access_key(view, indices, memory_scope)
        if key is None:
            return None
        return self.buffer_access_values.get(key)

    def map_buffer_access(
        self,
        source: object,
        view: ValueRef,
        indices: tuple[ValueRef, ...],
        memory_scope: str,
        ref: ValueRef,
    ) -> None:
        key = self.buffer_access_key(view, indices, memory_scope)
        if key is None:
            return
        self.buffer_access_values[key] = ref
        self.buffer_access_source_keys[self.source_key(source)] = key

    def map_buffer_access_key(
        self,
        key: TileLangBufferAccessKey,
        ref: ValueRef,
    ) -> None:
        self.buffer_access_values[key] = ref

    def map_named_buffer(
        self,
        buffer: object,
        ref: ValueRef,
    ) -> None:
        name = source_name(buffer, fallback="")
        if not name:
            return
        self.named_buffer_values[name] = ref
        self.named_buffer_sources[name] = buffer

    def mapped_named_buffer(
        self,
        source: object,
    ) -> tuple[ValueRef, object] | None:
        name = source_name(source, fallback="")
        if not name:
            return None
        ref = self.named_buffer_values.get(name)
        source_buffer = self.named_buffer_sources.get(name)
        if ref is None or source_buffer is None:
            return None
        return ref, source_buffer

    def buffer_access_source_key(
        self,
        source: object,
    ) -> TileLangBufferAccessKey | None:
        return self.buffer_access_source_keys.get(self.source_key(source))

    def invalidate_buffer_accesses(self, view: ValueRef) -> None:
        stale_keys = [
            key for key in self.buffer_access_values if key.view_id == view.id
        ]
        for key in stale_keys:
            self.buffer_access_values.pop(key, None)
        stale_source_keys = [
            source_lookup_key
            for source_lookup_key, key in self.buffer_access_source_keys.items()
            if key.view_id == view.id
        ]
        for source_lookup_key in stale_source_keys:
            self.buffer_access_source_keys.pop(source_lookup_key, None)

    def mapped_index_value(self, source: object) -> ValueRef | None:
        mapped_value = self.index_values.get(self.source_key(source))
        if mapped_value is not None:
            return mapped_value
        semantic_key = _semantic_var_key(source)
        if semantic_key is None:
            return None
        return self.semantic_index_values.get(semantic_key)

    def map_index_value(self, source: object, ref: ValueRef) -> None:
        self.index_values[self.source_key(source)] = ref
        semantic_key = _semantic_var_key(source)
        if semantic_key is not None:
            self.semantic_index_values[semantic_key] = ref
        if ref.name:
            self.names.capture(ref.name)

    @contextmanager
    def scoped_index_value(
        self,
        source: object,
        ref: ValueRef,
    ) -> Iterator[None]:
        """Temporarily bind a source index variable while importing an inline body."""

        source_lookup_key = self.source_key(source)
        semantic_key = _semantic_var_key(source)
        old_value = self.value_map.get(source_lookup_key)
        old_value_type = self.value_types.get(source_lookup_key)
        old_index = self.index_values.get(source_lookup_key)
        old_semantic_value = (
            self.semantic_values.get(semantic_key) if semantic_key is not None else None
        )
        old_semantic_value_type = (
            self.semantic_value_types.get(semantic_key)
            if semantic_key is not None
            else None
        )
        old_semantic_index = (
            self.semantic_index_values.get(semantic_key)
            if semantic_key is not None
            else None
        )
        had_value = source_lookup_key in self.value_map
        had_value_type = source_lookup_key in self.value_types
        had_index = source_lookup_key in self.index_values
        had_semantic_value = (
            semantic_key is not None and semantic_key in self.semantic_values
        )
        had_semantic_value_type = (
            semantic_key is not None and semantic_key in self.semantic_value_types
        )
        had_semantic_index = (
            semantic_key is not None and semantic_key in self.semantic_index_values
        )

        self.map_value(source, ref, "index")
        self.map_index_value(source, ref)
        try:
            yield
        finally:
            _restore_optional_mapping(
                self.value_map,
                source_lookup_key,
                had_value,
                old_value,
            )
            _restore_optional_mapping(
                self.value_types,
                source_lookup_key,
                had_value_type,
                old_value_type,
            )
            _restore_optional_mapping(
                self.index_values,
                source_lookup_key,
                had_index,
                old_index,
            )
            if semantic_key is not None:
                _restore_optional_mapping(
                    self.semantic_values,
                    semantic_key,
                    had_semantic_value,
                    old_semantic_value,
                )
                _restore_optional_mapping(
                    self.semantic_value_types,
                    semantic_key,
                    had_semantic_value_type,
                    old_semantic_value_type,
                )
                _restore_optional_mapping(
                    self.semantic_index_values,
                    semantic_key,
                    had_semantic_index,
                    old_semantic_index,
                )

    def can_emit_kernel_exit(self) -> bool:
        return (
            self.kernel_body_block is not None
            and self.builder.ir.insertion_block is self.kernel_body_block
        )

    def mark_pending_workgroup_memory_write(self) -> None:
        if not self.may_have_multiple_workitems():
            return
        self.pending_workgroup_memory_write = True

    def clear_pending_workgroup_memory_write(self) -> None:
        self.pending_workgroup_memory_write = False

    def flush_pending_workgroup_memory_barrier(self) -> bool:
        if not self.pending_workgroup_memory_write:
            return False
        self.builder.kernel.barrier(
            memory_space="workgroup",
            ordering="acq_rel",
            scope="workgroup",
        )
        self.clear_pending_workgroup_memory_write()
        return True

    def may_have_multiple_workitems(self) -> bool:
        static_count = 1
        for axis in ("threadIdx.x", "threadIdx.y", "threadIdx.z"):
            extent = self.static_topology_extent(axis)
            if extent is None:
                return True
            static_count *= extent
        return static_count > 1

    def fork(self, *, preview_block: object | None = None) -> TileLangConversionContext:
        return TileLangConversionContext(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
            type_converter=self.type_converter,
            target_preset=self.target_preset,
            float_fastmath_flags=self.float_fastmath_flags,
            index_values=dict(self.index_values),
            semantic_values=dict(self.semantic_values),
            semantic_value_types=dict(self.semantic_value_types),
            buffer_data_values=dict(self.buffer_data_values),
            buffer_data_buffers=dict(self.buffer_data_buffers),
            semantic_index_values=dict(self.semantic_index_values),
            reducer_infos=dict(self.reducer_infos),
            topology_values=dict(self.topology_values),
            topology_extent_values=dict(self.topology_extent_values),
            static_topology_extents=dict(self.static_topology_extents),
            distributed_indices=dict(self.distributed_indices),
            fragment_vectors=dict(self.fragment_vectors),
            matrix_fragments=dict(self.matrix_fragments),
            buffer_access_values=dict(self.buffer_access_values),
            buffer_access_source_keys=dict(self.buffer_access_source_keys),
            address_layout_preferences=dict(self.address_layout_preferences),
            address_layout_values=dict(self.address_layout_values),
            storage_schema_values=dict(self.storage_schema_values),
            physical_storage_values=dict(self.physical_storage_values),
            named_buffer_values=dict(self.named_buffer_values),
            named_buffer_sources=dict(self.named_buffer_sources),
            kernel_body_block=self.kernel_body_block,
            dense_layout=self.dense_layout,
            pending_workgroup_memory_write=self.pending_workgroup_memory_write,
        )

    def merge_child_records(self, child: SourceImportSession) -> None:
        SourceImportSession.merge_child_records(self, child)
        if isinstance(child, TileLangConversionContext):
            self.storage_schema_values.update(child.storage_schema_values)
            self.physical_storage_values.update(child.physical_storage_values)
            self.pending_workgroup_memory_write = (
                self.pending_workgroup_memory_write
                or child.pending_workgroup_memory_write
            )


_TRACKED_LOCAL_BUFFER_SCOPES = frozenset(("local", "local.fragment", "local.var"))


def _semantic_source_key(source: object) -> tuple[object, ...] | None:
    var_key = _semantic_var_key(source)
    if var_key is not None:
        return var_key
    return _semantic_buffer_key(source)


def _semantic_var_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) not in ("Var", "SizeVar"):
        return None
    source_dtype = dtype(source)
    if source_dtype == "handle" or source_dtype.endswith("*"):
        return None
    return ("var", source_name(source, fallback=str(source)), source_dtype)


def _semantic_buffer_data_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) not in ("Var", "SizeVar"):
        return None
    source_dtype = dtype(source)
    if source_dtype != "handle" and not source_dtype.endswith("*"):
        return None
    return ("buffer_data", source_name(source, fallback=str(source)), source_dtype)


def _semantic_buffer_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) != "Buffer":
        return None
    data = getattr(source, "data", None)
    return (
        "buffer",
        source_name(source, fallback=str(source)),
        dtype(source),
        _buffer_scope(source),
        source_name(data, fallback=str(data)),
        _semantic_sequence(getattr(source, "shape", ())),
    )


def _reducer_info_key(source: object) -> tuple[object, ...]:
    semantic_key = _semantic_buffer_data_key(source)
    if semantic_key is not None:
        return ("reducer_info", *semantic_key)
    semantic_key = _semantic_buffer_key(source)
    if semantic_key is not None:
        return ("reducer_info", *semantic_key)
    return ("reducer_info", source_key(source))


def address_layout_keys(buffer: object) -> tuple[object, ...]:
    """Return lookup keys that identify one TileLang buffer's address layout."""

    semantic_key = _semantic_buffer_key(buffer)
    if semantic_key is not None:
        return (semantic_key, source_key(buffer))
    return (source_key(buffer),)


def _storage_encoding_name(buffer: object | None) -> str:
    if buffer is None:
        return "storage"
    return f"{source_name(buffer, fallback='buffer')}_storage"


def _semantic_sequence(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, str | bytes):
        return ()
    if isinstance(value, Iterable):
        return tuple(_semantic_value(item) for item in value)
    return (_semantic_value(value),)


def _semantic_value(value: object) -> object:
    if isinstance(value, str | bytes | int | float | bool):
        return value
    if node_kind(value) in ("Var", "SizeVar"):
        return _semantic_var_key(value)
    if hasattr(value, "value"):
        return ("imm", str(value.value), dtype(value))
    return (
        node_kind(value),
        source_name(value, fallback=str(value)),
        dtype(value),
        str(value),
    )


def _integer_constant_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == "true":
            return 1
        if lower_value == "false":
            return 0
        return int(value, 0)
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"integer constant value must be integer-like, got {value!r}")


def _float_constant_value(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"floating-point constant value must be float-like, got {value!r}")


def _typed_constant_value(value: object, result_type: Type) -> Any:
    if result_type in (INDEX, OFFSET) or _is_integer_scalar_type(result_type):
        return _integer_constant_value(value)
    if _is_float_scalar_type(result_type):
        return _float_constant_value(value)
    return value


def _is_integer_scalar_type(value_type: Type) -> bool:
    return isinstance(value_type, ScalarType) and value_type.kind in (
        ScalarTypeKind.I1,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    )


def _is_float_scalar_type(value_type: Type) -> bool:
    return isinstance(value_type, ScalarType) and value_type.kind in (
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    )


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""


def _restore_optional_mapping(
    mapping: dict[Any, Any],
    key: Any,
    had_key: bool,
    old_value: Any,
) -> None:
    if had_key:
        mapping[key] = old_value
    else:
        mapping.pop(key, None)
