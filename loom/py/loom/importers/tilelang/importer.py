# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Library-form TileLang kernel importer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loom.builder import ValueRef
from loom.diagnostics import DiagnosticEngine
from loom.importers.core import (
    ImportBodyReport,
    ImportOptions,
    ImportResult,
    KernelArgumentSpec,
    KernelConfigArgumentSpec,
    KernelModuleShell,
    KernelModuleSpec,
    create_kernel_module,
    kernel_module_ops,
    normalize_launch_tuple,
    sanitize_symbol,
)
from loom.importers.tilelang.abi import (
    collect_dynamic_scalar_source_groups_from,
    extract_bindings,
    source_binding_keys,
)
from loom.importers.tilelang.analysis import collect_address_layout_preferences
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.defaults import build_default_registry
from loom.importers.tilelang.model import (
    TileLangBinding,
    TileLangImportInput,
    TileLangKernelFacts,
    normalize_tilelang_input,
    resolve_tilelang_input,
)
from loom.importers.tilelang.nodes import (
    attrs,
    mapping_items,
    node_text,
    source_name,
)
from loom.importers.tilelang.ops.topology import collect_thread_extents, integer_value
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import (
    INDEX,
    OFFSET,
    DynamicDim,
    Module,
    ShapedType,
    rebuild_value_metadata,
)
from loom.verify import verify_module

_TILELANG_FASTMATH_FLAGS = "reassoc|nnan|ninf|nsz|arcp|contract|afn"
_TILELANG_FASTMATH_PASS_CONFIG_KEY = "tl.enable_fast_math"
_TILELANG_FASTMATH_COMPILE_FLAG = "--use_fast_math"


@dataclass(frozen=True, slots=True)
class TileLangImportOptions(ImportOptions):
    """Options specific to TileLang kernel imports."""

    kernel: str | None = None
    target_preset: str | None = None


@dataclass(frozen=True, slots=True)
class _LaunchTopology:
    """TileLang launch topology before conversion into Loom SSA."""

    workgroup_count: tuple[object, object, object]
    workgroup_size: tuple[object, object, object]


def import_tilelang(
    source: object,
    *,
    options: TileLangImportOptions | None = None,
) -> ImportResult:
    options = options or TileLangImportOptions()
    diagnostics = DiagnosticEngine()
    normalized = normalize_tilelang_input(source)
    resolved_source = resolve_tilelang_input(normalized).source
    prim_func = _select_prim_func(
        resolved_source,
        options.kernel or normalized.name,
    )
    function_name = _function_name(prim_func, options.kernel or normalized.name)
    target_preset = options.target_preset or normalized.target or "tilelang.generic"
    float_fastmath_flags = _float_fastmath_flags(normalized, prim_func)
    type_converter = TileLangTypeConverter()
    bindings = extract_bindings(prim_func, type_converter)
    launch_topology = _launch_topology(prim_func)
    workgroup_size = _static_workgroup_size(launch_topology)
    loom_module, body_report, operation_counts = _build_loom_module(
        prim_func,
        bindings,
        function_name=function_name,
        target_preset=target_preset,
        launch_topology=launch_topology,
        diagnostics=diagnostics,
        type_converter=type_converter,
        float_fastmath_flags=float_fastmath_flags,
    )
    diagnostics.raise_if_errors()
    if options.verify_structure:
        verify_module(
            loom_module,
            ops=kernel_module_ops(target_preset),
            diagnostics=diagnostics,
        )
        diagnostics.raise_if_errors()
    facts = TileLangKernelFacts(
        function_name=function_name,
        target_preset=target_preset,
        workgroup_size=workgroup_size,
        bindings=bindings,
        operation_counts=operation_counts,
        converted_body=body_report,
    )
    return ImportResult(
        module=loom_module,
        diagnostics=diagnostics,
        report=facts if options.include_report else None,
    )


def import_tilelang_file(
    path: str | Path,
    *,
    options: TileLangImportOptions | None = None,
) -> ImportResult:
    raise ValueError(
        "TileLang imports require structured Python objects; "
        f"cannot import text file {path!s}"
    )


def _select_prim_func(source: object, requested_name: str | None) -> object:
    functions = getattr(source, "functions", None)
    if functions is None:
        return source
    matches: list[tuple[str, object]] = []
    for symbol, function in mapping_items(functions):
        name = source_name(symbol, fallback=str(symbol))
        if requested_name is None or name == requested_name:
            matches.append((name, function))
    if not matches:
        raise ValueError(f"TileLang module has no PrimFunc @{requested_name}")
    if len(matches) > 1:
        names = ", ".join(name for name, _function in matches)
        raise ValueError(f"TileLang module has multiple PrimFuncs: {names}")
    return matches[0][1]


def _function_name(prim_func: object, requested_name: str | None) -> str:
    if requested_name:
        return sanitize_symbol(requested_name, fallback="kernel")
    source_attrs = attrs(prim_func)
    for key in ("global_symbol", "sym_name"):
        if key in source_attrs:
            return sanitize_symbol(str(source_attrs[key]), fallback="kernel")
    return sanitize_symbol(source_name(prim_func, fallback="main"), fallback="kernel")


def _float_fastmath_flags(
    source: TileLangImportInput,
    prim_func: object,
) -> str | None:
    source_attrs = attrs(prim_func)
    if (
        _pass_config_enabled(
            getattr(source.source, "pass_configs", None),
            _TILELANG_FASTMATH_PASS_CONFIG_KEY,
        )
        or _pass_config_enabled(
            source.metadata.get("tilelang_pass_configs"),
            _TILELANG_FASTMATH_PASS_CONFIG_KEY,
        )
        or _pass_config_enabled(
            source_attrs.get("tilelang_pass_configs"),
            _TILELANG_FASTMATH_PASS_CONFIG_KEY,
        )
    ):
        return _TILELANG_FASTMATH_FLAGS
    if _compile_flag_enabled(
        source.metadata.get("tilelang_compile_flags"),
        _TILELANG_FASTMATH_COMPILE_FLAG,
    ) or _compile_flag_enabled(
        source_attrs.get("tilelang_compile_flags"),
        _TILELANG_FASTMATH_COMPILE_FLAG,
    ):
        return _TILELANG_FASTMATH_FLAGS
    return None


def _pass_config_enabled(configs: object, key: str) -> bool:
    for config_key, value in mapping_items(configs):
        if _pass_config_key_text(config_key) == key and _truthy_config_value(value):
            return True
    return False


def _pass_config_key_text(key: object) -> str:
    return str(getattr(key, "value", key))


def _truthy_config_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", "none", ""):
        return False
    return bool(value)


def _compile_flag_enabled(flags: object, flag: str) -> bool:
    if flags is None:
        return False
    if isinstance(flags, str):
        return flag in flags.split()
    if isinstance(flags, Mapping) or not isinstance(flags, Iterable):
        return False
    return any(str(item) == flag for item in flags)


def _build_loom_module(
    prim_func: object,
    bindings: tuple[TileLangBinding, ...],
    *,
    function_name: str,
    target_preset: str,
    launch_topology: _LaunchTopology,
    diagnostics: DiagnosticEngine,
    type_converter: TileLangTypeConverter,
    float_fastmath_flags: str | None,
) -> tuple[Module, ImportBodyReport, Counter[str]]:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset=target_preset,
            target_symbol=sanitize_symbol(target_preset, fallback="target"),
            export_symbol=function_name,
            callee=function_name,
            launch_config=None,
            arguments=[
                KernelArgumentSpec(
                    ordinal=binding.ordinal,
                    name=binding.name,
                    type=binding.type,
                )
                for binding in bindings
            ],
            config_arguments=_launch_config_arguments(bindings, launch_topology),
        )
    )
    converter = TileLangConverter(build_default_registry())
    address_layout_preferences = collect_address_layout_preferences(
        prim_func,
        target_preset=target_preset,
    )
    _build_launch_config(
        shell,
        bindings,
        launch_topology,
        diagnostics=diagnostics,
        type_converter=type_converter,
        target_preset=target_preset,
        converter=converter,
    )
    context = TileLangConversionContext(
        builder=shell.builder,
        diagnostics=diagnostics,
        preview_block=shell.body_block,
        type_converter=type_converter,
        target_preset=target_preset,
        kernel_body_block=shell.body_block,
        address_layout_preferences=address_layout_preferences,
        float_fastmath_flags=float_fastmath_flags,
    )
    _map_static_launch_topology(context, launch_topology)
    _capture_block_value_names(context, shell.body_block)
    with shell.builder.insertion_block(shell.body_block):
        mapped_arguments = _map_kernel_arguments(shell, bindings, context, converter)
        if mapped_arguments:
            converter.convert_stmt(getattr(prim_func, "body", None), context)
        shell.builder.kernel.return_()
    rebuild_value_metadata(shell.module)
    return shell.module, context.finish(), converter.operation_counts


def _build_launch_config(
    shell: KernelModuleShell,
    bindings: tuple[TileLangBinding, ...],
    launch_topology: _LaunchTopology,
    *,
    diagnostics: DiagnosticEngine,
    type_converter: TileLangTypeConverter,
    target_preset: str,
    converter: TileLangConverter,
) -> None:
    context = TileLangConversionContext(
        builder=shell.builder,
        diagnostics=diagnostics,
        preview_block=shell.config_block,
        type_converter=type_converter,
        target_preset=target_preset,
    )
    _capture_block_value_names(context, shell.config_block)
    with shell.builder.insertion_block(shell.config_block):
        _map_scalar_kernel_arguments(
            shell.config_arguments_by_ordinal,
            bindings,
            context,
        )
        workgroup_count = tuple(
            _convert_launch_extent(
                extent,
                context,
                converter,
                fallback_name=f"wg_count_{axis}",
            )
            for axis, extent in zip(
                ("x", "y", "z"),
                launch_topology.workgroup_count,
                strict=True,
            )
        )
        workgroup_size = tuple(
            _convert_launch_extent(
                extent,
                context,
                converter,
                fallback_name=f"wg_size_{axis}",
            )
            for axis, extent in zip(
                ("x", "y", "z"),
                launch_topology.workgroup_size,
                strict=True,
            )
        )
        shell.builder.kernel.config(
            workgroup_count_x=workgroup_count[0],
            workgroup_count_y=workgroup_count[1],
            workgroup_count_z=workgroup_count[2],
            workgroup_size_x=workgroup_size[0],
            workgroup_size_y=workgroup_size[1],
            workgroup_size_z=workgroup_size[2],
        )


def _map_static_launch_topology(
    context: TileLangConversionContext,
    launch_topology: _LaunchTopology,
) -> None:
    for source_tag, extent in zip(
        ("blockIdx.x", "blockIdx.y", "blockIdx.z"),
        launch_topology.workgroup_count,
        strict=True,
    ):
        static_extent = integer_value(extent)
        if static_extent is not None:
            context.map_topology_extent(source_tag, None, static_extent=static_extent)
    for source_tag, extent in zip(
        ("threadIdx.x", "threadIdx.y", "threadIdx.z"),
        launch_topology.workgroup_size,
        strict=True,
    ):
        static_extent = integer_value(extent)
        if static_extent is not None:
            context.map_topology_extent(source_tag, None, static_extent=static_extent)


def _convert_launch_extent(
    extent: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    fallback_name: str,
) -> ValueRef:
    immediate = integer_value(extent)
    if immediate is not None:
        return context.ensure_constant(
            str(immediate),
            "index",
            _index_constant_base_name(immediate),
        )
    converted = converter.convert_expr(extent, context, index_like=True)
    if converted is not None:
        return converted
    context.record_blocked(
        node_text(extent),
        f"launch extent `{fallback_name}` could not be imported",
    )
    return context.ensure_constant("1", "index", "c1")


def _index_constant_base_name(value: int) -> str:
    if value >= 0:
        return f"c{value}"
    return f"cm{-value}"


def _capture_block_value_names(
    context: TileLangConversionContext,
    block: object,
) -> None:
    for value_id in getattr(block, "arg_ids", ()):
        value = context.builder.module.values[value_id]
        if value.name:
            context.names.capture(value.name)


def _map_kernel_arguments(
    shell: KernelModuleShell,
    bindings: tuple[TileLangBinding, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> bool:
    zero = context.ensure_constant("0", "offset", "c0_bytes")
    _map_scalar_kernel_arguments(shell.body_arguments_by_ordinal, bindings, context)
    for binding in bindings:
        if binding.buffer is None:
            continue
        argument = shell.body_arguments_by_ordinal[binding.ordinal]
        buffer = _refine_kernel_buffer_argument(binding, argument, context)
        view_type = context.buffer_view_type(binding.buffer)
        dim_bindings = _dynamic_view_dimension_bindings(
            context,
            converter,
            binding.buffer,
            view_type,
        )
        if dim_bindings is None:
            return False
        view = context.builder.buffer.view(
            buffer=buffer,
            byte_offset=zero,
            results=[view_type],
            name=_buffer_view_name(binding, context),
        )
        context.bind_buffer_view_layout(view, binding.buffer)
        _bind_dynamic_view_dimensions(context, view, dim_bindings)
        context.map_value(binding.source, view, str(view_type))
        context.map_value(binding.buffer, view, str(view_type))
        data = getattr(binding.buffer, "data", None)
        if data is not None:
            context.map_buffer_data(data, view, buffer=binding.buffer)
        context.record_converted(
            f"param {binding.name}",
            f"{context.ssa(view)} = buffer.view",
        )
    return True


def _refine_kernel_buffer_argument(
    binding: TileLangBinding,
    argument: ValueRef,
    context: TileLangConversionContext,
) -> ValueRef:
    if not binding.noalias:
        return argument
    base_name = binding.name.removesuffix("_handle")
    if base_name == binding.name:
        base_name = binding.name
    buffers = context.builder.buffer.noalias(
        buffers=[argument],
        results=[argument.type],
        names=[context.reserve_name(f"{base_name}_noalias")],
    )
    buffer = buffers[0]
    context.record_converted(
        f"param {binding.name}",
        f"{context.ssa(buffer)} = buffer.assume.noalias",
    )
    return buffer


def _map_scalar_kernel_arguments(
    arguments_by_ordinal: Mapping[int, ValueRef],
    bindings: tuple[TileLangBinding, ...],
    context: TileLangConversionContext,
) -> None:
    for binding in bindings:
        argument = arguments_by_ordinal.get(binding.ordinal)
        if argument is None:
            continue
        if binding.buffer is not None:
            context.map_value(binding.source, argument, str(argument.type))
            continue
        context.map_value(binding.source, argument, str(argument.type))
        for alias in binding.aliases:
            context.map_value(alias, argument, str(argument.type))


def _launch_config_arguments(
    bindings: tuple[TileLangBinding, ...],
    launch_topology: _LaunchTopology,
) -> tuple[KernelConfigArgumentSpec, ...]:
    needed_keys = {
        key
        for group in collect_dynamic_scalar_source_groups_from(
            _launch_topology_extents(launch_topology)
        )
        for source in (group.source, *group.aliases)
        for key in source_binding_keys(source)
    }
    if not needed_keys:
        return ()
    config_arguments: list[KernelConfigArgumentSpec] = []
    for binding in bindings:
        if binding.buffer is not None:
            continue
        binding_keys = {
            key
            for source in (binding.source, *binding.aliases)
            for key in source_binding_keys(source)
        }
        if needed_keys.isdisjoint(binding_keys):
            continue
        config_arguments.append(
            KernelConfigArgumentSpec(
                ordinal=binding.ordinal,
                name=binding.name,
                type=binding.type,
            )
        )
    return tuple(config_arguments)


def _launch_topology_extents(launch_topology: _LaunchTopology) -> tuple[object, ...]:
    return (*launch_topology.workgroup_count, *launch_topology.workgroup_size)


def _buffer_view_name(
    binding: TileLangBinding,
    context: TileLangConversionContext,
) -> str:
    base_name = binding.name.removesuffix("_handle")
    if base_name == binding.name:
        base_name = f"{binding.name}_view"
    return context.reserve_name(base_name)


def _dynamic_view_dimension_bindings(
    context: TileLangConversionContext,
    converter: TileLangConverter,
    buffer: object,
    view_type: ShapedType,
) -> dict[int, ValueRef] | None:
    dim_bindings: dict[int, ValueRef] = {}
    shape = tuple(getattr(buffer, "shape", ()) or ())
    for position, dim in enumerate(view_type.dims):
        if not isinstance(dim, DynamicDim):
            continue
        if position >= len(shape):
            context.record_blocked(
                node_text(buffer),
                "dynamic TileLang buffer dimension has no source shape",
            )
            return None
        source_dim = shape[position]
        mapped_dim = context.mapped_index_value(source_dim)
        if mapped_dim is None:
            mapped = context.mapped(source_dim)
            if mapped is not None and mapped.type in (INDEX, OFFSET):
                mapped_dim = mapped
        if mapped_dim is None:
            mapped_dim = converter.convert_expr(source_dim, context, index_like=True)
        if mapped_dim is None:
            name = source_name(source_dim, fallback=str(source_dim))
            context.record_blocked(
                node_text(source_dim),
                f"dynamic TileLang buffer dimension `{name}` was not imported "
                "as an index value",
            )
            return None
        dim_bindings[position] = mapped_dim
    return dim_bindings


def _bind_dynamic_view_dimensions(
    context: TileLangConversionContext,
    view: ValueRef,
    dim_bindings: Mapping[int, ValueRef],
) -> None:
    if dim_bindings:
        context.builder.module.values[view.id].dim_bindings = {
            position: binding.id for position, binding in dim_bindings.items()
        }


def _launch_topology(prim_func: object) -> _LaunchTopology:
    source_attrs = attrs(prim_func)
    body_extents = collect_thread_extents(getattr(prim_func, "body", None))
    thread_extent = source_attrs.get("thread_extent")
    thread_extent_items = mapping_items(thread_extent)
    extents = {str(key): value for key, value in body_extents.items()}
    for key, value in thread_extent_items:
        extents[str(key)] = value
    return _LaunchTopology(
        workgroup_count=(
            extents.get("blockIdx.x", 1),
            extents.get("blockIdx.y", 1),
            extents.get("blockIdx.z", 1),
        ),
        workgroup_size=(
            extents.get("threadIdx.x", 1),
            extents.get("threadIdx.y", 1),
            extents.get("threadIdx.z", 1),
        ),
    )


def _static_workgroup_size(topology: _LaunchTopology) -> tuple[int, int, int]:
    values: list[int] = []
    for axis, extent in zip(("x", "y", "z"), topology.workgroup_size, strict=True):
        value = integer_value(extent)
        if value is None:
            raise ValueError(f"TileLang threadIdx.{axis} extent is not static")
        values.append(value)
    return normalize_launch_tuple(tuple(values))
