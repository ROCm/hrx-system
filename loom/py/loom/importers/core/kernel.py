# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared construction for kernel-shaped Loom importer outputs."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field

import loom
from loom.builder import ValueRef
from loom.dsl import Op
from loom.importers.core.names import NameAllocator, sanitize_symbol
from loom.ir import BUFFER_TYPE, INDEX, Block, Module, Region, Type


@dataclass(frozen=True, slots=True)
class KernelArgumentSpec:
    """One imported kernel ABI argument."""

    ordinal: int
    name: str
    type: Type = BUFFER_TYPE


@dataclass(frozen=True, slots=True)
class KernelConfigArgumentSpec:
    """One imported host launch configuration argument."""

    ordinal: int
    name: str
    type: Type = INDEX


@dataclass(frozen=True, slots=True)
class KernelLaunchConfigSpec:
    """Static launch configuration for simple kernel importers."""

    workgroup_count: tuple[int, ...] = (1, 1, 1)
    workgroup_size: tuple[int, ...] = (1, 1, 1)


@dataclass(frozen=True, slots=True)
class KernelModuleSpec:
    """Top-level target/kernel facts shared by foreign importers."""

    target_preset: str
    export_symbol: str
    callee: str
    arguments: Sequence[KernelArgumentSpec]
    config_arguments: Sequence[KernelConfigArgumentSpec] = ()
    target_symbol: str | None = None
    launch_config: KernelLaunchConfigSpec | None = field(
        default_factory=KernelLaunchConfigSpec
    )


@dataclass(frozen=True, slots=True)
class KernelModuleShell:
    """Partially-built Loom module plus handles for filling the kernel body."""

    module: Module
    builder: loom.LoomBuilder
    config: Region
    body: Region
    config_arguments_by_ordinal: dict[int, ValueRef]
    body_arguments_by_ordinal: dict[int, ValueRef]
    target_symbol: str
    kernel_symbol: str

    @property
    def body_block(self) -> Block:
        return self.body.blocks[0]

    @property
    def config_block(self) -> Block:
        return self.config.blocks[0]


@dataclass(frozen=True, slots=True)
class _AmdgpuTargetSelection:
    kind: str
    processor: str | None = None


def create_kernel_module(spec: KernelModuleSpec) -> KernelModuleShell:
    """Create a Loom module containing one target record and one kernel.def."""
    module, builder = loom.module_builder(ops=kernel_module_ops(spec.target_preset))
    target_symbol = spec.target_symbol or sanitize_symbol(
        spec.target_preset, fallback="target"
    )
    _build_target_record(builder, target_symbol, spec.target_preset)
    config_names = NameAllocator()
    config_arguments_by_ordinal = {
        argument.ordinal: builder.value(
            config_names.reserve_or_fresh(argument.name, fallback="arg"),
            argument.type,
        )
        for argument in spec.config_arguments
    }
    body_names = NameAllocator()
    body_arguments_by_ordinal = {
        argument.ordinal: builder.value(
            body_names.reserve_or_fresh(argument.name, fallback="arg"),
            argument.type,
        )
        for argument in spec.arguments
    }
    config = builder.region(
        args=[
            (
                config_arguments_by_ordinal[argument.ordinal].name,
                config_arguments_by_ordinal[argument.ordinal].type,
            )
            for argument in spec.config_arguments
        ]
    )
    body = builder.region()
    builder.kernel.def_(
        target=target_symbol,
        export_symbol=spec.export_symbol,
        callee=spec.callee,
        config_args=[
            (
                config_arguments_by_ordinal[argument.ordinal].name,
                config_arguments_by_ordinal[argument.ordinal].type,
            )
            for argument in spec.config_arguments
        ],
        args=[
            body_arguments_by_ordinal[argument.ordinal] for argument in spec.arguments
        ],
        config=config,
        body=body,
    )
    if spec.launch_config is not None:
        build_static_launch_config(builder, config, spec.launch_config)
    return KernelModuleShell(
        module=module,
        builder=builder,
        config=config,
        body=body,
        config_arguments_by_ordinal=_region_arguments_by_ordinal(
            builder,
            config,
            spec.config_arguments,
        ),
        body_arguments_by_ordinal=_region_arguments_by_ordinal(
            builder,
            body,
            spec.arguments,
        ),
        target_symbol=target_symbol,
        kernel_symbol=spec.callee,
    )


def build_static_launch_config(
    builder: loom.LoomBuilder,
    config: Region,
    launch_config: KernelLaunchConfigSpec,
) -> None:
    """Build a static kernel.launch.config terminator."""

    workgroup_count = normalize_launch_tuple(launch_config.workgroup_count)
    workgroup_size = normalize_launch_tuple(launch_config.workgroup_size)
    with builder.insertion_block(config.blocks[0]):
        count_x = builder.index.constant(
            value=workgroup_count[0],
            results=[INDEX],
            name="wg_count_x",
        )
        count_y = builder.index.constant(
            value=workgroup_count[1],
            results=[INDEX],
            name="wg_count_y",
        )
        count_z = builder.index.constant(
            value=workgroup_count[2],
            results=[INDEX],
            name="wg_count_z",
        )
        size_x = builder.index.constant(
            value=workgroup_size[0],
            results=[INDEX],
            name="wg_size_x",
        )
        size_y = builder.index.constant(
            value=workgroup_size[1],
            results=[INDEX],
            name="wg_size_y",
        )
        size_z = builder.index.constant(
            value=workgroup_size[2],
            results=[INDEX],
            name="wg_size_z",
        )
        builder.kernel.config(
            workgroup_count_x=count_x,
            workgroup_count_y=count_y,
            workgroup_count_z=count_z,
            workgroup_size_x=size_x,
            workgroup_size_y=size_y,
            workgroup_size_z=size_z,
        )


def normalize_workgroup_size(workgroup_size: tuple[int, ...]) -> tuple[int, int, int]:
    """Normalize source workgroup size facts to Loom's x/y/z tuple."""
    return normalize_launch_tuple(workgroup_size)


def normalize_launch_tuple(values: tuple[int, ...]) -> tuple[int, int, int]:
    """Normalize launch dimension facts to Loom's x/y/z tuple."""

    if len(values) > 3:
        raise ValueError(f"unsupported launch rank: {values}")
    normalized = list(values)
    while len(normalized) < 3:
        normalized.append(1)
    return (int(normalized[0]), int(normalized[1]), int(normalized[2]))


def _region_arguments_by_ordinal(
    builder: loom.LoomBuilder,
    region: Region,
    arguments: Sequence[KernelArgumentSpec | KernelConfigArgumentSpec],
) -> dict[int, ValueRef]:
    block = region.blocks[0]
    return {
        argument.ordinal: ValueRef(block.arg_ids[index], builder.ir)
        for index, argument in enumerate(arguments)
    }


def kernel_module_ops(target_preset: str) -> tuple[Op, ...]:
    """Return the op declarations needed for a kernel module target preset."""
    ops = loom.default_ops()
    if _amdgpu_target_selection(target_preset) is None:
        return ops
    from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS

    return (*ops, *ALL_AMDGPU_OPS)


def target_preset_amdgpu_kind(target_preset: str) -> str | None:
    """Return the canonical AMDGPU target kind selected by a target preset."""
    selection = _amdgpu_target_selection(target_preset)
    return selection.kind if selection is not None else None


def _build_target_record(
    builder: loom.LoomBuilder,
    target_symbol: str,
    target_preset: str,
) -> None:
    amdgpu_selection = _amdgpu_target_selection(target_preset)
    if amdgpu_selection is not None:
        from loom.target.arch.amdgpu.dialect import amdgpu_target

        attributes = {"symbol": target_symbol, "kind": amdgpu_selection.kind}
        if amdgpu_selection.processor is not None:
            attributes["processor"] = amdgpu_selection.processor
        builder.ir.build(
            amdgpu_target.name,
            attributes=attributes,
        )
        return
    builder.target.generic(symbol=target_symbol, kind="reference")


def _amdgpu_target_selection(target_preset: str) -> _AmdgpuTargetSelection | None:
    target_cpu = _target_cpu(target_preset)
    if target_cpu is None:
        return None
    from loom.target.arch.amdgpu.dialect import AmdgpuTargetKind
    from loom.target.arch.amdgpu.target_info import (
        amdgpu_default_target_record_info_for_descriptor_set,
        amdgpu_processor_info_by_name,
        amdgpu_target_record_info_for_processor,
    )

    available_kinds = {case.keyword for case in AmdgpuTargetKind.cases}

    target_record = amdgpu_target_record_info_for_processor(target_cpu)
    if target_record is not None and target_record.processor in available_kinds:
        return _AmdgpuTargetSelection(kind=target_record.processor)

    processor_info = amdgpu_processor_info_by_name(target_cpu)
    if processor_info is None or not processor_info.descriptor_set.key:
        return None
    target_record = amdgpu_default_target_record_info_for_descriptor_set(
        processor_info.descriptor_set.key
    )
    if target_record is None or target_record.processor not in available_kinds:
        return None
    return _AmdgpuTargetSelection(
        kind=target_record.processor,
        processor=target_cpu,
    )


def _target_cpu(target_preset: str) -> str | None:
    try:
        tokens = shlex.split(target_preset)
    except ValueError:
        tokens = target_preset.split()
    for index, token in enumerate(tokens):
        if token.startswith("-mcpu="):
            return token.split("=", 1)[1]
        if token.startswith("--mcpu="):
            return token.split("=", 1)[1]
        if token in ("-mcpu", "--mcpu") and index + 1 < len(tokens):
            return tokens[index + 1]
    return None
