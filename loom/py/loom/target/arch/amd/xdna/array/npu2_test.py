# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amd.xdna.array.model import (
    Provenance,
    RegisterModule,
    StreamDirection,
    StreamPort,
    TileKind,
    register_field_count,
    validate_array_family,
)
from loom.target.arch.amd.xdna.array.npu2 import (
    AIE_RT_SOURCE_COMMIT,
    MLIR_AIE_SOURCE_COMMIT,
    NPU2_ARRAY_FAMILY,
    REGISTER_DATABASE_VERSION,
)


def test_npu2_topology_and_resource_domains_are_complete() -> None:
    family = NPU2_ARRAY_FAMILY

    assert family.key == "amd.xdna.npu2"
    assert (family.column_count, family.row_count) == (8, 6)
    assert (family.column_shift, family.row_shift) == (25, 20)
    assert tuple(
        (tile.kind, tile.first_row, tile.row_count) for tile in family.tiles
    ) == (
        (TileKind.SHIM_NOC, 0, 1),
        (TileKind.MEMORY, 1, 1),
        (TileKind.COMPUTE, 2, 4),
    )
    assert {event.module: event.event_count for event in family.events} == {
        RegisterModule.CORE: 128,
        RegisterModule.COMPUTE_MEMORY: 128,
        RegisterModule.MEMORY_TILE: 161,
        RegisterModule.SHIM_PL: 128,
    }


def test_npu2_memory_distinguishes_local_storage_from_load_apertures() -> None:
    compute = next(
        tile for tile in NPU2_ARRAY_FAMILY.tiles if tile.kind is TileKind.COMPUTE
    )
    memory = next(
        tile for tile in NPU2_ARRAY_FAMILY.tiles if tile.kind is TileKind.MEMORY
    )

    assert (compute.memory.local_base, compute.memory.local_capacity) == (0, 64 * 1024)
    assert (compute.memory.program_base, compute.memory.program_capacity) == (
        0,
        16 * 1024,
    )
    assert {
        window.name: (
            window.base,
            window.capacity,
            window.lock_selector_base,
            window.owner_column_delta,
            window.owner_row_delta,
        )
        for window in compute.memory.load_windows
    } == {
        "south": (0x40000, 64 * 1024, 0, 0, -1),
        "west": (0x50000, 64 * 1024, 16, -1, 0),
        "north": (0x60000, 64 * 1024, 32, 0, 1),
        "self": (0x70000, 64 * 1024, 48, 0, 0),
    }
    assert (memory.memory.local_capacity, memory.memory.bank_count) == (512 * 1024, 8)


def test_npu2_stream_ordinals_match_programmable_register_order() -> None:
    port_ranges = {
        (row.tile_kind, row.direction, row.port): (row.ordinal, row.count)
        for row in NPU2_ARRAY_FAMILY.stream_ports
    }

    assert port_ranges[
        (TileKind.COMPUTE, StreamDirection.MASTER, StreamPort.NORTH)
    ] == (13, 6)
    assert port_ranges[(TileKind.COMPUTE, StreamDirection.SLAVE, StreamPort.SOUTH)] == (
        5,
        6,
    )
    assert port_ranges[(TileKind.MEMORY, StreamDirection.MASTER, StreamPort.DMA)] == (
        0,
        6,
    )
    assert port_ranges[
        (TileKind.SHIM_NOC, StreamDirection.SLAVE, StreamPort.SOUTH)
    ] == (2, 8)


def test_npu2_dma_encoding_and_stream_port_mappings_are_exact() -> None:
    dma = {tile.kind: tile.dma for tile in NPU2_ARRAY_FAMILY.tiles}

    shim = dma[TileKind.SHIM_NOC]
    compute = dma[TileKind.COMPUTE]
    assert shim is not None
    assert compute is not None
    assert (
        shim.address_encoding_shift,
        shim.transfer_length_granularity,
        shim.transfer_length_offset,
        shim.maximum_task_repeat_count,
    ) == (0, 4, 0, 256)
    assert (
        shim.memory_to_stream_port_base,
        shim.memory_to_stream_port_stride,
        shim.stream_to_memory_port_base,
        shim.stream_to_memory_port_stride,
    ) == (3, 4, 2, 1)
    assert (
        compute.address_encoding_shift,
        compute.memory_to_stream_port_base,
        compute.memory_to_stream_port_stride,
        compute.stream_to_memory_port_base,
        compute.stream_to_memory_port_stride,
    ) == (2, 0, 1, 0, 1)


def test_npu2_direct_dma_loopback_pairs_match_the_stream_switch() -> None:
    # The pinned aie-rt compute and memory validators permit DMA-to-DMA only
    # with matching indices. Shim DMA uses a separate mux and south ports.
    assert {
        tile.kind: tile.dma.loopback_channel_count
        for tile in NPU2_ARRAY_FAMILY.tiles
        if tile.dma is not None
    } == {TileKind.SHIM_NOC: 0, TileKind.MEMORY: 6, TileKind.COMPUTE: 2}


@pytest.mark.parametrize("loopback_channel_count", [-1, 3])
def test_validator_rejects_loopback_pairs_outside_dma_channels(
    loopback_channel_count: int,
) -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    assert compute.dma is not None
    invalid_compute = replace(
        compute,
        dma=replace(compute.dma, loopback_channel_count=loopback_channel_count),
    )
    with pytest.raises(ValueError, match="invalid DMA loopback channel range"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_register_patterns_cover_complete_seed_resource_families() -> None:
    family = NPU2_ARRAY_FAMILY

    # 49 base patterns plus the 4 core-module trace patterns (control0,
    # control1, event0, event1) added for hardware event trace.
    assert len(family.registers) == 53
    assert register_field_count(family) == 186
    assert all(
        pattern.provenance & (Provenance.AIE_RT | Provenance.REGISTER_DATABASE)
        == (Provenance.AIE_RT | Provenance.REGISTER_DATABASE)
        for pattern in family.registers
    )
    patterns = {pattern.key: pattern for pattern in family.registers}
    core_bd = patterns["compute_memory.dma.bd.word5"]
    assert core_bd.base_offset == 0x1D014
    assert core_bd.dimensions[0].count == 16
    assert core_bd.dimensions[0].stride == 0x20
    assert {field.name: field.mask for field in core_bd.fields}[
        "lock_acquire_value"
    ] == 0x00000FE0
    assert tuple(
        (dimension.name, dimension.count, dimension.stride)
        for dimension in patterns["memory_tile.stream.slave_slot"].dimensions
    ) == (("slave_port", 18, 0x10), ("slot", 4, 4))


def test_npu2_source_revisions_are_explicit_oracle_identities() -> None:
    assert AIE_RT_SOURCE_COMMIT == "8849e208bdcc533b20a0ed3f95c1ce961dee9c3a"
    assert MLIR_AIE_SOURCE_COMMIT == "db06374df9bf83d9fc557001ca213368aed15788"
    assert REGISTER_DATABASE_VERSION == "AM025-2024-11-13-1.1"


def test_validator_rejects_register_provenance_claims_missing_an_oracle() -> None:
    pattern = NPU2_ARRAY_FAMILY.registers[0]
    invalid_pattern = replace(pattern, provenance=Provenance.AIE_RT)
    invalid_family = replace(
        NPU2_ARRAY_FAMILY,
        registers=(invalid_pattern, *NPU2_ARRAY_FAMILY.registers[1:]),
    )

    with pytest.raises(ValueError, match="register provenance is incomplete"):
        validate_array_family(invalid_family)
