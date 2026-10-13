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
    RegisterAccess,
    RegisterModule,
    StreamDirection,
    StreamPort,
    TileKind,
    maximum_encoded_dma_transfer_length,
    tile_resource_totals,
    validate_array_family,
)
from loom.target.arch.amd.xdna.array.npu2 import NPU2_ARRAY_FAMILY


@pytest.mark.parametrize("tile_kind", tuple(TileKind))
def test_validator_rejects_dma_mapping_outside_stream_ports(
    tile_kind: TileKind,
) -> None:
    family = NPU2_ARRAY_FAMILY
    port = StreamPort.SOUTH if tile_kind is TileKind.SHIM_NOC else StreamPort.DMA
    capacity = next(
        row.count
        for row in family.stream_ports
        if row.tile_kind is tile_kind
        and row.direction is StreamDirection.SLAVE
        and row.port is port
    )
    tiles = tuple(
        replace(tile, dma=replace(tile.dma, memory_to_stream_port_base=capacity))
        if tile.kind is tile_kind
        else tile
        for tile in family.tiles
    )
    with pytest.raises(ValueError, match="DMA mapping exceeds"):
        validate_array_family(replace(family, tiles=tiles))


def test_validator_rejects_missing_routing_port() -> None:
    family = NPU2_ARRAY_FAMILY
    # The last ordinal row can disappear without breaking ordinal density.
    ports = tuple(
        row
        for row in family.stream_ports
        if not (
            row.tile_kind is TileKind.COMPUTE
            and row.direction is StreamDirection.MASTER
            and row.port is StreamPort.EAST
        )
    )
    with pytest.raises(ValueError, match="missing routing port east"):
        validate_array_family(replace(family, stream_ports=ports))


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

    assert tuple(
        (
            totals.physical_tile_count,
            totals.lock_count,
            totals.dma_channel_count_per_direction,
            totals.dma_buffer_descriptor_count,
        )
        for totals in (tile_resource_totals(family, tile) for tile in family.tiles)
    ) == (
        (8, 128, 16, 128),
        (8, 512, 48, 384),
        (32, 512, 64, 512),
    )


def test_validator_rejects_physical_plan_carrier_overflow() -> None:
    with pytest.raises(ValueError, match="physical-plan carrier overflows"):
        validate_array_family(replace(NPU2_ARRAY_FAMILY, column_count=(1 << 16) - 1))


def test_npu2_memory_distinguishes_local_storage_from_load_apertures() -> None:
    compute = next(
        tile for tile in NPU2_ARRAY_FAMILY.tiles if tile.kind is TileKind.COMPUTE
    )
    memory = next(
        tile for tile in NPU2_ARRAY_FAMILY.tiles if tile.kind is TileKind.MEMORY
    )

    assert (
        compute.memory.local_base,
        compute.memory.local_capacity,
        compute.memory.local_load_base,
    ) == (0, 64 * 1024, 0x70000)
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
    assert (
        memory.memory.local_capacity,
        memory.memory.local_load_base,
        memory.memory.bank_count,
    ) == (512 * 1024, 0x80000, 8)


def test_validator_requires_complete_self_load_window() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            load_windows=tuple(
                window
                for window in compute.memory.load_windows
                if window.owner_column_delta != 0 or window.owner_row_delta != 0
            ),
        ),
    )

    with pytest.raises(ValueError, match="no self load window covers local memory"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_requires_canonical_self_load_base() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    invalid_compute = replace(
        compute,
        memory=replace(compute.memory, local_load_base=0x30000),
    )

    with pytest.raises(ValueError, match="no self load window covers local memory"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_rejects_self_load_range_outside_tile_address() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    *other_windows, self_window = compute.memory.load_windows
    overflowing_base = (1 << NPU2_ARRAY_FAMILY.row_shift) - 64 * 1024 + 1
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            local_load_base=overflowing_base,
            load_windows=(*other_windows, replace(self_window, base=overflowing_base)),
        ),
    )

    with pytest.raises(ValueError, match="local-memory native carrier overflows"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_rejects_load_window_displacement_outside_native_carrier() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    south, *remaining_windows = compute.memory.load_windows
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            load_windows=(
                replace(south, owner_row_delta=-(1 << 7) - 1),
                *remaining_windows,
            ),
        ),
    )

    with pytest.raises(ValueError, match="load-window native carrier overflows"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_requires_unique_load_window_owner() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    south, west, north, self_window = compute.memory.load_windows
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            load_windows=(
                south,
                west,
                replace(
                    north,
                    owner_column_delta=south.owner_column_delta,
                    owner_row_delta=south.owner_row_delta,
                ),
                self_window,
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate load-window owner"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_requires_complete_visible_compute_window() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    south, *remaining_windows = compute.memory.load_windows
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            load_windows=(
                replace(south, capacity=compute.memory.local_capacity - 1),
                *remaining_windows,
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not cover owner local memory"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_requires_vertical_neighbor_load_windows() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    south, west, _north, self_window = compute.memory.load_windows
    invalid_compute = replace(
        compute,
        memory=replace(
            compute.memory,
            load_windows=(south, west, self_window),
        ),
    )

    with pytest.raises(ValueError, match="vertical neighbor load window"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lock_count": 1 << 8}, "invalid lock count"),
        ({"lock_value_minimum": -(1 << 7) - 1}, "native carrier overflows"),
        ({"lock_value_maximum": 1 << 7}, "native carrier overflows"),
        ({"lock_value_maximum": 0}, "invalid lock domain"),
    ],
)
def test_validator_rejects_locks_outside_native_planning_domain(
    changes: dict[str, int], message: str
) -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    invalid_compute = replace(compute, **changes)

    with pytest.raises(ValueError, match=message):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_npu2_compute_lock_domain_admits_ready_and_credit_values() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]

    assert compute.kind is TileKind.COMPUTE
    assert (compute.lock_value_minimum, compute.lock_value_maximum) == (-64, 63)


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


@pytest.mark.parametrize(
    ("tile_kind", "prefix"),
    [
        (TileKind.COMPUTE, "core"),
        (TileKind.MEMORY, "memory_tile"),
        (TileKind.SHIM_NOC, "shim_noc"),
    ],
)
@pytest.mark.parametrize("direction", list(StreamDirection))
def test_npu2_stream_ports_cover_configuration_registers(
    tile_kind: TileKind, prefix: str, direction: StreamDirection
) -> None:
    patterns = {pattern.key: pattern for pattern in NPU2_ARRAY_FAMILY.registers}
    configuration = patterns[f"{prefix}.stream.{direction.value}_config"]
    port_count = sum(
        port.count
        for port in NPU2_ARRAY_FAMILY.stream_ports
        if port.tile_kind is tile_kind and port.direction is direction
    )
    assert port_count == configuration.dimensions[0].count
    if direction is StreamDirection.SLAVE:
        assert port_count == patterns[f"{prefix}.stream.slave_slot"].dimensions[0].count


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

    assert {
        tile.kind: (
            tile.dma.transfer_length_field.bit_width,
            maximum_encoded_dma_transfer_length(tile),
        )
        for tile in NPU2_ARRAY_FAMILY.tiles
        if tile.dma is not None
    } == {
        TileKind.SHIM_NOC: (
            32,
            0xFFFFFFFF,
        ),
        TileKind.MEMORY: (
            17,
            0x1FFFF,
        ),
        TileKind.COMPUTE: (
            14,
            0x3FFF,
        ),
    }


def test_validator_rejects_dma_length_field_from_another_tile_kind() -> None:
    shim = NPU2_ARRAY_FAMILY.tiles[0]
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    assert shim.dma is not None
    assert compute.dma is not None
    invalid_compute = replace(
        compute,
        dma=replace(
            compute.dma,
            transfer_length_field=shim.dma.transfer_length_field,
        ),
    )

    with pytest.raises(ValueError, match="unavailable register module"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_rejects_unregistered_dma_length_field() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    assert compute.dma is not None
    invalid_compute = replace(
        compute,
        dma=replace(
            compute.dma,
            transfer_length_field=replace(compute.dma.transfer_length_field),
        ),
    )

    with pytest.raises(ValueError, match="unavailable canonical register field"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_rejects_signed_dma_length_field() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    assert compute.dma is not None
    signed_field = next(
        field
        for pattern in NPU2_ARRAY_FAMILY.registers
        if pattern.key == "compute_memory.dma.bd.word5"
        for field in pattern.fields
        if field.name == "lock_acquire_value"
    )
    invalid_compute = replace(
        compute,
        dma=replace(compute.dma, transfer_length_field=signed_field),
    )

    with pytest.raises(ValueError, match="DMA length field must be unsigned"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


def test_validator_rejects_dma_length_field_without_descriptor_coverage() -> None:
    compute = NPU2_ARRAY_FAMILY.tiles[-1]
    assert compute.dma is not None
    invalid_compute = replace(
        compute,
        dma=replace(
            compute.dma,
            buffer_descriptor_count=compute.dma.buffer_descriptor_count + 1,
        ),
    )

    with pytest.raises(ValueError, match="does not cover buffer descriptors"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(*NPU2_ARRAY_FAMILY.tiles[:-1], invalid_compute),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("maximum_task_repeat_count", 1 << 16, "u16 resource limit overflows"),
        ("address_dimension_count", 1 << 8, "u8 fact overflows"),
        ("address_maximum", 1 << 64, "address range exceeds u64"),
        ("step_size_bits", 32, "address dimension exceeds its u32 carrier"),
    ],
)
def test_validator_rejects_dma_facts_outside_native_domains(
    field_name: str, value: int, message: str
) -> None:
    shim = NPU2_ARRAY_FAMILY.tiles[0]
    assert shim.dma is not None
    invalid_shim = replace(
        shim,
        dma=replace(shim.dma, **{field_name: value}),
    )

    with pytest.raises(ValueError, match=message):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(invalid_shim, *NPU2_ARRAY_FAMILY.tiles[1:]),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("address_dimension_count", 4, "step fields disagree"),
        ("step_size_bits", 19, "step fields disagree"),
        ("wrap_bits", 9, "wrap fields disagree"),
    ],
)
def test_validator_rejects_dma_facts_disagreeing_with_descriptor_fields(
    field_name: str, value: int, message: str
) -> None:
    shim = NPU2_ARRAY_FAMILY.tiles[0]
    assert shim.dma is not None
    invalid_shim = replace(
        shim,
        dma=replace(shim.dma, **{field_name: value}),
    )

    with pytest.raises(ValueError, match=message):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(invalid_shim, *NPU2_ARRAY_FAMILY.tiles[1:]),
            )
        )


def test_validator_rejects_dma_repeat_fact_outside_queue_field() -> None:
    shim = NPU2_ARRAY_FAMILY.tiles[0]
    assert shim.dma is not None
    invalid_shim = replace(
        shim,
        dma=replace(shim.dma, maximum_task_repeat_count=257),
    )

    with pytest.raises(ValueError, match="queue repeat field does not cover tasks"):
        validate_array_family(
            replace(
                NPU2_ARRAY_FAMILY,
                tiles=(invalid_shim, *NPU2_ARRAY_FAMILY.tiles[1:]),
            )
        )


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


@pytest.mark.parametrize(
    ("key", "event_mask", "switch_count"),
    [
        ("core", 0x7F, 1),
        ("compute_memory", 0x7F, 1),
        ("memory_tile", 0xFF, 2),
        ("shim_pl", 0x7F, 2),
    ],
)
def test_trace_and_timer_modules_preserve_hardware_differences(
    key: str,
    event_mask: int,
    switch_count: int,
) -> None:
    # XAIE2PGBL_*_MODULE definitions and AM025 agree on the event domains,
    # mixed timer access, and one versus two broadcast switches.
    patterns = {pattern.key: pattern for pattern in NPU2_ARRAY_FAMILY.registers}
    control = patterns[f"{key}.trace_control0"]
    expected_masks = {"stop_event": event_mask << 24, "start_event": event_mask << 16}
    if key == "core":
        expected_masks["mode"] = 0x3
    assert {field.name: field.mask for field in control.fields} == expected_masks
    assert {
        field.name: (field.mask, field.access)
        for field in patterns[f"{key}.timer_control"].fields
    } == {
        "reset": (0x80000000, RegisterAccess.WRITE_ONLY),
        "reset_event": (event_mask << 8, RegisterAccess.READ_WRITE),
    }
    assert {
        field.name: (field.mask, field.access)
        for field in patterns[f"{key}.trace_status"].fields
    } == {
        "state": (0x300, RegisterAccess.READ_ONLY),
        "mode": (0x7, RegisterAccess.READ_ONLY),
    }
    for direction in ("south", "west", "north", "east"):
        pattern = patterns[f"{key}.event_broadcast.{direction}.value"]
        assert [
            (dimension.count, dimension.stride) for dimension in pattern.dimensions
        ] == [(switch_count, 0x40)]
        assert pattern.fields[0].access is RegisterAccess.READ_ONLY


def test_validator_rejects_register_provenance_claims_missing_an_oracle() -> None:
    pattern = NPU2_ARRAY_FAMILY.registers[0]
    invalid_pattern = replace(pattern, provenance=Provenance.AIE_RT)
    invalid_family = replace(
        NPU2_ARRAY_FAMILY,
        registers=(invalid_pattern, *NPU2_ARRAY_FAMILY.registers[1:]),
    )

    with pytest.raises(ValueError, match="register provenance is incomplete"):
        validate_array_family(invalid_family)


def test_validator_rejects_register_offsets_outside_tile_aperture() -> None:
    pattern = NPU2_ARRAY_FAMILY.registers[0]
    invalid_pattern = replace(pattern, base_offset=1 << NPU2_ARRAY_FAMILY.row_shift)
    invalid_family = replace(
        NPU2_ARRAY_FAMILY,
        registers=(invalid_pattern, *NPU2_ARRAY_FAMILY.registers[1:]),
    )

    with pytest.raises(ValueError, match="register offset exceeds tile aperture"):
        validate_array_family(invalid_family)
