# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validated physical-array and register-field source schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from itertools import product

_INT8_MINIMUM = -(1 << 7)
_INT8_MAXIMUM = (1 << 7) - 1
_UINT8_MAXIMUM = (1 << 8) - 1
_UINT16_MAXIMUM = (1 << 16) - 1
_UINT32_MAXIMUM = (1 << 32) - 1


class Provenance(IntFlag):
    """Independent sources supporting one physical fact."""

    AIE_RT = 1 << 0
    REGISTER_DATABASE = 1 << 1
    MLIR_AIE = 1 << 2
    XDNA_DRIVER = 1 << 3
    HARDWARE = 1 << 4


class Architecture(Enum):
    """Independent AIE instruction-set identity."""

    AIE2P = "aie2p"


class TileKind(Enum):
    """Physical NPU tile role."""

    SHIM_NOC = "shim_noc"
    MEMORY = "memory"
    COMPUTE = "compute"


class RegisterModule(Enum):
    """Independently addressed register module."""

    CORE = "core"
    COMPUTE_MEMORY = "compute_memory"
    MEMORY_TILE = "memory_tile"
    SHIM_NOC = "shim_noc"
    SHIM_PL = "shim_pl"


class StreamDirection(Enum):
    """Direction relative to the programmable stream switch."""

    MASTER = "master"
    SLAVE = "slave"


class StreamPort(Enum):
    """Architectural stream-switch port class."""

    CORE = "core"
    DMA = "dma"
    TILE_CONTROL = "tile_control"
    FIFO = "fifo"
    SOUTH = "south"
    WEST = "west"
    NORTH = "north"
    EAST = "east"
    TRACE = "trace"


class RegisterAccess(Enum):
    """Software-visible register access contract."""

    READ_WRITE = "read_write"
    WRITE_ONLY = "write_only"
    READ_ONLY = "read_only"


@dataclass(frozen=True, slots=True)
class AddressWindow:
    """One tile-relative load aperture mapped to canonical owner storage."""

    name: str
    base: int
    capacity: int
    lock_selector_base: int
    owner_column_delta: int
    owner_row_delta: int
    owner_kind: TileKind


@dataclass(frozen=True, slots=True)
class TileMemoryFacts:
    """Allocation geometry and load apertures for one tile kind."""

    local_base: int
    local_capacity: int
    local_load_base: int
    bank_count: int
    program_base: int
    program_capacity: int
    program_load_base: int
    load_windows: tuple[AddressWindow, ...]


@dataclass(frozen=True, slots=True)
class DmaEngineFacts:
    """Resource and field-width limits for one tile DMA engine."""

    buffer_descriptor_count: int
    maximum_task_repeat_count: int
    channel_count_per_direction: int
    loopback_channel_count: int
    address_dimension_count: int
    address_maximum: int
    address_alignment: int
    address_encoding_shift: int
    transfer_length_granularity: int
    transfer_length_offset: int
    transfer_length_field: RegisterField
    memory_to_stream_port_base: int
    memory_to_stream_port_stride: int
    stream_to_memory_port_base: int
    stream_to_memory_port_stride: int
    step_size_bits: int
    wrap_bits: int
    iteration_bits: int
    task_queue_depth: int
    supports_compression: bool
    supports_padding: bool
    supports_out_of_order: bool
    supports_tokens: bool
    supports_tlast_suppression: bool


@dataclass(frozen=True, slots=True)
class TileFacts:
    """All physical resources shared by tiles of one kind."""

    kind: TileKind
    first_row: int
    row_count: int
    memory: TileMemoryFacts
    lock_count: int
    lock_value_minimum: int
    lock_value_maximum: int
    packet_arbiter_count: int
    packet_master_select_count: int
    register_modules: tuple[RegisterModule, ...]
    dma: DmaEngineFacts | None


@dataclass(frozen=True, slots=True)
class TileResourceTotals:
    """Aggregate resources across all physical tiles of one kind."""

    physical_tile_count: int
    lock_count: int
    dma_channel_count_per_direction: int
    dma_buffer_descriptor_count: int


@dataclass(frozen=True, slots=True)
class EventModuleFacts:
    """Event identifier domain implemented by one register module."""

    module: RegisterModule
    event_count: int
    named_event_count: int


@dataclass(frozen=True, slots=True)
class StreamPortRange:
    """Contiguous channel range in one switch direction's ordinal space."""

    tile_kind: TileKind
    direction: StreamDirection
    port: StreamPort
    ordinal: int
    count: int
    # Legal master ordinals for each slave channel; empty for master ranges.
    connections: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RegisterDimension:
    """One indexed dimension in a regular register-address pattern."""

    name: str
    count: int
    stride: int


@dataclass(frozen=True, slots=True)
class RegisterField:
    """One software-visible field within a 32-bit register."""

    name: str
    least_significant_bit: int
    bit_width: int
    is_signed: bool = False
    # Hardware access is field-local, including registers with mixed access.
    access: RegisterAccess = RegisterAccess.READ_WRITE

    @property
    def mask(self) -> int:
        """Returns the field mask in its containing register."""
        return ((1 << self.bit_width) - 1) << self.least_significant_bit


@dataclass(frozen=True, slots=True)
class RegisterPattern:
    """Regular indexed register family and its supported semantic fields."""

    key: str
    module: RegisterModule
    base_offset: int
    dimensions: tuple[RegisterDimension, ...]
    fields: tuple[RegisterField, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ArrayFamily:
    """Complete materialized physical-array facts shared by device profiles."""

    key: str
    revision: int
    architecture: Architecture
    column_count: int
    row_count: int
    column_shift: int
    row_shift: int
    address_generation_granularity_bits: int
    tiles: tuple[TileFacts, ...]
    events: tuple[EventModuleFacts, ...]
    stream_ports: tuple[StreamPortRange, ...]
    registers: tuple[RegisterPattern, ...]
    provenance: Provenance


def tile_resource_totals(family: ArrayFamily, tile: TileFacts) -> TileResourceTotals:
    """Derives family-wide resource totals for one tile-kind row."""
    physical_tile_count = family.column_count * tile.row_count
    dma = tile.dma
    return TileResourceTotals(
        physical_tile_count=physical_tile_count,
        lock_count=physical_tile_count * tile.lock_count,
        dma_channel_count_per_direction=(
            physical_tile_count * dma.channel_count_per_direction if dma else 0
        ),
        dma_buffer_descriptor_count=(
            physical_tile_count * dma.buffer_descriptor_count if dma else 0
        ),
    )


def _validate_tile_memory(tile: TileFacts, row_shift: int) -> None:
    memory = tile.memory
    if (
        memory.local_base < 0
        or memory.local_capacity < 0
        or memory.local_load_base < 0
        or memory.bank_count < 0
    ):
        raise ValueError(f"{tile.kind.value}: invalid local-memory geometry")
    if (
        memory.local_base > _UINT32_MAXIMUM
        or memory.local_capacity > _UINT32_MAXIMUM
        or memory.local_load_base > _UINT32_MAXIMUM
        or memory.local_base + memory.local_capacity > 1 << 32
        or memory.local_load_base + memory.local_capacity > 1 << 32
        or memory.local_load_base + memory.local_capacity > 1 << row_shift
        or memory.bank_count > _UINT8_MAXIMUM
        or len(memory.load_windows) > _UINT8_MAXIMUM
    ):
        raise ValueError(f"{tile.kind.value}: local-memory native carrier overflows")
    if memory.local_capacity == 0 and memory.bank_count != 0:
        raise ValueError(f"{tile.kind.value}: empty local memory declares banks")
    if memory.local_capacity and (
        memory.bank_count == 0 or memory.local_capacity % memory.bank_count
    ):
        raise ValueError(f"{tile.kind.value}: local-memory banks do not partition")
    if (
        memory.program_base < 0
        or memory.program_capacity < 0
        or memory.program_load_base < 0
        or memory.program_base > _UINT32_MAXIMUM
        or memory.program_capacity > _UINT32_MAXIMUM
        or memory.program_load_base > _UINT32_MAXIMUM
        or memory.program_base + memory.program_capacity > 1 << 32
        or memory.program_load_base + memory.program_capacity > 1 << 32
        or memory.program_load_base + memory.program_capacity > 1 << row_shift
    ):
        raise ValueError(f"{tile.kind.value}: invalid program-memory geometry")
    if tile.kind is not TileKind.COMPUTE and memory.program_capacity != 0:
        raise ValueError(f"{tile.kind.value}: non-compute tile has program memory")
    names = [window.name for window in memory.load_windows]
    if len(names) != len(set(names)):
        raise ValueError(f"{tile.kind.value}: duplicate load-window name")
    owner_windows = [
        (window.owner_kind, window.owner_column_delta, window.owner_row_delta)
        for window in memory.load_windows
    ]
    if len(owner_windows) != len(set(owner_windows)):
        raise ValueError(f"{tile.kind.value}: duplicate load-window owner")
    self_windows = tuple(
        window
        for window in memory.load_windows
        if window.owner_kind is tile.kind
        and window.owner_column_delta == 0
        and window.owner_row_delta == 0
    )
    if memory.local_capacity and (
        len(self_windows) != 1
        or self_windows[0].base != memory.local_load_base
        or self_windows[0].capacity < memory.local_capacity
    ):
        raise ValueError(f"{tile.kind.value}: no self load window covers local memory")
    if not memory.local_capacity and memory.local_load_base != 0:
        raise ValueError(f"{tile.kind.value}: empty local memory has a load base")
    for index, window in enumerate(memory.load_windows):
        if (
            not window.name
            or window.base < 0
            or window.capacity <= 0
            or window.lock_selector_base < 0
        ):
            raise ValueError(f"{tile.kind.value}: invalid load window")
        if (
            window.base > _UINT32_MAXIMUM
            or window.capacity > _UINT32_MAXIMUM
            or window.lock_selector_base > _UINT16_MAXIMUM
            or window.owner_column_delta < _INT8_MINIMUM
            or window.owner_column_delta > _INT8_MAXIMUM
            or window.owner_row_delta < _INT8_MINIMUM
            or window.owner_row_delta > _INT8_MAXIMUM
        ):
            raise ValueError(
                f"{tile.kind.value}.{window.name}: load-window native carrier overflows"
            )
        end = window.base + window.capacity
        if end > 1 << 32:
            raise ValueError(f"{tile.kind.value}.{window.name}: address overflow")
        for other in memory.load_windows[index + 1 :]:
            other_end = other.base + other.capacity
            if window.base < other_end and other.base < end:
                raise ValueError(
                    f"{tile.kind.value}: load windows {window.name} and "
                    f"{other.name} overlap"
                )


def _validate_load_window_resources(family: ArrayFamily, tile: TileFacts) -> None:
    tile_kinds = {row.kind: row for row in family.tiles}
    if tile.kind is TileKind.COMPUTE and tile.row_count > 1:
        owner_deltas = {
            (window.owner_kind, window.owner_column_delta, window.owner_row_delta)
            for window in tile.memory.load_windows
        }
        required_neighbors = {
            (TileKind.COMPUTE, 0, -1),
            (TileKind.COMPUTE, 0, 1),
        }
        if not required_neighbors.issubset(owner_deltas):
            raise ValueError(
                f"{tile.kind.value}: vertical neighbor load window is unavailable"
            )
    lock_ranges: list[tuple[int, int, str]] = []
    for window in tile.memory.load_windows:
        owner = tile_kinds.get(window.owner_kind)
        if owner is None:
            raise ValueError(
                f"{tile.kind.value}.{window.name}: lock owner kind is unavailable"
            )
        if (
            tile.kind is TileKind.COMPUTE
            and window.owner_kind is TileKind.COMPUTE
            and window.capacity < owner.memory.local_capacity
        ):
            raise ValueError(
                f"{tile.kind.value}.{window.name}: compute load window does not "
                "cover owner local memory"
            )
        lock_end = window.lock_selector_base + owner.lock_count
        if lock_end > 1 << 16:
            raise ValueError(
                f"{tile.kind.value}.{window.name}: lock selector range overflows"
            )
        for other_base, other_end, other_name in lock_ranges:
            if window.lock_selector_base < other_end and other_base < lock_end:
                raise ValueError(
                    f"{tile.kind.value}: lock windows {window.name} and "
                    f"{other_name} overlap"
                )
        lock_ranges.append((window.lock_selector_base, lock_end, window.name))


def _validate_dma(tile: TileFacts) -> None:
    dma = tile.dma
    if dma is None:
        return
    positive_values = (
        dma.buffer_descriptor_count,
        dma.maximum_task_repeat_count,
        dma.channel_count_per_direction,
        dma.address_dimension_count,
        dma.address_maximum,
        dma.address_alignment,
        dma.transfer_length_granularity,
        dma.memory_to_stream_port_stride,
        dma.stream_to_memory_port_stride,
        dma.step_size_bits,
        dma.wrap_bits,
        dma.iteration_bits,
        dma.task_queue_depth,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError(f"{tile.kind.value}: invalid DMA resource limit")
    if dma.address_maximum > (1 << 64) - 1:
        raise ValueError(f"{tile.kind.value}: DMA address range exceeds u64")
    if (
        dma.buffer_descriptor_count > (1 << 16) - 1
        or dma.maximum_task_repeat_count > (1 << 16) - 1
    ):
        raise ValueError(f"{tile.kind.value}: DMA u16 resource limit overflows")
    if not 0 <= dma.loopback_channel_count <= dma.channel_count_per_direction:
        raise ValueError(f"{tile.kind.value}: invalid DMA loopback channel range")
    byte_values = (
        dma.channel_count_per_direction,
        dma.loopback_channel_count,
        dma.address_dimension_count,
        dma.address_alignment,
        dma.address_encoding_shift,
        dma.transfer_length_granularity,
        dma.transfer_length_offset,
        dma.memory_to_stream_port_base,
        dma.memory_to_stream_port_stride,
        dma.stream_to_memory_port_base,
        dma.stream_to_memory_port_stride,
        dma.step_size_bits,
        dma.wrap_bits,
        dma.iteration_bits,
        dma.task_queue_depth,
    )
    if any(value < 0 or value > (1 << 8) - 1 for value in byte_values):
        raise ValueError(f"{tile.kind.value}: DMA u8 fact overflows")
    if dma.step_size_bits >= 32 or dma.wrap_bits >= 32:
        raise ValueError(
            f"{tile.kind.value}: DMA address dimension exceeds its u32 carrier"
        )
    if dma.address_alignment & (dma.address_alignment - 1):
        raise ValueError(f"{tile.kind.value}: DMA alignment is not a power of two")
    if dma.address_maximum % dma.address_alignment:
        raise ValueError(f"{tile.kind.value}: DMA address range is misaligned")
    if dma.address_encoding_shift < 0:
        raise ValueError(f"{tile.kind.value}: invalid DMA address encoding shift")
    if dma.transfer_length_granularity & (dma.transfer_length_granularity - 1):
        raise ValueError(
            f"{tile.kind.value}: DMA length granularity is not a power of two"
        )
    if dma.transfer_length_offset < 0:
        raise ValueError(f"{tile.kind.value}: invalid DMA length encoding offset")
    for base, stride in (
        (dma.memory_to_stream_port_base, dma.memory_to_stream_port_stride),
        (dma.stream_to_memory_port_base, dma.stream_to_memory_port_stride),
    ):
        if base < 0 or base + (dma.channel_count_per_direction - 1) * stride > 255:
            raise ValueError(f"{tile.kind.value}: invalid DMA stream-port mapping")


def _validate_stream_ports(family: ArrayFamily) -> None:
    keys = [(row.tile_kind, row.direction, row.port) for row in family.stream_ports]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate stream-port range")
    tile_kinds = {tile.kind for tile in family.tiles}
    for row in family.stream_ports:
        if (
            row.tile_kind not in tile_kinds
            or row.ordinal < 0
            or row.ordinal > _UINT8_MAXIMUM
            or row.count <= 0
            or row.count > _UINT8_MAXIMUM
            or row.ordinal + row.count > 1 << 8
        ):
            raise ValueError("invalid stream-port range")
    for tile_kind in tile_kinds:
        for direction in StreamDirection:
            rows = sorted(
                (
                    row
                    for row in family.stream_ports
                    if row.tile_kind is tile_kind and row.direction is direction
                ),
                key=lambda row: row.ordinal,
            )
            expected_ordinal = 0
            for row in rows:
                if row.ordinal != expected_ordinal:
                    raise ValueError(
                        f"{tile_kind.value}.{direction.value}: stream ordinals "
                        "are not dense"
                    )
                expected_ordinal += row.count
            if expected_ordinal > 32:
                raise ValueError("stream ordinals exceed connection-mask width")

    ranges = {
        (row.tile_kind, row.direction, row.port): row for row in family.stream_ports
    }
    for tile in family.tiles:
        routing_ports = (StreamPort.NORTH, StreamPort.SOUTH)
        if tile.kind is not TileKind.MEMORY:
            routing_ports += (StreamPort.WEST, StreamPort.EAST)
        for direction in StreamDirection:
            for port in (*routing_ports, StreamPort.TILE_CONTROL):
                if (tile.kind, direction, port) not in ranges:
                    raise ValueError(
                        f"{tile.kind.value}.{direction.value}: missing routing "
                        f"port {port.value}"
                    )
        dma = tile.dma
        if dma is None:
            continue
        port = StreamPort.SOUTH if tile.kind is TileKind.SHIM_NOC else StreamPort.DMA
        for direction, base, stride in (
            (
                StreamDirection.SLAVE,
                dma.memory_to_stream_port_base,
                dma.memory_to_stream_port_stride,
            ),
            (
                StreamDirection.MASTER,
                dma.stream_to_memory_port_base,
                dma.stream_to_memory_port_stride,
            ),
        ):
            row = ranges.get((tile.kind, direction, port))
            if (
                row is None
                or base + (dma.channel_count_per_direction - 1) * stride >= row.count
            ):
                raise ValueError(
                    f"{tile.kind.value}.{direction.value}: DMA mapping exceeds "
                    f"stream port {port.value}"
                )
    for row in family.stream_ports:
        expected_count = row.count if row.direction is StreamDirection.SLAVE else 0
        master_count = sum(
            master.count
            for master in family.stream_ports
            if master.tile_kind is row.tile_kind
            and master.direction is StreamDirection.MASTER
        )
        if len(row.connections) != expected_count or any(
            mask < 0 or mask >= 1 << master_count for mask in row.connections
        ):
            raise ValueError("invalid stream connection masks")


def _register_offsets(pattern: RegisterPattern) -> tuple[int, ...]:
    if not pattern.dimensions:
        return (pattern.base_offset,)
    return tuple(
        pattern.base_offset
        + sum(
            index * dimension.stride
            for index, dimension in zip(indices, pattern.dimensions, strict=True)
        )
        for indices in product(
            *(range(dimension.count) for dimension in pattern.dimensions)
        )
    )


def _validate_registers(family: ArrayFamily) -> None:
    pattern_keys = [pattern.key for pattern in family.registers]
    if len(pattern_keys) != len(set(pattern_keys)):
        raise ValueError("duplicate register-pattern key")
    available_modules = {
        module for tile in family.tiles for module in tile.register_modules
    }
    occupied_offsets: dict[RegisterModule, dict[int, str]] = {
        module: {} for module in available_modules
    }
    field_keys: set[str] = set()
    required_provenance = Provenance.AIE_RT | Provenance.REGISTER_DATABASE
    for pattern in family.registers:
        if not pattern.key or pattern.module not in available_modules:
            raise ValueError(f"{pattern.key!r}: unavailable register module")
        if pattern.base_offset < 0 or pattern.base_offset % 4:
            raise ValueError(f"{pattern.key}: invalid register base offset")
        if pattern.provenance & required_provenance != required_provenance:
            raise ValueError(f"{pattern.key}: register provenance is incomplete")
        dimension_names = [dimension.name for dimension in pattern.dimensions]
        if len(dimension_names) != len(set(dimension_names)):
            raise ValueError(f"{pattern.key}: duplicate register dimension")
        for dimension in pattern.dimensions:
            if not dimension.name or dimension.count <= 0 or dimension.stride <= 0:
                raise ValueError(f"{pattern.key}: invalid register dimension")
            if dimension.stride % 4:
                raise ValueError(f"{pattern.key}: register stride is misaligned")
        offsets = _register_offsets(pattern)
        if len(offsets) != len(set(offsets)) or max(offsets) > 0xFFFFFFFF - 3:
            raise ValueError(f"{pattern.key}: register offsets collide or overflow")
        if max(offsets) + 4 > 1 << family.row_shift:
            raise ValueError(f"{pattern.key}: register offset exceeds tile aperture")
        module_offsets = occupied_offsets[pattern.module]
        for offset in offsets:
            previous = module_offsets.get(offset)
            if previous is not None:
                raise ValueError(
                    f"{pattern.key}: register offset 0x{offset:x} collides with "
                    f"{previous}"
                )
            module_offsets[offset] = pattern.key
        field_names = [field.name for field in pattern.fields]
        if not field_names or len(field_names) != len(set(field_names)):
            raise ValueError(f"{pattern.key}: invalid register-field names")
        occupied_mask = 0
        for field in pattern.fields:
            key = f"{pattern.key}.{field.name}"
            if key in field_keys:
                raise ValueError(f"duplicate register-field key {key}")
            field_keys.add(key)
            if (
                not field.name
                or field.least_significant_bit < 0
                or field.bit_width <= 0
                or field.least_significant_bit + field.bit_width > 32
            ):
                raise ValueError(f"{key}: invalid bit range")
            if field.mask & occupied_mask:
                raise ValueError(f"{pattern.key}: register fields overlap")
            occupied_mask |= field.mask


def _resolve_register_field(
    family: ArrayFamily, expected_field: RegisterField
) -> tuple[RegisterPattern, RegisterField]:
    for pattern in family.registers:
        for field in pattern.fields:
            if field is expected_field:
                return pattern, field
    raise ValueError(f"{expected_field.name!r}: unavailable canonical register field")


def _validate_dma_register_fields(family: ArrayFamily, tile: TileFacts) -> None:
    dma = tile.dma
    if dma is None:
        return
    pattern, field = _resolve_register_field(family, dma.transfer_length_field)
    if pattern.module not in tile.register_modules:
        raise ValueError(
            f"{tile.kind.value}: DMA length field uses unavailable register module"
        )
    if field.is_signed:
        raise ValueError(f"{tile.kind.value}: DMA length field must be unsigned")
    descriptor_dimensions = tuple(
        dimension
        for dimension in pattern.dimensions
        if dimension.name == "buffer_descriptor"
    )
    if (
        len(descriptor_dimensions) != 1
        or descriptor_dimensions[0].count != dma.buffer_descriptor_count
    ):
        raise ValueError(
            f"{tile.kind.value}: DMA length field does not cover buffer descriptors"
        )

    descriptor_patterns = tuple(
        candidate
        for candidate in family.registers
        if candidate.module is pattern.module
        and any(
            dimension.name == "buffer_descriptor"
            and dimension.count == dma.buffer_descriptor_count
            for dimension in candidate.dimensions
        )
    )

    def indexed_fields(suffix: str) -> dict[int, RegisterField]:
        result: dict[int, RegisterField] = {}
        for descriptor_pattern in descriptor_patterns:
            for candidate in descriptor_pattern.fields:
                prefix, separator, candidate_suffix = candidate.name.partition("_")
                if (
                    separator
                    and candidate_suffix == suffix
                    and len(prefix) > 1
                    and prefix[0] == "d"
                    and prefix[1:].isdigit()
                ):
                    index = int(prefix[1:])
                    if index in result:
                        raise ValueError(
                            f"{tile.kind.value}: duplicate DMA {suffix} dimension"
                        )
                    result[index] = candidate
        return result

    step_fields = indexed_fields("step_size")
    expected_step_indices = set(range(dma.address_dimension_count))
    if set(step_fields) != expected_step_indices or any(
        field.bit_width != dma.step_size_bits for field in step_fields.values()
    ):
        raise ValueError(
            f"{tile.kind.value}: DMA step fields disagree with address dimensions"
        )

    wrap_fields = indexed_fields("wrap")
    expected_wrap_indices = set(range(dma.address_dimension_count - 1))
    if set(wrap_fields) != expected_wrap_indices or any(
        field.bit_width != dma.wrap_bits for field in wrap_fields.values()
    ):
        raise ValueError(
            f"{tile.kind.value}: DMA wrap fields disagree with address dimensions"
        )

    queue_patterns = tuple(
        candidate
        for candidate in family.registers
        if candidate.module is pattern.module
        and any(field.name == "start_bd_id" for field in candidate.fields)
    )
    if len(queue_patterns) != 2:
        raise ValueError(f"{tile.kind.value}: DMA queue directions are incomplete")
    for queue_pattern in queue_patterns:
        queue_fields = {field.name: field for field in queue_pattern.fields}
        start_field = queue_fields.get("start_bd_id")
        repeat_field = queue_fields.get("repeat_count")
        token_field = queue_fields.get("enable_token_issue")
        channel_dimensions = tuple(
            dimension
            for dimension in queue_pattern.dimensions
            if dimension.name == "channel"
        )
        if (
            start_field is None
            or repeat_field is None
            or token_field is None
            or start_field.is_signed
            or repeat_field.is_signed
            or token_field.is_signed
            or len(channel_dimensions) != 1
            or channel_dimensions[0].count != dma.channel_count_per_direction
        ):
            raise ValueError(
                f"{tile.kind.value}: DMA queue fields disagree with resources"
            )
        if dma.buffer_descriptor_count > 1 << start_field.bit_width:
            raise ValueError(
                f"{tile.kind.value}: DMA queue start field does not cover descriptors"
            )
        if dma.maximum_task_repeat_count > 1 << repeat_field.bit_width:
            raise ValueError(
                f"{tile.kind.value}: DMA queue repeat field does not cover tasks"
            )
        if token_field.bit_width != 1:
            raise ValueError(f"{tile.kind.value}: DMA queue token field is not boolean")


def validate_array_family(family: ArrayFamily) -> None:
    """Validates one complete materialized physical-array fact set."""
    if not family.key or family.revision <= 0:
        raise ValueError("array-family identity is incomplete")
    if family.revision > _UINT32_MAXIMUM:
        raise ValueError(f"{family.key}: revision native carrier overflows")
    if family.column_count <= 0 or family.row_count <= 0:
        raise ValueError(f"{family.key}: invalid array geometry")
    if (
        family.column_count > _UINT16_MAXIMUM
        or family.row_count > _UINT16_MAXIMUM
        or len(family.tiles) > _UINT8_MAXIMUM
        or sum(len(tile.memory.load_windows) for tile in family.tiles) > _UINT16_MAXIMUM
        or len(family.events) > _UINT8_MAXIMUM
        or len(family.stream_ports) > _UINT8_MAXIMUM
        or family.address_generation_granularity_bits > _UINT8_MAXIMUM
        or int(family.provenance) < 0
        or int(family.provenance) > _UINT32_MAXIMUM
    ):
        raise ValueError(f"{family.key}: array native carrier overflows")
    if not 0 < family.row_shift < family.column_shift < 64:
        raise ValueError(f"{family.key}: invalid tile address shifts")
    if family.address_generation_granularity_bits <= 0:
        raise ValueError(f"{family.key}: invalid address-generation granularity")
    tile_kinds = [tile.kind for tile in family.tiles]
    if len(tile_kinds) != len(set(tile_kinds)):
        raise ValueError(f"{family.key}: duplicate tile-kind facts")
    covered_rows: set[int] = set()
    modules: set[RegisterModule] = set()
    for tile in family.tiles:
        if (
            tile.first_row < 0
            or tile.first_row > _UINT8_MAXIMUM
            or tile.row_count <= 0
            or tile.row_count > _UINT8_MAXIMUM
        ):
            raise ValueError(f"{tile.kind.value}: invalid tile rows")
        rows = set(range(tile.first_row, tile.first_row + tile.row_count))
        if max(rows) >= family.row_count or covered_rows & rows:
            raise ValueError(f"{tile.kind.value}: tile rows overlap or overflow")
        covered_rows |= rows
        if len(tile.register_modules) != len(set(tile.register_modules)):
            raise ValueError(f"{tile.kind.value}: duplicate register module")
        modules.update(tile.register_modules)
        if tile.lock_count <= 0 or tile.lock_count > _UINT8_MAXIMUM:
            raise ValueError(f"{tile.kind.value}: invalid lock count")
        if (
            tile.lock_value_minimum < _INT8_MINIMUM
            or tile.lock_value_maximum > _INT8_MAXIMUM
        ):
            raise ValueError(f"{tile.kind.value}: lock-value native carrier overflows")
        if tile.lock_value_minimum > 0 or tile.lock_value_maximum <= 0:
            raise ValueError(f"{tile.kind.value}: invalid lock domain")
        if (
            not 0 < tile.packet_arbiter_count <= 8
            or not 0 < tile.packet_master_select_count <= 4
        ):
            raise ValueError(f"{tile.kind.value}: invalid packet-switch resources")
        _validate_tile_memory(tile, family.row_shift)
        _validate_load_window_resources(family, tile)
        _validate_dma(tile)
    if covered_rows != set(range(family.row_count)):
        raise ValueError(f"{family.key}: tile rows do not cover the array")
    resource_totals = tuple(tile_resource_totals(family, tile) for tile in family.tiles)
    if any(
        value > _UINT32_MAXIMUM
        for totals in resource_totals
        for value in (
            totals.physical_tile_count,
            totals.lock_count,
            totals.dma_channel_count_per_direction,
            totals.dma_buffer_descriptor_count,
        )
    ):
        raise ValueError(f"{family.key}: aggregate tile resource carrier overflows")

    # Every physical channel consumes at least one unique directional DMA
    # endpoint or lock pair. Counting every family resource produces a
    # planner-policy-independent upper bound for all result-table carriers.
    maximum_channel_capacity = max(tile.lock_value_maximum for tile in family.tiles)
    maximum_dma_endpoint_count = sum(
        totals.dma_channel_count_per_direction * 2 for totals in resource_totals
    )
    maximum_lock_pair_count = sum(totals.lock_count // 2 for totals in resource_totals)
    maximum_channel_count = maximum_dma_endpoint_count + maximum_lock_pair_count
    maximum_route_count = maximum_dma_endpoint_count * (
        family.column_count + family.row_count
    )
    maximum_worker_storage_count = (
        sum(totals.physical_tile_count for totals in resource_totals) * _UINT8_MAXIMUM
    )
    if any(
        value > _UINT32_MAXIMUM
        for value in (
            maximum_channel_count,
            maximum_channel_count * maximum_channel_capacity,
            maximum_channel_count * 2,
            maximum_route_count,
            maximum_worker_storage_count,
        )
    ):
        raise ValueError(f"{family.key}: physical-plan carrier overflows")
    event_modules = [event.module for event in family.events]
    if not set(event_modules).issubset(modules) or len(event_modules) != len(
        set(event_modules)
    ):
        raise ValueError(f"{family.key}: event module is unavailable or duplicated")
    if any(
        event.event_count <= 0
        or event.named_event_count <= 0
        or event.named_event_count > event.event_count
        or event.event_count > _UINT16_MAXIMUM
        or event.named_event_count > _UINT16_MAXIMUM
        for event in family.events
    ):
        raise ValueError(f"{family.key}: invalid event domain")
    _validate_stream_ports(family)
    _validate_registers(family)
    for tile in family.tiles:
        _validate_dma_register_fields(family, tile)


def maximum_encoded_dma_transfer_length(tile: TileFacts) -> int:
    """Returns the largest transfer-length field value for ``tile``."""
    dma = tile.dma
    if dma is None:
        raise ValueError(f"{tile.kind.value}: tile has no DMA engine")
    return (1 << dma.transfer_length_field.bit_width) - 1


def register_field_count(family: ArrayFamily) -> int:
    """Returns the semantic register-field count in ``family``."""
    return sum(len(pattern.fields) for pattern in family.registers)
