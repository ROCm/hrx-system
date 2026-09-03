# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validated physical-array and configuration-register source schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag
from itertools import product


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
    """Independently addressed configuration-register module."""

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
    bank_count: int
    program_base: int
    program_capacity: int
    load_windows: tuple[AddressWindow, ...]


@dataclass(frozen=True, slots=True)
class DmaEngineFacts:
    """Resource and field-width limits for one tile DMA engine."""

    buffer_descriptor_count: int
    channel_count_per_direction: int
    address_dimension_count: int
    address_maximum: int
    address_alignment: int
    step_size_bits: int
    wrap_bits: int
    iteration_bits: int
    task_queue_depth: int
    supports_compression: bool
    supports_padding: bool
    supports_out_of_order: bool
    supports_tokens: bool
    supports_repeat: bool
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
    register_modules: tuple[RegisterModule, ...]
    dma: DmaEngineFacts | None


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


@dataclass(frozen=True, slots=True)
class RegisterDimension:
    """One indexed dimension in a regular register-address pattern."""

    name: str
    count: int
    stride: int


@dataclass(frozen=True, slots=True)
class RegisterField:
    """One semantically writable field within a 32-bit register."""

    name: str
    least_significant_bit: int
    bit_width: int
    is_signed: bool = False

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
    access: RegisterAccess
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


def _validate_tile_memory(tile: TileFacts) -> None:
    memory = tile.memory
    if memory.local_base < 0 or memory.local_capacity < 0 or memory.bank_count < 0:
        raise ValueError(f"{tile.kind.value}: invalid local-memory geometry")
    if memory.local_capacity == 0 and memory.bank_count != 0:
        raise ValueError(f"{tile.kind.value}: empty local memory declares banks")
    if memory.local_capacity and (
        memory.bank_count == 0 or memory.local_capacity % memory.bank_count
    ):
        raise ValueError(f"{tile.kind.value}: local-memory banks do not partition")
    if memory.program_base < 0 or memory.program_capacity < 0:
        raise ValueError(f"{tile.kind.value}: invalid program-memory geometry")
    if tile.kind is not TileKind.COMPUTE and memory.program_capacity != 0:
        raise ValueError(f"{tile.kind.value}: non-compute tile has program memory")
    names = [window.name for window in memory.load_windows]
    if len(names) != len(set(names)):
        raise ValueError(f"{tile.kind.value}: duplicate load-window name")
    for index, window in enumerate(memory.load_windows):
        if (
            not window.name
            or window.base < 0
            or window.capacity <= 0
            or window.lock_selector_base < 0
        ):
            raise ValueError(f"{tile.kind.value}: invalid load window")
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


def _validate_lock_windows(family: ArrayFamily, tile: TileFacts) -> None:
    tile_kinds = {row.kind: row for row in family.tiles}
    lock_ranges: list[tuple[int, int, str]] = []
    for window in tile.memory.load_windows:
        owner = tile_kinds.get(window.owner_kind)
        if owner is None:
            raise ValueError(
                f"{tile.kind.value}.{window.name}: lock owner kind is unavailable"
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
        dma.channel_count_per_direction,
        dma.address_dimension_count,
        dma.address_maximum,
        dma.address_alignment,
        dma.step_size_bits,
        dma.wrap_bits,
        dma.iteration_bits,
        dma.task_queue_depth,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError(f"{tile.kind.value}: invalid DMA resource limit")
    if dma.address_alignment & (dma.address_alignment - 1):
        raise ValueError(f"{tile.kind.value}: DMA alignment is not a power of two")
    if dma.address_maximum % dma.address_alignment:
        raise ValueError(f"{tile.kind.value}: DMA address range is misaligned")


def _validate_stream_ports(family: ArrayFamily) -> None:
    keys = [(row.tile_kind, row.direction, row.port) for row in family.stream_ports]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate stream-port range")
    tile_kinds = {tile.kind for tile in family.tiles}
    for row in family.stream_ports:
        if row.tile_kind not in tile_kinds or row.ordinal < 0 or row.count <= 0:
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


def validate_array_family(family: ArrayFamily) -> None:
    """Validates one complete materialized physical-array fact set."""
    if not family.key or family.revision <= 0:
        raise ValueError("array-family identity is incomplete")
    if family.column_count <= 0 or family.row_count <= 0:
        raise ValueError(f"{family.key}: invalid array geometry")
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
        if tile.first_row < 0 or tile.row_count <= 0:
            raise ValueError(f"{tile.kind.value}: invalid tile rows")
        rows = set(range(tile.first_row, tile.first_row + tile.row_count))
        if max(rows) >= family.row_count or covered_rows & rows:
            raise ValueError(f"{tile.kind.value}: tile rows overlap or overflow")
        covered_rows |= rows
        if len(tile.register_modules) != len(set(tile.register_modules)):
            raise ValueError(f"{tile.kind.value}: duplicate register module")
        modules.update(tile.register_modules)
        if tile.lock_count <= 0 or tile.lock_value_minimum >= tile.lock_value_maximum:
            raise ValueError(f"{tile.kind.value}: invalid lock domain")
        _validate_tile_memory(tile)
        _validate_lock_windows(family, tile)
        _validate_dma(tile)
    if covered_rows != set(range(family.row_count)):
        raise ValueError(f"{family.key}: tile rows do not cover the array")
    event_modules = [event.module for event in family.events]
    if not set(event_modules).issubset(modules) or len(event_modules) != len(
        set(event_modules)
    ):
        raise ValueError(f"{family.key}: event module is unavailable or duplicated")
    if any(
        event.event_count <= 0
        or event.named_event_count <= 0
        or event.named_event_count > event.event_count
        for event in family.events
    ):
        raise ValueError(f"{family.key}: invalid event domain")
    _validate_stream_ports(family)
    _validate_registers(family)


def register_field_count(family: ArrayFamily) -> int:
    """Returns the semantic register-field count in ``family``."""
    return sum(len(pattern.fields) for pattern in family.registers)
