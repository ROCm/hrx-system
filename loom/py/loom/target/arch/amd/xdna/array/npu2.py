# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete NPU2 array facts used by Strix-family AIE2P devices."""

from __future__ import annotations

from loom.target.arch.amd.xdna.array.model import (
    AddressWindow,
    Architecture,
    ArrayFamily,
    DmaEngineFacts,
    EventModuleFacts,
    Provenance,
    RegisterAccess,
    RegisterDimension,
    RegisterField,
    RegisterModule,
    RegisterPattern,
    StreamDirection,
    StreamPort,
    StreamPortRange,
    TileFacts,
    TileKind,
    TileMemoryFacts,
    validate_array_family,
)

# Pinned upstream sources used to reconstruct and independently compare this
# table. These identify the source revisions; they are not hashes of this file.
AIE_RT_SOURCE_COMMIT = "8849e208bdcc533b20a0ed3f95c1ce961dee9c3a"
MLIR_AIE_SOURCE_COMMIT = "db06374df9bf83d9fc557001ca213368aed15788"
REGISTER_DATABASE_VERSION = "AM025-2024-11-13-1.1"

_ARRAY_PROVENANCE = Provenance.AIE_RT | Provenance.MLIR_AIE
_REGISTER_PROVENANCE = Provenance.AIE_RT | Provenance.REGISTER_DATABASE


def _field(
    name: str,
    least_significant_bit: int,
    bit_width: int = 1,
    *,
    signed: bool = False,
) -> RegisterField:
    return RegisterField(name, least_significant_bit, bit_width, signed)


def _pattern(
    key: str,
    module: RegisterModule,
    base_offset: int,
    fields: tuple[RegisterField, ...],
    *dimensions: RegisterDimension,
    access: RegisterAccess = RegisterAccess.READ_WRITE,
) -> RegisterPattern:
    return RegisterPattern(
        key=key,
        module=module,
        base_offset=base_offset,
        access=access,
        dimensions=dimensions,
        fields=fields,
        provenance=_REGISTER_PROVENANCE,
    )


def _dma_buffer_descriptor_patterns(
    *,
    key: str,
    module: RegisterModule,
    base_offset: int,
    buffer_descriptor_count: int,
    words: tuple[tuple[RegisterField, ...], ...],
) -> tuple[RegisterPattern, ...]:
    dimension = RegisterDimension("buffer_descriptor", buffer_descriptor_count, 0x20)
    return tuple(
        _pattern(
            f"{key}.word{word_ordinal}",
            module,
            base_offset + word_ordinal * 4,
            fields,
            dimension,
        )
        for word_ordinal, fields in enumerate(words)
    )


_COMPUTE_DMA_BD_PATTERNS = _dma_buffer_descriptor_patterns(
    key="compute_memory.dma.bd",
    module=RegisterModule.COMPUTE_MEMORY,
    base_offset=0x1D000,
    buffer_descriptor_count=16,
    words=(
        (
            _field("base_address", 14, 14),
            _field("buffer_length", 0, 14),
        ),
        (
            _field("enable_compression", 31),
            _field("enable_packet", 30),
            _field("out_of_order_bd_id", 24, 6),
            _field("packet_id", 19, 5),
            _field("packet_type", 16, 3),
        ),
        (
            _field("d1_step_size", 13, 13),
            _field("d0_step_size", 0, 13),
        ),
        (
            _field("d1_wrap", 21, 8),
            _field("d0_wrap", 13, 8),
            _field("d2_step_size", 0, 13),
        ),
        (
            _field("iteration_current", 19, 6),
            _field("iteration_wrap", 13, 6),
            _field("iteration_step_size", 0, 13),
        ),
        (
            _field("tlast_suppress", 31),
            _field("next_bd", 27, 4),
            _field("use_next_bd", 26),
            _field("valid_bd", 25),
            _field("lock_release_value", 18, 7, signed=True),
            _field("lock_release_id", 13, 4),
            _field("lock_acquire_enable", 12),
            _field("lock_acquire_value", 5, 7, signed=True),
            _field("lock_acquire_id", 0, 4),
        ),
    ),
)

_MEMORY_DMA_BD_PATTERNS = _dma_buffer_descriptor_patterns(
    key="memory_tile.dma.bd",
    module=RegisterModule.MEMORY_TILE,
    base_offset=0xA0000,
    buffer_descriptor_count=48,
    words=(
        (
            _field("enable_packet", 31),
            _field("packet_type", 28, 3),
            _field("packet_id", 23, 5),
            _field("out_of_order_bd_id", 17, 6),
            _field("buffer_length", 0, 17),
        ),
        (
            _field("d0_zero_before", 26, 6),
            _field("next_bd", 20, 6),
            _field("use_next_bd", 19),
            _field("base_address", 0, 19),
        ),
        (
            _field("tlast_suppress", 31),
            _field("d0_wrap", 17, 10),
            _field("d0_step_size", 0, 17),
        ),
        (
            _field("d1_zero_before", 27, 5),
            _field("d1_wrap", 17, 10),
            _field("d1_step_size", 0, 17),
        ),
        (
            _field("enable_compression", 31),
            _field("d2_zero_before", 27, 4),
            _field("d2_wrap", 17, 10),
            _field("d2_step_size", 0, 17),
        ),
        (
            _field("d2_zero_after", 28, 4),
            _field("d1_zero_after", 23, 5),
            _field("d0_zero_after", 17, 6),
            _field("d3_step_size", 0, 17),
        ),
        (
            _field("iteration_current", 23, 6),
            _field("iteration_wrap", 17, 6),
            _field("iteration_step_size", 0, 17),
        ),
        (
            _field("valid_bd", 31),
            _field("lock_release_value", 24, 7, signed=True),
            _field("lock_release_id", 16, 8),
            _field("lock_acquire_enable", 15),
            _field("lock_acquire_value", 8, 7, signed=True),
            _field("lock_acquire_id", 0, 8),
        ),
    ),
)

_SHIM_DMA_BD_PATTERNS = _dma_buffer_descriptor_patterns(
    key="shim_noc.dma.bd",
    module=RegisterModule.SHIM_NOC,
    base_offset=0x1D000,
    buffer_descriptor_count=16,
    words=(
        (_field("buffer_length", 0, 32),),
        (_field("base_address_low", 2, 30),),
        (
            _field("enable_packet", 30),
            _field("out_of_order_bd_id", 24, 6),
            _field("packet_id", 19, 5),
            _field("packet_type", 16, 3),
            _field("base_address_high", 0, 16),
        ),
        (
            _field("secure_access", 30),
            _field("d0_wrap", 20, 10),
            _field("d0_step_size", 0, 20),
        ),
        (
            _field("burst_length", 30, 2),
            _field("d1_wrap", 20, 10),
            _field("d1_step_size", 0, 20),
        ),
        (
            _field("smid", 28, 4),
            _field("axi_cache", 24, 4),
            _field("axi_qos", 20, 4),
            _field("d2_step_size", 0, 20),
        ),
        (
            _field("iteration_current", 26, 6),
            _field("iteration_wrap", 20, 6),
            _field("iteration_step_size", 0, 20),
        ),
        (
            _field("tlast_suppress", 31),
            _field("next_bd", 27, 4),
            _field("use_next_bd", 26),
            _field("valid_bd", 25),
            _field("lock_release_value", 18, 7, signed=True),
            _field("lock_release_id", 13, 4),
            _field("lock_acquire_enable", 12),
            _field("lock_acquire_value", 5, 7, signed=True),
            _field("lock_acquire_id", 0, 4),
        ),
    ),
)


def _dma_channel_patterns(
    *,
    key: str,
    module: RegisterModule,
    s2mm_base: int,
    mm2s_base: int,
    queue_name: str,
    channel_count: int,
    bd_id_bits: int,
    shim: bool = False,
) -> tuple[RegisterPattern, ...]:
    channel = RegisterDimension("channel", channel_count, 0x8)
    s2mm_fields = (
        _field("fot_mode", 16, 2),
        _field("controller_id", 8, 8),
        _field("enable_out_of_order", 3),
        *(
            (_field("pause_stream", 2), _field("pause_memory", 1))
            if shim
            else (_field("decompression_enable", 4), _field("reset", 1))
        ),
    )
    mm2s_fields = (
        _field("controller_id", 8, 8),
        *(
            (_field("pause_stream", 2), _field("pause_memory", 1))
            if shim
            else (_field("compression_enable", 4), _field("reset", 1))
        ),
    )
    queue_fields = (
        _field("enable_token_issue", 31),
        _field("repeat_count", 16, 8),
        _field("start_bd_id", 0, bd_id_bits),
    )
    return (
        _pattern(f"{key}.s2mm.control", module, s2mm_base, s2mm_fields, channel),
        _pattern(
            f"{key}.s2mm.{queue_name}",
            module,
            s2mm_base + 4,
            queue_fields,
            channel,
            access=RegisterAccess.WRITE_ONLY,
        ),
        _pattern(f"{key}.mm2s.control", module, mm2s_base, mm2s_fields, channel),
        _pattern(
            f"{key}.mm2s.{queue_name}",
            module,
            mm2s_base + 4,
            queue_fields,
            channel,
            access=RegisterAccess.WRITE_ONLY,
        ),
    )


_DMA_CHANNEL_PATTERNS = (
    *_dma_channel_patterns(
        key="compute_memory.dma.channel",
        module=RegisterModule.COMPUTE_MEMORY,
        s2mm_base=0x1DE00,
        mm2s_base=0x1DE10,
        queue_name="start_queue",
        channel_count=2,
        bd_id_bits=4,
    ),
    *_dma_channel_patterns(
        key="memory_tile.dma.channel",
        module=RegisterModule.MEMORY_TILE,
        s2mm_base=0xA0600,
        mm2s_base=0xA0630,
        queue_name="start_queue",
        channel_count=6,
        bd_id_bits=6,
    ),
    *_dma_channel_patterns(
        key="shim_noc.dma.channel",
        module=RegisterModule.SHIM_NOC,
        s2mm_base=0x1D200,
        mm2s_base=0x1D210,
        queue_name="task_queue",
        channel_count=2,
        bd_id_bits=4,
        shim=True,
    ),
)


def _stream_register_patterns(
    *,
    key: str,
    module: RegisterModule,
    base_offset: int,
    master_count: int,
    slave_count: int,
) -> tuple[RegisterPattern, ...]:
    return (
        _pattern(
            f"{key}.stream.master_config",
            module,
            base_offset,
            (
                _field("enable", 31),
                _field("packet_enable", 30),
                _field("drop_header", 7),
                _field("configuration", 0, 7),
            ),
            RegisterDimension("master_port", master_count, 4),
        ),
        _pattern(
            f"{key}.stream.slave_config",
            module,
            base_offset + 0x100,
            (_field("enable", 31), _field("packet_enable", 30)),
            RegisterDimension("slave_port", slave_count, 4),
        ),
        _pattern(
            f"{key}.stream.slave_slot",
            module,
            base_offset + 0x200,
            (
                _field("packet_id", 24, 5),
                _field("packet_mask", 16, 5),
                _field("enable", 8),
                _field("master_select", 4, 2),
                _field("arbiter", 0, 3),
            ),
            RegisterDimension("slave_port", slave_count, 0x10),
            RegisterDimension("slot", 4, 4),
        ),
    )


_STREAM_REGISTER_PATTERNS = (
    *_stream_register_patterns(
        key="core",
        module=RegisterModule.CORE,
        base_offset=0x3F000,
        master_count=23,
        slave_count=25,
    ),
    *_stream_register_patterns(
        key="memory_tile",
        module=RegisterModule.MEMORY_TILE,
        base_offset=0xB0000,
        master_count=17,
        slave_count=18,
    ),
    *_stream_register_patterns(
        key="shim_noc",
        module=RegisterModule.SHIM_PL,
        base_offset=0x3F000,
        master_count=22,
        slave_count=23,
    ),
)


def _lock_pattern(
    key: str,
    module: RegisterModule,
    base_offset: int,
    count: int,
) -> RegisterPattern:
    return _pattern(
        f"{key}.lock.value",
        module,
        base_offset,
        (_field("value", 0, 6),),
        RegisterDimension("lock", count, 0x10),
    )


_FIXED_REGISTER_PATTERNS = (
    _lock_pattern("compute_memory", RegisterModule.COMPUTE_MEMORY, 0x1F000, 16),
    _lock_pattern("memory_tile", RegisterModule.MEMORY_TILE, 0xC0000, 64),
    _lock_pattern("shim_noc", RegisterModule.SHIM_NOC, 0x14000, 16),
    _pattern(
        "core.control",
        RegisterModule.CORE,
        0x32000,
        (_field("reset", 1), _field("enable", 0)),
    ),
    _pattern(
        "shim_noc.mux_config",
        RegisterModule.SHIM_NOC,
        0x1F000,
        (
            _field("south7", 14, 2),
            _field("south6", 12, 2),
            _field("south3", 10, 2),
            _field("south2", 8, 2),
        ),
    ),
    _pattern(
        "shim_noc.demux_config",
        RegisterModule.SHIM_NOC,
        0x1F004,
        (
            _field("south5", 10, 2),
            _field("south4", 8, 2),
            _field("south3", 6, 2),
            _field("south2", 4, 2),
        ),
    ),
)


def _stream_ranges(
    tile_kind: TileKind,
    direction: StreamDirection,
    rows: tuple[tuple[StreamPort, int], ...],
) -> tuple[StreamPortRange, ...]:
    ordinal = 0
    result: list[StreamPortRange] = []
    for port, count in rows:
        result.append(StreamPortRange(tile_kind, direction, port, ordinal, count))
        ordinal += count
    return tuple(result)


_STREAM_PORTS = (
    *_stream_ranges(
        TileKind.COMPUTE,
        StreamDirection.MASTER,
        (
            (StreamPort.CORE, 1),
            (StreamPort.DMA, 2),
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.FIFO, 1),
            (StreamPort.SOUTH, 4),
            (StreamPort.WEST, 4),
            (StreamPort.NORTH, 6),
            (StreamPort.EAST, 4),
        ),
    ),
    *_stream_ranges(
        TileKind.COMPUTE,
        StreamDirection.SLAVE,
        (
            (StreamPort.CORE, 1),
            (StreamPort.DMA, 2),
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.FIFO, 1),
            (StreamPort.SOUTH, 6),
            (StreamPort.WEST, 4),
            (StreamPort.NORTH, 4),
            (StreamPort.EAST, 4),
            (StreamPort.TRACE, 1),
        ),
    ),
    *_stream_ranges(
        TileKind.MEMORY,
        StreamDirection.MASTER,
        (
            (StreamPort.DMA, 6),
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.SOUTH, 4),
            (StreamPort.NORTH, 6),
        ),
    ),
    *_stream_ranges(
        TileKind.MEMORY,
        StreamDirection.SLAVE,
        (
            (StreamPort.DMA, 6),
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.SOUTH, 6),
            (StreamPort.NORTH, 4),
            (StreamPort.TRACE, 1),
        ),
    ),
    *_stream_ranges(
        TileKind.SHIM_NOC,
        StreamDirection.MASTER,
        (
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.FIFO, 1),
            (StreamPort.SOUTH, 6),
            (StreamPort.WEST, 4),
            (StreamPort.NORTH, 6),
            (StreamPort.EAST, 4),
        ),
    ),
    *_stream_ranges(
        TileKind.SHIM_NOC,
        StreamDirection.SLAVE,
        (
            (StreamPort.TILE_CONTROL, 1),
            (StreamPort.FIFO, 1),
            (StreamPort.SOUTH, 8),
            (StreamPort.WEST, 4),
            (StreamPort.NORTH, 4),
            (StreamPort.EAST, 4),
            (StreamPort.TRACE, 2),
        ),
    ),
)


NPU2_ARRAY_FAMILY = ArrayFamily(
    key="amd.xdna.npu2",
    revision=1,
    architecture=Architecture.AIE2P,
    column_count=8,
    row_count=6,
    column_shift=25,
    row_shift=20,
    address_generation_granularity_bits=32,
    tiles=(
        TileFacts(
            kind=TileKind.SHIM_NOC,
            first_row=0,
            row_count=1,
            memory=TileMemoryFacts(0, 0, 0, 0, 0, ()),
            lock_count=16,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.SHIM_NOC, RegisterModule.SHIM_PL),
            dma=DmaEngineFacts(
                buffer_descriptor_count=16,
                channel_count_per_direction=2,
                address_dimension_count=3,
                address_maximum=0x1000000000000,
                address_alignment=4,
                step_size_bits=20,
                wrap_bits=10,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=False,
                supports_padding=False,
                supports_out_of_order=True,
                supports_tokens=True,
                supports_repeat=True,
                supports_tlast_suppression=True,
            ),
        ),
        TileFacts(
            kind=TileKind.MEMORY,
            first_row=1,
            row_count=1,
            memory=TileMemoryFacts(
                local_base=0,
                local_capacity=512 * 1024,
                bank_count=8,
                program_base=0,
                program_capacity=0,
                load_windows=(
                    AddressWindow("west", 0x00000, 512 * 1024, -1, 0, TileKind.MEMORY),
                    AddressWindow("self", 0x80000, 512 * 1024, 0, 0, TileKind.MEMORY),
                    AddressWindow("east", 0x100000, 512 * 1024, 1, 0, TileKind.MEMORY),
                ),
            ),
            lock_count=64,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.MEMORY_TILE,),
            dma=DmaEngineFacts(
                buffer_descriptor_count=48,
                channel_count_per_direction=6,
                address_dimension_count=4,
                address_maximum=0x180000,
                address_alignment=4,
                step_size_bits=17,
                wrap_bits=10,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=True,
                supports_padding=True,
                supports_out_of_order=True,
                supports_tokens=True,
                supports_repeat=True,
                supports_tlast_suppression=True,
            ),
        ),
        TileFacts(
            kind=TileKind.COMPUTE,
            first_row=2,
            row_count=4,
            memory=TileMemoryFacts(
                local_base=0,
                local_capacity=64 * 1024,
                bank_count=4,
                program_base=0,
                program_capacity=16 * 1024,
                load_windows=(
                    AddressWindow("south", 0x40000, 64 * 1024, 0, -1, TileKind.COMPUTE),
                    AddressWindow("west", 0x50000, 64 * 1024, -1, 0, TileKind.COMPUTE),
                    AddressWindow("north", 0x60000, 64 * 1024, 0, 1, TileKind.COMPUTE),
                    AddressWindow("self", 0x70000, 64 * 1024, 0, 0, TileKind.COMPUTE),
                ),
            ),
            lock_count=16,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.CORE, RegisterModule.COMPUTE_MEMORY),
            dma=DmaEngineFacts(
                buffer_descriptor_count=16,
                channel_count_per_direction=2,
                address_dimension_count=3,
                address_maximum=0x20000,
                address_alignment=4,
                step_size_bits=13,
                wrap_bits=8,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=True,
                supports_padding=False,
                supports_out_of_order=True,
                supports_tokens=True,
                supports_repeat=True,
                supports_tlast_suppression=True,
            ),
        ),
    ),
    events=(
        EventModuleFacts(RegisterModule.CORE, 128, 127),
        EventModuleFacts(RegisterModule.COMPUTE_MEMORY, 128, 122),
        EventModuleFacts(RegisterModule.MEMORY_TILE, 161, 161),
        EventModuleFacts(RegisterModule.SHIM_PL, 128, 128),
    ),
    stream_ports=_STREAM_PORTS,
    registers=(
        *_COMPUTE_DMA_BD_PATTERNS,
        *_MEMORY_DMA_BD_PATTERNS,
        *_SHIM_DMA_BD_PATTERNS,
        *_DMA_CHANNEL_PATTERNS,
        *_STREAM_REGISTER_PATTERNS,
        *_FIXED_REGISTER_PATTERNS,
    ),
    provenance=_ARRAY_PROVENANCE,
)

validate_array_family(NPU2_ARRAY_FAMILY)
