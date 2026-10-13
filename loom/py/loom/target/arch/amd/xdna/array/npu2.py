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
    access: RegisterAccess = RegisterAccess.READ_WRITE,
) -> RegisterField:
    return RegisterField(name, least_significant_bit, bit_width, signed, access)


def _pattern(
    key: str,
    module: RegisterModule,
    base_offset: int,
    fields: tuple[RegisterField, ...],
    *dimensions: RegisterDimension,
) -> RegisterPattern:
    return RegisterPattern(
        key=key,
        module=module,
        base_offset=base_offset,
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


_COMPUTE_DMA_TRANSFER_LENGTH_FIELD = _field("buffer_length", 0, 14)
_MEMORY_DMA_TRANSFER_LENGTH_FIELD = _field("buffer_length", 0, 17)
_SHIM_DMA_TRANSFER_LENGTH_FIELD = _field("buffer_length", 0, 32)


_COMPUTE_DMA_BD_PATTERNS = _dma_buffer_descriptor_patterns(
    key="compute_memory.dma.bd",
    module=RegisterModule.COMPUTE_MEMORY,
    base_offset=0x1D000,
    buffer_descriptor_count=16,
    words=(
        (
            _field("base_address", 14, 14),
            _COMPUTE_DMA_TRANSFER_LENGTH_FIELD,
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
            _MEMORY_DMA_TRANSFER_LENGTH_FIELD,
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
        (_SHIM_DMA_TRANSFER_LENGTH_FIELD,),
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
    s2mm_status: int,
    mm2s_status: int,
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
        _field("enable_token_issue", 31, access=RegisterAccess.WRITE_ONLY),
        _field("repeat_count", 16, 8, access=RegisterAccess.WRITE_ONLY),
        _field("start_bd_id", 0, bd_id_bits, access=RegisterAccess.WRITE_ONLY),
    )
    # AIE2P's DMA wait-for-done contract observes every outstanding-task state,
    # including stalls while the channel-running bit is clear.
    status_fields = (
        _field("task_queue_size", 20, 3, access=RegisterAccess.READ_ONLY),
        _field("channel_running", 19, access=RegisterAccess.READ_ONLY),
        _field("stalled_lock_acquire", 2, access=RegisterAccess.READ_ONLY),
        _field("stalled_lock_release", 3, access=RegisterAccess.READ_ONLY),
    )
    s2mm_status_fields = (
        *status_fields,
        _field("stalled_stream_starvation", 4, access=RegisterAccess.READ_ONLY),
        _field("stalled_token_or_count_fifo", 5, access=RegisterAccess.READ_ONLY),
    )
    mm2s_status_fields = (
        *status_fields,
        _field("stalled_stream_backpressure", 4, access=RegisterAccess.READ_ONLY),
        _field("stalled_completion_token", 5, access=RegisterAccess.READ_ONLY),
    )
    status_channel = RegisterDimension("channel", channel_count, 4)
    return (
        _pattern(f"{key}.s2mm.control", module, s2mm_base, s2mm_fields, channel),
        _pattern(
            f"{key}.s2mm.{queue_name}",
            module,
            s2mm_base + 4,
            queue_fields,
            channel,
        ),
        _pattern(f"{key}.mm2s.control", module, mm2s_base, mm2s_fields, channel),
        _pattern(
            f"{key}.mm2s.{queue_name}",
            module,
            mm2s_base + 4,
            queue_fields,
            channel,
        ),
        _pattern(
            f"{key}.s2mm.status",
            module,
            s2mm_status,
            s2mm_status_fields,
            status_channel,
        ),
        _pattern(
            f"{key}.mm2s.status",
            module,
            mm2s_status,
            mm2s_status_fields,
            status_channel,
        ),
    )


_DMA_CHANNEL_PATTERNS = (
    *_dma_channel_patterns(
        key="compute_memory.dma.channel",
        module=RegisterModule.COMPUTE_MEMORY,
        s2mm_base=0x1DE00,
        mm2s_base=0x1DE10,
        s2mm_status=0x1DF00,
        mm2s_status=0x1DF10,
        queue_name="start_queue",
        channel_count=2,
        bd_id_bits=4,
    ),
    *_dma_channel_patterns(
        key="memory_tile.dma.channel",
        module=RegisterModule.MEMORY_TILE,
        s2mm_base=0xA0600,
        mm2s_base=0xA0630,
        s2mm_status=0xA0660,
        mm2s_status=0xA0680,
        queue_name="start_queue",
        channel_count=6,
        bd_id_bits=6,
    ),
    *_dma_channel_patterns(
        key="shim_noc.dma.channel",
        module=RegisterModule.SHIM_NOC,
        s2mm_base=0x1D200,
        mm2s_base=0x1D210,
        s2mm_status=0x1D220,
        mm2s_status=0x1D228,
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
        *(
            _pattern(
                f"{key}.stream.event_port_selection{word}",
                module,
                base_offset + 0xF00 + word * 4,
                tuple(
                    field
                    for slot in range(4)
                    for field in (
                        _field(f"port{word * 4 + slot}_master", slot * 8 + 5),
                        _field(f"port{word * 4 + slot}_id", slot * 8, 5),
                    )
                ),
            )
            for word in range(2)
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


def _event_register_patterns(
    *,
    key: str,
    module: RegisterModule,
    base_offset: int,
    event_bits: int,
    broadcast_switch_count: int,
) -> tuple[RegisterPattern, ...]:
    # The pinned aie-rt module definitions and AM025 register database share
    # this layout. Memory tiles have 8-bit events; the other modules use 7.
    # Only the core trace unit supports PC and instruction-execution modes.
    return (
        _pattern(
            f"{key}.timer_control",
            module,
            base_offset,
            (
                _field("reset", 31, access=RegisterAccess.WRITE_ONLY),
                _field("reset_event", 8, event_bits),
            ),
        ),
        _pattern(
            f"{key}.event_generate",
            module,
            base_offset + 0x08,
            (_field("event", 0, event_bits, access=RegisterAccess.WRITE_ONLY),),
        ),
        _pattern(
            f"{key}.event_broadcast",
            module,
            base_offset + 0x10,
            (_field("event", 0, event_bits),),
            RegisterDimension("broadcast_channel", 16, 4),
        ),
        *(
            _pattern(
                f"{key}.event_broadcast.{direction}.{action}",
                module,
                base_offset + 0x50 + direction_index * 0x10 + action_offset,
                (_field("channels", 0, 16, access=access),),
                RegisterDimension("broadcast_switch", broadcast_switch_count, 0x40),
            )
            for direction_index, direction in enumerate(
                ("south", "west", "north", "east")
            )
            for action, action_offset, access in (
                ("set", 0, RegisterAccess.WRITE_ONLY),
                ("clear", 4, RegisterAccess.WRITE_ONLY),
                ("value", 8, RegisterAccess.READ_ONLY),
            )
        ),
        _pattern(
            f"{key}.trace_control0",
            module,
            base_offset + 0xD0,
            (
                _field("stop_event", 24, event_bits),
                _field("start_event", 16, event_bits),
                *((_field("mode", 0, 2),) if module is RegisterModule.CORE else ()),
            ),
        ),
        _pattern(
            f"{key}.trace_control1",
            module,
            base_offset + 0xD4,
            (_field("packet_type", 12, 3), _field("packet_id", 0, 5)),
        ),
        _pattern(
            f"{key}.trace_status",
            module,
            base_offset + 0xD8,
            (
                _field("state", 8, 2, access=RegisterAccess.READ_ONLY),
                _field("mode", 0, 3, access=RegisterAccess.READ_ONLY),
            ),
        ),
        *(
            _pattern(
                f"{key}.trace_event{word}",
                module,
                base_offset + 0xE0 + word * 4,
                tuple(
                    _field(f"event{word * 4 + slot}", slot * 8, event_bits)
                    for slot in range(4)
                ),
            )
            for word in range(2)
        ),
        *(
            _pattern(
                f"{key}.timer_{name}",
                module,
                base_offset + offset,
                (_field("value", 0, 32, access=access),),
            )
            for name, offset, access in (
                ("trigger_low", 0xF0, RegisterAccess.READ_WRITE),
                ("trigger_high", 0xF4, RegisterAccess.READ_WRITE),
                ("low", 0xF8, RegisterAccess.READ_ONLY),
                ("high", 0xFC, RegisterAccess.READ_ONLY),
            )
        ),
        _pattern(
            f"{key}.combo_event_inputs",
            module,
            base_offset + 0x400,
            tuple(
                _field(f"event_{name}", index * 8, event_bits)
                for index, name in enumerate(("a", "b", "c", "d"))
            ),
        ),
        _pattern(
            f"{key}.combo_event_control",
            module,
            base_offset + 0x404,
            tuple(_field(f"combo{index}", index * 8, 2) for index in range(3)),
        ),
        _pattern(
            f"{key}.edge_event_control",
            module,
            base_offset + 0x408,
            tuple(
                field
                for index in range(2)
                for field in (
                    _field(f"event{index}", index * 16, event_bits),
                    _field(f"rising{index}", index * 16 + 9),
                    _field(f"falling{index}", index * 16 + 10),
                )
            ),
        ),
    )


_EVENT_REGISTER_PATTERNS = (
    *_event_register_patterns(
        key="core",
        module=RegisterModule.CORE,
        base_offset=0x34000,
        event_bits=7,
        broadcast_switch_count=1,
    ),
    *_event_register_patterns(
        key="compute_memory",
        module=RegisterModule.COMPUTE_MEMORY,
        base_offset=0x14000,
        event_bits=7,
        broadcast_switch_count=1,
    ),
    *_event_register_patterns(
        key="memory_tile",
        module=RegisterModule.MEMORY_TILE,
        base_offset=0x94000,
        event_bits=8,
        broadcast_switch_count=2,
    ),
    *_event_register_patterns(
        key="shim_pl",
        module=RegisterModule.SHIM_PL,
        base_offset=0x34000,
        event_bits=7,
        broadcast_switch_count=2,
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
            (StreamPort.TRACE, 2),
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
            (StreamPort.TRACE, 1),
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
            memory=TileMemoryFacts(
                local_base=0,
                local_capacity=0,
                local_load_base=0,
                bank_count=0,
                program_base=0,
                program_capacity=0,
                program_load_base=0,
                load_windows=(),
            ),
            lock_count=16,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.SHIM_NOC, RegisterModule.SHIM_PL),
            dma=DmaEngineFacts(
                buffer_descriptor_count=16,
                maximum_task_repeat_count=256,
                channel_count_per_direction=2,
                loopback_channel_count=0,
                address_dimension_count=3,
                address_maximum=0x1000000000000,
                address_alignment=4,
                address_encoding_shift=0,
                transfer_length_granularity=4,
                transfer_length_offset=0,
                transfer_length_field=_SHIM_DMA_TRANSFER_LENGTH_FIELD,
                memory_to_stream_port_base=3,
                memory_to_stream_port_stride=4,
                stream_to_memory_port_base=2,
                stream_to_memory_port_stride=1,
                step_size_bits=20,
                wrap_bits=10,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=False,
                supports_padding=False,
                supports_out_of_order=True,
                supports_tokens=True,
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
                local_load_base=0x80000,
                bank_count=8,
                program_base=0,
                program_capacity=0,
                program_load_base=0,
                load_windows=(
                    AddressWindow(
                        "west", 0x00000, 512 * 1024, 0, -1, 0, TileKind.MEMORY
                    ),
                    AddressWindow(
                        "self", 0x80000, 512 * 1024, 64, 0, 0, TileKind.MEMORY
                    ),
                    AddressWindow(
                        "east", 0x100000, 512 * 1024, 128, 1, 0, TileKind.MEMORY
                    ),
                ),
            ),
            lock_count=64,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.MEMORY_TILE,),
            dma=DmaEngineFacts(
                buffer_descriptor_count=48,
                maximum_task_repeat_count=256,
                channel_count_per_direction=6,
                # The memory switch connects only equal DMA channel indices.
                loopback_channel_count=6,
                address_dimension_count=4,
                address_maximum=0x180000,
                address_alignment=4,
                address_encoding_shift=2,
                transfer_length_granularity=4,
                transfer_length_offset=0,
                transfer_length_field=_MEMORY_DMA_TRANSFER_LENGTH_FIELD,
                memory_to_stream_port_base=0,
                memory_to_stream_port_stride=1,
                stream_to_memory_port_base=0,
                stream_to_memory_port_stride=1,
                step_size_bits=17,
                wrap_bits=10,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=True,
                supports_padding=True,
                supports_out_of_order=True,
                supports_tokens=True,
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
                local_load_base=0x70000,
                bank_count=4,
                program_base=0,
                program_capacity=16 * 1024,
                program_load_base=0x20000,
                load_windows=(
                    AddressWindow(
                        "south", 0x40000, 64 * 1024, 0, 0, -1, TileKind.COMPUTE
                    ),
                    AddressWindow(
                        "west", 0x50000, 64 * 1024, 16, -1, 0, TileKind.COMPUTE
                    ),
                    AddressWindow(
                        "north", 0x60000, 64 * 1024, 32, 0, 1, TileKind.COMPUTE
                    ),
                    AddressWindow(
                        "self", 0x70000, 64 * 1024, 48, 0, 0, TileKind.COMPUTE
                    ),
                ),
            ),
            lock_count=16,
            lock_value_minimum=-64,
            lock_value_maximum=63,
            register_modules=(RegisterModule.CORE, RegisterModule.COMPUTE_MEMORY),
            dma=DmaEngineFacts(
                buffer_descriptor_count=16,
                maximum_task_repeat_count=256,
                channel_count_per_direction=2,
                # The compute switch connects only equal DMA channel indices.
                loopback_channel_count=2,
                address_dimension_count=3,
                address_maximum=0x20000,
                address_alignment=4,
                address_encoding_shift=2,
                transfer_length_granularity=4,
                transfer_length_offset=0,
                transfer_length_field=_COMPUTE_DMA_TRANSFER_LENGTH_FIELD,
                memory_to_stream_port_base=0,
                memory_to_stream_port_stride=1,
                stream_to_memory_port_base=0,
                stream_to_memory_port_stride=1,
                step_size_bits=13,
                wrap_bits=8,
                iteration_bits=6,
                task_queue_depth=4,
                supports_compression=True,
                supports_padding=False,
                supports_out_of_order=True,
                supports_tokens=True,
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
        *_EVENT_REGISTER_PATTERNS,
        *_FIXED_REGISTER_PATTERNS,
    ),
    provenance=_ARRAY_PROVENANCE,
)

validate_array_family(NPU2_ARRAY_FAMILY)
