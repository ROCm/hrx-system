# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.amd.xdna.aie2p.trace_decoder import (
    EVENT_SYNC_CYCLES,
    decode_mode0_trace,
    to_chrome_trace_json,
    to_json,
)


# Expected cycles come from mlir-aie's parse.py timer loop.


def _pack_as_words(msb_first: bytes) -> bytes:
    """Turns MSB-first opcode bytes into the little-endian word-packed form
    decode_mode0_trace expects, padding to a word with no-op Skip bytes.
    """
    padded = msb_first + bytes([0xFE]) * (-len(msb_first) % 4)
    packed = bytearray(len(padded))
    for i in range(0, len(padded), 4):
        packed[i], packed[i + 1], packed[i + 2], packed[i + 3] = (
            padded[i + 3],
            padded[i + 2],
            padded[i + 1],
            padded[i],
        )
    return bytes(packed)


def test_single0_decodes_a_three_bit_slot_and_four_bit_delta() -> None:
    # 0b0_101_0011: slot=5, delta=3.
    trace = decode_mode0_trace(_pack_as_words(bytes([0x53])))
    assert [(e.slot, e.cycle) for e in trace.events] == [(5, 4)]


def test_single1_decodes_a_ten_bit_delta() -> None:
    # 0b100_010_01, 0b00101100: slot=2, delta=0b01_00101100=300.
    trace = decode_mode0_trace(_pack_as_words(bytes([0x89, 0x2C])))
    assert [(e.slot, e.cycle) for e in trace.events] == [(2, 301)]


def test_single2_decodes_an_eighteen_bit_delta() -> None:
    # 0b101_001_11, 0xFF, 0xFF: slot=1, delta=0b11_11111111_11111111=262143.
    trace = decode_mode0_trace(_pack_as_words(bytes([0xA7, 0xFF, 0xFF])))
    assert [(e.slot, e.cycle) for e in trace.events] == [(1, 0x40000)]


def test_multiple0_shares_one_delta_across_its_slot_mask() -> None:
    # 0b1100_0001, 0b0001_0010: mask=0b00010001 (slots 0 and 4), delta=2.
    trace = decode_mode0_trace(_pack_as_words(bytes([0xC1, 0x12])))
    assert [(e.slot, e.cycle) for e in trace.events] == [(0, 3), (4, 3)]


def test_event_sync_carries_a_full_eighteen_bit_range_not_the_field_width() -> None:
    # Event_Sync adds the delta field's range, 1 << 18.
    single0_event0_delta5 = bytes([0x05])
    trace = decode_mode0_trace(_pack_as_words(bytes([0xFF]) + single0_event0_delta5))
    assert EVENT_SYNC_CYCLES == 1 << 18
    assert [(e.slot, e.cycle) for e in trace.events] == [(0, EVENT_SYNC_CYCLES + 6)]


def test_repeat0_replays_the_previous_commands_delta_pattern() -> None:
    # Single0 slot=3 delta=4, then Repeat0 x2 replays the same delta.
    single0_slot3_delta4 = bytes([0x34])
    repeat0_two_times = bytes([0b1110_0010])
    trace = decode_mode0_trace(
        _pack_as_words(single0_slot3_delta4 + repeat0_two_times)
    )
    assert [(e.slot, e.cycle) for e in trace.events] == [(3, 5), (3, 10), (3, 15)]


def test_repeat_of_a_zero_delta_command_advances_cycle_without_new_events() -> None:
    # Repeating a zero-delta record advances time without new events.
    single0_slot2_delta0 = bytes([0x20])
    repeat0_five_times = bytes([0b1110_0101])
    single0_slot2_delta1 = bytes([0x21])
    trace = decode_mode0_trace(
        _pack_as_words(
            single0_slot2_delta0 + repeat0_five_times + single0_slot2_delta1
        )
    )
    assert [(e.slot, e.cycle) for e in trace.events] == [(2, 1), (2, 8)]


def test_start_record_seeds_the_cycle_counter_from_its_timer_value() -> None:
    start_timer_42 = bytes([0xF0]) + (42).to_bytes(7, "big")
    single0_event1_delta1 = bytes([0x11])
    trace = decode_mode0_trace(
        _pack_as_words(start_timer_42 + single0_event1_delta1)
    )
    assert trace.start_timer_value == 42
    assert [(e.slot, e.cycle) for e in trace.events] == [(1, 44)]


def test_empty_stream_decodes_to_no_events() -> None:
    trace = decode_mode0_trace(b"")
    assert trace.events == []
    assert trace.start_timer_value is None


def test_to_json_reports_slot_cycle_and_the_fixed_event_name() -> None:
    trace = decode_mode0_trace(_pack_as_words(bytes([0x53])))  # slot=5
    rendered = to_json(trace)
    assert rendered["events"] == [
        {"cycle": 4, "slot": 5, "event": "core_instr_lock_acquire_req"}
    ]


def test_to_chrome_trace_json_emits_one_instant_event_per_decoded_event() -> None:
    trace = decode_mode0_trace(_pack_as_words(bytes([0x53])))
    rendered = to_chrome_trace_json(trace, tile_name="core(0,2)")
    assert rendered["traceEvents"] == [
        {
            "name": "core_instr_lock_acquire_req",
            "cat": "trace",
            "ph": "i",
            "ts": 4,
            "pid": 0,
            "tid": "core(0,2)",
            "s": "t",
        }
    ]


# First 64 words of a traced single-worker copy captured on Strix (17f0:10),
# sha256 f498f0122f4dba4c354aea67f513be6318ec7c3f11fd063f595be46c0ff4d605.
_REAL_CAPTURE_PREFIX = bytes.fromhex(
    "00000200000000f028b7010010dc14a4fe00b7d810c213d810c1e10097daffdb"
    "000002000010c2000010c1e1ffdbffdb10c454d810c1e1000010c4000010c1e1"
    "000002000010c2e90010c1e1e10010c2fe0010c1ffdbffdb10c454d810c1e100"
    "000002000010c4000010c1e10010c2e9fe10c1e1ffdbffdbffdbffdbffdbffdb"
    "00000200ffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdb"
    "00000200ffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdb"
    "00000200ffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdb"
    "00000200ffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdbffdb"
)


def test_real_capture_prefix_decodes_the_documented_events() -> None:
    trace = decode_mode0_trace(_REAL_CAPTURE_PREFIX)
    assert trace.start_timer_value == 0x1B728
    assert len(trace.events) == 88
    assert [(e.slot, e.cycle, e.name) for e in trace.events[:8]] == [
        (0, 1, "core_active"),
        (0, 4, "core_active"),
        (0, 5, "core_active"),
        (0, 6, "core_active"),
        (1, 117765, "core_disabled"),
        (1, 117766, "core_disabled"),
        (0, 117950, "core_active"),
        (0, 117970, "core_active"),
    ]
    assert [(e.slot, e.cycle, e.name) for e in trace.events[-4:]] == [
        (0, 173118, "core_active"),
        (0, 173121, "core_active"),
        (0, 173122, "core_active"),
        (0, 173123, "core_active"),
    ]
