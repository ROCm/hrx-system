# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Decodes an AIE2P core trace buffer written in mode 0 (event-time).

The grammar and timing follow mlir-aie's python/utils/trace (utils.py
convert_to_commands, parse.py timer loop); each 32-bit word is read MSB-first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

# Event_Sync: the 18-bit delta field wrapped once.
EVENT_SYNC_CYCLES = 1 << 18

# Slot order of kEventSlots in aie2p/array/program.c.
DEFAULT_SLOT_EVENTS = (
    "core_active",
    "core_disabled",
    "core_memory_stall",
    "core_stream_stall",
    "core_lock_stall",
    "core_instr_lock_acquire_req",
    "core_instr_lock_release_req",
    "none",
)


@dataclass
class TraceEvent:
    slot: int
    cycle: int
    name: str = ""


@dataclass
class DecodedTrace:
    events: list[TraceEvent] = field(default_factory=list)
    start_timer_value: int | None = None


class TraceStreamError(ValueError):
    pass


def _slot_names(events: int) -> list[int]:
    return [i for i in range(8) if (events >> i) & 1]


def _reassemble_trace_words(data: bytes) -> bytes:
    word_count = len(data) // 4
    reassembled = bytearray(word_count * 4)
    for i in range(word_count):
        w = data[4 * i : 4 * i + 4]
        reassembled[4 * i : 4 * i + 4] = bytes((w[3], w[2], w[1], w[0]))
    return bytes(reassembled)


def decode_mode0_trace(data: bytes) -> DecodedTrace:
    """Decodes one tile's trace buffer into events with cumulative cycles."""
    result = DecodedTrace()
    data = _reassemble_trace_words(data)
    cursor = 0
    n = len(data)
    cycle = 0
    last_command: tuple[list[int], int] | None = None

    def emit(slot: int, delta: int) -> None:
        nonlocal cycle
        cycle += delta
        result.events.append(TraceEvent(slot, cycle, DEFAULT_SLOT_EVENTS[slot]))

    def emit_command(slots: list[int], delta: int) -> None:
        nonlocal last_command
        # A record spans one tick plus its delta.
        for i, slot in enumerate(slots):
            emit(slot, delta + 1 if i == 0 else 0)
        last_command = (slots, delta)

    def replay_last(repeats: int) -> None:
        nonlocal cycle
        if last_command is None:
            return
        slots, delta = last_command
        # Repeating a zero-delta record only advances time.
        if delta == 0:
            cycle += repeats
            return
        for _ in range(repeats):
            for i, slot in enumerate(slots):
                emit(slot, delta + 1 if i == 0 else 0)

    while cursor < n:
        b = data[cursor]
        if (b & 0b11111011) == 0b11110000:
            if cursor + 8 > n:
                raise TraceStreamError("truncated Start record")
            timer_value = 0
            for i in range(7):
                timer_value = (timer_value << 8) | data[cursor + 1 + i]
            result.start_timer_value = timer_value
            cycle = timer_value
            cursor += 8
            continue
        # DC padding
        if (b & 0b11111100) == 0b11011100:
            cursor += 4
            continue
        # Skip
        if b == 0b11111110:
            cursor += 1
            continue
        # Event_Sync
        if b == 0b11111111:
            cycle += EVENT_SYNC_CYCLES
            cursor += 1
            continue
        # Repeat0
        if (b & 0b11110000) == 0b11100000:
            repeats = b & 0b1111
            cursor += 1
            replay_last(repeats)
            continue
        if (b & 0b11111100) == 0b11011000:
            if cursor + 2 > n:
                raise TraceStreamError("truncated Repeat1 record")
            repeats = ((b & 0b11) << 8) | data[cursor + 1]
            cursor += 2
            replay_last(repeats)
            continue
        # Single0
        if (b & 0b10000000) == 0:
            slot = (b >> 4) & 0b111
            delta = b & 0b1111
            emit_command([slot], delta)
            cursor += 1
            continue
        if (b & 0b11100000) == 0b10000000:
            if cursor + 2 > n:
                raise TraceStreamError("truncated Single1 record")
            slot = (b >> 2) & 0b111
            delta = ((b & 0b11) << 8) | data[cursor + 1]
            emit_command([slot], delta)
            cursor += 2
            continue
        if (b & 0b11100000) == 0b10100000:
            if cursor + 3 > n:
                raise TraceStreamError("truncated Single2 record")
            slot = (b >> 2) & 0b111
            delta = ((b & 0b11) << 16) | (data[cursor + 1] << 8) | data[cursor + 2]
            emit_command([slot], delta)
            cursor += 3
            continue
        if (b & 0b11110000) == 0b11000000:
            if cursor + 2 > n:
                raise TraceStreamError("truncated Multiple0 record")
            mask = ((b & 0b1111) << 4) | (data[cursor + 1] >> 4)
            delta = data[cursor + 1] & 0b1111
            emit_command(_slot_names(mask), delta)
            cursor += 2
            continue
        if (b & 0b11111100) == 0b11010000:
            if cursor + 3 > n:
                raise TraceStreamError("truncated Multiple1 record")
            mask = ((b & 0b11) << 6) | (data[cursor + 1] >> 2)
            delta = ((data[cursor + 1] & 0b11) << 8) | data[cursor + 2]
            emit_command(_slot_names(mask), delta)
            cursor += 3
            continue
        if (b & 0b11111100) == 0b11010100:
            if cursor + 4 > n:
                raise TraceStreamError("truncated Multiple2 record")
            mask = ((b & 0b11) << 6) | (data[cursor + 1] >> 2)
            delta = (
                ((data[cursor + 1] & 0b11) << 16)
                | (data[cursor + 2] << 8)
                | data[cursor + 3]
            )
            emit_command(_slot_names(mask), delta)
            cursor += 4
            continue
        # Unassigned code: stop rather than misparse it.
        break

    return result


def to_json(trace: DecodedTrace) -> dict:
    return {
        "start_timer_value": trace.start_timer_value,
        "events": [
            {"cycle": e.cycle, "slot": e.slot, "event": e.name}
            for e in trace.events
        ],
    }


def to_chrome_trace_json(trace: DecodedTrace, tile_name: str = "core") -> dict:
    """Chrome/Perfetto instant events; `ts` is in core cycles."""
    return {
        "traceEvents": [
            {
                "name": e.name,
                "cat": "trace",
                "ph": "i",
                "ts": e.cycle,
                "pid": 0,
                "tid": tile_name,
                "s": "t",
            }
            for e in trace.events
        ]
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Raw trace buffer file (binary).")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument(
        "--chrome-trace",
        action="store_true",
        help="Emit Chrome/Perfetto trace-event JSON instead of the plain form.",
    )
    parser.add_argument(
        "--tile-name",
        default="core",
        help="Label for the traced tile in --chrome-trace output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    with open(args.input, "rb") as f:
        data = f.read()
    trace = decode_mode0_trace(data)
    output = (
        to_chrome_trace_json(trace, args.tile_name)
        if args.chrome_trace
        else to_json(trace)
    )
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
