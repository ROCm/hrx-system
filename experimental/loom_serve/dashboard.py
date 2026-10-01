# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read-only terminal dashboard for observe.py JSONL, live or recorded.

The viewer never connects to a model or samples this machine. Its bounded state
contains the latest measurements, aggregate epoch counters and short histories.
--snapshot prints the same view; --at selects a recorded elapsed time.
"""

import argparse
import collections
import curses
import json
import math
import sys
import time
from pathlib import Path


def text(value):
    # Logs include application output. Neither terminal escapes nor wide glyphs
    # may change the display geometry or issue commands to the terminal.
    return "".join(
        character
        if " " <= character <= "~"
        else character.encode("unicode_escape").decode("ascii")
        for character in str(value)
    )


def number(value, suffix="", digits=1):
    return "--" if value is None else f"{value:,.{digits}f}{suffix}"


def size(value):
    return "--" if value is None else f"{value / 2**30:.2f} GiB"


def history(values, maximum=None):
    if not values:
        return "(waiting)"
    maximum = maximum or max(values) or 1
    levels = " .:-=+*#%@"
    return "".join(levels[min(9, max(0, int(value / maximum * 9)))] for value in values)


class RunView:
    def __init__(self):
        self.last = None
        self.latest = {}
        self.event_times = {}
        self.epoch_totals = collections.Counter()
        self.output_rates = collections.deque(maxlen=60)
        self.fill_history = collections.deque(maxlen=60)
        self.messages = collections.deque(maxlen=6)

    def apply(self, record):
        self.last = record
        data = record["data"]
        event = data["event"]
        if event in (
            "run",
            "run_end",
            "ready",
            "heartbeat",
            "epoch",
            "system",
            "system_inventory",
        ):
            self.latest[event] = data
            self.event_times[event] = record["elapsed_ns"]
        if event == "epoch":
            self.epoch_totals["epochs"] += 1
            for key in (
                "prefill_tokens",
                "decode_tokens",
                "selected_tokens_including_eos",
                "traversals",
                "model_ms",
                "token_capacity",
            ):
                self.epoch_totals[key] += data[key]
            self.fill_history.append(
                (data["prefill_tokens"] + data["decode_tokens"])
                / data["token_capacity"]
            )
        elif event == "heartbeat":
            self.output_rates.append(data["interval_output_tokens_per_second"])
        elif event == "diagnostic":
            self.messages.append(text(data["text"])[:512])
        elif event in ("admit", "complete", "cancel", "evict"):
            self.messages.append(text(json.dumps(data, separators=(",", ":")))[:512])


class LogReader:
    def __init__(self, stream):
        self.stream = stream
        self.sequence = 0
        self.elapsed_ns = 0
        self.pending = False

    def poll(self, view, through_ns=None, limit=512):
        count = 0
        while count < limit:
            offset = self.stream.tell()
            line = self.stream.readline()
            if not line.endswith(b"\n"):
                # A writer can be between write and flush, including halfway
                # through a UTF-8 character. Only complete records are parsed.
                self.stream.seek(offset)
                self.pending = bool(line)
                break
            self.pending = False
            try:
                record = json.loads(line)
                if record["version"] != 1:
                    raise ValueError("unsupported observer log version")
                if record["sequence"] != self.sequence:
                    raise ValueError(f"expected sequence {self.sequence}")
                elapsed_ns = record["elapsed_ns"]
                if not isinstance(elapsed_ns, int) or elapsed_ns < self.elapsed_ns:
                    raise ValueError("invalid monotonic elapsed timestamp")
                if not isinstance(record["unix_time_ns"], int):
                    raise ValueError("invalid wall timestamp")
                if not isinstance(record["data"]["event"], str):
                    raise ValueError("missing event name")
                if through_ns is not None and elapsed_ns > through_ns:
                    self.stream.seek(offset)
                    break
                view.apply(record)
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
                raise ValueError(
                    f"log record {self.sequence} at byte {offset}: {error}"
                ) from error
            self.sequence += 1
            self.elapsed_ns = elapsed_ns
            count += 1
        return count


def render(view, all_sensors=False, now_ns=None, width=None):
    if view.last is None:
        return ["LOOM SERVE  |  waiting for a complete log record"]
    latest = view.latest
    ended = latest.get("run_end")
    elapsed_ns = view.last["elapsed_ns"]
    lag = 0 if now_ns is None or ended else max(0, now_ns - view.last["unix_time_ns"])
    cursor_ns = elapsed_ns + lag
    state = (
        f"FINISHED exit={ended['return_code']}"
        if ended
        else "RECORDED (no run_end at cursor)"
        if now_ns is None
        else f"record age {lag / 1e9:.1f}s (no run_end)"
    )
    hostname = latest.get("run", {}).get("host", {}).get("node", "unknown host")
    lines = [
        f"LOOM SERVE  |  {text(hostname)}  |  +{elapsed_ns / 1e9:.1f}s  |  {state}"
    ]
    ready = latest.get("ready", {})
    lines.append(
        f"Scheduler {text(ready.get('scheduler', '--'))}  rows {ready.get('rows', '--')}"
        f"  packing {text(ready.get('packing', '--'))}  cached shapes {ready.get('shape_count', '--')}"
        f"  record #{view.last['sequence']}"
    )

    def age(event):
        when = view.event_times.get(event)
        return "--" if when is None else number((cursor_ns - when) / 1e9, "s")

    heartbeat = latest.get("heartbeat", {})
    epoch = latest.get("epoch", {})
    totals = view.epoch_totals
    lines.extend(
        [
            "",
            "SCHEDULING",
            f"Phase {text(heartbeat.get('phase', 'loading'))}  heartbeat age {age('heartbeat')}"
            f"  last completed epoch age {age('epoch')}",
            f"Rows active {heartbeat.get('active_rows', '--')}  prefill {heartbeat.get('prefill_rows', '--')}"
            f"  decode {heartbeat.get('decode_rows', '--')}  backpressured {heartbeat.get('backpressured_rows', '--')}",
            f"Interval rates: output {number(heartbeat.get('interval_output_tokens_per_second'), ' tok/s')}"
            f"  prefill {number(heartbeat.get('interval_prefill_tokens_per_second'), ' tok/s')}  (output includes EOS)",
            f"Totals: {totals['prefill_tokens']:,} prompt inputs + {totals['decode_tokens']:,} decode inputs"
            f" -> {totals['selected_tokens_including_eos']:,} selected outputs",
            f"Completed {totals['epochs']:,} epochs / {totals['traversals']:,} traversals"
            f"  model wall {totals['model_ms'] / 1000:.2f}s",
        ]
    )
    if epoch:
        useful = epoch["prefill_tokens"] + epoch["decode_tokens"]
        fill = useful / epoch["token_capacity"]
        filled = round(fill * 24)
        lines.append(
            f"Last epoch {epoch['epoch']}: {epoch['spans']} spans  "
            f"{epoch['prefill_tokens']}P + {epoch['decode_tokens']}D / {epoch['token_capacity']} slots"
            f"  [{('#' * filled).ljust(24, '.')}] {fill:.0%}  {epoch['model_ms']:.1f}ms"
        )
    lines.extend(
        [
            "Committed fill |"
            + history(view.fill_history, 1)
            + "|  0..100% (not GPU utilization)",
            "Output rate   |"
            + history(view.output_rates)
            + "|  peak "
            + number(max(view.output_rates, default=0), " tok/s"),
            "",
            f"SYSTEM  (sample age {age('system')})",
        ]
    )
    sample = latest.get("system", {})
    host = sample.get("host", {})
    memory = host.get("memory_bytes", {})
    process = sample.get("process", {})
    lines.append(
        f"CPU {number(host.get('cpu_busy_percent'), '%')}  load {text(host.get('load_average', '--'))}"
        f"  server {number(process.get('cpu_percent'), '%')} / {size(process.get('resident_bytes'))} RSS"
    )
    lines.append(
        f"RAM available {size(memory.get('MemAvailable'))} / {size(memory.get('MemTotal'))}"
        f"  swap free {size(memory.get('SwapFree'))} / {size(memory.get('SwapTotal'))}"
        f"  log disk free {size(sample.get('log_filesystem', {}).get('free_bytes'))}"
    )
    lines.append(
        "Pressure some/10s: "
        + "  ".join(
            f"{resource} {number(sample.get(resource + '_pressure', {}).get('some', {}).get('avg10'), '%')}"
            for resource in ("cpu", "memory", "io")
        )
    )
    inventory = latest.get("system_inventory", {})
    channels = inventory.get("channels", [])
    values = sample.get("values", [])
    if sample and len(values) != len(channels):
        raise ValueError("system sample does not match channel inventory")
    if not all_sensors:
        for name in (
            "scaling_governor",
            "energy_performance_preference",
            "scaling_cur_freq",
        ):
            matching = [
                value
                for channel, value in zip(channels, values)
                if channel["label"].startswith("cpu/")
                and channel["label"].endswith("/" + name)
            ]
            available = [value for value in matching if value is not None]
            if available:
                detail = (
                    f"{min(available):.0f}..{max(available):.0f} MHz"
                    if name == "scaling_cur_freq"
                    else ", ".join(sorted(set(available)))
                )
                lines.append(
                    f"CPU {name}: {text(detail)} ({len(available)}/{len(matching)} policies)"
                )
    for name, error in sample.get("errors", {}).items():
        lines.append(f"Unavailable {text(name)}: {text(error)}")
    for error in inventory.get("errors", []):
        lines.append(f"Discovery error: {text(error)}")
    sensor_start = len(lines)
    lines.extend(
        [
            "",
            "SENSORS / POWER"
            + f" | {len(sample.get('unavailable', {}))} unavailable"
            + (" | all channels" if all_sensors else " | a: all"),
        ]
    )
    sensor_rows = list(enumerate(zip(channels, values)))
    if not all_sensors:
        priorities = ("amdgpu/", "k10temp/", "nvme/", "firmware/", "card", "accel")
        sensor_rows.sort(
            key=lambda row: next(
                (
                    priority
                    for priority, prefix in enumerate(priorities)
                    if row[1][0]["label"].startswith(prefix)
                ),
                len(priorities),
            )
        )
    for index, (channel, value) in sensor_rows:
        label, unit = channel["label"], channel["unit"]
        interesting = (
            unit in ("degC", "W", "rpm")
            and label.endswith(("/input", "/average", "/cap"))
        ) or any(
            label.endswith("/" + name)
            for name in (
                "platform_profile",
                "runtime_status",
                "control",
                "gpu_busy_percent",
                "power_dpm_force_performance_level",
                "pp_dpm_sclk",
                "pp_dpm_mclk",
            )
        )
        if not all_sensors and not interesting:
            continue
        if (
            not all_sensors
            and isinstance(value, str)
            and label.endswith(("/pp_dpm_sclk", "/pp_dpm_mclk"))
        ):
            active = [line for line in value.splitlines() if "*" in line]
            if active:
                value = "; ".join(active)
        detail = (
            "unavailable: "
            + sample.get("unavailable", {}).get(str(index), "no reading")
            if value is None
            else (
                size(value)
                if unit == "bytes"
                else text(value)
                if unit == "text"
                else number(value, " " + unit)
            )
        )
        label_width = 48 if all_sensors else 29
        lines.append(
            f"[{index:3}] {text(label):{label_width}.{label_width}} {text(detail)}"
        )
        if all_sensors:
            lines.append("      " + text(channel["path"]))
    if width is not None and width >= 120 and not all_sensors:
        # Keep the device measurements beside the scheduler on wide terminals.
        # Narrow terminals retain the complete, vertically scrollable view.
        left_width = width - 53
        left, right = lines[2:sensor_start], lines[sensor_start:]
        lines = lines[:2] + [
            f"{(left[index] if index < len(left) else '')[:left_width]:{left_width}} | "
            + (right[index] if index < len(right) else "")
            for index in range(max(len(left), len(right)))
        ]
    lines.extend(["", "RECENT EVENTS"])
    lines.extend(view.messages or ["(none)"])
    lines = [text(line) for line in lines]
    return [line[:width] for line in lines] if width is not None else lines


def display(screen, reader, view, through_ns, all_sensors):
    screen.timeout(250)
    screen.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    offset = 0
    while True:
        reader.poll(view, through_ns)
        height, width = screen.getmaxyx()
        lines = render(
            view, all_sensors, time.time_ns() if through_ns is None else None, width
        )
        available = max(1, height - 2)
        offset = min(offset, max(0, len(lines) - available))
        screen.erase()
        for row, line in enumerate(lines[offset : offset + available]):
            if row < height - 1:
                style = curses.A_NORMAL
                if line.startswith(
                    ("LOOM SERVE", "SCHEDULING", "SYSTEM", "SENSORS", "RECENT")
                ):
                    style = curses.A_BOLD | (
                        curses.color_pair(1) if curses.has_colors() else 0
                    )
                screen.addnstr(row, 0, text(line), max(0, width - 1), style)
        if height > 1:
            screen.addnstr(
                height - 1,
                0,
                f"q quit | arrows/PgUp/PgDn scroll | a all sensors | {offset + 1}-{min(offset + available, len(lines))}/{len(lines)}",
                max(0, width - 1),
                curses.A_REVERSE,
            )
        screen.refresh()
        key = screen.getch()
        if key in (ord("q"), 27):
            return
        if key == ord("a"):
            all_sensors = not all_sensors
            offset = 0
        elif key in (curses.KEY_DOWN, ord("j")):
            offset += 1
        elif key in (curses.KEY_UP, ord("k")):
            offset = max(0, offset - 1)
        elif key == curses.KEY_NPAGE:
            offset += available
        elif key == curses.KEY_PPAGE:
            offset = max(0, offset - available)
        elif key == curses.KEY_HOME:
            offset = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--snapshot", action="store_true", help="Print a view and exit."
    )
    parser.add_argument(
        "--all-sensors",
        action="store_true",
        help="Show every channel with its source path.",
    )
    parser.add_argument(
        "--at", type=float, help="Stop at this recorded elapsed time in seconds."
    )
    parser.add_argument("--width", type=int, help="Snapshot layout width in columns.")
    arguments = parser.parse_args()
    if arguments.at is not None and (
        not math.isfinite(arguments.at) or arguments.at < 0
    ):
        parser.error("at must be finite and nonnegative")
    if arguments.width is not None and arguments.width < 1:
        parser.error("width must be positive")
    through_ns = None if arguments.at is None else int(arguments.at * 1e9)
    with arguments.log.open("rb") as stream:
        reader, view = LogReader(stream), RunView()
        if arguments.snapshot:
            while reader.poll(view, through_ns):
                pass
            print("\n".join(render(view, arguments.all_sensors, width=arguments.width)))
            if reader.pending:
                print(
                    "Trailing partial record was not included in this snapshot.",
                    file=sys.stderr,
                )
        else:
            if not sys.stdout.isatty():
                parser.error("interactive mode needs a terminal; use --snapshot")
            curses.wrapper(display, reader, view, through_ns, arguments.all_sensors)


if __name__ == "__main__":
    main()
