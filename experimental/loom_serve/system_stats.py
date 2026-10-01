# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read-only Linux system measurements for the model-runner observer.

Inventory records describe source paths, units and runtime-power guards once.
Samples use that inventory's stable channel order. A failed or suspended source
has a null reading and a reason, never a plausible zero or a stale value.
"""

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Channel:
    # Exact kernel-provided source, retained in the inventory for attribution.
    path: Path
    # Human-readable device and channel identity, not a claim of wall power.
    label: str
    # Physical unit after scaling, or "text" for policy/state attributes.
    unit: str
    # Conversion from the kernel ABI unit; None preserves opaque policy text.
    scale: float | None = None
    # Runtime state read before a device attribute that could wake the device.
    runtime_status: Path | None = None


HWMON = re.compile(
    r"(temp|power|fan|in|curr|freq|energy)(\d+)_"
    r"(input|average|cap|cap_min|cap_max|crit|max|min|alarm)$"
)
UNITS = {
    "temp": ("degC", 0.001),
    "power": ("W", 0.000001),
    "fan": ("rpm", 1),
    "in": ("V", 0.001),
    "curr": ("A", 0.001),
    "freq": ("MHz", 0.000001),
    "energy": ("J", 0.000001),
}


def read_text(path):
    return path.read_text().strip()


def runtime_guard(path):
    for ancestor in path.resolve().parents:
        candidate = ancestor / "power/runtime_status"
        if candidate.is_file() and read_text(candidate) != "unsupported":
            return candidate
    return None


class SystemStats:
    def __init__(self, proc=Path("/proc"), sys=Path("/sys")):
        self.proc = proc
        self.sys = sys
        self.channels = []
        self.inventory_errors = []
        self.previous_cpu = None
        self.previous_process = None
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self._discover()

    def _add(self, path, label, unit="text", scale=None, guard=None):
        if path.is_file():
            self.channels.append(Channel(path, label, unit, scale, guard))

    def _discover(self):
        for directory in sorted((self.sys / "class/hwmon").glob("hwmon*")):
            try:
                name = read_text(directory / "name")
                for path in sorted(directory.iterdir()):
                    match = HWMON.fullmatch(path.name)
                    if not match:
                        continue
                    kind, index, attribute = match.groups()
                    label_path = directory / f"{kind}{index}_label"
                    label = (
                        read_text(label_path)
                        if label_path.is_file()
                        else f"{kind}{index}"
                    )
                    unit, scale = UNITS[kind]
                    if attribute == "alarm":
                        unit, scale = "bool", 1
                    self._add(
                        path,
                        f"{name}/{label}/{attribute}",
                        unit,
                        scale,
                        runtime_guard(path),
                    )
            except OSError as error:
                self.inventory_errors.append(
                    {"path": str(directory), "error": str(error)}
                )
        for directory in sorted(
            (self.sys / "devices/system/cpu/cpufreq").glob("policy*")
        ):
            for name in (
                "scaling_driver",
                "scaling_governor",
                "energy_performance_preference",
                "affected_cpus",
            ):
                self._add(directory / name, f"cpu/{directory.name}/{name}")
            for name in (
                "scaling_cur_freq",
                "scaling_min_freq",
                "scaling_max_freq",
                "cpuinfo_max_freq",
                "bios_limit",
            ):
                self._add(
                    directory / name, f"cpu/{directory.name}/{name}", "MHz", 0.001
                )
        for relative in (
            "devices/system/cpu/amd_pstate/status",
            "devices/system/cpu/cpufreq/boost",
            "firmware/acpi/platform_profile",
            "firmware/acpi/platform_profile_choices",
        ):
            self._add(self.sys / relative, relative)
        for device_class, pattern in (("drm", "card[0-9]*"), ("accel", "accel[0-9]*")):
            for directory in sorted((self.sys / "class" / device_class).glob(pattern)):
                if not re.fullmatch(r"(?:card|accel)\d+", directory.name):
                    continue
                device = directory / "device"
                guard = device / "power/runtime_status"
                guard = guard if guard.is_file() else None
                for name in (
                    "runtime_status",
                    "control",
                    "autosuspend_delay_ms",
                    "runtime_active_time",
                    "runtime_suspended_time",
                ):
                    self._add(device / "power" / name, f"{directory.name}/power/{name}")
                for name in (
                    "power_dpm_force_performance_level",
                    "power_dpm_state",
                    "pp_dpm_sclk",
                    "pp_dpm_mclk",
                    "current_link_speed",
                    "current_link_width",
                ):
                    self._add(device / name, f"{directory.name}/{name}", guard=guard)
                for name, unit in (
                    ("gpu_busy_percent", "%"),
                    ("mem_busy_percent", "%"),
                    ("mem_info_vram_total", "bytes"),
                    ("mem_info_vram_used", "bytes"),
                    ("mem_info_gtt_total", "bytes"),
                    ("mem_info_gtt_used", "bytes"),
                ):
                    self._add(device / name, f"{directory.name}/{name}", unit, 1, guard)
        for directory in sorted((self.sys / "class/powercap").glob("*")):
            for name, unit, scale in (
                ("energy_uj", "J", 0.000001),
                ("max_energy_range_uj", "J", 0.000001),
                ("constraint_0_power_limit_uw", "W", 0.000001),
                ("constraint_1_power_limit_uw", "W", 0.000001),
                ("enabled", "bool", 1),
                ("name", "text", None),
            ):
                self._add(directory / name, f"{directory.name}/{name}", unit, scale)

    def inventory(self):
        return {
            "event": "system_inventory",
            "platform": "linux",
            "channels": [
                {
                    "path": str(channel.path),
                    "label": channel.label,
                    "unit": channel.unit,
                    "runtime_status": str(channel.runtime_status)
                    if channel.runtime_status
                    else None,
                }
                for channel in self.channels
            ],
            "errors": self.inventory_errors,
        }

    def _host(self):
        memory = {}
        for line in read_text(self.proc / "meminfo").splitlines():
            key, value = line.split(":", 1)
            fields = value.split()
            if fields and len(fields) == 2 and fields[1] == "kB":
                memory[key] = int(fields[0]) * 1024
        fields = read_text(self.proc / "stat").splitlines()[0].split()
        if fields[0] != "cpu" or len(fields) < 9:
            raise ValueError("missing aggregate CPU counters")
        ticks = [int(value) for value in fields[1:9]]
        total, idle = sum(ticks), ticks[3] + ticks[4]
        busy = None
        if self.previous_cpu and total > self.previous_cpu[0]:
            busy = 100 * (
                1 - (idle - self.previous_cpu[1]) / (total - self.previous_cpu[0])
            )
        self.previous_cpu = (total, idle)
        return {
            "memory_bytes": memory,
            "cpu_busy_percent": busy,
            "load_average": [
                float(value) for value in read_text(self.proc / "loadavg").split()[:3]
            ],
        }

    def _process(self, pid):
        # comm may itself contain spaces and parentheses. The final ')' ends it.
        text = read_text(self.proc / str(pid) / "stat")
        fields = text[text.rindex(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        now = time.monotonic_ns()
        percent = None
        if self.previous_process and ticks >= self.previous_process[1]:
            elapsed = now - self.previous_process[0]
            if elapsed:
                percent = (
                    (ticks - self.previous_process[1])
                    / self.clock_ticks
                    * 1e11
                    / elapsed
                )
        self.previous_process = (now, ticks)
        # CPU percent follows top: one fully busy CPU is 100%, not 100 / CPUs.
        return {
            "pid": pid,
            "cpu_percent": percent,
            "resident_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            "threads": int(fields[17]),
        }

    def sample(self, pid, log_directory):
        values = []
        unavailable = {}
        runtime_states = {}
        for index, channel in enumerate(self.channels):
            value = None
            try:
                guard = channel.runtime_status
                if guard:
                    if guard not in runtime_states:
                        runtime_states[guard] = read_text(guard)
                    if runtime_states[guard] in ("suspended", "suspending", "resuming"):
                        unavailable[str(index)] = f"runtime_{runtime_states[guard]}"
                        values.append(None)
                        continue
                raw = read_text(channel.path)
                value = raw if channel.scale is None else int(raw) * channel.scale
            except (OSError, ValueError) as error:
                unavailable[str(index)] = str(error)
            values.append(value)
        result = {
            "event": "system",
            "values": values,
            "unavailable": unavailable,
            "errors": {},
        }
        for name, operation in (
            ("host", self._host),
            ("process", lambda: self._process(pid)),
        ):
            try:
                result[name] = operation()
            except (OSError, ValueError, IndexError) as error:
                result["errors"][name] = str(error)
        for resource in ("cpu", "memory", "io"):
            try:
                pressure = {}
                for line in read_text(self.proc / "pressure" / resource).splitlines():
                    kind, *fields = line.split()
                    pressure[kind] = {
                        key: int(value) if key == "total" else float(value)
                        for key, value in (field.split("=", 1) for field in fields)
                    }
                result[f"{resource}_pressure"] = pressure
            except (OSError, ValueError) as error:
                result["errors"][f"{resource}_pressure"] = str(error)
        try:
            usage = shutil.disk_usage(log_directory)
            result["log_filesystem"] = {
                "total_bytes": usage.total,
                "free_bytes": usage.free,
            }
        except OSError as error:
            result["errors"]["log_filesystem"] = str(error)
        return result
