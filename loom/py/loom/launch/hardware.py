# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Local hardware facts and profile compatibility.

Detection is best effort: every probe is optional, an absent tool answers
nothing, and a machine with only a CPU still yields a usable description. The
result decides which device URI a launch defaults to and how a profile is
ranked for this machine; it never overrides a profile's own requirements.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from loom.launch.values import human_bytes

_GFX_ARCH = re.compile(r"\bgfx[0-9a-f]{3,6}\b")
_MEMINFO = "/proc/meminfo"
_CPUINFO = "/proc/cpuinfo"
_NPU_DEVICE = "/dev/accel/accel0"

# A runner answers one command's standard output, or nothing when unavailable.
Runner = Callable[[list[str]], "str | None"]


@dataclass(frozen=True)
class Device:
    """One execution device the launcher may select."""

    name: str
    driver: str
    uri: str
    gfx_arch: str | None = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "driver": self.driver,
            "uri": self.uri,
            "gfx_arch": self.gfx_arch,
        }


@dataclass(frozen=True)
class Hardware:
    """What this machine offers a Loom deployment."""

    system: str
    machine: str
    cpu: str | None = None
    total_memory_bytes: int | None = None
    devices: tuple[Device, ...] = ()
    npu: bool = False

    def gpu(self) -> Device | None:
        for device in self.devices:
            if device.driver == "amdgpu":
                return device
        return None

    def gfx_arch(self) -> str | None:
        device = self.gpu()
        return device.gfx_arch if device else None

    def default_device_uri(self) -> str:
        """The URI a single-device launch selects when none was requested."""
        device = self.gpu()
        return device.uri if device else "task"

    def describe(self) -> dict:
        return {
            "system": self.system,
            "machine": self.machine,
            "cpu": self.cpu,
            "total_memory_bytes": self.total_memory_bytes,
            "memory": human_bytes(self.total_memory_bytes),
            "gpu": self.gpu().describe() if self.gpu() else None,
            "npu": self.npu,
            "devices": [device.describe() for device in self.devices],
            "default_device_uri": self.default_device_uri(),
        }


@dataclass(frozen=True)
class Compatibility:
    """How well one profile fits this machine, with the evidence either way."""

    score: float
    fits_memory: bool | None = None
    device_match: bool | None = None
    notes: tuple[str, ...] = ()

    def describe(self) -> dict:
        return {
            "score": round(self.score, 3),
            "fits_memory": self.fits_memory,
            "device_match": self.device_match,
            "notes": list(self.notes),
        }


def detect(
    runner: Runner | None = None,
    files: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> Hardware:
    """Describes this machine, using only the probes that answer."""
    runner = runner or _run
    system = system if system is not None else platform.system()
    machine = machine if machine is not None else platform.machine()
    gpu_name, gfx_arch = _amd_gpu(runner, files)
    devices: list[Device] = []
    if gfx_arch or gpu_name:
        devices.append(
            Device(
                name=gpu_name or gfx_arch or "AMD GPU",
                driver="amdgpu",
                uri="amdgpu://",
                gfx_arch=gfx_arch,
            )
        )
    return Hardware(
        system=system,
        machine=machine,
        cpu=_cpu_name(files),
        total_memory_bytes=_memory_total(files),
        devices=tuple(devices),
        npu=_exists(files, _NPU_DEVICE),
    )


def evaluate(hardware: Hardware, profile: object) -> Compatibility:
    """Scores one profile for this machine; missing evidence is not failure."""
    notes: list[str] = []
    score = 0.4
    fits_memory: bool | None = None
    memory_bytes = getattr(profile, "memory_bytes", None)
    if isinstance(memory_bytes, int) and memory_bytes > 0:
        total = hardware.total_memory_bytes
        if total:
            fits_memory = memory_bytes <= int(total * 0.9)
            if fits_memory:
                score += 0.25
                notes.append(
                    f"resident {human_bytes(memory_bytes)} fits {human_bytes(total)}"
                )
            else:
                score -= 0.35
                notes.append(
                    f"resident {human_bytes(memory_bytes)} does not fit "
                    f"{human_bytes(total)}"
                )
        else:
            notes.append(f"resident {human_bytes(memory_bytes)}; total unknown")
    device_match: bool | None = None
    hint = getattr(profile, "gfx_arch", None)
    if hint:
        local = hardware.gfx_arch()
        device_match = hint == local
        if device_match:
            score += 0.2
            notes.append(f"measured on {hint}, matching this machine")
        else:
            score -= 0.15
            notes.append(
                f"measured on {hint}, this machine reports {local or 'no AMD GPU'}"
            )
    gates = getattr(profile, "gates", None) or {}
    if gates:
        not_passing = sorted(
            name
            for name, record in gates.items()
            if getattr(record, "status", None) != "pass"
        )
        if not_passing:
            score -= 0.1
            notes.append("gates not passing: " + ", ".join(not_passing))
        else:
            score += 0.15
            notes.append(f"all {len(gates)} recorded gates passed")
    return Compatibility(
        score=max(0.0, min(1.0, score)),
        fits_memory=fits_memory,
        device_match=device_match,
        notes=tuple(notes),
    )


def _read(files: Mapping[str, str] | None, path: str) -> str | None:
    if files is not None:
        return files.get(path)
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            return file.read()
    except OSError:
        return None


def _exists(files: Mapping[str, str] | None, path: str) -> bool:
    """Reports a path's presence without opening it: device nodes block."""
    if files is not None:
        return path in files
    try:
        return os.path.exists(path)
    except OSError:
        return False


def _memory_total(files: Mapping[str, str] | None) -> int | None:
    text = _read(files, _MEMINFO) or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * 1024
    return None


def _cpu_name(files: Mapping[str, str] | None) -> str | None:
    text = _read(files, _CPUINFO) or ""
    for line in text.splitlines():
        if line.startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip() or None
    return None


def _amd_gpu(
    runner: Runner, files: Mapping[str, str] | None
) -> tuple[str | None, str | None]:
    text = runner(["rocminfo"]) or ""
    gfx_arch: str | None = None
    name: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if name is None and stripped.startswith("Marketing Name:"):
            name = stripped.split(":", 1)[1].strip() or None
        if gfx_arch is None:
            match = _GFX_ARCH.search(stripped)
            if match:
                gfx_arch = match.group(0)
    if gfx_arch is None and name is None and _drm_amdgpu(files):
        name = "AMD GPU"
    return name, gfx_arch


def _drm_amdgpu(files: Mapping[str, str] | None) -> bool:
    if not files:
        return False
    for path, content in files.items():
        if path.endswith("device/uevent") and content and "amdgpu" in content:
            return True
    return False


def _run(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout
