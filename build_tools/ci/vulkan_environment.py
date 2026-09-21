#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selects the CI render node and verifies Vulkan device zero uses that hardware.

The runner controller may name its comma-separated allocation in
PARENT_GPU_DEVICES. Prefer discrete hardware among the allocated devices, or
among all accessible devices when there is no explicit allocation.
Mesa selects the physical card by PCI address, independently of its product ID.
Khronos vulkaninfo supplies native device properties before the CTS runs.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderDevice:
    # Device node exposed to the job.
    path: Path
    # PCI domain, bus, device and function identifying this physical card.
    pci_address: str
    # Kernel-reported PCI product ID, used for diagnostics and identity checking.
    device_id: int
    # Device number from the opened render node, including major and minor.
    device_number: int

    @property
    def dri_prime(self) -> str:
        return f"pci-{self.pci_address.replace(':', '_').replace('.', '_')}!"


def discover_render_devices(sysfs_drm_path: Path, dri_path: Path) -> list[RenderDevice]:
    """Inventories sysfs devices and checks actual job access to AMD render nodes."""
    devices = []
    print(f"DRM render nodes (uid={os.getuid()}, gid={os.getgid()}):", flush=True)
    for entry in sorted(sysfs_drm_path.glob("renderD[0-9]*")):
        node_path = dri_path / entry.name
        try:
            pci_path = (entry / "device").resolve(strict=True)
            vendor_id = int((pci_path / "vendor").read_text(), 16)
            device_id = int((pci_path / "device").read_text(), 16)
            driver = (pci_path / "driver").resolve(strict=True).name
            print(
                f"  {node_path}: PCI {pci_path.name} "
                f"{vendor_id:04x}:{device_id:04x}, driver={driver}",
                flush=True,
            )
            if vendor_id != 0x1002:
                continue
            descriptor = os.open(node_path, os.O_RDWR | os.O_CLOEXEC)
            try:
                node_stat = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if not stat.S_ISCHR(node_stat.st_mode):
                raise RuntimeError(f"{node_path} is not a character device")
        except OSError as error:
            print(f"  {node_path}: unavailable: {error}", flush=True)
            continue
        devices.append(
            RenderDevice(node_path, pci_path.name, device_id, node_stat.st_rdev)
        )
    return devices


def render_device_candidates(
    devices: list[RenderDevice], allocation: str | None
) -> list[RenderDevice]:
    if allocation:
        accessible = {device.path: device for device in devices}
        candidates = []
        for path in allocation.split(","):
            device = accessible.get(Path(path))
            if device is None:
                raise RuntimeError(
                    f"PARENT_GPU_DEVICES={allocation!r} contains an inaccessible "
                    f"AMD render node {path!r}; refusing to test a different GPU"
                )
            candidates.append(device)
        return candidates
    if not devices:
        raise RuntimeError("No accessible AMD DRM render node is exposed to this job")
    return devices


def validate_vulkan_device(profile: dict, selected: RenderDevice) -> str:
    """Validates native identity and returns the driver's physical device type."""
    properties = profile["capabilities"]["device"]["properties"]
    device = properties["VkPhysicalDeviceProperties"]
    version = device["apiVersion"]
    print(
        f"Vulkan device 0: {device['deviceName']}, {device['deviceType']}, "
        f"PCI {device['vendorID']:04x}:{device['deviceID']:04x}, "
        f"API {version >> 22}.{(version >> 12) & 0x3FF}.{version & 0xFFF}",
        flush=True,
    )
    if device["deviceType"] not in (
        "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    ):
        raise RuntimeError("Vulkan device 0 is not real GPU hardware")
    if (device["vendorID"], device["deviceID"]) != (0x1002, selected.device_id):
        raise RuntimeError("Vulkan device 0 does not match the selected AMD PCI device")
    drm = properties.get("VkPhysicalDeviceDrmPropertiesEXT", {})
    if not drm.get("hasRender"):
        raise RuntimeError("Vulkan device 0 does not report a DRM render node")
    actual = (drm["renderMajor"], drm["renderMinor"])
    expected = (os.major(selected.device_number), os.minor(selected.device_number))
    if actual != expected:
        raise RuntimeError(
            f"Vulkan device 0 uses DRM {actual[0]}:{actual[1]}, "
            f"expected {expected[0]}:{expected[1]} ({selected.path})"
        )
    print(f"Verified hardware render node {selected.path} ({actual[0]}:{actual[1]}).")
    return device["deviceType"]


def check_environment(sysfs_drm_path: Path, dri_path: Path) -> None:
    allocation = os.environ.get("PARENT_GPU_DEVICES")
    print(f"Runner GPU allocation: {allocation or '(not specified)'}", flush=True)
    candidates = render_device_candidates(
        discover_render_devices(sysfs_drm_path, dri_path), allocation
    )
    # A private runtime directory keeps this headless probe independent of a
    # desktop session. The process supervisor also bounds native driver calls.
    with tempfile.TemporaryDirectory(prefix="iree-vulkan-") as temporary_dir:
        for candidate in candidates:
            print(f"Checking Vulkan with DRI_PRIME={candidate.dri_prime}", flush=True)
            profile_path = Path(temporary_dir) / f"{candidate.path.name}.json"
            subprocess.run(
                ["vulkaninfo", "--json=0", "--output", str(profile_path)],
                env={
                    **os.environ,
                    "DRI_PRIME": candidate.dri_prime,
                    "XDG_RUNTIME_DIR": temporary_dir,
                },
                check=True,
            )
            device_type = validate_vulkan_device(
                json.loads(profile_path.read_text()), candidate
            )
            if (
                len(candidates) == 1
                or device_type == "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
            ):
                selected = candidate
                break
        else:
            raise RuntimeError(
                "Multiple AMD GPUs are exposed without a discrete GPU; "
                "PARENT_GPU_DEVICES must identify the job's render node"
            )
    print(f"Selected {selected.path}: DRI_PRIME={selected.dri_prime}", flush=True)
    # Later workflow steps receive only a selection verified by the native
    # loader. ci.py carries it into both Bazel test actions and CMake tests.
    if github_env := os.environ.get("GITHUB_ENV"):
        with Path(github_env).open("a") as environment_file:
            environment_file.write(f"DRI_PRIME={selected.dri_prime}\n")


if __name__ == "__main__":
    try:
        check_environment(Path("/sys/class/drm"), Path("/dev/dri"))
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"Vulkan hardware preflight failed: {error}")
