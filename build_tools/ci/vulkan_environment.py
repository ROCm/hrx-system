#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verifies Vulkan device zero uses the render node granted to this job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from build_tools.ci import gpu_assignment


@dataclass(frozen=True)
class RenderDevice:
    # Device node exposed to the job.
    path: Path
    # PCI domain, bus, device and function identifying this physical card.
    pci_address: str
    # Kernel-reported PCI vendor ID, paired with device_id for Vulkan verification.
    vendor_id: int
    # Kernel-reported PCI product ID, used for diagnostics and identity checking.
    device_id: int
    # Device number of the granted render node, including major and minor.
    device_number: int

    @property
    def dri_prime(self) -> str:
        return f"pci-{self.pci_address.replace(':', '_').replace('.', '_')}!"


def assigned_render_devices(pci_vendor: int | None) -> list[RenderDevice]:
    devices = []
    for path, number in gpu_assignment.assigned_render_nodes().items():
        pci_path = (gpu_assignment.DRM_CLASS / path.name / "device").resolve(
            strict=True
        )
        vendor_id = int((pci_path / "vendor").read_text(), 16)
        device_id = int((pci_path / "device").read_text(), 16)
        if pci_vendor is not None and vendor_id != pci_vendor:
            print(
                f"Skipping {path}: PCI {pci_path.name} {vendor_id:04x}:{device_id:04x}"
            )
            continue
        devices.append(RenderDevice(path, pci_path.name, vendor_id, device_id, number))
    if not devices:
        raise ValueError(
            f"No assigned render node matches PCI vendor {pci_vendor:#06x}"
        )
    return devices


def open_render_device(device: RenderDevice) -> None:
    print(
        f"Opening {device.path}: PCI {device.pci_address} "
        f"{device.vendor_id:04x}:{device.device_id:04x}",
        flush=True,
    )
    descriptor = os.open(device.path, os.O_RDWR | os.O_CLOEXEC)
    try:
        if os.fstat(descriptor).st_rdev != device.device_number:
            raise RuntimeError(f"Render device changed while opening {device.path}")
    finally:
        os.close(descriptor)


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
    if (device["vendorID"], device["deviceID"]) != (
        selected.vendor_id,
        selected.device_id,
    ):
        raise RuntimeError(
            f"Vulkan device 0 does not match the selected PCI device "
            f"{selected.vendor_id:04x}:{selected.device_id:04x} ({selected.path})"
        )
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


def check_environment(
    profile_output: Path | None = None, *, pci_vendor: int | None = None
) -> None:
    candidates = assigned_render_devices(pci_vendor)
    # A private runtime directory keeps this headless probe independent of a
    # desktop session. The process supervisor also bounds native driver calls.
    with tempfile.TemporaryDirectory(prefix="iree-vulkan-") as temporary_dir:
        for candidate in candidates:
            open_render_device(candidate)
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
            profile = profile_path.read_bytes()
            device_type = validate_vulkan_device(json.loads(profile), candidate)
            if (
                len(candidates) == 1
                or device_type == "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
            ):
                selected = candidate
                break
        else:
            raise RuntimeError(
                "Multiple assigned GPUs are integrated; assign the render node for this job"
            )
    print(f"Selected {selected.path}: DRI_PRIME={selected.dri_prime}", flush=True)
    if profile_output is not None:
        profile_output.write_bytes(profile)
    # Later workflow steps receive only a selection verified by the native
    # loader. ci.py carries it into both Bazel test actions and CMake tests.
    if github_env := os.environ.get("GITHUB_ENV"):
        with Path(github_env).open("a") as environment_file:
            environment_file.write(f"DRI_PRIME={selected.dri_prime}\n")
            if selected.vendor_id == 0x1002:
                native_identity = gpu_assignment.native_identity(selected.device_number)
                environment_file.write(f"HAL_AMD_GPU_ASSIGNMENT={native_identity}\n")
                environment_file.write(
                    f"AMDF_CTS_GPU_NATIVE_IDENTITY={native_identity}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pci-vendor",
        type=lambda value: int(value, 16),
        help="Require an assigned GPU from this PCI vendor (hexadecimal).",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Save the selected device's vulkaninfo JSON.",
    )
    args = parser.parse_args()
    try:
        check_environment(args.profile_output, pci_vendor=args.pci_vendor)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"Vulkan hardware preflight failed: {error}")
