# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validated deployment-profile source schema."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.amd.xdna.array.model import (
    ArrayFamily,
    Provenance,
    validate_array_family,
)


@dataclass(frozen=True, slots=True)
class PciIdentity:
    """Exact PCI identity selecting one deployment profile."""

    vendor_id: int
    device_id: int
    revision: int


@dataclass(frozen=True, slots=True)
class FirmwareProtocol:
    """Firmware-facing configuration protocol selected by the profile."""

    identity: int
    minimum_major: int
    minimum_minor: int
    device_revision: int
    transaction_device_generation: int


@dataclass(frozen=True, slots=True)
class DeviceLimits:
    """Driver-visible resource limits outside individual array tiles."""

    minimum_device_memory_alignment: int
    hardware_context_limit: int
    context_limit: int
    temporal_contexts_only: bool


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Complete reproducible deployment profile selected by device identity."""

    key: str
    revision: int
    identity: int
    display_name: str
    pci: PciIdentity
    array_family: ArrayFamily
    physical_column_origin: int
    available_column_mask: int
    minimum_partition_column_count: int
    firmware: FirmwareProtocol
    native_elf_abi_major: int
    native_elf_abi_minor: int
    limits: DeviceLimits
    provenance: Provenance


def validate_device_profile(profile: DeviceProfile) -> None:
    """Validates one immutable deployment profile and its selected family."""
    validate_array_family(profile.array_family)
    if (
        not profile.key
        or not profile.display_name
        or profile.revision <= 0
        or profile.identity <= 0
        or profile.identity > 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("device-profile identity is incomplete")
    if profile.pci.vendor_id <= 0 or profile.pci.vendor_id > 0xFFFF:
        raise ValueError(f"{profile.key}: invalid PCI vendor ID")
    if profile.pci.device_id <= 0 or profile.pci.device_id > 0xFFFF:
        raise ValueError(f"{profile.key}: invalid PCI device ID")
    if profile.pci.revision < 0 or profile.pci.revision > 0xFF:
        raise ValueError(f"{profile.key}: invalid PCI revision")
    if profile.physical_column_origin < 0 or profile.physical_column_origin > 0xFFFF:
        raise ValueError(f"{profile.key}: invalid physical column origin")
    column_count = profile.array_family.column_count
    if profile.physical_column_origin + column_count > 1 << 16:
        raise ValueError(f"{profile.key}: physical columns overflow coordinates")
    if column_count > 64:
        raise ValueError(f"{profile.key}: column mask exceeds 64 bits")
    known_column_mask = (1 << column_count) - 1
    if (
        profile.available_column_mask <= 0
        or profile.available_column_mask & ~known_column_mask
    ):
        raise ValueError(f"{profile.key}: invalid available-column mask")
    if not 0 < profile.minimum_partition_column_count <= column_count:
        raise ValueError(f"{profile.key}: invalid minimum partition width")
    if profile.available_column_mask != known_column_mask:
        raise ValueError(f"{profile.key}: initial profile requires contiguous columns")
    firmware = profile.firmware
    if (
        firmware.identity <= 0
        or firmware.identity > 0xFFFFFFFFFFFFFFFF
        or firmware.minimum_major <= 0
        or firmware.minimum_minor < 0
        or firmware.device_revision <= 0
        or firmware.transaction_device_generation <= 0
    ):
        raise ValueError(f"{profile.key}: invalid firmware protocol")
    if profile.native_elf_abi_major <= 0 or profile.native_elf_abi_minor < 0:
        raise ValueError(f"{profile.key}: invalid native ELF ABI")
    limits = profile.limits
    if (
        limits.minimum_device_memory_alignment <= 0
        or limits.minimum_device_memory_alignment
        & (limits.minimum_device_memory_alignment - 1)
        or limits.hardware_context_limit <= 0
        or limits.context_limit < limits.hardware_context_limit
    ):
        raise ValueError(f"{profile.key}: invalid device limits")
    required_provenance = Provenance.XDNA_DRIVER | Provenance.HARDWARE
    if profile.provenance & required_provenance != required_provenance:
        raise ValueError(f"{profile.key}: profile provenance is incomplete")


def resolve_pci_profile(
    profiles: tuple[DeviceProfile, ...],
    vendor_id: int,
    device_id: int,
    revision: int,
) -> DeviceProfile:
    """Resolves one exact PCI identity or fails without a fallback profile."""
    matches = tuple(
        profile
        for profile in profiles
        if profile.pci == PciIdentity(vendor_id, device_id, revision)
    )
    if len(matches) != 1:
        raise ValueError(
            f"unsupported XDNA PCI identity {vendor_id:04x}:{device_id:04x} "
            f"revision {revision:02x}"
        )
    return matches[0]
