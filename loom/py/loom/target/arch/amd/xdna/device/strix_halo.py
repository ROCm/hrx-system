# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strix Halo XDNA deployment profile."""

from __future__ import annotations

from loom.target.arch.amd.xdna.array.model import Provenance
from loom.target.arch.amd.xdna.array.npu2 import NPU2_ARRAY_FAMILY
from loom.target.arch.amd.xdna.device.model import (
    DeviceLimits,
    DeviceProfile,
    FirmwareProtocol,
    PciIdentity,
    validate_device_profile,
)

# The XDNA driver revision maps PCI 17f0:11 to the npu5/Strix Halo profile.
XDNA_DRIVER_SOURCE_COMMIT = "c8471cb3bbff3621bbe72cf7c9b3278f6fc23dc2"

# ASCII `SXHALO`, followed by the incompatible profile identity revision.
STRIX_HALO_PROFILE_ID = 0x535848414C4F0001

# ASCII `NPU2`, followed by the minimum 6.12 firmware protocol revision.
NPU2_FIRMWARE_ABI_ID = 0x4E5055320006000C


STRIX_HALO_PROFILE = DeviceProfile(
    key="amd.xdna.strix_halo.17f0_11",
    revision=1,
    identity=STRIX_HALO_PROFILE_ID,
    display_name="NPU Strix Halo",
    pci=PciIdentity(vendor_id=0x1022, device_id=0x17F0, revision=0x11),
    array_family=NPU2_ARRAY_FAMILY,
    physical_column_origin=0,
    available_column_mask=0xFF,
    minimum_partition_column_count=1,
    firmware=FirmwareProtocol(
        identity=NPU2_FIRMWARE_ABI_ID,
        minimum_major=6,
        minimum_minor=12,
        device_revision=5,
        transaction_device_generation=4,
    ),
    native_elf_abi_major=1,
    native_elf_abi_minor=0,
    limits=DeviceLimits(
        minimum_device_memory_alignment=32 * 1024,
        hardware_context_limit=16,
        context_limit=32,
        temporal_contexts_only=True,
    ),
    provenance=(
        Provenance.AIE_RT
        | Provenance.MLIR_AIE
        | Provenance.XDNA_DRIVER
        | Provenance.HARDWARE
    ),
)

DEVICE_PROFILES = (STRIX_HALO_PROFILE,)

validate_device_profile(STRIX_HALO_PROFILE)
