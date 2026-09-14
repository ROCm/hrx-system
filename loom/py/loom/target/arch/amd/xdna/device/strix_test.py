# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amd.xdna.device.model import (
    resolve_pci_profile,
    validate_device_profile,
)
from loom.target.arch.amd.xdna.device.strix import (
    DEVICE_PROFILES,
    NPU2_FIRMWARE_ABI_ID,
    STRIX_HALO_PROFILE,
    STRIX_HALO_PROFILE_ID,
    STRIX_PROFILE,
    STRIX_PROFILE_ID,
    XDNA_DRIVER_SOURCE_COMMIT,
)


def test_strix_halo_profile_reproduces_observed_device_identity() -> None:
    profile = STRIX_HALO_PROFILE

    assert profile.key == "amd.xdna.strix_halo.17f0_11"
    assert profile.identity == STRIX_HALO_PROFILE_ID == 0x535848414C4F0001
    assert (profile.pci.vendor_id, profile.pci.device_id, profile.pci.revision) == (
        0x1022,
        0x17F0,
        0x11,
    )
    assert profile.display_name == "NPU Strix Halo"
    assert profile.available_column_mask == 0xFF
    assert profile.array_family.column_count == 8


def test_strix_profiles_share_array_and_protocol_facts() -> None:
    profile = STRIX_PROFILE

    assert profile.identity == STRIX_PROFILE_ID == 0x5354524958000001
    assert profile.identity != STRIX_HALO_PROFILE.identity
    assert profile.array_family is STRIX_HALO_PROFILE.array_family
    assert profile.firmware is STRIX_HALO_PROFILE.firmware
    assert profile.limits is STRIX_HALO_PROFILE.limits
    assert profile.firmware.identity == NPU2_FIRMWARE_ABI_ID
    assert (profile.firmware.minimum_major, profile.firmware.minimum_minor) == (6, 12)
    assert profile.firmware.transaction_device_generation == 4
    assert XDNA_DRIVER_SOURCE_COMMIT == "c8471cb3bbff3621bbe72cf7c9b3278f6fc23dc2"


def test_pci_resolution_is_exact_and_has_no_revision_fallback() -> None:
    assert (
        resolve_pci_profile(DEVICE_PROFILES, 0x1022, 0x17F0, 0x11) is STRIX_HALO_PROFILE
    )
    assert resolve_pci_profile(DEVICE_PROFILES, 0x1022, 0x17F0, 0x10) is STRIX_PROFILE

    with pytest.raises(ValueError, match="unsupported XDNA PCI identity"):
        resolve_pci_profile(DEVICE_PROFILES, 0x1022, 0x17F0, 0x12)


def test_physical_array_coordinates_must_fit_the_profile_encoding() -> None:
    invalid_profile = replace(STRIX_HALO_PROFILE, physical_column_origin=0xFFFF)
    with pytest.raises(ValueError, match="physical columns overflow coordinates"):
        validate_device_profile(invalid_profile)
