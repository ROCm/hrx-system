// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/device/profile.h"

#include "loom/target/arch/amd/xdna/device/profile_tables.inl"

const loom_xdna_device_profile_t* loom_xdna_device_profile_lookup(
    iree_string_view_t key) {
  key = iree_string_view_trim(key);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomXdnaDeviceProfiles);
       ++i) {
    const loom_xdna_device_profile_t* profile = &kLoomXdnaDeviceProfiles[i];
    if (iree_string_view_equal(key, iree_make_cstring_view(profile->key))) {
      return profile;
    }
  }
  return NULL;
}

iree_status_t loom_xdna_device_profile_resolve_pci(
    uint16_t vendor_id, uint16_t device_id, uint8_t revision,
    const loom_xdna_device_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomXdnaDeviceProfiles);
       ++i) {
    const loom_xdna_device_profile_t* profile = &kLoomXdnaDeviceProfiles[i];
    if (profile->pci_vendor_id == vendor_id &&
        profile->pci_device_id == device_id &&
        profile->pci_revision == revision) {
      *out_profile = profile;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unsupported XDNA PCI identity %04x:%04x revision %02x", vendor_id,
      device_id, revision);
}

iree_status_t loom_xdna_device_profile_resolve_identity(
    uint64_t profile_identity, uint32_t profile_revision,
    uint64_t firmware_abi_identity,
    const loom_xdna_device_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomXdnaDeviceProfiles);
       ++i) {
    const loom_xdna_device_profile_t* profile = &kLoomXdnaDeviceProfiles[i];
    if (profile->identity == profile_identity &&
        profile->revision == profile_revision &&
        profile->firmware_abi_identity == firmware_abi_identity) {
      *out_profile = profile;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unsupported XDNA profile 0x%016" PRIx64 " revision %" PRIu32
      " with firmware ABI 0x%016" PRIx64,
      profile_identity, profile_revision, firmware_abi_identity);
}

const loom_xdna_array_family_t* loom_xdna_device_profile_array_family(
    const loom_xdna_device_profile_t* profile) {
  IREE_ASSERT_ARGUMENT(profile);
  return profile->array_family_resolver();
}

iree_status_t loom_xdna_device_profile_validate_partition(
    const loom_xdna_device_profile_t* profile, uint16_t physical_column_origin,
    uint16_t column_count) {
  IREE_ASSERT_ARGUMENT(profile);
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(profile);
  if (column_count < profile->minimum_partition_column_count ||
      column_count > family->column_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA partition width %u is outside [%u, %u] for profile %s",
        column_count, profile->minimum_partition_column_count,
        family->column_count, profile->key);
  }
  if (physical_column_origin < profile->physical_column_origin) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA partition begins before profile %s",
                            profile->key);
  }
  const uint32_t relative_origin =
      physical_column_origin - profile->physical_column_origin;
  if (relative_origin + column_count > family->column_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA partition exceeds profile %s", profile->key);
  }
  const uint64_t partition_mask = ((UINT64_C(1) << column_count) - 1)
                                  << relative_origin;
  if ((profile->available_column_mask & partition_mask) != partition_mask) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "XDNA partition includes an unavailable column");
  }
  return iree_ok_status();
}
