// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable AMD XDNA deployment profiles.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_DEVICE_PROFILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_DEVICE_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Driver-visible resource limits outside individual array tiles.
typedef struct loom_xdna_device_limits_t {
  // Minimum byte alignment of device allocations.
  uint32_t minimum_device_memory_alignment;
  // Maximum simultaneously resident hardware contexts.
  uint8_t hardware_context_limit;
  // Maximum driver contexts.
  uint8_t context_limit;
  // Whether the profile admits only temporal context sharing.
  bool temporal_contexts_only;
} loom_xdna_device_limits_t;

// Resolves the immutable array family selected by one device profile.
typedef const loom_xdna_array_family_t*(
    IREE_API_PTR* loom_xdna_array_family_resolver_t)(void);

// Complete reproducible deployment profile selected by device identity.
typedef struct loom_xdna_device_profile_t {
  // Stable semantic profile key.
  const char* key;
  // Human-readable device name.
  const char* display_name;
  // Stable incompatible device-profile identity.
  uint64_t identity;
  // Stable firmware/configuration protocol identity.
  uint64_t firmware_abi_identity;
  // Bit set of physical columns available to partitions.
  uint64_t available_column_mask;
  // Function resolving the selected immutable array family.
  loom_xdna_array_family_resolver_t array_family_resolver;
  // Incompatible device-profile revision.
  uint32_t revision;
  // Independent sources supporting the profile.
  loom_xdna_provenance_bits_t provenance_bits;
  // PCI vendor identifier.
  uint16_t pci_vendor_id;
  // PCI device identifier.
  uint16_t pci_device_id;
  // Exact PCI revision selecting this profile.
  uint8_t pci_revision;
  // Physical column represented by profile-relative column zero.
  uint8_t physical_column_origin;
  // Minimum legal contiguous partition width.
  uint8_t minimum_partition_column_count;
  // Minimum supported firmware protocol major version.
  uint8_t firmware_protocol_major;
  // Minimum supported firmware protocol minor version.
  uint8_t firmware_protocol_minor;
  // Firmware-reported device revision identifier.
  uint8_t firmware_device_revision;
  // Device-generation value serialized in transaction streams.
  uint8_t transaction_device_generation;
  // Native XDNA ELF ABI major version.
  uint8_t native_elf_abi_major;
  // Native XDNA ELF ABI minor version.
  uint8_t native_elf_abi_minor;
  // Driver-visible resource limits.
  loom_xdna_device_limits_t limits;
  // Driver version used by the retained hardware witness.
  const char* qualified_driver_version;
  // Firmware build used by the retained hardware witness.
  const char* qualified_firmware_version;
  // XRT version used by the retained hardware witness.
  const char* qualified_xrt_version;
  // XRT source revision used by the retained hardware witness.
  const char* qualified_xrt_source_commit;
  // Pinned XDNA driver source revision defining this profile.
  const char* xdna_driver_source_commit;
} loom_xdna_device_profile_t;

// Resolves an exact PCI identity without a device-family fallback.
iree_status_t loom_xdna_device_profile_resolve_pci(
    uint16_t vendor_id, uint16_t device_id, uint8_t revision,
    const loom_xdna_device_profile_t** out_profile);

// Resolves an exact serialized profile and firmware ABI identity.
iree_status_t loom_xdna_device_profile_resolve_identity(
    uint64_t profile_identity, uint32_t profile_revision,
    uint64_t firmware_abi_identity,
    const loom_xdna_device_profile_t** out_profile);

// Returns the immutable array family selected by |profile|.
const loom_xdna_array_family_t* loom_xdna_device_profile_array_family(
    const loom_xdna_device_profile_t* profile);

// Validates one contiguous physical-column partition against |profile|.
iree_status_t loom_xdna_device_profile_validate_partition(
    const loom_xdna_device_profile_t* profile, uint16_t physical_column_origin,
    uint16_t column_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_DEVICE_PROFILE_H_
