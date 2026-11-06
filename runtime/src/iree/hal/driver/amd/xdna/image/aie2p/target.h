// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native execution requirements for an admitted AIE2P context.

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE2P_TARGET_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE2P_TARGET_H_

#include "iree/base/api.h"

// Immutable context facts. Image admission compares these values with declared
// image requirements; it does not decode or certify native instruction
// semantics.
typedef struct iree_hal_amd_xdna_aie2p_target_t {
  // Identities shared by compatible native execution profiles.
  struct {
    // Incompatible revision of the device profile.
    uint32_t device_profile_revision;
    // Complete execution-profile identity.
    uint64_t device_profile_id;
    // Native firmware protocol identity.
    uint64_t firmware_abi_id;
  } identity;
  // Context-relative geometry available to the executable.
  struct {
    // Number of admitted columns.
    uint16_t column_count;
    // Number of rows in the array family.
    uint16_t row_count;
  } context;
  // Required firmware instruction-address alignment in bytes.
  uint32_t instruction_alignment;
} iree_hal_amd_xdna_aie2p_target_t;

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE2P_TARGET_H_
