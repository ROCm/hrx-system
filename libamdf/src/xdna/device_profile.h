// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_DEVICE_PROFILE_H_
#define AMDF_SRC_XDNA_DEVICE_PROFILE_H_

#include "amdf/xdna.h"
#include "libamdf/src/xdna/bootstrap.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Target-native execution services carried by a resolved device profile.
typedef uint64_t amdf_xdna_execution_capabilities_t;
enum amdf_xdna_execution_capability_bits_e {
  // Version 1 transaction interpreter, command packet, and bootstrap contract.
  AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1 = UINT64_C(1) << 0,
  // Native ERT ELF instruction-range submission without a bootstrap CU.
  AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS = UINT64_C(1) << 1,
};

// Execution profile combining target encodings and effective array layout.
typedef struct amdf_xdna_device_profile_t {
  // Borrowed live-device properties, finalized before publication.
  const amdf_xdna_device_info_t* info;
  // Target-native services whose complete contracts are available below.
  amdf_xdna_execution_capabilities_t execution_capabilities;
  // Borrowed process-lifetime interpreter bootstrap, or NULL when unsupported.
  const amdf_xdna_bootstrap_t* bootstrap;
  // Firmware heap extent and alignment in bytes.
  uint32_t firmware_heap_byte_length;
  // Native DRAM address interpretation for direct shim DMA descriptors.
  struct {
    // Offset added to the native buffer address for shim DMA access.
    uint32_t byte_offset;
    // Number of significant bits in the complete translated DMA address.
    uint32_t address_bit_count;
  } dma;
  // Target-native transaction properties fixed for this endpoint identity.
  struct {
    // AIE-RT device-generation value encoded in transaction headers.
    uint8_t device_generation;
  } transaction;
  // Physical row classes reported by the native driver.
  struct {
    // First shim row in physical coordinates.
    uint32_t shim_origin;
    // Number of contiguous shim rows.
    uint32_t shim_count;
    // First memory-tile row in physical coordinates.
    uint32_t memory_origin;
    // Number of contiguous memory-tile rows.
    uint32_t memory_count;
    // First core-tile row in physical coordinates.
    uint32_t core_origin;
    // Number of contiguous core-tile rows.
    uint32_t core_count;
  } rows;
} amdf_xdna_device_profile_t;

// Copies passive architecture and compiler identity for a recognized endpoint.
// Returns false without modifying the output for an unrecognized identity.
bool amdf_xdna_query_endpoint_info(const amdf_endpoint_info_t* endpoint_info,
                                   amdf_xdna_endpoint_info_t* out_info);

// Initializes complete architecture and memory encodings for device creation.
// Passive recognition alone is insufficient; an unavailable device profile
// returns false without modifying either output. Geometry and scheduling are
// resolved by activation before the owner is published. The profile borrows
// `device_info`, which must outlive every native child.
bool amdf_xdna_device_profile_initialize(
    const amdf_endpoint_info_t* endpoint_info,
    amdf_xdna_device_info_t* device_info,
    amdf_xdna_device_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_DEVICE_PROFILE_H_
