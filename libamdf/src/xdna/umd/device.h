// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_DEVICE_H_
#define AMDF_SRC_XDNA_UMD_DEVICE_H_

#include "amdf/xdna.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/xdna/device_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_xdna_umd_device_t amdf_xdna_umd_device_t;
struct amdf_memory_reservation_profile_t;

// Array layout reported by the driver after device activation.
typedef struct amdf_xdna_umd_tile_metadata_t {
  // Number of addressable array columns.
  uint32_t column_count;
  // Total rows, including shim, memory and core tiles.
  uint32_t row_count;
  // First physical core row.
  uint32_t core_origin;
  // Number of contiguous core rows.
  uint32_t core_count;
  // First physical memory row.
  uint32_t memory_origin;
  // Number of contiguous memory rows.
  uint32_t memory_count;
  // First physical shim row.
  uint32_t shim_origin;
  // Number of contiguous shim rows.
  uint32_t shim_count;
} amdf_xdna_umd_tile_metadata_t;

// Native result used to publish one successfully materialized XDNA device.
typedef struct amdf_xdna_umd_device_result_t {
  // Opaque identity of the live native address-domain device.
  amdf_device_id_t id;
  // Provider epoch invalidating native state after reset.
  uint64_t reset_epoch;
  // Effective binding context placement contracts after native qualification.
  amdf_xdna_placement_modes_t placement_modes;
  // Native layout used by all contexts created on this device.
  amdf_xdna_umd_tile_metadata_t tiles;
} amdf_xdna_umd_device_result_t;

// Expected context admission through the implemented native provider.
typedef struct amdf_xdna_umd_context_capabilities_t {
  // Scheduling contracts supported for the target-native execution path.
  amdf_xdna_scheduling_modes_t scheduling_modes;
  // Binding physical placement contracts available to those contexts.
  amdf_xdna_placement_modes_t placement_modes;
  // Optional transaction operations implemented by this admission path.
  amdf_xdna_instruction_features_t instruction_features;
} amdf_xdna_umd_context_capabilities_t;

// Returns expected context admission contracts from the profile and implemented
// native provider. Performs no allocation, native query or device activation.
amdf_xdna_umd_context_capabilities_t amdf_xdna_umd_query_context_capabilities(
    const amdf_xdna_device_profile_t* profile);

// Creates one native XDNA ordinary-address domain and allocation namespace.
amdf_status_t amdf_xdna_umd_device_create(
    amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_device_profile_t* profile, amdf_allocator_t host_allocator,
    amdf_xdna_umd_device_t** out_device,
    amdf_xdna_umd_device_result_t* out_result);

// Releases native XDNA device state in reverse ownership order.
amdf_status_t amdf_xdna_umd_device_destroy(amdf_xdna_umd_device_t* device);

// Resolves a live ordinary-domain mapping contract from cached native facts.
// Unsupported providers return UNSUPPORTED; queries perform no native work.
amdf_status_t amdf_xdna_umd_device_query_memory_reservation_profile(
    amdf_xdna_umd_device_t* device, uint32_t address_domain_ordinal,
    struct amdf_memory_reservation_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_DEVICE_H_
