// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_ENDPOINT_PROFILE_H_
#define AMDF_SRC_GPU_ENDPOINT_PROFILE_H_

#include <stdbool.h>
#include <stdint.h>

#include "amdf/gpu.h"
#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Expected native support and implemented features for one lifetime policy.
typedef struct amdf_gpu_lifetime_properties_t {
  // Whether the provider implements this policy; activation qualifies its ABI.
  bool supported;
  // Expected features under this lifetime policy.
  amdf_gpu_device_features_t features;
} amdf_gpu_lifetime_properties_t;

enum { AMDF_GPU_QUEUE_FAMILY_CAPACITY = 4 };

enum { AMDF_GPU_MEMORY_PROFILE_CAPACITY = 3 };

// Native service properties for one constructible GPU queue family.
typedef struct amdf_gpu_queue_family_properties_t {
  // Native command representation accepted by this family.
  amdf_queue_command_type_t command_type;
  // Version defining commands and direct publication when available.
  uint32_t format_version;
  // Native packet encoding features within the command format.
  amdf_queue_format_features_t format_features;
  // Implemented publication mechanisms.
  amdf_queue_publication_modes_t publication_modes;
  // Semantic operations accepted by this family.
  amdf_queue_roles_t roles;
  // Semantic cache operations encoded by this family.
  amdf_cache_operations_t cache_operations;
  // Range and global forms available for cache operations.
  amdf_cache_transition_kinds_t cache_transition_kinds;
  // Atomic operations encoded by this family.
  amdf_atomic_capabilities_t atomic_capabilities;
  // Direct producer operations supported by user queues.
  amdf_user_queue_capabilities_t user_queue_capabilities;
  // Optional operations supported by kernel queues.
  amdf_kernel_queue_capabilities_t kernel_queue_capabilities;
  // Supported direct-publication reservation protocols.
  amdf_queue_producer_modes_t producer_modes;
  // Supported scheduling priorities for user queues.
  amdf_queue_priority_capabilities_t priority_capabilities;
  // Optional sidecar metadata command format.
  amdf_queue_metadata_format_t metadata;
  // Minimum supported power-of-two primary user ring length in bytes.
  uint64_t minimum_ring_byte_length;
  // Maximum supported power-of-two primary user ring length in bytes.
  uint64_t maximum_ring_byte_length;
  // Required primary user ring alignment in bytes.
  uint64_t ring_byte_length_alignment;
} amdf_gpu_queue_family_properties_t;

// Provider-neutral facts used to qualify one immutable GPU profile.
typedef struct amdf_gpu_endpoint_properties_t {
  // Exact Graphics IP identity.
  struct {
    // GFX IP major version.
    uint32_t major;
    // GFX IP minor version.
    uint32_t minor;
    // GFX IP stepping.
    uint32_t stepping;
  } gfx_ip;
  // HSA/KFD ASIC revision used for physical target selection.
  uint32_t asic_revision;
  // Active compute geometry and per-compute-unit limits.
  struct {
    // Number of lanes in one hardware wavefront.
    uint32_t wavefront_size;
    // Total active compute units across all XCCs.
    uint32_t compute_unit_count;
    // Maximum resident hardware waves per compute unit.
    uint32_t maximum_wave_count_per_compute_unit;
    // Effective scratch wave slots per compute unit.
    uint32_t maximum_scratch_wave_count_per_compute_unit;
    // Local data share capacity per compute unit in bytes.
    uint64_t local_data_share_byte_length;
  } compute;
  // Multi-chiplet command-processor topology.
  struct {
    // Number of active XCCs represented by the endpoint.
    uint32_t xcc_count;
    // Uniform number of shader engines within each XCC.
    uint32_t shader_engine_count_per_xcc;
  } topology;
  // Number of constructible native queue families.
  uint32_t queue_family_count;
  // Exact native queue services implemented by the selected UMD.
  amdf_gpu_queue_family_properties_t
      queue_families[AMDF_GPU_QUEUE_FAMILY_CAPACITY];
  // Capabilities indexed by amdf_native_lifetime_t.
  amdf_gpu_lifetime_properties_t native_lifetimes[2];
} amdf_gpu_endpoint_properties_t;

// Immutable qualified GPU profile owned by one core endpoint.
typedef struct amdf_gpu_endpoint_profile_t {
  // Public target and compute properties copied by the GPU extension.
  amdf_gpu_endpoint_info_t info;
  // Number of validated public queue-family records.
  uint32_t queue_family_count;
  // Public queue-family records with dense endpoint-local ordinals.
  amdf_queue_family_info_t queue_families[AMDF_GPU_QUEUE_FAMILY_CAPACITY];
  // Capabilities indexed by amdf_native_lifetime_t.
  amdf_gpu_lifetime_properties_t native_lifetimes[2];
  // Expected memory contracts qualified without constructing a native device.
  struct {
    // Number of complete profiles; zero when memory operations are unavailable.
    uint32_t count;
    // Complete native profiles for this instance's lifetime policy.
    amdf_memory_native_profile_t values[AMDF_GPU_MEMORY_PROFILE_CAPACITY];
  } memory;
} amdf_gpu_endpoint_profile_t;

// Validates and normalizes |properties| into |out_profile|.
bool amdf_gpu_endpoint_profile_initialize(
    const amdf_gpu_endpoint_properties_t* properties,
    amdf_gpu_endpoint_profile_t* out_profile);

// Returns the borrowed public information stored in |profile|.
const amdf_gpu_endpoint_info_t* amdf_gpu_endpoint_profile_get_info(
    const amdf_gpu_endpoint_profile_t* profile);

// Copies cached features for the instance's validated lifetime on success.
amdf_status_t amdf_gpu_endpoint_profile_query_device_features(
    const amdf_gpu_endpoint_profile_t* profile,
    amdf_native_lifetime_t native_lifetime,
    amdf_gpu_device_features_t* out_features);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_ENDPOINT_PROFILE_H_
