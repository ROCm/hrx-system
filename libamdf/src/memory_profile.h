// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_PROFILE_H_
#define AMDF_SRC_MEMORY_PROFILE_H_

#include "amdf/amdf.h"

// Properties of physical backing, independent of a consumer's access.
#define AMDF_MEMORY_BACKING_FLAGS                                  \
  (AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_LOCAL | \
   AMDF_MEMORY_FLAG_SHAREABLE)

// Properties established separately for each device consumer.
#define AMDF_MEMORY_ACCESS_FLAGS                                     \
  (AMDF_MEMORY_FLAG_QUEUE_STORAGE | AMDF_MEMORY_FLAG_HOST_COHERENT | \
   AMDF_MEMORY_FLAG_DEVICE_ADDRESS)

// One native placement/acquisition contract for a single consumer. Scope
// selection composes these native facts into backing and access capabilities;
// this is not a public allocation owner or an independently owned attachment.
typedef struct amdf_memory_native_profile_t {
  // Dense ordinal within the native implementation's profile set.
  uint32_t ordinal;
  // Physical placement produced by the native operation.
  amdf_memory_class_t memory_class;
  // Supported acquisition, host mapping and transport operations.
  amdf_memory_profile_roles_t roles;
  // Backing and access properties guaranteed by this native operation.
  amdf_memory_flags_t guaranteed_flags;
  // Backing and access properties the native operation can establish.
  amdf_memory_flags_t supported_flags;
  // Permissions unconditionally established by this native operation.
  amdf_memory_access_t guaranteed_device_access;
  // Permissions that can be requested from this native operation.
  amdf_memory_access_t supported_device_access;
  // Atomic operations on naturally aligned 32-bit words.
  amdf_atomic_operations_t atomic_operations_32;
  // Atomic operations on naturally aligned 64-bit words.
  amdf_atomic_operations_t atomic_operations_64;
  // Complete numeric envelope common to the produced address kinds.
  amdf_memory_address_capabilities_t device_address;
  // Native allocation limits, or zero without CREATE.
  amdf_memory_construction_capabilities_t allocation;
  // Borrowed-page registration limits, or zero without REGISTER.
  amdf_memory_construction_capabilities_t registration;
  // External backing import limits, or zero without IMPORT.
  amdf_memory_construction_capabilities_t import;
  // Explicit CPU view limits, or zero without HOST_MAP.
  amdf_host_mapping_capabilities_t host_mapping;
  // Number of initialized transport records.
  uint32_t external_memory_support_count;
  // Native transport contracts accepted by this operation.
  amdf_external_memory_support_t
      external_memory_support[AMDF_MEMORY_PROFILE_EXTERNAL_SUPPORT_CAPACITY];
  // Address interfaces established when DEVICE_ADDRESS is achieved.
  amdf_memory_address_kinds_t address_kinds;
  // Instance-qualified protocol for preparing one backing across consumers.
  struct {
    // Matching non-NULL operations admit qualification through this protocol.
    // Projects the candidate's complete access to the selected backing using
    // cached metadata only. False leaves the output unchanged. The candidate
    // and output may alias. This performs no allocation or native operation.
    bool (*query_access)(const struct amdf_memory_native_profile_t* backing,
                         const struct amdf_memory_native_profile_t* candidate,
                         struct amdf_memory_native_profile_t* out_profile);
    // Immutable native metadata borrowed from the queried endpoint or device.
    const void* data;
  } construction;
} amdf_memory_native_profile_t;

// Explicit consumer group selected for one native backing preparation.
typedef struct amdf_memory_native_group_t {
  // Number of participating access records, always at least one.
  uint32_t access_count;
  // Borrowed caller access ordinals, with the native owner first.
  const uint32_t* access_ordinals;
  // Borrowed selected profiles indexed by the original caller access ordinal.
  const amdf_memory_native_profile_t* profiles;
} amdf_memory_native_group_t;

// Validated geometry and permissions passed to one native backing preparer.
typedef struct amdf_memory_native_create_info_t {
  // Exact permissions for the native consumer.
  amdf_memory_access_t device_access;
  // Required backing and native access properties.
  amdf_memory_flags_t required_flags;
  // Requested logical byte length.
  uint64_t byte_length;
  // Requested power-of-two minimum alignment, or zero for profile policy.
  uint64_t minimum_alignment;
  // Borrowed host pages for REGISTER; NULL for allocation.
  void* registered_host_pointer;
} amdf_memory_native_create_info_t;

// Validated permissions passed to one native external-backing preparer.
typedef struct amdf_memory_native_import_info_t {
  // Exact permissions for the native consumer.
  amdf_memory_access_t device_access;
  // Required backing and native access properties.
  amdf_memory_flags_t required_flags;
  // Requested power-of-two minimum alignment, or zero for profile policy.
  uint64_t minimum_alignment;
} amdf_memory_native_import_info_t;

#endif  // AMDF_SRC_MEMORY_PROFILE_H_
