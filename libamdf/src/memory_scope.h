// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_SCOPE_H_
#define AMDF_SRC_MEMORY_SCOPE_H_

#include "libamdf/src/memory_profile.h"

// Cold operations for storage qualified by a live execution owner. The owner
// embeds its scope; these callbacks neither create nor retain execution state.
typedef struct amdf_memory_scope_vtable_t {
  // Copies the complete private allocation contract without native operations.
  void (*query_profile)(amdf_memory_scope_t* scope,
                        amdf_memory_native_profile_t* out_profile);
  // Prepares native state in the memory owner, including partial progress on
  // failure. Common construction owns rollback before publication.
  amdf_status_t (*prepare)(amdf_memory_scope_t* scope, amdf_memory_t* memory,
                           const amdf_memory_native_profile_t* profile,
                           const amdf_memory_native_create_info_t* create_info,
                           amdf_memory_info_t* out_info);
} amdf_memory_scope_vtable_t;

// Storage descriptors are embedded in their existing lifetime owner. They
// neither enumerate, retain nor track allocated memory resources.
struct amdf_memory_scope_t {
  // Storage locality and the active owner variant.
  amdf_memory_scope_kind_t kind;
  // Existing owner whose lifetime bounds this borrowed descriptor.
  union {
    // Owner of a SYSTEM scope.
    amdf_instance_t* instance;
    // Owner and physical location of a LOCAL scope.
    amdf_endpoint_t* endpoint;
    // Explicitly live execution owner of a PRIVATE scope.
    struct {
      // Ordinary-address-domain device borrowed by the execution owner.
      amdf_device_t* device;
      // Cold native storage operations supplied by that owner.
      const amdf_memory_scope_vtable_t* vtable;
      // Execution owner borrowed for allocation and context-qualified use.
      void* owner;
    } private_storage;
  } owner;
};

// Information boundary used to qualify one complete requested consumer set.
typedef enum amdf_memory_access_query_kind_e {
  AMDF_MEMORY_ACCESS_QUERY_EXPECTED,
  AMDF_MEMORY_ACCESS_QUERY_LIVE,
} amdf_memory_access_query_kind_t;

typedef struct amdf_memory_access_query_t {
  // Selects cached passive facts or effective facts on explicitly live devices.
  amdf_memory_access_query_kind_t kind;
  // Number of caller-ordered consumer requests.
  uint32_t count;
  // Borrowed requests consumed during the cold profile/construction operation.
  union {
    // Intended device consumers before activation.
    const amdf_memory_endpoint_access_t* endpoints;
    // Explicitly initialized consumers, kept live by the caller.
    const amdf_memory_device_access_t* devices;
  } accesses;
} amdf_memory_access_query_t;

// Cold construction selection. Native profiles are retained through setup so
// preparation consumes the selected contracts without repeating selection.
typedef struct amdf_memory_scope_plan_t {
  // Exact borrowed scope selected by the caller, retained through construction.
  amdf_memory_scope_t* scope;
  // Allocator owning temporary native profile storage.
  amdf_allocator_t host_allocator;
  // Complete backing capabilities of the selected scope contract.
  amdf_memory_profile_t profile;
  // Number of requested device accesses; zero selects CPU-only backing.
  uint32_t access_count;
  // Native backing preparer in caller order; zero for CPU-only backing.
  uint32_t backing_access_ordinal;
  // Number of consumers prepared together by the backing's native owner.
  uint32_t backing_access_count;
  // Borrowed slice of plan storage naming that group, with its owner first.
  uint32_t* backing_access_ordinals;
  // Native ownership slot for each access; ordinary independent accesses own
  // themselves, while coordinated consumers share the backing owner's slot.
  uint32_t* native_owner_ordinals;
  // Native transport used between preparers; zero when none is required.
  amdf_external_memory_type_t shared_external_type;
  // One selected native profile per access, or one host profile at zero count.
  amdf_memory_native_profile_t* native_profiles;
} amdf_memory_scope_plan_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the existing provider instance borrowed by a valid scope.
amdf_instance_t* amdf_memory_scope_instance(const amdf_memory_scope_t* scope);

// Enumerates borrowed instance-owned system storage descriptors.
amdf_status_t AMDF_CALL amdf_instance_enumerate_memory_scopes(
    amdf_instance_t* instance, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count);

// Enumerates borrowed endpoint-owned physical storage descriptors.
amdf_status_t AMDF_CALL amdf_endpoint_enumerate_memory_scopes(
    amdf_endpoint_t* endpoint, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count);

// Enumerates native-private descriptors already owned by a live device.
amdf_status_t AMDF_CALL amdf_device_enumerate_memory_scopes(
    amdf_device_t* device, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count);

// Copies complete immutable storage facts without native operations.
amdf_status_t AMDF_CALL amdf_memory_scope_query_info(
    amdf_memory_scope_t* scope, amdf_memory_scope_info_t* out_info);

// Queries complete backing and caller-ordered expected access capabilities.
amdf_status_t AMDF_CALL amdf_memory_scope_query_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_endpoint_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities);

// Queries complete backing and caller-ordered live access capabilities.
amdf_status_t AMDF_CALL amdf_memory_scope_query_device_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities);

// Validates the complete consumer set and selects native contracts before any
// backing allocation. An optional external value constrains import selection.
// Success publishes owned temporary metadata; failure leaves output unchanged.
amdf_status_t amdf_memory_scope_plan_initialize(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal,
    const amdf_memory_access_query_t* query,
    const amdf_external_memory_t* external_memory,
    amdf_memory_scope_plan_t* out_plan);

// Releases only the temporary selection metadata, never native resources.
void amdf_memory_scope_plan_deinitialize(amdf_memory_scope_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_MEMORY_SCOPE_H_
