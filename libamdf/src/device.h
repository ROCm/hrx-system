// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_DEVICE_H_
#define AMDF_SRC_DEVICE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_device_vtable_t {
  // Copies one immutable memory profile into caller-private result storage.
  amdf_status_t (*query_memory_profile)(
      amdf_device_t* device, uint32_t memory_profile_ordinal,
      amdf_memory_native_profile_t* out_profile);
  // Prepares native state in the already-allocated memory owner. Every return
  // leaves partial state there for common rollback; only success fills facts.
  amdf_status_t (*memory_prepare)(
      amdf_memory_t* memory, const amdf_memory_native_group_t* group,
      const amdf_memory_native_create_info_t* create_info,
      amdf_memory_info_t* out_info);
  // Acquires an independent native backing reference without consuming input.
  // Every return leaves partial state in the selected slot for common rollback;
  // only success publishes native backing facts and complete access facts.
  amdf_status_t (*memory_prepare_import)(
      amdf_memory_t* memory, uint32_t access_ordinal,
      const amdf_memory_native_profile_t* profile,
      const amdf_memory_native_import_info_t* import_info,
      const amdf_external_memory_t* external_memory,
      amdf_memory_info_t* out_info);
  // Releases the exact native state owned by a device implementation.
  amdf_status_t (*destroy_native)(amdf_device_t* device);
} amdf_device_vtable_t;

struct amdf_device_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the device is published.
  const amdf_device_vtable_t* vtable;
  // Endpoint borrowed for the lifetime of this device.
  amdf_endpoint_t* endpoint;
  // Provider instance defining the scope of provider-local identities.
  amdf_instance_t* provider_instance;
  // Exact engine family implementing this device.
  amdf_engine_kind_t engine_kind;
  // Number of live queues, programs, and contexts borrowing this device.
  // Memory dependencies are caller preconditions and are not counted.
  amdf_child_tracker_t children;
};

// Initializes an unpublished device base and borrows its endpoint.
amdf_status_t amdf_device_initialize(amdf_device_t* device,
                                     const amdf_device_vtable_t* vtable,
                                     amdf_endpoint_t* endpoint,
                                     amdf_engine_kind_t engine_kind);

// Releases the endpoint borrow held by an unpublished or torn-down device.
void amdf_device_deinitialize(amdf_device_t* device);

// Returns true when a device is implemented by `expected_engine_kind`.
bool amdf_device_is_engine(const amdf_device_t* device,
                           amdf_engine_kind_t expected_engine_kind);

// Returns true when both devices share one provider-instance identity scope.
bool amdf_device_shares_provider_instance(const amdf_device_t* lhs,
                                          const amdf_device_t* rhs);

// Returns the host allocator copied by the device.
amdf_allocator_t amdf_device_host_allocator(const amdf_device_t* device);

// Registers one child that borrows `device`.
amdf_status_t amdf_device_register_child(amdf_device_t* device);

// Releases one child borrow.
void amdf_device_unregister_child(amdf_device_t* device);

// Destroys a device after dependent memory and tracked children are released.
amdf_status_t AMDF_CALL amdf_device_destroy(amdf_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_DEVICE_H_
