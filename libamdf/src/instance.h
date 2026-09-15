// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_INSTANCE_H_
#define AMDF_SRC_INSTANCE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/platform/instance.h"

struct amdf_gpu_umd_instance_t;

struct amdf_instance_t {
  // Immutable lifetime policy for native resources created by this instance.
  amdf_native_lifetime_t native_lifetime;
  // Host allocator copied for this instance and all of its children.
  amdf_allocator_t host_allocator;
  // Platform implementation owned by this instance.
  amdf_platform_instance_t* platform;
  // Shared GPU native connection state, prepared by explicit device creation.
  // The platform native-state lock protects preparation; provider teardown
  // releases it after all endpoints close. NULL means no separate state.
  struct amdf_gpu_umd_instance_t* gpu;
  // Fixed borrowed system-storage descriptor, not a memory-resource registry.
  amdf_memory_scope_t system_memory_scope;
  // Number of open children borrowing this instance.
  amdf_child_tracker_t children;
};

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a provider instance with an immutable native lifetime policy.
amdf_status_t AMDF_CALL
amdf_instance_create(const amdf_instance_create_info_t* create_info,
                     amdf_instance_t** out_instance);

// Destroys an instance with no remaining children.
amdf_status_t AMDF_CALL amdf_instance_destroy(amdf_instance_t* instance);

// Enumerates a bounded snapshot of independently selectable AMD endpoints.
amdf_status_t AMDF_CALL amdf_endpoint_enumerate(
    amdf_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count);

// Returns the platform implementation borrowed by the instance.
amdf_platform_instance_t* amdf_instance_platform(amdf_instance_t* instance);

// Returns the host allocator copied by the instance.
amdf_allocator_t amdf_instance_host_allocator(const amdf_instance_t* instance);

// Returns the validated native lifetime policy copied by the instance.
amdf_native_lifetime_t amdf_instance_native_lifetime(
    const amdf_instance_t* instance);

// Registers an endpoint that borrows the instance.
amdf_status_t amdf_instance_register_endpoint(amdf_instance_t* instance);

// Unregisters one previously registered endpoint.
void amdf_instance_unregister_endpoint(amdf_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_INSTANCE_H_
