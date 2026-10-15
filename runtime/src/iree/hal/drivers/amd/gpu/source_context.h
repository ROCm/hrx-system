// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_SOURCE_CONTEXT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_SOURCE_CONTEXT_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/device_event.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_source_context_t
    iree_hal_amd_gpu_source_context_t;

// Device-owned immutable source metadata. Reports can outlive the executable
// which emitted them; registered addresses stay valid through the last drain.
typedef struct iree_hal_amd_gpu_source_context_registry_t {
  // Allocator owning contexts and their inline source bytes.
  iree_allocator_t host_allocator;
  // Serializes cold executable registration; readers never take this lock.
  iree_slim_mutex_t mutex;
  // Owned contexts released after all feedback producers and consumers join.
  iree_hal_amd_gpu_source_context_t* first;
  // Last assigned device-local executable identity; zero denotes unknown.
  uint64_t last_executable_id;
} iree_hal_amd_gpu_source_context_registry_t;

void iree_hal_amd_gpu_source_context_registry_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_source_context_registry_t* out_registry);

void iree_hal_amd_gpu_source_context_registry_deinitialize(
    iree_hal_amd_gpu_source_context_registry_t* registry);

// Validates and copies an optional Loom LSIT v1 source table. Empty tables
// retain executable identity without source locations. Success returns an
// immutable borrow dominated by registry; failure publishes nothing. This is
// load work.
iree_status_t iree_hal_amd_gpu_source_context_registry_register(
    iree_hal_amd_gpu_source_context_registry_t* registry,
    iree_const_byte_span_t site_table,
    const iree_hal_amd_gpu_source_context_t** out_context);

uint64_t iree_hal_amd_gpu_source_context_executable_id(
    const iree_hal_amd_gpu_source_context_t* context);

// Resolves a dense site ID directly. Returned strings and producer payload are
// borrowed from the retained context. Unknown sites leave out_site empty.
bool iree_hal_amd_gpu_source_context_resolve(
    const iree_hal_amd_gpu_source_context_t* context, uint64_t site_id,
    iree_hal_device_event_site_t* out_site);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_SOURCE_CONTEXT_H_
