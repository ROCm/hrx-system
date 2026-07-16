// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_UTILS_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_device_spec_builder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_spec_builder_storage_t
    iree_hal_device_spec_builder_storage_t;

// Mutable builder used by HAL driver implementations to construct immutable
// device specs during device creation.
typedef struct iree_hal_device_spec_builder_t {
  // Host allocator used for builder-owned intermediate storage.
  iree_allocator_t host_allocator;
  // Private heap storage owned by the builder implementation.
  iree_hal_device_spec_builder_storage_t* storage;
} iree_hal_device_spec_builder_t;

// Initializes a stack-allocated builder in |out_builder|.
IREE_API_EXPORT void iree_hal_device_spec_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_device_spec_builder_t* out_builder);

// Deinitializes |builder| and releases any copied intermediate state.
IREE_API_EXPORT void iree_hal_device_spec_builder_deinitialize(
    iree_hal_device_spec_builder_t* builder);

// Copies the logical device identity facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_identity(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_identity_spec_t* identity);

// Copies the memory facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_memory(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_memory_spec_t* memory);

// Copies the virtual memory facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_virtual_memory(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_virtual_memory_spec_t* virtual_memory);

// Copies the queue facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_queues(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_queue_spec_t* queues);

// Copies the dispatch facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_dispatch(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_dispatch_spec_t* dispatch);

// Copies the timing and profiling facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_timing(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_timing_spec_t* timing);

// Copies the executable facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_executables(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_executable_spec_t* executables);

// Adds an executable target to |builder|.
//
// A target with the same family, key, and kind as an existing target is merged
// by unioning its physical-device affinity and flags. Duplicate targets must
// have the same priority.
IREE_API_EXPORT iree_status_t
iree_hal_device_spec_builder_add_executable_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_executable_target_t* target);

// Copies the sanitizer facet into |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_set_sanitizer(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_sanitizer_spec_t* sanitizer);

// Adds a driver-local extension facet to |builder|.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_spec_facet_t* facet);

// Validates and freezes |builder| into an immutable device spec.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_builder_finalize(
    iree_hal_device_spec_builder_t* builder, iree_hal_device_spec_t** out_spec);

// Creates an immutable device spec containing only logical identity facts.
//
// Driver implementations use this when they have not yet populated richer
// capability facts but still need to satisfy the core invariant that every HAL
// device has a cached immutable spec.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_create_minimal(
    iree_string_view_t logical_device_id, iree_string_view_t display_name,
    iree_string_view_t driver_id, iree_string_view_t backend_id,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_DEVICE_SPEC_BUILDER_H_
