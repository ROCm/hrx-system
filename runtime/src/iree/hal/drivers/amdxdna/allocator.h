// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executable creation (active != 0) counts a live executable and resumes
// reuse. Executable destruction (active == 0) drops that count and, at zero,
// frees the reuse cache. Other allocator types are ignored.
// Returns 1 when this release drops the last live executable.
int iree_hal_amdxdna_allocator_set_model_reuse(iree_hal_allocator_t* allocator,
                                               int active);

// Returns the index of the smallest capacity that can satisfy |required_size|,
// or IREE_HOST_SIZE_MAX when none fits.
iree_host_size_t iree_hal_amdxdna_allocator_find_best_fit(
    const iree_device_size_t* capacities, iree_host_size_t capacity_count,
    iree_device_size_t required_size);

typedef struct iree_hal_amdxdna_host_recycler_block_t
    iree_hal_amdxdna_host_recycler_block_t;

// Bounded best-fit recycler used for short-lived executable descriptor
// storage. Cached blocks carry their capacity in an internal aligned header;
// callers see a normal iree_allocator_t and require no public HAL/API changes.
typedef struct iree_hal_amdxdna_host_recycler_t {
  iree_allocator_t backing_allocator;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_host_recycler_block_t* free_list;
  iree_host_size_t retained_bytes;
  iree_host_size_t retained_count;
  iree_host_size_t max_retained_bytes;
  iree_host_size_t max_retained_count;
} iree_hal_amdxdna_host_recycler_t;

void iree_hal_amdxdna_host_recycler_initialize(
    iree_allocator_t backing_allocator, iree_host_size_t max_retained_bytes,
    iree_host_size_t max_retained_count,
    iree_hal_amdxdna_host_recycler_t* out_recycler);

void iree_hal_amdxdna_host_recycler_deinitialize(
    iree_hal_amdxdna_host_recycler_t* recycler);

iree_allocator_t iree_hal_amdxdna_host_recycler_allocator(
    iree_hal_amdxdna_host_recycler_t* recycler);

// Creates a buffer allocator used for persistent allocations.
iree_status_t iree_hal_amdxdna_allocator_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_
