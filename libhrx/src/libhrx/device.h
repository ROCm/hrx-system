// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_LIBHRX_DEVICE_H_
#define LIBHRX_SRC_LIBHRX_DEVICE_H_

#include "buffer_table.h"
#include "hrx_runtime.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "mem_pool_backing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_allocator_s {
  // Borrowed legacy native allocator, retained once by the device.
  iree_hal_allocator_t* hal_allocator;
  // Borrowed parent; public allocator references retain this device.
  hrx_device_t device;
} hrx_allocator_s;

// Inline view owned by the immutable group. Public device references retain
// the group; the group never takes a public reference to its own views.
typedef struct hrx_device_s {
  // Borrowed lifetime owner for this view and every other device in the group.
  hrx_device_group_t group;
  // Execution processor class exposed by the selected driver.
  hrx_accelerator_type_t type;
  // Index within the owning group.
  int ordinal;
  // Native device borrowed from |hal_device_group|.
  iree_hal_device_t* hal_device;
  // Device-owned HRX handles in family-major provisioned queue order.
  struct hrx_queue_s* queues;
  // Borrowed provisioned queue for device-scoped compatibility operations.
  iree_hal_queue_t* dispatch_queue;
  // Borrowed native topology owner, retained once by |group|.
  iree_hal_device_group_t* hal_device_group;
  // Lazy compatibility backing; explicitly constructed pools select their own.
  hrx_mem_pool_backing_t mem_pool_backing;
  // Whether this device participates in the runtime's profile session.
  bool profiling_active;
  // Inline compatibility allocator view.
  hrx_allocator_s allocator;
  // Borrowed-pointer registry used by compatibility callers.
  hrx_buffer_table_t buffer_table;
  // Arena blocks shared by graph recordings on this device.
  iree_arena_block_pool_t block_pool;
  // NUL-terminated human-readable device name.
  char name[128];
  // NUL-terminated canonical executable target key.
  char architecture[64];
} hrx_device_s;

// Initializes an inline view after native group sealing. All fields required
// for deinitialization are valid even when initialization fails.
iree_status_t hrx_device_initialize(hrx_device_group_t group, size_t index,
                                    hrx_accelerator_type_t type,
                                    hrx_device_t device);
void hrx_device_deinitialize(hrx_device_t device);
// Consumes a terminal failure and publishes it through the owning event sink.
void hrx_device_report_failure(hrx_device_t device, iree_status_t status);

// Returns the borrowed HAL device underlying |device|, or NULL when |device|
// is NULL.
iree_hal_device_t* hrx_device_hal(hrx_device_t device);

// Returns recording modes required by the device's active HRX profile session.
// OR these into modes used to create command buffers that may contain
// dispatches. The session is established before publishing the device and ends
// at teardown.
iree_hal_command_buffer_mode_t hrx_device_profile_command_buffer_mode(
    hrx_device_t device);

// Queries immutable total device memory from the HAL device spec. Returns OK
// with |out_known| false when the spec does not describe a known capacity.
hrx_status_t hrx_device_query_total_memory_from_spec(
    hrx_device_t device, bool* out_known, iree_device_size_t* out_total);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_DEVICE_H_
