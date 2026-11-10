// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_LIBHRX_RUNTIME_H_
#define LIBHRX_SRC_LIBHRX_RUNTIME_H_

#include "hrx_runtime.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;

// Selects and orders physical devices before any device is activated. Entries
// may be reordered or removed in place; their opaque IDs and borrowed strings
// remain unchanged. An empty result is a valid selection. The callback and its
// context are borrowed only for initialization and are never retained.
typedef struct hrx_gpu_device_selector_t {
  // Caller-owned selection policy, or NULL when no context is required.
  void* self;
  // Optional cold-path selector. The input excludes aggregate pseudo-devices.
  iree_status_t (*select)(void* self, iree_string_view_t driver_name,
                          iree_allocator_t host_allocator,
                          iree_host_size_t* device_count,
                          iree_hal_device_info_t* device_infos);
} hrx_gpu_device_selector_t;

// Internal initialization inputs consumed before publishing the device group.
typedef struct hrx_gpu_initialize_options_t {
  // Public HRX initialization flags.
  uint32_t flags;
  // Immutable creation extension chain and provider data borrowed until
  // hrx_gpu_shutdown().
  const iree_hal_device_create_params_extension_t* device_extensions;
  // Frontend-owned device selection; zero preserves the driver's order.
  hrx_gpu_device_selector_t device_selector;
} hrx_gpu_initialize_options_t;

// Initializes only the selected GPUs, then seals their shared device group.
hrx_status_t hrx_gpu_initialize_with_options(
    const hrx_gpu_initialize_options_t* options);

// Drains completed profile records after a host synchronization boundary.
// Inactive or absent devices are a no-op. Serializes producer flushes and
// writes to the runtime's shared profile file, without holding execution locks
// or waiting for other queues. File errors are returned to the synchronizing
// caller.
iree_status_t hrx_runtime_profile_flush(hrx_device_t device);

// Returns the process-wide borrowed proactor pool, or NULL before runtime
// initialization and after shutdown.
iree_async_proactor_pool_t* hrx_runtime_proactor_pool(void);

// Returns the configured application event sink translated to the HAL event
// ABI. Returns false when HRX would otherwise discard device events.
bool hrx_runtime_try_get_hal_device_event_sink(
    iree_hal_device_event_sink_t* out_sink);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_RUNTIME_H_
