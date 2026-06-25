// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_READBACK_H_
#define EXPERIMENTAL_ID4_TOOLING_READBACK_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Host-owned bytes read back from one HAL buffer binding.
typedef struct id4_tooling_host_bytes_t {
  // Number of bytes in data.
  iree_host_size_t length;
  // Host allocation containing length bytes.
  uint8_t* data;
} id4_tooling_host_bytes_t;

// Options for asynchronously reading one HAL buffer binding to host memory.
typedef struct id4_tooling_readback_buffer_binding_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // HAL device owning the source buffer and queue.
  iree_hal_device_t* device;
  // Queue affinity used for the transfer submission.
  iree_hal_queue_affinity_t queue_affinity;
  // Source buffer binding to read.
  iree_hal_buffer_binding_t binding;
  // Semaphores that the readback transfer waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Host allocator used for the returned byte allocation.
  iree_allocator_t host_allocator;
} id4_tooling_readback_buffer_binding_options_t;

// Releases any storage owned by |bytes|.
void id4_tooling_host_bytes_deinitialize(id4_tooling_host_bytes_t* bytes,
                                         iree_allocator_t host_allocator);

// Reads one HAL buffer binding to a host allocation.
iree_status_t id4_tooling_readback_buffer_binding(
    const id4_tooling_readback_buffer_binding_options_t* options,
    id4_tooling_host_bytes_t* out_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_READBACK_H_
