// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_WEBGPU_WEBGPU_BUFFER_H_
#define IREE_HAL_DRIVER_WEBGPU_WEBGPU_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/webgpu/handle_table.h"
#include "iree/hal/driver/webgpu/webgpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_buffer_t
//===----------------------------------------------------------------------===//

enum iree_hal_webgpu_buffer_binding_e {
  IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE = 0,
  IREE_HAL_WEBGPU_BUFFER_BINDING_COUNT = 1,
};

// Returns the native binding table layout published by WebGPU buffers.
const iree_hal_buffer_binding_layout_t* iree_hal_webgpu_buffer_binding_layout(
    void);

// Wraps a WebGPU GPUBuffer (referenced by bridge handle) as an IREE HAL
// buffer. The buffer tracks its bridge handle and maintains a shadow buffer
// in wasm linear memory for host-side mapping operations.
//
// For HOST_LOCAL|DEVICE_VISIBLE buffers (staging write), the GPU buffer is
// created with mappedAtCreation:true and a shadow buffer is allocated
// immediately. The host writes into the shadow, and on unmap the shadow
// contents are copied to the GPU mapped range before the GPU buffer is
// unmapped.
//
// For HOST_VISIBLE|DEVICE_LOCAL buffers (staging read), the GPU buffer must
// be async-mapped via mapAsync before the host can read. On map_range the
// GPU mapped range is copied into a shadow buffer; on unmap_range the shadow
// is freed and the GPU buffer is unmapped.
//
// Device-local buffers (no HOST_VISIBLE) do not support host mapping.

// Creates a WebGPU buffer wrapping a bridge handle.
//
// |buffer_handle| is the bridge handle for the GPUBuffer itself.
// |mapped_at_creation| indicates whether the GPU buffer was created with
// mappedAtCreation:true (staging write path).
//
// If |mapped_at_creation| is true, a shadow buffer is immediately allocated
// in wasm linear memory. The caller can then map the buffer and write into
// the shadow.
iree_status_t iree_hal_webgpu_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_hal_webgpu_handle_t buffer_handle, bool mapped_at_creation,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the WebGPU bridge handle for the underlying GPUBuffer.
iree_hal_webgpu_handle_t iree_hal_webgpu_buffer_handle(
    const iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_WEBGPU_WEBGPU_BUFFER_H_
