// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_allocator.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_allocator_t
//===----------------------------------------------------------------------===//

// WebGPU exposes three memory heaps corresponding to the three buffer
// categories defined by the GPUBufferUsage constraints:
//
// 1. Device-local: STORAGE | COPY_SRC | COPY_DST (plus UNIFORM, INDIRECT).
//    Not host-mappable. Used for GPU compute buffers.
//
// 2. Staging write: MAP_WRITE | COPY_SRC. Created with mappedAtCreation:true.
//    Host writes data, unmaps to trigger GPU upload, then copies to
//    device-local. This is the HOST_LOCAL | DEVICE_VISIBLE path.
//
// 3. Staging read: MAP_READ | COPY_DST. Created normally, mapped after GPU
//    copies results into it via mapAsync. Host reads results from the mapped
//    range.
//
// WebGPU constrains mappable buffer usage: MAP_READ can only combine with
// COPY_DST, and MAP_WRITE can only combine with COPY_SRC. This means a
// single buffer cannot be both mappable and usable as a storage binding.

#define IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT 3

typedef struct iree_hal_webgpu_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_webgpu_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_webgpu_allocator_vtable;

static iree_hal_webgpu_allocator_t* iree_hal_webgpu_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_allocator_vtable);
  return (iree_hal_webgpu_allocator_t*)base_value;
}

iree_status_t iree_hal_webgpu_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_webgpu_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_webgpu_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_webgpu_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_allocator_t* allocator =
      iree_hal_webgpu_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_webgpu_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_webgpu_allocator_t* allocator =
      (iree_hal_webgpu_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static void iree_hal_webgpu_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  (void)base_allocator;
  (void)out_statistics;
}

static iree_status_t iree_hal_webgpu_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_ASSERT_ARGUMENT(out_count);

  // WebGPU devices do not expose heap size information. We report a large max
  // allocation size; the actual limit depends on the device's maxBufferSize
  // limit (typically 256MB–2GB), which is checked at buffer creation time by
  // the WebGPU implementation itself.
  const iree_device_size_t max_allocation_size = ~(iree_device_size_t)0;
  const iree_device_size_t min_alignment = 256;

  *out_count = IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT;
  if (capacity < IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  int i = 0;

  // Heap 0: Device-local memory.
  // GPU-optimal storage for compute dispatch. Not host-mappable.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Heap 1: Staging write (host→GPU upload).
  // HOST_LOCAL | DEVICE_VISIBLE with MAP_WRITE | COPY_SRC.
  // Created with mappedAtCreation:true for immediate host write access.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Heap 2: Staging read (GPU→host download).
  // HOST_VISIBLE | DEVICE_VISIBLE with MAP_READ | COPY_DST.
  // Mapped via mapAsync after GPU writes results into this buffer.
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  IREE_ASSERT(i == IREE_HAL_WEBGPU_ALLOCATOR_HEAP_COUNT);
  return iree_ok_status();
}

static const iree_hal_allocator_vtable_t iree_hal_webgpu_allocator_vtable = {
    .destroy = iree_hal_webgpu_allocator_destroy,
    .host_allocator = iree_hal_webgpu_allocator_host_allocator,
    .query_statistics = iree_hal_webgpu_allocator_query_statistics,
    .query_memory_heaps = iree_hal_webgpu_allocator_query_memory_heaps,
};
