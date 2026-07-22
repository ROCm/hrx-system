// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/rocm_device_heap.h"

#include <stddef.h>
#include <string.h>

#include "common/direct_transfer.h"
#include "common/internal.h"

// Device malloc's heap layout is defined by heap_t in the device library:
// https://github.com/ROCm/llvm-project/blob/52226beb248fcdd136d084307a12207d2fc00220/amd/device-libs/ockl/src/dm.cl#L23-L179
// This host mirror derives the initialized field offsets from that ABI instead
// of embedding offsets computed from its cache-line-padded members.
#define IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE (128 * 1024)
#define IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT 16
#define IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA 256
#define IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE (2 * 1024 * 1024)

typedef struct iree_hal_streaming_rocm_malloc_cache_line_t {
  // Storage for one atomic value and its device-library cache-line padding.
  uint8_t storage[128];
} iree_hal_streaming_rocm_malloc_cache_line_t;
static_assert(sizeof(iree_hal_streaming_rocm_malloc_cache_line_t) == 128,
              "device malloc cache-line record must be 128 bytes");

typedef struct iree_hal_streaming_rocm_malloc_sdata_t {
  // Device address of a second-level sdata array.
  uint64_t array;
  // Device address of the tracked slab.
  uint64_t slab_address;
  // Number of allocated blocks in the slab.
  uint32_t used_block_count;
  // Tail padding preserving the device structure alignment.
  uint32_t padding;
} iree_hal_streaming_rocm_malloc_sdata_t;
static_assert(sizeof(iree_hal_streaming_rocm_malloc_sdata_t) == 24,
              "device malloc slab record must be 24 bytes");

typedef struct iree_hal_streaming_rocm_malloc_heap_prefix_t {
  // Per-kind search cursors.
  iree_hal_streaming_rocm_malloc_cache_line_t
      starts[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT];
  // Per-kind allocated slab counts.
  iree_hal_streaming_rocm_malloc_cache_line_t
      allocated_counts[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT];
  // Per-kind recordable slab counts initialized by the host.
  iree_hal_streaming_rocm_malloc_cache_line_t
      recordable_counts[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT];
  // Per-kind most recent slab-allocation timestamps.
  iree_hal_streaming_rocm_malloc_cache_line_t
      allocation_times[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT];
  // Per-kind most recent recordable-growth timestamps.
  iree_hal_streaming_rocm_malloc_cache_line_t
      growth_times[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT];
  // First-level slab tracking records.
  iree_hal_streaming_rocm_malloc_sdata_t
      slab_data[IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT]
               [IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA];
  // Next address in the initial slab window.
  uint64_t initial_slabs;
  // End address of the initial slab window.
  uint64_t initial_slabs_end;
  // Start address of the initial slab window.
  uint64_t initial_slabs_start;
} iree_hal_streaming_rocm_malloc_heap_prefix_t;
static_assert(
    offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t, recordable_counts) ==
        2 * IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT *
            sizeof(iree_hal_streaming_rocm_malloc_cache_line_t),
    "device malloc recordable-count offset must match the device ABI");
static_assert(offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t,
                       initial_slabs) ==
                  5 * IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT *
                          sizeof(iree_hal_streaming_rocm_malloc_cache_line_t) +
                      IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT *
                          IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA *
                          sizeof(iree_hal_streaming_rocm_malloc_sdata_t),
              "device malloc initial-slab offset must match the device ABI");
static_assert(sizeof(iree_hal_streaming_rocm_malloc_heap_prefix_t) <=
                  IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
              "device malloc heap prefix must fit in its allocation");

static void iree_hal_streaming_rocm_store_u32(uint8_t* storage,
                                              iree_host_size_t offset,
                                              uint32_t value) {
  memcpy(storage + offset, &value, sizeof(value));
}

static void iree_hal_streaming_rocm_store_u64(uint8_t* storage,
                                              iree_host_size_t offset,
                                              uint64_t value) {
  memcpy(storage + offset, &value, sizeof(value));
}

static iree_status_t iree_hal_streaming_rocm_populate_malloc_heap_header(
    void* heap_storage, uint64_t initial_slabs_device_ptr,
    iree_device_size_t initial_slab_bytes) {
  if (IREE_UNLIKELY(initial_slab_bytes >
                    UINT64_MAX - initial_slabs_device_ptr)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ROCm device malloc initial slab range overflow");
  }

  uint8_t* heap = (uint8_t*)heap_storage;
  memset(heap, 0, IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE);

  for (uint32_t i = 0; i < IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT; ++i) {
    const iree_host_size_t offset =
        offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t,
                 recordable_counts) +
        (iree_host_size_t)i *
            sizeof(iree_hal_streaming_rocm_malloc_cache_line_t);
    iree_hal_streaming_rocm_store_u32(heap, offset,
                                      IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA);
  }

  const uint64_t initial_slabs_start = initial_slabs_device_ptr;
  const uint64_t initial_slabs_end =
      initial_slabs_start + (uint64_t)initial_slab_bytes;
  iree_hal_streaming_rocm_store_u64(
      heap,
      offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t, initial_slabs),
      initial_slabs_start);
  iree_hal_streaming_rocm_store_u64(
      heap,
      offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t, initial_slabs_end),
      initial_slabs_end);
  iree_hal_streaming_rocm_store_u64(
      heap,
      offsetof(iree_hal_streaming_rocm_malloc_heap_prefix_t,
               initial_slabs_start),
      initial_slabs_start);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_rocm_allocate_device_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_usage_t usage,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer,
    uint64_t* out_device_ptr) {
  *out_buffer = NULL;
  *out_device_ptr = 0;

  iree_hal_buffer_params_t params = {
      .usage = usage | IREE_HAL_BUFFER_USAGE_SHARING_EXPORT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = context->queue_affinity,
  };

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      context->device_allocator, params, allocation_size, &buffer));

  iree_hal_external_buffer_t external_buffer;
  iree_status_t status = iree_hal_allocator_export_buffer(
      context->device_allocator, buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    *out_device_ptr = external_buffer.handle.device_allocation.ptr;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t
iree_hal_streaming_context_initialize_rocm_device_runtime_locked(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_device_size_t requested_heap_size =
      (iree_device_size_t)context->limits.malloc_heap_size;
  const iree_device_size_t initial_slab_count = iree_device_size_ceil_div(
      iree_max(
          requested_heap_size,
          (iree_device_size_t)IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE),
      IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE);
  iree_device_size_t initial_slab_bytes = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul(
          initial_slab_count,
          (iree_device_size_t)IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE,
          &initial_slab_bytes))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ROCm device malloc heap size overflow");
  }

  iree_hal_buffer_t* heap_buffer = NULL;
  iree_hal_buffer_t* initial_slabs_buffer = NULL;
  uint64_t heap_device_ptr = 0;
  uint64_t initial_slabs_device_ptr = 0;
  uint8_t* heap_header = NULL;

  iree_status_t status = iree_hal_streaming_rocm_allocate_device_buffer(
      context,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE, &heap_buffer, &heap_device_ptr);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_rocm_allocate_device_buffer(
        context, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE, initial_slab_bytes,
        &initial_slabs_buffer, &initial_slabs_device_ptr);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator,
                                   IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
                                   (void**)&heap_header);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_rocm_populate_malloc_heap_header(
        heap_header, initial_slabs_device_ptr, initial_slab_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_direct_transfer_h2d(
        context, heap_header, heap_buffer, /*target_offset=*/0,
        IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE);
  }

  if (iree_status_is_ok(status)) {
    context->rocm_device_runtime.malloc_heap_buffer = heap_buffer;
    context->rocm_device_runtime.malloc_initial_slabs_buffer =
        initial_slabs_buffer;
    iree_atomic_store(&context->rocm_device_runtime.malloc_heap_device_ptr,
                      heap_device_ptr, iree_memory_order_release);
    heap_buffer = NULL;
    initial_slabs_buffer = NULL;
  }

  iree_allocator_free(context->host_allocator, heap_header);
  iree_hal_buffer_release(initial_slabs_buffer);
  iree_hal_buffer_release(heap_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_rocm_device_malloc_heap(
    iree_hal_streaming_context_t* context, uint64_t* out_heap_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_heap_device_ptr);
  *out_heap_device_ptr = 0;
  const uint64_t cached_heap_device_ptr =
      iree_hal_streaming_rocm_device_runtime_cached_heap(
          &context->rocm_device_runtime);
  if (cached_heap_device_ptr != 0) {
    *out_heap_device_ptr = cached_heap_device_ptr;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status = iree_ok_status();
  if (!context->rocm_device_runtime.malloc_heap_buffer) {
    status = iree_hal_streaming_context_initialize_rocm_device_runtime_locked(
        context);
  }
  if (iree_status_is_ok(status)) {
    *out_heap_device_ptr = iree_hal_streaming_rocm_device_runtime_cached_heap(
        &context->rocm_device_runtime);
  }
  iree_slim_mutex_unlock(&context->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
