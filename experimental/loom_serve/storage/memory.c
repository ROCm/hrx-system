// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/memory.h"

struct loom_serve_memory_pool_t {
  // Allocator owning pool, buffer, and slab metadata.
  iree_allocator_t host_allocator;
  // Retained physical allocation domain.
  iree_hal_allocator_t* device_allocator;
  // Queue families permitted to access all reservations.
  iree_hal_queue_family_affinity_t queue_affinity;
  // Common independent physical allocation size in bytes.
  iree_device_size_t slab_size;
  // Minimum address alignment guaranteed by virtual reservations.
  iree_device_size_t minimum_alignment;
  // Physical admission ceiling; UINT64_MAX means unbounded.
  uint64_t limit_bytes;
  // Accounting across all buffers, with physical allocations counted once.
  loom_serve_memory_statistics_t statistics;
  // Borrowed owner of cold capacity reclamation, separate from slab mechanics.
  loom_serve_memory_reclaimer_t reclaimer;
};

typedef struct memory_slab_t {
  // Owned physical handle, or null for uncommitted storage.
  iree_hal_physical_memory_t* physical;
  // Whether physical is currently mapped into its virtual slot.
  bool mapped;
  // Whether a current maintenance range requires this slab.
  bool keep;
} memory_slab_t;

struct loom_serve_virtual_buffer_t {
  // Borrowed physical owner outliving this reservation.
  loom_serve_memory_pool_t* pool;
  // Borrowed caller accounting group, independent of the shared pool budget.
  loom_serve_memory_statistics_t* statistics;
  // Owned stable virtual reservation, independent of physical commitment.
  iree_hal_buffer_t* handle;
  // Number of equally sized slots spanning the reservation.
  iree_host_size_t slab_count;
  // Physical ownership and maintenance marks for each virtual slot.
  memory_slab_t slabs[];
};

static iree_hal_buffer_params_t memory_params(void) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return params;
}

iree_status_t loom_serve_memory_pool_create(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_affinity,
    iree_device_size_t slab_size, uint64_t limit_bytes,
    loom_serve_memory_pool_t** out_pool, iree_allocator_t host_allocator) {
  *out_pool = NULL;
  iree_device_size_t minimum = 0, recommended = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_query_granularity(
      device_allocator, memory_params(), &minimum, &recommended));
  if (!slab_size) {
    slab_size = recommended;
  }
  if (!slab_size || slab_size > INT64_MAX || slab_size % minimum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "slab size must be a multiple of %" PRIu64,
                            minimum);
  }
  loom_serve_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  pool->host_allocator = host_allocator;
  pool->device_allocator = device_allocator;
  iree_hal_allocator_retain(device_allocator);
  pool->queue_affinity = queue_affinity;
  pool->slab_size = slab_size;
  pool->minimum_alignment = minimum;
  pool->limit_bytes = limit_bytes ? limit_bytes : UINT64_MAX;
  *out_pool = pool;
  return iree_ok_status();
}

void loom_serve_memory_pool_destroy(loom_serve_memory_pool_t* pool) {
  if (!pool) {
    return;
  }
  IREE_ASSERT(!pool->statistics.reserved_bytes &&
              !pool->statistics.committed_bytes);
  iree_hal_allocator_release(pool->device_allocator);
  iree_allocator_free(pool->host_allocator, pool);
}

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    const loom_serve_memory_pool_t* pool) {
  return pool->statistics;
}

void loom_serve_memory_pool_set_reclaimer(
    loom_serve_memory_pool_t* pool, loom_serve_memory_reclaimer_t reclaimer) {
  pool->reclaimer = reclaimer;
}

iree_status_t loom_serve_memory_pool_prepare(loom_serve_memory_pool_t* pool,
                                             uint64_t additional_bytes,
                                             bool* out_admitted) {
  *out_admitted = false;
  if (additional_bytes > pool->limit_bytes) {
    return iree_ok_status();
  }
  const uint64_t target_bytes = pool->limit_bytes - additional_bytes;
  if (pool->statistics.committed_bytes > target_bytes && pool->reclaimer.fn) {
    IREE_RETURN_IF_ERROR(
        pool->reclaimer.fn(pool->reclaimer.user_data, target_bytes));
  }
  *out_admitted = pool->statistics.committed_bytes <= target_bytes;
  return iree_ok_status();
}

iree_status_t loom_serve_virtual_buffer_create(
    loom_serve_memory_pool_t* pool, iree_device_size_t length,
    iree_device_size_t alignment, loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!length || length > INT64_MAX - pool->slab_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual buffer extent");
  }
  if (!alignment || pool->minimum_alignment % alignment) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "virtual reservation cannot guarantee alignment %" PRIu64, alignment);
  }
  const uint64_t count = (length + pool->slab_size - 1) / pool->slab_size;
  if (count > (IREE_HOST_SIZE_MAX - sizeof(loom_serve_virtual_buffer_t)) /
                  sizeof(memory_slab_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "virtual buffer metadata is too large");
  }
  loom_serve_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*buffer) + count * sizeof(memory_slab_t),
      (void**)&buffer));
  buffer->pool = pool;
  buffer->statistics = statistics;
  buffer->slab_count = (iree_host_size_t)count;
  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      pool->device_allocator, pool->queue_affinity, count * pool->slab_size,
      &buffer->handle);
  if (iree_status_is_ok(status)) {
    pool->statistics.reserved_bytes += count * pool->slab_size;
    statistics->reserved_bytes += count * pool->slab_size;
    *out_buffer = buffer;
  } else {
    iree_allocator_free(pool->host_allocator, buffer);
  }
  return status;
}

iree_hal_buffer_t* loom_serve_virtual_buffer_handle(
    const loom_serve_virtual_buffer_t* buffer) {
  return buffer->handle;
}

static iree_status_t memory_slab_release(loom_serve_virtual_buffer_t* buffer,
                                         iree_host_size_t index) {
  loom_serve_memory_pool_t* pool = buffer->pool;
  memory_slab_t* slab = &buffer->slabs[index];
  if (slab->mapped) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_unmap(
        pool->device_allocator, buffer->handle, index * pool->slab_size,
        pool->slab_size));
    slab->mapped = false;
  }
  if (slab->physical) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_physical_memory_free(
        pool->device_allocator, slab->physical));
    slab->physical = NULL;
    pool->statistics.committed_bytes -= pool->slab_size;
    pool->statistics.released_bytes += pool->slab_size;
    buffer->statistics->committed_bytes -= pool->slab_size;
    buffer->statistics->released_bytes += pool->slab_size;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_virtual_buffer_commit(
    loom_serve_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  loom_serve_memory_pool_t* pool = buffer->pool;
  const iree_host_size_t begin = offset / pool->slab_size;
  const iree_host_size_t end = (offset + length - 1) / pool->slab_size + 1;
  uint64_t additional_bytes = 0;
  for (iree_host_size_t i = begin; i < end; ++i) {
    if (!buffer->slabs[i].physical) {
      additional_bytes += pool->slab_size;
    }
  }
  bool admitted = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_memory_pool_prepare(pool, additional_bytes, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical pool cannot commit %" PRIu64 " bytes",
                            additional_bytes);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = begin; i < end && iree_status_is_ok(status); ++i) {
    memory_slab_t* slab = &buffer->slabs[i];
    if (slab->physical) {
      continue;
    }
    status = iree_hal_allocator_physical_memory_allocate(
        pool->device_allocator, memory_params(), pool->slab_size,
        pool->host_allocator, &slab->physical);
    if (iree_status_is_ok(status)) {
      pool->statistics.committed_bytes += pool->slab_size;
      pool->statistics.peak_bytes = iree_max(pool->statistics.peak_bytes,
                                             pool->statistics.committed_bytes);
      buffer->statistics->committed_bytes += pool->slab_size;
      buffer->statistics->peak_bytes = iree_max(
          buffer->statistics->peak_bytes, buffer->statistics->committed_bytes);
      status = iree_hal_allocator_virtual_memory_map(
          pool->device_allocator, buffer->handle, i * pool->slab_size,
          slab->physical, 0, pool->slab_size);
      slab->mapped = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_allocator_virtual_memory_protect(
          pool->device_allocator, buffer->handle, i * pool->slab_size,
          pool->slab_size, pool->queue_affinity,
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
          IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
    }
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(status, memory_slab_release(buffer, i));
    }
  }
  return status;
}

void loom_serve_virtual_buffer_begin_trim(loom_serve_virtual_buffer_t* buffer) {
  for (iree_host_size_t i = 0; i < buffer->slab_count; ++i) {
    buffer->slabs[i].keep = false;
  }
}

void loom_serve_virtual_buffer_keep(loom_serve_virtual_buffer_t* buffer,
                                    iree_device_size_t offset,
                                    iree_device_size_t length) {
  const iree_device_size_t slab_size = buffer->pool->slab_size;
  const iree_host_size_t end = (offset + length - 1) / slab_size + 1;
  for (iree_host_size_t i = offset / slab_size; i < end; ++i) {
    buffer->slabs[i].keep = true;
  }
}

iree_status_t loom_serve_virtual_buffer_trim(
    loom_serve_virtual_buffer_t* buffer) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < buffer->slab_count && iree_status_is_ok(status); ++i) {
    if (!buffer->slabs[i].keep) {
      status = memory_slab_release(buffer, i);
    }
  }
  return status;
}

iree_status_t loom_serve_virtual_buffer_destroy(
    loom_serve_virtual_buffer_t* buffer) {
  if (!buffer) {
    return iree_ok_status();
  }
  loom_serve_memory_pool_t* pool = buffer->pool;
  loom_serve_virtual_buffer_begin_trim(buffer);
  IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_trim(buffer));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_release(
      pool->device_allocator, buffer->handle));
  pool->statistics.reserved_bytes -= buffer->slab_count * pool->slab_size;
  buffer->statistics->reserved_bytes -= buffer->slab_count * pool->slab_size;
  iree_allocator_free(pool->host_allocator, buffer);
  return iree_ok_status();
}
