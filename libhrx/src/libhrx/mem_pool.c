// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Memory pool implementation. Wraps IREE HAL pools to provide stream-ordered
// memory-management policy. Pool lifecycle, allocation, and attribute
// management are implemented here; async alloc/free sequencing remains in the
// binding layer because it requires stream host callback support.

#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"
#include "iree/hal/memory/tlsf_pool.h"

//===----------------------------------------------------------------------===//
// Pool configuration
//===----------------------------------------------------------------------===//

// Default range length for GPU HIP/CUDA allocation pools.
static const iree_device_size_t HRX_MEM_POOL_GPU_RANGE_LENGTH_DEFAULT =
    (iree_device_size_t)16 * 1024 * 1024 * 1024;

// Minimum range length for GPU HIP/CUDA allocation pools.
static const iree_device_size_t HRX_MEM_POOL_GPU_RANGE_LENGTH_MIN =
    (iree_device_size_t)256 * 1024 * 1024;

// Default range length for CPU/local allocation pools.
static const iree_device_size_t HRX_MEM_POOL_CPU_RANGE_LENGTH_DEFAULT =
    (iree_device_size_t)64 * 1024 * 1024;

// Minimum byte alignment for HRX memory-pool reservations.
static const iree_device_size_t HRX_MEM_POOL_ALIGNMENT = 256;

static iree_status_t hrx_mem_pool_parse_range_length_env(
    const char* name, bool* out_found, iree_device_size_t* out_length) {
  *out_found = false;
  *out_length = 0;

  const char* value = getenv(name);
  if (!value || !value[0]) return iree_ok_status();

  char* end = NULL;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (value[0] < '0' || value[0] > '9' || !end || *end != '\0' || parsed == 0 ||
      parsed > (unsigned long long)IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s must be a positive byte count, got '%s'", name,
                            value);
  }

  *out_found = true;
  *out_length = (iree_device_size_t)parsed;
  return iree_ok_status();
}

static iree_status_t hrx_mem_pool_query_range_length(
    hrx_mem_pool_t pool, iree_device_size_t min_allocation_size,
    iree_device_size_t* out_range_length) {
  bool has_env_range_length = false;
  iree_device_size_t range_length = 0;
  IREE_RETURN_IF_ERROR(hrx_mem_pool_parse_range_length_env(
      "HRX_MEM_POOL_BYTES", &has_env_range_length, &range_length));
  if (!has_env_range_length) {
    IREE_RETURN_IF_ERROR(hrx_mem_pool_parse_range_length_env(
        "HRX_HIP_POOL_BYTES", &has_env_range_length, &range_length));
  }

  if (!has_env_range_length) {
    if (pool->device->type == HRX_ACCELERATOR_GPU) {
      bool total_memory_known = false;
      iree_device_size_t total_memory = 0;
      hrx_status_t memory_status = hrx_device_query_total_memory_from_spec(
          pool->device, &total_memory_known, &total_memory);
      IREE_RETURN_IF_ERROR(hrx_status_to_iree(memory_status));
      range_length = HRX_MEM_POOL_GPU_RANGE_LENGTH_DEFAULT;
      if (total_memory_known && total_memory > 0) {
        range_length = total_memory - total_memory / 4;
        range_length =
            iree_max(range_length, HRX_MEM_POOL_GPU_RANGE_LENGTH_MIN);
      }
    } else {
      range_length = HRX_MEM_POOL_CPU_RANGE_LENGTH_DEFAULT;
    }
  }

  range_length = iree_max(range_length, min_allocation_size);
  if (!iree_device_size_checked_align(range_length, HRX_MEM_POOL_ALIGNMENT,
                                      &range_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory pool range length overflows alignment");
  }
  *out_range_length = range_length;
  return iree_ok_status();
}

static void hrx_mem_pool_refresh_stats_locked(hrx_mem_pool_t pool) {
  if (!pool->hal_pool) return;
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool->hal_pool, &stats);
  pool->reserved_mem_current = stats.bytes_committed;
  pool->reserved_mem_high =
      iree_max(pool->reserved_mem_high, pool->reserved_mem_current);
  pool->used_mem_current = stats.bytes_reserved;
  pool->used_mem_high = iree_max(pool->used_mem_high, pool->used_mem_current);
}

static iree_status_t hrx_mem_pool_ensure_hal_pool_locked(
    hrx_mem_pool_t pool, iree_device_size_t min_allocation_size) {
  if (pool->hal_pool) return iree_ok_status();

  iree_hal_queue_pool_backend_t backend;
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      pool->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, &backend));
  if (!backend.slab_provider || !backend.notification) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL queue-pool backend returned an incomplete pool bundle");
  }

  iree_device_size_t range_length = 0;
  IREE_RETURN_IF_ERROR(hrx_mem_pool_query_range_length(
      pool, min_allocation_size, &range_length));

  iree_hal_tlsf_pool_options_t options = {0};
  options.tlsf_options.range_length = range_length;
  options.tlsf_options.alignment = HRX_MEM_POOL_ALIGNMENT;
  options.tlsf_options.frontier_capacity =
      IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
  options.asan = backend.asan;
  options.trace_name = iree_make_cstring_view("hrx-mem-pool");

  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_create(
      options, backend.slab_provider, backend.notification, backend.epoch_query,
      iree_allocator_system(), &pool->hal_pool));
  hrx_mem_pool_refresh_stats_locked(pool);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

hrx_status_t hrx_mem_pool_create(hrx_device_t device,
                                 const hrx_mem_pool_props_t* props,
                                 hrx_mem_pool_t* out_pool) {
  if (!device || !props || !out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device, props, or out_pool is NULL");
  }

  hrx_mem_pool_s* pool = (hrx_mem_pool_s*)calloc(1, sizeof(hrx_mem_pool_s));
  if (!pool) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate mem pool");
  }

  iree_atomic_ref_count_init(&pool->ref_count);
  pool->device = device;
  hrx_device_retain(pool->device);
  pool->props = *props;
  pool->release_threshold = 0;
  pool->reuse_allow_internal_dependencies = false;
  pool->reuse_follow_event_dependencies = true;
  pool->reuse_allow_opportunistic = false;
  pool->reserved_mem_current = 0;
  pool->reserved_mem_high = 0;
  pool->used_mem_current = 0;
  pool->used_mem_high = 0;
  pool->platform_handle = NULL;
  iree_slim_mutex_initialize(&pool->mutex);

  // Check if the device allocator supports virtual memory.
  pool->supports_virtual_memory = false;
  pool->vm_page_size_min = 0;
  pool->vm_page_size_recommended = 0;

  iree_hal_allocator_t* hal_allocator = device->allocator.hal_allocator;
  if (iree_hal_allocator_supports_virtual_memory(hal_allocator)) {
    pool->supports_virtual_memory = true;

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    iree_status_t vm_status =
        iree_hal_allocator_virtual_memory_query_granularity(
            hal_allocator, params, &pool->vm_page_size_min,
            &pool->vm_page_size_recommended);
    if (!iree_status_is_ok(vm_status)) {
      hrx_status_t status = hrx_status_from_iree(vm_status);
      hrx_device_release(pool->device);
      iree_slim_mutex_deinitialize(&pool->mutex);
      free(pool);
      return status;
    }
  }

  *out_pool = pool;
  return hrx_ok_status();
}

static void hrx_mem_pool_destroy(hrx_mem_pool_s* pool) {
  iree_hal_pool_release(pool->hal_pool);
  hrx_device_release(pool->device);
  iree_slim_mutex_deinitialize(&pool->mutex);
  free(pool);
}

void hrx_mem_pool_retain(hrx_mem_pool_t pool) {
  if (pool) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void hrx_mem_pool_release(hrx_mem_pool_t pool) {
  if (pool && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    hrx_mem_pool_destroy(pool);
  }
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

hrx_status_t hrx_mem_pool_get_attribute(hrx_mem_pool_t pool,
                                        hrx_mem_pool_attr_t attr,
                                        uint64_t* out_value) {
  if (!pool || !out_value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "pool or out_value is NULL");
  }
  *out_value = 0;

  iree_slim_mutex_lock(&pool->mutex);
  hrx_mem_pool_refresh_stats_locked(pool);

  hrx_status_t status = hrx_ok_status();
  switch (attr) {
    case HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      *out_value = pool->reuse_follow_event_dependencies ? 1 : 0;
      break;
    case HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      *out_value = pool->reuse_allow_internal_dependencies ? 1 : 0;
      break;
    case HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      *out_value = pool->reuse_allow_opportunistic ? 1 : 0;
      break;
    case HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      *out_value = pool->release_threshold;
      break;
    case HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT:
      *out_value = pool->reserved_mem_current;
      break;
    case HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH:
      *out_value = pool->reserved_mem_high;
      break;
    case HRX_MEM_POOL_ATTR_USED_MEM_CURRENT:
      *out_value = pool->used_mem_current;
      break;
    case HRX_MEM_POOL_ATTR_USED_MEM_HIGH:
      *out_value = pool->used_mem_high;
      break;
    default:
      status = hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "invalid memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

hrx_status_t hrx_mem_pool_set_attribute(hrx_mem_pool_t pool,
                                        hrx_mem_pool_attr_t attr,
                                        uint64_t value) {
  if (!pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }

  iree_slim_mutex_lock(&pool->mutex);

  hrx_status_t status = hrx_ok_status();
  switch (attr) {
    case HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      pool->reuse_follow_event_dependencies = value != 0;
      break;
    case HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      pool->reuse_allow_internal_dependencies = value != 0;
      break;
    case HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      pool->reuse_allow_opportunistic = value != 0;
      break;
    case HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      pool->release_threshold = value;
      break;
    default:
      status = hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "invalid or read-only memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

//===----------------------------------------------------------------------===//
// Trim
//===----------------------------------------------------------------------===//

hrx_status_t hrx_mem_pool_trim(hrx_mem_pool_t pool, size_t min_bytes_to_keep) {
  if (!pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }
  (void)min_bytes_to_keep;
  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status =
      pool->hal_pool ? iree_hal_pool_trim(pool->hal_pool) : iree_ok_status();
  if (iree_status_is_ok(status)) {
    hrx_mem_pool_refresh_stats_locked(pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return hrx_status_from_iree(status);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

static iree_status_t hrx_mem_pool_allocate_hal_buffer(
    hrx_mem_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_timeout_t timeout,
    iree_hal_pool_t** out_hal_pool, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_hal_pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_hal_pool = NULL;
  *out_buffer = NULL;
  if (!pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation size must be > 0");
  }

  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status = hrx_mem_pool_ensure_hal_pool_locked(pool, size);
  iree_hal_pool_t* hal_pool = pool->hal_pool;
  if (iree_status_is_ok(status)) {
    iree_hal_pool_retain(hal_pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (!iree_status_is_ok(status)) return status;

  status = iree_hal_pool_allocate_buffer(hal_pool, params, size,
                                         /*requester_frontier=*/NULL, timeout,
                                         out_buffer);
  if (iree_status_is_ok(status)) {
    *out_hal_pool = hal_pool;
    iree_slim_mutex_lock(&pool->mutex);
    hrx_mem_pool_refresh_stats_locked(pool);
    iree_slim_mutex_unlock(&pool->mutex);
  } else {
    iree_hal_pool_release(hal_pool);
  }
  return status;
}

hrx_status_t hrx_mem_pool_allocate_buffer(hrx_mem_pool_t pool,
                                          hrx_buffer_params_t params,
                                          size_t size, hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_mem_pool_allocate_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "buffer is NULL"));
  }
  *buffer = NULL;

  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_affinity = (iree_hal_queue_affinity_t)params.queue_affinity,
  };

  iree_hal_pool_t* hal_pool = NULL;
  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = hrx_mem_pool_allocate_hal_buffer(
      pool, hal_params, (iree_device_size_t)size, iree_infinite_timeout(),
      &hal_pool, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_buffer_t buf = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(hrx_buffer_s),
                                 (void**)&buf);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(hal_buffer);
    iree_hal_pool_release(hal_pool);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->hal_pool = hal_pool;
  buf->device = pool->device;
  hrx_device_retain(buf->device);
  buf->mem_type = params.type;
  buf->size = size;
  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}
