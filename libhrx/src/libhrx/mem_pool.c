// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Memory pool implementation. Wraps IREE HAL pools to provide stream-ordered
// memory-management policy. Pool lifecycle, allocation, and attribute
// management are implemented here; async alloc/free sequencing remains in the
// binding layer because it requires stream host callback support.

#include "mem_pool.h"

#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"
#include "iree/hal/memory/asan.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "vmm_slab_provider.h"

//===----------------------------------------------------------------------===//
// Pool configuration
//===----------------------------------------------------------------------===//

// Default slab length for growable GPU allocation pools.
static const iree_device_size_t HRX_MEM_POOL_GPU_SLAB_LENGTH_DEFAULT =
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
    hrx_mem_pool_t pool, iree_device_size_t* out_range_length) {
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
      // TLSF grows by this range length one slab at a time. A slab is fully
      // committed on first use, so its size bounds per-pool idle memory rather
      // than expressing a fraction of total device memory.
      range_length = HRX_MEM_POOL_GPU_SLAB_LENGTH_DEFAULT;
    } else {
      range_length = HRX_MEM_POOL_CPU_RANGE_LENGTH_DEFAULT;
    }
  }

  if (!iree_device_size_checked_align(range_length, HRX_MEM_POOL_ALIGNMENT,
                                      &range_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory pool range length overflows alignment");
  }
  *out_range_length = range_length;
  return iree_ok_status();
}

static void hrx_mem_pool_refresh_stats_locked(hrx_mem_pool_t pool) {
  iree_hal_pool_stats_t tlsf_stats = {0};
  iree_hal_pool_stats_t oversized_stats = {0};
  if (pool->hal_pool) {
    iree_hal_pool_query_stats(pool->hal_pool, &tlsf_stats);
  }
  if (pool->oversized_hal_pool) {
    iree_hal_pool_query_stats(pool->oversized_hal_pool, &oversized_stats);
  }
  const uint64_t previous_reserved_mem_current = pool->reserved_mem_current;
  pool->reserved_mem_current =
      tlsf_stats.bytes_committed > UINT64_MAX - oversized_stats.bytes_committed
          ? UINT64_MAX
          : tlsf_stats.bytes_committed + oversized_stats.bytes_committed;
  if (pool->reserved_mem_current > previous_reserved_mem_current) {
    pool->reserved_mem_high =
        iree_max(pool->reserved_mem_high, pool->reserved_mem_current);
  }
}

// Detaches both allocation classes only after every reservation has retired.
// Callers must hold |pool->mutex| and release returned references after
// unlocking.
static void hrx_mem_pool_take_idle_hal_pools_locked(
    hrx_mem_pool_t pool, iree_hal_pool_t** out_hal_pool,
    iree_hal_pool_t** out_oversized_hal_pool) {
  *out_hal_pool = NULL;
  *out_oversized_hal_pool = NULL;
  if ((!pool->hal_pool && !pool->oversized_hal_pool) ||
      pool->inflight_allocation_count != 0) {
    return;
  }

  iree_hal_pool_stats_t tlsf_stats = {0};
  iree_hal_pool_stats_t oversized_stats = {0};
  if (pool->hal_pool) {
    iree_hal_pool_query_stats(pool->hal_pool, &tlsf_stats);
  }
  if (pool->oversized_hal_pool) {
    iree_hal_pool_query_stats(pool->oversized_hal_pool, &oversized_stats);
  }
  if (tlsf_stats.bytes_reserved != 0 || oversized_stats.bytes_reserved != 0 ||
      pool->allocation_budget_current != 0) {
    return;
  }

  *out_hal_pool = pool->hal_pool;
  *out_oversized_hal_pool = pool->oversized_hal_pool;
  pool->hal_pool = NULL;
  pool->oversized_hal_pool = NULL;
  pool->reserved_mem_current = 0;
  pool->used_mem_current = 0;
}

// Device-local pools use the allocator VMM contract when it is available. This
// gives every pool slab a stable device address while preserving the existing
// TLSF policy for suballocation, trimming, and reservation accounting. ASAN
// owns a separate slab lifecycle, so its configured device provider remains
// authoritative for instrumented allocations.
static bool hrx_mem_pool_uses_virtual_memory_slabs(
    const hrx_mem_pool_t pool,
    const iree_hal_queue_pool_backend_t* queue_pool_backend) {
  return pool->device->type == HRX_ACCELERATOR_GPU &&
         iree_hal_allocator_supports_virtual_memory(
             pool->device->allocator.hal_allocator) &&
         !iree_hal_asan_pool_options_is_enabled(&queue_pool_backend->asan);
}

static iree_status_t hrx_mem_pool_ensure_hal_pools_locked(hrx_mem_pool_t pool) {
  if (pool->hal_pool && pool->oversized_hal_pool) return iree_ok_status();
  IREE_ASSERT(!pool->hal_pool);
  IREE_ASSERT(!pool->oversized_hal_pool);

  iree_hal_queue_pool_backend_t backend;
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      pool->device->hal_device,
      iree_hal_queue_family(pool->device->transfer_queue), &backend));
  if (!backend.notification) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL queue-pool backend returned no allocation "
                            "notification");
  }

  iree_device_size_t range_length = 0;
  IREE_RETURN_IF_ERROR(hrx_mem_pool_query_range_length(pool, &range_length));

  iree_hal_slab_provider_t* slab_provider = backend.slab_provider;
  bool owns_slab_provider = false;
  if (hrx_mem_pool_uses_virtual_memory_slabs(pool, &backend)) {
    const iree_hal_buffer_params_t physical_buffer_params = {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
    };
    IREE_RETURN_IF_ERROR(hrx_vmm_slab_provider_create(
        pool->device->allocator.hal_allocator, physical_buffer_params,
        iree_allocator_system(), &slab_provider));
    owns_slab_provider = true;
  }
  if (!slab_provider) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL memory pool has no slab provider");
  }

  iree_hal_tlsf_pool_options_t options = {0};
  options.tlsf_options.range_length = range_length;
  options.tlsf_options.alignment = HRX_MEM_POOL_ALIGNMENT;
  options.tlsf_options.frontier_capacity =
      IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
  options.asan = backend.asan;
  options.budget_limit = 0;
  options.trace_name = iree_make_cstring_view("hrx-mem-pool");

  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_tlsf_pool_create(
      options, slab_provider, backend.notification, backend.epoch_query,
      iree_allocator_system(), &hal_pool);
  if (!iree_status_is_ok(status)) {
    if (owns_slab_provider) iree_hal_slab_provider_release(slab_provider);
    return status;
  }

  iree_hal_passthrough_pool_options_t oversized_options = {
      .asan = backend.asan,
      .trace_name = iree_make_cstring_view("hrx-mem-pool-oversized"),
  };
  iree_hal_pool_t* oversized_hal_pool = NULL;
  status = iree_hal_passthrough_pool_create(
      oversized_options, slab_provider, backend.notification,
      iree_allocator_system(), &oversized_hal_pool);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release(hal_pool);
    if (owns_slab_provider) iree_hal_slab_provider_release(slab_provider);
    return status;
  }
  if (owns_slab_provider) iree_hal_slab_provider_release(slab_provider);

  pool->hal_pool = hal_pool;
  pool->oversized_hal_pool = oversized_hal_pool;
  pool->suballocation_max_size = range_length;
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
  pool->inflight_allocation_count = 0;
  pool->reuse_allow_internal_dependencies = true;
  pool->reuse_follow_event_dependencies = true;
  pool->reuse_allow_opportunistic = true;
  pool->reserved_mem_current = 0;
  pool->reserved_mem_high = 0;
  pool->used_mem_current = 0;
  pool->used_mem_high = 0;
  pool->platform_handle = NULL;
  iree_slim_mutex_initialize(&pool->mutex);

  *out_pool = pool;
  return hrx_ok_status();
}

static void hrx_mem_pool_destroy(hrx_mem_pool_s* pool) {
  iree_hal_pool_release(pool->hal_pool);
  iree_hal_pool_release(pool->oversized_hal_pool);
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
    case HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH:
      if (value != 0) {
        status = hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "reserved memory high watermark must reset to zero");
      } else {
        pool->reserved_mem_high = 0;
      }
      break;
    case HRX_MEM_POOL_ATTR_USED_MEM_HIGH:
      if (value != 0) {
        status =
            hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "used memory high watermark must reset to zero");
      } else {
        pool->used_mem_high = 0;
      }
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

  iree_hal_pool_t* idle_hal_pool = NULL;
  iree_hal_pool_t* idle_oversized_hal_pool = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status = pool->hal_pool ? iree_hal_tlsf_pool_trim_to(
                                              pool->hal_pool, min_bytes_to_keep)
                                        : iree_ok_status();
  if (iree_status_is_ok(status)) {
    hrx_mem_pool_refresh_stats_locked(pool);
    if (min_bytes_to_keep == 0) {
      hrx_mem_pool_take_idle_hal_pools_locked(pool, &idle_hal_pool,
                                              &idle_oversized_hal_pool);
    }
  }
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_pool_release(idle_hal_pool);
  iree_hal_pool_release(idle_oversized_hal_pool);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_mem_pool_release_unused(hrx_mem_pool_t pool) {
  if (!pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "pool is NULL");
  }

  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_pool_t* idle_hal_pool = NULL;
  iree_hal_pool_t* idle_oversized_hal_pool = NULL;
  iree_status_t status =
      pool->hal_pool
          ? iree_hal_tlsf_pool_trim_to(pool->hal_pool, pool->release_threshold)
          : iree_ok_status();
  if (iree_status_is_ok(status)) {
    hrx_mem_pool_refresh_stats_locked(pool);
    if (pool->release_threshold == 0) {
      hrx_mem_pool_take_idle_hal_pools_locked(pool, &idle_hal_pool,
                                              &idle_oversized_hal_pool);
    }
  }
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_pool_release(idle_hal_pool);
  iree_hal_pool_release(idle_oversized_hal_pool);
  return hrx_status_from_iree(status);
}

void hrx_mem_pool_record_logical_allocation(hrx_mem_pool_t pool, size_t size) {
  if (!pool || size == 0) return;

  iree_slim_mutex_lock(&pool->mutex);
  if (size >= UINT64_MAX - pool->used_mem_current) {
    pool->used_mem_current = UINT64_MAX;
  } else {
    pool->used_mem_current += size;
  }
  pool->used_mem_high = iree_max(pool->used_mem_high, pool->used_mem_current);
  iree_slim_mutex_unlock(&pool->mutex);
}

void hrx_mem_pool_record_logical_free(hrx_mem_pool_t pool, size_t size) {
  if (!pool || size == 0) return;

  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(pool->used_mem_current >= size);
  pool->used_mem_current -= size;
  iree_slim_mutex_unlock(&pool->mutex);
}

void hrx_mem_pool_release_allocation_budget(hrx_mem_pool_t pool, size_t size) {
  if (!pool || size == 0) return;

  iree_slim_mutex_lock(&pool->mutex);
  IREE_ASSERT(pool->allocation_budget_current >= size);
  pool->allocation_budget_current -= size;
  iree_slim_mutex_unlock(&pool->mutex);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

static iree_status_t hrx_mem_pool_allocate_hal_buffer(
    hrx_mem_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_hal_pool_t** out_hal_pool,
    iree_hal_buffer_t** out_buffer) {
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
  iree_status_t status = hrx_mem_pool_ensure_hal_pools_locked(pool);
  if (iree_status_is_ok(status) && pool->props.max_size != 0) {
    if (pool->allocation_budget_current > pool->props.max_size ||
        size > pool->props.max_size - pool->allocation_budget_current) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "memory pool allocation exceeds max size");
    } else {
      pool->allocation_budget_current += size;
    }
  }
  iree_hal_pool_t* hal_pool = NULL;
  if (iree_status_is_ok(status)) {
    hal_pool = size <= pool->suballocation_max_size ? pool->hal_pool
                                                    : pool->oversized_hal_pool;
    iree_hal_pool_retain(hal_pool);
    ++pool->inflight_allocation_count;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (!iree_status_is_ok(status)) return status;

  status = iree_hal_pool_allocate_buffer(hal_pool, params, size,
                                         /*requester_frontier=*/NULL,
                                         iree_immediate_timeout(), out_buffer);
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
    iree_status_free(status);
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "memory pool has no immediately reusable capacity for allocation");
  }
  iree_slim_mutex_lock(&pool->mutex);
  --pool->inflight_allocation_count;
  if (!iree_status_is_ok(status) && pool->props.max_size != 0) {
    IREE_ASSERT(pool->allocation_budget_current >= size);
    pool->allocation_budget_current -= size;
  }
  if (iree_status_is_ok(status)) {
    hrx_mem_pool_refresh_stats_locked(pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);

  if (iree_status_is_ok(status)) {
    *out_hal_pool = hal_pool;
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

  iree_hal_queue_family_affinity_t queue_family_affinity = 0;
  iree_status_t status = hrx_hal_queue_affinity_to_family_affinity(
      pool->device->hal_device, params.queue_affinity, &queue_family_affinity);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }
  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_family_affinity = queue_family_affinity,
  };

  iree_hal_pool_t* hal_pool = NULL;
  iree_hal_buffer_t* hal_buffer = NULL;
  status = hrx_mem_pool_allocate_hal_buffer(
      pool, hal_params, (iree_device_size_t)size, &hal_pool, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_buffer_t buf = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(hrx_buffer_s),
                                 (void**)&buf);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(hal_buffer);
    iree_hal_pool_release(hal_pool);
    if (pool->props.max_size != 0) {
      hrx_mem_pool_release_allocation_budget(pool, size);
    }
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
  if (pool->props.max_size != 0) {
    hrx_mem_pool_retain(pool);
    buf->allocation_budget_pool = pool;
    buf->allocation_budget_size = size;
  }
  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}
