// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Memory pool implementation. Wraps IREE virtual memory APIs to provide
// stream-ordered memory management. Pool lifecycle and attribute management
// are fully implemented here; async alloc/free paths are in the binding
// layer since they require stream host callback support.

#include <stdlib.h>

#include "hrx_internal.h"

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
      pool->supports_virtual_memory = false;
      iree_status_ignore(vm_status);
    }
  }

  *out_pool = pool;
  return hrx_ok_status();
}

static void hrx_mem_pool_destroy(hrx_mem_pool_s* pool) {
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
  return hrx_ok_status();
}
