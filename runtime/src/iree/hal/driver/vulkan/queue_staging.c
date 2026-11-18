// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/queue_staging.h"

#include <string.h>

#include "iree/hal/driver/vulkan/allocator.h"

iree_status_t iree_hal_vulkan_queue_staging_ring_allocate(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t slot_size, uint32_t slot_count,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_queue_staging_ring_t** out_ring) {
  *out_ring = NULL;
  if (slot_count == 0 || slot_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan staging ring requires non-zero capacity");
  }
  if (slot_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan staging ring slot size exceeds host addressable size");
  }
  if (slot_size > IREE_DEVICE_SIZE_MAX / slot_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring allocation size overflows");
  }

  iree_host_size_t slots_size = 0;
  if (!iree_host_size_checked_mul(slot_count,
                                  sizeof(iree_hal_vulkan_queue_staging_slot_t),
                                  &slots_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring slot table overflows");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(**out_ring), slots_size,
                                  &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan staging ring metadata overflows");
  }

  iree_hal_vulkan_queue_staging_ring_t* ring = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&ring));
  memset(ring, 0, total_size);
  iree_slim_mutex_initialize(&ring->mutex);
  ring->host_allocator = host_allocator;
  ring->slot_size = slot_size;
  ring->slot_count = slot_count;
  ring->waiter_tail = &ring->waiter_head;

  const iree_device_size_t allocation_size = slot_size * slot_count;
  const iree_hal_vulkan_buffer_spec_t buffer_spec = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
      .queue_family_affinity = queue_family_affinity,
  };
  iree_status_t status = iree_hal_vulkan_allocator_allocate_staging_buffer(
      device_allocator, &buffer_spec, allocation_size, ring->host_allocator,
      &ring->host_allocation, &ring->buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        ring->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_MAP_FLAG_NONE,
        /*byte_offset=*/0, allocation_size, &ring->mapping);
  }
  if (iree_status_is_ok(status)) {
    uint8_t* host_pointer = ring->host_allocation.data
                                ? ring->host_allocation.data
                                : ring->mapping.contents.data;
    for (uint32_t i = 0; i < slot_count; ++i) {
      const iree_device_size_t slot_offset = slot_size * i;
      ring->slots[i].ordinal = i;
      ring->slots[i].buffer_offset = slot_offset;
      ring->slots[i].host_span = iree_make_byte_span(
          host_pointer + slot_offset, (iree_host_size_t)slot_size);
    }
    *out_ring = ring;
  } else {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&ring->mapping));
    iree_hal_buffer_release(ring->buffer);
    iree_allocator_free_aligned(ring->host_allocator,
                                ring->host_allocation.data);
    iree_slim_mutex_deinitialize(&ring->mutex);
    iree_allocator_free(host_allocator, ring);
  }
  return status;
}

void iree_hal_vulkan_queue_staging_ring_free(
    iree_hal_vulkan_queue_staging_ring_t* ring) {
  if (!ring) {
    return;
  }
  IREE_CHECK_OK(iree_hal_buffer_unmap_range(&ring->mapping));
  iree_hal_buffer_release(ring->buffer);
  iree_allocator_free_aligned(ring->host_allocator, ring->host_allocation.data);
  iree_slim_mutex_deinitialize(&ring->mutex);
  iree_allocator_free(ring->host_allocator, ring);
}

bool iree_hal_vulkan_queue_staging_ring_try_acquire(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t** out_slot) {
  *out_slot = NULL;
  iree_slim_mutex_lock(&ring->mutex);
  for (uint32_t i = 0; i < ring->slot_count; ++i) {
    if (!ring->slots[i].in_use) {
      ring->slots[i].in_use = true;
      *out_slot = &ring->slots[i];
      break;
    }
  }
  iree_slim_mutex_unlock(&ring->mutex);
  return *out_slot != NULL;
}

bool iree_hal_vulkan_queue_staging_ring_queue_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter,
    iree_hal_vulkan_queue_staging_waiter_fn_t fn, void* user_data,
    iree_hal_resource_t* resource) {
  iree_slim_mutex_lock(&ring->mutex);
  bool should_queue = true;
  for (uint32_t i = 0; i < ring->slot_count; ++i) {
    if (!ring->slots[i].in_use) {
      should_queue = false;
      break;
    }
  }
  if (should_queue && !waiter->is_queued) {
    waiter->next = NULL;
    waiter->fn = fn;
    waiter->user_data = user_data;
    waiter->resource = resource;
    waiter->is_queued = true;
    if (waiter->resource) {
      iree_hal_resource_retain(waiter->resource);
    }
    *ring->waiter_tail = waiter;
    ring->waiter_tail = &waiter->next;
  }
  iree_slim_mutex_unlock(&ring->mutex);
  return should_queue;
}

void iree_hal_vulkan_queue_staging_ring_cancel_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter) {
  iree_hal_resource_t* resource = NULL;
  iree_slim_mutex_lock(&ring->mutex);
  iree_hal_vulkan_queue_staging_waiter_t** link = &ring->waiter_head;
  while (*link) {
    if (*link == waiter) {
      *link = waiter->next;
      if (!waiter->next) {
        ring->waiter_tail = link;
      }
      waiter->next = NULL;
      waiter->is_queued = false;
      resource = waiter->resource;
      waiter->resource = NULL;
      break;
    }
    link = &(*link)->next;
  }
  iree_slim_mutex_unlock(&ring->mutex);
  iree_hal_resource_release(resource);
}

void iree_hal_vulkan_queue_staging_ring_release(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t* slot) {
  iree_slim_mutex_lock(&ring->mutex);
  slot->in_use = false;
  while (ring->waiter_head) {
    bool has_capacity = false;
    for (uint32_t i = 0; i < ring->slot_count; ++i) {
      has_capacity |= !ring->slots[i].in_use;
    }
    if (!has_capacity) {
      break;
    }

    // Detach one complete callback before unlocking. A competing producer may
    // reuse the placed waiter immediately, including its link and resource.
    iree_hal_vulkan_queue_staging_waiter_t* waiter = ring->waiter_head;
    ring->waiter_head = waiter->next;
    if (!ring->waiter_head) {
      ring->waiter_tail = &ring->waiter_head;
    }
    waiter->next = NULL;
    waiter->is_queued = false;
    iree_hal_vulkan_queue_staging_waiter_fn_t fn = waiter->fn;
    void* user_data = waiter->user_data;
    iree_hal_resource_t* resource = waiter->resource;
    waiter->resource = NULL;
    iree_slim_mutex_unlock(&ring->mutex);
    fn(user_data);
    iree_hal_resource_release(resource);
    iree_slim_mutex_lock(&ring->mutex);
  }
  iree_slim_mutex_unlock(&ring->mutex);
}
