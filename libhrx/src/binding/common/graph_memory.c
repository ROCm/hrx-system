// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph_memory.h"

#include "common/internal.h"
#include "common/memory.h"

// Each graph allocation owns a stable virtual address. Physical backing moves
// between these reservations only while unmapped, which keeps every pointer
// passed to device code valid without rewriting kernel arguments.
struct iree_hal_streaming_graph_memory_allocation_t {
  // Reference count held by graph nodes and the returned device address.
  iree_atomic_ref_count_t ref_count;

  // Serializes mapping state and returned-pointer ownership for this address.
  iree_slim_mutex_t mutex;

  // Context that owns the graph allocation.
  iree_hal_streaming_context_t* context;

  // HRX allocator used for virtual reservation ownership.
  hrx_allocator_t allocator;

  // HRX virtual address reservation retained independently of its wrapper.
  hrx_buffer_t virtual_reservation;

  // Wrapper that owns and publishes the virtual address reservation.
  iree_hal_streaming_buffer_t* virtual_buffer;

  // Logical allocation size requested by the graph node.
  iree_device_size_t logical_size;

  // Page-aligned reservation and physical allocation size.
  iree_device_size_t reservation_size;

  // Physical backing currently mapped at |virtual_buffer|, if any.
  iree_hal_physical_memory_t* physical_memory;

  // True while |physical_memory| is mapped and has read-write access.
  bool is_mapped;

  // True after an allocation node has successfully mapped physical backing.
  bool has_executed;

  // True while the returned device address owns a reference to this record.
  bool has_pointer_reference;

  // True while one graph owns a free node for this allocation.
  bool has_free_node_reference;
};

struct iree_hal_streaming_graph_memory_physical_block_t {
  // Next cached block on the same device.
  iree_hal_streaming_graph_memory_physical_block_t* next;

  // Context/allocator that created and can remap this physical handle.
  iree_hal_streaming_context_t* context;

  // Unmapped physical allocation available for reuse.
  iree_hal_physical_memory_t* physical_memory;

  // Allocation size in bytes.
  iree_device_size_t size;
};

static iree_hal_buffer_params_t iree_hal_streaming_graph_memory_buffer_params(
    void) {
  return (iree_hal_buffer_params_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
  };
}

static void iree_hal_streaming_graph_memory_add_current(uint64_t* value,
                                                        uint64_t amount) {
  *value = amount > UINT64_MAX - *value ? UINT64_MAX : *value + amount;
}

static void iree_hal_streaming_graph_memory_subtract_current(uint64_t* value,
                                                             uint64_t amount) {
  *value = iree_min(*value, amount) == amount ? *value - amount : 0;
}

static void iree_hal_streaming_graph_memory_record_physical_allocate_locked(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  iree_hal_streaming_graph_memory_add_current(
      &device->graph_memory_used_current, size);
  iree_hal_streaming_graph_memory_add_current(
      &device->graph_memory_reserved_current, size);
  device->graph_memory_reserved_high =
      iree_max(device->graph_memory_reserved_high,
               device->graph_memory_reserved_current);
}

static void iree_hal_streaming_graph_memory_record_physical_free_locked(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  iree_hal_streaming_graph_memory_subtract_current(
      &device->graph_memory_used_current, size);
  iree_hal_streaming_graph_memory_subtract_current(
      &device->graph_memory_reserved_current, size);
}

static void iree_hal_streaming_graph_memory_record_map_locked(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  iree_hal_streaming_graph_memory_add_current(
      &device->graph_memory_mapped_current, size);
  device->graph_memory_used_high = iree_max(
      device->graph_memory_used_high, device->graph_memory_mapped_current);
}

static void iree_hal_streaming_graph_memory_record_unmap_locked(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  iree_hal_streaming_graph_memory_subtract_current(
      &device->graph_memory_mapped_current, size);
}

static iree_hal_streaming_graph_memory_physical_block_t*
iree_hal_streaming_graph_memory_take_cached_block_locked(
    iree_hal_streaming_device_t* device, iree_hal_streaming_context_t* context,
    iree_device_size_t size) {
  iree_hal_streaming_graph_memory_physical_block_t** previous_next =
      &device->graph_memory_cached_physical_blocks;
  while (*previous_next) {
    iree_hal_streaming_graph_memory_physical_block_t* block = *previous_next;
    if (block->context == context && block->size == size) {
      *previous_next = block->next;
      block->next = NULL;
      return block;
    }
    previous_next = &block->next;
  }
  return NULL;
}

static iree_status_t iree_hal_streaming_graph_memory_cache_block(
    iree_hal_streaming_device_t* device, iree_hal_streaming_context_t* context,
    iree_hal_physical_memory_t* physical_memory, iree_device_size_t size) {
  iree_hal_streaming_graph_memory_physical_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(context->host_allocator,
                                             sizeof(*block), (void**)&block));
  block->context = context;
  iree_hal_streaming_context_retain(context);
  block->physical_memory = physical_memory;
  block->size = size;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  block->next = device->graph_memory_cached_physical_blocks;
  device->graph_memory_cached_physical_blocks = block;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_memory_free_physical(
    iree_hal_streaming_device_t* device, iree_hal_streaming_context_t* context,
    iree_hal_physical_memory_t* physical_memory, iree_device_size_t size) {
  iree_status_t status = iree_hal_allocator_physical_memory_free(
      context->device_allocator, physical_memory);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&device->graph_memory_mutex);
    iree_hal_streaming_graph_memory_record_physical_free_locked(device, size);
    iree_slim_mutex_unlock(&device->graph_memory_mutex);
  }
  return status;
}

static iree_status_t iree_hal_streaming_graph_memory_allocation_map(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  iree_hal_streaming_context_t* context = allocation->context;
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&allocation->mutex);
  if (allocation->is_mapped) {
    iree_slim_mutex_unlock(&allocation->mutex);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "graph allocation virtual address is already mapped");
  }

  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_physical_block_t* cached_block =
      iree_hal_streaming_graph_memory_take_cached_block_locked(
          device, context, allocation->reservation_size);
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  iree_hal_physical_memory_t* physical_memory =
      cached_block ? cached_block->physical_memory : NULL;
  const bool was_cached = cached_block != NULL;
  if (cached_block) {
    iree_hal_streaming_context_release(cached_block->context);
    iree_allocator_free(context->host_allocator, cached_block);
  }

  iree_status_t status = iree_ok_status();
  bool is_mapped = false;
  if (!physical_memory) {
    status = iree_hal_allocator_physical_memory_allocate(
        context->device_allocator,
        iree_hal_streaming_graph_memory_buffer_params(),
        allocation->reservation_size, context->host_allocator,
        &physical_memory);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&device->graph_memory_mutex);
      iree_hal_streaming_graph_memory_record_physical_allocate_locked(
          device, allocation->reservation_size);
      iree_slim_mutex_unlock(&device->graph_memory_mutex);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_virtual_memory_map(
        context->device_allocator, allocation->virtual_buffer->buffer,
        /*virtual_offset=*/0, physical_memory, /*physical_offset=*/0,
        allocation->reservation_size);
    is_mapped = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_virtual_memory_protect(
        context->device_allocator, allocation->virtual_buffer->buffer,
        /*virtual_offset=*/0, allocation->reservation_size,
        IREE_HAL_QUEUE_AFFINITY_ANY,
        IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
        IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
  }
  if (iree_status_is_ok(status)) {
    allocation->physical_memory = physical_memory;
    allocation->is_mapped = true;
    allocation->has_executed = true;
    if (!allocation->has_pointer_reference) {
      iree_atomic_ref_count_inc(&allocation->ref_count);
      allocation->has_pointer_reference = true;
    }
    iree_slim_mutex_lock(&device->graph_memory_mutex);
    iree_hal_streaming_graph_memory_record_map_locked(
        device, allocation->reservation_size);
    iree_slim_mutex_unlock(&device->graph_memory_mutex);
  } else if (physical_memory) {
    if (is_mapped) {
      iree_status_t unmap_status = iree_hal_allocator_virtual_memory_unmap(
          context->device_allocator, allocation->virtual_buffer->buffer,
          /*virtual_offset=*/0, allocation->reservation_size);
      if (!iree_status_is_ok(unmap_status)) {
        // The failed protection leaves a live mapping. Preserve both handles
        // so later graph teardown can retry instead of releasing mapped VMM.
        allocation->physical_memory = physical_memory;
        allocation->is_mapped = true;
        iree_slim_mutex_lock(&device->graph_memory_mutex);
        iree_hal_streaming_graph_memory_record_map_locked(
            device, allocation->reservation_size);
        iree_slim_mutex_unlock(&device->graph_memory_mutex);
        iree_slim_mutex_unlock(&allocation->mutex);
        return iree_status_join(status, unmap_status);
      }
    }
    if (was_cached) {
      iree_status_t cache_status = iree_hal_streaming_graph_memory_cache_block(
          device, context, physical_memory, allocation->reservation_size);
      if (!iree_status_is_ok(cache_status)) {
        iree_status_t free_status =
            iree_hal_streaming_graph_memory_free_physical(
                device, context, physical_memory, allocation->reservation_size);
        status = iree_status_join(status, cache_status);
        if (!iree_status_is_ok(free_status)) {
          status = iree_status_join(status, free_status);
        }
      }
    } else {
      iree_status_t free_status = iree_hal_streaming_graph_memory_free_physical(
          device, context, physical_memory, allocation->reservation_size);
      if (!iree_status_is_ok(free_status)) {
        status = iree_status_join(status, free_status);
      }
    }
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return status;
}

static iree_status_t iree_hal_streaming_graph_memory_allocation_unmap_impl(
    iree_hal_streaming_graph_memory_allocation_t* allocation,
    bool allow_unmapped) {
  IREE_ASSERT_ARGUMENT(allocation);
  iree_hal_streaming_context_t* context = allocation->context;
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&allocation->mutex);
  if (!allocation->is_mapped || !allocation->physical_memory) {
    iree_slim_mutex_unlock(&allocation->mutex);
    if (allow_unmapped) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "graph allocation virtual address is not mapped");
  }

  iree_hal_physical_memory_t* physical_memory = allocation->physical_memory;
  iree_status_t status = iree_hal_allocator_virtual_memory_unmap(
      context->device_allocator, allocation->virtual_buffer->buffer,
      /*virtual_offset=*/0, allocation->reservation_size);
  if (iree_status_is_ok(status)) {
    allocation->physical_memory = NULL;
    allocation->is_mapped = false;
    iree_slim_mutex_lock(&device->graph_memory_mutex);
    iree_hal_streaming_graph_memory_record_unmap_locked(
        device, allocation->reservation_size);
    iree_slim_mutex_unlock(&device->graph_memory_mutex);
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_cache_block(
        device, context, physical_memory, allocation->reservation_size);
    if (!iree_status_is_ok(status)) {
      iree_status_t free_status = iree_hal_streaming_graph_memory_free_physical(
          device, context, physical_memory, allocation->reservation_size);
      if (!iree_status_is_ok(free_status)) {
        status = iree_status_join(status, free_status);
      }
    }
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_memory_allocation_unmap(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  return iree_hal_streaming_graph_memory_allocation_unmap_impl(
      allocation, /*allow_unmapped=*/false);
}

iree_status_t iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  return iree_hal_streaming_graph_memory_allocation_unmap_impl(
      allocation, /*allow_unmapped=*/true);
}

iree_status_t iree_hal_streaming_graph_memory_allocation_create(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_graph_memory_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph allocation size must be nonzero");
  }
  if (!iree_hal_allocator_supports_virtual_memory(context->device_allocator)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device allocator does not support virtual memory");
  }

  const iree_hal_buffer_params_t params =
      iree_hal_streaming_graph_memory_buffer_params();
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_query_granularity(
      context->device_allocator, params, &minimum_page_size,
      &recommended_page_size));
  const iree_device_size_t page_size =
      recommended_page_size ? recommended_page_size : minimum_page_size;
  if (IREE_UNLIKELY(!iree_device_size_is_valid_alignment(page_size))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocator returned an invalid VMM page size");
  }
  iree_device_size_t reservation_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(size, page_size,
                                                    &reservation_size))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "graph allocation size overflows VMM page alignment");
  }

  hrx_allocator_t allocator =
      hrx_device_allocator(context->device_entry->hrx_device);
  hrx_buffer_t virtual_buffer = NULL;
  IREE_RETURN_IF_ERROR(HRX_CALL(hrx_allocator_virtual_memory_reserve(
      allocator, /*affinity=*/0, reservation_size, &virtual_buffer)));

  iree_hal_streaming_buffer_t* virtual_wrapper = NULL;
  iree_status_t status = iree_hal_streaming_memory_wrap_virtual_reservation(
      context, virtual_buffer, &virtual_wrapper);
  if (!iree_status_is_ok(status)) {
    return iree_status_join(
        status, HRX_CALL(hrx_allocator_virtual_memory_release(allocator,
                                                              virtual_buffer)));
  }

  iree_hal_streaming_graph_memory_allocation_t* allocation = NULL;
  status = iree_allocator_malloc(context->host_allocator, sizeof(*allocation),
                                 (void**)&allocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_memory_release_wrapped_buffer(virtual_wrapper);
    return iree_status_join(
        status, HRX_CALL(hrx_allocator_virtual_memory_release(allocator,
                                                              virtual_buffer)));
  }
  memset(allocation, 0, sizeof(*allocation));
  iree_atomic_ref_count_init(&allocation->ref_count);
  iree_slim_mutex_initialize(&allocation->mutex);
  allocation->context = context;
  iree_hal_streaming_context_retain(context);
  allocation->allocator = allocator;
  allocation->virtual_reservation = virtual_buffer;
  allocation->virtual_buffer = virtual_wrapper;
  allocation->logical_size = size;
  allocation->reservation_size = reservation_size;
  allocation->has_pointer_reference = true;
  virtual_wrapper->logical_size = size;
  virtual_wrapper->graph_memory_allocation = allocation;

  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  ++device->graph_memory_allocation_count;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  *out_allocation = allocation;
  return iree_ok_status();
}

void iree_hal_streaming_graph_memory_allocation_retain(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (allocation) iree_atomic_ref_count_inc(&allocation->ref_count);
}

void iree_hal_streaming_graph_memory_allocation_release(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (!allocation || iree_atomic_ref_count_dec(&allocation->ref_count) != 1) {
    return;
  }

  iree_status_t status =
      iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(allocation);
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
  iree_hal_streaming_context_t* context = allocation->context;
  const iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  IREE_ASSERT(device->graph_memory_allocation_count > 0);
  const bool trim_cache = --device->graph_memory_allocation_count == 0;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);

  allocation->virtual_buffer->graph_memory_allocation = NULL;
  iree_hal_streaming_memory_prepare_virtual_reservation_release(
      allocation->virtual_buffer);
  status = HRX_CALL(hrx_allocator_virtual_memory_release(
      allocation->allocator, allocation->virtual_reservation));
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
  allocation->virtual_reservation = NULL;
  iree_hal_streaming_memory_release_wrapped_buffer(allocation->virtual_buffer);
  iree_slim_mutex_deinitialize(&allocation->mutex);
  iree_hal_streaming_context_release(context);
  iree_allocator_free(host_allocator, allocation);

  if (trim_cache) {
    iree_status_t status = iree_hal_streaming_graph_memory_trim(device);
    if (!iree_status_is_ok(status)) {
      iree_status_abort(status);
    }
  }
}

bool iree_hal_streaming_graph_memory_allocation_claim_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (!allocation) return false;
  iree_slim_mutex_lock(&allocation->mutex);
  const bool was_claimed =
      allocation->has_executed && allocation->has_pointer_reference;
  if (was_claimed) allocation->has_pointer_reference = false;
  iree_slim_mutex_unlock(&allocation->mutex);
  return was_claimed;
}

void iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(allocation);
  iree_slim_mutex_lock(&allocation->mutex);
  IREE_ASSERT(!allocation->has_pointer_reference);
  allocation->has_pointer_reference = true;
  iree_slim_mutex_unlock(&allocation->mutex);
}

bool iree_hal_streaming_graph_memory_allocation_claim_unexecuted_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (!allocation) return false;
  iree_slim_mutex_lock(&allocation->mutex);
  const bool was_claimed =
      !allocation->has_executed && allocation->has_pointer_reference;
  if (was_claimed) allocation->has_pointer_reference = false;
  iree_slim_mutex_unlock(&allocation->mutex);
  return was_claimed;
}

bool iree_hal_streaming_graph_memory_allocation_try_claim_free_node_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (!allocation) return false;
  iree_slim_mutex_lock(&allocation->mutex);
  const bool was_claimed = !allocation->has_free_node_reference;
  if (was_claimed) allocation->has_free_node_reference = true;
  iree_slim_mutex_unlock(&allocation->mutex);
  return was_claimed;
}

void iree_hal_streaming_graph_memory_allocation_release_free_node_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(allocation);
  iree_slim_mutex_lock(&allocation->mutex);
  IREE_ASSERT(allocation->has_free_node_reference);
  allocation->has_free_node_reference = false;
  iree_slim_mutex_unlock(&allocation->mutex);
}

void* iree_hal_streaming_graph_memory_allocation_device_pointer(
    const iree_hal_streaming_graph_memory_allocation_t* allocation) {
  return allocation ? (void*)allocation->virtual_buffer->device_ptr : NULL;
}

bool iree_hal_streaming_graph_memory_allocation_is_mapped(
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  if (!allocation) return false;
  iree_slim_mutex_lock(&allocation->mutex);
  const bool is_mapped = allocation->is_mapped;
  iree_slim_mutex_unlock(&allocation->mutex);
  return is_mapped;
}

iree_status_t iree_hal_streaming_graph_memory_allocation_lookup(
    iree_hal_streaming_context_t* context, uint64_t ptr,
    iree_hal_streaming_graph_memory_allocation_t** out_allocation) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_allocation);
  *out_allocation = NULL;
  iree_hal_streaming_buffer_ref_t ref;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_lookup(context, ptr, &ref));
  if (ref.offset != 0 || !ref.buffer->graph_memory_allocation) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "pointer is not a graph allocation base address");
  }
  iree_hal_streaming_graph_memory_allocation_t* allocation =
      ref.buffer->graph_memory_allocation;
  iree_hal_streaming_graph_memory_allocation_retain(allocation);
  *out_allocation = allocation;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_memory_allocation_map_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  return iree_hal_streaming_graph_memory_allocation_map(
      (iree_hal_streaming_graph_memory_allocation_t*)user_data);
}

iree_status_t iree_hal_streaming_graph_memory_allocation_unmap_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  iree_hal_streaming_graph_memory_allocation_t* allocation =
      (iree_hal_streaming_graph_memory_allocation_t*)user_data;
  if (!iree_hal_streaming_graph_memory_allocation_claim_pointer_reference(
          allocation)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "graph allocation has no live returned-pointer reference");
  }
  iree_status_t status =
      iree_hal_streaming_graph_memory_allocation_unmap(allocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
        allocation);
    return status;
  }
  iree_hal_streaming_graph_memory_allocation_release(allocation);
  return iree_ok_status();
}

uint64_t iree_hal_streaming_graph_memory_used_current(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_used_current;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_used_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_used_high;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_reserved_current(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_reserved_current;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

uint64_t iree_hal_streaming_graph_memory_reserved_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  uint64_t value = device->graph_memory_reserved_high;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return value;
}

void iree_hal_streaming_graph_memory_reset_used_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  device->graph_memory_used_high = 0;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

void iree_hal_streaming_graph_memory_reset_reserved_high(
    iree_hal_streaming_device_t* device) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  device->graph_memory_reserved_high = 0;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

iree_status_t iree_hal_streaming_graph_memory_trim(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  // Allocation teardown may trim automatically while the application issues
  // an explicit trim. Serialize the complete operation so an explicit trim
  // cannot observe an already-detached cache and return before its physical
  // blocks and accounting have been released.
  iree_slim_mutex_lock(&device->graph_memory_trim_mutex);
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_physical_block_t* blocks =
      device->graph_memory_cached_physical_blocks;
  device->graph_memory_cached_physical_blocks = NULL;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);

  iree_status_t status = iree_ok_status();
  while (blocks) {
    iree_hal_streaming_graph_memory_physical_block_t* block = blocks;
    blocks = block->next;
    iree_status_t free_status = iree_hal_allocator_physical_memory_free(
        block->context->device_allocator, block->physical_memory);
    if (iree_status_is_ok(free_status)) {
      const iree_allocator_t host_allocator = block->context->host_allocator;
      iree_slim_mutex_lock(&device->graph_memory_mutex);
      iree_hal_streaming_graph_memory_record_physical_free_locked(device,
                                                                  block->size);
      iree_slim_mutex_unlock(&device->graph_memory_mutex);
      iree_hal_streaming_context_release(block->context);
      iree_allocator_free(host_allocator, block);
    } else {
      iree_slim_mutex_lock(&device->graph_memory_mutex);
      block->next = device->graph_memory_cached_physical_blocks;
      device->graph_memory_cached_physical_blocks = block;
      iree_slim_mutex_unlock(&device->graph_memory_mutex);
      status = iree_status_join(status, free_status);
    }
  }
  iree_slim_mutex_unlock(&device->graph_memory_trim_mutex);
  return status;
}
