// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph_memory.h"

#include "common/context.h"
#include "common/device.h"
#include "common/hrx_bridge.h"
#include "common/memory.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "common/virtual_buffer.h"
#include "libhrx/src/libhrx/hrx_internal.h"

// Cached storage remains an ordinary queue allocation. Its last unmap point
// qualifies reuse without completion callbacks. No cache lock is held during
// submission or a native operation.
typedef struct iree_hal_streaming_graph_memory_block_t {
  // Next idle block owned by the frontend device.
  struct iree_hal_streaming_graph_memory_block_t* next;
  // Retained physical allocation, independent of every virtual address.
  iree_hal_buffer_t* buffer;
  // Borrowed device-owned source used to qualify this block.
  iree_hal_pool_t* source;
  // Exact mapping extent in bytes.
  iree_device_size_t size;
  // Retained completion of allocation or the last unmap of these bytes.
  iree_hal_semaphore_t* ready;
  // Value on ready required before another mapping can use these bytes.
  uint64_t ready_value;
} iree_hal_streaming_graph_memory_block_t;

struct iree_hal_streaming_graph_allocation_t {
  // Ref-counted owner shared by graph nodes and an active returned pointer.
  iree_hal_resource_t resource;
  // Guards host enqueue state, never a completion wait.
  iree_slim_mutex_t mutex;
  // Borrowed context retained by the registered virtual-buffer wrapper.
  iree_hal_streaming_context_t* context;
  // Borrowed frontend device owning physical placement and the shared cache.
  iree_hal_streaming_device_t* device;
  // Retained native source; construction acquires no physical memory.
  iree_hal_pool_t* source;
  // Owned pointer-table registration and virtual address reservation.
  iree_hal_streaming_buffer_t* wrapper;
  // Retained timeline for accepted allocation and unmap steps.
  iree_hal_semaphore_t* timeline;
  // Last accepted value on timeline.
  uint64_t value;
  // Physical bytes owned by the currently enqueued allocation, or NULL.
  iree_hal_streaming_graph_memory_block_t* block;
  // Logical byte count reported for the returned pointer.
  iree_device_size_t size;
  // Page-aligned backing and reservation extent.
  iree_device_size_t mapping_size;
  // Number of prepared queue/program mapping descriptors.
  iree_host_size_t mapping_count;
  // Owned bind descriptors followed by unbind descriptors. Desired and prior
  // sources are patched under mutex during queue capture.
  iree_hal_memory_mapping_t* mappings;
};

static void iree_hal_streaming_graph_allocation_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_streaming_graph_allocation_t* allocation = (void*)resource;
  const iree_allocator_t allocator = allocation->context->host_allocator;
  // An active allocation owns its own pointer reference. Nodes can disappear
  // while an application still uses the address, but cannot release backing.
  IREE_ASSERT(!allocation->block);
  iree_hal_semaphore_release(allocation->timeline);
  iree_hal_pool_release(allocation->source);
  iree_allocator_free(allocator, allocation->mappings);
  iree_slim_mutex_deinitialize(&allocation->mutex);
  iree_hal_streaming_memory_release_wrapped_buffer(allocation->wrapper);
  iree_allocator_free(allocator, allocation);
}

static const iree_hal_resource_vtable_t
    iree_hal_streaming_graph_allocation_vtable = {
        .destroy = iree_hal_streaming_graph_allocation_destroy,
};

void iree_hal_streaming_graph_allocation_retain(
    iree_hal_streaming_graph_allocation_t* allocation) {
  iree_hal_resource_retain(allocation);
}

void iree_hal_streaming_graph_allocation_release(
    iree_hal_streaming_graph_allocation_t* allocation) {
  iree_hal_resource_release(allocation);
}

iree_status_t iree_hal_streaming_graph_allocation_create(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_host_size_t device_count, const hrx_device_t* devices,
    const iree_hal_memory_access_t* accesses,
    iree_hal_streaming_graph_allocation_t** out_allocation,
    void** out_pointer) {
  *out_allocation = NULL;
  *out_pointer = NULL;
  const iree_hal_streaming_virtual_buffer_scope_t scope = {device_count,
                                                           devices, 0};
  const iree_hal_streaming_virtual_buffer_source_t source = {
      .device = devices[0], .memory_type = HRX_MEMORY_TYPE_DEVICE_LOCAL};
  const iree_hal_streaming_virtual_buffer_sources_t sources = {1, &source};
  hrx_virtual_buffer_capabilities_t capabilities;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_virtual_buffer_query_capabilities(
      scope, sources, &capabilities));
  if (!capabilities.mapping_granularity ||
      !(capabilities.supported_flags &
        HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "graph allocations require shared virtual addresses");
  }
  iree_device_size_t mapping_size = 0;
  if (!iree_device_size_checked_align(size, capabilities.mapping_granularity,
                                      &mapping_size) ||
      mapping_size > capabilities.max_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "graph allocation exceeds reservation limits");
  }
  hrx_buffer_t reservation = NULL;
  const hrx_virtual_buffer_options_t options = {
      .byte_length = mapping_size,
      .mapping_granularity = capabilities.mapping_granularity,
      .flags = HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
  };
  IREE_RETURN_IF_ERROR(iree_hal_streaming_virtual_buffer_create(
      scope, sources, &options, &reservation));

  iree_hal_streaming_graph_allocation_t* allocation = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*allocation), (void**)&allocation);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_streaming_graph_allocation_vtable,
                                 &allocation->resource);
    iree_slim_mutex_initialize(&allocation->mutex);
    allocation->context = context;
    allocation->device = iree_hal_streaming_device_entry(devices[0]->ordinal);
    allocation->size = size;
    allocation->mapping_size = mapping_size;
    status = hrx_mem_pool_backing_retain_allocation_source(
        devices[0], HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_MAPPING_SOURCE, &allocation->source);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &allocation->timeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memory_wrap_virtual_reservation(
        context, reservation, size, &allocation->wrapper);
  }
  iree_host_size_t mapping_count = 0;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    const iree_hal_device_queue_spec_t* queues = iree_hal_device_spec_queues(
        iree_hal_device_spec(devices[i]->hal_device));
    for (iree_host_size_t j = 0; j < queues->family_count; ++j) {
      const iree_hal_queue_family_t* family =
          iree_hal_device_queue_family(devices[i]->hal_device, j);
      mapping_count +=
          1 + !!iree_any_bit_set(iree_hal_queue_family_spec(family)->role_flags,
                                 IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        context->host_allocator, 2 * mapping_count,
        sizeof(*allocation->mappings), (void**)&allocation->mappings);
  }
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_device_queue_spec_t* queues = iree_hal_device_spec_queues(
        iree_hal_device_spec(devices[i]->hal_device));
    for (iree_host_size_t j = 0;
         j < queues->family_count && iree_status_is_ok(status); ++j) {
      const iree_hal_queue_family_t* family =
          iree_hal_device_queue_family(devices[i]->hal_device, j);
      for (uint32_t site = 0; site < 2 && iree_status_is_ok(status); ++site) {
        if (site == 1 &&
            !iree_any_bit_set(iree_hal_queue_family_spec(family)->role_flags,
                              IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
          continue;
        }
        const iree_hal_memory_scope_t memory_scope = {
            family->memory.domain, family->memory.queue_scope_id + site};
        uint32_t projection = 0;
        status = iree_hal_virtual_buffer_resolve_projection(
            reservation->hal_buffer, memory_scope, &projection);
        if (!iree_status_is_ok(status)) {
          break;
        }
        // The unused unbind half is construction scratch until the unique
        // native projections have been selected. Queue and program scopes
        // commonly share one page table and cannot be mapped twice in a batch.
        uint32_t* projections =
            (uint32_t*)(allocation->mappings + mapping_count);
        iree_host_size_t index = 0;
        while (index < allocation->mapping_count &&
               projections[index] != projection) {
          ++index;
        }
        if (index < allocation->mapping_count) {
          if (allocation->mappings[index].access != accesses[i]) {
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "shared device translations require matching access");
          }
          continue;
        }
        projections[index] = projection;
        allocation->mappings[allocation->mapping_count++] =
            (iree_hal_memory_mapping_t){
                .target = reservation->hal_buffer,
                .scope = memory_scope,
                .target_range = {0, mapping_size},
                .access = accesses[i],
                .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
            };
      }
    }
  }
  hrx_buffer_release(reservation);
  if (iree_status_is_ok(status)) {
    mapping_count = allocation->mapping_count;
    for (iree_host_size_t i = 0; i < mapping_count; ++i) {
      allocation->mappings[mapping_count + i] = allocation->mappings[i];
      allocation->mappings[mapping_count + i].access = 0;
      allocation->mappings[mapping_count + i].operation =
          IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
    }
    allocation->wrapper->graph_allocation = allocation;
    *out_pointer = (void*)(uintptr_t)allocation->wrapper->device_ptr;
    *out_allocation = allocation;
  } else {
    iree_hal_streaming_graph_allocation_release(allocation);
  }
  return status;
}

static void iree_hal_streaming_graph_memory_cache(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_graph_memory_block_t* block) {
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  block->next = device->graph_memory_blocks;
  device->graph_memory_blocks = block;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
}

static iree_status_t iree_hal_streaming_graph_memory_take(
    iree_hal_streaming_graph_allocation_t* allocation,
    iree_hal_streaming_graph_memory_block_t** out_block) {
  *out_block = NULL;
  iree_hal_streaming_device_t* device = allocation->device;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_block_t** link = &device->graph_memory_blocks;
  while (*link && iree_status_is_ok(status)) {
    iree_hal_streaming_graph_memory_block_t* block = *link;
    if (block->size == allocation->mapping_size &&
        block->source == allocation->source) {
      // Replays already depend on this address's preceding unmap. Reuse from
      // another address requires observed completion: adding a dependency on
      // an unrelated graph could block the work that lets that graph finish.
      bool reusable = block->ready == allocation->timeline;
      if (!reusable) {
        uint64_t completed = 0;
        status = iree_hal_semaphore_query(block->ready, &completed);
        reusable = iree_status_is_ok(status) && completed >= block->ready_value;
      }
      if (reusable) {
        *link = block->next;
        block->next = NULL;
        *out_block = block;
        break;
      }
    }
    link = &block->next;
  }
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  return status;
}

// Queue operations copy the descriptors and semaphore lists before returning.
// The graph compiler's at-most-eight workstreams and the stream's completion
// points leave room for the one allocation-local point in this fixed storage.
static iree_status_t iree_hal_streaming_graph_allocation_map(
    iree_hal_streaming_graph_allocation_t* allocation, iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_streaming_graph_memory_block_t* block, bool bind) {
  iree_hal_semaphore_t* wait_semaphores[16];
  uint64_t wait_values[16];
  iree_hal_semaphore_t* signal_semaphores[16];
  uint64_t signal_values[16];
  IREE_ASSERT(waits.count < IREE_ARRAYSIZE(wait_semaphores));
  IREE_ASSERT(signals.count < IREE_ARRAYSIZE(signal_semaphores));
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    wait_semaphores[i] = waits.semaphores[i];
    wait_values[i] = waits.payload_values[i];
  }
  iree_host_size_t wait_count = waits.count;
  if (allocation->value) {
    wait_semaphores[wait_count] = allocation->timeline;
    wait_values[wait_count++] = allocation->value;
  }
  // A fresh block or same-address reuse is covered by the allocation timeline.
  // Cross-address cache reuse requires completed unmapping before selection.
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    signal_semaphores[i] = signals.semaphores[i];
    signal_values[i] = signals.payload_values[i];
  }
  signal_semaphores[signals.count] = allocation->timeline;
  signal_values[signals.count] = allocation->value + 1;
  iree_hal_memory_mapping_t* mappings =
      allocation->mappings + (bind ? 0 : allocation->mapping_count);
  for (iree_host_size_t i = 0; i < allocation->mapping_count; ++i) {
    mappings[i].source = bind ? block->buffer : NULL;
    mappings[i].previous_source = bind ? NULL : block->buffer;
  }
  iree_status_t status = iree_hal_queue_map(
      queue,
      (iree_hal_semaphore_list_t){wait_count, wait_semaphores, wait_values},
      (iree_hal_semaphore_list_t){signals.count + 1, signal_semaphores,
                                  signal_values},
      allocation->mapping_count, mappings, IREE_HAL_QUEUE_MAP_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    ++allocation->value;
    if (block->ready != allocation->timeline) {
      iree_hal_semaphore_retain(allocation->timeline);
      iree_hal_semaphore_release(block->ready);
      block->ready = allocation->timeline;
    }
    block->ready_value = allocation->value;
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_allocation_lookup(
    iree_hal_streaming_context_t* context, uint64_t pointer,
    iree_hal_streaming_graph_allocation_t** out_allocation) {
  *out_allocation = NULL;
  iree_hal_streaming_buffer_ref_t ref = {0};
  bool found = iree_hal_streaming_memory_try_lookup(context, pointer, &ref);
  if (!found) {
    // Like ordinary HIP frees, graph frees locate the owning context even when
    // the caller has selected another context. Nodes or the live pointer keep
    // the address registration alive throughout a valid call.
    iree_hal_streaming_device_registry_t* registry =
        iree_hal_streaming_device_registry();
    iree_slim_mutex_lock(&registry->context_list.mutex);
    for (iree_hal_streaming_context_t* candidate = registry->context_list.head;
         candidate && !found; candidate = candidate->context_list_entry.next) {
      if (candidate != context) {
        found = iree_hal_streaming_memory_try_lookup(candidate, pointer, &ref);
      }
    }
    iree_slim_mutex_unlock(&registry->context_list.mutex);
  }
  if (!found || !ref.buffer->graph_allocation) {
    return iree_ok_status();
  }
  if (ref.offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph free requires the allocation base");
  }
  iree_hal_streaming_graph_allocation_retain(ref.buffer->graph_allocation);
  *out_allocation = ref.buffer->graph_allocation;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_allocation_free(
    iree_hal_streaming_graph_allocation_t* allocation,
    iree_hal_streaming_stream_t* stream) {
  if (stream) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_flush(stream));
    iree_slim_mutex_lock(&stream->mutex);
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
    iree_hal_streaming_submission_t submission = {0};
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_submission_prepare_locked(
          stream, stream->queue, signal_value,
          IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_t* accepted = NULL;
      uint64_t accepted_value = 0;
      status = iree_hal_streaming_graph_allocation_enqueue(
          allocation, submission.queue,
          (iree_hal_semaphore_list_t){wait_value ? 1 : 0,
                                      &stream->timeline_semaphore, &wait_value},
          submission.signals, IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_NONE,
          &accepted, &accepted_value);
    }
    iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                iree_status_is_ok(status));
    iree_slim_mutex_unlock(&stream->mutex);
    return status;
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_context_synchronize_all());
  iree_hal_semaphore_t* accepted = NULL;
  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_allocation_enqueue(
      allocation, allocation->context->queue, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(),
      IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_NONE, &accepted, &value));
  return iree_hal_semaphore_wait(accepted, value, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static void iree_hal_streaming_graph_memory_block_release(
    iree_allocator_t allocator,
    iree_hal_streaming_graph_memory_block_t* block) {
  iree_hal_buffer_release(block->buffer);
  iree_hal_semaphore_release(block->ready);
  iree_allocator_free(allocator, block);
}

iree_status_t iree_hal_streaming_graph_memory_trim(
    iree_hal_streaming_device_t* device) {
  const iree_allocator_t allocator =
      iree_hal_streaming_device_registry()->host_allocator;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_block_t* blocks = device->graph_memory_blocks;
  device->graph_memory_blocks = NULL;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  iree_status_t status = iree_ok_status();
  while (blocks) {
    iree_hal_streaming_graph_memory_block_t* block = blocks;
    blocks = block->next;
    uint64_t completed = 0;
    iree_status_t block_status =
        iree_hal_semaphore_query(block->ready, &completed);
    const bool ready = completed >= block->ready_value;
    if (iree_status_is_ok(block_status) && ready) {
      block_status = iree_hal_queue_dealloca(
          device->hrx_device->dispatch_queue, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), 1, &block->buffer);
    }
    if (ready && iree_status_is_ok(block_status)) {
      iree_slim_mutex_lock(&device->graph_memory_mutex);
      device->graph_memory_reserved_current -= block->size;
      iree_slim_mutex_unlock(&device->graph_memory_mutex);
      iree_hal_streaming_graph_memory_block_release(allocator, block);
    } else {
      iree_hal_streaming_graph_memory_cache(device, block);
    }
    status = iree_status_join(status, block_status);
  }
  return status;
}

void iree_hal_streaming_graph_memory_deinitialize(
    iree_hal_streaming_device_t* device) {
  const iree_allocator_t allocator =
      iree_hal_streaming_device_registry()->host_allocator;
  while (device->graph_memory_blocks) {
    iree_hal_streaming_graph_memory_block_t* block =
        device->graph_memory_blocks;
    device->graph_memory_blocks = block->next;
    iree_hal_streaming_graph_memory_block_release(allocator, block);
  }
}

static iree_status_t iree_hal_streaming_graph_allocation_unmap_locked(
    iree_hal_streaming_graph_allocation_t* allocation, iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals) {
  if (!allocation->block) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph address has no live allocation");
  }
  iree_hal_streaming_graph_memory_block_t* block = allocation->block;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_allocation_map(
      allocation, queue, waits, signals, block, false));
  allocation->block = NULL;
  iree_hal_streaming_device_t* device = allocation->device;
  iree_slim_mutex_lock(&device->graph_memory_mutex);
  device->graph_memory_used_current -= allocation->size;
  iree_slim_mutex_unlock(&device->graph_memory_mutex);
  iree_hal_streaming_graph_memory_cache(device, block);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_allocation_enqueue(
    iree_hal_streaming_graph_allocation_t* allocation, iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_streaming_graph_allocation_flags_t flags,
    iree_hal_semaphore_t** out_accepted_semaphore,
    uint64_t* out_accepted_value) {
  *out_accepted_semaphore = NULL;
  *out_accepted_value = 0;
  const bool allocate = iree_any_bit_set(
      flags, IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_ALLOCATE);
  const bool auto_free = iree_any_bit_set(
      flags, IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_AUTO_FREE);
  iree_slim_mutex_lock(&allocation->mutex);
  const uint64_t initial_value = allocation->value;
  iree_status_t status = iree_ok_status();
  bool release_pointer = false;
  if (!allocate || (auto_free && allocation->block)) {
    status = iree_hal_streaming_graph_allocation_unmap_locked(
        allocation, queue, waits,
        allocate ? iree_hal_semaphore_list_empty() : signals);
    release_pointer = iree_status_is_ok(status);
  }
  if (allocate && allocation->block && iree_status_is_ok(status)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "graph allocation is still live from a prior launch");
  }
  iree_hal_streaming_graph_memory_block_t* block = NULL;
  if (allocate && iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_take(allocation, &block);
    if (iree_status_is_ok(status) && !block) {
      status = iree_allocator_malloc(allocation->context->host_allocator,
                                     sizeof(*block), (void**)&block);
      if (iree_status_is_ok(status)) {
        block->source = allocation->source;
        block->size = allocation->mapping_size;
        const iree_hal_pool_reservation_request_t request = {.allocation_size =
                                                                 block->size};
        uint64_t ready_value = allocation->value + 1;
        iree_hal_semaphore_t* wait_semaphores[16];
        uint64_t wait_values[16];
        IREE_ASSERT(waits.count < IREE_ARRAYSIZE(wait_semaphores));
        for (iree_host_size_t i = 0; i < waits.count; ++i) {
          wait_semaphores[i] = waits.semaphores[i];
          wait_values[i] = waits.payload_values[i];
        }
        iree_host_size_t wait_count = waits.count;
        if (allocation->value) {
          wait_semaphores[wait_count] = allocation->timeline;
          wait_values[wait_count++] = allocation->value;
        }
        status = iree_hal_queue_alloca(
            queue,
            (iree_hal_semaphore_list_t){wait_count, wait_semaphores,
                                        wait_values},
            (iree_hal_semaphore_list_t){1, &allocation->timeline, &ready_value},
            allocation->source, 1, &request, &block->buffer);
        if (iree_status_is_ok(status)) {
          allocation->value = ready_value;
          block->ready = allocation->timeline;
          block->ready_value = ready_value;
          iree_hal_semaphore_retain(block->ready);
          iree_slim_mutex_lock(&allocation->device->graph_memory_mutex);
          allocation->device->graph_memory_reserved_current += block->size;
          allocation->device->graph_memory_reserved_high =
              iree_max(allocation->device->graph_memory_reserved_high,
                       allocation->device->graph_memory_reserved_current);
          iree_slim_mutex_unlock(&allocation->device->graph_memory_mutex);
        } else {
          iree_allocator_free(allocation->context->host_allocator, block);
          block = NULL;
        }
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_graph_allocation_map(allocation, queue, waits,
                                                       signals, block, true);
    }
    if (iree_status_is_ok(status)) {
      allocation->block = block;
      iree_hal_streaming_graph_allocation_retain(allocation);
      iree_slim_mutex_lock(&allocation->device->graph_memory_mutex);
      allocation->device->graph_memory_used_current += allocation->size;
      allocation->device->graph_memory_used_high =
          iree_max(allocation->device->graph_memory_used_high,
                   allocation->device->graph_memory_used_current);
      iree_slim_mutex_unlock(&allocation->device->graph_memory_mutex);
    } else if (block) {
      iree_hal_streaming_graph_memory_cache(allocation->device, block);
    }
  }
  if (allocation->value != initial_value) {
    *out_accepted_semaphore = allocation->timeline;
    *out_accepted_value = allocation->value;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  if (release_pointer) {
    iree_hal_streaming_graph_allocation_release(allocation);
  }
  return status;
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
