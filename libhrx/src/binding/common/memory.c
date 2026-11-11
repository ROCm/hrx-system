// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/memory.h"

#include <stdio.h>

#include "common/context.h"
#include "common/device.h"
#include "common/direct_transfer.h"
#include "common/graph.h"
#include "common/graph_memory.h"
#include "common/hrx_bridge.h"
#include "common/mem_pool.h"
#include "common/peer.h"
#include "common/stream.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/notification.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"
#include "libhrx/src/libhrx/mem_pool_backing.h"
#include "libhrx/src/libhrx/transfer.h"

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_buffer_query_device_pointer(
    hrx_buffer_t buffer, hrx_device_t device, void** out_pointer) {
  *out_pointer = NULL;
  if (buffer->device_ptr) {
    *out_pointer = buffer->device_ptr;
    return iree_ok_status();
  }
  if (!buffer->hal_buffer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer has no device address");
  }
  const iree_hal_buffer_memory_view_t memory =
      iree_hal_buffer_memory_view(buffer->hal_buffer);
  if (memory.contract) {
    const iree_hal_queue_family_t* family =
        iree_hal_queue_family(device->dispatch_queue);
    const hrx_memory_scope_t scope = {
        .domain = family->memory.domain,
        .id = family->memory.queue_scope_id + 1,
    };
    uint64_t address = 0;
    IREE_RETURN_IF_ERROR(HRX_CALL(
        hrx_buffer_query_address(buffer, scope,
                                 device->type == HRX_ACCELERATOR_CPU
                                     ? HRX_BUFFER_INTERFACE_HOST
                                     : HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                 &address)));
    *out_pointer = (void*)(uintptr_t)address;
    return iree_ok_status();
  }

  // A scoped mapping grants access only to its interval and cannot establish
  // a whole-buffer address. Export leaves any active mapping untouched.
  iree_hal_external_storage_t storage = {0};
  iree_hal_buffer_range_t logical_range = {0};
  iree_status_t status = iree_hal_buffer_export(
      buffer->hal_buffer, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS),
      &storage, &logical_range);
  if (iree_status_is_ok(status)) {
    *out_pointer = (void*)(uintptr_t)(storage.payload.device_address +
                                      logical_range.byte_offset);
  }
  iree_hal_external_storage_release(&storage);
  return status;
}

static iree_status_t iree_hal_streaming_command_buffer_barrier(
    iree_hal_command_buffer_t* command_buffer, iree_hal_barrier_flags_t flags) {
  static const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = flags,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  return iree_hal_command_buffer_barrier(command_buffer, &execution_barrier);
}

bool iree_hal_streaming_memory_try_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  size_t offset = 0;
  const bool found =
      hrx_buffer_table_try_find(&context->buffer_table, device_ptr, NULL,
                                &offset, (void**)&out_ref->buffer);
  out_ref->offset = (iree_device_size_t)offset;
  return found;
}

static iree_status_t iree_hal_streaming_buffer_ref_validate_range(
    const iree_hal_streaming_buffer_ref_t* ref, iree_device_size_t size) {
  if (!ref->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer reference is empty");
  }
  if (ref->offset > ref->buffer->size ||
      size > ref->buffer->size - ref->offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer reference range exceeds allocation");
  }
  return iree_ok_status();
}

typedef struct iree_hal_streaming_host_memcpy_callback_data_t {
  void* dst;
  const void* src;
  iree_device_size_t count;
} iree_hal_streaming_host_memcpy_callback_data_t;

static void iree_hal_streaming_host_memcpy_callback(void* user_data) {
  iree_hal_streaming_host_memcpy_callback_data_t* callback_data =
      (iree_hal_streaming_host_memcpy_callback_data_t*)user_data;
  memcpy(callback_data->dst, callback_data->src, callback_data->count);
}

typedef struct iree_hal_streaming_capture_fill_t {
  iree_hal_streaming_deviceptr_t dst;
  iree_device_size_t length;
  const void* pattern;
  iree_host_size_t pattern_length;
} iree_hal_streaming_capture_fill_t;

static iree_status_t iree_hal_streaming_capture_record_fill(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_fill_t* capture =
      (iree_hal_streaming_capture_fill_t*)user_data;
  if (capture->pattern_length != 1 && capture->pattern_length != 2 &&
      capture->pattern_length != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pattern length %zu",
                            capture->pattern_length);
  }
  uint32_t pattern_value = 0;
  memcpy(&pattern_value, capture->pattern, capture->pattern_length);
  return iree_hal_streaming_graph_add_fill_ptr_node(
      graph, dependencies, dependency_count, capture->dst, pattern_value,
      capture->pattern_length, capture->length / capture->pattern_length,
      out_terminal_node);
}

typedef struct iree_hal_streaming_capture_copy_t {
  iree_hal_streaming_deviceptr_t dst;
  iree_hal_streaming_deviceptr_t src;
  iree_device_size_t size;
} iree_hal_streaming_capture_copy_t;

static iree_status_t iree_hal_streaming_capture_record_copy(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_copy_t* capture =
      (iree_hal_streaming_capture_copy_t*)user_data;
  return iree_hal_streaming_graph_add_copy_ptr_node(
      graph, dependencies, dependency_count, capture->dst, capture->src,
      capture->size, out_terminal_node);
}

typedef struct iree_hal_streaming_capture_host_to_device_t {
  iree_hal_streaming_deviceptr_t dst;
  const void* src;
  iree_device_size_t size;
} iree_hal_streaming_capture_host_to_device_t;

static iree_status_t iree_hal_streaming_capture_build_host_to_device(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_host_to_device_t* capture =
      (iree_hal_streaming_capture_host_to_device_t*)user_data;
  iree_hal_streaming_buffer_ref_t source_ref;
  if (iree_hal_streaming_memory_try_lookup(
          graph->context, (iree_hal_streaming_deviceptr_t)capture->src,
          &source_ref)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_ref_validate_range(
        &source_ref, capture->size));
    if (source_ref.buffer->memory_kind ==
        IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "host-to-device source is not host memory");
    }
    return iree_hal_streaming_graph_add_copy_ptr_node(
        graph, dependencies, dependency_count, capture->dst,
        (iree_hal_streaming_deviceptr_t)capture->src, capture->size,
        out_terminal_node);
  }

  iree_hal_streaming_host_memcpy_callback_data_t* callback_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &graph->arena, sizeof(*callback_data), (void**)&callback_data));
  callback_data->dst = NULL;
  callback_data->src = capture->src;
  callback_data->count = capture->size;

  iree_hal_streaming_graph_node_t* callback_node = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_host_call_node(
      graph, dependencies, dependency_count,
      iree_hal_streaming_host_memcpy_callback, callback_data, &callback_node));
  callback_node->attrs.host.user_data_size = sizeof(*callback_data);

  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_staged_copy_node(
      graph, dependencies, dependency_count, callback_node,
      IREE_HAL_STREAMING_GRAPH_STAGING_UPLOAD, capture->dst, capture->size,
      capture->size,
      offsetof(iree_hal_streaming_host_memcpy_callback_data_t, dst),
      out_terminal_node));
  iree_hal_streaming_graph_attach_copy_callback(*out_terminal_node,
                                                callback_node, false);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_capture_record_host_to_device(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  const iree_hal_streaming_graph_copy_checkpoint_t checkpoint =
      iree_hal_streaming_graph_copy_checkpoint(graph);
  iree_status_t status = iree_hal_streaming_capture_build_host_to_device(
      graph, dependencies, dependency_count, user_data, out_terminal_node);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_rollback_copy(graph, checkpoint);
  }
  return status;
}

typedef struct iree_hal_streaming_capture_device_to_host_t {
  void* dst;
  iree_hal_streaming_deviceptr_t src;
  iree_device_size_t size;
} iree_hal_streaming_capture_device_to_host_t;

static iree_status_t iree_hal_streaming_capture_build_device_to_host(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  iree_hal_streaming_capture_device_to_host_t* capture =
      (iree_hal_streaming_capture_device_to_host_t*)user_data;
  iree_hal_streaming_buffer_ref_t destination_ref;
  if (iree_hal_streaming_memory_try_lookup(
          graph->context, (iree_hal_streaming_deviceptr_t)capture->dst,
          &destination_ref)) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_ref_validate_range(
        &destination_ref, capture->size));
    if (destination_ref.buffer->memory_kind ==
        IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device-to-host destination is not host memory");
    }
    return iree_hal_streaming_graph_add_copy_ptr_node(
        graph, dependencies, dependency_count,
        (iree_hal_streaming_deviceptr_t)capture->dst, capture->src,
        capture->size, out_terminal_node);
  }

  iree_hal_streaming_graph_node_t* copy_node = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_staged_copy_node(
      graph, dependencies, dependency_count, NULL,
      IREE_HAL_STREAMING_GRAPH_STAGING_DOWNLOAD, capture->src, capture->size,
      capture->size,
      offsetof(iree_hal_streaming_host_memcpy_callback_data_t, src),
      &copy_node));

  iree_hal_streaming_host_memcpy_callback_data_t* callback_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &graph->arena, sizeof(*callback_data), (void**)&callback_data));
  callback_data->dst = capture->dst;
  callback_data->src = NULL;
  callback_data->count = capture->size;

  iree_hal_streaming_graph_node_t* copy_dependencies[] = {copy_node};
  iree_hal_streaming_graph_node_t* callback_node = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_add_host_call_node(
      graph, copy_dependencies, IREE_ARRAYSIZE(copy_dependencies),
      iree_hal_streaming_host_memcpy_callback, callback_data, &callback_node));
  iree_hal_streaming_graph_attach_copy_callback(copy_node, callback_node, true);
  callback_node->attrs.host.user_data_size = sizeof(*callback_data);
  *out_terminal_node = copy_node;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_capture_record_device_to_host(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node) {
  const iree_hal_streaming_graph_copy_checkpoint_t checkpoint =
      iree_hal_streaming_graph_copy_checkpoint(graph);
  iree_status_t status = iree_hal_streaming_capture_build_device_to_host(
      graph, dependencies, dependency_count, user_data, out_terminal_node);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_rollback_copy(graph, checkpoint);
  }
  return status;
}

static void iree_hal_streaming_buffer_free(iree_hal_streaming_buffer_t* buffer);

static void iree_hal_streaming_buffer_activate_pool_allocation(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->allocation_pool || buffer->is_pool_allocation_live) {
    return;
  }
  iree_hal_streaming_mem_pool_record_logical_allocation(buffer->allocation_pool,
                                                        (size_t)buffer->size);
  buffer->is_pool_allocation_live = true;
}

static void iree_hal_streaming_buffer_release_pool_allocation(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->allocation_pool || !buffer->is_pool_allocation_live) {
    return;
  }
  iree_hal_streaming_mem_pool_record_logical_free(buffer->allocation_pool,
                                                  (size_t)buffer->size);
  buffer->is_pool_allocation_live = false;
}

static void iree_hal_streaming_memory_account_device_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->context || !buffer->context->device_entry ||
      buffer->memory_kind != IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
    return;
  }
  iree_hal_streaming_device_t* device = buffer->context->device_entry;
  const uint64_t size = buffer->size;
  const uint64_t total_memory = device->total_memory;
  uint64_t observed =
      iree_atomic_load(&device->free_memory, iree_memory_order_relaxed);
  uint64_t desired = 0;
  do {
    desired = size >= total_memory || observed >= total_memory - size
                  ? total_memory
                  : observed + size;
  } while (!iree_atomic_compare_exchange_weak(
      &device->free_memory, &observed, desired, iree_memory_order_relaxed,
      iree_memory_order_relaxed));
}

static void iree_hal_streaming_memory_account_device_allocation(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  if (!device) {
    return;
  }
  uint64_t observed =
      iree_atomic_load(&device->free_memory, iree_memory_order_relaxed);
  uint64_t desired = 0;
  do {
    desired = observed > size ? observed - size : 0;
  } while (!iree_atomic_compare_exchange_weak(
      &device->free_memory, &observed, desired, iree_memory_order_relaxed,
      iree_memory_order_relaxed));
}

static void iree_hal_streaming_buffer_set_context(
    iree_hal_streaming_buffer_t* buffer, iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_context_ownership_t ownership) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(context);
  buffer->context = context;
  buffer->context_ownership = ownership;
  if (ownership == IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED) {
    iree_hal_streaming_context_retain(context);
  }
}

static void iree_hal_streaming_buffer_release_context(
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  iree_hal_streaming_context_t* context = buffer->context;
  iree_hal_streaming_buffer_context_ownership_t ownership =
      buffer->context_ownership;
  buffer->context = NULL;
  buffer->context_ownership = IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED;
  if (ownership == IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED) {
    IREE_ASSERT_ARGUMENT(context);
    iree_hal_streaming_context_release(context);
  }
}

// Wraps an HRX buffer in a stream buffer and caches exported pointer metadata.
static iree_status_t iree_hal_streaming_buffer_wrap_hrx_buffer(
    iree_hal_streaming_context_t* context, hrx_buffer_t hrx_buf,
    iree_device_size_t logical_size, int memory_type,
    iree_hal_streaming_memory_kind_t memory_kind, void* imported_host_ptr,
    iree_hal_streaming_mem_pool_t* allocation_pool,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(hrx_buf);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void**)&wrapper));
  memset(wrapper, 0, sizeof(*wrapper));

  // Initialize wrapper.
  wrapper->hrx_buf = hrx_buf;
  hrx_buffer_retain(wrapper->hrx_buf);
  wrapper->buffer = hrx_buf->hal_buffer;
  iree_hal_streaming_buffer_set_context(wrapper, context, context_ownership);
  wrapper->allocation_pool = allocation_pool;
  if (wrapper->allocation_pool) {
    iree_hal_streaming_mem_pool_retain(wrapper->allocation_pool);
  }
  wrapper->is_pool_allocation_live = false;
  wrapper->memory_type = wrapper->buffer
                             ? (int)iree_hal_buffer_memory_type(wrapper->buffer)
                             : memory_type;
  wrapper->host_register_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  wrapper->imported_host_allocation = imported_host_ptr != NULL;
  wrapper->memory_kind = memory_kind;
  iree_hal_streaming_allocation_preparation_initialize(&wrapper->preparation);
  wrapper->has_host_mapping = false;
  memset(&wrapper->host_mapping, 0, sizeof(wrapper->host_mapping));
  wrapper->managed_page_count = 0;
  wrapper->managed_read_mostly_pages = NULL;
  wrapper->managed_preferred_locations = NULL;
  wrapper->managed_accessed_by_device_masks = NULL;
  wrapper->managed_last_prefetch_locations = NULL;
  wrapper->managed_coherency_modes = NULL;
  iree_slim_mutex_initialize(&wrapper->context_import_mutex);
  wrapper->context_imports = NULL;
  wrapper->ipc_handle = NULL;
  wrapper->size = logical_size;
  wrapper->logical_size = wrapper->size;
  iree_hal_streaming_buffer_activate_pool_allocation(wrapper);

  // Initialize unified memory attributes.
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;  // Unspecified initially.
  wrapper->accessed_by_device_mask = 0;
  wrapper->last_prefetch_location = -2;  // Never prefetched.
  wrapper->coherency_mode = 0;           // Fine-grain by default.

  // HIP pointers can occur inside by-value kernel arguments and must be real
  // device addresses. The binding selects this device's prepared program
  // binding; external-storage transport support is independent of address
  // access.
  void* device_pointer = NULL;
  iree_status_t status = iree_hal_streaming_buffer_query_device_pointer(
      hrx_buf, context->device_entry->hrx_device, &device_pointer);
  wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)device_pointer;

  if (imported_host_ptr) {
    wrapper->host_ptr = imported_host_ptr;
  }
  // Optional mapping is resolved by the allocator. Only the achieved grant
  // permits a mapping attempt; a failure with that grant is an allocation
  // error.
  if (iree_status_is_ok(status) && wrapper->buffer && !wrapper->host_ptr &&
      iree_all_bits_set(iree_hal_buffer_allowed_usage(wrapper->buffer),
                        IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT)) {
    status = iree_hal_buffer_map_range(
        wrapper->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        iree_hal_buffer_allowed_access(wrapper->buffer),
        IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS, 0, wrapper->size,
        &wrapper->host_mapping);
    if (iree_status_is_ok(status)) {
      wrapper->host_ptr = wrapper->host_mapping.contents.data;
      wrapper->has_host_mapping = true;
    }
  }

  if (iree_status_is_ok(status)) {
    // Register buffer in context's mapping table.
    status = HRX_CALL(hrx_buffer_table_insert(
        &context->buffer_table, wrapper->device_ptr, wrapper->host_ptr,
        wrapper->size, wrapper->hrx_buf, wrapper));
  }

  if (iree_status_is_ok(status)) {
    *out_wrapper = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Wraps a HAL buffer in a stream buffer and caches information.
static iree_status_t iree_hal_streaming_buffer_wrap(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    int memory_type, iree_hal_streaming_memory_kind_t memory_kind,
    void* imported_host_ptr, iree_hal_streaming_mem_pool_t* allocation_pool,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_buffer_t hrx_buf = NULL;
  hrx_device_t hrx_dev =
      context->device_entry ? context->device_entry->hrx_device : NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_buffer_create_from_hal(
              buffer, hrx_dev ? hrx_dev->group : NULL,
              (hrx_memory_type_t)memory_type,
              (size_t)iree_hal_buffer_byte_length(buffer), NULL, &hrx_buf));

  iree_status_t status = iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, hrx_buf, hrx_buf->size, memory_type, memory_kind,
      imported_host_ptr, allocation_pool, context_ownership, out_wrapper);
  hrx_buffer_release(hrx_buf);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Frees a buffer wrapper and releases the underlying HRX buffer.
static void iree_hal_streaming_buffer_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = buffer->context->host_allocator;
  iree_allocator_free(host_allocator, buffer->managed_read_mostly_pages);
  buffer->managed_read_mostly_pages = NULL;
  iree_allocator_free(host_allocator, buffer->managed_preferred_locations);
  buffer->managed_preferred_locations = NULL;
  iree_allocator_free(host_allocator, buffer->managed_accessed_by_device_masks);
  buffer->managed_accessed_by_device_masks = NULL;
  iree_allocator_free(host_allocator, buffer->managed_last_prefetch_locations);
  buffer->managed_last_prefetch_locations = NULL;
  iree_allocator_free(host_allocator, buffer->managed_coherency_modes);
  buffer->managed_coherency_modes = NULL;
  buffer->managed_page_count = 0;
  iree_slim_mutex_lock(&buffer->context_import_mutex);
  iree_hal_streaming_context_import_t* context_imports =
      buffer->context_imports;
  buffer->context_imports = NULL;
  iree_slim_mutex_unlock(&buffer->context_import_mutex);
  while (context_imports) {
    iree_hal_streaming_context_import_t* import = context_imports;
    context_imports = import->next;
    iree_hal_buffer_release(import->buffer);
    iree_hal_pool_release(import->pool);
    iree_hal_streaming_context_release(import->context);
    iree_allocator_free(host_allocator, import);
  }
  iree_slim_mutex_deinitialize(&buffer->context_import_mutex);
  iree_hal_streaming_allocation_preparation_deinitialize(&buffer->preparation);
  if (buffer->has_host_mapping) {
    iree_status_t status = iree_hal_buffer_unmap_range(&buffer->host_mapping);
    if (!iree_status_is_ok(status)) {
      // Final release has no error channel. Report the terminal unmap error.
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
    memset(&buffer->host_mapping, 0, sizeof(buffer->host_mapping));
    buffer->has_host_mapping = false;
  }
  iree_hal_streaming_buffer_release_pool_allocation(buffer);
  if (buffer->hrx_buf) {
    hrx_buffer_release(buffer->hrx_buf);
    buffer->hrx_buf = NULL;
    buffer->buffer = NULL;
  }
  iree_hal_streaming_mem_pool_release(buffer->allocation_pool);
  buffer->allocation_pool = NULL;
  iree_hal_streaming_buffer_release_context(buffer);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_memory_wrap_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  return iree_hal_streaming_buffer_wrap(
      context, buffer, (int)iree_hal_buffer_memory_type(buffer),
      IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
      /*imported_host_ptr=*/NULL, /*allocation_pool=*/NULL, context_ownership,
      out_buffer);
}

iree_status_t iree_hal_streaming_memory_wrap_virtual_reservation(
    iree_hal_streaming_context_t* context, hrx_buffer_t virtual_buffer,
    iree_device_size_t logical_size, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, virtual_buffer, logical_size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
      /*imported_host_ptr=*/NULL, /*allocation_pool=*/NULL,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, out_buffer);
}

void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) {
    return;
  }
  hrx_buffer_table_remove(&buffer->context->buffer_table, buffer->device_ptr);
  iree_hal_streaming_buffer_free(buffer);
}

void iree_hal_streaming_memory_prepare_virtual_reservation_release(
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(buffer->hrx_buf);
  hrx_buffer_t virtual_buffer = buffer->hrx_buf;
  buffer->hrx_buf = NULL;
  buffer->buffer = NULL;
  hrx_buffer_release(virtual_buffer);
}

void iree_hal_streaming_memory_restore_virtual_reservation(
    iree_hal_streaming_buffer_t* buffer, hrx_buffer_t virtual_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(!buffer->hrx_buf);
  buffer->hrx_buf = virtual_buffer;
  hrx_buffer_retain(virtual_buffer);
  buffer->buffer = virtual_buffer->hal_buffer;
}

static void iree_hal_streaming_temporary_host_buffer_free(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) {
    return;
  }
  hrx_buffer_table_remove(&context->buffer_table, buffer->device_ptr);
  iree_hal_streaming_buffer_free(buffer);
}

iree_hal_streaming_deviceptr_t iree_hal_streaming_buffer_device_pointer(
    iree_hal_streaming_buffer_t* buffer) {
  return buffer ? buffer->device_ptr : 0;
}

iree_status_t iree_hal_streaming_memory_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  return iree_hal_streaming_memory_try_lookup(context, device_ptr, out_ref)
             ? iree_ok_status()
             : iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  size_t offset = 0;
  IREE_RETURN_IF_ERROR(HRX_CALL(hrx_buffer_table_find_range(
      &context->buffer_table, device_ptr, (size_t)size, NULL, &offset,
      (void**)&out_ref->buffer)));
  out_ref->offset = (iree_device_size_t)offset;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range_across_contexts(
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_context = NULL;
  memset(out_ref, 0, sizeof(*out_ref));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  bool found = false;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_hal_streaming_buffer_ref_t candidate_ref;
    iree_status_t status = iree_hal_streaming_memory_lookup_range(
        context, device_ptr, size, &candidate_ref);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_context_retain(context);
      *out_context = context;
      *out_ref = candidate_ref;
      found = true;
      break;
    }
    iree_status_ignore(status);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return found ? iree_ok_status()
               : iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

static hrx_status_t iree_hal_streaming_buffer_preparation_acquire_callback(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data) {
  (void)offset;
  (void)user_data;
  iree_hal_streaming_buffer_t* buffer =
      (iree_hal_streaming_buffer_t*)entry->user_data;
  if (!buffer) {
    return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                           "allocation has no streaming wrapper");
  }
  if (!iree_hal_streaming_allocation_preparation_try_acquire(
          &buffer->preparation)) {
    return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                           "allocation is closing");
  }
  return hrx_ok_status();
}

static void iree_hal_streaming_memory_release_table_ref(
    hrx_buffer_table_retained_ref_t* table_ref) {
  if (!table_ref || !table_ref->buffer) {
    return;
  }
  iree_hal_streaming_buffer_t* owner_wrapper =
      (iree_hal_streaming_buffer_t*)table_ref->user_data;
  hrx_buffer_release(table_ref->buffer);
  if (owner_wrapper) {
    iree_hal_streaming_allocation_preparation_release(
        &owner_wrapper->preparation);
  }
  memset(table_ref, 0, sizeof(*table_ref));
}

// Consumes |table_ref| and transfers its allocation lease to |out_ref|.
static iree_status_t iree_hal_streaming_memory_adopt_table_ref(
    iree_hal_streaming_context_t* owner_context,
    hrx_buffer_table_retained_ref_t* table_ref,
    iree_hal_streaming_retained_buffer_ref_t* out_ref) {
  memset(out_ref, 0, sizeof(*out_ref));
  iree_hal_streaming_buffer_t* owner_wrapper =
      (iree_hal_streaming_buffer_t*)table_ref->user_data;
  if (IREE_UNLIKELY(!table_ref->buffer || !table_ref->buffer->hal_buffer ||
                    !owner_wrapper)) {
    iree_hal_streaming_memory_release_table_ref(table_ref);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "registered allocation has no HAL buffer");
  }

  out_ref->owner_wrapper = owner_wrapper;
  out_ref->owner = table_ref->buffer;
  table_ref->buffer = NULL;
  out_ref->buffer = out_ref->owner->hal_buffer;
  iree_hal_buffer_retain(out_ref->buffer);
  out_ref->offset = (iree_device_size_t)table_ref->offset;
  out_ref->memory_type = iree_hal_buffer_memory_type(out_ref->buffer);
  out_ref->device_pointer = table_ref->device_ptr;
  out_ref->host_pointer = table_ref->host_ptr;
  out_ref->allocation_size = table_ref->size;
  out_ref->host_register_flags = owner_wrapper->host_register_flags;
  out_ref->is_cross_context = false;
  out_ref->owner_context = owner_context;
  iree_hal_streaming_context_retain(owner_context);
  memset(table_ref, 0, sizeof(*table_ref));
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range_retain(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_retained_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));

  hrx_buffer_table_retained_ref_t table_ref;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find_range_retain_if(
      &context->buffer_table, device_ptr, (size_t)size,
      iree_hal_streaming_buffer_preparation_acquire_callback,
      /*callback_user_data=*/NULL, &table_ref));
  if (!iree_status_is_ok(status)) {
    return status;
  }
  return iree_hal_streaming_memory_adopt_table_ref(context, &table_ref,
                                                   out_ref);
}

static bool iree_hal_streaming_context_has_enabled_peer(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  bool enabled = false;
  iree_slim_mutex_lock(&context->mutex);
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      enabled = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&context->mutex);
  return enabled;
}

iree_status_t iree_hal_streaming_memory_buffer_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_buffer_t* buffer, bool allow_peer_device_allocation,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(execution_context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  if (buffer->context == execution_context) {
    *out_buffer = buffer->buffer;
    return iree_ok_status();
  }

  const bool import_host_allocation =
      buffer->memory_kind != IREE_HAL_STREAMING_MEMORY_KIND_DEVICE;
  const bool import_device_allocation =
      allow_peer_device_allocation &&
      buffer->memory_kind == IREE_HAL_STREAMING_MEMORY_KIND_DEVICE;
  if (!import_host_allocation && !import_device_allocation) {
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  if ((buffer->memory_kind == IREE_HAL_STREAMING_MEMORY_KIND_MANAGED) &&
      (!buffer->host_ptr ||
       (uint64_t)(uintptr_t)buffer->host_ptr != buffer->device_ptr)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cross-device managed memory requires one stable host/device address");
  }
  if (!buffer->buffer || buffer->device_ptr == 0 || buffer->size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation is missing device import metadata");
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&buffer->context_import_mutex);
  for (iree_hal_streaming_context_import_t* import = buffer->context_imports;
       import; import = import->next) {
    if (import->context == execution_context) {
      *out_buffer = import->buffer;
      iree_slim_mutex_unlock(&buffer->context_import_mutex);
      return iree_ok_status();
    }
  }

  iree_hal_external_storage_t external_buffer = {
      .storage_class = iree_hal_make_external_storage_class(
          import_host_allocation
              ? IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER
              : IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS),
      .byte_length = buffer->size,
      .maintenance_range = iree_hal_make_buffer_range(0, buffer->size),
  };
  if (import_host_allocation) {
    external_buffer.payload.host_pointer = buffer->host_ptr;
    external_buffer.host_properties =
        (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
         IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
  } else {
    external_buffer.payload.device_address = buffer->device_ptr;
  }

  iree_hal_pool_t* import_pool = NULL;
  iree_hal_buffer_t* imported_buffer = NULL;
  status = hrx_mem_pool_backing_retain_import_source(
      execution_context->device_entry->hrx_device,
      external_buffer.storage_class, &import_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_import_buffer(
        import_pool, &external_buffer,
        iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER), &imported_buffer);
  }
  iree_hal_streaming_context_import_t* import = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(buffer->context->host_allocator,
                                   sizeof(*import), (void**)&import);
  }
  if (iree_status_is_ok(status)) {
    import->next = buffer->context_imports;
    import->context = execution_context;
    iree_hal_streaming_context_retain(execution_context);
    import->buffer = imported_buffer;
    import->pool = import_pool;
    buffer->context_imports = import;
    imported_buffer = NULL;
    import_pool = NULL;
    *out_buffer = import->buffer;
  }
  iree_slim_mutex_unlock(&buffer->context_import_mutex);
  iree_hal_buffer_release(imported_buffer);
  iree_hal_pool_release(import_pool);
  return status;
}

static iree_status_t iree_hal_streaming_memory_prepare_ref_for_context(
    iree_hal_streaming_context_t* execution_context, bool peer_access_enabled,
    iree_hal_streaming_retained_buffer_ref_t* ref) {
  if (ref->owner_context == execution_context) {
    return iree_ok_status();
  }

  const bool is_device_import =
      ref->owner_wrapper->memory_kind == IREE_HAL_STREAMING_MEMORY_KIND_DEVICE;
  if (is_device_import && !peer_access_enabled) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "device-local stream value target requires enabled peer access");
  }

  iree_hal_buffer_t* imported_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_buffer_for_context(
      execution_context, ref->owner_wrapper,
      /*allow_peer_device_allocation=*/true, &imported_buffer));
  iree_hal_buffer_retain(imported_buffer);
  iree_hal_buffer_release(ref->buffer);
  ref->buffer = imported_buffer;
  ref->is_cross_context = true;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range_retain_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_retained_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(execution_context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  const iree_hal_streaming_memory_range_request_t request = {
      .address = device_ptr,
      .length = size,
  };
  iree_hal_streaming_memory_range_match_t match;
  iree_host_size_t ref_count = 0;
  iree_status_t status =
      iree_hal_streaming_memory_lookup_ranges_retain_for_context(
          execution_context, 1, &request, 1, out_ref, &ref_count, &match);
  if (iree_status_is_ok(status)) {
    out_ref->offset = match.offset;
  }
  return status;
}

void iree_hal_streaming_retained_buffer_ref_deinitialize(
    iree_hal_streaming_retained_buffer_ref_t* ref) {
  if (!ref) {
    return;
  }
  iree_hal_streaming_buffer_t* owner_wrapper = ref->owner_wrapper;
  iree_hal_buffer_release(ref->buffer);
  hrx_buffer_release(ref->owner);
  iree_hal_streaming_context_release(ref->owner_context);
  memset(ref, 0, sizeof(*ref));
  if (owner_wrapper) {
    iree_hal_streaming_allocation_preparation_release(
        &owner_wrapper->preparation);
  }
}

enum { IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT = 16 };

typedef struct iree_hal_streaming_memory_range_workspace_t {
  // Compacted unresolved requests for one allocation table.
  hrx_buffer_table_range_request_t* table_requests;
  // Original request ordinal corresponding to each compacted request.
  iree_host_size_t* request_indices;
  // Unique references retained from one allocation table.
  hrx_buffer_table_retained_ref_t* table_refs;
  // Per-request matches returned by one allocation table.
  hrx_buffer_table_range_match_t* table_matches;
  // Heap storage backing arrays larger than the inline capacity.
  void* allocated_storage;
  // Inline request storage.
  hrx_buffer_table_range_request_t
      inline_table_requests[IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT];
  // Inline original-ordinal storage.
  iree_host_size_t
      inline_request_indices[IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT];
  // Inline retained-reference storage.
  hrx_buffer_table_retained_ref_t
      inline_table_refs[IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT];
  // Inline match storage.
  hrx_buffer_table_range_match_t
      inline_table_matches[IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT];
} iree_hal_streaming_memory_range_workspace_t;

static iree_status_t iree_hal_streaming_memory_range_workspace_initialize(
    iree_hal_streaming_context_t* context, iree_host_size_t request_count,
    iree_hal_streaming_memory_range_workspace_t* out_workspace) {
  memset(out_workspace, 0, sizeof(*out_workspace));
  out_workspace->table_requests = out_workspace->inline_table_requests;
  out_workspace->request_indices = out_workspace->inline_request_indices;
  out_workspace->table_refs = out_workspace->inline_table_refs;
  out_workspace->table_matches = out_workspace->inline_table_matches;
  if (request_count <= IREE_HAL_STREAMING_MEMORY_INLINE_RANGE_COUNT) {
    return iree_ok_status();
  }

  iree_host_size_t requests_size = 0;
  iree_host_size_t indices_offset = 0;
  iree_host_size_t indices_size = 0;
  iree_host_size_t refs_offset = 0;
  iree_host_size_t refs_size = 0;
  iree_host_size_t matches_offset = 0;
  iree_host_size_t matches_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(request_count,
                                      sizeof(*out_workspace->table_requests),
                                      &requests_size) ||
          !iree_host_size_checked_align(requests_size, iree_max_align_t,
                                        &indices_offset) ||
          !iree_host_size_checked_mul(request_count,
                                      sizeof(*out_workspace->request_indices),
                                      &indices_size) ||
          !iree_host_size_checked_add(indices_offset, indices_size,
                                      &refs_offset) ||
          !iree_host_size_checked_align(refs_offset, iree_max_align_t,
                                        &refs_offset) ||
          !iree_host_size_checked_mul(
              request_count, sizeof(*out_workspace->table_refs), &refs_size) ||
          !iree_host_size_checked_add(refs_offset, refs_size,
                                      &matches_offset) ||
          !iree_host_size_checked_align(matches_offset, iree_max_align_t,
                                        &matches_offset) ||
          !iree_host_size_checked_mul(request_count,
                                      sizeof(*out_workspace->table_matches),
                                      &matches_size) ||
          !iree_host_size_checked_add(matches_offset, matches_size,
                                      &total_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "range lookup workspace size overflow");
  }

  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      context->host_allocator, total_size, &out_workspace->allocated_storage));
  uint8_t* storage = (uint8_t*)out_workspace->allocated_storage;
  out_workspace->table_requests = (hrx_buffer_table_range_request_t*)storage;
  out_workspace->request_indices =
      (iree_host_size_t*)(storage + indices_offset);
  out_workspace->table_refs =
      (hrx_buffer_table_retained_ref_t*)(storage + refs_offset);
  out_workspace->table_matches =
      (hrx_buffer_table_range_match_t*)(storage + matches_offset);
  return iree_ok_status();
}

static void iree_hal_streaming_memory_range_workspace_deinitialize(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_memory_range_workspace_t* workspace) {
  iree_allocator_free(context->host_allocator, workspace->allocated_storage);
  memset(workspace, 0, sizeof(*workspace));
}

static iree_status_t iree_hal_streaming_memory_lookup_ranges_in_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_context_t* owner_context, iree_host_size_t request_count,
    const iree_hal_streaming_memory_range_request_t* requests,
    iree_hal_streaming_memory_range_workspace_t* workspace,
    iree_host_size_t* inout_ref_count,
    iree_hal_streaming_retained_buffer_ref_t* out_refs,
    iree_hal_streaming_memory_range_match_t* out_matches) {
  iree_host_size_t unresolved_count = 0;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    if (out_matches[i].ref_index != IREE_HOST_SIZE_MAX) {
      continue;
    }
    workspace->table_requests[unresolved_count] =
        (hrx_buffer_table_range_request_t){
            .address = requests[i].address,
            .length = (size_t)requests[i].length,
        };
    workspace->request_indices[unresolved_count] = i;
    ++unresolved_count;
  }
  if (unresolved_count == 0) {
    return iree_ok_status();
  }

  size_t table_ref_count = 0;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find_ranges_retain_if(
      &owner_context->buffer_table, (size_t)unresolved_count,
      workspace->table_requests,
      iree_hal_streaming_buffer_preparation_acquire_callback,
      /*callback_user_data=*/NULL, (size_t)unresolved_count,
      workspace->table_refs, &table_ref_count, workspace->table_matches));
  if (!iree_status_is_ok(status)) {
    for (size_t i = 0; i < table_ref_count; ++i) {
      iree_hal_streaming_memory_release_table_ref(&workspace->table_refs[i]);
    }
    return status;
  }

  const iree_host_size_t ref_base = *inout_ref_count;
  const bool peer_access_enabled = table_ref_count == 0 ||
                                   owner_context == execution_context ||
                                   iree_hal_streaming_context_has_enabled_peer(
                                       execution_context, owner_context);
  size_t table_ref_index = 0;
  for (; table_ref_index < table_ref_count; ++table_ref_index) {
    status = iree_hal_streaming_memory_adopt_table_ref(
        owner_context, &workspace->table_refs[table_ref_index],
        &out_refs[*inout_ref_count]);
    if (!iree_status_is_ok(status)) {
      break;
    }
    ++*inout_ref_count;
    status = iree_hal_streaming_memory_prepare_ref_for_context(
        execution_context, peer_access_enabled,
        &out_refs[*inout_ref_count - 1]);
    if (!iree_status_is_ok(status)) {
      break;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (size_t i = table_ref_index + 1; i < table_ref_count; ++i) {
      iree_hal_streaming_memory_release_table_ref(&workspace->table_refs[i]);
    }
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }

  for (iree_host_size_t i = 0; i < unresolved_count; ++i) {
    const hrx_buffer_table_range_match_t table_match =
        workspace->table_matches[i];
    if (table_match.ref_index == SIZE_MAX) {
      continue;
    }
    out_matches[workspace->request_indices[i]] =
        (iree_hal_streaming_memory_range_match_t){
            .ref_index = ref_base + table_match.ref_index,
            .offset = (iree_device_size_t)table_match.offset,
        };
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_ranges_retain_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_host_size_t request_count,
    const iree_hal_streaming_memory_range_request_t* requests,
    iree_host_size_t ref_capacity,
    iree_hal_streaming_retained_buffer_ref_t* out_refs,
    iree_host_size_t* out_ref_count,
    iree_hal_streaming_memory_range_match_t* out_matches) {
  IREE_ASSERT_ARGUMENT(execution_context);
  IREE_ASSERT_ARGUMENT(out_ref_count);
  *out_ref_count = 0;
  if (request_count == 0 || !requests || !out_refs || !out_matches) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range lookup requires non-empty storage");
  }
  if (ref_capacity < request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "retained reference storage is too small");
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    out_matches[i] = (iree_hal_streaming_memory_range_match_t){
        .ref_index = IREE_HOST_SIZE_MAX,
        .offset = 0,
    };
    if (requests[i].length == 0 || requests[i].length > SIZE_MAX ||
        requests[i].length > UINT64_MAX - requests[i].address) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "range lookup request is invalid");
    }
  }

  iree_hal_streaming_memory_range_workspace_t workspace;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_range_workspace_initialize(
      execution_context, request_count, &workspace));

  iree_status_t status = iree_hal_streaming_memory_lookup_ranges_in_context(
      execution_context, execution_context, request_count, requests, &workspace,
      out_ref_count, out_refs, out_matches);

  enum { IREE_HAL_STREAMING_INLINE_CONTEXT_COUNT = 8 };
  iree_hal_streaming_context_t*
      inline_contexts[IREE_HAL_STREAMING_INLINE_CONTEXT_COUNT];
  iree_hal_streaming_context_t** contexts = inline_contexts;
  iree_hal_streaming_context_t** allocated_contexts = NULL;
  iree_host_size_t context_count = 0;
  if (iree_status_is_ok(status)) {
    bool all_resolved = true;
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      all_resolved &= out_matches[i].ref_index != IREE_HOST_SIZE_MAX;
    }
    if (!all_resolved) {
      iree_hal_streaming_device_registry_t* device_registry =
          iree_hal_streaming_device_registry();
      if (!device_registry) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "HAL stream layer not initialized");
      } else {
        iree_slim_mutex_lock(&device_registry->context_list.mutex);
        for (iree_hal_streaming_context_t* context =
                 device_registry->context_list.head;
             context; context = context->context_list_entry.next) {
          if (context != execution_context) {
            ++context_count;
          }
        }
        if (context_count > IREE_ARRAYSIZE(inline_contexts)) {
          iree_host_size_t contexts_size = 0;
          if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                  context_count, sizeof(*contexts), &contexts_size))) {
            status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                      "context snapshot size overflow");
          } else {
            status = iree_allocator_malloc(execution_context->host_allocator,
                                           contexts_size,
                                           (void**)&allocated_contexts);
          }
          if (iree_status_is_ok(status)) {
            contexts = allocated_contexts;
          }
        }
        if (iree_status_is_ok(status)) {
          iree_host_size_t index = 0;
          for (iree_hal_streaming_context_t* context =
                   device_registry->context_list.head;
               context; context = context->context_list_entry.next) {
            if (context == execution_context) {
              continue;
            }
            contexts[index++] = context;
            iree_hal_streaming_context_retain(context);
          }
          context_count = index;
        } else {
          context_count = 0;
        }
        iree_slim_mutex_unlock(&device_registry->context_list.mutex);
      }
    }
  }

  for (iree_host_size_t i = 0; i < context_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_memory_lookup_ranges_in_context(
        execution_context, contexts[i], request_count, requests, &workspace,
        out_ref_count, out_refs, out_matches);
  }
  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  iree_allocator_free(execution_context->host_allocator, allocated_contexts);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      if (out_matches[i].ref_index == IREE_HOST_SIZE_MAX) {
        status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
        break;
      }
    }
  }
  iree_hal_streaming_memory_range_workspace_deinitialize(execution_context,
                                                         &workspace);
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < *out_ref_count; ++i) {
      iree_hal_streaming_retained_buffer_ref_deinitialize(&out_refs[i]);
    }
    *out_ref_count = 0;
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_matches[i] = (iree_hal_streaming_memory_range_match_t){
          .ref_index = IREE_HOST_SIZE_MAX,
          .offset = 0,
      };
    }
  }
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  if (iree_any_bit_set(flags, IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED;
  }
  iree_hal_pool_t* pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      context->device_entry->hrx_device, (hrx_memory_type_t)memory_type,
      HRX_BUFFER_USAGE_DEFAULT, &pool);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(size, /*min_alignment=*/64);
    status = iree_hal_pool_allocate_buffer(pool, request,
                                           iree_infinite_timeout(), &buffer);
  }

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_wrap(
        context, buffer, (int)memory_type,
        IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
        /*imported_host_ptr=*/NULL,
        /*allocation_pool=*/NULL, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
        &wrapper);
  }

  // Release our reference (wrapper holds its own).
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
    iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                        size);
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_device_from_pool(
    iree_hal_streaming_context_t* context, iree_hal_streaming_mem_pool_t* pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_any_bit_set(flags, IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "memory-pool allocations cannot satisfy uncached device memory");
  }
  hrx_buffer_t hrx_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_mem_pool_allocate_buffer(pool, size, &hrx_buffer));

  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, hrx_buffer, size, (int)hrx_buffer->mem_type,
      IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
      /*imported_host_ptr=*/NULL, pool,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, &wrapper);
  hrx_buffer_release(hrx_buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
    iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                        size);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

struct iree_hal_streaming_deferred_device_free_t {
  // Next pending async free in the owning context's reuse registry.
  iree_hal_streaming_deferred_device_free_t* next;
  // Context owning the registry and allocation bookkeeping.
  iree_hal_streaming_context_t* owner_context;
  // Stable identifier of the stream that orders this logical free.
  unsigned long long source_stream_id;
  // Timeline value reached after this free's host completion callback.
  uint64_t completion_value;
  // Detached allocation wrapper to release on the stream timeline.
  iree_hal_streaming_buffer_t* buffer;
  // True while this operation is visible in |owner_context|'s registry.
  bool is_registered;
  // True when the host call ran after all stream prerequisites completed.
  bool is_ready;
  // True after the queue has released its terminal resource reference.
  bool is_terminal;
  // True when a later stream-ordered allocation adopted |buffer|.
  bool is_reused;
  // Reuse policy captured when the free is enqueued.
  bool allow_opportunistic;
};

typedef struct iree_hal_streaming_pending_free_terminal_t {
  // Resource retained by the accepted host call until callback completion or
  // cancellation. It transitively owns the operation's context retain.
  iree_hal_resource_t resource;
  // Context-owned operation notified when the queued call becomes terminal.
  iree_hal_streaming_deferred_device_free_t* free_op;
} iree_hal_streaming_pending_free_terminal_t;

static void iree_hal_streaming_pending_free_terminal_destroy(
    iree_hal_resource_t* resource);

static const iree_hal_resource_vtable_t
    iree_hal_streaming_pending_free_terminal_vtable = {
        .destroy = iree_hal_streaming_pending_free_terminal_destroy,
};

static void iree_hal_streaming_pending_free_destroy(
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_allocator_free(iree_allocator_system(), free_op);
}

static iree_status_t iree_hal_streaming_buffer_free_and_trim_pool(
    iree_hal_streaming_buffer_t* buffer, bool trim_to_release_threshold) {
  // Pool-backed buffers retain their pool. Preserve it across buffer
  // destruction so that any now-unused backing storage can be released.
  iree_hal_streaming_mem_pool_t* allocation_pool = buffer->allocation_pool;
  if (allocation_pool) {
    iree_hal_streaming_mem_pool_retain(allocation_pool);
  }
  iree_hal_streaming_buffer_free(buffer);

  iree_status_t status = iree_ok_status();
  if (allocation_pool && trim_to_release_threshold) {
    status = iree_hal_streaming_mem_pool_release_unused(allocation_pool);
  }
  iree_hal_streaming_mem_pool_release(allocation_pool);
  return status;
}

static iree_status_t iree_hal_streaming_pending_free_release_buffer(
    iree_hal_streaming_deferred_device_free_t* free_op,
    bool trim_to_release_threshold) {
  iree_status_t status = iree_hal_streaming_buffer_free_and_trim_pool(
      free_op->buffer, trim_to_release_threshold);
  iree_hal_streaming_pending_free_destroy(free_op);
  return status;
}

static void iree_hal_streaming_pending_free_remove_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current && *current != free_op) {
    current = &(*current)->next;
  }
  IREE_ASSERT(*current == free_op);
  *current = free_op->next;
  free_op->next = NULL;
  free_op->is_registered = false;
}

static void iree_hal_streaming_pending_free_terminal_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_streaming_pending_free_terminal_t* terminal =
      (iree_hal_streaming_pending_free_terminal_t*)resource;
  iree_hal_streaming_deferred_device_free_t* free_op = terminal->free_op;
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_hal_streaming_buffer_t* buffer = NULL;
  bool destroy_free_op = false;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->is_terminal = true;
  if (free_op->is_reused) {
    destroy_free_op = true;
  } else if (!free_op->is_ready || free_op->allow_opportunistic) {
    if (free_op->is_registered) {
      iree_hal_streaming_pending_free_remove_locked(context, free_op);
    }
    buffer = free_op->buffer;
    free_op->buffer = NULL;
    destroy_free_op = true;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  if (buffer) {
    iree_status_t status = iree_hal_streaming_buffer_free_and_trim_pool(
        buffer, /*trim_to_release_threshold=*/true);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
  }
  if (destroy_free_op) {
    iree_hal_streaming_pending_free_destroy(free_op);
  }
  iree_allocator_free(iree_allocator_system(), terminal);
  iree_hal_streaming_context_release(context);
}

static void iree_hal_streaming_pending_free_register(
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->next = context->pending_free_head;
  context->pending_free_head = free_op;
  free_op->is_registered = true;
  iree_slim_mutex_unlock(&context->pending_free_mutex);
}

static iree_status_t iree_hal_streaming_memory_try_reuse_pending_free(
    iree_hal_streaming_context_t* context, iree_hal_streaming_mem_pool_t* pool,
    iree_device_size_t size, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t** out_buffer) {
  *out_buffer = NULL;

  bool destroy_free_op = false;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current) {
    iree_hal_streaming_deferred_device_free_t* free_op = *current;
    iree_hal_streaming_buffer_t* buffer = free_op->buffer;
    bool can_reuse = buffer->allocation_pool == pool && buffer->size == size;
    if (can_reuse && free_op->source_stream_id != stream->stream_id) {
      const iree_hal_streaming_mem_pool_reuse_t reuse =
          iree_hal_streaming_mem_pool_query_reuse(pool);
      can_reuse =
          (reuse.follow_event_dependencies && free_op->completion_value != 0 &&
           iree_hal_streaming_stream_has_memory_reuse_dependency(
               stream, free_op->source_stream_id, free_op->completion_value)) ||
          (reuse.allow_opportunistic && free_op->is_ready);
    }
    if (can_reuse) {
      iree_hal_streaming_allocation_preparation_reopen(&buffer->preparation);
      hrx_status_t insert_status = hrx_buffer_table_insert(
          &context->buffer_table, buffer->device_ptr, buffer->host_ptr,
          buffer->size, buffer->hrx_buf, buffer);
      if (!hrx_status_is_ok(insert_status)) {
        iree_hal_streaming_allocation_preparation_begin_close(
            &buffer->preparation);
        iree_slim_mutex_unlock(&context->pending_free_mutex);
        return HRX_CALL(insert_status);
      }
      *current = free_op->next;
      free_op->next = NULL;
      free_op->is_registered = false;
      free_op->is_reused = true;
      free_op->buffer = NULL;
      destroy_free_op = free_op->is_terminal;
      iree_hal_streaming_buffer_activate_pool_allocation(buffer);
      iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                          buffer->size);
      *out_buffer = buffer;
      iree_slim_mutex_unlock(&context->pending_free_mutex);
      if (destroy_free_op) {
        iree_hal_streaming_pending_free_destroy(free_op);
      }
      return iree_ok_status();
    }
    current = &free_op->next;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_device_from_pool_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_mem_pool_t* pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_try_reuse_pending_free(
      context, pool, size, stream, out_buffer));
  if (*out_buffer) {
    return iree_ok_status();
  }
  return iree_hal_streaming_memory_allocate_device_from_pool(
      context, pool, size, flags, out_buffer);
}

iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t* context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t* out_pitch, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_pitch);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_pitch = 0;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Match HIP's observed pitched allocation granularity.
  const iree_device_size_t alignment = 512;
  iree_device_size_t pitch = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul_add(width_bytes, 1,
                                                      alignment - 1, &pitch))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pitched allocation width overflows");
  }
  pitch = pitch / alignment * alignment;

  // Element sizes of 4, 8, or 16 bytes preserve coalesced access assumptions.
  // The allocation contract does not require enforcing that here.

  // Calculate total size.
  iree_device_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(pitch, height, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pitched allocation size overflows");
  }

  // Allocate the buffer with the calculated total size.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, total_size, 0, &buffer);

  if (iree_status_is_ok(status)) {
    *out_pitch = pitch;
    *out_buffer = buffer;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_streaming_buffer_is_device_freeable_base(
    const iree_hal_streaming_buffer_t* buffer,
    iree_hal_streaming_deviceptr_t ptr, size_t offset) {
  if (!buffer || offset != 0) {
    return false;
  }
  if (buffer->device_ptr == ptr &&
      buffer->memory_kind == IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
    return true;
  }
  if (buffer->memory_kind != IREE_HAL_STREAMING_MEMORY_KIND_MANAGED) {
    return false;
  }
  if (buffer->device_ptr == ptr) {
    return true;
  }
  return buffer->host_ptr &&
         (iree_hal_streaming_deviceptr_t)(uintptr_t)buffer->host_ptr == ptr;
}

static hrx_status_t iree_hal_streaming_device_allocation_close_callback(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data) {
  const iree_hal_streaming_deviceptr_t ptr =
      *(const iree_hal_streaming_deviceptr_t*)user_data;
  iree_hal_streaming_buffer_t* buffer =
      (iree_hal_streaming_buffer_t*)entry->user_data;
  if (!iree_hal_streaming_buffer_is_device_freeable_base(buffer, ptr, offset)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           buffer && buffer->device_ptr == ptr && offset == 0
                               ? "pointer is not a device allocation"
                               : "device pointer is not an allocation base");
  }
  iree_hal_streaming_allocation_preparation_begin_close(&buffer->preparation);
  return hrx_ok_status();
}

typedef enum iree_hal_streaming_host_close_kind_e {
  IREE_HAL_STREAMING_HOST_CLOSE_KIND_FREE = 0,
  IREE_HAL_STREAMING_HOST_CLOSE_KIND_UNREGISTER = 1,
} iree_hal_streaming_host_close_kind_t;

typedef struct iree_hal_streaming_host_close_params_t {
  void* pointer;
  iree_hal_streaming_host_close_kind_t kind;
} iree_hal_streaming_host_close_params_t;

static hrx_status_t iree_hal_streaming_host_allocation_close_callback(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data) {
  const iree_hal_streaming_host_close_params_t* params =
      (const iree_hal_streaming_host_close_params_t*)user_data;
  iree_hal_streaming_buffer_t* buffer =
      (iree_hal_streaming_buffer_t*)entry->user_data;
  if (!buffer || buffer->host_ptr != params->pointer || offset != 0 ||
      buffer->memory_kind != IREE_HAL_STREAMING_MEMORY_KIND_HOST) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "host pointer is not an allocation base");
  }
  if (params->kind == IREE_HAL_STREAMING_HOST_CLOSE_KIND_FREE) {
    if (buffer->imported_host_allocation) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "registered host memory must be unregistered");
    }
  } else if (!buffer->imported_host_allocation) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "host pointer is not a registered allocation base");
  }
  iree_hal_streaming_allocation_preparation_begin_close(&buffer->preparation);
  return hrx_ok_status();
}

static iree_status_t iree_hal_streaming_memory_take_from_context(
    iree_hal_streaming_context_t* context, uint64_t pointer,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    iree_hal_streaming_buffer_t** out_buffer) {
  hrx_buffer_table_entry_t entry;
  iree_status_t status = HRX_CALL(hrx_buffer_table_remove_reserved_if(
      &context->buffer_table, pointer, callback, callback_user_data, &entry,
      /*out_offset=*/NULL));
  if (iree_status_is_ok(status)) {
    *out_buffer = (iree_hal_streaming_buffer_t*)entry.user_data;
  }
  return status;
}

static iree_status_t iree_hal_streaming_memory_take_allocation_context(
    iree_hal_streaming_context_t* preferred_context, uint64_t pointer,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(preferred_context);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_context = NULL;
  *out_buffer = NULL;

  iree_status_t status = iree_hal_streaming_memory_take_from_context(
      preferred_context, pointer, callback, callback_user_data, out_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_context_retain(preferred_context);
    *out_context = preferred_context;
    return status;
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return status;
  }
  iree_status_free(status);
  status = iree_status_from_code(IREE_STATUS_NOT_FOUND);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context == preferred_context) {
      continue;
    }
    status = iree_hal_streaming_memory_take_from_context(
        context, pointer, callback, callback_user_data, out_buffer);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_context_retain(context);
      *out_context = context;
      break;
    }
    if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
      break;
    }
    iree_status_free(status);
    status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return status;
}

static iree_status_t iree_hal_streaming_memory_restore_taken_allocation(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_t* buffer) {
  iree_hal_streaming_allocation_preparation_reopen(&buffer->preparation);
  hrx_status_t status = hrx_buffer_table_insert_reserved(
      &context->buffer_table, buffer->device_ptr, buffer->host_ptr,
      buffer->size, buffer->hrx_buf, buffer);
  if (!hrx_status_is_ok(status)) {
    iree_hal_streaming_allocation_preparation_begin_close(&buffer->preparation);
  }
  return HRX_CALL(status);
}

static iree_status_t iree_hal_streaming_memory_lookup_host_allocation(
    iree_hal_streaming_context_t* preferred_context, void* pointer,
    iree_hal_streaming_retained_buffer_ref_t* out_ref) {
  const iree_hal_streaming_deviceptr_t address =
      (iree_hal_streaming_deviceptr_t)(uintptr_t)pointer;
  iree_status_t status = iree_hal_streaming_memory_lookup_range_retain(
      preferred_context, address, /*size=*/1, out_ref);
  if (iree_status_is_ok(status)) {
    if (out_ref->host_pointer && out_ref->owner_wrapper->memory_kind ==
                                     IREE_HAL_STREAMING_MEMORY_KIND_HOST) {
      return status;
    }
    iree_hal_streaming_retained_buffer_ref_deinitialize(out_ref);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pointer is not a host allocation");
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return status;
  }
  iree_status_free(status);
  status = iree_status_from_code(IREE_STATUS_NOT_FOUND);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context == preferred_context) {
      continue;
    }
    status = iree_hal_streaming_memory_lookup_range_retain(context, address,
                                                           /*size=*/1, out_ref);
    if (iree_status_is_ok(status) ||
        iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
      break;
    }
    iree_status_free(status);
    status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  if (iree_status_is_ok(status) &&
      (!out_ref->host_pointer || out_ref->owner_wrapper->memory_kind !=
                                     IREE_HAL_STREAMING_MEMORY_KIND_HOST)) {
    iree_hal_streaming_retained_buffer_ref_deinitialize(out_ref);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pointer is not a host allocation");
  }
  return status;
}

iree_status_t iree_hal_streaming_memory_discard_unpublished_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);

  iree_hal_streaming_buffer_ref_t ref;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_lookup(context, ptr, &ref));
  if (!ref.buffer || ref.offset != 0 || ref.buffer->device_ptr != ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "discard pointer is not an allocation base");
  }
  IREE_RETURN_IF_ERROR(
      HRX_CALL(hrx_buffer_table_remove(&context->buffer_table, ptr)));
  iree_hal_streaming_memory_account_device_free(ref.buffer);
  return iree_hal_streaming_buffer_free_and_trim_pool(
      ref.buffer, /*trim_to_release_threshold=*/true);
}

iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr) {
  if (!ptr) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_allocation_t* graph_allocation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocation_lookup(context, ptr,
                                                     &graph_allocation));
  if (graph_allocation) {
    iree_status_t graph_status =
        iree_hal_streaming_graph_allocation_free(graph_allocation, NULL);
    iree_hal_streaming_graph_allocation_release(graph_allocation);
    IREE_TRACE_ZONE_END(z0);
    return graph_status;
  }

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_memory_take_allocation_context(
      context, ptr, iree_hal_streaming_device_allocation_close_callback, &ptr,
      &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // The table removal closes admission atomically. Existing preparers hold an
  // allocation-local lease and cannot be overtaken by synchronization.
  iree_hal_streaming_allocation_preparation_await_idle(&wrapper->preparation);
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);

  // A HIP pointer may be hidden in native kernargs or device memory, so freeing
  // cannot rely on launch-time binding discovery. The synchronization above
  // waits every active context before releasing the allocation.

  // Update free memory tracking.
  iree_hal_streaming_memory_account_device_free(wrapper);

  // Synchronous frees return an entirely idle pool's backing storage to the
  // allocator. Ordinary device allocations have no pool to trim.
  status = iree_hal_streaming_buffer_free_and_trim_pool(
      wrapper, /*trim_to_release_threshold=*/true);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_deferred_device_free(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* call_context) {
  (void)args;
  (void)call_context;
  iree_hal_streaming_deferred_device_free_t* free_op =
      (iree_hal_streaming_deferred_device_free_t*)user_data;
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->is_ready = true;
  iree_slim_mutex_unlock(&context->pending_free_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_pending_free_release_list(
    iree_hal_streaming_deferred_device_free_t* completed_frees,
    bool trim_to_release_threshold) {
  iree_status_t status = iree_ok_status();
  while (completed_frees) {
    iree_hal_streaming_deferred_device_free_t* free_op = completed_frees;
    completed_frees = free_op->next;
    status =
        iree_status_join(status, iree_hal_streaming_pending_free_release_buffer(
                                     free_op, trim_to_release_threshold));
  }
  return status;
}

iree_status_t iree_hal_streaming_memory_release_completed_async_frees(
    iree_hal_streaming_stream_t* stream) {
  // Context synchronization can retain a stream snapshot while public stream
  // destruction detaches it. Retain under the stream lock so cleanup either
  // owns a live context or leaves its deferred frees with the context registry.
  iree_hal_streaming_context_t* context = NULL;
  if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
    return iree_ok_status();
  }
  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current) {
    iree_hal_streaming_deferred_device_free_t* free_op = *current;
    if (free_op->source_stream_id == stream->stream_id && free_op->is_ready &&
        free_op->is_terminal) {
      *current = free_op->next;
      free_op->next = completed_frees;
      free_op->is_registered = false;
      completed_frees = free_op;
    } else {
      current = &free_op->next;
    }
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  iree_status_t status = iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/true);
  iree_hal_streaming_context_release(context);
  return status;
}

iree_status_t iree_hal_streaming_memory_release_terminal_async_frees(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  while (context->pending_free_head) {
    iree_hal_streaming_deferred_device_free_t* free_op =
        context->pending_free_head;
    IREE_ASSERT(free_op->is_terminal,
                "a nonterminal asynchronous free retained its context");
    context->pending_free_head = free_op->next;
    free_op->next = completed_frees;
    free_op->is_registered = false;
    completed_frees = free_op;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  return iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/true);
}

iree_status_t iree_hal_streaming_memory_release_completed_async_frees_from_pool(
    iree_hal_streaming_mem_pool_t* pool) {
  if (!pool) {
    return iree_ok_status();
  }

  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  // Pool trimming is an explicit cold-path operation. Scan contexts here
  // instead of imposing a process-wide registry on every allocation.
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_slim_mutex_lock(&context->pending_free_mutex);
    iree_hal_streaming_deferred_device_free_t** current =
        &context->pending_free_head;
    while (*current) {
      iree_hal_streaming_deferred_device_free_t* free_op = *current;
      if (free_op->is_ready && free_op->is_terminal && free_op->buffer &&
          free_op->buffer->allocation_pool == pool) {
        *current = free_op->next;
        free_op->next = completed_frees;
        free_op->is_registered = false;
        completed_frees = free_op;
      } else {
        current = &free_op->next;
      }
    }
    iree_slim_mutex_unlock(&context->pending_free_mutex);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  return iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/false);
}

iree_status_t iree_hal_streaming_memory_free_device_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream) {
  if (!ptr) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_allocation_t* graph_allocation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocation_lookup(context, ptr,
                                                     &graph_allocation));
  if (graph_allocation) {
    iree_status_t graph_status =
        iree_hal_streaming_graph_allocation_free(graph_allocation, stream);
    iree_hal_streaming_graph_allocation_release(graph_allocation);
    IREE_TRACE_ZONE_END(z0);
    return graph_status;
  }

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_memory_take_allocation_context(
      context, ptr, iree_hal_streaming_device_allocation_close_callback, &ptr,
      &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_hal_streaming_allocation_preparation_await_idle(&wrapper->preparation);

  if (owner_context != stream->context) {
    status = iree_hal_streaming_memory_restore_taken_allocation(owner_context,
                                                                wrapper);
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_join(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "stream-ordered free cannot be enqueued on a foreign "
                         "device context"),
        status);
  }

  // Non-pool allocations have no reusable backing and are retired as soon as
  // the stream reaches the free. Pool policy is sampled when the operation is
  // enqueued so terminal cleanup never needs to query a possibly reused
  // wrapper.
  const bool allow_opportunistic =
      !wrapper->allocation_pool ||
      iree_hal_streaming_mem_pool_query_reuse(wrapper->allocation_pool)
          .allow_opportunistic;

  iree_hal_streaming_deferred_device_free_t* free_op = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*free_op),
                                 (void**)&free_op);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  free_op->owner_context = owner_context;
  free_op->source_stream_id = stream->stream_id;
  // No event can safely reuse this allocation until the host call has been
  // queued and its stream timeline value is known.
  free_op->completion_value = UINT64_MAX;
  free_op->buffer = wrapper;
  free_op->next = NULL;
  free_op->is_registered = false;
  free_op->is_ready = false;
  free_op->is_terminal = false;
  free_op->is_reused = false;
  free_op->allow_opportunistic = allow_opportunistic;

  iree_hal_streaming_pending_free_terminal_t* terminal = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*terminal),
                                 (void**)&terminal);
  if (!iree_status_is_ok(status)) {
    free_op->buffer = NULL;
    iree_hal_streaming_pending_free_destroy(free_op);
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_resource_initialize(&iree_hal_streaming_pending_free_terminal_vtable,
                               &terminal->resource);
  terminal->free_op = free_op;
  uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = iree_hal_make_host_call_with_resource(
      iree_hal_streaming_deferred_device_free, free_op, &terminal->resource);
  status = iree_hal_streaming_queue_host_call(stream, call, args,
                                              IREE_HAL_HOST_CALL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);
    // The allocation is now unavailable to future streams. A later allocation
    // on this stream may adopt its backing before the callback completes.
    iree_hal_streaming_buffer_release_pool_allocation(wrapper);
    iree_hal_streaming_memory_account_device_free(wrapper);
    iree_slim_mutex_lock(&stream->mutex);
    free_op->completion_value = stream->pending_value;
    iree_slim_mutex_unlock(&stream->mutex);
    // Publish only after the callback has a stable completion value. The
    // caller's resource reference prevents terminal cleanup from racing this
    // publication even when the callback executes inline.
    iree_hal_streaming_pending_free_register(free_op);
    iree_hal_resource_release(&terminal->resource);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    free_op->buffer = NULL;
    iree_hal_resource_release(&terminal->resource);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef enum iree_hal_streaming_host_allocation_mode_e {
  IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_STAGING = 0,
  IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_COMMON_ADDRESS = 1,
  IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_FINE_GRAINED = 2,
} iree_hal_streaming_host_allocation_mode_t;

static iree_status_t iree_hal_streaming_memory_allocate_host_with_context_mode(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_host_allocation_mode_t mode,
    iree_device_size_t min_alignment,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_memory_kind_t memory_kind,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  const bool signal_memory = iree_any_bit_set(
      flags, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_SIGNAL_MEMORY);
  const bool fine_grained =
      mode == IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_FINE_GRAINED;
  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  const iree_host_size_t host_alignment =
      iree_max((iree_host_size_t)min_alignment,
               (iree_host_size_t)(signal_memory ? sizeof(uint64_t) : 4096));
  const iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

  iree_hal_pool_t* pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (signal_memory) {
    status = hrx_mem_pool_backing_retain_signal_pool(
        context->device_entry->hrx_device, &pool);
  } else if (mode != IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_STAGING) {
    status = hrx_mem_pool_backing_retain_common_host_source(
        context->device_entry->hrx_device, fine_grained,
        HRX_BUFFER_USAGE_DEFAULT, &pool);
  } else {
    status = hrx_mem_pool_backing_retain_allocation_source(
        context->device_entry->hrx_device, (hrx_memory_type_t)memory_type,
        HRX_BUFFER_USAGE_DEFAULT, &pool);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(allocation_size, host_alignment);
    status = iree_hal_pool_allocate_buffer(pool, request,
                                           iree_infinite_timeout(), &buffer);
  }

  iree_hal_streaming_buffer_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_wrap(
        context, buffer, (int)memory_type, memory_kind,
        /*imported_host_ptr=*/NULL,
        /*allocation_pool=*/NULL, context_ownership, &wrapper);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);

  if (iree_status_is_ok(status) && wrapper->host_ptr == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "host allocation did not export a host-visible pointer");
  }
  if (iree_status_is_ok(status)) {
    memset(wrapper->host_ptr, 0, allocation_size);
  }

  if (iree_status_is_ok(status)) {
    wrapper->imported_host_allocation = false;
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
  } else {
    if (wrapper) {
      hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);
      iree_hal_streaming_buffer_free(wrapper);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags, bool fine_grained,
    iree_hal_streaming_buffer_t** out_buffer) {
  const bool signal_memory = iree_any_bit_set(
      flags, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_SIGNAL_MEMORY);
  const iree_device_size_t min_alignment =
      signal_memory ? sizeof(uint64_t) : 64;
  return iree_hal_streaming_memory_allocate_host_with_context_mode(
      context, size, flags,
      fine_grained ? IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_FINE_GRAINED
                   : IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_COMMON_ADDRESS,
      min_alignment, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      IREE_HAL_STREAMING_MEMORY_KIND_HOST, out_buffer);
}

iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_allocate_host_with_context_mode(
      context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_STAGING, /*min_alignment=*/64,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      IREE_HAL_STREAMING_MEMORY_KIND_HOST, out_buffer);
}

static iree_status_t iree_hal_streaming_managed_metadata_allocate(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);

  const iree_host_size_t page_size = 4096;
  iree_host_size_t page_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
          buffer->size, 1, page_size - 1, &page_count))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed allocation size overflows page count");
  }
  page_count /= page_size;

  iree_host_size_t read_mostly_size = 0;
  iree_host_size_t location_size = 0;
  iree_host_size_t mask_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(page_count, sizeof(bool),
                                                &read_mostly_size) ||
                    !iree_host_size_checked_mul(page_count, sizeof(int32_t),
                                                &location_size) ||
                    !iree_host_size_checked_mul(page_count, sizeof(uint64_t),
                                                &mask_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed metadata size overflow");
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator, read_mostly_size,
                                   (void**)&buffer->managed_read_mostly_pages);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(context->host_allocator, location_size,
                              (void**)&buffer->managed_preferred_locations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        context->host_allocator, mask_size,
        (void**)&buffer->managed_accessed_by_device_masks);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(context->host_allocator, location_size,
                              (void**)&buffer->managed_last_prefetch_locations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator, location_size,
                                   (void**)&buffer->managed_coherency_modes);
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }

  memset(buffer->managed_read_mostly_pages, 0, read_mostly_size);
  memset(buffer->managed_accessed_by_device_masks, 0, mask_size);
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    buffer->managed_preferred_locations[i] = -2;
    buffer->managed_last_prefetch_locations[i] = -2;
    buffer->managed_coherency_modes[i] = 0;
  }
  buffer->managed_page_count = page_count;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_managed(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    unsigned int allocation_flags, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  iree_hal_streaming_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_memory_allocate_host_with_context_mode(
          context, allocation_size,
          IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
          IREE_HAL_STREAMING_HOST_ALLOCATION_MODE_FINE_GRAINED,
          /*min_alignment=*/4096, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
          IREE_HAL_STREAMING_MEMORY_KIND_MANAGED, &buffer));
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    buffer->host_register_flags =
        (iree_hal_streaming_host_register_flags_t)allocation_flags;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_managed_metadata_allocate(context, buffer);
  }
  if (!iree_status_is_ok(status)) {
    hrx_buffer_table_remove(&context->buffer_table, buffer->device_ptr);
    iree_hal_streaming_buffer_free(buffer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  *out_buffer = buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_repeat_pattern(void* dst,
                                              iree_device_size_t length,
                                              const void* pattern,
                                              iree_host_size_t pattern_length) {
  if (pattern_length == 1) {
    memset(dst, *(const uint8_t*)pattern, length);
    return;
  }
  uint8_t* dest = (uint8_t*)dst;
  for (iree_device_size_t i = 0; i < length; i += pattern_length) {
    iree_device_size_t copy_size = iree_min(pattern_length, length - i);
    memcpy(dest + i, pattern, copy_size);
  }
}

iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_host_close_params_t close_params = {
      .pointer = ptr,
      .kind = IREE_HAL_STREAMING_HOST_CLOSE_KIND_FREE,
  };
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_memory_take_allocation_context(
      context, (uint64_t)(uintptr_t)ptr,
      iree_hal_streaming_host_allocation_close_callback, &close_params,
      &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Host pointers may be hidden in native kernel arguments or other buffers.
  // Removing the entry blocks new operations. Wait its existing preparation
  // leases without holding a table lock, then establish the device-use
  // boundary.
  iree_hal_streaming_allocation_preparation_await_idle(&wrapper->preparation);
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);

  // Free wrapper and release its context ownership edge.
  iree_hal_streaming_buffer_free(wrapper);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_memory_register_host_with_context_mode(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_memory_kind_t memory_kind,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_external_storage_t external_buffer = {
      .storage_class = iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      .payload.host_pointer = ptr,
      .byte_length = (iree_device_size_t)size,
      .host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                          IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
      .maintenance_range =
          iree_hal_make_buffer_range(0, (iree_device_size_t)size),
  };
  iree_hal_pool_t* import_pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_import_source(
      context->device_entry->hrx_device, external_buffer.storage_class,
      &import_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_import_buffer(
        import_pool, &external_buffer,
        iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER), &buffer);
  }

  iree_hal_streaming_buffer_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_wrap(
        context, buffer,
        (int)(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        memory_kind, ptr, /*allocation_pool=*/NULL, context_ownership,
        &wrapper);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(import_pool);

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_managed_metadata_allocate(context, wrapper);
  }
  if (iree_status_is_ok(status)) {
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
  } else {
    if (wrapper) {
      hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);
      iree_hal_streaming_buffer_free(wrapper);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_register_host_with_context_mode(
      context, ptr, size, flags, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      IREE_HAL_STREAMING_MEMORY_KIND_HOST, out_buffer);
}

iree_status_t iree_hal_streaming_memory_import_managed(
    iree_hal_streaming_context_t* context, void* host_pointer,
    iree_host_size_t size, iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_register_host_with_context_mode(
      context, host_pointer, size,
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
      IREE_HAL_STREAMING_MEMORY_KIND_MANAGED, out_buffer);
}

iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_host_close_params_t close_params = {
      .pointer = ptr,
      .kind = IREE_HAL_STREAMING_HOST_CLOSE_KIND_UNREGISTER,
  };
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_memory_take_allocation_context(
      context, (uint64_t)(uintptr_t)ptr,
      iree_hal_streaming_host_allocation_close_callback, &close_params,
      &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Existing preparers finish recording before global synchronization starts;
  // neither wait runs under the registry or buffer-table lock.
  iree_hal_streaming_allocation_preparation_await_idle(&wrapper->preparation);
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_memory_restore_taken_allocation(
                    owner_context, wrapper));
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);

  // A captured graph may still retain this HAL import. Removing the public
  // registration prevents new lookups while the HAL refcount keeps the pages
  // pinned until the final graph or in-flight operation releases them.
  iree_hal_streaming_buffer_free(wrapper);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t* out_base, iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_base);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_base = 0;
  *out_size = 0;

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find(
      &context->buffer_table, ptr, NULL, NULL, (void**)&wrapper));
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (ptr >= wrapper->device_ptr && ptr - wrapper->device_ptr < wrapper->size) {
    *out_base = wrapper->device_ptr;
  } else if (wrapper->host_ptr) {
    *out_base = (iree_hal_streaming_deviceptr_t)wrapper->host_ptr;
  } else {
    *out_base = wrapper->device_ptr;
  }
  *out_size = wrapper->logical_size;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;

  iree_hal_streaming_retained_buffer_ref_t retained_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_host_allocation(
      context, ptr, &retained_ref);
  if (iree_status_is_ok(status)) {
    *out_flags = retained_ref.host_register_flags;
    iree_hal_streaming_retained_buffer_ref_deinitialize(&retained_ref);
  }

  return status;
}

iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_capture_fill_t capture = {
      .dst = dst,
      .length = length,
      .pattern = pattern,
      .pattern_length = pattern_length,
  };
  bool was_capturing = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_capture_try_record_node(
              stream, iree_hal_streaming_capture_record_fill, &capture,
              &was_capturing));
  if (was_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up the entire destination range. Managed allocations are process-wide
  // HIP pointers, so a device switch after allocation must still resolve them.
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_status_t lookup_status =
      iree_hal_streaming_memory_lookup_range(context, dst, length, &dst_ref);
  if (!iree_status_is_ok(lookup_status) &&
      iree_status_code(lookup_status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(lookup_status);
    lookup_status = iree_hal_streaming_memory_lookup_range_across_contexts(
        dst, length, &owner_context, &dst_ref);
    if (iree_status_is_ok(lookup_status)) {
      if (dst_ref.buffer->memory_kind !=
          IREE_HAL_STREAMING_MEMORY_KIND_MANAGED) {
        iree_hal_streaming_context_release(owner_context);
        owner_context = NULL;
        lookup_status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
      }
    }
  }
  if (!iree_status_is_ok(lookup_status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, lookup_status, "resolving `dst` buffer ref %p", (void*)dst);
  }

  if ((dst_ref.buffer->memory_kind == IREE_HAL_STREAMING_MEMORY_KIND_MANAGED) &&
      dst_ref.buffer->host_ptr) {
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    if (!iree_status_is_ok(sync_status)) {
      iree_hal_streaming_context_release(owner_context);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, sync_status);
    }
    uint8_t* dest = (uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset;
    iree_hal_streaming_repeat_pattern(dest, length, pattern, pattern_length);
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Fallback for imported host memory that could not be represented as a HAL
  // buffer.
  if (!dst_ref.buffer->buffer) {
    if (dst_ref.buffer->host_ptr) {
      uint8_t* dest = (uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset;
      iree_hal_streaming_repeat_pattern(dest, length, pattern, pattern_length);
      iree_hal_streaming_context_release(owner_context);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
    iree_hal_streaming_context_release(owner_context);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "no buffer available for memset"));
  }

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);

  iree_hal_buffer_ref_t target_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_fill_buffer(
        stream->command_buffer, target_ref, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(
        stream->command_buffer, IREE_HAL_BARRIER_FLAG_NONE);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  iree_hal_streaming_context_release(owner_context);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_enqueue_buffer_copy(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_ref_t src_ref,
    iree_hal_streaming_buffer_ref_t dst_ref, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(stream);
  if (!src_ref.buffer || !src_ref.buffer->buffer || !dst_ref.buffer ||
      !dst_ref.buffer->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy operands are not queue-compatible buffers");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_buffer_ref_validate_range(&src_ref, size));
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_buffer_ref_validate_range(&dst_ref, size));

  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = src_ref.buffer->buffer,
              .source_offset = src_ref.offset,
              .target_buffer = dst_ref.buffer->buffer,
              .target_offset = dst_ref.offset,
              .length = size,
              .flags = IREE_HAL_COPY_FLAG_NONE,
          },
  };
  return iree_hal_streaming_direct_transfer_enqueue(stream, 1, &operation);
}

iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_capture_copy_t capture = {
      .dst = dst,
      .src = src,
      .size = size,
  };
  bool was_capturing = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_capture_try_record_node(
              stream, iree_hal_streaming_capture_record_copy, &capture,
              &was_capturing));
  if (was_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  iree_status_t status =
      iree_hal_streaming_enqueue_buffer_copy(stream, src_ref, dst_ref, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_peer(
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst_context);
  IREE_ASSERT_ARGUMENT(src_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  bool can_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_device_can_access_peer(src_context->device_ordinal,
                                                    dst_context->device_ordinal,
                                                    &can_access));
  if (!can_access) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                             "P2P access not supported between devices %" PRIhsz
                             " and %" PRIhsz,
                             src_context->device_ordinal,
                             dst_context->device_ordinal));
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(dst_context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(src_context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  iree_status_t status =
      stream ? iree_hal_streaming_enqueue_buffer_copy(stream, src_ref, dst_ref,
                                                      size)
             : iree_hal_streaming_direct_transfer_d2d(
                   dst_context, src_ref.buffer->buffer, src_ref.offset,
                   dst_ref.buffer->buffer, dst_ref.offset, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_enqueue_host_update(
    iree_hal_streaming_stream_t* stream, const void* src,
    iree_hal_streaming_buffer_ref_t dst_ref, iree_device_size_t size) {
  if (!dst_ref.buffer->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host update destination is not a device buffer");
  }
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      .update =
          {
              .source_buffer = src,
              .target_buffer = dst_ref.buffer->buffer,
              .target_offset = dst_ref.offset,
              .length = size,
          },
  };
  return iree_hal_streaming_direct_transfer_enqueue(stream, 1, &operation);
}

static iree_status_t iree_hal_streaming_resolve_device_row(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t base,
    iree_device_size_t pitch, iree_device_size_t width, iree_host_size_t row,
    const char* endpoint_name, iree_hal_streaming_buffer_ref_t* out_ref) {
  iree_device_size_t row_offset = 0;
  iree_hal_streaming_deviceptr_t row_ptr = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul((iree_device_size_t)row,
                                                  pitch, &row_offset) ||
                    !iree_checked_add_u64(base, row_offset, &row_ptr))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s row address overflows", endpoint_name);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_memory_lookup_range(context, row_ptr, width, out_ref));
  if (!out_ref->buffer || !out_ref->buffer->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s row is not a queue-compatible buffer",
                            endpoint_name);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_host_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, const void* src, iree_device_size_t src_pitch,
    iree_device_size_t width, iree_host_size_t height,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (width > dst_pitch || width > src_pitch) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "copy width exceeds a row pitch"));
  }
  if (IREE_UNLIKELY((iree_host_size_t)(iree_device_size_t)height != height)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "H2D row count exceeds device address space"));
  }
  iree_host_size_t source_span = 0;
  if (IREE_UNLIKELY(
          src_pitch > IREE_HOST_SIZE_MAX || width > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(height - 1, src_pitch, &source_span) ||
          !iree_host_size_checked_add(source_span, width, &source_span) ||
          !iree_host_size_checked_add((iree_host_size_t)(uintptr_t)src,
                                      source_span, &source_span))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "H2D source row range overflows"));
  }

  iree_hal_streaming_buffer_ref_t first_source_ref = {0};
  const bool source_is_registered = iree_hal_streaming_memory_try_lookup(
      context, (iree_hal_streaming_deviceptr_t)src, &first_source_ref);
  iree_hal_transfer_operation_t* operations = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      context->host_allocator, height, sizeof(*operations),
      (void**)&operations);
  for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
       ++row) {
    iree_hal_streaming_buffer_ref_t destination_ref;
    status = iree_hal_streaming_resolve_device_row(
        context, dst, dst_pitch, width, row, "H2D destination",
        &destination_ref);
    if (iree_status_is_ok(status)) {
      operations[row] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy =
              {
                  .target_buffer = destination_ref.buffer->buffer,
                  .target_offset = destination_ref.offset,
                  .length = width,
              },
      };
    }
    if (iree_status_is_ok(status) && source_is_registered) {
      iree_hal_streaming_buffer_ref_t source_ref;
      status = iree_hal_streaming_resolve_device_row(
          context, (iree_hal_streaming_deviceptr_t)src, src_pitch, width, row,
          "H2D source", &source_ref);
      if (iree_status_is_ok(status) &&
          source_ref.buffer->memory_kind ==
              IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "H2D source row is not host memory");
      }
      if (iree_status_is_ok(status)) {
        operations[row].copy.source_buffer = source_ref.buffer->buffer;
        operations[row].copy.source_offset = source_ref.offset;
      }
    }
  }

  bool recorded_work = false;
  if (iree_status_is_ok(status) && source_is_registered) {
    status =
        iree_hal_streaming_direct_transfer_enqueue(stream, height, operations);
  } else if (iree_status_is_ok(status)) {
    // Unregistered uploads snapshot the caller's bytes while recording. They
    // cannot borrow the source after this function returns.
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_streaming_stream_begin_locked(stream);
    for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
         ++row) {
      iree_device_size_t remaining = width;
      iree_device_size_t chunk_offset = 0;
      while (remaining > 0 && iree_status_is_ok(status)) {
        const iree_device_size_t chunk_size = iree_min(
            remaining,
            (iree_device_size_t)IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
        const iree_hal_buffer_ref_t destination = iree_hal_make_buffer_ref(
            operations[row].copy.target_buffer,
            operations[row].copy.target_offset + chunk_offset, chunk_size);
        status = iree_hal_command_buffer_update_buffer(
            stream->command_buffer,
            (const uint8_t*)src + row * src_pitch + chunk_offset, 0,
            destination, IREE_HAL_UPDATE_FLAG_NONE);
        recorded_work |= iree_status_is_ok(status);
        chunk_offset += chunk_size;
        remaining -= chunk_size;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_command_buffer_barrier(
          stream->command_buffer, IREE_HAL_BARRIER_FLAG_NONE);
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  if (!iree_status_is_ok(status) && recorded_work) {
    status =
        iree_status_join(status, iree_hal_streaming_stream_synchronize(stream));
  }
  iree_allocator_free(context->host_allocator, operations);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Memory copy helper functions
//===----------------------------------------------------------------------===//
// Host coherence removes explicit host cache management, not device cache
// publication. Join producers before asking the native queue to make their
// writes visible and invalidate cached data before the CPU takes ownership.
// The empty transfer is a barrier and allocates no copy staging.
static iree_status_t iree_hal_streaming_synchronize_host_access(
    iree_hal_streaming_context_t* context) {
  IREE_RETURN_IF_ERROR(iree_hal_streaming_context_synchronize_all());
  return hrx_hal_queue_transfer_and_wait(
      context->queue,
      /*operation_count=*/0, NULL,
      iree_hal_streaming_context_wait_flags(context));
}

iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream) {
    iree_hal_streaming_capture_host_to_device_t capture = {
        .dst = dst,
        .src = src,
        .size = size,
    };
    bool was_capturing = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_try_record_node(
                stream, iree_hal_streaming_capture_record_host_to_device,
                &capture, &was_capturing));
    if (was_capturing) {
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Look up destination buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup(context, dst, &dst_ref);
  if (iree_status_is_ok(dst_status)) {
    dst_status = iree_hal_streaming_buffer_ref_validate_range(&dst_ref, size);
  }

  if (!iree_status_is_ok(dst_status)) {
    IREE_TRACE_ZONE_END(z0);
    return dst_status;
  }

  // A CPU mapping does not make a device allocation a host-copy destination.
  // On unified memory, CPU writes can make cache-resident GPU reads much
  // slower than reads after a native upload. Keep device uploads on the queue;
  // actual host and managed allocations retain their direct host copy.
  if (!stream && dst_ref.buffer->host_ptr &&
      dst_ref.buffer->memory_kind != IREE_HAL_STREAMING_MEMORY_KIND_DEVICE &&
      iree_all_bits_set((iree_hal_memory_type_t)dst_ref.buffer->memory_type,
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                            IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    iree_status_t status = iree_hal_streaming_synchronize_host_access(context);
    if (iree_status_is_ok(status)) {
      memcpy((uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset, src, size);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (stream) {
    iree_hal_streaming_buffer_ref_t src_ref;
    if (iree_hal_streaming_memory_try_lookup(
            context, (iree_hal_streaming_deviceptr_t)src, &src_ref)) {
      if (src_ref.buffer->memory_kind ==
          IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "host-to-device source is not host memory");
      }
      iree_status_t queue_status = iree_hal_streaming_enqueue_buffer_copy(
          stream, src_ref, dst_ref, size);
      IREE_TRACE_ZONE_END(z0);
      return queue_status;
    }
    if (size <= IREE_HAL_STREAMING_SMALL_TRANSFER_MAX_LENGTH) {
      iree_status_t queue_status =
          iree_hal_streaming_enqueue_host_update(stream, src, dst_ref, size);
      IREE_TRACE_ZONE_END(z0);
      return queue_status;
    }
    // A large pageable source stays borrowed through the blocking transaction.
    // The selected HAL queue owns transport staging; the frontend needs no
    // context-wide payload cache or lock across the copy and wait.
    const iree_hal_transfer_operation_t operation = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload =
            {
                .source = src,
                .target_buffer = dst_ref.buffer->buffer,
                .target_offset = dst_ref.offset,
                .length = size,
            },
    };
    iree_status_t status =
        iree_hal_streaming_direct_transfer_and_wait(stream, 1, &operation);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_status_t status = iree_hal_streaming_direct_transfer_h2d(
      context, src, dst_ref.buffer->buffer, dst_ref.offset, size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream) {
    iree_hal_streaming_capture_device_to_host_t capture = {
        .dst = dst,
        .src = src,
        .size = size,
    };
    bool was_capturing = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_try_record_node(
                stream, iree_hal_streaming_capture_record_device_to_host,
                &capture, &was_capturing));
    if (was_capturing) {
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Look up source buffer from device pointer.
  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup(context, src, &src_ref);
  if (iree_status_is_ok(src_status)) {
    src_status = iree_hal_streaming_buffer_ref_validate_range(&src_ref, size);
  }
  if (!iree_status_is_ok(src_status)) {
    IREE_TRACE_ZONE_END(z0);
    return src_status;
  }

  if (!stream && src_ref.buffer->host_ptr &&
      iree_all_bits_set((iree_hal_memory_type_t)src_ref.buffer->memory_type,
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                            IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    iree_status_t status = iree_hal_streaming_synchronize_host_access(context);
    if (iree_status_is_ok(status)) {
      memcpy(dst, (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset,
             size);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (stream) {
    iree_hal_streaming_buffer_ref_t dst_ref;
    if (iree_hal_streaming_memory_try_lookup(
            context, (iree_hal_streaming_deviceptr_t)dst, &dst_ref)) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_buffer_ref_validate_range(&dst_ref, size));
      if (dst_ref.buffer->memory_kind ==
          IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device-to-host destination is not host memory");
      }
    } else {
      const iree_hal_transfer_operation_t operation = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = src_ref.buffer->buffer,
                  .source_offset = src_ref.offset,
                  .target = dst,
                  .length = size,
              },
      };
      iree_status_t status =
          iree_hal_streaming_direct_transfer_and_wait(stream, 1, &operation);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    // Host pools require coherent access from every participating family. Copy
    // into the existing destination and publish device writes to the system
    // before completing the stream; no CPU staging or callout is required.
    iree_status_t status =
        iree_hal_streaming_enqueue_buffer_copy(stream, src_ref, dst_ref, size);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_direct_transfer_d2h(
              context, src_ref.buffer->buffer, src_ref.offset, dst, size));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_host_2d(
    iree_hal_streaming_context_t* context, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (width > dst_pitch || width > src_pitch) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "copy width exceeds a row pitch"));
  }
  if (IREE_UNLIKELY((iree_host_size_t)(iree_device_size_t)height != height)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "D2H row count exceeds device address space"));
  }
  iree_device_size_t destination_span = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul((iree_device_size_t)(height - 1),
                                        dst_pitch, &destination_span) ||
          !iree_device_size_checked_add(destination_span, width,
                                        &destination_span))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "D2H destination row address overflows"));
  }

  iree_hal_streaming_buffer_ref_t destination_ref = {0};
  const bool destination_is_registered = iree_hal_streaming_memory_try_lookup(
      context, (iree_hal_streaming_deviceptr_t)dst, &destination_ref);
  if (destination_is_registered) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_buffer_ref_validate_range(&destination_ref,
                                                         destination_span));
    if (destination_ref.buffer->memory_kind ==
        IREE_HAL_STREAMING_MEMORY_KIND_DEVICE) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "device-to-host destination is not host memory"));
    }
  }

  iree_hal_transfer_operation_t* operations = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      context->host_allocator, height, sizeof(*operations),
      (void**)&operations);
  for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
       ++row) {
    iree_hal_streaming_buffer_ref_t source_ref;
    status = iree_hal_streaming_resolve_device_row(
        context, src, src_pitch, width, row, "D2H source", &source_ref);
    if (iree_status_is_ok(status)) {
      if (destination_is_registered) {
        operations[row] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy =
                {
                    .source_buffer = source_ref.buffer->buffer,
                    .source_offset = source_ref.offset,
                    .target_buffer = destination_ref.buffer->buffer,
                    .target_offset = destination_ref.offset + row * dst_pitch,
                    .length = width,
                },
        };
      } else {
        operations[row] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = source_ref.buffer->buffer,
                    .source_offset = source_ref.offset,
                    .target = (uint8_t*)dst + row * dst_pitch,
                    .length = width,
                },
        };
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = destination_is_registered
                 ? iree_hal_streaming_direct_transfer_enqueue(stream, height,
                                                              operations)
                 : iree_hal_streaming_direct_transfer_and_wait(stream, height,
                                                               operations);
  }
  iree_allocator_free(context->host_allocator, operations);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!stream) {
    // Look up buffers from device pointers.
    iree_hal_streaming_buffer_ref_t dst_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
    iree_hal_streaming_buffer_ref_t src_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
        "resolving `src` buffer ref %p", (void*)src);

    // Transfer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_direct_transfer_d2d(
                context, src_ref.buffer->buffer, src_ref.offset,
                dst_ref.buffer->buffer, dst_ref.offset, size));
  } else {
    // Device-to-device copy is the same as memcpy with offset 0.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_memcpy(context, dst, src, size, stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (width > dst_pitch || width > src_pitch) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "copy width exceeds a row pitch"));
  }
  if (IREE_UNLIKELY((iree_host_size_t)(iree_device_size_t)height != height)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "D2D row count exceeds device address space"));
  }

  iree_hal_transfer_operation_t* operations = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      context->host_allocator, height, sizeof(*operations),
      (void**)&operations);
  for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
       ++row) {
    iree_hal_streaming_buffer_ref_t source_ref, destination_ref;
    status = iree_hal_streaming_resolve_device_row(
        context, src, src_pitch, width, row, "D2D source", &source_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_resolve_device_row(
          context, dst, dst_pitch, width, row, "D2D destination",
          &destination_ref);
    }
    if (iree_status_is_ok(status)) {
      operations[row] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy =
              {
                  .source_buffer = source_ref.buffer->buffer,
                  .source_offset = source_ref.offset,
                  .target_buffer = destination_ref.buffer->buffer,
                  .target_offset = destination_ref.offset,
                  .length = width,
              },
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_streaming_direct_transfer_enqueue(stream, height, operations);
  }
  iree_allocator_free(context->host_allocator, operations);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
