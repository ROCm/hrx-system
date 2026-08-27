// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/device.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/time.h"
#include "iree/hal/drivers/amdxdna/allocator.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/command_buffer.h"
#include "iree/hal/drivers/amdxdna/completion_queue.h"
#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"
#include "iree/hal/drivers/amdxdna/dispatch.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/semaphore.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/resource_set.h"

#define ARENA_BLOCK_SIZE (32 * 1024)

static const iree_hal_device_vtable_t iree_hal_amdxdna_device_vtable;

static bool iree_hal_amdxdna_semaphore_list_is_ready(
    iree_hal_semaphore_list_t semaphore_list);

static void iree_hal_amdxdna_device_initialize(
    iree_hal_amdxdna_device* device, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(device, 0, sizeof(*device));
  // Context and command caches are created after native caps are known; see
  // device_create.
  iree_atomic_store(&device->chain_max_slots, 0, iree_memory_order_relaxed);
  iree_atomic_store(&device->queue_epoch, 0, iree_memory_order_relaxed);

  iree_hal_resource_initialize(&iree_hal_amdxdna_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  device->power_mode_applied = false;

  iree_arena_block_pool_initialize(ARENA_BLOCK_SIZE, host_allocator,
                                   &device->block_pool);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdxdna_device_before_release_context(
    void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref) {
  iree_hal_amdxdna_device* device = (iree_hal_amdxdna_device*)user_data;
  iree_hal_amdxdna_native_queue_t* queue =
      iree_hal_amdxdna_native_context_ref_queue(context_ref);
  iree_hal_amdxdna_device_invalidate_command_caches_for_queue(device, queue);
}

static void iree_hal_amdxdna_device_deinitialize(
    iree_hal_amdxdna_device* device) {
  iree_hal_amdxdna_device_destroy_single_command_cache(device);
  iree_hal_amdxdna_device_destroy_chain_command_cache(device);
  iree_hal_amdxdna_device_context_cache_destroy(device->context_cache);
  device->context_cache = NULL;
}

static iree_status_t iree_hal_amdxdna_device_initialize_hal_resources(
    iree_hal_amdxdna_device* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_allocator_create(
      device->host_allocator, device->native_device,
      &device->device_allocator));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_async_queue_create(
      &device->block_pool, device->host_allocator, &device->async_queue));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_completion_queue_create(
      device->host_allocator, &device->completion_queue));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_initialize_async(
    iree_hal_amdxdna_device* device,
    const iree_hal_device_create_params_t* create_params) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  return iree_async_proactor_pool_get(device->proactor_pool, 0,
                                      &device->proactor);
}

// Creates the device's default slab-backed pool. Pattern mirrors
// iree_hal_sync_device_create_default_pool: CPU slab provider + proactor-
// backed notification + passthrough pool wrapping them. The slab provider
// owns CPU-side bookkeeping; amdxdna still allocates real BOs for explicit
// alloca, but the slab/notification pair is what the CTS Explicit*Pool* tests
// require to drive their pool epochs.
static iree_status_t iree_hal_amdxdna_device_create_default_pool(
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_slab_provider,
    iree_async_notification_t** out_notification, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  *out_slab_provider = NULL;
  *out_notification = NULL;
  *out_pool = NULL;

  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_cpu_slab_provider_create(host_allocator, &slab_provider));

  iree_async_notification_t* notification = NULL;
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t options = {0};
    status = iree_hal_passthrough_pool_create(
        options, slab_provider, notification, host_allocator, out_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_provider = slab_provider;
    *out_notification = notification;
    slab_provider = NULL;
    notification = NULL;
  }
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
  return status;
}

// Pool epoch query callback. The frontier_tracker is observed via the queue's
// per-axis epoch counter; when this returns true the test sees pool progress.
static bool iree_hal_amdxdna_device_query_pool_epoch(void* user_data,
                                                     iree_async_axis_t axis,
                                                     uint64_t epoch) {
  iree_async_frontier_tracker_t* tracker =
      (iree_async_frontier_tracker_t*)user_data;
  if (!tracker) return false;
  return iree_async_frontier_tracker_query_epoch(tracker, axis, epoch);
}

static iree_status_t iree_hal_amdxdna_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  (void)queue_affinity;
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (!device->default_slab_provider || !device->default_pool_notification) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna default pool not initialized");
  }
  out_backend->slab_provider = device->default_slab_provider;
  out_backend->notification = device->default_pool_notification;
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      iree_hal_amdxdna_device_query_pool_epoch, device->frontier_tracker};
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (!topology_info) {
    if (device->frontier_tracker) {
      iree_async_frontier_tracker_retire_axis(
          device->frontier_tracker, device->frontier_axis,
          iree_status_from_code(IREE_STATUS_CANCELLED));
      iree_hal_amdxdna_completion_queue_set_frontier(device->completion_queue,
                                                     NULL, 0, NULL);
      iree_hal_amdxdna_async_queue_set_frontier(device->async_queue, NULL, 0,
                                                NULL);
      iree_async_frontier_tracker_release(device->frontier_tracker);
      device->frontier_tracker = NULL;
      device->frontier_axis = 0;
    }
    memset(&device->topology_info, 0, sizeof(device->topology_info));
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* tracker = topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  if (!tracker) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "topology frontier tracker must be provided");
  }
  if (device->frontier_tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna topology info is already assigned; "
                            "clear it before assigning a new topology");
  }
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      tracker, axis, /*semaphore=*/NULL));
  device->topology_info = *topology_info;
  device->frontier_tracker = tracker;
  device->frontier_axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  iree_hal_amdxdna_async_queue_set_frontier(device->async_queue, tracker, axis,
                                            &device->queue_epoch);
  iree_hal_amdxdna_completion_queue_set_frontier(
      device->completion_queue, tracker, axis, &device->queue_epoch);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_load_executable(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;
  (void)target;

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_status_t status = iree_hal_amdxdna_native_executable_create(
      device->native_device, load_params, device->host_allocator,
      out_executable);
  if (iree_status_is_ok(status) &&
      device->native_caps.requires_executable_context_cache) {
    status = iree_hal_amdxdna_executable_preload_contexts(device,
                                                          *out_executable);
    if (!iree_status_is_ok(status)) {
      iree_hal_executable_release(*out_executable);
      *out_executable = NULL;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_command_buffer_create(
      device->device_allocator, &device->native_caps, mode, command_categories,
      queue_affinity, binding_capacity, &device->block_pool,
      device->host_allocator, out_command_buffer);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amdxdna_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  (void)semaphore;
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_amdxdna_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return iree_hal_amdxdna_semaphore_create(
      device->proactor, queue_affinity, initial_value, flags,
      device->host_allocator, out_semaphore);
}

typedef iree_status_t (*iree_hal_amdxdna_record_direct_command_fn_t)(
    iree_hal_command_buffer_t* command_buffer, void* user_data);

static iree_status_t iree_hal_amdxdna_record_direct_command(
    iree_hal_amdxdna_device* device,
    iree_hal_command_category_t command_categories,
    iree_hal_semaphore_list_t signal_list,
    iree_hal_amdxdna_completion_batch_t* completion_batch,
    iree_hal_amdxdna_record_direct_command_fn_t record_fn, void* user_data,
    bool* out_completion_submitted) {
  (void)signal_list;
  *out_completion_submitted = false;
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_command_buffer_mode_t mode = IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT;
  iree_status_t status = iree_ok_status();
  status = iree_hal_amdxdna_direct_command_buffer_create(
      device, mode, command_categories,
      /*binding_capacity=*/0, &device->block_pool, device->host_allocator,
      &command_buffer);
  if (iree_status_is_ok(status) && completion_batch) {
    iree_hal_amdxdna_direct_command_buffer_set_completion_batch(
        command_buffer, completion_batch);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = record_fn(command_buffer, user_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (completion_batch) {
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_completion_batch_record_error(completion_batch,
                                                     iree_status_clone(status));
      iree_status_ignore(status);
    }
    if (command_buffer) {
      iree_hal_amdxdna_completion_batch_retain_command_buffer(completion_batch,
                                                              command_buffer);
    }
    status = iree_hal_amdxdna_completion_batch_submit(completion_batch);
    *out_completion_submitted = true;
  }
  iree_hal_command_buffer_release(command_buffer);
  return status;
}

// Device queue entry-point contract:
//   * validate arguments and reject unsupported flags;
//   * retain resources and copy small submission metadata into op-owned memory;
//   * issue immediately only when all waits are already satisfied and the
//     operation can hand completion/signaling to a native completion batch;
//   * otherwise register wait timepoints and enqueue work to
//     async_queue/completion_queue.
//
// Queue entry points must not wait for native completion, run file-transfer
// loops, or perform large map/sync/memcpy operations on the caller thread.
// Ready dispatch submit remains asynchronous: it returns after driver
// submission, and native completion/signaling is owned by the completion queue.
static iree_status_t iree_hal_amdxdna_enqueue_signal_op(
    iree_hal_amdxdna_device* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_status_t status = iree_hal_amdxdna_async_queue_enqueue(
      device->async_queue, wait_semaphore_list, signal_semaphore_list,
      /*op_fn=*/NULL, /*cleanup_fn=*/NULL, /*user_data=*/NULL,
      /*retained_resources=*/NULL, /*retained_resource_count=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_validate_execute_flags(
    iree_hal_execute_flags_t flags) {
  const iree_hal_execute_flags_t supported_flags =
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported execute flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_create_binding_table_resource_set(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_resource_set_t** out_resource_set) {
  *out_resource_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_execute_flags(flags));
  if (!command_buffer || command_buffer->binding_count == 0 ||
      iree_any_bit_set(flags,
                       IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(binding_table.count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer requires at least %u "
                            "bindings but no binding table was provided",
                            command_buffer->binding_count);
  }
  if (IREE_UNLIKELY(binding_table.count < command_buffer->binding_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect command buffer requires at least %u bindings but only "
        "%" PRIhsz " were provided",
        command_buffer->binding_count, binding_table.count);
  }
  if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer binding table storage is "
                            "NULL for %" PRIhsz " bindings",
                            binding_table.count);
  }

  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(&device->block_pool, &resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert_strided(
        resource_set, command_buffer->binding_count, binding_table.bindings,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_set_freeze(resource_set);
    *out_resource_set = resource_set;
  } else {
    iree_hal_resource_set_free(resource_set);
  }
  return status;
}

typedef struct iree_hal_amdxdna_queue_execute_op_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_device* device;
  iree_allocator_t host_allocator;
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;
  iree_hal_resource_set_t* binding_resource_set;
  iree_hal_semaphore_list_t signal_list;
  iree_hal_amdxdna_completion_batch_t* completion_batch;
} iree_hal_amdxdna_queue_execute_op_t;

static iree_hal_amdxdna_queue_execute_op_t*
iree_hal_amdxdna_queue_execute_op_retain(
    iree_hal_amdxdna_queue_execute_op_t* op) {
  if (op) iree_atomic_ref_count_inc(&op->ref_count);
  return op;
}

static void iree_hal_amdxdna_queue_execute_op_destroy(
    iree_hal_amdxdna_queue_execute_op_t* op) {
  if (!op) return;
  iree_hal_semaphore_list_free(op->signal_list, op->host_allocator);
  iree_hal_amdxdna_completion_batch_destroy(op->completion_batch);
  iree_hal_resource_set_free(op->binding_resource_set);
  iree_hal_command_buffer_release(op->command_buffer);
  iree_allocator_free(op->host_allocator, op);
}

static void iree_hal_amdxdna_queue_execute_op_release(void* user_data) {
  iree_hal_amdxdna_queue_execute_op_t* op =
      (iree_hal_amdxdna_queue_execute_op_t*)user_data;
  if (!op) return;
  if (iree_atomic_ref_count_dec(&op->ref_count) == 1) {
    iree_hal_amdxdna_queue_execute_op_destroy(op);
  }
}

static iree_status_t iree_hal_amdxdna_validate_queue_execute_binding_table(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_binding_table_t* out_binding_table) {
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }
  if (IREE_UNLIKELY(command_buffer &&
                    !iree_hal_amdxdna_command_buffer_isa(command_buffer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_execute requires an amdxdna command buffer");
  }

  const iree_host_size_t binding_count =
      command_buffer ? command_buffer->binding_count : 0;
  if (binding_count == 0) {
    *out_binding_table = iree_hal_buffer_binding_table_empty();
  } else if (IREE_UNLIKELY(binding_table.count < binding_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "indirect command buffer requires at least %" PRIhsz
                            " bindings but only %" PRIhsz " were provided",
                            binding_count, binding_table.count);
  } else if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect command buffer binding table storage is "
                            "NULL for %" PRIhsz " bindings",
                            binding_table.count);
  } else {
    *out_binding_table = (iree_hal_buffer_binding_table_t){
        binding_count, binding_table.bindings};
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_queue_execute_op_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_semaphore_list_t signal_list,
    iree_hal_amdxdna_queue_execute_op_t** out_op) {
  *out_op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_execute_flags(flags));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_queue_execute_binding_table(
      command_buffer, binding_table, &binding_table));
  const iree_host_size_t binding_count = binding_table.count;

  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
          sizeof(iree_hal_amdxdna_queue_execute_op_t), binding_count,
          sizeof(iree_hal_buffer_binding_t), &total_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue_execute binding table copy is too large");
  }

  iree_hal_amdxdna_queue_execute_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(device->host_allocator, total_size, (void**)&op));
  memset(op, 0, total_size);
  iree_atomic_ref_count_init(&op->ref_count);
  op->device = device;
  op->host_allocator = device->host_allocator;
  op->command_buffer = command_buffer;
  op->signal_list = iree_hal_semaphore_list_empty();
  if (command_buffer) iree_hal_command_buffer_retain(command_buffer);

  iree_status_t status = iree_hal_amdxdna_create_binding_table_resource_set(
      device, command_buffer, binding_table, flags, &op->binding_resource_set);
  if (iree_status_is_ok(status) && binding_count > 0) {
    iree_hal_buffer_binding_t* bindings_copy =
        (iree_hal_buffer_binding_t*)((uint8_t*)op +
                                     sizeof(
                                         iree_hal_amdxdna_queue_execute_op_t));
    memcpy(bindings_copy, binding_table.bindings,
           binding_count * sizeof(*binding_table.bindings));
    op->binding_table.count = binding_count;
    op->binding_table.bindings = bindings_copy;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_list, device->host_allocator,
                                           &op->signal_list);
  }
  if (iree_status_is_ok(status) &&
      device->native_caps.submit_completion_is_deferred &&
      command_buffer) {
    status = iree_hal_amdxdna_completion_batch_create(
        device->completion_queue, signal_list, &op->completion_batch);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_execute_op_release(op);
    return status;
  }

  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_queue_execute_apply(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_semaphore_list_t signal_list,
    iree_hal_amdxdna_completion_batch_t* completion_batch,
    bool* out_completion_submitted) {
  *out_completion_submitted = false;
  if (!command_buffer) {
    if (completion_batch) {
      iree_status_t status =
          iree_hal_amdxdna_completion_batch_submit(completion_batch);
      *out_completion_submitted = true;
      return status;
    }
    return iree_ok_status();
  }
  iree_hal_command_buffer_t* direct_command_buffer = NULL;
  const iree_hal_command_buffer_mode_t mode =
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
      IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
  iree_status_t status = iree_ok_status();
  (void)signal_list;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_direct_command_buffer_create(
        device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
        /*binding_capacity=*/0, &device->block_pool, device->host_allocator,
        &direct_command_buffer);
  }
  if (iree_status_is_ok(status) && completion_batch) {
    iree_hal_amdxdna_direct_command_buffer_set_completion_batch(
        direct_command_buffer, completion_batch);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_command_buffer_apply(
        command_buffer, direct_command_buffer, binding_table);
  }
  if (completion_batch) {
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_completion_batch_record_error(completion_batch,
                                                     iree_status_clone(status));
      iree_status_ignore(status);
    }
    if (direct_command_buffer) {
      iree_hal_amdxdna_completion_batch_retain_command_buffer(
          completion_batch, direct_command_buffer);
    }
    status = iree_hal_amdxdna_completion_batch_submit(completion_batch);
    *out_completion_submitted = true;
  }
  iree_hal_command_buffer_release(direct_command_buffer);
  return status;
}

static iree_status_t iree_hal_amdxdna_queue_execute_submit_error_batch(
    iree_hal_amdxdna_queue_execute_op_t* op, iree_status_t status) {
  if (!op || !op->completion_batch) return status;
  iree_hal_amdxdna_completion_batch_record_error(op->completion_batch, status);
  iree_status_t submit_status =
      iree_hal_amdxdna_completion_batch_submit(op->completion_batch);
  op->completion_batch = NULL;
  return submit_status;
}

static iree_status_t iree_hal_amdxdna_queue_execute_wait_failure_fn(
    void* user_data, iree_status_t status) {
  return iree_hal_amdxdna_queue_execute_submit_error_batch(
      (iree_hal_amdxdna_queue_execute_op_t*)user_data, status);
}

static iree_status_t iree_hal_amdxdna_queue_execute_op_fn(void* user_data) {
  iree_hal_amdxdna_queue_execute_op_t* op =
      (iree_hal_amdxdna_queue_execute_op_t*)user_data;
  iree_hal_amdxdna_queue_execute_op_retain(op);
  bool completion_cleanup_owned = false;
  iree_status_t status = iree_ok_status();
  if (op->completion_batch) {
    status = iree_hal_amdxdna_completion_batch_add_cleanup(
        op->completion_batch, iree_hal_amdxdna_queue_execute_op_release, op);
    if (iree_status_is_ok(status)) {
      completion_cleanup_owned = true;
    }
  }
  bool completion_submitted = false;
  if (!iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_queue_execute_submit_error_batch(op, status);
    completion_submitted = true;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_queue_execute_apply(
        op->device, op->command_buffer, op->binding_table, op->signal_list,
        op->completion_batch, &completion_submitted);
    if (completion_submitted) op->completion_batch = NULL;
  }
  if (!completion_cleanup_owned) {
    iree_hal_amdxdna_queue_execute_op_release(op);
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_queue_execute_issue_ready(
    iree_hal_amdxdna_queue_execute_op_t* op) {
  iree_status_t status = iree_hal_amdxdna_queue_execute_op_fn(op);
  if (iree_status_is_deferred(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  iree_hal_amdxdna_queue_execute_op_release(op);
  return status;
}

static iree_status_t iree_hal_amdxdna_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)queue_affinity;

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  iree_hal_amdxdna_queue_execute_op_t* op = NULL;
  bool signal_list_handled = false;
  iree_status_t status = iree_hal_amdxdna_validate_execute_flags(flags);
  iree_hal_buffer_binding_table_t validated_binding_table =
      iree_hal_buffer_binding_table_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_validate_queue_execute_binding_table(
        command_buffer, binding_table, &validated_binding_table);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_queue_execute_op_create(
        device, command_buffer, validated_binding_table, flags,
        signal_semaphore_list, &op);
    if (iree_status_is_ok(status)) {
      bool completion_batch_owns_signal_list = false;
      iree_hal_semaphore_list_t async_signal_list = signal_semaphore_list;
      iree_hal_amdxdna_async_op_failure_fn_t failure_fn = NULL;
      if (op->completion_batch) {
        iree_hal_amdxdna_completion_batch_publish_signals(op->completion_batch);
        async_signal_list = iree_hal_semaphore_list_empty();
        failure_fn = iree_hal_amdxdna_queue_execute_wait_failure_fn;
        completion_batch_owns_signal_list = true;
        signal_list_handled = true;
      }
      bool direct_ready_submit = false;
      if (op->completion_batch) {
        direct_ready_submit =
            iree_hal_amdxdna_semaphore_list_is_ready(wait_semaphore_list);
      }
      if (direct_ready_submit) {
        status = iree_hal_amdxdna_queue_execute_issue_ready(op);
        op = NULL;
      } else {
        status = iree_hal_amdxdna_async_queue_enqueue_with_failure_handler(
            device->async_queue, wait_semaphore_list, async_signal_list,
            iree_hal_amdxdna_queue_execute_op_fn, failure_fn,
            iree_hal_amdxdna_queue_execute_op_release, op,
            /*retained_resources=*/NULL, /*retained_resource_count=*/0);
        if (iree_status_is_ok(status)) {
          op = NULL;  // async queue owns op.
        } else if (op->completion_batch) {
          iree_status_t enqueue_status = status;
          iree_status_ignore(iree_hal_amdxdna_queue_execute_submit_error_batch(
              op, iree_status_clone(status)));
          status = enqueue_status;
        }
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_queue_execute_op_release(op);
        op = NULL;
        if (!completion_batch_owns_signal_list) {
          iree_hal_semaphore_list_fail(signal_semaphore_list,
                                       iree_status_clone(status));
          signal_list_handled = true;
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_execute_op_release(op);
    if (!signal_list_handled) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_device_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_t* old_allocator = device->device_allocator;
  device->device_allocator = new_allocator;
  iree_hal_allocator_release(old_allocator);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_device_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_amdxdna_device_trim(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_amdxdna_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  return device->device_spec;
}

static iree_status_t iree_hal_amdxdna_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (iree_any_bit_set(requested_flags,
                       IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->device_spec, out_observation));
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_amdxdna_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_amdxdna_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  (void)base_device;
  (void)queue_affinity;
  (void)params;
  (void)out_channel;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_amdxdna_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  (void)flags;
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      /*proactor=*/NULL, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_amdxdna_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)pool;
  (void)flags;

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  // Allocate the buffer synchronously: the caller needs it returned now even
  // though the wait_semaphore_list may not yet be satisfied. This is the
  // standard "async placement" contract: the buffer object exists, but
  // signal_semaphore_list will not fire until the waits are resolved (which
  // is what the deferred async_queue task below ensures).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(device->device_allocator, params,
                                             allocation_size, out_buffer));
  iree_hal_amdxdna_buffer_mark_allocated(
      iree_hal_buffer_allocated_buffer(*out_buffer));

  // Tag the buffer with async placement so callers can route dealloca back to
  // this device/queue.
  //
  // The CTS BufferMetadata test asserts placement.queue_affinity has exactly
  // one bit set (so callers can use it as a target queue id directly). The
  // caller passes IREE_HAL_QUEUE_AFFINITY_ANY which is all-bits, so we
  // collapse to a single bit here. Today amdxdna reports a single physical
  // queue; the lowest-set-bit pick is therefore both deterministic and
  // correct. A multi-queue redesign would need a real scheduling decision
  // here instead of a fixed pick.
  // Once amdxdna reports more than one HAL queue this should become a
  // queue-affinity -> ordinal lookup/scheduling decision.
  iree_hal_queue_affinity_t selected =
      iree_hal_queue_affinity_is_empty(queue_affinity)
          ? 1
          : 1ull << iree_hal_queue_affinity_find_first_set(queue_affinity);
  (*out_buffer)->placement = (iree_hal_buffer_placement_t){
      base_device, selected, IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS, 0};

  // Defer signaling through the async queue even when waits are already ready:
  // device_queue_* entry points must not complete work inline.
  iree_status_t completion_status = iree_hal_amdxdna_enqueue_signal_op(
      device, wait_semaphore_list, signal_semaphore_list);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_buffer_release(*out_buffer);
    *out_buffer = NULL;
    IREE_TRACE_ZONE_END(z0);
    return completion_status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// op_fn that marks a buffer as deallocated. Runs on the async_queue worker
// when the op fires successfully. user_data is a borrowed buffer pointer;
// the buffer's lifetime is owned by the queue's retained_resources mechanism
// (separately retained by queue_dealloca below) so we do NOT release here.
// If the op is cancelled, the queue still releases the retained buffer; we
// just don't get to mark it deallocated, which is fine because the buffer
// is on its way out anyway.
static iree_status_t iree_hal_amdxdna_dealloca_op_fn(void* user_data) {
  iree_hal_buffer_t* buffer = (iree_hal_buffer_t*)user_data;
  iree_hal_amdxdna_buffer_mark_deallocated(buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_validate_live_buffer(
    iree_hal_buffer_t* buffer, const char* role) {
  if (iree_hal_amdxdna_buffer_is_deallocated(
          iree_hal_buffer_allocated_buffer(buffer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is a deallocated amdxdna buffer", role);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_validate_transfer_target(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const char* role) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_validate_live_buffer(target_buffer, role));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  return iree_hal_buffer_validate_range(target_buffer, target_offset, length);
}

static iree_status_t iree_hal_amdxdna_validate_transfer_source(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_device_size_t length, const char* role) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_validate_live_buffer(source_buffer, role));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  return iree_hal_buffer_validate_range(source_buffer, source_offset, length);
}

static iree_status_t iree_hal_amdxdna_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  (void)queue_affinity;
  (void)flags;
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  // Retain the buffer and hand it to the queue as a retained_resource. The
  // queue will release it on every termination path (success, cancellation,
  // wait failure), so the +1 retain is never leaked even if op_fn doesn't
  // run. user_data is the same buffer pointer (borrowed) for op_fn to use
  // while the retain is held.
  iree_hal_buffer_retain(buffer);
  iree_hal_resource_t* retained[] = {
      (iree_hal_resource_t*)buffer,
  };
  iree_status_t status = iree_hal_amdxdna_async_queue_enqueue(
      device->async_queue, wait_semaphore_list, signal_semaphore_list,
      iree_hal_amdxdna_dealloca_op_fn, /*cleanup_fn=*/NULL,
      /*user_data=*/buffer,
      /*retained_resources=*/retained,
      /*retained_resource_count=*/IREE_ARRAYSIZE(retained));
  if (!iree_status_is_ok(status)) {
    // Enqueue failed; caller still owns the +1 retain.
    iree_hal_buffer_release(buffer);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_unsupported_native_transfer(
    const char* operation) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna %s requires native blit support; host-emulated map/sync/memcpy "
      "transfers are not available on device queues",
      operation);
}

static iree_status_t iree_hal_amdxdna_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_target(
      target_buffer, target_offset, length, "queue_fill target"));
  if (length == 0) {
    iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
        base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }
  if (IREE_UNLIKELY(!pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_fill pattern is NULL");
  }
  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_fill pattern length must be 1, 2, or 4 "
                            "bytes (got %" PRIhsz ")",
                            pattern_length);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_FILL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill flags: 0x%" PRIx64, flags);
  }
  return iree_hal_amdxdna_unsupported_native_transfer("queue_fill");
}

static iree_status_t iree_hal_amdxdna_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_target(
      target_buffer, target_offset, length, "queue_update target"));
  if (length == 0) {
    iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
        base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }
  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_update source buffer is NULL");
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_UPDATE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported update flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(length > (iree_device_size_t)IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue_update length is too large");
  }
  iree_host_size_t host_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(source_offset > IREE_HOST_SIZE_MAX - host_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue_update source range overflows host address space "
        "(source_offset=%" PRIhsz ", length=%" PRIhsz ")",
        source_offset, host_length);
  }
  return iree_hal_amdxdna_unsupported_native_transfer("queue_update");
}

static iree_status_t iree_hal_amdxdna_validate_copy(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_source(
      source_buffer, source_offset, length, "queue_copy source"));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_target(
      target_buffer, target_offset, length, "queue_copy target"));
  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(iree_hal_buffer_test_overlap(source_buffer, source_offset,
                                                 length, target_buffer,
                                                 target_offset, length) !=
                    IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_copy(
      source_buffer, source_offset, target_buffer, target_offset, length,
      flags));

  if (length == 0) {
    iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
        base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }
  return iree_hal_amdxdna_unsupported_native_transfer("queue_copy");
}

static iree_status_t iree_hal_amdxdna_validate_file_range(
    iree_hal_file_t* file, const char* operation, uint64_t file_offset,
    iree_device_size_t length, iree_device_size_t* out_device_offset) {
  if (IREE_UNLIKELY(file_offset > UINT64_MAX - (uint64_t)length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s range [%" PRIu64 ", +%" PRIdsz
                            ") overflows file address space",
                            operation, file_offset, length);
  }
  const uint64_t file_length = iree_hal_file_length(file);
  const uint64_t file_end = file_offset + (uint64_t)length;
  if (file_length > 0 && file_end > file_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s range [%" PRIu64 ", %" PRIu64
                            ") exceeds file length %" PRIu64,
                            operation, file_offset, file_end, file_length);
  }
  if (out_device_offset) {
    if (IREE_UNLIKELY(file_offset > IREE_DEVICE_SIZE_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "%s storage-buffer offset %" PRIu64
                              " exceeds device address space",
                              operation, file_offset);
    }
    *out_device_offset = (iree_device_size_t)file_offset;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_target(
      target_buffer, target_offset, length, "queue_read target"));
  if (IREE_UNLIKELY(flags != IREE_HAL_READ_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported read flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (length == 0) {
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }

  iree_device_size_t source_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_file_range(
      source_file, "read", source_offset, length, &source_device_offset));
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (storage_buffer &&
      iree_all_bits_set(iree_hal_buffer_memory_type(storage_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_hal_amdxdna_device_queue_copy(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        storage_buffer, source_device_offset, target_buffer, target_offset,
        length, IREE_HAL_COPY_FLAG_NONE);
  }
  return iree_hal_amdxdna_unsupported_native_transfer("queue_read");
}

static iree_status_t iree_hal_amdxdna_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_transfer_source(
      source_buffer, source_offset, length, "queue_write source"));
  if (IREE_UNLIKELY(flags != IREE_HAL_WRITE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported write flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  if (length == 0) {
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }

  iree_device_size_t target_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_file_range(
      target_file, "write", target_offset, length, &target_device_offset));
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  if (storage_buffer &&
      iree_all_bits_set(iree_hal_buffer_memory_type(storage_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_hal_amdxdna_device_queue_copy(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_buffer, source_offset, storage_buffer, target_device_offset,
        length, IREE_HAL_COPY_FLAG_NONE);
  }
  return iree_hal_amdxdna_unsupported_native_transfer("queue_write");
}

typedef struct iree_hal_amdxdna_queue_host_call_op_t {
  iree_hal_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
  iree_allocator_t host_allocator;
  iree_hal_host_call_t call;
  uint64_t args[4];
  iree_hal_host_call_flags_t flags;
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_amdxdna_queue_host_call_op_t;

static iree_status_t iree_hal_amdxdna_queue_host_call_op_fn(void* user_data) {
  iree_hal_amdxdna_queue_host_call_op_t* op =
      (iree_hal_amdxdna_queue_host_call_op_t*)user_data;
  const bool is_nonblocking =
      iree_any_bit_set(op->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    iree_status_t status = iree_hal_semaphore_list_signal(
        op->signal_semaphore_list, /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) {
      return status;
    }
  }

  iree_hal_host_call_context_t context = {
      .device = op->device,
      .queue_affinity = op->queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : op->signal_semaphore_list,
  };
  iree_status_t status = op->call.fn(op->call.user_data, op->args, &context);
  if (is_nonblocking) {
    iree_status_ignore(status);
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  return status;
}

static void iree_hal_amdxdna_queue_host_call_op_cleanup(void* user_data) {
  iree_hal_amdxdna_queue_host_call_op_t* op =
      (iree_hal_amdxdna_queue_host_call_op_t*)user_data;
  if (!op) return;
  iree_hal_semaphore_list_free(op->signal_semaphore_list, op->host_allocator);
  iree_allocator_free(op->host_allocator, op);
}

static iree_status_t iree_hal_amdxdna_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  if (IREE_UNLIKELY(!call.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_host_call callback is NULL");
  }
  const iree_hal_host_call_flags_t supported_flags =
      IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
      IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE | IREE_HAL_HOST_CALL_FLAG_RELAXED;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported host call flags: 0x%" PRIx64, flags);
  }
  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  iree_hal_amdxdna_queue_host_call_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(device->host_allocator, sizeof(*op), (void**)&op));
  memset(op, 0, sizeof(*op));
  op->device = base_device;
  op->queue_affinity = queue_affinity;
  op->host_allocator = device->host_allocator;
  op->call = call;
  memcpy(op->args, args, sizeof(op->args));
  op->flags = flags;
  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, device->host_allocator,
      &op->signal_semaphore_list);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_async_queue_enqueue(
        device->async_queue, wait_semaphore_list, signal_semaphore_list,
        iree_hal_amdxdna_queue_host_call_op_fn,
        iree_hal_amdxdna_queue_host_call_op_cleanup, op,
        /*retained_resources=*/NULL, /*retained_resource_count=*/0);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_host_call_op_cleanup(op);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

typedef struct iree_hal_amdxdna_queue_dispatch_op_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_device* device;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_dispatch_plan_t plan;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
  iree_hal_semaphore_list_t signal_list;
  iree_hal_amdxdna_completion_batch_t* completion_batch;
  iree_hal_resource_t** retained_resources;
  iree_host_size_t retained_resource_count;
} iree_hal_amdxdna_queue_dispatch_op_t;

static iree_hal_amdxdna_queue_dispatch_op_t*
iree_hal_amdxdna_queue_dispatch_op_retain(
    iree_hal_amdxdna_queue_dispatch_op_t* op) {
  if (op) iree_atomic_ref_count_inc(&op->ref_count);
  return op;
}

static void iree_hal_amdxdna_queue_dispatch_op_cleanup(void* user_data);

static iree_status_t iree_hal_amdxdna_queue_dispatch_submit_error_batch(
    iree_hal_amdxdna_queue_dispatch_op_t* op, iree_status_t status) {
  if (!op || !op->completion_batch) return status;
  iree_hal_amdxdna_completion_batch_record_error(op->completion_batch, status);
  iree_status_t submit_status =
      iree_hal_amdxdna_completion_batch_submit(op->completion_batch);
  op->completion_batch = NULL;
  return submit_status;
}

static iree_status_t iree_hal_amdxdna_queue_dispatch_wait_failure_fn(
    void* user_data, iree_status_t status) {
  return iree_hal_amdxdna_queue_dispatch_submit_error_batch(
      (iree_hal_amdxdna_queue_dispatch_op_t*)user_data, status);
}

static iree_status_t iree_hal_amdxdna_apply_dispatch_plan(
    iree_hal_command_buffer_t* command_buffer, void* user_data) {
  iree_hal_amdxdna_queue_dispatch_op_t* op =
      (iree_hal_amdxdna_queue_dispatch_op_t*)user_data;
  return iree_hal_amdxdna_direct_command_buffer_dispatch_plan(
      command_buffer, &op->plan, op->constants, op->bindings, op->flags);
}

static bool iree_hal_amdxdna_semaphore_list_is_ready(
    iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(semaphore_list.semaphores[i], &current_value);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    if (current_value < semaphore_list.payload_values[i]) return false;
  }
  return true;
}

static iree_status_t iree_hal_amdxdna_queue_dispatch_op_fn(void* user_data) {
  iree_hal_amdxdna_queue_dispatch_op_t* op =
      (iree_hal_amdxdna_queue_dispatch_op_t*)user_data;
  iree_hal_amdxdna_queue_dispatch_op_retain(op);
  bool completion_cleanup_owned = false;
  iree_status_t status = iree_ok_status();
  if (op->completion_batch) {
    status = iree_hal_amdxdna_completion_batch_add_cleanup(
        op->completion_batch, iree_hal_amdxdna_queue_dispatch_op_cleanup, op);
    if (iree_status_is_ok(status)) {
      completion_cleanup_owned = true;
    }
  }
  bool completion_submitted = false;
  if (!iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_queue_dispatch_submit_error_batch(op, status);
    completion_submitted = true;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_record_direct_command(
        op->device, IREE_HAL_COMMAND_CATEGORY_DISPATCH, op->signal_list,
        op->completion_batch, iree_hal_amdxdna_apply_dispatch_plan, op,
        &completion_submitted);
    if (completion_submitted) op->completion_batch = NULL;
  }
  if (!completion_cleanup_owned) {
    iree_hal_amdxdna_queue_dispatch_op_cleanup(op);
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_queue_dispatch_issue_ready(
    iree_hal_amdxdna_queue_dispatch_op_t* op) {
  iree_status_t status = iree_hal_amdxdna_queue_dispatch_op_fn(op);
  const bool deferred = iree_status_is_deferred(status);
  if (deferred) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  iree_hal_amdxdna_queue_dispatch_op_cleanup(op);
  return status;
}

static void iree_hal_amdxdna_queue_dispatch_op_destroy(
    iree_hal_amdxdna_queue_dispatch_op_t* op) {
  if (!op) return;
  for (iree_host_size_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  iree_allocator_free(op->host_allocator, op->retained_resources);
  iree_hal_semaphore_list_free(op->signal_list, op->host_allocator);
  iree_hal_amdxdna_completion_batch_destroy(op->completion_batch);
  iree_allocator_free(op->host_allocator, op);
}

static void iree_hal_amdxdna_queue_dispatch_op_cleanup(void* user_data) {
  iree_hal_amdxdna_queue_dispatch_op_t* op =
      (iree_hal_amdxdna_queue_dispatch_op_t*)user_data;
  if (!op) return;
  if (iree_atomic_ref_count_dec(&op->ref_count) == 1) {
    iree_hal_amdxdna_queue_dispatch_op_destroy(op);
  }
}

static iree_status_t iree_hal_amdxdna_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  (void)queue_affinity;
  if (IREE_UNLIKELY(!executable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_dispatch executable is NULL");
  }
  if (IREE_UNLIKELY(bindings.count != 0 && !bindings.values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_dispatch bindings are NULL");
  }
  if (IREE_UNLIKELY(constants.data_length != 0 && !constants.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_dispatch constants are NULL");
  }

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_dispatch(&config, flags));
  if (iree_hal_amdxdna_dispatch_is_zero_workgroups(&config, flags)) {
    return iree_hal_amdxdna_enqueue_signal_op(device, wait_semaphore_list,
                                              signal_semaphore_list);
  }

  iree_host_size_t op_size = sizeof(iree_hal_amdxdna_queue_dispatch_op_t);
  iree_host_size_t constants_offset = 0;
  if (constants.data_length > 0) {
    op_size = iree_host_align(op_size, iree_max_align_t);
    constants_offset = op_size;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
            op_size, 1, constants.data_length, &op_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "queue_dispatch constants copy is too large");
    }
  }
  iree_host_size_t bindings_offset = 0;
  if (bindings.count > 0) {
    op_size = iree_host_align(op_size, iree_max_align_t);
    bindings_offset = op_size;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
            op_size, bindings.count, sizeof(*bindings.values), &op_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "queue_dispatch binding list is too large");
    }
  }

  iree_hal_amdxdna_queue_dispatch_op_t* op = NULL;
  bool signal_list_handled = false;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(device->host_allocator, op_size, (void**)&op));
  memset(op, 0, op_size);
  iree_atomic_ref_count_init(&op->ref_count);
  op->device = device;
  op->host_allocator = device->host_allocator;
  op->flags = flags;
  op->signal_list = iree_hal_semaphore_list_empty();
  iree_status_t status = iree_hal_amdxdna_dispatch_plan_initialize(
      &device->native_caps, executable, function, &op->plan);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(
        &signal_semaphore_list, device->host_allocator, &op->signal_list);
  }
  if (iree_status_is_ok(status) &&
      device->native_caps.submit_completion_is_deferred) {
    status = iree_hal_amdxdna_completion_batch_create(
        device->completion_queue, signal_semaphore_list, &op->completion_batch);
  }
  if (constants.data_length > 0) {
    uint8_t* constants_copy = (uint8_t*)op + constants_offset;
    memcpy(constants_copy, constants.data, constants.data_length);
    op->constants =
        iree_make_const_byte_span(constants_copy, constants.data_length);
  } else {
    op->constants = iree_const_byte_span_empty();
  }
  if (bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy =
        (iree_hal_buffer_ref_t*)((uint8_t*)op + bindings_offset);
    memcpy(bindings_copy, bindings.values,
           bindings.count * sizeof(*bindings.values));
    op->bindings.count = bindings.count;
    op->bindings.values = bindings_copy;
  } else {
    op->bindings = iree_hal_buffer_ref_list_empty();
  }

  iree_host_size_t retained_count = 1;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (bindings.values[i].buffer) ++retained_count;
  }
  iree_hal_resource_t** retained_resources = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(device->host_allocator, retained_count,
                                         sizeof(*retained_resources),
                                         (void**)&retained_resources);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t retained_index = 0;
    iree_hal_executable_retain(executable);
    retained_resources[retained_index++] = (iree_hal_resource_t*)executable;
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_buffer_t* buffer = bindings.values[i].buffer;
      if (buffer) {
        iree_hal_buffer_retain(buffer);
        retained_resources[retained_index++] = (iree_hal_resource_t*)buffer;
      }
    }
    op->retained_resources = retained_resources;
    op->retained_resource_count = retained_count;
    retained_resources = NULL;
    iree_hal_semaphore_list_t async_signal_list = signal_semaphore_list;
    iree_hal_amdxdna_async_op_failure_fn_t failure_fn = NULL;
    if (op->completion_batch) {
      iree_hal_amdxdna_completion_batch_publish_signals(op->completion_batch);
      async_signal_list = iree_hal_semaphore_list_empty();
      failure_fn = iree_hal_amdxdna_queue_dispatch_wait_failure_fn;
      signal_list_handled = true;
    }
    bool direct_ready_submit = false;
    if (op->completion_batch) {
      direct_ready_submit =
          iree_hal_amdxdna_semaphore_list_is_ready(wait_semaphore_list);
    }
    if (direct_ready_submit) {
      status = iree_hal_amdxdna_queue_dispatch_issue_ready(op);
      op = NULL;  // ready submit released queue-entry ownership.
    } else {
      status = iree_hal_amdxdna_async_queue_enqueue_with_failure_handler(
          device->async_queue, wait_semaphore_list, async_signal_list,
          iree_hal_amdxdna_queue_dispatch_op_fn, failure_fn,
          iree_hal_amdxdna_queue_dispatch_op_cleanup, op,
          /*retained_resources=*/NULL, /*retained_resource_count=*/0);
      if (iree_status_is_ok(status)) {
        op = NULL;  // async queue owns op.
      } else if (op->completion_batch) {
        iree_status_t enqueue_status = status;
        iree_hal_amdxdna_completion_batch_record_error(
            op->completion_batch, iree_status_clone(status));
        iree_status_ignore(
            iree_hal_amdxdna_completion_batch_submit(op->completion_batch));
        op->completion_batch = NULL;
        status = enqueue_status;
      }
    }
  }
  iree_allocator_free(device->host_allocator, retained_resources);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_queue_dispatch_op_cleanup(op);
    if (!signal_list_handled) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_device_queue_atomic_wait(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "amdxdna devices do not support atomic waits");
}

static iree_status_t iree_hal_amdxdna_device_queue_atomic_store(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "amdxdna devices do not support atomic stores");
}

static iree_status_t iree_hal_amdxdna_device_queue_atomic_rmw(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna devices do not support atomic read-modify-write");
}

static iree_status_t iree_hal_amdxdna_device_queue_timestamp(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)flags;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "amdxdna device-side timestamps not implemented");
}

static iree_status_t iree_hal_amdxdna_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  (void)base_device;
  (void)queue_affinity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_status_t iree_hal_amdxdna_device_profiling_flush(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_status_t iree_hal_amdxdna_device_profiling_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue profiling is not implemented");
}

static iree_string_view_t iree_hal_amdxdna_device_id(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->identifier;
}

static void iree_hal_amdxdna_device_destroy(iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  // Drain and shut down worker queues before tearing down the block pool (the
  // queues' pending ops live in arenas backed by the block pool).
  if (device->async_queue) {
    iree_hal_amdxdna_async_queue_destroy(device->async_queue);
    device->async_queue = NULL;
  }
  if (device->completion_queue) {
    iree_hal_amdxdna_completion_queue_destroy(device->completion_queue);
    device->completion_queue = NULL;
  }
  // Retire the frontier axis (if assigned) before releasing the tracker.
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->frontier_axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
  }
  if (device->default_pool) {
    iree_hal_pool_release(device->default_pool);
    device->default_pool = NULL;
  }
  if (device->default_slab_provider) {
    iree_hal_slab_provider_release(device->default_slab_provider);
    device->default_slab_provider = NULL;
  }
  if (device->default_pool_notification) {
    iree_async_notification_release(device->default_pool_notification);
    device->default_pool_notification = NULL;
  }
  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_device_spec_release(device->device_spec);
  if (device->proactor_pool) {
    iree_async_proactor_pool_release(device->proactor_pool);
  }
  if (device->power_mode_applied && device->native_device) {
    (void)iree_hal_amdxdna_native_device_c_set_power_mode(
        device->native_device, IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT);
    device->power_mode_applied = false;
  }
  // Drop cached native command BOs before tearing down the native device they
  // were allocated from. The destructor also calls these helpers, so this is
  // safe if an earlier failure path already cleared the caches.
  iree_hal_amdxdna_device_destroy_single_command_cache(device);
  iree_hal_amdxdna_device_destroy_chain_command_cache(device);
  // Drop cached native context refs before the native device they reference is
  // torn down. Per the IREE HAL lifetime contract, executables are released
  // before their device, so by the time we get here the cache holds the last
  // refs and clear() runs the native context destructors cleanly.
  // Lock defensively against the contract being violated (zero cost
  // uncontended) and so a future audit can't ask "is this clear racy?"
  iree_hal_amdxdna_device_context_cache_clear(device->context_cache);
  iree_hal_amdxdna_native_device_c_destroy(device->native_device);
  // Drop device-owned caches before freeing the storage. Save host_allocator
  // first; accessing `device->` after deinitialize is intentionally avoided.
  iree_allocator_t host_allocator = device->host_allocator;
  iree_hal_amdxdna_device_deinitialize(device);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
};

iree_hal_amdxdna_device* iree_hal_amdxdna_device_cast(
    iree_hal_device_t* base_device) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);
}

static iree_allocator_t iree_hal_amdxdna_device_host_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amdxdna_device_device_allocator(
    iree_hal_device_t* base_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_device* device = IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_device, iree_hal_amdxdna_device_vtable, iree_hal_amdxdna_device);

  IREE_TRACE_ZONE_END(z0);
  return device->device_allocator;
}

void iree_hal_amdxdna_device_options_initialize(
    struct iree_hal_amdxdna_device_params* out_options) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_options, 0, sizeof(*out_options));
  IREE_TRACE_ZONE_END(z0);
}

// Advertises the amdxdna executable family/key used by libhrx load_data.
// Must stay aligned with HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY/KEY.
static iree_status_t iree_hal_amdxdna_device_spec_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_device_spec_t** out_spec) {
  const iree_hal_physical_device_spec_t physical_device = {
      .identity =
          {
              .display_name = identifier,
              .backend_path = identifier,
          },
      .partition_count = 1,
      .physical_device_affinity = 1ull,
  };
  const iree_hal_device_identity_spec_t identity = {
      .logical_device_id = identifier,
      .display_name = identifier,
      .driver_id = IREE_SV("amdxdna"),
      .backend_id = IREE_SV("amdxdna"),
      .physical_device_count = 1,
      .physical_devices = &physical_device,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  const iree_hal_executable_target_t executable_target = {
      .family = IREE_SV("amdxdna"),
      .target_key = IREE_SV("amdxdna"),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      .priority = 100,
      .physical_device_affinity = 1ull,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_add_executable_target(
        &builder, &executable_target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

iree_status_t iree_hal_amdxdna_device_create(
    iree_string_view_t identifier,
    const struct iree_hal_amdxdna_device_params* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  IREE_TRACE_ZONE_BEGIN(z0);

  struct iree_hal_amdxdna_device_params resolved_options;
  iree_byte_span_t resolved_device_path_storage = iree_byte_span_empty();
  iree_hal_amdxdna_native_c_power_mode_t resolved_power_mode =
      IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT;
  bool should_set_power_mode = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_device_c_resolve_options(
              options, host_allocator, &resolved_options,
              &resolved_device_path_storage, &resolved_power_mode,
              &should_set_power_mode));

  iree_hal_amdxdna_device* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, resolved_device_path_storage.data);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_amdxdna_device_initialize(device, host_allocator);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + total_size - identifier.size);

  status = iree_hal_amdxdna_device_spec_create(
      identifier, device->host_allocator, &device->device_spec);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_device_c_create(
        &resolved_options, device->host_allocator, &device->native_device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_device_c_query_caps(device->native_device,
                                                         &device->native_caps);
  }
  if (iree_status_is_ok(status)) {
    // Now that caps are known, size the context cache to the device's
    // hardware-context budget (adaptive per architecture; falls back to a
    // conservative default when the budget is unknown).
    const iree_hal_amdxdna_context_cache_ops_t context_cache_ops = {
        .before_release_context =
            iree_hal_amdxdna_device_before_release_context,
    };
    device->context_cache =
        iree_hal_amdxdna_device_context_cache_create_with_ops(
            device->host_allocator, device->native_caps.max_hardware_contexts,
            &context_cache_ops, device);
    if (!device->context_cache) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to allocate amdxdna context cache");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_single_command_cache_create(
        device->host_allocator, &device->single_command_cache);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_chain_command_cache_create(
        device->host_allocator,
        device->native_caps.max_cached_chain_child_commands,
        &device->chain_command_cache);
  }
  if (iree_status_is_ok(status) && should_set_power_mode) {
    status = iree_hal_amdxdna_native_device_c_set_power_mode(
        device->native_device, resolved_power_mode);
    if (iree_status_is_ok(status) &&
        resolved_power_mode != IREE_HAL_AMDXDNA_NATIVE_C_POWER_MODE_DEFAULT) {
      device->power_mode_applied = true;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_initialize_hal_resources(device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_initialize_async(device, create_params);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_device_create_default_pool(
        device->proactor, device->host_allocator,
        &device->default_slab_provider, &device->default_pool_notification,
        &device->default_pool);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_device_destroy((iree_hal_device_t*)device);
    iree_allocator_free(host_allocator, resolved_device_path_storage.data);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_allocator_free(host_allocator, resolved_device_path_storage.data);
  *out_device = (iree_hal_device_t*)device;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_amdxdna_device_vtable = {
    .destroy = iree_hal_amdxdna_device_destroy,
    .id = iree_hal_amdxdna_device_id,
    .host_allocator = iree_hal_amdxdna_device_host_allocator,
    .device_allocator = iree_hal_amdxdna_device_device_allocator,
    .replace_device_allocator =
        iree_hal_amdxdna_device_replace_device_allocator,
    .replace_channel_provider =
        iree_hal_amdxdna_device_replace_channel_provider,
    .trim = iree_hal_amdxdna_device_trim,
    .device_spec = iree_hal_amdxdna_device_spec,
    .sample_observation = iree_hal_amdxdna_device_sample_observation,
    .topology_info = iree_hal_amdxdna_device_topology_info,
    .refine_topology_edge = iree_hal_amdxdna_device_refine_topology_edge,
    .assign_topology_info = iree_hal_amdxdna_device_assign_topology_info,
    .create_channel = iree_hal_amdxdna_device_create_channel,
    .create_command_buffer = iree_hal_amdxdna_device_create_command_buffer,
    .load_executable = iree_hal_amdxdna_device_load_executable,
    .import_file = iree_hal_amdxdna_device_import_file,
    .create_semaphore = iree_hal_amdxdna_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_amdxdna_device_query_semaphore_compatibility,
    // Pool-backed queue alloca is not supported. Returning UNIMPLEMENTED here
    // makes pool-using tests fail gracefully instead of NULL-deref crashing
    // through the vtable.
    .query_queue_pool_backend =
        iree_hal_amdxdna_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_amdxdna_device_queue_alloca,
    .queue_dealloca = iree_hal_amdxdna_device_queue_dealloca,
    .queue_fill = iree_hal_amdxdna_device_queue_fill,
    .queue_update = iree_hal_amdxdna_device_queue_update,
    .queue_copy = iree_hal_amdxdna_device_queue_copy,
    .queue_read = iree_hal_amdxdna_device_queue_read,
    .queue_write = iree_hal_amdxdna_device_queue_write,
    .queue_host_call = iree_hal_amdxdna_device_queue_host_call,
    .queue_dispatch = iree_hal_amdxdna_device_queue_dispatch,
    .queue_execute = iree_hal_amdxdna_device_queue_execute,
    .queue_atomic_wait = iree_hal_amdxdna_device_queue_atomic_wait,
    .queue_atomic_store = iree_hal_amdxdna_device_queue_atomic_store,
    .queue_atomic_rmw = iree_hal_amdxdna_device_queue_atomic_rmw,
    .queue_timestamp = iree_hal_amdxdna_device_queue_timestamp,
    .queue_flush = iree_hal_amdxdna_device_queue_flush,
    // Returning UNIMPLEMENTED here signals callers (e.g. the CTS profiling
    // tests) to skip profiling-dependent assertions instead of treating the
    // begin as a successful no-op and then failing because no events were
    // recorded.
    .profiling_begin = iree_hal_amdxdna_device_profiling_begin,
    .profiling_flush = iree_hal_amdxdna_device_profiling_flush,
    .profiling_end = iree_hal_amdxdna_device_profiling_end,
    .external_capture_begin = NULL,
    .external_capture_end = NULL,
};
