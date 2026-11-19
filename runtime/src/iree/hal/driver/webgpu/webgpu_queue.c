// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_queue.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/operation.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/js/proactor.h"
#include "iree/hal/driver/webgpu/webgpu.h"
#include "iree/hal/driver/webgpu/webgpu_buffer.h"
#include "iree/hal/driver/webgpu/webgpu_command_buffer.h"
#include "iree/hal/driver/webgpu/webgpu_executable.h"
#include "iree/hal/driver/webgpu/webgpu_file_transfer.h"
#include "iree/hal/driver/webgpu/webgpu_imports.h"
#include "iree/hal/driver/webgpu/webgpu_semaphore.h"

// WebGPU owns resource visibility within its memory domain. GPU submissions,
// host mapping/unmapping, and file staging already establish the required
// boundaries before signaling completion. Explicit queue barriers therefore
// require no extra native command, and empty lists cannot elide that native
// synchronization. This backend rejects ranged actions that WebGPU cannot
// encode without promoting their resource scope.

static iree_status_t iree_hal_webgpu_queue_validate_barriers(
    const iree_hal_queue_barriers_t* barriers) {
  if (!barriers) {
    return iree_ok_status();
  }
  const iree_hal_barrier_list_t* lists[2] = {barriers->before, barriers->after};
  for (iree_host_size_t boundary = 0; boundary < 2; ++boundary) {
    if (!lists[boundary]) {
      continue;
    }
    for (iree_host_size_t i = 0; i < lists[boundary]->count; ++i) {
      if (iree_hal_memory_effects_requires_resources(
              lists[boundary]->values[i].effects)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "WebGPU queues do not support ranged memory transitions");
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_queue_t
//===----------------------------------------------------------------------===//

static void iree_hal_webgpu_queue_destroy(iree_hal_queue_t* base_queue) {
  iree_hal_webgpu_queue_deinitialize((iree_hal_webgpu_queue_t*)base_queue);
}

static iree_status_t iree_hal_webgpu_queue_barrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_validate_barriers(barriers));
  (void)flags;
  return iree_hal_webgpu_queue_submit_execute(
      (iree_hal_webgpu_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, /*command_buffer=*/NULL,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_webgpu_queue_execute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  return iree_hal_webgpu_queue_submit_execute(
      (iree_hal_webgpu_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, command_buffer, binding_table, flags);
}

static iree_status_t iree_hal_webgpu_queue_host_call(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_hal_webgpu_queue_submit_host_call(
      (iree_hal_webgpu_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_webgpu_queue_query_dispatch_concurrency(
    iree_hal_queue_t* base_queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  (void)base_queue;
  (void)executable;
  (void)function;
  (void)params;
  (void)flags;
  (void)out_concurrency;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU queue dispatch concurrency queries are not implemented");
}

static iree_status_t iree_hal_webgpu_queue_dispatch(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_validate_barriers(barriers));
  return iree_hal_webgpu_queue_submit_dispatch(
      (iree_hal_webgpu_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, executable, function, config, constants, bindings,
      flags);
}

static iree_status_t iree_hal_webgpu_queue_atomic_wait(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  (void)barriers;
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "WebGPU queues do not support atomic waits");
}

static iree_status_t iree_hal_webgpu_queue_atomic_store(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  (void)barriers;
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "WebGPU queues do not support atomic stores");
}

static iree_status_t iree_hal_webgpu_queue_atomic_rmw(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  (void)barriers;
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)params;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU queues do not support atomic read-modify-write");
}

static iree_status_t iree_hal_webgpu_queue_timestamp(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags) {
  (void)barriers;
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)flags;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "WebGPU device-side timestamps not implemented");
}

static iree_status_t iree_hal_webgpu_queue_flush(iree_hal_queue_t* base_queue) {
  (void)base_queue;
  // WebGPU's queue.submit() is not buffered, so there is no pending work to
  // flush.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_queue_alloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)pool;
  (void)request_count;
  (void)requests;
  (void)out_buffers;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU exact-queue allocation transactions are not implemented");
}

static iree_status_t iree_hal_webgpu_queue_dealloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers) {
  (void)base_queue;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)buffer_count;
  (void)buffers;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "WebGPU exact-queue deallocation transactions are not implemented");
}

static iree_status_t iree_hal_webgpu_queue_transfer(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_validate_barriers(barriers));
  return iree_hal_webgpu_queue_submit_transfer(
      (iree_hal_webgpu_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, operation_count, operations);
}

static iree_status_t iree_hal_webgpu_queue_read(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  (void)flags;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_validate_barriers(barriers));
  return iree_hal_webgpu_queue_submit_file_transfer(
      (iree_hal_webgpu_queue_t*)base, waits, signals, files, buffers, transfers,
      IREE_HAL_MEMORY_ACCESS_READ);
}

static iree_status_t iree_hal_webgpu_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_binding_table_t buffers,
    iree_hal_file_binding_table_t files,
    const iree_hal_file_transfer_list_t* transfers,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  (void)flags;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_validate_barriers(barriers));
  return iree_hal_webgpu_queue_submit_file_transfer(
      (iree_hal_webgpu_queue_t*)base, waits, signals, files, buffers, transfers,
      IREE_HAL_MEMORY_ACCESS_WRITE);
}

static const iree_hal_queue_vtable_t iree_hal_webgpu_queue_vtable = {
    .destroy = iree_hal_webgpu_queue_destroy,
    .barrier = iree_hal_webgpu_queue_barrier,
    .execute = iree_hal_webgpu_queue_execute,
    .host_call = iree_hal_webgpu_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_webgpu_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_webgpu_queue_dispatch,
    .atomic_wait = iree_hal_webgpu_queue_atomic_wait,
    .atomic_store = iree_hal_webgpu_queue_atomic_store,
    .atomic_rmw = iree_hal_webgpu_queue_atomic_rmw,
    .timestamp = iree_hal_webgpu_queue_timestamp,
    .flush = iree_hal_webgpu_queue_flush,
    .alloca = iree_hal_webgpu_queue_alloca,
    .dealloca = iree_hal_webgpu_queue_dealloca,
    .transfer = iree_hal_webgpu_queue_transfer,
    .read = iree_hal_webgpu_queue_read,
    .write = iree_hal_webgpu_queue_write,
};

iree_status_t iree_hal_webgpu_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_handle_t queue_handle,
    const iree_hal_webgpu_builtins_t* builtins, iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_allocator_t host_allocator, iree_hal_webgpu_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);

  out_queue->device_handle = device_handle;
  out_queue->queue_handle = queue_handle;
  out_queue->builtins = builtins;
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->axis = axis;
  iree_atomic_store(&out_queue->epoch, 0, iree_memory_order_relaxed);
  out_queue->host_allocator = host_allocator;

  // Initialize the shared block pool for instruction stream builders.
  iree_arena_block_pool_initialize(/*total_block_size=*/65536, host_allocator,
                                   &out_queue->block_pool);

  // Initialize the scratch builder for queue operations (dynamic_count = 0,
  // all slots are static).
  iree_status_t status = iree_hal_webgpu_builder_initialize(
      &out_queue->block_pool, /*dynamic_count=*/0, host_allocator,
      &out_queue->scratch_builder);
  if (!iree_status_is_ok(status)) {
    iree_arena_block_pool_deinitialize(&out_queue->block_pool);
  } else {
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_initialize(queue_family, &queue_params,
                              &iree_hal_webgpu_queue_vtable, &out_queue->base);
  }
  return status;
}

void iree_hal_webgpu_queue_deinitialize(iree_hal_webgpu_queue_t* queue) {
  iree_hal_webgpu_builder_deinitialize(&queue->scratch_builder);
  iree_arena_block_pool_deinitialize(&queue->block_pool);
}

static iree_hal_semaphore_list_t iree_hal_webgpu_semaphore_list_at_offsets(
    void* base, iree_host_size_t count, iree_host_size_t semaphores_offset,
    iree_host_size_t payload_values_offset) {
  return (iree_hal_semaphore_list_t){
      .count = count,
      .semaphores =
          (iree_hal_semaphore_t**)((uint8_t*)base + semaphores_offset),
      .payload_values = (uint64_t*)((uint8_t*)base + payload_values_offset),
  };
}

static void iree_hal_webgpu_semaphore_list_clone_into(
    iree_hal_semaphore_list_t source_list,
    iree_hal_semaphore_list_t target_list) {
  IREE_ASSERT(source_list.count == target_list.count);
  for (iree_host_size_t i = 0; i < source_list.count; ++i) {
    target_list.semaphores[i] = source_list.semaphores[i];
    iree_hal_semaphore_retain(target_list.semaphores[i]);
    target_list.payload_values[i] = source_list.payload_values[i];
  }
}

//===----------------------------------------------------------------------===//
// Epoch tracking
//===----------------------------------------------------------------------===//

// Atomically increments the epoch counter and returns the new value. Called at
// submit time to establish causal ordering. The frontier tracker is NOT
// advanced here — that happens at completion time (or immediately after for
// fast-path ops where submit IS completion).
static uint64_t iree_hal_webgpu_queue_reserve_epoch(
    iree_hal_webgpu_queue_t* queue) {
  return (uint64_t)iree_atomic_fetch_add(&queue->epoch, 1,
                                         iree_memory_order_acq_rel) +
         1;
}

// Advances the frontier tracker to the given epoch. Called at completion time
// (after GPU work finishes via onSubmittedWorkDone) or immediately after
// reserve_epoch for fast-path ops where submit IS completion.
static void iree_hal_webgpu_queue_advance_tracker(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch) {
  if (!queue->frontier_tracker) {
    return;
  }
  iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                      epoch);
}

//===----------------------------------------------------------------------===//
// Frontier construction
//===----------------------------------------------------------------------===//

// Builds a single-entry frontier on caller-provided stack storage for the
// queue's axis at the given epoch. Returns NULL if the queue has no frontier
// tracker (frontiers disabled). The returned pointer is valid for the lifetime
// of |out_frontier|.
static const iree_async_frontier_t* iree_hal_webgpu_queue_build_frontier(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch,
    iree_async_single_frontier_t* out_frontier) {
  if (!queue->frontier_tracker) {
    return NULL;
  }
  iree_async_single_frontier_initialize(out_frontier, queue->axis, epoch);
  return iree_async_single_frontier_as_const_frontier(out_frontier);
}

//===----------------------------------------------------------------------===//
// Scratch builder execution
//===----------------------------------------------------------------------===//

// Executes the scratch builder's instruction stream via the one-shot path
// (execute_instructions). Builds the binding table from the builder's slot map
// and passes it with the builtins descriptor to the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_scratch_builder(
    iree_hal_webgpu_queue_t* queue) {
  iree_hal_webgpu_builder_t* builder = &queue->scratch_builder;

  uint32_t total_slots = iree_hal_webgpu_builder_total_slot_count(builder);
  uint32_t static_count = iree_hal_webgpu_builder_static_slot_count(builder);

  // Build the flat binding table in wire format.
  // Use stack allocation for small tables (covers the vast majority of queue
  // ops which touch 1-3 buffers).
  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* entries = inline_entries;
  if (total_slots > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, total_slots,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t), (void**)&entries));
  }

  const iree_hal_webgpu_builder_slot_entry_t* slot_entries =
      iree_hal_webgpu_builder_static_slot_entries(builder);
  for (uint32_t i = 0; i < static_count; ++i) {
    entries[slot_entries[i].slot].gpu_buffer_handle =
        slot_entries[i].gpu_buffer_handle;
    entries[slot_entries[i].slot].base_offset = 0;
    entries[slot_entries[i].slot].reserved = 0;
    entries[slot_entries[i].slot].length =
        iree_hal_buffer_byte_length(slot_entries[i].buffer);
  }

  iree_hal_webgpu_isa_builtins_descriptor_t builtins_descriptor;
  iree_hal_webgpu_builtins_get_descriptor(queue->builtins,
                                          &builtins_descriptor);

  uint32_t result = iree_hal_webgpu_import_execute_instructions(
      queue->device_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)iree_hal_webgpu_builder_block_table(builder),
      iree_hal_webgpu_builder_block_count(builder),
      iree_hal_webgpu_builder_block_word_capacity(builder),
      iree_hal_webgpu_builder_last_block_word_count(builder),
      (uint32_t)(uintptr_t)entries, total_slots,
      (uint32_t)(uintptr_t)&builtins_descriptor);

  if (entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, entries);
  }

  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_instructions failed with code %u",
                            result);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Signal completion (proactor-driven onSubmittedWorkDone)
//===----------------------------------------------------------------------===//

// Holds an async operation and a snapshot of the signal semaphore list for
// deferred signaling after onSubmittedWorkDone completes. Allocated as a single
// block: [struct | semaphore_ptrs[] | payload_values[]].
typedef struct iree_hal_webgpu_signal_completion_t {
  iree_async_operation_t operation;
  iree_hal_semaphore_list_t signal_semaphore_list;
  iree_hal_webgpu_queue_t* queue;  // Borrowed, outlives completion.
  uint64_t epoch;
  iree_allocator_t allocator;
} iree_hal_webgpu_signal_completion_t;

// Completion callback invoked by the proactor when onSubmittedWorkDone fires.
// Builds a frontier for the queue's axis/epoch, signals (or fails) all
// semaphores with it, advances the frontier tracker, releases references,
// and frees the completion struct.
static void iree_hal_webgpu_signal_completion_fn(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_signal_completion_t* completion =
      (iree_hal_webgpu_signal_completion_t*)base_operation;

  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(
            completion->queue, completion->epoch, &frontier_storage);
    iree_status_ignore(iree_hal_semaphore_list_signal(
        completion->signal_semaphore_list, frontier));
  } else {
    iree_hal_semaphore_list_fail(completion->signal_semaphore_list, status);
  }

  iree_hal_webgpu_queue_advance_tracker(completion->queue, completion->epoch);
  iree_hal_semaphore_list_release(completion->signal_semaphore_list);
  iree_allocator_free(completion->allocator, completion);
}

// Registers an onSubmittedWorkDone callback that signals all semaphores in
// |signal_semaphore_list| when the currently submitted GPU work completes.
// The |epoch| is the pre-reserved epoch from reserve_epoch; the completion
// callback builds a frontier from it and advances the tracker.
// If the semaphore list is empty, advances the tracker immediately and returns.
static iree_status_t iree_hal_webgpu_queue_register_signal_completion(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) {
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
    return iree_ok_status();
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t semaphores_offset = 0;
  iree_host_size_t payload_values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_signal_completion_t), &total_size,
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &payload_values_offset)));
  iree_hal_webgpu_signal_completion_t* completion = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&completion));

  completion->signal_semaphore_list = iree_hal_webgpu_semaphore_list_at_offsets(
      completion, signal_semaphore_list.count, semaphores_offset,
      payload_values_offset);
  completion->queue = queue;
  completion->epoch = epoch;
  completion->allocator = queue->host_allocator;
  iree_hal_webgpu_semaphore_list_clone_into(signal_semaphore_list,
                                            completion->signal_semaphore_list);

  iree_async_operation_initialize(
      &completion->operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_signal_completion_fn,
      /*user_data=*/NULL);

  // Submit to the JS proactor's token table to get a completion token.
  uint32_t token = UINT32_MAX;
  iree_status_t status = iree_async_proactor_js_submit_external(
      queue->proactor, &completion->operation, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_release(completion->signal_semaphore_list);
    iree_allocator_free(queue->host_allocator, completion);
    return status;
  }

  // Register onSubmittedWorkDone with the JS bridge. When the GPU finishes
  // the submitted work, JS writes {token, status_code} to the completion
  // ring. The proactor's drain path dispatches our callback.
  iree_hal_webgpu_import_queue_on_submitted_work_done(queue->queue_handle,
                                                      token);
  return iree_ok_status();
}

// Returns true if all wait semaphores can be elided because this queue has
// already submitted (but possibly not completed) signals that will satisfy
// every wait. Uses the submitted signal provenance fields on each semaphore
// to check if GPU FIFO ordering guarantees the wait will be satisfied.
static bool iree_hal_webgpu_queue_can_elide_waits(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list) {
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_webgpu_semaphore_has_submitted_signal(
            wait_semaphore_list.semaphores[i], queue->axis,
            wait_semaphore_list.payload_values[i])) {
      return false;
    }
  }
  return true;
}

// Marks all semaphores in the signal list as having a pending submitted signal
// from this queue. Called after GPU work is submitted in fast-path queue
// operations so that subsequent same-queue ops can use FIFO wait elision.
static void iree_hal_webgpu_queue_mark_signals_submitted(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_webgpu_semaphore_mark_submitted_signal(
        signal_semaphore_list.semaphores[i], queue->axis,
        signal_semaphore_list.payload_values[i]);
  }
}

// Finalizes and executes the scratch builder, then registers signal completion.
// Used by fast-path queue operations (fill, update, copy, dispatch) that build
// a single instruction into the scratch builder and submit it synchronously.
//
// Reserves an epoch and registers an onSubmittedWorkDone completion that
// carries the epoch's frontier. The completion callback signals semaphores
// with the frontier and advances the frontier tracker. On success, also marks
// the signal semaphores with submitted provenance for FIFO wait elision.
//
// |status| is the accumulated status from the caller's builder commands.
// If already failed, skips finalize/execute and fails the signal semaphores.
static iree_status_t iree_hal_webgpu_queue_submit_scratch_and_signal(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_builder_finalize(&queue->scratch_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_queue_execute_scratch_builder(queue);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_queue_register_signal_completion(
        queue, epoch, signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_mark_signals_submitted(queue, signal_semaphore_list);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
  }
  return status;
}

// Forward declarations for queue_execute helpers used in the wait callback.
static iree_status_t iree_hal_webgpu_queue_execute_recording(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_handle_t recording_handle,
    iree_hal_buffer_binding_table_t binding_table);
static iree_status_t iree_hal_webgpu_queue_execute_one_shot(
    iree_hal_webgpu_queue_t* queue, iree_hal_command_buffer_t* command_buffer);

//===----------------------------------------------------------------------===//
// Unified async queue operation state
//===----------------------------------------------------------------------===//
//
// All async queue operations (except host_call, which has unique DEFERRED
// semantics) share a common state structure. The wait completion callback
// dispatches on op_type to perform the operation's work, then either signals
// inline (CPU-only ops) or transfers slab ownership to an embedded signal
// completion (GPU-submit ops) for GPU completion tracking.
//
// CPU-only (barrier):
//   wait → CPU work → signal inline → advance epoch → free slab.
//
// GPU-submit (transfer, dispatch, execute):
//   wait → GPU work → register native completion → [slab lives on]
//   → native completion → release resources → signal → advance epoch → free
//   slab.
// Downloads include the asynchronous map and host scatter before completion.

typedef enum iree_hal_webgpu_queue_op_type_e {
  // CPU-only: wait → CPU work → signal inline.
  IREE_HAL_WEBGPU_QUEUE_OP_BARRIER,
  // GPU-submit: wait → GPU work → signal on GPU completion.
  IREE_HAL_WEBGPU_QUEUE_OP_TRANSFER,
  IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH,
  IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE,
} iree_hal_webgpu_queue_op_type_t;

// Fixed-width bridge record consumed by queue_transfer in webgpu_imports.mjs.
// The native transfer owner retains the HAL buffers separately from these
// handles.
typedef struct iree_hal_webgpu_transfer_t {
  // HAL transfer operation kind.
  uint32_t type;
  // Native source buffer handle, or zero for host input and fills.
  uint32_t source_handle;
  // Native destination buffer handle, or zero for host output.
  uint32_t target_handle;
  // Fill pattern expanded to a complete word.
  uint32_t pattern;
  // Native source offset, including the HAL subspan origin.
  uint64_t source_offset;
  // Native target offset, including the HAL subspan origin.
  uint64_t target_offset;
  // Borrowed wasm address, or captured UPDATE storage within the same owner.
  uint64_t host_address;
  // Full-width geometry captured at admission.
  iree_hal_transfer_region_t region;
} iree_hal_webgpu_transfer_t;
static_assert(sizeof(iree_hal_webgpu_transfer_t) == 96,
              "transfer bridge layout");

typedef struct iree_hal_webgpu_transfer_buffers_t {
  // Retained device source, if any.
  iree_hal_buffer_t* source;
  // Retained device target, if any.
  iree_hal_buffer_t* target;
} iree_hal_webgpu_transfer_buffers_t;

typedef struct iree_hal_webgpu_queue_state_t {
  // Must be first — the proactor casts between base operation and this.
  iree_async_semaphore_wait_operation_t wait_operation;

  // Embedded signal operation for GPU-submit ops. After the wait callback
  // submits GPU work, this is initialized as a NOP external operation with a
  // proactor token. The slab stays alive until onSubmittedWorkDone fires and
  // the signal callback runs.
  iree_async_operation_t signal_operation;

  iree_hal_webgpu_queue_t* queue;

  // Pre-incremented epoch. The atomic counter is incremented at submit time
  // for causal ordering; the frontier tracker is advanced at completion time.
  uint64_t epoch;

  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;

  iree_hal_webgpu_queue_op_type_t op_type;

  union {
    struct {
      // Number of nonempty transfer records.
      iree_host_size_t count;
      // Bridge records in the trailing owner allocation.
      iree_hal_webgpu_transfer_t* records;
      // HAL buffers retained until all native and host copies retire.
      iree_hal_webgpu_transfer_buffers_t* buffers;
      // Downloads require host completion before dependent GPU publication.
      bool has_download;
    } transfer;

    struct {
      iree_hal_webgpu_handle_t pipeline_handle;
      iree_hal_webgpu_handle_t bind_group_layout_handle;
      uint32_t workgroup_count[3];
      iree_hal_executable_t* executable;  // Retained.
      iree_hal_buffer_ref_t* bindings;    // Points into trailing slab.
      uint32_t binding_count;
    } dispatch;

    struct {
      iree_hal_command_buffer_t* command_buffer;  // Retained.
      iree_hal_buffer_binding_t* binding_table;   // Points into trailing slab.
      iree_host_size_t binding_count;
    } execute;
  };

  iree_allocator_t allocator;

  // Trailing arrays (via IREE_STRUCT_LAYOUT):
  //   iree_async_semaphore_t* wait_semaphores[wait_count]
  //   uint64_t wait_values[wait_count]
  //   iree_hal_semaphore_t* signal_semaphores[signal_count]
  //   uint64_t signal_values[signal_count]
  //   Per-op trailing data:
  //     DISPATCH: iree_hal_buffer_ref_t bindings[binding_count]
  //     EXECUTE: iree_hal_buffer_binding_t binding_table[binding_count]
} iree_hal_webgpu_queue_state_t;

// Releases per-op retained resources from the state's union. Called on both
// success (after work is done) and failure (wait error, submit error) paths.
// Each op type retains resources in its submit function and releases them here.
static void iree_hal_webgpu_queue_op_release_resources(
    iree_hal_webgpu_queue_state_t* state) {
  switch (state->op_type) {
    case IREE_HAL_WEBGPU_QUEUE_OP_BARRIER:
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_TRANSFER:
      for (iree_host_size_t i = 0; i < state->transfer.count; ++i) {
        iree_hal_buffer_release(state->transfer.buffers[i].source);
        iree_hal_buffer_release(state->transfer.buffers[i].target);
      }
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH:
      iree_hal_executable_release(state->dispatch.executable);
      for (uint32_t i = 0; i < state->dispatch.binding_count; ++i) {
        iree_hal_buffer_release(state->dispatch.bindings[i].buffer);
      }
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE:
      iree_hal_command_buffer_release(state->execute.command_buffer);
      for (iree_host_size_t i = 0; i < state->execute.binding_count; ++i) {
        iree_hal_buffer_release(state->execute.binding_table[i].buffer);
      }
      break;
    default:
      break;
  }
}

// Releases an unaccepted operation. Synchronous rejection leaves the caller's
// signal semaphores untouched; asynchronous failures use the completion path.
static void iree_hal_webgpu_queue_state_discard(
    iree_hal_webgpu_queue_state_t* state) {
  iree_hal_webgpu_queue_op_release_resources(state);
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Signal completion callback for GPU-submit ops. Fires when
// onSubmittedWorkDone delivers after GPU work completes. Recovers the
// queue_state from the embedded signal_operation via offsetof.
static void iree_hal_webgpu_queue_op_signal_completion(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_state_t* state =
      (iree_hal_webgpu_queue_state_t*)((uint8_t*)base_operation -
                                       offsetof(iree_hal_webgpu_queue_state_t,
                                                signal_operation));

  iree_hal_webgpu_queue_op_release_resources(state);
  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(state->queue, state->epoch,
                                             &frontier_storage);
    status =
        iree_hal_semaphore_list_signal(state->signal_semaphore_list, frontier);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }

  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Registers the embedded signal_operation for onSubmittedWorkDone completion
// tracking. Called from GPU-submit cases in the wait callback after GPU work
// has been submitted. On success returns true — the caller returns immediately
// (slab ownership transfers to the signal callback). On failure: fails the
// signal semaphores and returns false — the caller should break to shared
// cleanup.
static bool iree_hal_webgpu_queue_op_register_embedded_signal(
    iree_hal_webgpu_queue_state_t* state) {
  iree_async_operation_initialize(
      &state->signal_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_queue_op_signal_completion, /*user_data=*/NULL);
  uint32_t token = UINT32_MAX;
  iree_status_t status = iree_async_proactor_js_submit_external(
      state->queue->proactor, &state->signal_operation, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
    return false;
  }
  iree_hal_webgpu_import_queue_on_submitted_work_done(
      state->queue->queue_handle, token);
  return true;
}

// Finalizes the scratch builder, executes it, and registers the embedded
// signal for GPU completion tracking. Called from GPU-submit cases in the wait
// callback after per-op builder commands have been recorded. Returns true on
// success (caller must return immediately — slab ownership transfers to the
// signal callback); false on failure (caller should break to shared cleanup).
//
// |work_status| is the accumulated status from the per-op builder calls.
// If already failed, skips finalize/execute and fails the signal semaphores.
static bool iree_hal_webgpu_queue_op_finalize_and_submit(
    iree_hal_webgpu_queue_state_t* state, iree_status_t work_status) {
  if (iree_status_is_ok(work_status)) {
    work_status =
        iree_hal_webgpu_builder_finalize(&state->queue->scratch_builder);
  }
  if (iree_status_is_ok(work_status)) {
    work_status = iree_hal_webgpu_queue_execute_scratch_builder(state->queue);
  }
  if (!iree_status_is_ok(work_status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, work_status);
    return false;
  }
  return iree_hal_webgpu_queue_op_register_embedded_signal(state);
}

// Registers native completion before invoking JS. One token spans every upload,
// device transfer, and map/scatter continuation in this finite transaction.
static iree_status_t iree_hal_webgpu_queue_transfer_begin(
    iree_hal_webgpu_queue_state_t* state) {
  iree_async_operation_initialize(
      &state->signal_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_queue_op_signal_completion, NULL);
  uint32_t token = UINT32_MAX;
  IREE_RETURN_IF_ERROR(iree_async_proactor_js_submit_external(
      state->queue->proactor, &state->signal_operation, &token));
  iree_hal_webgpu_isa_builtins_descriptor_t builtins;
  iree_hal_webgpu_builtins_get_descriptor(state->queue->builtins, &builtins);
  iree_hal_webgpu_import_queue_transfer(
      state->queue->device_handle, state->queue->queue_handle,
      (uint32_t)(uintptr_t)&builtins,
      (uint32_t)(uintptr_t)state->transfer.records, state->transfer.count,
      token);
  if (!state->transfer.has_download) {
    iree_hal_webgpu_queue_mark_signals_submitted(state->queue,
                                                 state->signal_semaphore_list);
  }
  return iree_ok_status();
}

// Wait completion callback for all unified async queue operations. Dispatches
// on op_type to perform the operation's work after input semaphores are
// satisfied. CPU-only ops signal inline and fall through to shared cleanup.
// GPU-submit ops submit GPU work and return, transferring slab ownership to
// the signal completion callback.
static void iree_hal_webgpu_queue_op_wait_completion(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t wait_status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_state_t* state =
      (iree_hal_webgpu_queue_state_t*)base_operation;

  // Wait semaphores consumed regardless of outcome.
  iree_hal_semaphore_list_release(state->wait_semaphore_list);

  if (!iree_status_is_ok(wait_status)) {
    // Propagate wait failure to signal semaphores and clean up everything.
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, wait_status);
    iree_hal_webgpu_queue_op_release_resources(state);
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_allocator_free(state->allocator, state);
    return;
  }

  // Build frontier once for all cases that signal inline. GPU-submit success
  // cases don't use it here (they build their own in the signal callback).
  iree_async_single_frontier_t frontier_storage;
  const iree_async_frontier_t* frontier = iree_hal_webgpu_queue_build_frontier(
      state->queue, state->epoch, &frontier_storage);

  switch (state->op_type) {
    case IREE_HAL_WEBGPU_QUEUE_OP_BARRIER:
      // CPU-only: no work — signal passes straight through.
      iree_status_ignore(iree_hal_semaphore_list_signal(
          state->signal_semaphore_list, frontier));
      break;

    case IREE_HAL_WEBGPU_QUEUE_OP_TRANSFER: {
      iree_status_t status = iree_hal_webgpu_queue_transfer_begin(state);
      if (iree_status_is_ok(status)) {
        return;
      }
      iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH: {
      // GPU-submit: scratch build dispatch → GPU execute → signal on
      // completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_hal_buffer_ref_list_t binding_list = {
          .values = state->dispatch.bindings,
          .count = state->dispatch.binding_count,
      };
      iree_status_t work_status =
          iree_hal_webgpu_builder_reset(&queue->scratch_builder);
      if (iree_status_is_ok(work_status)) {
        work_status = iree_hal_webgpu_builder_dispatch(
            &queue->scratch_builder, state->dispatch.pipeline_handle,
            state->dispatch.bind_group_layout_handle,
            state->dispatch.workgroup_count, binding_list);
      }
      if (iree_hal_webgpu_queue_op_finalize_and_submit(state, work_status)) {
        return;
      }
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE: {
      // GPU-submit: command buffer submit → signal on GPU completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_hal_webgpu_handle_t recording_handle =
          iree_hal_webgpu_command_buffer_recording_handle(
              state->execute.command_buffer);
      iree_status_t work_status;
      if (recording_handle) {
        iree_hal_buffer_binding_table_t binding_table = {
            .count = state->execute.binding_count,
            .bindings = state->execute.binding_table,
        };
        work_status = iree_hal_webgpu_queue_execute_recording(
            queue, recording_handle, binding_table);
      } else {
        work_status = iree_hal_webgpu_queue_execute_one_shot(
            queue, state->execute.command_buffer);
      }
      if (!iree_status_is_ok(work_status)) {
        iree_hal_semaphore_list_fail(state->signal_semaphore_list, work_status);
        break;
      }
      if (!iree_hal_webgpu_queue_op_register_embedded_signal(state)) {
        break;
      }
      return;
    }

      // GPU-submit cases that succeed return from their case above,
      // transferring slab ownership to the signal completion callback. Error
      // cases break to the shared cleanup below.

    default:
      // Ops that haven't been converted to the unified async pattern never
      // create queue_state_t slabs. Reaching here is a programming error.
      iree_hal_semaphore_list_fail(
          state->signal_semaphore_list,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "unexpected op type %d in queue wait callback",
                           (int)state->op_type));
      break;
  }

  // Shared cleanup for CPU-only completion and GPU-submit error recovery.
  // GPU-submit success cases return from their switch case before reaching
  // here.
  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  iree_hal_webgpu_queue_op_release_resources(state);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Initializes the common fields of a pre-allocated queue state slab.
// Sets up the wait operation with semaphore arrays pointing into the trailing
// slab (aliased via cast — valid because iree_async_semaphore_t is at offset 0
// in iree_hal_webgpu_semaphore_t), clones both semaphore lists, and initializes
// the operation base with the unified wait completion callback.
static void iree_hal_webgpu_queue_state_initialize(
    iree_hal_webgpu_queue_state_t* state,
    iree_hal_webgpu_queue_op_type_t op_type, iree_hal_webgpu_queue_t* queue,
    uint64_t epoch, const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t wait_semaphores_offset,
    iree_host_size_t wait_values_offset,
    iree_host_size_t signal_semaphores_offset,
    iree_host_size_t signal_values_offset, iree_allocator_t allocator) {
  state->op_type = op_type;
  state->queue = queue;
  state->epoch = epoch;
  state->allocator = allocator;

  // Clone the wait semaphore list into the trailing slab. The wait operation's
  // semaphore pointer array aliases the same memory.
  state->wait_semaphore_list = iree_hal_webgpu_semaphore_list_at_offsets(
      state, wait_semaphore_list.count, wait_semaphores_offset,
      wait_values_offset);
  iree_hal_webgpu_semaphore_list_clone_into(wait_semaphore_list,
                                            state->wait_semaphore_list);
  state->wait_operation.semaphores =
      (iree_async_semaphore_t**)state->wait_semaphore_list.semaphores;
  state->wait_operation.values = state->wait_semaphore_list.payload_values;
  state->wait_operation.count = wait_semaphore_list.count;
  state->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  state->wait_operation.satisfied_index = 0;

  // Clone the signal semaphore list into the trailing slab.
  state->signal_semaphore_list = iree_hal_webgpu_semaphore_list_at_offsets(
      state, signal_semaphore_list.count, signal_semaphores_offset,
      signal_values_offset);
  iree_hal_webgpu_semaphore_list_clone_into(signal_semaphore_list,
                                            state->signal_semaphore_list);

  // Initialize the wait operation base with the unified completion callback.
  iree_async_operation_initialize(
      &state->wait_operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_queue_op_wait_completion,
      /*user_data=*/NULL);
}

// Allocates and initializes a queue state slab for an async queue operation.
// Pre-increments the queue epoch, computes the slab layout (base struct +
// 4 trailing semaphore arrays + optional per-op trailing data), allocates the
// slab, and initializes common fields. On failure, fails the signal semaphores
// and returns the error.
//
// |trailing_count| and |trailing_element_size| specify an optional trailing
// array for per-op data (dispatch bindings, execute binding table). Pass 0
// for both when no extra trailing data is needed. If non-zero,
// |out_trailing_offset| receives the byte offset from the slab base.
static iree_status_t iree_hal_webgpu_queue_state_allocate(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_queue_op_type_t op_type,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t trailing_count, iree_host_size_t trailing_element_size,
    iree_host_size_t* out_trailing_offset,
    iree_hal_webgpu_queue_state_t** out_state) {
  *out_state = NULL;

  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);

  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_queue_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_async_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));

  // Append per-op trailing data after the semaphore arrays. STRUCT_LAYOUT
  // produces a max_align_t-aligned total_size, which satisfies alignment for
  // all per-op element types (they contain pointers and device_size_t fields,
  // both <= max_align_t).
  if (iree_status_is_ok(status) && trailing_count > 0) {
    if (out_trailing_offset) {
      *out_trailing_offset = total_size;
    }
    iree_host_size_t trailing_bytes = 0;
    if (!iree_host_size_checked_mul(trailing_count, trailing_element_size,
                                    &trailing_bytes) ||
        !iree_host_size_checked_add(total_size, trailing_bytes, &total_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "trailing allocation size overflow");
    }
  }

  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_webgpu_queue_state_t* state = NULL;
  status =
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&state);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_webgpu_queue_state_initialize(
      state, op_type, queue, epoch, wait_semaphore_list, signal_semaphore_list,
      wait_semaphores_offset, wait_values_offset, signal_semaphores_offset,
      signal_values_offset, queue->host_allocator);

  *out_state = state;
  return iree_ok_status();
}

// Submits an initialized queue state to the proactor. A rejected submission
// releases its state without affecting caller semaphores.
static iree_status_t iree_hal_webgpu_queue_state_submit(
    iree_hal_webgpu_queue_state_t* state) {
  iree_status_t status = iree_async_proactor_submit_one(
      state->queue->proactor, &state->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_state_discard(state);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Queue operations (scratch builder)
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_submit_transfer(
    iree_hal_webgpu_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_host_size_t count = 0;
  iree_host_size_t capture_length = 0;
  bool has_download = false;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const iree_device_size_t length =
        iree_hal_transfer_operation_length(&operations[i]);
    if (!length) {
      continue;
    }
    ++count;
    has_download |=
        operations[i].type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    if (operations[i].type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
        !iree_host_size_checked_add(capture_length, (iree_host_size_t)length,
                                    &capture_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "WebGPU transfer capture exceeds address space");
    }
  }
  if (!count) {
    return iree_hal_webgpu_queue_submit_execute(
        queue, waits, signals, NULL, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  iree_host_size_t size = 0, records_offset = 0, buffers_offset = 0,
                   data_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD(count, iree_hal_webgpu_transfer_t, &records_offset),
      IREE_STRUCT_FIELD(count, iree_hal_webgpu_transfer_buffers_t,
                        &buffers_offset),
      IREE_STRUCT_FIELD(capture_length, uint8_t, &data_offset)));
  iree_hal_webgpu_queue_state_t* state = NULL;
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_TRANSFER, waits, signals, 1, size,
      &offset, &state));
  uint8_t* storage = (uint8_t*)state + offset;
  state->transfer.count = count;
  state->transfer.records =
      (iree_hal_webgpu_transfer_t*)(storage + records_offset);
  state->transfer.buffers =
      (iree_hal_webgpu_transfer_buffers_t*)(storage + buffers_offset);
  state->transfer.has_download = has_download;
  uint8_t* data = storage + data_offset;
  count = 0;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const iree_hal_transfer_operation_t* operation = &operations[i];
    const iree_device_size_t length =
        iree_hal_transfer_operation_length(operation);
    if (!length) {
      continue;
    }
    iree_hal_webgpu_transfer_t* record = &state->transfer.records[count];
    iree_hal_webgpu_transfer_buffers_t* buffers =
        &state->transfer.buffers[count++];
    *record = (iree_hal_webgpu_transfer_t){
        .type = operation->type,
        .region = iree_hal_transfer_operation_region(operation),
    };
    *buffers = (iree_hal_webgpu_transfer_buffers_t){0};
    switch (operation->type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        buffers->target = operation->fill.target_buffer;
        record->target_offset = operation->fill.target_offset;
        memcpy(&record->pattern, operation->fill.pattern,
               operation->fill.pattern_length);
        if (operation->fill.pattern_length == 1) {
          record->pattern *= UINT32_C(0x01010101);
        }
        if (operation->fill.pattern_length == 2) {
          record->pattern |= record->pattern << 16;
        }
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        buffers->target = operation->update.target_buffer;
        record->target_offset = operation->update.target_offset;
        iree_hal_transfer_gather_host(
            (const uint8_t*)operation->update.source_buffer +
                operation->update.source_offset,
            &record->region, 0, length, data);
        record->host_address = (uintptr_t)data;
        record->region.source_strides.row_stride = record->region.row_length;
        record->region.source_strides.slice_stride =
            record->region.row_length * record->region.row_count;
        data += length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        buffers->source = operation->copy.source_buffer;
        record->source_offset = operation->copy.source_offset;
        buffers->target = operation->copy.target_buffer;
        record->target_offset = operation->copy.target_offset;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        buffers->target = operation->upload.target_buffer;
        record->target_offset = operation->upload.target_offset;
        record->host_address = (uintptr_t)operation->upload.source;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        buffers->source = operation->download.source_buffer;
        record->source_offset = operation->download.source_offset;
        record->host_address = (uintptr_t)operation->download.target;
        break;
    }
    if (buffers->source) {
      iree_hal_buffer_retain(buffers->source);
      record->source_handle = iree_hal_webgpu_buffer_handle(buffers->source);
      record->source_offset += iree_hal_buffer_byte_offset(buffers->source);
    }
    if (buffers->target) {
      iree_hal_buffer_retain(buffers->target);
      record->target_handle = iree_hal_webgpu_buffer_handle(buffers->target);
      record->target_offset += iree_hal_buffer_byte_offset(buffers->target);
    }
  }
  if (!waits.count || iree_hal_semaphore_list_poll(waits) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, waits)) {
    iree_status_t status = iree_hal_webgpu_queue_transfer_begin(state);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_release(state->wait_semaphore_list);
    } else {
      iree_hal_webgpu_queue_state_discard(state);
    }
    return status;
  }
  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// Host call (inline + proactor-driven async)
//===----------------------------------------------------------------------===//

// Executes a host call inline and handles the result status.
// On OK: signals all signal semaphores with the given frontier.
// On DEFERRED: the callback has cloned the signal list and will signal later.
//   The DEFERRED path does not use the frontier — the callback manages its own
//   signaling and can build a frontier itself if needed.
// On error: fails all signal semaphores with the error status.
// If NON_BLOCKING: signals semaphores before calling, ignores the result.
static iree_status_t iree_hal_webgpu_queue_execute_host_call(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* frontier, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  if (flags & IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING) {
    // Signal semaphores immediately, then fire the call. The callback cannot
    // observe the signal list and its result is ignored.
    iree_status_t signal_status =
        iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
    iree_hal_host_call_context_t context = {
        .queue = &queue->base,
        .signal_semaphore_list = iree_hal_semaphore_list_empty(),
    };
    iree_status_free(call.fn(call.user_data, args, &context));
    return signal_status;
  }

  iree_hal_host_call_context_t context = {
      .queue = &queue->base,
      .signal_semaphore_list = signal_semaphore_list,
  };
  iree_status_t call_status = call.fn(call.user_data, args, &context);
  if (iree_status_is_ok(call_status)) {
    return iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
  } else if (iree_status_code(call_status) == IREE_STATUS_DEFERRED) {
    // The callback has cloned the signal list and will signal later.
    iree_status_free(call_status);
    return iree_ok_status();
  } else {
    // Fail all signal semaphores with the error.
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(call_status));
    return call_status;
  }
}

// State for a deferred host call that waits on input semaphores via the
// proactor before executing. Allocated as a single slab:
//   [struct | wait iree_async_semaphore_t*[] | wait uint64_t[] |
//    signal iree_hal_semaphore_t*[] | signal uint64_t[]]
typedef struct iree_hal_webgpu_host_call_state_t {
  // Asynchronous wait operation submitted to the proactor.
  iree_async_semaphore_wait_operation_t wait_operation;

  // User callback and state.
  iree_hal_host_call_t call;

  // User-defined callback arguments.
  uint64_t args[4];

  // Flags controlling callback ordering and completion.
  iree_hal_host_call_flags_t flags;

  // Exact hardware queue issuing the callback. Borrowed from the device.
  iree_hal_webgpu_queue_t* queue;

  // Pre-incremented at submit for causal ordering.
  uint64_t epoch;

  // Semaphores signaled when the callback completes.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Semaphores that must reach their payloads before invoking the callback.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Allocator used for the state slab.
  iree_allocator_t allocator;
} iree_hal_webgpu_host_call_state_t;

// Completion callback invoked by the proactor when all wait semaphores are
// satisfied (or one has failed).
static void iree_hal_webgpu_host_call_completion_fn(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t wait_status, iree_async_completion_flags_t completion_flags) {
  iree_hal_webgpu_host_call_state_t* state =
      (iree_hal_webgpu_host_call_state_t*)base_operation;

  if (iree_status_is_ok(wait_status)) {
    // Wait succeeded — build frontier and execute the host call. This handles
    // signaling/failing the signal semaphores based on the call result.
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(state->queue, state->epoch,
                                             &frontier_storage);
    iree_status_free(iree_hal_webgpu_queue_execute_host_call(
        state->queue, state->signal_semaphore_list, frontier, state->call,
        state->args, state->flags));
  } else {
    // Wait itself failed (semaphore failure propagation). Fail all signal
    // semaphores with the wait failure status.
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, wait_status);
  }

  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);

  // Release both semaphore lists and free the slab.
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  iree_hal_resource_release(state->call.resource);
  iree_allocator_free(state->allocator, state);
}

iree_status_t iree_hal_webgpu_queue_submit_host_call(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  // Fast path: no wait semaphores or all already satisfied — execute inline.
  // NO FIFO elision — host calls execute on the CPU, not the GPU FIFO.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
    iree_status_t status = iree_hal_webgpu_queue_execute_host_call(
        queue, signal_semaphore_list, frontier, call, args, flags);
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
    return status;
  }

  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);

  // Async path: allocate a slab with the wait operation, call state, and
  // trailing arrays for both wait and signal semaphore lists.
  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_host_call_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_async_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  iree_hal_webgpu_host_call_state_t* state = NULL;
  status =
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  // Copy call parameters.
  state->call = call;
  iree_hal_resource_retain(state->call.resource);
  memcpy(state->args, args, sizeof(state->args));
  state->flags = flags;
  state->queue = queue;
  state->epoch = epoch;
  state->allocator = queue->host_allocator;

  // Set up the wait semaphore list and clone the caller's data into it.
  // The wait operation's semaphore pointers alias the same array: WebGPU
  // semaphores have iree_async_semaphore_t at offset 0, so the pointer bits
  // are identical regardless of which type they're cast to.
  state->wait_semaphore_list = iree_hal_webgpu_semaphore_list_at_offsets(
      state, wait_semaphore_list.count, wait_semaphores_offset,
      wait_values_offset);
  iree_hal_webgpu_semaphore_list_clone_into(wait_semaphore_list,
                                            state->wait_semaphore_list);
  state->wait_operation.semaphores =
      (iree_async_semaphore_t**)state->wait_semaphore_list.semaphores;
  state->wait_operation.values = state->wait_semaphore_list.payload_values;
  state->wait_operation.count = wait_semaphore_list.count;
  state->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  state->wait_operation.satisfied_index = 0;

  // Set up the signal semaphore list and clone the caller's data into it.
  state->signal_semaphore_list = iree_hal_webgpu_semaphore_list_at_offsets(
      state, signal_semaphore_list.count, signal_semaphores_offset,
      signal_values_offset);
  iree_hal_webgpu_semaphore_list_clone_into(signal_semaphore_list,
                                            state->signal_semaphore_list);

  // Initialize the wait operation base and submit to the proactor.
  iree_async_operation_initialize(
      &state->wait_operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_host_call_completion_fn,
      /*user_data=*/NULL);

  status = iree_async_proactor_submit_one(queue->proactor,
                                          &state->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                                 iree_status_clone(status));
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_hal_semaphore_list_release(state->wait_semaphore_list);
    iree_hal_resource_release(state->call.resource);
    iree_allocator_free(queue->host_allocator, state);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// queue_dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_submit_dispatch(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "exact work-item dispatch is not supported by WebGPU");
  }
  if (IREE_UNLIKELY(constants.data_length != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "WebGPU does not support push constants; lower constants to buffer "
        "bindings");
  }
  // Extract pipeline/bgl handles at submit time. These are bridge table
  // indices (uint32 values) that remain valid as long as the executable lives.
  iree_hal_webgpu_handle_t pipeline_handle =
      iree_hal_webgpu_executable_pipeline_handle(executable, function);
  iree_hal_webgpu_handle_t bind_group_layout_handle =
      iree_hal_webgpu_executable_bind_group_layout_handle(executable, function);

  // Fast path: waits already satisfied (or FIFO-elided) — execute
  // synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_status_t status =
        iree_hal_webgpu_builder_reset(&queue->scratch_builder);
    if (iree_status_is_ok(status)) {
      status = iree_hal_webgpu_builder_dispatch(
          &queue->scratch_builder, pipeline_handle, bind_group_layout_handle,
          config.workgroup_count, bindings);
    }
    return iree_hal_webgpu_queue_submit_scratch_and_signal(
        queue, signal_semaphore_list, status);
  }

  // Async path: capture params, retain executable + per-binding buffers,
  // snapshot bindings into trailing slab.
  iree_host_size_t bindings_offset = 0;
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH, wait_semaphore_list,
      signal_semaphore_list, bindings.count, sizeof(iree_hal_buffer_ref_t),
      &bindings_offset, &state));

  state->dispatch.pipeline_handle = pipeline_handle;
  state->dispatch.bind_group_layout_handle = bind_group_layout_handle;
  memcpy(state->dispatch.workgroup_count, config.workgroup_count,
         sizeof(state->dispatch.workgroup_count));

  iree_hal_executable_retain(executable);
  state->dispatch.executable = executable;

  // Snapshot bindings into the trailing slab and retain each buffer.
  state->dispatch.bindings =
      (iree_hal_buffer_ref_t*)((uint8_t*)state + bindings_offset);
  state->dispatch.binding_count = (uint32_t)bindings.count;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    state->dispatch.bindings[i] = bindings.values[i];
    if (bindings.values[i].buffer) {
      iree_hal_buffer_retain(bindings.values[i].buffer);
    }
  }

  return iree_hal_webgpu_queue_state_submit(state);
}

// Submits an async barrier (wait → signal inline) to the proactor using the
// unified queue state. Called when iree_hal_semaphore_list_poll indicates the
// wait semaphores are not yet satisfied. On success, ownership transfers to
// the proactor — the wait completion callback signals and frees the state.
static iree_status_t iree_hal_webgpu_queue_submit_barrier(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_BARRIER, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));
  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// queue_execute
//===----------------------------------------------------------------------===//

// Executes a reusable command buffer's cached recording. Dynamic bindings from
// the binding table are resolved to GPU buffer handles for the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_recording(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_handle_t recording_handle,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* dynamic_entries = inline_entries;
  if (binding_table.count > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, binding_table.count,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t),
        (void**)&dynamic_entries));
  }

  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    const iree_hal_buffer_binding_t* binding = &binding_table.bindings[i];
    if (binding->buffer) {
      iree_hal_buffer_t* allocated =
          iree_hal_buffer_allocated_buffer(binding->buffer);
      dynamic_entries[i].gpu_buffer_handle =
          iree_hal_webgpu_buffer_handle(allocated);
      dynamic_entries[i].reserved = 0;
      dynamic_entries[i].base_offset =
          iree_hal_buffer_byte_offset(binding->buffer) + binding->offset;
      dynamic_entries[i].length =
          binding->length == IREE_HAL_WHOLE_BUFFER
              ? iree_hal_buffer_byte_length(binding->buffer) - binding->offset
              : binding->length;
    } else {
      dynamic_entries[i].gpu_buffer_handle = 0;
      dynamic_entries[i].base_offset = 0;
      dynamic_entries[i].reserved = 0;
      dynamic_entries[i].length = 0;
    }
  }

  uint32_t result = iree_hal_webgpu_import_execute_recording(
      recording_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)dynamic_entries);

  if (dynamic_entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, dynamic_entries);
  }
  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_recording failed with code %u", result);
  }
  return iree_ok_status();
}

// Executes a ONE_SHOT command buffer's instruction stream directly. Builds the
// binding table from the builder's static slot map and passes it with the
// builtins descriptor to the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_one_shot(
    iree_hal_webgpu_queue_t* queue, iree_hal_command_buffer_t* command_buffer) {
  iree_hal_webgpu_builder_t* builder =
      iree_hal_webgpu_command_buffer_builder(command_buffer);

  uint32_t total_slots = iree_hal_webgpu_builder_total_slot_count(builder);
  uint32_t static_count = iree_hal_webgpu_builder_static_slot_count(builder);

  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* entries = inline_entries;
  if (total_slots > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, total_slots,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t), (void**)&entries));
  }

  const iree_hal_webgpu_builder_slot_entry_t* slot_entries =
      iree_hal_webgpu_builder_static_slot_entries(builder);
  for (uint32_t i = 0; i < static_count; ++i) {
    entries[slot_entries[i].slot].gpu_buffer_handle =
        slot_entries[i].gpu_buffer_handle;
    entries[slot_entries[i].slot].base_offset = 0;
    entries[slot_entries[i].slot].reserved = 0;
    entries[slot_entries[i].slot].length =
        iree_hal_buffer_byte_length(slot_entries[i].buffer);
  }

  iree_hal_webgpu_isa_builtins_descriptor_t builtins_descriptor;
  iree_hal_webgpu_builtins_get_descriptor(queue->builtins,
                                          &builtins_descriptor);

  uint32_t result = iree_hal_webgpu_import_execute_instructions(
      queue->device_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)iree_hal_webgpu_builder_block_table(builder),
      iree_hal_webgpu_builder_block_count(builder),
      iree_hal_webgpu_builder_block_word_capacity(builder),
      iree_hal_webgpu_builder_last_block_word_count(builder),
      (uint32_t)(uintptr_t)entries, total_slots,
      (uint32_t)(uintptr_t)&builtins_descriptor);

  if (entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, entries);
  }
  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_instructions failed with code %u",
                            result);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_queue_submit_execute(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  // Barrier-only submission: no command buffer, just wait->signal.
  // NO FIFO elision — barrier signals are CPU-side with no GPU FIFO backing,
  // so FIFO ordering does not guarantee the signal is visible to consumers.
  if (!command_buffer) {
    if (wait_semaphore_list.count == 0 ||
        iree_hal_semaphore_list_poll(wait_semaphore_list)) {
      uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
      iree_async_single_frontier_t frontier_storage;
      const iree_async_frontier_t* frontier =
          iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
      iree_status_t status =
          iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
      iree_hal_webgpu_queue_advance_tracker(queue, epoch);
      return status;
    }
    return iree_hal_webgpu_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  // Fast path: waits already satisfied (or FIFO-elided) — execute
  // synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_hal_webgpu_handle_t recording_handle =
        iree_hal_webgpu_command_buffer_recording_handle(command_buffer);
    iree_status_t status;
    if (recording_handle) {
      status = iree_hal_webgpu_queue_execute_recording(queue, recording_handle,
                                                       binding_table);
    } else {
      status = iree_hal_webgpu_queue_execute_one_shot(queue, command_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_webgpu_queue_register_signal_completion(
          queue, epoch, signal_semaphore_list);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_webgpu_queue_mark_signals_submitted(queue,
                                                   signal_semaphore_list);
    } else {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
    return status;
  }

  // Async path: retain command buffer, snapshot binding table into trailing
  // slab (retaining each buffer), submit wait.
  iree_host_size_t binding_table_offset = 0;
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE, wait_semaphore_list,
      signal_semaphore_list, binding_table.count,
      sizeof(iree_hal_buffer_binding_t), &binding_table_offset, &state));

  iree_hal_command_buffer_retain(command_buffer);
  state->execute.command_buffer = command_buffer;

  // Snapshot the binding table into the trailing slab and retain each buffer.
  state->execute.binding_table =
      (iree_hal_buffer_binding_t*)((uint8_t*)state + binding_table_offset);
  state->execute.binding_count = binding_table.count;
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    state->execute.binding_table[i] = binding_table.bindings[i];
    if (binding_table.bindings[i].buffer) {
      iree_hal_buffer_retain(binding_table.bindings[i].buffer);
    }
  }

  return iree_hal_webgpu_queue_state_submit(state);
}
