// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/queue.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/command_buffer.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/client/file.h"
#include "iree/hal/remote/client/semaphore.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/status_wire.h"

//===----------------------------------------------------------------------===//
// Pending signal batch
//===----------------------------------------------------------------------===//

// Batch header for pending signal contexts. A single allocation holds this
// header followed by N iree_hal_remote_pending_signal_t entries. The atomic
// counter tracks how many entries are still live; the last callback to
// decrement frees the entire batch.
typedef struct iree_hal_remote_pending_signal_batch_t {
  // Number of pending signal entries plus the submitter hold.
  iree_atomic_int32_t remaining;

  // Allocator used to free the batch allocation.
  iree_allocator_t host_allocator;

  // Number of pending signal entries in trailing storage.
  iree_host_size_t entry_count;

  // Byte offset from this header to the trailing signal entries.
  iree_host_size_t entries_offset;

  // Trailing iree_hal_remote_pending_signal_t entries.
} iree_hal_remote_pending_signal_batch_t;

// Per-semaphore signal context within a batch. Each entry holds a frontier
// waiter that fires when the server's ADVANCE echoes the submission epoch.
typedef struct iree_hal_remote_pending_signal_t {
  // Waiter registered with the device frontier tracker.
  iree_async_frontier_waiter_t waiter;
  // Semaphore signaled when |frontier| is reached.
  iree_hal_semaphore_t* semaphore;
  // Timeline value signaled on |semaphore|.
  uint64_t value;
  // Parent batch owning this entry.
  iree_hal_remote_pending_signal_batch_t* batch;
  // Single-entry frontier waited by |waiter|.
  iree_async_single_frontier_t frontier;
} iree_hal_remote_pending_signal_t;

// Fired by the frontier tracker when the signal frontier is satisfied.
// Signals the proxy semaphore to the target value. The last callback to
// complete frees the entire batch allocation.
static void iree_hal_remote_pending_signal_callback(void* user_data,
                                                    iree_status_t status) {
  iree_hal_remote_pending_signal_t* pending =
      (iree_hal_remote_pending_signal_t*)user_data;
  if (iree_status_is_ok(status)) {
    iree_status_t signal_status = iree_hal_semaphore_signal(
        pending->semaphore, pending->value,
        iree_async_single_frontier_as_const_frontier(&pending->frontier));
    iree_status_ignore(signal_status);
  } else {
    // Frontier wait failed (axis error). Propagate by failing the semaphore.
    iree_hal_semaphore_fail(pending->semaphore, status);
  }
  iree_hal_semaphore_release(pending->semaphore);
  iree_hal_remote_pending_signal_batch_t* batch = pending->batch;
  if (iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_allocator_free(batch->host_allocator, batch);
  }
}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

// Maximum host-allocation file slice sent inline on the queue channel.
//
// Larger client-local files use the bulk channel so latency-sensitive queue
// traffic is not trapped behind file payloads. This conservative bound leaves
// room for queue frame headers and frontiers under the default frame size.
#define IREE_HAL_REMOTE_CLIENT_FILE_INLINE_UPDATE_MAX_LENGTH \
  (IREE_NET_QUEUE_FRAME_DEFAULT_MAX_SIZE / 2)

static iree_hal_remote_pending_signal_t*
iree_hal_remote_pending_signal_batch_entries(
    iree_hal_remote_pending_signal_batch_t* batch) {
  return (iree_hal_remote_pending_signal_t*)((uint8_t*)batch +
                                             batch->entries_offset);
}

// Allocates and retains all signal waiter state before transport admission.
static iree_status_t iree_hal_remote_client_prepare_signal_waiters(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_remote_pending_signal_batch_t** out_batch) {
  *out_batch = NULL;
  if (signal_semaphore_list.count == 0) {
    return iree_ok_status();
  }

  iree_hal_remote_pending_signal_batch_t* batch = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t entries_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*batch), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          signal_semaphore_list.count, iree_hal_remote_pending_signal_t,
          iree_alignof(iree_hal_remote_pending_signal_t), &entries_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(device->host_allocator, total_size,
                                             (void**)&batch));

  iree_atomic_store(&batch->remaining, 1, iree_memory_order_relaxed);
  batch->host_allocator = device->host_allocator;
  batch->entry_count = signal_semaphore_list.count;
  batch->entries_offset = entries_offset;

  iree_hal_remote_pending_signal_t* entries =
      (iree_hal_remote_pending_signal_t*)((uint8_t*)batch + entries_offset);
  memset(entries, 0, signal_semaphore_list.count * sizeof(*entries));
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    entries[i].semaphore = signal_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(entries[i].semaphore);
    entries[i].value = signal_semaphore_list.payload_values[i];
    entries[i].batch = batch;
  }

  *out_batch = batch;
  return iree_ok_status();
}

// Releases a prepared batch that has not registered any frontier waiters.
static void iree_hal_remote_client_discard_signal_waiters(
    iree_hal_remote_pending_signal_batch_t* batch) {
  if (!batch) return;
  iree_hal_remote_pending_signal_t* entries =
      iree_hal_remote_pending_signal_batch_entries(batch);
  for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
    iree_hal_semaphore_release(entries[i].semaphore);
  }
  iree_allocator_free(batch->host_allocator, batch);
}

// Publishes and consumes prepared signal waiters at the assigned epoch.
static iree_status_t iree_hal_remote_client_register_signal_waiters(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_pending_signal_batch_t* batch, iree_async_axis_t axis,
    uint64_t epoch) {
  if (!batch) return iree_ok_status();
  iree_hal_remote_pending_signal_t* entries =
      iree_hal_remote_pending_signal_batch_entries(batch);

  iree_status_t status = iree_ok_status();
  iree_host_size_t registered_count = 0;
  for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
    iree_hal_remote_pending_signal_t* pending = &entries[i];
    iree_async_single_frontier_initialize(&pending->frontier, axis, epoch);

    // Add a ref for this waiter before registration. If registration fails
    // we undo the ref and release the semaphore.
    iree_atomic_fetch_add(&batch->remaining, 1, iree_memory_order_relaxed);

    status = iree_async_frontier_tracker_wait(
        device->frontier_tracker,
        iree_async_single_frontier_as_frontier(&pending->frontier),
        iree_hal_remote_pending_signal_callback, pending, &pending->waiter);
    if (!iree_status_is_ok(status)) {
      iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_relaxed);
      iree_hal_semaphore_release(pending->semaphore);
      break;
    }

    // Record the (value → axis, epoch) mapping on the semaphore AFTER the
    // waiter is successfully registered. Recording before registration would
    // leave a stale mapping if registration fails — a subsequent operation
    // would encode the stale epoch in its wait frontier, and the server would
    // fail with NOT_FOUND because the COMMAND was never sent.
    iree_hal_remote_client_semaphore_record_epoch(pending->semaphore,
                                                  pending->value, axis, epoch);

    ++registered_count;
  }

  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = registered_count + 1; i < batch->entry_count;
         ++i) {
      iree_hal_semaphore_release(entries[i].semaphore);
    }
  }

  // Release the submitter hold. Registered waiters now own the batch.
  if (iree_atomic_fetch_sub(&batch->remaining, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_allocator_free(batch->host_allocator, batch);
  }

  return status;
}

IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_remote_frontier_storage_t, 16);

// Adds or merges a wait frontier entry while preserving sorted frontier order.
static iree_status_t iree_hal_remote_add_wait_entry(
    iree_async_frontier_entry_t* entries, iree_host_size_t max_entry_count,
    iree_host_size_t* entry_count, iree_async_axis_t axis, uint64_t epoch) {
  if (epoch == 0) return iree_ok_status();
  iree_host_size_t insert_index = 0;
  while (insert_index < *entry_count && entries[insert_index].axis < axis) {
    ++insert_index;
  }
  if (insert_index < *entry_count && entries[insert_index].axis == axis) {
    entries[insert_index].epoch = iree_max(entries[insert_index].epoch, epoch);
    return iree_ok_status();
  }
  if (*entry_count >= max_entry_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "wait frontier exceeds max entry count %" PRIhsz,
                            max_entry_count);
  }
  for (iree_host_size_t i = *entry_count; i > insert_index; --i) {
    entries[i] = entries[i - 1];
  }
  entries[insert_index] = (iree_async_frontier_entry_t){
      .axis = axis,
      .epoch = epoch,
  };
  ++*entry_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_merge_wait_frontier(
    iree_async_frontier_entry_t* entries, iree_host_size_t max_entry_count,
    iree_host_size_t* entry_count, const iree_async_frontier_t* frontier) {
  if (!frontier) return iree_ok_status();
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_remote_add_wait_entry(
        entries, max_entry_count, entry_count, frontier->entries[i].axis,
        frontier->entries[i].epoch));
  }
  return iree_ok_status();
}

// Builds a wait frontier from a HAL wait_semaphore_list by looking up
// epoch mappings on each proxy semaphore. Populates |out_entries| with up to
// |max_entries| frontier entries. Semaphores that are already satisfied
// (current value >= wait value) are skipped. Unsatisfied semaphores without
// an epoch mapping (host-signaled, cross-device) are reported via
// |out_first_gate|: the first such semaphore and value that must be locally
// gated before the COMMAND can be sent.
static iree_status_t iree_hal_remote_client_device_build_wait_frontier(
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_async_frontier_entry_t* out_entries, iree_host_size_t max_entries,
    iree_host_size_t* out_entry_count, iree_hal_semaphore_t** out_first_gate,
    uint64_t* out_first_gate_value) {
  *out_entry_count = 0;
  *out_first_gate = NULL;
  *out_first_gate_value = 0;
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphore_list.semaphores[i];
    uint64_t value = wait_semaphore_list.payload_values[i];

    // If the semaphore is already satisfied, no wait needed.
    uint64_t current_value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore, &current_value);
    if (iree_status_is_ok(query_status) && current_value >= value) continue;
    IREE_RETURN_IF_ERROR(query_status);

    // Look up the epoch mapping on the proxy semaphore.
    iree_async_axis_t axis = 0;
    uint64_t epoch = 0;
    if (!iree_hal_remote_client_semaphore_lookup_epoch(semaphore, value, &axis,
                                                       &epoch)) {
      // No epoch mapping: this semaphore will be signaled by something
      // other than a prior queue operation on this device (host signal,
      // cross-device dependency, deferred operation whose epoch hasn't
      // been assigned yet). Report it as a gate.
      if (!*out_first_gate) {
        *out_first_gate = semaphore;
        *out_first_gate_value = value;
      }
      continue;
    }

    IREE_RETURN_IF_ERROR(iree_hal_remote_add_wait_entry(
        out_entries, max_entries, out_entry_count, axis, epoch));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Deferred submit
//===----------------------------------------------------------------------===//

// Context for a deferred queue operation submission. Allocated when
// submit_queue_op encounters an unsatisfied wait semaphore without an epoch
// mapping (host→device dependency, cross-device dependency, or dependency on
// a deferred operation whose epoch hasn't been assigned yet).
//
// A timepoint is registered on the gate semaphore. When it fires (semaphore
// satisfied), the submit is retried with the saved arguments. The previously-
// gated semaphore is now satisfied and gets skipped. If more gates remain,
// another deferred context is allocated (recursive, converges after at most
// N retries for N gate semaphores).
typedef struct iree_hal_remote_deferred_submit_t {
  // Timepoint registered on the first host/cross-device gate semaphore.
  iree_async_semaphore_timepoint_t timepoint;
  // Borrowed device pointer; the device outlives deferred submissions.
  iree_hal_remote_client_device_t* device;
  // Allocator used to free this deferred submission allocation.
  iree_allocator_t host_allocator;

  // Number of retained wait semaphores in trailing storage.
  iree_host_size_t wait_count;
  // Number of retained signal semaphores in trailing storage.
  iree_host_size_t signal_count;
  // Number of required wait frontier entries in trailing storage.
  iree_host_size_t required_wait_entry_count;
  // Number of retained HAL resources in trailing storage.
  iree_host_size_t resource_count;
  // Byte offset of wait semaphores in trailing storage.
  iree_host_size_t wait_semaphores_offset;
  // Byte offset of wait values in trailing storage.
  iree_host_size_t wait_values_offset;
  // Byte offset of signal semaphores in trailing storage.
  iree_host_size_t signal_semaphores_offset;
  // Byte offset of signal values in trailing storage.
  iree_host_size_t signal_values_offset;
  // Byte offset of required wait frontier entries in trailing storage.
  iree_host_size_t required_wait_entries_offset;
  // Byte offset of retained resources in trailing storage.
  iree_host_size_t resources_offset;
  // Byte offset of queue payload bytes in trailing storage.
  iree_host_size_t payload_offset;
  // Trailing layout:
  //   iree_hal_semaphore_t* wait_semaphores[wait_count]
  //   uint64_t wait_values[wait_count]
  //   iree_hal_semaphore_t* signal_semaphores[signal_count]
  //   uint64_t signal_values[signal_count]
  //   iree_async_frontier_entry_t required_wait_entries[entry_count]
  //   iree_hal_resource_t* resources[resource_count]
  //   uint8_t payload_data[payload_length]
  // Number of payload bytes in trailing storage.
  iree_host_size_t payload_length;
} iree_hal_remote_deferred_submit_t;

typedef struct iree_hal_remote_queue_payload_writer_t {
  // Callback that serializes the payload into |target|.
  iree_status_t (*write)(void* user_data, iree_byte_span_t target);
  // User data passed to |write|.
  void* user_data;
  // Exact number of bytes written by |write|.
  iree_host_size_t payload_length;
} iree_hal_remote_queue_payload_writer_t;

typedef struct iree_hal_remote_queue_resource_list_t {
  // Callback that writes resource pointers into |target_resources|.
  iree_status_t (*write)(void* user_data,
                         iree_hal_resource_t** target_resources);
  // User data passed to |write|.
  void* user_data;
  // Exact number of resource pointers written by |write|.
  iree_host_size_t resource_count;
} iree_hal_remote_queue_resource_list_t;

typedef struct iree_hal_remote_resource_ptr_list_t {
  // Resource pointer values to copy.
  iree_hal_resource_t* const* values;
  // Number of resource pointer values to copy.
  iree_host_size_t count;
} iree_hal_remote_resource_ptr_list_t;

static void iree_hal_remote_retain_resources(
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources) {
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    iree_hal_resource_retain(resources[i]);
  }
}

static void iree_hal_remote_release_resources(iree_host_size_t resource_count,
                                              iree_hal_resource_t** resources) {
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    iree_hal_resource_release(resources[i]);
  }
}

static iree_status_t iree_hal_remote_write_resource_ptrs(
    void* user_data, iree_hal_resource_t** target_resources) {
  const iree_hal_remote_resource_ptr_list_t* source_resources =
      (const iree_hal_remote_resource_ptr_list_t*)user_data;
  memcpy(target_resources, source_resources->values,
         source_resources->count * sizeof(*target_resources));
  return iree_ok_status();
}

static iree_hal_remote_queue_resource_list_t
iree_hal_remote_make_resource_ptr_list(
    iree_hal_resource_t* const* resources, iree_host_size_t resource_count,
    iree_hal_remote_resource_ptr_list_t* out_storage) {
  out_storage->values = resources;
  out_storage->count = resource_count;
  iree_hal_remote_queue_resource_list_t resource_list = {
      .write = iree_hal_remote_write_resource_ptrs,
      .user_data = out_storage,
      .resource_count = resource_count,
  };
  return resource_list;
}

static iree_hal_remote_queue_resource_list_t
iree_hal_remote_empty_resource_list(void) {
  iree_hal_remote_queue_resource_list_t resource_list = {0};
  return resource_list;
}

static iree_status_t iree_hal_remote_payload_total_length(
    iree_async_span_list_t op_payload, iree_host_size_t* out_total_length) {
  iree_host_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < op_payload.count; ++i) {
    if (!iree_host_size_checked_add(total_length, op_payload.values[i].length,
                                    &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue op payload size overflow");
    }
  }
  *out_total_length = total_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_write_payload_spans(
    void* user_data, iree_byte_span_t target) {
  const iree_async_span_list_t* op_payload =
      (const iree_async_span_list_t*)user_data;
  iree_host_size_t offset = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < op_payload->count && iree_status_is_ok(status); ++i) {
    iree_host_size_t next_offset = 0;
    if (!iree_host_size_checked_add(offset, op_payload->values[i].length,
                                    &next_offset) ||
        next_offset > target.data_length) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "queue op payload write exceeds target span");
      break;
    }
    memcpy(target.data + offset, iree_async_span_ptr(op_payload->values[i]),
           op_payload->values[i].length);
    offset = next_offset;
  }
  return status;
}

static iree_status_t iree_hal_remote_make_span_payload_writer(
    iree_async_span_list_t* op_payload,
    iree_hal_remote_queue_payload_writer_t* out_payload_writer) {
  iree_host_size_t payload_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_payload_total_length(*op_payload, &payload_length));
  out_payload_writer->write = iree_hal_remote_write_payload_spans;
  out_payload_writer->user_data = op_payload;
  out_payload_writer->payload_length = payload_length;
  return iree_ok_status();
}

// Forward declaration: submit_queue_op and the deferred callback are
// mutually recursive.
static iree_status_t iree_hal_remote_client_device_submit_queue_op_writer(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* required_wait_frontier,
    iree_hal_remote_queue_payload_writer_t payload_writer,
    iree_hal_remote_queue_resource_list_t resource_list, uint64_t* out_epoch);

static void iree_hal_remote_deferred_submit_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_deferred_submit_t* deferred =
      (iree_hal_remote_deferred_submit_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unpack the trailing layout.
  uint8_t* base = (uint8_t*)deferred;
  iree_hal_semaphore_t** wait_semaphores =
      (iree_hal_semaphore_t**)(base + deferred->wait_semaphores_offset);
  uint64_t* wait_values = (uint64_t*)(base + deferred->wait_values_offset);
  iree_hal_semaphore_t** signal_semaphores =
      (iree_hal_semaphore_t**)(base + deferred->signal_semaphores_offset);
  uint64_t* signal_values = (uint64_t*)(base + deferred->signal_values_offset);
  iree_async_frontier_entry_t* required_wait_entries =
      (iree_async_frontier_entry_t*)(base +
                                     deferred->required_wait_entries_offset);
  iree_hal_resource_t** resources =
      (iree_hal_resource_t**)(base + deferred->resources_offset);
  uint8_t* payload_data = base + deferred->payload_offset;

  if (iree_status_is_ok(status)) {
    // Re-invoke submit. The previously-gated semaphore is now satisfied
    // and will be skipped by build_wait_frontier.
    iree_hal_semaphore_list_t wait_list = {
        .count = deferred->wait_count,
        .semaphores = wait_semaphores,
        .payload_values = wait_values,
    };
    iree_hal_semaphore_list_t signal_list = {
        .count = deferred->signal_count,
        .semaphores = signal_semaphores,
        .payload_values = signal_values,
    };
    iree_hal_remote_frontier_storage_t required_wait_frontier_storage;
    iree_async_frontier_t* required_wait_frontier = NULL;
    if (deferred->required_wait_entry_count > 0) {
      iree_async_frontier_initialize(
          iree_async_fixed_frontier_as_frontier(
              &required_wait_frontier_storage),
          (uint8_t)deferred->required_wait_entry_count);
      memcpy(required_wait_frontier_storage.entries, required_wait_entries,
             deferred->required_wait_entry_count *
                 sizeof(iree_async_frontier_entry_t));
      required_wait_frontier = iree_async_fixed_frontier_as_frontier(
          &required_wait_frontier_storage);
    }
    iree_async_span_t payload_span =
        iree_async_span_from_ptr(payload_data, deferred->payload_length);
    iree_async_span_list_t payload =
        deferred->payload_length > 0
            ? iree_async_span_list_make(&payload_span, 1)
            : iree_async_span_list_empty();
    iree_hal_remote_queue_payload_writer_t payload_writer = {
        .write = iree_hal_remote_write_payload_spans,
        .user_data = &payload,
        .payload_length = deferred->payload_length,
    };
    iree_hal_remote_resource_ptr_list_t resource_ptr_list = {
        .values = resources,
        .count = deferred->resource_count,
    };
    iree_hal_remote_queue_resource_list_t resource_list = {
        .write = iree_hal_remote_write_resource_ptrs,
        .user_data = &resource_ptr_list,
        .resource_count = deferred->resource_count,
    };
    status = iree_hal_remote_client_device_submit_queue_op_writer(
        deferred->device, wait_list, signal_list, required_wait_frontier,
        payload_writer, resource_list, /*out_epoch=*/NULL);
  }

  if (!iree_status_is_ok(status)) {
    // Gate failed or re-submit failed. Fail all signal semaphores.
    for (iree_host_size_t i = 0; i < deferred->signal_count; ++i) {
      iree_hal_semaphore_fail(signal_semaphores[i], iree_status_clone(status));
    }
    iree_status_ignore(status);
  }

  // Release retained semaphores.
  for (iree_host_size_t i = 0; i < deferred->wait_count; ++i) {
    iree_hal_semaphore_release(wait_semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < deferred->signal_count; ++i) {
    iree_hal_semaphore_release(signal_semaphores[i]);
  }
  iree_hal_remote_release_resources(deferred->resource_count, resources);
  iree_allocator_free(deferred->host_allocator, deferred);
  IREE_TRACE_ZONE_END(z0);
}

// Creates a deferred submit context and registers a timepoint on |gate|.
// Deep-copies the wait/signal semaphore lists and op payload. Returns OK
// immediately; the actual COMMAND send happens when the gate fires.
static iree_status_t iree_hal_remote_client_device_defer_submit(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* required_wait_frontier,
    iree_hal_remote_queue_payload_writer_t payload_writer,
    iree_hal_remote_queue_resource_list_t resource_list,
    iree_hal_semaphore_t* gate, uint64_t gate_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_host_size_t required_wait_entries_offset = 0;
  iree_host_size_t resources_offset = 0;
  iree_host_size_t payload_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_deferred_submit_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(wait_semaphore_list.count,
                                iree_hal_semaphore_t*, 1,
                                &wait_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(wait_semaphore_list.count, uint64_t, 1,
                                &wait_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(signal_semaphore_list.count,
                                iree_hal_semaphore_t*, 1,
                                &signal_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signal_semaphore_list.count, uint64_t, 1,
                                &signal_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          required_wait_frontier ? required_wait_frontier->entry_count : 0,
          iree_async_frontier_entry_t, 1, &required_wait_entries_offset),
      IREE_STRUCT_FIELD_ALIGNED(resource_list.resource_count,
                                iree_hal_resource_t*, 1, &resources_offset),
      IREE_STRUCT_FIELD_ALIGNED(payload_writer.payload_length, uint8_t, 1,
                                &payload_offset));

  iree_hal_remote_deferred_submit_t* deferred = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(device->host_allocator, total_size,
                                   (void**)&deferred);
  }

  iree_hal_semaphore_t** wait_semaphores = NULL;
  uint64_t* wait_values = NULL;
  iree_hal_semaphore_t** signal_semaphores = NULL;
  uint64_t* signal_values = NULL;
  bool semaphores_retained = false;
  iree_hal_resource_t** retained_resources = NULL;
  bool resources_retained = false;
  if (iree_status_is_ok(status)) {
    memset(deferred, 0, sizeof(*deferred));
    deferred->device = device;
    deferred->host_allocator = device->host_allocator;
    deferred->wait_count = wait_semaphore_list.count;
    deferred->signal_count = signal_semaphore_list.count;
    deferred->required_wait_entry_count =
        required_wait_frontier ? required_wait_frontier->entry_count : 0;
    deferred->resource_count = resource_list.resource_count;
    deferred->wait_semaphores_offset = wait_semaphores_offset;
    deferred->wait_values_offset = wait_values_offset;
    deferred->signal_semaphores_offset = signal_semaphores_offset;
    deferred->signal_values_offset = signal_values_offset;
    deferred->required_wait_entries_offset = required_wait_entries_offset;
    deferred->resources_offset = resources_offset;
    deferred->payload_offset = payload_offset;
    deferred->payload_length = payload_writer.payload_length;

    uint8_t* base = (uint8_t*)deferred;
    wait_semaphores = (iree_hal_semaphore_t**)(base + wait_semaphores_offset);
    wait_values = (uint64_t*)(base + wait_values_offset);
    signal_semaphores =
        (iree_hal_semaphore_t**)(base + signal_semaphores_offset);
    signal_values = (uint64_t*)(base + signal_values_offset);
    iree_async_frontier_entry_t* required_wait_entries =
        (iree_async_frontier_entry_t*)(base + required_wait_entries_offset);
    retained_resources = (iree_hal_resource_t**)(base + resources_offset);
    uint8_t* payload_target = base + payload_offset;

    status = payload_writer.write(
        payload_writer.user_data,
        iree_make_byte_span(payload_target, payload_writer.payload_length));

    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
        wait_semaphores[i] = wait_semaphore_list.semaphores[i];
        iree_hal_semaphore_retain(wait_semaphores[i]);
        wait_values[i] = wait_semaphore_list.payload_values[i];
      }
      for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
        signal_semaphores[i] = signal_semaphore_list.semaphores[i];
        iree_hal_semaphore_retain(signal_semaphores[i]);
        signal_values[i] = signal_semaphore_list.payload_values[i];
      }
      semaphores_retained = true;
      if (resource_list.resource_count > 0) {
        status =
            resource_list.write(resource_list.user_data, retained_resources);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_retain_resources(resource_list.resource_count,
                                         retained_resources);
        resources_retained = true;
      }
      if (iree_status_is_ok(status)) {
        if (deferred->required_wait_entry_count > 0) {
          memcpy(required_wait_entries, required_wait_frontier->entries,
                 deferred->required_wait_entry_count *
                     sizeof(iree_async_frontier_entry_t));
        }

        deferred->timepoint.callback = iree_hal_remote_deferred_submit_callback;
        deferred->timepoint.user_data = deferred;
        status = iree_async_semaphore_acquire_timepoint(
            (iree_async_semaphore_t*)gate, gate_value, &deferred->timepoint);
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    if (semaphores_retained) {
      iree_hal_semaphore_list_t retained_wait_list = {
          .count = wait_semaphore_list.count,
          .semaphores = wait_semaphores,
          .payload_values = wait_values,
      };
      iree_hal_semaphore_list_release(retained_wait_list);
      iree_hal_semaphore_list_t retained_signal_list = {
          .count = signal_semaphore_list.count,
          .semaphores = signal_semaphores,
          .payload_values = signal_values,
      };
      iree_hal_semaphore_list_release(retained_signal_list);
    }
    if (resources_retained) {
      iree_hal_remote_release_resources(resource_list.resource_count,
                                        retained_resources);
    }
    if (deferred) {
      iree_allocator_free(device->host_allocator, deferred);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Common submission path
//===----------------------------------------------------------------------===//

// Common submission path for all queue operations. Handles wait frontier
// encoding, host/cross-device gates, epoch assignment, transport submission,
// signal waiter registration, and error-path semaphore failure.
static iree_status_t iree_hal_remote_client_device_submit_queue_op_writer(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* required_wait_frontier,
    iree_hal_remote_queue_payload_writer_t payload_writer,
    iree_hal_remote_queue_resource_list_t resource_list, uint64_t* out_epoch) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_epoch) *out_epoch = 0;

  iree_async_frontier_entry_t wait_entries[16];
  iree_host_size_t wait_entry_count = 0;
  iree_hal_semaphore_t* gate = NULL;
  uint64_t gate_value = 0;
  iree_status_t status = iree_hal_remote_client_device_build_wait_frontier(
      wait_semaphore_list, wait_entries, IREE_ARRAYSIZE(wait_entries),
      &wait_entry_count, &gate, &gate_value);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_merge_wait_frontier(
        wait_entries, IREE_ARRAYSIZE(wait_entries), &wait_entry_count,
        required_wait_frontier);
  }

  bool deferred = false;
  if (iree_status_is_ok(status) && gate) {
    deferred = true;
    status = iree_hal_remote_client_device_defer_submit(
        device, wait_semaphore_list, signal_semaphore_list,
        required_wait_frontier, payload_writer, resource_list, gate,
        gate_value);
  }

  uint64_t epoch = 0;
  if (iree_status_is_ok(status) && !deferred) {
    iree_hal_remote_pending_signal_batch_t* signal_batch = NULL;
    status = iree_hal_remote_client_prepare_signal_waiters(
        device, signal_semaphore_list, &signal_batch);

    iree_net_queue_channel_t* queue_channel = NULL;
    iree_net_queue_channel_command_reservation_t reservation;
    if (iree_status_is_ok(status)) {
      queue_channel = (iree_net_queue_channel_t*)iree_atomic_load(
          &device->queue_channel, iree_memory_order_acquire);
      if (!queue_channel) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "queue channel not available");
      } else {
        status = iree_net_queue_channel_begin_command(
            queue_channel, /*stream_id=*/0, (uint8_t)wait_entry_count,
            /*signal_frontier_entry_count=*/1, payload_writer.payload_length,
            &reservation);
      }
    }
    if (iree_status_is_ok(status)) {
      if (wait_entry_count > 0) {
        memcpy(reservation.wait_entries, wait_entries,
               wait_entry_count * sizeof(iree_async_frontier_entry_t));
      }
      status = payload_writer.write(payload_writer.user_data,
                                    reservation.command_payload);
      if (!iree_status_is_ok(status)) {
        iree_net_queue_channel_abort_command(queue_channel, &reservation);
      }
    }
    if (iree_status_is_ok(status)) {
      // Epoch assignment occurs only after transport admission and payload
      // construction. Deferred callbacks and application threads may race.
      epoch = (uint64_t)iree_atomic_fetch_add(&device->next_submission_epoch, 1,
                                              iree_memory_order_relaxed);
      reservation.signal_entries[0] = (iree_async_frontier_entry_t){
          .axis = device->remote_queue_axis,
          .epoch = epoch,
      };
      status = iree_hal_remote_client_register_signal_waiters(
          device, signal_batch, device->remote_queue_axis, epoch);
      signal_batch = NULL;
      if (iree_status_is_ok(status)) {
        status =
            iree_net_queue_channel_commit_command(queue_channel, &reservation);
      } else {
        iree_net_queue_channel_abort_command(queue_channel, &reservation);
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_remote_client_device_fail(device, iree_status_clone(status));
      } else if (out_epoch) {
        *out_epoch = epoch;
      }
    }
    iree_hal_remote_client_discard_signal_waiters(signal_batch);
  }

  if (!iree_status_is_ok(status) && signal_semaphore_list.count > 0) {
    // Fail ALL signal semaphores, not just the registered ones. If
    // register_signal_waiters failed partway through, unregistered
    // semaphores would hang forever. Failing an already-failed semaphore
    // is a no-op (monotonic failure).
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_device_submit_queue_op_resources(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* required_wait_frontier,
    iree_async_span_list_t op_payload, iree_host_size_t resource_count,
    iree_hal_resource_t* const* resources, uint64_t* out_epoch) {
  iree_hal_remote_queue_payload_writer_t payload_writer;
  iree_status_t status =
      iree_hal_remote_make_span_payload_writer(&op_payload, &payload_writer);
  iree_hal_remote_resource_ptr_list_t resource_ptr_list;
  iree_hal_remote_queue_resource_list_t resource_list =
      iree_hal_remote_make_resource_ptr_list(resources, resource_count,
                                             &resource_ptr_list);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_submit_queue_op_writer(
        device, wait_semaphore_list, signal_semaphore_list,
        required_wait_frontier, payload_writer, resource_list, out_epoch);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_submit_queue_op(
    iree_hal_remote_client_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* required_wait_frontier,
    iree_async_span_list_t op_payload, uint64_t* out_epoch) {
  iree_hal_remote_queue_payload_writer_t payload_writer;
  iree_status_t status =
      iree_hal_remote_make_span_payload_writer(&op_payload, &payload_writer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_submit_queue_op_writer(
        device, wait_semaphore_list, signal_semaphore_list,
        required_wait_frontier, payload_writer,
        iree_hal_remote_empty_resource_list(), out_epoch);
  }
  return status;
}

// Resolves a caller-provided affinity set to the single logical remote stream
// used for proxy buffer placement. The protocol currently exposes one ordered
// remote queue axis, so an unconstrained or multi-queue request is represented
// locally as queue 0 until the wire format grows per-operation queue routing.
static iree_hal_queue_affinity_t iree_hal_remote_client_resolve_queue_affinity(
    iree_hal_queue_affinity_t operation_affinity,
    iree_hal_queue_affinity_t buffer_affinity) {
  iree_hal_queue_affinity_t affinity =
      buffer_affinity ? buffer_affinity : operation_affinity;
  if (iree_hal_queue_affinity_is_empty(affinity) ||
      iree_hal_queue_affinity_is_any(affinity)) {
    return 1ull;
  }
  return 1ull << iree_hal_queue_affinity_find_first_set(affinity);
}

typedef struct iree_hal_remote_execute_payload_t {
  // Command buffer being executed.
  iree_hal_command_buffer_t* command_buffer;
  // Caller-provided binding table.
  iree_hal_buffer_binding_table_t binding_table;
  // Execute flags copied to the wire payload.
  iree_hal_execute_flags_t flags;
  // Inline command stream for one-shot command buffers.
  iree_const_byte_span_t command_stream;
  // Byte offset of the binding table in the wire payload.
  iree_host_size_t bindings_offset;
  // Byte offset of the inline command stream in the wire payload.
  iree_host_size_t stream_offset;
  // True when the command stream is inlined instead of referencing a resource.
  bool is_one_shot;
} iree_hal_remote_execute_payload_t;

static iree_status_t iree_hal_remote_write_execute_payload(
    void* user_data, iree_byte_span_t target) {
  iree_hal_remote_execute_payload_t* payload =
      (iree_hal_remote_execute_payload_t*)user_data;
  memset(target.data, 0, target.data_length);

  iree_hal_remote_command_buffer_execute_op_t* op =
      (iree_hal_remote_command_buffer_execute_op_t*)target.data;
  op->header.type = IREE_HAL_REMOTE_QUEUE_OP_COMMAND_BUFFER_EXECUTE;
  if (payload->is_one_shot) {
    op->header.flags = IREE_HAL_REMOTE_EXECUTE_FLAG_INLINE_COMMAND_STREAM;
  } else {
    op->command_buffer_id = iree_hal_remote_client_command_buffer_resource_id(
        payload->command_buffer);
  }
  op->binding_count = (uint16_t)payload->binding_table.count;
  op->execute_flags = payload->flags;

  iree_hal_remote_binding_t* wire_bindings =
      (iree_hal_remote_binding_t*)(target.data + payload->bindings_offset);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < payload->binding_table.count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_binding_t* binding =
        &payload->binding_table.bindings[i];
    if (binding->buffer) {
      status = iree_hal_remote_client_buffer_resolve_wire_range(
          binding->buffer, binding->offset, binding->length,
          &wire_bindings[i].buffer_id, &wire_bindings[i].offset,
          &wire_bindings[i].length);
    } else {
      wire_bindings[i].offset = binding->offset;
      wire_bindings[i].length = binding->length;
    }
  }
  if (iree_status_is_ok(status) && payload->command_stream.data_length > 0) {
    memcpy(target.data + payload->stream_offset, payload->command_stream.data,
           payload->command_stream.data_length);
  }
  return status;
}

static iree_status_t iree_hal_remote_prepare_execute_payload(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_remote_execute_payload_t* out_payload,
    iree_hal_remote_queue_payload_writer_t* out_payload_writer) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->command_buffer = command_buffer;
  out_payload->binding_table = binding_table;
  out_payload->flags = flags;
  out_payload->is_one_shot = iree_all_bits_set(
      command_buffer->mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT);
  out_payload->command_stream =
      out_payload->is_one_shot
          ? iree_hal_remote_client_command_buffer_stream(command_buffer)
          : iree_const_byte_span_empty();

  iree_status_t status = iree_ok_status();
  if (binding_table.count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command buffer execute binding count %" PRIhsz
                              " exceeds wire limit %u",
                              binding_table.count, (unsigned)UINT16_MAX);
  }

  iree_host_size_t payload_length = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_command_buffer_execute_op_t), &payload_length,
        IREE_STRUCT_FIELD(binding_table.count, iree_hal_remote_binding_t,
                          &out_payload->bindings_offset),
        IREE_STRUCT_FIELD(out_payload->command_stream.data_length, uint8_t,
                          &out_payload->stream_offset));
  }

  if (iree_status_is_ok(status)) {
    out_payload_writer->write = iree_hal_remote_write_execute_payload;
    out_payload_writer->user_data = out_payload;
    out_payload_writer->payload_length = payload_length;
  }
  return status;
}

static iree_status_t iree_hal_remote_write_execute_resources(
    void* user_data, iree_hal_resource_t** target_resources) {
  const iree_hal_remote_execute_payload_t* payload =
      (const iree_hal_remote_execute_payload_t*)user_data;
  target_resources[0] = (iree_hal_resource_t*)payload->command_buffer;
  for (iree_host_size_t i = 0; i < payload->binding_table.count; ++i) {
    target_resources[1 + i] =
        (iree_hal_resource_t*)payload->binding_table.bindings[i].buffer;
  }
  return iree_ok_status();
}

typedef struct iree_hal_remote_dispatch_payload_t {
  // Executable containing the dispatch entry point.
  iree_hal_executable_t* executable;
  // Function token within |executable|.
  iree_hal_executable_function_t function;
  // Workgroup configuration copied to the wire payload.
  iree_hal_dispatch_config_t config;
  // Dispatch constants copied after the fixed header.
  iree_const_byte_span_t constants;
  // Caller-provided dispatch binding refs.
  iree_hal_buffer_ref_list_t bindings;
  // Dispatch flags copied to the wire payload.
  iree_hal_dispatch_flags_t flags;
  // Unpadded constant byte length.
  iree_host_size_t constants_size;
  // Constant byte length padded to the binding table alignment.
  iree_host_size_t constants_padded;
  // Byte offset of the binding table in the wire payload.
  iree_host_size_t bindings_offset;
  // Number of 32-bit constants encoded in the wire payload.
  uint16_t constant_count;
  // Number of bindings encoded in the wire payload.
  uint16_t binding_count;
} iree_hal_remote_dispatch_payload_t;

static iree_status_t iree_hal_remote_write_dispatch_payload(
    void* user_data, iree_byte_span_t target) {
  iree_hal_remote_dispatch_payload_t* payload =
      (iree_hal_remote_dispatch_payload_t*)user_data;
  memset(target.data, 0, target.data_length);

  iree_hal_remote_dispatch_op_t* op =
      (iree_hal_remote_dispatch_op_t*)target.data;
  op->header.type = IREE_HAL_REMOTE_QUEUE_OP_DISPATCH;
  op->executable_id =
      iree_hal_remote_client_executable_resource_id(payload->executable);
  op->function_value = payload->function.value;
  memcpy(op->config.workgroup_size, payload->config.workgroup_size,
         sizeof(payload->config.workgroup_size));
  memcpy(op->config.workgroup_count, payload->config.workgroup_count,
         sizeof(payload->config.workgroup_count));
  op->config.dynamic_workgroup_local_memory =
      payload->config.dynamic_workgroup_local_memory;
  op->constant_count = payload->constant_count;
  op->binding_count = payload->binding_count;
  op->dispatch_flags = payload->flags;

  iree_status_t status = iree_ok_status();
  const iree_hal_buffer_ref_t workgroup_count_ref =
      payload->config.workgroup_count_ref;
  if (workgroup_count_ref.buffer) {
    status = iree_hal_remote_client_buffer_resolve_wire_range(
        workgroup_count_ref.buffer, workgroup_count_ref.offset,
        workgroup_count_ref.length, &op->config.workgroup_count_buffer_id,
        &op->config.workgroup_count_offset, &op->config.workgroup_count_length);
  } else {
    op->config.workgroup_count_offset = workgroup_count_ref.offset;
    op->config.workgroup_count_length = workgroup_count_ref.length;
  }
  op->config.workgroup_count_buffer_slot = workgroup_count_ref.buffer_slot;

  uint8_t* constants_data = target.data + sizeof(iree_hal_remote_dispatch_op_t);
  if (payload->constants_size > 0) {
    memcpy(constants_data, payload->constants.data, payload->constants_size);
  }

  iree_hal_remote_binding_t* wire_bindings =
      (iree_hal_remote_binding_t*)(target.data + payload->bindings_offset);
  for (uint16_t i = 0; i < payload->binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* ref = &payload->bindings.values[i];
    if (ref->buffer) {
      status = iree_hal_remote_client_buffer_resolve_wire_range(
          ref->buffer, ref->offset, ref->length, &wire_bindings[i].buffer_id,
          &wire_bindings[i].offset, &wire_bindings[i].length);
    } else {
      wire_bindings[i].offset = ref->offset;
      wire_bindings[i].length = ref->length;
    }
    wire_bindings[i].buffer_slot = ref->buffer_slot;
  }
  return status;
}

static iree_status_t iree_hal_remote_prepare_dispatch_payload(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_remote_dispatch_payload_t* out_payload,
    iree_hal_remote_queue_payload_writer_t* out_payload_writer) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->executable = executable;
  out_payload->function = function;
  out_payload->config = config;
  out_payload->constants = constants;
  out_payload->bindings = bindings;
  out_payload->flags = flags;

  iree_status_t status = iree_ok_status();
  if ((constants.data_length % sizeof(uint32_t)) != 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch constants length %" PRIhsz
                              " is not a multiple of %zu bytes",
                              constants.data_length, sizeof(uint32_t));
  }
  if (iree_status_is_ok(status) &&
      constants.data_length / sizeof(uint32_t) > UINT16_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch constant count %" PRIhsz " exceeds wire limit %u",
        constants.data_length / sizeof(uint32_t), (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && bindings.count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "dispatch binding count %" PRIhsz
                              " exceeds wire limit %u",
                              bindings.count, (unsigned)UINT16_MAX);
  }

  if (iree_status_is_ok(status)) {
    out_payload->constant_count =
        (uint16_t)(constants.data_length / sizeof(uint32_t));
    out_payload->binding_count = (uint16_t)bindings.count;
    out_payload->constants_size = constants.data_length;
  }

  iree_host_size_t constants_offset = 0;
  iree_host_size_t payload_length = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_dispatch_op_t), &payload_length,
        IREE_STRUCT_FIELD(out_payload->constant_count, uint32_t,
                          &constants_offset),
        IREE_STRUCT_FIELD_ALIGNED(out_payload->binding_count,
                                  iree_hal_remote_binding_t, 8,
                                  &out_payload->bindings_offset));
  }
  if (iree_status_is_ok(status)) {
    out_payload->constants_padded =
        out_payload->bindings_offset - constants_offset;
  }

  if (iree_status_is_ok(status)) {
    out_payload_writer->write = iree_hal_remote_write_dispatch_payload;
    out_payload_writer->user_data = out_payload;
    out_payload_writer->payload_length = payload_length;
  }
  return status;
}

static iree_status_t iree_hal_remote_write_dispatch_resources(
    void* user_data, iree_hal_resource_t** target_resources) {
  const iree_hal_remote_dispatch_payload_t* payload =
      (const iree_hal_remote_dispatch_payload_t*)user_data;
  target_resources[0] = (iree_hal_resource_t*)payload->executable;
  target_resources[1] =
      (iree_hal_resource_t*)payload->config.workgroup_count_ref.buffer;
  for (iree_host_size_t i = 0; i < payload->bindings.count; ++i) {
    target_resources[2 + i] =
        (iree_hal_resource_t*)payload->bindings.values[i].buffer;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (!command_buffer) {
    status = iree_hal_remote_client_device_submit_queue_op(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, iree_async_span_list_empty(),
        /*out_epoch=*/NULL);
  } else {
    iree_hal_remote_execute_payload_t payload;
    iree_hal_remote_queue_payload_writer_t payload_writer;
    status = iree_hal_remote_prepare_execute_payload(
        command_buffer, binding_table, flags, &payload, &payload_writer);
    iree_hal_remote_queue_resource_list_t resource_list = {
        .write = iree_hal_remote_write_execute_resources,
        .user_data = &payload,
        .resource_count = 1 + binding_table.count,
    };
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_client_device_submit_queue_op_writer(
          device, wait_semaphore_list, signal_semaphore_list,
          /*required_wait_frontier=*/NULL, payload_writer, resource_list,
          /*out_epoch=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_atomic_wait(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_queue_atomic_wait_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_WAIT;
  op.target.queue_affinity = (uint64_t)queue_affinity;
  op.params.value = params.value;
  op.params.mask = params.mask;
  op.params.flags = (uint32_t)params.flags;
  op.params.width = (uint8_t)params.width;
  op.params.condition = (uint8_t)params.condition;
  op.params.reserved = params.reserved;

  uint64_t target_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &op.target.buffer_id,
      &op.target.offset, &target_length);
  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_atomic_store(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_queue_atomic_store_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_STORE;
  op.target.queue_affinity = (uint64_t)queue_affinity;
  op.params.value = params.value;
  op.params.flags = (uint32_t)params.flags;
  op.params.width = (uint8_t)params.width;
  memcpy(op.params.reserved, params.reserved, sizeof(op.params.reserved));

  uint64_t target_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &op.target.buffer_id,
      &op.target.offset, &target_length);
  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_atomic_rmw(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_queue_atomic_rmw_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_RMW;
  op.target.queue_affinity = (uint64_t)queue_affinity;
  op.params.operand = params.operand;
  op.params.flags = (uint32_t)params.flags;
  op.params.width = (uint8_t)params.width;
  op.params.operation = (uint8_t)params.operation;
  op.params.reserved = params.reserved;

  uint64_t target_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &op.target.buffer_id,
      &op.target.offset, &target_length);
  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_timestamp(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_queue_timestamp_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_QUEUE_TIMESTAMP;
  op.target.queue_affinity = (uint64_t)queue_affinity;
  op.flags = (uint64_t)flags;

  uint64_t target_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset, sizeof(uint64_t), &op.target.buffer_id,
      &op.target.offset, &target_length);
  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_async_frontier_t*
iree_hal_remote_client_make_frontier_from_entries(
    const iree_async_frontier_entry_t* entries, iree_host_size_t entry_count,
    iree_hal_remote_frontier_storage_t* out_storage) {
  if (entry_count == 0) return NULL;
  memset(out_storage, 0, sizeof(*out_storage));
  iree_async_frontier_initialize(
      iree_async_fixed_frontier_as_frontier(out_storage), (uint8_t)entry_count);
  memcpy(out_storage->entries, entries,
         entry_count * sizeof(iree_async_frontier_entry_t));
  return iree_async_fixed_frontier_as_const_frontier(out_storage);
}

static iree_status_t iree_hal_remote_client_device_materialize_pool_reservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation, iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* backing_buffer = NULL;
  iree_status_t status = iree_hal_pool_materialize_reservation(
      pool, params, reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &backing_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_set_backing(buffer, backing_buffer);
  }
  iree_hal_buffer_release(backing_buffer);

  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_buffer_attach_reservation(buffer, pool, reservation);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_materialize_pool_buffer(
    iree_hal_remote_client_device_t* device, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_placement_flags_t placement_flags,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_remote_client_buffer_create_unbacked(
      device, &params, allocation_size, placement_flags, device->host_allocator,
      &buffer);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_materialize_pool_reservation(
        pool, params, reservation, buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

typedef struct iree_hal_remote_pool_alloca_retry_t {
  // Wait operation armed on the pool notification.
  iree_async_notification_wait_operation_t wait_op;
  // Device submitting the eventual queue allocation barrier.
  iree_hal_remote_client_device_t* device;
  // Allocator used for cloned lists and retry state.
  iree_allocator_t host_allocator;
  // Pool retried until a reservation is available.
  iree_hal_pool_t* pool;
  // Logical buffer returned to the caller while retry is pending.
  iree_hal_buffer_t* buffer;
  // Buffer parameters used for materialization.
  iree_hal_buffer_params_t params;
  // Requested allocation size in bytes.
  iree_device_size_t allocation_size;
  // Placement flags for the logical buffer.
  iree_hal_buffer_placement_flags_t placement_flags;
  // Queue alloca flags controlling hidden wait behavior.
  iree_hal_alloca_flags_t flags;
  // Pool reservation flags derived from |flags|.
  iree_hal_pool_reserve_flags_t reserve_flags;
  // Cloned user wait semaphores for the final barrier.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Cloned user signal semaphores for the final barrier.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_remote_pool_alloca_retry_t;

static void iree_hal_remote_pool_alloca_retry_destroy(
    iree_hal_remote_pool_alloca_retry_t* retry) {
  if (!retry) return;
  iree_hal_semaphore_list_free(retry->wait_semaphore_list,
                               retry->host_allocator);
  iree_hal_semaphore_list_free(retry->signal_semaphore_list,
                               retry->host_allocator);
  iree_hal_buffer_release(retry->buffer);
  iree_hal_pool_release(retry->pool);
  iree_allocator_free(retry->host_allocator, retry);
}

static iree_status_t iree_hal_remote_pool_alloca_retry_try(
    iree_hal_remote_pool_alloca_retry_t** inout_retry);

static void iree_hal_remote_pool_alloca_retry_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_remote_pool_alloca_retry_t* retry =
      (iree_hal_remote_pool_alloca_retry_t*)user_data;

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_pool_alloca_retry_try(&retry);
  }
  if (retry) {
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(retry->signal_semaphore_list,
                                   iree_status_clone(status));
    }
    iree_hal_remote_pool_alloca_retry_destroy(retry);
  }
  iree_status_free(status);
}

static iree_status_t iree_hal_remote_pool_alloca_retry_wait(
    iree_hal_remote_pool_alloca_retry_t* retry,
    iree_async_notification_t* notification, uint32_t wait_token) {
  iree_async_notification_wait_operation_t* wait_op = &retry->wait_op;
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_remote_pool_alloca_retry_callback, retry);
  wait_op->notification = notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait_token;
  return iree_async_proactor_submit_one(retry->device->proactor,
                                        &wait_op->base);
}

static iree_status_t iree_hal_remote_pool_alloca_retry_submit(
    iree_hal_remote_pool_alloca_retry_t* retry,
    const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* acquire_info,
    iree_hal_pool_acquire_result_t acquire_result) {
  const iree_async_frontier_t* reservation_failure_frontier =
      acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
          ? acquire_info->wait_frontier
          : NULL;

  iree_status_t status =
      iree_hal_remote_client_device_materialize_pool_reservation(
          retry->pool, retry->params, reservation, retry->buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_submit_queue_op(
        retry->device, retry->wait_semaphore_list, retry->signal_semaphore_list,
        reservation_failure_frontier, iree_async_span_list_empty(),
        /*out_epoch=*/NULL);
  }
  if (!iree_status_is_ok(status)) {
    if (iree_hal_remote_client_buffer_has_reservation(retry->buffer)) {
      iree_hal_remote_client_buffer_release_reservation(
          retry->buffer, reservation_failure_frontier);
    } else {
      iree_hal_pool_release_reservation(retry->pool, reservation,
                                        reservation_failure_frontier);
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_pool_alloca_retry_on_acquire(
    iree_hal_remote_pool_alloca_retry_t** inout_retry,
    const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* acquire_info,
    iree_hal_pool_acquire_result_t acquire_result,
    iree_async_notification_t* notification, uint32_t wait_token) {
  iree_hal_remote_pool_alloca_retry_t* retry = *inout_retry;
  iree_status_t status = iree_ok_status();
  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      status = iree_hal_remote_pool_alloca_retry_submit(
          retry, reservation, acquire_info, acquire_result);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      if (iree_all_bits_set(retry->flags,
                            IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
        status = iree_hal_remote_pool_alloca_retry_submit(
            retry, reservation, acquire_info, acquire_result);
      } else {
        iree_hal_pool_release_reservation(retry->pool, reservation,
                                          acquire_info->wait_frontier);
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "queue_alloca recycled pool memory requires "
                             "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
      }
      break;
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET: {
      if (notification) {
        status = iree_hal_remote_pool_alloca_retry_wait(retry, notification,
                                                        wait_token);
        if (iree_status_is_ok(status)) {
          *inout_retry = NULL;
        }
      } else {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "queue_alloca exhausted pool did not provide a notification");
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "unrecognized pool acquire result %u",
                                acquire_result);
      break;
  }
  return status;
}

static iree_status_t iree_hal_remote_pool_alloca_retry_try(
    iree_hal_remote_pool_alloca_retry_t** inout_retry) {
  iree_hal_remote_pool_alloca_retry_t* retry = *inout_retry;
  iree_async_notification_t* notification =
      iree_hal_pool_notification(retry->pool);
  uint32_t wait_token = 0;
  if (notification) {
    wait_token = iree_async_notification_begin_observe(notification);
  }

  iree_hal_pool_reservation_t reservation;
  memset(&reservation, 0, sizeof(reservation));
  iree_hal_pool_acquire_info_t acquire_info;
  memset(&acquire_info, 0, sizeof(acquire_info));
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  iree_status_t status = iree_hal_pool_acquire_reservation(
      retry->pool, retry->allocation_size,
      retry->params.min_alignment ? retry->params.min_alignment : 1,
      /*requester_frontier=*/NULL, retry->reserve_flags, &reservation,
      &acquire_info, &acquire_result);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_pool_alloca_retry_on_acquire(
        inout_retry, &reservation, &acquire_info, acquire_result, notification,
        wait_token);
  }
  if (notification) {
    iree_async_notification_end_observe(notification);
  }
  return status;
}

static iree_status_t iree_hal_remote_pool_alloca_retry_create(
    iree_hal_remote_client_device_t* device, iree_hal_pool_t* pool,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_remote_client_buffer_create_unbacked(
      device, &params, allocation_size, placement_flags, device->host_allocator,
      &buffer);

  iree_hal_remote_pool_alloca_retry_t* retry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(device->host_allocator, sizeof(*retry),
                                   (void**)&retry);
  }
  if (iree_status_is_ok(status)) {
    memset(retry, 0, sizeof(*retry));
    retry->device = device;
    retry->host_allocator = device->host_allocator;
    retry->pool = pool;
    iree_hal_pool_retain(pool);
    retry->buffer = buffer;
    iree_hal_buffer_retain(buffer);
    retry->params = params;
    retry->allocation_size = allocation_size;
    retry->placement_flags = placement_flags;
    retry->flags = flags;
    retry->reserve_flags = reserve_flags;
    status = iree_hal_semaphore_list_clone(&wait_semaphore_list,
                                           device->host_allocator,
                                           &retry->wait_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_semaphore_list,
                                           device->host_allocator,
                                           &retry->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_pool_alloca_retry_try(&retry);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    buffer = NULL;
  }
  iree_hal_remote_pool_alloca_retry_destroy(retry);
  iree_hal_buffer_release(buffer);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  const iree_hal_queue_affinity_t effective_queue_affinity =
      iree_hal_remote_client_resolve_queue_affinity(queue_affinity,
                                                    params.queue_affinity);
  params.queue_affinity = effective_queue_affinity;
  iree_hal_buffer_placement_flags_t placement_flags =
      IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS;
  if (iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    placement_flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME;
  }

  if (IREE_UNLIKELY(pool)) {
    iree_async_frontier_entry_t wait_entries[16];
    iree_host_size_t wait_entry_count = 0;
    iree_hal_semaphore_t* gate = NULL;
    uint64_t gate_value = 0;
    iree_status_t status = iree_hal_remote_client_device_build_wait_frontier(
        wait_semaphore_list, wait_entries, IREE_ARRAYSIZE(wait_entries),
        &wait_entry_count, &gate, &gate_value);
    (void)gate;
    (void)gate_value;

    iree_hal_remote_frontier_storage_t requester_frontier_storage;
    const iree_async_frontier_t* requester_frontier = NULL;
    if (iree_status_is_ok(status)) {
      requester_frontier = iree_hal_remote_client_make_frontier_from_entries(
          wait_entries, wait_entry_count, &requester_frontier_storage);
    }

    iree_hal_pool_reserve_flags_t reserve_flags =
        IREE_HAL_POOL_RESERVE_FLAG_NONE;
    if (iree_all_bits_set(flags,
                          IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
      reserve_flags |= IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER;
    }

    iree_hal_pool_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    iree_hal_pool_acquire_info_t acquire_info;
    memset(&acquire_info, 0, sizeof(acquire_info));
    iree_hal_pool_acquire_result_t acquire_result =
        IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    if (iree_status_is_ok(status)) {
      status = iree_hal_pool_acquire_reservation(
          pool, allocation_size,
          params.min_alignment ? params.min_alignment : 1, requester_frontier,
          reserve_flags, &reservation, &acquire_info, &acquire_result);
    }

    bool reservation_acquired = false;
    bool retry_required = false;
    iree_hal_buffer_t* buffer = NULL;
    const iree_async_frontier_t* reservation_failure_frontier = NULL;
    if (iree_status_is_ok(status)) {
      switch (acquire_result) {
        case IREE_HAL_POOL_ACQUIRE_OK:
        case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
          reservation_acquired = true;
          break;
        case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
          reservation_acquired = true;
          reservation_failure_frontier = acquire_info.wait_frontier;
          if (!iree_all_bits_set(
                  flags, IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
            status = iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "queue_alloca recycled pool memory requires "
                "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
          }
          break;
        case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
        case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
          if (iree_all_bits_set(
                  flags, IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
            retry_required = true;
          } else {
            status = iree_make_status(
                IREE_STATUS_RESOURCE_EXHAUSTED,
                "queue_alloca explicit pool could not reserve %" PRIdsz
                " bytes (result %u)",
                allocation_size, acquire_result);
          }
          break;
        default:
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "unrecognized pool acquire result %u",
                                    acquire_result);
          break;
      }
    }

    if (iree_status_is_ok(status) && retry_required) {
      status = iree_hal_remote_pool_alloca_retry_create(
          device, pool, wait_semaphore_list, signal_semaphore_list, params,
          allocation_size, flags, reserve_flags, placement_flags, out_buffer);
    } else if (iree_status_is_ok(status)) {
      status = iree_hal_remote_client_device_materialize_pool_buffer(
          device, pool, params, allocation_size, placement_flags, &reservation,
          &buffer);
    }
    if (iree_status_is_ok(status) && !retry_required) {
      status = iree_hal_remote_client_device_submit_queue_op(
          device, wait_semaphore_list, signal_semaphore_list,
          acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
              ? acquire_info.wait_frontier
              : NULL,
          iree_async_span_list_empty(), /*out_epoch=*/NULL);
    }

    if (iree_status_is_ok(status) && !retry_required) {
      *out_buffer = buffer;
    } else {
      if (buffer) {
        iree_hal_remote_client_buffer_release_reservation(
            buffer, reservation_failure_frontier);
        iree_hal_buffer_release(buffer);
      } else if (reservation_acquired) {
        iree_hal_pool_release_reservation(pool, &reservation,
                                          reservation_failure_frontier);
      }
    }

    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Assign a unique provisional resource ID for this allocation.
  uint16_t generation = (uint16_t)iree_atomic_fetch_add(
      &device->next_provisional_generation, 1, iree_memory_order_relaxed);
  iree_hal_remote_resource_id_t provisional_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, generation);

  // Create a buffer proxy with the provisional ID. The buffer is immediately
  // usable in subsequent queue ops (frontier ordering guarantees that the
  // server processes the alloca before any op that references this buffer).
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_remote_client_buffer_create(
      device, provisional_id, &params, allocation_size,
      /*byte_length=*/allocation_size, placement_flags, device->host_allocator,
      &buffer);

  // Register in the provisional buffer table so on_advance can resolve the
  // provisional_id to the server's canonical ID.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_register_provisional(
        device, provisional_id, buffer);
  }

  // Build and send the COMMAND frame.
  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_alloca_op_t op;
    memset(&op, 0, sizeof(op));
    op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_ALLOCA;
    op.pool = 0;
    op.params.usage = params.usage;
    op.params.access = (uint16_t)params.access;
    op.params.type = params.type;
    op.params.queue_affinity = params.queue_affinity;
    op.params.min_alignment = (uint64_t)params.min_alignment;
    op.allocation_size = (uint64_t)allocation_size;
    op.alloca_flags = flags;
    op.provisional_buffer_id = provisional_id;

    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_buffer_dealloca_op_t op;
  memset(&op, 0, sizeof(op));
  uint64_t dealloca_epoch = 0;
  iree_status_t status = iree_ok_status();
  if (iree_hal_remote_client_buffer_has_reservation(buffer)) {
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, iree_async_span_list_empty(),
        IREE_ARRAYSIZE(resources), resources, &dealloca_epoch);
  } else {
    op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_DEALLOCA;
    uint64_t buffer_offset = 0;
    status = iree_hal_remote_client_buffer_resolve_wire_ref(
        buffer, 0, &op.buffer_id, &buffer_offset);
    op.dealloca_flags = flags;

    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    if (iree_status_is_ok(status)) {
      iree_hal_resource_t* resources[1] = {
          (iree_hal_resource_t*)buffer,
      };
      status = iree_hal_remote_client_device_submit_queue_op_resources(
          device, wait_semaphore_list, signal_semaphore_list,
          /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
          resources, &dealloca_epoch);
    }
  }
  if (iree_status_is_ok(status) && dealloca_epoch != 0) {
    iree_async_single_frontier_t death_frontier;
    iree_async_single_frontier_initialize(
        &death_frontier, device->remote_queue_axis, dealloca_epoch);
    iree_hal_remote_client_buffer_release_reservation(
        buffer, iree_async_single_frontier_as_const_frontier(&death_frontier));
    iree_hal_remote_client_buffer_mark_deallocated(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_fill_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_FILL;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset, length, &op.target_buffer_id,
      &op.target_offset, &op.length);
  op.pattern_length = (uint8_t)pattern_length;
  op.fill_flags = flags;
  // Copy pattern (1, 2, or 4 bytes) into the 4-byte field, zero-extended.
  memcpy(&op.pattern, pattern, pattern_length);

  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_update_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_UPDATE;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset, length, &op.target_buffer_id,
      &op.target_offset, &op.length);
  op.update_flags = flags;

  // The inline source data follows the op struct. Build two spans: the op
  // header and the source data. Padding to 8-byte alignment is handled by
  // the queue channel's frame builder.
  if (iree_status_is_ok(status)) {
    iree_async_span_t spans[2] = {
        iree_async_span_from_ptr(&op, sizeof(op)),
        iree_async_span_from_ptr(
            (void*)((const uint8_t*)source_buffer + source_offset),
            (iree_host_size_t)op.length),
    };
    iree_async_span_list_t payload = {spans, 2};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  iree_hal_remote_buffer_copy_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_BUFFER_COPY;
  uint64_t source_length = 0;
  uint64_t target_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      source_buffer, source_offset, length, &op.source_buffer_id,
      &op.source_offset, &source_length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_resolve_wire_range(
        target_buffer, target_offset, length, &op.target_buffer_id,
        &op.target_offset, &target_length);
  }
  op.length = length == IREE_HAL_WHOLE_BUFFER
                  ? iree_min(source_length, target_length)
                  : source_length;
  op.copy_flags = flags;

  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[2] = {
        (iree_hal_resource_t*)source_buffer,
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_device_queue_read_client_file_now(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_read_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  const uint64_t source_length = (uint64_t)length;
  uint64_t source_end = source_offset;
  const bool source_range_overflow = source_offset > UINT64_MAX - source_length;
  if (!source_range_overflow) {
    source_end = source_offset + source_length;
  }

  if (source_range_overflow || source_offset > source_file_view->length ||
      source_length > source_file_view->length - source_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "remote queue_read source range [%" PRIu64 ", %" PRIu64
        ") exceeds client file length %" PRIu64,
        source_offset, source_end, source_file_view->length);
  }

  if (source_file_view->kind ==
          IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION &&
      length <= IREE_HAL_REMOTE_CLIENT_FILE_INLINE_UPDATE_MAX_LENGTH) {
    return iree_hal_remote_client_device_queue_update(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_file_view->host_allocation.data, (iree_host_size_t)source_offset,
        target_buffer, target_offset, length, (iree_hal_update_flags_t)flags);
  }

  iree_hal_remote_client_file_read_op_t op;
  memset(&op, 0, sizeof(op));
  op.header.type = IREE_HAL_REMOTE_QUEUE_OP_CLIENT_FILE_READ;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      target_buffer, target_offset, length, &op.target_buffer_id,
      &op.target_offset, &op.length);

  uint64_t transfer_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_bulk_begin_file_read(
        device, source_file, source_file_view, source_offset, op.length,
        &transfer_id);
  }
  op.transfer_id = transfer_id;
  op.read_flags = flags;
  if (iree_status_is_ok(status)) {
    iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
    iree_async_span_list_t payload = {&span, 1};
    iree_hal_resource_t* resources[1] = {
        (iree_hal_resource_t*)target_buffer,
    };
    status = iree_hal_remote_client_device_submit_queue_op_resources(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload, IREE_ARRAYSIZE(resources),
        resources, /*out_epoch=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_bulk_upload_file_read(device, transfer_id);
  }
  if (!iree_status_is_ok(status) && transfer_id != 0) {
    iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
  }
  return status;
}

typedef struct iree_hal_remote_deferred_file_read_t {
  // Number of active timepoint callbacks plus the submitter hold.
  iree_atomic_int32_t pending_count;
  // Number of wait timepoints that still need to complete successfully.
  iree_atomic_int32_t remaining_wait_count;
  // Whether a wait or submit failure has already been reported to signals.
  iree_atomic_int32_t failure_reported;
  // Borrowed device pointer; the device outlives deferred queue work.
  iree_hal_remote_client_device_t* device;
  // Allocator used for this deferred read allocation.
  iree_allocator_t host_allocator;
  // Queue affinity supplied by the caller.
  iree_hal_queue_affinity_t queue_affinity;
  // Retained source file backing the client-local bytes.
  iree_hal_file_t* source_file;
  // Source file view captured when the caller submitted the read.
  iree_hal_remote_client_file_view_t source_file_view;
  // Byte offset in the source file.
  uint64_t source_offset;
  // Retained target buffer receiving the queued read contents.
  iree_hal_buffer_t* target_buffer;
  // Byte offset in the target buffer.
  iree_device_size_t target_offset;
  // Number of bytes to read.
  iree_device_size_t length;
  // Queue read flags supplied by the caller.
  iree_hal_read_flags_t flags;
  // Cloned wait list retained while timepoints are active.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Cloned signal list used by the deferred queue operation or failure path.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Offset of trailing iree_async_semaphore_timepoint_t storage.
  iree_host_size_t timepoints_offset;
} iree_hal_remote_deferred_file_read_t;

static iree_async_semaphore_timepoint_t*
iree_hal_remote_deferred_file_read_timepoints(
    iree_hal_remote_deferred_file_read_t* deferred) {
  return (iree_async_semaphore_timepoint_t*)((uint8_t*)deferred +
                                             deferred->timepoints_offset);
}

static void iree_hal_remote_deferred_file_read_destroy(
    iree_hal_remote_deferred_file_read_t* deferred) {
  if (!deferred) return;
  if (!iree_hal_semaphore_list_is_empty(deferred->wait_semaphore_list)) {
    iree_hal_semaphore_list_free(deferred->wait_semaphore_list,
                                 deferred->host_allocator);
  }
  if (!iree_hal_semaphore_list_is_empty(deferred->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(deferred->signal_semaphore_list,
                                 deferred->host_allocator);
  }
  iree_hal_buffer_release(deferred->target_buffer);
  iree_hal_file_release(deferred->source_file);
  iree_allocator_free(deferred->host_allocator, deferred);
}

static void iree_hal_remote_deferred_file_read_release(
    iree_hal_remote_deferred_file_read_t* deferred) {
  if (iree_atomic_fetch_sub(&deferred->pending_count, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_remote_deferred_file_read_destroy(deferred);
  }
}

static void iree_hal_remote_deferred_file_read_report_failure(
    iree_hal_remote_deferred_file_read_t* deferred, iree_status_t status) {
  int32_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &deferred->failure_reported, &expected, 1, iree_memory_order_acq_rel,
          iree_memory_order_acquire)) {
    iree_hal_semaphore_list_fail(deferred->signal_semaphore_list, status);
  } else {
    iree_status_ignore(status);
  }
}

static void iree_hal_remote_deferred_file_read_submit(
    iree_hal_remote_deferred_file_read_t* deferred) {
  if (iree_atomic_load(&deferred->failure_reported,
                       iree_memory_order_acquire)) {
    return;
  }

  iree_status_t status =
      iree_hal_remote_client_device_queue_read_client_file_now(
          (iree_hal_device_t*)deferred->device, deferred->queue_affinity,
          iree_hal_semaphore_list_empty(), deferred->signal_semaphore_list,
          deferred->source_file, &deferred->source_file_view,
          deferred->source_offset, deferred->target_buffer,
          deferred->target_offset, deferred->length, deferred->flags);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_deferred_file_read_report_failure(deferred, status);
  }
}

static void iree_hal_remote_deferred_file_read_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_deferred_file_read_t* deferred =
      (iree_hal_remote_deferred_file_read_t*)user_data;
  (void)timepoint;

  if (iree_status_is_ok(status)) {
    if (iree_atomic_fetch_sub(&deferred->remaining_wait_count, 1,
                              iree_memory_order_acq_rel) == 1) {
      iree_hal_remote_deferred_file_read_submit(deferred);
    }
  } else {
    iree_hal_remote_deferred_file_read_report_failure(deferred, status);
  }
  iree_hal_remote_deferred_file_read_release(deferred);
}

static iree_status_t iree_hal_remote_client_device_defer_queue_read_client_file(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_read_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (wait_semaphore_list.count > (iree_host_size_t)INT32_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "remote deferred queue_read wait count exceeds int32_t range");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t timepoints_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_deferred_file_read_t), &total_size,
        IREE_STRUCT_FIELD_ALIGNED(
            wait_semaphore_list.count, iree_async_semaphore_timepoint_t,
            iree_alignof(iree_async_semaphore_timepoint_t),
            &timepoints_offset));
  }

  iree_hal_remote_deferred_file_read_t* deferred = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(device->host_allocator, total_size,
                                   (void**)&deferred);
  }

  if (iree_status_is_ok(status)) {
    memset(deferred, 0, sizeof(*deferred));
    iree_atomic_store(&deferred->pending_count, 1, iree_memory_order_relaxed);
    iree_atomic_store(&deferred->remaining_wait_count,
                      (int32_t)wait_semaphore_list.count,
                      iree_memory_order_relaxed);
    iree_atomic_store(&deferred->failure_reported, 0,
                      iree_memory_order_relaxed);
    deferred->device = device;
    deferred->host_allocator = device->host_allocator;
    deferred->queue_affinity = queue_affinity;
    deferred->source_file = source_file;
    iree_hal_file_retain(deferred->source_file);
    deferred->source_file_view = *source_file_view;
    deferred->source_offset = source_offset;
    deferred->target_buffer = target_buffer;
    iree_hal_buffer_retain(deferred->target_buffer);
    deferred->target_offset = target_offset;
    deferred->length = length;
    deferred->flags = flags;
    deferred->timepoints_offset = timepoints_offset;

    status = iree_hal_semaphore_list_clone(&wait_semaphore_list,
                                           deferred->host_allocator,
                                           &deferred->wait_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_semaphore_list,
                                           deferred->host_allocator,
                                           &deferred->signal_semaphore_list);
  }

  bool registration_started = false;
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_timepoint_t* timepoints =
        iree_hal_remote_deferred_file_read_timepoints(deferred);
    for (iree_host_size_t i = 0;
         i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
      iree_atomic_fetch_add(&deferred->pending_count, 1,
                            iree_memory_order_relaxed);
      registration_started = true;
      timepoints[i].callback = iree_hal_remote_deferred_file_read_callback;
      timepoints[i].user_data = deferred;
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i], &timepoints[i]);
      if (!iree_status_is_ok(status)) {
        iree_hal_remote_deferred_file_read_release(deferred);
      }
    }
  }

  if (!iree_status_is_ok(status) && registration_started) {
    iree_hal_remote_deferred_file_read_report_failure(
        deferred, iree_status_clone(status));
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  if (deferred) {
    if (iree_status_is_ok(status)) {
      iree_hal_remote_deferred_file_read_release(deferred);
    } else {
      iree_hal_remote_deferred_file_read_destroy(deferred);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_device_queue_read_client_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_read_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list)) {
    return iree_hal_remote_client_device_queue_read_client_file_now(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_file, source_file_view, source_offset, target_buffer,
        target_offset, length, flags);
  }

  return iree_hal_remote_client_device_defer_queue_read_client_file(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_file_view, source_offset, target_buffer,
      target_offset, length, flags);
}

iree_status_t iree_hal_remote_client_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));

  iree_status_t status =
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ);
  if (iree_status_is_ok(status) && flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported read flags: 0x%" PRIx64, flags);
  }

  iree_hal_remote_client_file_view_t source_file_view;
  if (iree_status_is_ok(status) && length == 0) {
    status = iree_hal_device_queue_barrier(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  } else if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_client_file_resolve(source_file, &source_file_view);
    if (iree_status_is_ok(status)) {
      switch (source_file_view.kind) {
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION:
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE:
          status = iree_hal_remote_client_device_queue_read_client_file(
              base_device, queue_affinity, wait_semaphore_list,
              signal_semaphore_list, source_file, &source_file_view,
              source_offset, target_buffer, target_offset, length, flags);
          break;
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_REMOTE_FILE:
          if (source_file_view.length != 0 &&
              (source_offset > source_file_view.length ||
               (uint64_t)length > source_file_view.length - source_offset)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "remote queue_read source range exceeds server file length");
          }
          if (iree_status_is_ok(status)) {
            iree_hal_remote_file_read_op_t op;
            memset(&op, 0, sizeof(op));
            op.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_READ;
            op.source_file_id = source_file_view.remote_file_id;
            op.source_offset = source_offset;
            status = iree_hal_remote_client_buffer_resolve_wire_range(
                target_buffer, target_offset, length, &op.target_buffer_id,
                &op.target_offset, &op.length);
            op.read_flags = flags;
            if (iree_status_is_ok(status)) {
              iree_async_span_t span =
                  iree_async_span_from_ptr(&op, sizeof(op));
              iree_async_span_list_t payload = {&span, 1};
              iree_hal_resource_t* resources[2] = {
                  (iree_hal_resource_t*)target_buffer,
                  (iree_hal_resource_t*)source_file,
              };
              status = iree_hal_remote_client_device_submit_queue_op_resources(
                  device, wait_semaphore_list, signal_semaphore_list,
                  /*required_wait_frontier=*/NULL, payload,
                  IREE_ARRAYSIZE(resources), resources,
                  /*out_epoch=*/NULL);
              if (iree_status_is_ok(status)) {
                iree_hal_remote_client_file_mark_queue_referenced(source_file);
              }
            }
          }
          break;
        default:
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "unsupported remote queue_read file kind: %u",
                               (unsigned)source_file_view.kind);
          break;
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));

  iree_status_t status =
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE);
  if (iree_status_is_ok(status) && flags != IREE_HAL_WRITE_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported write flags: 0x%" PRIx64, flags);
  }

  iree_hal_remote_client_file_view_t target_file_view;
  if (iree_status_is_ok(status) && length == 0) {
    status = iree_hal_device_queue_barrier(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  } else if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_client_file_resolve(target_file, &target_file_view);
    if (iree_status_is_ok(status)) {
      switch (target_file_view.kind) {
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION:
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE: {
          iree_hal_remote_client_file_write_op_t op;
          memset(&op, 0, sizeof(op));
          op.header.type = IREE_HAL_REMOTE_QUEUE_OP_CLIENT_FILE_WRITE;
          status = iree_hal_remote_client_buffer_resolve_wire_range(
              source_buffer, source_offset, length, &op.source_buffer_id,
              &op.source_offset, &op.length);
          uint64_t transfer_id = 0;
          if (iree_status_is_ok(status)) {
            status = iree_hal_remote_client_bulk_begin_file_write(
                device, target_file, &target_file_view, target_offset,
                op.length, &transfer_id);
          }
          op.transfer_id = transfer_id;
          op.write_flags = flags;
          if (iree_status_is_ok(status)) {
            iree_async_span_t span = iree_async_span_from_ptr(&op, sizeof(op));
            iree_async_span_list_t payload = {&span, 1};
            iree_hal_resource_t* resources[1] = {
                (iree_hal_resource_t*)source_buffer,
            };
            status = iree_hal_remote_client_device_submit_queue_op_resources(
                device, wait_semaphore_list, signal_semaphore_list,
                /*required_wait_frontier=*/NULL, payload,
                IREE_ARRAYSIZE(resources), resources, /*out_epoch=*/NULL);
          }
          if (!iree_status_is_ok(status)) {
            iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
          }
          break;
        }
        case IREE_HAL_REMOTE_CLIENT_FILE_KIND_REMOTE_FILE:
          if (target_file_view.length != 0 &&
              (target_offset > target_file_view.length ||
               (uint64_t)length > target_file_view.length - target_offset)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "remote queue_write target range exceeds server file length");
          }
          if (iree_status_is_ok(status)) {
            iree_hal_remote_file_write_op_t op;
            memset(&op, 0, sizeof(op));
            op.header.type = IREE_HAL_REMOTE_QUEUE_OP_FILE_WRITE;
            status = iree_hal_remote_client_buffer_resolve_wire_range(
                source_buffer, source_offset, length, &op.source_buffer_id,
                &op.source_offset, &op.length);
            op.target_file_id = target_file_view.remote_file_id;
            op.target_offset = target_offset;
            op.write_flags = flags;
            if (iree_status_is_ok(status)) {
              iree_async_span_t span =
                  iree_async_span_from_ptr(&op, sizeof(op));
              iree_async_span_list_t payload = {&span, 1};
              iree_hal_resource_t* resources[2] = {
                  (iree_hal_resource_t*)source_buffer,
                  (iree_hal_resource_t*)target_file,
              };
              status = iree_hal_remote_client_device_submit_queue_op_resources(
                  device, wait_semaphore_list, signal_semaphore_list,
                  /*required_wait_frontier=*/NULL, payload,
                  IREE_ARRAYSIZE(resources), resources,
                  /*out_epoch=*/NULL);
              if (iree_status_is_ok(status)) {
                iree_hal_remote_client_file_mark_queue_referenced(target_file);
              }
            }
          }
          break;
        default:
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "unsupported remote queue_write file kind: %u",
                               (unsigned)target_file_view.kind);
          break;
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Host calls
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_host_call_state_t {
  // Timepoint registered on the current unsatisfied wait semaphore.
  iree_async_semaphore_timepoint_t timepoint;
  // Device the call was scheduled on. Retained for the state lifetime.
  iree_hal_device_t* device;
  // Host allocator used to free this state.
  iree_allocator_t host_allocator;
  // Queue affinity requested by the caller.
  iree_hal_queue_affinity_t queue_affinity;
  // Client-local callback and user data captured at submission.
  iree_hal_host_call_t call;
  // User arguments copied at submission.
  uint64_t args[4];
  // Host-call behavior flags captured at submission.
  iree_hal_host_call_flags_t flags;
  // First wait semaphore that may not yet be satisfied.
  iree_host_size_t wait_semaphore_index;
  // Retained wait semaphores that gate callback issue.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Retained signal semaphores completed by the callback.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_remote_host_call_state_t;

static void iree_hal_remote_host_call_state_release(
    iree_hal_remote_host_call_state_t* state) {
  if (state) {
    iree_allocator_t host_allocator = state->host_allocator;
    iree_hal_semaphore_list_release(state->wait_semaphore_list);
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_hal_resource_release(state->call.resource);
    iree_hal_device_release(state->device);
    iree_allocator_free(host_allocator, state);
  }
}

static iree_status_t iree_hal_remote_issue_host_call(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const bool is_nonblocking =
      iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  iree_status_t status = iree_ok_status();
  if (is_nonblocking) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }

  iree_status_t call_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    iree_hal_host_call_context_t context = {
        .device = device,
        .queue_affinity = queue_affinity,
        .signal_semaphore_list = is_nonblocking
                                     ? iree_hal_semaphore_list_empty()
                                     : signal_semaphore_list,
    };
    call_status = call.fn(call.user_data, args, &context);
  }

  if (iree_status_is_ok(status)) {
    if (is_nonblocking || iree_status_is_deferred(call_status)) {
      iree_status_ignore(call_status);
    } else if (iree_status_is_ok(call_status)) {
      status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                              /*frontier=*/NULL);
    } else {
      iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
      call_status = iree_ok_status();
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_host_call_arm_next_wait(
    iree_hal_remote_host_call_state_t* state, bool* out_state_consumed);

static void iree_hal_remote_host_call_wait_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_remote_host_call_state_t* state =
      (iree_hal_remote_host_call_state_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    bool state_consumed = false;
    status = iree_hal_remote_host_call_arm_next_wait(state, &state_consumed);
    if (state_consumed) state = NULL;
  }

  if (state) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
    iree_hal_remote_host_call_state_release(state);
  }

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_host_call_arm_next_wait(
    iree_hal_remote_host_call_state_t* state, bool* out_state_consumed) {
  *out_state_consumed = false;
  iree_status_t status = iree_ok_status();
  iree_hal_semaphore_t* gate_semaphore = NULL;
  uint64_t gate_value = 0;
  for (iree_host_size_t i = state->wait_semaphore_index;
       i < state->wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
    iree_hal_semaphore_t* semaphore = state->wait_semaphore_list.semaphores[i];
    uint64_t value = state->wait_semaphore_list.payload_values[i];
    uint64_t current_value = 0;
    status = iree_hal_semaphore_query(semaphore, &current_value);
    if (iree_status_is_ok(status) && current_value < value) {
      state->wait_semaphore_index = i;
      gate_semaphore = semaphore;
      gate_value = value;
      break;
    } else if (iree_status_is_ok(status)) {
      state->wait_semaphore_index = i + 1;
    }
  }

  if (iree_status_is_ok(status) && gate_semaphore) {
    state->timepoint.callback = iree_hal_remote_host_call_wait_callback;
    state->timepoint.user_data = state;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)gate_semaphore, gate_value, &state->timepoint);
    if (iree_status_is_ok(status)) {
      *out_state_consumed = true;
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_remote_issue_host_call(
        state->device, state->queue_affinity, state->signal_semaphore_list,
        state->call, state->args, state->flags);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
      status = iree_ok_status();
    }
    iree_hal_remote_host_call_state_release(state);
    *out_state_consumed = true;
  }
  return status;
}

static iree_status_t iree_hal_remote_host_call_state_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags,
    iree_hal_remote_host_call_state_t** out_state) {
  *out_state = NULL;

  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_host_call_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_hal_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));

  iree_hal_remote_host_call_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&state);
  }
  if (iree_status_is_ok(status)) {
    memset(state, 0, total_size);
    state->device = device;
    iree_hal_device_retain(device);
    state->host_allocator = host_allocator;
    state->queue_affinity = queue_affinity;
    state->call = call;
    iree_hal_resource_retain(state->call.resource);
    memcpy(state->args, args, sizeof(state->args));
    state->flags = flags;

    char* state_storage = (char*)state;
    state->wait_semaphore_list = (iree_hal_semaphore_list_t){
        .count = wait_semaphore_list.count,
        .semaphores =
            (iree_hal_semaphore_t**)(state_storage + wait_semaphores_offset),
        .payload_values = (uint64_t*)(state_storage + wait_values_offset),
    };
    state->signal_semaphore_list = (iree_hal_semaphore_list_t){
        .count = signal_semaphore_list.count,
        .semaphores =
            (iree_hal_semaphore_t**)(state_storage + signal_semaphores_offset),
        .payload_values = (uint64_t*)(state_storage + signal_values_offset),
    };

    memcpy(state->wait_semaphore_list.semaphores,
           wait_semaphore_list.semaphores,
           wait_semaphore_list.count *
               sizeof(state->wait_semaphore_list.semaphores[0]));
    memcpy(state->wait_semaphore_list.payload_values,
           wait_semaphore_list.payload_values,
           wait_semaphore_list.count *
               sizeof(state->wait_semaphore_list.payload_values[0]));
    iree_hal_semaphore_list_retain(state->wait_semaphore_list);

    memcpy(state->signal_semaphore_list.semaphores,
           signal_semaphore_list.semaphores,
           signal_semaphore_list.count *
               sizeof(state->signal_semaphore_list.semaphores[0]));
    memcpy(state->signal_semaphore_list.payload_values,
           signal_semaphore_list.payload_values,
           signal_semaphore_list.count *
               sizeof(state->signal_semaphore_list.payload_values[0]));
    iree_hal_semaphore_list_retain(state->signal_semaphore_list);

    *out_state = state;
  }

  return status;
}

static iree_status_t iree_hal_remote_validate_host_call(
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  const iree_hal_host_call_flags_t known_flags =
      IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
      IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE | IREE_HAL_HOST_CALL_FLAG_RELAXED;
  if (!call.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call callback must be non-null");
  } else if (!args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call args must be non-null");
  } else if (iree_any_bit_set(flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported host_call flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_remote_validate_host_call(call, args, flags);
  if (iree_status_is_ok(status) &&
      (wait_semaphore_list.count == 0 ||
       iree_hal_semaphore_list_poll(wait_semaphore_list))) {
    status = iree_hal_remote_issue_host_call(
        base_device, queue_affinity, signal_semaphore_list, call, args, flags);
  } else if (iree_status_is_ok(status)) {
    iree_hal_remote_host_call_state_t* state = NULL;
    status = iree_hal_remote_host_call_state_create(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        call, args, flags, &state);
    if (iree_status_is_ok(status)) {
      bool state_consumed = false;
      status = iree_hal_remote_host_call_arm_next_wait(state, &state_consumed);
      if (!state_consumed && !iree_status_is_ok(status)) {
        iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                                     iree_status_clone(status));
        iree_hal_remote_host_call_state_release(state);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_dispatch_payload_t payload;
  iree_hal_remote_queue_payload_writer_t payload_writer;
  iree_status_t status = iree_hal_remote_prepare_dispatch_payload(
      executable, function, config, constants, bindings, flags, &payload,
      &payload_writer);
  iree_hal_remote_queue_resource_list_t resource_list = {
      .write = iree_hal_remote_write_dispatch_resources,
      .user_data = &payload,
      .resource_count = 2 + bindings.count,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_submit_queue_op_writer(
        device, wait_semaphore_list, signal_semaphore_list,
        /*required_wait_frontier=*/NULL, payload_writer, resource_list,
        /*out_epoch=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // All sends are immediate (no batching). Nothing to flush.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue channel callbacks
//===----------------------------------------------------------------------===//

// Deserializes the server-side error status from an error ADVANCE payload.
// Tries the full status_wire format first (preserves source locations, message,
// annotations); falls back to code-only if the wire data is missing, truncated,
// or fails to deserialize. Returns a non-OK status that the caller owns.
static iree_status_t iree_hal_remote_client_device_deserialize_advance_error(
    const iree_hal_remote_advance_payload_t* advance_payload,
    iree_const_byte_span_t advance_data) {
  iree_status_code_t code = (iree_status_code_t)advance_payload->status_code;
  if (advance_payload->status_wire_length > 0) {
    iree_host_size_t entries_size =
        (iree_host_size_t)advance_payload->resolution_count *
        sizeof(iree_hal_remote_resolution_entry_t);
    iree_host_size_t wire_offset =
        sizeof(iree_hal_remote_advance_payload_t) + entries_size;
    iree_host_size_t wire_length =
        (iree_host_size_t)advance_payload->status_wire_length;
    iree_host_size_t required = 0;
    if (iree_host_size_checked_add(wire_offset, wire_length, &required) &&
        advance_data.data_length >= required) {
      iree_const_byte_span_t wire_data = iree_make_const_byte_span(
          advance_data.data + wire_offset, wire_length);
      iree_status_t server_status = iree_ok_status();
      iree_status_t deserialize_status =
          iree_net_status_wire_deserialize(wire_data, &server_status);
      if (iree_status_is_ok(deserialize_status)) {
        return server_status;
      }
      iree_status_ignore(deserialize_status);
    }
  }
  return iree_status_from_code(code);
}

// Client receives ADVANCE frames when server-side operations complete.
// Advances the frontier tracker for each entry in the signal frontier, which
// dispatches any waiters whose frontiers are now satisfied.
static iree_status_t iree_hal_remote_client_device_on_advance(
    void* user_data, const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t advance_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!signal_frontier || signal_frontier->entry_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ADVANCE frame with empty signal frontier");
  }

  // Parse the advance payload if present.
  const iree_hal_remote_advance_payload_t* advance_payload = NULL;
  if (advance_data.data_length >= sizeof(iree_hal_remote_advance_payload_t)) {
    advance_payload =
        (const iree_hal_remote_advance_payload_t*)advance_data.data;
  }

  // Process resolution entries BEFORE advancing the frontier. This ensures
  // that when the frontier advance fires semaphore signals and the
  // application wakes, all buffer proxies have their resolved resource_ids.
  if (advance_payload && advance_payload->resolution_count > 0) {
    iree_host_size_t entries_size =
        (iree_host_size_t)advance_payload->resolution_count *
        sizeof(iree_hal_remote_resolution_entry_t);
    if (advance_data.data_length >=
        sizeof(iree_hal_remote_advance_payload_t) + entries_size) {
      const iree_hal_remote_resolution_entry_t* entries =
          (const iree_hal_remote_resolution_entry_t*)(advance_payload + 1);
      for (uint16_t i = 0; i < advance_payload->resolution_count; ++i) {
        iree_hal_buffer_t* buffer =
            iree_hal_remote_client_device_resolve_provisional(
                device, entries[i].provisional_id);
        if (buffer) {
          iree_hal_remote_client_buffer_set_resource_id(buffer,
                                                        entries[i].resolved_id);
        }
      }
    }
  }

  // Check for server-side error. A non-zero status_code means the operation
  // failed — deserialize and propagate through the semaphore failure path.
  if (advance_payload && advance_payload->status_code != 0) {
    iree_status_t server_status =
        iree_hal_remote_client_device_deserialize_advance_error(advance_payload,
                                                                advance_data);
    // fail_axis takes ownership of the status. Clone for all entries except
    // the last, which receives the original.
    for (uint8_t i = 0; i < signal_frontier->entry_count; ++i) {
      bool is_last = (i == signal_frontier->entry_count - 1);
      iree_async_frontier_tracker_fail_axis(
          device->frontier_tracker, signal_frontier->entries[i].axis,
          is_last ? server_status : iree_status_clone(server_status));
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Success path: advance the frontier tracker (fires semaphore signals).
  for (uint8_t i = 0; i < signal_frontier->entry_count; ++i) {
    iree_async_frontier_tracker_advance(device->frontier_tracker,
                                        signal_frontier->entries[i].axis,
                                        signal_frontier->entries[i].epoch);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Client does not receive COMMAND frames (only servers do).
static iree_status_t iree_hal_remote_client_device_on_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "client does not accept COMMAND frames");
}

// Transport error on the queue channel endpoint.
static void iree_hal_remote_client_device_on_queue_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_device_fail(device, status);

  IREE_TRACE_ZONE_END(z0);
}

// Called when the queue endpoint is ready after session bootstrap. Creates the
// queue channel, then starts bulk endpoint provisioning.
void iree_hal_remote_client_device_on_queue_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_fail(device, status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Create header pool and queue channel into locals first; bulk endpoint
  // provisioning publishes it only after both production channels are ready.
  iree_async_buffer_pool_t* header_pool = NULL;
  iree_net_queue_channel_t* queue_channel = NULL;

  // Create header pool for queue frame header + frontier encoding.
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE, device->host_allocator,
      &header_pool);

  // Create queue channel with client-side callbacks.
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_callbacks_t callbacks = {
        .on_command = iree_hal_remote_client_device_on_command,
        .on_advance = iree_hal_remote_client_device_on_advance,
        .on_transport_error =
            iree_hal_remote_client_device_on_queue_transport_error,
        .user_data = device,
    };

    status = iree_net_queue_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS, header_pool, callbacks,
        device->host_allocator, &queue_channel);
    header_pool = NULL;  // queue_channel_create consumes the pool.
  }

  // Activate the channel to begin receiving frames.
  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_activate(queue_channel);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_client_device_open_bulk_endpoint(device, queue_channel);
    if (iree_status_is_ok(status)) {
      queue_channel = NULL;  // Ownership transferred to the bulk callback.
    }
  }

  if (!iree_status_is_ok(status)) {
    // Cleanup on failure. Channel owns the pool if it was created
    // successfully; otherwise we must release the pool ourselves.
    if (queue_channel) {
      iree_net_queue_channel_release(queue_channel);
    } else {
      iree_async_buffer_pool_release(header_pool);
    }

    iree_hal_remote_client_device_fail(device, status);
  }

  IREE_TRACE_ZONE_END(z0);
}
