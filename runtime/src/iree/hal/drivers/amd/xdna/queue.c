// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue.h"

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/queue_storage.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"

typedef struct iree_hal_amd_xdna_operation_t iree_hal_amd_xdna_operation_t;

// Arena-owned registration for one deferred semaphore wait.
typedef struct iree_hal_amd_xdna_wait_entry_t {
  // Intrusive timepoint owned by the semaphore until its callback fires.
  iree_async_semaphore_timepoint_t timepoint;
  // Operation awaiting all registered timepoints.
  iree_hal_amd_xdna_operation_t* operation;
  // Retained semaphore kept live until this callback returns.
  iree_hal_semaphore_t* semaphore;
} iree_hal_amd_xdna_wait_entry_t;

// Native FIFO positions, distinct from host operation readiness.
typedef struct iree_hal_amd_xdna_pending_t {
  // Accepted invocation retaining its command, buffers and executable.
  iree_hal_amd_xdna_operation_t* operation;
  // Actual opaque point returned by native acceptance.
  uint64_t submission;
  // Native-only causal epoch assigned after acceptance.
  uint64_t epoch;
} iree_hal_amd_xdna_pending_t;

// Intrusive ready work; submissions with unsatisfied waits are not linked here.
typedef struct iree_hal_amd_xdna_ready_list_t {
  // Oldest eligible operation, or NULL.
  iree_hal_amd_xdna_operation_t* head;
  // Newest eligible operation, or NULL.
  iree_hal_amd_xdna_operation_t* tail;
} iree_hal_amd_xdna_ready_list_t;

typedef struct iree_hal_amd_xdna_queue_t {
  // HAL identity and borrowed family.
  iree_hal_queue_t base;
  // Allocator owning the queue and cold native ring storage.
  iree_allocator_t host_allocator;
  // Reusable operation metadata and captured payload blocks.
  iree_hal_amd_xdna_queue_storage_t storage;
  // Borrowed device owner dominating all native children.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed shared completion owner.
  iree_async_proactor_t* proactor;
  // Native queue, released after accepted resources retire.
  amdf_kernel_queue_t* handle;
  // Owned native wake destination, live through every requested notification.
  iree_async_event_t* event;
  // Borrowed native descriptor naming event's signal primitive.
  amdf_native_event_t native_event;
  // Reusable one-shot wait, used only by the proactor owner.
  iree_async_event_wait_operation_t event_wait;
  // Eligible host producers progress independently of native pending work.
  iree_hal_amd_xdna_ready_list_t host_ready;
  // Eligible invocations waiting for native capacity.
  iree_hal_amd_xdna_ready_list_t dispatch_ready;
  // Placed native pending ring, sized to the provider's prepared capacity.
  struct {
    // Accepted invocations in native FIFO order.
    iree_hal_amd_xdna_pending_t* entries;
    // Number of placed pending entries.
    uint32_t capacity;
    // Index of the oldest accepted native invocation.
    uint32_t head;
    // Number of accepted invocations awaiting checked retirement.
    uint32_t count;
  } pending;
  // True while the one-shot native notification and proactor wait are armed.
  bool observing;
  // Sticky native observation/execution failure, owned by this queue.
  iree_status_t failure_status;
  // Retained topology tracker; NULL before group assignment.
  iree_async_frontier_tracker_t* tracker;
  // Queue axis assigned by its group.
  iree_async_axis_t axis;
  // Last accepted native epoch, never advanced by host work or unresolved
  // waits.
  uint64_t epoch;
  // Complete causal lower bound of the accepted native FIFO.
  iree_hal_amd_xdna_frontier_state_t accepted_frontier;
} iree_hal_amd_xdna_queue_t;

typedef enum iree_hal_amd_xdna_operation_kind_e {
  IREE_HAL_AMD_XDNA_OPERATION_BARRIER,
  IREE_HAL_AMD_XDNA_OPERATION_TRANSFER,
  IREE_HAL_AMD_XDNA_OPERATION_DISPATCH,
} iree_hal_amd_xdna_operation_kind_t;

// One captured host transfer descriptor.
typedef struct iree_hal_amd_xdna_transfer_t {
  // Next descriptor in transaction capture order, or NULL.
  struct iree_hal_amd_xdna_transfer_t* next;
  // Public descriptor with retained logical buffers.
  iree_hal_transfer_operation_t operation;
  // Inline storage for the active FILL pattern.
  uint8_t fill_pattern[4];
  // Chunked captured bytes for the active UPDATE operation.
  iree_hal_amd_xdna_queue_payload_t update_payload;
} iree_hal_amd_xdna_transfer_t;

struct iree_hal_amd_xdna_operation_t {
  // Placed admission callback that resolves causal state on the queue owner.
  iree_async_operation_t admission;
  // Placed callback returning resolved software waits to the queue owner.
  iree_async_operation_t wait_completion;
  // Registered timepoints plus one registration sentinel.
  iree_atomic_int32_t wait_count;
  // First failure transferred from a resolved timepoint callback.
  iree_atomic_intptr_t wait_status;
  // Borrowed queue kept live by |device|.
  iree_hal_amd_xdna_queue_t* queue;
  // Retained device dominating the queue through arena return.
  iree_hal_device_t* device;
  // Queue-owned arenas containing this operation and captured payloads.
  iree_hal_amd_xdna_queue_capture_t capture;
  // Intrusive readiness linkage, independent of proactor linkage.
  iree_hal_amd_xdna_operation_t* next;
  // Captured wait semaphores, retained through readiness.
  iree_hal_semaphore_list_t waits;
  // Captured signal semaphores, retained through final publication.
  iree_hal_semaphore_list_t signals;
  // Owning terminal operation status.
  iree_status_t status;
  // Causal lower bound retained through final publication.
  iree_hal_amd_xdna_frontier_state_t frontier;
  // Active operation payload.
  iree_hal_amd_xdna_operation_kind_t kind;
  union {
    // Finite native function invocation.
    struct {
      // Retained executable owning mutable command backing.
      iree_hal_executable_t* executable;
      // Borrowed function owned by executable.
      iree_hal_amd_xdna_function_t* function;
      // Exclusive prepared storage returned only after rejection or retirement.
      iree_hal_amd_xdna_invocation_t* invocation;
      // Number of captured binding rows.
      iree_host_size_t binding_count;
      // Native views with retained logical buffers in trailing slab storage.
      iree_hal_amd_xdna_executable_binding_t* bindings;
    } dispatch;
    // Host-mapped transfer transaction.
    struct {
      // First captured descriptor in transaction order, or NULL.
      iree_hal_amd_xdna_transfer_t* head;
      // Final captured descriptor used while appending, or NULL.
      iree_hal_amd_xdna_transfer_t* tail;
    } transfer;
  };
};

static const iree_hal_queue_vtable_t iree_hal_amd_xdna_queue_vtable;

static void iree_hal_amd_xdna_queue_report(iree_hal_amd_xdna_queue_t* queue,
                                           iree_status_code_t status_code,
                                           iree_string_view_t message) {
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = status_code,
      .message = message,
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = IREE_SV("xdna");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(queue->context->event_sink, &event);
}

// Transfer descriptors have already passed the common HAL public boundary.
static void iree_hal_amd_xdna_transfer_buffers(
    const iree_hal_amd_xdna_transfer_t* transfer,
    iree_hal_buffer_t** out_source, iree_hal_buffer_t** out_target) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  *out_source = NULL;
  *out_target = NULL;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        break;
      }
      *out_target = operation->fill.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        break;
      }
      *out_target = operation->update.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        break;
      }
      *out_source = operation->copy.source_buffer;
      *out_target = operation->copy.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        break;
      }
      *out_target = operation->upload.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        break;
      }
      *out_source = operation->download.source_buffer;
      break;
  }
}

static void iree_hal_amd_xdna_operation_release_resources(
    iree_hal_amd_xdna_operation_t* operation) {
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH) {
    for (iree_host_size_t i = 0; i < operation->dispatch.binding_count; ++i) {
      iree_hal_buffer_release(
          operation->dispatch.bindings[i].buffer_ref.buffer);
    }
    iree_hal_amd_xdna_invocation_release(operation->dispatch.invocation);
    iree_hal_executable_release(operation->dispatch.executable);
  } else if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_TRANSFER) {
    for (iree_hal_amd_xdna_transfer_t* transfer = operation->transfer.head;
         transfer; transfer = transfer->next) {
      iree_hal_buffer_t* source = NULL;
      iree_hal_buffer_t* target = NULL;
      iree_hal_amd_xdna_transfer_buffers(transfer, &source, &target);
      iree_hal_buffer_release(source);
      iree_hal_buffer_release(target);
    }
  }
  iree_hal_semaphore_list_release(operation->waits);
}

// Returns both arenas before releasing the device that owns their pools.
static void iree_hal_amd_xdna_operation_deallocate(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_device_t* device = operation->device;
  iree_hal_amd_xdna_queue_capture_t capture = operation->capture;
  operation = NULL;
  iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
  iree_hal_device_release(device);
}

static iree_status_t iree_hal_amd_xdna_operation_signal(
    iree_hal_amd_xdna_operation_t* operation, iree_hal_device_t* queue_device) {
  const iree_async_frontier_t* frontier =
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier);
  for (iree_host_size_t i = 0; i < operation->signals.count; ++i) {
    iree_hal_semaphore_t* semaphore = operation->signals.semaphores[i];
    const uint64_t value = operation->signals.payload_values[i];
    iree_status_t status = iree_ok_status();
    if (operation->frontier.exact &&
        iree_hal_amd_xdna_semaphore_is_local(semaphore, queue_device)) {
      status = iree_async_semaphore_publish_untainted(
          (iree_async_semaphore_t*)semaphore, value, frontier);
    } else {
      status = iree_hal_semaphore_signal(semaphore, value, frontier);
    }
    if (!iree_status_is_ok(status)) {
      return status;
    }
  }
  return iree_ok_status();
}

// The operation's device reference keeps queue storage live if a signal waiter
// releases its public device reference inline.
static void iree_hal_amd_xdna_operation_complete(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_amd_xdna_queue_report(operation->queue,
                                   iree_status_code(operation->status),
                                   iree_status_message(operation->status));
  }
  iree_hal_amd_xdna_operation_release_resources(operation);
  if (iree_status_is_ok(operation->status)) {
    operation->status =
        iree_hal_amd_xdna_operation_signal(operation, operation->device);
  }
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_semaphore_list_fail(operation->signals, operation->status);
  }
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_xdna_operation_deallocate(operation);
}

static iree_status_t iree_hal_amd_xdna_update_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  iree_hal_buffer_mapping_t target_mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      operation->update.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_MAP_FLAG_DISCARD,
      operation->update.target_offset, operation->update.length,
      &target_mapping));
  iree_hal_amd_xdna_queue_payload_copy(&transfer->update_payload,
                                       target_mapping.contents.data);
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(
          iree_hal_buffer_memory_type(operation->update.target_buffer),
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_flush_range(&target_mapping, 0,
                                                 IREE_HAL_WHOLE_BUFFER);
  }
  status =
      iree_status_join(status, iree_hal_buffer_unmap_range(&target_mapping));
  return status;
}

static iree_status_t iree_hal_amd_xdna_transfer_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        return iree_ok_status();
      }
      return iree_hal_buffer_map_fill(
          operation->fill.target_buffer, operation->fill.target_offset,
          operation->fill.length, operation->fill.pattern,
          operation->fill.pattern_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        return iree_ok_status();
      }
      return iree_hal_amd_xdna_update_execute(transfer);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        return iree_ok_status();
      }
      return iree_hal_buffer_map_copy(
          operation->copy.source_buffer, operation->copy.source_offset,
          operation->copy.target_buffer, operation->copy.target_offset,
          operation->copy.length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        return iree_ok_status();
      }
      return iree_hal_buffer_map_write(
          operation->upload.target_buffer, operation->upload.target_offset,
          operation->upload.source, operation->upload.length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        return iree_ok_status();
      }
      return iree_hal_buffer_map_read(
          operation->download.source_buffer, operation->download.source_offset,
          operation->download.target, operation->download.length);
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
}

// A failed observer cannot prove accepted device references have retired. Keep
// each remaining operation and its parent graph live, and fail completion
// edges. Native progress is never inferred from a failure signal.
static void iree_hal_amd_xdna_queue_abandon_pending(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  queue->failure_status = iree_status_join(queue->failure_status, status);
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(queue->failure_status),
                                 iree_status_message(queue->failure_status));
  while (queue->pending.count) {
    iree_hal_amd_xdna_pending_t* pending =
        &queue->pending.entries[queue->pending.head];
    iree_hal_semaphore_list_fail(pending->operation->signals,
                                 iree_status_clone(queue->failure_status));
    pending->operation = NULL;
    queue->pending.head = (queue->pending.head + 1) % queue->pending.capacity;
    --queue->pending.count;
  }
}

static iree_status_t iree_hal_amd_xdna_queue_arm(
    iree_hal_amd_xdna_queue_t* queue) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_request_notification(
          queue->handle, queue->pending.entries[queue->pending.head].submission,
          &queue->native_event),
      "kernel_queue_request_notification"));
  iree_async_operation_initialize(&queue->event_wait.base,
                                  IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  queue->event_wait.base.completion_fn, queue);
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_submit_one(queue->proactor, &queue->event_wait.base));
  queue->observing = true;
  return iree_ok_status();
}

static bool iree_hal_amd_xdna_queue_has_work(iree_hal_amd_xdna_queue_t* queue) {
  return queue->pending.count || queue->host_ready.head ||
         queue->dispatch_ready.head;
}

static void iree_hal_amd_xdna_queue_publish_signals(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_amd_xdna_operation_t* operation,
    uint64_t epoch) {
  if (!queue->tracker) {
    return;
  }
  iree_hal_device_t* queue_device =
      iree_hal_queue_family_device(queue->base.queue_family);
  const iree_async_frontier_t* frontier =
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier);
  for (iree_host_size_t i = 0; i < operation->signals.count; ++i) {
    iree_hal_semaphore_t* semaphore = operation->signals.semaphores[i];
    if (!iree_hal_amd_xdna_semaphore_is_local(semaphore, queue_device)) {
      continue;
    }
    iree_hal_amd_xdna_semaphore_publish_signal(
        semaphore, queue->axis, frontier, operation->frontier.exact, epoch,
        operation->signals.payload_values[i]);
  }
}

// Runs semaphore-ready work. Only the proactor owner touches scheduling state.
// Native capacity never prevents eligible host producers from running. A
// successor retains the queue across publication; without one, publication is
// the final queue access and the caller may immediately destroy the device.
static void iree_hal_amd_xdna_queue_pump(iree_hal_amd_xdna_queue_t* queue) {
  while (true) {
    iree_hal_amd_xdna_ready_list_t* ready = &queue->host_ready;
    if (!ready->head) {
      if (queue->pending.count == queue->pending.capacity) {
        return;
      }
      ready = &queue->dispatch_ready;
    }
    iree_hal_amd_xdna_operation_t* operation = ready->head;
    if (!operation) {
      return;
    }
    ready->head = operation->next;
    if (!ready->head) {
      ready->tail = NULL;
    }
    operation->next = NULL;
    if (iree_status_is_ok(operation->status) &&
        !iree_status_is_ok(queue->failure_status)) {
      operation->status = iree_status_clone(queue->failure_status);
    }
    if (iree_status_is_ok(operation->status)) {
      switch (operation->kind) {
        case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
          break;
        case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
          for (iree_hal_amd_xdna_transfer_t* transfer =
                   operation->transfer.head;
               transfer && iree_status_is_ok(operation->status);
               transfer = transfer->next) {
            operation->status = iree_hal_amd_xdna_transfer_execute(transfer);
          }
          break;
        case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH: {
          // This operation may have waited in the ready list while earlier
          // invocations were accepted. Physical FIFO order includes that
          // complete accepted prefix.
          iree_hal_amd_xdna_frontier_state_merge(&operation->frontier,
                                                 &queue->accepted_frontier);
          amdf_xdna_kernel_command_t command = {0};
          operation->status = iree_hal_amd_xdna_function_prepare(
              operation->dispatch.function, operation->dispatch.bindings,
              &operation->dispatch.invocation, &command);
          uint64_t submission = 0;
          if (iree_status_is_ok(operation->status)) {
            const amdf_xdna_kernel_queue_submission_info_t submission_info = {
                .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
                .structure_size = sizeof(submission_info),
                .command_count = 1,
                .commands = &command,
            };
            operation->status = IREE_HAL_AMD_STATUS_FROM_AMDF(
                queue->context->xdna->kernel_queue_submit(
                    queue->handle, &submission_info, &submission),
                "xdna.kernel_queue_submit");
          }
          if (iree_status_is_ok(operation->status)) {
            const uint64_t epoch = queue->epoch + 1;
            if (queue->tracker) {
              iree_hal_amd_xdna_frontier_state_advance(&operation->frontier,
                                                       queue->axis, epoch);
            } else {
              operation->frontier.exact = false;
            }
            iree_hal_amd_xdna_queue_publish_signals(queue, operation, epoch);
            iree_hal_amd_xdna_frontier_state_copy(&operation->frontier,
                                                  &queue->accepted_frontier);
            queue->epoch = epoch;
            const uint32_t tail = (queue->pending.head + queue->pending.count) %
                                  queue->pending.capacity;
            queue->pending.entries[tail] = (iree_hal_amd_xdna_pending_t){
                .operation = operation,
                .submission = submission,
                .epoch = epoch,
            };
            ++queue->pending.count;
            if (!queue->observing) {
              iree_status_t status = iree_hal_amd_xdna_queue_arm(queue);
              if (!iree_status_is_ok(status)) {
                iree_hal_amd_xdna_queue_abandon_pending(queue, status);
              }
            }
            // Native execution or the safe-leak failure owner holds this
            // operation. Neither path can release its storage here.
            operation = NULL;
          }
          break;
        }
      }
    }
    if (operation) {
      const bool has_work = iree_hal_amd_xdna_queue_has_work(queue);
      iree_hal_amd_xdna_operation_complete(operation);
      if (!has_work) {
        return;
      }
    }
  }
}

static void iree_hal_amd_xdna_queue_native_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_queue_t* queue = user_data;
  queue->observing = false;
  amdf_kernel_queue_status_t checked = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(checked),
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        queue->context->api->kernel_queue_refresh_status(queue->handle,
                                                         &checked),
        "kernel_queue_refresh_status");
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_abandon_pending(queue, status);
  } else {
    if (!amdf_status_is_ok(checked.terminal_status)) {
      queue->failure_status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          checked.terminal_status, "XDNA command outcome");
    }
    while (queue->pending.count &&
           queue->pending.entries[queue->pending.head].submission <=
               checked.retired_submission) {
      iree_hal_amd_xdna_pending_t* pending =
          &queue->pending.entries[queue->pending.head];
      iree_hal_amd_xdna_operation_t* operation = pending->operation;
      operation->status = iree_status_clone(queue->failure_status);
      if (queue->tracker && iree_status_is_ok(operation->status)) {
        iree_async_frontier_tracker_advance(queue->tracker, queue->axis,
                                            pending->epoch);
      }
      pending->operation = NULL;
      queue->pending.head = (queue->pending.head + 1) % queue->pending.capacity;
      --queue->pending.count;
      const bool has_work = iree_hal_amd_xdna_queue_has_work(queue);
      iree_hal_amd_xdna_operation_complete(operation);
      if (!has_work) {
        return;
      }
    }
    if (queue->pending.count) {
      if (checked.state != AMDF_QUEUE_STATE_ACTIVE) {
        iree_hal_amd_xdna_queue_abandon_pending(
            queue, iree_status_clone(queue->failure_status));
      } else {
        status = iree_hal_amd_xdna_queue_arm(queue);
        if (!iree_status_is_ok(status)) {
          iree_hal_amd_xdna_queue_abandon_pending(queue, status);
        }
      }
    }
  }
  iree_hal_amd_xdna_queue_pump(queue);
}

static void iree_hal_amd_xdna_queue_enqueue_ready(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  operation->status = status;
  operation->next = NULL;
  iree_hal_amd_xdna_ready_list_t* ready =
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH &&
              iree_status_is_ok(status)
          ? &queue->dispatch_ready
          : &queue->host_ready;
  if (ready->tail) {
    ready->tail->next = operation;
  } else {
    ready->head = operation;
  }
  ready->tail = operation;
  iree_hal_amd_xdna_queue_pump(queue);
}

static iree_status_t iree_hal_amd_xdna_operation_resolve_waits(
    iree_hal_amd_xdna_operation_t* operation,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution) {
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  const bool is_dispatch =
      operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH;
  const iree_hal_amd_xdna_frontier_state_t* accepted_state =
      is_dispatch ? &queue->accepted_frontier : NULL;
  const iree_hal_amd_xdna_wait_resolution_flags_t flags =
      is_dispatch ? IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO
                  : IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE;
  return iree_hal_amd_xdna_frontier_resolve_waits(
      iree_hal_queue_family_device(queue->base.queue_family), operation->waits,
      accepted_state, flags, &operation->frontier, out_resolution);
}

static void iree_hal_amd_xdna_operation_record_wait_status(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &operation->wait_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

// Schedules the sole wait completion owner. A failed handoff cannot safely
// destroy queue-owned storage from an arbitrary semaphore callback thread.
static void iree_hal_amd_xdna_operation_schedule_wait_completion(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &operation->wait_completion);
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_t wait_status = (iree_status_t)iree_atomic_exchange(
      &operation->wait_status, 0, iree_memory_order_acquire);
  status = iree_status_join(wait_status, status);
  iree_hal_amd_xdna_queue_report(operation->queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_hal_semaphore_list_fail(operation->signals, status);
}

static void iree_hal_amd_xdna_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amd_xdna_wait_entry_t* entry = user_data;
  iree_hal_amd_xdna_operation_t* operation = entry->operation;
  iree_hal_semaphore_t* semaphore = entry->semaphore;
  iree_hal_amd_xdna_operation_record_wait_status(operation, status);
  const int32_t previous_count = iree_atomic_fetch_sub(
      &operation->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    iree_hal_amd_xdna_operation_schedule_wait_completion(operation);
  }
  // No entry or operation access is permitted after the completion handoff.
  iree_hal_semaphore_release(semaphore);
}

// Registers arena-owned timepoints for every wait. The extra count sentinel
// prevents synchronous callbacks from completing the operation until all
// registrations have either succeeded or been accounted for.
static void iree_hal_amd_xdna_operation_register_waits(
    iree_hal_amd_xdna_operation_t* operation) {
  const iree_hal_semaphore_list_t waits = operation->waits;
  iree_atomic_store(&operation->wait_count, (int32_t)waits.count + 1,
                    iree_memory_order_release);
  iree_status_t status = iree_ok_status();
  iree_host_size_t i = 0;
  while (i < waits.count && iree_status_is_ok(status)) {
    iree_hal_amd_xdna_wait_entry_t* entry = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*entry), (void**)&entry);
    if (iree_status_is_ok(status)) {
      memset(entry, 0, sizeof(*entry));
      entry->operation = operation;
      entry->semaphore = waits.semaphores[i];
      iree_hal_semaphore_retain(entry->semaphore);
      entry->timepoint.callback = iree_hal_amd_xdna_wait_entry_resolved;
      entry->timepoint.user_data = entry;
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)entry->semaphore, waits.payload_values[i],
          &entry->timepoint);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_release(entry->semaphore);
      }
    }
    if (iree_status_is_ok(status)) {
      ++i;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_record_wait_status(operation, status);
    const int32_t unregistered_count = (int32_t)(waits.count - i);
    iree_atomic_fetch_sub(&operation->wait_count, unregistered_count,
                          iree_memory_order_acq_rel);
  }
  const int32_t previous_count = iree_atomic_fetch_sub(
      &operation->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    iree_hal_amd_xdna_operation_schedule_wait_completion(operation);
  }
}

static void iree_hal_amd_xdna_queue_wait_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_status_t wait_status = (iree_status_t)iree_atomic_exchange(
      &operation->wait_status, 0, iree_memory_order_acquire);
  status = iree_status_join(status, wait_status);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_xdna_wait_resolution_t resolution =
        IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER;
    status = iree_hal_amd_xdna_operation_resolve_waits(operation, &resolution);
    if (iree_status_is_ok(status) &&
        resolution != IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "completed XDNA software wait remained unresolved");
    }
  }
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
}

static void iree_hal_amd_xdna_queue_admit(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_resolve_waits(operation, &resolution);
  }
  if (iree_status_is_ok(status) &&
      resolution == IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER) {
    iree_hal_amd_xdna_operation_register_waits(operation);
    return;
  }
  iree_hal_amd_xdna_queue_enqueue_ready(operation, status);
}

static iree_status_t iree_hal_amd_xdna_operation_capture_semaphores(
    iree_hal_amd_xdna_operation_t* operation, iree_hal_semaphore_list_t source,
    iree_hal_semaphore_list_t* out_list) {
  *out_list = iree_hal_semaphore_list_empty();
  if (!source.count) {
    return iree_ok_status();
  }
  iree_host_size_t size = 0;
  iree_host_size_t semaphores_offset = 0;
  iree_host_size_t values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size,
      IREE_STRUCT_FIELD(source.count, iree_hal_semaphore_t*,
                        &semaphores_offset),
      IREE_STRUCT_FIELD(source.count, uint64_t, &values_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, size, (void**)&storage));
  *out_list = (iree_hal_semaphore_list_t){
      .count = source.count,
      .semaphores = (iree_hal_semaphore_t**)(storage + semaphores_offset),
      .payload_values = (uint64_t*)(storage + values_offset),
  };
  for (iree_host_size_t i = 0; i < source.count; ++i) {
    out_list->semaphores[i] = source.semaphores[i];
    out_list->payload_values[i] = source.payload_values[i];
  }
  iree_hal_semaphore_list_retain(*out_list);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_operation_create(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_amd_xdna_operation_kind_t kind,
    iree_hal_amd_xdna_operation_t** out_operation) {
  *out_operation = NULL;
  iree_hal_amd_xdna_queue_capture_t capture;
  iree_hal_amd_xdna_queue_capture_initialize(&queue->storage, &capture);
  iree_hal_amd_xdna_operation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &capture, sizeof(*operation), (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
    return status;
  }
  memset(operation, 0, sizeof(*operation));
  operation->queue = queue;
  operation->device = iree_hal_queue_family_device(queue->base.queue_family);
  operation->capture = capture;
  operation->kind = kind;
  iree_hal_device_retain(operation->device);
  status = iree_hal_amd_xdna_operation_capture_semaphores(operation, waits,
                                                          &operation->waits);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_capture_semaphores(
        operation, signals, &operation->signals);
  }
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(&operation->admission,
                                    IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    iree_hal_amd_xdna_queue_admit, operation);
    iree_async_operation_initialize(
        &operation->wait_completion, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_wait_complete,
        operation);
    iree_atomic_store(&operation->wait_count, 0, iree_memory_order_relaxed);
    iree_atomic_store(&operation->wait_status, 0, iree_memory_order_relaxed);
    *out_operation = operation;
  } else {
    iree_hal_semaphore_list_release(operation->signals);
    iree_hal_semaphore_list_release(operation->waits);
    iree_hal_amd_xdna_operation_deallocate(operation);
  }
  return status;
}

static void iree_hal_amd_xdna_operation_discard(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_operation_release_resources(operation);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_xdna_operation_deallocate(operation);
}

static iree_status_t iree_hal_amd_xdna_operation_submit(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &operation->admission);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_BARRIER, &operation));
  return iree_hal_amd_xdna_operation_submit(operation);
}

static iree_status_t iree_hal_amd_xdna_queue_dispatch(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
  if ((flags & ~supported_flags) || config.workgroup_count_ref.buffer ||
      config.dynamic_workgroup_local_memory || constants.data_length) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA dispatch requires direct buffer bindings");
  }
  for (uint32_t i = 0; i < 3; ++i) {
    if (config.workgroup_count[i] > 1 || config.workgroup_size[i] > 1) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA function is one compiled array invocation");
    }
  }
  if (!config.workgroup_count[0] || !config.workgroup_count[1] ||
      !config.workgroup_count[2]) {
    return iree_hal_amd_xdna_queue_barrier(base, waits, signals, 0);
  }
  iree_host_size_t binding_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &binding_length,
      IREE_STRUCT_FIELD_FAM(bindings.count,
                            iree_hal_amd_xdna_executable_binding_t)));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_DISPATCH, &operation));
  void* binding_storage = NULL;
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, binding_length, &binding_storage);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_resolve(
        executable, function, bindings, binding_storage,
        &operation->dispatch.function);
  }
  if (iree_status_is_ok(status)) {
    operation->dispatch.executable = executable;
    iree_hal_executable_retain(executable);
    operation->dispatch.binding_count = bindings.count;
    operation->dispatch.bindings = binding_storage;
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_buffer_retain(operation->dispatch.bindings[i].buffer_ref.buffer);
    }
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_TRANSFER, &operation));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_transfer_t* captured = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*captured), (void**)&captured);
    if (!iree_status_is_ok(status)) {
      break;
    }
    *captured = (iree_hal_amd_xdna_transfer_t){0};
    captured->operation = operations[i];
    if (captured->operation.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
        captured->operation.fill.length) {
      memcpy(captured->fill_pattern, captured->operation.fill.pattern,
             captured->operation.fill.pattern_length);
      captured->operation.fill.pattern = captured->fill_pattern;
    } else if (captured->operation.type ==
                   IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
               captured->operation.update.length) {
      const uint8_t* source =
          (const uint8_t*)captured->operation.update.source_buffer +
          captured->operation.update.source_offset;
      status = iree_hal_amd_xdna_queue_capture_payload(
          &operation->capture,
          iree_make_const_byte_span(
              source, (iree_host_size_t)captured->operation.update.length),
          &captured->update_payload);
      captured->operation.update.source_buffer = NULL;
      captured->operation.update.source_offset = 0;
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_t* source = NULL;
      iree_hal_buffer_t* target = NULL;
      iree_hal_amd_xdna_transfer_buffers(captured, &source, &target);
      iree_hal_buffer_retain(source);
      iree_hal_buffer_retain(target);
      if (operation->transfer.tail) {
        operation->transfer.tail->next = captured;
      } else {
        operation->transfer.head = captured;
      }
      operation->transfer.tail = captured;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  if (tracker) {
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_tracker_register_axis(tracker, axis, NULL));
    iree_async_frontier_tracker_retain(tracker);
  }
  if (queue->tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->tracker);
  }
  queue->tracker = tracker;
  queue->axis = axis;
  queue->epoch = 0;
  iree_hal_amd_xdna_frontier_state_initialize(&queue->accepted_frontier);
  return iree_ok_status();
}

bool iree_hal_amd_xdna_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  if (queue->handle) {
    amdf_status_t status =
        queue->context->api->kernel_queue_destroy(queue->handle);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(queue->context, status,
                                       "kernel_queue_destroy");
      // BUSY preserves the native queue and its borrowed notification target.
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        return false;
      }
    }
    queue->handle = NULL;
  }
  return true;
}

static void iree_hal_amd_xdna_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  if (!iree_hal_amd_xdna_queue_shutdown(base)) {
    return;
  }
  if (queue->tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(queue->tracker);
  }
  iree_status_free(queue->failure_status);
  iree_async_event_release(queue->event);
  iree_allocator_free(queue->host_allocator, queue->pending.entries);
  iree_hal_amd_xdna_queue_storage_deinitialize(&queue->storage);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_hal_amd_xdna_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue) {
  *out_queue = NULL;
  iree_hal_amd_xdna_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  iree_hal_amd_xdna_queue_storage_initialize(host_allocator, &queue->storage);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_initialize(family, &params, &iree_hal_amd_xdna_queue_vtable,
                            &queue->base);
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->proactor = proactor;
  iree_hal_amd_xdna_frontier_state_initialize(&queue->accepted_frontier);
  const amdf_xdna_kernel_queue_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .queue_family_ordinal = context->queue_family_ordinal,
      .maximum_pending_submission_count = 0,
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->kernel_queue_create(context->handle, &create_info,
                                         &queue->handle),
      "xdna.kernel_queue_create");
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->kernel_queue_query_info(queue->handle, &info),
        "kernel_queue_query_info");
  }
  if (iree_status_is_ok(status)) {
    queue->pending.capacity = info.maximum_pending_submission_count;
    iree_host_size_t pending_size = 0;
    status =
        IREE_STRUCT_LAYOUT(0, &pending_size,
                           IREE_STRUCT_FIELD_FAM(queue->pending.capacity,
                                                 iree_hal_amd_xdna_pending_t));
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(host_allocator, pending_size,
                                     (void**)&queue->pending.entries);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_event_create(proactor, &queue->event);
  }
  if (iree_status_is_ok(status)) {
#if defined(IREE_ASYNC_HAVE_EVENTFD)
    queue->native_event.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    queue->native_event.payload.file_descriptor =
        queue->event->native.signal_primitive.value.fd;
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    queue->native_event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
    queue->native_event.payload.native_handle =
        (void*)queue->event->native.signal_primitive.value.win32_handle;
#endif
    if (!iree_any_bit_set(info.notification_types,
                          UINT64_C(1) << queue->native_event.type)) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "XDNA queue has no compatible native wake destination");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(
        &queue->event_wait.base, IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_native_complete,
        queue);
    queue->event_wait.event = queue->event;
    *out_queue = &queue->base;
  } else {
    iree_hal_queue_release(&queue->base);
  }
  return status;
}

void iree_hal_amd_xdna_queue_trim(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (iree_hal_amd_xdna_queue_t*)base;
  iree_hal_amd_xdna_queue_storage_trim(&queue->storage);
}

static iree_status_t iree_hal_amd_xdna_queue_flush(iree_hal_queue_t* queue) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_queue_execute(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support execute");
}

static iree_status_t iree_hal_amd_xdna_queue_host_call(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support host_call");
}

static iree_status_t iree_hal_amd_xdna_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "XDNA queue does not support query_dispatch_concurrency");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_wait(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_wait");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_store(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_store");
}

static iree_status_t iree_hal_amd_xdna_queue_atomic_rmw(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support atomic_rmw");
}

static iree_status_t iree_hal_amd_xdna_queue_timestamp(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support timestamp");
}

static iree_status_t iree_hal_amd_xdna_queue_alloca(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support alloca");
}

static iree_status_t iree_hal_amd_xdna_queue_dealloca(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support dealloca");
}

static iree_status_t iree_hal_amd_xdna_queue_read(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support read");
}

static iree_status_t iree_hal_amd_xdna_queue_write(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "XDNA queue does not support write");
}

static const iree_hal_queue_vtable_t iree_hal_amd_xdna_queue_vtable = {
    .destroy = iree_hal_amd_xdna_queue_destroy,
    .barrier = iree_hal_amd_xdna_queue_barrier,
    .execute = iree_hal_amd_xdna_queue_execute,
    .host_call = iree_hal_amd_xdna_queue_host_call,
    .query_dispatch_concurrency =
        iree_hal_amd_xdna_queue_query_dispatch_concurrency,
    .dispatch = iree_hal_amd_xdna_queue_dispatch,
    .atomic_wait = iree_hal_amd_xdna_queue_atomic_wait,
    .atomic_store = iree_hal_amd_xdna_queue_atomic_store,
    .atomic_rmw = iree_hal_amd_xdna_queue_atomic_rmw,
    .timestamp = iree_hal_amd_xdna_queue_timestamp,
    .flush = iree_hal_amd_xdna_queue_flush,
    .alloca = iree_hal_amd_xdna_queue_alloca,
    .dealloca = iree_hal_amd_xdna_queue_dealloca,
    .transfer = iree_hal_amd_xdna_queue_transfer,
    .read = iree_hal_amd_xdna_queue_read,
    .write = iree_hal_amd_xdna_queue_write,
};
