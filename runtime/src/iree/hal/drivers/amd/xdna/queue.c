// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue.h"

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/executable.h"

typedef struct iree_hal_amd_xdna_operation_t iree_hal_amd_xdna_operation_t;

typedef struct iree_hal_amd_xdna_queue_t {
  // HAL identity and borrowed family.
  iree_hal_queue_t base;
  // Allocator owning queue and submission slabs.
  iree_allocator_t host_allocator;
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
  // Semaphore-ready operations, touched only by the proactor owner.
  struct {
    // Oldest ready operation, or NULL.
    iree_hal_amd_xdna_operation_t* head;
    // Newest ready operation, or NULL.
    iree_hal_amd_xdna_operation_t* tail;
  } ready;
  // Accepted native invocation retaining all command and binding storage.
  iree_hal_amd_xdna_operation_t* active;
  // Actual opaque point returned by native acceptance.
  uint64_t active_submission;
  // Sticky native observation/execution failure, owned by this queue.
  iree_status_t failure_status;
  // Retained topology tracker; NULL before group assignment.
  iree_async_frontier_tracker_t* tracker;
  // Queue axis assigned by its group.
  iree_async_axis_t axis;
  // Completed serialized execution epoch, never reserved for unresolved waits.
  uint64_t epoch;
} iree_hal_amd_xdna_queue_t;

typedef enum iree_hal_amd_xdna_operation_kind_e {
  IREE_HAL_AMD_XDNA_OPERATION_BARRIER,
  IREE_HAL_AMD_XDNA_OPERATION_TRANSFER,
  IREE_HAL_AMD_XDNA_OPERATION_DISPATCH,
} iree_hal_amd_xdna_operation_kind_t;

struct iree_hal_amd_xdna_operation_t {
  // Readiness callback operation, with semaphore arrays in the trailing slab.
  iree_async_semaphore_wait_operation_t wait;
  // Retained queue through final native use and scheduling state access.
  iree_hal_amd_xdna_queue_t* queue;
  // Host owner remains usable after dropping the queue reference.
  iree_allocator_t host_allocator;
  // Intrusive readiness linkage, independent of proactor linkage.
  iree_hal_amd_xdna_operation_t* next;
  // Captured wait semaphores, retained through readiness.
  iree_hal_semaphore_list_t waits;
  // Captured signal semaphores, retained through final publication.
  iree_hal_semaphore_list_t signals;
  // Owning terminal operation status.
  iree_status_t status;
  // Completed causal lower bound; empty when topology is unassigned.
  iree_async_single_frontier_t frontier;
  // Active operation payload.
  iree_hal_amd_xdna_operation_kind_t kind;
  union {
    // Finite native function invocation.
    struct {
      // Retained executable owning mutable command backing.
      iree_hal_executable_t* executable;
      // Borrowed function owned by executable.
      iree_hal_amd_xdna_function_t* function;
      // Number of captured binding rows.
      iree_host_size_t binding_count;
      // Native views with retained logical buffers in trailing slab storage.
      iree_hal_amd_xdna_executable_binding_t* bindings;
    } dispatch;
    // Host-mapped transfer transaction.
    struct {
      // Number of captured operations.
      iree_host_size_t count;
      // Captured descriptors with owned update/pattern bytes in trailing
      // storage.
      iree_hal_transfer_operation_t* operations;
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
    const iree_hal_transfer_operation_t* operation,
    iree_hal_buffer_t** out_source, iree_hal_buffer_t** out_target) {
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
    iree_hal_executable_release(operation->dispatch.executable);
  } else if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_TRANSFER) {
    for (iree_host_size_t i = 0; i < operation->transfer.count; ++i) {
      iree_hal_buffer_t* source = NULL;
      iree_hal_buffer_t* target = NULL;
      iree_hal_amd_xdna_transfer_buffers(&operation->transfer.operations[i],
                                         &source, &target);
      iree_hal_buffer_release(source);
      iree_hal_buffer_release(target);
    }
  }
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_queue_release(&operation->queue->base);
}

// Final publication deliberately follows resource release and the last queue
// access. A waiter may destroy its device immediately after observing success.
static void iree_hal_amd_xdna_operation_complete(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_amd_xdna_queue_report(operation->queue,
                                   iree_status_code(operation->status),
                                   iree_status_message(operation->status));
  }
  iree_hal_amd_xdna_operation_release_resources(operation);
  if (iree_status_is_ok(operation->status)) {
    operation->status = iree_hal_semaphore_list_signal(
        operation->signals,
        iree_async_single_frontier_as_const_frontier(&operation->frontier));
  }
  if (!iree_status_is_ok(operation->status)) {
    iree_hal_semaphore_list_fail(operation->signals, operation->status);
  }
  iree_hal_semaphore_list_release(operation->signals);
  iree_allocator_free(operation->host_allocator, operation);
}

static iree_status_t iree_hal_amd_xdna_transfer_execute(
    const iree_hal_transfer_operation_t* operation) {
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
      return iree_hal_buffer_map_write(
          operation->update.target_buffer, operation->update.target_offset,
          operation->update.source_buffer, operation->update.length);
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

// A failed observer cannot prove that accepted device references are retired.
// Keep the active operation, queue, event, and parent graph live, and fail the
// completion edges. Native progress is never inferred from a failure signal.
static void iree_hal_amd_xdna_queue_abandon_active(
    iree_hal_amd_xdna_queue_t* queue, iree_status_t status) {
  queue->failure_status = status;
  iree_hal_amd_xdna_queue_report(queue, iree_status_code(status),
                                 iree_status_message(status));
  iree_hal_semaphore_list_fail(queue->active->signals,
                               iree_status_clone(status));
  queue->active = NULL;
}

static iree_status_t iree_hal_amd_xdna_queue_arm(
    iree_hal_amd_xdna_queue_t* queue) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      queue->context->api->kernel_queue_request_notification(
          queue->handle, queue->active_submission, &queue->native_event),
      "kernel_queue_request_notification"));
  iree_async_operation_initialize(&queue->event_wait.base,
                                  IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  queue->event_wait.base.completion_fn, queue);
  return iree_async_proactor_submit_one(queue->proactor,
                                        &queue->event_wait.base);
}

static void iree_hal_amd_xdna_operation_retire(
    iree_hal_amd_xdna_queue_t* queue,
    iree_hal_amd_xdna_operation_t* operation) {
  if (queue->tracker && iree_status_is_ok(operation->status)) {
    ++queue->epoch;
    iree_async_frontier_tracker_advance(queue->tracker, queue->axis,
                                        queue->epoch);
    iree_async_single_frontier_initialize(&operation->frontier, queue->axis,
                                          queue->epoch);
  }
}

// Runs semaphore-ready work. Only the proactor owner touches scheduling state.
// Each operation publishes completion before following work begins. A ready
// successor retains the queue across publication; without one, publication is
// the final queue access and the caller may immediately destroy the device.
static void iree_hal_amd_xdna_queue_pump(iree_hal_amd_xdna_queue_t* queue) {
  while (!queue->active && queue->ready.head) {
    iree_hal_amd_xdna_operation_t* operation = queue->ready.head;
    queue->ready.head = operation->next;
    if (!queue->ready.head) {
      queue->ready.tail = NULL;
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
          for (iree_host_size_t i = 0; i < operation->transfer.count &&
                                       iree_status_is_ok(operation->status);
               ++i) {
            operation->status = iree_hal_amd_xdna_transfer_execute(
                &operation->transfer.operations[i]);
          }
          break;
        case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH: {
          amdf_xdna_kernel_command_t command = {0};
          operation->status = iree_hal_amd_xdna_function_prepare(
              operation->dispatch.function, operation->dispatch.bindings,
              &command);
          if (iree_status_is_ok(operation->status)) {
            const amdf_xdna_kernel_queue_submission_info_t submission = {
                .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
                .structure_size = sizeof(submission),
                .command_count = 1,
                .commands = &command,
            };
            operation->status = IREE_HAL_AMD_STATUS_FROM_AMDF(
                queue->context->xdna->kernel_queue_submit(
                    queue->handle, &submission, &queue->active_submission),
                "xdna.kernel_queue_submit");
          }
          if (iree_status_is_ok(operation->status)) {
            queue->active = operation;
            iree_status_t status = iree_hal_amd_xdna_queue_arm(queue);
            if (!iree_status_is_ok(status)) {
              iree_hal_amd_xdna_queue_abandon_active(queue, status);
            }
            // Either native execution or the safe-leak failure owner now holds
            // this operation. Neither path can release it here.
            operation = NULL;
          }
          break;
        }
      }
    }
    if (operation) {
      iree_hal_amd_xdna_operation_retire(queue, operation);
      const bool has_ready = queue->ready.head != NULL;
      iree_hal_amd_xdna_operation_complete(operation);
      if (!has_ready) {
        return;
      }
    }
  }
}

static void iree_hal_amd_xdna_queue_native_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_queue_t* queue = user_data;
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
  iree_hal_amd_xdna_operation_t* completed = NULL;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_queue_abandon_active(queue, status);
  } else if (checked.retired_submission >= queue->active_submission) {
    completed = queue->active;
    queue->active = NULL;
    completed->status = IREE_HAL_AMD_STATUS_FROM_AMDF(checked.terminal_status,
                                                      "XDNA command outcome");
    if (!iree_status_is_ok(completed->status)) {
      queue->failure_status = iree_status_clone(completed->status);
    }
    iree_hal_amd_xdna_operation_retire(queue, completed);
  } else if (checked.state != AMDF_QUEUE_STATE_ACTIVE) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(checked.terminal_status,
                                           "XDNA queue terminated");
    iree_hal_amd_xdna_queue_abandon_active(queue, status);
  } else {
    status = iree_hal_amd_xdna_queue_arm(queue);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_xdna_queue_abandon_active(queue, status);
    }
  }
  if (completed) {
    const bool has_ready = queue->ready.head != NULL;
    iree_hal_amd_xdna_operation_complete(completed);
    if (!has_ready) {
      return;
    }
  }
  iree_hal_amd_xdna_queue_pump(queue);
}

static void iree_hal_amd_xdna_queue_ready(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  iree_hal_amd_xdna_queue_t* queue = operation->queue;
  operation->status = status;
  if (queue->ready.tail) {
    queue->ready.tail->next = operation;
  } else {
    queue->ready.head = operation;
  }
  queue->ready.tail = operation;
  iree_hal_amd_xdna_queue_pump(queue);
}

static iree_status_t iree_hal_amd_xdna_operation_create(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_length,
    iree_hal_amd_xdna_operation_t** out_operation, void** out_payload) {
  iree_host_size_t size = 0, wait_semaphores_offset = 0, wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0, signal_values_offset = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_operation_t), &size,
      IREE_STRUCT_FIELD(waits.count, iree_hal_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(waits.count, uint64_t, &wait_values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signals.count, uint64_t, &signal_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(payload_length, uint8_t, 16, &payload_offset)));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, size, (void**)&operation));
  operation->queue = queue;
  operation->host_allocator = queue->host_allocator;
  iree_hal_queue_retain(&queue->base);
  uint8_t* slab = (uint8_t*)operation;
  operation->waits = (iree_hal_semaphore_list_t){
      .count = waits.count,
      .semaphores = (iree_hal_semaphore_t**)(slab + wait_semaphores_offset),
      .payload_values = (uint64_t*)(slab + wait_values_offset),
  };
  operation->signals = (iree_hal_semaphore_list_t){
      .count = signals.count,
      .semaphores = (iree_hal_semaphore_t**)(slab + signal_semaphores_offset),
      .payload_values = (uint64_t*)(slab + signal_values_offset),
  };
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    operation->waits.semaphores[i] = waits.semaphores[i];
    operation->waits.payload_values[i] = waits.payload_values[i];
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
  }
  iree_hal_semaphore_list_retain(operation->waits);
  iree_hal_semaphore_list_retain(operation->signals);
  iree_async_operation_initialize(
      &operation->wait.base,
      waits.count ? IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT
                  : IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amd_xdna_queue_ready, operation);
  operation->wait.semaphores =
      (iree_async_semaphore_t**)operation->waits.semaphores;
  operation->wait.values = operation->waits.payload_values;
  operation->wait.count = waits.count;
  operation->wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
  *out_operation = operation;
  *out_payload = slab + payload_offset;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_operation_discard(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_operation_release_resources(operation);
  iree_hal_semaphore_list_release(operation->signals);
  iree_allocator_free(operation->host_allocator, operation);
}

static iree_status_t iree_hal_amd_xdna_operation_submit(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &operation->wait.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_barrier(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_queue_barrier_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals, 0, &operation,
      &payload));
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
  iree_host_size_t payload_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &payload_length,
      IREE_STRUCT_FIELD_FAM(bindings.count,
                            iree_hal_amd_xdna_executable_binding_t)));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals, payload_length,
      &operation, &payload));
  iree_status_t status = iree_hal_amd_xdna_executable_resolve(
      executable, function, bindings, payload, &operation->dispatch.function);
  if (iree_status_is_ok(status)) {
    operation->kind = IREE_HAL_AMD_XDNA_OPERATION_DISPATCH;
    operation->dispatch.executable = executable;
    iree_hal_executable_retain(executable);
    operation->dispatch.binding_count = bindings.count;
    operation->dispatch.bindings = payload;
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
  iree_host_size_t payload_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &payload_length,
      IREE_STRUCT_FIELD_FAM(count, iree_hal_transfer_operation_t)));
  const iree_host_size_t descriptor_length = payload_length;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_hal_transfer_operation_t* operation = &operations[i];
    iree_host_size_t length = 0;
    if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL) {
      length = operation->fill.length ? operation->fill.pattern_length : 0;
    } else if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) {
      length = operation->update.length;
    }
    if (!iree_host_size_checked_add(payload_length, length, &payload_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA transfer capture overflows host size");
    }
  }
  iree_hal_amd_xdna_operation_t* operation = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals, payload_length,
      &operation, &payload));
  operation->kind = IREE_HAL_AMD_XDNA_OPERATION_TRANSFER;
  operation->transfer.count = count;
  operation->transfer.operations = payload;
  uint8_t* data = (uint8_t*)payload + descriptor_length;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_transfer_operation_t* captured =
        &operation->transfer.operations[i];
    *captured = operations[i];
    if (captured->type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
        captured->fill.length) {
      memcpy(data, captured->fill.pattern, captured->fill.pattern_length);
      captured->fill.pattern = data;
      data += captured->fill.pattern_length;
    } else if (captured->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
               captured->update.length) {
      memcpy(data,
             (const uint8_t*)captured->update.source_buffer +
                 captured->update.source_offset,
             captured->update.length);
      captured->update.source_buffer = data;
      captured->update.source_offset = 0;
      data += captured->update.length;
    }
    iree_hal_buffer_t* source = NULL;
    iree_hal_buffer_t* target = NULL;
    iree_hal_amd_xdna_transfer_buffers(captured, &source, &target);
    iree_hal_buffer_retain(source);
    iree_hal_buffer_retain(target);
  }
  return iree_hal_amd_xdna_operation_submit(operation);
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
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_initialize(family, &params, &iree_hal_amd_xdna_queue_vtable,
                            &queue->base);
  queue->host_allocator = host_allocator;
  queue->context = context;
  queue->proactor = proactor;
  const amdf_xdna_kernel_queue_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .queue_family_ordinal = context->queue_family_ordinal,
      .maximum_pending_submission_count = 1,
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
