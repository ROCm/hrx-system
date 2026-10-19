// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/kernel_queue.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/wait.h"

enum {
  AMDF_GPU_WINDOWS_WAIT_SIGNALED = 0,
};

struct amdf_gpu_umd_kernel_queue_t {
  // Device and WKMI adapter borrowed for the lifetime of the queue.
  amdf_gpu_umd_device_t* device;
  // Private native queue state owned by the loaded WKMI bridge.
  amdf_wkmi_bridge_gpu_kernel_queue_t* native;
  // Required command-buffer base alignment in bytes.
  uint32_t command_buffer_alignment;
  // Maximum byte length accepted by one native submission.
  uint64_t maximum_command_buffer_byte_length;
  // Monitored progress-fence object owned by `native`.
  D3DKMT_HANDLE progress_fence;
  // CPU-readable monotonic hardware-queue progress value.
  const volatile uint64_t* progress_fence_pointer;
  // GPU-visible progress address retained for queue interoperability.
  uint64_t progress_fence_device_address;
  // Native execution context borrowed from the bridge queue.
  D3DKMT_HANDLE context;
  // Native hardware queue borrowed for ordering mapping completion.
  D3DKMT_HANDLE hardware_queue;
  // First failure after native acceptance, independent of retirement.
  amdf_atomic_uint64_t failure;
  // Serializes use of the reusable asynchronous wait event.
  SRWLOCK wait_lock;
  // Manual-reset event reused by externally serialized native waits.
  HANDLE wait_event;
  // Native point whose notification a waiter can reuse, or zero when idle.
  // Older registrations may still signal the same event after a consumed wake.
  uint64_t wait_event_submission;
};

amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_gpu_umd_kernel_queue_t** out_queue) {
  amdf_wkmi_bridge_gpu_queue_command_type_t native_command_type;
  switch (command_type) {
    case AMDF_QUEUE_COMMAND_TYPE_GPU_PM4:
      native_command_type = AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4;
      break;
    case AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA:
      native_command_type = AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA;
      break;
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_gpu_umd_kernel_queue_t* queue = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(amdf_gpu_umd_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->device = device;
  amdf_atomic_uint64_initialize(&queue->failure, AMDF_STATUS_OK);
  InitializeSRWLock(&queue->wait_lock);

  status = AMDF_STATUS_OK;
  queue->wait_event = CreateEventW(NULL, TRUE, FALSE, NULL);
  if (queue->wait_event == NULL) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }

  amdf_wkmi_bridge_gpu_kernel_queue_info_t queue_info = {0};
  amdf_wkmi_bridge_gpu_kernel_queue_create_info_t create_info = {
      .structure_size = sizeof(create_info),
      .device_handle = device->device,
      .command_type = native_command_type,
  };
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_wddm_wkmi_adapter_create_kernel_queue(
        &device->wkmi_adapter, &create_info, &queue->native, &queue_info);
  }
  if (amdf_status_is_ok(status)) {
    queue->command_buffer_alignment = queue_info.command_buffer_alignment;
    queue->maximum_command_buffer_byte_length =
        queue_info.maximum_command_buffer_byte_length;
    queue->progress_fence = queue_info.progress_fence_handle;
    queue->context = queue_info.context_handle;
    queue->hardware_queue = queue_info.queue_handle;
    queue->progress_fence_pointer = queue_info.progress_fence_pointer;
    queue->progress_fence_device_address =
        queue_info.progress_fence_device_address;
    *out_queue = queue;
  } else {
    if (queue->wait_event != NULL && !CloseHandle(queue->wait_event)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    amdf_free(device->host_allocator, queue);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_submit(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t submission) {
  const amdf_status_t terminal_status =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (command_buffer_address == 0 || command_buffer_byte_length == 0 ||
      command_buffer_address % queue->command_buffer_alignment != 0 ||
      command_buffer_byte_length % queue->command_buffer_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (command_buffer_byte_length > queue->maximum_command_buffer_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  MemoryBarrier();
  bool accepted = false;
  amdf_status_t status = amdf_gpu_wddm_wkmi_adapter_submit_kernel_queue(
      &queue->device->wkmi_adapter, queue->native, command_buffer_address,
      command_buffer_byte_length, submission, &accepted);
  if (!amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_observe_error(
        &queue->device->status, queue->device->kmt, queue->device->device,
        status);
    if (accepted) {
      // The command is already live. Publish its point and surface the failed
      // completion signal through queue status, never as retryable rejection.
      amdf_atomic_uint64_store_release(&queue->failure, status);
      return AMDF_STATUS_OK;
    }
  }
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  const amdf_status_t failure =
      amdf_atomic_uint64_load_acquire(&queue->failure);
  return amdf_status_is_ok(failure)
             ? amdf_kmt_device_status_query(&queue->device->status)
             : failure;
}

uint32_t amdf_gpu_umd_kernel_queue_mapping_capacity(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  // WDDM can block the caller beyond 128 queued updates. Counting the entire
  // unretired prefix is conservative and needs no second operation ledger.
  return queue->hardware_queue != 0 &&
                 queue->device->kmt->update_gpu_virtual_address != NULL &&
                 queue->device->kmt->wait_on_hardware_queue != NULL
             ? 128
             : 0;
}

amdf_status_t amdf_gpu_wddm_kernel_queue_update_memory(
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations, uint64_t submission,
    bool* mutation_started, bool* accepted) {
  amdf_status_t status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  D3DKMT_UPDATEGPUVIRTUALADDRESS update = {0};
  update.hDevice = queue->device->device;
  update.hContext = queue->context;
  update.hFenceObject = queue->progress_fence;
  update.NumOperations = operation_count;
  update.Operations = operations;
  update.FenceValue = submission - 1;
  *mutation_started = true;
  status = amdf_kmt_make_status(
      queue->device->kmt->update_gpu_virtual_address(&update));
  if (!amdf_status_is_ok(status)) {
    return amdf_kmt_device_status_observe_error(&queue->device->status,
                                                queue->device->kmt,
                                                queue->device->device, status);
  }
  *accepted = true;
  D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {0};
  wait.hHwQueue = queue->hardware_queue;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = &queue->progress_fence;
  wait.FenceValueArray = &submission;
  status =
      amdf_kmt_make_status(queue->device->kmt->wait_on_hardware_queue(&wait));
  if (!amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_observe_error(
        &queue->device->status, queue->device->kmt, queue->device->device,
        status);
    // The mapping owns its accepted point even though later commands cannot
    // safely follow it. Never turn that acceptance into a retryable rejection.
    amdf_atomic_uint64_store_release(&queue->failure, status);
  }
  return status;
}

uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  return amdf_kmt_device_status_query_fence_progress(
      &queue->device->status, queue->progress_fence_pointer);
}

amdf_native_event_types_t amdf_gpu_umd_kernel_queue_query_notification_types(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
}

amdf_status_t amdf_gpu_umd_kernel_queue_request_notification(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  if (native_submission == 0) {
    return amdf_platform_native_event_signal(event);
  }
  // Sampling progress detects the reset sentinel before the completion check.
  // A reset wakes native waiters but never establishes successful retirement.
  const uint64_t progress = amdf_gpu_umd_kernel_queue_query_progress(queue);
  const amdf_status_t terminal_status =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (progress >= native_submission) {
    return amdf_platform_native_event_signal(event);
  }
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {0};
  wait.hDevice = queue->device->device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = &queue->progress_fence;
  wait.FenceValueArray = &native_submission;
  wait.hAsyncEvent = event->payload.native_handle;
  const amdf_status_t status =
      amdf_kmt_make_status(queue->device->kmt->wait_from_cpu(&wait));
  return amdf_status_is_ok(status)
             ? status
             : amdf_kmt_device_status_observe_error(
                   &queue->device->status, queue->device->kmt,
                   queue->device->device, status);
}

amdf_status_t amdf_gpu_umd_kernel_queue_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  const amdf_status_t terminal_status =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  amdf_wait_budget_t remaining;
  while (amdf_gpu_umd_kernel_queue_query_progress(queue) < native_submission) {
    amdf_status_t status =
        amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (remaining.poll == 0) {
      break;
    }
    YieldProcessor();
  }
  // A finite waiter never blocks acquiring the reusable event behind an
  // infinite native wait. Contention consumes the same original deadline.
  for (;;) {
    const uint64_t progress = amdf_gpu_umd_kernel_queue_query_progress(queue);
    amdf_status_t status =
        amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
    if (!amdf_status_is_ok(status) || progress >= native_submission) {
      return status;
    }
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (TryAcquireSRWLockExclusive(&queue->wait_lock)) {
      break;
    }
    amdf_platform_wait_yield();
  }
  amdf_status_t status = AMDF_STATUS_OK;
  while (amdf_status_is_ok(status) &&
         amdf_gpu_umd_kernel_queue_query_progress(queue) < native_submission) {
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    if (remaining.timeout == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      break;
    }
    // A timed-out later wait cannot supply an earlier point's wake. Register
    // that earlier point independently; all wakes still require a fence check.
    if (queue->wait_event_submission == 0 ||
        native_submission < queue->wait_event_submission) {
      if (!ResetEvent(queue->wait_event)) {
        status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
        break;
      }
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {0};
      wait.hDevice = queue->device->device;
      wait.ObjectCount = 1;
      wait.ObjectHandleArray = &queue->progress_fence;
      wait.FenceValueArray = &native_submission;
      wait.hAsyncEvent = queue->wait_event;
      status = amdf_kmt_make_status(queue->device->kmt->wait_from_cpu(&wait));
      if (!amdf_status_is_ok(status)) {
        status = amdf_kmt_device_status_observe_error(
            &queue->device->status, queue->device->kmt, queue->device->device,
            status);
        break;
      }
      queue->wait_event_submission = native_submission;
    }
    // Native event registration may itself consume time, so sample again.
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    DWORD wait_milliseconds = INFINITE;
    if (remaining.timeout != AMDF_TIMEOUT_INFINITE) {
      const uint64_t milliseconds =
          remaining.timeout / UINT64_C(1000000) +
          (remaining.timeout % UINT64_C(1000000) != 0);
      wait_milliseconds =
          milliseconds >= MAXDWORD ? MAXDWORD - 1 : (DWORD)milliseconds;
    }
    const DWORD wait_result =
        WaitForSingleObject(queue->wait_event, wait_milliseconds);
    if (wait_result == WAIT_TIMEOUT) {
      continue;
    }
    if (wait_result != AMDF_GPU_WINDOWS_WAIT_SIGNALED) {
      status = wait_result == WAIT_FAILED
                   ? amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError())
                   : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      break;
    }
    queue->wait_event_submission = 0;
  }
  ReleaseSRWLockExclusive(&queue->wait_lock);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue) {
  amdf_status_t status = amdf_gpu_wddm_wkmi_adapter_destroy_kernel_queue(
      &queue->device->wkmi_adapter, queue->native);
  // Accepted work is retired and no waiter borrows the event. Its release is
  // independent of native queue removal; preserve the first cleanup error.
  if (!CloseHandle(queue->wait_event) && amdf_status_is_ok(status)) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  amdf_free(queue->device->host_allocator, queue);
  return status;
}
