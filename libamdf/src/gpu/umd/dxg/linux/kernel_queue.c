// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/kernel_queue.h"

#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/gpu/umd/dxg/adapter.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"

struct amdf_gpu_umd_kernel_queue_t {
  // Device and private bridge adapter borrowed for the queue lifetime.
  amdf_gpu_umd_device_t* device;
  // Private native queue state owned by the loaded ROCDXG bridge.
  amdf_dxg_bridge_gpu_kernel_queue_t* native;
  // Required command-buffer base and length alignment in bytes.
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
  pthread_mutex_t wait_mutex;
  // Nonblocking eventfd reused by externally serialized native waits.
  int wait_event;
  // Native point whose notification a waiter can reuse, or zero when idle.
  // Older registrations may still signal the same event after a consumed wake.
  uint64_t wait_event_submission;
  // Checked progress preserved before native fence storage is removed.
  uint64_t inactivated_progress;
  // First native removal result, reused after native becomes NULL.
  amdf_status_t inactivation_status;
};

static void amdf_gpu_linux_dxg_kernel_queue_unlock(
    amdf_gpu_umd_kernel_queue_t* queue) {
  const int error = pthread_mutex_unlock(&queue->wait_mutex);
  amdf_assert(error == 0);
  (void)error;
}

// Drains every coalesced wake from one nonblocking eventfd. Native progress,
// rather than event credits, determines completion.
static amdf_status_t amdf_gpu_linux_dxg_kernel_queue_drain_event(
    amdf_gpu_umd_kernel_queue_t* queue) {
  eventfd_t value = 0;
  for (;;) {
    if (eventfd_read(queue->wait_event, &value) == 0) {
      continue;
    }
    if (errno == EINTR) {
      continue;
    }
    return errno == EAGAIN ? AMDF_STATUS_OK : amdf_linux_error(errno);
  }
}

static amdf_status_t amdf_gpu_linux_dxg_kernel_queue_register_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    int event_descriptor) {
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
      .hDevice = queue->device->logical_device.native,
      .ObjectCount = 1,
      .ObjectHandleArray = &queue->progress_fence,
      .FenceValueArray = &native_submission,
      // Linux dxgkrnl interprets the HANDLE-sized field as an eventfd number.
      .hAsyncEvent = (HANDLE)(uintptr_t)(uint32_t)event_descriptor,
  };
  amdf_status_t status =
      amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_wait_from_cpu(
          queue->device->logical_device.kmt, &wait));
  if (!amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_device_status_observe_error(
        &queue->device->status, queue->device->logical_device.kmt,
        queue->device->logical_device.native, status);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_create(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_gpu_umd_kernel_queue_t** out_queue) {
  amdf_dxg_bridge_gpu_queue_command_type_t native_command_type;
  switch (command_type) {
    case AMDF_QUEUE_COMMAND_TYPE_GPU_PM4:
      native_command_type = AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4;
      break;
    case AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA:
      native_command_type = AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA;
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
  queue->wait_event = -1;
  amdf_atomic_uint64_initialize(&queue->failure, AMDF_STATUS_OK);

  const int mutex_error = pthread_mutex_init(&queue->wait_mutex, NULL);
  if (mutex_error != 0) {
    amdf_free(device->host_allocator, queue);
    return amdf_linux_error(mutex_error);
  }
  queue->wait_event = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (queue->wait_event < 0) {
    status = amdf_linux_error(errno);
  }

  amdf_dxg_bridge_gpu_kernel_queue_info_t queue_info = {0};
  const amdf_dxg_bridge_gpu_kernel_queue_create_info_t create_info = {
      .structure_size = sizeof(create_info),
      .device_handle = device->logical_device.native,
      .command_type = native_command_type,
  };
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_adapter_create_kernel_queue(
        &device->logical_device.bridge_adapter, &create_info, &queue->native,
        &queue_info);
  }
  if (amdf_status_is_ok(status)) {
    queue->command_buffer_alignment = queue_info.command_buffer_alignment;
    queue->maximum_command_buffer_byte_length =
        queue_info.maximum_command_buffer_byte_length;
    queue->progress_fence = queue_info.progress_fence_handle;
    queue->progress_fence_pointer = queue_info.progress_fence_pointer;
    queue->progress_fence_device_address =
        queue_info.progress_fence_device_address;
    queue->context = queue_info.context_handle;
    queue->hardware_queue = queue_info.queue_handle;
    *out_queue = queue;
    return AMDF_STATUS_OK;
  }

  const amdf_status_t close_status = amdf_linux_file_close(&queue->wait_event);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  const int destroy_error = pthread_mutex_destroy(&queue->wait_mutex);
  amdf_assert(destroy_error == 0);
  (void)destroy_error;
  amdf_free(device->host_allocator, queue);
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

  __atomic_thread_fence(__ATOMIC_RELEASE);
  bool accepted = false;
  amdf_status_t status = amdf_gpu_dxg_adapter_submit_kernel_queue(
      &queue->device->logical_device.bridge_adapter, queue->native,
      command_buffer_address, command_buffer_byte_length, submission,
      &accepted);
  if (!amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_device_status_observe_error(
        &queue->device->status, queue->device->logical_device.kmt,
        queue->device->logical_device.native, status);
    if (accepted) {
      // The command is live. Publish its point and surface the failed
      // completion signal through status, never as retryable rejection.
      amdf_atomic_uint64_store_release(&queue->failure, status);
      return AMDF_STATUS_OK;
    }
  }
  return status;
}

uint64_t amdf_gpu_umd_kernel_queue_query_progress(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  return amdf_gpu_linux_dxg_device_status_query_fence_progress(
      &queue->device->status, queue->progress_fence_pointer);
}

amdf_status_t amdf_gpu_umd_kernel_queue_query_terminal_status(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  const amdf_status_t failure =
      amdf_atomic_uint64_load_acquire(&queue->failure);
  return amdf_status_is_ok(failure)
             ? amdf_gpu_linux_dxg_device_status_query(&queue->device->status)
             : failure;
}

uint32_t amdf_gpu_umd_kernel_queue_mapping_capacity(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  // WDDM can block the caller beyond 128 queued updates. Counting the entire
  // unretired prefix is conservative and needs no second operation ledger.
  return queue->hardware_queue != 0 ? 128 : 0;
}

amdf_status_t amdf_gpu_linux_dxg_kernel_queue_update_memory(
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations, uint64_t submission,
    bool* mutation_started, bool* accepted) {
  amdf_status_t status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  D3DKMT_UPDATEGPUVIRTUALADDRESS update = {
      .hDevice = queue->device->logical_device.native,
      .hContext = queue->context,
      .hFenceObject = queue->progress_fence,
      .NumOperations = operation_count,
      .Operations = operations,
      .FenceValue = submission - 1,
  };
  *mutation_started = true;
  status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_update_gpu_virtual_address(
          queue->device->logical_device.kmt, &update));
  if (!amdf_status_is_ok(status)) {
    return amdf_gpu_linux_dxg_device_status_observe_error(
        &queue->device->status, queue->device->logical_device.kmt,
        queue->device->logical_device.native, status);
  }

  *accepted = true;
  D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {
      .hHwQueue = queue->hardware_queue,
      .ObjectCount = 1,
      .ObjectHandleArray = &queue->progress_fence,
      .FenceValueArray = &submission,
  };
  status =
      amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_wait_on_hardware_queue(
          queue->device->logical_device.kmt, &wait));
  if (!amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_device_status_observe_error(
        &queue->device->status, queue->device->logical_device.kmt,
        queue->device->logical_device.native, status);
    // The mapping owns its accepted point even though later commands cannot
    // safely follow it. Never turn acceptance into a retryable rejection.
    amdf_atomic_uint64_store_release(&queue->failure, status);
  }
  return status;
}

amdf_native_event_types_t amdf_gpu_umd_kernel_queue_query_notification_types(
    const amdf_gpu_umd_kernel_queue_t* queue) {
  (void)queue;
  return AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD;
}

amdf_status_t amdf_gpu_umd_kernel_queue_request_notification(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  if (native_submission == 0) {
    return amdf_platform_native_event_signal(event);
  }
  // Sampling progress detects the reset sentinel before the completion check.
  const uint64_t progress = amdf_gpu_umd_kernel_queue_query_progress(queue);
  const amdf_status_t terminal_status =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (progress >= native_submission) {
    return amdf_platform_native_event_signal(event);
  }
  return amdf_gpu_linux_dxg_kernel_queue_register_wait(
      queue, native_submission, (int)event->payload.file_descriptor);
}

amdf_status_t amdf_gpu_umd_kernel_queue_wait(
    amdf_gpu_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  amdf_status_t status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  amdf_wait_budget_t remaining;
  while (amdf_gpu_umd_kernel_queue_query_progress(queue) < native_submission) {
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
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
    __builtin_ia32_pause();
  }

  // A finite waiter never blocks behind an infinite native wait. Contention
  // consumes the same original deadline.
  for (;;) {
    const uint64_t progress = amdf_gpu_umd_kernel_queue_query_progress(queue);
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
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
    const int mutex_error = pthread_mutex_trylock(&queue->wait_mutex);
    if (mutex_error == 0) {
      break;
    }
    if (mutex_error != EBUSY) {
      return amdf_linux_error(mutex_error);
    }
    amdf_platform_wait_yield();
  }

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

    if (queue->wait_event_submission == 0 ||
        native_submission < queue->wait_event_submission) {
      status = amdf_gpu_linux_dxg_kernel_queue_drain_event(queue);
      if (!amdf_status_is_ok(status)) {
        break;
      }
      status = amdf_gpu_linux_dxg_kernel_queue_register_wait(
          queue, native_submission, queue->wait_event);
      if (!amdf_status_is_ok(status)) {
        break;
      }
      queue->wait_event_submission = native_submission;
    }

    // Native registration may itself consume time, so sample the deadline
    // again before entering poll.
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    int timeout_milliseconds = -1;
    if (remaining.timeout != AMDF_TIMEOUT_INFINITE) {
      const uint64_t milliseconds =
          remaining.timeout / UINT64_C(1000000) +
          (remaining.timeout % UINT64_C(1000000) != 0);
      timeout_milliseconds =
          milliseconds > INT_MAX ? INT_MAX : (int)milliseconds;
    }
    struct pollfd wait_event = {
        .fd = queue->wait_event,
        .events = POLLIN,
    };
    const int poll_result = poll(&wait_event, 1, timeout_milliseconds);
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      status = amdf_linux_error(errno);
      break;
    }
    if (poll_result == 0) {
      continue;
    }
    if ((wait_event.revents & POLLNVAL) != 0) {
      status = amdf_linux_error(EBADF);
      break;
    }
    if ((wait_event.revents & (POLLERR | POLLHUP)) != 0) {
      status = amdf_linux_error(EIO);
      break;
    }
    if ((wait_event.revents & POLLIN) != 0) {
      status = amdf_gpu_linux_dxg_kernel_queue_drain_event(queue);
      if (amdf_status_is_ok(status)) {
        queue->wait_event_submission = 0;
      }
    }
  }
  amdf_gpu_linux_dxg_kernel_queue_unlock(queue);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_destroy(
    amdf_gpu_umd_kernel_queue_t* queue) {
  amdf_status_t status = queue->inactivation_status;
  if (queue->native != NULL) {
    status = amdf_gpu_dxg_adapter_destroy_kernel_queue(
        &queue->device->logical_device.bridge_adapter, queue->native);
    queue->native = NULL;
  }
  const amdf_status_t close_status = amdf_linux_file_close(&queue->wait_event);
  if (amdf_status_is_ok(status)) {
    status = close_status;
  }
  const int mutex_error = pthread_mutex_destroy(&queue->wait_mutex);
  amdf_assert(mutex_error == 0);
  (void)mutex_error;
  const amdf_allocator_t host_allocator = queue->device->host_allocator;
  amdf_free(host_allocator, queue);
  return status;
}

amdf_status_t amdf_gpu_umd_kernel_queue_inactivate(
    amdf_gpu_umd_kernel_queue_t* queue) {
  if (queue->native == NULL) {
    return queue->inactivation_status;
  }
  // Native removal consumes the fence mapping. Preserve only independently
  // checked progress and never manufacture completion of aborted work.
  queue->inactivated_progress = amdf_gpu_umd_kernel_queue_query_progress(queue);
  queue->progress_fence_pointer = &queue->inactivated_progress;
  const amdf_status_t failure =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue);
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(
      &queue->failure, &expected,
      amdf_status_is_ok(failure)
          ? amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION)
          : failure);
  queue->inactivation_status = amdf_gpu_dxg_adapter_destroy_kernel_queue(
      &queue->device->logical_device.bridge_adapter, queue->native);
  // The bridge consumes its owner on every result. Never retry removal through
  // a native handle whose storage may already have been reused.
  queue->native = NULL;
  return queue->inactivation_status;
}
