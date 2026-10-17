// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/event.h"

#include <limits.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/instance.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/wait.h"

struct amdf_gpu_umd_event_t {
  // Device borrowed through native event destruction.
  amdf_gpu_umd_device_t* device;
  // Caller-serialized native wait state: signal, memory and hardware events.
  // The signal's durable age closes native check-to-wait and early-wake races.
  struct kfd_event_data events[3];
};

amdf_status_t amdf_gpu_umd_event_create(
    amdf_gpu_umd_device_t* device, amdf_gpu_umd_event_t** out_event,
    amdf_gpu_umd_event_result_t* out_result) {
  amdf_gpu_umd_event_t* event = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*event),
                  amdf_alignof(amdf_gpu_umd_event_t), (void**)&event);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  event->device = device;
  amdf_gpu_kfd_event_info_t info = {0};
  amdf_gpu_kfd_device_lock_memory(device, AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
  status = amdf_gpu_kfd_instance_create_event(
      device->instance, &device->topology, device->page_size, &info);
  amdf_gpu_kfd_device_unlock_memory(device);
  if (amdf_status_is_ok(status)) {
    event->events[0].event_id = info.signal_id;
    event->events[0].signal_event_data.last_event_age = 1;
    event->events[1].event_id = info.memory_exception_id;
    event->events[2].event_id = info.hardware_exception_id;
    *out_result = (amdf_gpu_umd_event_result_t){
        .format = AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL,
        .format_version = 1,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
        .mailbox_address = info.mailbox_address,
        .trigger_data = info.trigger_data,
        .interrupt_context = info.signal_id,
    };
    *out_event = event;
  } else {
    amdf_free(device->host_allocator, event);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_event_signal(amdf_gpu_umd_event_t* event) {
  struct kfd_ioctl_set_event_args set = {
      .event_id = event->events[0].event_id,
  };
  int result;
  do {
    result = ioctl(event->device->descriptor, AMDKFD_IOC_SET_EVENT, &set);
  } while (result == -1 && errno == EINTR);
  return result == 0 ? AMDF_STATUS_OK : amdf_linux_error(errno);
}

// KFD clamps finite millisecond values to INT32_MAX. Split longer waits into
// native chunks while preserving the caller's absolute deadline.
static uint32_t amdf_gpu_kfd_event_timeout_milliseconds(uint64_t nanoseconds) {
  if (nanoseconds == AMDF_TIMEOUT_INFINITE) {
    return UINT32_MAX;
  }
  const uint64_t milliseconds =
      nanoseconds / UINT64_C(1000000) + (nanoseconds % UINT64_C(1000000) != 0);
  return milliseconds < INT32_MAX ? (uint32_t)milliseconds : INT32_MAX;
}

amdf_status_t amdf_gpu_umd_event_wait(amdf_gpu_umd_event_t* event,
                                      uint64_t timeout_nanoseconds) {
  amdf_wait_deadline_t deadline = {0};
  if (timeout_nanoseconds != AMDF_TIMEOUT_INFINITE) {
    const amdf_status_t status =
        amdf_wait_deadline_initialize(timeout_nanoseconds, 0, &deadline);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  uint64_t remaining_nanoseconds = timeout_nanoseconds;
  while (true) {
    // WAIT_EVENTS copies payloads only for activated exceptions. These fields
    // have bounded native values; the sentinel identifies untouched entries.
    event->events[1].memory_exception_data.failure.NotPresent = UINT32_MAX;
    event->events[2].hw_exception_data.reset_type = UINT32_MAX;
    struct kfd_ioctl_wait_events_args wait = {
        .events_ptr = (uintptr_t)event->events,
        .num_events = 3,
        .timeout =
            amdf_gpu_kfd_event_timeout_milliseconds(remaining_nanoseconds),
    };
    if (ioctl(event->device->descriptor, AMDKFD_IOC_WAIT_EVENTS, &wait) != 0) {
      // An interrupted observation prompts the same predicate recheck as a
      // coalesced/spurious wake. Reentering here could hide already-consumed
      // notification state from the authoritative completion-record owner.
      return errno == EINTR ? AMDF_STATUS_OK : amdf_linux_error(errno);
    }
    if (wait.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
      if (event->events[1].memory_exception_data.failure.NotPresent !=
              UINT32_MAX ||
          event->events[2].hw_exception_data.reset_type != UINT32_MAX) {
        return amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
      }
      return AMDF_STATUS_OK;
    }
    if (wait.wait_result != KFD_IOC_WAIT_RESULT_TIMEOUT ||
        timeout_nanoseconds == AMDF_TIMEOUT_INFINITE) {
      return amdf_linux_error(EPROTO);
    }
    amdf_wait_budget_t remaining;
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    remaining_nanoseconds = remaining.timeout;
  }
}

amdf_status_t amdf_gpu_umd_event_destroy(amdf_gpu_umd_event_t* event) {
  struct kfd_ioctl_destroy_event_args destroy = {
      .event_id = event->events[0].event_id,
  };
  if (ioctl(event->device->descriptor, AMDKFD_IOC_DESTROY_EVENT, &destroy) !=
      0) {
    return amdf_linux_error(errno);
  }
  amdf_free(event->device->host_allocator, event);
  return AMDF_STATUS_OK;
}
