// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/event.h"

#include <limits.h>
#include <linux/kfd_ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/rocjitsu/buffer.h"
#include "libamdf/src/gpu/umd/rocjitsu/device.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/wait.h"

struct amdf_gpu_umd_event_t {
  // Source borrowed through the creating device until event destruction.
  amdf_rocjitsu_source_t* source;
  // Immutable ID shared by concurrent host signaling and waiting.
  uint32_t event_id;
  // Caller-serialized readiness age retained across native waits.
  uint64_t last_event_age;
};

amdf_status_t amdf_rocjitsu_event_source_release(
    amdf_rocjitsu_source_t* source) {
  amdf_rocjitsu_buffer_t* storage = source->events.storage;
  if (storage == NULL) {
    return AMDF_STATUS_OK;
  }
  if (source->events.ready) {
    rj_vm_unmap_t unmap = {
        .addr = (uintptr_t)storage->pointer,
        .length = storage->byte_length,
    };
    const amdf_status_t status = amdf_rocjitsu_source_unmap(source, &unmap);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    // This page-aligned table has no reservation padding. The explicit unmap
    // detaches EventState's borrow and consumes the entire host reservation.
    storage->reservation = NULL;
    storage->host_mapped = false;
    source->events.ready = false;
  }
  const amdf_status_t status = amdf_rocjitsu_buffer_destroy(storage);
  if (amdf_status_is_ok(status)) {
    source->events.storage = NULL;
  }
  return status;
}

static amdf_status_t amdf_rocjitsu_event_source_prepare(
    amdf_rocjitsu_source_t* source) {
  if (source->events.ready) {
    return AMDF_STATUS_OK;
  }
  // Incomplete construction has never published producer operands. Release
  // any retained partial backing before another preparation attempt.
  amdf_status_t status = amdf_rocjitsu_event_source_release(source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t* gpu_ids = NULL;
  status = amdf_malloc(source->host_allocator,
                       (size_t)source->gpu_count * sizeof(*gpu_ids),
                       amdf_alignof(uint32_t), (void**)&gpu_ids);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t i = 0; i < source->gpu_count; ++i) {
    gpu_ids[i] = source->gpus[i].native.gpu_id;
  }
  status = amdf_rocjitsu_buffer_prepare(
      source, source->gpu_count, gpu_ids, AMDF_MEMORY_CLASS_SYSTEM,
      (uint64_t)KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t), 4096,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      &source->events.storage);
  amdf_free(source->host_allocator, gpu_ids);
  source->events.ready = amdf_status_is_ok(status);
  return status;
}

amdf_status_t amdf_rocjitsu_event_create(
    amdf_gpu_umd_device_t* device, amdf_gpu_umd_event_t** out_event,
    amdf_gpu_umd_event_result_t* out_result) {
  amdf_rocjitsu_source_t* source = device->gpu->source;
  amdf_gpu_umd_event_t* event = NULL;
  amdf_status_t status =
      amdf_calloc(source->host_allocator, sizeof(*event),
                  amdf_alignof(amdf_gpu_umd_event_t), (void**)&event);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_platform_instance_lock_native(source->platform);
  status = amdf_rocjitsu_event_source_prepare(source);
  struct kfd_ioctl_create_event_args create = {0};
  if (amdf_status_is_ok(status)) {
    create.event_page_offset = source->events.storage->handle;
    create.event_type = KFD_IOC_EVENT_SIGNAL;
    create.auto_reset = 1;
    status = amdf_rocjitsu_source_command(source, AMDKFD_IOC_CREATE_EVENT,
                                          &create, sizeof(create));
  }
  amdf_platform_instance_unlock_native(source->platform);
  if (amdf_status_is_ok(status)) {
    event->source = source;
    event->event_id = create.event_id;
    event->last_event_age = 1;
    *out_result = (amdf_gpu_umd_event_result_t){
        .format = AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL,
        .format_version = 1,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
        .mailbox_address = (uintptr_t)source->events.storage->pointer +
                           (uint64_t)create.event_slot_index * sizeof(uint64_t),
        .trigger_data = create.event_trigger_data,
        .interrupt_context = create.event_id,
    };
    *out_event = event;
  } else {
    amdf_free(source->host_allocator, event);
  }
  return status;
}

amdf_status_t amdf_rocjitsu_event_signal(amdf_gpu_umd_event_t* event) {
  struct kfd_ioctl_set_event_args signal = {.event_id = event->event_id};
  return amdf_rocjitsu_source_command(event->source, AMDKFD_IOC_SET_EVENT,
                                      &signal, sizeof(signal));
}

static uint32_t amdf_rocjitsu_event_timeout_milliseconds(uint64_t nanoseconds) {
  if (nanoseconds == AMDF_TIMEOUT_INFINITE) {
    return UINT32_MAX;
  }
  const uint64_t milliseconds =
      nanoseconds / UINT64_C(1000000) + (nanoseconds % UINT64_C(1000000) != 0);
  return milliseconds < INT32_MAX ? (uint32_t)milliseconds : INT32_MAX;
}

amdf_status_t amdf_rocjitsu_event_wait(amdf_gpu_umd_event_t* event,
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
    struct {
      // The explicit transport expects its event array inline after the header.
      struct kfd_ioctl_wait_events_args arguments;
      // Native readiness age copied back only after a successful command.
      struct kfd_event_data event;
    } wait = {
        .arguments =
            {
                .num_events = 1,
                .timeout = amdf_rocjitsu_event_timeout_milliseconds(
                    remaining_nanoseconds),
            },
        .event =
            {
                .event_id = event->event_id,
                .signal_event_data = {.last_event_age = event->last_event_age},
            },
    };
    const amdf_status_t status = amdf_rocjitsu_source_command(
        event->source, AMDKFD_IOC_WAIT_EVENTS, &wait, sizeof(wait));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    event->last_event_age = wait.event.signal_event_data.last_event_age;
    if (wait.arguments.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
      return AMDF_STATUS_OK;
    }
    if (wait.arguments.wait_result != KFD_IOC_WAIT_RESULT_TIMEOUT ||
        timeout_nanoseconds == AMDF_TIMEOUT_INFINITE) {
      return amdf_linux_error(EPROTO);
    }
    amdf_wait_budget_t remaining;
    const amdf_status_t remaining_status =
        amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(remaining_status)) {
      return remaining_status;
    }
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    remaining_nanoseconds = remaining.timeout;
  }
}

amdf_status_t amdf_rocjitsu_event_destroy(amdf_gpu_umd_event_t* event) {
  struct kfd_ioctl_destroy_event_args destroy = {.event_id = event->event_id};
  const amdf_status_t status = amdf_rocjitsu_source_command(
      event->source, AMDKFD_IOC_DESTROY_EVENT, &destroy, sizeof(destroy));
  if (amdf_status_is_ok(status)) {
    amdf_free(event->source->host_allocator, event);
  }
  return status;
}
