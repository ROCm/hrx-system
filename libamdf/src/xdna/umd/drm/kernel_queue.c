// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include <drm/amdxdna_accel.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/xdna/umd/drm/context.h"
#include "libamdf/src/xdna/umd/drm/elf_packet.h"

enum {
  AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED = 0,
  AMDF_LINUX_XDNA_FIRST_POINT_CAPTURING = 1,
  AMDF_LINUX_XDNA_FIRST_POINT_CAPTURED = 2,
};

struct amdf_xdna_umd_kernel_queue_t {
  // Scheduling context exclusively leased by this public queue.
  amdf_xdna_umd_context_t* context;
  // Number of command buffers acquired during transactional construction.
  uint32_t packet_count;
  // Precreated binary object retaining the first native fence, or zero when
  // this context has already retired its first command before this lease.
  uint32_t first_syncobj;
  // Native notification representations qualified before queue publication.
  amdf_native_event_types_t notification_types;
  // Greatest encoded native point confirmed by a native wait or query.
  amdf_atomic_uint64_t progress;
  // Nonwaiting claim around the one-time capture of native point zero.
  amdf_atomic_uint32_t first_point;
  // Native command/result storage reused only after checked retirement.
  amdf_linux_xdna_buffer_t packets[];
};

// DRM has no separate EVENTFD capability bit. Handle zero cannot name a
// syncobj, so the supported ioctl rejects it before taking an event reference
// or allocating a callback. Missing ioctl/feature support remains explicit.
static amdf_status_t amdf_linux_xdna_probe_notification_types(
    int descriptor, amdf_native_event_types_t* out_types) {
  struct drm_syncobj_eventfd request = {.fd = -1};
  if (ioctl(descriptor, DRM_IOCTL_SYNCOBJ_EVENTFD, &request) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  switch (errno) {
    case ENOENT:
      *out_types = AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD;
      return AMDF_STATUS_OK;
    case EINVAL:
    case ENOTTY:
    case EOPNOTSUPP:
      *out_types = 0;
      return AMDF_STATUS_OK;
    default:
      return amdf_linux_error(errno);
  }
}

static void amdf_linux_xdna_kernel_queue_record_progress(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t completed) {
  uint64_t progress = amdf_atomic_uint64_load_acquire(&queue->progress);
  while (progress < completed && !amdf_atomic_uint64_compare_exchange_acq_rel(
                                     &queue->progress, &progress, completed)) {
  }
}

// DRM point zero means the current fence, not the historical first point.
// Capture that fence before either waiting on it or replacing it with the
// second submission. The destination object already exists; transfer neither
// allocates library state nor waits for execution.
static amdf_status_t amdf_linux_xdna_kernel_queue_capture_first_point(
    amdf_xdna_umd_kernel_queue_t* queue) {
  if (amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
    return AMDF_STATUS_OK;
  }
  uint32_t expected = AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(
          &queue->first_point, &expected,
          AMDF_LINUX_XDNA_FIRST_POINT_CAPTURING)) {
    return expected == AMDF_LINUX_XDNA_FIRST_POINT_CAPTURED
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  struct drm_syncobj_transfer transfer = {
      .src_handle = queue->context->completion_syncobj,
      .dst_handle = queue->first_syncobj,
  };
  const int result = ioctl(queue->context->device->descriptor,
                           DRM_IOCTL_SYNCOBJ_TRANSFER, &transfer);
  const int error = errno;
  amdf_atomic_uint32_store_release(
      &queue->first_point, result == 0
                               ? AMDF_LINUX_XDNA_FIRST_POINT_CAPTURED
                               : AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED);
  return result == 0 ? AMDF_STATUS_OK : amdf_linux_error(error);
}

static void amdf_linux_xdna_context_record_failure(
    amdf_xdna_umd_context_t* context, amdf_status_t failure) {
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(&context->terminal_status,
                                              &expected, failure);
}

static amdf_status_t amdf_linux_xdna_command_query_result(
    const amdf_linux_xdna_buffer_t* packet) {
  // The caller has native fence proof and holds the packet's lifetime borrow.
  // ERT state by itself is never proof that the kernel has stopped using it.
  const uint32_t state =
      __atomic_load_n((const uint32_t*)packet->host_pointer, __ATOMIC_ACQUIRE) &
      0xf;
  if (state == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  return state == 4 ? AMDF_STATUS_OK
                    : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, state);
}

static amdf_status_t amdf_linux_xdna_command_submit(
    amdf_xdna_umd_context_t* context, const amdf_linux_xdna_buffer_t* packet,
    uint64_t* out_sequence) {
  amdf_xdna_umd_device_t* device = context->device;
  if (context->last_native_sequence == UINT64_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  struct amdxdna_drm_exec_cmd submit = {
      .hwctx = context->handle,
      .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
      .cmd_handles = packet->handle,
      // Backing is resident for its allocation lifetime. Only the command BO
      // identifies this native submission; indirect data has no BO list.
      .args = 0,
      .cmd_count = 1,
      .arg_count = 0,
  };
  // No operation after native acceptance may turn this into a rejected submit.
  if (ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_EXEC_CMD, &submit) != 0) {
    return amdf_linux_error(errno);
  }
  // Zero is the public no-work sentinel. The native first sequence is zero;
  // encode it as one and decode only at the DRM wait boundary. The first-fence
  // snapshot protects DRM's special interpretation of point zero.
  context->last_native_sequence = submit.seq + 1;
  *out_sequence = context->last_native_sequence;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_linux_xdna_timeline_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t sequence,
    const amdf_wait_deadline_t* deadline) {
  amdf_xdna_umd_device_t* device = queue->context->device;
  const uint32_t syncobj =
      sequence == 1 ? queue->first_syncobj : queue->context->completion_syncobj;
  const uint64_t native_sequence = sequence - 1;
  for (;;) {
    amdf_wait_budget_t remaining;
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    struct drm_syncobj_timeline_wait wait = {
        .handles = (uintptr_t)&syncobj,
        .points = (uintptr_t)&native_sequence,
        .timeout_nsec =
            remaining.poll != 0 || remaining.timeout == 0
                ? 0
                : (deadline->timeout > INT64_MAX ? INT64_MAX
                                                 : (int64_t)deadline->timeout),
        .count_handles = 1,
        .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
                 DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
    };
    if (ioctl(device->descriptor, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait) ==
        0) {
      return AMDF_STATUS_OK;
    }
    const int error = errno;
    if (error != ETIME && error != EINTR) {
      return amdf_linux_error(error);
    }
    // Retry interrupted waits and explicit active polling with the original
    // absolute deadline. A timeout never authorizes releasing accepted work.
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (error == ETIME && remaining.poll != 0) {
      amdf_platform_wait_yield();
    }
  }
}

static amdf_status_t amdf_linux_xdna_kernel_queue_release_storage(
    amdf_xdna_umd_kernel_queue_t* queue) {
  const int descriptor = queue->context->device->descriptor;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < queue->packet_count; ++i) {
    const amdf_status_t release_status =
        amdf_linux_xdna_buffer_deinitialize(descriptor, &queue->packets[i]);
    if (amdf_status_is_ok(status)) {
      status = release_status;
    }
  }
  if (queue->first_syncobj != 0) {
    struct drm_syncobj_destroy destroy = {.handle = queue->first_syncobj};
    if (ioctl(descriptor, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy) != 0 &&
        amdf_status_is_ok(status)) {
      status = amdf_linux_error(errno);
    }
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context, uint32_t capacity,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_kernel_queue_t* queue = NULL;
  const size_t slot_count = capacity;
  if (slot_count > (SIZE_MAX - sizeof(*queue)) / sizeof(queue->packets[0])) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_status_t status =
      amdf_calloc(device->host_allocator,
                  sizeof(*queue) + slot_count * sizeof(queue->packets[0]),
                  amdf_alignof(amdf_xdna_umd_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&context->queue_leased,
                                                   &expected, 1)) {
    amdf_free(device->host_allocator, queue);
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  queue->context = context;
  amdf_atomic_uint64_initialize(&queue->progress,
                                context->last_native_sequence);
  amdf_atomic_uint32_initialize(&queue->first_point,
                                AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED);
  status = amdf_atomic_uint64_load_acquire(&context->terminal_status);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_probe_notification_types(
        device->descriptor, &queue->notification_types);
  }
  if (amdf_status_is_ok(status) && context->last_native_sequence == 0) {
    struct drm_syncobj_create create = {0};
    if (ioctl(device->descriptor, DRM_IOCTL_SYNCOBJ_CREATE, &create) != 0) {
      status = amdf_linux_error(errno);
    } else {
      queue->first_syncobj = create.handle;
    }
  }
  for (uint32_t i = 0; i < capacity && amdf_status_is_ok(status); ++i) {
    status =
        amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_CMD,
                                      device->page_size, &queue->packets[i]);
    if (amdf_status_is_ok(status)) {
      ++queue->packet_count;
      status = amdf_linux_xdna_buffer_attach(
          device->descriptor, device->page_size, device->page_size, NULL,
          &queue->packets[i]);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    const amdf_status_t release_status =
        amdf_linux_xdna_kernel_queue_release_storage(queue);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
    amdf_atomic_uint32_store_release(&context->queue_leased, 0);
    amdf_free(device->host_allocator, queue);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_native_submission) {
  amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&queue->context->terminal_status);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (queue->context->last_native_sequence == 1) {
    status = amdf_linux_xdna_kernel_queue_capture_first_point(queue);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  amdf_linux_xdna_buffer_t* packet = &queue->packets[slot];
  amdf_linux_xdna_elf_packet_build(instruction_address, instruction_byte_length,
                                   packet->host_pointer);
  amdf_linux_host_cache_transfer(packet->host_pointer,
                                 sizeof(amdf_linux_xdna_elf_packet_t),
                                 queue->context->device->cache_line_size);
  return amdf_linux_xdna_command_submit(queue->context, packet,
                                        out_native_submission);
}

amdf_native_event_types_t amdf_xdna_umd_kernel_queue_query_notification_types(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return queue->notification_types;
}

amdf_status_t amdf_xdna_umd_kernel_queue_request_notification(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event) {
  if (amdf_atomic_uint64_load_acquire(&queue->progress) >= native_submission) {
    return amdf_platform_native_event_signal(event);
  }
  if (native_submission == 1) {
    const amdf_status_t status =
        amdf_linux_xdna_kernel_queue_capture_first_point(queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      // A nonwaiting request must not leave its caller asleep without an arm
      // when the first-fence snapshot belongs to another thread.
      return amdf_platform_native_event_signal(event);
    }
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    // Capture can observe another caller's progress instead of transferring
    // a fence. In that case the still-empty binary snapshot is not a waiter.
    if (amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
      return amdf_platform_native_event_signal(event);
    }
  }
  struct drm_syncobj_eventfd request = {
      .handle = native_submission == 1 ? queue->first_syncobj
                                       : queue->context->completion_syncobj,
      .point = native_submission - 1,
      .fd = (int)event->payload.file_descriptor,
  };
  return ioctl(queue->context->device->descriptor, DRM_IOCTL_SYNCOBJ_EVENTFD,
               &request) == 0
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_atomic_uint64_load_acquire(&queue->progress);
}

amdf_status_t amdf_xdna_umd_kernel_queue_refresh_progress(
    amdf_xdna_umd_kernel_queue_t* queue) {
  uint64_t native_point = 0;
  struct drm_syncobj_timeline_array query = {
      .handles = (uintptr_t)&queue->context->completion_syncobj,
      .points = (uintptr_t)&native_point,
      .count_handles = 1,
  };
  const int descriptor = queue->context->device->descriptor;
  if (ioctl(descriptor, DRM_IOCTL_SYNCOBJ_QUERY, &query) != 0) {
    return amdf_linux_error(errno);
  }
  if (native_point != 0) {
    amdf_linux_xdna_kernel_queue_record_progress(queue, native_point + 1);
    return AMDF_STATUS_OK;
  }
  if (amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
    return AMDF_STATUS_OK;
  }
  // QUERY cannot distinguish an unsignaled first fence from signaled point
  // zero. Poll the exact captured fence instead of granting false retirement.
  amdf_status_t status =
      amdf_linux_xdna_kernel_queue_capture_first_point(queue);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint64_t point = 0;
  struct drm_syncobj_timeline_wait wait = {
      .handles = (uintptr_t)&queue->first_syncobj,
      .points = (uintptr_t)&point,
      .count_handles = 1,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
  };
  if (ioctl(descriptor, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait) == 0) {
    amdf_linux_xdna_kernel_queue_record_progress(queue, 1);
    return AMDF_STATUS_OK;
  }
  return errno == ETIME ? AMDF_STATUS_OK : amdf_linux_error(errno);
}

void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot) {
  const amdf_status_t status =
      amdf_linux_xdna_command_query_result(&queue->packets[slot]);
  if (!amdf_status_is_ok(status)) {
    amdf_linux_xdna_context_record_failure(queue->context, status);
  }
}

amdf_status_t amdf_xdna_umd_kernel_queue_query_terminal_status(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_atomic_uint64_load_acquire(&queue->context->terminal_status);
}

amdf_status_t amdf_xdna_umd_kernel_queue_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  if (amdf_atomic_uint64_load_acquire(&queue->progress) >= native_submission) {
    return AMDF_STATUS_OK;
  }
  if (native_submission == 1) {
    for (;;) {
      const amdf_status_t status =
          amdf_linux_xdna_kernel_queue_capture_first_point(queue);
      if (amdf_status_is_ok(status)) {
        break;
      }
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        return status;
      }
      amdf_wait_budget_t remaining;
      const amdf_status_t budget_status =
          amdf_wait_deadline_query_remaining(deadline, &remaining);
      if (!amdf_status_is_ok(budget_status)) {
        return budget_status;
      }
      if (remaining.timeout == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      }
      amdf_platform_wait_yield();
    }
    if (amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
      return AMDF_STATUS_OK;
    }
  }
  const amdf_status_t status =
      amdf_linux_xdna_timeline_wait(queue, native_submission, deadline);
  if (amdf_status_is_ok(status)) {
    amdf_linux_xdna_kernel_queue_record_progress(queue, native_submission);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue) {
  amdf_xdna_umd_device_t* device = queue->context->device;
  const amdf_status_t status =
      amdf_linux_xdna_kernel_queue_release_storage(queue);
  amdf_atomic_uint32_store_release(&queue->context->queue_leased, 0);
  amdf_free(device->host_allocator, queue);
  return status;
}
