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
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/xdna/umd/drm/context.h"
#include "libamdf/src/xdna/umd/drm/elf_packet.h"

enum {
  AMDF_LINUX_XDNA_FIRST_POINT_OPEN = 0,
  AMDF_LINUX_XDNA_FIRST_POINT_WAITING = 1,
  AMDF_LINUX_XDNA_FIRST_POINT_CLOSED = 2,
};

struct amdf_xdna_umd_kernel_queue_t {
  // Scheduling context exclusively leased by this public queue.
  amdf_xdna_umd_context_t* context;
  // Queue-owned CMD allocation reused only after checked native retirement.
  amdf_linux_xdna_buffer_t packet;
  // Greatest encoded native point confirmed by timeline wait.
  amdf_atomic_uint64_t progress;
  // Serializes native point-zero observers with the second submission. DRM
  // treats zero as the latest fence, so an old observer must leave before that
  // fence can change. The gate remains closed after the first retirement.
  amdf_atomic_uint32_t first_point;
};

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
  if (state == 0) return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
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
  // encode it as one and decode only at the DRM wait boundary. The first-point
  // gate separately protects DRM's special interpretation of point zero.
  context->last_native_sequence = submit.seq + 1;
  *out_sequence = context->last_native_sequence;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_linux_xdna_timeline_wait(
    amdf_xdna_umd_context_t* context, uint64_t sequence,
    const amdf_wait_deadline_t* deadline) {
  amdf_xdna_umd_device_t* device = context->device;
  const uint64_t native_sequence = sequence - 1;
  for (;;) {
    amdf_wait_budget_t remaining;
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) return status;
    struct drm_syncobj_timeline_wait wait = {
        .handles = (uintptr_t)&context->completion_syncobj,
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
    if (error != ETIME && error != EINTR) return amdf_linux_error(error);
    // Retry interrupted waits and explicit active polling with the original
    // absolute deadline. A timeout never authorizes releasing accepted work.
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (error == ETIME && remaining.poll != 0) amdf_platform_wait_yield();
  }
}

amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_kernel_queue_t* queue = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(amdf_xdna_umd_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) return status;
  amdf_atomic_uint64_initialize(&queue->progress, 0);
  amdf_atomic_uint32_initialize(&queue->first_point,
                                AMDF_LINUX_XDNA_FIRST_POINT_OPEN);
  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&context->queue_leased,
                                                   &expected, 1)) {
    amdf_free(device->host_allocator, queue);
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  status = amdf_atomic_uint64_load_acquire(&context->terminal_status);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_CMD,
                                           device->page_size, &queue->packet);
  }
  if (amdf_status_is_ok(status)) {
    status =
        amdf_linux_xdna_buffer_attach(device->descriptor, device->page_size,
                                      device->page_size, NULL, &queue->packet);
  }
  if (amdf_status_is_ok(status)) {
    queue->context = context;
    *out_queue = queue;
  } else {
    const amdf_status_t release_status =
        amdf_linux_xdna_buffer_deinitialize(device->descriptor, &queue->packet);
    if (!amdf_status_is_ok(release_status)) status = release_status;
    amdf_atomic_uint32_store_release(&context->queue_leased, 0);
    amdf_free(device->host_allocator, queue);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t instruction_address,
    uint32_t instruction_byte_length, uint64_t* out_native_submission) {
  const amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&queue->context->terminal_status);
  if (!amdf_status_is_ok(status)) return status;
  if (queue->context->last_native_sequence == 1) {
    uint32_t expected = AMDF_LINUX_XDNA_FIRST_POINT_OPEN;
    if (!amdf_atomic_uint32_compare_exchange_acq_rel(
            &queue->first_point, &expected,
            AMDF_LINUX_XDNA_FIRST_POINT_CLOSED) &&
        expected != AMDF_LINUX_XDNA_FIRST_POINT_CLOSED) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
  }
  amdf_linux_xdna_elf_packet_build(instruction_address, instruction_byte_length,
                                   queue->packet.host_pointer);
  amdf_linux_host_cache_transfer(queue->packet.host_pointer,
                                 sizeof(amdf_linux_xdna_elf_packet_t),
                                 queue->context->device->cache_line_size);
  return amdf_linux_xdna_command_submit(queue->context, &queue->packet,
                                        out_native_submission);
}

uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  return amdf_atomic_uint64_load_acquire(&queue->progress);
}

void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue) {
  const amdf_status_t status =
      amdf_linux_xdna_command_query_result(&queue->packet);
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
      uint32_t expected = AMDF_LINUX_XDNA_FIRST_POINT_OPEN;
      if (amdf_atomic_uint32_compare_exchange_acq_rel(
              &queue->first_point, &expected,
              AMDF_LINUX_XDNA_FIRST_POINT_WAITING)) {
        break;
      }
      if (expected == AMDF_LINUX_XDNA_FIRST_POINT_CLOSED ||
          amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
        return AMDF_STATUS_OK;
      }
      amdf_wait_budget_t remaining;
      const amdf_status_t status =
          amdf_wait_deadline_query_remaining(deadline, &remaining);
      if (!amdf_status_is_ok(status)) return status;
      if (remaining.timeout == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      }
      amdf_platform_wait_yield();
    }
    if (amdf_atomic_uint64_load_acquire(&queue->progress) >= 1) {
      amdf_atomic_uint32_store_release(&queue->first_point,
                                       AMDF_LINUX_XDNA_FIRST_POINT_OPEN);
      return AMDF_STATUS_OK;
    }
  }
  const amdf_status_t status = amdf_linux_xdna_timeline_wait(
      queue->context, native_submission, deadline);
  if (amdf_status_is_ok(status)) {
    uint64_t progress = amdf_atomic_uint64_load_acquire(&queue->progress);
    while (progress < native_submission &&
           !amdf_atomic_uint64_compare_exchange_acq_rel(
               &queue->progress, &progress, native_submission)) {
    }
  }
  if (native_submission == 1) {
    amdf_atomic_uint32_store_release(&queue->first_point,
                                     AMDF_LINUX_XDNA_FIRST_POINT_OPEN);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue) {
  amdf_xdna_umd_device_t* device = queue->context->device;
  const amdf_status_t status =
      amdf_linux_xdna_buffer_deinitialize(device->descriptor, &queue->packet);
  if (!amdf_status_is_ok(status)) return status;
  amdf_atomic_uint32_store_release(&queue->context->queue_leased, 0);
  amdf_free(device->host_allocator, queue);
  return AMDF_STATUS_OK;
}
