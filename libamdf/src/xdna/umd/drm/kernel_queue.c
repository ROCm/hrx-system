// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include <drm/amdxdna_accel.h>
#include <string.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/xdna/elf_packet.h"
#include "libamdf/src/xdna/umd/drm/context.h"

enum {
  AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED = 0,
  AMDF_LINUX_XDNA_FIRST_POINT_CAPTURING = 1,
  AMDF_LINUX_XDNA_FIRST_POINT_CAPTURED = 2,
};

// Packet storage and submission ledgers have the same capacity but independent
// ring positions. An accepted array owns one contiguous packet extent, wrapping
// at capacity, until its root result has been consumed.
typedef struct amdf_linux_xdna_kernel_queue_slot_t {
  // One native BO, indexed by the publisher's packet cursor.
  amdf_linux_xdna_buffer_t packet;
  // Result BO index for the public submission occupying this ledger slot.
  uint32_t root_packet;
  // Packet credits owned by that submission, including any chain envelope.
  uint32_t packet_count;
} amdf_linux_xdna_kernel_queue_slot_t;

struct amdf_xdna_umd_kernel_queue_t {
  // Scheduling context exclusively leased by this public queue.
  amdf_xdna_umd_context_t* context;
  // Number of command buffers acquired during transactional construction.
  uint32_t packet_count;
  // Publisher-owned cursor into the shared packet ring.
  uint32_t next_packet;
  // Publisher-owned aggregate count of accepted packet credits.
  uint64_t published_packets;
  // Aggregate credits returned only after exclusive result consumption.
  amdf_atomic_uint64_t retired_packets;
  // Precreated binary object retaining the first native fence, or zero when
  // this context has already retired its first command before this lease.
  uint32_t first_syncobj;
  // Native notification representations qualified before queue publication.
  amdf_native_event_types_t notification_types;
  // Greatest encoded native point confirmed by a native wait or query.
  amdf_atomic_uint64_t progress;
  // Nonwaiting claim around the one-time capture of native point zero.
  amdf_atomic_uint32_t first_point;
  // Native command/result storage reused only after checked retirement. DRM
  // submits a BO handle with no byte offset, and the kernel later reads and
  // completes that same BO. Independently pending commands require distinct
  // BOs even though their packets occupy less than a page.
  amdf_linux_xdna_kernel_queue_slot_t slots[];
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
  // EXEC_BUF interruptions precede scheduler admission and sequence assignment.
  // Restart that interrupted publication, never an accepted command or another
  // native rejection such as resource exhaustion.
  while (ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_EXEC_CMD, &submit) != 0) {
    const int error = errno;
    if (error != EINTR) {
      return amdf_linux_error(error);
    }
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
    const amdf_status_t release_status = amdf_linux_xdna_buffer_deinitialize(
        descriptor, &queue->slots[i].packet);
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
    amdf_xdna_instruction_features_t required_features,
    amdf_xdna_umd_kernel_queue_t** out_queue) {
  (void)required_features;
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_kernel_queue_t* queue = NULL;
  const size_t slot_count = capacity;
  if (slot_count > (SIZE_MAX - sizeof(*queue)) / sizeof(queue->slots[0])) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_status_t status =
      amdf_calloc(device->host_allocator,
                  sizeof(*queue) + slot_count * sizeof(queue->slots[0]),
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
  status = amdf_linux_xdna_context_prepare_execution(context);
  amdf_atomic_uint64_initialize(&queue->retired_packets, 0);
  amdf_atomic_uint64_initialize(&queue->progress,
                                context->last_native_sequence);
  amdf_atomic_uint32_initialize(&queue->first_point,
                                AMDF_LINUX_XDNA_FIRST_POINT_UNCAPTURED);
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
    status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_CMD,
                                           device->page_size,
                                           &queue->slots[i].packet);
    if (amdf_status_is_ok(status)) {
      ++queue->packet_count;
      status = amdf_linux_xdna_buffer_attach(
          device->descriptor, device->page_size, device->page_size, NULL,
          &queue->slots[i].packet);
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

uint32_t amdf_xdna_umd_kernel_queue_maximum_command_count(
    const amdf_xdna_umd_kernel_queue_t* queue) {
  const uint32_t available =
      queue->packet_count > 2 ? queue->packet_count - 1 : 1;
  return available < AMDF_XDNA_UMD_KERNEL_QUEUE_MAXIMUM_COMMAND_COUNT
             ? available
             : AMDF_XDNA_UMD_KERNEL_QUEUE_MAXIMUM_COMMAND_COUNT;
}

bool amdf_xdna_umd_kernel_queue_has_capacity(
    const amdf_xdna_umd_kernel_queue_t* queue, uint32_t command_count) {
  const uint64_t occupied =
      queue->published_packets -
      amdf_atomic_uint64_load_acquire(&queue->retired_packets);
  const uint32_t required = command_count + (command_count > 1 ? 1 : 0);
  return queue->packet_count - occupied >= required;
}

static uint32_t amdf_linux_xdna_kernel_queue_next_packet(
    const amdf_xdna_umd_kernel_queue_t* queue, uint32_t index) {
  return index + 1 == queue->packet_count ? 0 : index + 1;
}

amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot, uint32_t command_count,
    const amdf_xdna_umd_kernel_command_t* commands,
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
  const uint32_t root_index = queue->next_packet;
  amdf_linux_xdna_buffer_t* root = &queue->slots[root_index].packet;
  uint32_t next_packet = root_index;
  if (command_count > 1) {
    // ERT_CMD_CHAIN: NEW state, six fixed payload words, then unaligned
    // uint64_t child BO handles starting at byte 28. The kernel reads the
    // children asynchronously and writes the aggregate result to this root.
    uint32_t* words = root->host_pointer;
    memset(words, 0, 7 * sizeof(*words));
    words[0] = (19u << 23) | ((6u + 2u * command_count) << 12) | 1u;
    words[1] = command_count;
    next_packet = amdf_linux_xdna_kernel_queue_next_packet(queue, next_packet);
  }
  for (uint32_t i = 0; i < command_count; ++i) {
    amdf_linux_xdna_buffer_t* packet = &queue->slots[next_packet].packet;
    amdf_xdna_elf_packet_build(commands[i].instruction_address,
                               commands[i].instruction_byte_length,
                               packet->host_pointer);
    if (command_count > 1) {
      const uint64_t handle = packet->handle;
      memcpy((uint8_t*)root->host_pointer + 28 + i * sizeof(handle), &handle,
             sizeof(handle));
    }
    next_packet = amdf_linux_xdna_kernel_queue_next_packet(queue, next_packet);
  }
  // This packet is shared with the kernel CPU, which builds the firmware
  // message and writes the result. CPU cache coherence covers both directions;
  // NPU instruction/data cache maintenance is a separate caller operation.
  status = amdf_linux_xdna_command_submit(queue->context, root,
                                          out_native_submission);
  if (amdf_status_is_ok(status)) {
    const uint32_t packet_count = command_count + (command_count > 1 ? 1 : 0);
    queue->slots[slot].root_packet = root_index;
    queue->slots[slot].packet_count = packet_count;
    queue->next_packet = next_packet;
    queue->published_packets += packet_count;
  }
  return status;
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
  const amdf_status_t status = amdf_linux_xdna_command_query_result(
      &queue->slots[queue->slots[slot].root_packet].packet);
  if (!amdf_status_is_ok(status)) {
    amdf_linux_xdna_context_record_failure(queue->context, status);
  }
  // Common code serializes result consumption. Capture the ledger before
  // publishing credits that permit another submitter to reuse packet storage.
  const uint64_t retired =
      amdf_atomic_uint64_load_acquire(&queue->retired_packets) +
      queue->slots[slot].packet_count;
  amdf_atomic_uint64_store_release(&queue->retired_packets, retired);
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
