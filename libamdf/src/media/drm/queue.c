// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/queue.h"

#include <emmintrin.h>
#include <linux/sync_file.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/media/drm/command.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/wait.h"

typedef struct amdf_media_drm_dependency_t {
  // Core-owned consumer view and destruction operation.
  amdf_kernel_queue_native_dependency_t base;
  // Copied allocator for terminal metadata release.
  amdf_allocator_t host_allocator;
  // Borrowed consumer render file.
  int descriptor;
  // Independently owned imported sync-object handle in the consumer file.
  uint32_t handle;
} amdf_media_drm_dependency_t;

static amdf_status_t amdf_media_drm_sync_destroy(int descriptor,
                                                 uint32_t handle) {
  struct drm_syncobj_destroy destroy = {.handle = handle};
  return ioctl(descriptor, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy) == 0
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

static void amdf_media_drm_queue_query_status(
    amdf_kernel_queue_t* base, amdf_kernel_queue_status_t* out_status) {
  const amdf_media_drm_queue_t* queue = (const amdf_media_drm_queue_t*)base;
  out_status->retired_submission =
      amdf_atomic_uint64_load_acquire(&queue->retired);
  out_status->terminal_status =
      amdf_atomic_uint64_load_acquire(&queue->terminal_status);
  out_status->state = amdf_status_is_ok(out_status->terminal_status)
                          ? AMDF_QUEUE_STATE_ACTIVE
                          : AMDF_QUEUE_STATE_FAILED;
}

static amdf_status_t amdf_media_drm_fence_status(int descriptor,
                                                 uint32_t handle,
                                                 int32_t* out_status) {
  struct drm_syncobj_handle export = {
      .handle = handle,
      .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
      .fd = -1,
  };
  if (ioctl(descriptor, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &export) != 0) {
    return amdf_linux_error(errno);
  }
  // The binary object retains the original scheduler fence. Timeline query
  // and wait ioctls report signaling without preserving its execution error.
  struct sync_file_info info = {0};
  amdf_status_t status = ioctl(export.fd, SYNC_IOC_FILE_INFO, &info) == 0
                             ? AMDF_STATUS_OK
                             : amdf_linux_error(errno);
  int file = export.fd;
  const amdf_status_t cleanup = amdf_linux_file_close(&file);
  if (amdf_status_is_ok(status)) {
    status = cleanup;
  }
  if (amdf_status_is_ok(status)) {
    *out_status = info.status;
  }
  return status;
}

static amdf_status_t amdf_media_drm_queue_refresh_status(
    amdf_kernel_queue_t* base, amdf_kernel_queue_status_t* out_status) {
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base;
  uint32_t expected = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  if (amdf_atomic_uint32_compare_exchange_acq_rel(&queue->observing, &expected,
                                                  1)) {
    uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
    const uint64_t submitted =
        amdf_atomic_uint64_load_acquire(&queue->submitted);
    uint64_t completed = retired;
    if (retired < submitted &&
        amdf_status_is_ok(
            amdf_atomic_uint64_load_acquire(&queue->terminal_status))) {
      struct drm_syncobj_timeline_array query = {
          .handles = (uintptr_t)&queue->progress.handle,
          .points = (uintptr_t)&completed,
          .count_handles = 1,
      };
      if (ioctl(queue->progress.descriptor, DRM_IOCTL_SYNCOBJ_QUERY, &query) !=
          0) {
        status = amdf_linux_error(errno);
      }
    }
    if (completed > submitted) {
      completed = submitted;
    }
    bool stopped = false;
    while (amdf_status_is_ok(status) && !stopped && retired < completed) {
      const uint32_t handle =
          queue->original_fences[retired %
                                 base->info.maximum_pending_submission_count];
      int32_t native_status = 0;
      status = amdf_media_drm_fence_status(queue->progress.descriptor, handle,
                                           &native_status);
      if (amdf_status_is_ok(status)) {
        if (native_status > 0) {
          ++retired;
        } else {
          stopped = true;
          if (native_status < 0) {
            amdf_atomic_uint64_store_release(
                &queue->terminal_status,
                amdf_linux_error((int)(0u - (uint32_t)native_status)));
          }
        }
      }
    }
    amdf_atomic_uint64_store_release(&queue->retired, retired);
    amdf_atomic_uint32_store_release(&queue->observing, 0);
  }
  amdf_media_drm_queue_query_status(base, out_status);
  return status;
}

static amdf_status_t amdf_media_drm_queue_notify(
    amdf_kernel_queue_t* base, uint64_t submission,
    const amdf_native_event_t* event) {
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base;
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (submission <= amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_platform_native_event_signal(event);
  }
  struct drm_syncobj_eventfd notification = {
      .handle = queue->progress.handle,
      .point = submission,
      .fd = (int32_t)event->payload.file_descriptor,
  };
  return ioctl(queue->progress.descriptor, DRM_IOCTL_SYNCOBJ_EVENTFD,
               &notification) == 0
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

static amdf_status_t amdf_media_drm_queue_wait(amdf_kernel_queue_t* base,
                                               uint64_t submission,
                                               uint64_t timeout_nanoseconds,
                                               uint64_t poll_nanoseconds) {
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base;
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_wait_deadline_t deadline;
  amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_nanoseconds, &deadline);
  bool completed = false;
  while (amdf_status_is_ok(status) && !completed) {
    amdf_kernel_queue_status_t snapshot = {0};
    status = amdf_media_drm_queue_refresh_status(base, &snapshot);
    completed = snapshot.retired_submission >= submission;
    if (!amdf_status_is_ok(status) || completed) {
      break;
    }
    status = snapshot.terminal_status;
    amdf_wait_budget_t remaining;
    if (amdf_status_is_ok(status)) {
      status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    }
    if (!amdf_status_is_ok(status)) {
      break;
    }
    if (remaining.timeout == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    } else if (remaining.poll != 0) {
      _mm_pause();
    } else {
      struct drm_syncobj_timeline_wait wait = {
          .handles = (uintptr_t)&queue->progress.handle,
          .points = (uintptr_t)&submission,
          .timeout_nsec = deadline.timeout > INT64_MAX
                              ? INT64_MAX
                              : (int64_t)deadline.timeout,
          .count_handles = 1,
          .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
      };
      if (ioctl(queue->progress.descriptor, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
                &wait) != 0) {
        status = errno == ETIME
                     ? amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED)
                     : amdf_linux_error(errno);
      } else if (amdf_atomic_uint32_load_acquire(&queue->observing) != 0) {
        amdf_platform_wait_yield();
      }
    }
  }
  return status;
}

static uint64_t amdf_media_drm_queue_last_submission(
    const amdf_kernel_queue_t* base) {
  return amdf_atomic_uint64_load_acquire(
      &((const amdf_media_drm_queue_t*)base)->submitted);
}

static const amdf_kernel_queue_native_progress_t* amdf_media_drm_queue_progress(
    const amdf_kernel_queue_t* base) {
  return &((const amdf_media_drm_queue_t*)base)->progress.base;
}

static amdf_status_t amdf_media_drm_dependency_destroy(
    amdf_kernel_queue_native_dependency_t* base) {
  amdf_media_drm_dependency_t* dependency = (amdf_media_drm_dependency_t*)base;
  const amdf_status_t status =
      amdf_media_drm_sync_destroy(dependency->descriptor, dependency->handle);
  amdf_free(dependency->host_allocator, dependency);
  return status;
}

static amdf_status_t amdf_media_drm_dependency_create(
    amdf_kernel_queue_t* consumer,
    const amdf_kernel_queue_native_progress_t* base,
    amdf_kernel_queue_native_dependency_t** out_native) {
  if (base->protocol !=
      AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_DRM_SYNCOBJ_TIMELINE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_media_drm_progress_t* progress =
      (const amdf_media_drm_progress_t*)base;
  amdf_media_drm_dependency_t* dependency = NULL;
  amdf_status_t status = amdf_calloc(
      consumer->host_allocator, sizeof(*dependency),
      amdf_alignof(amdf_media_drm_dependency_t), (void**)&dependency);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  dependency->base.destroy = amdf_media_drm_dependency_destroy;
  dependency->base.wait_operation_count = 1;
  dependency->host_allocator = consumer->host_allocator;
  dependency->descriptor =
      ((amdf_media_drm_device_t*)consumer->device)->descriptor;
  struct drm_syncobj_handle transport = {.handle = progress->handle, .fd = -1};
  if (ioctl(progress->descriptor, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &transport) !=
      0) {
    status = amdf_linux_error(errno);
  }
  if (amdf_status_is_ok(status)) {
    transport.handle = 0;
    if (ioctl(dependency->descriptor, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE,
              &transport) != 0) {
      status = amdf_linux_error(errno);
    } else {
      dependency->handle = transport.handle;
    }
  }
  int descriptor = transport.fd;
  const amdf_status_t cleanup = amdf_linux_file_close(&descriptor);
  if (amdf_status_is_ok(status)) {
    status = cleanup;
  }
  if (!amdf_status_is_ok(status)) {
    if (dependency->handle != 0) {
      const amdf_status_t release = amdf_media_drm_sync_destroy(
          dependency->descriptor, dependency->handle);
      if (!amdf_status_is_ok(release)) {
        status = release;
      }
    }
    amdf_free(consumer->host_allocator, dependency);
    return status;
  }
  *out_native = &dependency->base;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_media_drm_queue_release_native(
    amdf_media_drm_queue_t* queue) {
  const int descriptor = queue->progress.descriptor;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < queue->original_fence_count; ++i) {
    const amdf_status_t release =
        amdf_media_drm_sync_destroy(descriptor, queue->original_fences[i]);
    if (amdf_status_is_ok(status)) {
      status = release;
    }
  }
  if (queue->progress.handle != 0) {
    const amdf_status_t release =
        amdf_media_drm_sync_destroy(descriptor, queue->progress.handle);
    if (amdf_status_is_ok(status)) {
      status = release;
    }
  }
  if (queue->context != 0) {
    union drm_amdgpu_ctx context = {
        .in = {.op = AMDGPU_CTX_OP_FREE_CTX, .ctx_id = queue->context}};
    if (ioctl(descriptor, DRM_IOCTL_AMDGPU_CTX, &context) != 0 &&
        amdf_status_is_ok(status)) {
      status = amdf_linux_error(errno);
    }
  }
  return status;
}

static amdf_status_t amdf_media_drm_queue_destroy(amdf_kernel_queue_t* base) {
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base;
  if (amdf_atomic_uint64_load_acquire(&queue->retired) !=
      amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  return amdf_media_drm_queue_release_native(queue);
}

static const amdf_kernel_queue_vtable_t amdf_media_drm_queue_operations = {
    .query_status = amdf_media_drm_queue_query_status,
    .refresh_status = amdf_media_drm_queue_refresh_status,
    .request_notification = amdf_media_drm_queue_notify,
    .wait = amdf_media_drm_queue_wait,
    .destroy_native = amdf_media_drm_queue_destroy,
    .query_last_submission = amdf_media_drm_queue_last_submission,
    .query_native_progress = amdf_media_drm_queue_progress,
    .create_native_dependency = amdf_media_drm_dependency_create,
};

static amdf_status_t amdf_media_drm_queue_notification_types(
    amdf_media_drm_queue_t* queue, amdf_native_event_types_t* out_types) {
  int descriptor = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  struct drm_syncobj_eventfd probe = {.handle = queue->original_fences[0],
                                      .fd = descriptor};
  const int result =
      ioctl(queue->progress.descriptor, DRM_IOCTL_SYNCOBJ_EVENTFD, &probe);
  amdf_status_t status = result == 0 || errno == ENOTTY || errno == EOPNOTSUPP
                             ? AMDF_STATUS_OK
                             : amdf_linux_error(errno);
  const amdf_status_t cleanup = amdf_linux_file_close(&descriptor);
  if (amdf_status_is_ok(status)) {
    status = cleanup;
  }
  if (amdf_status_is_ok(status)) {
    *out_types = result == 0 ? AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD : 0;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_media_drm_queue_create(
    amdf_device_t* base,
    const amdf_media_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue) {
  if ((create_info->required_capabilities &
       ~AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)base;
  const uint32_t capacity = create_info->maximum_pending_submission_count;
  const uint32_t wait_count = create_info->maximum_wait_count != 0
                                  ? create_info->maximum_wait_count
                                  : 16;
  if (wait_count >
      UINT32_MAX / (sizeof(struct drm_amdgpu_cs_chunk_syncobj) / 4)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const size_t fence_bytes =
      ((size_t)capacity * sizeof(uint32_t) + 7) & ~(size_t)7;
  amdf_media_drm_queue_t* queue = NULL;
  amdf_status_t status = amdf_calloc(
      base->host_allocator,
      sizeof(*queue) + fence_bytes + (size_t)wait_count * sizeof(*queue->waits),
      amdf_alignof(amdf_media_drm_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->progress.descriptor = device->descriptor;
  queue->progress.base.protocol =
      AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_DRM_SYNCOBJ_TIMELINE;
  // Both original and timeline signals attach the same native job fence. No
  // additional hardware submission or host signaling operation is introduced.
  queue->progress.base.signal_operation_count = 0;
  const amdf_media_drm_engine_t* engine =
      &device->engines[create_info->queue_family_ordinal];
  queue->ring = (uint32_t)__builtin_ctz(engine->information.available_rings);
  queue->ip_type = engine->ip_type;
  queue->original_fences = (uint32_t*)(queue + 1);
  queue->waits = (struct drm_amdgpu_cs_chunk_syncobj*)((uint8_t*)(queue + 1) +
                                                       fence_bytes);
  union drm_amdgpu_ctx context = {.in = {.op = AMDGPU_CTX_OP_ALLOC_CTX}};
  if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_CTX, &context) != 0) {
    status = amdf_linux_error(errno);
  } else {
    queue->context = context.out.alloc.ctx_id;
  }
  struct drm_syncobj_create timeline = {0};
  if (amdf_status_is_ok(status)) {
    if (ioctl(device->descriptor, DRM_IOCTL_SYNCOBJ_CREATE, &timeline) != 0) {
      status = amdf_linux_error(errno);
    } else {
      queue->progress.handle = timeline.handle;
    }
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < capacity; ++i) {
    struct drm_syncobj_create original = {.flags = DRM_SYNCOBJ_CREATE_SIGNALED};
    if (ioctl(device->descriptor, DRM_IOCTL_SYNCOBJ_CREATE, &original) != 0) {
      status = amdf_linux_error(errno);
    } else {
      queue->original_fences[queue->original_fence_count++] = original.handle;
    }
  }
  amdf_native_event_types_t notification_types = 0;
  if (amdf_status_is_ok(status)) {
    status =
        amdf_media_drm_queue_notification_types(queue, &notification_types);
  }
  if (amdf_status_is_ok(status)) {
    amdf_device_info_t identity = {0};
    base->vtable->query_info(base, &identity);
    const amdf_kernel_queue_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
        .structure_size = sizeof(info),
        .device_id = identity.id,
        .reset_epoch = identity.reset_epoch,
        .queue_family_ordinal = create_info->queue_family_ordinal,
        .command_type = AMDF_QUEUE_COMMAND_TYPE_MEDIA,
        .maximum_pending_submission_count = capacity,
        .maximum_command_count = 1,
        .notification_types = notification_types,
        .capabilities = create_info->required_capabilities,
        .maximum_wait_count = wait_count,
    };
    status = amdf_kernel_queue_initialize(
        &queue->base, &amdf_media_drm_queue_operations, base, &info);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup = amdf_media_drm_queue_release_native(queue);
    amdf_free(base->host_allocator, queue);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_queue = &queue->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_queue_submit(
    amdf_kernel_queue_t* base,
    const amdf_media_kernel_queue_submission_info_t* submission_info,
    uint64_t* accepted_submission) {
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base;
  amdf_media_drm_command_t* command =
      (amdf_media_drm_command_t*)submission_info->command;
  if (command->base.session->queue != base) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_drm_session_t* session =
      (amdf_media_drm_session_t*)command->base.session;
  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&queue->publishing,
                                                   &expected, 1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  const uint64_t submitted = amdf_atomic_uint64_load_acquire(&queue->submitted);
  amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&queue->terminal_status);
  if (amdf_status_is_ok(status)) {
    if (command->byte_length == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
    } else if (command->submission > retired ||
               submitted - retired >=
                   base->info.maximum_pending_submission_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    } else if (submitted == UINT64_MAX) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  if (amdf_status_is_ok(status) && queue->ip_type != AMDGPU_HW_IP_VCN_JPEG) {
    const amdf_media_command_operation_t prior = session->operation;
    const bool valid = command->operation == AMDF_MEDIA_COMMAND_OPERATION_BEGIN
                           ? prior == 0
                           : prior == AMDF_MEDIA_COMMAND_OPERATION_BEGIN;
    if (!valid) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
    }
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t submission = submitted + 1;
    for (uint32_t i = 0; i < submission_info->wait_count; ++i) {
      const amdf_kernel_queue_dependency_wait_t* wait =
          &submission_info->waits[i];
      const amdf_media_drm_dependency_t* native =
          (const amdf_media_drm_dependency_t*)wait->dependency->native;
      queue->waits[i] = (struct drm_amdgpu_cs_chunk_syncobj){
          .handle = native->handle, .point = wait->producer_submission};
    }
    const struct drm_amdgpu_cs_chunk_ib ib = {
        .va_start = command->storage->address,
        .ib_bytes = command->byte_length,
        .ip_type = queue->ip_type,
        .ring = queue->ring,
    };
    const struct drm_amdgpu_bo_list_in residency = {
        .bo_number = command->buffer_count,
        .bo_info_size = sizeof(command->buffers[0]),
        .bo_info_ptr = (uintptr_t)command->buffers,
    };
    const struct drm_amdgpu_cs_chunk_syncobj signals[2] = {
        {.handle =
             queue->original_fences
                 [submitted % base->info.maximum_pending_submission_count]},
        {.handle = queue->progress.handle, .point = submission},
    };
    const struct drm_amdgpu_cs_chunk chunks[4] = {
        {.chunk_id = AMDGPU_CHUNK_ID_IB,
         .length_dw = sizeof(ib) / 4,
         .chunk_data = (uintptr_t)&ib},
        {.chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES,
         .length_dw = sizeof(residency) / 4,
         .chunk_data = (uintptr_t)&residency},
        {.chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL,
         .length_dw = sizeof(signals) / 4,
         .chunk_data = (uintptr_t)signals},
        {.chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT,
         .length_dw = submission_info->wait_count * sizeof(queue->waits[0]) / 4,
         .chunk_data = (uintptr_t)queue->waits},
    };
    uint64_t pointers[4];
    for (uint32_t i = 0; i < 4; ++i) {
      pointers[i] = (uintptr_t)&chunks[i];
    }
    union drm_amdgpu_cs cs = {
        .in = {.ctx_id = queue->context,
               .num_chunks = submission_info->wait_count != 0 ? 4u : 3u,
               .chunks = (uintptr_t)pointers}};
    _mm_mfence();
    if (ioctl(queue->progress.descriptor, DRM_IOCTL_AMDGPU_CS, &cs) != 0) {
      status = amdf_linux_error(errno);
    } else {
      command->submission = submission;
      if (queue->ip_type != AMDGPU_HW_IP_VCN_JPEG &&
          command->operation != AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
        session->operation = command->operation;
      }
      amdf_kernel_queue_dependency_record_waits(
          submission_info->wait_count, submission_info->waits, submission);
      amdf_atomic_uint64_store_release(&queue->submitted, submission);
      *accepted_submission = submission;
    }
  }
  amdf_atomic_uint32_store_release(&queue->publishing, 0);
  return status;
}
