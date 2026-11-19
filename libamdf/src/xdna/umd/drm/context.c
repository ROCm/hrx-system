// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/context.h"

#include <drm/amdxdna_accel.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"
#include "libamdf/src/xdna/elf_packet.h"
#include "libamdf/src/xdna/umd/drm/context.h"

static amdf_status_t amdf_linux_xdna_context_release_admission(
    amdf_xdna_umd_context_t* context) {
  const int descriptor = context->device->descriptor;
  const amdf_status_t packet_status = amdf_linux_xdna_buffer_deinitialize(
      descriptor, &context->admission.packet);
  const amdf_status_t instruction_status = amdf_linux_xdna_buffer_deinitialize(
      descriptor, &context->admission.instructions);
  return amdf_status_is_ok(packet_status) ? instruction_status : packet_status;
}

// This complete, known-valid request tests the required transport itself.
// EINVAL here means that the native provider cannot admit this contract; it
// does not change how arbitrary caller submissions report their errors.
static amdf_status_t amdf_linux_xdna_context_submit_admission(
    amdf_xdna_umd_context_t* context, uint64_t* out_sequence) {
  struct amdxdna_drm_exec_cmd submit = {
      .hwctx = context->handle,
      .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
      .cmd_handles = context->admission.packet.handle,
      .cmd_count = 1,
  };
  while (ioctl(context->device->descriptor, DRM_IOCTL_AMDXDNA_EXEC_CMD,
               &submit) != 0) {
    const int error = errno;
    if (error == EINVAL || error == ENOTTY || error == EOPNOTSUPP) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    } else if (error != EINTR) {
      return amdf_linux_error(error);
    }
  }
  *out_sequence = submit.seq;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_linux_xdna_context_wait_admission(
    amdf_xdna_umd_context_t* context, uint64_t sequence) {
  struct drm_syncobj_timeline_wait wait = {
      .handles = (uintptr_t)&context->completion_syncobj,
      .points = (uintptr_t)&sequence,
      .timeout_nsec = INT64_MAX,
      .count_handles = 1,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
  };
  // The exclusive queue lease prevents another submission from replacing the
  // first fence. Point zero is therefore safe without a binary snapshot.
  while (ioctl(context->device->descriptor, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
               &wait) != 0) {
    const int error = errno;
    if (error != EINTR) {
      return amdf_linux_error(error);
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_context_prepare_execution(
    amdf_xdna_umd_context_t* context) {
  amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&context->terminal_status);
  if (!amdf_status_is_ok(status) || context->last_native_sequence != 0) {
    return status;
  }
  amdf_xdna_umd_device_t* device = context->device;
  const amdf_xdna_device_profile_t* profile = device->profile;
  const size_t alignment = profile->info->instruction.address_alignment;
  status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_DEV,
                                         alignment,
                                         &context->admission.instructions);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_attach(device->descriptor, alignment,
                                           device->page_size, &device->heap,
                                           &context->admission.instructions);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_create(device->descriptor, AMDXDNA_BO_CMD,
                                           device->page_size,
                                           &context->admission.packet);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_attach(device->descriptor,
                                           device->page_size, device->page_size,
                                           NULL, &context->admission.packet);
  }
  if (amdf_status_is_ok(status)) {
    void* bytes = context->admission.instructions.host_pointer;
    amdf_xdna_elf_admission_write(profile->transaction.device_generation,
                                  (uint8_t)profile->info->array.row_count,
                                  (uint8_t)profile->rows.memory_count, 0, 0,
                                  bytes);
    amdf_linux_host_cache_transfer(bytes, AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH,
                                   device->cache_line_size);
    amdf_xdna_elf_packet_build(context->admission.instructions.device_address,
                               AMDF_XDNA_ELF_ADMISSION_BYTE_LENGTH,
                               context->admission.packet.host_pointer);
  }
  bool retired = true;
  uint64_t sequence = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_context_submit_admission(context, &sequence);
    if (amdf_status_is_ok(status)) {
      retired = false;
      context->last_native_sequence = sequence + 1;
      status = amdf_linux_xdna_context_wait_admission(context, sequence);
      retired = amdf_status_is_ok(status);
    }
  }
  if (amdf_status_is_ok(status)) {
    const uint32_t state =
        __atomic_load_n((const uint32_t*)context->admission.packet.host_pointer,
                        __ATOMIC_ACQUIRE) &
        0xf;
    status = state == 4 ? AMDF_STATUS_OK
                        : amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, state);
    if (state == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  if (retired) {
    const amdf_status_t release_status =
        amdf_linux_xdna_context_release_admission(context);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  if (!amdf_status_is_ok(status)) {
    amdf_atomic_uint64_store_release(&context->terminal_status, status);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_context_destroy(amdf_xdna_umd_context_t* context) {
  // Caller-owned private memory and queues have already been released.
  if (context->handle != AMDXDNA_INVALID_CTX_HANDLE) {
    struct amdxdna_drm_destroy_hwctx destroy = {.handle = context->handle};
    if (ioctl(context->device->descriptor, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX,
              &destroy) != 0) {
      return amdf_linux_error(errno);
    }
    context->handle = AMDXDNA_INVALID_CTX_HANDLE;
  }
  // Successful native destruction has drained or canceled accepted admission
  // work, including when its original completion could not be observed.
  const amdf_status_t admission_status =
      amdf_linux_xdna_context_release_admission(context);
  if (!amdf_status_is_ok(admission_status)) {
    return admission_status;
  }
  if (context->completion_syncobj != 0) {
    struct drm_syncobj_destroy destroy = {
        .handle = context->completion_syncobj,
    };
    if (ioctl(context->device->descriptor, DRM_IOCTL_SYNCOBJ_DESTROY,
              &destroy) != 0) {
      return amdf_linux_error(errno);
    }
    context->completion_syncobj = 0;
  }
  amdf_free(context->device->host_allocator, context);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_context_create(
    amdf_xdna_umd_device_t* device,
    const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_umd_context_t** out_context,
    amdf_xdna_umd_context_result_t* out_result) {
  const amdf_xdna_device_profile_t* profile = device->profile;
  if ((create_info->acceptable_scheduling_modes &
       AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED) == 0 ||
      (create_info->physical_column_origin !=
           AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY &&
       create_info->physical_column_origin !=
           profile->info->array.column_origin)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_xdna_umd_context_t* context = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*context),
                  amdf_alignof(amdf_xdna_umd_context_t), (void**)&context);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  context->device = device;
  context->handle = AMDXDNA_INVALID_CTX_HANDLE;

  struct amdxdna_qos_info qos = {.priority = AMDXDNA_QOS_NORMAL_PRIORITY};
  // Full-array admission establishes fixed placement independently of the
  // opaque instructions callers subsequently submit.
  struct amdxdna_drm_create_hwctx create = {
      .qos_p = (uintptr_t)&qos,
      .num_tiles = profile->info->array.column_count * profile->rows.core_count,
  };
  if (ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &create) != 0) {
    status = amdf_linux_error(errno);
  } else {
    context->handle = create.handle;
    context->completion_syncobj = create.syncobj_handle;
    if (context->handle == AMDXDNA_INVALID_CTX_HANDLE ||
        context->completion_syncobj == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  if (amdf_status_is_ok(status)) {
    amdf_xdna_umd_context_result_t result = {0};
    result.id.words[0] = (uintptr_t)context;
    result.id.words[1] =
        ((uint64_t)context->completion_syncobj << 32) | context->handle;
    result.scheduling_mode = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    // Device activation queried the full array geometry. Successful native
    // admission of every compute tile leaves one candidate partition: the full
    // array at its origin. The kernel retains it across context restarts, but
    // may time-share it with other contexts.
    result.physical_column_origin = profile->info->array.column_origin;
    result.physical_column_count = profile->info->array.column_count;
    *out_result = result;
    *out_context = context;
  } else {
    const amdf_status_t release_status = amdf_xdna_umd_context_destroy(context);
    if (!amdf_status_is_ok(release_status)) {
      // No command or private memory exists before publication. Native handle
      // cleanup failure does not require retaining this host bookkeeping.
      amdf_free(device->host_allocator, context);
      status = release_status;
    }
  }
  return status;
}
