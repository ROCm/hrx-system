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
#include "libamdf/src/xdna/umd/drm/context.h"

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
  const amdf_xdna_endpoint_profile_t* profile = device->profile;
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
  if (!amdf_status_is_ok(status)) return status;
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
    // Device qualification matched the full array geometry. Successful native
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
