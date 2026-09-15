// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/reset_monitor.h"

#include <errno.h>
#include <sys/ioctl.h>

#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_gpu_kfd_reset_monitor_context_ioctl(
    void* user_data, int render_descriptor, union drm_amdgpu_ctx* context) {
  (void)user_data;
  if (ioctl(render_descriptor, DRM_IOCTL_AMDGPU_CTX, context) != 0) {
    return amdf_linux_error(errno);
  }
  return AMDF_STATUS_OK;
}

static const amdf_gpu_kfd_reset_monitor_native_api_t
    amdf_gpu_kfd_reset_monitor_native_api = {
        .context_ioctl = amdf_gpu_kfd_reset_monitor_context_ioctl,
};

const amdf_gpu_kfd_reset_monitor_native_api_t*
amdf_gpu_kfd_reset_monitor_default_native_api(void) {
  return &amdf_gpu_kfd_reset_monitor_native_api;
}

amdf_status_t amdf_gpu_kfd_reset_monitor_initialize(
    int render_descriptor,
    const amdf_gpu_kfd_reset_monitor_native_api_t* native_api,
    amdf_gpu_kfd_reset_monitor_t* out_monitor) {
  if (render_descriptor < 0 || native_api == NULL ||
      native_api->context_ioctl == NULL || out_monitor == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  union drm_amdgpu_ctx context = {
      .in =
          {
              .op = AMDGPU_CTX_OP_ALLOC_CTX,
              .priority = AMDGPU_CTX_PRIORITY_NORMAL,
          },
  };
  const amdf_status_t status = native_api->context_ioctl(
      native_api->user_data, render_descriptor, &context);
  if (!amdf_status_is_ok(status)) return status;

  amdf_gpu_kfd_reset_monitor_t monitor = {
      .native_api = native_api,
      .render_descriptor = render_descriptor,
      .context_identifier = context.out.alloc.ctx_id,
      .context_owned = true,
  };
  amdf_atomic_uint32_initialize(&monitor.reset_observed, 0);
  *out_monitor = monitor;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_reset_monitor_query(
    amdf_gpu_kfd_reset_monitor_t* monitor,
    amdf_gpu_kfd_reset_state_t* out_state) {
  if (monitor == NULL || out_state == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!monitor->context_owned) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  union drm_amdgpu_ctx context = {
      .in =
          {
              .op = AMDGPU_CTX_OP_QUERY_STATE2,
              .ctx_id = monitor->context_identifier,
          },
  };
  const amdf_status_t status = monitor->native_api->context_ioctl(
      monitor->native_api->user_data, monitor->render_descriptor, &context);
  if (!amdf_status_is_ok(status)) return status;

  if ((context.out.state.flags & AMDGPU_CTX_QUERY2_FLAGS_RESET) != 0) {
    amdf_atomic_uint32_store_release(&monitor->reset_observed, 1);
  }
  const amdf_gpu_kfd_reset_state_t state = {
      .reset_observed =
          amdf_atomic_uint32_load_acquire(&monitor->reset_observed) != 0,
      .reset_in_progress = (context.out.state.flags &
                            AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS) != 0,
  };
  *out_state = state;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_reset_monitor_deinitialize(
    amdf_gpu_kfd_reset_monitor_t* monitor) {
  if (monitor == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!monitor->context_owned) return AMDF_STATUS_OK;
  union drm_amdgpu_ctx context = {
      .in =
          {
              .op = AMDGPU_CTX_OP_FREE_CTX,
              .ctx_id = monitor->context_identifier,
          },
  };
  const amdf_status_t status = monitor->native_api->context_ioctl(
      monitor->native_api->user_data, monitor->render_descriptor, &context);
  if (amdf_status_is_ok(status)) {
    monitor->context_owned = false;
    monitor->context_identifier = 0;
  }
  return status;
}
