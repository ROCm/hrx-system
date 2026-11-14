// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/perfmon.h"

#if defined(IREE_PLATFORM_LINUX)

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

iree_status_t iree_hal_amd_gpu_perfmon_open(
    const amdf_endpoint_info_t* endpoint_info,
    iree_hal_amd_gpu_perfmon_t* out_perfmon) {
  *out_perfmon = (iree_hal_amd_gpu_perfmon_t){.descriptor = -1};
  if (endpoint_info->native_identity.type !=
      AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU endpoint has no Linux render node identity");
  }
  // The character-device directory names every node by its device number, so
  // the endpoint's own node is found without guessing a render minor.
  char path[64];
  snprintf(path, sizeof(path), "/dev/char/%u:%u",
           endpoint_info->native_identity.value.linux_device.major,
           endpoint_info->native_identity.value.linux_device.minor);
  const int descriptor = open(path, O_RDWR | O_CLOEXEC);
  if (descriptor < 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "cannot open GPU render node %s: %s", path,
                            strerror(errno));
  }
  out_perfmon->descriptor = descriptor;
  struct drm_amdgpu_info_device device = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&device,
      .return_size = sizeof(device),
      .query = AMDGPU_INFO_DEV_INFO,
  };
  if (ioctl(descriptor, DRM_IOCTL_AMDGPU_INFO, &query) != 0 ||
      !device.num_shader_engines) {
    const int error = errno;
    iree_hal_amd_gpu_perfmon_close(out_perfmon);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "cannot query GPU device geometry on %s: %s", path,
                            strerror(error));
  }
  out_perfmon->shader_engine_count = device.num_shader_engines;
  union drm_amdgpu_ctx context = {
      .in = {.op = AMDGPU_CTX_OP_ALLOC_CTX},
  };
  if (ioctl(descriptor, DRM_IOCTL_AMDGPU_CTX, &context) == 0) {
    out_perfmon->context_id = context.out.alloc.ctx_id;
    out_perfmon->has_context = true;
    union drm_amdgpu_ctx request = {
        .in = {.op = AMDGPU_CTX_OP_SET_STABLE_PSTATE,
               .ctx_id = out_perfmon->context_id,
               .flags = AMDGPU_CTX_STABLE_PSTATE_STANDARD},
    };
    // Another context may hold a different stable state; the kernel then
    // refuses this one and the session runs under the device's own policy.
    out_perfmon->stable_power_state =
        ioctl(descriptor, DRM_IOCTL_AMDGPU_CTX, &request) == 0;
  }
  return iree_ok_status();
}

void iree_hal_amd_gpu_perfmon_close(iree_hal_amd_gpu_perfmon_t* perfmon) {
  if (perfmon->descriptor < 0) {
    return;
  }
  if (perfmon->has_context) {
    union drm_amdgpu_ctx context = {
        .in = {.op = AMDGPU_CTX_OP_FREE_CTX, .ctx_id = perfmon->context_id},
    };
    ioctl(perfmon->descriptor, DRM_IOCTL_AMDGPU_CTX, &context);
  }
  close(perfmon->descriptor);
  *perfmon = (iree_hal_amd_gpu_perfmon_t){.descriptor = -1};
}

#else

iree_status_t iree_hal_amd_gpu_perfmon_open(
    const amdf_endpoint_info_t* endpoint_info,
    iree_hal_amd_gpu_perfmon_t* out_perfmon) {
  (void)endpoint_info;
  *out_perfmon = (iree_hal_amd_gpu_perfmon_t){.descriptor = -1};
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "GPU perfmon prerequisites require Linux DRM");
}

void iree_hal_amd_gpu_perfmon_close(iree_hal_amd_gpu_perfmon_t* perfmon) {
  (void)perfmon;
}

#endif  // IREE_PLATFORM_LINUX
