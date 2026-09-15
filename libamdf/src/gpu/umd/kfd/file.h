// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_FILE_H_
#define AMDF_SRC_GPU_UMD_KFD_FILE_H_

#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>

#include "libamdf/src/platform/linux/file.h"

// Opens one owned KFD file and queries its native ABI. Failure releases any
// partially acquired descriptor and leaves both outputs unchanged.
static inline amdf_status_t amdf_gpu_kfd_file_open(
    int* out_descriptor, struct kfd_ioctl_get_version_args* out_version) {
  int descriptor = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (descriptor < 0) return amdf_linux_error(errno);
  struct kfd_ioctl_get_version_args version = {0};
  amdf_status_t status = AMDF_STATUS_OK;
  if (ioctl(descriptor, AMDKFD_IOC_GET_VERSION, &version) != 0) {
    status = amdf_linux_error(errno);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    return amdf_status_is_ok(close_status) ? status : close_status;
  }
  *out_descriptor = descriptor;
  *out_version = version;
  return status;
}

#endif  // AMDF_SRC_GPU_UMD_KFD_FILE_H_
