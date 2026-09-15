// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_FILE_H_
#define AMDF_SRC_PLATFORM_LINUX_FILE_H_

#include <errno.h>
#include <unistd.h>

#include "amdf/amdf.h"

// Preserves the kernel's native failure code.
static inline amdf_status_t amdf_linux_error(int error) {
  return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, (uint32_t)error);
}

// Linux consumes a valid descriptor even when close reports an error. Clearing
// ownership before the call prevents a later teardown attempt from closing a
// descriptor that another thread has since opened with the same number.
static inline amdf_status_t amdf_linux_file_close(int* descriptor) {
  if (*descriptor < 0) return AMDF_STATUS_OK;
  const int value = *descriptor;
  *descriptor = -1;
  return close(value) == 0 ? AMDF_STATUS_OK : amdf_linux_error(errno);
}

#endif  // AMDF_SRC_PLATFORM_LINUX_FILE_H_
