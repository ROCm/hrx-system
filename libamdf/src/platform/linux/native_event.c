// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/native_event.h"

#include <sys/eventfd.h>

#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_platform_native_event_signal(
    const amdf_native_event_t* event) {
  if (eventfd_write((int)event->payload.file_descriptor, 1) == 0 ||
      errno == EAGAIN) {
    return AMDF_STATUS_OK;
  }
  return amdf_linux_error(errno);
}
