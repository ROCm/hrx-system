// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _POSIX_C_SOURCE 200809L
#include "libamdf/src/platform/wait.h"

#include <sched.h>
#include <time.h>

#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_platform_wait_query_time(uint64_t* out_nanoseconds) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return amdf_linux_error(errno);
  }
  *out_nanoseconds = (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
  return AMDF_STATUS_OK;
}

void amdf_platform_wait_yield(void) {
  // Linux sched_yield has no failure conditions.
  sched_yield();
}
