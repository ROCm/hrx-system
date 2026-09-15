// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/wait.h"

#include <windows.h>

amdf_status_t amdf_platform_wait_query_time(uint64_t* out_nanoseconds) {
  ULONGLONG now;
  QueryUnbiasedInterruptTimePrecise(&now);
  *out_nanoseconds = now * UINT64_C(100);
  return AMDF_STATUS_OK;
}

void amdf_platform_wait_yield(void) { SwitchToThread(); }
