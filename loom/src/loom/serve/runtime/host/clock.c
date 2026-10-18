// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/host/clock.h"

#include <time.h>

int64_t loom_serve_clock_now_unix_seconds(void) {
  // POSIX and the Windows CRT define time() in Unix seconds. On 64-bit
  // Windows it wraps _time64; FILETIME epoch conversion is not involved.
  return (int64_t)time(NULL);
}
