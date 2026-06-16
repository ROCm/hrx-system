// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shim_debug.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

static std::recursive_mutex s_debug_mutex;

struct debug_lock {
  std::lock_guard<std::recursive_mutex> m_lk;
  debug_lock();
};

debug_lock::debug_lock() : m_lk(s_debug_mutex) {}

void debugf(const char* format, ...) {
  debug_lock lk;
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  fflush(stdout);
}
