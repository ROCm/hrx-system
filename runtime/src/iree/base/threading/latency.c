// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/latency.h"

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_MACOS)

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

bool iree_latency_scope_begin(iree_latency_scope_options_t options,
                              iree_latency_scope_t* out_scope) {
  IREE_ASSERT_ARGUMENT(out_scope);
  out_scope->handle = 0;
  if (options.maximum_cpu_idle_latency_us < -1) {
    return false;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  const int32_t maximum_latency_us = options.maximum_cpu_idle_latency_us < 0
                                         ? 5
                                         : options.maximum_cpu_idle_latency_us;
  int descriptor;
  do {
    descriptor = open("/dev/cpu_dma_latency", O_WRONLY | O_CLOEXEC);
  } while (descriptor < 0 && errno == EINTR);
  if (descriptor < 0) {
    IREE_TRACE_ZONE_END(z0);
    return false;
  }

  // The kernel consumes one complete s32 request. A short write cannot be
  // continued as a second write, which would be a different request.
  ssize_t length;
  do {
    length = write(descriptor, &maximum_latency_us, sizeof(maximum_latency_us));
  } while (length < 0 && errno == EINTR);
  if (length != sizeof(maximum_latency_us)) {
    close(descriptor);
    IREE_TRACE_ZONE_END(z0);
    return false;
  }
  out_scope->handle = (uintptr_t)descriptor + 1;
  IREE_TRACE_ZONE_END(z0);
  return true;
#else
  IREE_TRACE_ZONE_END(z0);
  return false;
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
}

iree_status_t iree_latency_scope_end(iree_latency_scope_t* scope) {
  if (!scope->handle) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // Linux consumes the descriptor even if close reports EINTR. Retrying could
  // close another thread's newly reused descriptor.
  close((int)(scope->handle - 1));
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
  scope->handle = 0;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_MACOS
