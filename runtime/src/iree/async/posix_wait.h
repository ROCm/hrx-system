// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_POSIX_WAIT_H_
#define IREE_ASYNC_POSIX_WAIT_H_

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_BSD) || \
    defined(IREE_PLATFORM_APPLE)

#include <poll.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a remaining wait to the relative timespec accepted by POSIX waits.
// Only infinity produces NULL. A time32 libc or kernel ABI can bound one
// sleep. The caller retains the original absolute deadline across such slices.
static inline const struct timespec* iree_async_posix_timeout_timespec(
    iree_duration_t duration_ns, struct timespec* storage) {
  if (duration_ns == IREE_DURATION_INFINITE) {
    return NULL;
  }
  int64_t seconds = duration_ns / 1000000000;
  if ((sizeof(time_t) == 4 || sizeof(long) == 4) && seconds > INT32_MAX) {
    storage->tv_sec = INT32_MAX;
    storage->tv_nsec = 999999999;
  } else {
    storage->tv_sec = (time_t)seconds;
    storage->tv_nsec = (long)(duration_ns % 1000000000);
  }
  return storage;
}

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
// Opens a close-on-exec kqueue, returning -1 with errno set on failure.
int iree_async_posix_kqueue(void);
#endif

#if defined(IREE_PLATFORM_APPLE)
// Darwin has no ppoll. fds[count] is a spare slot for a one-shot timer.
// A poll owner may supply its own empty kqueue to reuse across waits; -1
// opens a temporary one only if this finite wait would block. Returns poll's
// descriptor count, revents, and errno, excluding the private timer.
int iree_async_darwin_poll_until(struct pollfd* fds, nfds_t count,
                                 iree_time_t deadline_ns, int timer_fd);
#else
// Waits for descriptor readiness until a monotonic deadline, with poll's
// result, revents, and errno conventions.
int iree_async_posix_poll_until(struct pollfd* fds, nfds_t count,
                                iree_time_t deadline_ns);
#endif  // !IREE_PLATFORM_APPLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // POSIX platforms

#endif  // IREE_ASYNC_POSIX_WAIT_H_
