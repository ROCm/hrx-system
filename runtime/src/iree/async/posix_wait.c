// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // ppoll
#endif

#include "iree/async/posix_wait.h"

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
#include <errno.h>
#include <fcntl.h>
#include <sys/event.h>
#include <unistd.h>

int iree_async_posix_kqueue(void) {
#if defined(IREE_PLATFORM_APPLE)
  // XNU creates every kqueue close-on-exec and close-on-fork.
  return kqueue();
#elif defined(IREE_PLATFORM_NETBSD) || defined(KQUEUE_CLOEXEC)
  return kqueue1(O_CLOEXEC);
#else
  int fd = kqueue();
  if (fd >= 0 && fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
    int error = errno;
    close(fd);
    errno = error;
    return -1;
  }
  return fd;
#endif
}
#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD

#if defined(IREE_PLATFORM_APPLE)
int iree_async_darwin_poll_until(struct pollfd* fds, nfds_t count,
                                 iree_time_t deadline_ns, int timer_fd) {
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return poll(fds, count, -1);
  }
  int result = poll(fds, count, 0);
  if (result != 0 || deadline_ns == IREE_TIME_INFINITE_PAST) {
    return result;
  }
  iree_duration_t remaining_ns =
      iree_absolute_deadline_to_timeout_ns(deadline_ns);
  if (remaining_ns == 0) {
    return 0;
  }

  bool owns_timer = timer_fd < 0;
  if (owns_timer) {
    timer_fd = iree_async_posix_kqueue();
    if (timer_fd < 0) {
      return -1;
    }
    remaining_ns = iree_absolute_deadline_to_timeout_ns(deadline_ns);
    if (remaining_ns == 0) {
      close(timer_fd);
      return poll(fds, count, 0);
    }
  }
  // The timer represents an explicit wait deadline. Ordinary EVFILT_TIMER
  // coalescing can sleep longer than even the corresponding millisecond poll.
  struct kevent64_s timer;
  EV_SET64(&timer, 1, EVFILT_TIMER, EV_ADD | EV_ONESHOT,
           NOTE_NSECONDS | NOTE_CRITICAL | NOTE_MACH_CONTINUOUS_TIME,
           remaining_ns, 0, 0, 0);
  result = kevent64(timer_fd, &timer, 1, NULL, 0, 0, NULL);
  if (result == 0) {
    fds[count] = (struct pollfd){.fd = timer_fd, .events = POLLIN};
    result = poll(fds, count + 1, -1);
    if (result > 0 && fds[count].revents) {
      --result;
    }
    if (!owns_timer) {
      // poll observes readability without consuming the timer. Remove it
      // whether I/O, expiry, or interruption ended this wait.
      int error = errno;
      timer.flags = EV_DELETE;
      if (kevent64(timer_fd, &timer, 1, NULL, 0, 0, NULL) < 0) {
        return -1;
      }
      if (result < 0) {
        errno = error;
      }
    }
  }
  if (owns_timer) {
    int error = errno;
    close(timer_fd);
    if (result < 0) {
      errno = error;
    }
  }
  return result;
}
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_BSD)

int iree_async_posix_poll_until(struct pollfd* fds, nfds_t count,
                                iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    return poll(fds, count, 0);
  }
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return poll(fds, count, -1);
  }
  struct timespec storage;
  const struct timespec* timeout = iree_async_posix_timeout_timespec(
      iree_absolute_deadline_to_timeout_ns(deadline_ns), &storage);
#if defined(IREE_PLATFORM_NETBSD)
  return pollts(fds, count, timeout, NULL);
#else
  return ppoll(fds, count, timeout, NULL);
#endif  // IREE_PLATFORM_NETBSD
}

#endif  // POSIX platforms
