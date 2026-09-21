// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

#include <errno.h>
#include <sys/eventfd.h>
#include <unistd.h>

IREE_API_EXPORT iree_status_t
iree_async_event_native_initialize(iree_async_event_native_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (fd < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed");
  }
  out_event->wait_primitive = iree_async_primitive_from_fd(fd);
  out_event->signal_primitive = out_event->wait_primitive;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_event_native_set(
    const iree_async_event_native_t* event) {
  // eventfd: write a 64-bit value to signal. The kernel accumulates values
  // until read. Writing 1 is idiomatic; the actual value doesn't matter for
  // our binary signal semantics.
  uint64_t value = 1;
  ssize_t result = 0;
  do {
    result = write(event->signal_primitive.value.fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  // A saturated counter is already readable. Other outcomes violate the
  // initialized resource's lifetime or native event contract.
  IREE_ASSERT(result == sizeof(value) || (result < 0 && errno == EAGAIN),
              "eventfd write failed during event publication");
}

#elif defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

IREE_API_EXPORT iree_status_t
iree_async_event_native_initialize(iree_async_event_native_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  int pipe_fds[2];
  if (pipe(pipe_fds) < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pipe creation failed for event");
  }
  iree_status_t status = iree_ok_status();
  for (int i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    int flags = fcntl(pipe_fds[i], F_GETFL);
    if (flags < 0 || fcntl(pipe_fds[i], F_SETFL, flags | O_NONBLOCK) < 0 ||
        fcntl(pipe_fds[i], F_SETFD, FD_CLOEXEC) < 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "setting native event pipe flags");
    }
  }
  if (iree_status_is_ok(status)) {
    out_event->wait_primitive = iree_async_primitive_from_fd(pipe_fds[0]);
    out_event->signal_primitive = iree_async_primitive_from_fd(pipe_fds[1]);
  } else {
    close(pipe_fds[1]);
    close(pipe_fds[0]);
  }
  return status;
}

IREE_API_EXPORT void iree_async_event_native_set(
    const iree_async_event_native_t* event) {
  // pipe: write a single byte to signal. The read end becomes readable.
  uint8_t value = 1;
  ssize_t result = 0;
  do {
    result = write(event->signal_primitive.value.fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  // A full pipe is already readable. Both ends remain owned while signaling,
  // including after a peer releases its copies.
  IREE_ASSERT(result == sizeof(value) || (result < 0 && errno == EAGAIN),
              "pipe write failed during event publication");
}

#endif  // eventfd or pipe

IREE_API_EXPORT iree_status_t
iree_async_event_native_consume(const iree_async_event_native_t* event) {
  ssize_t result = 0;
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
  // One read consumes all accumulated eventfd signals. A signal racing after
  // that read remains pending for the next callback.
  uint64_t value = 0;
  do {
    result = read(event->wait_primitive.value.fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
#else
  // Pipe writes can accumulate across multiple reads. Stop when the
  // nonblocking read reports that no more bytes are available.
  uint8_t values[256];
  do {
    result = read(event->wait_primitive.value.fd, values, sizeof(values));
  } while (result > 0 || (result < 0 && errno == EINTR));
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
  if (result < 0 && errno != EAGAIN) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "event signal read failed");
  }
  if (result == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "event signaling endpoint closed");
  }
  return iree_ok_status();
}

#endif  // POSIX platforms
