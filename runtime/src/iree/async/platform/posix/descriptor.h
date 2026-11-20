// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_POSIX_DESCRIPTOR_H_
#define IREE_ASYNC_PLATFORM_POSIX_DESCRIPTOR_H_

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "iree/base/api.h"

// Checks that the descriptor names a socket of the requested stream or
// datagram kind before changing its flags or accepting ownership. This lets
// close distinguish sockets from files with fallible filesystem flushes.
static inline iree_status_t iree_async_posix_prepare_socket_import(
    int fd, int expected_socket_type) {
  int socket_type = 0;
  socklen_t size = sizeof(socket_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &size) < 0) {
    int error = errno;
    return iree_make_status(error == ENOTSOCK || error == EBADF
                                ? IREE_STATUS_INVALID_ARGUMENT
                                : iree_status_code_from_errno(error),
                            "querying socket type for fd %d: %s", fd,
                            strerror(error));
  }
  if (socket_type != expected_socket_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fd %d has socket type %d, expected %d", fd,
                            socket_type, expected_socket_type);
  }
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0 ||
      (!(flags & O_NONBLOCK) && fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)) {
    int error = errno;
    return iree_make_status(iree_status_code_from_errno(error),
                            "setting fd %d nonblocking: %s", fd,
                            strerror(error));
  }
  return iree_ok_status();
}

// Transfers an owned file or socket descriptor to the native close call.
// Linux and Darwin/BSD detach the descriptor before reporting a flush error,
// including an interrupted flush. Its number can already have been reused.
static inline iree_status_t iree_async_posix_close_fd(int* owned_fd) {
  int fd = *owned_fd;
  *owned_fd = -1;
#if defined(POSIX_CLOSE_RESTART)
  // POSIX.1-2024 lets the caller request an unconditional release.
  int result = posix_close(fd, 0);
#else
  int result = close(fd);
#endif  // POSIX_CLOSE_RESTART
  if (IREE_UNLIKELY(result < 0)) {
    int error = errno;
    return iree_make_status(iree_status_code_from_errno(error),
                            "close(%d) failed (errno %d): %s", fd, error,
                            strerror(error));
  }
  return iree_ok_status();
}

#endif  // IREE_ASYNC_PLATFORM_POSIX_DESCRIPTOR_H_
