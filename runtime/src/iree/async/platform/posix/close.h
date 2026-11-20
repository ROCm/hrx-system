// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_POSIX_CLOSE_H_
#define IREE_ASYNC_PLATFORM_POSIX_CLOSE_H_

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "iree/base/api.h"

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
    // Caller cancellation happens before issuing close. ECANCELED from the
    // syscall is a native error even when the caller accepts cancellation.
    iree_status_code_t code = error == ECANCELED
                                  ? IREE_STATUS_ABORTED
                                  : iree_status_code_from_errno(error);
    return iree_make_status(code, "close(%d) failed (errno %d): %s", fd, error,
                            strerror(error));
  }
  return iree_ok_status();
}

#endif  // IREE_ASYNC_PLATFORM_POSIX_CLOSE_H_
