// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/pollable_fd.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>

IREE_API_EXPORT iree_status_code_t
iree_net_rdma_pollable_fd_status_code_from_errno(int error) {
  if (error == ENODEV) return IREE_STATUS_UNAVAILABLE;
  return iree_status_code_from_errno(error);
}

IREE_API_EXPORT iree_status_t iree_net_rdma_pollable_fd_status_from_errno(
    const char* file, uint32_t line, int error, const char* call) {
  if (error == 0) return iree_ok_status();
  return iree_make_status_with_location(
      file, line, iree_net_rdma_pollable_fd_status_code_from_errno(error),
      "[%s] errno %d: %s", call, error, strerror(error));
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_pollable_fd_initialize(int file_descriptor) {
  int flags = fcntl(file_descriptor, F_GETFL);
  if (flags == -1) {
    return iree_net_rdma_pollable_fd_status_from_errno(__FILE__, __LINE__,
                                                       errno, "fcntl(F_GETFL)");
  }
  if (fcntl(file_descriptor, F_SETFL, flags | O_NONBLOCK) == -1) {
    return iree_net_rdma_pollable_fd_status_from_errno(
        __FILE__, __LINE__, errno, "fcntl(F_SETFL, O_NONBLOCK)");
  }

  int descriptor_flags = fcntl(file_descriptor, F_GETFD);
  if (descriptor_flags == -1) {
    return iree_net_rdma_pollable_fd_status_from_errno(__FILE__, __LINE__,
                                                       errno, "fcntl(F_GETFD)");
  }
  if (fcntl(file_descriptor, F_SETFD, descriptor_flags | FD_CLOEXEC) == -1) {
    return iree_net_rdma_pollable_fd_status_from_errno(
        __FILE__, __LINE__, errno, "fcntl(F_SETFD, FD_CLOEXEC)");
  }

  return iree_ok_status();
}
