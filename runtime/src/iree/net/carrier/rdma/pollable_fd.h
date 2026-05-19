// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pollable Linux file descriptor helpers for rdma-core objects.
//
// rdma_cm event channels and ibverbs completion channels expose kernel file
// descriptors that must be integrated with the IREE proactor. This component
// owns the shared descriptor contract: nonblocking operation, close-on-exec,
// and RDMA-specific errno translation.

#ifndef IREE_NET_CARRIER_RDMA_POLLABLE_FD_H_
#define IREE_NET_CARRIER_RDMA_POLLABLE_FD_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maps an errno value from rdma-core descriptor setup/drain operations.
//
// ENODEV is treated as UNAVAILABLE instead of NOT_FOUND because it usually
// means the kernel/provider side needed for RDMA event delivery is absent.
IREE_API_EXPORT iree_status_code_t
iree_net_rdma_pollable_fd_status_code_from_errno(int error);

// Returns an IREE status for |error| with source location and syscall context.
IREE_API_EXPORT iree_status_t iree_net_rdma_pollable_fd_status_from_errno(
    const char* file, uint32_t line, int error, const char* call);

// Sets O_NONBLOCK and FD_CLOEXEC on |file_descriptor|.
IREE_API_EXPORT iree_status_t
iree_net_rdma_pollable_fd_initialize(int file_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_POLLABLE_FD_H_
