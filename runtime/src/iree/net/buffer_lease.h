// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_BUFFER_LEASE_H_
#define IREE_NET_BUFFER_LEASE_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Allocates an unregistered host buffer and transfers it through |out_lease|.
//
// This is intended for receive-side copy paths that need message ownership but
// must not consume registered buffers reserved for preposted I/O. The lease is
// move-only: transfer it by value and clear the source. Release the final owner
// with iree_async_buffer_lease_release().
iree_status_t iree_net_buffer_lease_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_async_buffer_lease_t* out_lease);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_BUFFER_LEASE_H_
