// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_TRANSPORT_FACTORY_H_
#define IREE_HAL_REMOTE_CLIENT_TRANSPORT_FACTORY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;

// Creates the compiled-in transport factory selected by |driver_name|.
//
// |driver_name| must have the canonical `remote-<transport>` form used by
// remote HAL device URIs, such as `remote-tcp` or `remote-rdma`. Available
// transports are selected through //runtime/config/net:transports for Bazel or
// IREE_NET_TRANSPORT_* for CMake.
IREE_API_EXPORT iree_status_t iree_hal_remote_client_transport_factory_create(
    iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_TRANSPORT_FACTORY_H_
