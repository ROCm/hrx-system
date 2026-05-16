// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_H_

#include "iree/hal/remote/client/device.h"
#include "iree/net/channel/queue/queue_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opens the bulk endpoint after the queue channel has been created.
//
// On success this consumes |queue_channel| and publishes it only after the bulk
// channel is also ready. On failure the caller still owns |queue_channel|.
iree_status_t iree_hal_remote_client_device_open_bulk_endpoint(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* queue_channel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_H_
