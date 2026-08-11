// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Win32 named-pipe peer process resolution.

#ifndef IREE_NET_CARRIER_SHM_PIPE_PEER_WIN32_H_
#define IREE_NET_CARRIER_SHM_PIPE_PEER_WIN32_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(IREE_PLATFORM_WINDOWS)

// Opens the process at the opposite end of the connected named pipe |channel|.
// The pipe endpoint is the sole authority for peer identity. On success,
// |out_peer_process| owns a process HANDLE opened with PROCESS_DUP_HANDLE.
iree_status_t iree_net_shm_win32_pipe_open_peer_process(
    iree_async_primitive_t channel, iree_async_primitive_t* out_peer_process);

#endif  // IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_PIPE_PEER_WIN32_H_
