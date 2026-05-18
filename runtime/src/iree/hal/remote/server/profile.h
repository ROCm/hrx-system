// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_PROFILE_H_
#define IREE_HAL_REMOTE_SERVER_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/remote/server/session.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handles PROFILING_BEGIN.
iree_status_t iree_hal_remote_server_handle_profiling_begin(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length);

// Handles PROFILING_FLUSH.
iree_status_t iree_hal_remote_server_handle_profiling_flush(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length);

// Handles PROFILING_END.
iree_status_t iree_hal_remote_server_handle_profiling_end(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_PROFILE_H_
