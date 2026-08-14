// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_ATOMIC_H_
#define IREE_HAL_REMOTE_SERVER_ATOMIC_H_

#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/commands.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;

// Validates and submits a direct queue atomic wait operation.
iree_status_t iree_hal_remote_server_queue_atomic_wait(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, iree_const_byte_span_t command_data);

// Validates and submits a direct queue atomic store operation.
iree_status_t iree_hal_remote_server_queue_atomic_store(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, iree_const_byte_span_t command_data);

// Validates and submits a direct queue atomic RMW operation.
iree_status_t iree_hal_remote_server_queue_atomic_rmw(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, iree_const_byte_span_t command_data);

// Records a structurally validated atomic wait command.
iree_status_t iree_hal_remote_server_command_buffer_atomic_wait(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_wait_cmd_t* command);

// Records a structurally validated atomic store command.
iree_status_t iree_hal_remote_server_command_buffer_atomic_store(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_store_cmd_t* command);

// Records a structurally validated atomic RMW command.
iree_status_t iree_hal_remote_server_command_buffer_atomic_rmw(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_rmw_cmd_t* command);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_ATOMIC_H_
