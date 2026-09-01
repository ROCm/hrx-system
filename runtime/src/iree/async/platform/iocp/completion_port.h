// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_IOCP_COMPLETION_PORT_H_
#define IREE_ASYNC_PLATFORM_IOCP_COMPLETION_PORT_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#if defined(IREE_PLATFORM_WINDOWS)
// clang-format off
#include <winsock2.h>
#include <windows.h>
// clang-format on
#endif  // IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Owns an IOCP handle and the cross-thread mechanism used to wake its poll
// owner if a synthetic completion cannot be posted.
typedef struct iree_async_iocp_completion_port_t {
  // Native IOCP handle owned by this state object.
  uintptr_t handle;

  // Duplicated handle to the lifetime poll-owner thread. Published by its
  // first poll call and closed during deinitialization.
  iree_atomic_intptr_t poll_thread_handle;

  // Coalesced indication that a failed post requested an APC fallback wake.
  iree_atomic_int32_t fallback_wake_pending;
} iree_async_iocp_completion_port_t;

#if defined(IREE_PLATFORM_WINDOWS)

// Initializes |out_completion_port| to own |handle|.
void iree_async_iocp_completion_port_initialize(
    HANDLE handle, iree_async_iocp_completion_port_t* out_completion_port);

// Closes the poll-owner and completion-port handles.
void iree_async_iocp_completion_port_deinitialize(
    iree_async_iocp_completion_port_t* completion_port);

// Publishes a durable handle to the calling thread as the poll owner.
// Repeated calls from the same owner are no-ops.
iree_status_t iree_async_iocp_completion_port_bind_poll_thread(
    iree_async_iocp_completion_port_t* completion_port);

// Attempts to post one packet and captures GetLastError immediately on
// failure. Does not allocate and is safe for Windows-managed callbacks.
bool iree_async_iocp_completion_port_try_post(
    iree_async_iocp_completion_port_t* completion_port, DWORD bytes_transferred,
    ULONG_PTR completion_key, LPOVERLAPPED overlapped, DWORD* out_error_code);

// Posts one packet or returns an owned status describing the transport error.
iree_status_t iree_async_iocp_completion_port_post(
    iree_async_iocp_completion_port_t* completion_port, DWORD bytes_transferred,
    ULONG_PTR completion_key, LPOVERLAPPED overlapped);

// Requests an alertable-wait APC after state representing a failed post has
// been made visible to the poll thread. Multiple requests coalesce.
void iree_async_iocp_completion_port_request_fallback_wake(
    iree_async_iocp_completion_port_t* completion_port);

// Posts the normal wake sentinel, falling back to an APC if posting fails.
void iree_async_iocp_completion_port_wake(
    iree_async_iocp_completion_port_t* completion_port);

// Consumes a pending fallback wake request.
bool iree_async_iocp_completion_port_consume_fallback_wake(
    iree_async_iocp_completion_port_t* completion_port);

#endif  // IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_COMPLETION_PORT_H_
