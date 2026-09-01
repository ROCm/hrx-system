// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/completion_port.h"

#if defined(IREE_PLATFORM_WINDOWS)

static VOID CALLBACK
iree_async_iocp_completion_port_fallback_apc(ULONG_PTR data) {
  (void)data;
}

void iree_async_iocp_completion_port_initialize(
    HANDLE handle, iree_async_iocp_completion_port_t* out_completion_port) {
  out_completion_port->handle = (uintptr_t)handle;
  iree_atomic_store(&out_completion_port->poll_thread_handle, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_completion_port->fallback_wake_pending, 0,
                    iree_memory_order_relaxed);
}

void iree_async_iocp_completion_port_deinitialize(
    iree_async_iocp_completion_port_t* completion_port) {
  HANDLE poll_thread_handle = (HANDLE)iree_atomic_exchange(
      &completion_port->poll_thread_handle, 0, iree_memory_order_acq_rel);
  if (poll_thread_handle != NULL) {
    if (!CloseHandle(poll_thread_handle)) iree_abort();
  }
  if (completion_port->handle != 0) {
    if (!CloseHandle((HANDLE)completion_port->handle)) iree_abort();
    completion_port->handle = 0;
  }
}

iree_status_t iree_async_iocp_completion_port_bind_poll_thread(
    iree_async_iocp_completion_port_t* completion_port) {
  if (iree_atomic_load(&completion_port->poll_thread_handle,
                       iree_memory_order_acquire) != 0) {
    return iree_ok_status();
  }

  HANDLE poll_thread_handle = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                       GetCurrentProcess(), &poll_thread_handle, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD error_code = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error_code),
                            "DuplicateHandle for IOCP poll owner failed "
                            "(error %lu)",
                            (unsigned long)error_code);
  }

  intptr_t expected_handle = 0;
  if (!iree_atomic_compare_exchange_strong(
          &completion_port->poll_thread_handle, &expected_handle,
          (intptr_t)poll_thread_handle, iree_memory_order_release,
          iree_memory_order_acquire)) {
    if (!CloseHandle(poll_thread_handle)) iree_abort();
  }
  return iree_ok_status();
}

bool iree_async_iocp_completion_port_try_post(
    iree_async_iocp_completion_port_t* completion_port, DWORD bytes_transferred,
    ULONG_PTR completion_key, LPOVERLAPPED overlapped, DWORD* out_error_code) {
  if (PostQueuedCompletionStatus((HANDLE)completion_port->handle,
                                 bytes_transferred, completion_key,
                                 overlapped)) {
    *out_error_code = ERROR_SUCCESS;
    return true;
  }
  *out_error_code = GetLastError();
  return false;
}

iree_status_t iree_async_iocp_completion_port_post(
    iree_async_iocp_completion_port_t* completion_port, DWORD bytes_transferred,
    ULONG_PTR completion_key, LPOVERLAPPED overlapped) {
  DWORD error_code = ERROR_SUCCESS;
  if (iree_async_iocp_completion_port_try_post(
          completion_port, bytes_transferred, completion_key, overlapped,
          &error_code)) {
    return iree_ok_status();
  }
  return iree_make_status(iree_status_code_from_win32_error(error_code),
                          "PostQueuedCompletionStatus failed (error %lu)",
                          (unsigned long)error_code);
}

void iree_async_iocp_completion_port_request_fallback_wake(
    iree_async_iocp_completion_port_t* completion_port) {
  if (iree_atomic_exchange(&completion_port->fallback_wake_pending, 1,
                           iree_memory_order_acq_rel) != 0) {
    return;
  }

  HANDLE poll_thread_handle = (HANDLE)iree_atomic_load(
      &completion_port->poll_thread_handle, iree_memory_order_acquire);
  if (poll_thread_handle == NULL) {
    // No poll can be blocked before its owner publishes a thread handle. Its
    // first poll consumes fallback_wake_pending before waiting.
    return;
  }

  if (QueueUserAPC(iree_async_iocp_completion_port_fallback_apc,
                   poll_thread_handle, 0) != 0) {
    return;
  }

  // A duplicated live thread handle has THREAD_SET_CONTEXT access. Failure to
  // queue its APC therefore means the poll-interrupt invariant was violated.
  // An exited poll owner needs no wake; a live owner cannot be left blocked
  // after the state it must observe has already committed.
  DWORD wait_result = WaitForSingleObject(poll_thread_handle, 0);
  if (wait_result != WAIT_OBJECT_0) {
    iree_abort();
  }
}

void iree_async_iocp_completion_port_wake(
    iree_async_iocp_completion_port_t* completion_port) {
  DWORD error_code = ERROR_SUCCESS;
  if (iree_async_iocp_completion_port_try_post(completion_port, 0, 0, NULL,
                                               &error_code)) {
    return;
  }
  iree_async_iocp_completion_port_request_fallback_wake(completion_port);
}

bool iree_async_iocp_completion_port_consume_fallback_wake(
    iree_async_iocp_completion_port_t* completion_port) {
  return iree_atomic_exchange(&completion_port->fallback_wake_pending, 0,
                              iree_memory_order_acq_rel) != 0;
}

#endif  // IREE_PLATFORM_WINDOWS
