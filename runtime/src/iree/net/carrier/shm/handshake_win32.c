// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Windows handshake handle exchange: passes HANDLEs across processes using
// DuplicateHandle over overlapped ReadFile/WriteFile on a named pipe.
//
// Each handshake message consists of:
//   1. The fixed-size header (same as POSIX) written as pipe data.
//   2. A Windows-specific payload: reserved bytes + raw HANDLE values.
//
// The receiver derives the sending process from the connected named pipe,
// opens it with PROCESS_DUP_HANDLE access, then calls DuplicateHandle for each
// handle to copy it into the receiver's address space. This works for any
// handle type (file mappings, Events, etc.) without requiring named objects.
//
// Named pipes opened with FILE_FLAG_OVERLAPPED require overlapped I/O for
// ReadFile/WriteFile. Each function creates a manual-reset event for the
// OVERLAPPED structure and waits on both I/O completion and cooperative
// cancellation.

#include "iree/net/carrier/shm/handshake.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <string.h>
#include <windows.h>

#include "iree/net/carrier/shm/pipe_peer_win32.h"

// Maximum number of handles sent in a single handshake message.
// OFFER sends 3 (shm_region, wake_epoch_shm, signal_primitive).
// ACCEPT sends 2 (wake_epoch_shm, signal_primitive).
#define MAX_HANDSHAKE_HANDLES 3

// Windows-specific payload appended after the standard header on the wire.
// Raw HANDLE values are meaningful only in the process identified by the
// connected named pipe.
typedef struct iree_net_shm_handshake_win32_payload_t {
  // Reserved and required to be zero.
  uint32_t reserved;
  // Number of valid entries in |handles|.
  uint32_t handle_count;
  // Raw HANDLE values in the sending process.
  uint64_t handles[MAX_HANDSHAKE_HANDLES];
} iree_net_shm_handshake_win32_payload_t;
static_assert(sizeof(iree_net_shm_handshake_win32_payload_t) == 32, "");

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Extracts the raw uint64 value from an iree_shm_handle_t.
// Returns 0 for invalid handles.
static uint64_t iree_shm_handle_to_uint64(iree_shm_handle_t handle) {
  if (!iree_shm_handle_is_valid(handle)) return 0;
  return handle.value;
}

static iree_shm_handle_t iree_shm_handle_from_uint64(uint64_t value) {
  iree_shm_handle_t handle;
  handle.value = value;
  return handle;
}

// Extracts the raw uint64 value from an iree_async_primitive_t.
// Returns 0 for NONE primitives or non-WIN32_HANDLE types.
static uint64_t iree_async_primitive_to_uint64(
    iree_async_primitive_t primitive) {
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) return 0;
  return (uint64_t)primitive.value.win32_handle;
}

//===----------------------------------------------------------------------===//
// Overlapped pipe I/O helpers
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_shm_handshake_win32_cancelled_status(void) {
  return iree_make_status(IREE_STATUS_CANCELLED, "SHM handshake cancelled");
}

static bool iree_net_shm_handshake_win32_is_peer_disconnect(DWORD error) {
  return error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED ||
         error == ERROR_NO_DATA;
}

static iree_status_t iree_net_shm_handshake_win32_peer_disconnected_status(
    const char* operation_name) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "SHM handshake peer disconnected during %s",
                          operation_name);
}

// Waits until |overlapped| completes or cancellation is requested. The
// OVERLAPPED operation is always retired before return so its stack storage is
// no longer reachable by the kernel.
static iree_status_t iree_net_shm_handshake_win32_await_overlapped(
    HANDLE channel, HANDLE event, OVERLAPPED* overlapped,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    const char* operation_name, DWORD* out_transferred) {
  HANDLE wait_handles[2] = {event, NULL};
  DWORD wait_handle_count = 1;
  if (cancellation && cancellation->interrupt_primitive.type ==
                          IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    wait_handles[1] =
        (HANDLE)cancellation->interrupt_primitive.value.win32_handle;
    wait_handle_count = 2;
  }

  DWORD wait_result = WaitForMultipleObjects(wait_handle_count, wait_handles,
                                             /*bWaitAll=*/FALSE, INFINITE);
  bool cancellation_won =
      wait_handle_count == 2 && wait_result == WAIT_OBJECT_0 + 1;
  iree_status_t status = iree_ok_status();
  if (cancellation_won) {
    if (!CancelIoEx(channel, overlapped)) {
      DWORD cancel_error = GetLastError();
      if (cancel_error != ERROR_NOT_FOUND) {
        status = iree_make_status(
            iree_status_code_from_win32_error(cancel_error),
            "CancelIoEx failed for SHM handshake %s", operation_name);
      }
    }
    // CancelIoEx only requests cancellation. Wait for the I/O event so the
    // OVERLAPPED structure is no longer owned by the kernel.
    wait_result = WaitForSingleObject(event, INFINITE);
  }

  if (wait_result != WAIT_OBJECT_0) {
    status = iree_status_join(
        status, iree_make_status(
                    iree_status_code_from_win32_error(GetLastError()),
                    "waiting for SHM handshake %s failed", operation_name));
    return status;
  }

  DWORD transferred = 0;
  if (!GetOverlappedResult(channel, overlapped, &transferred, FALSE)) {
    DWORD result_error = GetLastError();
    if (!(cancellation_won && result_error == ERROR_OPERATION_ABORTED)) {
      iree_status_t result_status =
          iree_net_shm_handshake_win32_is_peer_disconnect(result_error)
              ? iree_net_shm_handshake_win32_peer_disconnected_status(
                    operation_name)
              : iree_make_status(
                    iree_status_code_from_win32_error(result_error),
                    "SHM handshake %s failed", operation_name);
      status = iree_status_join(status, result_status);
    }
  }
  if (cancellation_won ||
      iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
    status = iree_status_join(iree_net_shm_handshake_win32_cancelled_status(),
                              status);
  }
  *out_transferred = transferred;
  return status;
}

// Writes exactly |length| bytes to the pipe using overlapped WriteFile.
// |event| is a manual-reset event for the OVERLAPPED structure (caller-owned,
// reused across calls).
static iree_status_t iree_net_shm_handshake_win32_send_all(
    HANDLE channel, HANDLE event, const void* data, DWORD length,
    const iree_net_shm_handshake_cancellation_t* cancellation) {
  const char* cursor = (const char*)data;
  DWORD remaining = length;
  while (remaining > 0) {
    if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      return iree_net_shm_handshake_win32_cancelled_status();
    }
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = event;
    ResetEvent(event);

    DWORD written = 0;
    if (!WriteFile(channel, cursor, remaining, &written, &overlapped)) {
      DWORD error = GetLastError();
      if (error == ERROR_IO_PENDING) {
        IREE_RETURN_IF_ERROR(iree_net_shm_handshake_win32_await_overlapped(
            channel, event, &overlapped, cancellation, "send", &written));
      } else if (iree_net_shm_handshake_cancellation_is_requested(
                     cancellation)) {
        return iree_net_shm_handshake_win32_cancelled_status();
      } else if (iree_net_shm_handshake_win32_is_peer_disconnect(error)) {
        return iree_net_shm_handshake_win32_peer_disconnected_status("send");
      } else {
        return iree_make_status(iree_status_code_from_win32_error(error),
                                "handshake WriteFile failed");
      }
    }
    cursor += written;
    remaining -= written;
  }
  return iree_ok_status();
}

// Reads exactly |length| bytes from the pipe using overlapped ReadFile.
// |event| is a manual-reset event for the OVERLAPPED structure (caller-owned
// and reused across calls).
static iree_status_t iree_net_shm_handshake_win32_recv_all(
    HANDLE channel, HANDLE event, void* data, DWORD length,
    const iree_net_shm_handshake_cancellation_t* cancellation) {
  char* cursor = (char*)data;
  DWORD remaining = length;
  while (remaining > 0) {
    if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      return iree_net_shm_handshake_win32_cancelled_status();
    }
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(overlapped));
    overlapped.hEvent = event;
    ResetEvent(event);

    DWORD bytes_read = 0;
    if (!ReadFile(channel, cursor, remaining, &bytes_read, &overlapped)) {
      DWORD error = GetLastError();
      if (error == ERROR_IO_PENDING) {
        IREE_RETURN_IF_ERROR(iree_net_shm_handshake_win32_await_overlapped(
            channel, event, &overlapped, cancellation, "receive", &bytes_read));
      } else if (iree_net_shm_handshake_cancellation_is_requested(
                     cancellation)) {
        return iree_net_shm_handshake_win32_cancelled_status();
      } else if (iree_net_shm_handshake_win32_is_peer_disconnect(error)) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "handshake peer disconnected during recv "
                                "(%lu of %lu bytes received)",
                                (unsigned long)(length - remaining),
                                (unsigned long)length);
      } else {
        return iree_make_status(iree_status_code_from_win32_error(error),
                                "handshake ReadFile failed");
      }
    }
    if (bytes_read == 0) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "handshake peer disconnected during recv "
                              "(%lu of %lu bytes received)",
                              (unsigned long)(length - remaining),
                              (unsigned long)length);
    }
    cursor += bytes_read;
    remaining -= bytes_read;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Send/recv with DuplicateHandle
//===----------------------------------------------------------------------===//

iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles) {
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid Windows handle");
  }
  HANDLE channel_handle = (HANDLE)channel.value.win32_handle;

  // Build the Windows-specific payload with raw handle values. The receiver
  // derives our process identity from the connected named pipe.
  iree_net_shm_handshake_win32_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.handle_count = 0;

  // Collect handles in the same order as POSIX: shm_region, epoch, signal.
  uint64_t shm_raw = iree_shm_handle_to_uint64(handles->shm_region);
  if (shm_raw != 0) {
    payload.handles[payload.handle_count++] = shm_raw;
  }
  uint64_t epoch_raw = iree_shm_handle_to_uint64(handles->wake_epoch_shm);
  if (epoch_raw != 0) {
    payload.handles[payload.handle_count++] = epoch_raw;
  }
  uint64_t signal_raw =
      iree_async_primitive_to_uint64(handles->signal_primitive);
  if (signal_raw != 0) {
    payload.handles[payload.handle_count++] = signal_raw;
  }

  // Create event for overlapped I/O.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "handshake send CreateEvent failed");
  }

  // Send header, then payload.
  iree_status_t status = iree_net_shm_handshake_win32_send_all(
      channel_handle, event, header, (DWORD)sizeof(*header), cancellation);
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_win32_send_all(
        channel_handle, event, &payload, (DWORD)sizeof(payload), cancellation);
  }

  CloseHandle(event);
  return status;
}

iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles) {
  memset(out_header, 0, sizeof(*out_header));
  *out_handles = iree_net_shm_handshake_handles_empty();

  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid Windows handle");
  }
  HANDLE channel_handle = (HANDLE)channel.value.win32_handle;

  // Create event for overlapped I/O.
  HANDLE event = CreateEventW(NULL, /*bManualReset=*/TRUE,
                              /*bInitialState=*/FALSE, NULL);
  if (!event) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "handshake recv CreateEvent failed");
  }

  // Receive the header.
  iree_status_t status = iree_net_shm_handshake_win32_recv_all(
      channel_handle, event, out_header, (DWORD)sizeof(*out_header),
      cancellation);

  // Receive the Windows-specific payload.
  iree_net_shm_handshake_win32_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_win32_recv_all(
        channel_handle, event, &payload, (DWORD)sizeof(payload), cancellation);
  }

  CloseHandle(event);

  if (!iree_status_is_ok(status)) return status;

  // Validate the payload before opening the peer process or duplicating
  // anything. READY intentionally has no handle payload.
  if (payload.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake payload reserved field must be 0");
  }
  if (payload.handle_count > MAX_HANDSHAKE_HANDLES) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "handshake payload has invalid handle count: %u",
                            payload.handle_count);
  }

  uint32_t expected_handle_count = 0;
  switch (out_header->type) {
    case IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER:
      expected_handle_count = 3;
      break;
    case IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT:
      expected_handle_count = 2;
      break;
    case IREE_NET_SHM_HANDSHAKE_MESSAGE_READY:
      expected_handle_count = 0;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown handshake message type: %u",
                              (unsigned)out_header->type);
  }
  if (payload.handle_count != expected_handle_count) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "handshake message type %u expected %u handles, "
                            "got %u",
                            (unsigned)out_header->type, expected_handle_count,
                            payload.handle_count);
  }
  if (expected_handle_count == 0) return iree_ok_status();

  iree_async_primitive_t peer_process = iree_async_primitive_none();
  IREE_RETURN_IF_ERROR(
      iree_net_shm_win32_pipe_open_peer_process(channel, &peer_process));
  HANDLE peer_process_handle = (HANDLE)peer_process.value.win32_handle;

  // Duplicate each handle from the pipe peer's process into ours.
  HANDLE local_handles[MAX_HANDSHAKE_HANDLES];
  memset(local_handles, 0, sizeof(local_handles));
  for (uint32_t i = 0; i < payload.handle_count; ++i) {
    HANDLE source = (HANDLE)(uintptr_t)payload.handles[i];
    if (!DuplicateHandle(peer_process_handle, source, GetCurrentProcess(),
                         &local_handles[i], 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      // Close any handles we already duplicated.
      for (uint32_t j = 0; j < i; ++j) {
        CloseHandle(local_handles[j]);
      }
      DWORD error = GetLastError();
      iree_async_primitive_close(&peer_process);
      return iree_make_status(
          iree_status_code_from_win32_error(error),
          "DuplicateHandle failed for SHM peer handle %u (source=0x%llx)", i,
          (unsigned long long)payload.handles[i]);
    }
  }
  iree_async_primitive_close(&peer_process);

  // Unpack handles based on message type. Counts were validated before any
  // handles were duplicated.
  if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER) {
    out_handles->shm_region =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[0]);
    out_handles->wake_epoch_shm =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[1]);
    out_handles->signal_primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)local_handles[2]);
  } else if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT) {
    out_handles->wake_epoch_shm =
        iree_shm_handle_from_uint64((uint64_t)(uintptr_t)local_handles[0]);
    out_handles->signal_primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)local_handles[1]);
  }

  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
