// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/pipe_peer_win32.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <windows.h>

iree_status_t iree_net_shm_win32_pipe_open_peer_process(
    iree_async_primitive_t channel, iree_async_primitive_t* out_peer_process) {
  IREE_ASSERT_ARGUMENT(out_peer_process);
  *out_peer_process = iree_async_primitive_none();
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM channel is not a Win32 HANDLE");
  }

  HANDLE pipe_handle = (HANDLE)channel.value.win32_handle;
  DWORD pipe_flags = 0;
  if (!GetNamedPipeInfo(pipe_handle, &pipe_flags, NULL, NULL, NULL)) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to query SHM named-pipe endpoint");
  }

  ULONG peer_process_id = 0;
  BOOL query_succeeded =
      iree_any_bit_set(pipe_flags, PIPE_SERVER_END)
          ? GetNamedPipeClientProcessId(pipe_handle, &peer_process_id)
          : GetNamedPipeServerProcessId(pipe_handle, &peer_process_id);
  if (!query_succeeded) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "failed to query SHM named-pipe peer process");
  }
  if (peer_process_id == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "SHM named pipe reported process ID 0 for peer");
  }

  HANDLE peer_process = OpenProcess(PROCESS_DUP_HANDLE, FALSE, peer_process_id);
  if (!peer_process) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to open SHM named-pipe peer process %lu for handle transfer",
        (unsigned long)peer_process_id);
  }
  *out_peer_process =
      iree_async_primitive_from_win32_handle((uintptr_t)peer_process);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
