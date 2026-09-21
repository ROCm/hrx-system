// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"

#if defined(IREE_PLATFORM_WINDOWS)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#include <windows.h>

IREE_API_EXPORT iree_status_t
iree_async_event_native_initialize(iree_async_event_native_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  HANDLE handle = CreateEventW(NULL, FALSE, FALSE, NULL);
  if (!handle) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateEvent failed");
  }
  out_event->wait_primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)handle);
  out_event->signal_primitive = out_event->wait_primitive;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_event_native_set(
    const iree_async_event_native_t* event) {
  // Win32 event: SetEvent signals the event. If it's already signaled, this
  // is a no-op (for manual-reset events) or still succeeds (for auto-reset).
  HANDLE handle = (HANDLE)event->signal_primitive.value.win32_handle;
  BOOL result = SetEvent(handle);
  IREE_ASSERT(result, "event handle released before publication");
}

IREE_API_EXPORT iree_status_t
iree_async_event_native_consume(const iree_async_event_native_t* event) {
  // The native wait consumed the auto-reset signal before dispatch. A second
  // wait or ResetEvent here could consume a newer signal awaiting rearm.
  (void)event;
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
