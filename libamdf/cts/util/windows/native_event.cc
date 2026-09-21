// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/native_event.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace amdf::cts {

amdf_native_event_type_t NativeEventType() {
  return AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
}

::testing::AssertionResult CreateNativeEvent(amdf_native_event_t* out_event) {
  HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (handle == nullptr) {
    return ::testing::AssertionFailure() << "CreateEvent: " << GetLastError();
  }
  *out_event = {};
  out_event->type = NativeEventType();
  out_event->payload.native_handle = handle;
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult DestroyNativeEvent(amdf_native_event_t* event) {
  const BOOL result = CloseHandle(event->payload.native_handle);
  *event = {};
  return result ? ::testing::AssertionSuccess()
                : (::testing::AssertionFailure()
                   << "CloseHandle: " << GetLastError());
}

::testing::AssertionResult TryConsumeNativeEvent(
    const amdf_native_event_t& event, bool* out_ready) {
  const DWORD result = WaitForSingleObject(event.payload.native_handle, 0);
  if (result != WAIT_OBJECT_0 && result != WAIT_TIMEOUT) {
    return ::testing::AssertionFailure()
           << "WaitForSingleObject: " << GetLastError();
  }
  *out_ready = result == WAIT_OBJECT_0;
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult WaitNativeEvent(const amdf_native_event_t& event) {
  const DWORD result =
      WaitForSingleObject(event.payload.native_handle, INFINITE);
  return result == WAIT_OBJECT_0
             ? ::testing::AssertionSuccess()
             : (::testing::AssertionFailure()
                << "WaitForSingleObject: " << GetLastError());
}

}  // namespace amdf::cts
