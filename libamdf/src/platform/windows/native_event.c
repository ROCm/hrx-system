// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/native_event.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

amdf_status_t amdf_platform_native_event_signal(
    const amdf_native_event_t* event) {
  return SetEvent(event->payload.native_handle)
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}
