// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/event.h"

amdf_native_event_t iree_hal_amd_fabric_event_native_handle(
    const iree_async_event_native_t* event) {
  amdf_native_event_t handle = {0};
#if defined(IREE_ASYNC_HAVE_EVENTFD)
  handle.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
  handle.payload.file_descriptor = event->signal_primitive.value.fd;
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
  handle.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
  handle.payload.native_handle =
      (void*)event->signal_primitive.value.win32_handle;
#else
  (void)event;
#endif  // Native signal destination.
  return handle;
}
