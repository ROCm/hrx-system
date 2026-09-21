// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"

#if !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID) && \
    !defined(IREE_PLATFORM_APPLE) && !defined(IREE_PLATFORM_BSD) &&     \
    !defined(IREE_PLATFORM_WINDOWS)

IREE_API_EXPORT iree_status_t
iree_async_event_native_initialize(iree_async_event_native_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "native events not supported on this platform");
}

IREE_API_EXPORT void iree_async_event_native_set(
    const iree_async_event_native_t* event) {
  (void)event;
  IREE_ASSERT_UNREACHABLE("native event initialization is unavailable");
}

IREE_API_EXPORT iree_status_t
iree_async_event_native_consume(const iree_async_event_native_t* event) {
  (void)event;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not supported on this platform");
}

#endif  // Unsupported platforms
