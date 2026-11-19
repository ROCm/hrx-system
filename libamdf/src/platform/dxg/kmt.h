// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_DXG_KMT_H_
#define AMDF_SRC_PLATFORM_DXG_KMT_H_

// Native KMT ABI shared by Windows and WSL. Each binding retains its own
// transport, status conversion, and synchronous argument lifetime.
#if defined(_WIN32)
#include "libamdf/src/platform/windows/kmt_api.h"
typedef amdf_kmt_api_t amdf_dxg_kmt_t;
#define AMDF_DXG_KMT_CALL(kmt, name, arguments) \
  amdf_kmt_make_status((kmt)->name(arguments))

// A failed Windows release leaves the native owner available for retry.
static inline bool amdf_dxg_kmt_destroy_consumed(amdf_status_t status) {
  return amdf_status_is_ok(status);
}
#else
#include "libamdf/src/platform/linux/dxg/kmt.h"
typedef amdf_linux_dxg_kmt_t amdf_dxg_kmt_t;
#define AMDF_DXG_KMT_CALL(kmt, name, arguments) \
  amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_##name(kmt, arguments))

// dxgkrnl removes valid context and fence handles from the guest table before
// it sends the host destroy request. Even a failed host reply consumes them.
static inline bool amdf_dxg_kmt_destroy_consumed(amdf_status_t status) {
  (void)status;
  return true;
}
#endif

#endif  // AMDF_SRC_PLATFORM_DXG_KMT_H_
