// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_KMT_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_KMT_H_

#include <assert.h>
#include <sys/ioctl.h>

// Microsoft's WSL headers use MSVC warning pragmas around portable C types.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#include <basetsd.h>
#include <dxg/d3dkmthk.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "libamdf/src/platform/linux/dxg/connection.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Issues the complete published KMT operation family consumed by libamdf.
// Argument storage is borrowed only for the synchronous ioctl invocation.
// QueryAdapterInfo forwards payload bytes exactly; string-bearing queries need
// an explicit UTF-16 layout instead of the Linux wchar_t declarations.
#define AMDF_LINUX_DXG_KMT_OPERATION(name, type, operation)                \
  static inline NTSTATUS amdf_linux_dxg_kmt_##name(                        \
      const amdf_linux_dxg_connection_t* connection, type* arguments) {    \
    return (NTSTATUS)amdf_linux_dxg_connection_invoke(                     \
        (void*)connection, _IOWR('G', operation, type), (void*)arguments); \
  }
#include "libamdf/src/platform/linux/dxg/kmt_operations.inc"
#undef AMDF_LINUX_DXG_KMT_OPERATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_KMT_H_
