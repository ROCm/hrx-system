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
#if !defined(__cplusplus)
#undef C_ASSERT
#define C_ASSERT(expression) _Static_assert((expression), #expression)
#endif
#include <dxg/d3dkmthk.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "libamdf/src/platform/linux/dxg/connection.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed state required to issue typed KMT operations on one connection.
typedef struct amdf_linux_dxg_kmt_t {
  // Instance-owned connection used by every synchronous operation.
  const amdf_linux_dxg_connection_t* connection;
  // Borrowed host allocator used for temporary Windows wire-layout conversions.
  const amdf_allocator_t* host_allocator;
} amdf_linux_dxg_kmt_t;

// Issues the complete fixed-layout KMT operation family consumed by libamdf.
// Argument storage is borrowed only for the synchronous ioctl invocation.
#define AMDF_LINUX_DXG_KMT_OPERATION(name, type, operation)  \
  static inline NTSTATUS amdf_linux_dxg_kmt_##name(          \
      const amdf_linux_dxg_kmt_t* kmt, type* arguments) {    \
    return (NTSTATUS)amdf_linux_dxg_connection_invoke(       \
        (void*)kmt->connection, _IOWR('G', operation, type), \
        (void*)arguments);                                   \
  }
#include "libamdf/src/platform/linux/dxg/kmt_operations.inc"
#undef AMDF_LINUX_DXG_KMT_OPERATION

// Queries adapter information while translating Linux wchar_t declarations
// to the Windows UTF-16 wire layouts required by /dev/dxg.
NTSTATUS amdf_linux_dxg_kmt_query_adapter_info(
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_QUERYADAPTERINFO* arguments);

// Issues a bridge-supplied KMT request through the typed Linux boundary.
// String-bearing QueryAdapterInfo payloads receive their required conversion;
// every fixed-layout operation is forwarded unchanged.
uint32_t amdf_linux_dxg_kmt_invoke(void* context, uint64_t request,
                                   void* arguments);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_KMT_H_
