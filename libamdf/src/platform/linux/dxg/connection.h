// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_CONNECTION_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_CONNECTION_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Explicit connection to the Linux DXG transport for one provider instance.
typedef struct amdf_linux_dxg_connection_t {
  // Owned /dev/dxg descriptor, or -1 after teardown consumes it.
  int descriptor;
} amdf_linux_dxg_connection_t;

// Opens one instance-owned /dev/dxg connection. Failure preserves the output.
amdf_status_t amdf_linux_dxg_connection_initialize(
    amdf_linux_dxg_connection_t* out_connection);

// Closes the connection. Linux consumes the descriptor even when close fails.
amdf_status_t amdf_linux_dxg_connection_deinitialize(
    amdf_linux_dxg_connection_t* connection);

// Issues one trusted published DXG ioctl and returns its NTSTATUS result.
// The request encodes the exact argument type and the call completes before
// this function returns. Kernel errno failures use Microsoft's libdxg mapping.
uint32_t amdf_linux_dxg_connection_invoke(void* context, uint64_t request,
                                          void* arguments);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_CONNECTION_H_
