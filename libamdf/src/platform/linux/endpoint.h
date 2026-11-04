// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_ENDPOINT_H_
#define AMDF_SRC_PLATFORM_LINUX_ENDPOINT_H_

#include "libamdf/src/platform/linux/endpoint_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opens a fresh native file after verifying the endpoint identity. This is
// deliberately not dup: DRM GEM handle tables and HWCTX ownership are per file.
// Failure leaves the output unchanged.
amdf_status_t amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t* endpoint, int* out_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_LINUX_ENDPOINT_H_
