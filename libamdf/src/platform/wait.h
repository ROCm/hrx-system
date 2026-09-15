// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WAIT_H_
#define AMDF_SRC_PLATFORM_WAIT_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reads the monotonic clock used for native waits, excluding suspended time.
// Failure preserves any accepted device work and is returned to the waiter.
amdf_status_t amdf_platform_wait_query_time(uint64_t* out_nanoseconds);

// Yields to another runnable thread while contending on host retirement state.
void amdf_platform_wait_yield(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_WAIT_H_
