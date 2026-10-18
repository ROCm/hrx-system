// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_HOST_CLOCK_H_
#define LOOM_SERVE_RUNTIME_HOST_CLOCK_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns wall-clock Unix seconds for externally visible protocol metadata.
// Clock adjustments may move this value in either direction. Scheduling,
// deadlines, and elapsed durations use the monotonic iree_time_now instead.
int64_t loom_serve_clock_now_unix_seconds(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_HOST_CLOCK_H_
