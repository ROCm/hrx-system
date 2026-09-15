// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_SNAPSHOT_H_
#define AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_SNAPSHOT_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Enumerates a snapshot and normalizes its supported AMD endpoints. Every
// returned adapter gets one close attempt. Native close failure is reported
// without publishing outputs or retaining the snapshot for later cleanup.
amdf_status_t amdf_windows_endpoint_snapshot_enumerate(
    const amdf_kmt_api_t* api, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count,
    amdf_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_SNAPSHOT_H_
