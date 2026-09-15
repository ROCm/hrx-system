// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_HOST_CACHE_H_
#define AMDF_SRC_PLATFORM_WINDOWS_HOST_CACHE_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the host cache line length supported by cache control operations.
uint32_t amdf_windows_host_cache_line_size(void);

// Transfers ownership of every host cache line intersecting the range.
amdf_status_t amdf_windows_host_cache_control(
    amdf_host_cache_operation_t operation, void* pointer, uint64_t byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_HOST_CACHE_H_
