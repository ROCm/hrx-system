// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_HOST_MEMORY_H_
#define AMDF_SRC_PLATFORM_HOST_MEMORY_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates write-back system virtual memory. The validated length is a
// multiple of the platform host allocation granularity; no accelerator is
// involved.
amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer);

// Releases exactly one successful host allocation. Failure retains that range.
amdf_status_t amdf_platform_host_memory_free(void* pointer,
                                             uint64_t byte_length);

// Qualifies explicit CPU cache maintenance and returns its cache-line size.
amdf_status_t amdf_platform_host_memory_query_cache_line_size(
    uint32_t* out_line_size);

// Writes back and invalidates the validated range with full fences before and
// after the operation. The line size is the previously qualified host fact.
amdf_status_t amdf_platform_host_memory_cache_control(void* pointer,
                                                      uint64_t byte_length,
                                                      uint32_t line_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_HOST_MEMORY_H_
