// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_DYNAMIC_LOGGING_H_
#define LIBHRX_SRC_BINDING_HIP_DYNAMIC_LOGGING_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Indicates whether API-category messages pass the active level and mask
// filters. Kept visible to this private header so disabled logging requires
// only an inline relaxed atomic load.
extern iree_atomic_int32_t hrx_hip_dynamic_logging_api_enabled;

// Writes one API-category message when dynamic logging is enabled. Callers
// should use HRX_HIP_DYNAMIC_LOG so format arguments are not evaluated while
// logging is disabled.
IREE_PRINTF_ATTRIBUTE(1, 2)
void hrx_hip_dynamic_logging_write(const char* format, ...);

#define HRX_HIP_DYNAMIC_LOG(format, ...)                                     \
  do {                                                                       \
    if (IREE_UNLIKELY(iree_atomic_load(&hrx_hip_dynamic_logging_api_enabled, \
                                       iree_memory_order_relaxed) != 0))     \
      hrx_hip_dynamic_logging_write(format, ##__VA_ARGS__);                  \
  } while (0)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_DYNAMIC_LOGGING_H_
