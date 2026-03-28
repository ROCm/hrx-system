// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Direct HSA linkage — no dynamic symbol indirection.

#ifndef IREE_HAL_DRIVERS_HSA_STATUS_UTIL_H_
#define IREE_HAL_DRIVERS_HSA_STATUS_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "hsa_driver/hsa_headers.h"

// Converts a direct HSA call result into an iree_status_t.
//
// Usage:
//   iree_status_t status = IREE_HSA_CALL_TO_STATUS(hsa_foo(...));
#define IREE_HSA_CALL_TO_STATUS(expr, ...) \
  iree_hal_hsa_result_to_status((expr), __FILE__, __LINE__)

// Alias for consistency.
#define IREE_HSA_RESULT_TO_STATUS(result, ...) \
  iree_hal_hsa_result_to_status((result), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR with implicit hsa_status_t conversion.
#define IREE_HSA_RETURN_IF_ERROR(expr, ...)                          \
  IREE_RETURN_IF_ERROR(                                              \
      iree_hal_hsa_result_to_status((expr), __FILE__, __LINE__),     \
      __VA_ARGS__)

// IREE_RETURN_AND_END_ZONE_IF_ERROR with implicit conversion.
#define IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(zone_id, expr, ...)    \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                 \
      zone_id,                                                       \
      iree_hal_hsa_result_to_status((expr), __FILE__, __LINE__),     \
      __VA_ARGS__)

// IREE_IGNORE_ERROR with implicit conversion.
#define IREE_HSA_IGNORE_ERROR(expr) \
  IREE_IGNORE_ERROR(iree_hal_hsa_result_to_status((expr), __FILE__, __LINE__))

// Converts an hsa_status_t to an iree_status_t object.
iree_status_t iree_hal_hsa_result_to_status(hsa_status_t result,
                                            const char* file, uint32_t line);

#endif  // IREE_HAL_DRIVERS_HSA_STATUS_UTIL_H_
