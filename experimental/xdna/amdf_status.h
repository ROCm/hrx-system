// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conversion from portable libamdf status values to IREE statuses.

#ifndef IREE_EXPERIMENTAL_XDNA_AMDF_STATUS_H_
#define IREE_EXPERIMENTAL_XDNA_AMDF_STATUS_H_

#include "amdf/amdf.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts |amdf_status| to an IREE status attributed to |operation|.
//
// Success is returned without allocating. Failures preserve the complete
// libamdf domain and code in the diagnostic message while mapping portable API
// codes onto their closest IREE status code.
iree_status_t iree_hal_amd_status_from_amdf_status(const char* file,
                                                   uint32_t line,
                                                   amdf_status_t amdf_status,
                                                   const char* operation);

#define IREE_HAL_AMD_STATUS_FROM_AMDF(amdf_status, operation)             \
  iree_hal_amd_status_from_amdf_status(__FILE__, __LINE__, (amdf_status), \
                                       (operation))

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_XDNA_AMDF_STATUS_H_
