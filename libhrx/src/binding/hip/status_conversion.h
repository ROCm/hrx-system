// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_STATUS_CONVERSION_H_
#define LIBHRX_SRC_BINDING_HIP_STATUS_CONVERSION_H_

#include "binding/hip/api.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Consumes |status| and returns its HIP representation.
hipError_t iree_status_to_hip_result(iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_STATUS_CONVERSION_H_
