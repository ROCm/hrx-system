// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared lowering policy vocabulary for analysis request records.

#ifndef LOOM_ANALYSIS_POLICY_H_
#define LOOM_ANALYSIS_POLICY_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_lowering_policy_e {
  // Unknown or uninitialized lowering policy.
  LOOM_LOWERING_POLICY_UNKNOWN = 0,

  // Scalar or reference lowering is legal.
  LOOM_LOWERING_POLICY_REFERENCE_ALLOWED = 1,

  // Optimized lowering is preferred, but reference lowering is legal.
  LOOM_LOWERING_POLICY_VECTOR_PREFERRED = 2,

  // Any optimized lowering family may satisfy the request.
  LOOM_LOWERING_POLICY_OPTIMIZED_REQUIRED = 3,

  // A target primitive capability class must satisfy the request.
  LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED = 4,
} loom_lowering_policy_t;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_POLICY_H_
