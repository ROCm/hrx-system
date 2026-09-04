// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix result physical-representation planning policy.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_REPRESENTATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_REPRESENTATION_H_

#include "loom/codegen/low/lower/representation_observer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-plan observer selecting exact matrix result representations.
extern const loom_low_lower_source_plan_observer_t
    loom_amdgpu_matrix_representation_observer;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_REPRESENTATION_H_
