// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BOUNDARY_PROJECTION_APPLY_H_
#define LOOM_TRANSFORMS_BOUNDARY_PROJECTION_APPLY_H_

#include "loom/transforms/boundary/projection_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Applies one fully preflighted module-wide projection plan atomically.
iree_status_t loom_boundary_projection_apply(
    loom_boundary_projection_plan_t* plan);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PROJECTION_APPLY_H_
