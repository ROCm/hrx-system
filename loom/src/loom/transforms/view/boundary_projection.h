// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VIEW_BOUNDARY_PROJECTION_H_
#define LOOM_TRANSFORMS_VIEW_BOUNDARY_PROJECTION_H_

#include "loom/transforms/boundary/projection_rule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the compiler-owned retained-view transport rule.
const loom_boundary_projection_rule_t* loom_view_boundary_projection_rule(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VIEW_BOUNDARY_PROJECTION_H_
