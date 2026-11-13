// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low register-tuple projection across direct CFG boundaries.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_PROJECTION_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_PROJECTION_H_

#include "loom/transforms/boundary/projection_rule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the compiler-owned rule that projects partially sliced register
// tuples into unit-register components at CFG boundaries.
const loom_boundary_projection_rule_t*
loom_low_cfg_tuple_boundary_projection_rule(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_CFG_TUPLE_PROJECTION_H_
