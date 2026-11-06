// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar dialect adapters for shared combining semantics.

#ifndef LOOM_OPS_SCALAR_COMBINING_H_
#define LOOM_OPS_SCALAR_COMBINING_H_

#include "loom/ops/combining.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a combining kind to the scalar binary op that implements one pairwise
// combine step. Returns false if the kind has no scalar op representation.
bool loom_scalar_combining_kind_op(loom_combining_kind_t kind,
                                   loom_op_kind_t* out_op_kind);

// Returns the shared combining kind for a scalar floating-point extremum op,
// or COUNT_ for another operation kind.
loom_combining_kind_t loom_scalar_float_extremum_combining_kind(
    loom_op_kind_t op_kind);

// Builds the scalar operation for one verified combining step. Floating-point
// operations retain |fastmath_flags|; integer operations have exact semantics
// and ignore them. Returns the single operation result.
iree_status_t loom_scalar_combining_build(
    loom_builder_t* builder, loom_combining_kind_t kind, uint8_t fastmath_flags,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCALAR_COMBINING_H_
