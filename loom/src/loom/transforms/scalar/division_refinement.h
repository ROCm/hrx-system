// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SCALAR_DIVISION_REFINEMENT_H_
#define LOOM_TRANSFORMS_SCALAR_DIVISION_REFINEMENT_H_

#include "loom/ops/builder.h"

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

#ifdef __cplusplus
extern "C" {
#endif

// Applies an all-zero or all-one sign mask using wrapping arithmetic.
iree_status_t loom_scalar_division_apply_sign(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t sign,
    loom_type_t type, loom_location_id_t location, loom_value_id_t* out_value);

// Decomposes a signed scalar into its sign mask and unsigned magnitude. The
// minimum signed value retains its magnitude as the carrier's high bit.
iree_status_t loom_scalar_division_build_magnitude(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_sign,
    loom_value_id_t* out_magnitude);

// Refines the provided results of defined unsigned division. Each result
// pointer is optional. A signed-magnitude division additionally establishes
// that its remainder is representable as a nonnegative signed value.
iree_status_t loom_scalar_division_refine_unsigned_results(
    loom_builder_t* builder, loom_value_id_t dividend, loom_value_id_t divisor,
    loom_type_t type, bool signed_magnitudes, loom_location_id_t location,
    loom_value_id_t* quotient, loom_value_id_t* remainder);

// Retains numeric refinements and the runtime divisor bound on a reconstructed
// remainder. The optional fact table supplies facts established before the
// source operation is erased; semantic postconditions do not depend on it.
iree_status_t loom_scalar_division_refine_reconstructed_remainder(
    loom_builder_t* builder, const loom_value_fact_table_t* fact_table,
    loom_value_id_t original_result, loom_value_id_t reconstructed_result,
    loom_value_id_t divisor, loom_type_t result_type, bool is_signed,
    loom_location_id_t location, loom_value_id_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCALAR_DIVISION_REFINEMENT_H_
