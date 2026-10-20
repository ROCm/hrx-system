// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_BITWISE_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_BITWISE_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decomposes signed and unsigned extraction into fixed vector shifts and a
// result-width cast. Widening precedes alignment to preserve packed extension;
// narrowing follows extraction. Target contracts own packetization.
// Statically singleton vectors use the scalar lane recipe.
iree_status_t loom_vector_legalize_bitfield_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Decomposes rotation into vector shifts and bitwise operations in the logical
// element domain. Counts wrap modulo the element width without a full-width
// shift, including a zero count; boolean rotation is an identity.
iree_status_t loom_vector_legalize_rotation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_BITWISE_LEGALIZATION_H_
