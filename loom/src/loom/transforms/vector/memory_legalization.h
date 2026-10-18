// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_MEMORY_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_MEMORY_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Preserves whole vector accesses under uniform-element masks before falling
// back to per-lane reference lowering. Native target contracts take precedence.
iree_status_t loom_vector_legalize_non_dense_memory(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Applies the same uniform-mask guard to atomic updates, then delegates their
// per-lane ordering and update semantics to the shared atomic legalizer.
iree_status_t loom_vector_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_MEMORY_LEGALIZATION_H_
