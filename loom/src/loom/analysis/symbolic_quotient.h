// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Relational proofs for unsigned quotient values.

#ifndef LOOM_ANALYSIS_SYMBOLIC_QUOTIENT_H_
#define LOOM_ANALYSIS_SYMBOLIC_QUOTIENT_H_

#include "loom/analysis/symbolic_expr_proof.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attempts to prove a relation between |quotient_value| and |bound_value|
// using its launch extent or active path conditions.
iree_status_t loom_symbolic_expr_quotient_bound_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t quotient_value,
    loom_value_id_t bound_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_QUOTIENT_H_
