// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared proof helpers for index.cmp.
//
// These helpers centralize the comparison facts used by both fact inference
// and structural canonicalization. They intentionally prove only facts that are
// independent of backend integer width. Unsigned interval proofs require
// non-negative ranges so signed fact intervals can be reused without changing
// the ordering semantics.

#ifndef LOOM_OPS_INDEX_COMPARE_H_
#define LOOM_OPS_INDEX_COMPARE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Proves the result of comparing one SSA value to itself.
bool loom_index_cmp_same_value_result(uint8_t predicate, bool* out_result);

// Proves the comparison result from the two operand fact summaries.
bool loom_index_cmp_result_from_facts(uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts,
                                      bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_INDEX_COMPARE_H_
