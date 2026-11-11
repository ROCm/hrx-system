// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Numeric inference under retained CFG conditions in an immutable scope.

#ifndef LOOM_ANALYSIS_CONDITIONED_VALUE_FACTS_H_
#define LOOM_ANALYSIS_CONDITIONED_VALUE_FACTS_H_

#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Refines a fully computed function scope under dominating CFG edge conditions.
// Retained dominator trees establish each guard's scope; joins retain only
// dominating conditions. This does not compute cross-edge relational closure.
// Constraints apply to operand scratch and predicate references during the
// existing numeric solve. Incoming values retain their global facts; local
// result relations can name a counterpart's scoped numeric representative.
// The module's local-value scratch is leased only for this call. The resulting
// table requires whole-scope invalidation after IR edits and cannot be attached
// to an incremental rewriter. On failure, the caller invalidates the scope.
iree_status_t loom_conditioned_value_facts_compute(
    loom_value_fact_table_t* table, loom_module_t* module,
    loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITIONED_VALUE_FACTS_H_
