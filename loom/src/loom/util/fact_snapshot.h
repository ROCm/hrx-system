// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete auxiliary value publications retained across fact-only solves.

#ifndef LOOM_UTIL_FACT_SNAPSHOT_H_
#define LOOM_UTIL_FACT_SNAPSHOT_H_

#include "loom/util/fact_origin.h"
#include "loom/util/fact_relation.h"
#include "loom/util/fact_vector_partition.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supplements the numeric fact snapshot used by cyclic recomputation. Payload
// slices and canonical roots borrow the populated table scope. No IR mutation
// may occur while this snapshot is live: full SSA replacement can modify
// borrowed partition components in place.
typedef struct loom_value_fact_auxiliary_snapshot_t {
  // Complete materialization origins, including immutable layout bindings.
  loom_value_fact_origin_snapshot_t origins;
  // Exact and regular partition views with stable component storage.
  loom_value_fact_vector_partition_t partition;
  // Canonical retained signed integer relation sets.
  loom_value_fact_relation_roots_t relations;
  // Resolved SSA representative, independent of mutable publication tokens.
  loom_value_id_t identity;
  // Canonical set of transitive select-condition dependencies.
  loom_value_set_id_t selectors;
  // Per-execution uniformity independent of conservative global distribution.
  uint8_t definition_uniform_scope;
} loom_value_fact_auxiliary_snapshot_t;

// Captures value facts not represented in loom_value_facts_t. This performs
// indexed lookups only and neither allocates nor traverses source IR.
void loom_value_fact_table_snapshot_auxiliary(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_auxiliary_snapshot_t* out_snapshot);

// Compares publications from the same scope by their complete semantics.
bool loom_value_fact_auxiliary_snapshot_equal(
    const loom_value_fact_auxiliary_snapshot_t* lhs,
    const loom_value_fact_auxiliary_snapshot_t* rhs);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_SNAPSHOT_H_
