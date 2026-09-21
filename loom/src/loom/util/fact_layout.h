// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-local SSA materialization of explicit address-layout strides.
// Numeric layout summaries are independently transferable facts. These bindings
// name values in the current SSA scope and never travel with a fact extension.

#ifndef LOOM_UTIL_FACT_LAYOUT_H_
#define LOOM_UTIL_FACT_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_layout_origins_t
    loom_value_fact_layout_origins_t;

// Immutable per-axis stride bindings retained by the fact table.
typedef struct loom_value_fact_layout_strides_t {
  // SSA stride per axis, or INVALID for an axis described by numeric facts.
  const loom_value_id_t* values;
  // Layout rank, or zero when no scoped bindings are available.
  uint16_t count;
} loom_value_fact_layout_strides_t;

// Retains a copy of the per-axis SSA stride bindings for |value_id|. Static
// axes use LOOM_VALUE_ID_INVALID; their values live in the numeric summary.
// An empty slice clears the binding without allocating. Producers publish a
// nonempty slice only when at least one axis has an SSA stride. Recomputing a
// producer replaces its binding; unchanged bindings reuse their storage.
iree_status_t loom_value_fact_table_define_layout_strides(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_layout_strides_t strides);

// Returns the retained per-axis bindings, or an empty slice when the encoding
// has no scoped materialization. The slice borrows immutable fact-table storage
// until the populated scope is cleared. This is an indexed query, not an IR
// traversal, and does not infer bindings from numeric facts.
loom_value_fact_layout_strides_t loom_value_fact_table_query_layout_strides(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Shares the source's retained binding with a value that has the same physical
// layout in this scope. An unavailable source clears the result. This does not
// map values across calls or select among different control-flow predecessors.
iree_status_t loom_value_fact_table_forward_layout_strides(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id);

// Withdraws a binding when its producer is invalidated, without allocating.
void loom_value_fact_table_clear_layout_strides(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_LAYOUT_H_
