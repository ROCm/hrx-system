// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-local transitive select-condition dependencies of SSA values.

#ifndef LOOM_UTIL_FACT_SELECT_DEPENDENCIES_H_
#define LOOM_UTIL_FACT_SELECT_DEPENDENCIES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/type_dependencies.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_select_dependency_page_t
    loom_value_fact_select_dependency_page_t;

// Returns the canonical selector-set identity, or zero when empty. Roots are
// immutable and comparable within one populated fact-table scope.
loom_value_set_id_t loom_value_fact_table_select_dependency_root(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Begins increasing-order iteration over select conditions transitively
// controlling |value_id|. Returns the canonical set identity, or zero when
// empty; identities may be compared within one fact-table scope. The cursor
// and its results remain valid until the scope is cleared.
loom_value_set_id_t loom_value_fact_table_select_dependencies_begin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_cursor_t* out_cursor);

// Publishes the transitive select-condition summary for each result of |op|.
// Operand summaries, scalar Boolean selector roles, and predicate-list value
// references on fact identities contribute to the canonical result set.
// Loop/CFG block arguments are expression leaves, so numeric feedback does not
// change these structural summaries during an immutable solve.
// Summary changes set |inout_changed| when non-NULL without clearing an
// existing change.
iree_status_t loom_value_fact_table_propagate_select_dependencies(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_op_vtable_t* vtable, bool* inout_changed);

// Clones one value's complete selector set into the target's populated scope.
// Value IDs name the same module in both tables. The target owns its canonical
// set nodes independently of the source, including for numerically undefined
// values. An empty source set clears the target summary.
iree_status_t loom_value_fact_table_clone_select_dependencies(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_SELECT_DEPENDENCIES_H_
