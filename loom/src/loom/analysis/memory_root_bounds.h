// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-root bounds proofs for view-based memory accesses.

#ifndef LOOM_ANALYSIS_MEMORY_ROOT_BOUNDS_H_
#define LOOM_ANALYSIS_MEMORY_ROOT_BOUNDS_H_

#include "loom/error/emitter.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rejects |op| when its exclusive access byte end is not proven to fit in an
// exact storage-root extent underlying |view_value_id|. Roots without exact
// extents are left unresolved. |element_end_facts| is the exclusive linear
// element end relative to the view base.
iree_status_t loom_memory_root_bounds_verify_exact_root(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_value_id_t view_value_id, int64_t static_element_byte_count,
    loom_value_facts_t element_end_facts, bool* out_failed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_MEMORY_ROOT_BOUNDS_H_
