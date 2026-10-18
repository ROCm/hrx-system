// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-call argument facts retained for source-to-low planning.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_invocation_facts_t {
  // Conservative argument facts indexed by private callee argument value ID.
  loom_value_fact_table_t arguments;
} loom_low_source_invocation_facts_t;

// Computes private, acyclic source-call argument facts before source lowering.
//
// Every semantic caller participates in each callee argument meet. Exported
// and recursive callees retain their declared argument domains. Selected
// functions use their exact target facts while computing caller-local facts;
// unselected callers use target-independent inference.
iree_status_t loom_low_source_invocation_facts_compute(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_source_selection_list_t* selections,
    iree_arena_allocator_t* arena,
    loom_low_source_invocation_facts_t* out_facts);

// Returns the argument-fact inputs for one source function solve.
loom_value_fact_table_view_t loom_low_source_invocation_facts_for_function(
    const loom_low_source_invocation_facts_t* facts, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_
