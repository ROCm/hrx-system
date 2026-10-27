// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projection of authored Low representations into selected target contracts.

#ifndef LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_representation_projection_plan_t
    loom_low_representation_projection_plan_t;

// Module-local symbol index over the projection plans participating in one
// source-to-Low transaction. The plan vector and index belong to the planning
// arena. Symbol IDs must retain their identity until call planning completes.
typedef struct loom_low_representation_projection_index_t {
  // Borrowed plan vector in the module owner's publication order.
  const loom_low_representation_projection_plan_t* const* plans;
  // One-based plan ordinals by symbol ID; zero means no planned projection.
  const uint16_t* ordinals;
  // Number of symbols represented by ordinals; zero for an empty index.
  iree_host_size_t symbol_count;
} loom_low_representation_projection_index_t;

// Indexes existing plans without inspecting function bodies or reprojecting
// signatures. An empty plan vector allocates no storage. Nonempty vectors use
// two bytes per module symbol, independent of function body size.
iree_status_t loom_low_representation_projection_index_build(
    const loom_module_t* module,
    const loom_low_representation_projection_plan_t* const* plans,
    iree_host_size_t plan_count, iree_arena_allocator_t* arena,
    loom_low_representation_projection_index_t* out_index);

// Returns the planned projection for a symbol, or NULL when none is present.
// A NULL index represents standalone function lowering without module plans.
const loom_low_representation_projection_plan_t*
loom_low_representation_projection_index_find(
    const loom_low_representation_projection_index_t* index,
    loom_symbol_id_t symbol_id);

// Returns the exact descriptor set selected by the projection producer.
const loom_low_descriptor_set_t* loom_low_representation_projection_descriptors(
    const loom_low_representation_projection_plan_t* plan);

// Returns the immutable target facts borrowed by the projection producer.
const loom_target_facts_t* loom_low_representation_projection_target_facts(
    const loom_low_representation_projection_plan_t* plan);

// Returns a planned argument carrier without publishing its SSA type update.
// The authored function supplies the argument arity.
loom_type_t loom_low_representation_projection_argument_type(
    const loom_low_representation_projection_plan_t* plan, uint16_t index);

// Returns a planned result carrier without publishing its SSA type update.
// The authored function supplies the result arity.
loom_type_t loom_low_representation_projection_result_type(
    const loom_low_representation_projection_plan_t* plan, uint16_t index);

// Plans register types and descriptor packets for |target_facts|' exact
// representation without changing the authored function. The retained plan
// belongs to |arena|; construction scratch is released before returning. User
// contract failures emit diagnostics and return a NULL plan. Status reports
// infrastructure failures. The function must remain unchanged until execution.
// An already exact representation retains only its signature and target facts;
// its body needs no traversal, descriptor lookup, or per-value update storage.
iree_status_t loom_low_plan_function_representation(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    const loom_low_representation_projection_plan_t** out_plan);

// Applies the complete projection, including the function's representation
// attribute and descriptor effect summaries. No compatibility or type mapping
// decisions remain; only allocation failure can interrupt publication.
iree_status_t loom_low_apply_function_representation(
    loom_module_t* module,
    const loom_low_representation_projection_plan_t* plan, bool* out_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_
