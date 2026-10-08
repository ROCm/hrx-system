// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable analysis identity for one target-low function snapshot.

#ifndef LOOM_CODEGEN_LOW_FUNCTION_MODEL_H_
#define LOOM_CODEGEN_LOW_FUNCTION_MODEL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness_dataflow.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function_requirements.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/cfg_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_function_model_flag_bits_e {
  // Includes values defined in recursively nested function regions.
  LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE = 1u << 0,
};
typedef uint16_t loom_low_function_model_flags_t;

// Preparation state before constructing immutable control-flow analyses.
// Owns the local ordinal domain and the required physical-storage identities.
// Preparation rewrites preserve existing ordinals and update the identities
// when introducing independent values. A model consumes this ownership.
typedef struct loom_low_function_context_t {
  // Module containing the prepared function.
  loom_module_t* module;
  // Target-low function definition being prepared.
  const loom_op_t* function_op;
  // Executable function body covered by the local domain.
  loom_region_t* body;
  // Resolved target shared by preparation and snapshot consumers.
  loom_low_resolved_target_t target;
  // Stable function-local ordinals owned until transferred to a model.
  loom_local_value_domain_t value_domain;
  // Canonical required physical-storage owner for each local ordinal, or NULL
  // when every value is independent. Reserved tail entries initially own self.
  loom_value_ordinal_t* storage_origins;
  // Declared interfaces, storage and counts maintained during preparation.
  loom_low_function_requirements_t requirements;
  // Number of user-facing errors emitted while resolving the context.
  uint32_t error_count;
} loom_low_function_context_t;

// Shared immutable facts for one target-low function IR snapshot.
//
// The model owns the module value-ordinal scratch map until deinitialized.
// Callers must keep the function IR semantically immutable while the model is
// live and must deinitialize it before rewriting the function.
typedef struct loom_low_function_model_t {
  // Prepared target, stable value ordinals, and physical-storage identities.
  loom_low_function_context_t context;
  // Read-only control-flow graph for the function body.
  loom_cfg_graph_t cfg_graph;
  // Canonical block liveness shared by source and scheduled-order consumers.
  loom_liveness_dataflow_t liveness_dataflow;
  // Canonical loop intervals preserved from |cfg_graph|.
  loom_cfg_loop_forest_t loop_forest;
} loom_low_function_model_t;

// Resolves one function and its required storage identities before immutable
// analysis. The identity array reserves |additional_value_capacity| entries
// for preparation-created independent values without abandoning arena storage.
// The caller registers each created value in |out_context->value_domain| and
// maintains the identity result before handing the context to a model.
// When non-NULL, |resolved_target| supplies the producer-owned binding without
// repeating resolution; its borrowed contents must outlive the context/model.
iree_status_t loom_low_function_context_initialize(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_resolved_target_t* resolved_target,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, loom_low_function_model_flags_t flags,
    iree_host_size_t additional_value_capacity, iree_arena_allocator_t* arena,
    loom_low_function_context_t* out_context);

// Releases preparation ownership that has not been consumed by a model.
void loom_low_function_context_deinitialize(
    loom_low_function_context_t* context);

// Consumes |context| and builds immutable analyses over its current IR. The
// context is cleared even when analysis fails; deinitialize |out_model| to
// release its acquired domain on either outcome.
iree_status_t loom_low_function_model_build(
    loom_low_function_context_t* context, iree_arena_allocator_t* arena,
    loom_low_function_model_t* out_model);

// Initializes a model for one immutable target-low function snapshot.
//
// User target-binding failures are emitted through |emitter| and recorded in
// |out_model->context.error_count|. Infrastructure failures are returned as
// status. |function_target_facts| supplies invocation-refined facts that
// already include the function contract when non-NULL; otherwise the model
// resolves facts from the authored target witness. The supplied facts must
// outlive the model. |flags| selects whether the value domain covers only the
// immediate function region or its complete nested region tree. |arena| must
// outlive the model and every table derived from it.
iree_status_t loom_low_function_model_initialize(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, loom_low_function_model_flags_t flags,
    iree_arena_allocator_t* arena, loom_low_function_model_t* out_model);

// Releases the module value-ordinal scratch map owned by |model|.
void loom_low_function_model_deinitialize(loom_low_function_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FUNCTION_MODEL_H_
