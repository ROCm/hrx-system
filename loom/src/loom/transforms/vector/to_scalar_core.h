// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state shared by vector-to-scalar lowering clusters.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_CORE_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_CORE_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/to_scalar_options.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_VECTOR_TO_SCALAR_STATISTICS(V, statistics_type)   \
  V(statistics_type, ops_lowered, "ops-lowered",               \
    "Number of vector ops lowered.")                           \
  V(statistics_type, loops_created, "loops-created",           \
    "Number of scf.for loops created.")                        \
  V(statistics_type, lanes_materialized, "lanes-materialized", \
    "Number of scalar lane programs materialized.")

LOOM_PASS_STATISTICS_DECLARE(loom_vector_to_scalar_statistics,
                             loom_vector_to_scalar_statistics_t,
                             LOOM_VECTOR_TO_SCALAR_STATISTICS)

extern const loom_pass_statistic_layout_t
    loom_vector_to_scalar_statistics_layout;

static inline loom_vector_to_scalar_statistics_t*
loom_vector_to_scalar_statistics_if_available(loom_pass_t* pass) {
  return pass && pass->statistic_storage && pass->info &&
                 pass->info->statistic_layout ==
                     &loom_vector_to_scalar_statistics_layout
             ? loom_vector_to_scalar_statistics(pass)
             : NULL;
}

typedef struct loom_vector_to_scalar_descriptor_t
    loom_vector_to_scalar_descriptor_t;

typedef struct loom_vector_to_scalar_state_t {
  // Current pass instance owning statistics and transient arena state.
  loom_pass_t* pass;
  // Typed statistics storage for this pass invocation or state-local discard
  // storage when the scalarization helper is composed under another pass.
  loom_vector_to_scalar_statistics_t* statistics;
  // State-local discard storage used for composed helper invocations.
  loom_vector_to_scalar_statistics_t discarded_statistics;
  // Rewriter used to insert replacement lane IR and maintain use-def state.
  loom_rewriter_t* rewriter;
  // Vector op currently being scalarized.
  loom_op_t* op;
  // Descriptor that selects the lane-program family for |op|.
  const loom_vector_to_scalar_descriptor_t* descriptor;
  // Rewriter value checkpoint used to preserve result names on new values.
  loom_value_id_t value_checkpoint;
  // Result ordinal being scalarized for multi-result vector ops.
  uint16_t result_ordinal;
  // Vector result type currently being expanded into scalar lanes.
  loom_type_t vector_type;
  // Scalar type produced by each lane program.
  loom_type_t result_scalar_type;
  // Optional lowering behaviors enabled for this scalarization.
  loom_vector_to_scalar_flags_t flags;
  // Target-selected matrix fragment layout available to MMA fallback lowering.
  const loom_matrix_fragment_layout_t* matrix_fragment_layout;
  // Source location assigned to replacement ops.
  loom_location_id_t location;
} loom_vector_to_scalar_state_t;

static inline void loom_vector_to_scalar_state_bind_statistics(
    loom_vector_to_scalar_state_t* state, loom_pass_t* pass) {
  state->statistics = loom_vector_to_scalar_statistics_if_available(pass);
  if (!state->statistics) {
    state->statistics = &state->discarded_statistics;
  }
}

static inline void loom_vector_to_scalar_record_ops_lowered(
    loom_vector_to_scalar_state_t* state) {
  ++state->statistics->ops_lowered;
}

static inline void loom_vector_to_scalar_record_loop_created(
    loom_vector_to_scalar_state_t* state) {
  ++state->statistics->loops_created;
}

static inline void loom_vector_to_scalar_record_lane_materialized(
    loom_vector_to_scalar_state_t* state) {
  ++state->statistics->lanes_materialized;
}

// Returns the op defining |value_id|, or NULL when the value is not an op
// result that can be rematerialized from its defining operation.
static inline loom_op_t* loom_vector_to_scalar_value_def_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return NULL;
  }
  loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  return loom_value_def_op(value);
}

// Computes a static lane count representable by scalarized IR value slices.
iree_status_t loom_vector_to_scalar_static_element_count(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    uint16_t* out_element_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_CORE_H_
