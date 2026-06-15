// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent owned-resource lifetime analysis.
//
// The analysis interprets descriptor-backed ownership effects over a
// function-local value domain. It checks that owned resources are consumed,
// released, discarded, or escaped exactly once along every CFG path and that
// path joins do not hide an owned obligation on only some predecessors.

#ifndef LOOM_ANALYSIS_OWNERSHIP_LIFETIME_H_
#define LOOM_ANALYSIS_OWNERSHIP_LIFETIME_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/ownership.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ownership_lifetime_options_t {
  // Scratch arena for module summaries and transient analysis state.
  iree_arena_allocator_t* arena;
  // Structured diagnostic emitter for ownership lifetime failures.
  iree_diagnostic_emitter_t emitter;
  // Name of the phase reporting diagnostics.
  iree_string_view_t phase_name;
} loom_ownership_lifetime_options_t;

typedef enum loom_ownership_lifetime_materialization_policy_flag_bits_e {
  // Matching function-like arguments enter the analyzed body as owned values.
  LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_ARGUMENTS = 1u << 0,
  // Matching results from bodyless callees enter callers as owned values.
  LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_BODYLESS_RESULTS = 1u
                                                                          << 1,
} loom_ownership_lifetime_materialization_policy_flag_bits_t;

// Bitset of loom_ownership_lifetime_materialization_policy_flag_bits_t values.
typedef uint32_t loom_ownership_lifetime_materialization_policy_flags_t;

// Builds a release operation at the builder insertion point.
typedef iree_status_t (*loom_ownership_lifetime_build_release_op_fn_t)(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op);

// Target- or dialect-owned materialization policy for one resource family.
typedef struct loom_ownership_lifetime_materialization_policy_t {
  // Resource family whose values this policy materializes.
  loom_ownership_resource_family_t family;
  // ABI and materialization behavior for values matched by |family|.
  loom_ownership_lifetime_materialization_policy_flags_t flags;
  // Builder for explicit release operations.
  loom_ownership_lifetime_build_release_op_fn_t build_release;
  // Opaque payload passed to the builder callbacks.
  void* user_data;
} loom_ownership_lifetime_materialization_policy_t;

typedef struct loom_ownership_lifetime_materialize_options_t {
  // Scratch arena for module summaries and transient analysis state.
  iree_arena_allocator_t* arena;
  // Structured diagnostic emitter for ownership lifetime failures.
  iree_diagnostic_emitter_t emitter;
  // Name of the phase reporting diagnostics.
  iree_string_view_t phase_name;
  // Required resource-family policies active for this materialization stage.
  const loom_ownership_lifetime_materialization_policy_t* policies;
  // Number of entries in |policies|.
  iree_host_size_t policy_count;
} loom_ownership_lifetime_materialize_options_t;

typedef struct loom_ownership_lifetime_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of blocks checked for lifetime consistency.
  uint64_t blocks_checked;
  // Number of operations visited while checking ownership effects.
  uint64_t ops_checked;
  // Number of descriptor-backed ownership effects interpreted.
  uint64_t effects_checked;
  // Number of explicit release operations inserted.
  uint64_t releases_inserted;
  // Number of CFG edges split to preserve edge-local cleanup semantics.
  uint64_t edges_split;
} loom_ownership_lifetime_result_t;

// Analyzes owned-resource lifetimes for all function-like bodies in a module.
//
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|. The analysis returns OK for user IR failures so callers can use
// it as a pass, a target-lowering gate, or an importer diagnostic source.
// Infrastructure failures such as arena allocation failures are returned as
// status failures.
iree_status_t loom_ownership_lifetime_analyze_module(
    loom_module_t* module, const loom_ownership_lifetime_options_t* options,
    loom_ownership_lifetime_result_t* out_result);

// Materializes explicit resource lifetime operations for all function-like
// bodies in a module.
//
// This is the mutating production form of the lifetime stage. It runs the
// same summary solver as the non-mutating analyzer, records cleanup actions
// during the final stable transfer, and applies those actions after planning.
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|; infrastructure failures are returned as status failures.
iree_status_t loom_ownership_lifetime_materialize_module(
    loom_module_t* module,
    const loom_ownership_lifetime_materialize_options_t* options,
    loom_ownership_lifetime_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_OWNERSHIP_LIFETIME_H_
