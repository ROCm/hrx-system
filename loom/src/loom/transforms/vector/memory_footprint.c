// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/memory_footprint.h"

#include "loom/analysis/vector_memory_footprint.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loom/target/selection.h"

#define LOOM_VECTOR_MEMORY_FOOTPRINT_STATISTICS(V, statistics_type) \
  V(statistics_type, ops_checked, "ops-checked",                    \
    "Number of memory ops checked.")                                \
  V(statistics_type, ops_skipped, "ops-skipped",                    \
    "Number of memory ops skipped because no lane accesses memory.")

LOOM_PASS_STATISTICS_DEFINE(loom_vector_memory_footprint_statistics,
                            loom_vector_memory_footprint_statistics_t,
                            LOOM_VECTOR_MEMORY_FOOTPRINT_STATISTICS)

static const loom_pass_info_t loom_vector_memory_footprint_pass_info_storage = {
    .name = IREE_SVL("vector-memory-footprint"),
    .description = IREE_SVL(
        "Prove vector memory footprints are in bounds before lowering."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_vector_memory_footprint_statistics_layout,
};

const loom_pass_info_t* loom_vector_memory_footprint_pass_info(void) {
  return &loom_vector_memory_footprint_pass_info_storage;
}

static iree_status_t loom_vector_memory_footprint_fact_scope(
    loom_pass_t* pass, const loom_module_t* module, loom_func_like_t function,
    loom_pass_value_fact_scope_t* out_scope) {
  *out_scope = loom_pass_value_fact_scope_function(function);
  if (!pass->environment) {
    return iree_ok_status();
  }

  loom_target_bundle_storage_t* bundle_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->arena, sizeof(*bundle_storage),
                                           (void**)&bundle_storage));
  bool resolved = false;
  IREE_RETURN_IF_ERROR(loom_target_pass_capability_resolve_function_bundle(
      pass->environment, module, function, pass->diagnostic_emitter,
      pass->arena, &resolved, bundle_storage));
  if (resolved) {
    *out_scope = loom_pass_value_fact_scope_function_for_target(
        function, &bundle_storage->bundle, NULL);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_memory_footprint_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_pass_value_fact_scope_t fact_scope =
      loom_pass_value_fact_scope_function(function);
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_fact_scope(
      pass, module, function, &fact_scope));

  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_value_facts_acquire(pass, module, fact_scope, &fact_table));

  loom_vector_memory_footprint_result_t result = {0};
  const loom_vector_memory_footprint_options_t options = {
      .arena = pass->arena,
      .fact_table = fact_table,
      .emitter = pass->diagnostic_emitter,
  };
  iree_status_t status = loom_vector_memory_footprint_verify_function(
      module, function, &options, &result);
  if (pass->value_facts != NULL) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  IREE_RETURN_IF_ERROR(status);
  loom_vector_memory_footprint_statistics(pass)->ops_checked +=
      result.checked_op_count;
  loom_vector_memory_footprint_statistics(pass)->ops_skipped +=
      result.skipped_op_count;
  return iree_ok_status();
}
