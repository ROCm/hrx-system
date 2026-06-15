// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/greedy.h"

#include <string.h>

void loom_greedy_rewrite_driver_initialize(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    loom_pass_value_fact_owner_t* value_facts,
    loom_greedy_rewrite_driver_t* out_driver) {
  memset(out_driver, 0, sizeof(*out_driver));
  out_driver->module = module;
  out_driver->scratch_arena = scratch_arena;
  out_driver->value_facts = value_facts;
}

void loom_greedy_rewrite_driver_reset(loom_greedy_rewrite_driver_t* driver) {
  if (!driver) return;
  if (driver->rewriter_initialized) {
    loom_rewriter_deinitialize(&driver->rewriter);
    driver->rewriter_initialized = false;
  }
  if (driver->value_facts) {
    loom_pass_value_fact_owner_invalidate(driver->value_facts);
  }
  driver->latest_facts = NULL;
  if (driver->scratch_arena) {
    iree_arena_reset(driver->scratch_arena);
  }
}

void loom_greedy_rewrite_driver_deinitialize(
    loom_greedy_rewrite_driver_t* driver) {
  if (!driver) return;
  loom_greedy_rewrite_driver_reset(driver);
  memset(driver, 0, sizeof(*driver));
}

const loom_value_fact_table_t* loom_greedy_rewrite_driver_fact_table(
    const loom_greedy_rewrite_driver_t* driver) {
  return driver ? driver->latest_facts : NULL;
}

void loom_greedy_rewrite_result_record_rewriter_flags(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter) {
  if (!result || !rewriter) return;
  if (iree_any_bit_set(rewriter->flags, LOOM_REWRITER_FLAG_FACTS_CHANGED)) {
    result->facts_changed = true;
  }
  if (iree_any_bit_set(rewriter->flags, LOOM_REWRITER_FLAG_TYPE_CHANGED)) {
    result->types_changed = true;
  }
}

void loom_greedy_rewrite_result_record_change(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter,
    loom_greedy_rewrite_change_flags_t flags) {
  if (!result) return;
  result->changed = true;
  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);
  if (iree_any_bit_set(flags,
                       LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP)) {
    ++result->ops_modified;
  }
}

static iree_status_t loom_greedy_rewrite_enable_region_facts(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_greedy_rewrite_options_t* options) {
  if (!driver->value_facts) {
    if (options && options->seed_facts) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "seed facts require a greedy rewrite value-fact owner");
    }
    return iree_ok_status();
  }

  loom_value_fact_table_t* facts = NULL;
  const loom_target_bundle_t* target_bundle =
      options && options->seed_facts
          ? options->seed_facts->context.target_bundle
          : NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare(
      driver->value_facts, driver->module,
      loom_pass_value_fact_scope_region_for_target(function, region, parent_op,
                                                   target_bundle),
      &facts));
  driver->latest_facts = facts;
  return loom_rewriter_enable_region_analysis_with_seed_facts(
      &driver->rewriter, function, region, parent_op, facts,
      options ? options->seed_facts : NULL);
}

iree_status_t loom_greedy_rewrite_run_region(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_greedy_rewrite_options_t* options,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* out_result) {
  if (out_result) memset(out_result, 0, sizeof(*out_result));
  loom_greedy_rewrite_driver_reset(driver);
  if (!region) return iree_ok_status();

  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS;
  loom_greedy_rewrite_result_t result = {0};
  iree_status_t status = loom_rewriter_initialize(
      &driver->rewriter, driver->module, driver->scratch_arena);
  if (iree_status_is_ok(status)) {
    driver->rewriter_initialized = true;
    if (options) {
      driver->rewriter.materialize_constant = options->materialize_constant;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_greedy_rewrite_enable_region_facts(driver, function, region,
                                                     parent_op, options);
  }
  bool prepare_region_called = false;
  if (iree_status_is_ok(status) && callbacks && callbacks->prepare_region) {
    prepare_region_called = true;
    status = callbacks->prepare_region(callbacks->user_data, driver, function,
                                       region, parent_op);
  }

  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    status = loom_rewriter_seed_region(&driver->rewriter, region);
    if (!iree_status_is_ok(status)) break;

    bool any_changed = false;
    if (callbacks && callbacks->before_worklist) {
      bool changed = false;
      status = callbacks->before_worklist(callbacks->user_data, driver, region,
                                          &result, &changed);
      if (!iree_status_is_ok(status)) break;
      if (changed) {
        any_changed = true;
        if (callbacks->changed) {
          callbacks->changed(callbacks->user_data, driver);
        }
      }
    }

    loom_op_t* op = NULL;
    while ((op = loom_rewriter_pop(&driver->rewriter)) != NULL) {
      if (!callbacks || !callbacks->rewrite_op) continue;
      bool changed = false;
      status = callbacks->rewrite_op(callbacks->user_data, driver, op, &result,
                                     &changed);
      if (!iree_status_is_ok(status)) break;
      if (changed) {
        any_changed = true;
        if (callbacks->changed) {
          callbacks->changed(callbacks->user_data, driver);
        }
      }
    }
    if (!iree_status_is_ok(status)) break;
    if (!any_changed) break;
  }

  if (prepare_region_called && callbacks && callbacks->cleanup_region) {
    callbacks->cleanup_region(callbacks->user_data, driver);
  }
  if (iree_status_is_ok(status)) {
    result.boundary_maybe_changed =
        result.changed || result.facts_changed || result.types_changed;
    loom_rewriter_deinitialize(&driver->rewriter);
    driver->rewriter_initialized = false;
    if (out_result) *out_result = result;
    return iree_ok_status();
  }

  loom_greedy_rewrite_driver_reset(driver);
  return status;
}

typedef struct loom_pattern_rewrite_state_t {
  // Pattern array tried in declaration order.
  const loom_pattern_t* patterns;

  // Number of patterns in patterns.
  iree_host_size_t pattern_count;
} loom_pattern_rewrite_state_t;

static iree_status_t loom_greedy_rewrite_patterns_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_pattern_rewrite_state_t* state =
      (loom_pattern_rewrite_state_t*)user_data;
  for (iree_host_size_t i = 0; i < state->pattern_count; ++i) {
    const loom_pattern_t* pattern = &state->patterns[i];
    if (pattern->root_kind != op->kind) {
      continue;
    }
    driver->rewriter.flags = 0;
    IREE_RETURN_IF_ERROR(
        pattern->match_and_rewrite(pattern, op, &driver->rewriter));
    loom_greedy_rewrite_result_record_rewriter_flags(result, &driver->rewriter);
    if (iree_any_bit_set(driver->rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
      loom_greedy_rewrite_result_record_change(
          result, &driver->rewriter,
          LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
      *out_changed = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_greedy_rewrite_run_patterns(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    const loom_pattern_t* patterns, iree_host_size_t pattern_count,
    const loom_greedy_rewrite_options_t* options,
    loom_greedy_rewrite_result_t* out_result) {
  if (pattern_count > 0 && !patterns) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pattern array is required");
  }
  loom_pattern_rewrite_state_t state = {
      .patterns = patterns,
      .pattern_count = pattern_count,
  };
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .rewrite_op = loom_greedy_rewrite_patterns_op,
  };
  return loom_greedy_rewrite_run_region(
      driver, function, loom_func_like_body(function), function.op, options,
      &callbacks, out_result);
}

iree_status_t loom_greedy_rewrite(iree_arena_allocator_t* arena,
                                  loom_module_t* module,
                                  loom_func_like_t function,
                                  const loom_pattern_t* patterns,
                                  iree_host_size_t pattern_count,
                                  const loom_rewrite_config_t* config) {
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module, arena, NULL, &driver);
  loom_greedy_rewrite_options_t options = {
      .max_iterations = config ? config->max_iterations : 0,
  };
  iree_status_t status = loom_greedy_rewrite_run_patterns(
      &driver, function, patterns, pattern_count, &options, NULL);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  return status;
}
