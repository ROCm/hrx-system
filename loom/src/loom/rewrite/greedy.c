// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/greedy.h"

#include <string.h>

void loom_greedy_rewrite_driver_initialize(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    loom_value_fact_table_t* fact_table,
    loom_greedy_rewrite_driver_t* out_driver) {
  memset(out_driver, 0, sizeof(*out_driver));
  out_driver->module = module;
  out_driver->scratch_arena = scratch_arena;
  out_driver->fact_table = fact_table;
}

void loom_greedy_rewrite_driver_set_fact_table(
    loom_greedy_rewrite_driver_t* driver, loom_value_fact_table_t* fact_table) {
  IREE_ASSERT(!driver->rewriter_initialized);
  driver->fact_table = fact_table;
}

void loom_greedy_rewrite_driver_reset(loom_greedy_rewrite_driver_t* driver) {
  if (!driver) {
    return;
  }
  if (driver->rewriter_initialized) {
    loom_rewriter_deinitialize(&driver->rewriter);
    driver->rewriter_initialized = false;
  }
  if (driver->scratch_arena) {
    iree_arena_reset(driver->scratch_arena);
  }
}

void loom_greedy_rewrite_driver_deinitialize(
    loom_greedy_rewrite_driver_t* driver) {
  if (!driver) {
    return;
  }
  loom_greedy_rewrite_driver_reset(driver);
  memset(driver, 0, sizeof(*driver));
}

const loom_value_fact_table_t* loom_greedy_rewrite_driver_fact_table(
    const loom_greedy_rewrite_driver_t* driver) {
  return driver ? driver->fact_table : NULL;
}

void loom_greedy_rewrite_result_record_rewriter_flags(
    loom_greedy_rewrite_result_t* result, const loom_rewriter_t* rewriter) {
  if (!result || !rewriter) {
    return;
  }
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
  if (!result) {
    return;
  }
  result->changed = true;
  loom_greedy_rewrite_result_record_rewriter_flags(result, rewriter);
  if (iree_any_bit_set(flags,
                       LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP)) {
    ++result->ops_modified;
  }
}

static iree_status_t loom_greedy_rewrite_apply_prepared(
    loom_greedy_rewrite_driver_t* driver, loom_region_t* region,
    const loom_rewrite_pattern_registry_t* patterns,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_rewriter_propagate_value_facts(&driver->rewriter));
  loom_greedy_rewrite_result_record_rewriter_flags(result, &driver->rewriter);
  iree_arena_allocator_t arena;
  iree_arena_initialize(driver->scratch_arena->block_pool, &arena);
  loom_rewrite_action_t* actions = NULL;
  iree_status_t status = loom_rewrite_pattern_registry_prepare_region(
      patterns, driver->module, driver->fact_table, region, &arena, &actions);
  for (loom_rewrite_action_t* action = actions;
       iree_status_is_ok(status) && action; action = action->next) {
    driver->rewriter.flags = 0;
    status = action->apply(action, &driver->rewriter);
    loom_greedy_rewrite_result_record_change(
        result, &driver->rewriter,
        LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    if (callbacks && callbacks->changed) {
      callbacks->changed(callbacks->user_data, driver);
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_greedy_rewrite_run_region(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_greedy_rewrite_options_t* options,
    const loom_greedy_rewrite_callbacks_t* callbacks,
    loom_greedy_rewrite_result_t* out_result) {
  if (out_result) {
    memset(out_result, 0, sizeof(*out_result));
  }
  loom_greedy_rewrite_driver_reset(driver);
  if (!region) {
    return iree_ok_status();
  }

  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS;
  loom_greedy_rewrite_result_t result = {0};
  loom_rewriter_initialize(&driver->rewriter, driver->module,
                           driver->scratch_arena);
  driver->rewriter_initialized = true;
  loom_rewriter_attach_value_facts(&driver->rewriter, driver->fact_table);
  if (options) {
    driver->rewriter.constant_materializer = options->constant_materializer;
    driver->rewriter.pending_exact_relations_callback =
        options->pending_exact_relations_callback;
    driver->rewriter.math_policy = options->math_policy;
    driver->rewriter.math_target_bundle = options->math_target_bundle;
  }
  iree_status_t status = loom_rewriter_enable_worklist(&driver->rewriter);
  bool prepare_region_called = false;
  if (iree_status_is_ok(status) && callbacks && callbacks->prepare_region) {
    prepare_region_called = true;
    status = callbacks->prepare_region(callbacks->user_data, driver, function,
                                       region, parent_op);
  }
  if (iree_status_is_ok(status) && options && options->prepared_patterns &&
      options->prepared_patterns->pattern_count) {
    status = loom_greedy_rewrite_apply_prepared(
        driver, region, options->prepared_patterns, callbacks, &result);
  }

  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    status = loom_rewriter_seed_region(&driver->rewriter, region);
    if (!iree_status_is_ok(status)) {
      break;
    }

    bool any_changed = false;
    if (callbacks && callbacks->rewrite_op_without_facts) {
      loom_op_t* op = NULL;
      while ((op = loom_rewriter_pop(&driver->rewriter)) != NULL) {
        bool changed = false;
        status = callbacks->rewrite_op_without_facts(
            callbacks->user_data, driver, op, &result, &changed);
        if (!iree_status_is_ok(status)) {
          break;
        }
        if (changed) {
          any_changed = true;
          if (callbacks->changed) {
            callbacks->changed(callbacks->user_data, driver);
          }
        }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      status = loom_rewriter_propagate_value_facts(&driver->rewriter);
      loom_greedy_rewrite_result_record_rewriter_flags(&result,
                                                       &driver->rewriter);
      if (!iree_status_is_ok(status)) {
        break;
      }
      // Fact propagation schedules affected operations, but the complete
      // region rescan below supersedes that incremental work. Drain it first
      // so solver publication order cannot perturb the deterministic region
      // order seen by fact-observing rewrites.
      while (loom_rewriter_pop(&driver->rewriter) != NULL) {
      }
      status = loom_rewriter_seed_region(&driver->rewriter, region);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }

    if (callbacks && callbacks->before_worklist) {
      bool changed = false;
      status = loom_rewriter_propagate_value_facts(&driver->rewriter);
      loom_greedy_rewrite_result_record_rewriter_flags(&result,
                                                       &driver->rewriter);
      if (iree_status_is_ok(status)) {
        status = callbacks->before_worklist(callbacks->user_data, driver,
                                            region, &result, &changed);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (changed) {
        any_changed = true;
        if (callbacks->changed) {
          callbacks->changed(callbacks->user_data, driver);
        }
      }
    }

    loom_op_t* op = NULL;
    while ((op = loom_rewriter_pop(&driver->rewriter)) != NULL) {
      if (!callbacks || !callbacks->rewrite_op) {
        continue;
      }
      status = loom_rewriter_propagate_value_facts(&driver->rewriter);
      loom_greedy_rewrite_result_record_rewriter_flags(&result,
                                                       &driver->rewriter);
      if (!iree_status_is_ok(status)) {
        break;
      }
      // Patterns and any operations they create use the current region's
      // target. Optional math choices require that target's own policy.
      driver->rewriter.math_policy = options ? options->math_policy : NULL;
      const loom_target_facts_t* enclosing_target =
          driver->fact_table ? driver->fact_table->context.target_facts : NULL;
      if (driver->fact_table &&
          driver->fact_table->regions.has_independent_targets) {
        driver->fact_table->context.target_facts =
            loom_value_fact_table_block_target_facts(driver->fact_table,
                                                     op->parent_block);
        if (driver->fact_table->context.target_facts != enclosing_target) {
          driver->rewriter.math_policy = NULL;
        }
      }
      bool changed = false;
      status = callbacks->rewrite_op(callbacks->user_data, driver, op, &result,
                                     &changed);
      if (driver->fact_table) {
        driver->fact_table->context.target_facts = enclosing_target;
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (changed) {
        any_changed = true;
        if (callbacks->changed) {
          callbacks->changed(callbacks->user_data, driver);
        }
      }
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!any_changed) {
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    status = loom_rewriter_propagate_value_facts(&driver->rewriter);
    loom_greedy_rewrite_result_record_rewriter_flags(&result,
                                                     &driver->rewriter);
  }
  if (prepare_region_called && callbacks && callbacks->cleanup_region) {
    callbacks->cleanup_region(callbacks->user_data, driver);
  }
  if (iree_status_is_ok(status)) {
    result.boundary_maybe_changed =
        result.changed || result.facts_changed || result.types_changed;
    loom_rewriter_deinitialize(&driver->rewriter);
    driver->rewriter_initialized = false;
    if (out_result) {
      *out_result = result;
    }
    return iree_ok_status();
  }

  loom_greedy_rewrite_driver_reset(driver);
  return status;
}

typedef struct loom_pattern_rewrite_state_t {
  // Indexed pattern registry composed for this invocation.
  const loom_rewrite_pattern_registry_t* registry;

  // Invocation-owned state shared by registry callbacks.
  void* context;
} loom_pattern_rewrite_state_t;

static iree_status_t loom_greedy_rewrite_patterns_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_pattern_rewrite_state_t* state =
      (loom_pattern_rewrite_state_t*)user_data;
  driver->rewriter.flags = 0;
  IREE_RETURN_IF_ERROR(loom_rewrite_pattern_registry_apply(
      state->registry, state->context, op, &driver->rewriter, out_changed));
  loom_greedy_rewrite_result_record_rewriter_flags(result, &driver->rewriter);
  if (*out_changed) {
    loom_greedy_rewrite_result_record_change(
        result, &driver->rewriter,
        LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
  }
  return iree_ok_status();
}

iree_status_t loom_greedy_rewrite_run_patterns(
    loom_greedy_rewrite_driver_t* driver, loom_func_like_t function,
    const loom_rewrite_pattern_registry_t* pattern_registry,
    void* pattern_context, const loom_greedy_rewrite_options_t* options,
    loom_greedy_rewrite_result_t* out_result) {
  if (pattern_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rewrite pattern registry is required");
  }
  loom_pattern_rewrite_state_t state = {
      .registry = pattern_registry,
      .context = pattern_context,
  };
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .rewrite_op = loom_greedy_rewrite_patterns_op,
  };
  return loom_greedy_rewrite_run_region(
      driver, function, loom_func_like_body(function), function.op, options,
      &callbacks, out_result);
}

iree_status_t loom_greedy_rewrite(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_func_like_t function,
    const loom_rewrite_pattern_registry_t* pattern_registry,
    void* pattern_context, const loom_rewrite_config_t* config) {
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module, arena, /*fact_table=*/NULL,
                                        &driver);
  loom_greedy_rewrite_options_t options = {
      .max_iterations = config ? config->max_iterations : 0,
  };
  iree_status_t status = loom_greedy_rewrite_run_patterns(
      &driver, function, pattern_registry, pattern_context, &options, NULL);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  return status;
}
