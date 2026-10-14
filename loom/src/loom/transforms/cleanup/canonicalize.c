// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/canonicalize.h"

#include "loom/pass/options.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/transforms/cleanup/canonicalizer.h"
#include "loom/transforms/cleanup/integer_division.h"
#include "loom/transforms/cleanup/pass_environment.h"

static const loom_pass_option_def_t kCanonicalizeOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of worklist iterations.")},
};

#define LOOM_CANONICALIZE_STATISTICS(V, statistics_type)                       \
  V(statistics_type, ops_modified, "ops-modified",                             \
    "Number of ops simplified or combined.")                                   \
  V(statistics_type, type_propagation_conflicts, "type-propagation-conflicts", \
    "Number of type propagation candidate closures rejected as "               \
    "inconsistent.")                                                           \
  V(statistics_type, type_propagation_rejection_cache_hits,                    \
    "type-propagation-rejection-cache-hits",                                   \
    "Number of repeated rejected type candidates skipped within an "           \
    "iteration.")

LOOM_PASS_STATISTICS_DEFINE(loom_canonicalize_statistics,
                            loom_canonicalize_statistics_t,
                            LOOM_CANONICALIZE_STATISTICS)

static const loom_pass_info_t loom_canonicalize_pass_info_storage = {
    .name = IREE_SVL("canonicalize"),
    .description =
        IREE_SVL("Apply universal simplifications at any pipeline stage."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kCanonicalizeOptions,
    .option_count = IREE_ARRAYSIZE(kCanonicalizeOptions),
    .statistic_layout = &loom_canonicalize_statistics_layout,
};

static const loom_pass_info_t loom_combine_pass_info_storage = {
    .name = IREE_SVL("combine"),
    .description =
        IREE_SVL("Combine source operations before target legalization."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kCanonicalizeOptions,
    .option_count = IREE_ARRAYSIZE(kCanonicalizeOptions),
    .statistic_layout = &loom_canonicalize_statistics_layout,
};

const loom_pass_info_t* loom_canonicalize_pass_info(void) {
  return &loom_canonicalize_pass_info_storage;
}

const loom_pass_info_t* loom_combine_pass_info(void) {
  return &loom_combine_pass_info_storage;
}

typedef struct loom_canonicalize_pass_options_t {
  // Maximum worklist iterations. Zero selects the shared driver's default.
  uint32_t max_iterations;
} loom_canonicalize_pass_options_t;

typedef iree_status_t (*loom_canonicalizer_post_run_fn_t)(
    loom_module_t* module, loom_func_like_t function,
    const loom_value_fact_table_t* facts, iree_arena_allocator_t* arena,
    loom_canonicalizer_result_t* result, bool* out_changed);

static iree_status_t loom_canonicalize_parse_option(void* user_data,
                                                    iree_string_view_t name,
                                                    iree_string_view_t value) {
  loom_pass_t* pass = (loom_pass_t*)user_data;
  loom_canonicalize_pass_options_t* options =
      (loom_canonicalize_pass_options_t*)pass->state;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass '%.*s'",
          (int)pass->info->name.size, pass->info->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        pass->info->name, name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option 'max-iterations' must be greater than 0",
          (int)pass->info->name.size, pass->info->name.data);
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass '%.*s'",
                          (int)name.size, name.data, (int)pass->info->name.size,
                          pass->info->name.data);
}

iree_status_t loom_canonicalizer_pass_create(
    loom_pass_t* pass, iree_string_view_t options_string) {
  loom_canonicalize_pass_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  *options = (loom_canonicalize_pass_options_t){0};
  pass->state = options;
  if (pass->decoded_options) {
    // Both pass schemas contain only max-iterations. Option decoding validates
    // it and supplies zero for the absent option, selecting the default.
    options->max_iterations = pass->decoded_options->options[0].uint32_value;
    return iree_ok_status();
  }
  return loom_pass_options_parse(pass->info->name, options_string,
                                 (loom_pass_option_parse_callback_t){
                                     .fn = loom_canonicalize_parse_option,
                                     .user_data = pass,
                                 });
}

static iree_status_t loom_canonicalizer_run_pass(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_cleanup_pass_capability_t* cleanup_capability,
    loom_canonicalizer_pattern_registries_t patterns,
    loom_canonicalizer_post_run_fn_t post_run) {
  loom_canonicalizer_options_t run_options = {
      .patterns = patterns,
  };
  if (pass->state) {
    run_options.max_iterations =
        ((const loom_canonicalize_pass_options_t*)pass->state)->max_iterations;
  }
  loom_cleanup_canonicalizer_context_t context = {0};
  IREE_RETURN_IF_ERROR(
      loom_cleanup_pass_capability_resolve_canonicalizer_context(
          cleanup_capability, pass, module, function, &context));
  run_options.target_facts = context.target_facts;
  run_options.math_policy = context.math_policy;
  const loom_cleanup_pattern_registry_t* pattern_registry =
      loom_cleanup_pass_capability_pattern_registry(cleanup_capability);

  loom_canonicalizer_t canonicalizer;
  IREE_RETURN_IF_ERROR(loom_canonicalizer_initialize(
      module, pass->arena, pass->value_facts,
      pattern_registry ? pattern_registry->special_value_policy : NULL,
      pattern_registry ? pattern_registry->fact_refinement_policy : NULL,
      &canonicalizer));

  loom_canonicalizer_result_t result;
  iree_status_t status = loom_canonicalizer_run_function(
      &canonicalizer, function, &run_options, &result);
  if (iree_status_is_ok(status) && post_run) {
    bool post_run_changed = false;
    status = post_run(module, function,
                      loom_canonicalizer_fact_table(&canonicalizer),
                      pass->arena, &result, &post_run_changed);
    if (!iree_status_is_ok(status) || post_run_changed) {
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }
  if (iree_status_is_ok(status)) {
    if (result.changed) {
      loom_pass_mark_changed(pass);
    }
    loom_canonicalize_statistics_t* statistics =
        loom_canonicalize_statistics(pass);
    statistics->ops_modified += result.ops_modified;
    statistics->type_propagation_conflicts += result.type_propagation_conflicts;
    statistics->type_propagation_rejection_cache_hits +=
        result.type_propagation_rejection_cache_hits;
  }
  loom_canonicalizer_deinitialize(&canonicalizer);
  return status;
}

static iree_status_t loom_combine_integer_division(
    loom_module_t* module, loom_func_like_t function,
    const loom_value_fact_table_t* facts, iree_arena_allocator_t* arena,
    loom_canonicalizer_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_integer_division_combine_result_t combine_result;
  IREE_RETURN_IF_ERROR(loom_integer_division_combine(module, function, facts,
                                                     arena, &combine_result));
  if (combine_result.combined_operation_count != 0) {
    *out_changed = true;
    result->changed = true;
    result->boundary_maybe_changed = true;
    result->ops_modified += combine_result.retired_operation_count;
  }
  return iree_ok_status();
}

iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  const loom_cleanup_pass_capability_t* cleanup_capability =
      loom_cleanup_pass_capability_from_pass(pass);
  const loom_cleanup_pattern_registry_t* registry =
      loom_cleanup_pass_capability_pattern_registry(cleanup_capability);
  const loom_canonicalizer_pattern_registries_t patterns =
      loom_canonicalizer_pattern_registries_from_cleanup_registry(registry);
  return loom_canonicalizer_run_pass(pass, module, function, cleanup_capability,
                                     patterns, /*post_run=*/NULL);
}

iree_status_t loom_combine_run(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function) {
  const loom_cleanup_pass_capability_t* cleanup_capability =
      loom_cleanup_pass_capability_from_pass(pass);
  const loom_cleanup_pattern_registry_t* registry =
      loom_cleanup_pass_capability_pattern_registry(cleanup_capability);
  IREE_ASSERT(registry != NULL);
  IREE_ASSERT(registry->source_combine != NULL);
  loom_canonicalizer_pattern_registries_t patterns =
      loom_canonicalizer_pattern_registries_from_cleanup_registry(registry);
  patterns.post_canonicalization = registry->source_combine;
  return loom_canonicalizer_run_pass(pass, module, function, cleanup_capability,
                                     patterns, loom_combine_integer_division);
}
