// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/target_legalize_options.h"

#include <string.h>

#include "loom/codegen/low/transforms/pipeline/target_legalize.h"
#include "loom/pass/options.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/rewrite/greedy.h"

typedef struct loom_low_target_legalize_parse_context_t {
  // Mutable pass state being populated.
  loom_low_target_legalize_pass_state_t* state;
} loom_low_target_legalize_parse_context_t;

static const loom_pass_option_def_t kLowTargetLegalizeOptions[] = {
    {IREE_SVL("boundaries"),
     IREE_SVL("Boundary representation: logical or projected.")},
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of final legality diagnostics to emit; zero "
              "means no limit.")},
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of target legalization worklist iterations.")},
    {IREE_SVL("mode"),
     IREE_SVL(
         "Legalization phase: eager (partial rewrites), complete (finish "
         "rewrites), or final (finish rewrites and verify source legality).")},
    {IREE_SVL("policy"),
     IREE_SVL("Legalization strategy policy: prefer-native, reference-only, "
              "or require-native.")},
};

static const loom_pass_statistic_field_t kLowTargetLegalizeStatisticFields[] = {
    LOOM_LOW_TARGET_LEGALIZE_STATISTICS(LOOM_PASS_STATISTIC_LAYOUT_FIELD_,
                                        loom_low_target_legalize_statistics_t)};

const loom_pass_statistic_layout_t loom_low_target_legalize_statistics_layout =
    LOOM_PASS_STATISTIC_LAYOUT(loom_low_target_legalize_statistics_t,
                               kLowTargetLegalizeStatisticFields);

static const loom_pass_info_t loom_low_target_legalize_pass_info_storage = {
    .name = IREE_SVL("target-legalize"),
    .description =
        IREE_SVL("Rewrite unsupported target-bound source ops toward a legal "
                 "target-low contract."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowTargetLegalizeOptions,
    .option_count = IREE_ARRAYSIZE(kLowTargetLegalizeOptions),
    .statistic_layout = &loom_low_target_legalize_statistics_layout,
};

const loom_pass_info_t* loom_low_target_legalize_pass_info(void) {
  return &loom_low_target_legalize_pass_info_storage;
}

static iree_status_t loom_low_target_legalize_parse_max_errors(
    uint32_t max_errors, loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_errors_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-errors' for pass 'target-legalize'");
  }
  context->state->max_errors = max_errors;
  context->state->has_max_errors_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_max_iterations(
    uint32_t max_iterations,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_iterations_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-iterations' for pass 'target-legalize'");
  }
  if (max_iterations == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass 'target-legalize' option "
                            "'max-iterations' must be greater than 0");
  }
  context->state->max_iterations = max_iterations;
  context->state->has_max_iterations_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_mode(
    iree_string_view_t value,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_mode_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'mode' for pass "
                            "'target-legalize'");
  }
  if (iree_string_view_equal(value, IREE_SV("eager"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;
  } else if (iree_string_view_equal(value, IREE_SV("complete"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_FINAL;
  } else if (iree_string_view_equal(value, IREE_SV("final"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_FINAL;
    context->state->verify_source_legality = true;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-legalize option 'mode' expected 'eager' "
                            "or 'complete' or 'final', got '%.*s'",
                            (int)value.size, value.data);
  }
  context->state->has_mode_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_policy(
    iree_string_view_t value,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_policy_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'policy' for pass "
                            "'target-legalize'");
  }
  if (iree_string_view_equal(value, IREE_SV("prefer-native"))) {
    context->state->policy = LOOM_TARGET_LEGALIZATION_POLICY_PREFER_NATIVE;
  } else if (iree_string_view_equal(value, IREE_SV("reference-only"))) {
    context->state->policy = LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY;
  } else if (iree_string_view_equal(value, IREE_SV("require-native"))) {
    context->state->policy = LOOM_TARGET_LEGALIZATION_POLICY_REQUIRE_NATIVE;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-legalize option 'policy' expected 'prefer-native', "
        "'reference-only', or 'require-native', got '%.*s'",
        (int)value.size, value.data);
  }
  context->state->has_policy_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_boundaries(
    iree_string_view_t value,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_boundaries_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'boundaries' for pass "
                            "'target-legalize'");
  }
  if (iree_string_view_equal(value, IREE_SV("logical"))) {
    context->state->boundary_representation =
        LOOM_TARGET_BOUNDARY_REPRESENTATION_LOGICAL;
  } else if (iree_string_view_equal(value, IREE_SV("projected"))) {
    context->state->boundary_representation =
        LOOM_TARGET_BOUNDARY_REPRESENTATION_PROJECTED;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-legalize option 'boundaries' expected 'logical' or "
        "'projected', got '%.*s'",
        (int)value.size, value.data);
  }
  context->state->has_boundaries_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_target_legalize_parse_context_t* context =
      (loom_low_target_legalize_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-errors"))) {
    uint32_t max_errors = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_errors));
    return loom_low_target_legalize_parse_max_errors(max_errors, context);
  }
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    uint32_t max_iterations = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_iterations));
    return loom_low_target_legalize_parse_max_iterations(max_iterations,
                                                         context);
  }
  if (iree_string_view_equal(name, IREE_SV("mode"))) {
    return loom_low_target_legalize_parse_mode(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("boundaries"))) {
    return loom_low_target_legalize_parse_boundaries(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("policy"))) {
    return loom_low_target_legalize_parse_policy(value, context);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'target-legalize'",
                          (int)name.size, name.data);
}

iree_status_t loom_low_target_legalize_create(loom_pass_t* pass,
                                              iree_string_view_t options) {
  loom_low_target_legalize_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->max_errors = 20;
  state->max_iterations = LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS;
  state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;
  state->boundary_representation = LOOM_TARGET_BOUNDARY_REPRESENTATION_LOGICAL;
  state->policy = LOOM_TARGET_LEGALIZATION_POLICY_PREFER_NATIVE;

  loom_low_target_legalize_parse_context_t context = {
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("max-errors"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_errors(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_iterations(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_mode(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("boundaries"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_boundaries(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("policy"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_policy(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'target-legalize'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_low_target_legalize_parse_option,
                                    .user_data = &context,
                                }));
  }

  pass->state = state;
  return iree_ok_status();
}
