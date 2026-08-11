// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/legalize.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/rewrite/greedy.h"
#include "loom/target/math_policy.h"
#include "loom/target/pass_environment.h"
#include "loom/target/reporting/report.h"
#include "loom/transforms/math/patterns.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_option_def_t kMathLegalizeOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of worklist iterations.")},
};

#define LOOM_MATH_LEGALIZE_STATISTICS(V, statistics_type) \
  V(statistics_type, ops_rewritten, "ops-rewritten",      \
    "Number of math operations rewritten.")

LOOM_PASS_STATISTICS_DEFINE(loom_math_legalize_statistics,
                            loom_math_legalize_statistics_t,
                            LOOM_MATH_LEGALIZE_STATISTICS)

static const loom_pass_info_t loom_math_legalize_pass_info_storage = {
    .name = IREE_SVL("legalize-math"),
    .description = IREE_SVL("Rewrite semantic math ops to target-ready IR."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kMathLegalizeOptions,
    .option_count = IREE_ARRAYSIZE(kMathLegalizeOptions),
    .statistic_layout = &loom_math_legalize_statistics_layout,
};

const loom_pass_info_t* loom_math_legalize_pass_info(void) {
  return &loom_math_legalize_pass_info_storage;
}

typedef struct loom_math_legalize_options_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;
} loom_math_legalize_options_t;

typedef struct loom_math_legalize_state_t {
  // Current pass invocation.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Function-like op currently being rewritten.
  loom_func_like_t function;
  // Function target facts selected for |function|.
  const loom_target_facts_t* target_facts;
  // Target math policy selected for |function|, or NULL when unavailable.
  const loom_target_math_policy_t* policy;
  // Optional target compile report receiving math legalization rows.
  loom_target_compile_report_t* compile_report;
} loom_math_legalize_state_t;

#define LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(dialect, flag) \
  static_assert(LOOM_##dialect##_FASTMATHFLAGS_##flag ==       \
                    LOOM_TARGET_MATH_FASTMATH_FLAG_##flag,     \
                #dialect " fastmath flag " #flag " must match target math")

LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, REASSOC);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, NNAN);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, NINF);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, NSZ);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, ARCP);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, CONTRACT);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, AFN);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(SCALAR, FAST);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, REASSOC);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, NNAN);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, NINF);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, NSZ);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, ARCP);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, CONTRACT);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, AFN);
LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT(VECTOR, FAST);

#undef LOOM_MATH_LEGALIZE_FASTMATH_FLAG_ASSERT

static loom_target_math_fastmath_flags_t loom_math_legalize_fastmath_flags(
    uint8_t source_flags) {
  return (
      loom_target_math_fastmath_flags_t)(source_flags &
                                         LOOM_TARGET_MATH_FASTMATH_FLAG_FAST);
}

static iree_status_t loom_math_legalize_parse_option(void* user_data,
                                                     iree_string_view_t name,
                                                     iree_string_view_t value) {
  loom_math_legalize_options_t* options =
      (loom_math_legalize_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass 'legalize-math'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("legalize-math"), name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass 'legalize-math' option 'max-iterations' "
                              "must be greater than 0");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'legalize-math'",
                          (int)name.size, name.data);
}

iree_status_t loom_math_legalize_create(loom_pass_t* pass,
                                        iree_string_view_t options_string) {
  loom_math_legalize_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  memset(options, 0, sizeof(*options));
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        options->max_iterations = option->uint32_value;
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'legalize-math'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options_string,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_math_legalize_parse_option,
                                    .user_data = options,
                                }));
  }
  pass->state = options;
  return iree_ok_status();
}

static bool loom_math_legalize_scalar_result_query(
    const loom_module_t* module, const loom_op_t* op,
    loom_target_math_op_t math_op,
    loom_target_math_fastmath_flags_t fastmath_flags,
    loom_target_math_query_t* out_query) {
  const loom_value_id_t result = loom_op_results(op)[0];
  const loom_type_t result_type = loom_module_value_type(module, result);
  *out_query = (loom_target_math_query_t){
      .math_op = math_op,
      .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
      .value_type = result_type,
      .element_type = loom_type_element_type(result_type),
      .fastmath_flags = fastmath_flags,
  };
  return true;
}

static bool loom_math_legalize_vector_result_query(
    const loom_module_t* module, const loom_op_t* op,
    loom_target_math_op_t math_op,
    loom_target_math_fastmath_flags_t fastmath_flags,
    loom_target_math_query_t* out_query) {
  const loom_value_id_t result = loom_op_results(op)[0];
  const loom_type_t result_type = loom_module_value_type(module, result);
  *out_query = (loom_target_math_query_t){
      .math_op = math_op,
      .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
      .value_type = result_type,
      .element_type = loom_type_element_type(result_type),
      .fastmath_flags = fastmath_flags,
  };
  return true;
}

static loom_target_math_op_t loom_math_legalize_scalar_geluf_op(
    const loom_op_t* op) {
  switch (loom_scalar_geluf_variant(op)) {
    case LOOM_SCALAR_GELUF_VARIANT_ERF:
      return LOOM_TARGET_MATH_OP_GELUF_ERF;
    case LOOM_SCALAR_GELUF_VARIANT_TANH:
      return LOOM_TARGET_MATH_OP_GELUF_TANH;
    case LOOM_SCALAR_GELUF_VARIANT_LOGISTIC:
      return LOOM_TARGET_MATH_OP_GELUF_LOGISTIC;
    case LOOM_SCALAR_GELUF_VARIANT_COUNT_:
      break;
  }
  return LOOM_TARGET_MATH_OP_UNKNOWN;
}

static loom_target_math_op_t loom_math_legalize_vector_geluf_op(
    const loom_op_t* op) {
  switch (loom_vector_geluf_variant(op)) {
    case LOOM_VECTOR_GELUF_VARIANT_ERF:
      return LOOM_TARGET_MATH_OP_GELUF_ERF;
    case LOOM_VECTOR_GELUF_VARIANT_TANH:
      return LOOM_TARGET_MATH_OP_GELUF_TANH;
    case LOOM_VECTOR_GELUF_VARIANT_LOGISTIC:
      return LOOM_TARGET_MATH_OP_GELUF_LOGISTIC;
    case LOOM_VECTOR_GELUF_VARIANT_COUNT_:
      break;
  }
  return LOOM_TARGET_MATH_OP_UNKNOWN;
}

static loom_target_math_op_t loom_math_legalize_scalar_op_kind(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_EXPF:
      return LOOM_TARGET_MATH_OP_EXPF;
    case LOOM_OP_SCALAR_LOGF:
      return LOOM_TARGET_MATH_OP_LOGF;
    case LOOM_OP_SCALAR_LOG2F:
      return LOOM_TARGET_MATH_OP_LOG2F;
    case LOOM_OP_SCALAR_TANHF:
      return LOOM_TARGET_MATH_OP_TANHF;
    case LOOM_OP_SCALAR_POWF:
      return LOOM_TARGET_MATH_OP_POWF;
    case LOOM_OP_SCALAR_CEILF:
      return LOOM_TARGET_MATH_OP_CEILF;
    case LOOM_OP_SCALAR_FLOORF:
      return LOOM_TARGET_MATH_OP_FLOORF;
    case LOOM_OP_SCALAR_ROUNDF:
      return LOOM_TARGET_MATH_OP_ROUNDF;
    case LOOM_OP_SCALAR_ROUNDEVENF:
      return LOOM_TARGET_MATH_OP_ROUNDEVENF;
    case LOOM_OP_SCALAR_TRUNCF:
      return LOOM_TARGET_MATH_OP_TRUNCF;
    case LOOM_OP_SCALAR_SINF:
      return LOOM_TARGET_MATH_OP_SINF;
    case LOOM_OP_SCALAR_COSF:
      return LOOM_TARGET_MATH_OP_COSF;
    case LOOM_OP_SCALAR_SINTURNSF:
      return LOOM_TARGET_MATH_OP_SINTURNSF;
    case LOOM_OP_SCALAR_COSTURNSF:
      return LOOM_TARGET_MATH_OP_COSTURNSF;
    case LOOM_OP_SCALAR_ERFF:
      return LOOM_TARGET_MATH_OP_ERFF;
    case LOOM_OP_SCALAR_LOGISTICF:
      return LOOM_TARGET_MATH_OP_LOGISTICF;
    case LOOM_OP_SCALAR_SILUF:
      return LOOM_TARGET_MATH_OP_SILUF;
    case LOOM_OP_SCALAR_SOFTPLUSF:
      return LOOM_TARGET_MATH_OP_SOFTPLUSF;
    case LOOM_OP_SCALAR_GELUF:
      return loom_math_legalize_scalar_geluf_op(op);
    case LOOM_OP_SCALAR_ADDF:
      return LOOM_TARGET_MATH_OP_ADDF;
    case LOOM_OP_SCALAR_MULF:
      return LOOM_TARGET_MATH_OP_MULF;
    default:
      return LOOM_TARGET_MATH_OP_UNKNOWN;
  }
}

static loom_target_math_op_t loom_math_legalize_vector_op_kind(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_EXPF:
      return LOOM_TARGET_MATH_OP_EXPF;
    case LOOM_OP_VECTOR_LOGF:
      return LOOM_TARGET_MATH_OP_LOGF;
    case LOOM_OP_VECTOR_LOG2F:
      return LOOM_TARGET_MATH_OP_LOG2F;
    case LOOM_OP_VECTOR_TANHF:
      return LOOM_TARGET_MATH_OP_TANHF;
    case LOOM_OP_VECTOR_POWF:
      return LOOM_TARGET_MATH_OP_POWF;
    case LOOM_OP_VECTOR_CEILF:
      return LOOM_TARGET_MATH_OP_CEILF;
    case LOOM_OP_VECTOR_FLOORF:
      return LOOM_TARGET_MATH_OP_FLOORF;
    case LOOM_OP_VECTOR_ROUNDF:
      return LOOM_TARGET_MATH_OP_ROUNDF;
    case LOOM_OP_VECTOR_ROUNDEVENF:
      return LOOM_TARGET_MATH_OP_ROUNDEVENF;
    case LOOM_OP_VECTOR_TRUNCF:
      return LOOM_TARGET_MATH_OP_TRUNCF;
    case LOOM_OP_VECTOR_SINF:
      return LOOM_TARGET_MATH_OP_SINF;
    case LOOM_OP_VECTOR_COSF:
      return LOOM_TARGET_MATH_OP_COSF;
    case LOOM_OP_VECTOR_SINTURNSF:
      return LOOM_TARGET_MATH_OP_SINTURNSF;
    case LOOM_OP_VECTOR_COSTURNSF:
      return LOOM_TARGET_MATH_OP_COSTURNSF;
    case LOOM_OP_VECTOR_ERFF:
      return LOOM_TARGET_MATH_OP_ERFF;
    case LOOM_OP_VECTOR_LOGISTICF:
      return LOOM_TARGET_MATH_OP_LOGISTICF;
    case LOOM_OP_VECTOR_SILUF:
      return LOOM_TARGET_MATH_OP_SILUF;
    case LOOM_OP_VECTOR_SOFTPLUSF:
      return LOOM_TARGET_MATH_OP_SOFTPLUSF;
    case LOOM_OP_VECTOR_GELUF:
      return loom_math_legalize_vector_geluf_op(op);
    case LOOM_OP_VECTOR_ADDF:
      return LOOM_TARGET_MATH_OP_ADDF;
    case LOOM_OP_VECTOR_MULF:
      return LOOM_TARGET_MATH_OP_MULF;
    default:
      return LOOM_TARGET_MATH_OP_UNKNOWN;
  }
}

static bool loom_math_legalize_query_for_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_target_math_query_t* out_query) {
  loom_target_math_op_t math_op = loom_math_legalize_scalar_op_kind(op);
  if (math_op != LOOM_TARGET_MATH_OP_UNKNOWN) {
    return loom_math_legalize_scalar_result_query(
        module, op, math_op,
        loom_math_legalize_fastmath_flags(op->instance_flags), out_query);
  }

  math_op = loom_math_legalize_vector_op_kind(op);
  if (math_op != LOOM_TARGET_MATH_OP_UNKNOWN) {
    return loom_math_legalize_vector_result_query(
        module, op, math_op,
        loom_math_legalize_fastmath_flags(op->instance_flags), out_query);
  }

  return false;
}

static iree_string_view_t loom_math_legalize_function_name(
    const loom_math_legalize_state_t* state) {
  const loom_symbol_ref_t callee = loom_func_like_callee(state->function);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= state->module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol =
      &state->module->symbols.entries[callee.symbol_id];
  if (symbol->name_id >= state->module->strings.count) {
    return iree_string_view_empty();
  }
  return state->module->strings.entries[symbol->name_id];
}

static iree_status_t loom_math_legalize_record_report_row(
    loom_math_legalize_state_t* state, const loom_op_t* op,
    const loom_target_math_query_t* query,
    const loom_target_math_policy_decision_t* decision,
    loom_target_compile_report_math_action_t action, uint64_t created_op_count,
    uint64_t erased_op_count) {
  if (state->compile_report == NULL) {
    return iree_ok_status();
  }
  const loom_target_math_policy_decision_t empty_decision = {0};
  if (decision == NULL) {
    decision = &empty_decision;
  }
  const loom_target_bundle_storage_t* target_storage =
      &state->target_facts->storage;
  const loom_target_compile_report_math_row_t row = {
      .function_name = loom_math_legalize_function_name(state),
      .source_op_name = loom_op_name(state->module, op),
      .source_op_kind = op->kind,
      .target_bundle_name = target_storage->bundle.name,
      .target_config_name = target_storage->bundle.config
                                ? target_storage->bundle.config->name
                                : iree_string_view_empty(),
      .policy_name =
          state->policy ? state->policy->name : iree_string_view_empty(),
      .constraint_key = decision->constraint_key,
      .math_op = query->math_op,
      .lane_domain = query->lane_domain,
      .element_type = query->element_type,
      .action = action,
      .recipe = decision->recipe,
      .source_fastmath_flags = query->fastmath_flags,
      .recipe_fastmath_flags = decision->recipe_fastmath_flags,
      .created_op_count = created_op_count,
      .erased_op_count = erased_op_count,
  };
  return loom_target_compile_report_record_math_row(state->compile_report,
                                                    &row);
}

static iree_status_t loom_math_legalize_emit(
    loom_math_legalize_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_math_legalize_emit_missing_policy(
    loom_math_legalize_state_t* state, const loom_op_t* op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->module, op)),
      loom_param_string(state->pass->info->name),
      loom_param_string(state->target_facts->storage.config.contract_set_key),
  };
  return loom_math_legalize_emit(state, op, LOOM_ERR_LOWERING_034, params,
                                 IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_math_legalize_scalar_type_name(
    loom_scalar_type_t type) {
  const char* name = loom_scalar_type_name(type);
  return name ? iree_make_cstring_view(name) : IREE_SV("unknown");
}

static iree_status_t loom_math_legalize_emit_missing_recipe(
    loom_math_legalize_state_t* state, const loom_op_t* op,
    const loom_target_math_query_t* query,
    const loom_target_math_policy_decision_t* decision) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->module, op)),
      loom_param_string(state->pass->info->name),
      loom_param_string(state->policy->name),
      loom_param_string(loom_target_math_recipe_name(decision->recipe)),
      loom_param_string(loom_target_math_op_name(query->math_op)),
      loom_param_string(loom_target_math_lane_domain_name(query->lane_domain)),
      loom_param_string(
          loom_math_legalize_scalar_type_name(query->element_type)),
      loom_param_string(decision->constraint_key),
  };
  return loom_math_legalize_emit(state, op, LOOM_ERR_LOWERING_035, params,
                                 IREE_ARRAYSIZE(params));
}

static iree_status_t loom_math_legalize_emit_rejected(
    loom_math_legalize_state_t* state, const loom_op_t* op,
    const loom_target_math_query_t* query,
    const loom_target_math_policy_decision_t* decision) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->module, op)),
      loom_param_string(state->pass->info->name),
      loom_param_string(state->policy->name),
      loom_param_string(loom_target_math_op_name(query->math_op)),
      loom_param_string(loom_target_math_lane_domain_name(query->lane_domain)),
      loom_param_string(
          loom_math_legalize_scalar_type_name(query->element_type)),
      loom_param_string(decision->constraint_key),
  };
  return loom_math_legalize_emit(state, op, LOOM_ERR_LOWERING_036, params,
                                 IREE_ARRAYSIZE(params));
}

static bool loom_math_legalize_policy_action_is_known(
    loom_target_math_policy_action_t action) {
  switch (action) {
    case LOOM_TARGET_MATH_POLICY_ACTION_KEEP:
    case LOOM_TARGET_MATH_POLICY_ACTION_REWRITE:
    case LOOM_TARGET_MATH_POLICY_ACTION_REJECT:
      return true;
    case LOOM_TARGET_MATH_POLICY_ACTION_UNKNOWN:
      return false;
  }
  return false;
}

static iree_status_t loom_math_legalize_rewrite_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_math_legalize_state_t* state = (loom_math_legalize_state_t*)user_data;
  if (loom_pass_has_error_diagnostics(state->pass)) {
    return iree_ok_status();
  }

  loom_target_math_query_t query = {0};
  if (!loom_math_legalize_query_for_op(state->module, op, &query)) {
    return iree_ok_status();
  }

  if (state->policy == NULL) {
    IREE_RETURN_IF_ERROR(loom_math_legalize_record_report_row(
        state, op, &query, /*decision=*/NULL,
        LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_POLICY,
        /*created_op_count=*/0, /*erased_op_count=*/0));
    return loom_math_legalize_emit_missing_policy(state, op);
  }

  loom_target_math_policy_decision_t decision = {0};
  loom_target_math_policy_query(state->policy, &query, &decision);
  if (!loom_math_legalize_policy_action_is_known(decision.action)) {
    IREE_ASSERT_UNREACHABLE("target math policy returned unknown action");
    IREE_BUILTIN_UNREACHABLE();
  }

  if (decision.action == LOOM_TARGET_MATH_POLICY_ACTION_KEEP) {
    return iree_ok_status();
  }
  if (decision.action == LOOM_TARGET_MATH_POLICY_ACTION_REJECT) {
    IREE_RETURN_IF_ERROR(loom_math_legalize_record_report_row(
        state, op, &query, &decision,
        LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REJECTED,
        /*created_op_count=*/0, /*erased_op_count=*/0));
    return loom_math_legalize_emit_rejected(state, op, &query, &decision);
  }

  const loom_math_legalize_recipe_context_t context = {
      .pass = state->pass,
      .module = state->module,
      .query = query,
      .decision = decision,
  };
  driver->rewriter.flags = 0;
  const uint64_t created_op_count_before = driver->rewriter.created_op_count;
  const uint64_t erased_op_count_before = driver->rewriter.erased_op_count;
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_math_legalize_rewrite_recipe(
      &context, op, &driver->rewriter, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_math_legalize_record_report_row(
        state, op, &query, &decision,
        LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_RECIPE,
        /*created_op_count=*/0, /*erased_op_count=*/0));
    return loom_math_legalize_emit_missing_recipe(state, op, &query, &decision);
  }
  IREE_RETURN_IF_ERROR(loom_math_legalize_record_report_row(
      state, op, &query, &decision,
      LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN,
      driver->rewriter.created_op_count - created_op_count_before,
      driver->rewriter.erased_op_count - erased_op_count_before));
  loom_greedy_rewrite_result_record_rewriter_flags(result, &driver->rewriter);
  if (iree_any_bit_set(driver->rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_greedy_rewrite_result_record_change(
        result, &driver->rewriter,
        LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_math_legalize_run(loom_pass_t* pass, loom_module_t* module,
                                     loom_func_like_t function) {
  const loom_math_legalize_options_t* options =
      (const loom_math_legalize_options_t*)pass->state;
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }

  bool target_resolved = false;
  const loom_target_facts_t* target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_pass_resolve_function_facts(
      pass, module, function, &target_resolved, &target_facts));
  if (!target_resolved) {
    return iree_ok_status();
  }

  const loom_target_math_pass_capability_t* math_capability =
      loom_target_math_pass_capability_from_pass(pass);
  const loom_target_math_policy_registry_t* policy_registry =
      loom_target_math_pass_capability_policy_registry(math_capability);
  loom_target_compile_report_t* compile_report =
      loom_target_math_pass_capability_compile_report(math_capability);
  const loom_target_math_policy_t* policy =
      policy_registry ? loom_target_math_policy_registry_lookup_for_bundle(
                            policy_registry, &target_facts->storage.bundle)
                      : NULL;
  loom_math_legalize_state_t state = {
      .pass = pass,
      .module = module,
      .function = function,
      .target_facts = target_facts,
      .policy = policy,
      .compile_report = compile_report,
  };
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module, pass->arena, pass->value_facts,
                                        &driver);
  loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = options ? options->max_iterations : 0,
      .target_facts = target_facts,
  };
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .rewrite_op = loom_math_legalize_rewrite_op,
  };
  loom_greedy_rewrite_result_t result = {0};
  iree_status_t status = loom_greedy_rewrite_run_region(
      &driver, function, loom_func_like_body(function), function.op,
      &rewrite_options, &callbacks, &result);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (result.changed) {
    loom_pass_mark_changed(pass);
  }
  loom_math_legalize_statistics(pass)->ops_rewritten += result.ops_modified;
  return iree_ok_status();
}
