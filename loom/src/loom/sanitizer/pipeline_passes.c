// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/pipeline_passes.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/sanitizer/memory.h"
#include "loom/sanitizer/options.h"
#include "loom/sanitizer/site_location.h"
#include "loom/sanitizer/site_payload.h"

typedef struct loom_sanitizer_insert_assertions_state_t {
  // Check classes enabled for this pass invocation.
  loom_sanitizer_checks_t checks;
  // True when checks were supplied explicitly.
  bool has_checks_option;
} loom_sanitizer_insert_assertions_state_t;

#define LOOM_SANITIZER_INSERT_ASSERTIONS_STATISTICS(V, statistics_type)        \
  V(statistics_type, assume_ops_converted, "assume-ops-converted",             \
    "Number of assume ops replaced with sanitizer assertions or aliases.")     \
  V(statistics_type, access_assertions_inserted, "access-assertions-inserted", \
    "Number of sanitizer.assert.access ops inserted.")                         \
  V(statistics_type, value_assertions_inserted, "value-assertions-inserted",   \
    "Number of sanitizer.assert.value ops inserted.")                          \
  V(statistics_type, predicates_elided, "predicates-elided",                   \
    "Number of candidate predicates already proven before insertion.")         \
  V(statistics_type, fastmath_ops_instrumented, "fastmath-ops-instrumented",   \
    "Number of fast-math ops with input or result assertions inserted.")

LOOM_PASS_STATISTICS_DEFINE(loom_sanitizer_insert_assertions_statistics,
                            loom_sanitizer_insert_assertions_statistics_t,
                            LOOM_SANITIZER_INSERT_ASSERTIONS_STATISTICS)

static const loom_pass_option_def_t kSanitizerInsertAssertionsOptions[] = {
    {IREE_SVL("checks"),
     IREE_SVL("Sanitizer checks to insert: none, all, or a '|'-separated set "
              "of access, value, and operation.")},
};

static const loom_pass_info_t
    loom_sanitizer_insert_assertions_pass_info_storage = {
        .name = IREE_SVL("sanitizer-insert-assertions"),
        .description = IREE_SVL(
            "Insert semantic sanitizer assertions for enabled checks."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kSanitizerInsertAssertionsOptions,
        .option_count = IREE_ARRAYSIZE(kSanitizerInsertAssertionsOptions),
        .statistic_layout = &loom_sanitizer_insert_assertions_statistics_layout,
};

const loom_pass_info_t* loom_sanitizer_insert_assertions_pass_info(void) {
  return &loom_sanitizer_insert_assertions_pass_info_storage;
}

static iree_status_t loom_sanitizer_insert_assertions_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_sanitizer_insert_assertions_state_t* state =
      (loom_sanitizer_insert_assertions_state_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("checks"))) {
    if (state->has_checks_option) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'checks' for pass 'sanitizer-insert-assertions'");
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
        value, IREE_SV("sanitizer-insert-assertions option 'checks'"),
        &state->checks));
    state->has_checks_option = true;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'sanitizer-insert-assertions'",
      (int)name.size, name.data);
}

iree_status_t loom_sanitizer_insert_assertions_create(
    loom_pass_t* pass, iree_string_view_t options) {
  loom_sanitizer_insert_assertions_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->checks = LOOM_SANITIZER_CHECKS_KNOWN;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) continue;
      if (iree_string_view_equal(option->schema->name, IREE_SV("checks"))) {
        IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
            option->string_value,
            IREE_SV("sanitizer-insert-assertions option 'checks'"),
            &state->checks));
        state->has_checks_option = true;
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'sanitizer-insert-assertions'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_sanitizer_insert_assertions_parse_option,
            .user_data = state,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

static bool loom_sanitizer_checks_enabled(const loom_pass_t* pass,
                                          loom_sanitizer_checks_t checks) {
  const loom_sanitizer_insert_assertions_state_t* state =
      (const loom_sanitizer_insert_assertions_state_t*)pass->state;
  const loom_sanitizer_checks_t enabled =
      state ? state->checks : LOOM_SANITIZER_CHECKS_KNOWN;
  return iree_any_bit_set(enabled, checks);
}

static loom_sanitizer_check_kind_t loom_sanitizer_check_kind_for_predicate(
    loom_predicate_kind_t predicate_kind) {
  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_RELATION;
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
    case LOOM_PREDICATE_RANGE:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_RANGE;
    case LOOM_PREDICATE_MUL:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY;
    case LOOM_PREDICATE_POW2:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_POWER_OF_TWO;
    case LOOM_PREDICATE_NOT_NAN:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN;
    case LOOM_PREDICATE_NOT_INF:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_INF;
    case LOOM_PREDICATE_FINITE:
      return LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE;
    case LOOM_PREDICATE_COUNT_:
      return LOOM_SANITIZER_CHECK_KIND_UNKNOWN;
  }
  return LOOM_SANITIZER_CHECK_KIND_UNKNOWN;
}

static loom_sanitizer_check_kind_t loom_sanitizer_check_kind_for_predicates(
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  if (predicate_count == 0) return LOOM_SANITIZER_CHECK_KIND_UNKNOWN;
  loom_sanitizer_check_kind_t check_kind = LOOM_SANITIZER_CHECK_KIND_UNKNOWN;
  bool mixed_kinds = false;
  bool only_non_finite_parts = true;
  bool has_not_nan = false;
  bool has_not_inf = false;
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_kind_t predicate_kind = predicates[i].kind;
    if (predicate_kind == LOOM_PREDICATE_NOT_NAN) {
      has_not_nan = true;
    } else if (predicate_kind == LOOM_PREDICATE_NOT_INF) {
      has_not_inf = true;
    } else {
      only_non_finite_parts = false;
    }

    const loom_sanitizer_check_kind_t candidate_kind =
        loom_sanitizer_check_kind_for_predicate(predicate_kind);
    if (candidate_kind == LOOM_SANITIZER_CHECK_KIND_UNKNOWN) {
      return LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS;
    }
    if (check_kind == LOOM_SANITIZER_CHECK_KIND_UNKNOWN) {
      check_kind = candidate_kind;
    } else if (check_kind != candidate_kind) {
      mixed_kinds = true;
    }
  }
  if (mixed_kinds) {
    if (only_non_finite_parts && has_not_nan && has_not_inf) {
      return LOOM_SANITIZER_CHECK_KIND_VALUE_FINITE;
    }
    return LOOM_SANITIZER_CHECK_KIND_VALUE_CONSTRAINTS;
  }
  return check_kind;
}

static bool loom_sanitizer_find_value(loom_value_slice_t values,
                                      loom_value_id_t value_id,
                                      uint16_t* out_ordinal) {
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] != value_id) continue;
    if (out_ordinal) *out_ordinal = i;
    return true;
  }
  return false;
}

static bool loom_sanitizer_value_list_contains(const loom_value_id_t* values,
                                               uint16_t value_count,
                                               loom_value_id_t value_id,
                                               uint16_t* out_ordinal) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (values[i] != value_id) continue;
    if (out_ordinal) *out_ordinal = i;
    return true;
  }
  return false;
}

static iree_status_t loom_sanitizer_append_unique_value(
    loom_value_id_t* values, uint16_t capacity, loom_value_id_t value_id,
    uint16_t* inout_count, uint16_t* out_ordinal) {
  if (loom_sanitizer_value_list_contains(values, *inout_count, value_id,
                                         out_ordinal)) {
    return iree_ok_status();
  }
  if (*inout_count >= capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sanitizer assertion value list capacity exceeded");
  }
  uint16_t ordinal = *inout_count;
  values[ordinal] = value_id;
  *inout_count = ordinal + 1;
  if (out_ordinal) *out_ordinal = ordinal;
  return iree_ok_status();
}

static bool loom_sanitizer_type_accepts_integer_predicates(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_type_accepts_float_predicates(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_sanitizer_type_accepts_predicate(
    loom_type_t type, loom_predicate_kind_t predicate_kind) {
  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
      return loom_sanitizer_type_accepts_integer_predicates(type);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_sanitizer_type_accepts_float_predicates(type);
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_sanitizer_predicate_supported_for_values(
    const loom_module_t* module, loom_value_slice_t values,
    const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    if (predicate->args[i] < 0) return false;
    loom_value_id_t value_id = (loom_value_id_t)predicate->args[i];
    if (!loom_sanitizer_find_value(values, value_id, NULL)) return false;
    loom_type_t type = loom_module_value_type(module, value_id);
    if (!loom_sanitizer_type_accepts_predicate(type, predicate->kind)) {
      return false;
    }
  }
  return true;
}

static bool loom_sanitizer_predicate_arg_facts(
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_rewriter_t* rewriter, loom_value_slice_t values,
    loom_value_facts_t* out_facts) {
  if (argument_index >= predicate->arg_count) return false;
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      *out_facts = loom_value_facts_exact_i64(predicate->args[argument_index]);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      if (predicate->args[argument_index] < 0) return false;
      const loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (!loom_sanitizer_find_value(values, value_id, NULL)) {
        return false;
      }
      *out_facts = loom_rewriter_value_facts(rewriter, value_id);
      return true;
    }
    case LOOM_PRED_ARG_NONE:
    case LOOM_PRED_ARG_COUNT_:
      return false;
  }
  return false;
}

static bool loom_sanitizer_ranges_are_disjoint(loom_value_facts_t lhs,
                                               loom_value_facts_t rhs) {
  return lhs.range_hi < rhs.range_lo || rhs.range_hi < lhs.range_lo;
}

static bool loom_sanitizer_predicate_is_proven(
    const loom_predicate_t* predicate, loom_rewriter_t* rewriter,
    loom_value_slice_t values) {
  if (predicate->arg_count == 0 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE || predicate->args[0] < 0) {
    return false;
  }
  const loom_value_id_t target_value = (loom_value_id_t)predicate->args[0];
  if (!loom_sanitizer_find_value(values, target_value, NULL)) {
    return false;
  }
  loom_value_facts_t target_facts =
      loom_rewriter_value_facts(rewriter, target_value);

  loom_value_facts_t rhs_facts = {0};
  int64_t rhs_exact = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  switch (predicate->kind) {
    case LOOM_PREDICATE_EQ:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return loom_value_facts_is_exact(target_facts) &&
             loom_value_facts_is_exact(rhs_facts) &&
             target_facts.range_lo == rhs_facts.range_lo;
    case LOOM_PREDICATE_NE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      if (loom_value_facts_as_exact_i64(rhs_facts, &rhs_exact) &&
          rhs_exact == 0 && loom_value_facts_is_non_zero(target_facts)) {
        return true;
      }
      return loom_sanitizer_ranges_are_disjoint(target_facts, rhs_facts);
    case LOOM_PREDICATE_LT:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi < rhs_facts.range_lo;
    case LOOM_PREDICATE_LE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi <= rhs_facts.range_lo;
    case LOOM_PREDICATE_GT:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo > rhs_facts.range_hi;
    case LOOM_PREDICATE_GE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo >= rhs_facts.range_hi;
    case LOOM_PREDICATE_MUL:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &rhs_exact) ||
          rhs_exact == 0) {
        return false;
      }
      return loom_value_facts_divisible_by(target_facts, rhs_exact);
    case LOOM_PREDICATE_MIN:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo >= rhs_facts.range_hi;
    case LOOM_PREDICATE_MAX:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi <= rhs_facts.range_lo;
    case LOOM_PREDICATE_POW2:
      if (loom_value_facts_is_float(target_facts)) return false;
      return loom_value_facts_is_power_of_two(target_facts);
    case LOOM_PREDICATE_RANGE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &lower) ||
          !loom_sanitizer_predicate_arg_facts(predicate, 2, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &upper)) {
        return false;
      }
      return target_facts.range_lo >= lower && target_facts.range_hi <= upper;
    case LOOM_PREDICATE_NOT_NAN:
      return loom_value_facts_is_not_nan(target_facts);
    case LOOM_PREDICATE_NOT_INF:
      return loom_value_facts_is_not_inf(target_facts);
    case LOOM_PREDICATE_FINITE:
      return loom_value_facts_is_finite(target_facts) ||
             (loom_value_facts_is_not_nan(target_facts) &&
              loom_value_facts_is_not_inf(target_facts));
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_sanitizer_result_types_for_values(
    loom_module_t* module, loom_rewriter_t* rewriter, loom_value_slice_t values,
    loom_type_t** out_types) {
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, values.count, sizeof(*types), (void**)&types));
  for (uint16_t i = 0; i < values.count; ++i) {
    types[i] = loom_module_value_type(module, values.values[i]);
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_value_assertion(
    loom_module_t* module, loom_rewriter_t* rewriter, loom_value_slice_t values,
    const loom_predicate_t* predicates, uint16_t predicate_count,
    loom_sanitizer_provenance_kind_t provenance_kind,
    loom_location_id_t source_location, loom_op_t** out_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_result_types_for_values(
      module, rewriter, values, &result_types));
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_VALUE,
      .check_kind =
          loom_sanitizer_check_kind_for_predicates(predicates, predicate_count),
      .provenance_kind = provenance_kind,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_SCALAR,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  return loom_sanitizer_assert_value_build(
      &rewriter->builder, values.values, values.count, predicates,
      predicate_count, result_types, values.count, site_location, out_op);
}

static bool loom_sanitizer_value_slices_equal(loom_value_slice_t lhs,
                                              loom_value_slice_t rhs) {
  if (lhs.count != rhs.count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.values[i] != rhs.values[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_sanitizer_i64_arrays_equal(loom_attribute_t lhs,
                                            loom_attribute_t rhs) {
  if (loom_attr_is_absent(lhs) && loom_attr_is_absent(rhs)) {
    return true;
  }
  if (lhs.kind != LOOM_ATTR_I64_ARRAY || rhs.kind != LOOM_ATTR_I64_ARRAY ||
      lhs.count != rhs.count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.i64_array[i] != rhs.i64_array[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_sanitizer_preceded_by_matching_access_assertion(
    loom_op_t* op, loom_sanitizer_assert_access_kind_t kind,
    loom_value_id_t view, loom_value_slice_t indices,
    loom_attribute_t static_indices, loom_attribute_t static_extents) {
  loom_op_t* previous_op = op->prev_op;
  if (!previous_op || !loom_sanitizer_assert_access_isa(previous_op) ||
      loom_sanitizer_assert_access_kind(previous_op) != kind ||
      loom_sanitizer_assert_access_view(previous_op) != view) {
    return false;
  }
  return loom_sanitizer_value_slices_equal(
             loom_sanitizer_assert_access_indices(previous_op), indices) &&
         loom_sanitizer_i64_arrays_equal(
             loom_sanitizer_assert_access_static_indices(previous_op),
             static_indices) &&
         loom_sanitizer_i64_arrays_equal(
             loom_sanitizer_assert_access_static_extents(previous_op),
             static_extents);
}

static iree_status_t loom_sanitizer_emit_pass_diagnostic(
    loom_pass_t* pass, const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_sanitizer_emit_vector_access_unsupported(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
  };
  return loom_sanitizer_emit_pass_diagnostic(pass, op, LOOM_ERR_SUBRANGE_021,
                                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_vector_access_dynamic(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
  };
  return loom_sanitizer_emit_pass_diagnostic(pass, op, LOOM_ERR_SUBRANGE_019,
                                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_vector_access_rank(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* op,
    uint8_t vector_rank) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_i64(vector_rank),
  };
  return loom_sanitizer_emit_pass_diagnostic(pass, op, LOOM_ERR_SUBRANGE_020,
                                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_vector_static_extents(
    loom_pass_t* pass, loom_module_t* module,
    const loom_vector_memory_footprint_t* footprint,
    loom_attribute_t* out_static_extents) {
  *out_static_extents = loom_attr_absent();
  const uint8_t view_rank = footprint->vector_access.view_rank;
  int64_t* static_extents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(pass->arena, view_rank,
                                                 sizeof(*static_extents),
                                                 (void**)&static_extents));
  if (!loom_vector_memory_footprint_static_extents(footprint, static_extents,
                                                   view_rank)) {
    return loom_sanitizer_emit_vector_access_dynamic(pass, module,
                                                     footprint->access.op);
  }
  *out_static_extents = loom_attr_i64_array(static_extents, view_rank);
  return iree_ok_status();
}

static bool loom_sanitizer_access_kind_from_vector_footprint(
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t* out_kind) {
  const bool reads =
      iree_any_bit_set(footprint->flags, LOOM_VECTOR_MEMORY_FOOTPRINT_READS);
  const bool writes =
      iree_any_bit_set(footprint->flags, LOOM_VECTOR_MEMORY_FOOTPRINT_WRITES);
  if (reads && writes) {
    *out_kind = LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE;
    return true;
  } else if (writes) {
    *out_kind = LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE;
    return true;
  } else if (reads) {
    *out_kind = LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ;
    return true;
  }
  return false;
}

static iree_status_t loom_sanitizer_emit_vector_access_element_count(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* op,
    loom_type_t vector_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(pass->info->name),
      loom_param_type(vector_type),
  };
  return loom_sanitizer_emit_pass_diagnostic(pass, op, LOOM_ERR_SHAPE_007,
                                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_vector_static_lane_count(
    loom_pass_t* pass, loom_module_t* module,
    const loom_vector_memory_footprint_t* footprint, uint16_t* out_lane_count) {
  *out_lane_count = 0;
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(footprint->vector_type, &element_count)) {
    return loom_sanitizer_emit_vector_access_dynamic(pass, module,
                                                     footprint->access.op);
  }
  if (element_count > UINT16_MAX) {
    return loom_sanitizer_emit_vector_access_element_count(
        pass, module, footprint->access.op, footprint->vector_type);
  }
  *out_lane_count = (uint16_t)element_count;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_access_assertion(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_assert_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices,
    loom_attribute_t static_extents, loom_location_id_t source_location,
    loom_op_t** out_op) {
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  const loom_sanitizer_lane_policy_t lane_policy =
      loom_attr_is_absent(static_extents)
          ? LOOM_SANITIZER_LANE_POLICY_SCALAR
          : LOOM_SANITIZER_LANE_POLICY_ALL_LANES;
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_ACCESS,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = lane_policy,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  const loom_sanitizer_assert_access_build_flags_t build_flags =
      loom_attr_is_absent(static_extents)
          ? 0
          : LOOM_SANITIZER_ASSERT_ACCESS_BUILD_FLAG_HAS_STATIC_EXTENTS;
  return loom_sanitizer_assert_access_build(
      &rewriter->builder, build_flags, kind, view, indices.values,
      indices.count, static_indices.i64_array, static_indices.count,
      static_extents.i64_array, static_extents.count, site_location, out_op);
}

static loom_sanitizer_assert_accesses_kind_t
loom_sanitizer_repeated_access_kind(loom_sanitizer_assert_access_kind_t kind) {
  switch (kind) {
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ:
      return LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE:
      return LOOM_SANITIZER_ASSERT_ACCESSES_KIND_WRITE;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE:
      return LOOM_SANITIZER_ASSERT_ACCESSES_KIND_READ_WRITE;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_COUNT_:
      break;
  }
  return LOOM_SANITIZER_ASSERT_ACCESSES_KIND_COUNT_;
}

static iree_status_t loom_sanitizer_build_repeated_access_assertion(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_assert_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices,
    loom_attribute_t static_extents, loom_attribute_t static_strides,
    int64_t static_count, loom_location_id_t source_location,
    loom_op_t** out_op) {
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_ACCESS,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_ALL_LANES,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  return loom_sanitizer_assert_accesses_build(
      &rewriter->builder, loom_sanitizer_repeated_access_kind(kind), view,
      indices.values, indices.count, static_indices.i64_array,
      static_indices.count, static_extents.i64_array, static_extents.count,
      static_strides.i64_array, static_strides.count, static_count,
      site_location, out_op);
}

static iree_status_t loom_sanitizer_extract_vector_static_lane(
    loom_rewriter_t* rewriter, loom_value_id_t vector, loom_type_t vector_type,
    uint16_t lane_ordinal, loom_location_id_t location,
    loom_value_id_t* out_lane) {
  int64_t static_index = lane_ordinal;
  loom_type_t result_type =
      loom_type_scalar(loom_type_element_type(vector_type));
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, vector, NULL, 0, &static_index, 1, result_type,
      location, &extract_op));
  *out_lane = loom_vector_extract_result(extract_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_cast_to_index(loom_module_t* module,
                                                  loom_rewriter_t* rewriter,
                                                  loom_value_id_t value,
                                                  loom_location_id_t location,
                                                  loom_value_id_t* out_index) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t value_type = loom_module_value_type(module, value);
  if (loom_type_equal(value_type, index_type)) {
    *out_index = value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &rewriter->builder, value, value_type, index_type, location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_index_constant(
    loom_rewriter_t* rewriter, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_index) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &rewriter->builder, loom_attr_i64(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &constant_op));
  *out_index = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_index_add(
    loom_rewriter_t* rewriter, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_location_id_t location, loom_value_id_t* out_index) {
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      &rewriter->builder, lhs, rhs, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      location, &add_op));
  *out_index = loom_index_add_result(add_op);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_vector_atomic_lane_index(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint, uint16_t lane_ordinal,
    loom_value_slice_t* out_indices, loom_attribute_t* out_static_indices) {
  *out_indices = (loom_value_slice_t){0};
  *out_static_indices = loom_attr_absent();
  const uint8_t view_rank = footprint->vector_access.view_rank;
  if (footprint->static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      footprint->static_indices.count != view_rank ||
      footprint->offsets == LOOM_VALUE_ID_INVALID) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }
  uint16_t expected_dynamic_count = 0;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    if (footprint->static_indices.i64_array[axis] == INT64_MIN) {
      ++expected_dynamic_count;
    }
  }
  if (expected_dynamic_count != footprint->dynamic_indices.count) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }

  loom_type_t offset_type = loom_module_value_type(module, footprint->offsets);
  loom_value_id_t offset_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_extract_vector_static_lane(
      rewriter, footprint->offsets, offset_type, lane_ordinal,
      footprint->access.op->location, &offset_lane));
  loom_value_id_t offset_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_cast_to_index(
      module, rewriter, offset_lane, footprint->access.op->location,
      &offset_index));

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*static_indices),
                                                 (void**)&static_indices));
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*dynamic_indices),
                                                 (void**)&dynamic_indices));

  uint16_t source_dynamic_ordinal = 0;
  uint16_t dynamic_count = 0;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    const int64_t origin = footprint->static_indices.i64_array[axis];
    if (axis + 1 < view_rank) {
      static_indices[axis] = origin;
      if (origin == INT64_MIN) {
        dynamic_indices[dynamic_count++] =
            footprint->dynamic_indices.values[source_dynamic_ordinal++];
      }
      continue;
    }

    static_indices[axis] = INT64_MIN;
    loom_value_id_t base_index = LOOM_VALUE_ID_INVALID;
    if (origin == INT64_MIN) {
      base_index = footprint->dynamic_indices.values[source_dynamic_ordinal++];
    } else if (origin != 0) {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_index_constant(
          rewriter, origin, footprint->access.op->location, &base_index));
    }

    loom_value_id_t lane_index = offset_index;
    if (base_index != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_sanitizer_build_index_add(
          rewriter, base_index, offset_index, footprint->access.op->location,
          &lane_index));
    }
    dynamic_indices[dynamic_count++] = lane_index;
  }

  *out_indices = (loom_value_slice_t){
      .values = dynamic_indices,
      .count = dynamic_count,
  };
  *out_static_indices = loom_attr_i64_array(static_indices, view_rank);
  return iree_ok_status();
}

static bool loom_sanitizer_value_exact_positive_i64(loom_rewriter_t* rewriter,
                                                    loom_value_id_t value,
                                                    int64_t* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value);
  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value) || exact_value <= 0) {
    return false;
  }
  *out_value = exact_value;
  return true;
}

static iree_status_t loom_sanitizer_insert_access_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_assert_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices,
    loom_attribute_t static_extents, loom_location_id_t source_location) {
  loom_op_t* assert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_access_assertion(
      module, rewriter, kind, view, indices, static_indices, static_extents,
      source_location, &assert_op));
  (void)assert_op;
  loom_sanitizer_insert_assertions_statistics_t* statistics =
      loom_sanitizer_insert_assertions_statistics(pass);
  ++statistics->access_assertions_inserted;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_build_fragment_access_indices(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    const int64_t fragment_ordinals[3], loom_value_slice_t* out_indices,
    loom_attribute_t* out_static_indices) {
  *out_indices = (loom_value_slice_t){0};
  *out_static_indices = loom_attr_absent();
  const uint8_t view_rank = loom_type_rank(footprint->view_type);
  const uint8_t fragment_rank = loom_type_rank(footprint->vector_type);
  if (fragment_rank < 2 || fragment_rank > 3 || view_rank < fragment_rank ||
      footprint->static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      footprint->static_indices.count != view_rank) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*static_indices),
                                                 (void**)&static_indices));
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*dynamic_indices),
                                                 (void**)&dynamic_indices));

  const uint8_t first_fragment_axis = view_rank - fragment_rank;
  uint16_t source_dynamic_ordinal = 0;
  uint16_t dynamic_count = 0;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    const int64_t origin = footprint->static_indices.i64_array[axis];
    if (axis >= first_fragment_axis) {
      const int64_t axis_ordinal =
          fragment_ordinals[axis - first_fragment_axis];
      if (origin == INT64_MIN) {
        if (source_dynamic_ordinal >= footprint->dynamic_indices.count) {
          return loom_sanitizer_emit_vector_access_unsupported(
              pass, module, footprint->access.op);
        }
        loom_value_id_t index =
            footprint->dynamic_indices.values[source_dynamic_ordinal++];
        if (axis_ordinal != 0) {
          loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_sanitizer_build_index_constant(
              rewriter, axis_ordinal, footprint->access.op->location, &offset));
          IREE_RETURN_IF_ERROR(loom_sanitizer_build_index_add(
              rewriter, index, offset, footprint->access.op->location, &index));
        }
        static_indices[axis] = INT64_MIN;
        dynamic_indices[dynamic_count++] = index;
      } else {
        int64_t static_index = 0;
        if (!iree_checked_add_i64(origin, axis_ordinal, &static_index)) {
          return loom_sanitizer_emit_vector_access_unsupported(
              pass, module, footprint->access.op);
        }
        static_indices[axis] = static_index;
      }
      continue;
    }

    static_indices[axis] = origin;
    if (origin == INT64_MIN) {
      if (source_dynamic_ordinal >= footprint->dynamic_indices.count) {
        return loom_sanitizer_emit_vector_access_unsupported(
            pass, module, footprint->access.op);
      }
      dynamic_indices[dynamic_count++] =
          footprint->dynamic_indices.values[source_dynamic_ordinal++];
    }
  }
  if (source_dynamic_ordinal != footprint->dynamic_indices.count) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }

  *out_indices = (loom_value_slice_t){
      .values = dynamic_indices,
      .count = dynamic_count,
  };
  *out_static_indices = loom_attr_i64_array(static_indices, view_rank);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_insert_fragment_axis_assertions(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind, uint8_t contiguous_axis,
    uint8_t repeated_axis, const int64_t fragment_counts[3],
    const int64_t fragment_ordinals[3]) {
  const uint8_t view_rank = loom_type_rank(footprint->view_type);
  const uint8_t fragment_rank = loom_type_rank(footprint->vector_type);
  const uint8_t first_fragment_axis = view_rank - fragment_rank;
  int64_t* static_extents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*static_extents),
                                                 (void**)&static_extents));
  int64_t* static_strides = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, view_rank,
                                                 sizeof(*static_strides),
                                                 (void**)&static_strides));
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    const uint8_t fragment_axis = axis >= first_fragment_axis
                                      ? (uint8_t)(axis - first_fragment_axis)
                                      : UINT8_MAX;
    static_extents[axis] =
        fragment_axis == contiguous_axis ? fragment_counts[fragment_axis] : 1;
    static_strides[axis] = fragment_axis == repeated_axis ? 1 : 0;
  }

  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_fragment_access_indices(
      pass, module, rewriter, footprint, fragment_ordinals, &indices,
      &static_indices));
  loom_op_t* assertion_op = NULL;
  const int64_t repeated_count =
      repeated_axis != UINT8_MAX ? fragment_counts[repeated_axis] : 1;
  return loom_sanitizer_build_repeated_access_assertion(
      module, rewriter, kind, footprint->view, indices, static_indices,
      loom_attr_i64_array(static_extents, view_rank),
      loom_attr_i64_array(static_strides, view_rank), repeated_count,
      footprint->access.op->location, &assertion_op);
}

static iree_status_t loom_sanitizer_insert_fragment_scalar_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind,
    const int64_t fragment_ordinals[3]) {
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_fragment_access_indices(
      pass, module, rewriter, footprint, fragment_ordinals, &indices,
      &static_indices));
  return loom_sanitizer_insert_access_assertion(
      pass, module, rewriter, kind, footprint->view, indices, static_indices,
      loom_attr_absent(), footprint->access.op->location);
}

static bool loom_sanitizer_fragment_axis_static_stride(
    const loom_vector_memory_footprint_t* footprint, uint8_t fragment_axis,
    int64_t* out_stride) {
  *out_stride = 0;
  const uint8_t view_rank = footprint->vector_access.view_rank;
  const uint8_t fragment_rank = loom_type_rank(footprint->vector_type);
  if (fragment_rank < 2 || fragment_rank > 3 || view_rank < fragment_rank ||
      fragment_axis >= fragment_rank) {
    return false;
  }
  const uint8_t view_axis =
      (uint8_t)(view_rank - fragment_rank + fragment_axis);
  return loom_vector_memory_access_static_axis_stride(&footprint->vector_access,
                                                      view_axis, out_stride) &&
         *out_stride > 0;
}

static bool loom_sanitizer_fragment_axis_count(loom_rewriter_t* rewriter,
                                               loom_type_t vector_type,
                                               uint8_t axis,
                                               int64_t* out_count) {
  *out_count = 0;
  if (axis >= loom_type_rank(vector_type)) return false;
  if (!loom_type_dim_is_dynamic_at(vector_type, axis)) {
    *out_count = loom_type_dim_static_size_at(vector_type, axis);
    return *out_count > 0;
  }
  return loom_sanitizer_value_exact_positive_i64(
      rewriter, loom_type_dim_value_id_at(vector_type, axis), out_count);
}

static iree_status_t loom_sanitizer_try_instrument_fragment_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind) {
  const uint8_t fragment_rank = loom_type_rank(footprint->vector_type);
  if (fragment_rank < 2 || fragment_rank > 3 ||
      loom_type_rank(footprint->view_type) < fragment_rank) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }

  int64_t fragment_counts[3] = {1, 1, 1};
  int64_t element_count = 0;
  for (uint8_t axis = 0; axis < fragment_rank; ++axis) {
    if (!loom_sanitizer_fragment_axis_count(rewriter, footprint->vector_type,
                                            axis, &fragment_counts[axis])) {
      return loom_sanitizer_emit_vector_access_dynamic(pass, module,
                                                       footprint->access.op);
    }
    if (axis == 0) {
      element_count = fragment_counts[axis];
    } else if (!iree_checked_mul_i64(element_count, fragment_counts[axis],
                                     &element_count)) {
      return loom_sanitizer_emit_vector_access_unsupported(
          pass, module, footprint->access.op);
    }
  }
  if (element_count > UINT16_MAX) {
    return loom_sanitizer_emit_vector_access_unsupported(pass, module,
                                                         footprint->access.op);
  }

  uint8_t contiguous_axis = UINT8_MAX;
  uint8_t repeated_axis = UINT8_MAX;
  int64_t repeated_axis_count = 0;
  for (uint8_t axis = 0; axis < fragment_rank; ++axis) {
    int64_t axis_stride = 0;
    if (!loom_sanitizer_fragment_axis_static_stride(footprint, axis,
                                                    &axis_stride)) {
      continue;
    }
    if (axis_stride == 1 && contiguous_axis == UINT8_MAX) {
      contiguous_axis = axis;
    } else if (fragment_counts[axis] > repeated_axis_count) {
      repeated_axis = axis;
      repeated_axis_count = fragment_counts[axis];
    }
  }
  if (repeated_axis == contiguous_axis) repeated_axis = UINT8_MAX;

  int64_t outer_count = 1;
  for (uint8_t axis = 0; axis < fragment_rank; ++axis) {
    if (axis != contiguous_axis && axis != repeated_axis &&
        !iree_checked_mul_i64(outer_count, fragment_counts[axis],
                              &outer_count)) {
      return loom_sanitizer_emit_vector_access_unsupported(
          pass, module, footprint->access.op);
    }
  }
  loom_builder_set_before(&rewriter->builder, footprint->access.op);
  for (int64_t outer_ordinal = 0; outer_ordinal < outer_count;
       ++outer_ordinal) {
    int64_t fragment_ordinals[3] = {0, 0, 0};
    int64_t remaining_ordinal = outer_ordinal;
    for (uint8_t reverse_axis = fragment_rank; reverse_axis > 0;
         --reverse_axis) {
      const uint8_t axis = reverse_axis - 1;
      if (axis != contiguous_axis && axis != repeated_axis) {
        fragment_ordinals[axis] = remaining_ordinal % fragment_counts[axis];
        remaining_ordinal /= fragment_counts[axis];
      }
    }
    if (contiguous_axis != UINT8_MAX || repeated_axis != UINT8_MAX) {
      IREE_RETURN_IF_ERROR(loom_sanitizer_insert_fragment_axis_assertions(
          pass, module, rewriter, footprint, kind, contiguous_axis,
          repeated_axis, fragment_counts, fragment_ordinals));
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_insert_fragment_scalar_assertion(
          pass, module, rewriter, footprint, kind, fragment_ordinals));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_insert_vector_atomic_lane_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind, uint16_t lane_ordinal) {
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_vector_atomic_lane_index(
      pass, module, rewriter, footprint, lane_ordinal, &indices,
      &static_indices));
  return loom_sanitizer_insert_access_assertion(
      pass, module, rewriter, kind, footprint->view, indices, static_indices,
      loom_attr_absent(), footprint->access.op->location);
}

static iree_status_t loom_sanitizer_insert_masked_vector_atomic_lane_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind, uint16_t lane_ordinal) {
  loom_type_t mask_type = loom_module_value_type(module, footprint->mask);
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_sanitizer_extract_vector_static_lane(
      rewriter, footprint->mask, mask_type, lane_ordinal,
      footprint->access.op->location, &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition,
      NULL, 0, NULL, 0, footprint->access.op->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  IREE_RETURN_IF_ERROR(loom_sanitizer_insert_vector_atomic_lane_assertion(
      pass, module, rewriter, footprint, kind, lane_ordinal));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder, NULL, 0,
                                            footprint->access.op->location,
                                            &then_yield));
  loom_builder_restore(&rewriter->builder, saved);

  saved = loom_builder_enter_region(&rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder, NULL, 0,
                                            footprint->access.op->location,
                                            &else_yield));
  loom_builder_restore(&rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_vector_atomic_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_vector_memory_footprint_t* footprint,
    loom_sanitizer_assert_access_kind_t kind) {
  if (footprint->vector_access.vector_rank != 1) {
    return loom_sanitizer_emit_vector_access_rank(
        pass, module, footprint->access.op,
        footprint->vector_access.vector_rank);
  }
  uint16_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_sanitizer_vector_static_lane_count(
      pass, module, footprint, &lane_count));
  loom_builder_set_before(&rewriter->builder, footprint->access.op);
  const bool has_mask =
      footprint->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE;
  for (uint16_t lane_ordinal = 0; lane_ordinal < lane_count; ++lane_ordinal) {
    if (has_mask) {
      IREE_RETURN_IF_ERROR(
          loom_sanitizer_insert_masked_vector_atomic_lane_assertion(
              pass, module, rewriter, footprint, kind, lane_ordinal));
    } else {
      IREE_RETURN_IF_ERROR(loom_sanitizer_insert_vector_atomic_lane_assertion(
          pass, module, rewriter, footprint, kind, lane_ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  loom_sanitizer_assert_access_kind_t kind = 0;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  loom_attribute_t static_extents = loom_attr_absent();
  if (loom_view_load_isa(op)) {
    kind = LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ;
    view = loom_view_load_view(op);
    indices = loom_view_load_indices(op);
    static_indices = loom_view_load_static_indices(op);
  } else if (loom_view_store_isa(op)) {
    kind = LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE;
    view = loom_view_store_view(op);
    indices = loom_view_store_indices(op);
    static_indices = loom_view_store_static_indices(op);
  } else {
    loom_vector_memory_footprint_t footprint = {0};
    const loom_fact_context_t* fact_context =
        rewriter->fact_table ? &rewriter->fact_table->context : NULL;
    if (!loom_vector_memory_footprint_describe(fact_context, module, op,
                                               &footprint)) {
      return iree_ok_status();
    }
    if (!loom_sanitizer_access_view_is_observable(rewriter, footprint.view)) {
      return iree_ok_status();
    }
    if (!loom_sanitizer_access_kind_from_vector_footprint(&footprint, &kind)) {
      return loom_sanitizer_emit_vector_access_unsupported(pass, module, op);
    }
    switch (footprint.kind) {
      case LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE:
        break;
      case LOOM_VECTOR_MEMORY_FOOTPRINT_ATOMIC_PER_LANE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE:
        return loom_sanitizer_try_instrument_vector_atomic_access_op(
            pass, module, rewriter, &footprint, kind);
      case LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT:
        return loom_sanitizer_try_instrument_fragment_access_op(
            pass, module, rewriter, &footprint, kind);
      case LOOM_VECTOR_MEMORY_FOOTPRINT_NONE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_DENSE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_COMPRESS_EXPAND:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_PER_LANE_OFFSET:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_PER_LANE_OFFSET:
        return loom_sanitizer_emit_vector_access_unsupported(pass, module, op);
    }
    if (footprint.vector_access.vector_rank != 1) {
      return loom_sanitizer_emit_vector_access_rank(
          pass, module, op, footprint.vector_access.vector_rank);
    }
    view = footprint.view;
    indices = footprint.dynamic_indices;
    static_indices = footprint.static_indices;
    IREE_RETURN_IF_ERROR(loom_sanitizer_vector_static_extents(
        pass, module, &footprint, &static_extents));
  }
  if (!loom_sanitizer_access_view_is_observable(rewriter, view)) {
    return iree_ok_status();
  }
  if (loom_sanitizer_preceded_by_matching_access_assertion(
          op, kind, view, indices, static_indices, static_extents)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  return loom_sanitizer_insert_access_assertion(pass, module, rewriter, kind,
                                                view, indices, static_indices,
                                                static_extents, op->location);
}

static iree_status_t loom_sanitizer_filter_predicates(
    loom_module_t* module, loom_rewriter_t* rewriter, loom_value_slice_t values,
    loom_attribute_t predicates, loom_predicate_t** out_predicates,
    uint16_t* out_predicate_count, uint16_t* out_elided_count,
    bool* out_all_supported) {
  *out_predicates = NULL;
  *out_predicate_count = 0;
  *out_elided_count = 0;
  *out_all_supported = true;
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST || predicates.count == 0) {
    return iree_ok_status();
  }
  loom_predicate_t* filtered_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, predicates.count, sizeof(*filtered_predicates),
      (void**)&filtered_predicates));
  uint16_t filtered_count = 0;
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (!loom_sanitizer_predicate_supported_for_values(module, values,
                                                       predicate)) {
      *out_all_supported = false;
      continue;
    }
    if (loom_sanitizer_predicate_is_proven(predicate, rewriter, values)) {
      ++*out_elided_count;
      continue;
    }
    filtered_predicates[filtered_count++] = *predicate;
  }
  *out_predicates = filtered_predicates;
  *out_predicate_count = filtered_count;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_collect_assume_assert_values(
    loom_rewriter_t* rewriter, loom_value_slice_t assume_values,
    loom_attribute_t predicates, loom_value_slice_t* out_values) {
  uint32_t maximum_value_count = assume_values.count;
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST) {
    maximum_value_count += predicates.count * 3u;
  }
  if (maximum_value_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sanitizer assertion value list is too large");
  }
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, maximum_value_count, sizeof(*values), (void**)&values));
  uint16_t value_count = 0;
  for (uint16_t i = 0; i < assume_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_sanitizer_append_unique_value(
        values, (uint16_t)maximum_value_count, assume_values.values[i],
        &value_count, NULL));
  }
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST) {
    for (uint16_t predicate_index = 0; predicate_index < predicates.count;
         ++predicate_index) {
      const loom_predicate_t* predicate =
          &predicates.predicate_list[predicate_index];
      for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
           ++argument_index) {
        if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE ||
            predicate->args[argument_index] < 0) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_sanitizer_append_unique_value(
            values, (uint16_t)maximum_value_count,
            (loom_value_id_t)predicate->args[argument_index], &value_count,
            NULL));
      }
    }
  }
  *out_values = (loom_value_slice_t){
      .values = values,
      .count = value_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_replace_assume(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op, loom_value_slice_t assume_values,
    loom_value_slice_t assume_results, loom_attribute_t predicates) {
  if (assume_values.count != assume_results.count) {
    return iree_ok_status();
  }
  loom_value_slice_t assert_values = {0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_collect_assume_assert_values(
      rewriter, assume_values, predicates, &assert_values));

  loom_predicate_t* filtered_predicates = NULL;
  uint16_t filtered_count = 0;
  uint16_t elided_count = 0;
  bool all_predicates_supported = true;
  IREE_RETURN_IF_ERROR(loom_sanitizer_filter_predicates(
      module, rewriter, assert_values, predicates, &filtered_predicates,
      &filtered_count, &elided_count, &all_predicates_supported));
  if (!all_predicates_supported) return iree_ok_status();
  loom_sanitizer_insert_assertions_statistics_t* statistics =
      loom_sanitizer_insert_assertions_statistics(pass);
  statistics->predicates_elided += elided_count;
  if (filtered_count == 0) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, assume_values.values, assume_values.count));
    ++statistics->assume_ops_converted;
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* assert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_value_assertion(
      module, rewriter, assert_values, filtered_predicates, filtered_count,
      LOOM_SANITIZER_PROVENANCE_KIND_ASSUME, op->location, &assert_op));
  loom_value_slice_t assert_results =
      loom_sanitizer_assert_value_results(assert_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, assert_results.values, assume_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, assert_results.values, assume_results.count));
  ++statistics->assume_ops_converted;
  ++statistics->value_assertions_inserted;
  return iree_ok_status();
}

static uint8_t loom_sanitizer_scalar_fastmath_flags(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_REMF:
    case LOOM_OP_SCALAR_NEGF:
    case LOOM_OP_SCALAR_ABSF:
    case LOOM_OP_SCALAR_MINIMUMF:
    case LOOM_OP_SCALAR_MAXIMUMF:
    case LOOM_OP_SCALAR_MINNUMF:
    case LOOM_OP_SCALAR_MAXNUMF:
    case LOOM_OP_SCALAR_CLAMPF:
    case LOOM_OP_SCALAR_COPYSIGNF:
    case LOOM_OP_SCALAR_EXPF:
    case LOOM_OP_SCALAR_EXP2F:
    case LOOM_OP_SCALAR_EXPM1F:
    case LOOM_OP_SCALAR_LOGF:
    case LOOM_OP_SCALAR_LOG2F:
    case LOOM_OP_SCALAR_LOG10F:
    case LOOM_OP_SCALAR_LOG1PF:
    case LOOM_OP_SCALAR_POWF:
    case LOOM_OP_SCALAR_SQRTF:
    case LOOM_OP_SCALAR_RSQRTF:
    case LOOM_OP_SCALAR_CBRTF:
    case LOOM_OP_SCALAR_SINF:
    case LOOM_OP_SCALAR_COSF:
    case LOOM_OP_SCALAR_SINTURNSF:
    case LOOM_OP_SCALAR_COSTURNSF:
    case LOOM_OP_SCALAR_TANF:
    case LOOM_OP_SCALAR_ASINF:
    case LOOM_OP_SCALAR_ACOSF:
    case LOOM_OP_SCALAR_ATANF:
    case LOOM_OP_SCALAR_ATAN2F:
    case LOOM_OP_SCALAR_SINHF:
    case LOOM_OP_SCALAR_COSHF:
    case LOOM_OP_SCALAR_TANHF:
    case LOOM_OP_SCALAR_ASINHF:
    case LOOM_OP_SCALAR_ACOSHF:
    case LOOM_OP_SCALAR_ATANHF:
    case LOOM_OP_SCALAR_ERFF:
    case LOOM_OP_SCALAR_ERFCF:
    case LOOM_OP_SCALAR_LOGISTICF:
    case LOOM_OP_SCALAR_SILUF:
    case LOOM_OP_SCALAR_SOFTPLUSF:
    case LOOM_OP_SCALAR_GELUF:
    case LOOM_OP_SCALAR_FMAF:
    case LOOM_OP_SCALAR_CEILF:
    case LOOM_OP_SCALAR_FLOORF:
    case LOOM_OP_SCALAR_ROUNDF:
    case LOOM_OP_SCALAR_ROUNDEVENF:
    case LOOM_OP_SCALAR_TRUNCF:
    case LOOM_OP_SCALAR_CMPF:
      return op->instance_flags;
    default:
      return 0;
  }
}

static bool loom_sanitizer_fastmath_predicate_kind(
    uint8_t fastmath_flags, loom_predicate_kind_t* out_kind) {
  const bool no_nans =
      iree_any_bit_set(fastmath_flags, LOOM_SCALAR_FASTMATHFLAGS_NNAN);
  const bool no_infinities =
      iree_any_bit_set(fastmath_flags, LOOM_SCALAR_FASTMATHFLAGS_NINF);
  if (no_nans && no_infinities) {
    *out_kind = LOOM_PREDICATE_FINITE;
    return true;
  }
  if (no_nans) {
    *out_kind = LOOM_PREDICATE_NOT_NAN;
    return true;
  }
  if (no_infinities) {
    *out_kind = LOOM_PREDICATE_NOT_INF;
    return true;
  }
  return false;
}

static loom_predicate_t loom_sanitizer_make_unary_predicate(
    loom_predicate_kind_t kind, loom_value_id_t value) {
  return (loom_predicate_t){
      .kind = kind,
      .arg_count = 1,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE, LOOM_PRED_ARG_NONE},
      .args = {value, 0, 0},
  };
}

static bool loom_sanitizer_value_is_scalar_float(loom_module_t* module,
                                                 loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_sanitizer_assertion_has_predicate_for_value(
    loom_op_t* assert_op, loom_value_id_t value_id,
    loom_predicate_kind_t predicate_kind) {
  if (!loom_sanitizer_assert_value_isa(assert_op)) return false;
  loom_attribute_t predicates =
      loom_sanitizer_assert_value_predicates(assert_op);
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST) return false;
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (predicate->kind != predicate_kind || predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0) {
      continue;
    }
    if ((loom_value_id_t)predicate->args[0] == value_id) return true;
  }
  return false;
}

static bool loom_sanitizer_result_already_asserted(
    loom_module_t* module, loom_value_id_t value_id,
    loom_predicate_kind_t predicate_kind) {
  loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (loom_sanitizer_assertion_has_predicate_for_value(user_op, value_id,
                                                         predicate_kind)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_sanitizer_build_fastmath_value_assertion(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_value_slice_t values, loom_predicate_kind_t predicate_kind,
    loom_location_id_t location, loom_op_t** out_op) {
  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, values.count, sizeof(*predicates), (void**)&predicates));
  uint16_t predicate_count = 0;
  uint16_t elided_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    loom_predicate_t predicate =
        loom_sanitizer_make_unary_predicate(predicate_kind, values.values[i]);
    if (loom_sanitizer_predicate_is_proven(&predicate, rewriter, values)) {
      ++elided_count;
      continue;
    }
    predicates[predicate_count++] = predicate;
  }
  loom_sanitizer_insert_assertions_statistics_t* statistics =
      loom_sanitizer_insert_assertions_statistics(pass);
  statistics->predicates_elided += elided_count;
  if (predicate_count == 0) {
    *out_op = NULL;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_value_assertion(
      module, rewriter, values, predicates, predicate_count,
      LOOM_SANITIZER_PROVENANCE_KIND_FAST_MATH, location, out_op));
  ++statistics->value_assertions_inserted;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_instrument_fastmath_inputs(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op, loom_predicate_kind_t predicate_kind, bool* out_changed) {
  *out_changed = false;
  if (op->operand_count == 0) return iree_ok_status();
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->operand_count, sizeof(*values), (void**)&values));
  uint16_t value_count = 0;
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (!loom_sanitizer_value_is_scalar_float(module, operands[i])) continue;
    IREE_RETURN_IF_ERROR(loom_sanitizer_append_unique_value(
        values, op->operand_count, operands[i], &value_count, NULL));
  }
  if (value_count == 0) return iree_ok_status();

  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* assert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_fastmath_value_assertion(
      pass, module, rewriter,
      (loom_value_slice_t){
          .values = values,
          .count = value_count,
      },
      predicate_kind, op->location, &assert_op));
  if (!assert_op) return iree_ok_status();

  loom_value_slice_t checked_values =
      loom_sanitizer_assert_value_results(assert_op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    uint16_t value_ordinal = 0;
    if (!loom_sanitizer_value_list_contains(
            values, value_count, operands[operand_index], &value_ordinal)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, op, operand_index, checked_values.values[value_ordinal]));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_instrument_fastmath_results(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op, loom_predicate_kind_t predicate_kind, bool* out_changed) {
  *out_changed = false;
  if (op->result_count == 0) return iree_ok_status();
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*values), (void**)&values));
  uint16_t value_count = 0;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!loom_sanitizer_value_is_scalar_float(module, results[i])) continue;
    if (loom_sanitizer_result_already_asserted(module, results[i],
                                               predicate_kind)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_append_unique_value(
        values, op->result_count, results[i], &value_count, NULL));
  }
  if (value_count == 0) return iree_ok_status();

  loom_builder_set_after(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* assert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_fastmath_value_assertion(
      pass, module, rewriter,
      (loom_value_slice_t){
          .values = values,
          .count = value_count,
      },
      predicate_kind, op->location, &assert_op));
  if (!assert_op) return iree_ok_status();

  loom_value_slice_t checked_values =
      loom_sanitizer_assert_value_results(assert_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, checked_values.values, checked_values.count,
      value_checkpoint));
  for (uint16_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        rewriter, values[i], checked_values.values[i], assert_op));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_fastmath_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  uint8_t fastmath_flags = loom_sanitizer_scalar_fastmath_flags(op);
  loom_predicate_kind_t predicate_kind = LOOM_PREDICATE_COUNT_;
  if (!loom_sanitizer_fastmath_predicate_kind(fastmath_flags,
                                              &predicate_kind)) {
    return iree_ok_status();
  }
  bool inputs_changed = false;
  IREE_RETURN_IF_ERROR(loom_sanitizer_instrument_fastmath_inputs(
      pass, module, rewriter, op, predicate_kind, &inputs_changed));
  bool results_changed = false;
  IREE_RETURN_IF_ERROR(loom_sanitizer_instrument_fastmath_results(
      pass, module, rewriter, op, predicate_kind, &results_changed));
  if (inputs_changed || results_changed) {
    loom_sanitizer_insert_assertions_statistics_t* statistics =
        loom_sanitizer_insert_assertions_statistics(pass);
    ++statistics->fastmath_ops_instrumented;
  }
  return iree_ok_status();
}

iree_status_t loom_sanitizer_insert_assertions_run(loom_pass_t* pass,
                                                   loom_module_t* module,
                                                   loom_func_like_t function) {
  const bool access_checks_enabled =
      loom_sanitizer_checks_enabled(pass, LOOM_SANITIZER_CHECK_ACCESS);
  const bool value_checks_enabled =
      loom_sanitizer_checks_enabled(pass, LOOM_SANITIZER_CHECK_VALUE);
  if (!access_checks_enabled && !value_checks_enabled) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_value_fact_table_t* value_facts = NULL;
  iree_status_t status = loom_pass_value_facts_prepare(
      pass, module, loom_pass_value_fact_scope_function(function),
      &value_facts);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_enable_analysis(&rewriter, function, value_facts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_seed_function(&rewriter, function);
  }
  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) break;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
    if (access_checks_enabled) {
      status =
          loom_sanitizer_try_instrument_access_op(pass, module, &rewriter, op);
      if (!iree_status_is_ok(status)) continue;
      if (loom_pass_has_error_diagnostics(pass)) break;
    }
    if (!value_checks_enabled) continue;
    if (loom_scalar_assume_isa(op)) {
      status = loom_sanitizer_replace_assume(
          pass, module, &rewriter, op, loom_scalar_assume_values(op),
          loom_scalar_assume_results(op), loom_scalar_assume_predicates(op));
      continue;
    }
    if (loom_index_assume_isa(op)) {
      status = loom_sanitizer_replace_assume(
          pass, module, &rewriter, op, loom_index_assume_values(op),
          loom_index_assume_results(op), loom_index_assume_predicates(op));
      continue;
    }
    status =
        loom_sanitizer_try_instrument_fastmath_op(pass, module, &rewriter, op);
  }

  if (iree_status_is_ok(status) &&
      (rewriter.created_op_count != 0 || rewriter.erased_op_count != 0 ||
       iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED))) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status) && pass->value_facts) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}

static const loom_pass_info_t
    loom_sanitizer_materialize_assertions_pass_info_storage = {
        .name = IREE_SVL("sanitizer-materialize-assertions"),
        .description =
            IREE_SVL("Materialize sanitizer assertions for target reporting."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_sanitizer_materialize_assertions_pass_info(void) {
  return &loom_sanitizer_materialize_assertions_pass_info_storage;
}

iree_status_t loom_sanitizer_materialize_assertions_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  (void)pass;
  (void)module;
  (void)function;
  return iree_ok_status();
}
