// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

#define LOOM_SCF_UNROLL_STATISTICS(V, statistics_type)                   \
  V(statistics_type, loops_unrolled, "loops-unrolled",                   \
    "Number of scf.for loops unrolled.")                                 \
  V(statistics_type, iterations_materialized, "iterations-materialized", \
    "Number of loop body copies materialized.")                          \
  V(statistics_type, policies_cleared, "policies-cleared",               \
    "Number of no-op scf.for unroll policies removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_scf_unroll_statistics,
                            loom_scf_unroll_statistics_t,
                            LOOM_SCF_UNROLL_STATISTICS)

static const loom_pass_info_t loom_scf_unroll_pass_info_storage = {
    .name = IREE_SVL("unroll-scf-for"),
    .description = IREE_SVL("Consume local scf.for unroll policies."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_scf_unroll_statistics_layout,
};

const loom_pass_info_t* loom_scf_unroll_pass_info(void) {
  return &loom_scf_unroll_pass_info_storage;
}

iree_status_t loom_scf_unroll_create(loom_pass_t* pass,
                                     iree_string_view_t options_string) {
  (void)pass;
  if (!iree_string_view_is_empty(options_string)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass 'unroll-scf-for' takes no options");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Candidate collection
//===----------------------------------------------------------------------===//

#define LOOM_SCF_UNROLL_INITIAL_LOOP_CAPACITY 16

typedef struct loom_scf_unroll_loop_list_t {
  // Collected scf.for operations in traversal order.
  loom_op_t** ops;
  // Number of collected loop operations.
  iree_host_size_t count;
  // Allocated loop pointer capacity.
  iree_host_size_t capacity;
} loom_scf_unroll_loop_list_t;

typedef struct loom_scf_unroll_collect_context_t {
  // Pass scratch arena used for the collected loop pointer list.
  iree_arena_allocator_t* arena;
  // Collected scf.for operations.
  loom_scf_unroll_loop_list_t* loops;
} loom_scf_unroll_collect_context_t;

static iree_status_t loom_scf_unroll_loop_list_initialize(
    iree_arena_allocator_t* arena, loom_scf_unroll_loop_list_t* list) {
  list->count = 0;
  list->capacity = LOOM_SCF_UNROLL_INITIAL_LOOP_CAPACITY;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_scf_unroll_loop_list_push(
    iree_arena_allocator_t* arena, loom_scf_unroll_loop_list_t* list,
    loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(loom_op_t*),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_collect_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_scf_unroll_collect_context_t* collect_context =
      (loom_scf_unroll_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_for_isa(op)) {
    return iree_ok_status();
  }
  return loom_scf_unroll_loop_list_push(collect_context->arena,
                                        collect_context->loops, op);
}

//===----------------------------------------------------------------------===//
// Unrolling
//===----------------------------------------------------------------------===//

typedef struct loom_scf_unroll_context_t {
  // Current pass invocation.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_scf_unroll_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for cloning and erasure.
  loom_rewriter_t* rewriter;
  // Value facts for the current function snapshot.
  loom_value_fact_table_t* fact_table;
  // True when the caller requested pass report detail rows.
  bool reports_enabled;
} loom_scf_unroll_context_t;

typedef enum loom_scf_unroll_trip_count_state_e {
  LOOM_SCF_UNROLL_TRIP_COUNT_EXACT = 0,
  LOOM_SCF_UNROLL_TRIP_COUNT_UNRESOLVED = 1,
  LOOM_SCF_UNROLL_TRIP_COUNT_NON_POSITIVE_STEP = 2,
  LOOM_SCF_UNROLL_TRIP_COUNT_OVERFLOW = 3,
  LOOM_SCF_UNROLL_TRIP_COUNT_TOO_LARGE = 4,
  LOOM_SCF_UNROLL_TRIP_COUNT_RANGE_DEPENDENT = 5,
} loom_scf_unroll_trip_count_state_t;

typedef enum loom_scf_unroll_lower_bound_kind_e {
  LOOM_SCF_UNROLL_LOWER_BOUND_STATIC = 0,
  LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC = 1,
} loom_scf_unroll_lower_bound_kind_t;

typedef struct loom_scf_unroll_trip_count_t {
  // Number of iterations proven for the loop.
  uint32_t count;
  // Positive static step used between adjacent induction values.
  int64_t step;
  // Describes whether |lower_i64| or |lower_value| defines the first IV.
  loom_scf_unroll_lower_bound_kind_t lower_kind;
  // Static lower bound used when |lower_kind| is STATIC.
  int64_t lower_i64;
  // Dynamic lower bound value used when |lower_kind| is DYNAMIC.
  loom_value_id_t lower_value;
  // Inclusive lower-bound fact range minimum used for dynamic proof.
  int64_t lower_range_min;
  // Inclusive lower-bound fact range maximum used for dynamic proof.
  int64_t lower_range_max;
} loom_scf_unroll_trip_count_t;

static bool loom_scf_unroll_exact_i64(const loom_value_fact_table_t* facts,
                                      loom_value_id_t value,
                                      int64_t* out_integer) {
  return loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(facts, value), out_integer);
}

static loom_scf_unroll_trip_count_state_t
loom_scf_unroll_compute_trip_count_i64(int64_t lower, int64_t upper,
                                       int64_t step, uint32_t* out_trip_count) {
  *out_trip_count = 0;
  if (step <= 0) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_NON_POSITIVE_STEP;
  }
  if (upper <= lower) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_EXACT;
  }

  int64_t span = 0;
  if ((lower > 0 && upper < INT64_MIN + lower) ||
      (lower < 0 && upper > INT64_MAX + lower)) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_OVERFLOW;
  }
  span = upper - lower;
  uint64_t trip_count = ((uint64_t)span + (uint64_t)step - 1) / (uint64_t)step;
  if (trip_count > UINT32_MAX) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_TOO_LARGE;
  }

  *out_trip_count = (uint32_t)trip_count;
  return LOOM_SCF_UNROLL_TRIP_COUNT_EXACT;
}

static iree_status_t loom_scf_unroll_emit_policy_error(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    iree_string_view_t field_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission);
}

static loom_scf_unroll_trip_count_state_t loom_scf_unroll_resolve_trip_count(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_scf_unroll_trip_count_t* out_trip_count) {
  *out_trip_count = (loom_scf_unroll_trip_count_t){
      .count = 0,
      .step = 0,
      .lower_kind = LOOM_SCF_UNROLL_LOWER_BOUND_STATIC,
      .lower_i64 = 0,
      .lower_value = LOOM_VALUE_ID_INVALID,
      .lower_range_min = 0,
      .lower_range_max = 0,
  };

  int64_t upper = 0;
  int64_t step = 0;
  if (!loom_scf_unroll_exact_i64(context->fact_table,
                                 loom_scf_for_upper_bound(op), &upper) ||
      !loom_scf_unroll_exact_i64(context->fact_table, loom_scf_for_step(op),
                                 &step)) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_UNRESOLVED;
  }
  if (step <= 0) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_NON_POSITIVE_STEP;
  }

  loom_value_id_t lower_value = loom_scf_for_lower_bound(op);
  loom_value_facts_t lower_facts =
      loom_value_fact_table_lookup(context->fact_table, lower_value);
  int64_t lower = 0;
  if (loom_value_facts_as_exact_i64(lower_facts, &lower)) {
    uint32_t trip_count = 0;
    loom_scf_unroll_trip_count_state_t state =
        loom_scf_unroll_compute_trip_count_i64(lower, upper, step, &trip_count);
    if (state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) return state;
    *out_trip_count = (loom_scf_unroll_trip_count_t){
        .count = trip_count,
        .step = step,
        .lower_kind = LOOM_SCF_UNROLL_LOWER_BOUND_STATIC,
        .lower_i64 = lower,
        .lower_value = LOOM_VALUE_ID_INVALID,
        .lower_range_min = lower,
        .lower_range_max = lower,
    };
    return LOOM_SCF_UNROLL_TRIP_COUNT_EXACT;
  }

  if (loom_value_facts_is_float(lower_facts) ||
      lower_facts.range_lo == INT64_MIN || lower_facts.range_hi == INT64_MAX) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_UNRESOLVED;
  }

  uint32_t lower_min_trip_count = 0;
  loom_scf_unroll_trip_count_state_t lower_min_state =
      loom_scf_unroll_compute_trip_count_i64(lower_facts.range_lo, upper, step,
                                             &lower_min_trip_count);
  if (lower_min_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    return lower_min_state;
  }
  uint32_t lower_max_trip_count = 0;
  loom_scf_unroll_trip_count_state_t lower_max_state =
      loom_scf_unroll_compute_trip_count_i64(lower_facts.range_hi, upper, step,
                                             &lower_max_trip_count);
  if (lower_max_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    return lower_max_state;
  }
  // With a positive step, trip count is monotonic in the lower bound; equal
  // endpoint counts prove every value in the lower-bound range has that count.
  if (lower_min_trip_count != lower_max_trip_count) {
    return LOOM_SCF_UNROLL_TRIP_COUNT_RANGE_DEPENDENT;
  }

  *out_trip_count = (loom_scf_unroll_trip_count_t){
      .count = lower_min_trip_count,
      .step = step,
      .lower_kind = LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC,
      .lower_i64 = 0,
      .lower_value = lower_value,
      .lower_range_min = lower_facts.range_lo,
      .lower_range_max = lower_facts.range_hi,
  };
  return LOOM_SCF_UNROLL_TRIP_COUNT_EXACT;
}

static iree_string_view_t loom_scf_unroll_lower_bound_kind_name(
    loom_scf_unroll_lower_bound_kind_t kind) {
  switch (kind) {
    case LOOM_SCF_UNROLL_LOWER_BOUND_STATIC:
      return IREE_SV("static");
    case LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC:
      return IREE_SV("dynamic");
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_scf_unroll_append_report_detail(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    iree_string_view_t policy, int64_t unroll_factor,
    const loom_scf_unroll_trip_count_t* trip_count) {
  if (!context->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[12];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"), IREE_SV("unrolled"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("op"), loom_op_name(context->module, op));
  fields[field_count++] =
      loom_pass_report_detail_string_field(IREE_SV("policy"), policy);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("trip_count"), trip_count->count);
  fields[field_count++] =
      loom_pass_report_detail_int64_field(IREE_SV("step"), trip_count->step);
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("lower_bound_kind"),
      loom_scf_unroll_lower_bound_kind_name(trip_count->lower_kind));
  if (unroll_factor >= 0) {
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("unroll_factor"), unroll_factor);
  }
  switch (trip_count->lower_kind) {
    case LOOM_SCF_UNROLL_LOWER_BOUND_STATIC:
      fields[field_count++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower"), trip_count->lower_i64);
      break;
    case LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC:
      fields[field_count++] = loom_pass_report_detail_uint64_field(
          IREE_SV("lower_value_id"), trip_count->lower_value);
      fields[field_count++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower_range_min"), trip_count->lower_range_min);
      fields[field_count++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower_range_max"), trip_count->lower_range_max);
      break;
  }
  return loom_pass_report_append_detail(context->pass, IREE_SV("scf-unroll"),
                                        fields, field_count);
}

static iree_status_t loom_scf_unroll_emit_trip_count_error(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_scf_unroll_trip_count_state_t state) {
  switch (state) {
    case LOOM_SCF_UNROLL_TRIP_COUNT_UNRESOLVED:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll"), 0,
          IREE_SV("exact static trip count"));
    case LOOM_SCF_UNROLL_TRIP_COUNT_NON_POSITIVE_STEP:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("step"), 0,
          IREE_SV("positive exact static step"));
    case LOOM_SCF_UNROLL_TRIP_COUNT_OVERFLOW:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll"), 0,
          IREE_SV("trip count arithmetic without signed i64 overflow"));
    case LOOM_SCF_UNROLL_TRIP_COUNT_TOO_LARGE:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll"), (int64_t)UINT32_MAX,
          IREE_SV("trip count representable as uint32"));
    case LOOM_SCF_UNROLL_TRIP_COUNT_RANGE_DEPENDENT:
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll"), 0,
          IREE_SV("static trip count independent of lower-bound range"));
    case LOOM_SCF_UNROLL_TRIP_COUNT_EXACT:
      return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_scf_unroll_policy_present(loom_op_t* op) {
  return loom_scf_for_unroll_factor_is_present(op) ||
         !loom_attr_is_absent(
             loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX]);
}

static bool loom_scf_unroll_shape_is_supported(loom_op_t* op,
                                               loom_op_t** out_yield) {
  *out_yield = NULL;
  loom_region_t* body = loom_scf_for_body(op);
  if (!body || body->block_count != 1) {
    return false;
  }
  loom_block_t* body_block = loom_region_entry_block(body);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  if (iter_args.count != op->result_count || !body_block ||
      body_block->arg_count != 1 + iter_args.count) {
    return false;
  }
  loom_op_t* yield = body_block->last_op;
  if (!yield || !loom_scf_yield_isa(yield)) {
    return false;
  }
  if (loom_scf_yield_values(yield).count != op->result_count) {
    return false;
  }
  *out_yield = yield;
  return true;
}

static iree_status_t loom_scf_unroll_build_iteration_index(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_value_id_t induction_variable,
    const loom_scf_unroll_trip_count_t* trip_count, uint32_t ordinal,
    loom_value_id_t* out_index) {
  *out_index = LOOM_VALUE_ID_INVALID;

  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)ordinal, trip_count->step, &scaled_step)) {
    return iree_ok_status();
  }

  if (trip_count->lower_kind == LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC) {
    if (scaled_step == 0) {
      *out_index = trip_count->lower_value;
      return iree_ok_status();
    }
    loom_op_t* offset_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &context->rewriter->builder, loom_attr_i64(scaled_step),
        loom_module_value_type(context->module, loom_scf_for_lower_bound(op)),
        op->location, &offset_op));
    loom_value_id_t offset = loom_index_constant_result(offset_op);
    loom_op_t* add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        &context->rewriter->builder, trip_count->lower_value, offset,
        loom_module_value_type(context->module, loom_scf_for_lower_bound(op)),
        op->location, &add_op));
    *out_index = loom_index_add_result(add_op);
    char suffix[32] = {0};
    int suffix_length =
        iree_snprintf(suffix, sizeof(suffix), "%" PRId64, scaled_step);
    if (suffix_length <= 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_ok_status();
    }
    return loom_rewriter_try_set_derived_value_name(
        context->rewriter, induction_variable, *out_index,
        iree_make_string_view(suffix, (iree_host_size_t)suffix_length));
  }

  int64_t iteration_value = 0;
  if (!iree_checked_add_i64(trip_count->lower_i64, scaled_step,
                            &iteration_value)) {
    return iree_ok_status();
  }
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(iteration_value),
      loom_module_value_type(context->module, loom_scf_for_lower_bound(op)),
      op->location, &constant_op));
  *out_index = loom_index_constant_result(constant_op);

  char suffix[32] = {0};
  int suffix_length =
      iree_snprintf(suffix, sizeof(suffix), "%" PRId64, iteration_value);
  if (suffix_length <= 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  return loom_rewriter_try_set_derived_value_name(
      context->rewriter, induction_variable, *out_index,
      iree_make_string_view(suffix, (iree_host_size_t)suffix_length));
}

static iree_status_t loom_scf_unroll_clone_iteration(
    loom_scf_unroll_context_t* context, const loom_block_t* body_block,
    loom_op_t* yield, loom_value_id_t iteration_index, uint32_t ordinal,
    const loom_value_id_t* carried_values, uint16_t carried_count,
    loom_value_id_t* next_carried_values) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, &context->module->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&remap, body_block->arg_ids[0], iteration_index));
  for (uint16_t i = 0; i < carried_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &remap, body_block->arg_ids[1 + i], carried_values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_ir_clone_block_ops(
      &context->rewriter->builder, body_block, &remap,
      &(loom_ir_clone_block_options_t){.omit_terminators = true}));

  if (ordinal > 0 &&
      iree_any_bit_set(context->rewriter->name_policy,
                       LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES)) {
    char suffix[32] = {0};
    int suffix_length =
        iree_snprintf(suffix, sizeof(suffix), "%" PRIu32, ordinal);
    if (suffix_length <= 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_ok_status();
    }
    iree_string_view_t suffix_view =
        iree_make_string_view(suffix, (iree_host_size_t)suffix_length);
    for (const loom_op_t* source_op = body_block->first_op;
         source_op && source_op != yield; source_op = source_op->next_op) {
      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      for (uint16_t i = 0; i < source_op->result_count; ++i) {
        loom_value_id_t source_result = source_results[i];
        if (source_result == LOOM_VALUE_ID_INVALID) continue;
        loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
        if (!loom_ir_remap_try_lookup_value(&remap, source_result,
                                            &target_result)) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_rewriter_clear_value_name(context->rewriter, target_result));
        IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
            context->rewriter, source_result, target_result, suffix_view));
      }
    }
  }

  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  for (uint16_t i = 0; i < carried_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &remap, yielded_values.values[i], &next_carried_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_copy_result_types(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (op->result_count == 0) return iree_ok_status();
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, op->result_count,
                                sizeof(*result_types), (void**)&result_types));
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_types[i] = loom_module_value_type(context->module, results[i]);
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_adjust_tied_results_for_policy_clear(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_value_slice_t iter_args, loom_tied_result_t** out_tied_results,
    uint16_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  if (op->tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, op->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  uint16_t old_iter_arg_offset = op->operand_count;
  if (iter_args.count > 0) {
    old_iter_arg_offset =
        (uint16_t)(iter_args.values - loom_op_const_operands(op));
  }
  const uint16_t new_iter_arg_offset = 3;
  const uint16_t old_unroll_factor_offset =
      (uint16_t)(old_iter_arg_offset + iter_args.count);
  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    if (tied_result.operand_index >= old_iter_arg_offset &&
        tied_result.operand_index < old_iter_arg_offset + iter_args.count) {
      tied_result.operand_index =
          (uint16_t)(new_iter_arg_offset +
                     (tied_result.operand_index - old_iter_arg_offset));
    } else if (loom_scf_for_unroll_factor_is_present(op) &&
               tied_result.operand_index == old_unroll_factor_offset) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll_factor"), tied_result.operand_index,
          IREE_SV("not tied to a result"));
    }
    tied_results[i] = tied_result;
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = op->tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_clear_policy(
    loom_scf_unroll_context_t* context, loom_op_t* op, loom_op_t* yield,
    bool* out_changed) {
  *out_changed = false;
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_copy_result_types(context, op, &result_types));

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_adjust_tied_results_for_policy_clear(
      context, op, iter_args, &tied_results, &tied_result_count));

  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter->builder, /*build_flags=*/0,
      loom_scf_for_lower_bound(op), loom_scf_for_upper_bound(op),
      loom_scf_for_step(op), iter_args.values, iter_args.count, result_types,
      op->result_count, tied_results, tied_result_count, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, op->location, &new_loop));

  loom_region_t* old_body = loom_scf_for_body(op);
  loom_block_t* old_block = loom_region_entry_block(old_body);
  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, new_loop, new_body);
  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(&context->rewriter->builder, yielded_values.values,
                           yielded_values.count, op->location, &new_yield));
  loom_builder_restore(&context->rewriter->builder, saved_ip);

  loom_block_t* new_block = loom_region_entry_block(new_body);
  for (uint16_t i = 0; i < old_block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(new_block, i)));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(new_block, i)));
  }

  loom_op_t* child_op = old_block->first_op;
  while (child_op && child_op != yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(context->rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, op, new_results.values, new_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, op, new_results.values, new_results.count));
  ++context->statistics->policies_cleared;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_try_unroll(
    loom_scf_unroll_context_t* context, loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      !loom_scf_unroll_policy_present(op)) {
    return iree_ok_status();
  }

  bool has_unroll_factor = loom_scf_for_unroll_factor_is_present(op);
  bool has_unroll_policy = !loom_attr_is_absent(
      loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX]);
  if (has_unroll_factor && has_unroll_policy) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("unroll"), 2,
        IREE_SV("either bare unroll or unroll factor, not both"));
  }

  loom_op_t* yield = NULL;
  if (!loom_scf_unroll_shape_is_supported(op, &yield)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("unroll"), 0,
        IREE_SV("single-block scf.for with matching scf.yield"));
  }

  int64_t unroll_factor = 0;
  if (has_unroll_factor) {
    loom_value_id_t unroll_factor_value = loom_scf_for_unroll_factor(op);
    if (!loom_scf_unroll_exact_i64(context->fact_table, unroll_factor_value,
                                   &unroll_factor)) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll_factor"), unroll_factor_value,
          IREE_SV("compile-time exact i64 value"));
    }
    if (unroll_factor < 0) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll_factor"), unroll_factor,
          IREE_SV("nonnegative unroll factor"));
    }
    if (unroll_factor <= 1) {
      return loom_scf_unroll_clear_policy(context, op, yield, out_changed);
    }
  }

  loom_scf_unroll_trip_count_t trip_count = {0};
  loom_scf_unroll_trip_count_state_t trip_count_state =
      loom_scf_unroll_resolve_trip_count(context, op, &trip_count);
  if (trip_count_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    return loom_scf_unroll_emit_trip_count_error(context, op, trip_count_state);
  }
  if (has_unroll_factor && (uint32_t)unroll_factor != trip_count.count) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("unroll_factor"), unroll_factor,
        IREE_SV("0, 1, or the exact full trip count"));
  }
  IREE_RETURN_IF_ERROR(loom_scf_unroll_append_report_detail(
      context, op, has_unroll_factor ? IREE_SV("factor") : IREE_SV("bare"),
      has_unroll_factor ? unroll_factor : -1, &trip_count));

  loom_region_t* body = loom_scf_for_body(op);
  loom_block_t* body_block = loom_region_entry_block(body);
  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_id_t* carried_values = NULL;
  loom_value_id_t* next_carried_values = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, op->result_count, sizeof(*carried_values),
        (void**)&carried_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, op->result_count, sizeof(*next_carried_values),
        (void**)&next_carried_values));
    memcpy(carried_values, iter_args.values,
           (iree_host_size_t)op->result_count * sizeof(*carried_values));
  }

  loom_value_id_t induction_variable = body_block->arg_ids[0];
  for (uint32_t ordinal = 0; ordinal < trip_count.count; ++ordinal) {
    loom_value_id_t iteration_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_iteration_index(
        context, op, induction_variable, &trip_count, ordinal,
        &iteration_index));
    if (iteration_index == LOOM_VALUE_ID_INVALID) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll"), ordinal,
          IREE_SV("iteration index representable as i64"));
    }
    IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_iteration(
        context, body_block, yield, iteration_index, ordinal, carried_values,
        op->result_count, next_carried_values));
    if (op->result_count > 0) {
      loom_value_id_t* temporary_values = carried_values;
      carried_values = next_carried_values;
      next_carried_values = temporary_values;
    }
  }

  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        context->rewriter, op, carried_values, op->result_count,
        value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, op, carried_values, op->result_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, op));
  }

  ++context->statistics->loops_unrolled;
  context->statistics->iterations_materialized += trip_count.count;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_process_function_once(
    loom_scf_unroll_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  *out_changed = false;
  loom_scf_unroll_loop_list_t loops = {0};
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_loop_list_initialize(context->pass->arena, &loops));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_scf_unroll_collect_context_t collect_context = {
      .arena = context->pass->arena,
      .loops = &loops,
  };
  IREE_RETURN_IF_ERROR(
      loom_walk_function(context->module, function, LOOM_WALK_PRE_ORDER,
                         (loom_walk_callback_t){
                             .fn = loom_scf_unroll_collect_loop,
                             .user_data = &collect_context,
                         },
                         context->pass->arena, &walk_result));

  for (iree_host_size_t i = 0; i < loops.count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(
        loom_scf_unroll_try_unroll(context, loops.ops[i], &changed));
    *out_changed = *out_changed || changed;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_unroll_run(loom_pass_t* pass, loom_module_t* module,
                                  loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  loom_scf_unroll_context_t context = {
      .pass = pass,
      .statistics = loom_scf_unroll_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .reports_enabled = loom_pass_report_is_enabled(pass),
  };

  iree_status_t status = iree_ok_status();
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status = loom_pass_value_facts_acquire(
        pass, module, loom_pass_value_fact_scope_function(function),
        &context.fact_table);
    if (!iree_status_is_ok(status)) break;
    status =
        loom_scf_unroll_process_function_once(&context, function, &changed);
    if (changed) {
      any_changed = true;
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
