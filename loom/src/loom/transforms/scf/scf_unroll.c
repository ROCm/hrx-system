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
#include "loom/ir/types.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/scf/scf_unroll_bounds.h"
#include "loom/transforms/scf/scf_unroll_tile.h"
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

static bool loom_scf_unroll_policy_present(loom_op_t* op) {
  return loom_scf_for_unroll_factor_is_present(op) ||
         !loom_attr_is_absent(
             loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX]) ||
         !loom_attr_is_absent(
             loom_op_attrs(op)[loom_scf_for_unroll_schedule_ATTR_INDEX]);
}

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
  // True when at least one collected loop requests unrolling.
  bool has_policy;
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
  if (!collect_context->has_policy) {
    collect_context->has_policy = loom_scf_unroll_policy_present(op);
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

typedef enum loom_scf_unroll_partial_unroll_flag_bits_e {
  LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL = 1u << 0,
  LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_TAIL_LOOP = 1u << 1,
} loom_scf_unroll_partial_unroll_flag_bits_t;
typedef uint32_t loom_scf_unroll_partial_unroll_flags_t;

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
    if (state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
      return state;
    }
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

static iree_string_view_t loom_scf_unroll_trip_count_state_name(
    loom_scf_unroll_trip_count_state_t state) {
  switch (state) {
    case LOOM_SCF_UNROLL_TRIP_COUNT_EXACT:
      return IREE_SV("exact");
    case LOOM_SCF_UNROLL_TRIP_COUNT_UNRESOLVED:
      return IREE_SV("unresolved");
    case LOOM_SCF_UNROLL_TRIP_COUNT_NON_POSITIVE_STEP:
      return IREE_SV("non_positive_step");
    case LOOM_SCF_UNROLL_TRIP_COUNT_OVERFLOW:
      return IREE_SV("overflow");
    case LOOM_SCF_UNROLL_TRIP_COUNT_TOO_LARGE:
      return IREE_SV("too_large");
    case LOOM_SCF_UNROLL_TRIP_COUNT_RANGE_DEPENDENT:
      return IREE_SV("range_dependent");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_scf_unroll_schedule_name(
    loom_scf_for_unroll_schedule_t schedule) {
  switch (schedule) {
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_LINEAR:
      return IREE_SV("linear");
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED:
      return IREE_SV("interleaved");
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_RECURRENCE:
      return IREE_SV("recurrence");
    case LOOM_SCF_FOR_UNROLL_SCHEDULE_COUNT_:
      break;
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_scf_unroll_tail_strategy_name(
    loom_scf_unroll_partial_unroll_flags_t flags) {
  if (iree_any_bit_set(flags, LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL)) {
    return IREE_SV("guarded_lanes");
  }
  if (iree_any_bit_set(flags, LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_TAIL_LOOP)) {
    return IREE_SV("tail_loop");
  }
  return IREE_SV("none");
}

static void loom_scf_unroll_append_lower_bound_report_fields(
    loom_pass_report_detail_field_t* fields, uint16_t* field_count,
    const loom_scf_unroll_trip_count_t* trip_count) {
  fields[(*field_count)++] = loom_pass_report_detail_string_field(
      IREE_SV("lower_bound_kind"),
      loom_scf_unroll_lower_bound_kind_name(trip_count->lower_kind));
  switch (trip_count->lower_kind) {
    case LOOM_SCF_UNROLL_LOWER_BOUND_STATIC:
      fields[(*field_count)++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower"), trip_count->lower_i64);
      break;
    case LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC:
      fields[(*field_count)++] = loom_pass_report_detail_uint64_field(
          IREE_SV("lower_value_id"), trip_count->lower_value);
      fields[(*field_count)++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower_range_min"), trip_count->lower_range_min);
      fields[(*field_count)++] = loom_pass_report_detail_int64_field(
          IREE_SV("lower_range_max"), trip_count->lower_range_max);
      break;
  }
}

static iree_status_t loom_scf_unroll_append_report_detail(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    iree_string_view_t policy, loom_scf_for_unroll_schedule_t schedule,
    int64_t unroll_factor, const loom_scf_unroll_trip_count_t* trip_count) {
  if (!context->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[11];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"), IREE_SV("unrolled"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("op"), loom_op_name(context->module, op));
  fields[field_count++] =
      loom_pass_report_detail_string_field(IREE_SV("policy"), policy);
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("schedule"), loom_scf_unroll_schedule_name(schedule));
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("trip_count"), trip_count->count);
  fields[field_count++] =
      loom_pass_report_detail_int64_field(IREE_SV("step"), trip_count->step);
  if (unroll_factor >= 0) {
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("unroll_factor"), unroll_factor);
  }
  loom_scf_unroll_append_lower_bound_report_fields(fields, &field_count,
                                                   trip_count);
  return loom_pass_report_append_detail(context->pass, IREE_SV("scf-unroll"),
                                        fields, field_count);
}

static iree_status_t loom_scf_unroll_append_partial_report_detail(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    int64_t unroll_factor, loom_scf_for_unroll_schedule_t schedule,
    int64_t step, loom_scf_unroll_trip_count_state_t trip_count_state,
    const loom_scf_unroll_trip_count_t* trip_count,
    loom_scf_unroll_partial_unroll_flags_t flags) {
  if (!context->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[13];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"), IREE_SV("stripmined"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("op"), loom_op_name(context->module, op));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("policy"), IREE_SV("factor"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("schedule"), loom_scf_unroll_schedule_name(schedule));
  fields[field_count++] = loom_pass_report_detail_int64_field(
      IREE_SV("unroll_factor"), unroll_factor);
  fields[field_count++] =
      loom_pass_report_detail_int64_field(IREE_SV("step"), step);
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("tail_strategy"), loom_scf_unroll_tail_strategy_name(flags));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("trip_count_state"),
      loom_scf_unroll_trip_count_state_name(trip_count_state));
  if (trip_count_state == LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    fields[field_count++] = loom_pass_report_detail_uint64_field(
        IREE_SV("trip_count"), trip_count->count);
    loom_scf_unroll_append_lower_bound_report_fields(fields, &field_count,
                                                     trip_count);
  }
  return loom_pass_report_append_detail(context->pass, IREE_SV("scf-unroll"),
                                        fields, field_count);
}

static iree_status_t loom_scf_unroll_append_policy_absent_report_detail(
    const loom_scf_unroll_context_t* context, loom_op_t* op) {
  if (!context->reports_enabled) {
    return iree_ok_status();
  }

  loom_scf_unroll_trip_count_t trip_count = {0};
  loom_scf_unroll_trip_count_state_t trip_count_state =
      loom_scf_unroll_resolve_trip_count(context, op, &trip_count);
  if (trip_count_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT ||
      trip_count.count <= 1) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[12];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"), IREE_SV("structured"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("op"), loom_op_name(context->module, op));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("policy"), IREE_SV("absent"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("trip_count_state"),
      loom_scf_unroll_trip_count_state_name(trip_count_state));
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("trip_count"), trip_count.count);
  fields[field_count++] =
      loom_pass_report_detail_int64_field(IREE_SV("step"), trip_count.step);
  loom_scf_unroll_append_lower_bound_report_fields(fields, &field_count,
                                                   &trip_count);
  return loom_pass_report_append_detail(context->pass, IREE_SV("scf-unroll"),
                                        fields, field_count);
}

static iree_status_t loom_scf_unroll_append_clear_report_detail(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    int64_t unroll_factor, loom_scf_for_unroll_schedule_t schedule) {
  if (!context->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[6];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"), IREE_SV("cleared"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("op"), loom_op_name(context->module, op));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("policy"), IREE_SV("factor"));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("schedule"), loom_scf_unroll_schedule_name(schedule));
  fields[field_count++] = loom_pass_report_detail_int64_field(
      IREE_SV("unroll_factor"), unroll_factor);
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("clear_reason"), IREE_SV("factor_le_one"));
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

static iree_status_t loom_scf_unroll_build_strided_iteration_index(
    loom_scf_unroll_context_t* context, loom_value_id_t induction_variable,
    loom_value_id_t base_index, loom_type_t index_type, int64_t step,
    uint32_t ordinal, loom_location_id_t location, loom_value_id_t* out_index) {
  *out_index = LOOM_VALUE_ID_INVALID;
  if (ordinal == 0) {
    *out_index = base_index;
    return iree_ok_status();
  }

  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)ordinal, step, &scaled_step)) {
    return iree_ok_status();
  }

  loom_op_t* offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(scaled_step), index_type,
      location, &offset_op));
  loom_value_id_t offset = loom_index_constant_result(offset_op);
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(&context->rewriter->builder,
                                            base_index, offset, index_type,
                                            location, &add_op));
  *out_index = loom_index_add_result(add_op);

  char suffix[32] = {0};
  int suffix_length =
      iree_snprintf(suffix, sizeof(suffix), "%" PRId64, scaled_step);
  if (suffix_length <= 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  return loom_rewriter_try_set_derived_value_name(
      context->rewriter, induction_variable, *out_index,
      iree_make_string_view(suffix, (iree_host_size_t)suffix_length));
}

static iree_status_t loom_scf_unroll_build_index_constant(
    loom_scf_unroll_context_t* context, loom_op_t* op, int64_t value,
    loom_type_t index_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(value), index_type,
      op->location, &constant_op));
  *out_value = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

// Initializes the same-module remap reused by every linear clone of one loop
// body. Each iteration overwrites the body arguments before cloning, and
// dominance ensures each cloned result mapping is overwritten before a later
// body operation can consume it. Keeping one sparse map avoids allocating and
// initializing an identical map for every materialized iteration.
static iree_status_t loom_scf_unroll_initialize_iteration_remap(
    loom_scf_unroll_context_t* context, loom_ir_remap_t* out_remap) {
  return loom_ir_remap_initialize(
      context->module, context->module, context->pass->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      out_remap);
}

// Returns true when cloning the loop body must materialize and map |value_id|.
// Operand and type references are maintained exactly. Attribute references do
// not have individual use records, so their maintained summary bit is
// intentionally conservative: a stale bit may retain an index but can never
// remove one that the cloned body requires.
static bool loom_scf_unroll_value_has_references(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return !loom_value_has_no_uses(value) ||
         loom_value_has_attribute_uses(value) ||
         loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_scf_unroll_clone_iteration(
    loom_scf_unroll_context_t* context, loom_ir_remap_t* remap,
    const loom_block_t* body_block, loom_op_t* yield, uint32_t ordinal,
    const loom_value_id_t* carried_values, uint16_t carried_count,
    loom_value_id_t* next_carried_values) {
  for (uint16_t i = 0; i < carried_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, body_block->arg_ids[1 + i], carried_values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_ir_clone_block_ops(
      &context->rewriter->builder, body_block, remap,
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
        if (source_result == LOOM_VALUE_ID_INVALID) {
          continue;
        }
        loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
        if (!loom_ir_remap_try_lookup_value(remap, source_result,
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
        remap, yielded_values.values[i], &next_carried_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_clone_guarded_iteration(
    loom_scf_unroll_context_t* context, loom_ir_remap_t* remap,
    const loom_block_t* body_block, loom_op_t* yield, loom_op_t* source,
    loom_value_id_t base, int64_t step, uint32_t ordinal,
    const loom_value_id_t* carried_values, uint16_t carried_count,
    const loom_type_t* result_types, loom_location_id_t location,
    loom_value_id_t* next_carried_values) {
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_value_id_t source_iv = body_block->arg_ids[0];
  const loom_type_t type = loom_module_value_type(context->module, source_iv);
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  const bool is_offset = scalar_type == LOOM_SCALAR_TYPE_OFFSET;
  const int64_t advance = step * ordinal;
  const loom_value_facts_t iv_facts =
      loom_value_fact_table_lookup(context->fact_table, source_iv);
  int64_t maximum_candidate = 0;
  bool candidate_fits =
      iree_checked_add_i64(iv_facts.range_hi, advance, &maximum_candidate);
  if (candidate_fits) {
    const loom_value_facts_t maximum_facts =
        loom_value_facts_exact_i64(maximum_candidate);
    candidate_fits =
        is_offset
            ? loom_index_value_facts_fit_unsigned_target_carrier(
                  &context->fact_table->context, scalar_type, maximum_facts)
            : loom_index_value_facts_fit_signed_target_carrier(
                  &context->fact_table->context, scalar_type, maximum_facts);
  }
  loom_value_id_t iteration_index = LOOM_VALUE_ID_INVALID;
  loom_op_t* comparison = NULL;
  if (candidate_fits) {
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_strided_iteration_index(
        context, source_iv, base, type, step, ordinal, location,
        &iteration_index));
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder,
        is_offset ? LOOM_INDEX_CMP_PREDICATE_ULT : LOOM_INDEX_CMP_PREDICATE_SLT,
        iteration_index, loom_scf_for_upper_bound(source), location,
        &comparison));
  } else {
    // The body is entered only with base < upper. Unsigned subtraction keeps
    // that positive distance exact even when signed bounds straddle zero.
    loom_op_t* remaining = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_sub_build(builder, loom_scf_for_upper_bound(source), base,
                             type, location, &remaining));
    loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_index_constant(
        context, source, advance, type, &offset));
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_ULT, offset,
        loom_index_sub_result(remaining), location, &comparison));
  }
  const loom_value_id_t condition = loom_index_cmp_result(comparison);
  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &context->rewriter->builder,
      carried_count > 0 ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION : 0, condition,
      result_types, carried_count, NULL, 0, location, &if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  if (loom_scf_unroll_value_has_references(context->module, source_iv)) {
    if (!candidate_fits) {
      IREE_RETURN_IF_ERROR(loom_scf_unroll_build_strided_iteration_index(
          context, source_iv, base, type, step, ordinal, location,
          &iteration_index));
      // This guard proves the materialized lane is inside the source domain.
      const loom_predicate_t predicates[] = {
          {.kind = LOOM_PREDICATE_GE,
           .arg_count = 2,
           .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                        LOOM_PRED_ARG_NONE},
           .args = {iteration_index, base, 0}},
          {.kind = LOOM_PREDICATE_LT,
           .arg_count = 2,
           .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                        LOOM_PRED_ARG_NONE},
           .args = {iteration_index, loom_scf_for_upper_bound(source), 0}},
      };
      loom_op_t* assume = NULL;
      IREE_RETURN_IF_ERROR(loom_index_assume_build(
          builder, &iteration_index, 1, predicates, IREE_ARRAYSIZE(predicates),
          &type, 1, location, &assume));
      const loom_value_id_t bounded_index =
          loom_index_assume_results(assume).values[0];
      IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
          context->rewriter, iteration_index, bounded_index));
      iteration_index = bounded_index;
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, source_iv, iteration_index));
  }
  IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_iteration(
      context, remap, body_block, yield, ordinal, carried_values, carried_count,
      next_carried_values));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&context->rewriter->builder,
                                            next_carried_values, carried_count,
                                            location, &then_yield));
  loom_builder_restore(&context->rewriter->builder, saved_ip);

  if (carried_count > 0) {
    saved_ip = loom_builder_enter_region(&context->rewriter->builder, if_op,
                                         loom_scf_if_else_region(if_op));
    loom_op_t* else_yield = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(&context->rewriter->builder,
                                              carried_values, carried_count,
                                              location, &else_yield));
    loom_builder_restore(&context->rewriter->builder, saved_ip);

    loom_value_slice_t if_results = loom_scf_if_results(if_op);
    memcpy(next_carried_values, if_results.values,
           (iree_host_size_t)carried_count * sizeof(*next_carried_values));
  }

  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_copy_result_types(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (op->result_count == 0) {
    return iree_ok_status();
  }
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
  if (op->tied_result_count == 0) {
    return iree_ok_status();
  }

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
  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    if (tied_result.operand_index >= old_iter_arg_offset &&
        tied_result.operand_index < old_iter_arg_offset + iter_args.count) {
      tied_result.operand_index =
          (uint16_t)(new_iter_arg_offset +
                     (tied_result.operand_index - old_iter_arg_offset));
    }
    tied_results[i] = tied_result;
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = op->tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_clear_policy(
    loom_scf_unroll_context_t* context, loom_op_t* op, loom_op_t* yield,
    int64_t unroll_factor, loom_scf_for_unroll_schedule_t schedule,
    bool* out_changed) {
  *out_changed = false;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_append_clear_report_detail(
      context, op, unroll_factor, schedule));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);

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
      loom_scf_for_step(op), iter_args.values, iter_args.count, tied_results,
      tied_result_count, /*pipeline_depth=*/LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      op->location, &new_loop));

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

static iree_status_t loom_scf_unroll_build_scaled_step(
    loom_scf_unroll_context_t* context, loom_op_t* op, int64_t step,
    uint32_t unroll_factor, loom_type_t index_type,
    loom_value_id_t* out_scaled_step) {
  *out_scaled_step = LOOM_VALUE_ID_INVALID;
  const int64_t scaled_step = step * (int64_t)unroll_factor;
  return loom_scf_unroll_build_index_constant(context, op, scaled_step,
                                              index_type, out_scaled_step);
}

static iree_status_t loom_scf_unroll_partial_unroll(
    loom_scf_unroll_context_t* context, loom_op_t* op, loom_op_t* yield,
    int64_t step, uint32_t unroll_factor,
    loom_scf_unroll_partial_unroll_flags_t flags, bool* out_changed) {
  *out_changed = false;

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_copy_result_types(context, op, &result_types));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_adjust_tied_results_for_policy_clear(
      context, op, iter_args, &tied_results, &tied_result_count));

  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_type_t index_type =
      loom_module_value_type(context->module, loom_scf_for_lower_bound(op));
  loom_value_id_t scaled_step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_build_scaled_step(
      context, op, step, unroll_factor, index_type, &scaled_step));

  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter->builder, /*build_flags=*/0,
      loom_scf_for_lower_bound(op), loom_scf_for_upper_bound(op), scaled_step,
      iter_args.values, iter_args.count, tied_results, tied_result_count,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &new_loop));

  loom_region_t* old_body = loom_scf_for_body(op);
  loom_block_t* old_block = loom_region_entry_block(old_body);
  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, new_loop, new_body);
  loom_block_t* new_block = loom_region_entry_block(new_body);
  for (uint16_t i = 0; i < old_block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(new_block, i)));
  }

  loom_value_id_t* carried_values = NULL;
  loom_value_id_t* next_carried_values = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, op->result_count, sizeof(*carried_values),
        (void**)&carried_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, op->result_count, sizeof(*next_carried_values),
        (void**)&next_carried_values));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      carried_values[i] = loom_block_arg_id(new_block, (uint16_t)(1 + i));
    }
  }

  loom_ir_remap_t iteration_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_initialize_iteration_remap(context, &iteration_remap));
  loom_value_id_t outer_index = loom_block_arg_id(new_block, 0);
  loom_value_id_t source_induction_variable = loom_block_arg_id(old_block, 0);
  const bool induction_variable_has_references =
      loom_scf_unroll_value_has_references(context->module,
                                           source_induction_variable);
  for (uint32_t ordinal = 0; ordinal < unroll_factor; ++ordinal) {
    const bool guard_iteration =
        ordinal > 0 &&
        iree_any_bit_set(flags, LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL);
    if (guard_iteration) {
      IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_guarded_iteration(
          context, &iteration_remap, old_block, yield, op, outer_index, step,
          ordinal, carried_values, op->result_count, result_types, op->location,
          next_carried_values));
    } else {
      if (induction_variable_has_references) {
        loom_value_id_t iteration_index = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_scf_unroll_build_strided_iteration_index(
            context, source_induction_variable, outer_index, index_type, step,
            ordinal, op->location, &iteration_index));
        IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
            &iteration_remap, source_induction_variable, iteration_index));
      }
      IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_iteration(
          context, &iteration_remap, old_block, yield, ordinal, carried_values,
          op->result_count, next_carried_values));
    }

    if (op->result_count > 0) {
      loom_value_id_t* temporary_values = carried_values;
      carried_values = next_carried_values;
      next_carried_values = temporary_values;
    }
  }

  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&context->rewriter->builder,
                                            carried_values, op->result_count,
                                            op->location, &new_yield));
  loom_builder_restore(&context->rewriter->builder, saved_ip);

  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, op, new_results.values, new_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, op, new_results.values, new_results.count));

  ++context->statistics->loops_unrolled;
  context->statistics->iterations_materialized += unroll_factor;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_emit_scheduled_tile(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_block_t* body_block,
    const loom_scf_unroll_trip_count_t* trip_count,
    loom_value_id_t iteration_upper_bound,
    const loom_value_id_t* initial_carried_values,
    loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena,
    loom_value_id_t* final_carried_values) {
  const loom_value_id_t induction_variable = body_block->arg_ids[0];
  loom_value_id_t* iteration_indices = NULL;
  const loom_value_id_t* remapped_iteration_indices = NULL;
  if (trip_count->count > 0 && loom_scf_unroll_value_has_references(
                                   context->module, induction_variable)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, trip_count->count, sizeof(*iteration_indices),
        (void**)&iteration_indices));
    for (uint32_t ordinal = 0; ordinal < trip_count->count; ++ordinal) {
      IREE_RETURN_IF_ERROR(loom_scf_unroll_build_iteration_index(
          context, op, induction_variable, trip_count, ordinal,
          &iteration_indices[ordinal]));
      if (iteration_indices[ordinal] == LOOM_VALUE_ID_INVALID) {
        return loom_scf_unroll_emit_policy_error(
            context, op, IREE_SV("schedule"), ordinal,
            IREE_SV("iteration index representable as i64"));
      }
    }

    remapped_iteration_indices = iteration_indices;
    // Dynamic partial unrolling replaces the source upper bound with an aligned
    // main-loop bound. Preserve the source bound on every materialized index so
    // consumers do not have to reconstruct the split arithmetic.
    if (iteration_upper_bound != LOOM_VALUE_ID_INVALID) {
      loom_predicate_t* predicates = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(scratch_arena, trip_count->count,
                                    sizeof(*predicates), (void**)&predicates));
      loom_type_t* result_types = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          scratch_arena, trip_count->count, sizeof(*result_types),
          (void**)&result_types));
      for (uint32_t ordinal = 0; ordinal < trip_count->count; ++ordinal) {
        predicates[ordinal] = (loom_predicate_t){
            .kind = LOOM_PREDICATE_LT,
            .arg_count = 2,
            .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                         LOOM_PRED_ARG_NONE},
            .args = {iteration_indices[ordinal], iteration_upper_bound, 0},
        };
        result_types[ordinal] =
            loom_module_value_type(context->module, iteration_indices[ordinal]);
      }
      loom_op_t* assume_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_assume_build(
          &context->rewriter->builder, iteration_indices, trip_count->count,
          predicates, trip_count->count, result_types, trip_count->count,
          op->location, &assume_op));
      remapped_iteration_indices = loom_index_assume_results(assume_op).values;
      for (uint32_t ordinal = 0; ordinal < trip_count->count; ++ordinal) {
        IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
            context->rewriter, iteration_indices[ordinal],
            remapped_iteration_indices[ordinal]));
      }
    }
  }
  return loom_scf_unroll_tile_emit(
      context->pass, context->rewriter, context->fact_table, op,
      remapped_iteration_indices, trip_count->count, initial_carried_values,
      schedule, scratch_arena, final_carried_values);
}

static iree_status_t loom_scf_unroll_full_unroll_scheduled_with_arena(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_scf_unroll_trip_count_t* trip_count,
    loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena, bool* out_changed) {
  *out_changed = false;

  loom_region_t* body = loom_scf_for_body(op);
  loom_block_t* body_block = loom_region_entry_block(body);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  const uint16_t carried_count = op->result_count;
  loom_value_id_t* final_carried_values = NULL;
  if (carried_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, carried_count, sizeof(*final_carried_values),
        (void**)&final_carried_values));
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&context->rewriter->builder);
  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  iree_status_t status = loom_scf_unroll_emit_scheduled_tile(
      context, op, body_block, trip_count, LOOM_VALUE_ID_INVALID,
      iter_args.values, schedule, scratch_arena, final_carried_values);
  loom_builder_restore(&context->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  if (loom_pass_has_error_diagnostics(context->pass)) {
    return iree_ok_status();
  }

  if (carried_count > 0) {
    for (uint16_t i = 0; i < carried_count; ++i) {
      if (final_carried_values[i] < context->module->values.count) {
        continue;
      }
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("schedule"), schedule,
          IREE_SV("acyclic loop-carried dependencies"));
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        context->rewriter, op, final_carried_values, carried_count,
        value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, op, final_carried_values, carried_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, op));
  }

  ++context->statistics->loops_unrolled;
  context->statistics->iterations_materialized += trip_count->count;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_full_unroll_scheduled(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_scf_unroll_trip_count_t* trip_count,
    loom_scf_for_unroll_schedule_t schedule, bool* out_changed) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(context->pass->arena->block_pool, &scratch_arena);
  iree_status_t status = loom_scf_unroll_full_unroll_scheduled_with_arena(
      context, op, trip_count, schedule, &scratch_arena, out_changed);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_scf_unroll_build_exact_scheduled_main_upper(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_scf_unroll_trip_count_t* trip_count, uint32_t unroll_factor,
    loom_type_t index_type, loom_value_id_t* out_scaled_step,
    loom_value_id_t* out_main_upper) {
  *out_scaled_step = LOOM_VALUE_ID_INVALID;
  *out_main_upper = LOOM_VALUE_ID_INVALID;

  IREE_RETURN_IF_ERROR(loom_scf_unroll_build_scaled_step(
      context, op, trip_count->step, unroll_factor, index_type,
      out_scaled_step));

  const uint32_t main_iteration_count =
      (trip_count->count / unroll_factor) * unroll_factor;
  // With no remainder the original upper bound terminates exactly the full
  // tiles. Materializing lower + main_span could overflow after the last IV.
  if (main_iteration_count == trip_count->count) {
    *out_main_upper = loom_scf_for_upper_bound(op);
    return iree_ok_status();
  }

  int64_t main_span = 0;
  if (!iree_checked_mul_i64((int64_t)main_iteration_count, trip_count->step,
                            &main_span)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), main_iteration_count,
        IREE_SV("main scheduled iteration span representable as i64"));
  }

  switch (trip_count->lower_kind) {
    case LOOM_SCF_UNROLL_LOWER_BOUND_STATIC: {
      int64_t main_upper = 0;
      if (!iree_checked_add_i64(trip_count->lower_i64, main_span,
                                &main_upper)) {
        return loom_scf_unroll_emit_policy_error(
            context, op, IREE_SV("schedule"), main_span,
            IREE_SV("main scheduled upper bound representable as i64"));
      }
      loom_op_t* upper_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &context->rewriter->builder, loom_attr_i64(main_upper), index_type,
          op->location, &upper_op));
      *out_main_upper = loom_index_constant_result(upper_op);
      return iree_ok_status();
    }
    case LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC:
      if (!loom_index_value_facts_fit_signed_target_carrier(
              &context->fact_table->context, loom_type_element_type(index_type),
              loom_value_facts_exact_i64(main_span))) {
        return loom_scf_unroll_build_dynamic_split(
            &context->rewriter->builder, context->fact_table, op,
            trip_count->step, unroll_factor, *out_scaled_step, out_main_upper);
      }
      if (main_span == 0) {
        *out_main_upper = trip_count->lower_value;
        return iree_ok_status();
      }
      loom_op_t* span_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &context->rewriter->builder, loom_attr_i64(main_span), index_type,
          op->location, &span_op));
      loom_op_t* upper_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &context->rewriter->builder, trip_count->lower_value,
          loom_index_constant_result(span_op), index_type, op->location,
          &upper_op));
      *out_main_upper = loom_index_add_result(upper_op);
      return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_clone_loop_body(
    loom_scf_unroll_context_t* context, const loom_block_t* source_block,
    loom_block_t* target_block, iree_arena_allocator_t* scratch_arena) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, scratch_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &remap));
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        context->rewriter, loom_block_arg_id(source_block, i),
        loom_block_arg_id(target_block, i)));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, loom_block_arg_id(source_block, i),
                                loom_block_arg_id(target_block, i)));
  }
  return loom_ir_clone_block_ops(&context->rewriter->builder, source_block,
                                 &remap, NULL);
}

static iree_status_t loom_scf_unroll_partial_unroll_scheduled_with_arena(
    loom_scf_unroll_context_t* context, loom_op_t* op, loom_op_t* yield,
    int64_t step, uint32_t unroll_factor,
    const loom_scf_unroll_trip_count_t* trip_count, bool tail_loop_required,
    loom_scf_for_unroll_schedule_t schedule,
    iree_arena_allocator_t* scratch_arena, bool* out_changed) {
  *out_changed = false;

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_adjust_tied_results_for_policy_clear(
      context, op, iter_args, &tied_results, &tied_result_count));

  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_type_t index_type =
      loom_module_value_type(context->module, loom_scf_for_lower_bound(op));

  loom_value_id_t scaled_step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t main_upper = LOOM_VALUE_ID_INVALID;
  if (trip_count) {
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_exact_scheduled_main_upper(
        context, op, trip_count, unroll_factor, index_type, &scaled_step,
        &main_upper));
  } else {
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_scaled_step(
        context, op, step, unroll_factor, index_type, &scaled_step));
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_dynamic_split(
        &context->rewriter->builder, context->fact_table, op, step,
        unroll_factor, scaled_step, &main_upper));
  }

  loom_op_t* main_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter->builder, /*build_flags=*/0,
      loom_scf_for_lower_bound(op), main_upper, scaled_step, iter_args.values,
      iter_args.count, tied_results, tied_result_count,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &main_loop));

  loom_region_t* old_body = loom_scf_for_body(op);
  loom_block_t* old_block = loom_region_entry_block(old_body);
  loom_region_t* main_body = loom_scf_for_body(main_loop);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, main_loop, main_body);
  loom_block_t* main_block = loom_region_entry_block(main_body);
  for (uint16_t i = 0; i < old_block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
        context->rewriter, loom_block_arg_id(old_block, i),
        loom_block_arg_id(main_block, i)));
  }

  loom_value_id_t* main_carried_values = NULL;
  loom_value_id_t* final_main_carried_values = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, op->result_count, sizeof(*main_carried_values),
        (void**)&main_carried_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, op->result_count, sizeof(*final_main_carried_values),
        (void**)&final_main_carried_values));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      main_carried_values[i] = loom_block_arg_id(main_block, (uint16_t)(1 + i));
    }
  }

  loom_scf_unroll_trip_count_t tile_trip_count = {
      .count = unroll_factor,
      .step = step,
      .lower_kind = LOOM_SCF_UNROLL_LOWER_BOUND_DYNAMIC,
      .lower_value = loom_block_arg_id(main_block, 0),
      .lower_range_min = 0,
      .lower_range_max = 0,
  };
  iree_status_t status = loom_scf_unroll_emit_scheduled_tile(
      context, op, old_block, &tile_trip_count,
      trip_count ? LOOM_VALUE_ID_INVALID : loom_scf_for_upper_bound(op),
      main_carried_values, schedule, scratch_arena, final_main_carried_values);
  if (iree_status_is_ok(status) &&
      !loom_pass_has_error_diagnostics(context->pass)) {
    loom_op_t* main_yield = NULL;
    status = loom_scf_yield_build(&context->rewriter->builder,
                                  final_main_carried_values, op->result_count,
                                  op->location, &main_yield);
  }
  loom_builder_restore(&context->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  if (loom_pass_has_error_diagnostics(context->pass)) {
    return iree_ok_status();
  }

  const loom_value_id_t* replacement_values = NULL;
  if (op->result_count > 0) {
    replacement_values = loom_scf_for_results(main_loop).values;
  }

  if (tail_loop_required) {
    loom_value_slice_t tail_iter_args = {0};
    if (op->result_count > 0) {
      tail_iter_args = loom_scf_for_results(main_loop);
    }
    loom_value_id_t tail_step = loom_scf_for_step(op);
    if (step != 1) {
      IREE_RETURN_IF_ERROR(loom_scf_unroll_build_index_constant(
          context, op, step, index_type, &tail_step));
    }
    loom_op_t* tail_loop = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_for_build(
        &context->rewriter->builder, /*build_flags=*/0, main_upper,
        loom_scf_for_upper_bound(op), tail_step, tail_iter_args.values,
        tail_iter_args.count, tied_results, tied_result_count,
        /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
        /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &tail_loop));
    loom_region_t* tail_body = loom_scf_for_body(tail_loop);
    saved_ip = loom_builder_enter_region(&context->rewriter->builder, tail_loop,
                                         tail_body);
    IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_loop_body(
        context, old_block, loom_region_entry_block(tail_body), scratch_arena));
    loom_builder_restore(&context->rewriter->builder, saved_ip);
    if (op->result_count > 0) {
      replacement_values = loom_scf_for_results(tail_loop).values;
    }
  }

  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        context->rewriter, op, replacement_values, op->result_count,
        value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, op, replacement_values, op->result_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, op));
  }

  ++context->statistics->loops_unrolled;
  context->statistics->iterations_materialized += unroll_factor;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_partial_unroll_scheduled(
    loom_scf_unroll_context_t* context, loom_op_t* op, loom_op_t* yield,
    int64_t step, uint32_t unroll_factor,
    const loom_scf_unroll_trip_count_t* trip_count, bool tail_loop_required,
    loom_scf_for_unroll_schedule_t schedule, bool* out_changed) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(context->pass->arena->block_pool, &scratch_arena);
  iree_status_t status = loom_scf_unroll_partial_unroll_scheduled_with_arena(
      context, op, yield, step, unroll_factor, trip_count, tail_loop_required,
      schedule, &scratch_arena, out_changed);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_scf_unroll_try_unroll(
    loom_scf_unroll_context_t* context, loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  if (!loom_scf_unroll_policy_present(op)) {
    return loom_scf_unroll_append_policy_absent_report_detail(context, op);
  }
  if (loom_scf_for_pipeline_depth_is_present(op)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("pipeline"), 0,
        IREE_SV("consumed by pipeline-scf-for before unroll-scf-for"));
  }

  bool has_unroll_factor = loom_scf_for_unroll_factor_is_present(op);
  bool has_unroll_policy = !loom_attr_is_absent(
      loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX]);
  bool has_unroll_schedule = !loom_attr_is_absent(
      loom_op_attrs(op)[loom_scf_for_unroll_schedule_ATTR_INDEX]);
  if (has_unroll_factor && has_unroll_policy) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("unroll"), 2,
        IREE_SV("either bare unroll or unroll factor, not both"));
  }
  if (has_unroll_schedule && !has_unroll_factor && !has_unroll_policy) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("schedule"), 0,
        IREE_SV("paired with bare unroll or unroll factor"));
  }
  loom_scf_for_unroll_schedule_t unroll_schedule =
      LOOM_SCF_FOR_UNROLL_SCHEDULE_LINEAR;
  if (has_unroll_schedule) {
    unroll_schedule = loom_scf_for_unroll_schedule(op);
  }
  const bool uses_scheduled_tile =
      unroll_schedule != LOOM_SCF_FOR_UNROLL_SCHEDULE_LINEAR;

  loom_op_t* yield = NULL;
  if (!loom_scf_unroll_shape_is_supported(op, &yield)) {
    return loom_scf_unroll_emit_policy_error(
        context, op, IREE_SV("unroll"), 0,
        IREE_SV("single-block scf.for with matching scf.yield"));
  }

  int64_t unroll_factor = 0;
  uint32_t unroll_factor_u32 = 0;
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
      return loom_scf_unroll_clear_policy(context, op, yield, unroll_factor,
                                          unroll_schedule, out_changed);
    }
    if (unroll_factor > UINT32_MAX) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll_factor"), unroll_factor,
          IREE_SV("representable as uint32"));
    }
    unroll_factor_u32 = (uint32_t)unroll_factor;
  }

  loom_scf_unroll_trip_count_t trip_count = {0};
  loom_scf_unroll_trip_count_state_t trip_count_state =
      loom_scf_unroll_resolve_trip_count(context, op, &trip_count);
  int64_t step = trip_count.step;
  if (trip_count_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    if (!has_unroll_factor) {
      return loom_scf_unroll_emit_trip_count_error(context, op,
                                                   trip_count_state);
    }
    if (!loom_scf_unroll_exact_i64(context->fact_table, loom_scf_for_step(op),
                                   &step)) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("step"), 0,
          IREE_SV("positive exact static step"));
    }
    if (step <= 0) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("step"), step,
          IREE_SV("positive exact static step"));
    }
  }
  if (has_unroll_factor &&
      (trip_count_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT ||
       unroll_factor_u32 != trip_count.count)) {
    int64_t scaled_step = 0;
    const loom_scalar_type_t scalar_type = loom_type_element_type(
        loom_module_value_type(context->module, loom_scf_for_lower_bound(op)));
    const bool product_fits =
        iree_checked_mul_i64(step, unroll_factor, &scaled_step);
    const loom_value_facts_t scaled_facts =
        loom_value_facts_exact_i64(scaled_step);
    const bool carrier_fits =
        scalar_type == LOOM_SCALAR_TYPE_OFFSET
            ? loom_index_value_facts_fit_unsigned_target_carrier(
                  &context->fact_table->context, scalar_type, scaled_facts)
            : loom_index_value_facts_fit_signed_target_carrier(
                  &context->fact_table->context, scalar_type, scaled_facts);
    if (!product_fits || !carrier_fits) {
      return loom_scf_unroll_emit_policy_error(
          context, op, IREE_SV("unroll_factor"), unroll_factor,
          IREE_SV("step * unroll factor representable in the address carrier"));
    }
  }
  if (trip_count_state != LOOM_SCF_UNROLL_TRIP_COUNT_EXACT) {
    if (uses_scheduled_tile) {
      IREE_RETURN_IF_ERROR(loom_scf_unroll_append_partial_report_detail(
          context, op, unroll_factor, unroll_schedule, step, trip_count_state,
          &trip_count, LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_TAIL_LOOP));
      return loom_scf_unroll_partial_unroll_scheduled(
          context, op, yield, step, unroll_factor_u32,
          /*trip_count=*/NULL, /*tail_loop_required=*/true, unroll_schedule,
          out_changed);
    }
    IREE_RETURN_IF_ERROR(loom_scf_unroll_append_partial_report_detail(
        context, op, unroll_factor, unroll_schedule, step, trip_count_state,
        &trip_count, LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL));
    return loom_scf_unroll_partial_unroll(
        context, op, yield, step, unroll_factor_u32,
        LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL, out_changed);
  }
  if (has_unroll_factor && unroll_factor_u32 != trip_count.count) {
    loom_scf_unroll_partial_unroll_flags_t partial_flags =
        trip_count.count % unroll_factor_u32 == 0
            ? 0
            : LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_GUARD_TAIL;
    if (uses_scheduled_tile) {
      partial_flags = trip_count.count % unroll_factor_u32 == 0
                          ? 0
                          : LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_TAIL_LOOP;
      IREE_RETURN_IF_ERROR(loom_scf_unroll_append_partial_report_detail(
          context, op, unroll_factor, unroll_schedule, trip_count.step,
          trip_count_state, &trip_count, partial_flags));
      return loom_scf_unroll_partial_unroll_scheduled(
          context, op, yield, trip_count.step, unroll_factor_u32, &trip_count,
          iree_any_bit_set(partial_flags,
                           LOOM_SCF_UNROLL_PARTIAL_UNROLL_FLAG_TAIL_LOOP),
          unroll_schedule, out_changed);
    }
    IREE_RETURN_IF_ERROR(loom_scf_unroll_append_partial_report_detail(
        context, op, unroll_factor, unroll_schedule, trip_count.step,
        trip_count_state, &trip_count, partial_flags));
    return loom_scf_unroll_partial_unroll(context, op, yield, trip_count.step,
                                          unroll_factor_u32, partial_flags,
                                          out_changed);
  }
  IREE_RETURN_IF_ERROR(loom_scf_unroll_append_report_detail(
      context, op, has_unroll_factor ? IREE_SV("factor") : IREE_SV("bare"),
      unroll_schedule, has_unroll_factor ? unroll_factor : -1, &trip_count));

  if (uses_scheduled_tile) {
    return loom_scf_unroll_full_unroll_scheduled(context, op, &trip_count,
                                                 unroll_schedule, out_changed);
  }

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

  loom_ir_remap_t iteration_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_initialize_iteration_remap(context, &iteration_remap));
  loom_value_id_t induction_variable = body_block->arg_ids[0];
  const bool induction_variable_has_references =
      loom_scf_unroll_value_has_references(context->module, induction_variable);
  for (uint32_t ordinal = 0; ordinal < trip_count.count; ++ordinal) {
    if (induction_variable_has_references) {
      loom_value_id_t iteration_index = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scf_unroll_build_iteration_index(
          context, op, induction_variable, &trip_count, ordinal,
          &iteration_index));
      if (iteration_index == LOOM_VALUE_ID_INVALID) {
        return loom_scf_unroll_emit_policy_error(
            context, op, IREE_SV("unroll"), ordinal,
            IREE_SV("iteration index representable as i64"));
      }
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
          &iteration_remap, induction_variable, iteration_index));
    }
    IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_iteration(
        context, &iteration_remap, body_block, yield, ordinal, carried_values,
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

  // Fact computation is proportional to the entire function. Avoid it when
  // the terminal convergence iteration has no loops or when all remaining
  // loops have no unroll policy. Report mode still resolves trip counts for
  // policy-free loops to explain why they remain structured.
  if (loops.count == 0 ||
      (!collect_context.has_policy && !context->reports_enabled)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      context->pass, context->module,
      loom_pass_value_fact_scope_function_for_target(
          function, loom_target_function_version_target_facts(
                        context->pass->function_version)),
      &context->fact_table));

  for (iree_host_size_t i = 0;
       i < loops.count && !loom_pass_has_error_diagnostics(context->pass);
       ++i) {
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
  loom_rewriter_initialize(&rewriter, module, pass->arena);

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
  while (iree_status_is_ok(status) && changed &&
         !loom_pass_has_error_diagnostics(pass)) {
    changed = false;
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
