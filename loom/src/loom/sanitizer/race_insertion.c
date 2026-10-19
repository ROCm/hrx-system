// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/race_insertion.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ops/atomic.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/options.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/sanitizer/memory.h"
#include "loom/sanitizer/options.h"
#include "loom/sanitizer/site_location.h"
#include "loom/sanitizer/site_payload.h"

typedef struct loom_sanitizer_insert_race_observations_state_t {
  // Check classes enabled for this pass invocation.
  loom_sanitizer_checks_t checks;
  // True when checks were supplied explicitly.
  bool has_checks_option;
} loom_sanitizer_insert_race_observations_state_t;

#define LOOM_SANITIZER_INSERT_RACE_OBSERVATIONS_STATISTICS(V, statistics_type) \
  V(statistics_type, access_observations_inserted,                             \
    "access-observations-inserted",                                            \
    "Number of sanitizer race-access observations inserted.")                  \
  V(statistics_type, sync_observations_inserted, "sync-observations-inserted", \
    "Number of sanitizer.race.sync ops inserted.")

LOOM_PASS_STATISTICS_DEFINE(
    loom_sanitizer_insert_race_observations_statistics,
    loom_sanitizer_insert_race_observations_statistics_t,
    LOOM_SANITIZER_INSERT_RACE_OBSERVATIONS_STATISTICS)

static const loom_pass_option_def_t kSanitizerInsertRaceObservationsOptions[] =
    {
        {IREE_SVL("checks"),
         IREE_SVL("Sanitizer checks to insert: none, all, race, or tsan.")},
};

static const loom_pass_info_t
    loom_sanitizer_insert_race_observations_pass_info_storage = {
        .name = IREE_SVL("sanitizer-insert-race-observations"),
        .description =
            IREE_SVL("Insert sanitizer race observations for enabled checks."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kSanitizerInsertRaceObservationsOptions,
        .option_count = IREE_ARRAYSIZE(kSanitizerInsertRaceObservationsOptions),
        .statistic_layout =
            &loom_sanitizer_insert_race_observations_statistics_layout,
};

const loom_pass_info_t* loom_sanitizer_insert_race_observations_pass_info(
    void) {
  return &loom_sanitizer_insert_race_observations_pass_info_storage;
}

static iree_status_t loom_sanitizer_insert_race_observations_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_sanitizer_insert_race_observations_state_t* state =
      (loom_sanitizer_insert_race_observations_state_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("checks"))) {
    if (state->has_checks_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate option 'checks' for pass "
                              "'sanitizer-insert-race-observations'");
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
        value, IREE_SV("sanitizer-insert-race-observations option 'checks'"),
        &state->checks));
    state->has_checks_option = true;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'sanitizer-insert-race-observations'",
      (int)name.size, name.data);
}

iree_status_t loom_sanitizer_insert_race_observations_create(
    loom_pass_t* pass, iree_string_view_t options) {
  loom_sanitizer_insert_race_observations_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->checks = LOOM_SANITIZER_CHECK_RACE;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("checks"))) {
        IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
            option->string_value,
            IREE_SV("sanitizer-insert-race-observations option 'checks'"),
            &state->checks));
        state->has_checks_option = true;
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'sanitizer-insert-race-observations'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_sanitizer_insert_race_observations_parse_option,
            .user_data = state,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

static bool loom_sanitizer_race_check_enabled(const loom_pass_t* pass) {
  const loom_sanitizer_insert_race_observations_state_t* state =
      (const loom_sanitizer_insert_race_observations_state_t*)pass->state;
  const loom_sanitizer_checks_t enabled =
      state ? state->checks : LOOM_SANITIZER_CHECK_RACE;
  return iree_any_bit_set(enabled, LOOM_SANITIZER_CHECK_RACE);
}

static bool loom_sanitizer_i64_arrays_equal(loom_attribute_t lhs,
                                            loom_attribute_t rhs) {
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

static bool loom_sanitizer_preceded_by_matching_race_access(
    loom_op_t* op, loom_sanitizer_race_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices, bool atomic,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope) {
  loom_op_t* previous_op = op->prev_op;
  if (!previous_op || !loom_sanitizer_race_access_isa(previous_op) ||
      loom_sanitizer_race_access_kind(previous_op) != kind ||
      loom_sanitizer_race_access_view(previous_op) != view ||
      loom_sanitizer_race_access_atomic(previous_op) != atomic) {
    return false;
  }
  if (atomic && (loom_sanitizer_race_access_ordering(previous_op) != ordering ||
                 loom_sanitizer_race_access_scope(previous_op) != scope)) {
    return false;
  }
  return loom_sanitizer_value_slices_equal(
             loom_sanitizer_race_access_indices(previous_op), indices) &&
         loom_sanitizer_i64_arrays_equal(
             loom_sanitizer_race_access_static_indices(previous_op),
             static_indices);
}

static bool loom_sanitizer_matching_fragment_race_access(
    const loom_op_t* candidate, loom_sanitizer_race_fragment_access_kind_t kind,
    loom_value_id_t fragment, loom_value_id_t view, loom_value_slice_t indices,
    loom_attribute_t static_indices, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns,
    loom_sanitizer_race_fragment_access_role_t role) {
  return candidate && loom_sanitizer_race_fragment_access_isa(candidate) &&
         loom_sanitizer_race_fragment_access_kind(candidate) == kind &&
         loom_sanitizer_race_fragment_access_fragment(candidate) == fragment &&
         loom_sanitizer_race_fragment_access_view(candidate) == view &&
         loom_sanitizer_race_fragment_access_blocks(candidate) == blocks &&
         loom_sanitizer_race_fragment_access_rows(candidate) == rows &&
         loom_sanitizer_race_fragment_access_columns(candidate) == columns &&
         loom_sanitizer_race_fragment_access_role(candidate) == role &&
         loom_sanitizer_value_slices_equal(
             loom_sanitizer_race_fragment_access_indices(candidate), indices) &&
         loom_sanitizer_i64_arrays_equal(
             loom_sanitizer_race_fragment_access_static_indices(candidate),
             static_indices);
}

static bool loom_sanitizer_followed_by_matching_race_sync(
    loom_op_t* op, loom_value_fact_memory_space_t memory_space,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope) {
  loom_op_t* next_op = op->next_op;
  return next_op && loom_sanitizer_race_sync_isa(next_op) &&
         loom_sanitizer_race_sync_memory_space(next_op) == memory_space &&
         loom_sanitizer_race_sync_ordering(next_op) == ordering &&
         loom_sanitizer_race_sync_scope(next_op) == scope;
}

static iree_status_t loom_sanitizer_emit_unsupported_race_memory_observation(
    loom_pass_t* pass, const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t reason) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(pass->info->name),
      loom_param_string(reason),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_046,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_sanitizer_build_race_access(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_race_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices, bool atomic,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope,
    loom_location_id_t source_location, loom_op_t** out_op) {
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_RACE,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  const loom_sanitizer_race_access_build_flags_t build_flags =
      atomic ? LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_ORDERING |
                   LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_SCOPE
             : 0;
  return loom_sanitizer_race_access_build(
      &rewriter->builder, build_flags, kind, view, indices.values,
      indices.count, static_indices.i64_array, static_indices.count, atomic,
      ordering, scope, site_location, out_op);
}

static iree_status_t loom_sanitizer_build_fragment_race_access(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_race_fragment_access_kind_t kind, loom_value_id_t fragment,
    loom_value_id_t view, loom_value_slice_t indices,
    loom_attribute_t static_indices, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns,
    loom_sanitizer_race_fragment_access_role_t role,
    loom_location_id_t source_location, loom_op_t** out_op) {
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_RACE,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  const loom_sanitizer_race_fragment_access_build_flags_t build_flags =
      blocks != LOOM_VALUE_ID_INVALID
          ? LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_BUILD_FLAG_HAS_BLOCKS
          : 0;
  return loom_sanitizer_race_fragment_access_build(
      &rewriter->builder, build_flags, kind, fragment, view, indices.values,
      indices.count, static_indices.i64_array, static_indices.count, blocks,
      rows, columns, role, site_location, out_op);
}

static loom_sanitizer_race_fragment_access_role_t
loom_sanitizer_race_fragment_access_role_from_vector(loom_vector_role_t role) {
  switch (role) {
    case LOOM_VECTOR_ROLE_LHS:
      return LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_LHS;
    case LOOM_VECTOR_ROLE_RHS:
      return LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_RHS;
    case LOOM_VECTOR_ROLE_INIT:
      return LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_INIT;
    case LOOM_VECTOR_ROLE_RESULT:
      return LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_RESULT;
    default:
      IREE_ASSERT_UNREACHABLE("verified vector fragment role");
      return LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_LHS;
  }
}

static iree_status_t loom_sanitizer_try_instrument_fragment_race_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op, bool* out_handled) {
  *out_handled = false;
  loom_sanitizer_race_fragment_access_kind_t kind = 0;
  loom_value_id_t fragment = LOOM_VALUE_ID_INVALID;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  loom_value_id_t blocks = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rows = LOOM_VALUE_ID_INVALID;
  loom_value_id_t columns = LOOM_VALUE_ID_INVALID;
  loom_sanitizer_race_fragment_access_role_t role = 0;
  const loom_op_t* existing_observation = NULL;
  bool insert_after = false;
  if (loom_vector_fragment_load_isa(op)) {
    kind = LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_KIND_READ;
    fragment = loom_vector_fragment_load_result(op);
    view = loom_vector_fragment_load_view(op);
    indices = loom_vector_fragment_load_indices(op);
    static_indices = loom_vector_fragment_load_static_indices(op);
    blocks = loom_vector_fragment_load_blocks(op);
    rows = loom_vector_fragment_load_rows(op);
    columns = loom_vector_fragment_load_columns(op);
    role = loom_sanitizer_race_fragment_access_role_from_vector(
        loom_vector_fragment_load_role(op));
    existing_observation = op->next_op;
    insert_after = true;
  } else if (loom_vector_fragment_store_isa(op)) {
    kind = LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_KIND_WRITE;
    fragment = loom_vector_fragment_store_value(op);
    view = loom_vector_fragment_store_view(op);
    indices = loom_vector_fragment_store_indices(op);
    static_indices = loom_vector_fragment_store_static_indices(op);
    blocks = loom_vector_fragment_store_blocks(op);
    rows = loom_vector_fragment_store_rows(op);
    columns = loom_vector_fragment_store_columns(op);
    role = loom_sanitizer_race_fragment_access_role_from_vector(
        loom_vector_fragment_store_role(op));
    existing_observation = op->prev_op;
  } else {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_query_view_memory_space(rewriter, view, &memory_space) ||
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (loom_sanitizer_matching_fragment_race_access(
          existing_observation, kind, fragment, view, indices, static_indices,
          blocks, rows, columns, role)) {
    return iree_ok_status();
  }

  if (insert_after) {
    loom_builder_set_after(&rewriter->builder, op);
  } else {
    loom_builder_set_before(&rewriter->builder, op);
  }
  loom_op_t* observation_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_fragment_race_access(
      module, rewriter, kind, fragment, view, indices, static_indices, blocks,
      rows, columns, role, op->location, &observation_op));
  (void)observation_op;
  ++loom_sanitizer_insert_race_observations_statistics(pass)
        ->access_observations_inserted;
  return iree_ok_status();
}

// Traverse logical vector axes without cloning the race detector per element.
// Typed view indexing retains the original layout, including strided rows.
static iree_status_t loom_sanitizer_build_vector_race_access(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_race_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices,
    loom_type_t vector_type, loom_location_id_t location) {
  const uint8_t view_rank = (uint8_t)static_indices.count;
  const uint8_t vector_rank = loom_type_rank(vector_type);
  const uint8_t first_vector_axis = (uint8_t)(view_rank - vector_rank);
  int64_t observed_static_indices[LOOM_TYPE_MAX_RANK] = {0};
  loom_value_id_t observed_dynamic_indices[LOOM_TYPE_MAX_RANK] = {0};
  uint16_t source_dynamic_ordinal = 0;
  uint16_t observed_dynamic_count = 0;
  uint8_t loop_count = 0;
  loom_builder_ip_t saved_ips[LOOM_TYPE_MAX_RANK];
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    const int64_t origin = static_indices.i64_array[axis];
    loom_value_id_t origin_value = LOOM_VALUE_ID_INVALID;
    if (origin == INT64_MIN) {
      origin_value = indices.values[source_dynamic_ordinal++];
    }
    const int64_t extent = axis < first_vector_axis
                               ? 1
                               : loom_type_dim_static_size_at(
                                     vector_type, axis - first_vector_axis);
    if (extent == 1) {
      observed_static_indices[axis] = origin;
      if (origin == INT64_MIN) {
        observed_dynamic_indices[observed_dynamic_count++] = origin_value;
      }
      continue;
    }
    loom_op_t* lower = NULL;
    loom_op_t* upper = NULL;
    loom_op_t* step = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &rewriter->builder, loom_attr_i64(0), index_type, location, &lower));
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(&rewriter->builder, loom_attr_i64(extent),
                                  index_type, location, &upper));
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &rewriter->builder, loom_attr_i64(1), index_type, location, &step));
    if (origin_value == LOOM_VALUE_ID_INVALID && origin != 0) {
      loom_op_t* origin_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_constant_build(&rewriter->builder, loom_attr_i64(origin),
                                    index_type, location, &origin_op));
      origin_value = loom_index_constant_result(origin_op);
    }
    loom_op_t* loop = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_for_build(
        &rewriter->builder, /*build_flags=*/0,
        loom_index_constant_result(lower), loom_index_constant_result(upper),
        loom_index_constant_result(step), /*init=*/NULL, 0,
        /*result_types=*/NULL, NULL, 0,
        /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
        /*unroll_policy=*/0, /*unroll_schedule=*/0, location, &loop));
    saved_ips[loop_count++] = loom_builder_enter_region(
        &rewriter->builder, loop, loom_scf_for_body(loop));
    loom_value_id_t index =
        loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
    if (origin_value != LOOM_VALUE_ID_INVALID) {
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &rewriter->builder, origin_value, index, index_type, location, &add));
      index = loom_index_add_result(add);
    }
    observed_static_indices[axis] = INT64_MIN;
    observed_dynamic_indices[observed_dynamic_count++] = index;
  }
  loom_op_t* observation = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_race_access(
      module, rewriter, kind, view,
      (loom_value_slice_t){observed_dynamic_indices, observed_dynamic_count},
      loom_attr_i64_array(observed_static_indices, view_rank),
      /*atomic=*/false, /*ordering=*/0, /*scope=*/0, location, &observation));
  while (loop_count) {
    loom_op_t* yield_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(&rewriter->builder, NULL, 0, location, &yield_op));
    --loop_count;
    loom_builder_restore(&rewriter->builder, saved_ips[loop_count]);
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_vector_race_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op, bool* out_handled) {
  *out_handled = false;
  loom_sanitizer_race_access_kind_t kind = 0;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  loom_type_t vector_type = {0};
  if (loom_vector_load_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ;
    view = loom_vector_load_view(op);
    indices = loom_vector_load_indices(op);
    static_indices = loom_vector_load_static_indices(op);
    vector_type = loom_module_value_type(module, loom_vector_load_result(op));
  } else if (loom_vector_store_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE;
    view = loom_vector_store_view(op);
    indices = loom_vector_store_indices(op);
    static_indices = loom_vector_store_static_indices(op);
    vector_type = loom_module_value_type(module, loom_vector_store_value(op));
  } else {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_query_view_memory_space(rewriter, view, &memory_space) ||
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (!loom_type_is_all_static(vector_type)) {
    return loom_sanitizer_emit_unsupported_race_memory_observation(
        pass, module, op,
        IREE_SV("local-memory vector operation has dynamic lane count"));
  }
  if (loom_type_has_static_zero_extent(vector_type)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_vector_race_access(
      module, rewriter, kind, view, indices, static_indices, vector_type,
      op->location));
  ++loom_sanitizer_insert_race_observations_statistics(pass)
        ->access_observations_inserted;
  return iree_ok_status();
}

static bool loom_sanitizer_unsupported_vector_memory_view(
    const loom_op_t* op, loom_value_id_t* out_view) {
  *out_view = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK:
      *out_view = loom_vector_load_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_STORE_MASK:
      *out_view = loom_vector_store_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      *out_view = loom_vector_load_expand_view(op);
      return true;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      *out_view = loom_vector_store_compress_view(op);
      return true;
    case LOOM_OP_VECTOR_GATHER:
      *out_view = loom_vector_gather_view(op);
      return true;
    case LOOM_OP_VECTOR_SCATTER:
      *out_view = loom_vector_scatter_view(op);
      return true;
    case LOOM_OP_VECTOR_GATHER_MASK:
      *out_view = loom_vector_gather_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      *out_view = loom_vector_scatter_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE:
      *out_view = loom_vector_atomic_reduce_view(op);
      return true;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      *out_view = loom_vector_atomic_reduce_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_ATOMIC_RMW:
      *out_view = loom_vector_atomic_rmw_view(op);
      return true;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      *out_view = loom_vector_atomic_rmw_mask_view(op);
      return true;
    case LOOM_OP_VECTOR_ATOMIC_CMPXCHG:
      *out_view = loom_vector_atomic_cmpxchg_view(op);
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_sanitizer_reject_unsupported_vector_race_access_op(
    loom_pass_t* pass, const loom_module_t* module,
    const loom_rewriter_t* rewriter, const loom_op_t* op, bool* out_handled) {
  *out_handled = false;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  if (!loom_sanitizer_unsupported_vector_memory_view(op, &view)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_query_view_memory_space(rewriter, view, &memory_space) ||
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  return loom_sanitizer_emit_unsupported_race_memory_observation(
      pass, module, op,
      IREE_SV("local-memory vector operation requires lane-activity-aware "
              "race observation"));
}

static iree_status_t loom_sanitizer_try_instrument_race_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_sanitizer_try_instrument_fragment_race_access_op(
      pass, module, rewriter, op, &handled));
  if (handled) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_sanitizer_try_instrument_vector_race_access_op(
      pass, module, rewriter, op, &handled));
  if (handled) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_sanitizer_reject_unsupported_vector_race_access_op(
      pass, module, rewriter, op, &handled));
  if (handled) {
    return iree_ok_status();
  }

  loom_sanitizer_race_access_kind_t kind = 0;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  bool atomic = false;
  loom_atomic_ordering_t ordering = 0;
  loom_atomic_scope_t scope = 0;
  if (loom_view_load_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ;
    view = loom_view_load_view(op);
    indices = loom_view_load_indices(op);
    static_indices = loom_view_load_static_indices(op);
  } else if (loom_view_store_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE;
    view = loom_view_store_view(op);
    indices = loom_view_store_indices(op);
    static_indices = loom_view_store_static_indices(op);
  } else if (loom_view_atomic_reduce_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_reduce_view(op);
    indices = loom_view_atomic_reduce_indices(op);
    static_indices = loom_view_atomic_reduce_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_reduce_ordering(op);
    scope = loom_view_atomic_reduce_scope(op);
  } else if (loom_view_atomic_rmw_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_rmw_view(op);
    indices = loom_view_atomic_rmw_indices(op);
    static_indices = loom_view_atomic_rmw_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_rmw_ordering(op);
    scope = loom_view_atomic_rmw_scope(op);
  } else if (loom_view_atomic_cmpxchg_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_cmpxchg_view(op);
    indices = loom_view_atomic_cmpxchg_indices(op);
    static_indices = loom_view_atomic_cmpxchg_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_cmpxchg_success_ordering(op);
    scope = loom_view_atomic_cmpxchg_scope(op);
  } else {
    return iree_ok_status();
  }

  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_query_view_memory_space(rewriter, view, &memory_space) ||
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (loom_sanitizer_preceded_by_matching_race_access(
          op, kind, view, indices, static_indices, atomic, ordering, scope)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* observe_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_race_access(
      module, rewriter, kind, view, indices, static_indices, atomic, ordering,
      scope, op->location, &observe_op));
  (void)observe_op;
  loom_sanitizer_insert_race_observations_statistics_t* statistics =
      loom_sanitizer_insert_race_observations_statistics(pass);
  ++statistics->access_observations_inserted;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_race_sync_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  if (!loom_kernel_barrier_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_fact_memory_space_t memory_space =
      loom_kernel_barrier_memory_space(op);
  const loom_atomic_ordering_t ordering = loom_kernel_barrier_ordering(op);
  const loom_atomic_scope_t scope = loom_kernel_barrier_scope(op);
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (scope != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sanitizer race observation does not yet support non-workgroup "
        "kernel.barrier scope");
  }
  if (loom_sanitizer_followed_by_matching_race_sync(op, memory_space, ordering,
                                                    scope)) {
    return iree_ok_status();
  }

  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_RACE,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, op->location, &payload, &site_location));
  loom_builder_set_after(&rewriter->builder, op);
  loom_op_t* observe_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_race_sync_build(&rewriter->builder, memory_space, scope,
                                     ordering, site_location, &observe_op));
  (void)observe_op;
  loom_sanitizer_insert_race_observations_statistics_t* statistics =
      loom_sanitizer_insert_race_observations_statistics(pass);
  ++statistics->sync_observations_inserted;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_insert_race_observations_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_sanitizer_race_check_enabled(pass)) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
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
  while (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) {
      break;
    }
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    status = loom_sanitizer_try_instrument_race_access_op(pass, module,
                                                          &rewriter, op);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (loom_pass_has_error_diagnostics(pass)) {
      continue;
    }
    status =
        loom_sanitizer_try_instrument_race_sync_op(pass, module, &rewriter, op);
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
