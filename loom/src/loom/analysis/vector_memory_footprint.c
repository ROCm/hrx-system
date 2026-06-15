// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/vector_memory_footprint.h"

#include <inttypes.h>
#include <stdio.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// State and helpers
//===----------------------------------------------------------------------===//

typedef struct loom_vector_memory_footprint_state_t {
  // Module whose function body is being checked.
  loom_module_t* module;

  // Caller-owned verification options.
  const loom_vector_memory_footprint_options_t* options;

  // Per-function value facts visible to footprint proof.
  const loom_value_fact_table_t* fact_table;

  // Per-function symbolic expression context sharing the fact table above.
  loom_symbolic_expr_context_t expression_context;

  // Result object receiving counters and proof failures.
  loom_vector_memory_footprint_result_t* result;

  // True once the current access has emitted a proof failure.
  bool current_access_failed;
} loom_vector_memory_footprint_state_t;

typedef struct loom_vector_memory_footprint_access_t {
  // Memory op being checked.
  const loom_op_t* op;

  // View value addressed by the memory op.
  loom_value_id_t view;

  // Vector payload type loaded, stored, or atomically updated.
  loom_type_t vector_type;

  // Full-rank logical origin indices with INT64_MIN entries for dynamic axes.
  loom_attribute_t static_indices;

  // Dynamic logical origin index values matching INT64_MIN static slots.
  loom_value_slice_t dynamic_indices;

  // Optional vector mask; invalid for unmasked memory ops.
  loom_value_id_t mask;

  // Optional lane offset vector for gather, scatter, and vector atomics.
  loom_value_id_t offsets;

  // Whether offsets participates in the logical footprint.
  bool has_offsets;
} loom_vector_memory_footprint_access_t;

static bool loom_vector_memory_footprint_should_suppress_error(
    const loom_vector_memory_footprint_state_t* state) {
  return state->options->max_errors != 0 &&
         state->result->error_count >= state->options->max_errors;
}

static iree_string_view_t loom_vector_memory_footprint_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) {
    return IREE_SV("<unknown>");
  }
  return loom_op_vtable_name(vtable);
}

static iree_string_view_t loom_vector_memory_footprint_origin_name(
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, char* origin_buffer,
    iree_host_size_t origin_buffer_capacity) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return IREE_SV("<unknown>");
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index != INT64_MIN) {
      if (i == axis) {
        int length = snprintf(origin_buffer, origin_buffer_capacity, "%" PRId64,
                              static_index);
        if (length <= 0 || (iree_host_size_t)length >= origin_buffer_capacity) {
          return IREE_SV("<unknown>");
        }
        return iree_make_string_view(origin_buffer, (iree_host_size_t)length);
      }
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return IREE_SV("<unknown>");
      }
      int length = snprintf(origin_buffer, origin_buffer_capacity, "%%%u",
                            (unsigned)dynamic_indices.values[dynamic_ordinal]);
      if (length <= 0 || (iree_host_size_t)length >= origin_buffer_capacity) {
        return IREE_SV("<unknown>");
      }
      return iree_make_string_view(origin_buffer, (iree_host_size_t)length);
    }
    ++dynamic_ordinal;
  }
  return IREE_SV("<unknown>");
}

static bool loom_vector_memory_footprint_origin_value(
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return false;
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index != INT64_MIN) {
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return false;
      }
      *out_value = dynamic_indices.values[dynamic_ordinal];
      return *out_value != LOOM_VALUE_ID_INVALID;
    }
    ++dynamic_ordinal;
  }
  return false;
}

static bool loom_vector_memory_footprint_lookup_config_decl_bound(
    const loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access, uint8_t view_axis,
    iree_string_view_t* out_config_key) {
  *out_config_key = iree_string_view_empty();
  if (access->view == LOOM_VALUE_ID_INVALID ||
      access->view >= state->module->values.count) {
    return false;
  }

  loom_type_t view_type = loom_module_value_type(state->module, access->view);
  if (view_axis >= loom_type_rank(view_type) ||
      !loom_type_dim_is_dynamic_at(view_type, view_axis)) {
    return false;
  }
  loom_value_id_t bound_id = loom_type_dim_value_id_at(view_type, view_axis);
  if (bound_id == LOOM_VALUE_ID_INVALID ||
      bound_id >= state->module->values.count) {
    return false;
  }

  const loom_value_t* bound_value = loom_module_value(state->module, bound_id);
  if (loom_value_is_block_arg(bound_value)) {
    return false;
  }
  const loom_op_t* bound_op = loom_value_def_op(bound_value);
  if (!bound_op || !loom_config_get_isa(bound_op)) {
    return false;
  }

  loom_symbol_ref_t config_ref = loom_config_get_config(bound_op);
  if (!loom_symbol_ref_is_valid(config_ref) || config_ref.module_id != 0 ||
      config_ref.symbol_id >= state->module->symbols.count) {
    return false;
  }
  const loom_symbol_t* config_symbol =
      &state->module->symbols.entries[config_ref.symbol_id];
  if (!loom_symbol_implements(config_symbol, LOOM_SYMBOL_INTERFACE_CONFIG) ||
      !config_symbol->defining_op ||
      !loom_config_decl_isa(config_symbol->defining_op) ||
      config_symbol->name_id >= state->module->strings.count) {
    return false;
  }

  *out_config_key = state->module->strings.entries[config_symbol->name_id];
  return true;
}

static iree_status_t loom_vector_memory_footprint_fail_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access, uint8_t view_axis,
    int64_t vector_axis, const loom_error_def_t* error) {
  char origin_buffer[32];
  iree_string_view_t origin = loom_vector_memory_footprint_origin_name(
      access->static_indices, access->dynamic_indices, view_axis, origin_buffer,
      IREE_ARRAYSIZE(origin_buffer));
  state->current_access_failed = true;
  if (loom_vector_memory_footprint_should_suppress_error(state)) {
    return iree_ok_status();
  }
  ++state->result->error_count;
  iree_string_view_t op_name =
      loom_vector_memory_footprint_op_name(state->module, access->op);
  if (error == LOOM_ERR_SUBRANGE_010 || error == LOOM_ERR_SUBRANGE_011) {
    iree_string_view_t config_key = iree_string_view_empty();
    if (loom_vector_memory_footprint_lookup_config_decl_bound(
            state, access, view_axis, &config_key)) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_i64(view_axis),
          loom_param_i64(vector_axis),
          loom_param_string(origin),
          loom_param_string(config_key),
          loom_param_string(
              IREE_SV("config_decl.where.vector_footprint_upper_bound")),
      };
      loom_diagnostic_emission_t emission = {
          .op = access->op,
          .error = LOOM_ERR_SUBRANGE_018,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      return iree_diagnostic_emit(state->options->emitter, &emission);
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_i64(view_axis),
      loom_param_i64(vector_axis),
      loom_param_string(origin),
  };
  loom_diagnostic_emission_t emission = {
      .op = access->op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_vector_memory_footprint_fail_unresolved_layout(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  state->current_access_failed = true;
  if (loom_vector_memory_footprint_should_suppress_error(state)) {
    return iree_ok_status();
  }
  ++state->result->error_count;
  iree_string_view_t op_name =
      loom_vector_memory_footprint_op_name(state->module, access->op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
  };
  loom_diagnostic_emission_t emission = {
      .op = access->op,
      .error = LOOM_ERR_SUBRANGE_007,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_vector_memory_footprint_fail_unsupported_offset_rank(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  state->current_access_failed = true;
  if (loom_vector_memory_footprint_should_suppress_error(state)) {
    return iree_ok_status();
  }
  ++state->result->error_count;
  iree_string_view_t op_name =
      loom_vector_memory_footprint_op_name(state->module, access->op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
  };
  loom_diagnostic_emission_t emission = {
      .op = access->op,
      .error = LOOM_ERR_SUBRANGE_017,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_vector_memory_footprint_fail_linear_span(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_error_def_t* error) {
  iree_string_view_t op_name =
      loom_vector_memory_footprint_op_name(state->module, access->op);
  state->current_access_failed = true;
  if (loom_vector_memory_footprint_should_suppress_error(state)) {
    return iree_ok_status();
  }
  ++state->result->error_count;
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
  };
  loom_diagnostic_emission_t emission = {
      .op = access->op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static bool loom_vector_memory_footprint_facts_i64_bounds(
    loom_value_facts_t facts, int64_t* out_lower, int64_t* out_upper) {
  if (loom_value_facts_is_float(facts)) {
    return false;
  }
  if (facts.range_lo == INT64_MIN || facts.range_hi == INT64_MAX) {
    return false;
  }
  *out_lower = facts.range_lo;
  *out_upper = facts.range_hi;
  return true;
}

static bool loom_vector_memory_footprint_facts_exact_i64(
    loom_value_facts_t facts, int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static int64_t loom_vector_memory_footprint_vector_axis(
    const loom_vector_memory_access_t* access, uint8_t view_axis) {
  return view_axis >= access->first_vector_axis
             ? view_axis - access->first_vector_axis
             : -1;
}

static iree_status_t loom_vector_memory_footprint_expr_from_value(
    loom_vector_memory_footprint_state_t* state, loom_value_id_t value,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_from_value(&state->expression_context, value,
                                       out_expression);
}

static iree_status_t loom_vector_memory_footprint_expr_add(
    loom_vector_memory_footprint_state_t* state,
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add(&state->expression_context, left, right,
                                out_expression);
}

static iree_status_t loom_vector_memory_footprint_expr_mul_i64(
    loom_vector_memory_footprint_state_t* state,
    const loom_symbolic_expr_t* expression, int64_t multiplier,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_mul_i64(&state->expression_context, expression,
                                    multiplier, out_expression);
}

static iree_status_t loom_vector_memory_footprint_expr_add_i64(
    loom_vector_memory_footprint_state_t* state,
    const loom_symbolic_expr_t* expression, int64_t value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t constant = {0};
  loom_symbolic_expr_constant(value, &constant);
  return loom_vector_memory_footprint_expr_add(state, expression, &constant,
                                               out_expression);
}

static bool loom_vector_memory_footprint_expr_exact_i64(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) {
    return false;
  }
  *out_value = expression->constant;
  return true;
}

static iree_status_t loom_vector_memory_footprint_prove_le(
    loom_vector_memory_footprint_state_t* state,
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    bool* out_proven) {
  loom_symbolic_proof_result_t result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(&state->expression_context,
                                                   left, right, &result));
  *out_proven = result == LOOM_SYMBOLIC_PROOF_TRUE;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_prove_equal(
    loom_vector_memory_footprint_state_t* state,
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    bool* out_proven) {
  bool left_le_right = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(state, left, right,
                                                             &left_le_right));
  if (!left_le_right) {
    *out_proven = false;
    return iree_ok_status();
  }
  bool right_le_left = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(state, right, left,
                                                             &right_le_left));
  *out_proven = right_le_left;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_dim_expr(
    loom_vector_memory_footprint_state_t* state, loom_type_t type, uint8_t axis,
    loom_symbolic_expr_t* out_expression) {
  if (!loom_type_dim_is_dynamic_at(type, axis)) {
    loom_symbolic_expr_constant(loom_type_dim_static_size_at(type, axis),
                                out_expression);
    return iree_ok_status();
  }
  return loom_vector_memory_footprint_expr_from_value(
      state, loom_type_dim_value_id_at(type, axis), out_expression);
}

static iree_status_t loom_vector_memory_footprint_origin_expr(
    loom_vector_memory_footprint_state_t* state,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return iree_ok_status();
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index != INT64_MIN) {
      if (i == axis) {
        loom_symbolic_expr_constant(static_index, out_expression);
        *out_known = true;
        return iree_ok_status();
      }
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_from_value(
          state, dynamic_indices.values[dynamic_ordinal], out_expression));
      *out_known = true;
      return iree_ok_status();
    }
    ++dynamic_ordinal;
  }
  return iree_ok_status();
}

static bool loom_vector_memory_footprint_mask_uniform_false(
    loom_vector_memory_footprint_state_t* state, loom_value_id_t mask) {
  if (mask == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(state->fact_table, mask);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&state->fact_table->context, facts,
                                             &uniform)) {
    int64_t value = 0;
    return loom_vector_memory_footprint_facts_exact_i64(uniform.element,
                                                        &value) &&
           value == 0;
  }
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(&state->fact_table->context,
                                                 facts, &lanes)) {
    return false;
  }
  if (lanes.count == 0) {
    return true;
  }
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    int64_t value = 0;
    if (!loom_vector_memory_footprint_facts_exact_i64(lanes.lanes[i], &value) ||
        value != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_memory_footprint_value_defines_mask_range(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t** out_op) {
  *out_op = NULL;
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (!op || !loom_vector_mask_range_isa(op)) {
    return false;
  }
  *out_op = op;
  return true;
}

static iree_status_t loom_vector_memory_footprint_tail_end_expr(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    const loom_symbolic_expr_t* origin, loom_symbolic_expr_t* out_end,
    bool* out_tail) {
  *out_tail = false;
  if (access->mask == LOOM_VALUE_ID_INVALID ||
      memory_access->vector_rank != 1 ||
      view_axis != memory_access->first_vector_axis) {
    return iree_ok_status();
  }

  const loom_op_t* mask_op = NULL;
  if (!loom_vector_memory_footprint_value_defines_mask_range(
          state->module, access->mask, &mask_op)) {
    return iree_ok_status();
  }

  loom_value_facts_t step_facts = loom_value_fact_table_lookup(
      state->fact_table, loom_vector_mask_range_step(mask_op));
  int64_t step = 0;
  if (!loom_vector_memory_footprint_facts_exact_i64(step_facts, &step) ||
      step != 1) {
    return iree_ok_status();
  }

  loom_symbolic_expr_t lower_bound = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_from_value(
      state, loom_vector_mask_range_lower_bound(mask_op), &lower_bound));
  bool lower_matches_origin = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_equal(
      state, &lower_bound, origin, &lower_matches_origin));
  if (!lower_matches_origin) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_from_value(
      state, loom_vector_mask_range_upper_bound(mask_op), out_end));
  *out_tail = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_axis_extent_expr(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    loom_symbolic_expr_t* out_expression) {
  if (view_axis < access->first_vector_axis) {
    loom_symbolic_expr_constant(1, out_expression);
    return iree_ok_status();
  }
  uint8_t vector_axis = view_axis - access->first_vector_axis;
  return loom_vector_memory_footprint_dim_expr(state, access->vector_type,
                                               vector_axis, out_expression);
}

static iree_status_t loom_vector_memory_footprint_axis_has_unit_extent(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    bool* out_unit_extent) {
  *out_unit_extent = false;
  loom_symbolic_expr_t extent = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_axis_extent_expr(
      state, access, view_axis, &extent));
  int64_t extent_value = 0;
  *out_unit_extent =
      loom_vector_memory_footprint_expr_exact_i64(&extent, &extent_value) &&
      extent_value == 1;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_prove_unit_axis_upper_bound(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    bool* out_proven) {
  *out_proven = false;
  bool unit_extent = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_axis_has_unit_extent(
      state, memory_access, view_axis, &unit_extent));
  if (!unit_extent ||
      !loom_type_dim_is_dynamic_at(memory_access->view_type, view_axis)) {
    return iree_ok_status();
  }

  loom_value_id_t origin_value = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_memory_footprint_origin_value(access->static_indices,
                                                 access->dynamic_indices,
                                                 view_axis, &origin_value)) {
    return iree_ok_status();
  }
  loom_value_id_t bound_value =
      loom_type_dim_value_id_at(memory_access->view_type, view_axis);
  if (bound_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_value_relation(
      &state->expression_context, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      origin_value, bound_value, &relation));
  *out_proven = relation == LOOM_SYMBOLIC_PROOF_TRUE;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_direct_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis) {
  loom_symbolic_expr_t origin = {0};
  bool origin_known = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_origin_expr(
      state, access->static_indices, access->dynamic_indices, view_axis,
      &origin, &origin_known));
  if (!origin_known) {
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis,
        loom_vector_memory_footprint_vector_axis(memory_access, view_axis),
        LOOM_ERR_SUBRANGE_008);
  }

  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  bool lower_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &zero, &origin, &lower_proven));
  if (!lower_proven) {
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis,
        loom_vector_memory_footprint_vector_axis(memory_access, view_axis),
        LOOM_ERR_SUBRANGE_009);
  }

  loom_symbolic_expr_t end = {0};
  bool has_tail_end = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_tail_end_expr(
      state, access, memory_access, view_axis, &origin, &end, &has_tail_end));
  if (!has_tail_end) {
    loom_symbolic_expr_t extent = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_axis_extent_expr(
        state, memory_access, view_axis, &extent));
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_expr_add(state, &origin, &extent, &end));
  }

  loom_symbolic_expr_t bound = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_dim_expr(
      state, memory_access->view_type, view_axis, &bound));
  bool upper_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &end, &bound, &upper_proven));
  if (!upper_proven) {
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_prove_unit_axis_upper_bound(
            state, access, memory_access, view_axis, &upper_proven));
  }
  if (!upper_proven) {
    bool unit_extent = false;
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_axis_has_unit_extent(
        state, memory_access, view_axis, &unit_extent));
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis,
        loom_vector_memory_footprint_vector_axis(memory_access, view_axis),
        unit_extent ? LOOM_ERR_SUBRANGE_011 : LOOM_ERR_SUBRANGE_010);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_direct(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  loom_type_t view_type = loom_module_value_type(state->module, access->view);
  loom_vector_memory_access_t memory_access = {0};
  const loom_fact_context_t* fact_context =
      state->fact_table ? &state->fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, state->module,
                                          view_type, access->vector_type,
                                          &memory_access)) {
    return iree_ok_status();
  }
  if (memory_access.layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN) {
    return loom_vector_memory_footprint_fail_unresolved_layout(state, access);
  }

  for (uint8_t axis = 0; axis < memory_access.view_rank; ++axis) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_check_direct_axis(
        state, access, &memory_access, axis));
    if (state->current_access_failed) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_origin_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis) {
  loom_symbolic_expr_t origin = {0};
  bool origin_known = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_origin_expr(
      state, access->static_indices, access->dynamic_indices, view_axis,
      &origin, &origin_known));
  if (!origin_known) {
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis, /*vector_axis=*/-1, LOOM_ERR_SUBRANGE_008);
  }

  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  bool lower_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &zero, &origin, &lower_proven));
  if (!lower_proven) {
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis, /*vector_axis=*/-1, LOOM_ERR_SUBRANGE_009);
  }

  loom_symbolic_expr_t exclusive_end = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
      state, &origin, 1, &exclusive_end));
  loom_symbolic_expr_t bound = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_dim_expr(
      state, memory_access->view_type, view_axis, &bound));
  bool upper_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &exclusive_end, &bound, &upper_proven));
  if (!upper_proven) {
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_prove_unit_axis_upper_bound(
            state, access, memory_access, view_axis, &upper_proven));
  }
  if (!upper_proven) {
    return loom_vector_memory_footprint_fail_axis(
        state, access, view_axis, /*vector_axis=*/-1, LOOM_ERR_SUBRANGE_011);
  }
  return iree_ok_status();
}

static bool loom_vector_memory_footprint_offset_bounds_from_facts(
    loom_vector_memory_footprint_state_t* state, loom_value_facts_t facts,
    int64_t* out_lower, int64_t* out_upper) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&state->fact_table->context, facts,
                                             &uniform)) {
    return loom_vector_memory_footprint_facts_i64_bounds(uniform.element,
                                                         out_lower, out_upper);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(&state->fact_table->context,
                                                 facts, &lanes)) {
    return false;
  }
  if (lanes.count == 0) {
    *out_lower = 0;
    *out_upper = -1;
    return true;
  }
  int64_t lower = INT64_MAX;
  int64_t upper = INT64_MIN;
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    int64_t lane_lower = 0;
    int64_t lane_upper = 0;
    if (!loom_vector_memory_footprint_facts_i64_bounds(
            lanes.lanes[i], &lane_lower, &lane_upper)) {
      return false;
    }
    if (lane_lower < lower) {
      lower = lane_lower;
    }
    if (lane_upper > upper) {
      upper = lane_upper;
    }
  }
  *out_lower = lower;
  *out_upper = upper;
  return true;
}

static bool loom_vector_memory_footprint_value_defines_iota(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t** out_op) {
  *out_op = NULL;
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (!op || !loom_vector_iota_isa(op)) {
    return false;
  }
  *out_op = op;
  return true;
}

static iree_status_t loom_vector_memory_footprint_iota_offset_bounds(
    loom_vector_memory_footprint_state_t* state, const loom_op_t* iota_op,
    loom_type_t offsets_type, loom_symbolic_expr_t* out_lower,
    loom_symbolic_expr_t* out_upper, bool* out_known) {
  *out_known = false;
  if (loom_type_rank(offsets_type) != 1) {
    return iree_ok_status();
  }

  loom_value_facts_t step_facts = loom_value_fact_table_lookup(
      state->fact_table, loom_vector_iota_step(iota_op));
  int64_t step = 0;
  if (!loom_vector_memory_footprint_facts_exact_i64(step_facts, &step)) {
    return iree_ok_status();
  }

  loom_symbolic_expr_t base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_from_value(
      state, loom_vector_iota_base(iota_op), &base));
  if (!loom_type_dim_is_dynamic_at(offsets_type, 0)) {
    int64_t lane_count = loom_type_dim_static_size_at(offsets_type, 0);
    if (lane_count <= 0) {
      loom_symbolic_expr_constant(0, out_lower);
      loom_symbolic_expr_constant(-1, out_upper);
      *out_known = true;
      return iree_ok_status();
    }
    int64_t last_delta = 0;
    if (!iree_checked_mul_i64(lane_count - 1, step, &last_delta)) {
      return iree_ok_status();
    }
    loom_symbolic_expr_t last = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
        state, &base, last_delta, &last));
    if (step >= 0) {
      *out_lower = base;
      *out_upper = last;
    } else {
      *out_lower = last;
      *out_upper = base;
    }
    *out_known = true;
    return iree_ok_status();
  }

  loom_value_facts_t count_facts = loom_value_fact_table_lookup(
      state->fact_table, loom_type_dim_value_id_at(offsets_type, 0));
  if (!loom_value_facts_is_positive(count_facts) || step < 0) {
    return iree_ok_status();
  }
  loom_symbolic_expr_t count = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_memory_footprint_dim_expr(state, offsets_type, 0, &count));
  loom_symbolic_expr_t count_minus_one = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
      state, &count, -1, &count_minus_one));
  loom_symbolic_expr_t scaled = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_mul_i64(
      state, &count_minus_one, step, &scaled));
  loom_symbolic_expr_t last = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_memory_footprint_expr_add(state, &base, &scaled, &last));
  *out_lower = base;
  *out_upper = last;
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_offset_bounds(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    loom_symbolic_expr_t* out_lower, loom_symbolic_expr_t* out_upper,
    bool* out_known) {
  *out_known = false;
  loom_type_t offsets_type =
      loom_module_value_type(state->module, access->offsets);
  const loom_op_t* iota_op = NULL;
  if (loom_vector_memory_footprint_value_defines_iota(
          state->module, access->offsets, &iota_op)) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_iota_offset_bounds(
        state, iota_op, offsets_type, out_lower, out_upper, out_known));
    if (*out_known) {
      return iree_ok_status();
    }
  }

  loom_value_facts_t facts =
      loom_value_fact_table_lookup(state->fact_table, access->offsets);
  int64_t lower = 0;
  int64_t upper = 0;
  if (!loom_vector_memory_footprint_offset_bounds_from_facts(state, facts,
                                                             &lower, &upper)) {
    return iree_ok_status();
  }
  loom_symbolic_expr_constant(lower, out_lower);
  loom_symbolic_expr_constant(upper, out_upper);
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_origin_element_offset_expr(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  loom_symbolic_expr_t sum = {0};
  loom_symbolic_expr_constant(0, &sum);
  for (uint8_t view_axis = 0; view_axis < memory_access->view_rank;
       ++view_axis) {
    int64_t stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(memory_access, view_axis,
                                                      &stride)) {
      return iree_ok_status();
    }
    if (stride == 0) {
      continue;
    }

    loom_symbolic_expr_t origin = {0};
    bool origin_known = false;
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_origin_expr(
        state, access->static_indices, access->dynamic_indices, view_axis,
        &origin, &origin_known));
    if (!origin_known) {
      return iree_ok_status();
    }

    loom_symbolic_expr_t contribution = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_mul_i64(
        state, &origin, stride, &contribution));
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
        state, &sum, &contribution, &sum));
  }
  *out_expression = sum;
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_storage_element_span_expr(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_access_t* memory_access,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  loom_symbolic_expr_t maximum_element_offset = {0};
  loom_symbolic_expr_constant(0, &maximum_element_offset);
  for (uint8_t view_axis = 0; view_axis < memory_access->view_rank;
       ++view_axis) {
    int64_t stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(memory_access, view_axis,
                                                      &stride)) {
      return iree_ok_status();
    }
    if (stride == 0) {
      continue;
    }

    loom_symbolic_expr_t dimension = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_dim_expr(
        state, memory_access->view_type, view_axis, &dimension));
    loom_symbolic_expr_t maximum_index = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
        state, &dimension, -1, &maximum_index));
    loom_symbolic_expr_t contribution = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_mul_i64(
        state, &maximum_index, stride, &contribution));
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
        state, &maximum_element_offset, &contribution,
        &maximum_element_offset));
  }
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
      state, &maximum_element_offset, 1, out_expression));
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_offsets(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  loom_type_t view_type = loom_module_value_type(state->module, access->view);
  loom_vector_memory_access_t memory_access = {0};
  const loom_fact_context_t* fact_context =
      state->fact_table ? &state->fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, state->module,
                                          view_type, access->vector_type,
                                          &memory_access)) {
    return iree_ok_status();
  }
  if (memory_access.layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN) {
    return loom_vector_memory_footprint_fail_unresolved_layout(state, access);
  }
  if (memory_access.vector_rank != 1) {
    return loom_vector_memory_footprint_fail_unsupported_offset_rank(state,
                                                                     access);
  }

  for (uint8_t view_axis = 0; view_axis < memory_access.view_rank;
       ++view_axis) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_check_origin_axis(
        state, access, &memory_access, view_axis));
    if (state->current_access_failed) {
      return iree_ok_status();
    }
  }

  loom_symbolic_expr_t origin_element_offset = {0};
  bool origin_element_offset_known = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_origin_element_offset_expr(
      state, access, &memory_access, &origin_element_offset,
      &origin_element_offset_known));
  if (!origin_element_offset_known) {
    return loom_vector_memory_footprint_fail_linear_span(state, access,
                                                         LOOM_ERR_SUBRANGE_012);
  }

  loom_symbolic_expr_t lower_offset = {0};
  loom_symbolic_expr_t upper_offset = {0};
  bool offsets_known = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_offset_bounds(
      state, access, &lower_offset, &upper_offset, &offsets_known));
  if (!offsets_known) {
    return loom_vector_memory_footprint_fail_linear_span(state, access,
                                                         LOOM_ERR_SUBRANGE_013);
  }

  loom_symbolic_expr_t minimum_access = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
      state, &origin_element_offset, &lower_offset, &minimum_access));
  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);
  bool lower_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &zero, &minimum_access, &lower_proven));
  if (!lower_proven) {
    return loom_vector_memory_footprint_fail_linear_span(state, access,
                                                         LOOM_ERR_SUBRANGE_014);
  }

  loom_symbolic_expr_t maximum_access = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
      state, &origin_element_offset, &upper_offset, &maximum_access));
  loom_symbolic_expr_t exclusive_end = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
      state, &maximum_access, 1, &exclusive_end));

  loom_symbolic_expr_t storage_span = {0};
  bool storage_span_known = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_storage_element_span_expr(
      state, &memory_access, &storage_span, &storage_span_known));
  if (!storage_span_known) {
    return loom_vector_memory_footprint_fail_linear_span(state, access,
                                                         LOOM_ERR_SUBRANGE_015);
  }

  bool upper_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &exclusive_end, &storage_span, &upper_proven));
  if (!upper_proven) {
    return loom_vector_memory_footprint_fail_linear_span(state, access,
                                                         LOOM_ERR_SUBRANGE_016);
  }
  return iree_ok_status();
}

static bool loom_vector_memory_footprint_access_empty(
    const loom_vector_memory_footprint_access_t* access) {
  return loom_type_has_static_zero_extent(access->vector_type);
}

static iree_status_t loom_vector_memory_footprint_check_access(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  state->current_access_failed = false;
  if (loom_vector_memory_footprint_access_empty(access) ||
      loom_vector_memory_footprint_mask_uniform_false(state, access->mask)) {
    ++state->result->skipped_op_count;
    return iree_ok_status();
  }

  ++state->result->checked_op_count;
  if (access->has_offsets) {
    return loom_vector_memory_footprint_check_offsets(state, access);
  }
  return loom_vector_memory_footprint_check_direct(state, access);
}

static bool loom_vector_memory_footprint_describe_op(
    loom_vector_memory_footprint_state_t* state, const loom_op_t* op,
    loom_vector_memory_footprint_access_t* out_access) {
  *out_access = (loom_vector_memory_footprint_access_t){
      .op = op,
      .view = LOOM_VALUE_ID_INVALID,
      .mask = LOOM_VALUE_ID_INVALID,
      .offsets = LOOM_VALUE_ID_INVALID,
  };
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      out_access->view = loom_vector_load_view(op);
      out_access->vector_type =
          loom_module_value_type(state->module, loom_vector_load_result(op));
      out_access->static_indices = loom_vector_load_static_indices(op);
      out_access->dynamic_indices = loom_vector_load_indices(op);
      return true;
    case LOOM_OP_VECTOR_STORE:
      out_access->view = loom_vector_store_view(op);
      out_access->vector_type =
          loom_module_value_type(state->module, loom_vector_store_value(op));
      out_access->static_indices = loom_vector_store_static_indices(op);
      out_access->dynamic_indices = loom_vector_store_indices(op);
      return true;
    case LOOM_OP_VECTOR_LOAD_MASK:
      out_access->view = loom_vector_load_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_load_mask_result(op));
      out_access->static_indices = loom_vector_load_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_load_mask_indices(op);
      out_access->mask = loom_vector_load_mask_mask(op);
      return true;
    case LOOM_OP_VECTOR_STORE_MASK:
      out_access->view = loom_vector_store_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_store_mask_value(op));
      out_access->static_indices = loom_vector_store_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_store_mask_indices(op);
      out_access->mask = loom_vector_store_mask_mask(op);
      return true;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      out_access->view = loom_vector_load_expand_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_load_expand_result(op));
      out_access->static_indices = loom_vector_load_expand_static_indices(op);
      out_access->dynamic_indices = loom_vector_load_expand_indices(op);
      out_access->mask = loom_vector_load_expand_mask(op);
      return true;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      out_access->view = loom_vector_store_compress_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_store_compress_value(op));
      out_access->static_indices =
          loom_vector_store_compress_static_indices(op);
      out_access->dynamic_indices = loom_vector_store_compress_indices(op);
      out_access->mask = loom_vector_store_compress_mask(op);
      return true;
    case LOOM_OP_VECTOR_GATHER:
      out_access->view = loom_vector_gather_view(op);
      out_access->vector_type =
          loom_module_value_type(state->module, loom_vector_gather_result(op));
      out_access->static_indices = loom_vector_gather_static_indices(op);
      out_access->dynamic_indices = loom_vector_gather_indices(op);
      out_access->offsets = loom_vector_gather_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_SCATTER:
      out_access->view = loom_vector_scatter_view(op);
      out_access->vector_type =
          loom_module_value_type(state->module, loom_vector_scatter_value(op));
      out_access->static_indices = loom_vector_scatter_static_indices(op);
      out_access->dynamic_indices = loom_vector_scatter_indices(op);
      out_access->offsets = loom_vector_scatter_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_GATHER_MASK:
      out_access->view = loom_vector_gather_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_gather_mask_result(op));
      out_access->static_indices = loom_vector_gather_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_gather_mask_indices(op);
      out_access->mask = loom_vector_gather_mask_mask(op);
      out_access->offsets = loom_vector_gather_mask_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      out_access->view = loom_vector_scatter_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_scatter_mask_value(op));
      out_access->static_indices = loom_vector_scatter_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_scatter_mask_indices(op);
      out_access->mask = loom_vector_scatter_mask_mask(op);
      out_access->offsets = loom_vector_scatter_mask_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE:
      out_access->view = loom_vector_atomic_reduce_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_atomic_reduce_value(op));
      out_access->static_indices = loom_vector_atomic_reduce_static_indices(op);
      out_access->dynamic_indices = loom_vector_atomic_reduce_indices(op);
      out_access->offsets = loom_vector_atomic_reduce_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      out_access->view = loom_vector_atomic_reduce_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_atomic_reduce_mask_value(op));
      out_access->static_indices =
          loom_vector_atomic_reduce_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_atomic_reduce_mask_indices(op);
      out_access->mask = loom_vector_atomic_reduce_mask_mask(op);
      out_access->offsets = loom_vector_atomic_reduce_mask_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_ATOMIC_RMW:
      out_access->view = loom_vector_atomic_rmw_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_atomic_rmw_value(op));
      out_access->static_indices = loom_vector_atomic_rmw_static_indices(op);
      out_access->dynamic_indices = loom_vector_atomic_rmw_indices(op);
      out_access->offsets = loom_vector_atomic_rmw_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      out_access->view = loom_vector_atomic_rmw_mask_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_atomic_rmw_mask_value(op));
      out_access->static_indices =
          loom_vector_atomic_rmw_mask_static_indices(op);
      out_access->dynamic_indices = loom_vector_atomic_rmw_mask_indices(op);
      out_access->mask = loom_vector_atomic_rmw_mask_mask(op);
      out_access->offsets = loom_vector_atomic_rmw_mask_offsets(op);
      out_access->has_offsets = true;
      return true;
    case LOOM_OP_VECTOR_ATOMIC_CMPXCHG:
      out_access->view = loom_vector_atomic_cmpxchg_view(op);
      out_access->vector_type = loom_module_value_type(
          state->module, loom_vector_atomic_cmpxchg_old(op));
      out_access->static_indices =
          loom_vector_atomic_cmpxchg_static_indices(op);
      out_access->dynamic_indices = loom_vector_atomic_cmpxchg_indices(op);
      out_access->offsets = loom_vector_atomic_cmpxchg_offsets(op);
      out_access->has_offsets = true;
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_vector_memory_footprint_check_op_tree(
    loom_vector_memory_footprint_state_t* state, loom_op_t* op) {
  if (loom_vector_memory_footprint_should_suppress_error(state)) {
    return iree_ok_status();
  }
  loom_vector_memory_footprint_access_t access = {0};
  if (loom_vector_memory_footprint_describe_op(state, op, &access)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_check_access(state, &access));
    if (loom_vector_memory_footprint_should_suppress_error(state)) {
      return iree_ok_status();
    }
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_vector_memory_footprint_check_op_tree(state, child_op));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_vector_memory_footprint_verify_function(
    loom_module_t* module, loom_func_like_t function,
    const loom_vector_memory_footprint_options_t* options,
    loom_vector_memory_footprint_result_t* out_result) {
  *out_result = (loom_vector_memory_footprint_result_t){0};

  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_vector_memory_footprint_state_t state = {
      .module = module,
      .options = options,
      .fact_table = options->fact_table,
      .result = out_result,
  };
  loom_symbolic_expr_context_initialize(
      module, state.fact_table, options->arena, &state.expression_context);

  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_vector_memory_footprint_check_op_tree(&state, op));
    }
  }
  return iree_ok_status();
}
