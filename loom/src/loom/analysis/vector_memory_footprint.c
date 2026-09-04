// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/vector_memory_footprint.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/cfg_condition_facts.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/memory_root_bounds.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/analysis/vector_memory_mask_bounds.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// State and helpers
//===----------------------------------------------------------------------===//

typedef struct loom_vector_memory_footprint_state_t {
  // Module whose function body is being checked.
  loom_module_t* module;

  // Root function region containing every verified access.
  loom_region_t* root_region;

  // Caller-owned verification options.
  const loom_vector_memory_footprint_options_t* options;

  // Call-scoped arena owning all verification analysis storage.
  iree_arena_allocator_t* arena;

  // Per-function value facts visible to footprint proof.
  const loom_value_fact_table_t* fact_table;

  // Per-function symbolic expression context sharing the fact table above.
  loom_symbolic_expr_context_t expression_context;

  // Lazily initialized dominance info for CFG path-sensitive checks.
  loom_dominance_info_t dominance;

  // Result object receiving counters and proof failures.
  loom_vector_memory_footprint_result_t* result;

  // True once dominance has been initialized.
  bool dominance_initialized;

  // True once the current access has emitted a proof failure.
  bool current_access_failed;
} loom_vector_memory_footprint_state_t;

typedef enum loom_vector_memory_footprint_access_kind_e {
  LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_DIRECT_VECTOR = 0,
  LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_OFFSET_VECTOR,
  LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR,
} loom_vector_memory_footprint_access_kind_t;

typedef struct loom_vector_memory_footprint_access_t {
  // Memory op being checked.
  const loom_op_t* op;

  // Mutually exclusive footprint shape selected by the memory op family.
  loom_vector_memory_footprint_access_kind_t kind;

  // View value addressed by the memory op.
  loom_value_id_t view;

  // Vector payload type loaded, stored, or atomically updated.
  loom_type_t vector_type;

  // Inline backing for a copied rank-3 fragment footprint type.
  loom_overflow_dim_t fragment_dimensions[3];

  // Optional physical storage scaling for one logical footprint axis.
  loom_vector_memory_footprint_axis_scale_t axis_scale;

  // Full-rank logical origin indices with INT64_MIN entries for dynamic axes.
  loom_attribute_t static_indices;

  // Dynamic logical origin index values matching INT64_MIN static slots.
  loom_value_slice_t dynamic_indices;

  // Optional vector mask; invalid for unmasked memory ops.
  loom_value_id_t mask;

  // Optional lane offset vector for gather, scatter, and vector atomics.
  loom_value_id_t offsets;
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

static iree_string_view_t loom_vector_memory_footprint_value_name(
    loom_value_id_t value, char* value_buffer,
    iree_host_size_t value_buffer_capacity) {
  if (value == LOOM_VALUE_ID_INVALID) {
    return IREE_SV("<unknown>");
  }
  int length =
      snprintf(value_buffer, value_buffer_capacity, "%%%u", (unsigned)value);
  if (length <= 0 || (iree_host_size_t)length >= value_buffer_capacity) {
    return IREE_SV("<unknown>");
  }
  return iree_make_string_view(value_buffer, (iree_host_size_t)length);
}

static bool loom_vector_memory_footprint_expr_exact_i64(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) {
    return false;
  }
  *out_value = expression->constant;
  return true;
}

static iree_string_view_t loom_vector_memory_footprint_i64_text(
    int64_t value, char* text_buffer, iree_host_size_t text_buffer_capacity) {
  int length = snprintf(text_buffer, text_buffer_capacity, "%" PRId64, value);
  if (length <= 0 || (iree_host_size_t)length >= text_buffer_capacity) {
    return IREE_SV("<unknown>");
  }
  return iree_make_string_view(text_buffer, (iree_host_size_t)length);
}

static iree_string_view_t loom_vector_memory_footprint_expr_text(
    const loom_symbolic_expr_t* expression, char* text_buffer,
    iree_host_size_t text_buffer_capacity) {
  int64_t exact_value = 0;
  if (loom_vector_memory_footprint_expr_exact_i64(expression, &exact_value)) {
    return loom_vector_memory_footprint_i64_text(exact_value, text_buffer,
                                                 text_buffer_capacity);
  }
  if (loom_symbolic_expr_is_linear(expression) && expression->constant == 0 &&
      expression->term_count == 1 && expression->terms[0].coefficient == 1) {
    return loom_vector_memory_footprint_value_name(
        expression->terms[0].value_id, text_buffer, text_buffer_capacity);
  }
  return IREE_SV("<dynamic>");
}

static iree_string_view_t
loom_vector_memory_footprint_required_origin_upper_bound_text(
    const loom_symbolic_expr_t* bound, const loom_symbolic_expr_t* extent,
    char* text_buffer, iree_host_size_t text_buffer_capacity) {
  int64_t bound_value = 0;
  int64_t extent_value = 0;
  int64_t required_origin_upper_bound = 0;
  if (!loom_vector_memory_footprint_expr_exact_i64(bound, &bound_value) ||
      !loom_vector_memory_footprint_expr_exact_i64(extent, &extent_value) ||
      !iree_checked_sub_i64(bound_value, extent_value,
                            &required_origin_upper_bound)) {
    return IREE_SV("<dynamic>");
  }
  return loom_vector_memory_footprint_i64_text(
      required_origin_upper_bound, text_buffer, text_buffer_capacity);
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
  if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR &&
      error == LOOM_ERR_SUBRANGE_008) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_i64(view_axis),
        loom_param_string(origin),
    };
    loom_diagnostic_emission_t emission = {
        .op = access->op,
        .error = LOOM_ERR_SUBRANGE_022,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(state->options->emitter, &emission);
  }
  if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR &&
      error == LOOM_ERR_SUBRANGE_009) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_i64(view_axis),
        loom_param_string(origin),
    };
    loom_diagnostic_emission_t emission = {
        .op = access->op,
        .error = LOOM_ERR_SUBRANGE_023,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(state->options->emitter, &emission);
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

static iree_status_t loom_vector_memory_footprint_fail_upper_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access, uint8_t view_axis,
    int64_t vector_axis, const loom_symbolic_expr_t* extent,
    const loom_symbolic_expr_t* bound, const loom_error_def_t* error,
    iree_string_view_t constraint_key) {
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

  iree_string_view_t config_key = iree_string_view_empty();
  if (loom_vector_memory_footprint_lookup_config_decl_bound(
          state, access, view_axis, &config_key)) {
    if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_i64(view_axis),
          loom_param_string(origin),
          loom_param_string(config_key),
          loom_param_string(
              IREE_SV("config_decl.where.memory_footprint_upper_bound")),
      };
      loom_diagnostic_emission_t emission = {
          .op = access->op,
          .error = LOOM_ERR_SUBRANGE_025,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      return iree_diagnostic_emit(state->options->emitter, &emission);
    }
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

  char view_buffer[32];
  char extent_buffer[32];
  char bound_buffer[32];
  char required_origin_upper_bound_buffer[32];
  iree_string_view_t view = loom_vector_memory_footprint_value_name(
      access->view, view_buffer, IREE_ARRAYSIZE(view_buffer));
  iree_string_view_t vector_extent = loom_vector_memory_footprint_expr_text(
      extent, extent_buffer, IREE_ARRAYSIZE(extent_buffer));
  iree_string_view_t view_bound = loom_vector_memory_footprint_expr_text(
      bound, bound_buffer, IREE_ARRAYSIZE(bound_buffer));
  iree_string_view_t required_origin_upper_bound =
      loom_vector_memory_footprint_required_origin_upper_bound_text(
          bound, extent, required_origin_upper_bound_buffer,
          IREE_ARRAYSIZE(required_origin_upper_bound_buffer));
  if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_i64(view_axis),
        loom_param_string(view),
        loom_param_string(origin),
        loom_param_string(view_bound),
        loom_param_string(required_origin_upper_bound),
        loom_param_string(IREE_SV("memory_footprint.scalar_axis_upper_bound")),
    };
    loom_diagnostic_emission_t emission = {
        .op = access->op,
        .error = LOOM_ERR_SUBRANGE_024,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(state->options->emitter, &emission);
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_i64(view_axis),
      loom_param_i64(vector_axis),
      loom_param_string(view),
      loom_param_string(origin),
      loom_param_string(vector_extent),
      loom_param_string(view_bound),
      loom_param_string(required_origin_upper_bound),
      loom_param_string(constraint_key),
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

static iree_status_t loom_vector_memory_footprint_check_root_element_end(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access,
    loom_value_facts_t element_end_facts) {
  if (memory_access->static_element_byte_count <= 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_memory_root_bounds_verify_exact_root(
      state->module, state->fact_table, state->options->emitter, access->op,
      access->view, memory_access->static_element_byte_count, element_end_facts,
      &state->current_access_failed));
  state->result->error_count += state->current_access_failed;
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

static bool loom_vector_memory_footprint_axis_scale_applies(
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis) {
  return view_axis >= memory_access->first_vector_axis &&
         access->axis_scale.vector_axis ==
             view_axis - memory_access->first_vector_axis;
}

static bool loom_vector_memory_footprint_axis_extent_is_group_aligned(
    const loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    uint16_t logical_element_count) {
  const uint8_t vector_axis = view_axis - memory_access->first_vector_axis;
  if (!loom_type_dim_is_dynamic_at(memory_access->vector_type, vector_axis)) {
    return loom_type_dim_static_size_at(memory_access->vector_type,
                                        vector_axis) %
               logical_element_count ==
           0;
  }
  const loom_value_id_t extent =
      loom_type_dim_value_id_at(memory_access->vector_type, vector_axis);
  if (extent == LOOM_VALUE_ID_INVALID || state->fact_table == NULL) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(state->fact_table, extent);
  return facts.known_divisor > 0 &&
         facts.known_divisor % logical_element_count == 0;
}

static iree_status_t loom_vector_memory_footprint_scale_axis_proof(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    const loom_symbolic_expr_t* origin, const loom_symbolic_expr_t* extent,
    const loom_symbolic_expr_t* bound, loom_symbolic_expr_t* out_end,
    loom_symbolic_expr_t* out_bound,
    loom_symbolic_expr_t* out_diagnostic_extent) {
  const int64_t storage_element_count =
      access->axis_scale.storage_element_count;
  const int64_t logical_element_count =
      access->axis_scale.logical_element_count;

  int64_t exact_extent = 0;
  int64_t rounded_extent = 0;
  int64_t physical_extent = 0;
  if (loom_vector_memory_footprint_expr_exact_i64(extent, &exact_extent) &&
      exact_extent >= 0 &&
      iree_checked_add_i64(exact_extent, logical_element_count - 1,
                           &rounded_extent) &&
      iree_checked_mul_i64(rounded_extent / logical_element_count,
                           storage_element_count, &physical_extent)) {
    loom_symbolic_expr_constant(physical_extent, out_diagnostic_extent);
    *out_bound = *bound;
    return loom_vector_memory_footprint_expr_add(
        state, origin, out_diagnostic_extent, out_end);
  }

  loom_symbolic_expr_t grouped_extent = *extent;
  if (!loom_vector_memory_footprint_axis_extent_is_group_aligned(
          state, memory_access, view_axis, (uint16_t)logical_element_count)) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add_i64(
        state, extent, logical_element_count - 1, &grouped_extent));
  }
  loom_symbolic_expr_t scaled_origin = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_mul_i64(
      state, origin, logical_element_count, &scaled_origin));
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_mul_i64(
      state, &grouped_extent, storage_element_count, out_diagnostic_extent));
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
      state, &scaled_origin, out_diagnostic_extent, out_end));
  return loom_vector_memory_footprint_expr_mul_i64(
      state, bound, logical_element_count, out_bound);
}

static iree_status_t
loom_vector_memory_footprint_prove_axis_upper_bound_from_origin_relation(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    const loom_symbolic_expr_t* extent, bool* out_proven) {
  *out_proven = false;

  loom_value_id_t origin_value = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_memory_footprint_origin_value(access->static_indices,
                                                 access->dynamic_indices,
                                                 view_axis, &origin_value)) {
    return iree_ok_status();
  }

  loom_symbolic_expr_t origin = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_value(&state->expression_context,
                                                origin_value, &origin));
  loom_symbolic_expr_t end = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_memory_footprint_expr_add(state, &origin, extent, &end));
  loom_symbolic_expr_t bound = {0};
  if (loom_type_dim_is_dynamic_at(memory_access->view_type, view_axis)) {
    loom_value_id_t bound_value =
        loom_type_dim_value_id_at(memory_access->view_type, view_axis);
    if (bound_value == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_value(&state->expression_context,
                                                  bound_value, &bound));
  } else {
    loom_symbolic_expr_constant(
        loom_type_dim_static_size_at(memory_access->view_type, view_axis),
        &bound);
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_memory_footprint_prove_le(state, &end, &bound, out_proven));
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_direct_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access,
    const loom_vector_memory_mask_bounds_t* mask_bounds, uint8_t view_axis,
    loom_value_facts_t* out_exclusive_end_facts) {
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

  loom_symbolic_expr_t extent = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_axis_extent_expr(
      state, memory_access, view_axis, &extent));
  loom_symbolic_expr_t full_end = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expr_add(
      state, &origin, &extent, &full_end));
  loom_symbolic_expr_t end = full_end;
  loom_value_facts_t end_facts = full_end.facts;
  bool has_tail_end = false;
  if (memory_access->vector_rank == 1 &&
      view_axis == memory_access->first_vector_axis) {
    loom_symbolic_expr_t tail_end = {0};
    IREE_RETURN_IF_ERROR(loom_vector_memory_mask_bounds_tail_end(
        &state->expression_context, mask_bounds, &origin, &tail_end,
        &has_tail_end));
    if (has_tail_end) {
      end = tail_end;
      loom_value_facts_minsi(&full_end.facts, &tail_end.facts, &end_facts);
    }
  }

  loom_symbolic_expr_t bound = {0};
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_dim_expr(
      state, memory_access->view_type, view_axis, &bound));
  loom_symbolic_expr_t proof_bound = bound;
  loom_symbolic_expr_t diagnostic_extent = extent;
  const bool axis_is_scaled = loom_vector_memory_footprint_axis_scale_applies(
      access, memory_access, view_axis);
  if (axis_is_scaled && !has_tail_end) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_scale_axis_proof(
        state, access, memory_access, view_axis, &origin, &extent, &bound, &end,
        &proof_bound, &diagnostic_extent));
    end_facts = end.facts;
  }
  bool upper_proven = false;
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
      state, &end, &proof_bound, &upper_proven));
  if (!upper_proven && has_tail_end) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_prove_le(
        state, &full_end, &proof_bound, &upper_proven));
  }
  if (!upper_proven && !axis_is_scaled) {
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_prove_axis_upper_bound_from_origin_relation(
            state, access, memory_access, view_axis, &extent, &upper_proven));
  }
  if (!upper_proven) {
    int64_t extent_value = 0;
    bool unit_extent =
        loom_vector_memory_footprint_expr_exact_i64(&extent, &extent_value) &&
        extent_value == 1;
    return loom_vector_memory_footprint_fail_upper_axis(
        state, access, view_axis,
        loom_vector_memory_footprint_vector_axis(memory_access, view_axis),
        &diagnostic_extent, &bound,
        unit_extent ? LOOM_ERR_SUBRANGE_011 : LOOM_ERR_SUBRANGE_010,
        unit_extent ? IREE_SV("vector_footprint.scalar_axis_upper_bound")
                    : IREE_SV("vector_footprint.full_vector_upper_bound"));
  }
  *out_exclusive_end_facts = end_facts;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_direct(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_mask_bounds_t* mask_bounds) {
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

  loom_value_facts_t
      axis_exclusive_end_facts[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint8_t axis = 0; axis < memory_access.view_rank; ++axis) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_check_direct_axis(
        state, access, &memory_access, mask_bounds, axis,
        &axis_exclusive_end_facts[axis]));
    if (state->current_access_failed) {
      return iree_ok_status();
    }
  }
  if (access->axis_scale.vector_axis == UINT8_MAX) {
    loom_value_facts_t element_end_facts = loom_value_facts_unknown();
    if (loom_vector_memory_access_linear_element_end_facts(
            &memory_access, axis_exclusive_end_facts, &element_end_facts)) {
      return loom_vector_memory_footprint_check_root_element_end(
          state, access, &memory_access, element_end_facts);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_origin_axis(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access,
    const loom_vector_memory_access_t* memory_access, uint8_t view_axis,
    loom_value_facts_t* out_exclusive_end_facts) {
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
    loom_symbolic_expr_t extent = {0};
    loom_symbolic_expr_constant(1, &extent);
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_prove_axis_upper_bound_from_origin_relation(
            state, access, memory_access, view_axis, &extent, &upper_proven));
  }
  if (!upper_proven) {
    loom_symbolic_expr_t extent = {0};
    loom_symbolic_expr_constant(1, &extent);
    return loom_vector_memory_footprint_fail_upper_axis(
        state, access, view_axis, /*vector_axis=*/-1, &extent, &bound,
        LOOM_ERR_SUBRANGE_011,
        IREE_SV("vector_footprint.scalar_axis_upper_bound"));
  }
  *out_exclusive_end_facts = exclusive_end.facts;
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_check_scalar(
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

  loom_value_facts_t
      axis_exclusive_end_facts[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint8_t axis = 0; axis < memory_access.view_rank; ++axis) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_check_origin_axis(
        state, access, &memory_access, axis, &axis_exclusive_end_facts[axis]));
    if (state->current_access_failed) {
      return iree_ok_status();
    }
  }
  loom_value_facts_t element_end_facts = loom_value_facts_unknown();
  if (loom_vector_memory_access_linear_element_end_facts(
          &memory_access, axis_exclusive_end_facts, &element_end_facts)) {
    return loom_vector_memory_footprint_check_root_element_end(
        state, access, &memory_access, element_end_facts);
  }
  return iree_ok_status();
}

static bool loom_vector_memory_footprint_iota_bounds_from_facts(
    loom_vector_memory_footprint_state_t* state, loom_value_facts_t facts,
    loom_type_t offsets_type, int64_t* out_lower, int64_t* out_upper) {
  if (loom_type_rank(offsets_type) != 1) {
    return false;
  }
  loom_value_fact_vector_iota_t iota = {0};
  if (!loom_value_facts_query_vector_iota(&state->fact_table->context, facts,
                                          &iota)) {
    return false;
  }

  int64_t base_lower = 0;
  int64_t base_upper = 0;
  int64_t step = 0;
  if (!loom_vector_memory_footprint_facts_i64_bounds(iota.base, &base_lower,
                                                     &base_upper) ||
      !loom_vector_memory_footprint_facts_exact_i64(iota.step, &step)) {
    return false;
  }

  int64_t lane_count_upper = 0;
  if (!loom_type_dim_is_dynamic_at(offsets_type, 0)) {
    lane_count_upper = loom_type_dim_static_size_at(offsets_type, 0);
    if (lane_count_upper <= 0) {
      *out_lower = 0;
      *out_upper = -1;
      return true;
    }
  } else {
    loom_value_facts_t count_facts = loom_value_fact_table_lookup(
        state->fact_table, loom_type_dim_value_id_at(offsets_type, 0));
    int64_t lane_count_lower = 0;
    if (!loom_value_facts_is_positive(count_facts) ||
        !loom_vector_memory_footprint_facts_i64_bounds(
            count_facts, &lane_count_lower, &lane_count_upper)) {
      return false;
    }
  }

  int64_t last_lane = 0;
  int64_t last_delta = 0;
  if (!iree_checked_sub_i64(lane_count_upper, 1, &last_lane) ||
      !iree_checked_mul_i64(last_lane, step, &last_delta)) {
    return false;
  }

  if (step >= 0) {
    if (!iree_checked_add_i64(base_upper, last_delta, out_upper)) {
      return false;
    }
    *out_lower = base_lower;
  } else {
    if (!iree_checked_add_i64(base_lower, last_delta, out_lower)) {
      return false;
    }
    *out_upper = base_upper;
  }
  return true;
}

static bool loom_vector_memory_footprint_offset_bounds_from_facts(
    loom_vector_memory_footprint_state_t* state, loom_value_facts_t facts,
    loom_type_t offsets_type, int64_t* out_lower, int64_t* out_upper) {
  if (loom_vector_memory_footprint_iota_bounds_from_facts(
          state, facts, offsets_type, out_lower, out_upper)) {
    return true;
  }

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
  if (!loom_vector_memory_footprint_offset_bounds_from_facts(
          state, facts, offsets_type, &lower, &upper)) {
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
    loom_value_facts_t axis_exclusive_end_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_check_origin_axis(
        state, access, &memory_access, view_axis, &axis_exclusive_end_facts));
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
  return loom_vector_memory_footprint_check_root_element_end(
      state, access, &memory_access, exclusive_end.facts);
}

static bool loom_vector_memory_footprint_access_empty(
    const loom_vector_memory_footprint_access_t* access) {
  return loom_type_has_static_zero_extent(access->vector_type);
}

static bool loom_vector_memory_footprint_value_is_scalar_element(
    const loom_vector_memory_footprint_state_t* state, loom_value_id_t value_id,
    loom_scalar_type_t element_type) {
  if (value_id >= state->module->values.count) return false;
  const loom_type_t value_type =
      loom_module_value_type(state->module, value_id);
  return loom_type_equal(value_type, loom_type_scalar(element_type));
}

static bool loom_vector_memory_footprint_access_is_scalar_element(
    const loom_vector_memory_footprint_state_t* state, const loom_op_t* op,
    loom_memory_access_t memory_access, loom_type_t view_type) {
  const loom_scalar_type_t element_type = loom_type_element_type(view_type);
  switch (loom_memory_access_operation_kind(memory_access)) {
    case LOOM_MEMORY_ACCESS_OPERATION_LOAD:
      if (op->result_count != 1) return false;
      return loom_vector_memory_footprint_value_is_scalar_element(
          state, loom_op_const_results(op)[0], element_type);
    case LOOM_MEMORY_ACCESS_OPERATION_STORE:
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_REDUCE:
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_RMW:
      return loom_vector_memory_footprint_value_is_scalar_element(
          state, loom_memory_access_value(memory_access), element_type);
    case LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG:
      return loom_vector_memory_footprint_value_is_scalar_element(
                 state, loom_memory_access_expected(memory_access),
                 element_type) &&
             loom_vector_memory_footprint_value_is_scalar_element(
                 state, loom_memory_access_replacement(memory_access),
                 element_type);
    case LOOM_MEMORY_ACCESS_OPERATION_PREFETCH:
    case LOOM_MEMORY_ACCESS_OPERATION_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_vector_memory_footprint_check_access(
    loom_vector_memory_footprint_state_t* state,
    const loom_vector_memory_footprint_access_t* access) {
  state->current_access_failed = false;
  loom_vector_memory_mask_bounds_t mask_bounds = {0};
  if (access->mask != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_mask_bounds_analyze(
        state->module, state->fact_table, &state->expression_context,
        access->mask, &mask_bounds));
  }
  if (loom_vector_memory_footprint_access_empty(access) ||
      mask_bounds.definitely_empty) {
    ++state->result->skipped_op_count;
    return iree_ok_status();
  }

  ++state->result->checked_op_count;
  if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR) {
    return loom_vector_memory_footprint_check_scalar(state, access);
  }
  if (access->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_OFFSET_VECTOR) {
    return loom_vector_memory_footprint_check_offsets(state, access);
  }
  return loom_vector_memory_footprint_check_direct(state, access, &mask_bounds);
}

static bool loom_vector_memory_footprint_describe_scalar_access_op(
    loom_vector_memory_footprint_state_t* state, const loom_op_t* op,
    loom_vector_memory_footprint_access_t* out_access) {
  loom_memory_access_t memory_access =
      loom_memory_access_cast(state->module, op);
  if (!loom_memory_access_isa(memory_access)) {
    return false;
  }
  const loom_value_id_t view = loom_memory_access_view(memory_access);
  if (view >= state->module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(state->module, view);
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) == 0) {
    return false;
  }
  if (!loom_vector_memory_footprint_access_is_scalar_element(
          state, op, memory_access, view_type)) {
    return false;
  }

  out_access->view = view;
  out_access->vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  out_access->kind = LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_SCALAR;
  out_access->static_indices = loom_memory_access_static_indices(memory_access);
  out_access->dynamic_indices =
      loom_memory_access_dynamic_indices(memory_access);
  return true;
}

static bool loom_vector_memory_footprint_describe_op(
    loom_vector_memory_footprint_state_t* state, const loom_op_t* op,
    loom_vector_memory_footprint_access_t* out_access) {
  *out_access = (loom_vector_memory_footprint_access_t){
      .op = op,
      .view = LOOM_VALUE_ID_INVALID,
      .mask = LOOM_VALUE_ID_INVALID,
      .offsets = LOOM_VALUE_ID_INVALID,
      .axis_scale.vector_axis = UINT8_MAX,
  };
  const loom_fact_context_t* fact_context =
      state->fact_table ? &state->fact_table->context : NULL;
  loom_vector_memory_footprint_t footprint = {0};
  if (loom_vector_memory_footprint_describe(fact_context, state->module, op,
                                            &footprint)) {
    out_access->view = footprint.view;
    out_access->vector_type = footprint.vector_type;
    if (footprint.kind == LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT &&
        loom_type_rank(footprint.vector_type) > 2) {
      memcpy(out_access->fragment_dimensions, footprint.fragment_dimensions,
             sizeof(out_access->fragment_dimensions));
      out_access->vector_type.dims[0] =
          (uint64_t)(uintptr_t)out_access->fragment_dimensions;
    }
    out_access->axis_scale = footprint.axis_scale;
    out_access->static_indices = footprint.static_indices;
    out_access->dynamic_indices = footprint.dynamic_indices;
    out_access->mask = footprint.mask;
    out_access->offsets = footprint.offsets;
    switch (footprint.kind) {
      case LOOM_VECTOR_MEMORY_FOOTPRINT_NONE:
        return false;
      case LOOM_VECTOR_MEMORY_FOOTPRINT_PER_LANE_OFFSET:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_PER_LANE_OFFSET:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_ATOMIC_PER_LANE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE:
        out_access->kind = LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_OFFSET_VECTOR;
        break;
      case LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_DENSE:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_COMPRESS_EXPAND:
      case LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT:
        out_access->kind = LOOM_VECTOR_MEMORY_FOOTPRINT_ACCESS_DIRECT_VECTOR;
        break;
    }
    return true;
  }
  return loom_vector_memory_footprint_describe_scalar_access_op(state, op,
                                                                out_access);
}

static iree_status_t loom_vector_memory_footprint_ensure_dominance(
    loom_vector_memory_footprint_state_t* state) {
  if (state->dominance_initialized) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_dominance_info_initialize_region(
      state->module, state->root_region, state->arena, &state->dominance));
  state->dominance_initialized = true;
  return iree_ok_status();
}

static void loom_vector_memory_footprint_append_condition_facts(
    const loom_condition_fact_set_t* source,
    loom_condition_fact_set_t* destination) {
  if (!source) return;
  for (iree_host_size_t i = 0; i < source->integer_relation_count &&
                               destination->integer_relation_count <
                                   destination->integer_relation_capacity;
       ++i) {
    destination->integer_relations[destination->integer_relation_count++] =
        source->integer_relations[i];
  }
}

static iree_status_t loom_vector_memory_footprint_condition_facts_copy(
    loom_vector_memory_footprint_state_t* state,
    const loom_condition_fact_set_t* base,
    const loom_condition_fact_set_t* additional,
    const loom_condition_fact_set_t** out_facts) {
  *out_facts = base;
  iree_host_size_t base_relation_count =
      base ? base->integer_relation_count : 0;
  iree_host_size_t additional_relation_count =
      additional ? additional->integer_relation_count : 0;
  if (additional_relation_count == 0) {
    *out_facts = base_relation_count == 0 ? NULL : base;
    return iree_ok_status();
  }
  iree_host_size_t relation_count =
      base_relation_count + additional_relation_count;
  if (relation_count == 0) {
    *out_facts = NULL;
    return iree_ok_status();
  }

  loom_condition_fact_set_t* facts = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(state->arena, sizeof(*facts), (void**)&facts));
  loom_condition_integer_relation_t* relations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, relation_count, sizeof(*relations), (void**)&relations));
  loom_condition_fact_set_initialize(relations, relation_count, facts);
  loom_vector_memory_footprint_append_condition_facts(base, facts);
  loom_vector_memory_footprint_append_condition_facts(additional, facts);
  *out_facts = facts;
  return iree_ok_status();
}

static iree_status_t
loom_vector_memory_footprint_condition_facts_for_assumption(
    loom_vector_memory_footprint_state_t* state,
    const loom_condition_fact_set_t* base, loom_value_id_t condition,
    bool assumed_truth, const loom_condition_fact_set_t** out_facts) {
  loom_condition_integer_relation_t relation_storage[32];
  loom_condition_fact_set_t additional;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &additional);
  if (state->fact_table) {
    bool complete = false;
    IREE_RETURN_IF_ERROR(loom_condition_facts_query(
        &state->expression_context.condition_query, state->fact_table,
        condition, assumed_truth, &additional, &complete));
    if (!complete) loom_condition_fact_set_reset(&additional);
  }
  return loom_vector_memory_footprint_condition_facts_copy(
      state, base, &additional, out_facts);
}

static iree_status_t loom_vector_memory_footprint_condition_facts_for_cfg_block(
    loom_vector_memory_footprint_state_t* state,
    const loom_condition_fact_set_t* base,
    const loom_cfg_block_entry_condition_facts_t* block_facts,
    const loom_condition_fact_set_t** out_facts) {
  loom_condition_fact_set_t additional = {
      .integer_relations =
          (loom_condition_integer_relation_t*)(block_facts
                                                   ? block_facts
                                                         ->integer_relations
                                                   : NULL),
      .integer_relation_count =
          block_facts ? block_facts->integer_relation_count : 0,
      .integer_relation_capacity =
          block_facts ? block_facts->integer_relation_count : 0,
  };
  return loom_vector_memory_footprint_condition_facts_copy(
      state, base, &additional, out_facts);
}

typedef enum loom_vector_memory_footprint_frame_kind_e {
  LOOM_VECTOR_MEMORY_FOOTPRINT_FRAME_REGION = 0,
  LOOM_VECTOR_MEMORY_FOOTPRINT_FRAME_BLOCK = 1,
} loom_vector_memory_footprint_frame_kind_t;

typedef struct loom_vector_memory_footprint_frame_t {
  // Frame category.
  loom_vector_memory_footprint_frame_kind_t kind;
  // Path facts active while processing this frame.
  const loom_condition_fact_set_t* condition_facts;
  union {
    // Region waiting to expand into block frames.
    loom_region_t* region;
    // Block scan state.
    struct {
      // Block being scanned.
      loom_block_t* block;
      // Next operation to visit in block.
      loom_op_t* op;
    } block;
  };
} loom_vector_memory_footprint_frame_t;

typedef struct loom_vector_memory_footprint_stack_t {
  // Arena-backed frame storage.
  loom_vector_memory_footprint_frame_t* frames;
  // Number of active frames.
  iree_host_size_t count;
  // Allocated frame capacity.
  iree_host_size_t capacity;
} loom_vector_memory_footprint_stack_t;

static iree_status_t loom_vector_memory_footprint_stack_initialize(
    iree_arena_allocator_t* arena,
    loom_vector_memory_footprint_stack_t* stack) {
  stack->count = 0;
  stack->capacity = 16;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(*stack->frames), (void**)&stack->frames);
}

static iree_status_t loom_vector_memory_footprint_stack_push(
    iree_arena_allocator_t* arena, loom_vector_memory_footprint_stack_t* stack,
    loom_vector_memory_footprint_frame_t frame) {
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(*stack->frames),
        &stack->capacity, (void**)&stack->frames));
  }
  stack->frames[stack->count++] = frame;
  return iree_ok_status();
}

static loom_vector_memory_footprint_frame_t*
loom_vector_memory_footprint_stack_top(
    loom_vector_memory_footprint_stack_t* stack) {
  return stack->count == 0 ? NULL : &stack->frames[stack->count - 1];
}

static void loom_vector_memory_footprint_stack_pop(
    loom_vector_memory_footprint_stack_t* stack) {
  if (stack->count > 0) --stack->count;
}

static iree_status_t loom_vector_memory_footprint_push_region(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_region_t* region,
    const loom_condition_fact_set_t* condition_facts) {
  if (!region) return iree_ok_status();
  return loom_vector_memory_footprint_stack_push(
      state->arena, stack,
      (loom_vector_memory_footprint_frame_t){
          .kind = LOOM_VECTOR_MEMORY_FOOTPRINT_FRAME_REGION,
          .condition_facts = condition_facts,
          .region = region,
      });
}

static iree_status_t loom_vector_memory_footprint_push_block(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_block_t* block,
    const loom_condition_fact_set_t* condition_facts) {
  if (!block) return iree_ok_status();
  return loom_vector_memory_footprint_stack_push(
      state->arena, stack,
      (loom_vector_memory_footprint_frame_t){
          .kind = LOOM_VECTOR_MEMORY_FOOTPRINT_FRAME_BLOCK,
          .condition_facts = condition_facts,
          .block =
              {
                  .block = block,
                  .op = block->first_op,
              },
      });
}

static iree_status_t loom_vector_memory_footprint_push_structured_blocks(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_region_t* region,
    const loom_condition_fact_set_t* condition_facts) {
  for (iree_host_size_t i = region->block_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_block(
        state, stack, loom_region_block(region, (uint16_t)(i - 1)),
        condition_facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_push_cfg_blocks(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_region_t* region,
    const loom_condition_fact_set_t* condition_facts) {
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_ensure_dominance(state));

  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(state->module, region, state->arena, &graph));
  if (graph.malformed) {
    return loom_vector_memory_footprint_push_structured_blocks(
        state, stack, region, condition_facts);
  }

  loom_cfg_condition_fact_table_t condition_fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_condition_fact_table_compute(
      state->module, &graph, state->fact_table, &state->dominance, state->arena,
      &condition_fact_table));
  for (iree_host_size_t i = graph.block_count; i > 0; --i) {
    uint16_t block_index = (uint16_t)(i - 1);
    if (!loom_cfg_graph_block_is_reachable(&graph, block_index)) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)graph.blocks[block_index].block;
    if (!block) continue;
    const loom_condition_fact_set_t* block_condition_facts = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_condition_facts_for_cfg_block(
            state, condition_facts,
            loom_cfg_condition_fact_table_block(&condition_fact_table,
                                                block_index),
            &block_condition_facts));
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_block(
        state, stack, block, block_condition_facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_footprint_expand_region(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_region_t* region,
    const loom_condition_fact_set_t* condition_facts) {
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return loom_vector_memory_footprint_push_cfg_blocks(state, stack, region,
                                                        condition_facts);
  }
  return loom_vector_memory_footprint_push_structured_blocks(
      state, stack, region, condition_facts);
}

static iree_status_t loom_vector_memory_footprint_push_op_regions(
    loom_vector_memory_footprint_state_t* state,
    loom_vector_memory_footprint_stack_t* stack, loom_op_t* op,
    const loom_condition_fact_set_t* condition_facts) {
  if (loom_scf_if_isa(op)) {
    const loom_condition_fact_set_t* else_facts = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_condition_facts_for_assumption(
            state, condition_facts, loom_scf_if_condition(op),
            /*assumed_truth=*/false, &else_facts));
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_region(
        state, stack, loom_scf_if_else_region(op), else_facts));
    const loom_condition_fact_set_t* then_facts = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_memory_footprint_condition_facts_for_assumption(
            state, condition_facts, loom_scf_if_condition(op),
            /*assumed_truth=*/true, &then_facts));
    return loom_vector_memory_footprint_push_region(
        state, stack, loom_scf_if_then_region(op), then_facts);
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = op->region_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_region(
        state, stack, regions[i - 1], condition_facts));
  }
  return iree_ok_status();
}

static void loom_vector_memory_footprint_activate_condition_facts(
    loom_vector_memory_footprint_state_t* state,
    const loom_condition_fact_set_t* condition_facts) {
  if (state->expression_context.condition_facts == condition_facts) return;
  state->expression_context.condition_facts = condition_facts;
  loom_symbolic_expr_context_reset(&state->expression_context);
}

static iree_status_t loom_vector_memory_footprint_check_with_stack(
    loom_vector_memory_footprint_state_t* state, loom_region_t* root_region) {
  loom_vector_memory_footprint_stack_t stack = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_memory_footprint_stack_initialize(state->arena, &stack));
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_region(
      state, &stack, root_region, /*condition_facts=*/NULL));

  while (stack.count > 0) {
    if (loom_vector_memory_footprint_should_suppress_error(state)) {
      break;
    }
    loom_vector_memory_footprint_frame_t* frame =
        loom_vector_memory_footprint_stack_top(&stack);
    if (frame->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_FRAME_REGION) {
      loom_region_t* region = frame->region;
      const loom_condition_fact_set_t* condition_facts = frame->condition_facts;
      loom_vector_memory_footprint_stack_pop(&stack);
      IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_expand_region(
          state, &stack, region, condition_facts));
      continue;
    }

    loom_op_t* op = frame->block.op;
    if (!op) {
      loom_vector_memory_footprint_stack_pop(&stack);
      continue;
    }
    frame->block.op = op->next_op;
    loom_vector_memory_footprint_activate_condition_facts(
        state, frame->condition_facts);

    loom_vector_memory_footprint_access_t access = {0};
    if (loom_vector_memory_footprint_describe_op(state, op, &access)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_memory_footprint_check_access(state, &access));
      if (loom_vector_memory_footprint_should_suppress_error(state)) {
        continue;
      }
    }
    IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_push_op_regions(
        state, &stack, op, frame->condition_facts));
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

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_vector_memory_footprint_state_t state = {
      .module = module,
      .root_region = body,
      .options = options,
      .arena = &arena,
      .fact_table = options->fact_table,
      .result = out_result,
  };
  loom_symbolic_expr_context_initialize(module, /*value_domain=*/NULL,
                                        state.fact_table, &arena,
                                        &state.expression_context);

  iree_status_t status =
      loom_vector_memory_footprint_check_with_stack(&state, body);
  iree_arena_deinitialize(&arena);
  return status;
}
