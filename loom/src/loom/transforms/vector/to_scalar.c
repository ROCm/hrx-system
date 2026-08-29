// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar.h"

#include "loom/analysis/contract.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/to_scalar_aggregates.h"
#include "loom/transforms/vector/to_scalar_core.h"
#include "loom/transforms/vector/to_scalar_descriptors.h"
#include "loom/transforms/vector/to_scalar_encoding.h"
#include "loom/transforms/vector/to_scalar_lanes.h"
#include "loom/transforms/vector/to_scalar_memory.h"
#include "loom/transforms/vector/to_scalar_mma.h"
#include "loom/transforms/vector/to_scalar_reductions.h"
#include "loom/transforms/vector/to_scalar_terms.h"
#include "loom/transforms/vector/to_scalar_transforms.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_statistic_field_t loom_vector_to_scalar_fields[] = {
    LOOM_VECTOR_TO_SCALAR_STATISTICS(LOOM_PASS_STATISTIC_LAYOUT_FIELD_,
                                     loom_vector_to_scalar_statistics_t)};

const loom_pass_statistic_layout_t loom_vector_to_scalar_statistics_layout =
    LOOM_PASS_STATISTIC_LAYOUT(loom_vector_to_scalar_statistics_t,
                               loom_vector_to_scalar_fields);

static const loom_pass_info_t loom_vector_to_scalar_pass_info_storage = {
    .name = IREE_SVL("vector-to-scalar"),
    .description =
        IREE_SVL("Expose vector lane semantics as scalar ops and scf loops."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_vector_to_scalar_statistics_layout,
};

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void) {
  return &loom_vector_to_scalar_pass_info_storage;
}

static const loom_pass_info_t
    loom_vector_reduce_axes_to_scalar_pass_info_storage = {
        .name = IREE_SVL("vector-reduce-axes-to-scalar"),
        .description =
            IREE_SVL("Expose vector.reduce.axes lane semantics as scalar ops "
                     "and scf loops."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_vector_to_scalar_statistics_layout,
};

const loom_pass_info_t* loom_vector_reduce_axes_to_scalar_pass_info(void) {
  return &loom_vector_reduce_axes_to_scalar_pass_info_storage;
}

static const loom_pass_info_t loom_vector_memory_to_scalar_pass_info_storage = {
    .name = IREE_SVL("vector-memory-to-scalar"),
    .description = IREE_SVL("Expose non-dense vector memory accesses as scalar "
                            "view operations."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_vector_to_scalar_statistics_layout,
};

const loom_pass_info_t* loom_vector_memory_to_scalar_pass_info(void) {
  return &loom_vector_memory_to_scalar_pass_info_storage;
}

iree_status_t loom_vector_to_scalar_static_element_count(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    uint16_t* out_element_count) {
  *out_element_count = 0;
  uint64_t element_count = 0;
  if (loom_type_static_element_count(type, &element_count) &&
      element_count <= UINT16_MAX) {
    *out_element_count = (uint16_t)element_count;
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->rewriter->module, state->op)),
      loom_param_string(state->pass->info->name),
      loom_param_type(type),
  };
  loom_diagnostic_emission_t emission = {
      .op = state->op,
      .error = LOOM_ERR_SHAPE_007,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_prepare_state(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_vector_to_scalar_descriptor_t* descriptor,
    uint16_t result_ordinal, loom_vector_to_scalar_state_t* out_state) {
  loom_value_id_t result = loom_op_results(op)[result_ordinal];
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  *out_state = (loom_vector_to_scalar_state_t){
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor =
          descriptor ? *descriptor : (loom_vector_to_scalar_descriptor_t){0},
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .result_ordinal = result_ordinal,
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(out_state, pass);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_replace_one_result(
    loom_vector_to_scalar_state_t* state, loom_value_id_t replacement) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, state->op, &replacement, 1, state->value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, state->op, &replacement, 1));
  loom_pass_mark_changed(state->pass);
  loom_vector_to_scalar_record_ops_lowered(state);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_replace_results(
    loom_vector_to_scalar_state_t* state, const loom_value_id_t* replacements,
    iree_host_size_t replacement_count) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, state->op, replacements, replacement_count,
      state->value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, state->op, replacements, replacement_count));
  loom_pass_mark_changed(state->pass);
  loom_vector_to_scalar_record_ops_lowered(state);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_erase_lowered_op(
    loom_vector_to_scalar_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, state->op));
  loom_pass_mark_changed(state->pass);
  loom_vector_to_scalar_record_ops_lowered(state);
  return iree_ok_status();
}

static loom_value_id_t loom_vector_to_scalar_memory_payload_value(
    const loom_module_t* module, loom_op_t* op,
    loom_memory_access_operation_kind_t required_kind) {
  loom_memory_access_t access = loom_memory_access_cast(module, op);
  if (!loom_memory_access_isa(access) ||
      loom_memory_access_operation_kind(access) != required_kind) {
    return LOOM_VALUE_ID_INVALID;
  }
  return loom_memory_access_value(access);
}

static iree_status_t loom_vector_to_scalar_lower_memory_store_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_value_id_t value = loom_vector_to_scalar_memory_payload_value(
      rewriter->module, op, LOOM_MEMORY_ACCESS_OPERATION_STORE);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_memory_store(&state));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_fragment_store_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  loom_value_id_t value = loom_vector_fragment_store_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_fragment_store(&state, out_handled));
  if (!*out_handled || loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_store_compress_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_value_id_t value = loom_vector_store_compress_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_store_compress(&state));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_reduce_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_value_id_t value = loom_vector_to_scalar_memory_payload_value(
      rewriter->module, op, LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_REDUCE);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_reduce(&state));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_rmw_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_type_t vector_type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_rmw(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_cmpxchg_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_type_t vector_type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_cmpxchg(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_scalar_extract(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  loom_value_id_t result = loom_vector_extract_result(op);
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (loom_type_is_vector(result_type)) {
    *out_handled = false;
    return iree_ok_status();
  }
  *out_handled = true;

  loom_vector_to_scalar_descriptor_t descriptor = {0};
  if (!loom_vector_to_scalar_resolve_descriptor(op->kind, &descriptor)) {
    IREE_ASSERT_UNREACHABLE("scalar vector.extract descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor = descriptor,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .result_scalar_type = result_type,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      &state, loom_vector_extract_static_indices(op),
      loom_vector_extract_indices(op), &explicit_terms, &explicit_count));
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      &state, explicit_terms, explicit_count, &source_indices));
  loom_value_id_t source = loom_vector_extract_source(op);
  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      &state, source, source_type, source_indices, &materialized,
      &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  if (!materialized) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_static_constant(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_constant_result(op));
  if (!loom_type_is_all_static(result_type)) return iree_ok_status();
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);

  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state, result_type, &element_count));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  for (uint16_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_constant_lane(&state, &elements[i]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, element_count, result_type, op->location,
      &from_elements_op));
  loom_value_id_t replacement =
      loom_vector_from_elements_result(from_elements_op);
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_static_poison(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_poison_result(op));
  if (!loom_type_is_all_static(result_type)) return iree_ok_status();
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);

  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state, result_type, &element_count));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  for (uint16_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_poison_lane(&state, &elements[i]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, element_count, result_type, op->location,
      &from_elements_op));
  loom_value_id_t replacement =
      loom_vector_from_elements_result(from_elements_op);
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_deinterleave(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_vector_to_scalar_descriptor_t descriptor = {0};
  if (!loom_vector_to_scalar_resolve_descriptor(op->kind, &descriptor)) {
    return iree_ok_status();
  }

  loom_value_id_t replacements[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  loom_vector_to_scalar_state_t first_state = {0};
  for (uint16_t i = 0; i < 2; ++i) {
    loom_vector_to_scalar_state_t state = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(
        pass, rewriter, op, &descriptor, i, &state));
    if (i == 0) {
      first_state = state;
    }
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_aggregate(&state, &replacements[i]));
    if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  }
  return loom_vector_to_scalar_replace_results(&first_state, replacements,
                                               IREE_ARRAYSIZE(replacements));
}

static iree_status_t loom_vector_to_scalar_skip_op(loom_pass_t* pass,
                                                   loom_rewriter_t* rewriter,
                                                   loom_op_t* op,
                                                   bool* out_handled) {
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_gate_insert_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  loom_type_t insert_result_type =
      loom_module_value_type(rewriter->module, loom_vector_insert_result(op));
  *out_handled = !loom_type_is_all_static(insert_result_type);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_splat_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_vector_to_scalar_state_t state = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_prepare_state(pass, rewriter, op, NULL, 0, &state));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_splat(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_transform_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_vector_to_scalar_state_t state = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_prepare_state(pass, rewriter, op, NULL, 0, &state));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_transform(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_reduce_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = loom_module_value_type(rewriter->module,
                                            loom_vector_reduce_input(op)),
      .result_scalar_type =
          loom_module_value_type(rewriter->module, loom_vector_reduce_init(op)),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_reduce(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_reduce_axes_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_reduce_axes_result(op));
  loom_type_t input_type = loom_module_value_type(
      rewriter->module, loom_vector_reduce_axes_input(op));
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type =
          loom_type_is_vector(result_type) ? result_type : input_type,
      .result_scalar_type = loom_type_is_vector(result_type)
                                ? loom_vector_to_scalar_lane_type(result_type)
                                : result_type,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_reduce_axes(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_dotf_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type =
          loom_module_value_type(rewriter->module, loom_vector_dotf_lhs(op)),
      .result_scalar_type =
          loom_module_value_type(rewriter->module, loom_vector_dotf_init(op)),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_dotf(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_mma_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type =
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op)),
      .result_scalar_type = loom_vector_to_scalar_lane_type(
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op))),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_mma(&state, out_handled, &replacement));
  if (!*out_handled || loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

typedef iree_status_t (*loom_vector_to_scalar_op_lowerer_t)(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled);

#define LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX(op_kind) \
  ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(op, lowerer) \
  [LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] = lowerer

static_assert(LOOM_OP_VECTOR_COUNT_ <= UINT8_MAX,
              "vector op indexes must fit in the lowerer lookup key");

static const loom_vector_to_scalar_op_lowerer_t
    kVectorToScalarOpLowerers[LOOM_OP_VECTOR_COUNT_] = {
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            DEINTERLEAVE, loom_vector_to_scalar_lower_deinterleave),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            STORE, loom_vector_to_scalar_lower_memory_store_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            STORE_MASK, loom_vector_to_scalar_lower_memory_store_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            FRAGMENT_STORE, loom_vector_to_scalar_lower_fragment_store_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            SCATTER, loom_vector_to_scalar_lower_memory_store_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            SCATTER_MASK, loom_vector_to_scalar_lower_memory_store_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            STORE_COMPRESS, loom_vector_to_scalar_lower_store_compress_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            ATOMIC_REDUCE, loom_vector_to_scalar_lower_atomic_reduce_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            ATOMIC_REDUCE_MASK, loom_vector_to_scalar_lower_atomic_reduce_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            ATOMIC_RMW, loom_vector_to_scalar_lower_atomic_rmw_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            ATOMIC_RMW_MASK, loom_vector_to_scalar_lower_atomic_rmw_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            ATOMIC_CMPXCHG, loom_vector_to_scalar_lower_atomic_cmpxchg_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            CONSTANT, loom_vector_to_scalar_lower_static_constant),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            POISON, loom_vector_to_scalar_lower_static_poison),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(EMPTY,
                                             loom_vector_to_scalar_skip_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            EXTRACT, loom_vector_to_scalar_lower_scalar_extract),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            INSERT, loom_vector_to_scalar_gate_insert_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            SPLAT, loom_vector_to_scalar_lower_splat_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            TRANSFORM, loom_vector_to_scalar_lower_transform_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            REDUCE, loom_vector_to_scalar_lower_reduce_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            REDUCE_AXES, loom_vector_to_scalar_lower_reduce_axes_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            DOTF, loom_vector_to_scalar_lower_dotf_op),
        LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW(
            MMA, loom_vector_to_scalar_lower_mma_op),
};

#undef LOOM_VECTOR_TO_SCALAR_OP_LOWERER_ROW
#undef LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX

static iree_status_t loom_vector_to_scalar_try_direct_lowerer(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (loom_op_dialect_id(op->kind) != LOOM_DIALECT_VECTOR)
    return iree_ok_status();
  const uint8_t op_index = loom_op_dialect_index(op->kind);
  if (op_index >= IREE_ARRAYSIZE(kVectorToScalarOpLowerers)) {
    return iree_ok_status();
  }
  const loom_vector_to_scalar_op_lowerer_t lowerer =
      kVectorToScalarOpLowerers[op_index];
  return lowerer ? lowerer(pass, rewriter, op, out_handled) : iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_descriptor_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  if (op->result_count != 1) return iree_ok_status();
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  if (!loom_type_is_vector(result_type)) return iree_ok_status();

  loom_vector_to_scalar_descriptor_t descriptor = {0};
  if (!loom_vector_to_scalar_resolve_descriptor(op->kind, &descriptor)) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(
      pass, rewriter, op, &descriptor, 0, &state));
  if ((descriptor.lane_kind == LOOM_VECTOR_TO_SCALAR_LANE_DECODE ||
       descriptor.lane_kind == LOOM_VECTOR_TO_SCALAR_LANE_ENCODE) &&
      loom_vector_to_scalar_encoding_rejection_bits(&state) !=
          LOOM_CONTRACT_REJECTION_NONE) {
    return iree_ok_status();
  }
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_aggregate(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

iree_status_t loom_vector_to_scalar_rewrite_op(loom_pass_t* pass,
                                               loom_rewriter_t* rewriter,
                                               loom_op_t* op,
                                               bool* out_rewritten) {
  *out_rewritten = false;
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_try_direct_lowerer(pass, rewriter, op, &handled));
  if (!handled) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_descriptor_op(pass, rewriter, op));
  }
  if (!loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_op(loom_pass_t* pass,
                                                    loom_rewriter_t* rewriter,
                                                    loom_op_t* op) {
  bool rewritten = false;
  return loom_vector_to_scalar_rewrite_op(pass, rewriter, op, &rewritten);
}

static iree_status_t loom_vector_reduce_axes_to_scalar_lower_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!loom_vector_reduce_axes_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  return loom_vector_to_scalar_lower_reduce_axes_op(pass, rewriter, op,
                                                    &handled);
}

iree_status_t loom_vector_reduce_to_scalar_rewrite_op(loom_pass_t* pass,
                                                      loom_rewriter_t* rewriter,
                                                      loom_op_t* op,
                                                      bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_reduce_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = loom_module_value_type(rewriter->module,
                                            loom_vector_reduce_input(op)),
      .result_scalar_type =
          loom_module_value_type(rewriter->module, loom_vector_reduce_init(op)),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_reduce(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, state.value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_reduce_axes_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_reduce_axes_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_reduce_axes_result(op));
  loom_type_t input_type = loom_module_value_type(
      rewriter->module, loom_vector_reduce_axes_input(op));
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type =
          loom_type_is_vector(result_type) ? result_type : input_type,
      .result_scalar_type = loom_type_is_vector(result_type)
                                ? loom_vector_to_scalar_lane_type(result_type)
                                : result_type,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_reduce_axes(&state, &replacement));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, state.value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_dotf_to_scalar_rewrite_op(loom_pass_t* pass,
                                                    loom_rewriter_t* rewriter,
                                                    loom_op_t* op,
                                                    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_dotf_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_dotf_op(pass, rewriter, op, &handled));
  if (handled && !loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_transform_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_transform_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_transform_op(pass, rewriter, op, &handled));
  if (handled && !loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_descriptor_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  loom_builder_set_before(&rewriter->builder, op);
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_descriptor_op(pass, rewriter, op));
  if (!loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_decode_to_scalar_rewrite_op(loom_pass_t* pass,
                                                      loom_rewriter_t* rewriter,
                                                      loom_op_t* op,
                                                      bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_decode_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_descriptor_op(pass, rewriter, op));
  if (!loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

uint32_t loom_vector_decode_to_scalar_reference_rejection_bits(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!loom_vector_decode_isa(op)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  loom_vector_to_scalar_descriptor_t descriptor = {0};
  if (!loom_vector_to_scalar_resolve_descriptor(op->kind, &descriptor)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  loom_value_id_t result = loom_vector_decode_result(op);
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor = descriptor,
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  return loom_vector_to_scalar_encoding_rejection_bits(&state);
}

iree_status_t loom_vector_mma_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_mma_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type =
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op)),
      .result_scalar_type = loom_vector_to_scalar_lane_type(
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op))),
      .flags = options.flags,
      .matrix_fragment_layout = options.matrix_fragment_layout,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_mma(&state, out_rewritten, &replacement));
  if (!*out_rewritten || loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, state.value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  return iree_ok_status();
}

uint32_t loom_vector_mma_to_scalar_reference_rejection_bits(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options) {
  if (!loom_vector_mma_isa(op)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .vector_type =
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op)),
      .result_scalar_type = loom_vector_to_scalar_lane_type(
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op))),
      .flags = options.flags,
      .matrix_fragment_layout = options.matrix_fragment_layout,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  return loom_vector_to_scalar_mma_reference_rejection_bits(&state);
}

uint32_t loom_vector_mma_to_scalar_reference_rejection_detail(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options) {
  if (!loom_vector_mma_isa(op)) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .vector_type =
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op)),
      .result_scalar_type = loom_vector_to_scalar_lane_type(
          loom_module_value_type(rewriter->module, loom_vector_mma_result(op))),
      .flags = options.flags,
      .matrix_fragment_layout = options.matrix_fragment_layout,
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  return loom_vector_to_scalar_mma_reference_rejection_detail(&state);
}

iree_status_t loom_vector_store_to_scalar_rewrite_op(loom_pass_t* pass,
                                                     loom_rewriter_t* rewriter,
                                                     loom_op_t* op,
                                                     bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_store_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_memory_store_op(
      pass, rewriter, op, &handled));
  if (handled && !loom_pass_has_error_diagnostics(pass)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_fragment_store_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_fragment_store_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_fragment_store_op(
      pass, rewriter, op, &handled));
  if (handled && !loom_pass_has_error_diagnostics(pass)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

uint32_t loom_vector_fragment_store_to_scalar_reference_rejection_bits(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!loom_vector_fragment_store_isa(op)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  const loom_value_id_t value = loom_vector_fragment_store_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_vector_to_scalar_state_initialize(&state, pass);
  return loom_vector_to_scalar_fragment_store_reference_rejection_bits(&state);
}

iree_status_t loom_vector_extract_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_extract_isa(op)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_scalar_extract(pass, rewriter, op, &handled));
  if (handled && !loom_pass_has_error_diagnostics(pass) &&
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_memory_to_scalar_lower_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_fact_context_t* fact_context =
      rewriter->fact_table ? &rewriter->fact_table->context : NULL;
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(fact_context, rewriter->module, op,
                                             &footprint)) {
    return iree_ok_status();
  }
  switch (footprint.kind) {
    case LOOM_VECTOR_MEMORY_FOOTPRINT_NONE:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT:
      return iree_ok_status();
    // Preserve vector atomics for target legalization; scalarizing them can
    // destroy native packed atomic selection.
    case LOOM_VECTOR_MEMORY_FOOTPRINT_ATOMIC_PER_LANE:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE:
      return iree_ok_status();
    case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_DENSE:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_COMPRESS_EXPAND:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_PER_LANE_OFFSET:
    case LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_PER_LANE_OFFSET:
      break;
  }

  loom_builder_set_before(&rewriter->builder, op);
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_try_direct_lowerer(pass, rewriter, op, &handled));
  if (handled) return iree_ok_status();
  return loom_vector_to_scalar_lower_descriptor_op(pass, rewriter, op);
}

typedef iree_status_t (*loom_vector_to_scalar_lower_op_fn_t)(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op);

typedef uint32_t loom_vector_to_scalar_run_flags_t;

enum loom_vector_to_scalar_run_flag_bits_t {
  LOOM_VECTOR_TO_SCALAR_RUN_FLAG_NONE = 0u,
  LOOM_VECTOR_TO_SCALAR_RUN_FLAG_ERASE_DEAD_OPS = 1u << 0,
};

static iree_status_t loom_vector_to_scalar_run_with_lowerer(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_vector_to_scalar_lower_op_fn_t lower_op,
    loom_vector_to_scalar_run_flags_t flags) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_value_fact_table_t* facts = NULL;
  iree_status_t status = loom_pass_value_facts_prepare(
      pass, module, loom_pass_value_fact_scope_function(function), &facts);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_enable_analysis(&rewriter, function, facts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_seed_function(&rewriter, function);
  }
  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) break;
    if (iree_any_bit_set(flags,
                         LOOM_VECTOR_TO_SCALAR_RUN_FLAG_ERASE_DEAD_OPS)) {
      bool erased = false;
      status = loom_rewriter_erase_if_dead(&rewriter, op, &erased);
      if (!iree_status_is_ok(status)) continue;
      if (erased) {
        loom_pass_mark_changed(pass);
        continue;
      }
    }
    status = lower_op(pass, &rewriter, op);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  return loom_vector_to_scalar_run_with_lowerer(
      pass, module, function, loom_vector_to_scalar_lower_op,
      LOOM_VECTOR_TO_SCALAR_RUN_FLAG_ERASE_DEAD_OPS);
}

iree_status_t loom_vector_reduce_axes_to_scalar_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function) {
  return loom_vector_to_scalar_run_with_lowerer(
      pass, module, function, loom_vector_reduce_axes_to_scalar_lower_op,
      LOOM_VECTOR_TO_SCALAR_RUN_FLAG_NONE);
}

iree_status_t loom_vector_memory_to_scalar_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  return loom_vector_to_scalar_run_with_lowerer(
      pass, module, function, loom_vector_memory_to_scalar_lower_op,
      LOOM_VECTOR_TO_SCALAR_RUN_FLAG_ERASE_DEAD_OPS);
}
