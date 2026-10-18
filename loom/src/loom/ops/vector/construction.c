// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/construction.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/fact_vector_partition.h"

iree_status_t loom_vector_project_regular_partition(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed) {
  *out_changed = false;
  const bool is_extract = loom_vector_extract_isa(op);
  if (!is_extract && !loom_vector_slice_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_op_const_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  if (loom_type_rank(source_type) != 1 ||
      (is_extract ? !loom_type_is_scalar(result_type)
                  : !loom_type_is_all_static(result_type))) {
    return iree_ok_status();
  }
  const loom_attribute_t offsets = is_extract
                                       ? loom_vector_extract_static_indices(op)
                                       : loom_vector_slice_static_offsets(op);
  const uint64_t length =
      is_extract ? 1 : loom_type_dim_static_size_at(result_type, 0);
  // An empty slice may start at the end of the source, outside any component.
  if (length == 0 || offsets.i64_array[0] == INT64_MIN) {
    return iree_ok_status();
  }
  const loom_value_fact_vector_partition_t partition =
      loom_value_fact_table_query_vector_partition(rewriter->fact_table,
                                                   source);
  if (partition.snapshot_lane_count == 0) {
    return iree_ok_status();
  }
  const uint64_t offset = offsets.i64_array[0];
  const loom_value_fact_vector_partition_component_t component =
      partition.snapshot_components[offset / partition.snapshot_lane_count];
  const int64_t component_offset = offset % partition.snapshot_lane_count;
  if (length > component.lane_count - component_offset) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = component.value;
  if (is_extract || component_offset != 0 ||
      !loom_type_equal(
          loom_module_value_type(rewriter->module, component.value),
          result_type)) {
    loom_op_t* projected = NULL;
    if (is_extract) {
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          &rewriter->builder, component.value, NULL, 0, &component_offset, 1,
          result_type, op->location, &projected));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, component.value, NULL, 0, &component_offset, 1,
          result_type, op->location, &projected));
    }
    replacement = loom_op_const_results(projected)[0];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_vector_fold_constant_lanes(loom_op_t* op,
                                              loom_rewriter_t* rewriter,
                                              bool* out_changed) {
  *out_changed = false;
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, result);
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(&rewriter->fact_table->context,
                                                 facts, &lanes)) {
    return iree_ok_status();
  }
  for (iree_host_size_t lane = 0; lane < lanes.count; ++lane) {
    if (!loom_value_facts_is_exact(lanes.lanes[lane])) {
      return iree_ok_status();
    }
  }

  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, result);
  const loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(result_type));
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t elements[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT];
  for (iree_host_size_t lane = 0; lane < lanes.count; ++lane) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_build_constant(rewriter, lanes.lanes[lane], element_type,
                                     op->location, &elements[lane]));
  }
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, lanes.count, result_type, op->location,
      &replacement_op));
  const loom_value_id_t replacement =
      loom_vector_from_elements_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_value_def_op(const loom_rewriter_t* rewriter,
                                     loom_value_id_t value_id,
                                     loom_op_t** out_op) {
  const loom_value_t* value = loom_module_value(rewriter->module, value_id);
  *out_op = loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  return *out_op != NULL;
}

#define LOOM_VECTOR_LANE_CHAIN_MAX_STEPS 8

typedef struct loom_vector_lane_chain_step_t {
  // Scalar unary op kind matched for this lane-chain step.
  loom_op_kind_t scalar_kind;
  // Per-instance flags copied from the scalar op.
  uint8_t instance_flags;
  // Scalar input type observed on the matched scalar op.
  loom_type_t input_type;
  // Scalar result type observed on the matched scalar op.
  loom_type_t result_type;
} loom_vector_lane_chain_step_t;

typedef struct loom_vector_lane_chain_t {
  // Source vector whose static lanes feed every reconstructed lane.
  loom_value_id_t source;
  // Type of the source vector.
  loom_type_t source_type;
  // Number of valid entries in steps.
  iree_host_size_t step_count;
  // Scalar unary steps from outermost producer toward the source extract.
  loom_vector_lane_chain_step_t steps[LOOM_VECTOR_LANE_CHAIN_MAX_STEPS];
} loom_vector_lane_chain_t;

static loom_type_t loom_vector_same_shape_with_element_type(
    loom_type_t shape_type, loom_scalar_type_t element_type) {
  loom_type_t type = shape_type;
  type.header = loom_type_make_header(LOOM_TYPE_VECTOR, element_type,
                                      loom_type_rank(shape_type),
                                      loom_type_flags(shape_type));
  return type;
}

static bool loom_vector_type_is_static_1d_vector(loom_type_t type,
                                                 int64_t* out_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      loom_type_dim_is_dynamic_at(type, 0)) {
    return false;
  }
  *out_lane_count = loom_type_dim_static_size_at(type, 0);
  return *out_lane_count >= 0;
}

static bool loom_vector_lane_chain_steps_equal(
    const loom_vector_lane_chain_step_t* lhs,
    const loom_vector_lane_chain_step_t* rhs) {
  return lhs->scalar_kind == rhs->scalar_kind &&
         lhs->instance_flags == rhs->instance_flags &&
         loom_type_equal(lhs->input_type, rhs->input_type) &&
         loom_type_equal(lhs->result_type, rhs->result_type);
}

static bool loom_vector_scalar_lane_chain_kind_supported(
    loom_op_kind_t scalar_kind) {
  switch (scalar_kind) {
    case LOOM_OP_INDEX_CAST:
    case LOOM_OP_SCALAR_CEILF:
    case LOOM_OP_SCALAR_FLOORF:
    case LOOM_OP_SCALAR_ROUNDF:
    case LOOM_OP_SCALAR_ROUNDEVENF:
    case LOOM_OP_SCALAR_TRUNCF:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
      return true;
    default:
      return false;
  }
}

static bool loom_vector_describe_scalar_lane_chain_step(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_vector_lane_chain_step_t* out_step, loom_value_id_t* out_input) {
  *out_step = (loom_vector_lane_chain_step_t){0};
  *out_input = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_scalar_lane_chain_kind_supported(source_op->kind)) {
    return false;
  }
  out_step->scalar_kind = source_op->kind;
  out_step->instance_flags = source_op->instance_flags;
  *out_input = loom_op_const_operands(source_op)[0];
  out_step->input_type = loom_module_value_type(module, *out_input);
  out_step->result_type =
      loom_module_value_type(module, loom_op_const_results(source_op)[0]);
  return loom_type_is_scalar(out_step->input_type) &&
         loom_type_is_scalar(out_step->result_type);
}

static bool loom_vector_value_is_static_lane_extract(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    iree_host_size_t expected_lane, loom_value_id_t* out_source,
    loom_type_t* out_source_type) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_type = loom_type_none();

  loom_op_t* extract_op = NULL;
  if (!loom_vector_value_def_op(rewriter, value_id, &extract_op) ||
      !loom_vector_extract_isa(extract_op) ||
      loom_vector_extract_indices(extract_op).count != 0) {
    return false;
  }

  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(extract_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      (iree_host_size_t)static_indices.i64_array[0] != expected_lane) {
    return false;
  }

  loom_value_id_t source = loom_vector_extract_source(extract_op);
  *out_source = source;
  *out_source_type = loom_module_value_type(rewriter->module, source);
  return true;
}

iree_status_t loom_vector_from_elements_fold_source(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed) {
  *out_changed = false;
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  int64_t lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &lane_count) ||
      lane_count == 0 || lane_count != (int64_t)elements.count) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_type_t source_type = loom_type_none();
  if (!loom_vector_value_is_static_lane_extract(rewriter, elements.values[0], 0,
                                                &source, &source_type) ||
      !loom_type_equal(source_type, result_type)) {
    return iree_ok_status();
  }
  for (iree_host_size_t lane = 1; lane < elements.count; ++lane) {
    loom_value_id_t lane_source = LOOM_VALUE_ID_INVALID;
    loom_type_t lane_source_type = loom_type_none();
    if (!loom_vector_value_is_static_lane_extract(
            rewriter, elements.values[lane], lane, &lane_source,
            &lane_source_type) ||
        lane_source != source) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &source, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_match_scalar_lane_chain(
    const loom_rewriter_t* rewriter, loom_value_id_t lane_value,
    iree_host_size_t lane_index, loom_vector_lane_chain_t* out_chain) {
  *out_chain = (loom_vector_lane_chain_t){0};
  out_chain->source = LOOM_VALUE_ID_INVALID;

  loom_value_id_t value = lane_value;
  while (out_chain->step_count < LOOM_VECTOR_LANE_CHAIN_MAX_STEPS) {
    loom_op_t* source_op = NULL;
    if (!loom_vector_value_def_op(rewriter, value, &source_op)) {
      break;
    }

    loom_vector_lane_chain_step_t step = {0};
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    if (!loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, source_op, &step, &input)) {
      break;
    }
    out_chain->steps[out_chain->step_count++] = step;
    value = input;
  }

  loom_op_t* overflow_op = NULL;
  if (out_chain->step_count == LOOM_VECTOR_LANE_CHAIN_MAX_STEPS &&
      loom_vector_value_def_op(rewriter, value, &overflow_op)) {
    loom_vector_lane_chain_step_t overflow_step = {0};
    loom_value_id_t overflow_input = LOOM_VALUE_ID_INVALID;
    if (loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, overflow_op, &overflow_step, &overflow_input)) {
      return false;
    }
  }

  return loom_vector_value_is_static_lane_extract(
      rewriter, value, lane_index, &out_chain->source, &out_chain->source_type);
}

static bool loom_vector_lane_chain_matches(
    const loom_vector_lane_chain_t* expected,
    const loom_vector_lane_chain_t* candidate) {
  if (candidate->source != expected->source ||
      candidate->step_count != expected->step_count ||
      !loom_type_equal(candidate->source_type, expected->source_type)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expected->step_count; ++i) {
    if (!loom_vector_lane_chain_steps_equal(&candidate->steps[i],
                                            &expected->steps[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_build_lane_chain_step(
    loom_builder_t* builder, const loom_vector_lane_chain_step_t* step,
    loom_value_id_t input, loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  switch (step->scalar_kind) {
    case LOOM_OP_INDEX_CAST:
      return loom_vector_index_cast_build(builder, input, input_type,
                                          result_type, location, out_op);
    case LOOM_OP_SCALAR_CEILF:
      return loom_vector_ceilf_build(builder, step->instance_flags, input,
                                     result_type, location, out_op);
    case LOOM_OP_SCALAR_FLOORF:
      return loom_vector_floorf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDF:
      return loom_vector_roundf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDEVENF:
      return loom_vector_roundevenf_build(builder, step->instance_flags, input,
                                          result_type, location, out_op);
    case LOOM_OP_SCALAR_TRUNCF:
      return loom_vector_truncf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_FPTOSI:
      return loom_vector_fptosi_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_FPTOUI:
      return loom_vector_fptoui_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_vector_trunci_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_SITOFP:
      return loom_vector_sitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_UITOFP:
      return loom_vector_uitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unsupported vector lane-chain step kind");
  return iree_ok_status();
}

iree_status_t loom_vector_from_elements_combine_lanes(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) {
    return iree_ok_status();
  }

  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  int64_t result_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &result_lane_count) ||
      result_lane_count != (int64_t)elements.count) {
    return iree_ok_status();
  }

  loom_vector_lane_chain_t expected_chain = {0};
  if (!loom_vector_match_scalar_lane_chain(rewriter, elements.values[0], 0,
                                           &expected_chain) ||
      expected_chain.step_count == 0) {
    return iree_ok_status();
  }
  int64_t source_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(expected_chain.source_type,
                                            &source_lane_count) ||
      source_lane_count != result_lane_count ||
      !loom_type_shape_equals(expected_chain.source_type, result_type)) {
    return iree_ok_status();
  }

  for (iree_host_size_t lane_index = 1; lane_index < elements.count;
       ++lane_index) {
    loom_vector_lane_chain_t candidate_chain = {0};
    if (!loom_vector_match_scalar_lane_chain(rewriter,
                                             elements.values[lane_index],
                                             lane_index, &candidate_chain) ||
        !loom_vector_lane_chain_matches(&expected_chain, &candidate_chain)) {
      return iree_ok_status();
    }
  }

  loom_type_t validated_input_type = expected_chain.source_type;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->input_type)) {
      return iree_ok_status();
    }
    validated_input_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      validated_input_type = result_type;
    }
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->result_type)) {
      return iree_ok_status();
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t replacement = expected_chain.source;
  loom_type_t input_type = expected_chain.source_type;
  loom_op_t* replacement_op = NULL;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    loom_type_t step_result_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      step_result_type = result_type;
    }

    IREE_RETURN_IF_ERROR(loom_vector_build_lane_chain_step(
        &rewriter->builder, step, replacement, input_type, step_result_type,
        op->location, &replacement_op));
    replacement = loom_op_const_results(replacement_op)[0];
    input_type = step_result_type;
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_match_static_scalar_insert(const loom_module_t* module,
                                                   const loom_op_t* op,
                                                   loom_type_t vector_type,
                                                   iree_host_size_t lane_count,
                                                   loom_value_id_t* out_value,
                                                   iree_host_size_t* out_lane) {
  if (!loom_vector_insert_isa(op) ||
      loom_vector_insert_indices(op).count != 0 ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_insert_result(op)),
          vector_type) ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_insert_dest(op)),
          vector_type)) {
    return false;
  }

  const loom_value_id_t value = loom_vector_insert_value(op);
  if (!loom_type_equal(loom_module_value_type(module, value),
                       loom_type_scalar(loom_type_element_type(vector_type)))) {
    return false;
  }

  const loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      (uint64_t)static_indices.i64_array[0] >= lane_count) {
    return false;
  }

  if (out_value) {
    *out_value = value;
  }
  if (out_lane) {
    *out_lane = (iree_host_size_t)static_indices.i64_array[0];
  }
  return true;
}

// Construction-only prefixes are subsumed by their terminal insert. Deferring
// them prevents the greedy driver from materializing and then immediately
// reopening one vector.from_elements for every lane in a linear chain.
static bool loom_vector_insert_has_only_construction_users(
    const loom_rewriter_t* rewriter, const loom_op_t* op,
    loom_type_t vector_type, iree_host_size_t lane_count) {
  const loom_value_id_t result = loom_vector_insert_result(op);
  const loom_value_t* result_value =
      loom_module_value(rewriter->module, result);
  const loom_use_t* uses = loom_value_uses(result_value);
  for (uint32_t i = 0; i < result_value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (!loom_vector_match_static_scalar_insert(
            rewriter->module, user_op, vector_type, lane_count,
            /*out_value=*/NULL, /*out_lane=*/NULL) ||
        loom_vector_insert_dest(user_op) != result) {
      return false;
    }
  }
  return true;
}

// Rejects ordinary partial updates before allocating lane storage. A chain with
// at least one insert per lane may completely overwrite an arbitrary base; a
// shorter chain is a constructor only when its base exposes every missing lane.
static bool loom_vector_insert_chain_may_form_elements(
    const loom_rewriter_t* rewriter, const loom_op_t* root_op,
    loom_type_t vector_type, iree_host_size_t lane_count) {
  const loom_op_t* insert_op = root_op;
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  iree_host_size_t insert_count = 0;
  while (insert_op != NULL && loom_vector_match_static_scalar_insert(
                                  rewriter->module, insert_op, vector_type,
                                  lane_count, /*out_value=*/NULL,
                                  /*out_lane=*/NULL)) {
    ++insert_count;
    base = loom_vector_insert_dest(insert_op);
    const loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    insert_op = base_op && loom_vector_insert_isa(base_op) ? base_op : NULL;
  }
  if (insert_count >= lane_count) {
    return true;
  }

  const loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
  if (base_op && loom_vector_splat_isa(base_op)) {
    return true;
  }
  return base_op && loom_vector_from_elements_isa(base_op) &&
         loom_vector_from_elements_elements(base_op).count == lane_count;
}

iree_status_t loom_vector_insert_combine_elements(loom_op_t* op,
                                                  loom_rewriter_t* rewriter,
                                                  bool* out_changed) {
  *out_changed = false;

  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_insert_result(op));
  int64_t lane_count_i64 = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &lane_count_i64) ||
      lane_count_i64 <= 0 || lane_count_i64 > UINT16_MAX) {
    return iree_ok_status();
  }
  const iree_host_size_t lane_count = (iree_host_size_t)lane_count_i64;

  if (!loom_vector_match_static_scalar_insert(
          rewriter->module, op, result_type, lane_count,
          /*out_value=*/NULL, /*out_lane=*/NULL) ||
      loom_vector_insert_has_only_construction_users(rewriter, op, result_type,
                                                     lane_count) ||
      !loom_vector_insert_chain_may_form_elements(rewriter, op, result_type,
                                                  lane_count)) {
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_value_id_scratch(
      rewriter->fact_table, lane_count, &elements));
  for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
    elements[lane] = LOOM_VALUE_ID_INVALID;
  }

  iree_host_size_t assigned_count = 0;
  loom_op_t* insert_op = op;
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  while (insert_op != NULL) {
    loom_value_id_t inserted_value = LOOM_VALUE_ID_INVALID;
    iree_host_size_t inserted_lane = 0;
    if (!loom_vector_match_static_scalar_insert(
            rewriter->module, insert_op, result_type, lane_count,
            &inserted_value, &inserted_lane)) {
      return iree_ok_status();
    }
    if (elements[inserted_lane] == LOOM_VALUE_ID_INVALID) {
      elements[inserted_lane] = inserted_value;
      ++assigned_count;
    }

    base = loom_vector_insert_dest(insert_op);
    if (assigned_count == lane_count) {
      break;
    }
    loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    insert_op = base_op && loom_vector_insert_isa(base_op) ? base_op : NULL;
  }

  if (assigned_count != lane_count) {
    loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    const loom_value_id_t* base_elements = NULL;
    loom_value_id_t splat_element = LOOM_VALUE_ID_INVALID;
    if (base_op && loom_vector_splat_isa(base_op)) {
      splat_element = loom_vector_splat_scalar(base_op);
    } else if (base_op && loom_vector_from_elements_isa(base_op)) {
      const loom_value_slice_t element_slice =
          loom_vector_from_elements_elements(base_op);
      if (element_slice.count != lane_count) {
        return iree_ok_status();
      }
      base_elements = element_slice.values;
    } else {
      return iree_ok_status();
    }

    for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
      if (elements[lane] == LOOM_VALUE_ID_INVALID) {
        elements[lane] = base_elements ? base_elements[lane] : splat_element;
      }
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, lane_count, result_type, op->location,
      &replacement_op));
  const loom_value_id_t replacement =
      loom_vector_from_elements_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}
