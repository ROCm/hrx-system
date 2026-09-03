// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"

typedef enum loom_pipeline_flow_field_kind_e {
  LOOM_PIPELINE_FLOW_FIELD_OPERAND = 0,
  LOOM_PIPELINE_FLOW_FIELD_RESULT = 1,
} loom_pipeline_flow_field_kind_t;

#define LOOM_PIPELINE_ARGUMENT_COUNT_UNCONSTRAINED UINT32_MAX

static iree_status_t loom_pipeline_emit(iree_diagnostic_emitter_t emitter,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_pipeline_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_pipeline_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_pipeline_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_pipeline_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_pipeline_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t actual_field, uint32_t actual_count,
    iree_string_view_t expected_field, uint32_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(actual_field),
      loom_param_u32(actual_count),
      loom_param_string(expected_field),
      loom_param_u32(expected_count),
  };
  return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                            IREE_ARRAYSIZE(params));
}

static bool loom_pipeline_type_is_opaque(const loom_module_t* module,
                                         loom_type_t type,
                                         iree_string_view_t name) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  return name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[name_id], name);
}

static bool loom_pipeline_type_is_group(const loom_module_t* module,
                                        loom_type_t type) {
  return loom_pipeline_type_is_opaque(module, type, IREE_SV("group"));
}

static bool loom_pipeline_type_is_flow_tile(const loom_module_t* module,
                                            loom_type_t type,
                                            loom_type_t* out_tile_type) {
  *out_tile_type = (loom_type_t){0};
  if (!loom_pipeline_flow_type_isa(type)) return false;
  const loom_type_id_t element_type_id =
      loom_pipeline_flow_type_element_type(type);
  const loom_type_t element_type = module->types.entries[element_type_id];
  if (!loom_type_is_tile(element_type)) return false;
  *out_tile_type = element_type;
  return true;
}

static bool loom_pipeline_tile_matches_view(loom_type_t tile_type,
                                            loom_type_t view_type,
                                            uint8_t view_axis_offset) {
  if (!loom_type_is_tile(tile_type) || !loom_type_is_view(view_type) ||
      loom_type_element_type(tile_type) != loom_type_element_type(view_type)) {
    return false;
  }
  const uint8_t tile_rank = loom_type_rank(tile_type);
  const uint8_t view_rank = loom_type_rank(view_type);
  if ((uint16_t)tile_rank + view_axis_offset != view_rank) return false;
  for (uint8_t axis = 0; axis < tile_rank; ++axis) {
    if (loom_type_dim(tile_type, axis) !=
        loom_type_dim(view_type, axis + view_axis_offset)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_pipeline_verify_operand_group(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t operand_name,
    loom_value_id_t value_id, bool* out_valid) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_valid = loom_pipeline_type_is_group(module, type);
  if (*out_valid) return iree_ok_status();
  return loom_pipeline_emit_operand_constraint(emitter, op, operand_name, type,
                                               IREE_SV("group"));
}

static iree_status_t loom_pipeline_verify_operand_flow(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t operand_name,
    loom_value_id_t value_id, loom_type_t* out_tile_type, bool* out_valid) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_valid = loom_pipeline_type_is_flow_tile(module, type, out_tile_type);
  if (*out_valid) return iree_ok_status();
  return loom_pipeline_emit_operand_constraint(
      emitter, op, operand_name, type, IREE_SV("pipeline.flow<tile<...>>"));
}

static iree_status_t loom_pipeline_verify_result_flow(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t result_name,
    loom_value_id_t value_id, loom_type_t* out_tile_type, bool* out_valid) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_valid = loom_pipeline_type_is_flow_tile(module, type, out_tile_type);
  if (*out_valid) return iree_ok_status();
  return loom_pipeline_emit_result_constraint(
      emitter, op, result_name, type, IREE_SV("pipeline.flow<tile<...>>"));
}

static iree_status_t loom_pipeline_verify_flow_sequence(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_name,
    loom_value_slice_t values, loom_pipeline_flow_field_kind_t field_kind,
    bool* out_valid) {
  *out_valid = true;
  for (uint16_t i = 0; i < values.count; ++i) {
    char name[48];
    iree_snprintf(name, sizeof(name), "%.*s %u", (int)field_name.size,
                  field_name.data, i);
    loom_type_t tile_type = {0};
    bool value_valid = false;
    iree_status_t status =
        field_kind == LOOM_PIPELINE_FLOW_FIELD_RESULT
            ? loom_pipeline_verify_result_flow(
                  module, op, emitter, iree_make_cstring_view(name),
                  values.values[i], &tile_type, &value_valid)
            : loom_pipeline_verify_operand_flow(
                  module, op, emitter, iree_make_cstring_view(name),
                  values.values[i], &tile_type, &value_valid);
    if (!iree_status_is_ok(status)) return status;
    if (!value_valid) {
      *out_valid = false;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_verify_entry(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_symbol_ref_t entry_ref,
    uint32_t expected_argument_count) {
  const loom_op_t* entry_op =
      module->symbols.entries[entry_ref.symbol_id].defining_op;
  const loom_func_like_t entry = loom_func_like_const_cast(module, entry_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(entry, &argument_count);
  if (expected_argument_count != LOOM_PIPELINE_ARGUMENT_COUNT_UNCONSTRAINED &&
      argument_count != expected_argument_count) {
    return loom_pipeline_emit_count_mismatch(
        emitter, op, IREE_SV("entry argument"), argument_count,
        IREE_SV("stage flow"), expected_argument_count);
  }
  if (entry_op->result_count != 0) {
    return loom_pipeline_emit_count_mismatch(
        emitter, op, IREE_SV("entry result"), entry_op->result_count,
        IREE_SV("buffer-ABI result"), 0);
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, argument_ids[i]);
    if (loom_type_satisfies_constraint(type, LOOM_TYPE_CONSTRAINT_BUFFER)) {
      continue;
    }
    char name[40];
    iree_snprintf(name, sizeof(name), "entry argument %u", i);
    return loom_pipeline_emit_operand_constraint(
        emitter, op, iree_make_cstring_view(name), type, IREE_SV("buffer"));
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_def_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  const loom_func_like_t pipeline = loom_func_like_const_cast(module, op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(pipeline, &argument_count);
  const int64_t specialization_count =
      loom_func_like_specialization_count(pipeline);
  if (specialization_count < 0 || specialization_count > argument_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("specialization_count")),
        loom_param_i64(specialization_count),
        loom_param_string(IREE_SV("between zero and the argument count")),
    };
    return loom_pipeline_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                              IREE_ARRAYSIZE(params));
  }
  for (uint16_t i = (uint16_t)specialization_count; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, argument_ids[i]);
    if (loom_type_satisfies_constraint(type, LOOM_TYPE_CONSTRAINT_BUFFER)) {
      continue;
    }
    char name[40];
    iree_snprintf(name, sizeof(name), "launch binding %u",
                  i - (uint16_t)specialization_count);
    return loom_pipeline_emit_operand_constraint(
        emitter, op, iree_make_cstring_view(name), type, IREE_SV("buffer"));
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_scatter_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_group(
      module, op, emitter, IREE_SV("group"), loom_pipeline_scatter_group(op),
      &valid));
  if (!valid) return iree_ok_status();
  loom_type_t tile_type = {0};
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_result_flow(
      module, op, emitter, IREE_SV("result"), loom_pipeline_scatter_result(op),
      &tile_type, &valid));
  if (!valid) return iree_ok_status();
  const loom_type_t source_type =
      loom_module_value_type(module, loom_pipeline_scatter_source(op));
  if (loom_type_rank(source_type) > 0 &&
      loom_pipeline_tile_matches_view(tile_type, source_type, 1)) {
    return iree_ok_status();
  }
  return loom_pipeline_emit_result_constraint(
      emitter, op, IREE_SV("result"),
      loom_module_value_type(module, loom_pipeline_scatter_result(op)),
      IREE_SV("pipeline.flow tile matching the source view suffix"));
}

iree_status_t loom_pipeline_read_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  bool valid = false;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_verify_operand_group(module, op, emitter, IREE_SV("group"),
                                         loom_pipeline_read_group(op), &valid));
  if (!valid) return iree_ok_status();
  loom_type_t tile_type = {0};
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_result_flow(
      module, op, emitter, IREE_SV("result"), loom_pipeline_read_result(op),
      &tile_type, &valid));
  if (!valid) return iree_ok_status();
  const loom_type_t source_type =
      loom_module_value_type(module, loom_pipeline_read_source(op));
  if (loom_pipeline_tile_matches_view(tile_type, source_type, 0)) {
    return iree_ok_status();
  }
  return loom_pipeline_emit_result_constraint(
      emitter, op, IREE_SV("result"),
      loom_module_value_type(module, loom_pipeline_read_result(op)),
      IREE_SV("pipeline.flow tile matching the complete source view"));
}

iree_status_t loom_pipeline_stage_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_group(
      module, op, emitter, IREE_SV("group"), loom_pipeline_stage_group(op),
      &valid));
  if (!valid) return iree_ok_status();
  const loom_value_slice_t inputs = loom_pipeline_stage_inputs(op);
  const loom_value_slice_t outputs = loom_pipeline_stage_outputs(op);
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_flow_sequence(
      module, op, emitter, IREE_SV("input"), inputs,
      LOOM_PIPELINE_FLOW_FIELD_OPERAND, &valid));
  if (!valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_flow_sequence(
      module, op, emitter, IREE_SV("output"), outputs,
      LOOM_PIPELINE_FLOW_FIELD_RESULT, &valid));
  if (!valid) return iree_ok_status();
  return loom_pipeline_verify_entry(module, op, emitter,
                                    loom_pipeline_stage_entry(op),
                                    (uint32_t)inputs.count + outputs.count);
}

iree_status_t loom_pipeline_buffer_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t tile_type = {0};
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_flow(
      module, op, emitter, IREE_SV("source"), loom_pipeline_buffer_source(op),
      &tile_type, &valid));
  return iree_ok_status();
}

iree_status_t loom_pipeline_reduce_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_group(
      module, op, emitter, IREE_SV("source group"),
      loom_pipeline_reduce_source_group(op), &valid));
  if (!valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_group(
      module, op, emitter, IREE_SV("target group"),
      loom_pipeline_reduce_target_group(op), &valid));
  if (!valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_flow_sequence(
      module, op, emitter, IREE_SV("source input"),
      loom_pipeline_reduce_source_inputs(op), LOOM_PIPELINE_FLOW_FIELD_OPERAND,
      &valid));
  if (!valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_flow_sequence(
      module, op, emitter, IREE_SV("target input"),
      loom_pipeline_reduce_target_inputs(op), LOOM_PIPELINE_FLOW_FIELD_OPERAND,
      &valid));
  if (!valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_flow_sequence(
      module, op, emitter, IREE_SV("output"), loom_pipeline_reduce_outputs(op),
      LOOM_PIPELINE_FLOW_FIELD_RESULT, &valid));
  if (!valid) return iree_ok_status();
  return loom_pipeline_verify_entry(module, op, emitter,
                                    loom_pipeline_reduce_entry(op),
                                    LOOM_PIPELINE_ARGUMENT_COUNT_UNCONSTRAINED);
}

iree_status_t loom_pipeline_write_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t tile_type = {0};
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_verify_operand_flow(
      module, op, emitter, IREE_SV("source"), loom_pipeline_write_source(op),
      &tile_type, &valid));
  if (!valid) return iree_ok_status();
  const loom_type_t target_type =
      loom_module_value_type(module, loom_pipeline_write_target(op));
  if (loom_pipeline_tile_matches_view(tile_type, target_type, 0)) {
    return iree_ok_status();
  }
  return loom_pipeline_emit_operand_constraint(
      emitter, op, IREE_SV("source"),
      loom_module_value_type(module, loom_pipeline_write_source(op)),
      IREE_SV("pipeline.flow tile matching the complete target view"));
}
