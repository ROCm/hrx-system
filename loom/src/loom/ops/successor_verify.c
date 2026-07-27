// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/successor_verify.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

static iree_status_t loom_ops_successor_verify_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static uint32_t loom_ops_successor_saturating_u32(iree_host_size_t value) {
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
}

static bool loom_ops_successor_value_is_defined(const loom_module_t* module,
                                                loom_value_id_t value_id) {
  return value_id != LOOM_VALUE_ID_INVALID && value_id < module->values.count;
}

static iree_status_t loom_ops_successor_emit_arg_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t op_name, uint8_t successor_index,
    iree_host_size_t actual_count, iree_host_size_t expected_count) {
  loom_diagnostic_field_ref_t successor_ref = loom_diagnostic_field_ref(
      LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, successor_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_with_field_ref(loom_param_u32(successor_index), successor_ref),
      loom_param_u32(loom_ops_successor_saturating_u32(actual_count)),
      loom_param_u32(loom_ops_successor_saturating_u32(expected_count)),
  };
  return loom_ops_successor_verify_emit(emitter, op, LOOM_ERR_STRUCTURE_025,
                                        params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_ops_successor_emit_arg_type_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t op_name, uint8_t successor_index,
    uint16_t argument_index, loom_type_t actual_type,
    loom_type_t expected_type) {
  loom_diagnostic_field_ref_t successor_ref = loom_diagnostic_field_ref(
      LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, successor_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_with_field_ref(loom_param_u32(successor_index), successor_ref),
      loom_param_u32(argument_index),
      loom_param_type(actual_type),
      loom_param_type(expected_type),
  };
  return loom_ops_successor_verify_emit(emitter, op, LOOM_ERR_STRUCTURE_026,
                                        params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_ops_verify_successor_args(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t op_name, uint8_t successor_index,
    const loom_block_t* target, const loom_value_id_t* args,
    iree_host_size_t arg_count) {
  if (!target) return iree_ok_status();
  if (arg_count != target->arg_count) {
    return loom_ops_successor_emit_arg_count_mismatch(
        emitter, op, op_name, successor_index, arg_count, target->arg_count);
  }
  loom_type_value_remap_t type_remap = {
      .source_values = target->arg_ids,
      .target_values = args,
      .count = target->arg_count,
  };
  for (uint16_t i = 0; i < target->arg_count; ++i) {
    loom_value_id_t actual_id = args[i];
    loom_value_id_t expected_id = loom_block_arg_id(target, i);
    if (!loom_ops_successor_value_is_defined(module, actual_id) ||
        !loom_ops_successor_value_is_defined(module, expected_id)) {
      continue;
    }
    loom_type_t actual_type = loom_module_value_type(module, actual_id);
    loom_type_t expected_type = loom_module_value_type(module, expected_id);
    if (!loom_type_equal_after_value_remap(expected_type, actual_type,
                                           &type_remap)) {
      return loom_ops_successor_emit_arg_type_mismatch(
          emitter, op, op_name, successor_index, i, actual_type, expected_type);
    }
  }
  return iree_ok_status();
}
