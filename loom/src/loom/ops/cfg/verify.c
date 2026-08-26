// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/successor_verify.h"

static iree_status_t loom_cfg_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_cfg_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_cfg_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_cfg_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_cfg_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_cfg_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_host_size_t actual_count,
    iree_string_view_t expected_field_name, iree_host_size_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32((uint32_t)iree_min(actual_count, UINT32_MAX)),
      loom_param_string(expected_field_name),
      loom_param_u32((uint32_t)iree_min(expected_count, UINT32_MAX)),
  };
  return loom_cfg_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_cfg_br_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t args = loom_cfg_br_args(op);
  return loom_ops_verify_successor_args(module, emitter, op, IREE_SV("cfg.br"),
                                        0, loom_cfg_br_dest(op), args.values,
                                        args.count);
}

iree_status_t loom_cfg_cond_br_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("cfg.cond_br"), 0,
      loom_cfg_cond_br_true_dest(op), NULL, 0));
  return loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("cfg.cond_br"), 1,
      loom_cfg_cond_br_false_dest(op), NULL, 0);
}

iree_status_t loom_cfg_switch_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  if (op->successor_count == 0) {
    return loom_cfg_emit_count_mismatch(
        emitter, op, IREE_SV("successors"), 0,
        IREE_SV("default destination plus case destinations"), 1);
  }

  loom_attribute_t case_keys = loom_cfg_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY) {
    return loom_cfg_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("case_keys"), case_keys.kind, LOOM_ATTR_I64_ARRAY);
  }
  if (case_keys.count == 0) {
    return loom_cfg_emit_attribute_value_constraint(
        emitter, op, IREE_SV("case_keys"), 0,
        IREE_SV("at least one explicit case key"));
  }
  if (!case_keys.i64_array) {
    return loom_cfg_emit_attribute_value_constraint(
        emitter, op, IREE_SV("case_keys"), case_keys.count,
        IREE_SV("non-null i64_array storage when count is non-zero"));
  }
  for (uint16_t i = 1; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] <= case_keys.i64_array[i - 1]) {
      return loom_cfg_emit_attribute_value_constraint(
          emitter, op, IREE_SV("case_keys"), case_keys.i64_array[i],
          IREE_SV("strictly increasing sorted unique case key"));
    }
  }

  loom_successor_slice_t case_dests = loom_cfg_switch_case_dests(op);
  if (case_dests.count != case_keys.count) {
    return loom_cfg_emit_count_mismatch(emitter, op, IREE_SV("case_dests"),
                                        case_dests.count, IREE_SV("case_keys"),
                                        case_keys.count);
  }

  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("cfg.switch"), 0,
      loom_cfg_switch_default_dest(op), NULL, 0));
  for (uint16_t i = 0; i < case_dests.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
        module, emitter, op, IREE_SV("cfg.switch"), i + 1, case_dests.blocks[i],
        NULL, 0));
  }
  return iree_ok_status();
}
