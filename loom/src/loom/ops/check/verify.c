// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/ancestry.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/function_contract_verify.h"

static iree_status_t loom_check_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_check_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t actual_field, uint16_t actual_count,
    iree_string_view_t expected_field, uint16_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(actual_field),
      loom_param_u32(actual_count),
      loom_param_string(expected_field),
      loom_param_u32(expected_count),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_check_emit_positive_count(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attribute_name, int64_t count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(attribute_name),
      loom_param_i64(count),
      loom_param_string(IREE_SV("a positive finite domain size")),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_check_emit_operand_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_check_emit_result_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_check_emit_type_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t actual_field, loom_type_t actual_type,
    iree_string_view_t expected_field, loom_type_t expected_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(actual_field),
      loom_param_type(actual_type),
      loom_param_string(expected_field),
      loom_param_type(expected_type),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                         IREE_ARRAYSIZE(params));
}

static bool loom_check_entropy_type_isa(const loom_module_t* module,
                                        loom_type_t type) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  return name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count &&
         iree_string_view_equal(
             loom_string_table_get(&module->strings, name_id),
             IREE_SV("check.entropy"));
}

static bool loom_check_index_type_isa(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
}

static bool loom_check_i64_type_isa(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

static iree_status_t loom_check_verify_domain_arguments(
    const loom_module_t* module, const loom_op_t* op, const loom_region_t* body,
    iree_string_view_t field_name, bool* out_valid,
    iree_diagnostic_emitter_t emitter) {
  *out_valid = false;
  const uint16_t argument_count = loom_region_entry_arg_count(body);
  if (argument_count != 2) {
    return loom_check_emit_count_mismatch(emitter, op, field_name,
                                          argument_count,
                                          IREE_SV("domain arguments"), 2);
  }
  const loom_type_t ordinal_type =
      loom_module_value_type(module, loom_region_entry_arg_id(body, 0));
  if (!loom_check_index_type_isa(ordinal_type)) {
    return loom_check_emit_operand_type(emitter, op, IREE_SV("domain ordinal"),
                                        ordinal_type, IREE_SV("index"));
  }
  const loom_type_t entropy_type =
      loom_module_value_type(module, loom_region_entry_arg_id(body, 1));
  if (!loom_check_entropy_type_isa(module, entropy_type)) {
    return loom_check_emit_operand_type(emitter, op, IREE_SV("domain entropy"),
                                        entropy_type, IREE_SV("check.entropy"));
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_check_verify_scenario_trials(
    const loom_op_t* op, const loom_region_t* body,
    iree_diagnostic_emitter_t emitter) {
  const loom_block_t* block = loom_region_const_entry_block(body);
  const loom_op_t* nested_op = NULL;
  loom_block_for_each_op(block, nested_op) {
    if (loom_check_trial_isa(nested_op)) {
      return iree_ok_status();
    }
  }
  return loom_check_emit_count_mismatch(emitter, op,
                                        IREE_SV("trial operations"), 0,
                                        IREE_SV("minimum trial operations"), 1);
}

iree_status_t loom_check_scenario_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  const loom_attribute_t count_attribute = loom_op_const_attrs(op)[2];
  const loom_region_t* body = loom_check_scenario_body(op);
  if (loom_attr_is_absent(count_attribute)) {
    const uint16_t argument_count = loom_region_entry_arg_count(body);
    if (argument_count == 0) {
      return loom_check_verify_scenario_trials(op, body, emitter);
    }
    return loom_check_emit_count_mismatch(
        emitter, op, IREE_SV("scenario body arguments"), argument_count,
        IREE_SV("unconfigured scenario body arguments"), 0);
  }

  const int64_t configuration_count =
      loom_check_scenario_configuration_count(op);
  if (configuration_count <= 0) {
    return loom_check_emit_positive_count(
        emitter, op, IREE_SV("configuration_count"), configuration_count);
  }
  bool arguments_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_verify_domain_arguments(
      module, op, body, IREE_SV("configuration arguments"), &arguments_valid,
      emitter));
  if (!arguments_valid) {
    return iree_ok_status();
  }
  return loom_check_verify_scenario_trials(op, body, emitter);
}

iree_status_t loom_check_trial_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  const int64_t trial_count = loom_check_trial_trial_count(op);
  if (trial_count <= 0) {
    return loom_check_emit_positive_count(emitter, op, IREE_SV("trial_count"),
                                          trial_count);
  }
  const loom_region_t* body = loom_check_trial_body(op);
  bool arguments_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_verify_domain_arguments(
      module, op, body, IREE_SV("trial arguments"), &arguments_valid, emitter));
  if (!arguments_valid) {
    return iree_ok_status();
  }
  const loom_block_t* block = loom_region_const_entry_block(body);
  if (block->op_count == 0) {
    return loom_check_emit_count_mismatch(emitter, op,
                                          IREE_SV("trial body operations"), 0,
                                          IREE_SV("terminating action"), 1);
  }
  const loom_op_t* terminator = loom_block_const_last_op(block);
  if (loom_check_compare_isa(terminator) || loom_check_invoke_isa(terminator)) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, terminator)),
      loom_param_string(IREE_SV("a check.compare or check.invoke terminator")),
      loom_param_string(IREE_SV("the end of a check.trial body")),
  };
  return loom_check_emit(emitter, terminator, LOOM_ERR_STRUCTURE_031, params,
                         IREE_ARRAYSIZE(params));
}

static const loom_symbol_t* loom_check_lookup_subject(
    const loom_module_t* module, loom_symbol_ref_t callee) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  return symbol->definition && symbol->defining_op ? symbol : NULL;
}

static iree_status_t loom_check_verify_configuration_values(
    const loom_module_t* module, const loom_op_t* action,
    loom_value_slice_t parameters, bool* out_valid,
    iree_diagnostic_emitter_t emitter) {
  *out_valid = false;
  const loom_op_t* trial = action->parent_op;
  for (uint16_t i = 0; i < parameters.count; ++i) {
    if (!loom_op_subtree_defines_value(module, trial, parameters.values[i])) {
      continue;
    }
    char field_name[32];
    iree_snprintf(field_name, sizeof(field_name), "call_parameters %u", i);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, action)),
        loom_param_string(iree_make_cstring_view(field_name)),
        loom_param_string(IREE_SV("a value defined outside the trial domain")),
    };
    return loom_check_emit(emitter, action, LOOM_ERR_STRUCTURE_032, params,
                           IREE_ARRAYSIZE(params));
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_check_verify_argument_group(
    const loom_module_t* module, const loom_op_t* action,
    loom_value_slice_t actual_values, const loom_value_id_t* expected_values,
    uint16_t expected_count, const loom_type_value_remap_t* value_remap,
    loom_function_call_argument_match_flags_t match_flags,
    iree_string_view_t actual_field, iree_string_view_t expected_field,
    bool* out_valid, iree_diagnostic_emitter_t emitter) {
  *out_valid = false;
  if (actual_values.count != expected_count) {
    return loom_check_emit_count_mismatch(emitter, action, actual_field,
                                          actual_values.count, expected_field,
                                          expected_count);
  }
  for (uint16_t i = 0; i < actual_values.count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, actual_values.values[i]);
    const loom_type_t expected_type =
        loom_module_value_type(module, expected_values[i]);
    bool matches = false;
    IREE_RETURN_IF_ERROR(loom_function_call_argument_type_matches(
        module, actual_type, expected_type, value_remap, match_flags,
        &matches));
    if (matches) {
      continue;
    }
    char actual_name[48];
    char expected_name[48];
    iree_snprintf(actual_name, sizeof(actual_name), "%.*s %u",
                  (int)actual_field.size, actual_field.data, i);
    iree_snprintf(expected_name, sizeof(expected_name), "%.*s %u",
                  (int)expected_field.size, expected_field.data, i);
    return loom_check_emit_type_mismatch(
        emitter, action, iree_make_cstring_view(actual_name), actual_type,
        iree_make_cstring_view(expected_name), expected_type);
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_check_verify_subject_operands(
    const loom_module_t* module, const loom_op_t* action,
    const loom_symbol_t* symbol, loom_value_slice_t parameters,
    loom_value_slice_t arguments, bool* out_valid,
    iree_diagnostic_emitter_t emitter) {
  *out_valid = false;
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE)) {
    if (parameters.count != 0) {
      return loom_check_emit_count_mismatch(
          emitter, action, IREE_SV("call parameters"), parameters.count,
          IREE_SV("callable parameters"), 0);
    }
    *out_valid = true;
    return iree_ok_status();
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    const loom_value_slice_t workload_arguments =
        loom_kernel_workload_arg_ids(module, symbol->defining_op);
    const loom_func_like_t function =
        loom_func_like_const_cast(module, symbol->defining_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* argument_ids =
        loom_func_like_arg_ids(function, &argument_count);
    const loom_type_value_remap_t argument_remap = {
        .source_values = argument_ids,
        .target_values = arguments.values,
        .count =
            arguments.count < argument_count ? arguments.count : argument_count,
        .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
    };
    const loom_type_value_remap_t workload_remap = {
        .source_values = workload_arguments.values,
        .target_values = parameters.values,
        .count = parameters.count < workload_arguments.count
                     ? parameters.count
                     : workload_arguments.count,
        .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
        .next = argument_remap.count ? &argument_remap : NULL,
    };
    const loom_type_value_remap_t* value_remap =
        workload_remap.count ? &workload_remap
                             : (argument_remap.count ? &argument_remap : NULL);
    bool group_valid = false;
    IREE_RETURN_IF_ERROR(loom_check_verify_argument_group(
        module, action, parameters, workload_arguments.values,
        workload_arguments.count, value_remap, /*match_flags=*/0,
        IREE_SV("call parameters"), IREE_SV("kernel workload arguments"),
        &group_valid, emitter));
    if (!group_valid) {
      return iree_ok_status();
    }
    return loom_check_verify_argument_group(
        module, action, arguments, argument_ids, argument_count, value_remap,
        LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
        IREE_SV("arguments"), IREE_SV("kernel ABI arguments"), out_valid,
        emitter);
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM |
                                         LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    const loom_func_like_t function =
        loom_func_like_const_cast(module, symbol->defining_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* argument_ids =
        loom_func_like_arg_ids(function, &argument_count);
    const int64_t specialization_count =
        loom_func_like_specialization_count(function);
    if (specialization_count < 0 || specialization_count > argument_count) {
      return iree_ok_status();
    }
    const uint16_t staged_count = (uint16_t)specialization_count;
    if (parameters.count == staged_count) {
      bool configurations_valid = false;
      IREE_RETURN_IF_ERROR(loom_check_verify_configuration_values(
          module, action, parameters, &configurations_valid, emitter));
      if (!configurations_valid) {
        return iree_ok_status();
      }
    }
    const loom_value_slice_t actual_values = {
        .values = loom_op_operands(action),
        .count = action->operand_count,
    };
    const loom_type_value_remap_t value_remap = {
        .source_values = argument_ids,
        .target_values = actual_values.values,
        .count = actual_values.count < argument_count ? actual_values.count
                                                      : argument_count,
        .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
    };
    bool group_valid = false;
    IREE_RETURN_IF_ERROR(loom_check_verify_argument_group(
        module, action, parameters, argument_ids, staged_count, &value_remap,
        /*match_flags=*/0, IREE_SV("call parameters"),
        IREE_SV("subject specialization arguments"), &group_valid, emitter));
    if (!group_valid) {
      return iree_ok_status();
    }
    return loom_check_verify_argument_group(
        module, action, arguments, argument_ids + staged_count,
        argument_count - staged_count, &value_remap,
        LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
        IREE_SV("arguments"), IREE_SV("subject launch bindings"), out_valid,
        emitter);
  }

  return iree_ok_status();
}

iree_status_t loom_check_expect_pair_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t actual_id = loom_op_const_operands(op)[0];
  const loom_value_id_t expected_id = loom_op_const_operands(op)[1];
  const loom_type_t actual_type = loom_module_value_type(module, actual_id);
  const loom_type_t expected_type = loom_module_value_type(module, expected_id);

  loom_type_value_remap_t pair_remap = {0};
  const loom_type_value_remap_t* value_remap = NULL;
  const loom_op_t* compare_op = op->parent_op;
  if (compare_op && loom_check_compare_isa(compare_op)) {
    const loom_attribute_t count_attribute = loom_op_const_attrs(compare_op)[1];
    const int64_t actual_count_i64 =
        loom_attr_is_absent(count_attribute)
            ? 0
            : loom_check_compare_actual_count(compare_op);
    const loom_block_t* block = loom_region_const_entry_block(
        loom_check_compare_comparison(compare_op));
    if (actual_count_i64 > 0 && actual_count_i64 <= UINT16_MAX / 2u &&
        block->arg_count == (uint16_t)actual_count_i64 * 2u) {
      const uint16_t actual_count = (uint16_t)actual_count_i64;
      pair_remap = (loom_type_value_remap_t){
          .source_values = block->arg_ids + actual_count,
          .target_values = block->arg_ids,
          .count = actual_count,
          .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
      };
      value_remap = &pair_remap;
    }
  }

  bool matches = false;
  IREE_RETURN_IF_ERROR(loom_function_call_argument_type_matches(
      module, actual_type, expected_type, value_remap, /*flags=*/0, &matches));
  if (matches) {
    return iree_ok_status();
  }
  return loom_check_emit_type_mismatch(emitter, op, IREE_SV("actual"),
                                       actual_type, IREE_SV("expected"),
                                       expected_type);
}

static bool loom_check_expectation_pair_values(const loom_op_t* op,
                                               loom_value_id_t* out_actual,
                                               loom_value_id_t* out_expected) {
  if (loom_check_expect_equal_isa(op) || loom_check_expect_bitwise_isa(op) ||
      loom_check_expect_close_isa(op)) {
    *out_actual = loom_op_const_operands(op)[0];
    *out_expected = loom_op_const_operands(op)[1];
    return true;
  }
  return false;
}

static iree_status_t loom_check_emit_result_origin(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t field_name,
    iree_string_view_t required_origin) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(field_name),
      loom_param_string(required_origin),
  };
  return loom_check_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_check_verify_result_pair_observation(
    const loom_module_t* module, const loom_op_t* compare_op,
    uint16_t result_index, loom_value_id_t actual_id,
    loom_value_id_t expected_id, iree_diagnostic_emitter_t emitter) {
  bool observed = false;
  const loom_value_t* actual_value = loom_module_value(module, actual_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(actual_value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    loom_value_id_t used_actual = LOOM_VALUE_ID_INVALID;
    loom_value_id_t used_expected = LOOM_VALUE_ID_INVALID;
    if (loom_check_expectation_pair_values(user_op, &used_actual,
                                           &used_expected)) {
      if (operand_index != 0) {
        return loom_check_emit_result_origin(
            module, emitter, user_op, IREE_SV("expected"),
            IREE_SV("the corresponding oracle result"));
      }
      if (used_expected != expected_id) {
        return loom_check_emit_result_origin(
            module, emitter, user_op, IREE_SV("expected"),
            IREE_SV("the oracle result paired with the actual result"));
      }
      observed = true;
      continue;
    }
    if (loom_check_expect_shape_isa(user_op) && operand_index == 0) {
      continue;
    }
    if (loom_check_expect_shape_isa(user_op)) {
      return loom_check_emit_result_origin(
          module, emitter, user_op, IREE_SV("shape dimensions"),
          IREE_SV("values defined outside the comparison result bindings"));
    }
  }

  const loom_value_t* expected_value = loom_module_value(module, expected_id);
  loom_value_for_each_use(expected_value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    loom_value_id_t used_actual = LOOM_VALUE_ID_INVALID;
    loom_value_id_t used_expected = LOOM_VALUE_ID_INVALID;
    if (loom_check_expectation_pair_values(user_op, &used_actual,
                                           &used_expected)) {
      if (operand_index != 1) {
        return loom_check_emit_result_origin(
            module, emitter, user_op, IREE_SV("actual"),
            IREE_SV("the corresponding target result"));
      }
      if (used_actual != actual_id) {
        return loom_check_emit_result_origin(
            module, emitter, user_op, IREE_SV("actual"),
            IREE_SV("the target result paired with the expected result"));
      }
      continue;
    }
    if (loom_check_expect_shape_isa(user_op) && operand_index == 0) {
      continue;
    }
    if (loom_check_expect_shape_isa(user_op)) {
      return loom_check_emit_result_origin(
          module, emitter, user_op, IREE_SV("shape dimensions"),
          IREE_SV("values defined outside the comparison result bindings"));
    }
  }

  if (observed) {
    return iree_ok_status();
  }
  char field_name[32];
  iree_snprintf(field_name, sizeof(field_name), "result pair %u", result_index);
  return loom_check_emit_result_origin(
      module, emitter, compare_op, iree_make_cstring_view(field_name),
      IREE_SV("an explicit paired check.expect operation"));
}

static iree_status_t loom_check_verify_compare_body(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_slice_t* out_actual_results,
    loom_value_slice_t* out_expected_results, bool* out_valid,
    iree_diagnostic_emitter_t emitter) {
  *out_valid = false;
  const loom_attribute_t count_attribute = loom_op_const_attrs(op)[1];
  const loom_region_t* comparison = loom_check_compare_comparison(op);
  const loom_block_t* block = loom_region_const_entry_block(comparison);
  const int64_t actual_count_i64 = loom_attr_is_absent(count_attribute)
                                       ? 0
                                       : loom_check_compare_actual_count(op);
  if (!loom_attr_is_absent(count_attribute) &&
      (actual_count_i64 <= 0 || actual_count_i64 > UINT16_MAX / 2u)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("actual_count")),
        loom_param_i64(actual_count_i64),
        loom_param_string(IREE_SV("a count in [1, 32767]")),
    };
    return loom_check_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                           IREE_ARRAYSIZE(params));
  }
  const uint16_t actual_count = (uint16_t)actual_count_i64;
  if (block->arg_count != actual_count * 2u) {
    return loom_check_emit_count_mismatch(
        emitter, op, IREE_SV("comparison arguments"), block->arg_count,
        IREE_SV("paired actual and expected arguments"), actual_count * 2u);
  }
  const loom_type_value_remap_t pair_remap = {
      .source_values = actual_count ? block->arg_ids + actual_count : NULL,
      .target_values = block->arg_ids,
      .count = actual_count,
      .flags = LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE,
  };
  for (uint16_t i = 0; i < actual_count; ++i) {
    const loom_type_t actual_type = loom_block_arg_type(module, block, i);
    const loom_type_t expected_type =
        loom_block_arg_type(module, block, actual_count + i);
    bool matches = false;
    IREE_RETURN_IF_ERROR(loom_function_call_argument_type_matches(
        module, actual_type, expected_type, &pair_remap, /*flags=*/0,
        &matches));
    if (matches) {
      continue;
    }
    return loom_check_emit_type_mismatch(
        emitter, op, IREE_SV("actual result"), actual_type,
        IREE_SV("expected result"), expected_type);
  }

  const loom_op_t* nested_op = NULL;
  loom_block_for_each_op(block, nested_op) {
    if (loom_check_expect_equal_isa(nested_op) ||
        loom_check_expect_bitwise_isa(nested_op) ||
        loom_check_expect_close_isa(nested_op) ||
        loom_check_expect_shape_isa(nested_op) ||
        loom_check_expect_event_isa(nested_op) ||
        loom_check_return_isa(nested_op)) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, nested_op)),
        loom_param_string(IREE_SV("an explicit check.expect.* operation")),
        loom_param_string(IREE_SV("a check.compare body")),
    };
    return loom_check_emit(emitter, nested_op, LOOM_ERR_STRUCTURE_031, params,
                           IREE_ARRAYSIZE(params));
  }
  for (uint16_t i = 0; i < actual_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_check_verify_result_pair_observation(
        module, op, i, block->arg_ids[i], block->arg_ids[actual_count + i],
        emitter));
  }
  *out_actual_results = (loom_value_slice_t){
      .values = block->arg_ids,
      .count = actual_count,
  };
  *out_expected_results = (loom_value_slice_t){
      .values = actual_count ? block->arg_ids + actual_count : NULL,
      .count = actual_count,
  };
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_check_verify_compare_subject(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    loom_value_slice_t parameters, loom_value_slice_t arguments,
    loom_value_slice_t results, iree_string_view_t result_field,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol = loom_check_lookup_subject(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }
  bool operands_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_verify_subject_operands(
      module, op, symbol, parameters, arguments, &operands_valid, emitter));
  if (!operands_valid) {
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE)) {
    return loom_function_call_contract_verify(
        module, op, callee, arguments, results,
        LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
        emitter);
  }
  if (results.count == 0) {
    return iree_ok_status();
  }
  return loom_check_emit_count_mismatch(
      emitter, op, result_field, results.count, IREE_SV("product results"), 0);
}

iree_status_t loom_check_compare_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t actual_results = {0};
  loom_value_slice_t expected_results = {0};
  bool comparison_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_verify_compare_body(
      module, op, &actual_results, &expected_results, &comparison_valid,
      emitter));
  if (!comparison_valid) {
    return iree_ok_status();
  }

  const loom_symbol_ref_t callee = loom_check_compare_callee(op);
  const loom_value_slice_t parameters = loom_check_compare_call_parameters(op);
  const loom_value_slice_t arguments = loom_check_compare_arguments(op);
  IREE_RETURN_IF_ERROR(loom_check_verify_compare_subject(
      module, op, callee, parameters, arguments, actual_results,
      IREE_SV("actual results"), emitter));

  loom_symbol_ref_t oracle_callee = loom_check_compare_oracle_callee(op);
  if (!loom_symbol_ref_is_valid(oracle_callee) ||
      (oracle_callee.module_id == callee.module_id &&
       oracle_callee.symbol_id == callee.symbol_id)) {
    return iree_ok_status();
  }
  return loom_check_verify_compare_subject(
      module, op, oracle_callee, parameters, arguments, expected_results,
      IREE_SV("expected results"), emitter);
}

iree_status_t loom_check_invoke_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_check_invoke_callee(op);
  const loom_symbol_t* symbol = loom_check_lookup_subject(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }
  const loom_value_slice_t parameters = loom_check_invoke_call_parameters(op);
  const loom_value_slice_t arguments = loom_check_invoke_arguments(op);
  bool operands_valid = false;
  IREE_RETURN_IF_ERROR(loom_check_verify_subject_operands(
      module, op, symbol, parameters, arguments, &operands_valid, emitter));
  if (!operands_valid) {
    return iree_ok_status();
  }
  const loom_value_slice_t results = loom_check_invoke_results(op);
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE)) {
    return loom_function_call_contract_verify(
        module, op, callee, arguments, results,
        LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
        emitter);
  }
  if (results.count == 0) {
    return iree_ok_status();
  }
  return loom_check_emit_count_mismatch(
      emitter, op, IREE_SV("declared results"), results.count,
      IREE_SV("product results"), 0);
}

iree_status_t loom_check_entropy_fork_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_type_t entropy_type =
      loom_module_value_type(module, loom_check_entropy_fork_entropy(op));
  if (loom_check_entropy_type_isa(module, entropy_type)) {
    return iree_ok_status();
  }
  return loom_check_emit_operand_type(emitter, op, IREE_SV("entropy"),
                                      entropy_type, IREE_SV("check.entropy"));
}

iree_status_t loom_check_entropy_read_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_type_t entropy_type =
      loom_module_value_type(module, loom_check_entropy_read_entropy(op));
  if (!loom_check_entropy_type_isa(module, entropy_type)) {
    return loom_check_emit_operand_type(emitter, op, IREE_SV("entropy"),
                                        entropy_type, IREE_SV("check.entropy"));
  }
  const loom_i64_array_t static_ordinals =
      loom_attr_as_i64_array(loom_check_entropy_read_static_ordinals(op));
  const loom_value_slice_t dynamic_ordinals =
      loom_check_entropy_read_ordinals(op);
  if (static_ordinals.count != 1) {
    return loom_check_emit_count_mismatch(
        emitter, op, IREE_SV("static ordinals"), static_ordinals.count,
        IREE_SV("entropy read ordinals"), 1);
  }
  if (static_ordinals.values[0] < 0 && static_ordinals.values[0] != INT64_MIN) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("static ordinal")),
        loom_param_i64(static_ordinals.values[0]),
        loom_param_string(IREE_SV("a non-negative ordinal")),
    };
    return loom_check_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                           IREE_ARRAYSIZE(params));
  }
  const uint16_t expected_dynamic_count =
      static_ordinals.values[0] == INT64_MIN ? 1 : 0;
  if (dynamic_ordinals.count != expected_dynamic_count) {
    return loom_check_emit_count_mismatch(
        emitter, op, IREE_SV("dynamic ordinals"), dynamic_ordinals.count,
        IREE_SV("dynamic ordinal sentinels"), expected_dynamic_count);
  }
  const loom_type_t result_type =
      loom_module_value_type(module, loom_check_entropy_read_result(op));
  if (loom_check_i64_type_isa(result_type)) {
    return iree_ok_status();
  }
  return loom_check_emit_result_type(emitter, op, IREE_SV("result"),
                                     result_type, IREE_SV("i64"));
}
