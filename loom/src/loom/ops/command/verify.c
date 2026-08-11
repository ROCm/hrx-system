// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/op_defs.h"

static iree_status_t loom_command_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_command_emit_launch_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* launch_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  const loom_diagnostic_emission_t emission = {
      .op = launch_op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_command_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* launch_op,
    const loom_op_t* definition_op, iree_string_view_t actual_field,
    uint16_t actual_count, iree_string_view_t expected_field,
    uint16_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(actual_field),
      loom_param_u32(actual_count),
      loom_param_string(expected_field),
      loom_param_u32(expected_count),
  };
  return loom_command_emit_launch_related(emitter, launch_op, definition_op,
                                          LOOM_ERR_STRUCTURE_013, params,
                                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_command_emit_pattern_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t pattern, iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("pattern")),
      loom_param_string(pattern),
      loom_param_string(expected_constraint),
  };
  return loom_command_emit(emitter, op, LOOM_ERR_STRUCTURE_027, params,
                           IREE_ARRAYSIZE(params));
}

iree_status_t loom_command_parameter_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  const iree_string_view_t pattern =
      module->strings.entries[loom_command_parameter_pattern(op)];
  if (pattern.size == 0) {
    return loom_command_emit_pattern_constraint(
        emitter, op, pattern, IREE_SV("a non-empty parameter key pattern"));
  }

  iree_host_size_t placeholder_count = 0;
  for (iree_host_size_t i = 0; i < pattern.size; ++i) {
    if (pattern.data[i] == '{') {
      if (i + 1 >= pattern.size || pattern.data[i + 1] != '}') {
        return loom_command_emit_pattern_constraint(
            emitter, op, pattern,
            IREE_SV("only complete '{}' index placeholders"));
      }
      ++placeholder_count;
      ++i;
    } else if (pattern.data[i] == '}') {
      return loom_command_emit_pattern_constraint(
          emitter, op, pattern,
          IREE_SV("only complete '{}' index placeholders"));
    }
  }

  const uint16_t substitution_count =
      loom_command_parameter_substitutions(op).count;
  if (placeholder_count != substitution_count) {
    char expected_constraint[96];
    iree_snprintf(expected_constraint, sizeof(expected_constraint),
                  "exactly %" PRIu16 " '{}' placeholders", substitution_count);
    return loom_command_emit_pattern_constraint(
        emitter, op, pattern, iree_make_cstring_view(expected_constraint));
  }
  return iree_ok_status();
}

static iree_status_t loom_command_verify_program_contract(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));

  loom_func_like_t program = loom_func_like_cast(module, (loom_op_t*)op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(program, &argument_count);
  const int64_t specialization_count =
      loom_func_like_specialization_count(program);
  if (specialization_count < 0 || specialization_count > argument_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("specialization_count")),
        loom_param_i64(specialization_count),
        loom_param_string(IREE_SV("between zero and the argument count")),
    };
    return loom_command_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
  }

  for (uint16_t i = (uint16_t)specialization_count; i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(module, argument_ids[i]);
    if (loom_type_satisfies_constraint(type, LOOM_TYPE_CONSTRAINT_BUFFER)) {
      continue;
    }
    char field_name[32];
    iree_snprintf(field_name, sizeof(field_name), "launch binding %u",
                  i - (uint16_t)specialization_count);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(iree_make_cstring_view(field_name)),
        loom_param_type(type),
        loom_param_string(IREE_SV("buffer")),
    };
    return loom_command_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                             IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_command_program_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_command_verify_program_contract(module, op, emitter);
}

iree_status_t loom_command_program_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_command_verify_program_contract(module, op, emitter);
}

iree_status_t loom_command_program_launch_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_command_program_launch_callee(op);
  const loom_op_t* definition_op =
      module->symbols.entries[callee.symbol_id].defining_op;
  loom_func_like_t program =
      loom_func_like_cast(module, (loom_op_t*)definition_op);

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(program);
  if (specialization_count_i64 < 0 ||
      specialization_count_i64 > argument_count) {
    // The defining operation owns this malformed external-input diagnostic.
    // Keep this consumer safe if it is visited first during module traversal.
    return iree_ok_status();
  }
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;
  const uint16_t binding_count = argument_count - specialization_count;

  const loom_value_slice_t specializations =
      loom_command_program_launch_specializations(op);
  if (specializations.count != specialization_count) {
    return loom_command_emit_count_mismatch(
        emitter, op, definition_op, IREE_SV("specialization"),
        specializations.count, IREE_SV("program specialization argument"),
        specialization_count);
  }
  const loom_value_slice_t bindings = loom_command_program_launch_bindings(op);
  if (bindings.count != binding_count) {
    return loom_command_emit_count_mismatch(
        emitter, op, definition_op, IREE_SV("binding"), bindings.count,
        IREE_SV("program launch binding"), binding_count);
  }

  const loom_type_value_remap_t value_remap = {
      .source_values = argument_ids,
      .target_values = loom_op_const_operands(op),
      .count = argument_count,
  };
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, loom_op_const_operands(op)[i]);
    const loom_type_t expected_type =
        loom_module_value_type(module, argument_ids[i]);
    if (loom_type_equal_after_value_remap(module, expected_type, actual_type,
                                          &value_remap)) {
      continue;
    }

    char actual_name[32];
    char expected_name[48];
    const bool is_specialization = i < specialization_count;
    const uint16_t group_index =
        is_specialization ? i : i - specialization_count;
    iree_snprintf(actual_name, sizeof(actual_name), "%s %u",
                  is_specialization ? "specialization" : "binding",
                  group_index);
    iree_snprintf(
        expected_name, sizeof(expected_name), "program %s %u",
        is_specialization ? "specialization argument" : "launch binding",
        group_index);
    const loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(
            loom_param_string(iree_make_cstring_view(actual_name)),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i)),
        loom_param_type(actual_type),
        loom_param_string(iree_make_cstring_view(expected_name)),
        loom_param_type(expected_type),
    };
    return loom_command_emit_launch_related(emitter, op, definition_op,
                                            LOOM_ERR_TYPE_001, params,
                                            IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}
