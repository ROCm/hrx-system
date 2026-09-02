// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/condition.h"
#include "loom/target/projection.h"

static iree_string_view_t loom_function_contract_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  return module->strings.entries[symbol->name_id];
}

typedef struct loom_function_contract_signature_t {
  // Operation defining the expected signature.
  const loom_op_t* definition_op;
  // Expected argument value IDs in signature order.
  const loom_value_id_t* argument_ids;
  // Number of expected arguments.
  uint16_t argument_count;
  // Expected result value IDs in signature order.
  const loom_value_id_t* result_ids;
  // Number of expected results.
  uint16_t result_count;
} loom_function_contract_signature_t;

typedef struct loom_function_contract_boundary_t {
  // Operation carrying the actual signature or call boundary.
  const loom_op_t* op;
  // Actual argument or operand value IDs in signature order.
  const loom_value_id_t* argument_ids;
  // Number of actual arguments or operands.
  uint16_t argument_count;
  // Actual result value IDs in signature order.
  const loom_value_id_t* result_ids;
  // Number of actual results.
  uint16_t result_count;
  // Diagnostic field kind for actual arguments.
  loom_diagnostic_field_kind_t argument_field_kind;
  // Diagnostic field kind for actual results.
  loom_diagnostic_field_kind_t result_field_kind;
  // Human-readable actual argument field prefix.
  const char* argument_prefix;
  // Human-readable actual result field prefix.
  const char* result_prefix;
} loom_function_contract_boundary_t;

static bool loom_function_contract_signature_contains_value(
    const loom_value_id_t* argument_ids, uint16_t argument_count,
    const loom_value_id_t* result_ids, uint16_t result_count,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (argument_ids[i] == value_id) return true;
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    if (result_ids[i] == value_id) return true;
  }
  return false;
}

static void loom_function_contract_format_predicate_argument(
    char* buffer, iree_host_size_t buffer_capacity, uint16_t predicate_index,
    uint8_t argument_index) {
  iree_snprintf(buffer, buffer_capacity, "predicates[%u].arg[%u]",
                predicate_index, argument_index);
}

static iree_string_view_t loom_function_contract_predicate_expected_type(
    uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return IREE_SV("floating-point scalar");
    default:
      return IREE_SV("integer, index, or offset scalar");
  }
}

static iree_status_t loom_function_contract_emit_predicate_origin_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, uint16_t predicate_index,
    uint8_t argument_index) {
  char field_name[40];
  loom_function_contract_format_predicate_argument(
      field_name, sizeof(field_name), predicate_index, argument_index);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_string(IREE_SV("a function argument or result")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_032,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_function_contract_emit_predicate_type_error(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    uint16_t predicate_index, uint8_t argument_index, loom_type_t actual_type,
    uint8_t predicate_kind) {
  char field_name[40];
  loom_function_contract_format_predicate_argument(
      field_name, sizeof(field_name), predicate_index, argument_index);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_type(actual_type),
      loom_param_string(
          loom_function_contract_predicate_expected_type(predicate_kind)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_003,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_function_contract_verify_predicates(
    const loom_module_t* module, const loom_op_t* op, loom_func_like_t function,
    iree_diagnostic_emitter_t emitter) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_id_t* result_ids = loom_op_const_results(op);
  for (uint16_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    const loom_predicate_t* predicate = &predicates[predicate_index];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
      return loom_function_contract_emit_predicate_origin_error(
          module, op, emitter, predicate_index, 0);
    }
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE) continue;
      const int64_t encoded_value_id = predicate->args[argument_index];
      if (encoded_value_id < 0 || encoded_value_id > UINT32_MAX) {
        return loom_function_contract_emit_predicate_origin_error(
            module, op, emitter, predicate_index, argument_index);
      }
      const loom_value_id_t value_id = (loom_value_id_t)encoded_value_id;
      if (!loom_function_contract_signature_contains_value(
              argument_ids, argument_count, result_ids, op->result_count,
              value_id)) {
        return loom_function_contract_emit_predicate_origin_error(
            module, op, emitter, predicate_index, argument_index);
      }
      const loom_type_t value_type = loom_module_value_type(module, value_id);
      // Registers without a semantic value type do not expose a numeric domain
      // to this target-independent verifier. The target-bound Low verifier
      // decides whether that register form is canonical for its register class.
      // Ordinary source values and typed registers retain the strict domain
      // check here.
      const bool has_verifiable_semantic_type =
          !loom_type_is_register(value_type) ||
          loom_type_register_has_value_type(value_type);
      if (has_verifiable_semantic_type &&
          !loom_predicate_kind_accepts_value_type(predicate->kind,
                                                  value_type)) {
        return loom_function_contract_emit_predicate_type_error(
            op, emitter, predicate_index, argument_index, value_type,
            predicate->kind);
      }
    }
  }
  return iree_ok_status();
}

static const loom_symbol_t* loom_function_contract_lookup_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (!symbol->definition || !symbol->defining_op) {
    return NULL;
  }
  return symbol;
}

static bool loom_function_contract_load_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_function_contract_signature_t* out_signature) {
  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return false;
  }
  out_signature->definition_op = symbol->defining_op;
  out_signature->argument_ids =
      loom_func_like_arg_ids(function, &out_signature->argument_count);
  out_signature->result_ids = loom_op_const_results(function.op);
  out_signature->result_count = function.op->result_count;
  return true;
}

static iree_status_t loom_function_contract_emit_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("contract defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_function_contract_emit_count_mismatch(
    const loom_module_t* module,
    const loom_function_contract_boundary_t* boundary,
    const loom_function_contract_signature_t* signature,
    iree_diagnostic_emitter_t emitter, const loom_error_def_t* error,
    uint16_t actual_count, uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, boundary->op)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_function_contract_emit_related(emitter, boundary->op,
                                             signature->definition_op, error,
                                             params, IREE_ARRAYSIZE(params));
}

static void loom_function_contract_format_field(
    char* buffer, iree_host_size_t buffer_capacity, const char* prefix,
    uint16_t field_index) {
  iree_snprintf(buffer, buffer_capacity, "%s %u", prefix, field_index);
}

static iree_status_t loom_function_contract_emit_type_mismatch(
    const loom_function_contract_boundary_t* boundary,
    const loom_function_contract_signature_t* signature,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t actual_field_kind,
    const char* actual_field_prefix, const char* expected_field_prefix,
    uint16_t field_index, loom_type_t actual_type, loom_type_t expected_type) {
  char actual_field_name[48];
  char expected_field_name[48];
  loom_function_contract_format_field(actual_field_name,
                                      sizeof(actual_field_name),
                                      actual_field_prefix, field_index);
  loom_function_contract_format_field(expected_field_name,
                                      sizeof(expected_field_name),
                                      expected_field_prefix, field_index);
  loom_diagnostic_param_t actual_field =
      loom_param_string(iree_make_cstring_view(actual_field_name));
  if (actual_field_kind != LOOM_DIAGNOSTIC_FIELD_NONE) {
    actual_field = loom_param_with_field_ref(
        actual_field,
        loom_diagnostic_field_ref(actual_field_kind, field_index));
  }
  loom_diagnostic_param_t params[] = {
      actual_field,
      loom_param_type(actual_type),
      loom_param_string(iree_make_cstring_view(expected_field_name)),
      loom_param_type(expected_type),
  };
  return loom_function_contract_emit_related(
      emitter, boundary->op, signature->definition_op, LOOM_ERR_TYPE_001,
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_function_contract_verify_boundary(
    const loom_module_t* module,
    const loom_function_contract_signature_t* signature,
    const loom_function_contract_boundary_t* boundary,
    iree_diagnostic_emitter_t emitter) {
  if (boundary->argument_count != signature->argument_count) {
    IREE_RETURN_IF_ERROR(loom_function_contract_emit_count_mismatch(
        module, boundary, signature, emitter, LOOM_ERR_STRUCTURE_001,
        boundary->argument_count, signature->argument_count));
  }
  if (boundary->result_count != signature->result_count) {
    IREE_RETURN_IF_ERROR(loom_function_contract_emit_count_mismatch(
        module, boundary, signature, emitter, LOOM_ERR_STRUCTURE_002,
        boundary->result_count, signature->result_count));
  }

  const uint16_t argument_count =
      boundary->argument_count < signature->argument_count
          ? boundary->argument_count
          : signature->argument_count;
  const uint16_t result_count = boundary->result_count < signature->result_count
                                    ? boundary->result_count
                                    : signature->result_count;
  const loom_type_value_remap_t result_remap = {
      .source_values = signature->result_ids,
      .target_values = boundary->result_ids,
      .count = result_count,
  };
  const loom_type_value_remap_t signature_remap = {
      .source_values = signature->argument_ids,
      .target_values = boundary->argument_ids,
      .count = argument_count,
      .next = &result_remap,
  };

  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, boundary->argument_ids[i]);
    const loom_type_t expected_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    if (loom_type_equal_after_value_remap(module, expected_type, actual_type,
                                          &signature_remap)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_function_contract_emit_type_mismatch(
        boundary, signature, emitter, boundary->argument_field_kind,
        boundary->argument_prefix, "contract argument", i, actual_type,
        expected_type));
  }

  for (uint16_t i = 0; i < result_count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, boundary->result_ids[i]);
    const loom_type_t expected_type =
        loom_module_value_type(module, signature->result_ids[i]);
    if (loom_type_equal_after_value_remap(module, expected_type, actual_type,
                                          &signature_remap)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_function_contract_emit_type_mismatch(
        boundary, signature, emitter, boundary->result_field_kind,
        boundary->result_prefix, "contract result", i, actual_type,
        expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_function_contract_verify_symbol_boundary(
    const loom_module_t* module, loom_symbol_ref_t contract_ref,
    const loom_function_contract_boundary_t* boundary,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol =
      loom_function_contract_lookup_symbol(module, contract_ref);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_function_contract_signature_t signature = {0};
  if (!loom_function_contract_load_signature(module, symbol, &signature)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_function_contract_symbol_name(module, symbol)),
        loom_param_string(
            loom_symbol_definition_descriptor_name(symbol->definition)),
        loom_param_string(IREE_SV("function-like contract")),
    };
    return loom_function_contract_emit_related(
        emitter, boundary->op, symbol->defining_op, LOOM_ERR_SYMBOL_003, params,
        IREE_ARRAYSIZE(params));
  }
  return loom_function_contract_verify_boundary(module, &signature, boundary,
                                                emitter);
}

iree_status_t loom_function_call_contract_verify(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    loom_value_slice_t operands, loom_value_slice_t results,
    iree_diagnostic_emitter_t emitter) {
  const loom_function_contract_boundary_t boundary = {
      .op = op,
      .argument_ids = operands.values,
      .argument_count = operands.count,
      .result_ids = results.values,
      .result_count = results.count,
      .argument_field_kind = LOOM_DIAGNOSTIC_FIELD_OPERAND,
      .result_field_kind = LOOM_DIAGNOSTIC_FIELD_RESULT,
      .argument_prefix = "operand",
      .result_prefix = "result",
  };
  return loom_function_contract_verify_symbol_boundary(module, callee,
                                                       &boundary, emitter);
}

static iree_status_t loom_function_provider_family_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_func_like_t provider = loom_func_like_const_cast(module, op);
  const loom_symbol_ref_t family = loom_func_like_template_family(provider);
  if (!loom_symbol_ref_is_valid(family)) {
    return iree_ok_status();
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(provider, &argument_count);
  const loom_function_contract_boundary_t boundary = {
      .op = op,
      .argument_ids = argument_ids,
      .argument_count = argument_count,
      .result_ids = loom_op_const_results(op),
      .result_count = op->result_count,
      .argument_field_kind = LOOM_DIAGNOSTIC_FIELD_NONE,
      .result_field_kind = LOOM_DIAGNOSTIC_FIELD_RESULT,
      .argument_prefix = "implementation argument",
      .result_prefix = "implementation result",
  };
  return loom_function_contract_verify_symbol_boundary(module, family,
                                                       &boundary, emitter);
}

static iree_status_t loom_template_provider_emit_non_identity_target_field(
    const loom_module_t* module, const loom_op_t* provider_op,
    const loom_symbol_t* provider_symbol, const loom_op_t* target_op,
    const loom_symbol_t* target_symbol, uint8_t target_attr_index,
    iree_diagnostic_emitter_t emitter) {
  const loom_op_vtable_t* target_vtable = loom_op_vtable(module, target_op);
  const iree_string_view_t field_name = loom_attr_descriptor_name(
      &target_vtable->attr_descriptors[target_attr_index]);
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_function_contract_symbol_name(module, provider_symbol)),
      loom_param_string(
          loom_function_contract_symbol_name(module, target_symbol)),
      loom_param_string(field_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("target field authored here"),
      .op = target_op,
      .field_ref = loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                             target_attr_index),
  }};
  loom_diagnostic_emission_t emission = {
      .op = provider_op,
      .error = LOOM_ERR_TARGET_067,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_function_verify_target_conditions(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_func_like_t function, iree_diagnostic_emitter_t emitter) {
  const loom_parameterized_attr_array_t requirements =
      loom_func_like_requires(function);
  for (iree_host_size_t i = 0; i < requirements.count; ++i) {
    const loom_target_condition_descriptor_t* descriptor =
        loom_target_condition_resolve(module->context, requirements.values[i]);
    const iree_string_view_t invalid_reason =
        descriptor != NULL
            ? loom_target_condition_validate(descriptor, requirements.values[i])
            : IREE_SV(
                  "parameterized attribute family is not a target "
                  "condition");
    if (!iree_string_view_is_empty(invalid_reason)) {
      const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
      const loom_symbol_t* function_symbol =
          &module->symbols.entries[function_ref.symbol_id];
      loom_diagnostic_param_t params[] = {
          loom_param_string(
              loom_function_contract_symbol_name(module, function_symbol)),
          loom_param_with_field_ref(
              loom_param_u32((uint32_t)i),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                        function.vtable->requires_attr_index)),
          loom_param_string(invalid_reason),
      };
      loom_diagnostic_emission_t emission = {
          .op = function_op,
          .error = LOOM_ERR_TARGET_068,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      return iree_diagnostic_emit(emitter, &emission);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  // Targetless functions are valid generic program representations. A compile
  // invocation may bind an exact target later, so source verification cannot
  // require an authored target attribute.
  loom_func_like_t function = loom_func_like_const_cast(module, op);
  IREE_ASSERT(loom_func_like_isa(function));
  IREE_RETURN_IF_ERROR(
      loom_function_contract_verify_predicates(module, op, function, emitter));
  return loom_function_verify_target_conditions(module, op, function, emitter);
}

iree_status_t loom_function_provider_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));

  loom_func_like_t provider = loom_func_like_const_cast(module, op);
  const loom_symbol_ref_t target_ref = loom_func_like_target(provider);
  if (loom_symbol_ref_is_valid(target_ref)) {
    const loom_symbol_t* target_symbol =
        &module->symbols.entries[target_ref.symbol_id];
    const loom_symbol_ref_t provider_ref = loom_func_like_callee(provider);
    const loom_symbol_t* provider_symbol =
        &module->symbols.entries[provider_ref.symbol_id];
    loom_target_like_t target =
        loom_target_like_cast(module, target_symbol->defining_op);
    if (loom_target_like_isa(target)) {
      const loom_target_like_descriptor_t* descriptor =
          loom_target_like_descriptor(target);
      for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
        const uint8_t attr_index = descriptor->projections[i].attr_index;
        if (loom_attr_is_absent(loom_op_const_attrs(target.op)[attr_index])) {
          continue;
        }
        return loom_template_provider_emit_non_identity_target_field(
            module, op, provider_symbol, target.op, target_symbol, attr_index,
            emitter);
      }
    }
  }
  return loom_function_provider_family_contract_verify(module, op, emitter);
}
