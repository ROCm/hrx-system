// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/low_invoke.h"

#include <inttypes.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbolic_value.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/decision/predicate.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/target/facts.h"
#include "loom/target/function_contract.h"
#include "loom/target/registers.h"

static iree_string_view_t loom_low_invoke_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<invalid>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_low_invoke_emit_contract_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_072, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_emit_count_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t value_kind,
    uint32_t actual_count, uint32_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(value_kind),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_073, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_emit_type_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t value_kind,
    loom_type_t actual_type, loom_type_t expected_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(value_kind),
      loom_param_type(actual_type),
      loom_param_type(expected_type),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_074, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_emit_representation_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t callee_contract,
    iree_string_view_t caller_target_contract) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(callee_contract),
      loom_param_string(caller_target_contract),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_075, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_emit_target_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t callee_target,
    iree_string_view_t caller_target) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(callee_target),
      loom_param_string(caller_target),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_076, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_emit_precondition_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, uint32_t predicate_index,
    uint8_t predicate_kind, loom_decision_truth_t proof_result) {
  const char* predicate_kind_name = loom_predicate_kind_name(predicate_kind);
  const iree_string_view_t proof_result_name =
      proof_result == LOOM_DECISION_TRUTH_FALSE ? IREE_SV("proven false")
                                                : IREE_SV("not proven");
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_u32(predicate_index),
      loom_param_string(predicate_kind_name != NULL
                            ? iree_make_cstring_view(predicate_kind_name)
                            : IREE_SV("<invalid>")),
      loom_param_string(proof_result_name),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_078, params, IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_low_invoke_function_descriptor_set(
    const loom_module_t* module, const loom_op_t* function_op) {
  const loom_string_id_t descriptor_set =
      loom_low_func_def_descriptor_set(function_op);
  if (descriptor_set >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[descriptor_set];
}

typedef struct loom_low_invoke_projection_t {
  // Representation contract used to interpret the authored helper body.
  const loom_low_descriptor_set_t* source_descriptor_set;
  // Representation contract selected for the concrete caller.
  const loom_low_descriptor_set_t* target_descriptor_set;
} loom_low_invoke_projection_t;

static iree_status_t loom_low_invoke_project_register_type(
    loom_module_t* module, const loom_low_invoke_projection_t* projection,
    loom_type_t source_type, loom_type_t* out_target_type) {
  *out_target_type = source_type;
  if (!loom_type_is_register(source_type) ||
      projection->source_descriptor_set == projection->target_descriptor_set) {
    return iree_ok_status();
  }

  const uint64_t source_descriptor_set_id =
      loom_low_register_type_descriptor_set_stable_id(source_type);
  if (source_descriptor_set_id !=
      projection->source_descriptor_set->stable_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low helper register type belongs to descriptor set 0x%016" PRIx64
        ", expected 0x%016" PRIx64,
        source_descriptor_set_id, projection->source_descriptor_set->stable_id);
  }

  const uint16_t source_class_id = loom_low_register_type_class_id(source_type);
  if (source_class_id >= projection->source_descriptor_set->reg_class_count) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low helper register class %u is outside descriptor set '%.*s'",
        (unsigned)source_class_id, (int)source_set_name.size,
        source_set_name.data);
  }

  const loom_low_reg_class_t* source_class =
      &projection->source_descriptor_set->reg_classes[source_class_id];
  const iree_string_view_t source_class_name = loom_low_descriptor_set_string(
      projection->source_descriptor_set, source_class->name_string_offset);
  uint16_t target_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (!loom_low_descriptor_set_lookup_register_class(
          projection->target_descriptor_set, source_class_name,
          &target_class_id, /*out_descriptor_register_class=*/NULL)) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_offset);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low representation '%.*s' claims compatibility with '%.*s' but "
        "register class '%.*s' is absent",
        (int)source_set_name.size, source_set_name.data,
        (int)target_set_name.size, target_set_name.data,
        (int)source_class_name.size, source_class_name.data);
  }

  const uint32_t unit_count = loom_low_register_type_unit_count(source_type);
  const loom_type_t* value_type = loom_type_register_value_type(source_type);
  if (value_type != NULL) {
    return loom_low_build_typed_register_type(
        module, projection->target_descriptor_set, target_class_id, unit_count,
        *value_type, out_target_type);
  }
  return loom_low_build_register_type(projection->target_descriptor_set,
                                      target_class_id, unit_count,
                                      out_target_type);
}

static iree_status_t loom_low_invoke_project_value_type(
    loom_module_t* module, const loom_low_invoke_projection_t* projection,
    loom_value_id_t value_id) {
  const loom_type_t source_type = loom_module_value_type(module, value_id);
  loom_type_t target_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_invoke_project_register_type(
      module, projection, source_type, &target_type));
  if (loom_type_equal(source_type, target_type)) {
    return iree_ok_status();
  }
  return loom_module_set_value_type(module, value_id, target_type);
}

static iree_status_t loom_low_invoke_project_descriptor(
    loom_op_t* op, uint8_t descriptor_attr_index,
    const loom_low_invoke_projection_t* projection) {
  if (projection->source_descriptor_set == projection->target_descriptor_set) {
    return iree_ok_status();
  }

  const uint32_t source_ordinal =
      loom_attr_as_scoped_enum(loom_op_const_attrs(op)[descriptor_attr_index]);
  const loom_low_descriptor_t* source_descriptor =
      loom_low_descriptor_set_descriptor_at(projection->source_descriptor_set,
                                            source_ordinal);
  if (source_descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low helper descriptor ordinal %u is outside its representation "
        "contract",
        source_ordinal);
  }
  const iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
      projection->source_descriptor_set, source_descriptor->key_string_offset);
  const uint32_t target_ordinal = loom_low_descriptor_set_lookup_descriptor(
      projection->target_descriptor_set, descriptor_key);
  if (target_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_offset);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low representation '%.*s' claims compatibility with '%.*s' but "
        "descriptor '%.*s' is absent",
        (int)source_set_name.size, source_set_name.data,
        (int)target_set_name.size, target_set_name.data,
        (int)descriptor_key.size, descriptor_key.data);
  }
  loom_op_attrs(op)[descriptor_attr_index] =
      loom_attr_scoped_enum(target_ordinal);
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_project_cloned_op(
    loom_module_t* module, loom_op_t* cloned_op,
    const loom_low_invoke_projection_t* projection) {
  if (loom_low_op_isa(cloned_op)) {
    IREE_RETURN_IF_ERROR(loom_low_invoke_project_descriptor(
        cloned_op, loom_low_op_descriptor_ATTR_INDEX, projection));
  } else if (loom_low_const_isa(cloned_op)) {
    IREE_RETURN_IF_ERROR(loom_low_invoke_project_descriptor(
        cloned_op, loom_low_const_descriptor_ATTR_INDEX, projection));
  }

  const loom_value_id_t* results = loom_op_const_results(cloned_op);
  for (uint16_t i = 0; i < cloned_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_project_value_type(module, projection, results[i]));
  }

  loom_region_t** regions = loom_op_regions(cloned_op);
  for (uint8_t region_index = 0; region_index < cloned_op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (region == NULL) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_invoke_project_value_type(
            module, projection, loom_block_arg_id(block, i)));
      }
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_low_invoke_project_cloned_op(module, child_op, projection));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_resolve_callee_target(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_symbol_ref_t callee_ref, const loom_op_t* callee_op,
    iree_string_view_t callee_name,
    const loom_target_facts_t** out_callee_target_facts,
    const loom_low_descriptor_set_t** out_callee_descriptor_set) {
  *out_callee_target_facts = NULL;
  *out_callee_descriptor_set = NULL;

  const iree_string_view_t callee_contract =
      loom_low_invoke_function_descriptor_set(context->module, callee_op);
  *out_callee_descriptor_set = loom_low_descriptor_registry_lookup(
      context->options->descriptor_registry, callee_contract);
  if (*out_callee_descriptor_set == NULL) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the helper representation contract is not linked"));
  }

  if (!loom_symbol_ref_is_valid(loom_low_func_def_target(callee_op))) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(
      &symbol_facts, loom_low_lower_context_emission_arena(context));
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, context->module, callee_ref, &base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (function_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified low helper '@%.*s' has no function symbol facts",
        (int)callee_name.size, callee_name.data);
  }

  bool target_valid = true;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_facts(
      context->module, &symbol_facts, function_facts, context->options->emitter,
      loom_low_lower_context_emission_arena(context), &target_valid,
      out_callee_target_facts));
  if (!target_valid) {
    *out_callee_descriptor_set = NULL;
    if (!loom_low_lower_context_should_stop(context)) {
      ++context->result->error_count;
    }
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_map_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  const uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (loom_type_kind(*out_low_type) != LOOM_TYPE_NONE) {
    return iree_ok_status();
  }
  if (context->result->error_count != previous_error_count) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result")),
      loom_param_u64(source_result),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_027, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_invoke_build_schedule_fence(
    loom_low_lower_context_t* context, loom_location_id_t location) {
  loom_op_t* fence_op = NULL;
  return loom_low_schedule_fence_build(loom_low_lower_context_builder(context),
                                       location, &fence_op);
}

static bool loom_low_invoke_find_argument(
    const loom_value_id_t* callee_arguments, uint16_t callee_argument_count,
    loom_value_id_t value_id, uint16_t* out_argument_index) {
  if (out_argument_index) *out_argument_index = 0;
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    if (callee_arguments[i] != value_id) continue;
    if (out_argument_index) *out_argument_index = i;
    return true;
  }
  return false;
}

static bool loom_low_invoke_predicate_integer_relation(
    uint8_t predicate_kind,
    loom_symbolic_integer_relation_t* out_integer_relation) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_PREDICATE_NE:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_PREDICATE_LT:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MIN:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_PREDICATE_MAX:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_low_invoke_decision_operand_as_integer(
    const loom_decision_predicate_operand_t* operand,
    loom_condition_integer_operand_t* out_integer_operand) {
  *out_integer_operand = (loom_condition_integer_operand_t){0};
  if (operand->identity != LOOM_DECISION_OPERAND_IDENTITY_NONE) {
    out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
    out_integer_operand->value_id = operand->identity;
    return true;
  }
  int64_t constant = 0;
  if (!loom_value_facts_as_exact_i64(operand->facts, &constant)) return false;
  out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
  out_integer_operand->constant = constant;
  return true;
}

static bool loom_low_invoke_caller_predicate_operand_as_integer(
    const loom_predicate_t* predicate, uint8_t argument_index,
    const loom_value_id_t* caller_arguments, uint16_t caller_argument_count,
    loom_condition_integer_operand_t* out_integer_operand) {
  *out_integer_operand = (loom_condition_integer_operand_t){0};
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
      out_integer_operand->constant = predicate->args[argument_index];
      return true;
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) return false;
      if (!loom_low_invoke_find_argument(caller_arguments,
                                         caller_argument_count,
                                         (loom_value_id_t)raw_value_id,
                                         /*out_argument_index=*/NULL)) {
        return false;
      }
      out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
      out_integer_operand->value_id = (loom_value_id_t)raw_value_id;
      return true;
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static iree_status_t loom_low_invoke_integer_operands_match(
    loom_low_lower_context_t* context,
    loom_condition_integer_operand_t known_operand,
    loom_condition_integer_operand_t queried_operand, bool* out_match) {
  *out_match = false;
  if (known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT &&
      queried_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    *out_match = known_operand.constant == queried_operand.constant;
    return iree_ok_status();
  }
  loom_symbolic_expr_context_t* expression_context =
      loom_low_lower_context_symbolic_expr_context(context);
  if (known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      queried_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return loom_symbolic_values_match(expression_context,
                                      known_operand.value_id,
                                      queried_operand.value_id, out_match);
  }

  const loom_condition_integer_operand_t value_operand =
      known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE
          ? known_operand
          : queried_operand;
  const loom_condition_integer_operand_t constant_operand =
      known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT
          ? known_operand
          : queried_operand;
  if (value_operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      constant_operand.kind != LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    return iree_ok_status();
  }
  loom_value_facts_t value_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_symbolic_value_lookup_condition_refined_facts(
      expression_context, value_operand.value_id, &value_facts));
  int64_t exact_value = 0;
  *out_match = loom_value_facts_as_exact_i64(value_facts, &exact_value) &&
               exact_value == constant_operand.constant;
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_caller_contract_proves_relation(
    loom_low_lower_context_t* context,
    const loom_condition_integer_relation_t* queried_relation,
    loom_decision_truth_t* out_proof_result) {
  *out_proof_result = LOOM_DECISION_TRUTH_UNKNOWN;
  uint16_t caller_argument_count = 0;
  const loom_value_id_t* caller_arguments =
      loom_func_like_arg_ids(context->source_function, &caller_argument_count);
  uint16_t caller_predicate_count = 0;
  const loom_predicate_t* caller_predicates = loom_func_like_predicates(
      context->source_function, &caller_predicate_count);
  for (uint16_t i = 0; i < caller_predicate_count; ++i) {
    // Only caller argument predicates are available at the invocation. A
    // predicate involving a function result is a postcondition, not evidence
    // for work still inside the function body.
    const loom_predicate_t* caller_predicate = &caller_predicates[i];
    loom_condition_integer_relation_t known_relation = {0};
    if (caller_predicate->arg_count != 2 ||
        !loom_low_invoke_predicate_integer_relation(caller_predicate->kind,
                                                    &known_relation.relation) ||
        !loom_low_invoke_caller_predicate_operand_as_integer(
            caller_predicate, 0, caller_arguments, caller_argument_count,
            &known_relation.left) ||
        !loom_low_invoke_caller_predicate_operand_as_integer(
            caller_predicate, 1, caller_arguments, caller_argument_count,
            &known_relation.right)) {
      continue;
    }

    bool left_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_invoke_integer_operands_match(
        context, known_relation.left, queried_relation->left, &left_matches));
    bool right_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_invoke_integer_operands_match(
        context, known_relation.right, queried_relation->right,
        &right_matches));
    bool implication_result = false;
    if (left_matches && right_matches &&
        loom_symbolic_integer_relation_implies(known_relation.relation,
                                               queried_relation->relation,
                                               &implication_result)) {
      *out_proof_result = implication_result ? LOOM_DECISION_TRUTH_TRUE
                                             : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }

    bool swapped_left_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_invoke_integer_operands_match(
        context, known_relation.left, queried_relation->right,
        &swapped_left_matches));
    bool swapped_right_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_invoke_integer_operands_match(
        context, known_relation.right, queried_relation->left,
        &swapped_right_matches));
    if (swapped_left_matches && swapped_right_matches &&
        loom_symbolic_integer_relation_implies(
            loom_symbolic_integer_relation_swap(known_relation.relation),
            queried_relation->relation, &implication_result)) {
      *out_proof_result = implication_result ? LOOM_DECISION_TRUTH_TRUE
                                             : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_prove_predicate(
    loom_low_lower_context_t* context, uint8_t predicate_kind,
    const loom_decision_predicate_operand_t predicate_operands[3],
    loom_decision_truth_t* out_proof_result) {
  // Scalar facts handle exact values, ranges, divisibility, and floating-point
  // classes. Relational predicates then use symbolic/path proofs and finally
  // implication from the caller's entry contract. None of these paths mutate
  // facts: low.assume is built only after this proof returns true.
  *out_proof_result =
      loom_decision_predicate_evaluate(predicate_kind, predicate_operands);
  if (*out_proof_result != LOOM_DECISION_TRUTH_UNKNOWN) {
    return iree_ok_status();
  }

  loom_condition_integer_relation_t queried_relation = {0};
  if (!loom_low_invoke_predicate_integer_relation(predicate_kind,
                                                  &queried_relation.relation) ||
      !loom_low_invoke_decision_operand_as_integer(&predicate_operands[0],
                                                   &queried_relation.left) ||
      !loom_low_invoke_decision_operand_as_integer(&predicate_operands[1],
                                                   &queried_relation.right)) {
    return iree_ok_status();
  }

  if (queried_relation.left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      queried_relation.right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_symbolic_proof_result_t symbolic_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
        loom_low_lower_context_symbolic_expr_context(context),
        queried_relation.relation, queried_relation.left.value_id,
        queried_relation.right.value_id, &symbolic_result));
    if (symbolic_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
      *out_proof_result = symbolic_result == LOOM_SYMBOLIC_PROOF_TRUE
                              ? LOOM_DECISION_TRUTH_TRUE
                              : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }
  }

  return loom_low_invoke_caller_contract_proves_relation(
      context, &queried_relation, out_proof_result);
}

static iree_status_t loom_low_invoke_resolve_precondition_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, const loom_predicate_t* predicate,
    uint8_t predicate_argument_index, const loom_value_id_t* callee_arguments,
    uint16_t callee_argument_count, loom_value_slice_t invoke_operands,
    loom_decision_predicate_operand_t* out_operand) {
  *out_operand = (loom_decision_predicate_operand_t){
      .facts = loom_value_facts_unknown(),
      .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
  };
  switch (
      (loom_predicate_arg_tag_t)predicate->arg_tags[predicate_argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_operand->facts =
          loom_value_facts_exact_i64(predicate->args[predicate_argument_index]);
      return iree_ok_status();
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[predicate_argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) {
        return loom_low_invoke_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("a helper precondition has an invalid SSA value"));
      }
      uint16_t callee_argument_index = 0;
      if (!loom_low_invoke_find_argument(
              callee_arguments, callee_argument_count,
              (loom_value_id_t)raw_value_id, &callee_argument_index)) {
        return loom_low_invoke_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("helper preconditions may reference only helper "
                    "arguments"));
      }
      if (callee_argument_index >= invoke_operands.count) {
        return loom_low_invoke_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("a helper precondition argument has no invocation "
                    "operand"));
      }
      const loom_value_id_t source_value =
          invoke_operands.values[callee_argument_index];
      out_operand->identity = source_value;
      return loom_symbolic_value_lookup_condition_refined_facts(
          loom_low_lower_context_symbolic_expr_context(context), source_value,
          &out_operand->facts);
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return loom_low_invoke_emit_contract_error(
          context, source_op, callee_name,
          IREE_SV("a helper precondition has an invalid argument kind"));
  }
}

static iree_status_t loom_low_invoke_prove_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, const loom_predicate_t* predicates,
    uint16_t predicate_count, const loom_value_id_t* callee_arguments,
    uint16_t callee_argument_count, loom_value_slice_t invoke_operands) {
  for (uint16_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    const loom_predicate_t* predicate = &predicates[predicate_index];
    const uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (expected_argument_count == UINT8_MAX ||
        predicate->arg_count != expected_argument_count ||
        predicate->arg_count > IREE_ARRAYSIZE(predicate->args)) {
      return loom_low_invoke_emit_contract_error(
          context, source_op, callee_name,
          IREE_SV("a helper precondition has an invalid predicate shape"));
    }

    loom_decision_predicate_operand_t predicate_operands[3] = {0};
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      IREE_RETURN_IF_ERROR(loom_low_invoke_resolve_precondition_operand(
          context, source_op, callee_name, predicate, argument_index,
          callee_arguments, callee_argument_count, invoke_operands,
          &predicate_operands[argument_index]));
    }
    loom_decision_truth_t proof_result = LOOM_DECISION_TRUTH_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_low_invoke_prove_predicate(
        context, predicate->kind, predicate_operands, &proof_result));
    if (proof_result != LOOM_DECISION_TRUTH_TRUE) {
      return loom_low_invoke_emit_precondition_error(
          context, source_op, callee_name, predicate_index, predicate->kind,
          proof_result);
    }
  }
  return iree_ok_status();
}

// Proves and materializes the helper's argument contract at the inline site.
// The function boundary disappears during required inlining, so each proven
// predicate becomes an ordinary Low fact identity before the body is cloned.
static iree_status_t loom_low_invoke_materialize_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, loom_func_like_t callee,
    const loom_value_id_t* callee_arguments, uint16_t callee_argument_count,
    loom_value_slice_t invoke_operands, loom_ir_remap_t* remap) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(callee, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_low_invoke_prove_argument_contract(
      context, source_op, callee_name, predicates, predicate_count,
      callee_arguments, callee_argument_count, invoke_operands));

  loom_predicate_t* remapped_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      remap, predicates, predicate_count, &remapped_predicates));

  loom_value_id_t* low_arguments = NULL;
  loom_type_t* low_argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, callee_argument_count, sizeof(*low_arguments),
      (void**)&low_arguments));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, callee_argument_count, sizeof(*low_argument_types),
      (void**)&low_argument_types));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, callee_arguments[i],
                                                     &low_arguments[i]));
    low_argument_types[i] = loom_module_value_type(
        loom_low_lower_context_module(context), low_arguments[i]);
  }

  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_assume_build(
      loom_low_lower_context_builder(context), low_arguments,
      callee_argument_count, remapped_predicates, predicate_count,
      low_argument_types, callee_argument_count, source_op->location,
      &assume_op));
  const loom_value_slice_t assumed_arguments =
      loom_low_assume_results(assume_op);
  return loom_ir_remap_map_values(
      remap, callee_arguments, assumed_arguments.values, callee_argument_count);
}

static iree_status_t loom_low_invoke_validate_body_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, const loom_op_t* body_op,
    bool lock_schedule) {
  const loom_op_vtable_t* vtable =
      loom_op_vtable(loom_low_lower_context_module(context), body_op);
  if (vtable != NULL && vtable->call_like != NULL) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("nested calls require their own target and representation "
                "projection"));
  }
  if (loom_low_resource_isa(body_op) || loom_low_live_in_isa(body_op)) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("function-entry resource imports cannot be moved to the "
                "invocation site"));
  }
  if (lock_schedule && body_op->region_count != 0) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("locked scheduling cannot be preserved inside nested "
                "regions"));
  }
  for (uint8_t region_index = 0; region_index < body_op->region_count;
       ++region_index) {
    const loom_region_t* region = loom_op_regions(body_op)[region_index];
    if (region == NULL) continue;
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      for (const loom_op_t* child_op = block->first_op; child_op != NULL;
           child_op = child_op->next_op) {
        IREE_RETURN_IF_ERROR(loom_low_invoke_validate_body_op(
            context, source_op, callee_name, child_op, lock_schedule));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_invoke_clone_body(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_block, bool lock_schedule,
    const loom_low_invoke_projection_t* projection, loom_ir_remap_t* remap) {
  const loom_op_t* source_terminator = loom_block_const_last_op(source_block);
  if (source_block->first_op == source_terminator) {
    return iree_ok_status();
  }
  if (lock_schedule) {
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build_schedule_fence(context, source_op->location));
  }
  for (const loom_op_t* body_op = source_block->first_op;
       body_op != source_terminator; body_op = body_op->next_op) {
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_clone_op(
        loom_low_lower_context_builder(context), body_op, remap, &cloned_op));
    IREE_RETURN_IF_ERROR(loom_low_invoke_project_cloned_op(
        loom_low_lower_context_module(context), cloned_op, projection));
    if (lock_schedule) {
      IREE_RETURN_IF_ERROR(
          loom_low_invoke_build_schedule_fence(context, source_op->location));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_invoke(loom_low_lower_context_t* context,
                                    const loom_op_t* source_op) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_symbol_ref_t callee_ref = loom_low_invoke_callee(source_op);
  const iree_string_view_t callee_name =
      loom_low_invoke_symbol_name(module, callee_ref);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= module->symbols.count) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the callee is not a module-local symbol"));
  }

  const loom_symbol_t* callee_symbol =
      &module->symbols.entries[callee_ref.symbol_id];
  const loom_op_t* callee_op = callee_symbol->defining_op;
  if (callee_op == NULL || !loom_low_func_def_isa(callee_op)) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the callee is not a low.func.def with a body"));
  }

  const uint8_t allocation_mode = loom_low_func_def_allocation(callee_op);
  if (allocation_mode != 0 && allocation_mode != LOOM_LOW_ALLOCATION_VIRTUAL) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("assigned or fixed register allocation cannot be transferred "
                "into another function"));
  }
  const bool lock_schedule =
      loom_low_func_def_schedule(callee_op) == LOOM_LOW_SCHEDULE_LOCKED;

  const loom_region_t* callee_body = loom_low_func_def_body(callee_op);
  if (callee_body == NULL || callee_body->block_count != 1) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("required inlining supports exactly one outer body block"));
  }
  const loom_block_t* callee_entry = loom_region_const_entry_block(callee_body);
  const loom_op_t* callee_return = loom_block_const_last_op(callee_entry);
  if (callee_return == NULL || !loom_low_return_isa(callee_return)) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the helper body does not end in low.return"));
  }
  for (const loom_op_t* body_op = callee_entry->first_op;
       body_op != callee_return; body_op = body_op->next_op) {
    IREE_RETURN_IF_ERROR(loom_low_invoke_validate_body_op(
        context, source_op, callee_name, body_op, lock_schedule));
  }

  const loom_target_facts_t* callee_target_facts = NULL;
  const loom_low_descriptor_set_t* callee_descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_invoke_resolve_callee_target(
      context, source_op, callee_ref, callee_op, callee_name,
      &callee_target_facts, &callee_descriptor_set));
  if (callee_descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_target_facts_t* caller_target_facts =
      loom_low_lower_context_target_facts(context);
  if (callee_target_facts != NULL &&
      !loom_target_facts_satisfy_specialization_requirement(
          caller_target_facts, callee_target_facts)) {
    return loom_low_invoke_emit_target_error(
        context, source_op, callee_name,
        loom_target_facts_identity_name(callee_target_facts),
        loom_target_facts_identity_name(caller_target_facts));
  }

  const iree_string_view_t callee_contract =
      loom_low_invoke_function_descriptor_set(module, callee_op);
  const loom_low_descriptor_set_t* caller_descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t caller_target_contract =
      loom_low_lower_context_bundle(context)->config->contract_set_key;
  if (!loom_low_descriptor_set_supports_target_contract(
          callee_descriptor_set, caller_target_contract)) {
    return loom_low_invoke_emit_representation_error(
        context, source_op, callee_name, callee_contract,
        caller_target_contract);
  }
  const loom_low_invoke_projection_t projection = {
      .source_descriptor_set = callee_descriptor_set,
      .target_descriptor_set = caller_descriptor_set,
  };

  const loom_func_like_t callee = loom_func_like_const_cast(module, callee_op);
  uint16_t callee_argument_count = 0;
  const loom_value_id_t* callee_arguments =
      loom_func_like_arg_ids(callee, &callee_argument_count);
  const loom_value_slice_t invoke_operands =
      loom_low_invoke_operands(source_op);
  if (invoke_operands.count != callee_argument_count) {
    return loom_low_invoke_emit_count_error(
        context, source_op, callee_name, IREE_SV("operand"),
        invoke_operands.count, callee_argument_count);
  }

  const loom_value_slice_t callee_results =
      loom_low_func_def_results(callee_op);
  const loom_value_slice_t invoke_results = loom_low_invoke_results(source_op);
  if (invoke_results.count != callee_results.count) {
    return loom_low_invoke_emit_count_error(
        context, source_op, callee_name, IREE_SV("result"),
        invoke_results.count, callee_results.count);
  }
  const loom_value_slice_t return_values =
      loom_low_return_values(callee_return);
  if (return_values.count != callee_results.count) {
    return loom_low_invoke_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the low.return payload does not match the helper signature"));
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      module, module, loom_low_lower_context_emission_arena(context), NULL,
      &remap));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    loom_value_id_t low_operand = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, invoke_operands.values[i], &low_operand));
    const loom_type_t actual_type = loom_module_value_type(module, low_operand);
    const loom_type_t source_expected_type =
        loom_module_value_type(module, callee_arguments[i]);
    loom_type_t expected_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_invoke_project_register_type(
        module, &projection, source_expected_type, &expected_type));
    if (!loom_type_equal(actual_type, expected_type)) {
      return loom_low_invoke_emit_type_error(context, source_op, callee_name,
                                             IREE_SV("operand"), actual_type,
                                             expected_type);
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_structural_operand(
        context, source_op, i, invoke_operands.values[i], expected_type,
        &low_operand));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, callee_arguments[i], low_operand));
  }

  IREE_RETURN_IF_ERROR(loom_low_invoke_materialize_argument_contract(
      context, source_op, callee_name, callee, callee_arguments,
      callee_argument_count, invoke_operands, &remap));

  for (uint16_t i = 0; i < invoke_results.count; ++i) {
    loom_type_t actual_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_invoke_map_result_type(
        context, source_op, invoke_results.values[i], &actual_type));
    if (loom_type_kind(actual_type) == LOOM_TYPE_NONE) {
      return iree_ok_status();
    }
    const loom_type_t source_expected_type =
        loom_module_value_type(module, callee_results.values[i]);
    loom_type_t expected_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_invoke_project_register_type(
        module, &projection, source_expected_type, &expected_type));
    if (!loom_type_equal(actual_type, expected_type)) {
      return loom_low_invoke_emit_type_error(context, source_op, callee_name,
                                             IREE_SV("result"), actual_type,
                                             expected_type);
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_invoke_clone_body(
      context, source_op, callee_entry, lock_schedule, &projection, &remap));

  for (uint16_t i = 0; i < invoke_results.count; ++i) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &remap, return_values.values[i], &low_result));
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, invoke_results.values[i], low_result));
  }
  return iree_ok_status();
}
