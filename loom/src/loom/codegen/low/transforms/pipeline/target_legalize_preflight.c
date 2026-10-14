// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/target_legalize_preflight.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"

void loom_low_target_legalize_preflight_initialize(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* target_bundle,
    iree_diagnostic_emitter_t emitter, uint32_t max_errors,
    loom_low_target_legalize_preflight_t* out_preflight) {
  *out_preflight = (loom_low_target_legalize_preflight_t){
      .module = module,
      .function = function,
      .target_bundle = target_bundle,
      .emitter = emitter,
      .max_errors = max_errors,
  };
}

static iree_string_view_t loom_low_target_legalize_preflight_function_name(
    const loom_low_target_legalize_preflight_t* preflight) {
  const loom_symbol_ref_t callee = loom_func_like_callee(preflight->function);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= preflight->module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol =
      &preflight->module->symbols.entries[callee.symbol_id];
  if (symbol->name_id >= preflight->module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&preflight->module->strings, symbol->name_id);
}

static iree_string_view_t loom_low_target_legalize_preflight_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_target_legalize_preflight_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return IREE_SV("<unknown>");
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (value->name_id == LOOM_STRING_ID_INVALID ||
      value->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return loom_string_table_get(&module->strings, value->name_id);
}

static bool loom_low_target_legalize_should_stop_preflight(
    const loom_low_target_legalize_preflight_t* preflight) {
  return preflight->max_errors != 0 &&
         preflight->error_count >= preflight->max_errors;
}

static bool loom_low_target_legalize_op_is_in_function_entry_block(
    const loom_low_target_legalize_preflight_t* preflight,
    const loom_op_t* op) {
  if (!op->parent_block) {
    return false;
  }
  const loom_region_t* body = loom_func_like_body(preflight->function);
  return op->parent_block->parent_region == body &&
         op->parent_block->region_index == 0;
}

static bool loom_low_target_legalize_predicate_const_range(
    const loom_predicate_t* predicate, int64_t* out_minimum,
    int64_t* out_maximum) {
  *out_minimum = INT64_MIN;
  *out_maximum = INT64_MAX;
  if (predicate->arg_count < 1 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
    return false;
  }
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
      if (predicate->arg_count < 2 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST) {
        return false;
      }
      *out_minimum = predicate->args[1];
      *out_maximum = predicate->args[1];
      return true;
    case LOOM_PREDICATE_LT:
      if (predicate->arg_count < 2 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST ||
          predicate->args[1] == INT64_MIN) {
        return false;
      }
      *out_maximum = predicate->args[1] - 1;
      return true;
    case LOOM_PREDICATE_LE:
      if (predicate->arg_count < 2 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST) {
        return false;
      }
      *out_maximum = predicate->args[1];
      return true;
    case LOOM_PREDICATE_GT:
      if (predicate->arg_count < 2 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST ||
          predicate->args[1] == INT64_MAX) {
        return false;
      }
      *out_minimum = predicate->args[1] + 1;
      return true;
    case LOOM_PREDICATE_GE:
      if (predicate->arg_count < 2 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST) {
        return false;
      }
      *out_minimum = predicate->args[1];
      return true;
    case LOOM_PREDICATE_RANGE:
      if (predicate->arg_count < 3 ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_CONST ||
          predicate->arg_tags[2] != LOOM_PRED_ARG_CONST) {
        return false;
      }
      *out_minimum = predicate->args[1];
      *out_maximum = predicate->args[2];
      return true;
    default:
      return false;
  }
}

static bool loom_low_target_legalize_lookup_assume_operand_facts(
    const loom_value_fact_table_t* fact_table, loom_value_slice_t values,
    const loom_predicate_t* predicate, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL || predicate->arg_count < 1 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE || predicate->args[0] < 0 ||
      predicate->args[0] > UINT32_MAX) {
    return false;
  }
  const loom_value_id_t target_value = (loom_value_id_t)predicate->args[0];
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] != target_value) {
      continue;
    }
    *out_facts = loom_value_fact_table_lookup(fact_table, target_value);
    loom_value_facts_t element_facts = loom_value_facts_unknown();
    if (loom_value_facts_query_all_equal_element(&fact_table->context,
                                                 *out_facts, &element_facts)) {
      *out_facts = element_facts;
    }
    return true;
  }
  return false;
}

static void loom_low_target_legalize_context_params(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle = preflight->target_bundle;
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const loom_target_export_plan_t* export_plan =
      bundle ? bundle->export_plan : NULL;
  params[0] = loom_param_string(loom_low_target_legalize_preflight_nonempty(
      bundle ? bundle->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_low_target_legalize_preflight_nonempty(
      export_plan ? export_plan->name : iree_string_view_empty(),
      IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_low_target_legalize_preflight_nonempty(
      config ? config->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[3] = loom_param_string(loom_low_target_legalize_preflight_nonempty(
      loom_low_target_legalize_preflight_function_name(preflight),
      IREE_SV("<module>")));
  params[4] = loom_param_string(loom_op_name(preflight->module, op));
}

static iree_status_t loom_low_target_legalize_emit_topology_assume_error(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    iree_string_view_t value_kind, iree_string_view_t axis,
    int64_t required_minimum, int64_t required_maximum, int64_t assumed_minimum,
    int64_t assumed_maximum) {
  loom_diagnostic_param_t params[11];
  loom_low_target_legalize_context_params(preflight, op, params);
  params[5] = loom_param_string(value_kind);
  params[6] = loom_param_string(axis);
  params[7] = loom_param_i64(required_minimum);
  params[8] = loom_param_i64(required_maximum);
  params[9] = loom_param_i64(assumed_minimum);
  params[10] = loom_param_i64(assumed_maximum);
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_058,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(preflight->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_emit_assume_range_conflict_error(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_predicate_t* predicate, loom_value_id_t value_id,
    loom_value_fact_predicate_conflict_t conflict) {
  const char* predicate_name = loom_predicate_kind_name(predicate->kind);
  loom_diagnostic_param_t params[11];
  loom_low_target_legalize_context_params(preflight, op, params);
  params[5] =
      loom_param_string(predicate_name ? iree_make_cstring_view(predicate_name)
                                       : IREE_SV("<unknown>"));
  params[6] = loom_param_string(loom_low_target_legalize_preflight_value_name(
      preflight->module, value_id));
  params[7] = loom_param_i64(conflict.required_minimum);
  params[8] = loom_param_i64(conflict.required_maximum);
  params[9] = loom_param_i64(conflict.known_minimum);
  params[10] = loom_param_i64(conflict.known_maximum);
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_059,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(preflight->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_emit_assume_exact_i64_error(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_predicate_t* predicate, loom_value_id_t value_id,
    loom_value_fact_predicate_conflict_t conflict) {
  const char* predicate_name = loom_predicate_kind_name(predicate->kind);
  loom_diagnostic_param_t params[8];
  loom_low_target_legalize_context_params(preflight, op, params);
  params[5] =
      loom_param_string(predicate_name ? iree_make_cstring_view(predicate_name)
                                       : IREE_SV("<unknown>"));
  params[6] = loom_param_string(loom_low_target_legalize_preflight_value_name(
      preflight->module, value_id));
  params[7] = loom_param_i64(conflict.known_value);
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_060,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(preflight->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_string_view_t loom_low_target_legalize_float_class_name(
    loom_value_fact_predicate_conflict_float_class_t float_class) {
  switch (float_class) {
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NAN:
      return IREE_SV("nan");
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_INFINITY:
      return IREE_SV("infinity");
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NONE:
      return IREE_SV("<unknown>");
  }
  return IREE_SV("<unknown>");
}

static iree_status_t loom_low_target_legalize_emit_assume_float_class_error(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_predicate_t* predicate, loom_value_id_t value_id,
    loom_value_fact_predicate_conflict_t conflict) {
  const char* predicate_name = loom_predicate_kind_name(predicate->kind);
  loom_diagnostic_param_t params[8];
  loom_low_target_legalize_context_params(preflight, op, params);
  params[5] =
      loom_param_string(predicate_name ? iree_make_cstring_view(predicate_name)
                                       : IREE_SV("<unknown>"));
  params[6] = loom_param_string(loom_low_target_legalize_preflight_value_name(
      preflight->module, value_id));
  params[7] = loom_param_string(
      loom_low_target_legalize_float_class_name(conflict.known_float_class));
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_061,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(preflight->emitter, &emission));
  ++preflight->error_count;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_emit_assume_conflict_error(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_predicate_t* predicate, loom_value_id_t value_id,
    loom_value_fact_predicate_conflict_t conflict) {
  switch (conflict.kind) {
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_RANGE:
      return loom_low_target_legalize_emit_assume_range_conflict_error(
          preflight, op, predicate, value_id, conflict);
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_EXACT_I64:
      return loom_low_target_legalize_emit_assume_exact_i64_error(
          preflight, op, predicate, value_id, conflict);
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS:
      return loom_low_target_legalize_emit_assume_float_class_error(
          preflight, op, predicate, value_id, conflict);
    case LOOM_VALUE_FACT_PREDICATE_CONFLICT_NONE:
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unknown assume predicate conflict kind %d",
                          (int)conflict.kind);
}

static iree_status_t loom_low_target_legalize_check_assume_predicate_conflicts(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table, loom_value_slice_t values,
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  if (loom_low_target_legalize_should_stop_preflight(preflight)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0;
       i < predicate_count &&
       !loom_low_target_legalize_should_stop_preflight(preflight);
       ++i) {
    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!loom_low_target_legalize_lookup_assume_operand_facts(
            fact_table, values, &predicates[i], &facts)) {
      continue;
    }
    loom_value_fact_predicate_conflict_t conflict = {0};
    if (!loom_value_facts_predicate_conflict(facts, &predicates[i],
                                             &conflict)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_target_legalize_emit_assume_conflict_error(
        preflight, op, &predicates[i], (loom_value_id_t)predicates[i].args[0],
        conflict));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_check_assume_topology_domain(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table, loom_value_slice_t values,
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  if (loom_low_target_legalize_should_stop_preflight(preflight)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0;
       i < predicate_count &&
       !loom_low_target_legalize_should_stop_preflight(preflight);
       ++i) {
    int64_t assumed_minimum = INT64_MIN;
    int64_t assumed_maximum = INT64_MAX;
    if (!loom_low_target_legalize_predicate_const_range(
            &predicates[i], &assumed_minimum, &assumed_maximum)) {
      continue;
    }

    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!loom_low_target_legalize_lookup_assume_operand_facts(
            fact_table, values, &predicates[i], &facts)) {
      continue;
    }
    const loom_value_fact_topology_domain_t* domain =
        loom_value_facts_topology_domain(facts);
    if (!domain) {
      continue;
    }
    if (assumed_minimum <= facts.range_lo &&
        assumed_maximum >= facts.range_hi) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_target_legalize_emit_topology_assume_error(
        preflight, op, domain->value_kind_name, domain->axis_name,
        facts.range_lo, facts.range_hi, assumed_minimum, assumed_maximum));
  }
  return iree_ok_status();
}

iree_status_t loom_low_target_legalize_preflight_check_assume(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table) {
  loom_value_slice_t values = {0};
  loom_attribute_t predicate_attr = loom_attr_absent();
  if (loom_index_assume_isa(op)) {
    values = loom_index_assume_values(op);
    predicate_attr = loom_index_assume_predicates(op);
  } else if (loom_scalar_assume_isa(op)) {
    values = loom_scalar_assume_values(op);
    predicate_attr = loom_scalar_assume_predicates(op);
  } else if (loom_low_assume_isa(op)) {
    values = loom_low_assume_values(op);
    predicate_attr = loom_low_assume_predicates(op);
  } else {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_check_assume_predicate_conflicts(
          preflight, op, fact_table, values, predicate_attr.predicate_list,
          predicate_attr.count));
  if (!loom_low_target_legalize_op_is_in_function_entry_block(preflight, op)) {
    return iree_ok_status();
  }
  return loom_low_target_legalize_check_assume_topology_domain(
      preflight, op, fact_table, values, predicate_attr.predicate_list,
      predicate_attr.count);
}
