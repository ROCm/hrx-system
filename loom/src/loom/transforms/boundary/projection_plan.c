// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/projection_plan.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/pass_environment.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/walk.h"

static bool loom_boundary_projection_rule_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  return !rule->function_applies ||
         rule->function_applies(rule, plan, function);
}

static iree_host_size_t loom_boundary_projection_rule_index(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule) {
  for (iree_host_size_t i = 0; i < plan->rules.count; ++i) {
    if (plan->rules.values[i] == rule) {
      return i;
    }
  }
  IREE_ASSERT(false);
  return IREE_HOST_SIZE_MAX;
}

void* loom_boundary_projection_rule_state(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule) {
  const iree_host_size_t index =
      loom_boundary_projection_rule_index(plan, rule);
  return plan->rule_states[index];
}

void loom_boundary_projection_set_rule_state(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule, void* state) {
  const iree_host_size_t index =
      loom_boundary_projection_rule_index(plan, rule);
  IREE_ASSERT(plan->rule_states[index] == NULL);
  plan->rule_states[index] = state;
}

void* loom_boundary_projection_function_rule_state(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_rule_t* rule) {
  const iree_host_size_t index =
      loom_boundary_projection_rule_index(plan, rule);
  return function->rule_states[index];
}

void loom_boundary_projection_set_function_rule_state(
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_rule_t* rule, void* state) {
  const iree_host_size_t index =
      loom_boundary_projection_rule_index(plan, rule);
  IREE_ASSERT(function->rule_states[index] == NULL);
  function->rule_states[index] = state;
}

void loom_boundary_projection_record(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule, int64_t projections,
    int64_t components) {
  const iree_host_size_t index =
      loom_boundary_projection_rule_index(plan, rule);
  plan->rule_statistics[index].projections += projections;
  plan->rule_statistics[index].components += components;
}

static iree_status_t loom_boundary_projection_plan_slot_schema(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  const loom_type_kind_t kind =
      loom_type_kind(loom_module_value_type(plan->module, value_id));
  const loom_boundary_projection_type_kind_bits_t kind_bit =
      LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(kind);
  for (iree_host_size_t i = 0; i < plan->rules.count; ++i) {
    const loom_boundary_projection_rule_t* rule = plan->rules.values[i];
    if (!iree_any_bit_set(rule->type_kind_bits, kind_bit) ||
        !loom_boundary_projection_rule_applies(rule, plan, function)) {
      continue;
    }
    loom_boundary_projection_schema_t schema = {0};
    bool claimed = false;
    IREE_RETURN_IF_ERROR(rule->plan_slot(rule, plan, function, role, value_id,
                                         block, &schema, &claimed));
    if (!claimed) {
      continue;
    }
    IREE_ASSERT(!*out_claimed &&
                "projection rules must not claim the same logical slot");
    IREE_ASSERT(schema.rule == rule);
    IREE_ASSERT(schema.component_count == 0 || schema.component_types != NULL);
    *out_schema = schema;
    *out_claimed = true;
  }
  return iree_ok_status();
}

static bool loom_boundary_projection_may_claim_slot(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_value_id_t value_id) {
  const loom_type_kind_t kind =
      loom_type_kind(loom_module_value_type(plan->module, value_id));
  const loom_boundary_projection_type_kind_bits_t kind_bit =
      LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(kind);
  for (iree_host_size_t i = 0; i < plan->rules.count; ++i) {
    const loom_boundary_projection_rule_t* rule = plan->rules.values[i];
    if (iree_any_bit_set(rule->type_kind_bits, kind_bit) &&
        loom_boundary_projection_rule_applies(rule, plan, function)) {
      return true;
    }
  }
  return false;
}

static bool loom_boundary_projection_value_has_nonoperand_uses(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_module_value_has_type_uses(module, value_id) ||
         loom_value_has_attribute_uses(value);
}

static bool loom_boundary_projection_type_requires_coupled_transport(
    loom_type_t type) {
  // An SSA encoding is another logical boundary value whose projection must
  // be coordinated with the dependent value. A callable projection rule does
  // not yet own that paired signature, so selecting either half would discard
  // the runtime layout contract.
  return loom_type_has_ssa_encoding(type);
}

static bool loom_boundary_projection_tie_uses_expanded_slot(
    const loom_boundary_projection_function_t* function,
    const loom_tied_result_t* tie) {
  if (function->argument_operand_offset != UINT16_MAX &&
      tie->operand_index >= function->argument_operand_offset) {
    const uint16_t argument_index =
        (uint16_t)(tie->operand_index - function->argument_operand_offset);
    if (argument_index < function->argument_count &&
        loom_boundary_projection_schema_is_projected(
            &function->argument_schemas[argument_index])) {
      return true;
    }
  }
  return tie->result_index < function->result_count &&
         loom_boundary_projection_schema_is_projected(
             &function->result_schemas[tie->result_index]);
}

static bool loom_boundary_projection_call_tie_uses_expanded_slot(
    loom_call_like_t call, const loom_boundary_projection_function_t* callee,
    const loom_tied_result_t* tie) {
  const uint16_t operand_offset = loom_call_like_operand_offset(call);
  const uint16_t result_offset = loom_call_like_result_offset(call);
  if (tie->operand_index >= operand_offset) {
    const uint16_t argument_index =
        (uint16_t)(tie->operand_index - operand_offset);
    if (argument_index < callee->argument_count &&
        loom_boundary_projection_schema_is_projected(
            &callee->argument_schemas[argument_index])) {
      return true;
    }
  }
  if (tie->result_index >= result_offset) {
    const uint16_t result_index = (uint16_t)(tie->result_index - result_offset);
    if (result_index < callee->result_count &&
        loom_boundary_projection_schema_is_projected(
            &callee->result_schemas[result_index])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_boundary_projection_plan_function_signature(
    loom_boundary_projection_plan_t* plan, loom_func_like_t function,
    loom_function_version_t* version,
    loom_boundary_projection_function_t* out_function) {
  memset(out_function, 0, sizeof(*out_function));
  out_function->function = function;
  out_function->version = version;
  out_function->argument_operand_offset = UINT16_MAX;
  if (plan->rules.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, plan->rules.count, sizeof(*out_function->rule_states),
        (void**)&out_function->rule_states));
    memset(out_function->rule_states, 0,
           plan->rules.count * sizeof(*out_function->rule_states));
  }
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &out_function->argument_count);
  if (out_function->argument_count != 0) {
    loom_value_id_t* argument_copy = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, out_function->argument_count, sizeof(*argument_copy),
        (void**)&argument_copy));
    memcpy(argument_copy, arguments,
           out_function->argument_count * sizeof(*argument_copy));
    out_function->arguments = argument_copy;
  }
  out_function->result_count = function.op->result_count;
  if (out_function->result_count != 0) {
    loom_value_id_t* result_copy = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, out_function->result_count,
                                  sizeof(*result_copy), (void**)&result_copy));
    memcpy(result_copy, loom_op_const_results(function.op),
           out_function->result_count * sizeof(*result_copy));
    out_function->results = result_copy;
  }
  out_function->selected = true;

  if (!loom_func_like_isa(function) || function.op->successor_count != 0) {
    out_function->selected = false;
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body) {
    const uint8_t body_region_index =
        loom_func_like_body_region_index(function);
    const loom_op_vtable_t* op_vtable =
        loom_op_vtable(plan->module, function.op);
    const loom_region_descriptor_t* body_descriptor =
        loom_op_vtable_region_descriptor(op_vtable, body_region_index);
    if (!body_descriptor ||
        body_descriptor->terminator == LOOM_OP_KIND_UNKNOWN ||
        function.op->region_count != 1 || body_region_index != 0) {
      out_function->selected = false;
      return iree_ok_status();
    }
    out_function->return_kind = body_descriptor->terminator;
  } else {
    if (function.op->region_count != 0 ||
        (function.vtable->args_operand_field_index == LOOM_OPERAND_INDEX_NONE &&
         function.op->operand_count != 0)) {
      out_function->selected = false;
      return iree_ok_status();
    }
    if (function.vtable->args_operand_field_index != LOOM_OPERAND_INDEX_NONE) {
      const loom_op_vtable_t* op_vtable =
          loom_op_vtable(plan->module, function.op);
      const loom_value_slice_t argument_span = loom_op_operand_field_span(
          op_vtable, function.op, function.vtable->args_operand_field_index);
      IREE_ASSERT_EQ(argument_span.count, out_function->argument_count);
      out_function->argument_operand_offset =
          (uint16_t)(argument_span.values -
                     loom_op_const_operands(function.op));
    }
    out_function->return_kind = LOOM_OP_KIND_UNKNOWN;
  }

  if (out_function->argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, out_function->argument_count,
                                  sizeof(*out_function->argument_schemas),
                                  (void**)&out_function->argument_schemas));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, out_function->argument_count,
                                  sizeof(*out_function->argument_indices),
                                  (void**)&out_function->argument_indices));
    memset(
        out_function->argument_schemas, 0,
        out_function->argument_count * sizeof(*out_function->argument_schemas));
  }
  uint32_t final_argument_count = 0;
  for (uint16_t i = 0; i < out_function->argument_count; ++i) {
    out_function->argument_indices[i] =
        final_argument_count <= UINT16_MAX ? (uint16_t)final_argument_count : 0;
    const loom_value_id_t argument = out_function->arguments[i];
    loom_boundary_projection_schema_t schema = {0};
    bool claimed = false;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
        plan, out_function, LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT,
        argument, /*block=*/NULL, &schema, &claimed));
    if (!claimed) {
      ++final_argument_count;
      continue;
    }
    const loom_type_t type = loom_module_value_type(plan->module, argument);
    out_function->argument_schemas[i] = schema;
    out_function->signature_changes = true;
    final_argument_count += schema.component_count;
    if (loom_boundary_projection_type_requires_coupled_transport(type) ||
        loom_boundary_projection_value_has_nonoperand_uses(plan->module,
                                                           argument)) {
      out_function->selected = false;
    }
  }
  if (final_argument_count > UINT16_MAX) {
    out_function->selected = false;
  } else {
    out_function->final_argument_count = (uint16_t)final_argument_count;
  }

  if (out_function->result_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, out_function->result_count,
                                  sizeof(*out_function->result_schemas),
                                  (void**)&out_function->result_schemas));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, out_function->result_count,
                                  sizeof(*out_function->result_indices),
                                  (void**)&out_function->result_indices));
    memset(out_function->result_schemas, 0,
           out_function->result_count * sizeof(*out_function->result_schemas));
  }
  uint32_t final_result_count = 0;
  for (uint16_t i = 0; i < out_function->result_count; ++i) {
    out_function->result_indices[i] =
        final_result_count <= UINT16_MAX ? (uint16_t)final_result_count : 0;
    const loom_value_id_t result = out_function->results[i];
    loom_boundary_projection_schema_t schema = {0};
    bool claimed = false;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
        plan, out_function, LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT,
        result, /*block=*/NULL, &schema, &claimed));
    if (!claimed) {
      ++final_result_count;
      continue;
    }
    const loom_type_t type = loom_module_value_type(plan->module, result);
    out_function->result_schemas[i] = schema;
    out_function->signature_changes = true;
    out_function->result_signature_changes = true;
    final_result_count += schema.component_count;
    if (loom_boundary_projection_type_requires_coupled_transport(type) ||
        loom_boundary_projection_value_has_nonoperand_uses(plan->module,
                                                           result) ||
        loom_module_value(plan->module, result)->use_count != 0) {
      out_function->selected = false;
    }
  }
  if (final_result_count > UINT16_MAX) {
    out_function->selected = false;
  } else {
    out_function->final_result_count = (uint16_t)final_result_count;
  }

  const loom_tied_result_t* ties = loom_op_tied_results(function.op);
  for (uint16_t i = 0; i < function.op->tied_result_count; ++i) {
    if (loom_boundary_projection_tie_uses_expanded_slot(out_function,
                                                        &ties[i])) {
      out_function->selected = false;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_plan_functions(
    loom_boundary_projection_plan_t* plan,
    const loom_function_version_list_t* version_list) {
  plan->function_index_count = plan->module->symbols.count;
  if (plan->function_index_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->function_index_count, sizeof(*plan->function_indices),
      (void**)&plan->function_indices));
  for (iree_host_size_t i = 0; i < plan->function_index_count; ++i) {
    plan->function_indices[i] = IREE_HOST_SIZE_MAX;
  }
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      plan->module, version_list, plan->arena, &plan->versions));

  iree_host_size_t function_count = 0;
  for (iree_host_size_t symbol_id = 0; symbol_id < plan->module->symbols.count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &plan->module->symbols.entries[symbol_id];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        symbol->defining_op == NULL) {
      continue;
    }
    ++function_count;
  }
  if (function_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, function_count,
                                                 sizeof(*plan->functions),
                                                 (void**)&plan->functions));
  memset(plan->functions, 0, function_count * sizeof(*plan->functions));

  for (iree_host_size_t symbol_id = 0; symbol_id < plan->module->symbols.count;
       ++symbol_id) {
    loom_function_version_t* version =
        loom_target_function_version_snapshot_handle_at(&plan->versions,
                                                        symbol_id);
    loom_symbol_t* symbol = &plan->module->symbols.entries[symbol_id];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        symbol->defining_op == NULL) {
      continue;
    }
    const iree_host_size_t index = plan->function_count++;
    plan->function_indices[symbol_id] = index;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_function_signature(
        plan, loom_func_like_cast(plan->module, symbol->defining_op), version,
        &plan->functions[index]));
  }
  IREE_ASSERT_EQ(plan->function_count, function_count);
  return iree_ok_status();
}

static iree_host_size_t loom_boundary_projection_function_index(
    const loom_boundary_projection_plan_t* plan, loom_symbol_ref_t symbol) {
  return loom_symbol_ref_is_valid(symbol) && symbol.module_id == 0 &&
                 symbol.symbol_id < plan->function_index_count
             ? plan->function_indices[symbol.symbol_id]
             : IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_boundary_projection_add_slot(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_value_id_t value_id,
    loom_boundary_projection_slot_role_t role, loom_block_t* block,
    loom_op_t* call_op,
    const loom_boundary_projection_schema_t* required_schema) {
  loom_boundary_projection_schema_t schema = {0};
  bool claimed = required_schema != NULL;
  if (required_schema) {
    schema = *required_schema;
  } else {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
        plan, function, role, value_id, block, &schema, &claimed));
  }
  if (!claimed) {
    return iree_ok_status();
  }
  if (function->candidate_count == function->candidate_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->candidate_count, function->candidate_count + 1,
        sizeof(*function->candidates), &function->candidate_capacity,
        (void**)&function->candidates));
  }
  loom_value_id_t* component_value_ids = NULL;
  if (schema.component_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, schema.component_count, sizeof(*component_value_ids),
        (void**)&component_value_ids));
  }
  for (uint16_t i = 0; i < schema.component_count; ++i) {
    component_value_ids[i] = LOOM_VALUE_ID_INVALID;
  }
  const loom_boundary_projection_slot_t candidate = {
      .value_id = value_id,
      .logical_type = loom_module_value_type(plan->module, value_id),
      .schema = schema,
      .block = block,
      .reconstruction_anchor = block ? block->first_op : NULL,
      .call_op = call_op,
      .component_value_ids = component_value_ids,
      .replacement_value_id = LOOM_VALUE_ID_INVALID,
      .role = role,
      .first_dependent = IREE_HOST_SIZE_MAX,
      .selected = true,
  };
  function->candidates[function->candidate_count++] = candidate;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_add_call(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_call_like_t call,
    iree_host_size_t callee_index) {
  if (function->call_count == function->call_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->call_count, function->call_count + 1,
        sizeof(*function->calls), &function->call_capacity,
        (void**)&function->calls));
  }
  function->calls[function->call_count++] = (loom_boundary_projection_call_t){
      .call = call,
      .callee_index = callee_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_add_return(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_op_t* op) {
  if (function->return_count == function->return_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->return_count, function->return_count + 1,
        sizeof(*function->returns), &function->return_capacity,
        (void**)&function->returns));
  }
  function->returns[function->return_count++] =
      (loom_boundary_projection_return_t){.op = op};
  return iree_ok_status();
}

typedef struct loom_boundary_projection_collect_t {
  // Whole-module plan used to resolve callees.
  loom_boundary_projection_plan_t* plan;
  // Function currently being walked.
  loom_boundary_projection_function_t* function;
} loom_boundary_projection_collect_t;

static int loom_boundary_projection_candidate_compare(const void* lhs,
                                                      const void* rhs) {
  const loom_boundary_projection_slot_t* left = lhs;
  const loom_boundary_projection_slot_t* right = rhs;
  return left->value_id < right->value_id   ? -1
         : left->value_id > right->value_id ? 1
                                            : 0;
}

static iree_status_t loom_boundary_projection_collect_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_boundary_projection_collect_t* collect = user_data;

  if (collect->function->result_signature_changes &&
      collect->function->return_kind != LOOM_OP_KIND_UNKNOWN &&
      op->kind == collect->function->return_kind &&
      op->parent_op == collect->function->function.op) {
    return loom_boundary_projection_add_return(collect->plan, collect->function,
                                               op);
  }

  const loom_call_like_t call = loom_call_like_cast(collect->plan->module, op);
  if (!loom_call_like_isa(call) ||
      loom_call_like_kind(call) != LOOM_CALL_LIKE_KIND_SEMANTIC) {
    return iree_ok_status();
  }
  const loom_symbol_ref_t callee = loom_call_like_callee(call);
  const loom_value_slice_t operands = loom_call_like_operands(call);
  const loom_value_slice_t results = loom_call_like_results(call);
  const iree_host_size_t callee_index =
      loom_boundary_projection_function_index(collect->plan, callee);
  if (callee_index == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  loom_boundary_projection_function_t* callee_plan =
      &collect->plan->functions[callee_index];
  if (!callee_plan->signature_changes) {
    return iree_ok_status();
  }
  if (operands.count != callee_plan->argument_count ||
      results.count != callee_plan->result_count) {
    collect->function->selected = false;
    callee_plan->selected = false;
    return iree_ok_status();
  }
  if (op->region_count != 0 || op->successor_count != 0) {
    collect->function->selected = false;
    callee_plan->selected = false;
    return iree_ok_status();
  }
  if ((uint32_t)loom_call_like_operand_offset(call) +
              callee_plan->final_argument_count >
          UINT16_MAX ||
      (uint32_t)loom_call_like_result_offset(call) +
              callee_plan->final_result_count >
          UINT16_MAX) {
    collect->function->selected = false;
    callee_plan->selected = false;
    return iree_ok_status();
  }
  const loom_tied_result_t* ties = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (loom_boundary_projection_call_tie_uses_expanded_slot(call, callee_plan,
                                                             &ties[i])) {
      collect->function->selected = false;
      callee_plan->selected = false;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_boundary_projection_add_call(
      collect->plan, collect->function, call, callee_index));
  for (uint16_t i = 0; i < results.count; ++i) {
    if (!loom_boundary_projection_schema_is_projected(
            &callee_plan->result_schemas[i])) {
      continue;
    }
    if (loom_boundary_projection_value_has_nonoperand_uses(
            collect->plan->module, results.values[i])) {
      collect->function->selected = false;
      callee_plan->selected = false;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_slot(
        collect->plan, collect->function, results.values[i],
        LOOM_BOUNDARY_PROJECTION_SLOT_CALL_RESULT, NULL, op,
        &callee_plan->result_schemas[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_collect_function(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  loom_region_t* body = loom_func_like_body(function->function);
  if (!body) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      plan->module, body, plan->arena, &function->domain));

  loom_boundary_projection_collect_t collect = {
      .plan = plan,
      .function = function,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      plan->module, function->function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_boundary_projection_collect_op, &collect},
      plan->arena, &walk_result));

  bool may_have_block_slot = false;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      may_have_block_slot |= loom_boundary_projection_may_claim_slot(
          plan, function, loom_block_arg_id(block, i));
    }
  }

  if (function->signature_changes || function->call_count != 0 ||
      function->candidate_count != 0 || may_have_block_slot) {
    const loom_target_function_version_t* target_version =
        loom_target_function_version_const_cast(function->version);
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
        plan->pass, plan->module,
        loom_pass_value_fact_scope_function_for_target(
            function->function,
            target_version ? target_version->function_target_facts : NULL),
        &function->facts));
    function->cfg =
        loom_value_fact_table_lookup_cfg_region(function->facts, body);
  }

  if (may_have_block_slot) {
    loom_region_for_each_block(body, block) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_add_slot(
            plan, function, loom_block_arg_id(block, i),
            LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT, block, NULL,
            /*required_schema=*/NULL));
      }
    }
  }

  if (function->candidate_count == 0 && function->call_count == 0 &&
      !function->signature_changes) {
    function->selected = false;
    return iree_ok_status();
  }
  if (function->candidate_count > 1) {
    qsort(function->candidates, function->candidate_count,
          sizeof(*function->candidates),
          loom_boundary_projection_candidate_compare);
    for (iree_host_size_t i = 1; i < function->candidate_count; ++i) {
      if (function->candidates[i - 1].value_id ==
          function->candidates[i].value_id) {
        function->selected = false;
      }
    }
  }
  return iree_ok_status();
}

iree_host_size_t loom_boundary_projection_slot_index(
    const loom_boundary_projection_function_t* function,
    loom_value_id_t value_id) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = function->candidate_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_value_id_t candidate = function->candidates[middle].value_id;
    if (candidate < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < function->candidate_count &&
                 function->candidates[begin].value_id == value_id
             ? begin
             : IREE_HOST_SIZE_MAX;
}

iree_status_t loom_boundary_projection_add_dependency(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, iree_host_size_t source,
    iree_host_size_t target, bool orders_reconstruction) {
  IREE_ASSERT_LT(source, function->candidate_count);
  IREE_ASSERT_LT(target, function->candidate_count);
  for (iree_host_size_t edge = function->candidates[source].first_dependent;
       edge != IREE_HOST_SIZE_MAX; edge = function->dependencies[edge].next) {
    loom_boundary_projection_dependency_t* dependency =
        &function->dependencies[edge];
    if (dependency->target == target) {
      dependency->orders_reconstruction |= orders_reconstruction;
      return iree_ok_status();
    }
  }
  if (function->dependency_count == function->dependency_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->dependency_count, function->dependency_count + 1,
        sizeof(*function->dependencies), &function->dependency_capacity,
        (void**)&function->dependencies));
  }
  const loom_boundary_projection_dependency_t dependency = {
      .target = target,
      .next = function->candidates[source].first_dependent,
      .orders_reconstruction = orders_reconstruction,
  };
  function->dependencies[function->dependency_count] = dependency;
  function->candidates[source].first_dependent = function->dependency_count++;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_plan_call_coordinates(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t call_index = 0; call_index < function->call_count;
       ++call_index) {
    loom_boundary_projection_call_t* call = &function->calls[call_index];
    loom_boundary_projection_function_t* callee =
        &plan->functions[call->callee_index];
    const loom_value_slice_t operands = loom_call_like_operands(call->call);
    if (operands.count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, operands.count, sizeof(*call->operand_sources),
          (void**)&call->operand_sources));
      memset(call->operand_sources, 0,
             operands.count * sizeof(*call->operand_sources));
    }
    for (uint16_t i = 0; i < operands.count; ++i) {
      bool planned = false;
      if (loom_boundary_projection_schema_is_projected(
              &callee->argument_schemas[i])) {
        const loom_boundary_projection_schema_t* schema =
            &callee->argument_schemas[i];
        IREE_RETURN_IF_ERROR(schema->rule->transport.plan_source(
            schema->rule, plan, function, /*destination=*/NULL, schema,
            operands.values[i], call->call.op, &call->operand_sources[i],
            &planned));
      }
      if (loom_boundary_projection_schema_is_projected(
              &callee->argument_schemas[i]) &&
          !planned) {
        function->selected = false;
        callee->selected = false;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_plan_return_coordinates(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t return_index = 0; return_index < function->return_count;
       ++return_index) {
    loom_boundary_projection_return_t* return_plan =
        &function->returns[return_index];
    const loom_value_slice_t operands = {
        .values = loom_op_operands(return_plan->op),
        .count = return_plan->op->operand_count,
    };
    if (operands.count != function->result_count) {
      function->selected = false;
      continue;
    }
    const loom_op_vtable_t* vtable =
        loom_op_vtable(plan->module, return_plan->op);
    if (return_plan->op->result_count != 0 ||
        return_plan->op->region_count != 0 ||
        return_plan->op->successor_count != 0 ||
        return_plan->op->tied_result_count != 0 ||
        loom_op_vtable_has_segmented_operands(vtable)) {
      function->selected = false;
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, operands.count, sizeof(*return_plan->sources),
        (void**)&return_plan->sources));
    memset(return_plan->sources, 0,
           operands.count * sizeof(*return_plan->sources));
    for (uint16_t i = 0; i < operands.count; ++i) {
      bool planned = false;
      if (loom_boundary_projection_schema_is_projected(
              &function->result_schemas[i])) {
        const loom_boundary_projection_schema_t* schema =
            &function->result_schemas[i];
        IREE_RETURN_IF_ERROR(schema->rule->transport.plan_source(
            schema->rule, plan, function, /*destination=*/NULL, schema,
            operands.values[i], return_plan->op, &return_plan->sources[i],
            &planned));
      }
      if (loom_boundary_projection_schema_is_projected(
              &function->result_schemas[i]) &&
          !planned) {
        function->selected = false;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_preflight_cfg_slot(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot) {
  if (!slot->selected ||
      slot->role != LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function->function);
  if (!body || slot->block == loom_region_entry_block(body)) {
    return iree_ok_status();
  }
  if (!function->cfg || function->cfg->graph.malformed) {
    slot->selected = false;
    return iree_ok_status();
  }

  const uint16_t block_index = slot->block->region_index;
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(&function->cfg->graph, block_index);
  slot->incoming_source_count = predecessor_edges.count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, predecessor_edges.count, sizeof(*slot->incoming_sources),
      (void**)&slot->incoming_sources));
  memset(slot->incoming_sources, 0,
         predecessor_edges.count * sizeof(*slot->incoming_sources));
  const uint16_t argument_index =
      loom_value_def_index(loom_module_value(plan->module, slot->value_id));
  for (iree_host_size_t edge_index = 0; edge_index < predecessor_edges.count;
       ++edge_index) {
    const loom_cfg_edge_info_t* edge = loom_cfg_graph_edge(
        &function->cfg->graph, predecessor_edges.values[edge_index]);
    loom_op_t* terminator = edge ? (loom_op_t*)edge->terminator : NULL;
    const loom_value_id_t* sources = NULL;
    uint16_t source_count = 0;
    const loom_op_vtable_t* vtable =
        terminator ? loom_op_vtable(plan->module, terminator) : NULL;
    if (!loom_cfg_terminator_payload_for_successor(terminator, slot->block,
                                                   &sources, &source_count) ||
        source_count != slot->block->arg_count ||
        terminator->successor_count != 1 || terminator->result_count != 0 ||
        terminator->region_count != 0 || terminator->tied_result_count != 0 ||
        loom_op_vtable_has_segmented_operands(vtable)) {
      slot->selected = false;
      continue;
    }
    bool planned = false;
    IREE_RETURN_IF_ERROR(slot->schema.rule->transport.plan_source(
        slot->schema.rule, plan, function, slot, &slot->schema,
        sources[argument_index], terminator,
        &slot->incoming_sources[edge_index], &planned));
    if (!planned) {
      slot->selected = false;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_plan_type_dependencies(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t source_index = 0;
       source_index < function->candidate_count; ++source_index) {
    loom_boundary_projection_slot_t* source =
        &function->candidates[source_index];
    if (!source->selected ||
        source->role != LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT) {
      continue;
    }
    loom_type_use_iterator_t users;
    loom_module_value_type_users(plan->module, source->value_id, &users);
    for (loom_value_id_t carrier_id = loom_type_users_next(&users);
         carrier_id != LOOM_VALUE_ID_INVALID;
         carrier_id = loom_type_users_next(&users)) {
      const loom_value_t* carrier_value =
          loom_module_value(plan->module, carrier_id);
      if (!loom_value_is_block_arg(carrier_value) ||
          loom_value_def_block(carrier_value) != source->block) {
        continue;
      }
      const iree_host_size_t carrier_index =
          loom_boundary_projection_slot_index(function, carrier_id);
      if (carrier_index == IREE_HOST_SIZE_MAX ||
          !function->candidates[carrier_index].selected) {
        source->selected = false;
        continue;
      }
      const loom_boundary_projection_schema_t* carrier_schema =
          &function->candidates[carrier_index].schema;
      bool dependency_removed = true;
      for (uint16_t component = 0; component < carrier_schema->component_count;
           ++component) {
        dependency_removed &= !loom_type_references_value(
            plan->module, carrier_schema->component_types[component],
            source->value_id);
      }
      if (!dependency_removed) {
        source->selected = false;
        continue;
      }
      // The two slots form one availability component, while only the
      // provider-to-carrier edge constrains reconstruction order.
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, source_index, carrier_index,
          /*orders_reconstruction=*/true));
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, carrier_index, source_index,
          /*orders_reconstruction=*/false));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_propagate_slot_rejections(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  if (function->candidate_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, function->candidate_count, sizeof(*queue), (void**)&queue));
  iree_host_size_t head = 0;
  iree_host_size_t tail = 0;
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    if (!function->candidates[i].selected) {
      queue[tail++] = i;
    }
  }
  while (head < tail) {
    const iree_host_size_t source = queue[head++];
    for (iree_host_size_t edge = function->candidates[source].first_dependent;
         edge != IREE_HOST_SIZE_MAX; edge = function->dependencies[edge].next) {
      const iree_host_size_t target = function->dependencies[edge].target;
      if (function->candidates[target].selected) {
        function->candidates[target].selected = false;
        queue[tail++] = target;
      }
    }
  }
  return iree_ok_status();
}

static void loom_boundary_projection_reject_block_slots(
    loom_boundary_projection_function_t* function, loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    const iree_host_size_t slot_index = loom_boundary_projection_slot_index(
        function, loom_block_arg_id(block, i));
    if (slot_index != IREE_HOST_SIZE_MAX) {
      function->candidates[slot_index].selected = false;
    }
  }
}

static void loom_boundary_projection_preflight_block_arities(
    loom_boundary_projection_function_t* function) {
  loom_region_t* body = loom_func_like_body(function->function);
  if (!body) {
    return;
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    uint32_t final_argument_count = block->arg_count;
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      const iree_host_size_t slot_index = loom_boundary_projection_slot_index(
          function, loom_block_arg_id(block, i));
      if (slot_index == IREE_HOST_SIZE_MAX ||
          !function->candidates[slot_index].selected) {
        continue;
      }
      final_argument_count =
          final_argument_count - 1u +
          function->candidates[slot_index].schema.component_count;
    }
    if (final_argument_count > UINT16_MAX) {
      loom_boundary_projection_reject_block_slots(function, block);
    }
  }
}

static iree_status_t loom_boundary_projection_plan_reconstruction_order(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  iree_host_size_t selected_count = 0;
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    selected_count += function->candidates[i].selected &&
                      function->candidates[i].role ==
                          LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT;
  }
  function->reconstruction_count = selected_count;
  if (selected_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t* indegrees = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, function->candidate_count,
                                sizeof(*indegrees), (void**)&indegrees));
  memset(indegrees, 0, function->candidate_count * sizeof(*indegrees));
  for (iree_host_size_t source = 0; source < function->candidate_count;
       ++source) {
    if (!function->candidates[source].selected) {
      continue;
    }
    for (iree_host_size_t edge = function->candidates[source].first_dependent;
         edge != IREE_HOST_SIZE_MAX; edge = function->dependencies[edge].next) {
      const loom_boundary_projection_dependency_t* dependency =
          &function->dependencies[edge];
      if (dependency->orders_reconstruction &&
          function->candidates[dependency->target].selected) {
        ++indegrees[dependency->target];
      }
    }
  }

  iree_host_size_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, selected_count, sizeof(*queue), (void**)&queue));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, selected_count, sizeof(*function->reconstruction_order),
      (void**)&function->reconstruction_order));
  iree_host_size_t head = 0;
  iree_host_size_t tail = 0;
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    if (function->candidates[i].selected &&
        function->candidates[i].role ==
            LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT &&
        indegrees[i] == 0) {
      queue[tail++] = i;
    }
  }
  iree_host_size_t ordered_count = 0;
  while (head < tail) {
    const iree_host_size_t source = queue[head++];
    function->reconstruction_order[ordered_count++] = source;
    for (iree_host_size_t edge = function->candidates[source].first_dependent;
         edge != IREE_HOST_SIZE_MAX; edge = function->dependencies[edge].next) {
      const loom_boundary_projection_dependency_t* dependency =
          &function->dependencies[edge];
      const iree_host_size_t target = dependency->target;
      if (!dependency->orders_reconstruction ||
          !function->candidates[target].selected) {
        continue;
      }
      IREE_ASSERT_GT(indegrees[target], 0);
      if (--indegrees[target] == 0) {
        queue[tail++] = target;
      }
    }
  }
  IREE_ASSERT_EQ(ordered_count, selected_count);
  return iree_ok_status();
}

static bool loom_boundary_projection_block_has_selected_slot(
    const loom_boundary_projection_function_t* function,
    const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    const iree_host_size_t slot_index = loom_boundary_projection_slot_index(
        function, loom_block_arg_id(block, i));
    if (slot_index != IREE_HOST_SIZE_MAX &&
        function->candidates[slot_index].selected) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_boundary_projection_build_block_plans(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  if (!function->cfg || function->cfg->graph.malformed) {
    return iree_ok_status();
  }
  const loom_cfg_graph_t* graph = &function->cfg->graph;
  iree_host_size_t block_count = 0;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    block_count += loom_boundary_projection_block_has_selected_slot(
        function, graph->blocks[block_index].block);
  }
  if (block_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, block_count,
                                                 sizeof(*function->blocks),
                                                 (void**)&function->blocks));
  memset(function->blocks, 0, block_count * sizeof(*function->blocks));

  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!loom_boundary_projection_block_has_selected_slot(function, block)) {
      continue;
    }
    loom_boundary_projection_block_t* block_plan =
        &function->blocks[function->block_count++];
    block_plan->block = block;
    block_plan->original_argument_count = block->arg_count;
    uint32_t final_argument_count = block->arg_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, block->arg_count, sizeof(*block_plan->original_arguments),
        (void**)&block_plan->original_arguments));
    memcpy(block_plan->original_arguments, block->arg_ids,
           block->arg_count * sizeof(*block->arg_ids));
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      const iree_host_size_t slot_index = loom_boundary_projection_slot_index(
          function, block_plan->original_arguments[i]);
      if (slot_index != IREE_HOST_SIZE_MAX &&
          function->candidates[slot_index].selected) {
        final_argument_count =
            final_argument_count - 1u +
            function->candidates[slot_index].schema.component_count;
      }
    }
    IREE_ASSERT_LE(final_argument_count, UINT16_MAX);
    block_plan->final_argument_count = (uint16_t)final_argument_count;

    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, block_index);
    block_plan->edge_count = predecessor_edges.count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, predecessor_edges.count, sizeof(*block_plan->edges),
        (void**)&block_plan->edges));
    memset(block_plan->edges, 0,
           predecessor_edges.count * sizeof(*block_plan->edges));
    for (iree_host_size_t edge_index = 0; edge_index < predecessor_edges.count;
         ++edge_index) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, predecessor_edges.values[edge_index]);
      loom_boundary_projection_edge_t* edge_plan =
          &block_plan->edges[edge_index];
      edge_plan->terminator = (loom_op_t*)edge->terminator;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, block->arg_count, sizeof(*edge_plan->sources),
          (void**)&edge_plan->sources));
      memset(edge_plan->sources, 0,
             block->arg_count * sizeof(*edge_plan->sources));
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        const iree_host_size_t slot_index = loom_boundary_projection_slot_index(
            function, block_plan->original_arguments[i]);
        if (slot_index == IREE_HOST_SIZE_MAX ||
            !function->candidates[slot_index].selected) {
          continue;
        }
        const loom_boundary_projection_slot_t* slot =
            &function->candidates[slot_index];
        IREE_ASSERT_EQ(slot->incoming_source_count, predecessor_edges.count);
        edge_plan->sources[i] = slot->incoming_sources[edge_index];
      }
    }
  }
  IREE_ASSERT_EQ(function->block_count, block_count);
  return iree_ok_status();
}

static iree_host_size_t loom_boundary_projection_component_root(
    iree_host_size_t* parents, iree_host_size_t index) {
  iree_host_size_t root = index;
  while (parents[root] != root) {
    root = parents[root];
  }
  while (parents[index] != index) {
    const iree_host_size_t parent = parents[index];
    parents[index] = root;
    index = parent;
  }
  return root;
}

static iree_status_t loom_boundary_projection_propagate_rejections(
    loom_boundary_projection_plan_t* plan) {
  if (plan->function_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t* parents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->function_count, sizeof(*parents), (void**)&parents));
  iree_host_size_t* component_sizes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->function_count, sizeof(*component_sizes),
      (void**)&component_sizes));
  bool* component_selected = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->function_count, sizeof(*component_selected),
      (void**)&component_selected));
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    parents[i] = i;
    component_sizes[i] = 1;
    component_selected[i] = true;
  }

  // Every rewritten semantic call couples its caller and callee signatures.
  // Union the undirected call components once so rejection is linear in the
  // planned call graph rather than repeatedly rescanning long call chains.
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    const loom_boundary_projection_function_t* caller = &plan->functions[i];
    for (iree_host_size_t j = 0; j < caller->call_count; ++j) {
      iree_host_size_t caller_root =
          loom_boundary_projection_component_root(parents, i);
      iree_host_size_t callee_root = loom_boundary_projection_component_root(
          parents, caller->calls[j].callee_index);
      if (caller_root == callee_root) {
        continue;
      }
      if (component_sizes[caller_root] < component_sizes[callee_root]) {
        const iree_host_size_t temporary = caller_root;
        caller_root = callee_root;
        callee_root = temporary;
      }
      parents[callee_root] = caller_root;
      component_sizes[caller_root] += component_sizes[callee_root];
    }
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    const iree_host_size_t root =
        loom_boundary_projection_component_root(parents, i);
    component_selected[root] &= plan->functions[i].selected;
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    const iree_host_size_t root =
        loom_boundary_projection_component_root(parents, i);
    plan->functions[i].selected = component_selected[root];
  }
  return iree_ok_status();
}

iree_status_t loom_boundary_projection_plan_prepare(
    loom_boundary_projection_plan_t* plan,
    const loom_function_version_list_t* version_list,
    loom_boundary_projection_rule_list_t rules) {
  IREE_ASSERT(rules.count == 0 || rules.values != NULL);
  for (iree_host_size_t i = 0; i < rules.count; ++i) {
    const loom_boundary_projection_rule_t* rule = rules.values[i];
    IREE_ASSERT(rule != NULL);
    IREE_ASSERT(!iree_string_view_is_empty(rule->name));
    IREE_ASSERT_NE(rule->type_kind_bits, 0);
    IREE_ASSERT(rule->plan_slot != NULL);
    IREE_ASSERT(rule->transport.plan_source != NULL);
    IREE_ASSERT(rule->transport.materialize_source != NULL);
    IREE_ASSERT(rule->transport.reconstruct != NULL);
    for (iree_host_size_t j = 0; j < i; ++j) {
      IREE_ASSERT_NE(rule, rules.values[j]);
    }
  }
  plan->rules = rules;
  if (rules.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, rules.count,
                                                   sizeof(*plan->rule_states),
                                                   (void**)&plan->rule_states));
    memset(plan->rule_states, 0, rules.count * sizeof(*plan->rule_states));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, rules.count, sizeof(*plan->rule_statistics),
        (void**)&plan->rule_statistics));
    memset(plan->rule_statistics, 0,
           rules.count * sizeof(*plan->rule_statistics));
  }
  for (iree_host_size_t i = 0; i < rules.count; ++i) {
    if (rules.values[i]->initialize) {
      IREE_RETURN_IF_ERROR(rules.values[i]->initialize(rules.values[i], plan));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_plan_functions(plan, version_list));
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    loom_boundary_projection_function_t* function = &plan->functions[i];
    iree_status_t status =
        loom_boundary_projection_collect_function(plan, function);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t rule_index = 0;
           rule_index < rules.count && iree_status_is_ok(status);
           ++rule_index) {
        const loom_boundary_projection_rule_t* rule = rules.values[rule_index];
        if (rule->prepare_function &&
            loom_boundary_projection_rule_applies(rule, plan, function)) {
          status = rule->prepare_function(rule, plan, function);
        }
      }
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      status = loom_boundary_projection_plan_call_coordinates(plan, function);
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      status = loom_boundary_projection_plan_return_coordinates(plan, function);
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      status = loom_boundary_projection_plan_type_dependencies(plan, function);
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      for (iree_host_size_t slot_index = 0;
           slot_index < function->candidate_count && iree_status_is_ok(status);
           ++slot_index) {
        status = loom_boundary_projection_preflight_cfg_slot(
            plan, function, &function->candidates[slot_index]);
      }
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      loom_boundary_projection_preflight_block_arities(function);
      status =
          loom_boundary_projection_propagate_slot_rejections(plan, function);
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      status =
          loom_boundary_projection_plan_reconstruction_order(plan, function);
    }
    if (iree_status_is_ok(status) && function->selected &&
        loom_func_like_body(function->function)) {
      status = loom_boundary_projection_build_block_plans(plan, function);
    }
    loom_local_value_domain_release(&function->domain);
    IREE_RETURN_IF_ERROR(status);
  }
  return loom_boundary_projection_propagate_rejections(plan);
}
