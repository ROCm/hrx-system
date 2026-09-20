// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/view_abi.h"

#include <limits.h>
#include <string.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/contracts/scalar_lower_rules.h"

typedef struct loom_x86_view_abi_state_t {
  // Function-local source value domain indexing |plans_by_value_ordinal|.
  const loom_local_value_domain_t* value_domain;
  // Sparse payload pointers indexed densely by local value ordinal. NULL until
  // the function contains a call carrying a view operand.
  const loom_low_source_memory_access_plan_t** plans_by_value_ordinal;
  // Terminal observer infrastructure failure transferred from |end|.
  iree_status_t status;
  // Descriptor that adds two GPR64 address components.
  const loom_low_descriptor_t* lea_add_descriptor;
  // Descriptor that adds a signed 32-bit displacement to a GPR64 address.
  const loom_low_descriptor_t* lea_disp_descriptor;
  // Descriptor that materializes an arbitrary 64-bit address component.
  const loom_low_descriptor_t* mov_imm64_descriptor;
  // Interned lea displacement attribute name.
  loom_string_id_t disp32_name_id;
  // Interned 64-bit immediate attribute name.
  loom_string_id_t imm64_name_id;
} loom_x86_view_abi_state_t;

static int loom_x86_view_abi_state_key;

// A non-NULL plan-table sentinel for a view whose address diagnostic was
// already emitted during source traversal.
static const loom_low_source_memory_access_plan_t
    loom_x86_rejected_view_abi_plan;

static iree_status_t loom_x86_view_abi_state_for_context(
    loom_low_lower_context_t* context, loom_x86_view_abi_state_t** out_state) {
  *out_state = NULL;
  return loom_low_lower_get_or_allocate_target_state(
      context, &loom_x86_view_abi_state_key, sizeof(**out_state),
      (void**)out_state);
}

static bool loom_x86_view_abi_call_operands(const loom_op_t* source_op,
                                            loom_value_slice_t* out_operands) {
  *out_operands = (loom_value_slice_t){0};
  if (loom_func_call_isa(source_op)) {
    *out_operands = loom_func_call_operands(source_op);
    return true;
  }
  if (loom_low_invoke_isa(source_op)) {
    *out_operands = loom_low_invoke_operands(source_op);
    return true;
  }
  return false;
}

static iree_status_t loom_x86_view_abi_observer_begin(
    void* user_data, loom_low_lower_context_t* context,
    void** out_observer_state) {
  (void)user_data;
  *out_observer_state = NULL;
  loom_x86_view_abi_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_view_abi_state_for_context(context, &state));
  *state = (loom_x86_view_abi_state_t){
      .value_domain = loom_low_lower_context_value_domain(context),
      .status = iree_ok_status(),
      .disp32_name_id = LOOM_STRING_ID_INVALID,
      .imm64_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_observer_state = state;
  return iree_ok_status();
}

static iree_status_t loom_x86_view_abi_retain_plan(
    loom_x86_view_abi_state_t* state, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(state->value_domain, source_value_id);
  if (state->plans_by_value_ordinal == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, state->value_domain->value_count,
        sizeof(*state->plans_by_value_ordinal),
        (void**)&state->plans_by_value_ordinal));
    memset(state->plans_by_value_ordinal, 0,
           state->value_domain->value_count *
               sizeof(*state->plans_by_value_ordinal));
  }
  if (state->plans_by_value_ordinal[value_ordinal] != NULL) {
    return iree_ok_status();
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_low_source_memory_access_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, 1, sizeof(*plan), (void**)&plan));
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build_view_address(
          view_regions, source_value_id, plan, &diagnostic)) {
    state->plans_by_value_ordinal[value_ordinal] =
        &loom_x86_rejected_view_abi_plan;
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_source_memory_access_rejection_key(
            diagnostic.rejection_bits)),
    };
    return loom_low_lower_emit_target_context_error(context, source_op,
                                                    LOOM_ERR_TARGET_032, params,
                                                    IREE_ARRAYSIZE(params));
  }

  state->plans_by_value_ordinal[value_ordinal] = plan;
  loom_low_lower_require_source_memory_offset_storage(context, plan);
  return iree_ok_status();
}

static void loom_x86_view_abi_observer_observe(
    void* observer_state, loom_low_lower_context_t* context,
    const loom_op_t* source_op) {
  loom_x86_view_abi_state_t* state = (loom_x86_view_abi_state_t*)observer_state;
  if (!iree_status_is_ok(state->status)) {
    return;
  }
  loom_value_slice_t operands = {0};
  if (!loom_x86_view_abi_call_operands(source_op, &operands)) {
    return;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < operands.count; ++i) {
    const loom_value_id_t source_value_id = operands.values[i];
    if (!loom_type_is_view(loom_module_value_type(module, source_value_id))) {
      continue;
    }
    state->status = loom_x86_view_abi_retain_plan(state, context, source_op,
                                                  source_value_id);
    if (!iree_status_is_ok(state->status) ||
        loom_low_lower_context_should_stop(context)) {
      return;
    }
  }
}

static iree_status_t loom_x86_view_abi_observer_end(
    void* observer_state, loom_low_lower_context_t* context) {
  (void)context;
  loom_x86_view_abi_state_t* state = (loom_x86_view_abi_state_t*)observer_state;
  iree_status_t status = state->status;
  state->status = iree_ok_status();
  return status;
}

const loom_low_lower_source_plan_observer_t
    loom_x86_view_abi_source_plan_observer = {
        .begin = loom_x86_view_abi_observer_begin,
        .observe = loom_x86_view_abi_observer_observe,
        .end = loom_x86_view_abi_observer_end,
        .user_data = NULL,
};

static const loom_low_descriptor_t* loom_x86_view_abi_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  const uint32_t ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  IREE_ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
  IREE_ASSERT(descriptor != NULL);
  return descriptor;
}

static void loom_x86_view_abi_resolve_descriptors(
    loom_low_lower_context_t* context, loom_x86_view_abi_state_t* state) {
  if (state->lea_add_descriptor != NULL) {
    return;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  state->lea_add_descriptor = loom_x86_view_abi_descriptor(
      descriptor_set, IREE_SV("x86.scalar.lea.add.gpr64"));
  state->lea_disp_descriptor = loom_x86_view_abi_descriptor(
      descriptor_set, IREE_SV("x86.scalar.lea.disp.gpr64"));
  state->mov_imm64_descriptor = loom_x86_view_abi_descriptor(
      descriptor_set, IREE_SV("x86.scalar.movimm.gpr64"));
}

static iree_status_t loom_x86_view_abi_emit_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t resolved = {
      .descriptor = descriptor,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved, operands, operand_count, attrs, &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_x86_view_abi_emit_add(
    loom_low_lower_context_t* context, loom_x86_view_abi_state_t* state,
    const loom_op_t* source_op, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_value_id_t* out_value_id) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_x86_view_abi_emit_op(
      context, source_op, state->lea_add_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), result_type,
      out_value_id);
}

static iree_status_t loom_x86_view_abi_emit_static_offset(
    loom_low_lower_context_t* context, loom_x86_view_abi_state_t* state,
    const loom_op_t* source_op, loom_value_id_t base, int64_t byte_offset,
    loom_type_t result_type, loom_value_id_t* out_value_id) {
  loom_module_t* module = loom_low_lower_context_module(context);
  if (byte_offset >= INT32_MIN && byte_offset <= INT32_MAX) {
    if (state->disp32_name_id == LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("disp32"),
                                                     &state->disp32_name_id));
    }
    const loom_named_attr_t attr = {
        .name_id = state->disp32_name_id,
        .value = loom_attr_i64(byte_offset),
    };
    return loom_x86_view_abi_emit_op(
        context, source_op, state->lea_disp_descriptor, &base, 1,
        loom_make_named_attr_slice(&attr, 1), result_type, out_value_id);
  }

  if (state->imm64_name_id == LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("imm64"),
                                                   &state->imm64_name_id));
  }
  const loom_named_attr_t attr = {
      .name_id = state->imm64_name_id,
      .value = loom_attr_i64(byte_offset),
  };
  const loom_low_lower_resolved_descriptor_t resolved = {
      .descriptor = state->mov_imm64_descriptor,
  };
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &resolved, loom_make_named_attr_slice(&attr, 1), result_type,
      source_op->location, &const_op));
  return loom_x86_view_abi_emit_add(context, state, source_op, base,
                                    loom_low_const_result(const_op),
                                    result_type, out_value_id);
}

iree_status_t loom_x86_materialize_view_abi_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id) {
  (void)user_data;
  (void)operand_index;
  *out_low_value_id = low_value_id;

  loom_value_slice_t call_operands = {0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_x86_view_abi_call_operands(source_op, &call_operands) ||
      !loom_type_is_view(loom_module_value_type(module, source_value_id))) {
    return iree_ok_status();
  }

  loom_x86_view_abi_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_view_abi_state_for_context(context, &state));
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(state->value_domain, source_value_id);
  IREE_ASSERT(state->plans_by_value_ordinal != NULL);
  const loom_low_source_memory_access_plan_t* plan =
      state->plans_by_value_ordinal[value_ordinal];
  IREE_ASSERT(plan != NULL && plan != &loom_x86_rejected_view_abi_plan,
              "x86 call view operand has no retained address plan");
  loom_x86_view_abi_resolve_descriptors(context, state);

  loom_value_id_t address = low_value_id;
  if (plan->dynamic_term_count != 0) {
    IREE_ASSERT_EQ(loom_x86_scalar_lower_rule_set
                       .source_memory_byte_offset_materializer_count,
                   1);
    const loom_low_lower_source_memory_byte_offset_materializer_t*
        materializer = loom_x86_scalar_lower_rule_set
                           .source_memory_byte_offset_materializers;
    loom_value_id_t dynamic_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_source_memory_byte_offset(
        context, &loom_x86_scalar_lower_rule_set, source_op, materializer, plan,
        &dynamic_offset));
    IREE_RETURN_IF_ERROR(loom_x86_view_abi_emit_add(
        context, state, source_op, address, dynamic_offset, required_low_type,
        &address));
  }
  if (plan->static_byte_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_x86_view_abi_emit_static_offset(
        context, state, source_op, address, plan->static_byte_offset,
        required_low_type, &address));
  }
  *out_low_value_id = address;
  return iree_ok_status();
}
