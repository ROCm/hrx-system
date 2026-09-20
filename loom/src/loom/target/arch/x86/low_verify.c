// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/low_verify.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/x86/error_catalog.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/sysv_abi.h"

typedef enum loom_x86_low_layout_state_e {
  LOOM_X86_LOW_LAYOUT_STATE_UNPARSED = 0,
  LOOM_X86_LOW_LAYOUT_STATE_VALID = 1,
  LOOM_X86_LOW_LAYOUT_STATE_INVALID = 2,
} loom_x86_low_layout_state_t;

typedef struct loom_x86_low_layout_record_t {
  // Descriptor set used to classify the function signature.
  const loom_low_descriptor_set_t* descriptor_set;
  // Parsed canonical layout borrowing module attribute storage.
  loom_x86_sysv_abi_layout_t layout;
  // Parse state for this symbol.
  loom_x86_low_layout_state_t state;
} loom_x86_low_layout_record_t;

typedef struct loom_x86_low_verify_module_state_t {
  // Module being verified.
  const loom_module_t* module;
  // Dense layout cache indexed by module-local symbol ID.
  loom_x86_low_layout_record_t* layouts;
} loom_x86_low_verify_module_state_t;

typedef struct loom_x86_low_verify_state_t {
  // Shared module-level cache.
  loom_x86_low_verify_module_state_t* module_state;
  // Resolved x86 target for this function.
  const loom_low_resolved_target_t* target;
  // Low function definition being verified.
  const loom_op_t* function_op;
  // Borrowed function symbol name for diagnostics.
  iree_string_view_t function_name;
} loom_x86_low_verify_state_t;

static iree_string_view_t loom_x86_low_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT_EQ(symbol_ref.module_id, 0);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static const loom_op_t* loom_x86_low_function_op(const loom_module_t* module,
                                                 loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT_EQ(symbol_ref.module_id, 0);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_op_t* function_op =
      module->symbols.entries[symbol_ref.symbol_id].defining_op;
  IREE_ASSERT(loom_low_func_def_isa(function_op) ||
              loom_low_func_decl_isa(function_op));
  return function_op;
}

static loom_named_attr_slice_t loom_x86_low_function_layout_attrs(
    const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_abi_layout(function_op);
  }
  IREE_ASSERT(loom_low_func_decl_isa(function_op));
  return loom_low_func_decl_abi_layout(function_op);
}

static iree_status_t loom_x86_low_emit_invalid_layout(
    loom_low_verify_context_t* context, const loom_op_t* diagnostic_op,
    iree_string_view_t function_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_name),
  };
  return loom_low_verify_context_emit(context, diagnostic_op, LOOM_ERR_X86_001,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_x86_low_resolve_layout(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    loom_symbol_ref_t function_ref, const loom_op_t* diagnostic_op,
    const loom_x86_sysv_abi_layout_t** out_layout) {
  *out_layout = NULL;
  loom_x86_low_layout_record_t* record =
      &state->module_state->layouts[function_ref.symbol_id];
  if (record->state == LOOM_X86_LOW_LAYOUT_STATE_VALID) {
    IREE_ASSERT_EQ(record->descriptor_set, state->target->descriptor_set);
    *out_layout = &record->layout;
    return iree_ok_status();
  }
  if (record->state == LOOM_X86_LOW_LAYOUT_STATE_INVALID) {
    return iree_ok_status();
  }

  const loom_op_t* function_op =
      loom_x86_low_function_op(state->module_state->module, function_ref);
  iree_status_t status = loom_x86_sysv_abi_function_layout_parse(
      state->module_state->module, state->target->descriptor_set, function_op,
      loom_low_verify_context_arena(context), &record->layout);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    if (status_code != IREE_STATUS_INVALID_ARGUMENT &&
        status_code != IREE_STATUS_FAILED_PRECONDITION &&
        status_code != IREE_STATUS_UNIMPLEMENTED) {
      return status;
    }
    record->state = LOOM_X86_LOW_LAYOUT_STATE_INVALID;
    iree_status_free(status);
    return loom_x86_low_emit_invalid_layout(
        context, diagnostic_op,
        loom_x86_low_symbol_name(state->module_state->module, function_ref));
  }

  record->descriptor_set = state->target->descriptor_set;
  record->state = LOOM_X86_LOW_LAYOUT_STATE_VALID;
  *out_layout = &record->layout;
  return iree_ok_status();
}

static iree_status_t loom_x86_low_emit_placement_mismatch(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op, uint32_t ordinal, bool actual_is_stack,
    bool expected_is_stack) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_operation_name(state->module_state->module, op)),
      loom_param_u32(ordinal),
      loom_param_string(actual_is_stack ? IREE_SV("stack memory")
                                        : IREE_SV("a register")),
      loom_param_string(expected_is_stack ? IREE_SV("stack memory")
                                          : IREE_SV("a register")),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_X86_002, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_x86_low_emit_offset_mismatch(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op, uint32_t ordinal, int64_t actual_offset,
    uint32_t expected_offset) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_operation_name(state->module_state->module, op)),
      loom_param_u32(ordinal),
      loom_param_i64(actual_offset),
      loom_param_u32(expected_offset),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_X86_003, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_x86_low_verify_stack_argument(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op, loom_symbol_ref_t function_ref, uint32_t ordinal,
    int64_t byte_offset) {
  const loom_x86_sysv_abi_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      loom_x86_low_resolve_layout(context, state, function_ref, op, &layout));
  if (layout == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(ordinal, layout->argument_count);
  const int64_t location = layout->argument_locations[ordinal];
  if (loom_x86_sysv_abi_location_is_register(location)) {
    return loom_x86_low_emit_placement_mismatch(context, state, op, ordinal,
                                                true, false);
  }
  const uint32_t expected_offset =
      loom_x86_sysv_abi_stack_location_offset(location);
  return byte_offset == (int64_t)expected_offset
             ? iree_ok_status()
             : loom_x86_low_emit_offset_mismatch(context, state, op, ordinal,
                                                 byte_offset, expected_offset);
}

static iree_status_t loom_x86_low_verify_stack_arg(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op) {
  const loom_symbol_ref_t function_ref =
      loom_func_like_callee(loom_func_like_const_cast(
          state->module_state->module, state->function_op));
  return loom_x86_low_verify_stack_argument(
      context, state, op, function_ref,
      (uint32_t)loom_low_func_stack_arg_ordinal(op),
      loom_low_func_stack_arg_byte_offset(op));
}

static iree_status_t loom_x86_low_verify_call_arg(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op) {
  return loom_x86_low_verify_stack_argument(
      context, state, op, loom_low_func_call_arg_callee(op),
      (uint32_t)loom_low_func_call_arg_ordinal(op),
      loom_low_func_call_arg_byte_offset(op));
}

static iree_status_t loom_x86_low_verify_call(
    loom_low_verify_context_t* context, loom_x86_low_verify_state_t* state,
    const loom_op_t* op) {
  const loom_value_slice_t stack_args = loom_low_func_call_stack_args(op);
  if (stack_args.count == 0) {
    return iree_ok_status();
  }

  const loom_x86_sysv_abi_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_low_resolve_layout(
      context, state, loom_low_func_call_callee(op), op, &layout));
  if (layout == NULL) {
    return iree_ok_status();
  }

  uint16_t stack_index = 0;
  for (uint32_t ordinal = 0; ordinal < layout->argument_count; ++ordinal) {
    bool actual_is_stack = false;
    if (stack_index < stack_args.count) {
      const loom_value_t* token = loom_module_value(
          state->module_state->module, stack_args.values[stack_index]);
      const loom_op_t* call_arg_op = loom_value_def_op(token);
      IREE_ASSERT(loom_low_func_call_arg_isa(call_arg_op));
      actual_is_stack =
          loom_low_func_call_arg_ordinal(call_arg_op) == (int64_t)ordinal;
    }
    const bool expected_is_stack = !loom_x86_sysv_abi_location_is_register(
        layout->argument_locations[ordinal]);
    if (actual_is_stack != expected_is_stack) {
      return loom_x86_low_emit_placement_mismatch(
          context, state, op, ordinal, actual_is_stack, expected_is_stack);
    }
    stack_index += actual_is_stack;
  }
  IREE_ASSERT_EQ(stack_index, stack_args.count);
  return iree_ok_status();
}

static iree_status_t loom_x86_low_begin_module(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_module_context_t* context, void** out_provider_state) {
  (void)provider;
  const loom_module_t* module = loom_low_verify_module_context_module(context);
  loom_x86_low_verify_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(loom_low_verify_module_context_arena(context),
                          sizeof(*state), (void**)&state));
  *state = (loom_x86_low_verify_module_state_t){
      .module = module,
  };
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        loom_low_verify_module_context_arena(context), module->symbols.count,
        sizeof(*state->layouts), (void**)&state->layouts));
    memset(state->layouts, 0, module->symbols.count * sizeof(*state->layouts));
  }
  *out_provider_state = state;
  return iree_ok_status();
}

static iree_status_t loom_x86_low_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (target->descriptor_set == NULL || target->target_facts == NULL ||
      target->target_facts->fact_type != &loom_x86_target_fact_type) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_verify_context_module(context);
  const loom_op_t* function_op = loom_low_verify_context_function_op(context);
  if (!loom_low_func_def_isa(function_op)) {
    return iree_ok_status();
  }
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  loom_x86_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  *state = (loom_x86_low_verify_state_t){
      .module_state = (loom_x86_low_verify_module_state_t*)
          loom_low_verify_context_provider_module_state(context),
      .target = target,
      .function_op = function_op,
      .function_name =
          loom_x86_low_symbol_name(module, loom_func_like_callee(function)),
  };
  *out_provider_state = state;

  if (loom_x86_low_function_layout_attrs(function_op).count != 0) {
    const loom_x86_sysv_abi_layout_t* layout = NULL;
    IREE_RETURN_IF_ERROR(loom_x86_low_resolve_layout(
        context, state, loom_func_like_callee(function), function_op, &layout));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_descriptor_packet_t* packet) {
  (void)provider;
  loom_x86_low_verify_state_t* state =
      (loom_x86_low_verify_state_t*)provider_state;
  if (state == NULL || loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  if (loom_low_func_stack_arg_isa(packet->op)) {
    return loom_x86_low_verify_stack_arg(context, state, packet->op);
  }
  if (loom_low_func_call_arg_isa(packet->op)) {
    return loom_x86_low_verify_call_arg(context, state, packet->op);
  }
  if (loom_low_func_call_isa(packet->op)) {
    return loom_x86_low_verify_call(context, state, packet->op);
  }
  return iree_ok_status();
}

const loom_low_verify_provider_t loom_x86_low_verify_provider = {
    .name = IREE_SVL("x86"),
    .begin_module = loom_x86_low_begin_module,
    .begin_function = loom_x86_low_begin_function,
    .verify_op = loom_x86_low_verify_op,
};
