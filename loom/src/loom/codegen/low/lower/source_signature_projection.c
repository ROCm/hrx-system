// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_signature_projection.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry.h"

struct loom_low_source_signature_projection_plan_t {
  // Source callable retaining the logical object signature.
  loom_func_like_t function;
  // Target callback copied from the selected immutable lowering policy.
  loom_low_lower_source_signature_callback_t callback;
  // Flat logical argument carriers used inside source-to-Low.
  const loom_type_t* logical_argument_types;
  // Number of flat logical argument carriers.
  uint16_t logical_argument_count;
  // Flat logical result carriers used inside source-to-Low.
  const loom_type_t* logical_result_types;
  // Number of flat logical result carriers.
  uint16_t logical_result_count;
  // Target-independent source object and terminal leaf layout.
  loom_c_abi_signature_layout_t signature_layout;
  // Physical target signature and opaque emission recipe.
  loom_low_lower_source_signature_target_plan_t target;
  // Canonical target-owned placement/layout attributes.
  loom_named_attr_slice_t abi_layout;
};

bool loom_low_source_signature_projection_required(loom_func_like_t function) {
  loom_func_c_abi_signature_t signature;
  return loom_func_c_abi_signature_cast(
      loom_func_like_source_signature(function), &signature);
}

iree_status_t loom_low_source_signature_projection_index_build(
    const loom_module_t* module,
    const loom_low_source_signature_projection_plan_t* const* plans,
    iree_host_size_t plan_count, iree_arena_allocator_t* arena,
    loom_low_source_signature_projection_index_t* out_index) {
  *out_index = (loom_low_source_signature_projection_index_t){0};
  if (plan_count == 0) {
    return iree_ok_status();
  }
  if (plan_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source signature projection count exceeds 65535");
  }
  uint16_t* ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*ordinals), (void**)&ordinals));
  memset(ordinals, 0, module->symbols.count * sizeof(*ordinals));
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_symbol_ref_t callee = loom_func_like_callee(plans[i]->function);
    IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
    IREE_ASSERT_EQ(ordinals[callee.symbol_id], 0);
    ordinals[callee.symbol_id] = (uint16_t)(i + 1);
  }
  *out_index = (loom_low_source_signature_projection_index_t){
      .plans = plans,
      .ordinals = ordinals,
      .symbol_count = module->symbols.count,
  };
  return iree_ok_status();
}

const loom_low_source_signature_projection_plan_t*
loom_low_source_signature_projection_index_find(
    const loom_low_source_signature_projection_index_t* index,
    loom_symbol_id_t symbol_id) {
  const uint16_t ordinal = index != NULL && symbol_id < index->symbol_count
                               ? index->ordinals[symbol_id]
                               : 0;
  return ordinal != 0 ? index->plans[ordinal - 1] : NULL;
}

static bool loom_low_source_signature_function_attr_present(
    loom_func_like_t function, uint8_t attr_index) {
  return attr_index != LOOM_ATTR_INDEX_NONE &&
         !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

static loom_target_abi_kind_t loom_low_source_signature_function_abi(
    loom_low_lower_context_t* context) {
  const loom_func_like_t function = context->source_function;
  if (loom_low_source_signature_function_attr_present(
          function, function.vtable->abi_attr_index)) {
    return (loom_target_abi_kind_t)loom_func_like_abi(function);
  }
  const loom_target_abi_kind_t target_abi =
      loom_low_lower_context_bundle(context)->export_plan->abi_kind;
  if (target_abi != LOOM_TARGET_ABI_UNKNOWN) {
    return target_abi;
  }
  // Internal and imported callees inherit execution target facts without the
  // artifact root's export overlay. A native target that owns a complete source
  // ABI projection still gives those retained C boundaries object semantics.
  return context->policy->import_decl_kind ==
                     LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE &&
                 context->policy->source_signature.plan != NULL &&
                 context->policy->source_signature.emit != NULL
             ? LOOM_TARGET_ABI_OBJECT_FUNCTION
             : LOOM_TARGET_ABI_UNKNOWN;
}

static iree_status_t loom_low_source_signature_emit_unavailable(
    loom_low_lower_context_t* context, iree_string_view_t reason) {
  return loom_low_lower_emit_source_signature_unsupported(
      context, context->source_function.op, reason);
}

static iree_status_t loom_low_source_signature_map_logical_types(
    loom_low_lower_context_t* context, const loom_value_id_t* values,
    uint16_t count, iree_arena_allocator_t* arena,
    const loom_type_t** out_types) {
  *out_types = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*types), (void**)&types));
  for (uint16_t i = 0; i < count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(context->module, values[i]);
    IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
        context, context->source_function.op, source_type, &types[i]));
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_low_source_signature_plan_layout(
    loom_low_lower_context_t* context,
    const loom_low_lower_source_signature_target_plan_t* target,
    iree_arena_allocator_t* scratch_arena,
    loom_named_attr_slice_t* out_layout) {
  *out_layout = loom_named_attr_slice_empty();
  loom_named_attr_slice_t layout = target->abi_layout;
  if (layout.count == 0) {
    if (context->policy->map_abi_layout.fn == NULL) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(context->policy->map_abi_layout.fn(
        context->policy->map_abi_layout.user_data, context,
        LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC, target->argument_types,
        target->argument_count, target->result_types, target->result_count,
        scratch_arena, &layout));
    if (context->result->error_count != 0) {
      return iree_ok_status();
    }
  }
  loom_attribute_t canonical_layout = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      context->module, layout, &canonical_layout));
  *out_layout = loom_attr_as_dict(canonical_layout);
  return iree_ok_status();
}

iree_status_t loom_low_plan_source_signature_projection(
    loom_module_t* module, loom_func_like_t function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_result_t* out_result,
    const loom_low_source_signature_projection_plan_t** out_plan) {
  *out_plan = NULL;
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  loom_func_c_abi_signature_t signature;
  if (!loom_func_c_abi_signature_cast(loom_func_like_source_signature(function),
                                      &signature)) {
    return iree_ok_status();
  }

  loom_target_low_source_binding_t binding = {0};
  IREE_RETURN_IF_ERROR(loom_target_low_source_binding_select(
      options->descriptor_registry,
      loom_target_facts_bundle(options->target_facts), &binding));

  loom_low_lowering_frame_t frame = {0};
  iree_arena_allocator_t function_arena;
  iree_arena_initialize(module->arena.block_pool, &function_arena);
  loom_low_lower_context_t context = {
      .function_arena = &function_arena,
      .lowering = &frame,
      .module = module,
      .source_function = function,
      .options = options,
      .policy = options->policy,
      .descriptor_set = binding.descriptor_set,
      .schedule_model = binding.schedule_model,
      .result = out_result,
  };
  out_result->descriptor_set = binding.descriptor_set;
  iree_arena_initialize(module->arena.block_pool, &context.analysis_arena);
  loom_condition_query_initialize(module, /*value_domain=*/NULL,
                                  &context.analysis_arena,
                                  &context.condition_query);

  // Source object contracts survive independently of the selected output ABI.
  // Kernel, VM, and other target ABIs retain their existing boundary planners;
  // only native object functions use the C calling-convention projection.
  if (loom_low_source_signature_function_abi(&context) !=
      LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    iree_arena_deinitialize(&context.analysis_arena);
    iree_arena_deinitialize(&function_arena);
    return iree_ok_status();
  }

  loom_low_source_signature_projection_plan_t* plan = NULL;
  iree_status_t status =
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan);
  if (iree_status_is_ok(status)) {
    memset(plan, 0, sizeof(*plan));
    plan->function = function;
    plan->callback = options->policy->source_signature;
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (iree_status_is_ok(status)) {
    plan->logical_argument_count = argument_count;
    status = loom_low_source_signature_map_logical_types(
        &context, arguments, argument_count, arena,
        &plan->logical_argument_types);
  }
  if (iree_status_is_ok(status)) {
    plan->logical_result_count = function.op->result_count;
    status = loom_low_source_signature_map_logical_types(
        &context, loom_op_const_results(function.op), function.op->result_count,
        arena, &plan->logical_result_types);
  }

  if (iree_status_is_ok(status) && out_result->error_count == 0 &&
      (plan->callback.plan == NULL || plan->callback.emit == NULL)) {
    status = loom_low_source_signature_emit_unavailable(
        &context, IREE_SV("target has no C source signature projection"));
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    status = loom_c_abi_signature_layout_build(&signature, &function_arena,
                                               arena, &plan->signature_layout);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    IREE_ASSERT_EQ(plan->signature_layout.argument_logical_count,
                   plan->logical_argument_count);
    IREE_ASSERT_EQ(plan->signature_layout.result_logical_count,
                   plan->logical_result_count);
    status = plan->callback.plan(
        plan->callback.user_data, &context, &plan->signature_layout,
        plan->logical_argument_types, plan->logical_argument_count,
        plan->logical_result_types, plan->logical_result_count, &function_arena,
        arena, &plan->target);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    IREE_ASSERT(plan->target.argument_count == 0 ||
                plan->target.argument_types != NULL);
    IREE_ASSERT(plan->target.result_count == 0 ||
                plan->target.result_types != NULL);
    IREE_ASSERT(plan->target.tied_result_count == 0 ||
                plan->target.tied_results != NULL);
    status = loom_low_source_signature_plan_layout(
        &context, &plan->target, &function_arena, &plan->abi_layout);
  }
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    *out_plan = plan;
  }
  iree_arena_deinitialize(&context.analysis_arena);
  iree_arena_deinitialize(&function_arena);
  return status;
}

const loom_type_t* loom_low_source_signature_projection_logical_argument_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count) {
  *out_count = plan->logical_argument_count;
  return plan->logical_argument_types;
}

const loom_type_t* loom_low_source_signature_projection_logical_result_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count) {
  *out_count = plan->logical_result_count;
  return plan->logical_result_types;
}

const loom_type_t* loom_low_source_signature_projection_physical_argument_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count) {
  *out_count = plan->target.argument_count;
  return plan->target.argument_types;
}

const loom_type_t* loom_low_source_signature_projection_physical_result_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count) {
  *out_count = plan->target.result_count;
  return plan->target.result_types;
}

const loom_tied_result_t* loom_low_source_signature_projection_tied_results(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count) {
  *out_count = plan->target.tied_result_count;
  return plan->target.tied_results;
}

loom_named_attr_slice_t loom_low_source_signature_projection_abi_layout(
    const loom_low_source_signature_projection_plan_t* plan) {
  return plan->abi_layout;
}

iree_status_t loom_low_source_signature_projection_emit(
    const loom_low_source_signature_projection_plan_t* plan,
    loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const loom_value_id_t* input_values,
    uint16_t input_count, loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state) {
  return plan->callback.emit(plan->callback.user_data, context, phase,
                             source_op, plan->target.target_data, input_values,
                             input_count, output_values, output_count,
                             state_arena, inout_state);
}
