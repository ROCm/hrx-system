// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_model.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/storage_identity.h"

iree_status_t loom_low_function_context_initialize(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_resolved_target_t* resolved_target,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, loom_low_function_model_flags_t flags,
    iree_host_size_t additional_value_capacity, iree_arena_allocator_t* arena,
    loom_low_function_context_t* out_context) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }
  *out_context = (loom_low_function_context_t){
      .module = module,
      .function_op = low_func_op,
      .body = loom_low_function_body((loom_op_t*)low_func_op),
  };
  IREE_ASSERT(out_context->body != NULL);

  if (resolved_target != NULL) {
    out_context->target = *resolved_target;
  } else {
    loom_symbol_fact_table_t symbol_facts = {0};
    loom_symbol_fact_table_initialize(&symbol_facts, arena);
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, &symbol_facts, low_func_op, function_target_facts,
        descriptor_registry, emitter, &out_context->target));
  }
  if (out_context->target.descriptor_set == NULL) {
    out_context->error_count = 1;
    return iree_ok_status();
  }

  if (iree_any_bit_set(flags, LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE)) {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
        module, out_context->body, arena, &out_context->value_domain));
  } else {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
        module, out_context->body, arena, &out_context->value_domain));
  }
  iree_status_t status = loom_low_storage_identity_build(
      &out_context->value_domain, additional_value_capacity, arena,
      &out_context->storage_origins);
  if (iree_status_is_ok(status)) {
    status = loom_low_function_requirements_build(
        module, out_context->body, arena, &out_context->requirements);
  }
  if (!iree_status_is_ok(status)) {
    loom_low_function_context_deinitialize(out_context);
  }
  return status;
}

void loom_low_function_context_deinitialize(
    loom_low_function_context_t* context) {
  loom_local_value_domain_release(&context->value_domain);
  *context = (loom_low_function_context_t){0};
}

iree_status_t loom_low_function_model_build(
    loom_low_function_context_t* context, iree_arena_allocator_t* arena,
    loom_low_function_model_t* out_model) {
  *out_model = (loom_low_function_model_t){.context = *context};
  *context = (loom_low_function_context_t){0};
  const loom_low_function_context_t* prepared = &out_model->context;
  if (prepared->error_count != 0) {
    return iree_ok_status();
  }
  out_model->cfg_graph = (loom_cfg_graph_t){
      .module = prepared->module,
      .region = prepared->body,
      .block_count = prepared->body->block_count,
  };
  iree_status_t status = iree_ok_status();
  if (prepared->body->block_count > 1 ||
      iree_any_bit_set(prepared->body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    status = loom_cfg_graph_build(prepared->module, prepared->body, arena,
                                  &out_model->cfg_graph);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cfg_loop_forest_build(&out_model->cfg_graph, arena,
                                        &out_model->loop_forest);
  }
  if (iree_status_is_ok(status)) {
    status = loom_liveness_dataflow_analyze(&prepared->value_domain,
                                            &out_model->cfg_graph, arena,
                                            &out_model->liveness_dataflow);
  }
  return status;
}

iree_status_t loom_low_function_model_initialize(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, loom_low_function_model_flags_t flags,
    iree_arena_allocator_t* arena, loom_low_function_model_t* out_model) {
  *out_model = (loom_low_function_model_t){0};
  loom_low_function_context_t context = {0};
  iree_status_t status = loom_low_function_context_initialize(
      module, low_func_op, NULL, function_target_facts, descriptor_registry,
      emitter, flags, /*additional_value_capacity=*/0, arena, &context);
  if (iree_status_is_ok(status)) {
    status = loom_low_function_model_build(&context, arena, out_model);
  }
  loom_low_function_context_deinitialize(&context);
  return status;
}

void loom_low_function_model_deinitialize(loom_low_function_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  loom_low_function_context_deinitialize(&model->context);
  *model = (loom_low_function_model_t){0};
}
