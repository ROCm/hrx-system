// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_model.h"

#include "loom/codegen/low/function.h"

iree_status_t loom_low_function_model_initialize(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection, iree_diagnostic_emitter_t emitter,
    loom_low_function_model_flags_t flags, iree_arena_allocator_t* arena,
    loom_low_function_model_t* out_model) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }
  *out_model = (loom_low_function_model_t){
      .module = module,
      .function_op = low_func_op,
      .body = loom_low_function_body((loom_op_t*)low_func_op),
  };
  IREE_ASSERT(out_model->body != NULL);

  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, descriptor_registry, target_selection, emitter,
      &out_model->target));
  if (out_model->target.descriptor_set == NULL) {
    out_model->error_count = 1;
    return iree_ok_status();
  }

  if (iree_any_bit_set(flags, LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE)) {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
        module, out_model->body, arena, &out_model->value_domain));
  } else {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
        module, out_model->body, arena, &out_model->value_domain));
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(out_model->body, block) {
    out_model->node_count += block->op_count;
  }
  return iree_ok_status();
}

void loom_low_function_model_deinitialize(loom_low_function_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  loom_local_value_domain_release(&model->value_domain);
  *model = (loom_low_function_model_t){0};
}
