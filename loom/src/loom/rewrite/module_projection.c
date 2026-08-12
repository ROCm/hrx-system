// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/module_projection.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"

static iree_status_t loom_ir_module_projection_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  const loom_ir_module_projection_t* projection =
      (const loom_ir_module_projection_t*)user_data;
  IREE_ASSERT(source_module == projection->source_module);
  IREE_ASSERT(target_module == projection->target_module);
  IREE_ASSERT(source_ref.module_id == 0);
  IREE_ASSERT(source_ref.symbol_id < projection->target_symbol_count);
  *out_target_ref = projection->target_symbols[source_ref.symbol_id];
  return iree_ok_status();
}

iree_status_t loom_ir_module_projection_initialize(
    const loom_module_t* source_module, loom_module_t* target_module,
    const loom_symbol_ref_t* target_symbols,
    iree_host_size_t target_symbol_count,
    loom_ir_module_projection_t* out_projection) {
  if (!source_module || !target_module || !out_projection) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source, target, and output projection are required");
  }
  *out_projection = (loom_ir_module_projection_t){0};
  if (source_module->context != target_module->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source and target module contexts must match");
  }
  if (target_symbol_count != source_module->symbols.count ||
      (target_symbol_count > 0 && !target_symbols)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module projection has %" PRIhsz
                            " symbols but source module has %" PRIhsz,
                            target_symbol_count, source_module->symbols.count);
  }
  for (iree_host_size_t i = 0; i < target_symbol_count; ++i) {
    const loom_symbol_ref_t target_ref = target_symbols[i];
    if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
        target_ref.symbol_id >= target_module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source symbol %" PRIhsz " has invalid target reference {%u, %u}", i,
          (unsigned)target_ref.module_id, (unsigned)target_ref.symbol_id);
    }
  }

  *out_projection = (loom_ir_module_projection_t){
      .source_module = source_module,
      .target_module = target_module,
      .target_symbols = target_symbols,
      .target_symbol_count = target_symbol_count,
  };
  return iree_ok_status();
}

iree_status_t loom_ir_module_projection_clone(
    loom_ir_module_projection_t* projection,
    iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      projection->source_module, projection->target_module, scratch_arena,
      &(loom_ir_remap_options_t){
          .remap_symbol = loom_ir_remap_symbol_callback_make(
              loom_ir_module_projection_remap_symbol, (void*)projection),
      },
      &projection->remap));
  loom_builder_t builder;
  loom_builder_initialize(
      projection->target_module, &projection->target_module->arena,
      loom_module_block(projection->target_module), &builder);
  return loom_ir_clone_block_ops(
      &builder, loom_region_const_entry_block(projection->source_module->body),
      &projection->remap, /*options=*/NULL);
}
