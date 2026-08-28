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

iree_status_t loom_ir_module_projection_track_operations(
    loom_ir_module_projection_t* projection,
    loom_ir_remap_op_projection_t* entries, iree_host_size_t entry_count) {
  if (projection == NULL || projection->source_module == NULL ||
      projection->target_module == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module projection is not initialized");
  }
  if (projection->remap.source_module != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot select operations after cloning begins");
  }
  if (entry_count != 0 && entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation projection count is non-zero but entries are NULL");
  }
  projection->operations.entries = entries;
  projection->operations.count = entry_count;
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
          .value_map_kind = LOOM_IR_REMAP_VALUE_MAP_SOURCE_INDEXED,
          .op_projection =
              {
                  .entries = projection->operations.entries,
                  .count = projection->operations.count,
              },
      },
      &projection->remap));
  loom_builder_t builder;
  loom_builder_initialize(
      projection->target_module, &projection->target_module->arena,
      loom_module_block(projection->target_module), &builder);
  IREE_RETURN_IF_ERROR(loom_ir_clone_block_ops(
      &builder, loom_region_const_entry_block(projection->source_module->body),
      &projection->remap, /*options=*/NULL));
  if (projection->remap.op_projection.cursor !=
      projection->remap.op_projection.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "module clone observed %zu of %zu selected source operations",
        projection->remap.op_projection.cursor,
        projection->remap.op_projection.count);
  }
  return iree_ok_status();
}

iree_status_t loom_ir_module_clone(
    const loom_module_t* source_module,
    const loom_ir_module_clone_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    iree_allocator_t allocator, loom_ir_module_projection_t* out_projection,
    loom_module_t** out_module) {
  *out_projection = (loom_ir_module_projection_t){0};
  *out_module = NULL;
  const loom_ir_module_clone_options_t clone_options =
      options ? *options : (loom_ir_module_clone_options_t){0};

  iree_host_size_t target_string_capacity = 0;
  iree_host_size_t target_symbol_capacity = 0;
  if (!iree_host_size_checked_add(source_module->strings.count,
                                  clone_options.additional_string_capacity,
                                  &target_string_capacity) ||
      !iree_host_size_checked_add(source_module->symbols.count,
                                  clone_options.additional_symbol_capacity,
                                  &target_symbol_capacity) ||
      target_symbol_capacity > LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "cloned module exceeds the target string or symbol capacity");
  }
  const iree_string_view_t module_name =
      source_module->name_id == LOOM_STRING_ID_INVALID
          ? IREE_SV("module")
          : source_module->strings.entries[source_module->name_id];
  const loom_module_size_hints_t hints = {
      .value_count = 0,
      .string_count = target_string_capacity,
      .type_count = source_module->types.count,
      .encoding_count = source_module->encodings.count,
      .source_count = source_module->sources.count,
      .symbol_count = target_symbol_capacity,
  };
  loom_module_t* target_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(source_module->context, module_name,
                                            block_pool, &hints, allocator,
                                            &target_module));
  target_module->flags = source_module->flags;

  loom_symbol_ref_t* target_symbols = NULL;
  iree_status_t status = iree_ok_status();
  if (source_module->symbols.count != 0) {
    status = iree_arena_allocate_array(
        scratch_arena, source_module->symbols.count, sizeof(*target_symbols),
        (void**)&target_symbols);
  }
  for (loom_symbol_id_t source_symbol_id = 0;
       source_symbol_id < source_module->symbols.count &&
       iree_status_is_ok(status);
       ++source_symbol_id) {
    const loom_symbol_t* source_symbol =
        &source_module->symbols.entries[source_symbol_id];
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(
        target_module, source_module->strings.entries[source_symbol->name_id],
        &target_name_id);
    loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    if (iree_status_is_ok(status)) {
      status = loom_module_add_symbol(target_module, target_name_id,
                                      &target_symbol_id);
    }
    if (iree_status_is_ok(status)) {
      target_module->symbols.entries[target_symbol_id].flags =
          source_symbol->flags;
      target_symbols[source_symbol_id] = (loom_symbol_ref_t){
          .module_id = 0,
          .symbol_id = target_symbol_id,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_module_projection_initialize(
        source_module, target_module, target_symbols,
        source_module->symbols.count, out_projection);
  }
  if (iree_status_is_ok(status) && clone_options.operations.count != 0) {
    status = loom_ir_module_projection_track_operations(
        out_projection, clone_options.operations.entries,
        clone_options.operations.count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_module_projection_clone(out_projection, scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    *out_module = target_module;
    target_module = NULL;
  }
  loom_module_free(target_module);
  return status;
}
