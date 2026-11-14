// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/pipeline_snapshot.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/pass/ops.h"

static iree_string_view_t loom_pass_pipeline_snapshot_normalize_symbol(
    iree_string_view_t symbol) {
  symbol = iree_string_view_trim(symbol);
  while (iree_string_view_starts_with_char(symbol, '@')) {
    symbol = iree_string_view_remove_prefix(symbol, 1);
  }
  return symbol;
}

iree_status_t loom_pass_pipeline_snapshot_initialize(
    const loom_module_t* source_module, iree_string_view_t pipeline_symbol,
    iree_string_view_t identifier, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_pass_pipeline_snapshot_t* out_snapshot) {
  *out_snapshot = (loom_pass_pipeline_snapshot_t){0};
  const iree_string_view_t symbol_name =
      loom_pass_pipeline_snapshot_normalize_symbol(pipeline_symbol);
  const loom_string_id_t name_id =
      loom_module_lookup_string(source_module, symbol_name);
  const loom_symbol_id_t symbol_id =
      name_id != LOOM_STRING_ID_INVALID
          ? loom_module_find_symbol(source_module, name_id)
          : LOOM_SYMBOL_ID_INVALID;
  const loom_op_t* source_op =
      symbol_id != LOOM_SYMBOL_ID_INVALID
          ? source_module->symbols.entries[symbol_id].defining_op
          : NULL;
  if (source_op == NULL || !loom_pass_pipeline_isa(source_op)) {
    const bool empty_name = iree_string_view_is_empty(symbol_name);
    const loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
        loom_param_string(source_op ? loom_op_vtable_name(loom_op_vtable(
                                          source_module, source_op))
                                    : iree_string_view_empty()),
        loom_param_string(IREE_SV("pass.pipeline")),
    };
    const loom_diagnostic_emission_t emission = {
        .module = source_module,
        .op = source_op,
        .error = empty_name  ? LOOM_ERR_SYMBOL_018
                 : source_op ? LOOM_ERR_SYMBOL_003
                             : LOOM_ERR_SYMBOL_017,
        .params = params,
        .param_count = empty_name  ? 0
                       : source_op ? 3
                                   : 1,
    };
    return iree_diagnostic_emit(diagnostic_emitter, &emission);
  }

  const loom_module_t* source_modules[] = {source_module};
  const iree_string_view_t root_symbols[] = {symbol_name};
  loom_symbol_ref_t target_root = loom_symbol_ref_null();
  iree_status_t status = loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = identifier,
          .diagnostic_emitter = diagnostic_emitter,
          .root_target_symbols = {.values = &target_root, .count = 1},
          .root_symbols =
              {
                  .count = IREE_ARRAYSIZE(root_symbols),
                  .values = root_symbols,
              },
      },
      block_pool, allocator, &out_snapshot->module);
  if (iree_status_is_ok(status) && out_snapshot->module != NULL) {
    out_snapshot->pipeline_op =
        out_snapshot->module->symbols.entries[target_root.symbol_id]
            .defining_op;
  }
  if (!iree_status_is_ok(status)) {
    loom_pass_pipeline_snapshot_deinitialize(out_snapshot);
  }
  return status;
}

void loom_pass_pipeline_snapshot_deinitialize(
    loom_pass_pipeline_snapshot_t* snapshot) {
  if (snapshot == NULL) {
    return;
  }
  loom_module_free(snapshot->module);
  *snapshot = (loom_pass_pipeline_snapshot_t){0};
}
