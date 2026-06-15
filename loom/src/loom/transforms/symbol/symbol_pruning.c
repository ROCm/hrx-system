// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_pruning.h"

#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"

static bool loom_symbol_pruning_function_is_source_entry(
    loom_func_like_t function) {
  return loom_symbol_ref_is_valid(loom_func_like_target(function)) &&
         (loom_func_def_isa(function.op) || loom_kernel_def_isa(function.op));
}

static bool loom_symbol_pruning_retain_target_source_entries(
    const loom_symbol_pruning_options_t* options) {
  return options &&
         iree_all_bits_set(options->flags,
                           LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES);
}

static bool loom_symbol_pruning_symbol_is_erasable_with_options(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_symbol_pruning_options_t* options) {
  if (!symbol || !symbol->defining_op) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return false;
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      return true;
    }
    if (loom_func_like_visibility(function) != 0) {
      return false;
    }
    if (loom_symbol_pruning_retain_target_source_entries(options) &&
        loom_symbol_pruning_function_is_source_entry(function)) {
      return false;
    }
    if (loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID ||
        loom_func_like_export_attrs(function).count > 0) {
      return false;
    }
    return true;
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    return true;
  }

  return false;
}

bool loom_symbol_pruning_symbol_is_erasable(const loom_module_t* module,
                                            const loom_symbol_t* symbol) {
  return loom_symbol_pruning_symbol_is_erasable_with_options(module, symbol,
                                                             NULL);
}

bool loom_symbol_pruning_symbol_is_root(void* user_data,
                                        const loom_module_t* module,
                                        loom_symbol_id_t symbol_id,
                                        const loom_symbol_t* symbol) {
  (void)symbol_id;
  const loom_symbol_pruning_options_t* options =
      (const loom_symbol_pruning_options_t*)user_data;
  return !loom_symbol_pruning_symbol_is_erasable_with_options(module, symbol,
                                                              options);
}

typedef struct loom_symbol_pruning_erasure_t {
  // Defining op that should be erased from the module body.
  loom_op_t* op;

  // True when |op| defines a function-like symbol.
  bool is_function_like;
} loom_symbol_pruning_erasure_t;

iree_status_t loom_symbol_pruning_erase_unreachable(
    loom_module_t* module, const loom_symbol_liveness_t* liveness,
    const loom_symbol_pruning_options_t* options, iree_arena_allocator_t* arena,
    loom_symbol_pruning_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_symbol_pruning_result_t){0};
  if (liveness->module != module ||
      liveness->symbol_count != module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol pruning liveness result does not match "
                            "the module symbol table");
  }
  if (module->symbols.count == 0) return iree_ok_status();

  loom_symbol_pruning_erasure_t* erasures = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*erasures), (void**)&erasures));

  iree_host_size_t erasure_count = 0;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    loom_symbol_id_t symbol_id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    if (loom_symbol_liveness_is_live(liveness, symbol_id)) continue;
    if (!loom_symbol_pruning_symbol_is_erasable_with_options(module, symbol,
                                                             options)) {
      continue;
    }
    erasures[erasure_count++] = (loom_symbol_pruning_erasure_t){
        .op = symbol->defining_op,
        .is_function_like =
            loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE),
    };
  }

  loom_symbol_pruning_result_t result = {0};
  for (iree_host_size_t i = 0; i < erasure_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_op_erase(module, erasures[i].op));
    ++result.symbol_count;
    if (erasures[i].is_function_like) {
      ++result.function_like_count;
    }
  }
  *out_result = result;
  return iree_ok_status();
}
