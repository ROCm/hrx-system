// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_version.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

const loom_function_version_type_t loom_target_function_version_type = {
    .name = IREE_SVL("target"),
};

loom_target_function_version_t* loom_target_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function) {
  return loom_target_function_version_cast(
      loom_function_version_list_find(list, function));
}

iree_status_t loom_target_function_version_snapshot_build(
    const loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    iree_arena_allocator_t* arena,
    loom_target_function_version_snapshot_t* out_snapshot) {
  *out_snapshot = (loom_target_function_version_snapshot_t){
      .symbol_count = module->symbols.count,
  };
  const iree_host_size_t function_version_count =
      function_versions != NULL ? function_versions->count : 0;
  if (function_version_count > LOOM_FUNCTION_VERSION_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "target function-version count exceeds the "
                            "ordinal domain");
  }

  if (module->symbols.count > 0) {
    iree_host_size_t handle_offset = 0;
    iree_host_size_t ordinal_offset = 0;
    iree_host_size_t storage_size = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &storage_size,
        IREE_STRUCT_FIELD(module->symbols.count, loom_function_version_t*,
                          &handle_offset),
        IREE_STRUCT_FIELD(module->symbols.count,
                          loom_function_version_ordinal_t, &ordinal_offset)));
    uint8_t* storage = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, storage_size, (void**)&storage));
    out_snapshot->version_handles_by_symbol =
        (loom_function_version_t**)(storage + handle_offset);
    out_snapshot->version_ordinals_by_symbol =
        (loom_function_version_ordinal_t*)(storage + ordinal_offset);
    memset(out_snapshot->version_handles_by_symbol, 0,
           module->symbols.count *
               sizeof(*out_snapshot->version_handles_by_symbol));
    memset(out_snapshot->version_ordinals_by_symbol, 0xFF,
           module->symbols.count *
               sizeof(*out_snapshot->version_ordinals_by_symbol));
  }
  for (iree_host_size_t i = 0; i < function_version_count; ++i) {
    loom_function_version_t* version_handle = function_versions->values[i];
    const loom_target_function_version_t* function_version =
        loom_target_function_version_const_cast(version_handle);
    if (function_version == NULL) continue;

    const loom_func_like_t function = function_version->base.function;
    const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
    if (!loom_func_like_isa(function) || function.vtable == NULL ||
        !loom_symbol_ref_is_valid(function_ref) ||
        function_ref.module_id != 0 ||
        function_ref.symbol_id >= module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target function version %zu does not name a module-local function "
          "symbol in the observed module",
          i);
    }
    const loom_symbol_t* symbol =
        &module->symbols.entries[function_ref.symbol_id];
    const loom_func_like_t live_function =
        loom_func_like_cast(module, symbol->defining_op);
    if (symbol->defining_op != function.op ||
        live_function.vtable != function.vtable) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target function version %zu does not name the live definition of "
          "module symbol %u",
          i, (unsigned)function_ref.symbol_id);
    }
    if (out_snapshot->version_handles_by_symbol[function_ref.symbol_id] !=
        NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "multiple target function versions name module symbol %u",
          (unsigned)function_ref.symbol_id);
    }
    out_snapshot->version_handles_by_symbol[function_ref.symbol_id] =
        version_handle;
    out_snapshot->version_ordinals_by_symbol[function_ref.symbol_id] =
        (loom_function_version_ordinal_t)i;
  }
  return iree_ok_status();
}
