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
  if (module->symbols.count == 0 || function_versions == NULL ||
      function_versions->count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count,
      sizeof(*out_snapshot->version_handles_by_symbol),
      (void**)&out_snapshot->version_handles_by_symbol));
  memset(
      out_snapshot->version_handles_by_symbol, 0,
      module->symbols.count * sizeof(*out_snapshot->version_handles_by_symbol));
  for (iree_host_size_t i = 0; i < function_versions->count; ++i) {
    loom_function_version_t* version_handle = function_versions->values[i];
    const loom_target_function_version_t* function_version =
        loom_target_function_version_const_cast(version_handle);
    if (function_version == NULL) continue;
    const loom_symbol_ref_t function_ref =
        loom_func_like_callee(function_version->base.function);
    out_snapshot->version_handles_by_symbol[function_ref.symbol_id] =
        version_handle;
  }
  return iree_ok_status();
}
