// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_requirements.h"

#include "loom/ops/low/ops.h"

iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements) {
  *out_requirements = (loom_low_function_requirements_t){0};
  const loom_op_t** resources = NULL;
  iree_host_size_t resource_capacity = 0;
  loom_low_storage_layout_builder_t storage_builder;
  loom_low_storage_layout_builder_initialize(&storage_builder);
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    out_requirements->node_count += block->op_count;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_resource_isa(op)) {
        const iree_host_size_t minimum_capacity =
            out_requirements->resource_count + 1;
        if (minimum_capacity > resource_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              arena, out_requirements->resource_count,
              iree_max(minimum_capacity, 4u), sizeof(*resources),
              &resource_capacity, (void**)&resources));
        }
        resources[out_requirements->resource_count++] = op;
      } else if (loom_low_storage_reserve_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_low_storage_layout_builder_append(
            module, op, arena, &storage_builder));
      } else if (loom_low_return_isa(op)) {
        ++out_requirements->return_count;
      }
    }
  }
  out_requirements->resources = resources;
  loom_low_storage_layout_builder_finish(&storage_builder,
                                         &out_requirements->storage_layout);
  return iree_ok_status();
}
