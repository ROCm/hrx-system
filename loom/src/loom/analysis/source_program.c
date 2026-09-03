// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_program.h"

#include <string.h>

typedef struct loom_source_program_build_state_t {
  // Program receiving indexed events.
  loom_source_program_t* program;
  // Arena owning the growing event array.
  iree_arena_allocator_t* arena;
} loom_source_program_build_state_t;

static iree_status_t loom_source_program_append_node(
    loom_source_program_build_state_t* state, loom_source_program_node_t node,
    loom_source_program_node_ordinal_t* out_ordinal) {
  loom_source_program_t* program = state->program;
  if (program->node_count == LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds node ordinal range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)program->node_count + 1;
  if (minimum_capacity > program->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, program->node_count, minimum_capacity,
        sizeof(*program->nodes), &program->node_capacity,
        (void**)&program->nodes));
  }
  const loom_source_program_node_ordinal_t ordinal = program->node_count++;
  program->nodes[ordinal] = node;
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_region(
    loom_source_program_build_state_t* state, const loom_region_t* region,
    const loom_op_t* context_op, uint16_t region_depth, bool is_root_region) {
  if (state->program->region_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds region count range");
  }
  ++state->program->region_count;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (state->program->block_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source program exceeds block count range");
    }
    const loom_block_t* block = loom_region_const_block(region, block_index);
    loom_source_program_node_ordinal_t block_ordinal =
        LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_source_program_append_node(
        state,
        (loom_source_program_node_t){
            .object = block,
            .context_op = context_op,
            .region_depth = region_depth,
            .kind = LOOM_SOURCE_PROGRAM_NODE_BLOCK,
            .flags = is_root_region && block_index == 0
                         ? LOOM_SOURCE_PROGRAM_NODE_ROOT_ENTRY_BLOCK
                         : 0,
        },
        &block_ordinal));
    ++state->program->block_count;

    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (state->program->operation_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source program exceeds operation count range");
      }
      loom_source_program_node_ordinal_t op_ordinal =
          LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
      IREE_RETURN_IF_ERROR(loom_source_program_append_node(
          state,
          (loom_source_program_node_t){
              .object = op,
              .region_depth = region_depth,
              .kind = LOOM_SOURCE_PROGRAM_NODE_OPERATION,
          },
          &op_ordinal));
      ++state->program->operation_count;

      if (op->region_count != 0 && region_depth == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source program exceeds region nesting range");
      }
      loom_region_t* const* child_regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (child_regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_source_program_index_region(
            state, child_regions[i], op, (uint16_t)(region_depth + 1),
            /*is_root_region=*/false));
      }
      state->program->nodes[op_ordinal].subtree_limit =
          state->program->node_count;
    }
    state->program->nodes[block_ordinal].subtree_limit =
        state->program->node_count;
  }
  return iree_ok_status();
}

iree_status_t loom_source_program_build(
    const loom_module_t* module, const loom_op_t* root_context_op,
    const loom_region_t* root_region,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_source_program_t* out_program) {
  IREE_ASSERT(module != NULL);
  IREE_ASSERT(root_region != NULL);
  IREE_ASSERT(value_domain != NULL);
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  IREE_ASSERT(value_domain->module == module);
  IREE_ASSERT(value_domain->region == root_region);
  IREE_ASSERT(iree_any_bit_set(value_domain->flags,
                               LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE));
  IREE_ASSERT(arena != NULL);
  IREE_ASSERT(out_program != NULL);
  *out_program = (loom_source_program_t){
      .module = module,
      .root_region = root_region,
      .value_domain = value_domain,
  };
  loom_source_program_build_state_t state = {
      .program = out_program,
      .arena = arena,
  };
  iree_status_t status = loom_source_program_index_region(
      &state, root_region, root_context_op, /*region_depth=*/0,
      /*is_root_region=*/true);
  if (!iree_status_is_ok(status)) {
    *out_program = (loom_source_program_t){0};
  }
  return status;
}
