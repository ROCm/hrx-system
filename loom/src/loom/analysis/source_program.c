// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_program.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_source_program_build_state_t {
  // Program receiving indexed events.
  loom_source_program_t* program;
  // Arena owning the growing event array.
  iree_arena_allocator_t* arena;
} loom_source_program_build_state_t;

static int loom_source_program_compare_value_relations(const void* lhs_ptr,
                                                       const void* rhs_ptr) {
  const loom_source_program_value_relation_t* lhs =
      (const loom_source_program_value_relation_t*)lhs_ptr;
  const loom_source_program_value_relation_t* rhs =
      (const loom_source_program_value_relation_t*)rhs_ptr;
  if (lhs->lhs != rhs->lhs) return lhs->lhs < rhs->lhs ? -1 : 1;
  if (lhs->rhs != rhs->rhs) return lhs->rhs < rhs->rhs ? -1 : 1;
  return 0;
}

static void loom_source_program_canonicalize_value_relations(
    loom_source_program_t* program) {
  if (program->value_relation_count < 2) return;
  qsort(program->value_relations, program->value_relation_count,
        sizeof(*program->value_relations),
        loom_source_program_compare_value_relations);
  uint32_t write_index = 1;
  for (uint32_t read_index = 1; read_index < program->value_relation_count;
       ++read_index) {
    const loom_source_program_value_relation_t previous =
        program->value_relations[write_index - 1];
    const loom_source_program_value_relation_t current =
        program->value_relations[read_index];
    if (previous.lhs == current.lhs && previous.rhs == current.rhs) continue;
    program->value_relations[write_index++] = current;
  }
  program->value_relation_count = write_index;
}

static iree_status_t loom_source_program_append_value_relation(
    loom_source_program_build_state_t* state, loom_value_id_t lhs_value_id,
    loom_value_id_t rhs_value_id) {
  if (lhs_value_id == rhs_value_id) return iree_ok_status();
  const loom_local_value_domain_t* value_domain = state->program->value_domain;
  loom_value_ordinal_t lhs =
      loom_local_value_domain_try_ordinal(value_domain, lhs_value_id);
  loom_value_ordinal_t rhs =
      loom_local_value_domain_try_ordinal(value_domain, rhs_value_id);
  if (lhs == LOOM_VALUE_ORDINAL_INVALID || rhs == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source value relation escapes the indexed value domain");
  }
  if (lhs > rhs) {
    const loom_value_ordinal_t temporary = lhs;
    lhs = rhs;
    rhs = temporary;
  }
  loom_source_program_t* program = state->program;
  if (program->value_relation_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds value relation range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)program->value_relation_count + 1;
  if (minimum_capacity > program->value_relation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, program->value_relation_count, minimum_capacity,
        sizeof(*program->value_relations), &program->value_relation_capacity,
        (void**)&program->value_relations));
  }
  program->value_relations[program->value_relation_count++] =
      (loom_source_program_value_relation_t){
          .lhs = lhs,
          .rhs = rhs,
      };
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_cfg_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t successor_index = 0; successor_index < op->successor_count;
       ++successor_index) {
    const loom_block_t* destination = successors[successor_index];
    const loom_value_id_t* payload = NULL;
    uint16_t payload_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(op, destination, &payload,
                                                   &payload_count)) {
      continue;
    }
    const uint16_t relation_count =
        iree_min(payload_count, destination->arg_count);
    for (uint16_t i = 0; i < relation_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_relation(
          state, payload[i], loom_block_arg_id(destination, i)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_local_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    if (tied.operand_index >= op->operand_count ||
        tied.result_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source op carries an invalid tied result");
    }
    IREE_RETURN_IF_ERROR(loom_source_program_append_value_relation(
        state, operands[tied.operand_index], results[tied.result_index]));
  }

  const loom_trait_flags_t traits =
      loom_op_effective_traits(state->program->module, op);
  if (loom_traits_are_fact_identity(traits)) {
    const uint16_t relation_count =
        iree_min(op->operand_count, op->result_count);
    for (uint16_t i = 0; i < relation_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_relation(
          state, operands[i], results[i]));
    }
  }
  if (loom_traits_are_value_alias(traits) && op->operand_count != 0 &&
      op->result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_source_program_append_value_relation(
        state, operands[0], results[0]));
  }
  return iree_ok_status();
}

static const loom_op_t* loom_source_program_region_terminator(
    const loom_region_t* region) {
  if (region == NULL || region->block_count != 1) return NULL;
  const loom_block_t* block = loom_region_const_entry_block(region);
  return block != NULL ? block->last_op : NULL;
}

static iree_status_t loom_source_program_join_value(
    loom_source_program_build_state_t* state, loom_value_id_t anchor,
    loom_value_id_t value) {
  if (anchor == LOOM_VALUE_ID_INVALID || value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_source_program_append_value_relation(state, anchor, value);
}

static iree_status_t loom_source_program_index_loop_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op,
    loom_loop_like_t loop) {
  const loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  const loom_value_id_t* results = loom_op_const_results(op);
  const uint16_t state_count = iree_min(iter_args.count, op->result_count);
  loom_region_t* body = loom_loop_like_body(loop);
  const loom_block_t* body_entry = body != NULL && body->block_count != 0
                                       ? loom_region_const_entry_block(body)
                                       : NULL;
  const uint16_t body_arg_offset =
      loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
          ? 0
          : (uint16_t)(loop.vtable->iv_block_arg_index + 1);
  const loom_op_t* body_terminator =
      loom_source_program_region_terminator(body);
  const loom_value_id_t* body_yields =
      body_terminator != NULL ? loom_op_const_operands(body_terminator) : NULL;

  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const loom_block_t* condition_entry =
      condition_region != NULL && condition_region->block_count != 0
          ? loom_region_const_entry_block(condition_region)
          : NULL;
  const loom_op_t* condition_terminator =
      loom_source_program_region_terminator(condition_region);
  const loom_value_id_t* condition_forwarded =
      condition_terminator != NULL && condition_terminator->operand_count != 0
          ? loom_op_const_operands(condition_terminator) + 1
          : NULL;
  const uint16_t condition_forwarded_count =
      condition_terminator != NULL && condition_terminator->operand_count != 0
          ? (uint16_t)(condition_terminator->operand_count - 1)
          : 0;

  for (uint16_t i = 0; i < state_count; ++i) {
    const loom_value_id_t anchor = iter_args.values[i];
    IREE_RETURN_IF_ERROR(
        loom_source_program_join_value(state, anchor, results[i]));
    if (body_entry != NULL && body_arg_offset + i < body_entry->arg_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, loom_block_arg_id(body_entry, body_arg_offset + i)));
    }
    if (body_terminator != NULL && i < body_terminator->operand_count) {
      IREE_RETURN_IF_ERROR(
          loom_source_program_join_value(state, anchor, body_yields[i]));
    }
    if (condition_entry != NULL && i < condition_entry->arg_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, loom_block_arg_id(condition_entry, i)));
    }
    if (condition_forwarded != NULL && i < condition_forwarded_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, condition_forwarded[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_region_branch_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op,
    loom_region_branch_t branch) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_op_t* terminator = loom_region_branch_region_terminator(
        state->program->module, branch, region_index);
    if (terminator == NULL) continue;
    const uint16_t relation_count =
        iree_min(terminator->operand_count, op->result_count);
    const loom_value_id_t* yielded = loom_op_const_operands(terminator);
    for (uint16_t i = 0; i < relation_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_relation(
          state, yielded[i], results[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_value_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_source_program_index_cfg_relations(state, op));
  IREE_RETURN_IF_ERROR(loom_source_program_index_local_relations(state, op));
  loom_loop_like_t loop =
      loom_loop_like_cast(state->program->module, (loom_op_t*)op);
  if (loom_loop_like_isa(loop)) {
    IREE_RETURN_IF_ERROR(
        loom_source_program_index_loop_relations(state, op, loop));
  }
  loom_region_branch_t branch =
      loom_region_branch_cast(state->program->module, (loom_op_t*)op);
  if (loom_region_branch_isa(branch)) {
    IREE_RETURN_IF_ERROR(
        loom_source_program_index_region_branch_relations(state, op, branch));
  }
  return iree_ok_status();
}

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
      IREE_RETURN_IF_ERROR(
          loom_source_program_index_value_relations(state, op));
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
  if (iree_status_is_ok(status)) {
    loom_source_program_canonicalize_value_relations(out_program);
  } else {
    *out_program = (loom_source_program_t){0};
  }
  return status;
}
