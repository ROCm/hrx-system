// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/predicate_lifetimes.h"

#include <string.h>

#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"
#include "loom/transforms/boundary/predicate_word.h"
#include "loom/util/fact_cfg.h"

typedef struct loom_predicate_escape_t {
  // Original operand, retained before any use-list mutation.
  loom_use_t use;
  // Consumer block in the retained CFG snapshot.
  uint16_t block_index;
  // Next escaping operand of the same definition.
  struct loom_predicate_escape_t* next;
} loom_predicate_escape_t;

typedef struct loom_predicate_capture_t {
  // Logical predicate whose exiting lanes must retain their final value.
  loom_value_id_t value;
  // Scalar or vector predicate shape.
  loom_type_t type;
  // Definition block for block arguments and ordinary results alike.
  loom_block_t* block;
  // Defining operation, or NULL for a block argument.
  loom_op_t* definition;
  // Ordinary operands outside the defining loop.
  loom_predicate_escape_t* escapes;
  // Next definition requiring durable transport.
  struct loom_predicate_capture_t* next;
} loom_predicate_capture_t;

typedef struct loom_predicate_reconstruction_t {
  // Most recent capture reconstructed in this consumer block.
  const loom_predicate_capture_t* capture;
  // Comparison result shared by that capture's uses in this block.
  loom_value_id_t value;
} loom_predicate_reconstruction_t;

static const loom_pass_info_t kPassInfo = {
    .name = IREE_SVL("project-loop-predicate-lifetimes"),
    .description = IREE_SVL("Preserve lane predicates across CFG loop exits."),
    .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_project_loop_predicate_lifetimes_pass_info(void) {
  return &kPassInfo;
}

static iree_status_t loom_predicate_plan_capture(
    loom_module_t* module, const loom_value_fact_cfg_region_t* cfg,
    uint16_t loop_index, uint32_t maximum_element_count,
    loom_value_id_t value_id, loom_block_t* block, loom_op_t* definition,
    iree_arena_allocator_t* arena, loom_predicate_capture_t** captures) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_predicate_word_supports_type(value->type, maximum_element_count)) {
    return iree_ok_status();
  }
  loom_predicate_capture_t* capture = NULL;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < value->use_count && iree_status_is_ok(status); ++i) {
    const loom_use_t use = loom_value_uses(value)[i];
    const loom_op_t* user = loom_use_user_op(use);
    if (user->parent_block->parent_region != cfg->graph.region ||
        loom_cfg_loop_nest_contains(&cfg->loops, loop_index,
                                    user->parent_block->region_index)) {
      continue;
    }
    if (capture == NULL) {
      status = iree_arena_allocate(arena, sizeof(*capture), (void**)&capture);
      if (iree_status_is_ok(status)) {
        *capture = (loom_predicate_capture_t){
            .value = value_id,
            .type = value->type,
            .block = block,
            .definition = definition,
            .next = *captures,
        };
        *captures = capture;
      }
    }
    loom_predicate_escape_t* escape = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(arena, sizeof(*escape), (void**)&escape);
    }
    if (iree_status_is_ok(status)) {
      *escape = (loom_predicate_escape_t){
          .use = use,
          .block_index = user->parent_block->region_index,
          .next = capture->escapes,
      };
      capture->escapes = escape;
    }
  }
  return status;
}

static iree_status_t loom_predicate_plan_lifetimes(
    loom_module_t* module, const loom_value_fact_cfg_region_t* cfg,
    uint32_t maximum_element_count, iree_arena_allocator_t* arena,
    loom_predicate_capture_t** out_captures) {
  *out_captures = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < cfg->graph.block_count && iree_status_is_ok(status); ++i) {
    const uint16_t loop_index = loom_cfg_loop_nest_innermost(&cfg->loops, i);
    if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
        loom_value_facts_is_subgroup_uniform(loom_value_fact_control_execution(
            cfg->control, cfg->loops.loops[loop_index].header_index))) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)cfg->graph.blocks[i].block;
    for (uint16_t j = 0; j < block->arg_count && iree_status_is_ok(status);
         ++j) {
      status = loom_predicate_plan_capture(
          module, cfg, loop_index, maximum_element_count, block->arg_ids[j],
          block, NULL, arena, out_captures);
    }
    for (loom_op_t* op = block->first_op; op && iree_status_is_ok(status);
         op = op->next_op) {
      for (uint16_t j = 0; j < op->result_count && iree_status_is_ok(status);
           ++j) {
        status = loom_predicate_plan_capture(
            module, cfg, loop_index, maximum_element_count,
            loom_op_results(op)[j], block, op, arena, out_captures);
      }
    }
  }
  return status;
}

static iree_status_t loom_predicate_apply_lifetimes(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* cfg,
    const loom_predicate_capture_t* captures) {
  loom_predicate_reconstruction_t* reconstructed = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, cfg->graph.block_count, sizeof(*reconstructed),
      (void**)&reconstructed));
  memset(reconstructed, 0, cfg->graph.block_count * sizeof(*reconstructed));
  iree_status_t status = iree_ok_status();
  for (const loom_predicate_capture_t* capture = captures;
       capture && iree_status_is_ok(status); capture = capture->next) {
    loom_builder_t* builder = &rewriter->builder;
    if (capture->definition) {
      loom_builder_set_after(builder, capture->definition);
    } else {
      loom_builder_set_before(builder, capture->block->first_op);
    }
    loom_value_id_t word = LOOM_VALUE_ID_INVALID;
    const loom_location_id_t location =
        capture->definition ? capture->definition->location
                            : capture->block->first_op->location;
    status = loom_predicate_word_materialize(
        builder, capture->value, loom_predicate_word_type(capture->type),
        location, &word);
    for (const loom_predicate_escape_t* escape = capture->escapes;
         escape && iree_status_is_ok(status); escape = escape->next) {
      loom_predicate_reconstruction_t* entry =
          &reconstructed[escape->block_index];
      if (entry->capture != capture) {
        loom_op_t* first =
            cfg->graph.blocks[escape->block_index].block->first_op;
        loom_builder_set_before(builder, first);
        status = loom_predicate_word_reconstruct(
            builder, word, capture->type, first->location, &entry->value);
        entry->capture = capture;
      }
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_set_operand(
            rewriter, loom_use_user_op(escape->use),
            loom_use_operand_index(escape->use), entry->value);
      }
    }
  }
  return status;
}

iree_status_t loom_project_loop_predicate_lifetimes_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(pass->function_version);
  if (version == NULL ||
      version->resolved_target.provider->loop_predicate_carrier !=
          LOOM_TARGET_LOOP_PREDICATE_CARRIER_I32) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body == NULL || body->block_count < 2) {
    return iree_ok_status();
  }
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_conditioned_function_for_target(
          function, version->function_target_facts),
      &facts));
  const loom_value_fact_cfg_region_t* cfg = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_get_or_build_cfg_region(facts, module, body, &cfg));
  if (cfg->loops.loop_count == 0) {
    return iree_ok_status();
  }
  loom_predicate_capture_t* captures = NULL;
  IREE_RETURN_IF_ERROR(loom_predicate_plan_lifetimes(
      module, cfg,
      version->resolved_target.provider
          ->loop_predicate_max_vector_element_count,
      pass->arena, &captures));
  if (captures == NULL) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status =
      loom_predicate_apply_lifetimes(&rewriter, cfg, captures);
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  if (iree_status_is_ok(status)) {
    loom_pass_mark_changed(pass);
  }
  return status;
}
