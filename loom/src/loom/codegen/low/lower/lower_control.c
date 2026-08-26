// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"

static iree_status_t loom_low_lower_expand_cfg_switch(loom_module_t* module,
                                                      loom_op_t* switch_op) {
  loom_attribute_t case_keys = loom_cfg_switch_case_keys(switch_op);
  loom_successor_slice_t case_dests = loom_cfg_switch_case_dests(switch_op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      case_keys.count != case_dests.count || case_keys.count == 0 ||
      case_keys.i64_array == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot expand malformed cfg.switch");
  }

  loom_block_t* source_block = switch_op->parent_block;
  loom_region_t* region = source_block->parent_region;
  const uint16_t source_block_index = source_block->region_index;
  const uint16_t dispatch_block_count = (uint16_t)(case_keys.count - 1);
  if (dispatch_block_count > UINT16_MAX - region->block_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "cfg.switch expansion exceeds the region block-count limit");
  }

  for (uint16_t i = 0; i < dispatch_block_count; ++i) {
    loom_block_t* dispatch_block = NULL;
    IREE_RETURN_IF_ERROR(loom_region_insert_block(
        module, region, (uint16_t)(source_block_index + i + 1),
        &dispatch_block));
  }
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, source_block, &builder);
  const loom_value_id_t selector = loom_cfg_switch_selector(switch_op);
  const loom_type_t selector_type = loom_module_value_type(module, selector);
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    loom_block_t* dispatch_block =
        i == 0 ? source_block : region->blocks[source_block_index + i];
    loom_block_t* false_dest = (uint16_t)(i + 1) < case_keys.count
                                   ? region->blocks[source_block_index + i + 1]
                                   : loom_cfg_switch_default_dest(switch_op);
    if (i == 0) {
      loom_builder_set_before(&builder, switch_op);
    } else {
      loom_builder_set_block(&builder, dispatch_block);
    }

    loom_op_t* key_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &builder, loom_attr_i64(case_keys.i64_array[i]), selector_type,
        switch_op->location, &key_op));
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        &builder, LOOM_INDEX_CMP_PREDICATE_EQ, selector,
        loom_index_constant_result(key_op), switch_op->location, &compare_op));
    loom_op_t* branch_op = NULL;
    IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(
        &builder, loom_index_cmp_result(compare_op), case_dests.blocks[i],
        false_dest, switch_op->location, &branch_op));
  }

  return loom_op_erase(module, switch_op);
}

static iree_status_t loom_low_lower_prepare_cfg_switches_in_region(
    loom_module_t* module, loom_region_t* region,
    const loom_target_facts_t* target_facts,
    const loom_low_lower_policy_t* policy, bool* out_changed) {
  const uint16_t original_block_count = region->block_count;
  for (uint16_t reverse_index = original_block_count; reverse_index > 0;
       --reverse_index) {
    loom_block_t* block = region->blocks[reverse_index - 1];
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t** child_regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (child_regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_prepare_cfg_switches_in_region(
            module, child_regions[i], target_facts, policy, out_changed));
      }
    }

    loom_op_t* terminator = block->last_op;
    if (!terminator || !loom_cfg_switch_isa(terminator)) {
      continue;
    }
    if (policy->switch_lowering.can_emit != NULL &&
        policy->switch_lowering.can_emit(policy->switch_lowering.user_data,
                                         module, terminator, target_facts)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_expand_cfg_switch(module, terminator));
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_prepare_cfg_switches(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_target_facts_t* target_facts,
    const loom_low_lower_policy_t* policy, bool* out_changed) {
  *out_changed = false;

  const bool has_query = policy->switch_lowering.can_emit != NULL;
  const bool has_emitter = policy->switch_lowering.emit != NULL;
  if (has_query != has_emitter) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low lowering switch policy must provide both query and emitter");
  }

  loom_region_t* body = loom_func_like_body(source_function);
  if (body == NULL) {
    return iree_ok_status();
  }
  return loom_low_lower_prepare_cfg_switches_in_region(
      module, body, target_facts, policy, out_changed);
}
