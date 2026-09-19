// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/ancestry.h"

#include "loom/ir/module.h"

bool loom_op_is_ancestor_of(const loom_op_t* ancestor_op, const loom_op_t* op) {
  if (!ancestor_op) {
    return false;
  }
  for (const loom_op_t* current_op = op; current_op;
       current_op = current_op->parent_op) {
    if (current_op == ancestor_op) {
      return true;
    }
  }
  return false;
}

static const loom_op_t* loom_region_parent_op(const loom_region_t* region) {
  if (!region) {
    return NULL;
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    if (block->first_op) {
      return block->first_op->parent_op;
    }
  }
  return NULL;
}

static bool loom_op_contains_region_slow(const loom_op_t* ancestor_op,
                                         const loom_region_t* target_region) {
  if (!ancestor_op || !target_region) {
    return false;
  }
  loom_region_t* const* regions = loom_op_regions(ancestor_op);
  for (uint8_t i = 0; i < ancestor_op->region_count; ++i) {
    const loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    if (region == target_region) {
      return true;
    }
    const loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        if (loom_op_contains_region_slow(child_op, target_region)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool loom_op_contains_block(const loom_op_t* ancestor_op,
                            const loom_block_t* block) {
  if (!ancestor_op || !block ||
      !loom_region_try_block_index(block->parent_region, block, NULL)) {
    return false;
  }
  const loom_op_t* parent_op = loom_region_parent_op(block->parent_region);
  return parent_op
             ? loom_op_is_ancestor_of(ancestor_op, parent_op)
             : loom_op_contains_region_slow(ancestor_op, block->parent_region);
}

bool loom_op_subtree_defines_value(const loom_module_t* module,
                                   const loom_op_t* ancestor_op,
                                   loom_value_id_t value_id) {
  if (!module || !ancestor_op || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value)
             ? loom_op_contains_block(ancestor_op, loom_value_def_block(value))
             : loom_op_is_ancestor_of(ancestor_op, loom_value_def_op(value));
}
