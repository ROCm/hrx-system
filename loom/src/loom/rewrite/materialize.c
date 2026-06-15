// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/materialize.h"

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/rewrite/rewriter.h"

static iree_status_t loom_ir_clone_value_name(loom_ir_remap_t* remap,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t target_value_id) {
  const loom_value_t* source_value =
      loom_module_value(remap->source_module, source_value_id);
  loom_value_t* target_value =
      loom_module_value(remap->target_module, target_value_id);
  return loom_ir_remap_string_id(remap, source_value->name_id,
                                 /*allow_invalid=*/true,
                                 &target_value->name_id);
}

static iree_status_t loom_ir_clone_block_label(loom_ir_remap_t* remap,
                                               const loom_block_t* source_block,
                                               loom_block_t* target_block) {
  return loom_ir_remap_string_id(remap, source_block->label_id,
                                 /*allow_invalid=*/true,
                                 &target_block->label_id);
}

static iree_status_t loom_ir_clone_block_args(loom_ir_remap_t* remap,
                                              const loom_block_t* source_block,
                                              loom_block_t* target_block) {
  if (source_block->arg_count == 0) return iree_ok_status();

  loom_value_id_t* target_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(remap->arena, source_block->arg_count,
                                sizeof(loom_value_id_t), (void**)&target_args));
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    loom_value_id_t target_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        remap->target_module, loom_type_none(), &target_arg));
    IREE_RETURN_IF_ERROR(
        loom_block_add_arg(remap->target_module, target_block, target_arg));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, loom_block_arg_id(source_block, i), target_arg));
    IREE_RETURN_IF_ERROR(loom_ir_clone_value_name(
        remap, loom_block_arg_id(source_block, i), target_arg));
    target_args[i] = target_arg;
  }
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap,
        loom_module_value_type(remap->source_module,
                               loom_block_arg_id(source_block, i)),
        &target_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        remap->target_module, target_args[i], target_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_results(loom_ir_remap_t* remap,
                                              const loom_op_t* source_op,
                                              loom_value_id_t* target_results) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    target_results[i] = LOOM_VALUE_ID_INVALID;
    if (source_results[i] == LOOM_VALUE_ID_INVALID) continue;
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        remap->target_module, loom_type_none(), &target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, source_results[i], target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_value_name(remap, source_results[i], target_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_operands(
    loom_ir_remap_t* remap, const loom_op_t* source_op,
    loom_value_id_t* target_operands) {
  const loom_op_vtable_t* source_vtable =
      loom_op_vtable(remap->source_module, source_op);
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  if (source_vtable && source_vtable->func_like &&
      source_vtable->func_like->args_as_operands) {
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      target_operands[i] = LOOM_VALUE_ID_INVALID;
      if (source_operands[i] == LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_module_define_value(
          remap->target_module, loom_type_none(), &target_operands[i]));
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(remap, source_operands[i],
                                                   target_operands[i]));
      IREE_RETURN_IF_ERROR(loom_ir_clone_value_name(remap, source_operands[i],
                                                    target_operands[i]));
    }
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      if (source_operands[i] == LOOM_VALUE_ID_INVALID) {
        continue;
      }
      loom_type_t target_type = {0};
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          remap,
          loom_module_value_type(remap->source_module, source_operands[i]),
          &target_type));
      IREE_RETURN_IF_ERROR(loom_module_set_value_type(
          remap->target_module, target_operands[i], target_type));
    }
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    if (source_operands[i] == LOOM_VALUE_ID_INVALID) {
      target_operands[i] = LOOM_VALUE_ID_INVALID;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, source_operands[i],
                                                     &target_operands[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_successors(
    loom_ir_remap_t* remap, const loom_op_t* source_op,
    loom_block_t** target_successors) {
  loom_block_t* const* source_successors = loom_op_const_successors(source_op);
  for (uint8_t i = 0; i < source_op->successor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_block(
        remap, source_successors[i], &target_successors[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_result_types(
    loom_ir_remap_t* remap, const loom_op_t* source_op,
    loom_value_id_t* target_results, loom_type_t* target_result_types) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    if (source_results[i] == LOOM_VALUE_ID_INVALID) {
      target_result_types[i] = loom_type_none();
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(remap->source_module, source_results[i]),
        &target_result_types[i]));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        remap->target_module, target_results[i], target_result_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_attrs(loom_ir_remap_t* remap,
                                            const loom_op_t* source_op,
                                            loom_attribute_t* target_attrs) {
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_attribute(remap, source_attrs[i], &target_attrs[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_region_skeleton(
    loom_ir_remap_t* remap, const loom_region_t* source_region,
    loom_region_t** out_target_region) {
  *out_target_region = NULL;
  loom_region_t* target_region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(
      remap->target_module, source_region->block_count, &target_region));
  target_region->flags = source_region->flags;

  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_region, block_index);
    loom_block_t* target_block = loom_region_block(target_region, block_index);
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_block(remap, source_block, target_block));
    target_block->flags = source_block->flags;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_block_label(remap, source_block, target_block));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_block_args(remap, source_block, target_block));
  }

  *out_target_region = target_region;
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_region_ops(
    loom_builder_t* builder, const loom_region_t* source_region,
    loom_ir_remap_t* remap, loom_region_t* target_region) {
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_op_t* parent_op = builder->ip.parent_op;
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_region->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_region, block_index);
    loom_block_t* target_block = loom_region_block(target_region, block_index);
    builder->ip.block = target_block;
    builder->ip.parent_op = parent_op;
    builder->ip.before_op = NULL;
    status = loom_ir_clone_block_ops(builder, source_block, remap,
                                     /*options=*/NULL);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

iree_status_t loom_ir_clone_op(loom_builder_t* builder,
                               const loom_op_t* source_op,
                               loom_ir_remap_t* remap,
                               loom_op_t** out_cloned_op) {
  *out_cloned_op = NULL;
  if (builder->module != remap->target_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "builder module must be the remap target module");
  }
  if (iree_any_bit_set(source_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot clone a dead op");
  }
  const loom_op_vtable_t* source_vtable =
      loom_op_vtable(remap->source_module, source_op);
  const loom_op_vtable_t* target_vtable =
      loom_op_vtable(remap->target_module, source_op);
  if (!target_vtable) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target module has no vtable for source op kind");
  }

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(remap, source_op->location, &target_location));

  loom_value_id_t* target_operands = NULL;
  if (source_op->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->operand_count, sizeof(loom_value_id_t),
        (void**)&target_operands));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_operands(remap, source_op, target_operands));
  }

  loom_block_t** target_successors = NULL;
  if (source_op->successor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->successor_count, sizeof(loom_block_t*),
        (void**)&target_successors));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_successors(remap, source_op, target_successors));
  }

  loom_region_t** target_regions = NULL;
  loom_region_t* const* source_regions = loom_op_regions(source_op);
  if (source_op->region_count > 0) {
    // Region entry arguments can appear in parent op result types and attrs
    // (for example dynamic function signatures), so map the region skeleton
    // before remapping parent op payloads.
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->region_count, sizeof(loom_region_t*),
        (void**)&target_regions));
    for (uint8_t i = 0; i < source_op->region_count; ++i) {
      target_regions[i] = NULL;
      if (!source_regions[i]) continue;
      IREE_RETURN_IF_ERROR(loom_ir_clone_region_skeleton(
          remap, source_regions[i], &target_regions[i]));
    }
  }

  loom_value_id_t* target_results = NULL;
  loom_type_t* target_result_types = NULL;
  if (source_op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->result_count, sizeof(loom_value_id_t),
        (void**)&target_results));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->result_count, sizeof(loom_type_t),
        (void**)&target_result_types));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_results(remap, source_op, target_results));
    IREE_RETURN_IF_ERROR(loom_ir_clone_op_result_types(
        remap, source_op, target_results, target_result_types));
  }

  loom_attribute_t* target_attrs = NULL;
  if (source_op->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->attribute_count, sizeof(loom_attribute_t),
        (void**)&target_attrs));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_attrs(remap, source_op, target_attrs));
  }

  loom_op_t* target_op = NULL;
  uint8_t operand_segment_count =
      loom_op_vtable_operand_segment_count(source_vtable);
  const uint16_t* operand_segment_counts =
      operand_segment_count > 0
          ? loom_op_const_operand_segment_counts(source_op)
          : NULL;
  if (loom_op_vtable_has_segmented_operands(target_vtable)) {
    if (source_op->successor_count > 0) {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
          builder, source_op->kind, source_op->operand_count,
          operand_segment_counts, operand_segment_count,
          source_op->result_count, source_op->successor_count,
          source_op->region_count, source_op->tied_result_count,
          source_op->attribute_count, target_location, &target_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
          builder, source_op->kind, source_op->operand_count,
          operand_segment_counts, operand_segment_count,
          source_op->result_count, source_op->region_count,
          source_op->tied_result_count, source_op->attribute_count,
          target_location, &target_op));
    }
  } else {
    if (source_op->successor_count > 0) {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
          builder, source_op->kind, source_op->operand_count,
          source_op->result_count, source_op->successor_count,
          source_op->region_count, source_op->tied_result_count,
          source_op->attribute_count, target_location, &target_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
          builder, source_op->kind, source_op->operand_count,
          source_op->result_count, source_op->region_count,
          source_op->tied_result_count, source_op->attribute_count,
          target_location, &target_op));
    }
  }
  target_op->instance_flags = source_op->instance_flags;
  target_op->traits = source_op->traits;
  if (source_op->operand_count > 0) {
    memcpy(
        loom_op_operands(target_op), target_operands,
        (iree_host_size_t)source_op->operand_count * sizeof(loom_value_id_t));
  }
  if (source_op->successor_count > 0) {
    memcpy(
        loom_op_successors(target_op), target_successors,
        (iree_host_size_t)source_op->successor_count * sizeof(loom_block_t*));
  }
  if (source_op->region_count > 0) {
    memcpy(loom_op_regions(target_op), target_regions,
           (iree_host_size_t)source_op->region_count * sizeof(loom_region_t*));
    for (uint8_t i = 0; i < source_op->region_count; ++i) {
      if (!target_regions[i]) continue;
      for (uint16_t block_index = 0;
           block_index < target_regions[i]->block_count; ++block_index) {
        target_regions[i]->blocks[block_index]->parent_region =
            target_regions[i];
      }
    }
  }
  if (source_op->result_count > 0) {
    memcpy(loom_op_results(target_op), target_results,
           (iree_host_size_t)source_op->result_count * sizeof(loom_value_id_t));
  }
  if (source_op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(target_op), loom_op_tied_results(source_op),
           (iree_host_size_t)source_op->tied_result_count *
               sizeof(loom_tied_result_t));
  }
  if (source_op->attribute_count > 0) {
    memcpy(loom_op_attrs(target_op), target_attrs,
           (iree_host_size_t)source_op->attribute_count *
               sizeof(loom_attribute_t));
  }

  if (source_op->region_count > 0) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    builder->ip.parent_op = target_op;
    iree_status_t status = iree_ok_status();
    for (uint8_t i = 0;
         i < source_op->region_count && iree_status_is_ok(status); ++i) {
      if (!source_regions[i]) continue;
      status = loom_ir_clone_region_ops(builder, source_regions[i], remap,
                                        target_regions[i]);
    }
    loom_builder_restore(builder, saved_ip);
    IREE_RETURN_IF_ERROR(status);
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target_op));
  *out_cloned_op = target_op;
  return iree_ok_status();
}

iree_status_t loom_ir_clone_block_ops(
    loom_builder_t* builder, const loom_block_t* source_block,
    loom_ir_remap_t* remap, const loom_ir_clone_block_options_t* options) {
  bool omit_terminators = options ? options->omit_terminators : false;
  for (const loom_op_t* source_op = source_block->first_op; source_op;
       source_op = source_op->next_op) {
    const loom_op_vtable_t* vtable =
        loom_op_vtable(remap->source_module, source_op);
    if (omit_terminators && vtable &&
        iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR)) {
      continue;
    }
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(builder, source_op, remap, &cloned_op));
  }
  return iree_ok_status();
}

iree_status_t loom_ir_clone_region(loom_builder_t* builder,
                                   const loom_region_t* source_region,
                                   loom_ir_remap_t* remap,
                                   loom_region_t** out_target_region) {
  *out_target_region = NULL;
  loom_region_t* target_region = NULL;
  IREE_RETURN_IF_ERROR(
      loom_ir_clone_region_skeleton(remap, source_region, &target_region));
  iree_status_t status =
      loom_ir_clone_region_ops(builder, source_region, remap, target_region);
  IREE_RETURN_IF_ERROR(status);
  *out_target_region = target_region;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Existing op remapping
//===----------------------------------------------------------------------===//

static iree_status_t loom_ir_remap_map_value_to_self(loom_ir_remap_t* remap,
                                                     loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) return iree_ok_status();
  loom_value_id_t mapped_value = LOOM_VALUE_ID_INVALID;
  if (loom_ir_remap_try_lookup_value(remap, value_id, &mapped_value)) {
    if (mapped_value == value_id) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot move value %%%u through remap to distinct value %%%u",
        (unsigned)value_id, (unsigned)mapped_value);
  }
  return loom_ir_remap_map_value(remap, value_id, value_id);
}

static iree_status_t loom_ir_remap_block_arg_values_to_self(
    loom_ir_remap_t* remap, const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value_to_self(remap, loom_block_arg_id(block, i)));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_op_subtree_values_to_self(
    loom_ir_remap_t* remap, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value_to_self(remap, results[i]));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_block_arg_values_to_self(remap, block));
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_ir_remap_op_subtree_values_to_self(remap, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_op_operands_in_place(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_ir_remap_t* remap) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_resolve_value(remap, operands[i], &target_value));
    if (target_value != operands[i]) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_operand(rewriter, op, i, target_value));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_op_successors_in_place(
    loom_op_t* op, loom_ir_remap_t* remap) {
  loom_block_t** successors = loom_op_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    loom_block_t* target_block = NULL;
    if (!loom_ir_remap_try_lookup_block(remap, successors[i], &target_block)) {
      continue;
    }
    successors[i] = target_block;
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_result_types_in_place(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_ir_remap_t* remap) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID) continue;
    loom_type_t source_type =
        loom_module_value_type(remap->source_module, result);
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(remap, source_type, &target_type));
    if (!loom_type_equal(source_type, target_type)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_value_type(rewriter, result, target_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_attrs_in_place(loom_rewriter_t* rewriter,
                                                  loom_op_t* op,
                                                  loom_ir_remap_t* remap) {
  const loom_attribute_t* source_attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    loom_attribute_t target_attr = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_attribute(remap, source_attrs[i], &target_attr));
    if (!loom_attribute_equal(&source_attrs[i], &target_attr)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_attr(rewriter, op, i, target_attr));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_remap_block_arg_types_in_place(
    loom_rewriter_t* rewriter, loom_block_t* block, loom_ir_remap_t* remap) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t arg_id = loom_block_arg_id(block, i);
    loom_type_t source_type =
        loom_module_value_type(remap->source_module, arg_id);
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(remap, source_type, &target_type));
    if (!loom_type_equal(source_type, target_type)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_value_type(rewriter, arg_id, target_type));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_ir_remap_op_references(loom_rewriter_t* rewriter,
                                          loom_op_t* op,
                                          loom_ir_remap_t* remap) {
  if (remap->source_module != rewriter->module ||
      remap->target_module != rewriter->module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "in-place op reference remapping requires same-module remap");
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_op_subtree_values_to_self(remap, op));

  IREE_RETURN_IF_ERROR(loom_ir_remap_op_operands_in_place(rewriter, op, remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_op_successors_in_place(op, remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_result_types_in_place(rewriter, op, remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_attrs_in_place(rewriter, op, remap));

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_block_arg_types_in_place(rewriter, block, remap));
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_ir_remap_op_references(rewriter, child_op, remap));
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Move materialization
//===----------------------------------------------------------------------===//

static bool loom_ir_move_should_omit_op(const loom_module_t* module,
                                        const loom_op_t* op,
                                        bool omit_terminators) {
  if (!omit_terminators) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static bool loom_ir_move_value_id_is_valid(const loom_module_t* module,
                                           loom_value_id_t value_id) {
  return module && value_id != LOOM_VALUE_ID_INVALID &&
         value_id < module->values.count;
}

static bool loom_ir_move_source_value_id_is_valid(const loom_ir_remap_t* remap,
                                                  loom_value_id_t value_id) {
  return remap &&
         loom_ir_move_value_id_is_valid(remap->source_module, value_id);
}

static const loom_op_t* loom_ir_move_region_owner_op(
    const loom_region_t* region) {
  if (!region) {
    return NULL;
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* first_op = block->first_op;
    if (first_op) {
      return first_op->parent_op;
    }
  }
  return NULL;
}

static bool loom_ir_move_region_contains_block_slow(
    const loom_region_t* region, const loom_block_t* target) {
  if (!region || !target) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (block == target) {
      return true;
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_ir_move_region_contains_block_slow(regions[i], target)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_ir_move_block_is_nested_under_op(const loom_op_t* root,
                                                  const loom_block_t* block) {
  if (!root || !block) {
    return false;
  }
  const loom_op_t* owner_op =
      loom_ir_move_region_owner_op(block->parent_region);
  if (owner_op) {
    for (const loom_op_t* current = owner_op; current;
         current = current->parent_op) {
      if (current == root) {
        return true;
      }
    }
    return false;
  }

  loom_region_t** regions = loom_op_regions(root);
  for (uint8_t i = 0; i < root->region_count; ++i) {
    if (loom_ir_move_region_contains_block_slow(regions[i], block)) {
      return true;
    }
  }
  return false;
}

static bool loom_ir_move_op_is_inside_moved_block(
    const loom_module_t* module, const loom_block_t* source_block,
    bool omit_terminators, const loom_op_t* op) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current->parent_block != source_block) continue;
    return !loom_ir_move_should_omit_op(module, current, omit_terminators);
  }
  return false;
}

static bool loom_ir_move_block_arg_is_inside_moved_op(
    const loom_module_t* module, const loom_block_t* source_block,
    bool omit_terminators, const loom_block_t* block) {
  loom_op_t* root_op = NULL;
  loom_block_for_each_op(source_block, root_op) {
    if (loom_ir_move_should_omit_op(module, root_op, omit_terminators)) {
      continue;
    }
    if (loom_ir_move_block_is_nested_under_op(root_op, block)) {
      return true;
    }
  }
  return false;
}

static bool loom_ir_move_value_moves_with_block(
    const loom_module_t* module, const loom_block_t* source_block,
    bool omit_terminators, loom_value_id_t value_id) {
  if (!loom_ir_move_value_id_is_valid(module, value_id)) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    return block != source_block &&
           loom_ir_move_block_arg_is_inside_moved_op(module, source_block,
                                                     omit_terminators, block);
  }
  return loom_ir_move_op_is_inside_moved_block(
      module, source_block, omit_terminators, loom_value_def_op(value));
}

typedef struct loom_ir_move_availability_t {
  // Availability analysis for the module being rewritten.
  const loom_availability_analysis_t* analysis;
  // Same-module remap applied before checking value availability.
  loom_ir_remap_t* remap;
  // Source block whose live non-omitted ops are being moved.
  const loom_block_t* source_block;
  // Insertion point that moved ops must be valid immediately before.
  const loom_op_t* before_op;
  // True to treat source terminators as outside the moved op set.
  bool omit_terminators;
} loom_ir_move_availability_t;

typedef struct loom_ir_move_type_query_t {
  // Move availability query shared across every source type reference.
  const loom_ir_move_availability_t* availability;
  // False once any walked type reference is unavailable at the insertion point.
  bool available;
} loom_ir_move_type_query_t;

static bool loom_ir_move_value_is_available(
    const loom_ir_move_availability_t* query, loom_value_id_t target_value) {
  if (loom_ir_move_value_moves_with_block(
          query->analysis->module, query->source_block, query->omit_terminators,
          target_value)) {
    return true;
  }
  return loom_availability_value_is_available_before_op(
      query->analysis, /*moving_root_op=*/NULL, query->before_op, target_value);
}

static iree_status_t loom_ir_move_remapped_value_is_available(
    const loom_ir_move_availability_t* query, loom_value_id_t source_value,
    bool* out_available) {
  *out_available = false;
  if (source_value == LOOM_VALUE_ID_INVALID) return iree_ok_status();
  loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_resolve_value(query->remap, source_value, &target_value));
  *out_available = loom_ir_move_value_is_available(query, target_value);
  return iree_ok_status();
}

static iree_status_t loom_ir_move_check_type_ref(loom_value_id_t source_value,
                                                 void* user_data) {
  loom_ir_move_type_query_t* query = (loom_ir_move_type_query_t*)user_data;
  bool available = false;
  IREE_RETURN_IF_ERROR(loom_ir_move_remapped_value_is_available(
      query->availability, source_value, &available));
  if (!available) query->available = false;
  return iree_ok_status();
}

static iree_status_t loom_ir_move_remapped_type_is_available(
    const loom_ir_move_availability_t* query, loom_type_t source_type,
    bool* out_available) {
  *out_available = false;
  loom_ir_move_type_query_t type_query = {
      .availability = query,
      .available = true,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      source_type, loom_ir_move_check_type_ref, &type_query));
  *out_available = type_query.available;
  return iree_ok_status();
}

static iree_status_t loom_ir_move_predicate_list_is_available(
    const loom_ir_move_availability_t* query, loom_attribute_t attr,
    bool* out_available) {
  *out_available = false;
  if (attr.count > 0 && !attr.predicate_list) return iree_ok_status();
  for (uint16_t i = 0; i < attr.count; ++i) {
    const loom_predicate_t* predicate = &attr.predicate_list[i];
    for (uint8_t j = 0; j < IREE_ARRAYSIZE(predicate->arg_tags); ++j) {
      if (predicate->arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
      if (predicate->args[j] < 0) return iree_ok_status();
      loom_value_id_t value_id = (loom_value_id_t)predicate->args[j];
      if (!loom_ir_move_source_value_id_is_valid(query->remap, value_id)) {
        return iree_ok_status();
      }
      bool available = false;
      IREE_RETURN_IF_ERROR(loom_ir_move_remapped_value_is_available(
          query, value_id, &available));
      if (!available) return iree_ok_status();
    }
  }
  *out_available = true;
  return iree_ok_status();
}

static iree_status_t loom_ir_move_attr_is_available(
    const loom_ir_move_availability_t* query, const loom_attribute_t* attr,
    uint8_t depth, bool* out_available) {
  *out_available = false;
  if (depth > LOOM_ATTR_DICT_MAX_NESTING_DEPTH || !attr) {
    return iree_ok_status();
  }
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_STRING:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_I64_ARRAY:
    case LOOM_ATTR_SYMBOL:
      *out_available = true;
      return iree_ok_status();
    case LOOM_ATTR_TYPE:
      if (attr->type_id == LOOM_TYPE_ID_INVALID ||
          attr->type_id >= query->remap->source_module->types.count) {
        return iree_ok_status();
      }
      return loom_ir_move_remapped_type_is_available(
          query, query->remap->source_module->types.entries[attr->type_id],
          out_available);
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_ir_move_predicate_list_is_available(query, *attr,
                                                      out_available);
    case LOOM_ATTR_DICT:
      if (attr->count > 0 && !attr->dict_entries) return iree_ok_status();
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_ir_move_attr_is_available(
            query, &attr->dict_entries[i].value, (uint8_t)(depth + 1),
            out_available));
        if (!*out_available) return iree_ok_status();
      }
      *out_available = true;
      return iree_ok_status();
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(query->remap->source_module, attr->encoding_id);
      if (!encoding) return iree_ok_status();
      if (encoding->attribute_count > 0 && !encoding->attributes) {
        return iree_ok_status();
      }
      for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_ir_move_attr_is_available(
            query, &encoding->attributes[i].value, (uint8_t)(depth + 1),
            out_available));
        if (!*out_available) return iree_ok_status();
      }
      *out_available = true;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_ir_move_block_arg_types_are_available(
    const loom_ir_move_availability_t* query, const loom_block_t* block,
    bool* out_available) {
  *out_available = false;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t arg_id = loom_block_arg_id(block, i);
    if (!loom_ir_move_source_value_id_is_valid(query->remap, arg_id)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_ir_move_remapped_type_is_available(
        query, loom_module_value_type(query->remap->source_module, arg_id),
        out_available));
    if (!*out_available) return iree_ok_status();
  }
  *out_available = true;
  return iree_ok_status();
}

static iree_status_t loom_ir_move_op_captures_are_available(
    const loom_ir_move_availability_t* query, const loom_op_t* op,
    bool* out_available) {
  *out_available = false;

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_move_remapped_value_is_available(
        query, operands[i], out_available));
    if (!*out_available) return iree_ok_status();
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (!loom_ir_move_source_value_id_is_valid(query->remap, result)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_ir_move_remapped_type_is_available(
        query, loom_module_value_type(query->remap->source_module, result),
        out_available));
    if (!*out_available) return iree_ok_status();
  }

  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_move_attr_is_available(query, &attrs[i], 0, out_available));
    if (!*out_available) return iree_ok_status();
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(loom_ir_move_block_arg_types_are_available(
          query, block, out_available));
      if (!*out_available) return iree_ok_status();
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(loom_ir_move_op_captures_are_available(
            query, child_op, out_available));
        if (!*out_available) return iree_ok_status();
      }
    }
  }

  *out_available = true;
  return iree_ok_status();
}

static iree_status_t loom_ir_move_block_captures_are_available(
    const loom_ir_move_availability_t* query, bool* out_available) {
  *out_available = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(query->source_block, op) {
    if (loom_ir_move_should_omit_op(query->analysis->module, op,
                                    query->omit_terminators)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_op_subtree_values_to_self(query->remap, op));
    IREE_RETURN_IF_ERROR(
        loom_ir_move_op_captures_are_available(query, op, out_available));
    if (!*out_available) return iree_ok_status();
  }
  *out_available = true;
  return iree_ok_status();
}

iree_status_t loom_ir_move_block_ops_before(
    loom_rewriter_t* rewriter, loom_block_t* source_block, loom_op_t* before_op,
    loom_ir_remap_t* remap, const loom_ir_move_block_options_t* options) {
  if (remap->source_module != rewriter->module ||
      remap->target_module != rewriter->module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "move materialization requires a same-module remap");
  }
  if (iree_any_bit_set(before_op->flags, LOOM_OP_FLAG_DEAD) ||
      !before_op->parent_block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "move materialization requires a live insertion op");
  }
  if (source_block == before_op->parent_block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "move materialization requires distinct source and insertion blocks");
  }

  bool omit_terminators = options ? options->omit_terminators : false;
  loom_availability_analysis_t availability = {0};
  IREE_RETURN_IF_ERROR(loom_availability_analysis_initialize(
      rewriter->module, rewriter->arena, &availability));
  loom_ir_move_availability_t query = {
      .analysis = &availability,
      .remap = remap,
      .source_block = source_block,
      .before_op = before_op,
      .omit_terminators = omit_terminators,
  };
  bool available = false;
  IREE_RETURN_IF_ERROR(
      loom_ir_move_block_captures_are_available(&query, &available));
  if (!available) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot move block before insertion point: remapped captures are not "
        "available");
  }

  loom_op_t* source_op = source_block->first_op;
  while (source_op) {
    loom_op_t* next_op = source_op->next_op;
    if (iree_any_bit_set(source_op->flags, LOOM_OP_FLAG_DEAD)) {
      source_op = next_op;
      continue;
    }
    if (loom_ir_move_should_omit_op(rewriter->module, source_op,
                                    omit_terminators)) {
      source_op = next_op;
      continue;
    }

    IREE_RETURN_IF_ERROR(
        loom_ir_remap_op_references(rewriter, source_op, remap));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, source_op, before_op));
    source_op = next_op;
  }

  return iree_ok_status();
}
