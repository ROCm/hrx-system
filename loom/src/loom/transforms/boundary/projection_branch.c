// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/projection_branch.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/rewrite/remap.h"

static loom_boundary_projection_slot_t* loom_boundary_projection_branch_slot(
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_branch_t* branch, uint16_t result_index) {
  const iree_host_size_t index = branch->candidates[result_index];
  if (index == IREE_HOST_SIZE_MAX || !function->candidates[index].selected) {
    return NULL;
  }
  return &function->candidates[index];
}

iree_status_t loom_boundary_projection_collect_branch(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_region_branch_t branch) {
  bool claimable = false;
  for (uint16_t i = 0; i < branch.op->result_count; ++i) {
    const loom_value_id_t value = loom_op_const_results(branch.op)[i];
    if (!loom_boundary_projection_may_claim_slot(
            plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_BRANCH_RESULT, value,
            /*block=*/NULL)) {
      continue;
    }
    claimable = true;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_provisional_slot(
        plan, function, value, LOOM_BOUNDARY_PROJECTION_SLOT_BRANCH_RESULT,
        /*block=*/NULL));
  }
  if (!claimable) {
    return iree_ok_status();
  }

  if (function->branch_count == function->branch_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->branch_count, function->branch_count + 1,
        sizeof(*function->branches), &function->branch_capacity,
        (void**)&function->branches));
  }
  loom_boundary_projection_branch_t* retained =
      &function->branches[function->branch_count++];
  *retained = (loom_boundary_projection_branch_t){.branch = branch};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, branch.op->region_count, sizeof(*retained->terminators),
      (void**)&retained->terminators));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, branch.op->result_count, sizeof(*retained->candidates),
      (void**)&retained->candidates));
  for (uint8_t i = 0; i < branch.op->region_count; ++i) {
    retained->terminators[i] =
        loom_region_branch_region_terminator(plan->module, branch, i);
  }
  return iree_ok_status();
}

void loom_boundary_projection_index_branches(
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t b = 0; b < function->branch_count; ++b) {
    loom_boundary_projection_branch_t* branch = &function->branches[b];
    const loom_op_t* op = branch->branch.op;
    for (uint16_t i = 0; i < op->result_count; ++i) {
      branch->candidates[i] = loom_boundary_projection_slot_index(
          function, loom_op_const_results(op)[i]);
    }
  }
}

iree_status_t loom_boundary_projection_select_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  // Select every result before planning yields so nested branches transport
  // components directly through the enclosing result's schema.
  for (iree_host_size_t b = 0; b < function->branch_count; ++b) {
    loom_boundary_projection_branch_t* branch = &function->branches[b];
    const loom_op_t* op = branch->branch.op;
    uint32_t result_count = 0;
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const loom_value_id_t value = loom_op_const_results(op)[i];
      const iree_host_size_t index = branch->candidates[i];
      loom_boundary_projection_schema_t schema = {0};
      bool claimed = false;
      if (index != IREE_HOST_SIZE_MAX) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
            plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_BRANCH_RESULT, value,
            /*block=*/NULL, &schema, &claimed));
      }
      if (claimed) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_finalize_provisional_slot(
            plan, function, index, &schema));
      }
      // Rejection may restore an eliminated result, so reserve its identity
      // arity even when the selected schema has no physical components.
      result_count += claimed ? iree_max(schema.component_count, 1) : 1;
    }
    if (result_count > UINT16_MAX) {
      for (uint16_t i = 0; i < op->result_count; ++i) {
        if (branch->candidates[i] != IREE_HOST_SIZE_MAX) {
          function->candidates[branch->candidates[i]].selected = false;
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_boundary_projection_plan_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t b = 0; b < function->branch_count; ++b) {
    loom_boundary_projection_branch_t* branch = &function->branches[b];
    const loom_op_t* op = branch->branch.op;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, (iree_host_size_t)op->result_count * op->region_count,
        sizeof(*branch->sources), (void**)&branch->sources));
    for (uint8_t region = 0; region < op->region_count; ++region) {
      loom_op_t* terminator = branch->terminators[region];
      for (uint16_t i = 0; i < op->result_count; ++i) {
        loom_boundary_projection_slot_t* slot =
            loom_boundary_projection_branch_slot(function, branch, i);
        if (!slot) {
          continue;
        }
        bool planned = false;
        IREE_RETURN_IF_ERROR(slot->schema.rule->transport.plan_source(
            slot->schema.rule, plan, function, slot, &slot->schema,
            loom_op_const_operands(terminator)[i], terminator,
            &branch->sources[(iree_host_size_t)region * op->result_count + i],
            &planned));
        if (!planned) {
          slot->selected = false;
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_boundary_projection_preallocate_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t b = 0; b < function->branch_count; ++b) {
    loom_boundary_projection_branch_t* branch = &function->branches[b];
    const loom_op_t* op = branch->branch.op;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, op->result_count, sizeof(*branch->offsets),
        (void**)&branch->offsets));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const loom_boundary_projection_slot_t* slot =
          loom_boundary_projection_branch_slot(function, branch, i);
      branch->offsets[i] = branch->result_count;
      branch->result_count += slot ? slot->schema.component_count : 1;
      branch->selected |= slot != NULL;
    }
    if (!branch->selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, branch->result_count, sizeof(*branch->result_ids),
        (void**)&branch->result_ids));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_boundary_projection_slot_t* slot =
          loom_boundary_projection_branch_slot(function, branch, i);
      const uint16_t count = slot ? slot->schema.component_count : 1;
      for (uint16_t component = 0; component < count; ++component) {
        loom_value_id_t* result =
            &branch->result_ids[branch->offsets[i] + component];
        IREE_RETURN_IF_ERROR(
            loom_module_define_value(plan->module, loom_type_none(), result));
        if (slot) {
          slot->component_value_ids[component] = *result;
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_branch_remap(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_branch_t* branch,
    loom_ir_remap_t* out_remap) {
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      plan->module, plan->module, plan->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      out_remap));
  const loom_op_t* source = branch->branch.op;
  for (uint16_t i = 0; i < source->result_count; ++i) {
    if (!loom_boundary_projection_branch_slot(function, branch, i)) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(out_remap, loom_op_const_results(source)[i],
                                  branch->result_ids[branch->offsets[i]]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_branch_result_types(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_branch_t* branch, loom_ir_remap_t* remap) {
  const loom_op_t* source = branch->branch.op;
  for (uint16_t i = 0; i < source->result_count; ++i) {
    const loom_boundary_projection_slot_t* slot =
        loom_boundary_projection_branch_slot(function, branch, i);
    const uint16_t count = slot ? slot->schema.component_count : 1;
    for (uint16_t component = 0; component < count; ++component) {
      const loom_type_t source_type =
          slot ? slot->schema.component_types[component]
               : loom_module_value_type(plan->module,
                                        loom_op_const_results(source)[i]);
      loom_type_t type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(remap, source_type, &type));
      IREE_RETURN_IF_ERROR(loom_module_set_value_type(
          plan->module, branch->result_ids[branch->offsets[i] + component],
          type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_branch_copy_comments(
    loom_module_t* module, const loom_op_t* source, const loom_op_t* target) {
  iree_host_size_t count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, source, &count);
  return count == 0
             ? iree_ok_status()
             : loom_module_attach_op_comments(module, target, comments, count);
}

static iree_status_t loom_boundary_projection_apply_branch(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_branch_t* branch) {
  loom_op_t* source = branch->branch.op;
  // RegionBranch results are forwarded by mutually exclusive yields; they do
  // not transfer linear ownership from an operand of the branch itself.
  IREE_ASSERT_EQ(source->tied_result_count, 0);
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_branch_remap(plan, function, branch, &remap));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_branch_result_types(
      plan, function, branch, &remap));
  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, source);
  loom_op_t* replacement = NULL;
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(
      loom_op_vtable(plan->module, source));
  IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
      builder, source->kind, source->operand_count,
      segment_count ? loom_op_const_operand_segment_counts(source) : NULL,
      segment_count, branch->result_count, source->region_count,
      /*tied_result_count=*/0, source->attribute_count, source->location,
      &replacement));
  replacement->instance_flags = source->instance_flags;
  replacement->traits = source->traits;
  replacement->intrinsic_write_spaces = source->intrinsic_write_spaces;
  replacement->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  memcpy(loom_op_operands(replacement), loom_op_const_operands(source),
         source->operand_count * sizeof(loom_value_id_t));
  if (source->attribute_count != 0) {
    memcpy(loom_op_attrs(replacement), loom_op_const_attrs(source),
           source->attribute_count * sizeof(loom_attribute_t));
  }
  if (branch->result_count != 0) {
    memcpy(loom_op_results(replacement), branch->result_ids,
           branch->result_count * sizeof(loom_value_id_t));
  }
  for (uint8_t region = 0; region < source->region_count; ++region) {
    loom_region_t* replacement_region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_module_allocate_region(plan->module, 1, &replacement_region));
    loom_op_set_region(replacement, region, replacement_region);
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, replacement));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_branch_copy_comments(
      plan->module, source, replacement));
  const loom_region_branch_t replacement_branch =
      loom_region_branch_cast(plan->module, replacement);
  loom_value_id_t* yields = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, branch->result_count, sizeof(*yields), (void**)&yields));
  for (uint8_t region = 0; region < source->region_count; ++region) {
    const loom_region_t* source_region = loom_op_regions(source)[region];
    loom_region_t* target_region = loom_op_regions(replacement)[region];
    target_region->flags = source_region->flags;
    target_region->source_flags = source_region->source_flags;
    loom_block_t* source_block =
        loom_region_entry_block(loom_op_regions(source)[region]);
    loom_block_t* target_block = loom_region_entry_block(target_region);
    target_block->label_id = source_block->label_id;
    target_block->flags = source_block->flags;
    iree_host_size_t comment_count = 0;
    const iree_string_view_t* comments =
        loom_module_block_comments(plan->module, source_block, &comment_count);
    if (comment_count != 0) {
      IREE_RETURN_IF_ERROR(loom_module_attach_block_comments(
          plan->module, target_block, comments, comment_count));
    }
    while (source_block->first_op != NULL) {
      IREE_RETURN_IF_ERROR(loom_rewriter_move_to_block_end(
          &plan->rewriter, source_block->first_op, target_block, replacement));
    }
    loom_op_t* terminator = branch->terminators[region];
    loom_builder_set_before(builder, terminator);
    for (uint16_t i = 0; i < source->result_count; ++i) {
      const loom_boundary_projection_slot_t* slot =
          loom_boundary_projection_branch_slot(function, branch, i);
      if (slot) {
        const loom_boundary_projection_source_t* recipe =
            &branch
                 ->sources[(iree_host_size_t)region * source->result_count + i];
        IREE_RETURN_IF_ERROR(slot->schema.rule->transport.materialize_source(
            slot->schema.rule, plan, function, recipe,
            yields ? yields + branch->offsets[i] : NULL));
      } else {
        yields[branch->offsets[i]] = loom_op_const_operands(terminator)[i];
      }
    }
    loom_op_t* new_terminator = NULL;
    IREE_RETURN_IF_ERROR(loom_region_branch_build_region_terminator(
        builder, plan->module, replacement_branch, region, yields,
        branch->result_count, terminator->location, &new_terminator));
    new_terminator->flags |=
        terminator->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_branch_copy_comments(
        plan->module, terminator, new_terminator));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(&plan->rewriter, terminator));
  }
  // Replace retained definitions first so projected result types can refer to
  // the new identities when their logical values are reconstructed.
  for (uint16_t i = 0; i < source->result_count; ++i) {
    if (!loom_boundary_projection_branch_slot(function, branch, i)) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_replace_definition(
          plan, function, loom_op_const_results(source)[i],
          branch->result_ids[branch->offsets[i]]));
    }
  }
  for (uint16_t i = 0; i < source->result_count; ++i) {
    loom_boundary_projection_slot_t* slot =
        loom_boundary_projection_branch_slot(function, branch, i);
    if (!slot) {
      continue;
    }
    loom_type_t logical_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, slot->logical_type, &logical_type));
    loom_builder_set_after(builder, replacement);
    if (slot->schema.destination_mode ==
        LOOM_BOUNDARY_PROJECTION_DESTINATION_RECONSTRUCT) {
      IREE_RETURN_IF_ERROR(slot->schema.rule->transport.reconstruct(
          slot->schema.rule, plan, function, slot, logical_type,
          source->location, &slot->replacement_value_id));
      IREE_RETURN_IF_ERROR(loom_boundary_projection_replace_definition(
          plan, function, slot->value_id, slot->replacement_value_id));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_boundary_projection_apply_branches(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t b = 0; b < function->branch_count; ++b) {
    if (function->branches[b].selected) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_apply_branch(
          plan, function, &function->branches[b]));
    }
  }
  return iree_ok_status();
}
