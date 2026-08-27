// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/low_mix.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/reporting/low_names.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/cfg_loop.h"

static bool loom_target_compile_report_low_branch_falls_through(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, const loom_block_t* dest) {
  const uint32_t dest_block_index = loom_low_packet_block_index(schedule, dest);
  return dest_block_index != LOOM_LOW_PACKET_INDEX_NONE &&
         dest_block_index == node->block_index + 1;
}

static uint64_t
loom_target_compile_report_low_structural_control_transfer_count(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  if (node->op == NULL) {
    return 0;
  }
  if (loom_low_return_isa(node->op)) {
    return 1;
  }
  if (loom_low_br_isa(node->op)) {
    return loom_target_compile_report_low_branch_falls_through(
               schedule, node, loom_low_br_dest(node->op))
               ? 0
               : 1;
  }
  if (!loom_low_cond_br_isa(node->op)) {
    return 0;
  }

  const loom_block_t* true_dest = loom_low_cond_br_true_dest(node->op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(node->op);
  const bool true_fallthrough =
      loom_target_compile_report_low_branch_falls_through(schedule, node,
                                                          true_dest);
  if (true_dest == false_dest) {
    return true_fallthrough ? 0 : 1;
  }
  const bool false_fallthrough =
      loom_target_compile_report_low_branch_falls_through(schedule, node,
                                                          false_dest);
  return true_fallthrough || false_fallthrough ? 1 : 2;
}

static uint64_t loom_target_compile_report_low_structural_move_count(
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_node_t* node) {
  if (node->op == NULL) {
    return 0;
  }
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR &&
      loom_low_br_isa(node->op)) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            allocation, node->source_ordinal);
    return group != NULL ? group->move_group.moves.count : 0;
  }
  if (!loom_low_copy_isa(node->op) && !loom_low_move_isa(node->op) &&
      !loom_low_slice_isa(node->op) && !loom_low_concat_isa(node->op)) {
    return 0;
  }
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          allocation, node->source_ordinal);
  return group != NULL ? group->move_group.moves.count : 0;
}

static uint64_t loom_target_compile_report_low_effect_byte_count(
    const loom_low_effect_t* effect, bool* out_known) {
  *out_known = effect->width_bits != 0 && (effect->width_bits % 8u) == 0;
  return *out_known ? effect->width_bits / 8u : 0;
}

static void loom_target_compile_report_accumulate_low_memory_effect(
    const loom_low_effect_t* effect,
    loom_low_instruction_class_flags_t instruction_classes,
    loom_target_compile_report_static_instruction_mix_t* mix) {
  if (effect->kind != LOOM_LOW_EFFECT_KIND_READ &&
      effect->kind != LOOM_LOW_EFFECT_KIND_WRITE) {
    return;
  }

  bool known_byte_width = false;
  const uint64_t byte_count = loom_target_compile_report_low_effect_byte_count(
      effect, &known_byte_width);
  uint64_t* total_byte_count = NULL;
  uint64_t* family_byte_count = NULL;
  if (effect->kind == LOOM_LOW_EFFECT_KIND_READ) {
    if (!known_byte_width) {
      ++mix->memory_read_unknown_width_count;
      return;
    }
    total_byte_count = &mix->memory_read_byte_count;
    if (iree_all_bits_set(instruction_classes,
                          LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_LOAD)) {
      family_byte_count = &mix->global_load_byte_count;
    } else if (iree_all_bits_set(instruction_classes,
                                 LOOM_LOW_INSTRUCTION_CLASS_FLAG_BUFFER_LOAD)) {
      family_byte_count = &mix->buffer_load_byte_count;
    } else if (iree_all_bits_set(instruction_classes,
                                 LOOM_LOW_INSTRUCTION_CLASS_FLAG_FLAT_MEMORY)) {
      family_byte_count = &mix->flat_read_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_LOCAL_MEMORY)) {
      family_byte_count = &mix->local_read_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_MEMORY)) {
      family_byte_count = &mix->scalar_read_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_PRIVATE_MEMORY)) {
      family_byte_count = &mix->private_read_byte_count;
    } else if (iree_any_bit_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_GENERIC_MEMORY |
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_MEMORY |
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_ATOMIC)) {
      family_byte_count = &mix->unclassified_read_byte_count;
    }
  } else {
    if (!known_byte_width) {
      ++mix->memory_write_unknown_width_count;
      return;
    }
    total_byte_count = &mix->memory_write_byte_count;
    if (iree_all_bits_set(instruction_classes,
                          LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_STORE)) {
      family_byte_count = &mix->global_store_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_BUFFER_STORE)) {
      family_byte_count = &mix->buffer_store_byte_count;
    } else if (iree_all_bits_set(instruction_classes,
                                 LOOM_LOW_INSTRUCTION_CLASS_FLAG_FLAT_MEMORY)) {
      family_byte_count = &mix->flat_write_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_LOCAL_MEMORY)) {
      family_byte_count = &mix->local_write_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_MEMORY)) {
      family_byte_count = &mix->scalar_write_byte_count;
    } else if (iree_all_bits_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_PRIVATE_MEMORY)) {
      family_byte_count = &mix->private_write_byte_count;
    } else if (iree_any_bit_set(
                   instruction_classes,
                   LOOM_LOW_INSTRUCTION_CLASS_FLAG_GENERIC_MEMORY |
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_MEMORY |
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_ATOMIC)) {
      family_byte_count = &mix->unclassified_write_byte_count;
    }
  }
  *total_byte_count += byte_count;
  if (family_byte_count != NULL) {
    *family_byte_count += byte_count;
  }
}

static void loom_target_compile_report_accumulate_low_memory_effects(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    loom_target_compile_report_static_instruction_mix_t* mix) {
  if (descriptor->effect_count == 0) {
    return;
  }
  IREE_ASSERT_LE((uint64_t)descriptor->effect_start + descriptor->effect_count,
                 descriptor_set->effect_count);
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const uint32_t effect_index = descriptor->effect_start + i;
    const loom_low_effect_t* effect = &descriptor_set->effects[effect_index];
    loom_target_compile_report_accumulate_low_memory_effect(
        effect, descriptor->instruction_class_flags, mix);
  }
}

void loom_target_compile_report_accumulate_low_node_static_mix(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_static_instruction_mix_t* mix) {
  mix->register_move_count +=
      loom_target_compile_report_low_structural_move_count(allocation, node);
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
    const uint64_t control_transfer_count =
        loom_target_compile_report_low_structural_control_transfer_count(
            schedule, node);
    mix->branch_count += control_transfer_count;
    mix->control_count += control_transfer_count;
    return;
  }
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
      node->descriptor == NULL) {
    return;
  }
  ++mix->descriptor_count;
  const loom_low_instruction_class_flags_t instruction_classes =
      node->descriptor->instruction_class_flags;
  loom_target_compile_report_accumulate_low_memory_effects(
      descriptor_set, node->descriptor, mix);
#define LOOM_ACCUMULATE_INSTRUCTION_CLASS_(field, flag)         \
  mix->field##_count +=                                         \
      iree_all_bits_set(instruction_classes,                    \
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_##flag) \
          ? 1                                                   \
          : 0
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(unknown, OTHER);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(scalar_alu, SCALAR_ALU);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(vector_alu, VECTOR_ALU);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(matrix, MATRIX);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(mfma, MFMA);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(smfmac, SMFMAC);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(wmma, WMMA);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(swmmac, SWMMAC);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(dot, DOT);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(global_memory, GLOBAL_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(global_load, GLOBAL_LOAD);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(global_store, GLOBAL_STORE);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(buffer_load, BUFFER_LOAD);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(buffer_store, BUFFER_STORE);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(flat_memory, FLAT_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(local_memory, LOCAL_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(scalar_memory, SCALAR_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(private_memory, PRIVATE_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(generic_memory, GENERIC_MEMORY);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(atomic, ATOMIC);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(branch, BRANCH);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(barrier, BARRIER);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(control, CONTROL);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(conversion, CONVERSION);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(cache, CACHE);
  LOOM_ACCUMULATE_INSTRUCTION_CLASS_(register_move, REGISTER_MOVE);
#undef LOOM_ACCUMULATE_INSTRUCTION_CLASS_
}

void loom_target_compile_report_accumulate_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source) {
  target->descriptor_count += source->descriptor_count;
  target->unknown_count += source->unknown_count;
  target->scalar_alu_count += source->scalar_alu_count;
  target->vector_alu_count += source->vector_alu_count;
  target->matrix_count += source->matrix_count;
  target->mfma_count += source->mfma_count;
  target->smfmac_count += source->smfmac_count;
  target->wmma_count += source->wmma_count;
  target->swmmac_count += source->swmmac_count;
  target->dot_count += source->dot_count;
  target->global_memory_count += source->global_memory_count;
  target->global_load_count += source->global_load_count;
  target->global_store_count += source->global_store_count;
  target->buffer_load_count += source->buffer_load_count;
  target->buffer_store_count += source->buffer_store_count;
  target->flat_memory_count += source->flat_memory_count;
  target->local_memory_count += source->local_memory_count;
  target->scalar_memory_count += source->scalar_memory_count;
  target->private_memory_count += source->private_memory_count;
  target->generic_memory_count += source->generic_memory_count;
  target->memory_read_unknown_width_count +=
      source->memory_read_unknown_width_count;
  target->memory_write_unknown_width_count +=
      source->memory_write_unknown_width_count;
  target->memory_read_byte_count += source->memory_read_byte_count;
  target->memory_write_byte_count += source->memory_write_byte_count;
  target->global_load_byte_count += source->global_load_byte_count;
  target->global_store_byte_count += source->global_store_byte_count;
  target->buffer_load_byte_count += source->buffer_load_byte_count;
  target->buffer_store_byte_count += source->buffer_store_byte_count;
  target->flat_read_byte_count += source->flat_read_byte_count;
  target->flat_write_byte_count += source->flat_write_byte_count;
  target->local_read_byte_count += source->local_read_byte_count;
  target->local_write_byte_count += source->local_write_byte_count;
  target->scalar_read_byte_count += source->scalar_read_byte_count;
  target->scalar_write_byte_count += source->scalar_write_byte_count;
  target->private_read_byte_count += source->private_read_byte_count;
  target->private_write_byte_count += source->private_write_byte_count;
  target->unclassified_read_byte_count += source->unclassified_read_byte_count;
  target->unclassified_write_byte_count +=
      source->unclassified_write_byte_count;
  target->atomic_count += source->atomic_count;
  target->branch_count += source->branch_count;
  target->barrier_count += source->barrier_count;
  target->control_count += source->control_count;
  target->conversion_count += source->conversion_count;
  target->cache_count += source->cache_count;
  target->register_move_count += source->register_move_count;
}

bool loom_target_compile_report_accumulate_scaled_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source,
    uint64_t scale) {
#define LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(field)             \
  do {                                                                        \
    uint64_t scaled_value = 0;                                                \
    if (!iree_checked_mul_u64(source->field, scale, &scaled_value) ||         \
        !iree_checked_add_u64(target->field, scaled_value, &target->field)) { \
      return false;                                                           \
    }                                                                         \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(descriptor_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(unknown_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(vector_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(matrix_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(mfma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(smfmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(wmma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(swmmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(dot_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(generic_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      memory_read_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      memory_write_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(memory_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(memory_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      unclassified_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      unclassified_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(atomic_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(branch_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(barrier_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(control_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(conversion_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(cache_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(register_move_count);
#undef LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD
  return true;
}

static bool loom_target_compile_report_low_exact_trip_count(
    const loom_value_fact_table_t* fact_table, loom_loop_like_t loop,
    uint64_t* out_trip_count) {
  *out_trip_count = 0;
  if (!loom_loop_like_isa(loop) || !loom_loop_like_has_counted_range(loop)) {
    return false;
  }
  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table,
                                       loom_loop_like_lower_bound(loop)),
          &lower_bound) ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table,
                                       loom_loop_like_upper_bound(loop)),
          &upper_bound) ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, loom_loop_like_step(loop)),
          &step) ||
      step <= 0) {
    return false;
  }
  if (upper_bound <= lower_bound) {
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper_bound, lower_bound, &span)) {
    return false;
  }
  const uint64_t unsigned_span = (uint64_t)span;
  const uint64_t unsigned_step = (uint64_t)step;
  *out_trip_count = ((unsigned_span - 1) / unsigned_step) + 1;
  return true;
}

static const loom_low_descriptor_t*
loom_target_compile_report_schedule_descriptor_for_op(
    const loom_low_schedule_table_t* schedule, const loom_op_t* op) {
  if (schedule == NULL || op == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->op == op) {
      return node->descriptor;
    }
  }
  return NULL;
}

static iree_string_view_t loom_target_compile_report_low_op_semantic_tag(
    const loom_low_schedule_table_t* schedule, const loom_op_t* op) {
  return loom_target_compile_report_descriptor_semantic_tag(
      schedule->target.descriptor_set,
      loom_target_compile_report_schedule_descriptor_for_op(schedule, op));
}

static bool loom_target_compile_report_value_exact_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  *out_value = 0;
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_value_facts_as_exact_i64(facts, out_value)) {
    return true;
  }
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  return loom_value_facts_query_all_equal_element(&fact_table->context, facts,
                                                  &element_facts) &&
         loom_value_facts_as_exact_i64(element_facts, out_value);
}

static const loom_named_attr_t* loom_target_compile_report_low_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id < module->strings.count &&
        iree_string_view_equal(module->strings.entries[attr->name_id], name)) {
      return attr;
    }
  }
  return NULL;
}

static bool loom_target_compile_report_low_op_i64_attr(
    const loom_module_t* module, const loom_op_t* op, iree_string_view_t name,
    int64_t* out_value) {
  const loom_named_attr_t* attr = loom_target_compile_report_low_find_attr(
      module, loom_low_op_attrs(op), name);
  if (attr == NULL || attr->value.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = loom_attr_as_i64(attr->value);
  return true;
}

static bool loom_target_compile_report_low_block_branch_to(
    const loom_block_t* block, const loom_block_t* dest,
    const loom_op_t** out_br) {
  const loom_op_t* terminator = block != NULL ? block->last_op : NULL;
  if (terminator == NULL || !loom_low_br_isa(terminator) ||
      loom_low_br_dest(terminator) != dest) {
    return false;
  }
  *out_br = terminator;
  return true;
}

static bool loom_target_compile_report_low_edge_copy_branch_arg(
    const loom_low_allocation_table_t* allocation,
    uint32_t branch_source_ordinal, const loom_block_t* dest,
    uint16_t arg_index, loom_value_id_t* out_value_id) {
  if (allocation == NULL || dest == NULL || arg_index >= dest->arg_count) {
    return false;
  }
  const loom_value_id_t destination_value_id =
      loom_block_arg_id(dest, arg_index);
  const loom_low_allocation_edge_copy_group_t* group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          allocation, branch_source_ordinal);
  if (group == NULL) return false;
  bool found = false;
  loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
  const iree_host_size_t copy_end =
      (iree_host_size_t)group->copy_start + (iree_host_size_t)group->copy_count;
  if (copy_end > allocation->edge_copy_count) return false;
  for (iree_host_size_t i = group->copy_start; i < copy_end; ++i) {
    const loom_low_allocation_edge_copy_t* copy = &allocation->edge_copies[i];
    if (copy->payload_index != arg_index ||
        copy->destination_value_id != destination_value_id) {
      continue;
    }
    if (found && copy->source_value_id != source_value_id) return false;
    found = true;
    source_value_id = copy->source_value_id;
  }
  if (!found) return false;
  *out_value_id = source_value_id;
  return true;
}

static bool loom_target_compile_report_low_branch_arg(
    const loom_low_allocation_table_t* allocation, const loom_op_t* branch_op,
    uint32_t branch_source_ordinal, const loom_block_t* dest,
    uint16_t arg_index, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  if (branch_op == NULL || arg_index >= dest->arg_count) {
    return false;
  }
  const loom_value_slice_t args = loom_low_br_args(branch_op);
  if (arg_index < args.count) {
    *out_value_id = args.values[arg_index];
    return true;
  }
  return loom_target_compile_report_low_edge_copy_branch_arg(
      allocation, branch_source_ordinal, dest, arg_index, out_value_id);
}

static bool loom_target_compile_report_low_add_step(
    const loom_low_schedule_table_t* schedule,
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t iv_id, int64_t* out_step) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || !loom_low_op_isa(defining_op)) {
    return false;
  }
  const iree_string_view_t tag =
      loom_target_compile_report_low_op_semantic_tag(schedule, defining_op);
  if (!iree_string_view_equal(tag, IREE_SV("integer.add.u32")) &&
      !iree_string_view_equal(tag, IREE_SV("integer.add.i32"))) {
    return false;
  }
  loom_value_slice_t operands = loom_low_op_operands(defining_op);
  if (operands.count == 1 && operands.values[0] == iv_id) {
    return loom_target_compile_report_low_op_i64_attr(
        module, defining_op, IREE_SV("imm32"), out_step);
  }
  if (operands.count != 2) {
    return false;
  }
  if (operands.values[0] == iv_id) {
    return loom_target_compile_report_value_exact_i64(
        fact_table, operands.values[1], out_step);
  }
  if (operands.values[1] == iv_id) {
    return loom_target_compile_report_value_exact_i64(
        fact_table, operands.values[0], out_step);
  }
  return false;
}

static bool loom_target_compile_report_low_header_upper_bound(
    const loom_low_schedule_table_t* schedule,
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    const loom_op_t* cond_br_op, loom_value_id_t iv_id,
    int64_t* out_upper_bound, bool* out_inclusive_bound) {
  *out_inclusive_bound = false;
  const loom_value_id_t condition = loom_low_cond_br_condition(cond_br_op);
  if (condition >= module->values.count) {
    return false;
  }
  const loom_value_t* condition_value = loom_module_value(module, condition);
  if (loom_value_is_block_arg(condition_value)) {
    return false;
  }
  const loom_op_t* compare_op = loom_value_def_op(condition_value);
  if (compare_op == NULL || !loom_low_op_isa(compare_op)) {
    return false;
  }
  const iree_string_view_t tag =
      loom_target_compile_report_low_op_semantic_tag(schedule, compare_op);
  const bool exclusive_upper =
      iree_string_view_equal(tag, IREE_SV("integer.compare.slt.i32")) ||
      iree_string_view_equal(tag, IREE_SV("integer.compare.ult.i32"));
  const bool inclusive_upper =
      iree_string_view_equal(tag, IREE_SV("integer.compare.sle.i32")) ||
      iree_string_view_equal(tag, IREE_SV("integer.compare.ule.i32"));
  if (!exclusive_upper && !inclusive_upper) {
    return false;
  }
  loom_value_slice_t operands = loom_low_op_operands(compare_op);
  if (operands.count != 2 || operands.values[0] != iv_id) {
    return false;
  }
  if (!loom_target_compile_report_value_exact_i64(
          fact_table, operands.values[1], out_upper_bound)) {
    return false;
  }
  *out_inclusive_bound = inclusive_upper;
  return true;
}

static bool loom_target_compile_report_low_compute_trip_count(
    int64_t lower_bound, int64_t upper_bound, bool inclusive_upper_bound,
    int64_t step, uint64_t* out_trip_count) {
  *out_trip_count = 0;
  if (step <= 0) {
    return false;
  }
  if (inclusive_upper_bound) {
    if (upper_bound == INT64_MAX) {
      return false;
    }
    ++upper_bound;
  }
  if (upper_bound <= lower_bound) {
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper_bound, lower_bound, &span)) {
    return false;
  }
  const uint64_t unsigned_span = (uint64_t)span;
  const uint64_t unsigned_step = (uint64_t)step;
  *out_trip_count = ((unsigned_span - 1) / unsigned_step) + 1;
  return true;
}

static uint32_t loom_target_compile_report_low_block_terminator_ordinal(
    const loom_low_schedule_table_t* schedule, uint16_t block_index) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  IREE_ASSERT_GT(block->node_count, 0);
  const uint32_t node_index = block->node_start + block->node_count - 1;
  IREE_ASSERT_LT(node_index, schedule->node_count);
  IREE_ASSERT(schedule->nodes[node_index].op == block->block->last_op);
  return schedule->nodes[node_index].source_ordinal;
}

static bool loom_target_compile_report_low_try_counted_loop(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    const loom_cfg_graph_t* graph, const loom_cfg_loop_interval_t* interval,
    uint64_t* out_trip_count) {
  *out_trip_count = 0;
  const loom_block_t* header = graph->blocks[interval->header_index].block;
  if (header == NULL || header->arg_count == 0 || header->last_op == NULL ||
      !loom_low_cond_br_isa(header->last_op)) {
    return false;
  }
  const loom_value_id_t iv_id = loom_block_arg_id(header, 0);
  const loom_cfg_edge_info_t* entry_edge =
      loom_cfg_graph_edge(graph, interval->entry_edge_index);
  const loom_cfg_edge_info_t* backedge =
      loom_cfg_graph_edge(graph, interval->backedge_edge_index);
  if (entry_edge == NULL || backedge == NULL) return false;

  const loom_op_t* initial_branch_op = NULL;
  if (!loom_target_compile_report_low_block_branch_to(
          graph->blocks[entry_edge->source_block_index].block, header,
          &initial_branch_op)) {
    return false;
  }
  const loom_op_t* body_backedge_op = NULL;
  if (!loom_target_compile_report_low_block_branch_to(
          graph->blocks[backedge->source_block_index].block, header,
          &body_backedge_op)) {
    return false;
  }

  const uint32_t backedge_source_ordinal =
      loom_target_compile_report_low_block_terminator_ordinal(
          schedule, backedge->source_block_index);
  loom_value_id_t body_backedge_arg = LOOM_VALUE_ID_INVALID;
  if (!loom_target_compile_report_low_branch_arg(
          allocation, body_backedge_op, backedge_source_ordinal, header,
          /*arg_index=*/0, &body_backedge_arg)) {
    return false;
  }

  const uint32_t entry_source_ordinal =
      loom_target_compile_report_low_block_terminator_ordinal(
          schedule, entry_edge->source_block_index);
  loom_value_id_t initial_arg = LOOM_VALUE_ID_INVALID;
  if (!loom_target_compile_report_low_branch_arg(
          allocation, initial_branch_op, entry_source_ordinal, header,
          /*arg_index=*/0, &initial_arg)) {
    return false;
  }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  bool inclusive_upper_bound = false;
  bool lower_ok = loom_target_compile_report_value_exact_i64(
      fact_table, initial_arg, &lower_bound);
  bool step_ok = loom_target_compile_report_low_add_step(
      schedule, fact_table, module, body_backedge_arg, iv_id, &step);
  bool upper_ok = loom_target_compile_report_low_header_upper_bound(
      schedule, fact_table, module, header->last_op, iv_id, &upper_bound,
      &inclusive_upper_bound);
  bool trip_ok = loom_target_compile_report_low_compute_trip_count(
      lower_bound, upper_bound, inclusive_upper_bound, step, out_trip_count);
  return lower_ok && step_ok && upper_ok && trip_ok;
}

static iree_status_t loom_target_compile_report_low_block_multipliers(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    iree_arena_allocator_t* arena, uint64_t** out_block_multipliers,
    bool* out_exact) {
  *out_block_multipliers = NULL;
  *out_exact = true;
  if (schedule->block_count == 0) {
    return iree_ok_status();
  }
  uint64_t* block_multipliers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                 sizeof(*block_multipliers),
                                                 (void**)&block_multipliers));
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  const loom_cfg_loop_forest_t* loop_forest = &schedule->loop_forest;
  if (graph->malformed || graph->block_count != schedule->block_count) {
    *out_exact = false;
    return iree_ok_status();
  }
  uint64_t* trip_counts = NULL;
  if (loop_forest->interval_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, loop_forest->interval_count,
                                  sizeof(*trip_counts), (void**)&trip_counts));
  }
  for (iree_host_size_t i = 0; i < loop_forest->interval_count; ++i) {
    if (!loom_target_compile_report_low_try_counted_loop(
            schedule, allocation, fact_table, module, graph,
            &loop_forest->intervals[i], &trip_counts[i])) {
      *out_exact = false;
      return iree_ok_status();
    }
  }
  *out_exact = loom_cfg_loop_forest_calculate_block_execution_counts(
      loop_forest, graph, trip_counts, block_multipliers);
  if (*out_exact) {
    *out_block_multipliers = block_multipliers;
  }
  return iree_ok_status();
}

bool loom_target_compile_report_low_node_execution_multiplier(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const uint64_t* block_multipliers, const loom_low_schedule_node_t* node,
    uint64_t* out_multiplier) {
  *out_multiplier = block_multipliers != NULL &&
                            node->block_index != LOOM_LOW_PACKET_INDEX_NONE
                        ? block_multipliers[node->block_index]
                        : 1;
  const loom_op_t* op = node->op;
  for (const loom_op_t* parent = op ? op->parent_op : NULL; parent;
       parent = parent->parent_op) {
    loom_loop_like_t loop = loom_loop_like_cast(module, (loom_op_t*)parent);
    if (!loom_loop_like_isa(loop)) {
      continue;
    }
    uint64_t trip_count = 0;
    if (!loom_target_compile_report_low_exact_trip_count(fact_table, loop,
                                                         &trip_count) ||
        !iree_checked_mul_u64(*out_multiplier, trip_count, out_multiplier)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_target_compile_report_low_dynamic_context_initialize(
    const loom_low_emission_frame_t* frame,
    loom_target_compile_report_low_dynamic_context_t* out_context) {
  *out_context = (loom_target_compile_report_low_dynamic_context_t){0};
  if (frame->module == NULL || frame->function_op == NULL ||
      frame->module->arena.block_pool == NULL) {
    return iree_ok_status();
  }

  iree_arena_initialize(frame->module->arena.block_pool, &out_context->arena);
  out_context->initialized = true;
  iree_status_t status = loom_value_fact_table_initialize(
      &out_context->fact_table, &out_context->arena,
      frame->module->values.count);
  loom_func_like_t function =
      loom_func_like_cast(frame->module, (loom_op_t*)frame->function_op);
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_compute_region(
        &out_context->fact_table, frame->module, function,
        (loom_region_t*)loom_low_function_const_body(frame->function_op),
        (loom_op_t*)frame->function_op);
  }
  bool block_multipliers_exact = true;
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_low_block_multipliers(
        &frame->schedule, &frame->allocation, &out_context->fact_table,
        frame->module, &out_context->arena, &out_context->block_multipliers,
        &block_multipliers_exact);
  }
  out_context->exact = iree_status_is_ok(status) && block_multipliers_exact;
  return status;
}

void loom_target_compile_report_low_dynamic_context_deinitialize(
    loom_target_compile_report_low_dynamic_context_t* context) {
  if (context->initialized) {
    iree_arena_deinitialize(&context->arena);
  }
  *context = (loom_target_compile_report_low_dynamic_context_t){0};
}

void loom_target_compile_report_record_low_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_target_compile_report_static_instruction_mix_t mix = {0};
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    loom_target_compile_report_accumulate_low_node_static_mix(
        &frame->schedule, &frame->allocation, descriptor_set, node, &mix);
  }
  loom_target_compile_report_record_static_instruction_mix(report, &mix);
}

iree_status_t loom_target_compile_report_record_low_dynamic_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context) {
  if (dynamic_context == NULL || !dynamic_context->exact) {
    return iree_ok_status();
  }
  loom_target_compile_report_static_instruction_mix_t mix = {0};
  bool exact = true;
  for (iree_host_size_t i = 0; exact && i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    uint64_t multiplier = 1;
    exact = loom_target_compile_report_low_node_execution_multiplier(
        frame->module, &dynamic_context->fact_table,
        dynamic_context->block_multipliers, node, &multiplier);
    if (!exact) {
      break;
    }
    loom_target_compile_report_static_instruction_mix_t node_mix = {0};
    loom_target_compile_report_accumulate_low_node_static_mix(
        &frame->schedule, &frame->allocation,
        frame->schedule.target.descriptor_set, node, &node_mix);
    exact = loom_target_compile_report_accumulate_scaled_static_mix(
        &mix, &node_mix, multiplier);
  }
  if (exact) {
    loom_target_compile_report_record_dynamic_instruction_mix(report, &mix);
  }
  return iree_ok_status();
}
