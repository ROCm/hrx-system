// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_classification.h"

#include <string.h>

#include "iree/base/bitfield.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

static iree_status_t loom_amdgpu_wait_classification_allocate(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_classification_t* classification) {
  const iree_host_size_t node_count = schedule->node_count;
  if (node_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*classification->node_states),
      (void**)&classification->node_states));
  memset(classification->node_states, 0,
         node_count * sizeof(*classification->node_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*classification->frontier_nodes),
      (void**)&classification->frontier_nodes));
  memset(classification->frontier_nodes, 0,
         node_count * sizeof(*classification->frontier_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*classification->completion_nodes),
      (void**)&classification->completion_nodes));
  memset(classification->completion_nodes, 0,
         node_count * sizeof(*classification->completion_nodes));
  return iree_ok_status();
}

static bool loom_amdgpu_wait_effect_is_dependency_memory(
    const loom_low_schedule_effect_use_t* effect_use) {
  if ((effect_use->effect_flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY) == 0) {
    return false;
  }
  switch (effect_use->memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return true;
    default:
      return false;
  }
}

static uint32_t loom_amdgpu_wait_effect_counter_mask(
    const loom_low_schedule_effect_use_t* effect) {
  if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    return 0;
  }
  return loom_amdgpu_wait_counter_mask(effect->counter_id);
}

static bool loom_amdgpu_wait_effect_requires_system_scope_store_drain(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_effect_use_t* effect) {
  if (effect->memory_space != LOOM_LOW_MEMORY_SPACE_GLOBAL &&
      effect->memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    return false;
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[effect->node_index];
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (descriptor == NULL) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor);
  if (iree_any_bit_set(descriptor_view->instruction_class_flags,
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_ATOMIC)) {
    return false;
  }
  const loom_low_memory_access_summary_t* access =
      loom_low_memory_access_map_lookup(schedule->memory_accesses, node->op,
                                        effect->effect_ordinal);
  if (access != NULL &&
      iree_any_bit_set(access->source_flags,
                       LOOM_LOW_MEMORY_ACCESS_SOURCE_FLAG_ATOMIC_OBSERVATION)) {
    return false;
  }
  const loom_amdgpu_descriptor_immediate_slots_t immediate_slots =
      loom_amdgpu_descriptor_immediate_slots(descriptor_set, descriptor);
  if (immediate_slots.cache_scope ==
      LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE) {
    return false;
  }
  IREE_ASSERT_LT(immediate_slots.cache_scope, descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start +
                                  immediate_slots.cache_scope];
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(schedule, effect->node_index);
  const loom_attribute_t scope_attr =
      loom_low_packet_immediate_attr(&packet, immediate);
  const int64_t scope = loom_attr_is_absent(scope_attr)
                            ? immediate->default_value
                            : scope_attr.i64;
  return scope == LOOM_CACHE_SCOPE_SYSTEM;
}

static loom_amdgpu_structural_packet_flags_t
loom_amdgpu_wait_classification_classify_structural_node(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_STRUCTURAL) {
    return 0;
  }
  return loom_amdgpu_structural_packet_analyze(
             schedule, allocation, node,
             LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION)
      .flags;
}

// Keep the compact hazard walk out of the inlined classifier so its schedule
// range remains resident instead of being reloaded under caller register
// pressure.
IREE_ATTRIBUTE_NOINLINE static void
loom_amdgpu_wait_classification_classify_hazards(
    const loom_low_schedule_table_t* schedule,
    loom_amdgpu_wait_classification_t* classification) {
  const loom_low_schedule_hazard_use_t* hazard_uses = schedule->hazard_uses;
  const iree_host_size_t hazard_use_count = schedule->hazard_use_count;
  for (iree_host_size_t i = 0; i < hazard_use_count; ++i) {
    const loom_low_schedule_hazard_use_t* hazard = &hazard_uses[i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) {
      continue;
    }
    IREE_ASSERT_EQ(hazard->reference_kind,
                   LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER);
    IREE_ASSERT_LT(hazard->node_index, schedule->node_count);
    const uint32_t counter_mask =
        loom_amdgpu_wait_counter_mask(hazard->reference_id);
    classification->node_states[hazard->node_index].hazard_counter_mask |=
        counter_mask;
  }
}

static void loom_amdgpu_wait_classification_classify_effects(
    const loom_low_schedule_table_t* schedule,
    loom_amdgpu_wait_classification_t* classification) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          schedule->target.descriptor_set->descriptor_set_ordinal);
  const bool waits_before_system_scope_stores =
      loom_amdgpu_descriptor_set_info_has_flags(
          descriptor_set_info,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_WAITS_BEFORE_SYSTEM_SCOPE_STORES);
  const loom_low_schedule_effect_use_t* effect_uses = schedule->effect_uses;
  const iree_host_size_t effect_use_count = schedule->effect_use_count;
  for (iree_host_size_t i = 0; i < effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &effect_uses[i];
    IREE_ASSERT_LT(effect->node_index, schedule->node_count);
    loom_amdgpu_wait_node_state_t* node_state =
        &classification->node_states[effect->node_index];
    loom_amdgpu_wait_frontier_node_t* frontier_node =
        &classification->frontier_nodes[effect->node_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          // Counter-backed external reads produce asynchronous results without
          // participating in memory alias dependencies.
          const uint32_t counter_mask =
              loom_amdgpu_wait_effect_counter_mask(effect);
          frontier_node->read_counter_mask |= counter_mask;
          frontier_node->external_counter_mask |= counter_mask;
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ;
        frontier_node->read_space_flags |=
            loom_amdgpu_wait_memory_space_flag(effect->memory_space);
        const uint32_t counter_mask =
            loom_amdgpu_wait_effect_counter_mask(effect);
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ;
        } else {
          frontier_node->read_counter_mask |= counter_mask;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP ||
              effect->memory_space == LOOM_LOW_MEMORY_SPACE_GENERIC) {
            node_state->workgroup_access_counter_mask |= counter_mask;
          }
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        if (waits_before_system_scope_stores &&
            loom_amdgpu_wait_effect_requires_system_scope_store_drain(schedule,
                                                                      effect)) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_SYSTEM_SCOPE_STORE_DRAIN;
          node_state->barrier_counter_mask |=
              LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD |
              LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM |
              LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
        }
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          const uint32_t counter_mask =
              loom_amdgpu_wait_effect_counter_mask(effect);
          frontier_node->write_counter_mask |= counter_mask;
          frontier_node->external_counter_mask |= counter_mask;
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->write_space_flags |=
            loom_amdgpu_wait_memory_space_flag(effect->memory_space);
        const uint32_t counter_mask =
            loom_amdgpu_wait_effect_counter_mask(effect);
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE;
        } else {
          frontier_node->write_counter_mask |= counter_mask;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP ||
              effect->memory_space == LOOM_LOW_MEMORY_SPACE_GENERIC) {
            node_state->workgroup_access_counter_mask |= counter_mask;
          }
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_BARRIER:
        if (loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          const uint32_t counter_mask =
              loom_amdgpu_wait_effect_counter_mask(effect);
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          } else {
            node_state->barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          }
        }
        break;
      case LOOM_LOW_EFFECT_KIND_COUNTER: {
        if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT;
          break;
        }
        const uint32_t counter_mask =
            loom_amdgpu_wait_counter_mask(effect->counter_id);
        if (node_state->explicit_wait_counter_mask == 0) {
          node_state->state.wait_bounds_index =
              (uint32_t)classification->wait_bounds_count++;
        }
        node_state->explicit_wait_counter_mask |= counter_mask;
        break;
      }
      default:
        break;
    }
  }
}

static bool loom_amdgpu_wait_classification_descriptor_has_xcnt_source_lease(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL || descriptor->storage_lease_count == 0) {
    return false;
  }
  IREE_ASSERT_LE(descriptor->storage_lease_start,
                 descriptor_set->storage_lease_count);
  IREE_ASSERT_LE(
      descriptor->storage_lease_count,
      descriptor_set->storage_lease_count - descriptor->storage_lease_start);
  for (uint16_t i = 0; i < descriptor->storage_lease_count; ++i) {
    const loom_low_descriptor_storage_lease_t* lease =
        &descriptor_set->storage_leases[descriptor->storage_lease_start + i];
    if (lease->kind == LOOM_LOW_STORAGE_LEASE_SOURCE_READ &&
        lease->release_class_id == LOOM_AMDGPU_WAIT_COUNTER_X) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_wait_classification_descriptor_writes_exec(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return false;
  }
  IREE_ASSERT_LE(descriptor->operand_start, descriptor_set->operand_count);
  IREE_ASSERT_LE(descriptor->operand_count,
                 descriptor_set->operand_count - descriptor->operand_start);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      continue;
    }
    IREE_ASSERT_LE(operand->reg_class_alt_start,
                   descriptor_set->reg_class_alt_count);
    IREE_ASSERT_LE(
        operand->reg_class_alt_count,
        descriptor_set->reg_class_alt_count - operand->reg_class_alt_start);
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint16_t reg_class_id =
          descriptor_set->reg_class_alts[operand->reg_class_alt_start + j]
              .reg_class_id;
      if (reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_EXEC) {
        return true;
      }
    }
  }
  return false;
}

static bool
loom_amdgpu_wait_classification_structural_node_implicitly_drains_xcnt(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return false;
  }
  if (loom_low_return_isa(op)) {
    return true;
  }
  if (loom_low_br_isa(op)) {
    const uint32_t destination_block_index =
        loom_low_packet_block_index(schedule, loom_low_br_dest(op));
    return destination_block_index != node->block_index + 1;
  }
  if (loom_low_cond_br_isa(op)) {
    const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
    const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
    if (true_dest != false_dest) {
      return true;
    }
    const uint32_t destination_block_index =
        loom_low_packet_block_index(schedule, true_dest);
    return destination_block_index != node->block_index + 1;
  }
  return false;
}

static void loom_amdgpu_wait_classification_finish_nodes(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_wait_packet_target_t* wait_packet_target,
    loom_amdgpu_wait_classification_t* classification) {
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const bool has_valu_trans_use_depctr =
      loom_amdgpu_processor_properties_have_scheduling(
          processor_properties,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
  IREE_ASSERT_LT(LOOM_AMDGPU_WAIT_COUNTER_MASK_X,
                 wait_packet_target->selection_count);
  const bool supports_xcnt =
      wait_packet_target->selections[LOOM_AMDGPU_WAIT_COUNTER_MASK_X]
          .covered_counter_mask == LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_node_state_t* node_state = &classification->node_states[i];
    loom_amdgpu_wait_frontier_node_t* frontier_node =
        &classification->frontier_nodes[i];
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    const loom_amdgpu_wait_memory_space_flags_t generic_space =
        loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
    const loom_amdgpu_wait_memory_space_flags_t workgroup_spaces =
        generic_space |
        loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_WORKGROUP);
    if (node->descriptor == NULL &&
        !iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY)) {
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ;
        frontier_node->read_space_flags |= generic_space;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->write_space_flags |= generic_space;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                             LOOM_TRAIT_UNKNOWN_EFFECTS |
                                             LOOM_TRAIT_CONVERGENT |
                                             LOOM_TRAIT_OBSERVABLE_EFFECT)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ |
                             LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->read_space_flags |= generic_space;
        frontier_node->write_space_flags |= generic_space;
      }
    }
    // A volatile memory instruction is an ordering consumer even when its
    // descriptor only reads. Retain that distinction for both local edges and
    // the incoming memory frontier across CFG blocks.
    if (node->descriptor != NULL &&
        iree_any_bit_set(node->op->instance_flags,
                         LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ |
                           LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
      frontier_node->read_space_flags |= generic_space;
      frontier_node->write_space_flags |= generic_space;
    }
    const loom_amdgpu_descriptor_traits_t descriptor_traits =
        loom_amdgpu_descriptor_traits(descriptor_set, node->descriptor);
    if (supports_xcnt &&
        iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
      const loom_low_descriptor_view_t* descriptor_view =
          loom_low_descriptor_set_descriptor_view(descriptor_set,
                                                  node->descriptor);
      if (iree_any_bit_set(descriptor_view->instruction_class_flags,
                           LOOM_LOW_INSTRUCTION_CLASS_FLAG_ATOMIC) ||
          iree_any_bit_set(node->op->instance_flags,
                           LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
        // These accesses cannot be repeated after a Gfx125x page-fault replay.
        // Ordinary dependency waits may already drain translations; the barrier
        // consumes their progress before deciding whether XCNT needs a wait.
        node_state->barrier_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
      }
    }
    if (loom_amdgpu_wait_classification_descriptor_has_xcnt_source_lease(
            descriptor_set, node->descriptor)) {
      node_state->source_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
      node_state->flags |= iree_any_bit_set(node_state->hazard_counter_mask,
                                            LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM)
                               ? LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER
                               : LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_PRODUCER;
      frontier_node->xcnt_group_flags =
          iree_any_bit_set(node_state->flags,
                           LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER)
              ? LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM
              : LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;
    }
    if (supports_xcnt && loom_amdgpu_wait_classification_descriptor_writes_exec(
                             descriptor_set, node->descriptor)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_WRITES_EXEC;
    }
    if (supports_xcnt &&
        (iree_any_bit_set(descriptor_traits,
                          LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN) ||
         loom_amdgpu_wait_classification_structural_node_implicitly_drains_xcnt(
             schedule, node))) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_IMPLICIT_DRAIN;
      node_state->implicit_wait_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
    }
    frontier_node->vmem_result_order_class =
        loom_amdgpu_descriptor_vmem_result_order_class(descriptor_set,
                                                       node->descriptor);
    if (node->result_count != 0 && frontier_node->vmem_result_order_class !=
                                       LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE) {
      ++classification->vmem_result_node_count;
    }
    if (node->descriptor != NULL && node->result_count != 0) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL;
    }
    const loom_amdgpu_wait_node_state_flags_t flags = node_state->flags;
    const bool has_generic_counter_effect = iree_any_bit_set(
        flags, LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT);
    if (has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask |= node_state->hazard_counter_mask;
    }
    if (node_state->explicit_wait_counter_mask != 0 &&
        !has_generic_counter_effect) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_node(schedule, i);
      node_state
          ->explicit_wait_counter_mask = loom_amdgpu_wait_packet_decode_bounds(
          descriptor_set, &packet, wait_packet_target,
          &classification->wait_bounds[node_state->state.wait_bounds_index]);
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT;
    }
    IREE_ASSERT(node_state->explicit_wait_counter_mask == 0 ||
                node_state->hazard_counter_mask != 0);
    if (iree_any_bit_set(flags,
                         LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ)) {
      const uint32_t default_read_counter_mask =
          node_state->hazard_counter_mask &
          LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY;
      IREE_ASSERT_NE(default_read_counter_mask, 0u);
      frontier_node->read_counter_mask |= default_read_counter_mask;
      if (iree_any_bit_set(frontier_node->read_space_flags, workgroup_spaces)) {
        node_state->workgroup_access_counter_mask |= default_read_counter_mask;
      }
    }
    IREE_ASSERT_EQ(
        frontier_node->read_counter_mask & ~node_state->hazard_counter_mask,
        0u);
    if (iree_any_bit_set(
            flags, LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE)) {
      const uint32_t default_write_counter_mask =
          node_state->hazard_counter_mask &
          LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY;
      IREE_ASSERT_NE(default_write_counter_mask, 0u);
      frontier_node->write_counter_mask |= default_write_counter_mask;
      if (iree_any_bit_set(frontier_node->write_space_flags,
                           workgroup_spaces)) {
        node_state->workgroup_access_counter_mask |= default_write_counter_mask;
      }
    }
    IREE_ASSERT_EQ(
        frontier_node->write_counter_mask & ~node_state->hazard_counter_mask,
        0u);
    const uint32_t memory_counter_mask =
        frontier_node->read_counter_mask | frontier_node->write_counter_mask;
    if (iree_any_bit_set(memory_counter_mask,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS) &&
        iree_any_bit_set(memory_counter_mask,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM) &&
        iree_any_bit_set(
            frontier_node->read_space_flags | frontier_node->write_space_flags,
            generic_space) &&
        !loom_amdgpu_processor_properties_have_scheduling(
            processor_properties,
            LOOM_AMDGPU_PROCESSOR_SCHEDULING_FLAT_COUNTERS_IN_ORDER)) {
      node_state->flags |=
          LOOM_AMDGPU_WAIT_NODE_STATE_UNORDERED_FLAT_COMPLETION;
    }
    const loom_amdgpu_structural_packet_flags_t structural_flags =
        loom_amdgpu_wait_classification_classify_structural_node(
            schedule, allocation, (uint32_t)i);
    if (node->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL &&
        !iree_any_bit_set(structural_flags,
                          LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_ZERO_NATIVE_WORK;
    }
    if (iree_any_bit_set(
            structural_flags,
            LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES;
      ++classification->forwarding_node_count;
    }
    if (node->result_count != 0 &&
        iree_any_bit_set(structural_flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS;
    }
    if (has_valu_trans_use_depctr &&
        iree_any_bit_set(flags, LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL)) {
      node_state->trans_result_counter_mask |=
          LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
      ++classification->trans_result_node_count;
    }
    const uint32_t producer_counter_mask =
        frontier_node->read_counter_mask | frontier_node->write_counter_mask |
        node_state->trans_result_counter_mask | node_state->source_counter_mask;
    classification->completion_nodes[i] = (loom_amdgpu_wait_completion_node_t){
        .producer_counter_mask = producer_counter_mask,
        .write_counter_mask = frontier_node->write_counter_mask,
        .reset_counter_mask = node_state->explicit_wait_counter_mask |
                              node_state->implicit_wait_counter_mask |
                              node_state->barrier_counter_mask,
        .hazard_counter_mask = node_state->hazard_counter_mask,
        .workgroup_access_counter_mask =
            node_state->workgroup_access_counter_mask,
        .workgroup_barrier_counter_mask =
            node_state->workgroup_barrier_counter_mask,
    };
    // The payload indexes either immutable wait bounds or mutable issue state.
    IREE_ASSERT(!iree_any_bit_set(node_state->flags,
                                  LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT) ||
                classification->completion_nodes[i].producer_counter_mask == 0);
    if (producer_counter_mask != 0) {
      IREE_ASSERT_LT(classification->producer_state_count, UINT32_MAX);
      node_state->state.producer_state_ordinal =
          (uint32_t)++classification->producer_state_count;
    }
  }
}

// Classification has one hot caller. Preserve its caller specialization while
// retaining an external definition for ordinary non-LTO builds.
IREE_ATTRIBUTE_ALWAYS_INLINE extern inline iree_status_t
loom_amdgpu_wait_classification_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_wait_packet_target_t* wait_packet_target,
    iree_arena_allocator_t* arena,
    loom_amdgpu_wait_classification_t* IREE_RESTRICT out_classification) {
  *out_classification = (loom_amdgpu_wait_classification_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_classification_allocate(
      schedule, arena, out_classification));
  loom_amdgpu_wait_classification_classify_hazards(schedule,
                                                   out_classification);
  loom_amdgpu_wait_classification_classify_effects(schedule,
                                                   out_classification);
  if (out_classification->wait_bounds_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_classification->wait_bounds_count,
                                  sizeof(*out_classification->wait_bounds),
                                  (void**)&out_classification->wait_bounds));
  }
  loom_amdgpu_wait_classification_finish_nodes(
      schedule, allocation, processor_properties, wait_packet_target,
      out_classification);
  return iree_ok_status();
}
