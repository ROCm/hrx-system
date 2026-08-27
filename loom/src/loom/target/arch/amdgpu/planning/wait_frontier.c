// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_frontier.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/util/cfg_graph.h"

enum {
  LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED = 1u << 0,
  LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_RESOLVED = 1u << 1,
  LOOM_AMDGPU_WAIT_VMEM_RESULT_STATE_FLAG_PENDING = 1u << 0,
  LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK =
      ((1u << LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT) - 1u)
      << LOOM_LOW_MEMORY_SPACE_GENERIC,
  LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT = 8,
  LOOM_AMDGPU_WAIT_VMEM_RESULT_BITS_PER_UNIT = 4,
  LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD =
      64 / LOOM_AMDGPU_WAIT_VMEM_RESULT_BITS_PER_UNIT,
  LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD = 64,
};

static_assert(LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT <= 8,
              "memory-space frontier flags must fit in one byte");
static_assert(LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT <= 8,
              "memory frontier counter masks must fit in one byte");
static_assert(LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL ==
                  (1u << LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT) - 1u,
              "memory frontier counter masks must use dense low bits");
static_assert(sizeof(loom_amdgpu_wait_memory_state_t) ==
                  LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT * sizeof(uint16_t),
              "memory frontier state must not acquire padding");
static_assert(LOOM_AMDGPU_VMEM_RESULT_ORDER_CLASS_COUNT - 1 ==
                  LOOM_AMDGPU_WAIT_VMEM_RESULT_BITS_PER_UNIT,
              "VMEM result classes must fit the packed frontier state");

loom_amdgpu_wait_memory_space_flags_t loom_amdgpu_wait_memory_space_flag(
    loom_low_memory_space_t memory_space) {
  const loom_low_memory_space_t normalized_space =
      loom_low_memory_access_normalize_space(memory_space);
  return (loom_amdgpu_wait_memory_space_flags_t)(1u << normalized_space);
}

static bool loom_amdgpu_wait_memory_state_union_changed(
    loom_amdgpu_wait_memory_state_t* target,
    const loom_amdgpu_wait_memory_state_t* source) {
  bool changed = false;
  for (uint32_t space = 0; space < LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT;
       ++space) {
    const uint16_t access_counter_masks = target->access_counter_masks[space] |
                                          source->access_counter_masks[space];
    changed |= access_counter_masks != target->access_counter_masks[space];
    target->access_counter_masks[space] = access_counter_masks;
  }
  return changed;
}

static bool loom_amdgpu_wait_memory_state_is_empty(
    const loom_amdgpu_wait_memory_state_t* state) {
  for (uint32_t space = 0; space < LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT;
       ++space) {
    if (state->access_counter_masks[space] != 0) return false;
  }
  return true;
}

static uint64_t* loom_amdgpu_wait_frontier_storage_lease_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->storage_leases.word_count;
}

static const uint64_t*
loom_amdgpu_wait_frontier_const_storage_lease_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->storage_leases.word_count;
}

static const uint64_t*
loom_amdgpu_wait_frontier_const_storage_lease_counter_words(
    const loom_amdgpu_wait_frontier_t* frontier, uint32_t counter_slot) {
  IREE_ASSERT_LT(counter_slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return frontier->storage_leases.release_membership.words +
         counter_slot * frontier->storage_leases.word_count;
}

static bool loom_amdgpu_wait_storage_lease_state_union_changed(
    uint64_t* target, const uint64_t* source, iree_host_size_t word_count) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    const uint64_t result = target[i] | source[i];
    changed |= result != target[i];
    target[i] = result;
  }
  return changed;
}

static bool loom_amdgpu_wait_storage_lease_state_is_empty(
    const uint64_t* words, iree_host_size_t word_count) {
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    if (words[i] != 0) return false;
  }
  return true;
}

static bool loom_amdgpu_wait_storage_lease_state_test(
    const uint64_t* words, iree_host_size_t lease_index) {
  const iree_host_size_t word_index =
      lease_index / LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
  const uint32_t bit_index =
      (uint32_t)(lease_index % LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD);
  return (words[word_index] & (UINT64_C(1) << bit_index)) != 0;
}

static void loom_amdgpu_wait_storage_lease_state_set(
    uint64_t* words, iree_host_size_t lease_index) {
  const iree_host_size_t word_index =
      lease_index / LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
  const uint32_t bit_index =
      (uint32_t)(lease_index % LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD);
  words[word_index] |= UINT64_C(1) << bit_index;
}

static void loom_amdgpu_wait_storage_lease_state_clear(
    uint64_t* words, iree_host_size_t lease_index) {
  const iree_host_size_t word_index =
      lease_index / LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
  const uint32_t bit_index =
      (uint32_t)(lease_index % LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD);
  words[word_index] &= ~(UINT64_C(1) << bit_index);
}

static loom_amdgpu_wait_xcnt_group_flags_t
loom_amdgpu_wait_frontier_storage_lease_xcnt_group(
    const loom_amdgpu_wait_frontier_t* frontier, iree_host_size_t lease_index) {
  IREE_ASSERT_LT(lease_index, frontier->storage_leases.lease_count);
  const loom_low_allocation_storage_lease_t* lease =
      &frontier->allocation->storage_lease_instances[lease_index];
  IREE_ASSERT_LT(lease->lease_record_index,
                 frontier->allocation->storage_leases.record_count);
  const loom_low_storage_lease_record_t* record =
      &frontier->allocation->storage_leases.records[lease->lease_record_index];
  if (record->release_scope !=
          LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS ||
      record->release_class_id != LOOM_AMDGPU_WAIT_COUNTER_X) {
    return 0;
  }
  IREE_ASSERT_LT(record->node_index, frontier->schedule->node_count);
  return frontier->nodes[record->node_index].xcnt_group_flags;
}

static void loom_amdgpu_wait_storage_lease_state_drain(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    uint32_t counter_mask) {
  counter_mask &= LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  while (counter_mask != 0) {
    const uint32_t counter_slot =
        (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
    const uint64_t* release_counter_words =
        loom_amdgpu_wait_frontier_const_storage_lease_counter_words(
            frontier, counter_slot);
    for (iree_host_size_t word_index = 0;
         word_index < frontier->storage_leases.word_count; ++word_index) {
      words[word_index] &= ~release_counter_words[word_index];
    }
    counter_mask &= counter_mask - 1;
  }
}

static bool loom_amdgpu_wait_storage_lease_state_union_after_drain_changed(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* target,
    const uint64_t* source, uint32_t counter_mask) {
  if (counter_mask == 0) {
    return loom_amdgpu_wait_storage_lease_state_union_changed(
        target, source, frontier->storage_leases.word_count);
  }
  bool changed = false;
  // The inner mask traversal is bounded by the eight architectural wait
  // counters, keeping this linear in the storage-lease word count.
  for (iree_host_size_t word_index = 0;
       word_index < frontier->storage_leases.word_count; ++word_index) {
    uint64_t retained_source = source[word_index];
    uint32_t remaining_counter_mask =
        counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
    while (remaining_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(remaining_counter_mask);
      const uint64_t* release_counter_words =
          loom_amdgpu_wait_frontier_const_storage_lease_counter_words(
              frontier, counter_slot);
      retained_source &= ~release_counter_words[word_index];
      remaining_counter_mask &= remaining_counter_mask - 1;
    }
    const uint64_t result = target[word_index] | retained_source;
    changed |= result != target[word_index];
    target[word_index] = result;
  }
  return changed;
}

static void loom_amdgpu_wait_storage_lease_state_drain_xcnt_groups(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    loom_amdgpu_wait_xcnt_group_flags_t group_flags) {
  for (iree_host_size_t lease_index = 0;
       lease_index < frontier->storage_leases.lease_count; ++lease_index) {
    if (!iree_any_bit_set(group_flags,
                          loom_amdgpu_wait_frontier_storage_lease_xcnt_group(
                              frontier, lease_index))) {
      continue;
    }
    const iree_host_size_t word_index =
        lease_index / LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
    const uint32_t bit_index =
        (uint32_t)(lease_index % LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD);
    words[word_index] &= ~(UINT64_C(1) << bit_index);
  }
}

static void loom_amdgpu_wait_memory_state_drain(
    loom_amdgpu_wait_memory_state_t* state, uint32_t counter_mask) {
  if (counter_mask == 0) return;
  const uint8_t retained_counter_mask = (uint8_t)~counter_mask;
  const uint16_t retained_access_counter_masks =
      (uint16_t)retained_counter_mask |
      ((uint16_t)retained_counter_mask
       << LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT);
  for (uint32_t space = 0; space < LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT;
       ++space) {
    state->access_counter_masks[space] &= retained_access_counter_masks;
  }
}

static void loom_amdgpu_wait_memory_state_add_access(
    loom_amdgpu_wait_memory_state_t* state,
    loom_amdgpu_wait_memory_space_flags_t producer_space_flags,
    uint16_t access_counter_masks) {
  if (producer_space_flags == 0 || access_counter_masks == 0) return;
  IREE_ASSERT_EQ(
      (uint32_t)producer_space_flags & ~LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK,
      0u);
  producer_space_flags >>= LOOM_LOW_MEMORY_SPACE_GENERIC;
  state->access_counter_masks[0] |= access_counter_masks;
  if (iree_any_bit_set(producer_space_flags, 1u)) {
    for (uint32_t space_index = 1;
         space_index < LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT; ++space_index) {
      state->access_counter_masks[space_index] |= access_counter_masks;
    }
    return;
  }
  while (producer_space_flags != 0) {
    const uint32_t space_index =
        (uint32_t)iree_math_count_trailing_zeros_u32(producer_space_flags);
    IREE_ASSERT_LT(space_index, LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT);
    state->access_counter_masks[space_index] |= access_counter_masks;
    producer_space_flags &= producer_space_flags - 1;
  }
}

static void loom_amdgpu_wait_memory_state_add_node(
    loom_amdgpu_wait_memory_state_t* state,
    const loom_amdgpu_wait_frontier_node_t* node, uint32_t read_counter_mask,
    uint32_t write_counter_mask) {
  if ((read_counter_mask | write_counter_mask) == 0) return;
  IREE_ASSERT_EQ((read_counter_mask | write_counter_mask) &
                     ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL,
                 0u);
  loom_amdgpu_wait_memory_state_add_access(state, node->read_space_flags,
                                           (uint8_t)read_counter_mask);
  loom_amdgpu_wait_memory_state_add_access(
      state, node->write_space_flags,
      (uint16_t)(uint8_t)write_counter_mask
          << LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT);
}

static uint8_t loom_amdgpu_wait_vmem_result_order_class_flag(
    loom_amdgpu_vmem_result_order_class_t order_class) {
  IREE_ASSERT_GT(order_class, LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE);
  IREE_ASSERT_LT(order_class, LOOM_AMDGPU_VMEM_RESULT_ORDER_CLASS_COUNT);
  return (uint8_t)(1u << (order_class - 1u));
}

static bool loom_amdgpu_wait_frontier_map_vector_assignment(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_low_allocation_assignment_t* assignment,
    iree_host_size_t* out_unit_base, iree_host_size_t* out_unit_count) {
  *out_unit_base = 0;
  *out_unit_count = 0;
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_count == 0) {
    return false;
  }
  iree_host_size_t unit_base = assignment->location_base;
  iree_host_size_t unit_limit = frontier->vmem_results.vgpr_unit_count;
  const loom_amdgpu_reg_class_traits_t reg_class_traits =
      loom_amdgpu_reg_class_traits(frontier->schedule->target.descriptor_set,
                                   assignment->descriptor_reg_class_id);
  if (iree_any_bit_set(reg_class_traits, LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR)) {
    unit_base += frontier->vmem_results.vgpr_unit_count;
    unit_limit += frontier->vmem_results.agpr_unit_count;
  } else if (assignment->descriptor_reg_class_id !=
             LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return false;
  }
  IREE_ASSERT_LE(unit_base + assignment->location_count, unit_limit);
  *out_unit_base = unit_base;
  *out_unit_count = assignment->location_count;
  return true;
}

static uint64_t* loom_amdgpu_wait_frontier_vmem_result_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->vmem_results.word_count;
}

static const uint64_t* loom_amdgpu_wait_frontier_const_vmem_result_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->vmem_results.word_count;
}

static bool loom_amdgpu_wait_vmem_result_state_union_changed(
    uint64_t* target, const uint64_t* source, iree_host_size_t word_count) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    const uint64_t result = target[i] | source[i];
    changed |= result != target[i];
    target[i] = result;
  }
  return changed;
}

static bool loom_amdgpu_wait_vmem_result_state_is_empty(
    const uint64_t* words, iree_host_size_t word_count) {
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    if (words[i] != 0) return false;
  }
  return true;
}

static uint8_t loom_amdgpu_wait_vmem_result_state_query(
    const uint64_t* words, iree_host_size_t unit_base,
    iree_host_size_t unit_count) {
  uint8_t order_class_flags = 0;
  for (iree_host_size_t i = 0; i < unit_count; ++i) {
    const iree_host_size_t unit = unit_base + i;
    const iree_host_size_t word_index =
        unit / LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD;
    const uint32_t bit_offset =
        (uint32_t)(unit % LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD) *
        LOOM_AMDGPU_WAIT_VMEM_RESULT_BITS_PER_UNIT;
    order_class_flags |= (uint8_t)(words[word_index] >> bit_offset) & 0xFu;
  }
  return order_class_flags;
}

static void loom_amdgpu_wait_vmem_result_state_publish(
    uint64_t* words, iree_host_size_t unit_base, iree_host_size_t unit_count,
    loom_amdgpu_vmem_result_order_class_t order_class) {
  const uint64_t order_class_flag =
      loom_amdgpu_wait_vmem_result_order_class_flag(order_class);
  for (iree_host_size_t i = 0; i < unit_count; ++i) {
    const iree_host_size_t unit = unit_base + i;
    const iree_host_size_t word_index =
        unit / LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD;
    const uint32_t bit_offset =
        (uint32_t)(unit % LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD) *
        LOOM_AMDGPU_WAIT_VMEM_RESULT_BITS_PER_UNIT;
    words[word_index] |= order_class_flag << bit_offset;
  }
}

static void loom_amdgpu_wait_frontier_publish_node_vmem_results(
    loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    uint32_t node_index) {
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &frontier->nodes[node_index];
  if (frontier_node->vmem_result_order_class ==
      LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE) {
    return;
  }
  const loom_low_schedule_node_t* schedule_node =
      &frontier->schedule->nodes[node_index];
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(schedule_node);
  for (uint16_t i = 0; i < schedule_node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            frontier->allocation, result_ordinals[i], NULL);
    iree_host_size_t unit_base = 0;
    iree_host_size_t unit_count = 0;
    if (!loom_amdgpu_wait_frontier_map_vector_assignment(
            frontier, assignment, &unit_base, &unit_count)) {
      continue;
    }
    loom_amdgpu_wait_vmem_result_state_publish(
        words, unit_base, unit_count, frontier_node->vmem_result_order_class);
  }
}

static void loom_amdgpu_wait_frontier_publish_packet_storage_leases(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    iree_host_size_t packet_index, uint32_t excluded_counter_mask,
    iree_host_size_t* inout_next_storage_lease_index) {
  while (*inout_next_storage_lease_index <
         frontier->storage_leases.lease_count) {
    const iree_host_size_t lease_index = *inout_next_storage_lease_index;
    const loom_low_storage_lease_record_t* record =
        &frontier->allocation->storage_leases.records[lease_index];
    if (record->packet_index != packet_index) break;
    ++*inout_next_storage_lease_index;
    if (record->release_scope !=
            LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS ||
        !loom_amdgpu_wait_counter_id_is_valid(record->release_class_id) ||
        iree_any_bit_set(
            excluded_counter_mask,
            loom_amdgpu_wait_counter_mask(record->release_class_id))) {
      continue;
    }
    loom_amdgpu_wait_storage_lease_state_set(words, lease_index);
  }
}

static iree_host_size_t loom_amdgpu_wait_frontier_storage_lease_lower_bound(
    const loom_amdgpu_wait_frontier_t* frontier,
    iree_host_size_t packet_index) {
  iree_host_size_t first_lease_index = 0;
  iree_host_size_t lease_index_limit = frontier->storage_leases.lease_count;
  while (first_lease_index < lease_index_limit) {
    const iree_host_size_t middle_lease_index =
        first_lease_index + (lease_index_limit - first_lease_index) / 2;
    const loom_low_storage_lease_record_t* record =
        &frontier->allocation->storage_leases.records[middle_lease_index];
    if (record->packet_index < packet_index) {
      first_lease_index = middle_lease_index + 1;
    } else {
      lease_index_limit = middle_lease_index;
    }
  }
  return first_lease_index;
}

static void loom_amdgpu_wait_frontier_apply_static_xcnt_producer(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* storage_lease_words,
    loom_amdgpu_wait_xcnt_group_flags_t* xcnt_group_flags,
    loom_amdgpu_wait_xcnt_group_flags_t producer_group_flags) {
  if (producer_group_flags == 0) return;
  const loom_amdgpu_wait_xcnt_group_flags_t other_group_flags =
      *xcnt_group_flags &
      (loom_amdgpu_wait_xcnt_group_flags_t)~producer_group_flags;
  if (other_group_flags != 0 && storage_lease_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain_xcnt_groups(
        frontier, storage_lease_words, other_group_flags);
  }
  *xcnt_group_flags = producer_group_flags;
}

static void loom_amdgpu_wait_frontier_build_local_states(
    loom_amdgpu_wait_frontier_t* frontier,
    const uint32_t* planned_block_drain_counter_masks) {
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  iree_host_size_t next_storage_lease_index = 0;
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    loom_amdgpu_wait_memory_state_t* memory_state =
        frontier->memory.static_outgoing_states == NULL
            ? NULL
            : &frontier->memory.static_outgoing_states[block_index];
    uint64_t* vmem_result_words =
        frontier->vmem_results.static_outgoing_words == NULL
            ? NULL
            : loom_amdgpu_wait_frontier_vmem_result_block_words(
                  frontier, frontier->vmem_results.static_outgoing_words,
                  block_index);
    uint64_t* storage_lease_words =
        frontier->storage_leases.static_outgoing_words == NULL
            ? NULL
            : loom_amdgpu_wait_frontier_storage_lease_block_words(
                  frontier, frontier->storage_leases.static_outgoing_words,
                  block_index);
    loom_amdgpu_wait_xcnt_group_flags_t* xcnt_group_flags =
        frontier->xcnt.static_outgoing_flags == NULL
            ? NULL
            : &frontier->xcnt.static_outgoing_flags[block_index];
    uint32_t block_drain_counter_mask = 0;
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      const loom_amdgpu_wait_frontier_node_t* node =
          &frontier->nodes[node_index];
      uint32_t drain_counter_mask = node->drain_counter_mask;
      if (planned_block_drain_counter_masks != NULL &&
          schedule->nodes[node_index].op == block->block->last_op) {
        drain_counter_mask |= planned_block_drain_counter_masks[block_index];
      }
      block_drain_counter_mask |= drain_counter_mask;
      if (memory_state != NULL) {
        loom_amdgpu_wait_memory_state_drain(memory_state, drain_counter_mask);
        loom_amdgpu_wait_memory_state_add_node(memory_state, node,
                                               node->read_counter_mask,
                                               node->write_counter_mask);
      }
      if (vmem_result_words != NULL) {
        if (iree_any_bit_set(drain_counter_mask,
                             LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
          memset(
              vmem_result_words, 0,
              frontier->vmem_results.word_count * sizeof(*vmem_result_words));
        }
        loom_amdgpu_wait_frontier_publish_node_vmem_results(
            frontier, vmem_result_words, node_index);
      }
      if (storage_lease_words != NULL) {
        loom_amdgpu_wait_storage_lease_state_drain(
            frontier, storage_lease_words, drain_counter_mask);
      }
      if (xcnt_group_flags != NULL) {
        if (iree_any_bit_set(drain_counter_mask,
                             LOOM_AMDGPU_WAIT_COUNTER_MASK_X)) {
          *xcnt_group_flags = 0;
        }
        loom_amdgpu_wait_frontier_apply_static_xcnt_producer(
            frontier, storage_lease_words, xcnt_group_flags,
            node->xcnt_group_flags);
      }
      if (storage_lease_words != NULL) {
        loom_amdgpu_wait_frontier_publish_packet_storage_leases(
            frontier, storage_lease_words, packet_index,
            /*excluded_counter_mask=*/0, &next_storage_lease_index);
      }
    }
    frontier->block_drain_counter_masks[block_index] = block_drain_counter_mask;
  }
}

static bool loom_amdgpu_wait_frontier_block_state_union_changed(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t target_block,
    uint16_t source_block) {
  bool changed = false;
  const uint32_t drain_counter_mask =
      frontier->block_drain_counter_masks[target_block];
  if (frontier->memory.static_outgoing_states != NULL) {
    loom_amdgpu_wait_memory_state_t incoming_state =
        frontier->memory.static_outgoing_states[source_block];
    loom_amdgpu_wait_memory_state_drain(&incoming_state, drain_counter_mask);
    changed |= loom_amdgpu_wait_memory_state_union_changed(
        &frontier->memory.static_outgoing_states[target_block],
        &incoming_state);
  }
  if (frontier->vmem_results.static_outgoing_words != NULL &&
      !iree_any_bit_set(drain_counter_mask,
                        LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
    changed |= loom_amdgpu_wait_vmem_result_state_union_changed(
        loom_amdgpu_wait_frontier_vmem_result_block_words(
            frontier, frontier->vmem_results.static_outgoing_words,
            target_block),
        loom_amdgpu_wait_frontier_const_vmem_result_block_words(
            frontier, frontier->vmem_results.static_outgoing_words,
            source_block),
        frontier->vmem_results.word_count);
  }
  if (frontier->storage_leases.static_outgoing_words != NULL) {
    changed |= loom_amdgpu_wait_storage_lease_state_union_after_drain_changed(
        frontier,
        loom_amdgpu_wait_frontier_storage_lease_block_words(
            frontier, frontier->storage_leases.static_outgoing_words,
            target_block),
        loom_amdgpu_wait_frontier_const_storage_lease_block_words(
            frontier, frontier->storage_leases.static_outgoing_words,
            source_block),
        drain_counter_mask);
  }
  if (frontier->xcnt.static_outgoing_flags != NULL &&
      !iree_any_bit_set(drain_counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_X)) {
    const loom_amdgpu_wait_xcnt_group_flags_t flags =
        frontier->xcnt.static_outgoing_flags[target_block] |
        frontier->xcnt.static_outgoing_flags[source_block];
    changed |= flags != frontier->xcnt.static_outgoing_flags[target_block];
    frontier->xcnt.static_outgoing_flags[target_block] = flags;
  }
  return changed;
}

static bool loom_amdgpu_wait_frontier_block_state_is_empty(
    const loom_amdgpu_wait_frontier_t* frontier, uint16_t block_index) {
  if (frontier->memory.static_outgoing_states != NULL &&
      !loom_amdgpu_wait_memory_state_is_empty(
          &frontier->memory.static_outgoing_states[block_index])) {
    return false;
  }
  if (frontier->vmem_results.static_outgoing_words != NULL &&
      !loom_amdgpu_wait_vmem_result_state_is_empty(
          loom_amdgpu_wait_frontier_const_vmem_result_block_words(
              frontier, frontier->vmem_results.static_outgoing_words,
              block_index),
          frontier->vmem_results.word_count)) {
    return false;
  }
  if (frontier->storage_leases.static_outgoing_words != NULL &&
      !loom_amdgpu_wait_storage_lease_state_is_empty(
          loom_amdgpu_wait_frontier_const_storage_lease_block_words(
              frontier, frontier->storage_leases.static_outgoing_words,
              block_index),
          frontier->storage_leases.word_count)) {
    return false;
  }
  return frontier->xcnt.static_outgoing_flags == NULL ||
         frontier->xcnt.static_outgoing_flags[block_index] == 0;
}

static void loom_amdgpu_wait_frontier_worklist_push(
    uint16_t block_index, uint16_t* worklist, uint32_t block_count,
    uint32_t* tail, uint32_t* count, uint8_t* block_flags) {
  if (iree_any_bit_set(block_flags[block_index],
                       LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED)) {
    return;
  }
  worklist[*tail] = block_index;
  if (++*tail == block_count) *tail = 0;
  ++*count;
  block_flags[block_index] |= LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED;
}

static void loom_amdgpu_wait_frontier_propagate_static_states(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t* worklist) {
  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  const uint32_t block_count = (uint32_t)frontier->schedule->block_count;
  if (block_count <= 1 || graph->blocks == NULL) return;

  uint32_t head = 0;
  uint32_t tail = 0;
  uint32_t count = 0;
  for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index) ||
        loom_amdgpu_wait_frontier_block_state_is_empty(frontier, block_index)) {
      continue;
    }
    loom_amdgpu_wait_frontier_worklist_push(block_index, worklist, block_count,
                                            &tail, &count,
                                            frontier->block_flags);
  }

  while (count != 0) {
    const uint16_t block_index = worklist[head];
    if (++head == block_count) head = 0;
    --count;
    frontier->block_flags[block_index] &=
        (uint8_t)~LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED;
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t successor_index = successors.values[i];
      if (!loom_cfg_graph_block_is_reachable(graph, successor_index) ||
          !loom_amdgpu_wait_frontier_block_state_union_changed(
              frontier, successor_index, block_index)) {
        continue;
      }
      loom_amdgpu_wait_frontier_worklist_push(successor_index, worklist,
                                              block_count, &tail, &count,
                                              frontier->block_flags);
    }
  }
}

iree_status_t loom_amdgpu_wait_frontier_initialize(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_frontier_node_t* nodes,
    iree_host_size_t vgpr_unit_count, iree_host_size_t agpr_unit_count,
    const uint32_t* planned_block_drain_counter_masks,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_frontier_t* out_frontier) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontier);
  IREE_ASSERT(schedule->node_count == 0 || nodes != NULL);
  *out_frontier = (loom_amdgpu_wait_frontier_t){
      .schedule = schedule,
      .allocation = allocation,
      .nodes = nodes,
      .vmem_results =
          {
              .vgpr_unit_count = vgpr_unit_count,
              .agpr_unit_count = agpr_unit_count,
          },
      .storage_leases =
          {
              .lease_count = allocation == NULL
                                 ? 0
                                 : allocation->storage_lease_instance_count,
          },
      .active_block_index = UINT16_MAX,
  };

  bool has_memory_producer = false;
  bool has_vmem_result_producer = false;
  bool has_xcnt_producer = false;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    has_memory_producer |=
        nodes[i].read_counter_mask != 0 || nodes[i].write_counter_mask != 0;
    has_vmem_result_producer |=
        nodes[i].vmem_result_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
    has_xcnt_producer |= nodes[i].xcnt_group_flags != 0;
  }

  const bool has_cross_block_state =
      schedule->block_count > 1 && schedule->cfg_graph.blocks != NULL;
  const iree_host_size_t vector_unit_count = vgpr_unit_count + agpr_unit_count;
  if (has_cross_block_state && has_vmem_result_producer && allocation != NULL &&
      vector_unit_count != 0) {
    out_frontier->vmem_results.word_count =
        (vector_unit_count + LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD - 1) /
        LOOM_AMDGPU_WAIT_VMEM_RESULT_UNITS_PER_WORD;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_frontier->vmem_results.word_count,
        sizeof(*out_frontier->vmem_results.active_words),
        (void**)&out_frontier->vmem_results.active_words));
    memset(out_frontier->vmem_results.active_words, 0,
           out_frontier->vmem_results.word_count *
               sizeof(*out_frontier->vmem_results.active_words));
  }
  if (has_cross_block_state && out_frontier->storage_leases.lease_count != 0) {
    out_frontier->storage_leases.word_count =
        (out_frontier->storage_leases.lease_count +
         LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD - 1) /
        LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_frontier->storage_leases.word_count,
        sizeof(*out_frontier->storage_leases.active_words),
        (void**)&out_frontier->storage_leases.active_words));
    memset(out_frontier->storage_leases.active_words, 0,
           out_frontier->storage_leases.word_count *
               sizeof(*out_frontier->storage_leases.active_words));
    const iree_host_size_t release_counter_word_count =
        LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT *
        out_frontier->storage_leases.word_count;
    if (out_frontier->storage_leases.word_count == 1) {
      out_frontier->storage_leases.release_membership.words =
          out_frontier->storage_leases.release_membership.inline_words;
    } else {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, release_counter_word_count,
          sizeof(*out_frontier->storage_leases.release_membership.words),
          (void**)&out_frontier->storage_leases.release_membership.words));
      memset(
          out_frontier->storage_leases.release_membership.words, 0,
          release_counter_word_count *
              sizeof(*out_frontier->storage_leases.release_membership.words));
    }
    for (iree_host_size_t lease_index = 0;
         lease_index < out_frontier->storage_leases.lease_count;
         ++lease_index) {
      const loom_low_allocation_storage_lease_t* lease =
          &allocation->storage_lease_instances[lease_index];
      IREE_ASSERT_LT(lease->lease_record_index,
                     allocation->storage_leases.record_count);
      const loom_low_storage_lease_record_t* record =
          &allocation->storage_leases.records[lease->lease_record_index];
      if (record->release_scope !=
              LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS ||
          !loom_amdgpu_wait_counter_id_is_valid(record->release_class_id)) {
        continue;
      }
      const uint32_t counter_slot =
          loom_amdgpu_wait_counter_slot_from_id(record->release_class_id);
      uint64_t* release_counter_words =
          out_frontier->storage_leases.release_membership.words +
          counter_slot * out_frontier->storage_leases.word_count;
      loom_amdgpu_wait_storage_lease_state_set(release_counter_words,
                                               lease_index);
    }
  }

  if (!has_cross_block_state ||
      (!has_memory_producer &&
       out_frontier->vmem_results.active_words == NULL &&
       out_frontier->storage_leases.active_words == NULL &&
       !has_xcnt_producer)) {
    return iree_ok_status();
  }

  if (has_memory_producer) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count,
        sizeof(*out_frontier->memory.static_outgoing_states),
        (void**)&out_frontier->memory.static_outgoing_states));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count,
        sizeof(*out_frontier->memory.resolved_outgoing_states),
        (void**)&out_frontier->memory.resolved_outgoing_states));
    memset(out_frontier->memory.static_outgoing_states, 0,
           schedule->block_count *
               sizeof(*out_frontier->memory.static_outgoing_states));
    memset(out_frontier->memory.resolved_outgoing_states, 0,
           schedule->block_count *
               sizeof(*out_frontier->memory.resolved_outgoing_states));
  }
  if (out_frontier->vmem_results.active_words != NULL) {
    const iree_host_size_t state_word_count =
        schedule->block_count * out_frontier->vmem_results.word_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state_word_count,
        sizeof(*out_frontier->vmem_results.static_outgoing_words),
        (void**)&out_frontier->vmem_results.static_outgoing_words));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state_word_count,
        sizeof(*out_frontier->vmem_results.resolved_outgoing_words),
        (void**)&out_frontier->vmem_results.resolved_outgoing_words));
    memset(out_frontier->vmem_results.static_outgoing_words, 0,
           state_word_count *
               sizeof(*out_frontier->vmem_results.static_outgoing_words));
    memset(out_frontier->vmem_results.resolved_outgoing_words, 0,
           state_word_count *
               sizeof(*out_frontier->vmem_results.resolved_outgoing_words));
  }
  if (out_frontier->storage_leases.active_words != NULL) {
    const iree_host_size_t state_word_count =
        schedule->block_count * out_frontier->storage_leases.word_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state_word_count,
        sizeof(*out_frontier->storage_leases.static_outgoing_words),
        (void**)&out_frontier->storage_leases.static_outgoing_words));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state_word_count,
        sizeof(*out_frontier->storage_leases.resolved_outgoing_words),
        (void**)&out_frontier->storage_leases.resolved_outgoing_words));
    memset(out_frontier->storage_leases.static_outgoing_words, 0,
           state_word_count *
               sizeof(*out_frontier->storage_leases.static_outgoing_words));
    memset(out_frontier->storage_leases.resolved_outgoing_words, 0,
           state_word_count *
               sizeof(*out_frontier->storage_leases.resolved_outgoing_words));
  }
  if (has_xcnt_producer) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count,
        sizeof(*out_frontier->xcnt.static_outgoing_flags),
        (void**)&out_frontier->xcnt.static_outgoing_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count,
        sizeof(*out_frontier->xcnt.resolved_outgoing_flags),
        (void**)&out_frontier->xcnt.resolved_outgoing_flags));
    memset(out_frontier->xcnt.static_outgoing_flags, 0,
           schedule->block_count *
               sizeof(*out_frontier->xcnt.static_outgoing_flags));
    memset(out_frontier->xcnt.resolved_outgoing_flags, 0,
           schedule->block_count *
               sizeof(*out_frontier->xcnt.resolved_outgoing_flags));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count,
      sizeof(*out_frontier->block_drain_counter_masks),
      (void**)&out_frontier->block_drain_counter_masks));
  memset(
      out_frontier->block_drain_counter_masks, 0,
      schedule->block_count * sizeof(*out_frontier->block_drain_counter_masks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*out_frontier->block_flags),
      (void**)&out_frontier->block_flags));
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*worklist), (void**)&worklist));
  memset(out_frontier->block_flags, 0,
         schedule->block_count * sizeof(*out_frontier->block_flags));

  loom_amdgpu_wait_frontier_build_local_states(
      out_frontier, planned_block_drain_counter_masks);
  loom_amdgpu_wait_frontier_propagate_static_states(out_frontier, worklist);
  return iree_ok_status();
}

void loom_amdgpu_wait_frontier_begin_block(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t block_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(block_index < frontier->schedule->block_count);
  IREE_ASSERT(frontier->active_block_index == UINT16_MAX);
  frontier->memory.active_state = (loom_amdgpu_wait_memory_state_t){0};
  if (frontier->vmem_results.active_words != NULL) {
    memset(frontier->vmem_results.active_words, 0,
           frontier->vmem_results.word_count *
               sizeof(*frontier->vmem_results.active_words));
  }
  if (frontier->storage_leases.active_words != NULL) {
    memset(frontier->storage_leases.active_words, 0,
           frontier->storage_leases.word_count *
               sizeof(*frontier->storage_leases.active_words));
  }
  frontier->vmem_results.active_flags = 0;
  frontier->xcnt.active_flags = 0;
  frontier->active_block_index = block_index;

  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  if (frontier->block_flags == NULL || graph->blocks == NULL) return;
  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) continue;
    const bool predecessor_resolved =
        iree_any_bit_set(frontier->block_flags[predecessor_index],
                         LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_RESOLVED);
    if (frontier->memory.static_outgoing_states != NULL) {
      const loom_amdgpu_wait_memory_state_t* predecessor_state =
          predecessor_resolved
              ? &frontier->memory.resolved_outgoing_states[predecessor_index]
              : &frontier->memory.static_outgoing_states[predecessor_index];
      loom_amdgpu_wait_memory_state_union_changed(
          &frontier->memory.active_state, predecessor_state);
    }
    if (frontier->vmem_results.static_outgoing_words != NULL) {
      const uint64_t* predecessor_words =
          predecessor_resolved
              ? loom_amdgpu_wait_frontier_const_vmem_result_block_words(
                    frontier, frontier->vmem_results.resolved_outgoing_words,
                    predecessor_index)
              : loom_amdgpu_wait_frontier_const_vmem_result_block_words(
                    frontier, frontier->vmem_results.static_outgoing_words,
                    predecessor_index);
      loom_amdgpu_wait_vmem_result_state_union_changed(
          frontier->vmem_results.active_words, predecessor_words,
          frontier->vmem_results.word_count);
    }
    if (frontier->storage_leases.static_outgoing_words != NULL) {
      const uint64_t* predecessor_words =
          predecessor_resolved
              ? loom_amdgpu_wait_frontier_const_storage_lease_block_words(
                    frontier, frontier->storage_leases.resolved_outgoing_words,
                    predecessor_index)
              : loom_amdgpu_wait_frontier_const_storage_lease_block_words(
                    frontier, frontier->storage_leases.static_outgoing_words,
                    predecessor_index);
      loom_amdgpu_wait_storage_lease_state_union_changed(
          frontier->storage_leases.active_words, predecessor_words,
          frontier->storage_leases.word_count);
    }
    if (frontier->xcnt.static_outgoing_flags != NULL) {
      frontier->xcnt.active_flags |=
          predecessor_resolved
              ? frontier->xcnt.resolved_outgoing_flags[predecessor_index]
              : frontier->xcnt.static_outgoing_flags[predecessor_index];
    }
  }
  if (frontier->vmem_results.active_words != NULL &&
      !loom_amdgpu_wait_vmem_result_state_is_empty(
          frontier->vmem_results.active_words,
          frontier->vmem_results.word_count)) {
    frontier->vmem_results.active_flags |=
        LOOM_AMDGPU_WAIT_VMEM_RESULT_STATE_FLAG_PENDING;
  }
}

uint32_t loom_amdgpu_wait_frontier_memory_query(
    const loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  if (space_flags == 0 || access_flags == 0) return 0;
  IREE_ASSERT_EQ(
      (uint32_t)space_flags & ~LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK, 0u);
  space_flags >>= LOOM_LOW_MEMORY_SPACE_GENERIC;
  uint32_t counter_mask = 0;
  while (space_flags != 0) {
    const uint32_t space_index =
        (uint32_t)iree_math_count_trailing_zeros_u32(space_flags);
    IREE_ASSERT_LT(space_index, LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT);
    const uint16_t access_counter_masks =
        frontier->memory.active_state.access_counter_masks[space_index];
    if (iree_any_bit_set(access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ)) {
      counter_mask |= (uint32_t)(uint8_t)access_counter_masks;
    }
    if (iree_any_bit_set(access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
      counter_mask |= (uint32_t)(access_counter_masks >>
                                 LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT);
    }
    space_flags &= space_flags - 1;
  }
  return counter_mask;
}

uint32_t loom_amdgpu_wait_frontier_memory_dependency_mask(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_frontier_node_t* node) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_ARGUMENT(node);
  if (frontier->memory.static_outgoing_states == NULL ||
      (node->read_space_flags == 0 && node->write_space_flags == 0)) {
    return 0;
  }
  const uint32_t prior_writes =
      node->read_space_flags == 0
          ? 0
          : loom_amdgpu_wait_frontier_memory_query(
                frontier, node->read_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
  const uint32_t prior_reads =
      node->write_space_flags == 0
          ? 0
          : loom_amdgpu_wait_frontier_memory_query(
                frontier, node->write_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ);
  return prior_writes | prior_reads;
}

loom_amdgpu_vmem_result_order_class_t
loom_amdgpu_wait_frontier_query_vmem_result(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  if (frontier->vmem_results.active_words == NULL ||
      !iree_any_bit_set(frontier->vmem_results.active_flags,
                        LOOM_AMDGPU_WAIT_VMEM_RESULT_STATE_FLAG_PENDING)) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
  iree_host_size_t unit_base = 0;
  iree_host_size_t unit_count = 0;
  if (!loom_amdgpu_wait_frontier_map_vector_assignment(
          frontier, assignment, &unit_base, &unit_count)) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
  const uint8_t order_class_flags = loom_amdgpu_wait_vmem_result_state_query(
      frontier->vmem_results.active_words, unit_base, unit_count);
  if (order_class_flags == 0) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
  for (loom_amdgpu_vmem_result_order_class_t order_class =
           LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER;
       order_class < LOOM_AMDGPU_VMEM_RESULT_ORDER_CLASS_COUNT; ++order_class) {
    if (order_class_flags ==
        loom_amdgpu_wait_vmem_result_order_class_flag(order_class)) {
      return order_class;
    }
  }
  return LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN;
}

bool loom_amdgpu_wait_frontier_storage_lease_is_active(
    const loom_amdgpu_wait_frontier_t* frontier, iree_host_size_t lease_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_LT(lease_index, frontier->storage_leases.lease_count);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  return frontier->storage_leases.active_words != NULL &&
         loom_amdgpu_wait_storage_lease_state_test(
             frontier->storage_leases.active_words, lease_index);
}

void loom_amdgpu_wait_frontier_retire_storage_lease(
    loom_amdgpu_wait_frontier_t* frontier, iree_host_size_t lease_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_LT(lease_index, frontier->storage_leases.lease_count);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  IREE_ASSERT(frontier->storage_leases.active_words != NULL);
  loom_amdgpu_wait_storage_lease_state_clear(
      frontier->storage_leases.active_words, lease_index);
}

loom_amdgpu_wait_xcnt_group_flags_t
loom_amdgpu_wait_frontier_active_xcnt_groups(
    const loom_amdgpu_wait_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  return frontier->xcnt.active_flags;
}

void loom_amdgpu_wait_frontier_prepare_xcnt_producer(
    loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_xcnt_group_flags_t group_flags) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  IREE_ASSERT(group_flags == LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM ||
              group_flags == LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM);
  const loom_amdgpu_wait_xcnt_group_flags_t other_group_flags =
      frontier->xcnt.active_flags &
      (loom_amdgpu_wait_xcnt_group_flags_t)~group_flags;
  if (other_group_flags != 0 && frontier->storage_leases.active_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain_xcnt_groups(
        frontier, frontier->storage_leases.active_words, other_group_flags);
  }
  frontier->xcnt.active_flags &= group_flags;
}

void loom_amdgpu_wait_frontier_note_xcnt_producer(
    loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_xcnt_group_flags_t group_flags) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  IREE_ASSERT(group_flags == LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM ||
              group_flags == LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM);
  IREE_ASSERT(frontier->xcnt.active_flags == 0 ||
              frontier->xcnt.active_flags == group_flags);
  frontier->xcnt.active_flags = group_flags;
}

void loom_amdgpu_wait_frontier_drain(loom_amdgpu_wait_frontier_t* frontier,
                                     uint32_t counter_mask) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  loom_amdgpu_wait_memory_state_drain(&frontier->memory.active_state,
                                      counter_mask);
  if (frontier->vmem_results.active_words != NULL &&
      iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
    memset(frontier->vmem_results.active_words, 0,
           frontier->vmem_results.word_count *
               sizeof(*frontier->vmem_results.active_words));
    frontier->vmem_results.active_flags = 0;
  }
  if (frontier->storage_leases.active_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain(
        frontier, frontier->storage_leases.active_words, counter_mask);
  }
  if (iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_X)) {
    frontier->xcnt.active_flags = 0;
  }
}

void loom_amdgpu_wait_frontier_end_block(
    loom_amdgpu_wait_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  const uint16_t block_index = frontier->active_block_index;
  loom_amdgpu_wait_memory_state_t* outgoing_memory_state =
      frontier->memory.resolved_outgoing_states == NULL
          ? NULL
          : &frontier->memory.resolved_outgoing_states[block_index];
  if (outgoing_memory_state != NULL) {
    *outgoing_memory_state = frontier->memory.active_state;
  }
  uint64_t* outgoing_vmem_result_words =
      frontier->vmem_results.resolved_outgoing_words == NULL
          ? NULL
          : loom_amdgpu_wait_frontier_vmem_result_block_words(
                frontier, frontier->vmem_results.resolved_outgoing_words,
                block_index);
  if (outgoing_vmem_result_words != NULL) {
    memcpy(outgoing_vmem_result_words, frontier->vmem_results.active_words,
           frontier->vmem_results.word_count *
               sizeof(*outgoing_vmem_result_words));
  }
  uint64_t* outgoing_storage_lease_words =
      frontier->storage_leases.resolved_outgoing_words == NULL
          ? NULL
          : loom_amdgpu_wait_frontier_storage_lease_block_words(
                frontier, frontier->storage_leases.resolved_outgoing_words,
                block_index);
  if (outgoing_storage_lease_words != NULL) {
    memcpy(outgoing_storage_lease_words, frontier->storage_leases.active_words,
           frontier->storage_leases.word_count *
               sizeof(*outgoing_storage_lease_words));
  }
  loom_amdgpu_wait_xcnt_group_flags_t* outgoing_xcnt_group_flags =
      frontier->xcnt.resolved_outgoing_flags == NULL
          ? NULL
          : &frontier->xcnt.resolved_outgoing_flags[block_index];
  if (outgoing_xcnt_group_flags != NULL) {
    *outgoing_xcnt_group_flags = frontier->xcnt.active_flags;
  }
  if (outgoing_memory_state != NULL || outgoing_vmem_result_words != NULL ||
      outgoing_storage_lease_words != NULL) {
    const loom_low_schedule_block_t* block =
        &frontier->schedule->blocks[block_index];
    iree_host_size_t next_storage_lease_index =
        outgoing_storage_lease_words == NULL
            ? 0
            : loom_amdgpu_wait_frontier_storage_lease_lower_bound(
                  frontier, block->scheduled_node_start);
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      const uint32_t node_index =
          frontier->schedule->scheduled_node_indices[packet_index];
      const loom_amdgpu_wait_frontier_node_t* node =
          &frontier->nodes[node_index];
      if (outgoing_memory_state != NULL) {
        const uint32_t read_counter_mask =
            node->read_counter_mask &
            ~node->drained_after_production_counter_mask;
        const uint32_t write_counter_mask =
            node->write_counter_mask &
            ~node->drained_after_production_counter_mask;
        loom_amdgpu_wait_memory_state_add_node(
            outgoing_memory_state, node, read_counter_mask, write_counter_mask);
      }
      if (outgoing_vmem_result_words != NULL &&
          !iree_any_bit_set(node->drained_after_production_counter_mask,
                            LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
        loom_amdgpu_wait_frontier_publish_node_vmem_results(
            frontier, outgoing_vmem_result_words, node_index);
      }
      if (outgoing_storage_lease_words != NULL) {
        loom_amdgpu_wait_frontier_publish_packet_storage_leases(
            frontier, outgoing_storage_lease_words, packet_index,
            node->drained_after_production_counter_mask,
            &next_storage_lease_index);
      }
    }
  }
  if (frontier->block_flags != NULL) {
    frontier->block_flags[block_index] |=
        LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_RESOLVED;
  }
  frontier->memory.active_state = (loom_amdgpu_wait_memory_state_t){0};
  if (frontier->vmem_results.active_words != NULL) {
    memset(frontier->vmem_results.active_words, 0,
           frontier->vmem_results.word_count *
               sizeof(*frontier->vmem_results.active_words));
  }
  if (frontier->storage_leases.active_words != NULL) {
    memset(frontier->storage_leases.active_words, 0,
           frontier->storage_leases.word_count *
               sizeof(*frontier->storage_leases.active_words));
  }
  frontier->vmem_results.active_flags = 0;
  frontier->xcnt.active_flags = 0;
  frontier->active_block_index = UINT16_MAX;
}
