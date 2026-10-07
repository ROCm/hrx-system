// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_frontier.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_congruence.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/target/arch/amdgpu/planning/wait_frontier_budget.h"
#include "loom/util/cfg_graph.h"

enum {
  LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED = 1u << 0,
  LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_RESOLVED = 1u << 1,
  LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_SUCCESSOR = 1u << 0,
  LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_PREDECESSOR = 1u << 1,
  LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK =
      ((1u << LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT) - 1u)
      << LOOM_LOW_MEMORY_SPACE_GENERIC,
  LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT = 8,
  LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD = 64,
  LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD = 64,
};

static_assert(LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT <= 8,
              "memory-space frontier flags must fit in one byte");
static_assert(LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT <= 8,
              "memory frontier counter masks must fit in one byte");
static_assert(LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL ==
                  (1u << LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT) - 1u,
              "memory frontier counter masks must use dense low bits");
static_assert(LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT ==
                  LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MEMORY_SPACE_COUNT,
              "precise budget must track every coarse memory-space bit");
static_assert((LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM |
               LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM) ==
                  (1u << LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_XCNT_GROUP_COUNT) -
                      1u,
              "precise budget must track every XCNT group bit");
static_assert(sizeof(loom_amdgpu_wait_memory_state_t) ==
                  LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT * sizeof(uint16_t),
              "memory frontier state must not acquire padding");
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
    if (state->access_counter_masks[space] != 0) {
      return false;
    }
  }
  return true;
}

static uint64_t* loom_amdgpu_wait_frontier_precise_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->memory.precise_word_count;
}

static const uint64_t* loom_amdgpu_wait_frontier_const_precise_block_words(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* states,
    iree_host_size_t block_index) {
  return states + block_index * frontier->memory.precise_word_count;
}

static bool loom_amdgpu_wait_precise_state_union_changed(
    uint64_t* target, const uint64_t* source, iree_host_size_t word_count) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    const uint64_t result = target[i] | source[i];
    changed |= result != target[i];
    target[i] = result;
  }
  return changed;
}

static bool loom_amdgpu_wait_precise_state_is_empty(
    const uint64_t* words, iree_host_size_t word_count) {
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    if (words[i] != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_wait_precise_state_test(const uint64_t* words,
                                                iree_host_size_t access_index,
                                                uint32_t counter_slot) {
  const iree_host_size_t bit_index =
      access_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT + counter_slot;
  return (words[bit_index / LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD] &
          (UINT64_C(1) << (bit_index %
                           LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD))) != 0;
}

static void loom_amdgpu_wait_precise_state_set(uint64_t* words,
                                               iree_host_size_t access_index,
                                               uint32_t counter_mask) {
  counter_mask &= LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  while (counter_mask != 0) {
    const uint32_t counter_slot =
        (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
    const iree_host_size_t bit_index =
        access_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT + counter_slot;
    words[bit_index / LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD] |=
        UINT64_C(1) << (bit_index % LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD);
    counter_mask &= counter_mask - 1;
  }
}

static void loom_amdgpu_wait_precise_state_drain(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    uint32_t counter_mask) {
  counter_mask &= LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  if (counter_mask == 0 || words == NULL) {
    return;
  }
  for (iree_host_size_t access_index = 0;
       access_index < frontier->memory.precise_access_count; ++access_index) {
    uint32_t remaining_counter_mask = counter_mask;
    while (remaining_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(remaining_counter_mask);
      const iree_host_size_t bit_index =
          access_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT + counter_slot;
      words[bit_index / LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD] &= ~(
          UINT64_C(1) << (bit_index % LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD));
      remaining_counter_mask &= remaining_counter_mask - 1;
    }
  }
}

static bool loom_amdgpu_wait_precise_state_union_after_drain_changed(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* target,
    const uint64_t* source, uint32_t counter_mask) {
  if (counter_mask == 0) {
    return loom_amdgpu_wait_precise_state_union_changed(
        target, source, frontier->memory.precise_word_count);
  }
  bool changed = false;
  for (iree_host_size_t access_index = 0;
       access_index < frontier->memory.precise_access_count; ++access_index) {
    uint32_t retained_counter_mask =
        frontier->memory.precise_accesses[access_index].producer_counter_mask &
        ~counter_mask;
    while (retained_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(retained_counter_mask);
      if (loom_amdgpu_wait_precise_state_test(source, access_index,
                                              counter_slot) &&
          !loom_amdgpu_wait_precise_state_test(target, access_index,
                                               counter_slot)) {
        loom_amdgpu_wait_precise_state_set(
            target, access_index,
            loom_amdgpu_wait_counter_mask_from_slot(counter_slot));
        changed = true;
      }
      retained_counter_mask &= retained_counter_mask - 1;
    }
  }
  return changed;
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

static void loom_amdgpu_wait_storage_lease_state_update_word(
    uint64_t* words, iree_host_size_t word_index, uint64_t word,
    loom_low_allocation_storage_lease_selection_t* selection) {
  if (selection != NULL) {
    uint64_t changed_bits = words[word_index] ^ word;
    while (changed_bits != 0) {
      const uint32_t bit_index =
          (uint32_t)iree_math_count_trailing_zeros_u64(changed_bits);
      const uint32_t lease_index =
          (uint32_t)(word_index * LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD +
                     bit_index);
      loom_low_allocation_storage_lease_selection_set_active(
          selection, lease_index, (word & (UINT64_C(1) << bit_index)) != 0);
      changed_bits &= changed_bits - 1;
    }
  }
  words[word_index] = word;
}

static bool loom_amdgpu_wait_storage_lease_state_union_changed(
    uint64_t* target, const uint64_t* source, iree_host_size_t word_count,
    loom_low_allocation_storage_lease_selection_t* selection) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    const uint64_t result = target[i] | source[i];
    changed |= result != target[i];
    loom_amdgpu_wait_storage_lease_state_update_word(target, i, result,
                                                     selection);
  }
  return changed;
}

static bool loom_amdgpu_wait_storage_lease_state_is_empty(
    const uint64_t* words, iree_host_size_t word_count) {
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    if (words[i] != 0) {
      return false;
    }
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
    uint64_t* words, iree_host_size_t lease_index,
    loom_low_allocation_storage_lease_selection_t* selection) {
  const iree_host_size_t word_index =
      lease_index / LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD;
  const uint32_t bit_index =
      (uint32_t)(lease_index % LOOM_AMDGPU_WAIT_STORAGE_LEASES_PER_WORD);
  loom_amdgpu_wait_storage_lease_state_update_word(
      words, word_index, words[word_index] & ~(UINT64_C(1) << bit_index),
      selection);
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
    uint32_t counter_mask,
    loom_low_allocation_storage_lease_selection_t* selection) {
  counter_mask &= LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  while (counter_mask != 0) {
    const uint32_t counter_slot =
        (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
    const uint64_t* release_counter_words =
        loom_amdgpu_wait_frontier_const_storage_lease_counter_words(
            frontier, counter_slot);
    for (iree_host_size_t word_index = 0;
         word_index < frontier->storage_leases.word_count; ++word_index) {
      loom_amdgpu_wait_storage_lease_state_update_word(
          words, word_index,
          words[word_index] & ~release_counter_words[word_index], selection);
    }
    counter_mask &= counter_mask - 1;
  }
}

static bool loom_amdgpu_wait_storage_lease_state_union_after_drain_changed(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* target,
    const uint64_t* source, const uint64_t* completed_words,
    uint32_t counter_mask) {
  if (counter_mask == 0 && completed_words == NULL) {
    return loom_amdgpu_wait_storage_lease_state_union_changed(
        target, source, frontier->storage_leases.word_count,
        /*selection=*/NULL);
  }
  bool changed = false;
  // The inner mask traversal is bounded by the eight architectural wait
  // counters, keeping this linear in the storage-lease word count.
  for (iree_host_size_t word_index = 0;
       word_index < frontier->storage_leases.word_count; ++word_index) {
    uint64_t retained_source = source[word_index];
    if (completed_words != NULL) {
      retained_source &= ~completed_words[word_index];
    }
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
    loom_amdgpu_wait_xcnt_group_flags_t group_flags,
    loom_low_allocation_storage_lease_selection_t* selection) {
  for (iree_host_size_t lease_index = 0;
       lease_index < frontier->storage_leases.lease_count; ++lease_index) {
    if (!iree_any_bit_set(group_flags,
                          loom_amdgpu_wait_frontier_storage_lease_xcnt_group(
                              frontier, lease_index))) {
      continue;
    }
    loom_amdgpu_wait_storage_lease_state_clear(words, lease_index, selection);
  }
}

static void loom_amdgpu_wait_memory_state_drain(
    loom_amdgpu_wait_memory_state_t* state, uint32_t counter_mask) {
  if (counter_mask == 0) {
    return;
  }
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
  if (producer_space_flags == 0 || access_counter_masks == 0) {
    return;
  }
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
  if ((read_counter_mask | write_counter_mask) == 0) {
    return;
  }
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

static bool loom_amdgpu_wait_frontier_effect_is_dependency_memory(
    const loom_low_schedule_effect_use_t* effect) {
  if (!iree_any_bit_set(effect->effect_flags,
                        LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
    return false;
  }
  switch (effect->memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return true;
    default:
      return false;
  }
}

static loom_amdgpu_wait_memory_access_flags_t
loom_amdgpu_wait_frontier_effect_access_flags(
    const loom_low_schedule_effect_use_t* effect) {
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE;
    default:
      return 0;
  }
}

static uint32_t loom_amdgpu_wait_frontier_effect_counter_mask(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_low_schedule_effect_use_t* effect) {
  if (effect->counter_id != LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    return loom_amdgpu_wait_counter_mask(effect->counter_id);
  }
  const loom_amdgpu_wait_frontier_node_t* node =
      &frontier->nodes[effect->node_index];
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return node->read_counter_mask;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return node->write_counter_mask;
    default:
      return 0;
  }
}

static const loom_low_memory_access_summary_t*
loom_amdgpu_wait_frontier_classify_effect_summary(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_effect_use_t* effect,
    bool* out_generic_summary) {
  if (out_generic_summary != NULL) {
    *out_generic_summary = false;
  }
  if (schedule->memory_accesses == NULL ||
      !loom_amdgpu_wait_frontier_effect_is_dependency_memory(effect) ||
      loom_amdgpu_wait_frontier_effect_access_flags(effect) == 0 ||
      effect->node_index >= schedule->node_count) {
    return NULL;
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[effect->node_index];
  if (node->descriptor == NULL || node->op == NULL ||
      iree_any_bit_set(node->op->instance_flags,
                       LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
    return NULL;
  }
  const loom_low_memory_access_summary_t* summary =
      loom_low_memory_access_map_lookup(schedule->memory_accesses, node->op,
                                        effect->effect_ordinal);
  if (summary == NULL) {
    return NULL;
  }
  if (loom_low_memory_access_normalize_space(summary->memory_space) ==
      LOOM_LOW_MEMORY_SPACE_GENERIC) {
    if (out_generic_summary != NULL) {
      *out_generic_summary = true;
    }
    return NULL;
  }
  return summary;
}

static const loom_low_memory_access_summary_t*
loom_amdgpu_wait_frontier_precise_effect_summary(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_effect_use_t* effect) {
  return loom_amdgpu_wait_frontier_classify_effect_summary(
      schedule, effect, /*out_generic_summary=*/NULL);
}

static void loom_amdgpu_wait_frontier_add_coarse_effect(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_low_schedule_effect_use_t* effect,
    loom_amdgpu_wait_frontier_node_t* coarse_node) {
  if (!loom_amdgpu_wait_frontier_effect_is_dependency_memory(effect)) {
    return;
  }
  const uint32_t counter_mask =
      loom_amdgpu_wait_frontier_effect_counter_mask(frontier, effect);
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      coarse_node->read_space_flags |=
          loom_amdgpu_wait_memory_space_flag(effect->memory_space);
      coarse_node->read_counter_mask |= counter_mask;
      break;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      coarse_node->write_space_flags |=
          loom_amdgpu_wait_memory_space_flag(effect->memory_space);
      coarse_node->write_counter_mask |= counter_mask;
      break;
    default:
      break;
  }
}

static bool loom_amdgpu_wait_frontier_add_allocation_bytes(
    iree_host_size_t count, iree_host_size_t element_size,
    iree_host_size_t* inout_total_bytes) {
  iree_host_size_t bytes = 0;
  return iree_host_size_checked_mul(count, element_size, &bytes) &&
         iree_host_size_checked_add(*inout_total_bytes, bytes,
                                    inout_total_bytes);
}

static bool loom_amdgpu_wait_frontier_edge_is_backedge(
    const loom_cfg_graph_t* graph, uint16_t source_block,
    uint16_t target_block) {
  const loom_cfg_block_info_t* source = &graph->blocks[source_block];
  const loom_cfg_block_info_t* target = &graph->blocks[target_block];
  return source->preorder != UINT16_MAX && target->preorder != UINT16_MAX &&
         target->preorder <= source->preorder &&
         source->preorder < target->preorder_end;
}

// Exact state may cross only edges that cannot participate in a cycle. The
// graph's reverse postorder is a topological order after DFS backedges are
// removed; validating that property makes corrupt-but-in-range DFS intervals a
// conservative coarse-state fallback instead of an unsound alias proof.
// Successor and predecessor spans must also describe the same edge multiset:
// static propagation consumes successors while dynamic block entry consumes
// predecessors.
static bool loom_amdgpu_wait_frontier_validate_acyclic_edge_partition(
    const loom_cfg_graph_t* graph, uint32_t* block_ranks,
    uint16_t* preorder_stack, uint8_t* edge_seen_flags) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(block_ranks);
  IREE_ASSERT_ARGUMENT(preorder_stack);
  if (graph->edge_count != 0 &&
      (graph->edges == NULL || graph->successor_indices == NULL ||
       graph->successor_edge_indices == NULL ||
       graph->predecessor_indices == NULL ||
       graph->predecessor_edge_indices == NULL || edge_seen_flags == NULL)) {
    return false;
  }
  memset(block_ranks, 0xFF, graph->block_count * sizeof(*block_ranks));
  if (graph->edge_count != 0) {
    memset(edge_seen_flags, 0, graph->edge_count * sizeof(*edge_seen_flags));
  }

  iree_host_size_t reachable_count = 0;
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    reachable_count += graph->blocks[i].reachable;
  }
  if (reachable_count == 0) {
    return false;
  }
  for (iree_host_size_t block_index = 0; block_index < graph->block_count;
       ++block_index) {
    const loom_cfg_block_info_t* block = &graph->blocks[block_index];
    if (block->successor_start > graph->edge_count ||
        block->successor_edge_start > graph->edge_count ||
        block->successor_count > graph->edge_count - block->successor_start ||
        block->successor_count >
            graph->edge_count - block->successor_edge_start ||
        block->predecessor_start > graph->edge_count ||
        block->predecessor_edge_start > graph->edge_count ||
        block->predecessor_count >
            graph->edge_count - block->predecessor_start ||
        block->predecessor_count >
            graph->edge_count - block->predecessor_edge_start) {
      return false;
    }
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, (uint16_t)block_index);
    const loom_cfg_edge_index_span_t successor_edges =
        loom_cfg_graph_successor_edges(graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t target_index = successors.values[i];
      const loom_cfg_edge_index_t edge_index = successor_edges.values[i];
      if (target_index >= graph->block_count ||
          edge_index >= graph->edge_count ||
          iree_any_bit_set(edge_seen_flags[edge_index],
                           LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_SUCCESSOR)) {
        return false;
      }
      const loom_cfg_edge_info_t* edge = &graph->edges[edge_index];
      if (edge->source_block_index != block_index ||
          edge->target_block_index != target_index ||
          edge->successor_index != i) {
        return false;
      }
      edge_seen_flags[edge_index] |=
          LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_SUCCESSOR;
    }
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, (uint16_t)block_index);
    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const uint16_t source_index = predecessors.values[i];
      const loom_cfg_edge_index_t edge_index = predecessor_edges.values[i];
      if (source_index >= graph->block_count ||
          edge_index >= graph->edge_count ||
          iree_any_bit_set(edge_seen_flags[edge_index],
                           LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_PREDECESSOR)) {
        return false;
      }
      const loom_cfg_edge_info_t* edge = &graph->edges[edge_index];
      if (edge->source_block_index != source_index ||
          edge->target_block_index != block_index) {
        return false;
      }
      edge_seen_flags[edge_index] |=
          LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_PREDECESSOR;
    }
    if (!block->reachable) {
      if (block->preorder != UINT16_MAX || block->preorder_end != 0 ||
          block->parent != UINT16_MAX) {
        return false;
      }
      continue;
    }
    if (block->preorder == UINT16_MAX || block->preorder >= reachable_count ||
        block->preorder_end <= block->preorder ||
        block->preorder_end > reachable_count ||
        block_ranks[block->preorder] != UINT32_MAX) {
      return false;
    }
    block_ranks[block->preorder] = (uint32_t)block_index;
  }
  const uint8_t required_edge_flags =
      LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_SUCCESSOR |
      LOOM_AMDGPU_WAIT_FRONTIER_EDGE_SEEN_PREDECESSOR;
  for (iree_host_size_t edge_index = 0; edge_index < graph->edge_count;
       ++edge_index) {
    if (edge_seen_flags[edge_index] != required_edge_flags) {
      return false;
    }
  }
  if (block_ranks[0] != 0) {
    return false;
  }

  iree_host_size_t stack_count = 0;
  for (iree_host_size_t preorder = 0; preorder < reachable_count; ++preorder) {
    const uint16_t block_index = (uint16_t)block_ranks[preorder];
    const loom_cfg_block_info_t* block = &graph->blocks[block_index];
    while (stack_count != 0 &&
           preorder >=
               graph->blocks[preorder_stack[stack_count - 1]].preorder_end) {
      --stack_count;
    }
    if (preorder == 0) {
      if (block->parent != UINT16_MAX ||
          block->preorder_end != reachable_count) {
        return false;
      }
    } else {
      if (stack_count == 0 ||
          block->parent != preorder_stack[stack_count - 1]) {
        return false;
      }
      const loom_cfg_block_info_t* parent = &graph->blocks[block->parent];
      if (block->preorder_end > parent->preorder_end) {
        return false;
      }
      bool parent_edge_found = false;
      const loom_cfg_block_index_span_t predecessors =
          loom_cfg_graph_predecessors(graph, block_index);
      for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
        parent_edge_found |= predecessors.values[i] == block->parent;
      }
      if (!parent_edge_found) {
        return false;
      }
    }
    preorder_stack[stack_count++] = block_index;
  }

  memset(block_ranks, 0xFF, graph->block_count * sizeof(*block_ranks));
  if (graph->reverse_postorder.count != reachable_count ||
      (reachable_count != 0 && graph->reverse_postorder.values == NULL)) {
    return false;
  }
  for (iree_host_size_t rank = 0; rank < reachable_count; ++rank) {
    const uint16_t block_index = graph->reverse_postorder.values[rank];
    if (block_index >= graph->block_count ||
        !graph->blocks[block_index].reachable ||
        block_ranks[block_index] != UINT32_MAX) {
      return false;
    }
    block_ranks[block_index] = (uint32_t)rank;
  }
  for (iree_host_size_t source_index = 0; source_index < graph->block_count;
       ++source_index) {
    const loom_cfg_block_info_t* source = &graph->blocks[source_index];
    if (!source->reachable) {
      continue;
    }
    if (block_ranks[source_index] == UINT32_MAX) {
      return false;
    }
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, (uint16_t)source_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t target_index = successors.values[i];
      if (!graph->blocks[target_index].reachable ||
          block_ranks[target_index] == UINT32_MAX ||
          (!loom_amdgpu_wait_frontier_edge_is_backedge(
               graph, (uint16_t)source_index, target_index) &&
           block_ranks[source_index] >= block_ranks[target_index])) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_amdgpu_wait_frontier_budget_add_assign(uint64_t value,
                                                        uint64_t* inout_value) {
  return iree_checked_add_u64(*inout_value, value, inout_value);
}

static bool loom_amdgpu_wait_frontier_budget_u64_from_size(
    iree_host_size_t value, uint64_t* out_value) {
  *out_value = (uint64_t)value;
  return (iree_host_size_t)*out_value == value;
}

typedef struct loom_amdgpu_wait_frontier_budget_node_t {
  loom_amdgpu_wait_memory_space_flags_t coarse_read_space_flags;
  loom_amdgpu_wait_memory_space_flags_t coarse_write_space_flags;
  loom_amdgpu_wait_memory_space_flags_t precise_read_space_flags;
  loom_amdgpu_wait_memory_space_flags_t precise_write_space_flags;
} loom_amdgpu_wait_frontier_budget_node_t;

static bool loom_amdgpu_wait_frontier_block_completion_mask(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const uint32_t* planned_block_drain_counter_masks, uint16_t block_index,
    uint32_t* out_completion_mask) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  if (block->scheduled_node_start > schedule->scheduled_node_count ||
      block->scheduled_node_count >
          schedule->scheduled_node_count - block->scheduled_node_start ||
      (block->scheduled_node_count != 0 &&
       schedule->scheduled_node_indices == NULL) ||
      (planned_block_drain_counter_masks != NULL && block->block == NULL)) {
    return false;
  }
  uint32_t completion_mask = 0;
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start + i;
    const uint32_t node_index = schedule->scheduled_node_indices[packet_index];
    if (node_index >= schedule->node_count ||
        schedule->nodes[node_index].block_index != block_index) {
      return false;
    }
    const uint32_t node_completion_mask =
        completion_nodes[node_index].reset_counter_mask |
        completion_nodes[node_index].completed_before_block_exit_counter_mask;
    if ((node_completion_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL) != 0) {
      return false;
    }
    completion_mask |= node_completion_mask;
    if (planned_block_drain_counter_masks != NULL &&
        schedule->nodes[node_index].op == block->block->last_op) {
      const uint32_t planned_mask =
          planned_block_drain_counter_masks[block_index];
      if ((planned_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL) != 0) {
        return false;
      }
      completion_mask |= planned_mask;
    }
  }
  *out_completion_mask = completion_mask;
  return true;
}

static bool loom_amdgpu_wait_frontier_collect_precise_budget(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const uint32_t* planned_block_drain_counter_masks,
    const loom_amdgpu_wait_frontier_precise_runtime_bounds_t* runtime_bounds,
    loom_amdgpu_wait_frontier_budget_node_t* budget_nodes,
    iree_host_size_t dependency_count,
    iree_host_size_t expected_precise_access_count,
    iree_host_size_t expected_precise_word_count,
    iree_host_size_t allocation_byte_count,
    loom_amdgpu_wait_frontier_precise_budget_input_t* out_input) {
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  *out_input = (loom_amdgpu_wait_frontier_precise_budget_input_t){
      .memory_space_count = LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT,
      .collector_heavy_path_possible = false,
  };
  if (!loom_amdgpu_wait_frontier_budget_u64_from_size(
          expected_precise_access_count, &out_input->precise_access_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          expected_precise_word_count, &out_input->precise_word_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          frontier->storage_leases.lease_count,
          &out_input->storage_lease_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->block_count, &out_input->block_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          allocation_byte_count, &out_input->allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->effect_use_count, &out_input->effect_use_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          graph->edge_count, &out_input->cfg_edge_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->scheduled_node_count,
          &out_input->scheduled_node_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->node_count, &out_input->node_count)) {
    return false;
  }
  if (!loom_amdgpu_wait_frontier_budget_u64_from_size(
          dependency_count, &out_input->dependency_count)) {
    return false;
  }
  out_input->dependency_read_query_count = 0;
  out_input->dependency_write_query_count = 0;
  out_input->barrier_query_count = runtime_bounds->barrier_query_count;
  out_input->program_exit_query_count =
      runtime_bounds->program_exit_query_count;
  out_input->producer_completion_call_count =
      runtime_bounds->producer_completion_call_count;
  out_input->producer_completion_guard_reject_count =
      runtime_bounds->producer_completion_guard_reject_count;
  out_input->producer_completion_full_path_count =
      runtime_bounds->producer_completion_full_path_count;
  out_input->producer_completion_full_path_read_space_count =
      runtime_bounds->producer_completion_full_path_read_space_count;
  out_input->producer_completion_full_path_write_space_count =
      runtime_bounds->producer_completion_full_path_write_space_count;
  out_input->direct_memory_query_count =
      runtime_bounds->direct_memory_query_count;
  out_input->unclassified_memory_query_count =
      runtime_bounds->unclassified_memory_query_count;
  out_input->direct_producer_completion_count =
      runtime_bounds->direct_producer_completion_count;
  out_input->unclassified_producer_completion_count =
      runtime_bounds->unclassified_producer_completion_count;

  uint64_t scanned_precise_access_count = 0;
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    bool generic_summary = false;
    const loom_low_memory_access_summary_t* summary =
        loom_amdgpu_wait_frontier_classify_effect_summary(
            schedule, effect, &generic_summary);
    out_input->collector_heavy_path_possible |= generic_summary;
    if (summary == NULL) {
      if (loom_amdgpu_wait_frontier_effect_is_dependency_memory(effect)) {
        loom_amdgpu_wait_memory_space_flags_t* coarse_space_flags = NULL;
        switch (effect->kind) {
          case LOOM_LOW_EFFECT_KIND_READ:
            coarse_space_flags =
                &budget_nodes[effect->node_index].coarse_read_space_flags;
            break;
          case LOOM_LOW_EFFECT_KIND_WRITE:
            coarse_space_flags =
                &budget_nodes[effect->node_index].coarse_write_space_flags;
            break;
          default:
            break;
        }
        if (coarse_space_flags != NULL) {
          *coarse_space_flags |=
              loom_amdgpu_wait_memory_space_flag(effect->memory_space);
        }
      }
      continue;
    }
    if (!loom_amdgpu_wait_frontier_budget_add_assign(
            1, &scanned_precise_access_count)) {
      return false;
    }
    const uint32_t producer_counter_mask =
        loom_amdgpu_wait_frontier_effect_counter_mask(frontier, effect) &
        LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
    if (!loom_amdgpu_wait_frontier_budget_add_assign(
            (uint64_t)iree_math_count_ones_u32(producer_counter_mask),
            &out_input->producer_counter_bit_count)) {
      return false;
    }

    uint64_t* access_count = NULL;
    uint64_t* producer_bit_count = NULL;
    uint64_t* term_sum = NULL;
    switch (loom_amdgpu_wait_frontier_effect_access_flags(effect)) {
      case LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ:
        access_count = &out_input->precise_read_count;
        producer_bit_count = &out_input->read_producer_counter_bit_count;
        term_sum = &out_input->read_effective_term_sum;
        budget_nodes[effect->node_index].precise_read_space_flags |=
            loom_amdgpu_wait_memory_space_flag(summary->memory_space);
        break;
      case LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE:
        access_count = &out_input->precise_write_count;
        producer_bit_count = &out_input->write_producer_counter_bit_count;
        term_sum = &out_input->write_effective_term_sum;
        budget_nodes[effect->node_index].precise_write_space_flags |=
            loom_amdgpu_wait_memory_space_flag(summary->memory_space);
        break;
      default:
        return false;
    }
    if (!loom_amdgpu_wait_frontier_budget_add_assign(1, access_count) ||
        !loom_amdgpu_wait_frontier_budget_add_assign(
            (uint64_t)iree_math_count_ones_u32(producer_counter_mask),
            producer_bit_count)) {
      return false;
    }
    const loom_symbolic_expr_t* form =
        summary->relative_interval == NULL
            ? NULL
            : loom_symbolic_congruence_form(
                  &summary->relative_interval->origin);
    if (form == NULL || !loom_symbolic_expr_is_linear(form)) {
      continue;
    }
    uint64_t term_count = 0;
    if ((form->term_count != 0 && form->terms == NULL) ||
        !loom_amdgpu_wait_frontier_budget_u64_from_size(form->term_count,
                                                        &term_count) ||
        !loom_amdgpu_wait_frontier_budget_add_assign(term_count, term_sum)) {
      return false;
    }
    out_input->maximum_effective_term_count =
        iree_max(out_input->maximum_effective_term_count, term_count);
  }
  if (scanned_precise_access_count != out_input->precise_access_count) {
    return false;
  }

  const uint32_t valid_xcnt_group_mask = LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM |
                                         LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM;
  uint32_t observed_xcnt_group_mask = 0;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_frontier_budget_node_t* budget_node = &budget_nodes[i];
    const loom_low_schedule_node_t* schedule_node = &schedule->nodes[i];
    const loom_amdgpu_wait_frontier_node_t* node = &frontier->nodes[i];
    if (schedule_node->descriptor == NULL ||
        (schedule_node->op != NULL &&
         iree_any_bit_set(schedule_node->op->instance_flags,
                          LOOM_MEMORY_ACCESS_FLAG_VOLATILE))) {
      budget_node->coarse_read_space_flags |= node->read_space_flags;
      budget_node->coarse_write_space_flags |= node->write_space_flags;
    }
    const loom_amdgpu_wait_memory_space_flags_t represented_read_space_flags =
        budget_node->coarse_read_space_flags |
        budget_node->precise_read_space_flags;
    const loom_amdgpu_wait_memory_space_flags_t represented_write_space_flags =
        budget_node->coarse_write_space_flags |
        budget_node->precise_write_space_flags;
    if (!iree_all_bits_set(represented_read_space_flags,
                           node->read_space_flags)) {
      budget_node->coarse_read_space_flags |= node->read_space_flags;
    }
    if (!iree_all_bits_set(represented_write_space_flags,
                           node->write_space_flags)) {
      budget_node->coarse_write_space_flags |= node->write_space_flags;
    }
    observed_xcnt_group_mask |= node->xcnt_group_flags;
  }
  if ((observed_xcnt_group_mask & ~valid_xcnt_group_mask) != 0) {
    return false;
  }
  out_input->xcnt_group_count = (uint64_t)iree_math_count_ones_u32(
      observed_xcnt_group_mask & valid_xcnt_group_mask);

  for (iree_host_size_t block_index = 0; block_index < graph->block_count;
       ++block_index) {
    if (graph->blocks[block_index].reachable &&
        !loom_amdgpu_wait_frontier_budget_add_assign(
            1, &out_input->reachable_block_count)) {
      return false;
    }
  }
  for (iree_host_size_t target_index = 0; target_index < graph->block_count;
       ++target_index) {
    uint32_t completion_mask = 0;
    if (!loom_amdgpu_wait_frontier_block_completion_mask(
            schedule, completion_nodes, planned_block_drain_counter_masks,
            (uint16_t)target_index, &completion_mask)) {
      return false;
    }
    out_input->collector_heavy_path_possible |= completion_mask != 0;
    if (!graph->blocks[target_index].reachable) {
      continue;
    }
    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, (uint16_t)target_index);
    for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
      const loom_cfg_edge_info_t* edge =
          &graph->edges[predecessor_edges.values[i]];
      if (!graph->blocks[edge->source_block_index].reachable) {
        continue;
      }
      uint64_t* edge_count = &out_input->unfiltered_forward_edge_count;
      if (loom_amdgpu_wait_frontier_edge_is_backedge(
              graph, edge->source_block_index, edge->target_block_index)) {
        edge_count = &out_input->backedge_count;
      } else if (completion_mask != 0) {
        edge_count = &out_input->filtered_forward_edge_count;
      }
      if (!loom_amdgpu_wait_frontier_budget_add_assign(1, edge_count)) {
        return false;
      }
    }
  }

  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (!loom_amdgpu_wait_frontier_budget_add_assign(
              budget_nodes[node_index].coarse_read_space_flags != 0,
              &out_input->dependency_read_query_count) ||
          !loom_amdgpu_wait_frontier_budget_add_assign(
              budget_nodes[node_index].coarse_write_space_flags != 0,
              &out_input->dependency_write_query_count)) {
        return false;
      }
      uint32_t reset_mask = completion_nodes[node_index].reset_counter_mask;
      if (planned_block_drain_counter_masks != NULL &&
          schedule->nodes[node_index].op == block->block->last_op) {
        reset_mask |= planned_block_drain_counter_masks[block_index];
      }
      if (reset_mask == 0) {
        continue;
      }
      if (!loom_amdgpu_wait_frontier_budget_add_assign(
              1, &out_input->local_reset_event_count) ||
          !loom_amdgpu_wait_frontier_budget_add_assign(
              (uint64_t)iree_math_count_ones_u32(reset_mask),
              &out_input->local_reset_counter_bit_count)) {
        return false;
      }
    }
  }

  if (!iree_checked_mul_u64(
          out_input->producer_completion_full_path_count,
          out_input->precise_access_count,
          &out_input->producer_complete_precise_access_visit_count) ||
      !iree_checked_mul_u64(
          out_input->producer_completion_full_path_count,
          out_input->storage_lease_count,
          &out_input->producer_complete_storage_lease_visit_count)) {
    return false;
  }
  return true;
}

static iree_status_t loom_amdgpu_wait_frontier_build_precise_memory_metadata(
    loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const uint32_t* planned_block_drain_counter_masks,
    iree_host_size_t dependency_count,
    const loom_amdgpu_wait_frontier_precise_runtime_bounds_t* runtime_bounds,
    iree_arena_allocator_t* arena) {
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  frontier->memory.coarse_nodes = frontier->nodes;
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  if (schedule->memory_accesses == NULL || schedule->effect_use_count == 0 ||
      schedule->effect_uses == NULL || schedule->node_count == 0 ||
      schedule->nodes == NULL || schedule->blocks == NULL ||
      completion_nodes == NULL || graph->malformed || graph->blocks == NULL ||
      graph->block_count != schedule->block_count) {
    return iree_ok_status();
  }
  uint64_t effect_use_count = 0;
  uint64_t block_count = 0;
  uint64_t cfg_edge_count = 0;
  uint64_t scheduled_node_count = 0;
  uint64_t node_count = 0;
  uint64_t dependency_count_u64 = 0;
  uint64_t pre_admission_visits = 0;
  if (runtime_bounds == NULL || !runtime_bounds->arithmetic_valid ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->effect_use_count, &effect_use_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(schedule->block_count,
                                                       &block_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(graph->edge_count,
                                                       &cfg_edge_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(
          schedule->scheduled_node_count, &scheduled_node_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(schedule->node_count,
                                                       &node_count) ||
      !loom_amdgpu_wait_frontier_budget_u64_from_size(dependency_count,
                                                       &dependency_count_u64) ||
      !loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
          effect_use_count, block_count, cfg_edge_count, scheduled_node_count,
          node_count, dependency_count_u64, &pre_admission_visits) ||
      pre_admission_visits >
          LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRE_ADMISSION_VISITS) {
    return iree_ok_status();
  }
  iree_host_size_t precise_access_count = 0;
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    if (effect->node_index >= schedule->node_count ||
        effect->block_index >= schedule->block_count ||
        schedule->nodes[effect->node_index].block_index !=
            effect->block_index ||
        (effect->counter_id != LOOM_AMDGPU_WAIT_COUNTER_NONE &&
         !loom_amdgpu_wait_counter_id_is_valid(effect->counter_id))) {
      return iree_ok_status();
    }
    if (loom_amdgpu_wait_frontier_precise_effect_summary(schedule, effect) !=
            NULL &&
        (!iree_host_size_checked_add(precise_access_count, 1,
                                     &precise_access_count) ||
         precise_access_count >
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ACCESS_COUNT)) {
      return iree_ok_status();
    }
  }
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    const loom_cfg_block_info_t* block = &graph->blocks[i];
    if (block->reachable && (block->preorder == UINT16_MAX ||
                             block->preorder_end <= block->preorder ||
                             block->preorder_end > graph->block_count)) {
      return iree_ok_status();
    }
  }
  if (precise_access_count == 0 || precise_access_count > UINT32_MAX ||
      schedule->node_count >= UINT32_MAX) {
    return iree_ok_status();
  }

  iree_host_size_t precise_bit_count = 0;
  iree_host_size_t rounded_bit_count = 0;
  iree_host_size_t index_count = 0;
  if (!iree_host_size_checked_mul(precise_access_count,
                                  LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT,
                                  &precise_bit_count) ||
      !iree_host_size_checked_add(precise_bit_count,
                                  LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD - 1,
                                  &rounded_bit_count) ||
      !iree_host_size_checked_add(schedule->node_count, 1, &index_count)) {
    return iree_ok_status();
  }
  const iree_host_size_t precise_word_count =
      rounded_bit_count / LOOM_AMDGPU_WAIT_PRECISE_BITS_PER_WORD;
  iree_host_size_t state_word_count = 0;
  iree_host_size_t total_state_word_count = 0;
  iree_host_size_t total_allocation_byte_count = 0;
  if (!iree_host_size_checked_mul(schedule->block_count, precise_word_count,
                                  &state_word_count) ||
      !iree_host_size_checked_mul(state_word_count, 2,
                                  &total_state_word_count) ||
      !iree_host_size_checked_add(total_state_word_count, precise_word_count,
                                  &total_state_word_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          total_state_word_count, sizeof(uint64_t),
          &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          schedule->node_count, sizeof(loom_amdgpu_wait_frontier_node_t),
          &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          precise_access_count, sizeof(loom_amdgpu_wait_memory_access_t),
          &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          index_count, sizeof(uint32_t), &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          schedule->node_count, sizeof(uint32_t),
          &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          graph->block_count, sizeof(uint32_t), &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          graph->block_count, sizeof(uint16_t), &total_allocation_byte_count) ||
      !loom_amdgpu_wait_frontier_add_allocation_bytes(
          graph->edge_count, sizeof(uint8_t), &total_allocation_byte_count)) {
    return iree_ok_status();
  }
  if ((uint64_t)total_allocation_byte_count >
      LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALLOCATION_BYTES) {
    return iree_ok_status();
  }

  uint32_t* block_ranks = NULL;
  uint16_t* preorder_stack = NULL;
  uint8_t* edge_seen_flags = NULL;
  loom_amdgpu_wait_frontier_budget_node_t* budget_nodes = NULL;
  const iree_arena_checkpoint_t validation_checkpoint =
      iree_arena_checkpoint_save(arena);
  iree_status_t status = iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*block_ranks), (void**)&block_ranks);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, graph->block_count,
                                       sizeof(*preorder_stack),
                                       (void**)&preorder_stack);
  }
  if (graph->edge_count != 0) {
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(arena, graph->edge_count,
                                         sizeof(*edge_seen_flags),
                                         (void**)&edge_seen_flags);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, schedule->node_count,
                                       sizeof(*budget_nodes),
                                       (void**)&budget_nodes);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_checkpoint_restore(&validation_checkpoint);
    return status;
  }
  memset(budget_nodes, 0, schedule->node_count * sizeof(*budget_nodes));
  const bool graph_is_valid =
      loom_amdgpu_wait_frontier_validate_acyclic_edge_partition(
          graph, block_ranks, preorder_stack, edge_seen_flags);
  loom_amdgpu_wait_frontier_precise_budget_input_t budget_input;
  const bool budget_is_admitted =
      graph_is_valid &&
      loom_amdgpu_wait_frontier_collect_precise_budget(
          frontier, completion_nodes, planned_block_drain_counter_masks,
          runtime_bounds, budget_nodes, dependency_count,
          precise_access_count, precise_word_count, total_allocation_byte_count,
          &budget_input) &&
      loom_amdgpu_wait_frontier_precise_budget_is_admitted(&budget_input);
  iree_arena_checkpoint_restore(&validation_checkpoint);
  if (!budget_is_admitted) {
    return iree_ok_status();
  }

  loom_amdgpu_wait_frontier_node_t* coarse_nodes = NULL;
  loom_amdgpu_wait_memory_access_t* precise_accesses = NULL;
  uint32_t* precise_access_indices_by_node = NULL;
  uint32_t* next_precise_access_indices = NULL;
  uint64_t* precise_active_words = NULL;
  uint64_t* precise_static_outgoing_words = NULL;
  uint64_t* precise_resolved_outgoing_words = NULL;
  const iree_arena_checkpoint_t precise_checkpoint =
      iree_arena_checkpoint_save(arena);
  status = iree_arena_allocate_array(arena, schedule->node_count,
                                     sizeof(*coarse_nodes),
                                     (void**)&coarse_nodes);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, precise_access_count,
                                       sizeof(*precise_accesses),
                                       (void**)&precise_accesses);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        arena, index_count, sizeof(*precise_access_indices_by_node),
        (void**)&precise_access_indices_by_node);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        arena, schedule->node_count, sizeof(*next_precise_access_indices),
        (void**)&next_precise_access_indices);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        arena, precise_word_count, sizeof(*precise_active_words),
        (void**)&precise_active_words);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        arena, state_word_count, sizeof(*precise_static_outgoing_words),
        (void**)&precise_static_outgoing_words);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        arena, state_word_count, sizeof(*precise_resolved_outgoing_words),
        (void**)&precise_resolved_outgoing_words);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_checkpoint_restore(&precise_checkpoint);
    return status;
  }

  memset(precise_access_indices_by_node, 0,
         index_count * sizeof(*precise_access_indices_by_node));
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    if (loom_amdgpu_wait_frontier_precise_effect_summary(schedule, effect) !=
        NULL) {
      ++precise_access_indices_by_node[effect->node_index + 1];
    }
  }
  for (iree_host_size_t i = 1; i < index_count; ++i) {
    precise_access_indices_by_node[i] += precise_access_indices_by_node[i - 1];
  }
  memcpy(next_precise_access_indices, precise_access_indices_by_node,
         schedule->node_count * sizeof(*next_precise_access_indices));

  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    coarse_nodes[i] = frontier->nodes[i];
    coarse_nodes[i].read_counter_mask = 0;
    coarse_nodes[i].write_counter_mask = 0;
    coarse_nodes[i].read_space_flags = 0;
    coarse_nodes[i].write_space_flags = 0;
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->descriptor == NULL ||
        (node->op != NULL &&
         iree_any_bit_set(node->op->instance_flags,
                          LOOM_MEMORY_ACCESS_FLAG_VOLATILE))) {
      coarse_nodes[i].read_counter_mask = frontier->nodes[i].read_counter_mask;
      coarse_nodes[i].write_counter_mask =
          frontier->nodes[i].write_counter_mask;
      coarse_nodes[i].read_space_flags = frontier->nodes[i].read_space_flags;
      coarse_nodes[i].write_space_flags = frontier->nodes[i].write_space_flags;
    }
  }

  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    const loom_low_memory_access_summary_t* summary =
        loom_amdgpu_wait_frontier_precise_effect_summary(schedule, effect);
    if (summary == NULL) {
      loom_amdgpu_wait_frontier_add_coarse_effect(
          frontier, effect, &coarse_nodes[effect->node_index]);
      continue;
    }
    const uint32_t access_index =
        next_precise_access_indices[effect->node_index]++;
    IREE_ASSERT_LT(access_index, precise_access_count);
    precise_accesses[access_index] = (loom_amdgpu_wait_memory_access_t){
        .summary = summary,
        .producer_counter_mask = (loom_amdgpu_wait_counter_mask_t)
            loom_amdgpu_wait_frontier_effect_counter_mask(frontier, effect),
        .space_flags =
            loom_amdgpu_wait_memory_space_flag(summary->memory_space),
        .access_flags = loom_amdgpu_wait_frontier_effect_access_flags(effect),
    };
  }

  // Any memory space not accounted for by descriptor effects stays coarse. Do
  // not use aggregate node counter masks for this check: those may also include
  // counter-backed external effects that do not participate in memory aliasing.
  for (iree_host_size_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    const loom_amdgpu_wait_frontier_node_t* node = &frontier->nodes[node_index];
    loom_amdgpu_wait_frontier_node_t* coarse_node = &coarse_nodes[node_index];
    loom_amdgpu_wait_memory_space_flags_t represented_read_space_flags =
        coarse_node->read_space_flags;
    loom_amdgpu_wait_memory_space_flags_t represented_write_space_flags =
        coarse_node->write_space_flags;
    const uint32_t first = precise_access_indices_by_node[node_index];
    const uint32_t end = precise_access_indices_by_node[node_index + 1];
    for (uint32_t access_index = first; access_index < end; ++access_index) {
      const loom_amdgpu_wait_memory_access_t* access =
          &precise_accesses[access_index];
      if (iree_any_bit_set(access->access_flags,
                           LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ)) {
        represented_read_space_flags |= access->space_flags;
      }
      if (iree_any_bit_set(access->access_flags,
                           LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
        represented_write_space_flags |= access->space_flags;
      }
    }
    if (!iree_all_bits_set(represented_read_space_flags,
                           node->read_space_flags)) {
      coarse_node->read_counter_mask |= node->read_counter_mask;
      coarse_node->read_space_flags |= node->read_space_flags;
    }
    if (!iree_all_bits_set(represented_write_space_flags,
                           node->write_space_flags)) {
      coarse_node->write_counter_mask |= node->write_counter_mask;
      coarse_node->write_space_flags |= node->write_space_flags;
    }
  }

  memset(precise_active_words, 0,
         precise_word_count * sizeof(*precise_active_words));
  memset(precise_static_outgoing_words, 0,
         state_word_count * sizeof(*precise_static_outgoing_words));
  memset(precise_resolved_outgoing_words, 0,
         state_word_count * sizeof(*precise_resolved_outgoing_words));
  frontier->memory.coarse_nodes = coarse_nodes;
  frontier->memory.precise_accesses = precise_accesses;
  frontier->memory.precise_access_indices_by_node =
      precise_access_indices_by_node;
  frontier->memory.precise_access_count = precise_access_count;
  frontier->memory.precise_word_count = precise_word_count;
  frontier->memory.precise_static_outgoing_words =
      precise_static_outgoing_words;
  frontier->memory.precise_resolved_outgoing_words =
      precise_resolved_outgoing_words;
  frontier->memory.precise_active_words = precise_active_words;
  return iree_ok_status();
}

static void loom_amdgpu_wait_precise_state_add_node(
    const loom_amdgpu_wait_frontier_t* frontier, uint64_t* words,
    uint32_t node_index, uint32_t excluded_counter_mask) {
  if (words == NULL) {
    return;
  }
  const uint32_t first =
      frontier->memory.precise_access_indices_by_node[node_index];
  const uint32_t end =
      frontier->memory.precise_access_indices_by_node[node_index + 1];
  for (uint32_t access_index = first; access_index < end; ++access_index) {
    loom_amdgpu_wait_precise_state_set(
        words, access_index,
        frontier->memory.precise_accesses[access_index].producer_counter_mask &
            ~excluded_counter_mask);
  }
}

static uint32_t loom_amdgpu_wait_memory_state_query(
    const loom_amdgpu_wait_memory_state_t* state,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags) {
  IREE_ASSERT_EQ(
      (uint32_t)space_flags & ~LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK, 0u);
  space_flags >>= LOOM_LOW_MEMORY_SPACE_GENERIC;
  uint32_t counter_mask = 0;
  while (space_flags != 0) {
    const uint32_t space_index =
        (uint32_t)iree_math_count_trailing_zeros_u32(space_flags);
    IREE_ASSERT_LT(space_index, LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT);
    const uint16_t access_counter_masks =
        state->access_counter_masks[space_index];
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

static uint32_t loom_amdgpu_wait_precise_state_query(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* words,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags) {
  if (words == NULL || space_flags == 0 || access_flags == 0) {
    return 0;
  }
  const loom_amdgpu_wait_memory_space_flags_t generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  const bool query_is_generic = iree_any_bit_set(space_flags, generic_space);
  uint32_t counter_mask = 0;
  for (iree_host_size_t access_index = 0;
       access_index < frontier->memory.precise_access_count; ++access_index) {
    const loom_amdgpu_wait_memory_access_t* access =
        &frontier->memory.precise_accesses[access_index];
    if (!iree_any_bit_set(access->access_flags, access_flags) ||
        (!query_is_generic &&
         !iree_any_bit_set(access->space_flags, space_flags))) {
      continue;
    }
    uint32_t pending_counter_mask = access->producer_counter_mask;
    while (pending_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(pending_counter_mask);
      if (loom_amdgpu_wait_precise_state_test(words, access_index,
                                              counter_slot)) {
        counter_mask |= loom_amdgpu_wait_counter_mask_from_slot(counter_slot);
      }
      pending_counter_mask &= pending_counter_mask - 1;
    }
  }
  return counter_mask;
}

static uint32_t loom_amdgpu_wait_precise_state_dependency_query(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* words,
    const loom_amdgpu_wait_memory_access_t* consumer_access) {
  if (words == NULL) {
    return 0;
  }
  loom_amdgpu_wait_memory_access_flags_t producer_access_flags = 0;
  if (iree_any_bit_set(consumer_access->access_flags,
                       LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ)) {
    producer_access_flags |= LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE;
  }
  if (iree_any_bit_set(consumer_access->access_flags,
                       LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
    producer_access_flags |= LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ;
  }
  uint32_t counter_mask = 0;
  for (iree_host_size_t producer_index = 0;
       producer_index < frontier->memory.precise_access_count;
       ++producer_index) {
    const loom_amdgpu_wait_memory_access_t* producer_access =
        &frontier->memory.precise_accesses[producer_index];
    // Precise bits are never propagated across DFS backedges. Producer and
    // consumer therefore belong to one captured invocation with no CFG
    // re-entry, which is the caller-side proof required by this comparison.
    if (!iree_any_bit_set(producer_access->access_flags,
                          producer_access_flags) ||
        !loom_low_memory_access_summaries_may_alias(
            producer_access->summary, consumer_access->summary,
            LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION)) {
      continue;
    }
    uint32_t pending_counter_mask = producer_access->producer_counter_mask;
    while (pending_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(pending_counter_mask);
      if (loom_amdgpu_wait_precise_state_test(words, producer_index,
                                              counter_slot)) {
        counter_mask |= loom_amdgpu_wait_counter_mask_from_slot(counter_slot);
      }
      pending_counter_mask &= pending_counter_mask - 1;
    }
  }
  return counter_mask;
}

static void loom_amdgpu_wait_precise_state_collapse(
    const loom_amdgpu_wait_frontier_t* frontier, const uint64_t* words,
    uint32_t drain_counter_mask, loom_amdgpu_wait_memory_state_t* state) {
  if (words == NULL) {
    return;
  }
  for (iree_host_size_t access_index = 0;
       access_index < frontier->memory.precise_access_count; ++access_index) {
    const loom_amdgpu_wait_memory_access_t* access =
        &frontier->memory.precise_accesses[access_index];
    uint32_t pending_counter_mask =
        access->producer_counter_mask & ~drain_counter_mask;
    uint32_t retained_counter_mask = 0;
    while (pending_counter_mask != 0) {
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(pending_counter_mask);
      if (loom_amdgpu_wait_precise_state_test(words, access_index,
                                              counter_slot)) {
        retained_counter_mask |=
            loom_amdgpu_wait_counter_mask_from_slot(counter_slot);
      }
      pending_counter_mask &= pending_counter_mask - 1;
    }
    if (iree_any_bit_set(access->access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
      retained_counter_mask <<= LOOM_AMDGPU_WAIT_MEMORY_WRITE_COUNTER_SHIFT;
    }
    loom_amdgpu_wait_memory_state_add_access(state, access->space_flags,
                                             (uint16_t)retained_counter_mask);
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
    if (record->packet_index != packet_index) {
      break;
    }
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
  if (producer_group_flags == 0) {
    return;
  }
  const loom_amdgpu_wait_xcnt_group_flags_t other_group_flags =
      *xcnt_group_flags &
      (loom_amdgpu_wait_xcnt_group_flags_t)~producer_group_flags;
  if (other_group_flags != 0 && storage_lease_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain_xcnt_groups(
        frontier, storage_lease_words, other_group_flags, /*selection=*/NULL);
  }
  *xcnt_group_flags = producer_group_flags;
}

static void loom_amdgpu_wait_frontier_build_local_states(
    loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
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
    uint64_t* precise_memory_words =
        frontier->memory.precise_static_outgoing_words == NULL
            ? NULL
            : loom_amdgpu_wait_frontier_precise_block_words(
                  frontier, frontier->memory.precise_static_outgoing_words,
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
    uint32_t incoming_completion_counter_mask = 0;
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      const loom_amdgpu_wait_frontier_node_t* node =
          &frontier->nodes[node_index];
      const loom_amdgpu_wait_frontier_node_t* coarse_node =
          &frontier->memory.coarse_nodes[node_index];
      const uint32_t completed_counter_mask =
          completion_nodes[node_index].completed_before_block_exit_counter_mask;
      uint32_t drain_counter_mask =
          completion_nodes[node_index].reset_counter_mask;
      if (planned_block_drain_counter_masks != NULL &&
          schedule->nodes[node_index].op == block->block->last_op) {
        drain_counter_mask |= planned_block_drain_counter_masks[block_index];
      }
      // Completing any local producer also completes incoming work in its
      // counter class, while later local producers may remain pending.
      incoming_completion_counter_mask |=
          drain_counter_mask | completed_counter_mask;
      if (memory_state != NULL) {
        loom_amdgpu_wait_memory_state_drain(memory_state, drain_counter_mask);
        loom_amdgpu_wait_memory_state_add_node(
            memory_state, coarse_node,
            coarse_node->read_counter_mask & ~completed_counter_mask,
            coarse_node->write_counter_mask & ~completed_counter_mask);
      }
      if (precise_memory_words != NULL) {
        loom_amdgpu_wait_precise_state_drain(frontier, precise_memory_words,
                                             drain_counter_mask);
        loom_amdgpu_wait_precise_state_add_node(
            frontier, precise_memory_words, node_index, completed_counter_mask);
      }
      if (storage_lease_words != NULL) {
        loom_amdgpu_wait_storage_lease_state_drain(
            frontier, storage_lease_words, drain_counter_mask,
            /*selection=*/NULL);
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
            frontier, storage_lease_words, packet_index, completed_counter_mask,
            &next_storage_lease_index);
      }
    }
    frontier->incoming_completion_counter_masks[block_index] =
        incoming_completion_counter_mask;
  }
}

// Local completion and full incoming drains already filter these counters.
// Only dependencies carrying otherwise-pending results need exact lease bits.
static uint32_t loom_amdgpu_wait_frontier_incoming_dependency_counter_mask(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const loom_amdgpu_wait_dependency_t* dependency) {
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  const uint16_t consumer_block =
      schedule->nodes[dependency->consumer_node].block_index;
  if (schedule->nodes[dependency->producer_node].block_index ==
      consumer_block) {
    return 0;
  }
  return dependency->counter_mask &
         ~completion_nodes[dependency->producer_node]
              .completed_before_block_exit_counter_mask &
         ~frontier->incoming_completion_counter_masks[consumer_block];
}

// A cross-block use completes its producer's incoming result instance. Keep
// that fact in the static transfer so a backedge cannot resurrect its lease.
// Dependency construction follows pending values only through coalesced edges;
// materialized copies complete their source at the copy. A use of an older,
// copied value therefore cannot retire a newly issued instance of its producer.
//
// This transfer filters incoming bits, never locally generated instances or
// unrelated work in the same counter. Reusing an already-completed SSA value
// need not emit a wait and therefore proves no counter-wide reset.
static iree_status_t loom_amdgpu_wait_frontier_build_result_completions(
    loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const loom_amdgpu_wait_dependency_t* dependencies,
    iree_host_size_t dependency_count, iree_arena_allocator_t* arena) {
  if (frontier->storage_leases.word_count == 0) {
    return iree_ok_status();
  }
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  iree_host_size_t first_dependency = 0;
  while (first_dependency < dependency_count &&
         loom_amdgpu_wait_frontier_incoming_dependency_counter_mask(
             frontier, completion_nodes, &dependencies[first_dependency]) ==
             0) {
    ++first_dependency;
  }
  if (first_dependency == dependency_count) {
    return iree_ok_status();
  }
  const iree_host_size_t word_count =
      schedule->block_count * frontier->storage_leases.word_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(uint64_t),
      (void**)&frontier->storage_leases.completed_incoming_words));
  memset(frontier->storage_leases.completed_incoming_words, 0,
         word_count * sizeof(uint64_t));
  // Lease records are contiguous by producer in scheduled packet order. Index
  // each range once instead of searching the record table for every use.
  uint32_t* first_leases = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->node_count,
                                                 sizeof(*first_leases),
                                                 (void**)&first_leases));
  memset(first_leases, 0xFF, schedule->node_count * sizeof(*first_leases));
  const loom_low_storage_lease_record_t* records =
      frontier->allocation->storage_leases.records;
  for (iree_host_size_t i = frontier->storage_leases.lease_count; i > 0; --i) {
    first_leases[records[i - 1].node_index] = (uint32_t)(i - 1);
  }
  frontier->storage_leases.first_indices_by_node = first_leases;
  for (iree_host_size_t i = first_dependency; i < dependency_count; ++i) {
    const loom_amdgpu_wait_dependency_t* dependency = &dependencies[i];
    const uint32_t counter_mask =
        loom_amdgpu_wait_frontier_incoming_dependency_counter_mask(
            frontier, completion_nodes, dependency);
    if (counter_mask == 0) {
      continue;
    }
    const uint16_t consumer_block =
        schedule->nodes[dependency->consumer_node].block_index;
    uint64_t* completed_words =
        loom_amdgpu_wait_frontier_storage_lease_block_words(
            frontier, frontier->storage_leases.completed_incoming_words,
            consumer_block);
    for (iree_host_size_t lease_index = first_leases[dependency->producer_node];
         lease_index < frontier->storage_leases.lease_count &&
         records[lease_index].node_index == dependency->producer_node;
         ++lease_index) {
      const loom_low_storage_lease_record_t* record = &records[lease_index];
      if (record->kind == LOOM_LOW_STORAGE_LEASE_RESULT_WRITE &&
          record->release_scope ==
              LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS &&
          loom_amdgpu_wait_counter_id_is_valid(record->release_class_id) &&
          iree_any_bit_set(counter_mask, loom_amdgpu_wait_counter_mask(
                                             record->release_class_id))) {
        loom_amdgpu_wait_storage_lease_state_set(completed_words, lease_index);
      }
    }
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_frontier_block_state_union_changed(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t target_block,
    uint16_t source_block) {
  bool changed = false;
  const uint32_t drain_counter_mask =
      frontier->incoming_completion_counter_masks[target_block];
  const bool is_backedge =
      frontier->memory.precise_static_outgoing_words != NULL &&
      loom_amdgpu_wait_frontier_edge_is_backedge(&frontier->schedule->cfg_graph,
                                                 source_block, target_block);
  if (frontier->memory.static_outgoing_states != NULL) {
    loom_amdgpu_wait_memory_state_t incoming_state =
        frontier->memory.static_outgoing_states[source_block];
    loom_amdgpu_wait_memory_state_drain(&incoming_state, drain_counter_mask);
    if (is_backedge) {
      loom_amdgpu_wait_precise_state_collapse(
          frontier,
          loom_amdgpu_wait_frontier_const_precise_block_words(
              frontier, frontier->memory.precise_static_outgoing_words,
              source_block),
          drain_counter_mask, &incoming_state);
    }
    changed |= loom_amdgpu_wait_memory_state_union_changed(
        &frontier->memory.static_outgoing_states[target_block],
        &incoming_state);
  }
  if (frontier->memory.precise_static_outgoing_words != NULL && !is_backedge) {
    changed |= loom_amdgpu_wait_precise_state_union_after_drain_changed(
        frontier,
        loom_amdgpu_wait_frontier_precise_block_words(
            frontier, frontier->memory.precise_static_outgoing_words,
            target_block),
        loom_amdgpu_wait_frontier_const_precise_block_words(
            frontier, frontier->memory.precise_static_outgoing_words,
            source_block),
        drain_counter_mask);
  }
  if (frontier->storage_leases.static_outgoing_words != NULL) {
    const uint64_t* completed_words =
        frontier->storage_leases.completed_incoming_words == NULL
            ? NULL
            : loom_amdgpu_wait_frontier_const_storage_lease_block_words(
                  frontier, frontier->storage_leases.completed_incoming_words,
                  target_block);
    changed |= loom_amdgpu_wait_storage_lease_state_union_after_drain_changed(
        frontier,
        loom_amdgpu_wait_frontier_storage_lease_block_words(
            frontier, frontier->storage_leases.static_outgoing_words,
            target_block),
        loom_amdgpu_wait_frontier_const_storage_lease_block_words(
            frontier, frontier->storage_leases.static_outgoing_words,
            source_block),
        completed_words, drain_counter_mask);
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
  if (frontier->memory.precise_static_outgoing_words != NULL &&
      !loom_amdgpu_wait_precise_state_is_empty(
          loom_amdgpu_wait_frontier_const_precise_block_words(
              frontier, frontier->memory.precise_static_outgoing_words,
              block_index),
          frontier->memory.precise_word_count)) {
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
  if (++*tail == block_count) {
    *tail = 0;
  }
  ++*count;
  block_flags[block_index] |= LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_QUEUED;
}

static void loom_amdgpu_wait_frontier_propagate_static_states(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t* worklist) {
  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  const uint32_t block_count = (uint32_t)frontier->schedule->block_count;
  if (block_count <= 1 || graph->blocks == NULL) {
    return;
  }

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
    if (++head == block_count) {
      head = 0;
    }
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
    const loom_amdgpu_wait_completion_node_t* completion_nodes,
    const loom_amdgpu_wait_dependency_t* dependencies,
    iree_host_size_t dependency_count,
    const uint32_t* planned_block_drain_counter_masks,
    const loom_amdgpu_wait_frontier_precise_runtime_bounds_t* runtime_bounds,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_frontier_t* out_frontier) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontier);
  IREE_ASSERT(schedule->node_count == 0 || nodes != NULL);
  *out_frontier = (loom_amdgpu_wait_frontier_t){
      .schedule = schedule,
      .allocation = allocation,
      .nodes = nodes,
      .memory = {.coarse_nodes = nodes},
      .storage_leases =
          {
              .lease_count = allocation == NULL
                                 ? 0
                                 : allocation->storage_lease_instance_count,
          },
      .active_block_index = UINT16_MAX,
  };

  bool has_memory_producer = false;
  bool has_xcnt_producer = false;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    has_memory_producer |=
        nodes[i].read_counter_mask != 0 || nodes[i].write_counter_mask != 0;
    has_xcnt_producer |= nodes[i].xcnt_group_flags != 0;
  }

  const bool has_cross_block_state =
      schedule->block_count > 1 && schedule->cfg_graph.blocks != NULL;
  if (has_cross_block_state && out_frontier->storage_leases.lease_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_selection_initialize(
        allocation->storage_lease_unit_index, arena,
        &out_frontier->storage_leases.active_selection));
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
       out_frontier->storage_leases.active_words == NULL &&
       !has_xcnt_producer)) {
    return iree_ok_status();
  }

  if (has_memory_producer) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_frontier_build_precise_memory_metadata(
            out_frontier, completion_nodes, planned_block_drain_counter_masks,
            dependency_count, runtime_bounds, arena));
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
      sizeof(*out_frontier->incoming_completion_counter_masks),
      (void**)&out_frontier->incoming_completion_counter_masks));
  memset(out_frontier->incoming_completion_counter_masks, 0,
         schedule->block_count *
             sizeof(*out_frontier->incoming_completion_counter_masks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*out_frontier->block_flags),
      (void**)&out_frontier->block_flags));
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*worklist), (void**)&worklist));
  memset(out_frontier->block_flags, 0,
         schedule->block_count * sizeof(*out_frontier->block_flags));

  loom_amdgpu_wait_frontier_build_local_states(
      out_frontier, completion_nodes, planned_block_drain_counter_masks);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_frontier_build_result_completions(
      out_frontier, completion_nodes, dependencies, dependency_count, arena));
  loom_amdgpu_wait_frontier_propagate_static_states(out_frontier, worklist);
  return iree_ok_status();
}

void loom_amdgpu_wait_frontier_begin_block(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t block_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(block_index < frontier->schedule->block_count);
  IREE_ASSERT(frontier->active_block_index == UINT16_MAX);
  frontier->memory.active_state = (loom_amdgpu_wait_memory_state_t){0};
  if (frontier->memory.precise_active_words != NULL) {
    memset(frontier->memory.precise_active_words, 0,
           frontier->memory.precise_word_count *
               sizeof(*frontier->memory.precise_active_words));
  }
  if (frontier->storage_leases.active_words != NULL) {
    memset(frontier->storage_leases.active_words, 0,
           frontier->storage_leases.word_count *
               sizeof(*frontier->storage_leases.active_words));
  }
  frontier->xcnt.active_flags = 0;
  frontier->xcnt.drained_group_flags = 0;
  frontier->incoming_drain_counter_mask = 0;
  frontier->active_block_index = block_index;

  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  if (frontier->block_flags == NULL || graph->blocks == NULL) {
    return;
  }
  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
      continue;
    }
    const bool predecessor_resolved =
        iree_any_bit_set(frontier->block_flags[predecessor_index],
                         LOOM_AMDGPU_WAIT_FRONTIER_BLOCK_FLAG_RESOLVED);
    const bool is_backedge = loom_amdgpu_wait_frontier_edge_is_backedge(
        graph, predecessor_index, block_index);
    const bool use_resolved_state =
        predecessor_resolved &&
        (!is_backedge || frontier->memory.precise_active_words == NULL);
    if (frontier->memory.static_outgoing_states != NULL) {
      const loom_amdgpu_wait_memory_state_t* predecessor_state =
          use_resolved_state
              ? &frontier->memory.resolved_outgoing_states[predecessor_index]
              : &frontier->memory.static_outgoing_states[predecessor_index];
      loom_amdgpu_wait_memory_state_union_changed(
          &frontier->memory.active_state, predecessor_state);
    }
    if (frontier->memory.precise_active_words != NULL) {
      const uint64_t* predecessor_words =
          use_resolved_state
              ? loom_amdgpu_wait_frontier_const_precise_block_words(
                    frontier, frontier->memory.precise_resolved_outgoing_words,
                    predecessor_index)
              : loom_amdgpu_wait_frontier_const_precise_block_words(
                    frontier, frontier->memory.precise_static_outgoing_words,
                    predecessor_index);
      if (is_backedge) {
        loom_amdgpu_wait_precise_state_collapse(frontier, predecessor_words,
                                                /*drain_counter_mask=*/0,
                                                &frontier->memory.active_state);
      } else {
        loom_amdgpu_wait_precise_state_union_changed(
            frontier->memory.precise_active_words, predecessor_words,
            frontier->memory.precise_word_count);
      }
    }
    if (frontier->storage_leases.static_outgoing_words != NULL) {
      const uint64_t* predecessor_words =
          use_resolved_state
              ? loom_amdgpu_wait_frontier_const_storage_lease_block_words(
                    frontier, frontier->storage_leases.resolved_outgoing_words,
                    predecessor_index)
              : loom_amdgpu_wait_frontier_const_storage_lease_block_words(
                    frontier, frontier->storage_leases.static_outgoing_words,
                    predecessor_index);
      loom_amdgpu_wait_storage_lease_state_union_changed(
          frontier->storage_leases.active_words, predecessor_words,
          frontier->storage_leases.word_count,
          &frontier->storage_leases.active_selection);
    }
    if (frontier->xcnt.static_outgoing_flags != NULL) {
      frontier->xcnt.active_flags |=
          use_resolved_state
              ? frontier->xcnt.resolved_outgoing_flags[predecessor_index]
              : frontier->xcnt.static_outgoing_flags[predecessor_index];
    }
  }
}

uint32_t loom_amdgpu_wait_frontier_memory_query(
    const loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  if (space_flags == 0 || access_flags == 0) {
    return 0;
  }
  IREE_ASSERT_EQ(
      (uint32_t)space_flags & ~LOOM_AMDGPU_WAIT_MEMORY_SPACE_FLAG_MASK, 0u);
  return loom_amdgpu_wait_memory_state_query(&frontier->memory.active_state,
                                             space_flags, access_flags) |
         loom_amdgpu_wait_precise_state_query(
             frontier, frontier->memory.precise_active_words, space_flags,
             access_flags);
}

uint32_t loom_amdgpu_wait_frontier_memory_dependency_mask(
    const loom_amdgpu_wait_frontier_t* frontier, uint32_t consumer_node) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_LT(consumer_node, frontier->schedule->node_count);
  const loom_amdgpu_wait_frontier_node_t* node =
      &frontier->nodes[consumer_node];
  if (frontier->memory.static_outgoing_states == NULL ||
      (node->read_space_flags == 0 && node->write_space_flags == 0)) {
    return 0;
  }
  const loom_amdgpu_wait_frontier_node_t* coarse_node =
      &frontier->memory.coarse_nodes[consumer_node];
  const uint32_t prior_writes =
      coarse_node->read_space_flags == 0
          ? 0
          : loom_amdgpu_wait_frontier_memory_query(
                frontier, coarse_node->read_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
  const uint32_t prior_reads =
      coarse_node->write_space_flags == 0
          ? 0
          : loom_amdgpu_wait_frontier_memory_query(
                frontier, coarse_node->write_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ);
  uint32_t counter_mask = prior_writes | prior_reads;
  if (frontier->memory.precise_access_indices_by_node == NULL) {
    return counter_mask;
  }
  const uint32_t first =
      frontier->memory.precise_access_indices_by_node[consumer_node];
  const uint32_t end =
      frontier->memory.precise_access_indices_by_node[consumer_node + 1];
  for (uint32_t access_index = first; access_index < end; ++access_index) {
    const loom_amdgpu_wait_memory_access_t* consumer_access =
        &frontier->memory.precise_accesses[access_index];
    if (iree_any_bit_set(consumer_access->access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ)) {
      counter_mask |= loom_amdgpu_wait_memory_state_query(
          &frontier->memory.active_state, consumer_access->space_flags,
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
    }
    if (iree_any_bit_set(consumer_access->access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
      counter_mask |= loom_amdgpu_wait_memory_state_query(
          &frontier->memory.active_state, consumer_access->space_flags,
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ);
    }
    counter_mask |= loom_amdgpu_wait_precise_state_dependency_query(
        frontier, frontier->memory.precise_active_words, consumer_access);
  }
  return counter_mask;
}

bool loom_amdgpu_wait_frontier_producer_is_complete(
    const loom_amdgpu_wait_frontier_t* frontier, uint32_t producer_node,
    uint32_t counter_mask) {
  const loom_amdgpu_wait_frontier_node_t* node =
      &frontier->nodes[producer_node];
  const uint32_t tracked_counter_mask =
      node->read_counter_mask | node->write_counter_mask;
  if (frontier->memory.static_outgoing_states == NULL ||
      !iree_all_bits_set(tracked_counter_mask, counter_mask)) {
    return false;
  }
  const loom_amdgpu_wait_frontier_node_t* coarse_node =
      &frontier->memory.coarse_nodes[producer_node];
  uint32_t memory_tracked_counter_mask = 0;
  if (coarse_node->read_space_flags != 0) {
    memory_tracked_counter_mask |= coarse_node->read_counter_mask;
  }
  if (coarse_node->write_space_flags != 0) {
    memory_tracked_counter_mask |= coarse_node->write_counter_mask;
  }
  if (frontier->memory.precise_access_indices_by_node != NULL) {
    const uint32_t first =
        frontier->memory.precise_access_indices_by_node[producer_node];
    const uint32_t end =
        frontier->memory.precise_access_indices_by_node[producer_node + 1];
    for (uint32_t access_index = first; access_index < end; ++access_index) {
      memory_tracked_counter_mask |=
          frontier->memory.precise_accesses[access_index].producer_counter_mask;
    }
  }
  const uint32_t pending_reads = loom_amdgpu_wait_frontier_memory_query(
      frontier, node->read_space_flags,
      LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ);
  const uint32_t pending_writes = loom_amdgpu_wait_frontier_memory_query(
      frontier, node->write_space_flags,
      LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
  // Memory evidence cannot retire a non-memory result that shares its counter.
  // Exact result leases must prove that part complete independently.
  const uint32_t pending_counter_mask =
      ((pending_reads | pending_writes) & counter_mask) |
      (counter_mask &
       (~memory_tracked_counter_mask | node->external_counter_mask));
  if (pending_counter_mask == 0) {
    return true;
  }
  if (frontier->storage_leases.first_indices_by_node == NULL) {
    return false;
  }
  // Incoming result bits retain exact producer completion across copies and
  // joins. A newer load in the same memory space must not obscure that fact.
  // Missing result records provide no proof, and every matching result lease
  // must be complete before its counter class can be considered complete.
  const loom_low_storage_lease_record_t* records =
      frontier->allocation->storage_leases.records;
  uint32_t completed_result_mask = 0;
  uint32_t pending_result_mask = 0;
  for (iree_host_size_t lease_index =
           frontier->storage_leases.first_indices_by_node[producer_node];
       lease_index < frontier->storage_leases.lease_count &&
       records[lease_index].node_index == producer_node;
       ++lease_index) {
    const loom_low_storage_lease_record_t* record = &records[lease_index];
    if (record->kind != LOOM_LOW_STORAGE_LEASE_RESULT_WRITE ||
        record->release_scope !=
            LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS ||
        !loom_amdgpu_wait_counter_id_is_valid(record->release_class_id)) {
      continue;
    }
    const uint32_t result_counter_mask =
        loom_amdgpu_wait_counter_mask(record->release_class_id);
    if (loom_amdgpu_wait_storage_lease_state_test(
            frontier->storage_leases.active_words, lease_index)) {
      pending_result_mask |= result_counter_mask;
    } else {
      completed_result_mask |= result_counter_mask;
    }
  }
  return iree_all_bits_set(completed_result_mask & ~pending_result_mask,
                           pending_counter_mask);
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
      frontier->storage_leases.active_words, lease_index,
      &frontier->storage_leases.active_selection);
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
      (loom_amdgpu_wait_xcnt_group_flags_t) ~(
          group_flags | frontier->xcnt.drained_group_flags);
  if (other_group_flags != 0 && frontier->storage_leases.active_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain_xcnt_groups(
        frontier, frontier->storage_leases.active_words, other_group_flags,
        &frontier->storage_leases.active_selection);
  }
  frontier->xcnt.drained_group_flags |= other_group_flags;
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
  if (iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_X)) {
    frontier->xcnt.active_flags = 0;
    frontier->xcnt.drained_group_flags = LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM |
                                         LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM;
  }
  // Incoming state only loses members while processing a block. Each counter
  // therefore clears its incoming bitmaps at most once, regardless of how many
  // waits complete subsequently issued local producers.
  counter_mask &= ~frontier->incoming_drain_counter_mask;
  frontier->incoming_drain_counter_mask |= counter_mask;
  loom_amdgpu_wait_memory_state_drain(&frontier->memory.active_state,
                                      counter_mask);
  loom_amdgpu_wait_precise_state_drain(
      frontier, frontier->memory.precise_active_words, counter_mask);
  if (frontier->storage_leases.active_words != NULL) {
    loom_amdgpu_wait_storage_lease_state_drain(
        frontier, frontier->storage_leases.active_words, counter_mask,
        &frontier->storage_leases.active_selection);
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
  uint64_t* outgoing_precise_memory_words =
      frontier->memory.precise_resolved_outgoing_words == NULL
          ? NULL
          : loom_amdgpu_wait_frontier_precise_block_words(
                frontier, frontier->memory.precise_resolved_outgoing_words,
                block_index);
  if (outgoing_precise_memory_words != NULL) {
    memcpy(outgoing_precise_memory_words, frontier->memory.precise_active_words,
           frontier->memory.precise_word_count *
               sizeof(*outgoing_precise_memory_words));
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
  if (outgoing_memory_state != NULL || outgoing_precise_memory_words != NULL ||
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
        const loom_amdgpu_wait_frontier_node_t* coarse_node =
            &frontier->memory.coarse_nodes[node_index];
        const uint32_t read_counter_mask =
            coarse_node->read_counter_mask &
            ~node->drained_after_production_counter_mask;
        const uint32_t write_counter_mask =
            coarse_node->write_counter_mask &
            ~node->drained_after_production_counter_mask;
        loom_amdgpu_wait_memory_state_add_node(outgoing_memory_state,
                                               coarse_node, read_counter_mask,
                                               write_counter_mask);
      }
      if (outgoing_precise_memory_words != NULL) {
        loom_amdgpu_wait_precise_state_add_node(
            frontier, outgoing_precise_memory_words, node_index,
            node->drained_after_production_counter_mask);
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
  if (frontier->memory.precise_active_words != NULL) {
    memset(frontier->memory.precise_active_words, 0,
           frontier->memory.precise_word_count *
               sizeof(*frontier->memory.precise_active_words));
  }
  if (frontier->storage_leases.active_words != NULL) {
    for (iree_host_size_t word_index = 0;
         word_index < frontier->storage_leases.word_count; ++word_index) {
      loom_amdgpu_wait_storage_lease_state_update_word(
          frontier->storage_leases.active_words, word_index, 0,
          &frontier->storage_leases.active_selection);
    }
  }
  frontier->xcnt.active_flags = 0;
  frontier->active_block_index = UINT16_MAX;
}
