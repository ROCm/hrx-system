// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/replay_clause.h"

#include <string.h>

#include "loom/target/arch/amdgpu/facts.h"

struct loom_amdgpu_replay_clause_t {
  // Successful physical allocation supplying indexed packet assignments.
  const loom_low_allocation_table_t* allocation;
  // Disjoint register-bank offsets indexed by descriptor register class.
  uint32_t* bank_offsets;
  // Read plane followed by write plane for the current memory clause.
  uint64_t* bits;
  // Number of words in each physical-register plane.
  iree_host_size_t word_count;
  // Memory-class traits requiring software replay protection on this target.
  loom_amdgpu_descriptor_traits_t enabled_classes;
  // Current clause's memory-class trait, or zero after a native separator.
  loom_amdgpu_descriptor_traits_t memory_class;
  // Last native memory node in the current clause.
  uint32_t last_node;
  // The clause contains at least one physical register definition.
  bool has_definitions;
  // A register appears in both the clause's read and write planes.
  bool overlap;
};

iree_status_t loom_amdgpu_replay_clause_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_descriptor_traits_t function_traits,
    iree_arena_allocator_t* arena, loom_amdgpu_replay_clause_t** out_state) {
  *out_state = NULL;
  const loom_amdgpu_target_facts_t* facts =
      (const loom_amdgpu_target_facts_t*)schedule->target.target_facts;
  const loom_amdgpu_processor_properties_t* processor =
      facts->properties.processor;
  // Selectable feature state alone cannot describe always-on replay: those
  // processors have no target-ID mode selector but still advertise XNACK in
  // their ELF policy. Explicit OFF is admitted only on selectable processors.
  if (facts->properties.amdhsa_features.xnack ==
          LOOM_AMDGPU_TARGET_FEATURE_OFF ||
      !iree_any_bit_set(processor->elf.feature_flags,
                        LOOM_AMDGPU_ELF_FEATURE_XNACK_MASK_V4)) {
    return iree_ok_status();
  }
  loom_amdgpu_descriptor_traits_t enabled_classes =
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY;
  if (iree_any_bit_set(processor->features.scheduling,
                       LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_REPLAY_CLAUSES)) {
    enabled_classes |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY;
  }
  enabled_classes &= function_traits;
  if (enabled_classes == 0) {
    return iree_ok_status();
  }

  uint32_t register_count = 0;
  for (iree_host_size_t i = 0; i < allocation->physical_extents.count; ++i) {
    register_count +=
        iree_max(allocation->physical_extents.ends_by_reg_class[i],
                 allocation->physical_extents.fixed_ends_by_reg_class[i]);
  }
  const iree_host_size_t word_count =
      iree_host_size_ceil_div(register_count, 64);
  const iree_host_size_t bank_bytes = iree_host_align(
      allocation->physical_extents.count * sizeof(uint32_t), sizeof(uint64_t));
  loom_amdgpu_replay_clause_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*state) + bank_bytes + 2 * word_count * sizeof(uint64_t),
      (void**)&state));
  *state = (loom_amdgpu_replay_clause_t){
      .allocation = allocation,
      .bank_offsets = (uint32_t*)(state + 1),
      .bits = (uint64_t*)((uint8_t*)(state + 1) + bank_bytes),
      .word_count = word_count,
      .enabled_classes = enabled_classes,
  };
  register_count = 0;
  for (iree_host_size_t i = 0; i < allocation->physical_extents.count; ++i) {
    state->bank_offsets[i] = register_count;
    register_count +=
        iree_max(allocation->physical_extents.ends_by_reg_class[i],
                 allocation->physical_extents.fixed_ends_by_reg_class[i]);
  }
  *out_state = state;
  return iree_ok_status();
}

void loom_amdgpu_replay_clause_break(loom_amdgpu_replay_clause_t* state) {
  state->memory_class = 0;
}

static void loom_amdgpu_replay_clause_begin(
    loom_amdgpu_replay_clause_t* state,
    loom_amdgpu_descriptor_traits_t memory_class) {
  state->memory_class = memory_class;
  state->has_definitions = false;
  state->overlap = false;
  memset(state->bits, 0, 2 * state->word_count * sizeof(uint64_t));
}

static void loom_amdgpu_replay_clause_record_assignment(
    loom_amdgpu_replay_clause_t* state,
    const loom_low_allocation_assignment_t* assignment, bool writes) {
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return;
  }
  uint32_t base = state->bank_offsets[assignment->descriptor_reg_class_id] +
                  assignment->location_base;
  uint32_t remaining = assignment->location_count;
  uint64_t* own = state->bits + (writes ? state->word_count : 0);
  const uint64_t* other = state->bits + (writes ? 0 : state->word_count);
  while (remaining != 0) {
    const unsigned offset = base % 64;
    const unsigned count = iree_min(remaining, 64 - offset);
    const uint64_t mask = (UINT64_MAX >> (64 - count)) << offset;
    state->overlap |= (other[base / 64] & mask) != 0;
    own[base / 64] |= mask;
    base += count;
    remaining -= count;
  }
  state->has_definitions |= writes;
}

static void loom_amdgpu_replay_clause_record_packet(
    loom_amdgpu_replay_clause_t* state, const loom_low_packet_view_t* packet) {
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    loom_amdgpu_replay_clause_record_assignment(
        state, loom_low_packet_operand_assignment(state->allocation, packet, i),
        false);
  }
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    loom_amdgpu_replay_clause_record_assignment(
        state, loom_low_packet_result_assignment(state->allocation, packet, i),
        true);
  }
}

// Keep the non-memory fast path in the sole native traversal caller while
// retaining an external definition for non-LTO builds.
extern IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t loom_amdgpu_replay_clause_apply(
    loom_amdgpu_replay_clause_t* state, const loom_low_packet_view_t* packet,
    loom_amdgpu_descriptor_traits_t traits, uint64_t instruction_count) {
  if (instruction_count == 0) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  const loom_amdgpu_descriptor_traits_t memory_class =
      traits & state->enabled_classes;
  if (memory_class == 0) {
    loom_amdgpu_replay_clause_break(state);
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  if (memory_class != state->memory_class) {
    loom_amdgpu_replay_clause_begin(state, memory_class);
  }
  const bool had_definitions = state->has_definitions;
  loom_amdgpu_replay_clause_record_packet(state, packet);
  uint32_t previous_node = LOOM_LOW_PACKET_INDEX_NONE;
  // Single instructions may reuse their own inputs where the descriptor
  // permits it. Replay of a longer clause must preserve every source. Loads
  // followed by stores also need separation even without register overlap.
  if (had_definitions &&
      (state->overlap ||
       iree_any_bit_set(packet->node->traits, LOOM_TRAIT_WRITES_MEMORY))) {
    previous_node = state->last_node;
    loom_amdgpu_replay_clause_begin(state, memory_class);
    loom_amdgpu_replay_clause_record_packet(state, packet);
  }
  state->last_node = packet->node_index;
  return previous_node;
}
