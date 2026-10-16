// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/short_wait.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/cfg_dataflow.h"

enum {
  LOOM_AMDGPU_SHORT_WAIT_STORE_VALU = 1u << 0,
  LOOM_AMDGPU_SHORT_WAIT_STORE_OTHER = 1u << 1,
  LOOM_AMDGPU_SHORT_WAIT_STORES = 3u,
  LOOM_AMDGPU_SHORT_WAIT_CHANNEL_COUNT = 20,
  LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES = 5,
  // Generated instructions have at most two physical results per native issue,
  // including the two coincident VOPD results. A packet may also retain
  // one store payload or pipeline release. All obligations expire after at
  // most five slots.
  LOOM_AMDGPU_SHORT_WAIT_SOURCE_COUNT = 3 * LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES,
};

// Required native instruction separation for each independent release channel.
static const uint8_t loom_amdgpu_short_wait_cycles[] = {
    2, 1, 2, 1, 2, 2, 1, 1, 5, 4, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2,
};

static_assert(IREE_ARRAYSIZE(loom_amdgpu_short_wait_cycles) ==
                  LOOM_AMDGPU_SHORT_WAIT_CHANNEL_COUNT,
              "every physical release channel has a bounded lifetime");

typedef struct loom_amdgpu_short_wait_source_t {
  // Position immediately after the native producer.
  uint64_t end_position;
  // Scheduled producer node.
  uint32_t node_index;
  // First register in the disjoint S/V/A/VCC/M0 index space.
  uint32_t base;
  // Number of contiguous physical registers.
  uint16_t count;
  // Reasons requiring at least plane + 1 slots after the producer.
  loom_amdgpu_short_wait_reasons_t reasons[LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES];
} loom_amdgpu_short_wait_source_t;

typedef struct loom_amdgpu_short_wait_prefix_t {
  // Packet before which an incoming dependency is delayed.
  uint32_t packet_index;
  // Channels queried at this native position.
  loom_amdgpu_short_wait_reasons_t reasons;
  // Packet start relative to the block's bounded prefix.
  uint8_t packet_position;
} loom_amdgpu_short_wait_prefix_t;

typedef struct loom_amdgpu_short_wait_block_t {
  // Consumers in the first five native instruction slots.
  loom_amdgpu_short_wait_prefix_t prefix[LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES];
  // Locally opened windows still live at block exit.
  loom_amdgpu_short_wait_source_t* sources;
  // Number of locally opened windows still live at block exit.
  uint8_t source_count;
  // Guaranteed issue progress, saturated at the maximum lifetime.
  uint8_t cycles;
  // The first native instruction activates incoming VALU-to-memory hazards.
  bool leading_dgemm;
  // Incoming channels explicitly retired inside this block.
  loom_amdgpu_short_wait_reasons_t drained_reasons;
  // Channels represented in this block's outgoing residual planes.
  loom_amdgpu_short_wait_reasons_t outgoing_reasons;
} loom_amdgpu_short_wait_block_t;

struct loom_amdgpu_short_wait_t {
  // Schedule providing the verified CFG and final packet identities.
  const loom_low_schedule_table_t* schedule;
  // Final physical allocation consumed by payload and result queries.
  const loom_low_allocation_table_t* allocation;
  // Arena owning transient summaries and the dataflow worklist.
  iree_arena_allocator_t* arena;
  // CFG summaries, or NULL when the function has no edges.
  loom_amdgpu_short_wait_block_t* blocks;
  // Prefix query and outgoing residual planes for each block.
  uint64_t* bits;
  // Queries at the current native packet's bounded instruction offsets.
  uint64_t* queries;
  // Populated query channels at each native offset of the current packet.
  loom_amdgpu_short_wait_reasons_t
      query_reasons[LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES];
  // Channels needed by any retained block prefix or transfer.
  loom_amdgpu_short_wait_reasons_t required_reasons;
  // Channels published by any locally produced outgoing window.
  loom_amdgpu_short_wait_reasons_t outgoing_reasons;
  // Number of words in one physical-register bit plane.
  iree_host_size_t word_count;
  // Starting plane of each channel; disabled channels occupy no planes.
  uint8_t channel_offsets[LOOM_AMDGPU_SHORT_WAIT_CHANNEL_COUNT];
  // Total planes in one prefix-query or outgoing-residual summary.
  uint8_t plane_count;
  // Local pending physical releases, independent of allocated register count.
  loom_amdgpu_short_wait_source_t sources[LOOM_AMDGPU_SHORT_WAIT_SOURCE_COUNT];
  // Populated local source prefix, including reusable expired entries.
  uint8_t source_count;
  // The pending packet activates still-fresh VALU-to-memory hazards.
  bool pending_dgemm;
  // Store payload, DOT result, or pipeline release retained until commit.
  loom_amdgpu_short_wait_source_t pending_source;
  // Native issue progress guaranteed by the pending packet.
  uint64_t pending_cycles;
  // Guaranteed native instruction position in the current block.
  uint64_t position;
  // Beginning of each physical bank in the combined index space.
  struct {
    // Accumulator registers follow the architectural vector extent.
    uint32_t agpr;
    // Scalar registers follow both vector banks.
    uint32_t sgpr;
    // Allocated scalar prefix length before ABI-fixed locations.
    uint32_t sgpr_allocated_count;
    // Physical beginning of the descriptor's ABI-fixed scalar range.
    uint32_t sgpr_fixed_base;
    // VCC follows ordinary and ABI-fixed scalar registers.
    uint32_t vcc;
    // M0 has independent storage after VCC.
    uint32_t m0;
  } banks;
  // Current region block index.
  uint16_t block_index;
  // Accumulator class, or UINT8_MAX when the descriptor set has none.
  uint8_t agpr_reg_class;
  // Target-owned M0 register class.
  uint8_t m0_reg_class;
  // Enabled fixed ALU and store-retention reasons.
  loom_amdgpu_short_wait_reasons_t enabled_reasons;
  // Incoming channels retired before the current packet.
  loom_amdgpu_short_wait_reasons_t drained_reasons;
};

static uint64_t* loom_amdgpu_short_wait_block_bits(
    const loom_amdgpu_short_wait_t* state, uint16_t block_index,
    unsigned plane) {
  return state->bits +
         (block_index * 2 * state->plane_count + plane) * state->word_count;
}

static void loom_amdgpu_short_wait_set_range(uint64_t* bits, uint32_t base,
                                             uint32_t count) {
  for (uint32_t i = base; i < base + count; ++i) {
    bits[i / 64] |= UINT64_C(1) << (i % 64);
  }
}

static bool loom_amdgpu_short_wait_map_range(
    const loom_amdgpu_short_wait_t* state, uint16_t register_class,
    uint32_t* base) {
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return true;
  }
  if (register_class == state->agpr_reg_class) {
    *base += state->banks.agpr;
    return true;
  }
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    if (*base >= state->banks.sgpr_allocated_count) {
      *base += state->banks.sgpr_allocated_count - state->banks.sgpr_fixed_base;
    }
    *base += state->banks.sgpr;
    return true;
  }
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VCC) {
    *base += state->banks.vcc;
    return true;
  }
  if (register_class == state->m0_reg_class) {
    *base += state->banks.m0;
    return true;
  }
  return false;
}

static loom_amdgpu_wait_state_reason_t loom_amdgpu_short_wait_reason(
    unsigned channel) {
  static const uint8_t reasons[] = {
      LOOM_AMDGPU_WAIT_STATE_REASON_STORE_DATA_REUSE,
      LOOM_AMDGPU_WAIT_STATE_REASON_STORE_DATA_REUSE,
      LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_DIV_FMAS_CONDITION,
      LOOM_AMDGPU_WAIT_STATE_REASON_M0_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_PIPELINE,
      LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_MEMORY,
  };
  return (loom_amdgpu_wait_state_reason_t)reasons[channel];
}

static loom_amdgpu_short_wait_reasons_t loom_amdgpu_short_wait_select_reasons(
    loom_amdgpu_processor_scheduling_bits_t scheduling,
    loom_amdgpu_descriptor_traits_t traits, uint8_t dot_identity_mask) {
  loom_amdgpu_short_wait_reasons_t reasons =
      (loom_amdgpu_descriptor_matrix_pass_count(traits) != 0
           ? (1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE)
           : 0) |
      ((uint32_t)dot_identity_mask << LOOM_AMDGPU_SHORT_WAIT_DOT_SHIFT);
  if (iree_any_bit_set(
          scheduling,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_FIXED_DATA_WAIT_STATES)) {
    reasons |= LOOM_AMDGPU_SHORT_WAIT_M0_READ;
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP)) {
      reasons |= (1u << LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ);
    }
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
      reasons |= LOOM_AMDGPU_SHORT_WAIT_VMEM_SGPR_READ;
    }
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_DIV_FMAS)) {
      reasons |= LOOM_AMDGPU_SHORT_WAIT_DIV_FMAS_CONDITION;
    }
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_PERMLANE)) {
      reasons |= LOOM_AMDGPU_SHORT_WAIT_PERMLANE_READ;
    }
  }
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL) &&
      iree_any_bit_set(
          scheduling,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES)) {
    reasons |= (1u << LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE);
  }
  if (iree_any_bit_set(
          scheduling,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES)) {
    reasons |= (1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ);
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE)) {
      reasons |= (1u << LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ);
    }
  }
  if (iree_any_bit_set(
          traits,
          LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA |
              LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING) &&
      iree_any_bit_set(
          scheduling,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES)) {
    reasons |= (1u << LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE);
  }
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM)) {
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_FMA64)) {
      reasons |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE;
    }
    if (iree_any_bit_set(
            scheduling,
            LOOM_AMDGPU_PROCESSOR_SCHEDULING_DGEMM_MEMORY_INTERLOCK) &&
        iree_any_bit_set(
            traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK)) {
      reasons |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY;
    }
  }
  return reasons;
}

iree_status_t loom_amdgpu_short_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_processor_scheduling_bits_t scheduling,
    loom_amdgpu_descriptor_traits_t traits, uint8_t dot_identity_mask,
    iree_arena_allocator_t* arena, loom_amdgpu_short_wait_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_short_wait_reasons_t reasons =
      loom_amdgpu_short_wait_select_reasons(scheduling, traits,
                                            dot_identity_mask);
  const loom_amdgpu_descriptor_set_info_t* info =
      loom_amdgpu_target_info_descriptor_set_at(
          schedule->target.descriptor_set->descriptor_set_ordinal);
  if (iree_any_bit_set(
          info->flags,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_STORE_DATA_WAIT_STATES)) {
    reasons |= LOOM_AMDGPU_SHORT_WAIT_STORES;
  }
  if (reasons == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_short_wait_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->schedule = schedule;
  state->allocation = allocation;
  state->arena = arena;
  state->enabled_reasons = reasons;
  for (unsigned channel = 0; channel < LOOM_AMDGPU_SHORT_WAIT_CHANNEL_COUNT;
       ++channel) {
    state->channel_offsets[channel] = state->plane_count;
    if (reasons & (1u << channel)) {
      state->plane_count += loom_amdgpu_short_wait_cycles[channel];
    }
  }
  state->agpr_reg_class =
      loom_amdgpu_agpr_reg_class_id(schedule->target.descriptor_set);
  for (uint16_t i = 0; i < schedule->target.descriptor_set->reg_class_count;
       ++i) {
    if (iree_any_bit_set(
            loom_amdgpu_reg_class_traits(schedule->target.descriptor_set, i),
            LOOM_AMDGPU_REG_CLASS_TRAIT_M0)) {
      state->m0_reg_class = (uint8_t)i;
      break;
    }
  }
  state->banks.agpr = allocation->physical_extents
                          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  state->banks.sgpr =
      state->banks.agpr + (state->agpr_reg_class == UINT8_MAX
                               ? 0
                               : allocation->physical_extents
                                     .ends_by_reg_class[state->agpr_reg_class]);
  state->banks.sgpr_allocated_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
  const loom_low_reg_class_t* sgpr =
      &schedule->target.descriptor_set
           ->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
  state->banks.sgpr_fixed_base = loom_low_reg_class_fixed_location_base(sgpr);
  state->banks.vcc = state->banks.sgpr + state->banks.sgpr_allocated_count +
                     loom_low_reg_class_fixed_location_count(sgpr);
  state->banks.m0 = state->banks.vcc + 1;
  const uint32_t pipeline_count =
      (reasons & LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE) != 0;
  state->word_count = (state->banks.m0 + 1 + pipeline_count + 63) / 64;
  if (schedule->cfg_graph.edge_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                   sizeof(*state->blocks),
                                                   (void**)&state->blocks));
    memset(state->blocks, 0, schedule->block_count * sizeof(*state->blocks));
    const iree_host_size_t word_count =
        schedule->block_count * (2 * state->plane_count) * state->word_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*state->bits), (void**)&state->bits));
    memset(state->bits, 0, word_count * sizeof(*state->bits));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state->plane_count * state->word_count, sizeof(*state->queries),
        (void**)&state->queries));
  }
  *out_state = state;
  return iree_ok_status();
}

void loom_amdgpu_short_wait_begin_block(loom_amdgpu_short_wait_t* state,
                                        uint16_t block_index) {
  state->block_index = block_index;
  state->position = 0;
  state->drained_reasons = 0;
  state->source_count = 0;
}

static loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_query_range(
    loom_amdgpu_short_wait_t* state, uint32_t base, uint16_t count,
    uint32_t instruction_offset, loom_amdgpu_short_wait_reasons_t reasons) {
  loom_amdgpu_short_wait_match_t match = {0};
  if (state->blocks != NULL && state->position + instruction_offset <
                                   LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES) {
    unsigned incoming_reasons = reasons & ~state->drained_reasons;
    while (incoming_reasons) {
      const unsigned channel =
          iree_math_count_trailing_zeros_u32(incoming_reasons);
      incoming_reasons &= incoming_reasons - 1;
      if (state->position + instruction_offset <
          loom_amdgpu_short_wait_cycles[channel]) {
        uint64_t* queries = state->queries + (state->channel_offsets[channel] +
                                              instruction_offset) *
                                                 state->word_count;
        if (!(state->query_reasons[instruction_offset] & (1u << channel))) {
          memset(queries, 0, state->word_count * sizeof(*queries));
          state->query_reasons[instruction_offset] |= 1u << channel;
        }
        loom_amdgpu_short_wait_set_range(queries, base, count);
      }
    }
  }
  for (unsigned i = 0; i < state->source_count; ++i) {
    const loom_amdgpu_short_wait_source_t* source = &state->sources[i];
    const uint64_t observed =
        state->position + instruction_offset - source->end_position;
    if (observed >= LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES ||
        base >= source->base + source->count || source->base >= base + count ||
        !(source->reasons[observed] & reasons)) {
      continue;
    }
    for (unsigned plane = (unsigned)observed;
         plane < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES; ++plane) {
      const unsigned matched = source->reasons[plane] & reasons;
      const unsigned residual = plane + 1 - (unsigned)observed;
      if (matched && residual > match.cycles) {
        match = (loom_amdgpu_short_wait_match_t){
            .reason = loom_amdgpu_short_wait_reason(
                iree_math_count_trailing_zeros_u32(matched)),
            .producer_node = source->node_index,
            .required_cycles = (uint16_t)(plane + 1),
            .observed_cycles = (uint16_t)observed,
            .cycles = (uint16_t)residual,
        };
      }
    }
  }
  return match;
}

loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_query(
    loom_amdgpu_short_wait_t* state, uint16_t register_class, uint32_t base,
    uint16_t count, uint32_t instruction_offset,
    loom_amdgpu_short_wait_reasons_t reasons) {
  reasons &= state->enabled_reasons;
  if (!reasons ||
      !loom_amdgpu_short_wait_map_range(state, register_class, &base)) {
    return (loom_amdgpu_short_wait_match_t){0};
  }
  return loom_amdgpu_short_wait_query_range(state, base, count,
                                            instruction_offset, reasons);
}

static loom_amdgpu_short_wait_source_t* loom_amdgpu_short_wait_source(
    loom_amdgpu_short_wait_t* state, uint32_t producer_node, uint32_t base,
    uint16_t count, uint64_t end_position) {
  unsigned free_index = state->source_count;
  unsigned index = 0;
  for (; index < state->source_count; ++index) {
    loom_amdgpu_short_wait_source_t* source = &state->sources[index];
    if (source->end_position == end_position &&
        source->node_index == producer_node && source->base == base &&
        source->count == count) {
      break;
    }
    const uint64_t elapsed = end_position - source->end_position;
    if (elapsed >= LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES ||
        !source->reasons[elapsed]) {
      free_index = index;
    }
  }
  if (index == state->source_count) {
    // Descriptor generation bounds simultaneous physical outputs. Reusing an
    // occupied row would silently discard a still-live hardware obligation.
    IREE_ASSERT_LT(free_index, LOOM_AMDGPU_SHORT_WAIT_SOURCE_COUNT);
    index = free_index;
    if (index == state->source_count) {
      ++state->source_count;
    }
    state->sources[index] = (loom_amdgpu_short_wait_source_t){
        .end_position = end_position,
        .node_index = producer_node,
        .base = base,
        .count = count,
    };
  }
  return &state->sources[index];
}

void loom_amdgpu_short_wait_record(loom_amdgpu_short_wait_t* state,
                                   uint32_t producer_node,
                                   uint16_t register_class, uint32_t base,
                                   uint16_t count,
                                   uint32_t instruction_end_offset,
                                   loom_amdgpu_short_wait_reasons_t reasons,
                                   uint16_t cycles) {
  reasons &= state->enabled_reasons;
  if (!reasons ||
      !loom_amdgpu_short_wait_map_range(state, register_class, &base)) {
    return;
  }
  const uint64_t end_position = state->position + instruction_end_offset;
  loom_amdgpu_short_wait_source_t* source = loom_amdgpu_short_wait_source(
      state, producer_node, base, count, end_position);
  for (unsigned plane = 0; plane < cycles; ++plane) {
    source->reasons[plane] |= reasons;
  }
}

loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_begin_packet(
    loom_amdgpu_short_wait_t* state, const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits) {
  state->pending_source = (loom_amdgpu_short_wait_source_t){0};
  const bool is_dgemm =
      iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM);
  state->pending_dgemm = is_dgemm && (state->enabled_reasons &
                                      LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY);
  state->pending_cycles =
      packet->descriptor != NULL
          ? 1
          : loom_amdgpu_structural_packet_minimum_instruction_count(structural);
  if (state->blocks != NULL &&
      state->position < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES) {
    memset(state->query_reasons, 0, sizeof(state->query_reasons));
  }
  const loom_amdgpu_short_wait_reasons_t dot_identity =
      (uint32_t)loom_amdgpu_descriptor_dot_identity_mask(traits)
      << LOOM_AMDGPU_SHORT_WAIT_DOT_SHIFT;
  const loom_amdgpu_short_wait_reasons_t dot_reasons =
      state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_DOT_MASK;
  if (is_dgemm &&
      (state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE)) {
    state->pending_source = (loom_amdgpu_short_wait_source_t){
        .node_index = packet->node_index,
        .base = state->banks.m0 + 1,
        .count = 1,
        .reasons = {LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE,
                    LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE},
    };
  }
  if (dot_identity) {
    const loom_low_allocation_assignment_t* result =
        loom_low_packet_result_assignment(state->allocation, packet, 0);
    state->pending_source = (loom_amdgpu_short_wait_source_t){
        .node_index = packet->node_index,
        .base = result->location_base,
        .count = result->location_count,
        .reasons = {dot_identity, dot_identity, dot_identity},
    };
  }
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY) &&
      (state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_STORES)) {
    const loom_amdgpu_store_data_wait_t binding =
        loom_amdgpu_descriptor_store_data_wait(
            state->schedule->target.descriptor_set, packet->descriptor);
    if (binding.valu_cycles != 0) {
      const loom_low_allocation_assignment_t* payload =
          loom_low_packet_descriptor_operand_assignment(
              state->allocation, packet, binding.operand_index);
      uint32_t base = payload->location_base;
      loom_amdgpu_short_wait_map_range(state, payload->descriptor_reg_class_id,
                                       &base);
      state->pending_source = (loom_amdgpu_short_wait_source_t){
          .node_index = packet->node_index,
          .base = base,
          .count = payload->location_count,
          .reasons = {LOOM_AMDGPU_SHORT_WAIT_STORE_VALU |
                          (binding.other_cycles != 0
                               ? LOOM_AMDGPU_SHORT_WAIT_STORE_OTHER
                               : 0),
                      binding.valu_cycles == 2
                          ? LOOM_AMDGPU_SHORT_WAIT_STORE_VALU
                          : 0},
      };
    }
  }
  loom_amdgpu_short_wait_match_t match = {0};
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_FMA64) &&
      (state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE)) {
    match = loom_amdgpu_short_wait_query_range(
        state, state->banks.m0 + 1, 1, 0,
        LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE);
  }
  const loom_amdgpu_short_wait_reasons_t memory_reasons =
      iree_any_bit_set(traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK)
          ? state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY
          : 0;
  if (!(state->enabled_reasons & LOOM_AMDGPU_SHORT_WAIT_STORES) &&
      !dot_reasons && !memory_reasons) {
    return match;
  }
  const bool dot_consumer =
      !iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX) &&
      (iree_any_bit_set(
           traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU |
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER) ||
       structural->vector_alu_instruction_count != 0);
  if (structural->moves.count != 0) {
    for (iree_host_size_t i = 0; i < structural->moves.count && i < 3; ++i) {
      const loom_low_move_t* move =
          &state->allocation->moves[structural->moves.start + i];
      if (dot_reasons && move->source.location_kind ==
                             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
        loom_amdgpu_short_wait_match_t candidate = loom_amdgpu_short_wait_query(
            state, move->source.descriptor_reg_class_id, move->source.location,
            1, (uint32_t)i, dot_reasons);
        if (candidate.cycles > match.cycles) {
          match = candidate;
        }
      }
      loom_amdgpu_short_wait_match_t candidate = loom_amdgpu_short_wait_query(
          state, move->destination.descriptor_reg_class_id,
          move->destination.location, 1, (uint32_t)i,
          LOOM_AMDGPU_SHORT_WAIT_STORE_VALU | dot_reasons);
      if (candidate.cycles > match.cycles) {
        match = candidate;
      }
    }
  } else if (state->pending_cycles != 0) {
    if ((dot_reasons && dot_consumer) || memory_reasons) {
      for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
        const loom_low_allocation_assignment_t* operand =
            loom_low_packet_operand_assignment(state->allocation, packet, i);
        if (operand != NULL &&
            operand->location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
          // Generation fixes DOT's accumulator at input two. Other inputs and
          // every ordinary consumer retain all pending native identities.
          const loom_amdgpu_short_wait_reasons_t reasons =
              (dot_consumer
                   ? (i == 2 ? dot_reasons & ~dot_identity : dot_reasons)
                   : 0) |
              memory_reasons;
          const loom_amdgpu_short_wait_match_t candidate =
              loom_amdgpu_short_wait_query(
                  state, operand->descriptor_reg_class_id,
                  operand->location_base, operand->location_count, 0, reasons);
          if (candidate.cycles > match.cycles) {
            match = candidate;
          }
        }
      }
    }
    const loom_amdgpu_short_wait_reasons_t reasons =
        (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU |
                                      LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX) ||
                 structural->vector_alu_instruction_count != 0
             ? LOOM_AMDGPU_SHORT_WAIT_STORE_VALU
             : LOOM_AMDGPU_SHORT_WAIT_STORE_OTHER) |
        (dot_consumer ? dot_reasons & ~dot_identity : 0);
    for (uint16_t i = 0; i < packet->node->result_count; ++i) {
      const loom_low_allocation_assignment_t* result =
          loom_low_packet_result_assignment(state->allocation, packet, i);
      if (result != NULL &&
          result->location_kind ==
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
        loom_amdgpu_short_wait_match_t candidate = loom_amdgpu_short_wait_query(
            state, result->descriptor_reg_class_id, result->location_base,
            result->location_count, 0, reasons);
        if (candidate.cycles > match.cycles) {
          match = candidate;
        }
      }
    }
  }
  return match;
}

void loom_amdgpu_short_wait_advance(loom_amdgpu_short_wait_t* state,
                                    uint64_t cycles) {
  state->position += cycles;
}

void loom_amdgpu_short_wait_drain_stores(loom_amdgpu_short_wait_t* state) {
  state->drained_reasons |= LOOM_AMDGPU_SHORT_WAIT_STORES;
  for (unsigned i = 0; i < state->source_count; ++i) {
    state->sources[i].reasons[0] &= ~LOOM_AMDGPU_SHORT_WAIT_STORES;
    state->sources[i].reasons[1] &= ~LOOM_AMDGPU_SHORT_WAIT_STORES;
  }
}

void loom_amdgpu_short_wait_commit(loom_amdgpu_short_wait_t* state,
                                   const loom_low_packet_view_t* packet) {
  if (state->pending_dgemm) {
    // Only the immediately preceding VALU can retain a dependency after this
    // DGEMM issues. Keep that path relationship on the producer's existing row;
    // independently joined VALU and DGEMM facts would invent hazards.
    for (unsigned i = 0; i < state->source_count; ++i) {
      loom_amdgpu_short_wait_source_t* source = &state->sources[i];
      if (source->end_position == state->position &&
          (source->reasons[1] &
           (1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE))) {
        source->reasons[0] |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY;
        source->reasons[1] |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY;
      }
    }
    if (state->blocks != NULL && state->position == 0) {
      state->blocks[state->block_index].leading_dgemm = true;
      state->required_reasons |=
          1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE;
    }
  }
  if (state->blocks != NULL) {
    loom_amdgpu_short_wait_block_t* block = &state->blocks[state->block_index];
    for (unsigned offset = 0;
         state->position + offset < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES;
         ++offset) {
      const unsigned position = (unsigned)state->position + offset;
      loom_amdgpu_short_wait_reasons_t committed_reasons = 0;
      unsigned reasons = state->query_reasons[offset];
      while (reasons) {
        const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
        reasons &= reasons - 1;
        if (position >= loom_amdgpu_short_wait_cycles[channel]) {
          continue;
        }
        committed_reasons |= 1u << channel;
        uint64_t* prefix = loom_amdgpu_short_wait_block_bits(
            state, state->block_index,
            state->channel_offsets[channel] + position);
        const uint64_t* queries =
            state->queries +
            (state->channel_offsets[channel] + offset) * state->word_count;
        for (iree_host_size_t word = 0; word < state->word_count; ++word) {
          prefix[word] |= queries[word];
        }
      }
      if (committed_reasons) {
        block->prefix[position] = (loom_amdgpu_short_wait_prefix_t){
            .packet_index = (uint32_t)packet->packet_index,
            .reasons = block->prefix[position].reasons | committed_reasons,
            .packet_position = (uint8_t)state->position,
        };
        state->required_reasons |= committed_reasons;
      }
    }
  }
  state->position += state->pending_cycles;
  if (state->pending_source.reasons[0]) {
    loom_amdgpu_short_wait_source_t* source = loom_amdgpu_short_wait_source(
        state, state->pending_source.node_index, state->pending_source.base,
        state->pending_source.count, state->position);
    for (unsigned plane = 0; plane < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES;
         ++plane) {
      source->reasons[plane] |= state->pending_source.reasons[plane];
    }
  }
}

static void loom_amdgpu_short_wait_publish_bits(
    loom_amdgpu_short_wait_t* state, uint16_t block_index,
    const loom_amdgpu_short_wait_source_t* source, unsigned elapsed) {
  if (elapsed < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES) {
    state->blocks[block_index].outgoing_reasons |= source->reasons[elapsed];
    state->outgoing_reasons |= source->reasons[elapsed];
  }
  for (unsigned plane = elapsed; plane < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES;
       ++plane) {
    unsigned reasons = source->reasons[plane];
    while (reasons) {
      const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
      reasons &= reasons - 1;
      loom_amdgpu_short_wait_set_range(
          loom_amdgpu_short_wait_block_bits(
              state, block_index,
              state->plane_count + state->channel_offsets[channel] + plane -
                  elapsed),
          source->base, source->count);
    }
  }
}

iree_status_t loom_amdgpu_short_wait_end_block(
    loom_amdgpu_short_wait_t* state) {
  if (state->blocks == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_short_wait_block_t* block = &state->blocks[state->block_index];
  block->cycles =
      (uint8_t)iree_min(state->position, LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES);
  block->drained_reasons = state->drained_reasons;
  unsigned live_count = 0;
  for (unsigned i = 0; i < state->source_count; ++i) {
    loom_amdgpu_short_wait_source_t* source = &state->sources[i];
    const uint64_t elapsed = state->position - source->end_position;
    if (elapsed >= LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES ||
        !source->reasons[elapsed]) {
      continue;
    }
    for (unsigned plane = 0; plane < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES;
         ++plane) {
      source->reasons[plane] =
          plane + elapsed < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES
              ? source->reasons[plane + elapsed]
              : 0;
    }
    loom_amdgpu_short_wait_publish_bits(state, state->block_index, source, 0);
    state->sources[live_count++] = *source;
  }
  if (live_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, live_count,
                                                   sizeof(*block->sources),
                                                   (void**)&block->sources));
    memcpy(block->sources, state->sources,
           live_count * sizeof(*block->sources));
  }
  block->source_count = (uint8_t)live_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_short_wait_transfer(void* user_data,
                                                     uint16_t source_index,
                                                     uint16_t target_index,
                                                     bool* out_changed) {
  loom_amdgpu_short_wait_t* state = (loom_amdgpu_short_wait_t*)user_data;
  loom_amdgpu_short_wait_block_t* target = &state->blocks[target_index];
  *out_changed = false;
  unsigned reasons = state->blocks[source_index].outgoing_reasons &
                     state->required_reasons & ~target->drained_reasons;
  while (reasons) {
    const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
    reasons &= reasons - 1;
    bool changed = false;
    for (unsigned plane = 0;
         plane + target->cycles < loom_amdgpu_short_wait_cycles[channel];
         ++plane) {
      uint64_t* output = loom_amdgpu_short_wait_block_bits(
          state, target_index,
          state->plane_count + state->channel_offsets[channel] + plane);
      const uint64_t* input = loom_amdgpu_short_wait_block_bits(
          state, source_index,
          state->plane_count + state->channel_offsets[channel] + plane +
              target->cycles);
      for (iree_host_size_t word = 0; word < state->word_count; ++word) {
        changed |= (input[word] & ~output[word]) != 0;
        output[word] |= input[word];
      }
    }
    if (changed) {
      target->outgoing_reasons |= 1u << channel;
      *out_changed = true;
    }
  }
  if (target->leading_dgemm && target->cycles == 1 &&
      (state->blocks[source_index].outgoing_reasons &
       (1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE))) {
    const uint64_t* input = loom_amdgpu_short_wait_block_bits(
        state, source_index,
        state->plane_count +
            state->channel_offsets
                [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE] +
            1);
    uint64_t* output = loom_amdgpu_short_wait_block_bits(
        state, target_index,
        state->plane_count +
            state->channel_offsets[LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY_SHIFT]);
    for (iree_host_size_t word = 0; word < state->word_count; ++word) {
      *out_changed |= (input[word] & ~output[word]) != 0;
      output[word] |= input[word];
      if (output[word]) {
        target->outgoing_reasons |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_short_wait_resolve(
    loom_amdgpu_short_wait_t* state, loom_amdgpu_short_wait_emit_fn_t emit,
    void* user_data) {
  if (state->blocks == NULL ||
      !(state->required_reasons & state->outgoing_reasons)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_dataflow_solve(&state->schedule->cfg_graph,
                                               loom_amdgpu_short_wait_transfer,
                                               state, state->arena));
  uint64_t* incoming = state->queries;
  loom_amdgpu_wait_state_t pending = {0};
  bool has_pending = false;
  iree_status_t status = iree_ok_status();
  for (uint16_t index = 0;
       index < state->schedule->block_count && iree_status_is_ok(status);
       ++index) {
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(&state->schedule->cfg_graph, index);
    loom_amdgpu_short_wait_reasons_t incoming_reasons = 0;
    memset(incoming, 0,
           state->plane_count * state->word_count * sizeof(*incoming));
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      if (!loom_cfg_graph_block_is_reachable(&state->schedule->cfg_graph,
                                             predecessors.values[i])) {
        continue;
      }
      unsigned reasons =
          state->blocks[predecessors.values[i]].outgoing_reasons &
          state->required_reasons;
      incoming_reasons |= reasons;
      while (reasons) {
        const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
        reasons &= reasons - 1;
        const uint64_t* output = loom_amdgpu_short_wait_block_bits(
            state, predecessors.values[i],
            state->plane_count + state->channel_offsets[channel]);
        uint64_t* input =
            incoming + state->channel_offsets[channel] * state->word_count;
        for (iree_host_size_t word = 0;
             word < loom_amdgpu_short_wait_cycles[channel] * state->word_count;
             ++word) {
          input[word] |= output[word];
        }
      }
    }
    loom_amdgpu_short_wait_block_t* block = &state->blocks[index];
    if (block->leading_dgemm &&
        (incoming_reasons &
         (1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE))) {
      // Project the leading DGEMM's channel transition into the input planes.
      // Memory can first query it at position one. Normal delay aging removes
      // the dependency when a wait before or after DGEMM supplies that slot.
      const uint64_t* potential =
          incoming + (state->channel_offsets
                          [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE] +
                      1) *
                         state->word_count;
      uint64_t* activated =
          incoming +
          state->channel_offsets[LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY_SHIFT] *
              state->word_count;
      for (iree_host_size_t word = 0; word < state->word_count; ++word) {
        activated[word] |= potential[word];
        activated[state->word_count + word] |= potential[word];
      }
      incoming_reasons |= LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY;
    }
    uint8_t inserted[LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES] = {0};
    unsigned delay = 0;
    for (unsigned position = 0;
         position + delay < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES &&
         iree_status_is_ok(status);
         ++position) {
      unsigned residual = 0;
      unsigned required = 0;
      unsigned matched_channel = 0;
      unsigned reasons = block->prefix[position].reasons & incoming_reasons;
      while (reasons) {
        const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
        reasons &= reasons - 1;
        if (position + delay >= loom_amdgpu_short_wait_cycles[channel]) {
          continue;
        }
        const unsigned channel_offset = state->channel_offsets[channel];
        const uint64_t* queries = loom_amdgpu_short_wait_block_bits(
            state, index, channel_offset + position);
        for (unsigned plane = position + delay;
             plane < loom_amdgpu_short_wait_cycles[channel]; ++plane) {
          for (iree_host_size_t word = 0; word < state->word_count; ++word) {
            if ((queries[word] &
                 incoming[(channel_offset + plane) * state->word_count +
                          word]) &&
                plane + 1 - position - delay > residual) {
              matched_channel = channel;
              required = plane + 1;
              residual = plane + 1 - position - delay;
            }
          }
        }
      }
      if (residual == 0) {
        continue;
      }
      const loom_amdgpu_short_wait_prefix_t* prefix = &block->prefix[position];
      const loom_low_packet_view_t packet =
          loom_low_packet_at(state->schedule, prefix->packet_index);
      if (has_pending && pending.node_index == packet.node_index) {
        pending.cycle_count += (uint16_t)residual;
        pending.required_cycle_count += (uint16_t)residual;
      } else {
        if (has_pending) {
          status = emit(user_data, &pending);
        }
        if (!iree_status_is_ok(status)) {
          break;
        }
        pending = (loom_amdgpu_wait_state_t){
            .reason = loom_amdgpu_short_wait_reason(matched_channel),
            .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP,
            .block_index = index,
            .node_index = packet.node_index,
            .scheduled_ordinal = packet.node->scheduled_ordinal,
            .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
            .consumer_node = packet.node->source_ordinal,
            .required_cycle_count = (uint16_t)required,
            .observed_cycle_count = (uint16_t)(position + delay),
            .cycle_count = (uint16_t)residual,
        };
        has_pending = true;
      }
      inserted[prefix->packet_position] += (uint8_t)residual;
      delay += residual;
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    // Refine forward-edge outputs with actual waits. Backedges retain the
    // conservative fixed point; only progress after a local producer counts.
    unsigned reasons =
        (block->outgoing_reasons | incoming_reasons) & state->required_reasons;
    block->outgoing_reasons = 0;
    while (reasons) {
      const unsigned channel = iree_math_count_trailing_zeros_u32(reasons);
      reasons &= reasons - 1;
      const unsigned channel_offset = state->channel_offsets[channel];
      for (unsigned plane = 0; plane < loom_amdgpu_short_wait_cycles[channel];
           ++plane) {
        uint64_t* outgoing = loom_amdgpu_short_wait_block_bits(
            state, index, state->plane_count + channel_offset + plane);
        const unsigned shifted = plane + block->cycles + delay;
        for (iree_host_size_t word = 0; word < state->word_count; ++word) {
          outgoing[word] =
              shifted < loom_amdgpu_short_wait_cycles[channel] &&
                      !(block->drained_reasons & (1u << channel))
                  ? incoming[(channel_offset + shifted) * state->word_count +
                             word]
                  : 0;
          if (outgoing[word]) {
            block->outgoing_reasons |= 1u << channel;
          }
        }
      }
    }
    for (unsigned i = 0; i < block->source_count; ++i) {
      const loom_amdgpu_short_wait_source_t* source = &block->sources[i];
      unsigned elapsed = 0;
      for (unsigned position = 0; position < LOOM_AMDGPU_SHORT_WAIT_MAX_CYCLES;
           ++position) {
        if (position >= source->end_position) {
          elapsed += inserted[position];
        }
      }
      loom_amdgpu_short_wait_publish_bits(state, index, source, elapsed);
    }
  }
  if (iree_status_is_ok(status) && has_pending) {
    status = emit(user_data, &pending);
  }
  return status;
}
