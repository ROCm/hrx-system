// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_coexecution.h"

#include <string.h>

#include "iree/base/bitfield.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/cfg_graph.h"

enum {
  LOOM_AMDGPU_MATRIX_COEXECUTION_NODE_CHUNK_CAPACITY = 64,
  LOOM_AMDGPU_MATRIX_COEXECUTION_CONSUMER_CHUNK_CAPACITY = 128,
};

typedef enum loom_amdgpu_matrix_coexecution_channel_e {
  // Result storage protected from a later matrix source read.
  LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_MATRIX_RESULT = 0,
  // Result storage protected from a later ordinary VALU read or write.
  LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT = 1,
  // Source storage protected from a later ordinary VALU write.
  LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_SOURCE = 2,
  // Number of independent physical release channels.
  LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT = 3,
} loom_amdgpu_matrix_coexecution_channel_t;

typedef enum loom_amdgpu_matrix_coexecution_query_flag_bits_e {
  // The queried packet is a matrix consumer.
  LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_MATRIX_CONSUMER = 1u << 0,
  // Retain the queried access in the active static block summary.
  LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_RETAIN_CONSUMER = 1u << 1,
} loom_amdgpu_matrix_coexecution_query_flag_bits_t;
typedef uint8_t loom_amdgpu_matrix_coexecution_query_flags_t;

typedef struct loom_amdgpu_matrix_coexecution_family_layout_t {
  // Descriptor operand containing the matrix result.
  uint8_t result_operand_index;
  // First descriptor operand protected as a matrix source.
  uint8_t source_operand_start;
  // Number of consecutive protected matrix source operands.
  uint8_t source_operand_count;
} loom_amdgpu_matrix_coexecution_family_layout_t;

static const loom_amdgpu_matrix_coexecution_family_layout_t
    kSourceLayouts[LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_COUNT] = {
#include "loom/target/arch/amdgpu/matrix_coexecution_source_layouts.inl"
};

// One retained source packet in final scheduled order.
typedef struct loom_amdgpu_matrix_coexecution_source_t {
  // Scheduled packet index opening the release window.
  uint32_t packet_index;
  // Schedule node opening the release window.
  uint32_t producer_node;
  // Descriptor operand containing the matrix result.
  uint8_t result_operand_index;
  // First protected matrix source descriptor operand.
  uint8_t source_operand_start;
  // Number of consecutive protected matrix source operands.
  uint8_t source_operand_count;
  // Required vector issues before a dependent matrix packet.
  uint8_t matrix_issue_distance;
  // Required vector issues before a dependent ordinary VALU packet.
  uint8_t vector_issue_distance;
} loom_amdgpu_matrix_coexecution_source_t;

// Dense scratch state for one physical VGPR in the active traversal.
typedef struct loom_amdgpu_matrix_coexecution_active_state_t {
  // Absolute vector issue position releasing each channel.
  uint64_t release_positions[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
  // Original release distance by channel.
  uint8_t required[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
  // Active traversal epoch owning this state.
  uint32_t epoch;
  // Source schedule node by channel.
  uint32_t producer_nodes[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
  // Release channels consumed by accesses in the active static block.
  uint8_t consumer_channels;
} loom_amdgpu_matrix_coexecution_active_state_t;

// Sparse outgoing state for one physical VGPR.
typedef struct loom_amdgpu_matrix_coexecution_frontier_node_t {
  // Next physical VGPR state in the block frontier.
  struct loom_amdgpu_matrix_coexecution_frontier_node_t* next;
  // Physical VGPR index represented by this row.
  uint32_t vgpr_index;
  // Outstanding vector issue slots by release channel.
  uint8_t remaining[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
  // Original release distance by channel. Zero denotes conservative static
  // state with no retained provenance.
  uint8_t required[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
  // Source schedule node by channel.
  uint32_t producer_nodes[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT];
} loom_amdgpu_matrix_coexecution_frontier_node_t;

typedef struct loom_amdgpu_matrix_coexecution_node_chunk_t {
  // Number of populated rows.
  uint16_t count;
  // Sparse frontier rows.
  loom_amdgpu_matrix_coexecution_frontier_node_t
      nodes[LOOM_AMDGPU_MATRIX_COEXECUTION_NODE_CHUNK_CAPACITY];
} loom_amdgpu_matrix_coexecution_node_chunk_t;

// Sparse physical access summary for one block.
typedef struct loom_amdgpu_matrix_coexecution_consumer_t {
  // Next physical VGPR access in the block.
  struct loom_amdgpu_matrix_coexecution_consumer_t* next;
  // Physical VGPR index accessed by the block.
  uint32_t vgpr_index;
  // Release channels consumed by accesses to the physical VGPR.
  uint8_t channels;
} loom_amdgpu_matrix_coexecution_consumer_t;

typedef struct loom_amdgpu_matrix_coexecution_consumer_chunk_t {
  // Number of populated rows.
  uint16_t count;
  // Sparse physical access rows.
  loom_amdgpu_matrix_coexecution_consumer_t
      consumers[LOOM_AMDGPU_MATRIX_COEXECUTION_CONSUMER_CHUNK_CAPACITY];
} loom_amdgpu_matrix_coexecution_consumer_chunk_t;

typedef enum loom_amdgpu_matrix_coexecution_block_flag_bits_e {
  // Block is present in the static propagation worklist.
  LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_QUEUED = 1u << 0,
  // Block has a refined outgoing state from final wait-state traversal.
  LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_RESOLVED = 1u << 1,
} loom_amdgpu_matrix_coexecution_block_flag_bits_t;
typedef uint8_t loom_amdgpu_matrix_coexecution_block_flags_t;

typedef struct loom_amdgpu_matrix_coexecution_block_t {
  // Conservative outgoing state after static CFG propagation.
  loom_amdgpu_matrix_coexecution_frontier_node_t* static_outgoing;
  // Refined outgoing state after final wait-state traversal.
  loom_amdgpu_matrix_coexecution_frontier_node_t* resolved_outgoing;
  // Physical accesses that discharge incoming release channels.
  loom_amdgpu_matrix_coexecution_consumer_t* consumers;
  // Saturated native vector issue count for the block.
  uint8_t vector_issue_count;
  // Propagation and resolution flags.
  loom_amdgpu_matrix_coexecution_block_flags_t flags;
} loom_amdgpu_matrix_coexecution_block_t;

struct loom_amdgpu_matrix_coexecution_t {
  // Schedule table being planned.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying final physical locations and moves.
  const loom_low_allocation_table_t* allocation;
  // Descriptor set selected by the scheduled function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Selected processor-profile rule table.
  const loom_amdgpu_matrix_coexecution_profile_info_t* profile;
  // Arena owning transient frontier nodes.
  iree_arena_allocator_t* arena;
  // Retained qualified source rows in scheduled packet order.
  loom_amdgpu_matrix_coexecution_source_t* sources;
  // Allocated entries in |sources|.
  iree_host_size_t source_capacity;
  // Populated entries in |sources|.
  iree_host_size_t source_count;
  // Next source consumed by final wait-state traversal.
  iree_host_size_t source_cursor;
  // Per-block sparse frontier state, or NULL for a single-block schedule.
  loom_amdgpu_matrix_coexecution_block_t* blocks;
  // Dense active scratch state indexed by physical VGPR.
  loom_amdgpu_matrix_coexecution_active_state_t* active_states;
  // Physical VGPR indices touched in the active epoch.
  uint32_t* active_vgpr_indices;
  // Number of entries in |active_vgpr_indices|.
  iree_host_size_t active_vgpr_count;
  // Multi-block scratch lookup from physical VGPR to one outgoing frontier.
  loom_amdgpu_matrix_coexecution_frontier_node_t** frontier_lookup;
  // Number of physical VGPR units in the release lattice.
  uint32_t vgpr_count;
  // Active scratch-state epoch.
  uint32_t active_epoch;
  // Absolute vector issue position in the active block.
  uint64_t current_issue_position;
  // Circular static propagation worklist.
  uint16_t* worklist;
  // Current frontier-node allocation chunk.
  loom_amdgpu_matrix_coexecution_node_chunk_t* node_chunk;
  // Current consumer-node allocation chunk.
  loom_amdgpu_matrix_coexecution_consumer_chunk_t* consumer_chunk;
  // Current block index, or UINT16_MAX outside block processing.
  uint16_t active_block_index;
};

static bool loom_amdgpu_matrix_coexecution_descriptor_is_source(
    const loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(coexecution->descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_COEXECUTION_SOURCE);
}

static uint8_t loom_amdgpu_matrix_coexecution_saturating_add(uint8_t lhs,
                                                             uint16_t rhs,
                                                             uint8_t limit) {
  const uint8_t available = (uint8_t)(limit - lhs);
  return rhs >= available ? limit : (uint8_t)(lhs + rhs);
}

static uint16_t loom_amdgpu_matrix_coexecution_saturate_issue_count(
    const loom_amdgpu_matrix_coexecution_t* coexecution, uint64_t issue_count) {
  return issue_count >= coexecution->profile->maximum_issue_distance
             ? coexecution->profile->maximum_issue_distance
             : (uint16_t)issue_count;
}

static void loom_amdgpu_matrix_coexecution_reset_active(
    loom_amdgpu_matrix_coexecution_t* coexecution) {
  ++coexecution->active_epoch;
  if (coexecution->active_epoch == 0) {
    memset(coexecution->active_states, 0,
           (iree_host_size_t)coexecution->vgpr_count *
               sizeof(*coexecution->active_states));
    coexecution->active_epoch = 1;
  }
  coexecution->active_vgpr_count = 0;
  coexecution->current_issue_position = 0;
}

static loom_amdgpu_matrix_coexecution_active_state_t*
loom_amdgpu_matrix_coexecution_get_active(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint32_t vgpr_index) {
  IREE_ASSERT_LT(vgpr_index, coexecution->vgpr_count);
  loom_amdgpu_matrix_coexecution_active_state_t* state =
      &coexecution->active_states[vgpr_index];
  if (state->epoch == coexecution->active_epoch) {
    return state;
  }
  *state = (loom_amdgpu_matrix_coexecution_active_state_t){
      .epoch = coexecution->active_epoch,
  };
  coexecution->active_vgpr_indices[coexecution->active_vgpr_count++] =
      vgpr_index;
  return state;
}

static const loom_amdgpu_matrix_coexecution_active_state_t*
loom_amdgpu_matrix_coexecution_try_active(
    const loom_amdgpu_matrix_coexecution_t* coexecution, uint32_t vgpr_index) {
  IREE_ASSERT_LT(vgpr_index, coexecution->vgpr_count);
  const loom_amdgpu_matrix_coexecution_active_state_t* state =
      &coexecution->active_states[vgpr_index];
  return state->epoch == coexecution->active_epoch ? state : NULL;
}

static loom_amdgpu_matrix_coexecution_active_state_t*
loom_amdgpu_matrix_coexecution_try_active_mutable(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint32_t vgpr_index) {
  IREE_ASSERT_LT(vgpr_index, coexecution->vgpr_count);
  loom_amdgpu_matrix_coexecution_active_state_t* state =
      &coexecution->active_states[vgpr_index];
  return state->epoch == coexecution->active_epoch ? state : NULL;
}

static void loom_amdgpu_matrix_coexecution_publish_range(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_matrix_coexecution_channel_t channel, uint8_t issue_distance,
    uint32_t producer_node) {
  IREE_ASSERT(loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_VGPR));
  IREE_ASSERT_LE(assignment->location_base + assignment->location_count,
                 coexecution->vgpr_count);
  const uint64_t release_position =
      coexecution->current_issue_position + issue_distance;
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_matrix_coexecution_active_state_t* state =
        loom_amdgpu_matrix_coexecution_get_active(
            coexecution, assignment->location_base + i);
    if (release_position > state->release_positions[channel]) {
      state->release_positions[channel] = release_position;
      state->required[channel] = issue_distance;
      state->producer_nodes[channel] = producer_node;
    } else if (release_position == state->release_positions[channel] &&
               state->producer_nodes[channel] != producer_node) {
      state->producer_nodes[channel] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }
}

static void loom_amdgpu_matrix_coexecution_publish_source_assignment(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_matrix_coexecution_channel_t channel, uint8_t issue_distance,
    uint32_t producer_node) {
  if (!loom_low_allocation_assignment_is_physical_register_class(
          assignment, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return;
  }
  loom_amdgpu_matrix_coexecution_publish_range(coexecution, assignment, channel,
                                               issue_distance, producer_node);
}

static void loom_amdgpu_matrix_coexecution_publish_source(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_matrix_coexecution_source_t* source) {
  const loom_low_allocation_assignment_t* result =
      loom_low_packet_descriptor_operand_assignment(
          coexecution->allocation, packet, source->result_operand_index);
  loom_amdgpu_matrix_coexecution_publish_range(
      coexecution, result, LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_MATRIX_RESULT,
      source->matrix_issue_distance, source->producer_node);
  loom_amdgpu_matrix_coexecution_publish_range(
      coexecution, result, LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT,
      source->vector_issue_distance, source->producer_node);
  for (uint8_t i = 0; i < source->source_operand_count; ++i) {
    const loom_low_allocation_assignment_t* input =
        loom_low_packet_descriptor_operand_assignment(
            coexecution->allocation, packet,
            (uint16_t)(source->source_operand_start + i));
    loom_amdgpu_matrix_coexecution_publish_source_assignment(
        coexecution, input, LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_SOURCE,
        source->vector_issue_distance, source->producer_node);
  }
}

static const loom_amdgpu_matrix_coexecution_source_t*
loom_amdgpu_matrix_coexecution_append_source(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet) {
  IREE_ASSERT_LT(coexecution->source_count, coexecution->source_capacity);
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  loom_amdgpu_matrix_coexecution_source_kind_t source_kind =
      LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA;
  if (iree_any_bit_set(descriptor->instruction_class_flags,
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_SWMMAC)) {
    source_kind = LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC;
  } else if (iree_any_bit_set(descriptor->instruction_class_flags,
                              LOOM_LOW_INSTRUCTION_CLASS_FLAG_WMMA)) {
    source_kind = LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA;
  } else {
    IREE_ASSERT_UNREACHABLE(
        "matrix coexecution source trait requires WMMA or SWMMAC");
  }
  const loom_low_schedule_class_t* schedule_class =
      &coexecution->descriptor_set
           ->schedule_classes[packet->descriptor->schedule_class_id];
  IREE_ASSERT_LE(schedule_class->latency_cycles, UINT8_MAX);
  const uint8_t latency_cycles = (uint8_t)schedule_class->latency_cycles;
  const loom_amdgpu_matrix_coexecution_release_t* release =
      &coexecution->profile->releases[source_kind][latency_cycles];
  IREE_ASSERT_NE(release->matrix_issue_distance, 0);
  const loom_amdgpu_matrix_coexecution_family_layout_t* layout =
      &kSourceLayouts[source_kind];
  loom_amdgpu_matrix_coexecution_source_t* source =
      &coexecution->sources[coexecution->source_count++];
  *source = (loom_amdgpu_matrix_coexecution_source_t){
      .packet_index = (uint32_t)packet->packet_index,
      .producer_node = packet->node_index,
      .result_operand_index = layout->result_operand_index,
      .source_operand_start = layout->source_operand_start,
      .source_operand_count = layout->source_operand_count,
      .matrix_issue_distance = release->matrix_issue_distance,
      .vector_issue_distance = release->vector_issue_distance,
  };
  return source;
}

static const loom_amdgpu_matrix_coexecution_source_t*
loom_amdgpu_matrix_coexecution_current_source(
    const loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet) {
  if (coexecution->source_cursor == coexecution->source_count) {
    return NULL;
  }
  const loom_amdgpu_matrix_coexecution_source_t* source =
      &coexecution->sources[coexecution->source_cursor];
  IREE_ASSERT_LE(packet->packet_index, source->packet_index);
  return source->packet_index == packet->packet_index ? source : NULL;
}

static iree_status_t loom_amdgpu_matrix_coexecution_allocate_frontier_node(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    loom_amdgpu_matrix_coexecution_frontier_node_t** out_node) {
  if (coexecution->node_chunk == NULL ||
      coexecution->node_chunk->count ==
          LOOM_AMDGPU_MATRIX_COEXECUTION_NODE_CHUNK_CAPACITY) {
    loom_amdgpu_matrix_coexecution_node_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(coexecution->arena, sizeof(*chunk),
                                             (void**)&chunk));
    *chunk = (loom_amdgpu_matrix_coexecution_node_chunk_t){0};
    coexecution->node_chunk = chunk;
  }
  *out_node = &coexecution->node_chunk->nodes[coexecution->node_chunk->count++];
  **out_node = (loom_amdgpu_matrix_coexecution_frontier_node_t){0};
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_matrix_coexecution_allocate_consumer(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    loom_amdgpu_matrix_coexecution_consumer_t** out_consumer) {
  if (coexecution->consumer_chunk == NULL ||
      coexecution->consumer_chunk->count ==
          LOOM_AMDGPU_MATRIX_COEXECUTION_CONSUMER_CHUNK_CAPACITY) {
    loom_amdgpu_matrix_coexecution_consumer_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(coexecution->arena, sizeof(*chunk),
                                             (void**)&chunk));
    *chunk = (loom_amdgpu_matrix_coexecution_consumer_chunk_t){0};
    coexecution->consumer_chunk = chunk;
  }
  *out_consumer = &coexecution->consumer_chunk
                       ->consumers[coexecution->consumer_chunk->count++];
  **out_consumer = (loom_amdgpu_matrix_coexecution_consumer_t){0};
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_matrix_coexecution_retain_consumers(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    loom_amdgpu_matrix_coexecution_block_t* block) {
  for (iree_host_size_t i = 0; i < coexecution->active_vgpr_count; ++i) {
    const uint32_t vgpr_index = coexecution->active_vgpr_indices[i];
    const uint8_t channels =
        coexecution->active_states[vgpr_index].consumer_channels;
    if (channels == 0) continue;
    loom_amdgpu_matrix_coexecution_consumer_t* consumer = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_allocate_consumer(
        coexecution, &consumer));
    consumer->vgpr_index = vgpr_index;
    consumer->channels = channels;
    consumer->next = block->consumers;
    block->consumers = consumer;
  }
  return iree_ok_status();
}

static void loom_amdgpu_matrix_coexecution_apply_consumers(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_amdgpu_matrix_coexecution_consumer_t* consumers) {
  for (const loom_amdgpu_matrix_coexecution_consumer_t* consumer = consumers;
       consumer != NULL; consumer = consumer->next) {
    loom_amdgpu_matrix_coexecution_active_state_t* active =
        loom_amdgpu_matrix_coexecution_try_active_mutable(coexecution,
                                                          consumer->vgpr_index);
    if (active == NULL) continue;
    for (uint32_t channel = 0;
         channel < LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT; ++channel) {
      if ((consumer->channels & (1u << channel)) == 0) continue;
      active->release_positions[channel] = 0;
      active->required[channel] = 0;
      active->producer_nodes[channel] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }
}

static void loom_amdgpu_matrix_coexecution_merge_active_frontier(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_amdgpu_matrix_coexecution_frontier_node_t* frontier) {
  for (const loom_amdgpu_matrix_coexecution_frontier_node_t* node = frontier;
       node != NULL; node = node->next) {
    loom_amdgpu_matrix_coexecution_active_state_t* active =
        loom_amdgpu_matrix_coexecution_get_active(coexecution,
                                                  node->vgpr_index);
    for (uint32_t channel = 0;
         channel < LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT; ++channel) {
      if (node->remaining[channel] == 0) continue;
      const uint64_t release_position =
          coexecution->current_issue_position + node->remaining[channel];
      const uint8_t required = node->required[channel] != 0
                                   ? node->required[channel]
                                   : node->remaining[channel];
      const uint32_t producer_node = node->required[channel] != 0
                                         ? node->producer_nodes[channel]
                                         : LOOM_LOW_SCHEDULE_NODE_NONE;
      if (release_position > active->release_positions[channel]) {
        active->release_positions[channel] = release_position;
        active->required[channel] = required;
        active->producer_nodes[channel] = producer_node;
      } else if (release_position == active->release_positions[channel] &&
                 active->producer_nodes[channel] != producer_node) {
        active->producer_nodes[channel] = LOOM_LOW_SCHEDULE_NODE_NONE;
      }
    }
  }
}

static iree_status_t loom_amdgpu_matrix_coexecution_merge_outgoing(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    loom_amdgpu_matrix_coexecution_frontier_node_t** inout_frontier,
    bool preserve_provenance, bool* out_changed) {
  *out_changed = false;
  for (loom_amdgpu_matrix_coexecution_frontier_node_t* node = *inout_frontier;
       node != NULL; node = node->next) {
    IREE_ASSERT_LT(node->vgpr_index, coexecution->vgpr_count);
    IREE_ASSERT(coexecution->frontier_lookup[node->vgpr_index] == NULL);
    coexecution->frontier_lookup[node->vgpr_index] = node;
  }
  for (iree_host_size_t i = 0; i < coexecution->active_vgpr_count; ++i) {
    const uint32_t vgpr_index = coexecution->active_vgpr_indices[i];
    const loom_amdgpu_matrix_coexecution_active_state_t* active =
        &coexecution->active_states[vgpr_index];
    uint8_t remaining[LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT] = {0};
    bool has_remaining = false;
    for (uint32_t channel = 0;
         channel < LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT; ++channel) {
      if (active->release_positions[channel] <=
          coexecution->current_issue_position) {
        continue;
      }
      const uint64_t residual = active->release_positions[channel] -
                                coexecution->current_issue_position;
      IREE_ASSERT_LE(residual, coexecution->profile->maximum_issue_distance);
      remaining[channel] = (uint8_t)residual;
      has_remaining = true;
    }
    if (!has_remaining) continue;

    loom_amdgpu_matrix_coexecution_frontier_node_t* node =
        coexecution->frontier_lookup[vgpr_index];
    if (node == NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_matrix_coexecution_allocate_frontier_node(coexecution,
                                                                &node));
      node->vgpr_index = vgpr_index;
      node->next = *inout_frontier;
      *inout_frontier = node;
      coexecution->frontier_lookup[vgpr_index] = node;
      *out_changed = true;
    }
    for (uint32_t channel = 0;
         channel < LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_COUNT; ++channel) {
      if (remaining[channel] > node->remaining[channel]) {
        node->remaining[channel] = remaining[channel];
        if (preserve_provenance) {
          node->required[channel] = active->required[channel];
          node->producer_nodes[channel] = active->producer_nodes[channel];
        }
        *out_changed = true;
      } else if (preserve_provenance && remaining[channel] != 0 &&
                 remaining[channel] == node->remaining[channel] &&
                 node->producer_nodes[channel] !=
                     active->producer_nodes[channel]) {
        node->producer_nodes[channel] = LOOM_LOW_SCHEDULE_NODE_NONE;
      }
    }
  }
  for (loom_amdgpu_matrix_coexecution_frontier_node_t* node = *inout_frontier;
       node != NULL; node = node->next) {
    coexecution->frontier_lookup[node->vgpr_index] = NULL;
  }
  return iree_ok_status();
}

static void loom_amdgpu_matrix_coexecution_worklist_push(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t block_index,
    uint32_t block_count, uint32_t* tail, uint32_t* count) {
  loom_amdgpu_matrix_coexecution_block_t* block =
      &coexecution->blocks[block_index];
  if (iree_any_bit_set(block->flags,
                       LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_QUEUED)) {
    return;
  }
  coexecution->worklist[*tail] = block_index;
  if (++*tail == block_count) *tail = 0;
  ++*count;
  block->flags |= LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_QUEUED;
}

static void loom_amdgpu_matrix_coexecution_note_channel(
    const loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_amdgpu_matrix_coexecution_active_state_t* state,
    loom_amdgpu_matrix_coexecution_channel_t channel,
    uint16_t preceding_issue_count, bool matrix_consumer,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  const uint64_t consumer_position =
      coexecution->current_issue_position + preceding_issue_count;
  if (state == NULL || state->release_positions[channel] <= consumer_position) {
    return;
  }
  const uint64_t residual =
      state->release_positions[channel] - consumer_position;
  IREE_ASSERT_LE(residual, UINT16_MAX);
  if (residual <= inout_match->residual_issue_count) return;
  IREE_ASSERT_LE(residual, state->required[channel]);
  inout_match->producer_node = state->producer_nodes[channel];
  inout_match->required_issue_count = state->required[channel];
  inout_match->observed_issue_count =
      (uint16_t)(state->required[channel] - residual);
  inout_match->residual_issue_count = (uint16_t)residual;
  inout_match->matrix_consumer = matrix_consumer;
}

static void loom_amdgpu_matrix_coexecution_query_range(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint32_t base,
    uint32_t count, loom_amdgpu_matrix_coexecution_channel_t channel,
    uint16_t preceding_issue_count,
    loom_amdgpu_matrix_coexecution_query_flags_t query_flags,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  IREE_ASSERT_LE(base + count, coexecution->vgpr_count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t vgpr_index = base + i;
    const loom_amdgpu_matrix_coexecution_active_state_t* state = NULL;
    if (iree_any_bit_set(
            query_flags,
            LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_RETAIN_CONSUMER)) {
      loom_amdgpu_matrix_coexecution_active_state_t* mutable_state =
          loom_amdgpu_matrix_coexecution_get_active(coexecution, vgpr_index);
      mutable_state->consumer_channels |= (uint8_t)(1u << channel);
      state = mutable_state;
    } else {
      state =
          loom_amdgpu_matrix_coexecution_try_active(coexecution, vgpr_index);
    }
    const bool matrix_consumer = iree_any_bit_set(
        query_flags, LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_MATRIX_CONSUMER);
    loom_amdgpu_matrix_coexecution_note_channel(coexecution, state, channel,
                                                preceding_issue_count,
                                                matrix_consumer, inout_match);
  }
}

static void loom_amdgpu_matrix_coexecution_query_assignment(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_matrix_coexecution_channel_t channel,
    uint16_t preceding_issue_count,
    loom_amdgpu_matrix_coexecution_query_flags_t query_flags,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  if (!loom_low_allocation_assignment_is_physical_register_class(
          assignment, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return;
  }
  loom_amdgpu_matrix_coexecution_query_range(
      coexecution, assignment->location_base, assignment->location_count,
      channel, preceding_issue_count, query_flags, inout_match);
}

static void loom_amdgpu_matrix_coexecution_query_node(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_schedule_node_t* node, bool retain_consumers,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  const loom_amdgpu_matrix_coexecution_query_flags_t query_flags =
      retain_consumers
          ? LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_RETAIN_CONSUMER
          : 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    loom_amdgpu_matrix_coexecution_query_assignment(
        coexecution,
        loom_low_allocation_assignment_for_value_ordinal(
            coexecution->allocation, operand_ordinals[i], NULL),
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT, 0, query_flags,
        inout_match);
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            coexecution->allocation, result_ordinals[i], NULL);
    loom_amdgpu_matrix_coexecution_query_assignment(
        coexecution, assignment,
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT, 0, query_flags,
        inout_match);
    loom_amdgpu_matrix_coexecution_query_assignment(
        coexecution, assignment,
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_SOURCE, 0, query_flags,
        inout_match);
  }
}

static uint16_t loom_amdgpu_matrix_coexecution_query_moves(
    loom_amdgpu_matrix_coexecution_t* coexecution, loom_low_move_range_t moves,
    bool retain_consumers,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  const loom_amdgpu_matrix_coexecution_query_flags_t query_flags =
      retain_consumers
          ? LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_RETAIN_CONSUMER
          : 0;
  uint16_t vector_issue_count = 0;
  for (iree_host_size_t i = 0; i < moves.count; ++i) {
    const loom_low_move_t* move =
        &coexecution->allocation->moves[moves.start + i];
    if (move->destination.descriptor_reg_class_id !=
        LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      continue;
    }
    if (move->source.descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      loom_amdgpu_matrix_coexecution_query_range(
          coexecution, move->source.location, 1,
          LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT,
          vector_issue_count, query_flags, inout_match);
    }
    loom_amdgpu_matrix_coexecution_query_range(
        coexecution, move->destination.location, 1,
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_RESULT, vector_issue_count,
        query_flags, inout_match);
    loom_amdgpu_matrix_coexecution_query_range(
        coexecution, move->destination.location, 1,
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_VALU_SOURCE, vector_issue_count,
        query_flags, inout_match);
    if (vector_issue_count < coexecution->profile->maximum_issue_distance) {
      ++vector_issue_count;
    }
  }
  return vector_issue_count;
}

static void loom_amdgpu_matrix_coexecution_query_matrix(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_matrix_coexecution_source_t* source,
    bool retain_consumers,
    loom_amdgpu_matrix_coexecution_match_t* inout_match) {
  const loom_amdgpu_matrix_coexecution_query_flags_t query_flags =
      LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_MATRIX_CONSUMER |
      (retain_consumers
           ? LOOM_AMDGPU_MATRIX_COEXECUTION_QUERY_FLAG_RETAIN_CONSUMER
           : 0);
  for (uint8_t i = 0; i < source->source_operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(
            coexecution->allocation, packet,
            (uint16_t)(source->source_operand_start + i));
    loom_amdgpu_matrix_coexecution_query_assignment(
        coexecution, assignment,
        LOOM_AMDGPU_MATRIX_COEXECUTION_CHANNEL_MATRIX_RESULT, 0, query_flags,
        inout_match);
  }
}

static void loom_amdgpu_matrix_coexecution_inspect_packet_impl(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural_info,
    const loom_amdgpu_matrix_coexecution_source_t* source,
    bool retain_consumers, loom_amdgpu_matrix_coexecution_match_t* inout_match,
    uint16_t* out_vector_issue_count) {
  *out_vector_issue_count = 0;
  if (packet->descriptor != NULL) {
    IREE_ASSERT(structural_info == NULL);
    const loom_amdgpu_descriptor_traits_t traits =
        loom_amdgpu_descriptor_traits(coexecution->descriptor_set,
                                      packet->descriptor);
    if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE)) {
      *out_vector_issue_count = 1;
    }
    if (source != NULL) {
      loom_amdgpu_matrix_coexecution_query_matrix(
          coexecution, packet, source, retain_consumers, inout_match);
      return;
    }
    const bool is_ordinary_vector =
        iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU) &&
        !iree_any_bit_set(packet->descriptor->instruction_class_flags,
                          LOOM_LOW_INSTRUCTION_CLASS_FLAG_WMMA |
                              LOOM_LOW_INSTRUCTION_CLASS_FLAG_SWMMAC |
                              LOOM_LOW_INSTRUCTION_CLASS_FLAG_LDSDMA);
    if (is_ordinary_vector) {
      loom_amdgpu_matrix_coexecution_query_node(coexecution, packet->node,
                                                retain_consumers, inout_match);
    }
    return;
  }

  IREE_ASSERT(structural_info != NULL);
  if (structural_info->vector_alu_instruction_count != 0) {
    loom_amdgpu_matrix_coexecution_query_node(coexecution, packet->node,
                                              retain_consumers, inout_match);
  }
  const uint16_t move_issue_count = loom_amdgpu_matrix_coexecution_query_moves(
      coexecution, structural_info->moves, retain_consumers, inout_match);
  *out_vector_issue_count = loom_amdgpu_matrix_coexecution_saturate_issue_count(
      coexecution,
      structural_info->vector_alu_instruction_count + move_issue_count);
}

iree_status_t loom_amdgpu_matrix_coexecution_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_matrix_coexecution_profile_t profile,
    iree_arena_allocator_t* arena,
    loom_amdgpu_matrix_coexecution_t** out_coexecution) {
  *out_coexecution = NULL;
  const loom_amdgpu_matrix_coexecution_profile_info_t* profile_model =
      loom_amdgpu_target_info_matrix_coexecution_profile(profile);
  if (profile_model->releases == NULL) return iree_ok_status();
  const iree_host_size_t source_capacity =
      schedule->matrix_coexecution_source_use_count;
  if (source_capacity == 0) return iree_ok_status();
  const uint32_t vgpr_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  IREE_ASSERT_NE(vgpr_count, 0);
  IREE_ASSERT_LE(schedule->block_count, UINT16_MAX);

  loom_amdgpu_matrix_coexecution_t* coexecution = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*coexecution), (void**)&coexecution));
  *coexecution = (loom_amdgpu_matrix_coexecution_t){
      .schedule = schedule,
      .allocation = allocation,
      .descriptor_set = schedule->target.descriptor_set,
      .profile = profile_model,
      .arena = arena,
      .source_capacity = source_capacity,
      .vgpr_count = vgpr_count,
      .active_block_index = UINT16_MAX,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source_capacity, sizeof(*coexecution->sources),
      (void**)&coexecution->sources));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, vgpr_count, sizeof(*coexecution->active_states),
      (void**)&coexecution->active_states));
  memset(coexecution->active_states, 0,
         (iree_host_size_t)vgpr_count * sizeof(*coexecution->active_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, vgpr_count, sizeof(*coexecution->active_vgpr_indices),
      (void**)&coexecution->active_vgpr_indices));
  if (schedule->block_count > 1) {
    IREE_ASSERT(schedule->cfg_graph.blocks != NULL);
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, vgpr_count, sizeof(*coexecution->frontier_lookup),
        (void**)&coexecution->frontier_lookup));
    memset(
        coexecution->frontier_lookup, 0,
        (iree_host_size_t)vgpr_count * sizeof(*coexecution->frontier_lookup));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count, sizeof(*coexecution->blocks),
        (void**)&coexecution->blocks));
    memset(coexecution->blocks, 0,
           schedule->block_count * sizeof(*coexecution->blocks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, schedule->block_count, sizeof(*coexecution->worklist),
        (void**)&coexecution->worklist));
  }
  *out_coexecution = coexecution;
  return iree_ok_status();
}

void loom_amdgpu_matrix_coexecution_begin_static_block(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t block_index) {
  IREE_ASSERT(coexecution->active_block_index == UINT16_MAX);
  IREE_ASSERT_LT(block_index, coexecution->schedule->block_count);
  loom_amdgpu_matrix_coexecution_reset_active(coexecution);
  coexecution->active_block_index = block_index;
}

static void loom_amdgpu_matrix_coexecution_advance_static(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t issue_count) {
  IREE_ASSERT(coexecution->blocks != NULL);
  loom_amdgpu_matrix_coexecution_advance(coexecution, issue_count);
  loom_amdgpu_matrix_coexecution_block_t* block =
      &coexecution->blocks[coexecution->active_block_index];
  block->vector_issue_count = loom_amdgpu_matrix_coexecution_saturating_add(
      block->vector_issue_count, issue_count,
      coexecution->profile->maximum_issue_distance);
}

void loom_amdgpu_matrix_coexecution_commit_static_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural_info) {
  IREE_ASSERT_EQ(packet->node->block_index, coexecution->active_block_index);
  const loom_amdgpu_matrix_coexecution_source_t* source = NULL;
  if (packet->descriptor != NULL &&
      loom_amdgpu_matrix_coexecution_descriptor_is_source(coexecution,
                                                          packet->descriptor)) {
    source = loom_amdgpu_matrix_coexecution_append_source(coexecution, packet);
  }
  if (coexecution->blocks == NULL) return;

  loom_amdgpu_matrix_coexecution_match_t match = {
      .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  uint16_t vector_issue_count = 0;
  loom_amdgpu_matrix_coexecution_inspect_packet_impl(
      coexecution, packet, structural_info, source,
      /*retain_consumers=*/true, &match, &vector_issue_count);
  loom_amdgpu_matrix_coexecution_advance_static(coexecution,
                                                match.residual_issue_count);
  loom_amdgpu_matrix_coexecution_advance_static(coexecution,
                                                vector_issue_count);
  if (source != NULL) {
    loom_amdgpu_matrix_coexecution_publish_source(coexecution, packet, source);
  }
}

void loom_amdgpu_matrix_coexecution_commit_static_vopd_pair(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* first_packet,
    const loom_low_packet_view_t* second_packet) {
  IREE_ASSERT(first_packet->descriptor != NULL);
  IREE_ASSERT(second_packet->descriptor != NULL);
  IREE_ASSERT(!loom_amdgpu_matrix_coexecution_descriptor_is_source(
      coexecution, first_packet->descriptor));
  IREE_ASSERT(!loom_amdgpu_matrix_coexecution_descriptor_is_source(
      coexecution, second_packet->descriptor));
  if (coexecution->blocks == NULL) return;

  loom_amdgpu_matrix_coexecution_match_t match = {
      .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  uint16_t first_issue_count = 0;
  uint16_t second_issue_count = 0;
  loom_amdgpu_matrix_coexecution_inspect_packet_impl(
      coexecution, first_packet, /*structural_info=*/NULL, /*source=*/NULL,
      /*retain_consumers=*/true, &match, &first_issue_count);
  loom_amdgpu_matrix_coexecution_inspect_packet_impl(
      coexecution, second_packet, /*structural_info=*/NULL, /*source=*/NULL,
      /*retain_consumers=*/true, &match, &second_issue_count);
  IREE_ASSERT_EQ(first_issue_count, 1);
  IREE_ASSERT_EQ(second_issue_count, 1);
  loom_amdgpu_matrix_coexecution_advance_static(coexecution,
                                                match.residual_issue_count);
  loom_amdgpu_matrix_coexecution_advance_static(coexecution, 1);
}

iree_status_t loom_amdgpu_matrix_coexecution_end_static_block(
    loom_amdgpu_matrix_coexecution_t* coexecution) {
  IREE_ASSERT_LT(coexecution->active_block_index,
                 coexecution->schedule->block_count);
  if (coexecution->blocks != NULL) {
    loom_amdgpu_matrix_coexecution_block_t* block =
        &coexecution->blocks[coexecution->active_block_index];
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_merge_outgoing(
        coexecution, &block->static_outgoing,
        /*preserve_provenance=*/false, &changed));
    (void)changed;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_matrix_coexecution_retain_consumers(coexecution, block));
  }
  coexecution->active_block_index = UINT16_MAX;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_matrix_coexecution_finalize_static(
    loom_amdgpu_matrix_coexecution_t* coexecution) {
  IREE_ASSERT_EQ(coexecution->source_count, coexecution->source_capacity);
  if (coexecution->blocks == NULL) return iree_ok_status();
  const loom_cfg_graph_t* graph = &coexecution->schedule->cfg_graph;
  const uint32_t block_count = (uint32_t)coexecution->schedule->block_count;
  uint32_t head = 0;
  uint32_t tail = 0;
  uint32_t count = 0;
  for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
    if (loom_cfg_graph_block_is_reachable(graph, block_index)) {
      loom_amdgpu_matrix_coexecution_worklist_push(coexecution, block_index,
                                                   block_count, &tail, &count);
    }
  }
  while (count != 0) {
    const uint16_t block_index = coexecution->worklist[head];
    if (++head == block_count) head = 0;
    --count;
    loom_amdgpu_matrix_coexecution_block_t* block =
        &coexecution->blocks[block_index];
    block->flags &=
        (loom_amdgpu_matrix_coexecution_block_flags_t)~LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_QUEUED;

    loom_amdgpu_matrix_coexecution_reset_active(coexecution);
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const uint16_t predecessor_index = predecessors.values[i];
      if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
        continue;
      }
      loom_amdgpu_matrix_coexecution_merge_active_frontier(
          coexecution, coexecution->blocks[predecessor_index].static_outgoing);
    }
    loom_amdgpu_matrix_coexecution_apply_consumers(coexecution,
                                                   block->consumers);
    loom_amdgpu_matrix_coexecution_advance(coexecution,
                                           block->vector_issue_count);
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_merge_outgoing(
        coexecution, &block->static_outgoing,
        /*preserve_provenance=*/false, &changed));
    if (!changed) continue;
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t successor_index = successors.values[i];
      if (loom_cfg_graph_block_is_reachable(graph, successor_index)) {
        loom_amdgpu_matrix_coexecution_worklist_push(
            coexecution, successor_index, block_count, &tail, &count);
      }
    }
  }
  loom_amdgpu_matrix_coexecution_reset_active(coexecution);
  return iree_ok_status();
}

void loom_amdgpu_matrix_coexecution_begin_block(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t block_index) {
  IREE_ASSERT(coexecution->active_block_index == UINT16_MAX);
  IREE_ASSERT_LT(block_index, coexecution->schedule->block_count);
  loom_amdgpu_matrix_coexecution_reset_active(coexecution);
  coexecution->active_block_index = block_index;
  if (coexecution->blocks == NULL) return;
  const loom_cfg_graph_t* graph = &coexecution->schedule->cfg_graph;
  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) continue;
    const loom_amdgpu_matrix_coexecution_block_t* predecessor =
        &coexecution->blocks[predecessor_index];
    const bool resolved = iree_any_bit_set(
        predecessor->flags, LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_RESOLVED);
    loom_amdgpu_matrix_coexecution_merge_active_frontier(
        coexecution, resolved ? predecessor->resolved_outgoing
                              : predecessor->static_outgoing);
  }
}

void loom_amdgpu_matrix_coexecution_inspect_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural_info,
    loom_amdgpu_matrix_coexecution_match_t* inout_match,
    uint16_t* out_vector_issue_count) {
  const loom_amdgpu_matrix_coexecution_source_t* source =
      loom_amdgpu_matrix_coexecution_current_source(coexecution, packet);
  loom_amdgpu_matrix_coexecution_inspect_packet_impl(
      coexecution, packet, structural_info, source,
      /*retain_consumers=*/false, inout_match, out_vector_issue_count);
}

void loom_amdgpu_matrix_coexecution_advance(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t issue_count) {
  coexecution->current_issue_position += issue_count;
}

void loom_amdgpu_matrix_coexecution_commit_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet, uint16_t vector_issue_count) {
  loom_amdgpu_matrix_coexecution_advance(coexecution, vector_issue_count);
  const loom_amdgpu_matrix_coexecution_source_t* source =
      loom_amdgpu_matrix_coexecution_current_source(coexecution, packet);
  if (source == NULL) return;
  loom_amdgpu_matrix_coexecution_publish_source(coexecution, packet, source);
  ++coexecution->source_cursor;
}

void loom_amdgpu_matrix_coexecution_commit_vopd_pair(
    loom_amdgpu_matrix_coexecution_t* coexecution) {
  loom_amdgpu_matrix_coexecution_advance(coexecution, 1);
}

iree_status_t loom_amdgpu_matrix_coexecution_end_block(
    loom_amdgpu_matrix_coexecution_t* coexecution) {
  IREE_ASSERT_LT(coexecution->active_block_index,
                 coexecution->schedule->block_count);
  if (coexecution->blocks != NULL) {
    loom_amdgpu_matrix_coexecution_block_t* block =
        &coexecution->blocks[coexecution->active_block_index];
    IREE_ASSERT(!iree_any_bit_set(
        block->flags, LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_RESOLVED));
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_merge_outgoing(
        coexecution, &block->resolved_outgoing,
        /*preserve_provenance=*/true, &changed));
    (void)changed;
    block->flags |= LOOM_AMDGPU_MATRIX_COEXECUTION_BLOCK_FLAG_RESOLVED;
  }
  if (coexecution->active_block_index + 1 ==
      coexecution->schedule->block_count) {
    IREE_ASSERT_EQ(coexecution->source_cursor, coexecution->source_count);
  }
  coexecution->active_block_index = UINT16_MAX;
  return iree_ok_status();
}
