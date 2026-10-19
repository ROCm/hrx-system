// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/sgpr_read_wait.h"

#include <string.h>

#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/util/cfg_dataflow.h"
#include "loom/util/segmented_storage.h"

#define LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT 16u
#define LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT 64u
#define LOOM_AMDGPU_SGPR_READ_WAIT_VCC (UINT64_C(3) << (106 - 64))

typedef struct loom_amdgpu_sgpr_read_wait_destinations_t {
  // Pending scalar-written destinations by physical DWORD.
  uint64_t scalar[2];
  // Pending vector-written destinations by physical DWORD.
  uint64_t vector[2];
} loom_amdgpu_sgpr_read_wait_destinations_t;

typedef struct loom_amdgpu_sgpr_read_wait_block_t {
  // Pair-expanded VALU reads generated in this block.
  uint64_t tracked[2];
  // Pair-expanded VALU reads reaching this block through any predecessor.
  uint64_t incoming_tracked[2];
  // Locally generated pending writes; replaced by exact output after resolve.
  loom_amdgpu_sgpr_read_wait_destinations_t generated;
  // Incoming writes consumed or completed in this block.
  loom_amdgpu_sgpr_read_wait_destinations_t killed;
  // Conservatively joined predecessor write obligations.
  loom_amdgpu_sgpr_read_wait_destinations_t incoming;
  // First retained native event in this block.
  iree_host_size_t event_start;
  // Number of retained native events in this block.
  iree_host_size_t event_count;
  // The generated row now contains resolved outgoing state.
  bool resolved;
} loom_amdgpu_sgpr_read_wait_block_t;

struct loom_amdgpu_sgpr_read_wait_t {
  // Final schedule owning packet coordinates and the immutable CFG.
  const loom_low_schedule_table_t* schedule;
  // Scoped arena owning all analysis storage.
  iree_arena_allocator_t* arena;
  // Fixed-size block summaries in pool-sized segments.
  loom_segmented_storage_t blocks;
  // Relevant native instruction facts in pool-sized segments.
  loom_segmented_storage_t events;
  // Number of populated native instruction facts.
  iree_host_size_t event_count;
};

static loom_amdgpu_sgpr_read_wait_block_t* loom_amdgpu_sgpr_read_wait_block(
    loom_amdgpu_sgpr_read_wait_t* state, uint32_t index) {
  loom_amdgpu_sgpr_read_wait_block_t* segment =
      (loom_amdgpu_sgpr_read_wait_block_t*)loom_segmented_storage_segment(
          &state->blocks,
          index / LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT);
  return &segment[index % LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT];
}

static const loom_amdgpu_sgpr_read_wait_event_t*
loom_amdgpu_sgpr_read_wait_event(const loom_amdgpu_sgpr_read_wait_t* state,
                                 iree_host_size_t index) {
  const loom_amdgpu_sgpr_read_wait_event_t* segment =
      (const loom_amdgpu_sgpr_read_wait_event_t*)
          loom_segmented_storage_const_segment(
              &state->events,
              (uint32_t)(index /
                         LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT));
  return &segment[index % LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT];
}

static uint64_t loom_amdgpu_sgpr_read_wait_pairs(uint64_t reads) {
  const uint64_t even = UINT64_C(0x5555555555555555);
  return reads | ((reads & even) << 1) | ((reads & ~even) >> 1);
}

iree_status_t loom_amdgpu_sgpr_read_wait_allocate(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_sgpr_read_wait_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sgpr_read_wait_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  *state = (loom_amdgpu_sgpr_read_wait_t){
      .schedule = schedule,
      .arena = arena,
  };
  loom_segmented_storage_initialize(
      LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT *
          sizeof(loom_amdgpu_sgpr_read_wait_block_t),
      iree_alignof(loom_amdgpu_sgpr_read_wait_block_t), &state->blocks);
  loom_segmented_storage_initialize(
      LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT *
          sizeof(loom_amdgpu_sgpr_read_wait_event_t),
      iree_alignof(loom_amdgpu_sgpr_read_wait_event_t), &state->events);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t index = 0;
       index < schedule->block_count && iree_status_is_ok(status);
       index += LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT) {
    void* segment = NULL;
    status = loom_segmented_storage_append(&state->blocks, arena, &segment);
    if (iree_status_is_ok(status)) {
      memset(segment, 0,
             LOOM_AMDGPU_SGPR_READ_WAIT_BLOCKS_PER_SEGMENT *
                 sizeof(loom_amdgpu_sgpr_read_wait_block_t));
    }
  }
  if (iree_status_is_ok(status)) {
    *out_state = state;
  }
  return status;
}

iree_status_t loom_amdgpu_sgpr_read_wait_collect(
    loom_amdgpu_sgpr_read_wait_t* state, uint16_t block_index,
    const loom_amdgpu_sgpr_read_wait_event_t* event) {
  if ((event->reads[0] | event->reads[1] | event->writes[0] | event->writes[1] |
       event->drain_fields) == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t offset =
      state->event_count % LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT;
  void* segment = NULL;
  if (offset == 0) {
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&state->events, state->arena, &segment));
  } else {
    segment = loom_segmented_storage_segment(
        &state->events,
        (uint32_t)(state->event_count /
                   LOOM_AMDGPU_SGPR_READ_WAIT_EVENTS_PER_SEGMENT));
  }
  ((loom_amdgpu_sgpr_read_wait_event_t*)segment)[offset] = *event;
  loom_amdgpu_sgpr_read_wait_block_t* block =
      loom_amdgpu_sgpr_read_wait_block(state, block_index);
  if (block->event_count == 0) {
    block->event_start = state->event_count;
  }
  ++block->event_count;
  ++state->event_count;
  if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR) {
    for (unsigned word = 0; word < 2; ++word) {
      block->tracked[word] |=
          loom_amdgpu_sgpr_read_wait_pairs(event->reads[word]);
    }
  }
  return iree_ok_status();
}

static void loom_amdgpu_sgpr_read_wait_set_range(uint64_t bits[2],
                                                 uint16_t register_class,
                                                 uint32_t base, uint32_t count,
                                                 bool scalar_write) {
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VCC) {
    base = 106;
    count = scalar_write ? 1 : 2;
  } else if (register_class != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return;
  }
  while (count != 0) {
    const unsigned offset = base % 64;
    const unsigned length = iree_min(count, 64 - offset);
    bits[base / 64] |= (UINT64_MAX >> (64 - length)) << offset;
    base += length;
    count -= length;
  }
}

static void loom_amdgpu_sgpr_read_wait_set_assignment(
    uint64_t bits[2], const loom_low_allocation_assignment_t* assignment,
    bool scalar_write) {
  if (assignment != NULL &&
      assignment->location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_amdgpu_sgpr_read_wait_set_range(
        bits, assignment->descriptor_reg_class_id, assignment->location_base,
        assignment->location_count, scalar_write);
  }
}

static void loom_amdgpu_sgpr_read_wait_packet_operands(
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_sgpr_read_wait_event_t* event) {
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    loom_amdgpu_sgpr_read_wait_set_assignment(
        event->reads, loom_low_packet_operand_assignment(allocation, packet, i),
        false);
  }
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    loom_amdgpu_sgpr_read_wait_set_assignment(
        event->writes, loom_low_packet_result_assignment(allocation, packet, i),
        event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR);
  }
}

iree_status_t loom_amdgpu_sgpr_read_wait_collect_packet(
    loom_amdgpu_sgpr_read_wait_t* state,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    const loom_low_packet_view_t* second_packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits) {
  loom_amdgpu_sgpr_read_wait_event_t event = {
      .packet_index = packet->packet_index,
  };
  if (packet->descriptor == NULL) {
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < structural->moves.count && iree_status_is_ok(status); ++i) {
      const loom_low_move_t* move =
          &allocation->moves[structural->moves.start + i];
      event = (loom_amdgpu_sgpr_read_wait_event_t){
          .packet_index = packet->packet_index,
          .instruction_offset = (uint32_t)i,
          .kind = move->destination.descriptor_reg_class_id ==
                              LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
                          move->source.descriptor_reg_class_id ==
                              LOOM_AMDGPU_REG_CLASS_ID_VGPR
                      ? LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR
                      : LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR,
      };
      if (move->source.location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
        loom_amdgpu_sgpr_read_wait_set_range(
            event.reads, move->source.descriptor_reg_class_id,
            move->source.location, 1, false);
      }
      if (move->destination.location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
        loom_amdgpu_sgpr_read_wait_set_range(
            event.writes, move->destination.descriptor_reg_class_id,
            move->destination.location, 1,
            event.kind == LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR);
      }
      status = loom_amdgpu_sgpr_read_wait_collect(
          state, (uint16_t)packet->node->block_index, &event);
    }
    return status;
  }
  // SMEM and non-FLAT VMEM interlock scalar destinations. GLOBAL and SCRATCH
  // use the FLAT path, so their memory issue alone supplies no such progress.
  const uint16_t format =
      loom_low_descriptor_layout(state->schedule->target.descriptor_set,
                                 packet->descriptor)
          ->encoding_format_id;
  if ((traits & LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY) ||
      (loom_amdgpu_encoding_format_flags(format) &
       LOOM_AMDGPU_ENCODING_FORMAT_FLAG_BUFFER_ADDRESS) ||
      format == LOOM_AMDGPU_ENCODING_FORMAT_VIMAGE ||
      format == LOOM_AMDGPU_ENCODING_FORMAT_VSAMPLE) {
    event.kind = LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN;
    event.drain_fields = LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST |
                         LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST |
                         LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC;
  } else if (traits & (LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU |
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    event.kind = (traits & LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)
                     ? LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR
                     : LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR;
    loom_amdgpu_sgpr_read_wait_packet_operands(allocation, packet, &event);
    if (second_packet != NULL) {
      loom_amdgpu_sgpr_read_wait_packet_operands(allocation, second_packet,
                                                 &event);
    }
  }
  return loom_amdgpu_sgpr_read_wait_collect(
      state, (uint16_t)packet->node->block_index, &event);
}

iree_status_t loom_amdgpu_sgpr_read_wait_collect_wait(
    loom_amdgpu_sgpr_read_wait_t* state, const loom_low_packet_view_t* packet,
    uint16_t bound) {
  const loom_amdgpu_sgpr_read_wait_event_t event = {
      .packet_index = packet->packet_index,
      .kind = LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN,
      .drain_fields =
          ((bound & 0x0001) == 0 ? LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST : 0) |
          ((bound & 0x0e00) == 0 ? LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST : 0) |
          ((bound & 0x0002) == 0 ? LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC : 0),
  };
  return loom_amdgpu_sgpr_read_wait_collect(
      state, (uint16_t)packet->node->block_index, &event);
}

static iree_status_t loom_amdgpu_sgpr_read_wait_transfer_pairs(
    void* user_data, uint16_t source_index, uint16_t target_index,
    bool* out_changed) {
  loom_amdgpu_sgpr_read_wait_t* state = user_data;
  const loom_amdgpu_sgpr_read_wait_block_t* source =
      loom_amdgpu_sgpr_read_wait_block(state, source_index);
  loom_amdgpu_sgpr_read_wait_block_t* target =
      loom_amdgpu_sgpr_read_wait_block(state, target_index);
  *out_changed = false;
  for (unsigned word = 0; word < 2; ++word) {
    const uint64_t incoming =
        source->tracked[word] | source->incoming_tracked[word];
    const uint64_t added = incoming & ~target->incoming_tracked[word];
    *out_changed |= (added & ~target->tracked[word]) != 0;
    target->incoming_tracked[word] |= incoming;
  }
  return iree_ok_status();
}

static void loom_amdgpu_sgpr_read_wait_drain(
    loom_amdgpu_sgpr_read_wait_destinations_t* destinations, uint8_t fields) {
  if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST) {
    destinations->scalar[0] = destinations->scalar[1] = 0;
  }
  if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST) {
    destinations->vector[0] = 0;
    destinations->vector[1] &= LOOM_AMDGPU_SGPR_READ_WAIT_VCC;
  }
  if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC) {
    destinations->vector[1] &= ~LOOM_AMDGPU_SGPR_READ_WAIT_VCC;
  }
}

static void loom_amdgpu_sgpr_read_wait_summarize(
    loom_amdgpu_sgpr_read_wait_t* state,
    loom_amdgpu_sgpr_read_wait_block_t* block) {
  uint64_t tracked[2] = {block->incoming_tracked[0],
                         block->incoming_tracked[1]};
  for (iree_host_size_t index = 0; index < block->event_count; ++index) {
    const loom_amdgpu_sgpr_read_wait_event_t* event =
        loom_amdgpu_sgpr_read_wait_event(state, block->event_start + index);
    if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN) {
      loom_amdgpu_sgpr_read_wait_drain(&block->generated, event->drain_fields);
      loom_amdgpu_sgpr_read_wait_destinations_t survivors = {
          .scalar = {UINT64_MAX, UINT64_MAX},
          .vector = {UINT64_MAX, UINT64_MAX},
      };
      loom_amdgpu_sgpr_read_wait_drain(&survivors, event->drain_fields);
      for (unsigned word = 0; word < 2; ++word) {
        block->killed.scalar[word] |= ~survivors.scalar[word];
        block->killed.vector[word] |= ~survivors.vector[word];
      }
      continue;
    }
    for (unsigned word = 0; word < 2; ++word) {
      // A consumed pending destination is guarded during resolution. The
      // summary cannot assume that this conditional wait also cleared other
      // destinations; their obligations remain conservative across edges.
      block->generated.scalar[word] &= ~event->reads[word];
      block->generated.vector[word] &= ~event->reads[word];
      block->killed.scalar[word] |= event->reads[word];
      block->killed.vector[word] |= event->reads[word];
      if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR) {
        tracked[word] |= loom_amdgpu_sgpr_read_wait_pairs(event->reads[word]);
        block->generated.vector[word] |= event->writes[word] & tracked[word];
      } else {
        block->generated.scalar[word] |= event->writes[word] & tracked[word];
      }
    }
  }
}

static loom_amdgpu_sgpr_read_wait_destinations_t
loom_amdgpu_sgpr_read_wait_output(
    const loom_amdgpu_sgpr_read_wait_block_t* block) {
  loom_amdgpu_sgpr_read_wait_destinations_t output = block->generated;
  if (!block->resolved) {
    for (unsigned word = 0; word < 2; ++word) {
      output.scalar[word] |=
          block->incoming.scalar[word] & ~block->killed.scalar[word];
      output.vector[word] |=
          block->incoming.vector[word] & ~block->killed.vector[word];
    }
  }
  return output;
}

static iree_status_t loom_amdgpu_sgpr_read_wait_transfer_writes(
    void* user_data, uint16_t source_index, uint16_t target_index,
    bool* out_changed) {
  loom_amdgpu_sgpr_read_wait_t* state = user_data;
  const loom_amdgpu_sgpr_read_wait_destinations_t output =
      loom_amdgpu_sgpr_read_wait_output(
          loom_amdgpu_sgpr_read_wait_block(state, source_index));
  loom_amdgpu_sgpr_read_wait_block_t* target =
      loom_amdgpu_sgpr_read_wait_block(state, target_index);
  *out_changed = false;
  for (unsigned word = 0; word < 2; ++word) {
    const uint64_t added_scalar =
        output.scalar[word] & ~target->incoming.scalar[word];
    const uint64_t added_vector =
        output.vector[word] & ~target->incoming.vector[word];
    *out_changed |= (added_scalar & ~(target->killed.scalar[word] |
                                      target->generated.scalar[word])) != 0 ||
                    (added_vector & ~(target->killed.vector[word] |
                                      target->generated.vector[word])) != 0;
    target->incoming.scalar[word] |= output.scalar[word];
    target->incoming.vector[word] |= output.vector[word];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sgpr_read_wait_resolve_block(
    loom_amdgpu_sgpr_read_wait_t* state, uint16_t block_index,
    loom_amdgpu_sgpr_read_wait_emit_fn_t emit, void* user_data) {
  loom_amdgpu_sgpr_read_wait_block_t* block =
      loom_amdgpu_sgpr_read_wait_block(state, block_index);
  loom_amdgpu_sgpr_read_wait_destinations_t pending = {0};
  const loom_cfg_block_index_span_t predecessors =
      state->schedule->cfg_graph.edge_count != 0
          ? loom_cfg_graph_predecessors(&state->schedule->cfg_graph,
                                        block_index)
          : (loom_cfg_block_index_span_t){0};
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    if (!state->schedule->cfg_graph.blocks[predecessors.values[i]].reachable) {
      continue;
    }
    const loom_amdgpu_sgpr_read_wait_destinations_t output =
        loom_amdgpu_sgpr_read_wait_output(
            loom_amdgpu_sgpr_read_wait_block(state, predecessors.values[i]));
    for (unsigned word = 0; word < 2; ++word) {
      pending.scalar[word] |= output.scalar[word];
      pending.vector[word] |= output.vector[word];
    }
  }
  uint64_t tracked[2] = {block->incoming_tracked[0],
                         block->incoming_tracked[1]};
  iree_status_t status = iree_ok_status();
  // Local writes retain an exact producer; incoming obligations may join
  // several predecessors and intentionally have no single producer node.
  uint32_t origins[2][128];
  memset(origins, 0xff, sizeof(origins));
  for (iree_host_size_t index = 0;
       index < block->event_count && iree_status_is_ok(status); ++index) {
    const loom_amdgpu_sgpr_read_wait_event_t* event =
        loom_amdgpu_sgpr_read_wait_event(state, block->event_start + index);
    if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN) {
      loom_amdgpu_sgpr_read_wait_drain(&pending, event->drain_fields);
      continue;
    }
    uint8_t fields = 0;
    uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    for (unsigned word = 0; word < 2; ++word) {
      const uint64_t scalar_match = event->reads[word] & pending.scalar[word];
      if (scalar_match) {
        fields |= LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST;
        producer_node =
            origins[0][word * 64 +
                       iree_math_count_trailing_zeros_u64(scalar_match)];
      }
      if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR) {
        const uint64_t matched = event->reads[word] & pending.vector[word];
        const uint64_t vcc = word == 1 ? LOOM_AMDGPU_SGPR_READ_WAIT_VCC : 0;
        if (matched & vcc) {
          fields |= LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC;
        }
        if (matched & ~vcc) {
          fields |= LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST;
        }
        if (matched && producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
          producer_node =
              origins[1]
                     [word * 64 + iree_math_count_trailing_zeros_u64(matched)];
        }
      } else {
        pending.vector[word] &= ~event->reads[word];
      }
    }
    if (fields != 0) {
      uint16_t immediate = 0xffff;
      if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST) {
        immediate &= ~0x0001;
      }
      if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST) {
        immediate &= ~0x0e00;
      }
      if (fields & LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC) {
        immediate &= ~0x0002;
      }
      const loom_low_packet_view_t packet =
          loom_low_packet_at(state->schedule, event->packet_index);
      const loom_amdgpu_wait_state_t wait_state = {
          .reason = LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY,
          .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU,
          .block_index = block_index,
          .node_index = packet.node_index,
          .scheduled_ordinal = packet.node->scheduled_ordinal,
          .instruction_offset = event->instruction_offset,
          .producer_node = producer_node,
          .consumer_node = packet.node_index,
          .required_cycle_count = 1,
          .cycle_count = 1,
          .immediate = immediate,
      };
      status = emit(user_data, &wait_state);
      loom_amdgpu_sgpr_read_wait_drain(&pending, fields);
    }
    for (unsigned word = 0; word < 2; ++word) {
      if (event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR) {
        tracked[word] |= loom_amdgpu_sgpr_read_wait_pairs(event->reads[word]);
        pending.vector[word] |= event->writes[word] & tracked[word];
      } else {
        pending.scalar[word] |= event->writes[word] & tracked[word];
      }
      uint64_t writes = event->writes[word] & tracked[word];
      const unsigned writer =
          event->kind == LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR ? 1 : 0;
      while (writes != 0) {
        const unsigned bit = iree_math_count_trailing_zeros_u64(writes);
        origins[writer][word * 64 + bit] =
            state->schedule->scheduled_node_indices[event->packet_index];
        writes &= writes - 1;
      }
    }
  }
  block->generated = pending;
  block->resolved = true;
  return status;
}

iree_status_t loom_amdgpu_sgpr_read_wait_resolve(
    loom_amdgpu_sgpr_read_wait_t* state,
    loom_amdgpu_sgpr_read_wait_emit_fn_t emit, void* user_data) {
  if (state->event_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_dataflow_solve(
      &state->schedule->cfg_graph, loom_amdgpu_sgpr_read_wait_transfer_pairs,
      state, state->arena));
  for (iree_host_size_t i = 0; i < state->schedule->block_count; ++i) {
    loom_amdgpu_sgpr_read_wait_summarize(
        state, loom_amdgpu_sgpr_read_wait_block(state, (uint32_t)i));
  }
  IREE_RETURN_IF_ERROR(loom_cfg_dataflow_solve(
      &state->schedule->cfg_graph, loom_amdgpu_sgpr_read_wait_transfer_writes,
      state, state->arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->schedule->block_count && iree_status_is_ok(status); ++i) {
    status = loom_amdgpu_sgpr_read_wait_resolve_block(state, (uint16_t)i, emit,
                                                      user_data);
  }
  return status;
}
