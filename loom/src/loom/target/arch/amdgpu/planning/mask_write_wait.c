// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/mask_write_wait.h"

#include <string.h>

#include "loom/util/segmented_storage.h"

// GFX11 has 106 ordinary SGPR DWORDs. VCC has its own allocation class and
// occupies the last bit of this compact analysis domain, not an SGPR alias.
#define LOOM_AMDGPU_MASK_WRITE_VCC_BIT (UINT64_C(1) << 63)

// GFX11 DEPCTR fields. All other fields retain their unconstrained maxima,
// including va_vdst and vm_vsrc; reserved bits remain zero.
enum {
  LOOM_AMDGPU_MASK_WRITE_DEPCTR_DEFAULT = 0xff9f,
  LOOM_AMDGPU_MASK_WRITE_SA_SDST = 0x0001,
  LOOM_AMDGPU_MASK_WRITE_VA_VCC = 0x0002,
  LOOM_AMDGPU_MASK_WRITE_VA_SDST = 0x0e00,
};

typedef struct loom_amdgpu_mask_write_block_t {
  // Mask reads after the last scalar VALU read or overwrite of each register.
  uint64_t generated[2];
  // Incoming mask reads cleared by scalar VALU reads or register overwrites.
  uint64_t killed[2];
  // Mask reads reaching the block through any predecessor.
  uint64_t incoming[2];
  // Whether this block already has a pending propagation visit.
  bool queued;
} loom_amdgpu_mask_write_block_t;

typedef struct loom_amdgpu_mask_write_event_t {
  // First local overwrites which could match an incoming mask read.
  uint64_t incoming[2];
  // Packet owning this physical overwrite.
  uint32_t packet_index;
  // Native instruction boundary immediately after the overwrite.
  uint32_t instruction_offset;
  // DEPCTR fields already required by local mask reads, or zero.
  uint16_t fields;
  // Whether the overwrite issues on VALU rather than SALU.
  bool is_vector;
} loom_amdgpu_mask_write_event_t;

#define LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT \
  (4096u / sizeof(loom_amdgpu_mask_write_event_t))

struct loom_amdgpu_mask_write_wait_t {
  // Schedule owning final packet order and the retained CFG.
  const loom_low_schedule_table_t* schedule;
  // Final physical allocation and resolved movement sequences.
  const loom_low_allocation_table_t* allocation;
  // Arena owning all transient collection and propagation storage.
  iree_arena_allocator_t* arena;
  // Block summaries in the schedule's region order.
  loom_amdgpu_mask_write_block_t* blocks;
  // Sparse overwrite candidates in final instruction order.
  loom_segmented_storage_t events;
  // Number of populated overwrite candidates.
  iree_host_size_t event_count;
};

iree_status_t loom_amdgpu_mask_write_wait_create(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_mask_write_wait_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_mask_write_wait_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  *state = (loom_amdgpu_mask_write_wait_t){
      .schedule = schedule,
      .allocation = allocation,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                 sizeof(*state->blocks),
                                                 (void**)&state->blocks));
  memset(state->blocks, 0, schedule->block_count * sizeof(*state->blocks));
  loom_segmented_storage_initialize(
      LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT *
          sizeof(loom_amdgpu_mask_write_event_t),
      iree_alignof(loom_amdgpu_mask_write_event_t), &state->events);
  *out_state = state;
  return iree_ok_status();
}

static void loom_amdgpu_mask_write_set_range(uint64_t bits[2],
                                             uint16_t register_class,
                                             uint32_t base, uint32_t count) {
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VCC) {
    bits[1] |= LOOM_AMDGPU_MASK_WRITE_VCC_BIT;
  } else if (register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    for (uint32_t i = base; i < base + count; ++i) {
      bits[i / 64] |= UINT64_C(1) << (i % 64);
    }
  }
}

static void loom_amdgpu_mask_write_set_assignment(
    uint64_t bits[2], const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_kind ==
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_amdgpu_mask_write_set_range(bits, assignment->descriptor_reg_class_id,
                                     assignment->location_base,
                                     assignment->location_count);
  }
}

static uint16_t loom_amdgpu_mask_write_fields(const uint64_t bits[2],
                                              bool is_vector) {
  if ((bits[0] | bits[1]) == 0) {
    return 0;
  }
  if (!is_vector) {
    return LOOM_AMDGPU_MASK_WRITE_SA_SDST;
  }
  uint16_t fields = (bits[1] & LOOM_AMDGPU_MASK_WRITE_VCC_BIT)
                        ? LOOM_AMDGPU_MASK_WRITE_VA_VCC
                        : 0;
  if (bits[0] | (bits[1] & ~LOOM_AMDGPU_MASK_WRITE_VCC_BIT)) {
    fields |= LOOM_AMDGPU_MASK_WRITE_VA_SDST;
  }
  return fields;
}

static iree_status_t loom_amdgpu_mask_write_record(
    loom_amdgpu_mask_write_wait_t* state, const loom_low_packet_view_t* packet,
    uint32_t instruction_offset, const uint64_t writes[2],
    loom_amdgpu_descriptor_traits_t traits) {
  loom_amdgpu_mask_write_block_t* block =
      &state->blocks[packet->node->block_index];
  loom_amdgpu_mask_write_event_t event = {
      .packet_index = packet->packet_index,
      .instruction_offset = instruction_offset,
      .is_vector =
          iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU),
  };
  uint64_t local[2];
  for (unsigned word = 0; word < 2; ++word) {
    local[word] = writes[word] & block->generated[word];
    event.incoming[word] = writes[word] & ~block->killed[word];
    // Exposed SALU VCC definitions write vcc_lo. Keep the full-mask read
    // live until a VALU definition replaces both halves in wave64 mode.
    const uint64_t kills = word == 1 && !event.is_vector
                               ? writes[word] & ~LOOM_AMDGPU_MASK_WRITE_VCC_BIT
                               : writes[word];
    block->generated[word] &= ~kills;
    block->killed[word] |= kills;
  }
  if (!iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU |
                                    LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    return iree_ok_status();
  }
  event.fields = loom_amdgpu_mask_write_fields(local, event.is_vector);
  if (event.fields == 0 && (event.incoming[0] | event.incoming[1]) == 0) {
    return iree_ok_status();
  }
  const uint32_t segment_index =
      (uint32_t)(state->event_count /
                 LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT);
  const iree_host_size_t offset =
      state->event_count % LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT;
  void* segment = NULL;
  if (offset == 0) {
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&state->events, state->arena, &segment));
  } else {
    segment = loom_segmented_storage_segment(&state->events, segment_index);
  }
  ((loom_amdgpu_mask_write_event_t*)segment)[offset] = event;
  ++state->event_count;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_mask_write_wait_collect(
    loom_amdgpu_mask_write_wait_t* state, const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits) {
  iree_status_t status = iree_ok_status();
  if (packet->descriptor == NULL) {
    for (uint32_t i = 0;
         i < structural->moves.count && iree_status_is_ok(status); ++i) {
      const loom_low_move_t* move =
          &state->allocation->moves[structural->moves.start + i];
      if (move->destination.descriptor_reg_class_id !=
          LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
        continue;
      }
      uint64_t writes[2] = {0};
      loom_amdgpu_mask_write_set_range(
          writes, move->destination.descriptor_reg_class_id,
          move->destination.location, 1);
      status = loom_amdgpu_mask_write_record(
          state, packet, i + 1, writes,
          LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU);
    }
    return status;
  }

  loom_amdgpu_mask_write_block_t* block =
      &state->blocks[packet->node->block_index];
  uint64_t predicate_reads[2] = {0};
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    const loom_low_descriptor_set_t* descriptor_set =
        state->schedule->target.descriptor_set;
    bool resets_mask_reads = false;
    for (uint16_t i = packet->descriptor->result_count;
         i < packet->descriptor->operand_count; ++i) {
      const loom_low_operand_t* operand =
          loom_low_descriptor_operand(descriptor_set, packet->descriptor, i);
      uint16_t register_class = LOOM_LOW_REG_CLASS_NONE;
      if (loom_low_descriptor_operand_maps_to_packet_operand(
              descriptor_set, packet->descriptor, i)) {
        const loom_low_allocation_assignment_t* assignment =
            loom_low_packet_descriptor_operand_assignment(state->allocation,
                                                          packet, i);
        register_class = assignment->descriptor_reg_class_id;
        if (operand->role == LOOM_LOW_OPERAND_ROLE_PREDICATE) {
          loom_amdgpu_mask_write_set_assignment(predicate_reads, assignment);
        }
      } else if (iree_any_bit_set(operand->flags,
                                  LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
        register_class =
            descriptor_set->reg_class_alts[operand->reg_class_alt_start]
                .reg_class_id;
      }
      resets_mask_reads |=
          register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
          (register_class != LOOM_LOW_REG_CLASS_NONE &&
           iree_any_bit_set(
               loom_amdgpu_reg_class_traits(descriptor_set, register_class),
               LOOM_AMDGPU_REG_CLASS_TRAIT_VCC |
                   LOOM_AMDGPU_REG_CLASS_TRAIT_M0));
    }
    // Reading SGPR, VCC or M0 on VALU replaces the prior lane-mask latch
    // before this instruction writes its results. EXEC and constants do not.
    if (resets_mask_reads) {
      block->generated[0] = block->generated[1] = 0;
      block->killed[0] = block->killed[1] = UINT64_MAX;
    }
  }

  uint64_t writes[2] = {0};
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    loom_amdgpu_mask_write_set_assignment(
        writes,
        loom_low_packet_result_assignment(state->allocation, packet, i));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_mask_write_record(state, packet, 1, writes, traits));
  // Publish reads after overwrites: a carry instruction can read and replace
  // the same mask, and its read must still protect the next overwrite.
  block->generated[0] |= predicate_reads[0];
  block->generated[1] |= predicate_reads[1];
  return iree_ok_status();
}

static void loom_amdgpu_mask_write_propagate(
    loom_amdgpu_mask_write_wait_t* state, uint16_t* worklist) {
  const loom_cfg_graph_t* graph = &state->schedule->cfg_graph;
  iree_host_size_t count = graph->reverse_postorder.count;
  // Pop in reverse postorder so acyclic edges normally need only one visit.
  for (iree_host_size_t i = 0; i < count; ++i) {
    const uint16_t index = graph->reverse_postorder.values[count - i - 1];
    worklist[i] = index;
    state->blocks[index].queued = true;
  }
  while (count != 0) {
    const uint16_t index = worklist[--count];
    loom_amdgpu_mask_write_block_t* block = &state->blocks[index];
    block->queued = false;
    const uint64_t outgoing[2] = {
        block->generated[0] | (block->incoming[0] & ~block->killed[0]),
        block->generated[1] | (block->incoming[1] & ~block->killed[1]),
    };
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t successor = successors.values[i];
      loom_amdgpu_mask_write_block_t* next = &state->blocks[successor];
      if (((outgoing[0] & ~next->incoming[0]) |
           (outgoing[1] & ~next->incoming[1])) == 0) {
        continue;
      }
      next->incoming[0] |= outgoing[0];
      next->incoming[1] |= outgoing[1];
      if (!next->queued) {
        worklist[count++] = successor;
        next->queued = true;
      }
    }
  }
}

iree_status_t loom_amdgpu_mask_write_wait_resolve(
    loom_amdgpu_mask_write_wait_t* state,
    loom_amdgpu_mask_write_wait_emit_fn_t emit, void* user_data) {
  if (state->event_count == 0) {
    return iree_ok_status();
  }
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->schedule->block_count,
                                sizeof(*worklist), (void**)&worklist));
  loom_amdgpu_mask_write_propagate(state, worklist);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->event_count && iree_status_is_ok(status); ++i) {
    const loom_amdgpu_mask_write_event_t* segment =
        (const loom_amdgpu_mask_write_event_t*)
            loom_segmented_storage_const_segment(
                &state->events,
                (uint32_t)(i / LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT));
    const loom_amdgpu_mask_write_event_t* event =
        &segment[i % LOOM_AMDGPU_MASK_WRITE_EVENTS_PER_SEGMENT];
    const loom_low_packet_view_t packet =
        loom_low_packet_at(state->schedule, event->packet_index);
    const loom_amdgpu_mask_write_block_t* block =
        &state->blocks[packet.node->block_index];
    const uint64_t incoming[2] = {
        event->incoming[0] & block->incoming[0],
        event->incoming[1] & block->incoming[1],
    };
    const uint16_t fields = event->fields | loom_amdgpu_mask_write_fields(
                                                incoming, event->is_vector);
    if (fields == 0) {
      continue;
    }
    const loom_amdgpu_wait_state_t wait_state = {
        .reason = LOOM_AMDGPU_WAIT_STATE_REASON_MASK_WRITE,
        .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR,
        .block_index = packet.node->block_index,
        .node_index = packet.node_index,
        .scheduled_ordinal = packet.node->scheduled_ordinal,
        .instruction_offset = event->instruction_offset,
        .producer_node = packet.node_index,
        .consumer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
        .immediate =
            (uint16_t)(LOOM_AMDGPU_MASK_WRITE_DEPCTR_DEFAULT & ~fields),
    };
    status = emit(user_data, &wait_state);
  }
  return status;
}
