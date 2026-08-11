// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent packet views over scheduled and allocated low functions.
//
// The packet layer is the emitter-facing join between schedule and allocation
// tables. It intentionally does not serialize, print types, or know about any
// target backend. Native, SPIR-V, WebAssembly, and diagnostic emitters consume
// this view without copying the schedule/allocation join logic into each
// backend or routing through JSON.

#ifndef LOOM_CODEGEN_LOW_PACKET_H_
#define LOOM_CODEGEN_LOW_PACKET_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent packet indices.
#define LOOM_LOW_PACKET_INDEX_NONE UINT32_MAX

// Returns the named descriptor-attribute slice for |op| when it is a
// descriptor-backed low packet op. When non-NULL, |out_attrs_attr_index|
// receives the operation attribute field index that owns the dictionary for
// diagnostics.
bool loom_low_packet_try_op_attrs(const loom_op_t* op,
                                  loom_named_attr_slice_t* out_attrs,
                                  uint16_t* out_attrs_attr_index);

// One scheduled packet in emitter order.
typedef struct loom_low_packet_view_t {
  // Packet ordinal in the final scheduled stream.
  iree_host_size_t packet_index;
  // Schedule-node index represented by this packet.
  uint32_t node_index;
  // Schedule node represented by this packet.
  const loom_low_schedule_node_t* node;
  // Descriptor row for descriptor-backed packets, or NULL for structural ops.
  const loom_low_descriptor_t* descriptor;
} loom_low_packet_view_t;

// Returns true when |packet| is compile-time-only and has no emitted target
// instruction. These packets remain in schedule and report tables so their
// structural position is observable, but final emitters omit them.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline bool
loom_low_packet_is_compile_time_only(const loom_low_packet_view_t* packet) {
  return loom_traits_are_compile_time_only(packet->node->traits);
}

// Returns the number of packets in a successful schedule.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_host_size_t
loom_low_packet_count(const loom_low_schedule_table_t* schedule) {
  return schedule->scheduled_node_count;
}

// Returns the packet at |packet_index| in a successful schedule.
//
// |packet_index| must be derived from |schedule|.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline loom_low_packet_view_t
loom_low_packet_at(const loom_low_schedule_table_t* schedule,
                   iree_host_size_t packet_index) {
  IREE_ASSERT_LT(packet_index, schedule->scheduled_node_count);
  const uint32_t node_index = schedule->scheduled_node_indices[packet_index];
  IREE_ASSERT_LT(node_index, schedule->node_count);
  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  return (loom_low_packet_view_t){
      /*.packet_index=*/packet_index,
      /*.node_index=*/node_index,
      /*.node=*/node,
      /*.descriptor=*/node->descriptor,
  };
}

// Returns the packet at |scheduled_ordinal| in |block_index|.
//
// Both indices must be derived from |schedule|.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline loom_low_packet_view_t
loom_low_packet_at_block_ordinal(const loom_low_schedule_table_t* schedule,
                                 uint32_t block_index,
                                 uint32_t scheduled_ordinal) {
  IREE_ASSERT_LT(block_index, schedule->block_count);
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  IREE_ASSERT_LT(scheduled_ordinal, block->scheduled_node_count);
  const iree_host_size_t packet_index =
      (iree_host_size_t)block->scheduled_node_start + scheduled_ordinal;
  return loom_low_packet_at(schedule, packet_index);
}

// Returns the allocation assignment for a descriptor operand in a verified
// packet from a successful allocation. The packet and allocation must describe
// the same immutable low function.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline const loom_low_allocation_assignment_t*
loom_low_packet_descriptor_operand_assignment(
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet, uint16_t descriptor_operand_index) {
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      allocation->target.descriptor_set;
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  if (descriptor_operand_index < descriptor->result_count) {
    const uint16_t result_index = descriptor_operand->source_value_index;
    IREE_ASSERT_LT(result_index, packet->node->result_count);
    const loom_value_ordinal_t value_ordinal =
        loom_low_schedule_node_const_result_ordinals(
            packet->node)[result_index];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(allocation,
                                                         value_ordinal, NULL);
    IREE_ASSERT(assignment != NULL);
    return assignment;
  }
  const uint16_t packet_operand_index =
      loom_low_descriptor_operand_packet_index(descriptor_set, descriptor,
                                               descriptor_operand_index);
  IREE_ASSERT_LT(packet_operand_index, packet->node->operand_count);
  const loom_value_ordinal_t value_ordinal =
      loom_low_schedule_node_const_operand_ordinals(
          packet->node)[packet_operand_index];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(allocation,
                                                       value_ordinal, NULL);
  IREE_ASSERT(assignment != NULL);
  return assignment;
}

// Returns the named descriptor-attribute slice for |packet|, or an empty slice
// for structural packets.
loom_named_attr_slice_t loom_low_packet_attrs(
    const loom_low_packet_view_t* packet);

// Optional selected asm-form table for scheduled packets. Target legality or
// target emitters populate this table when descriptor-backed packets have
// multiple legal asm forms. Entries are indexed by packet ordinal; structural
// packets and descriptor packets that should use their unique canonical form
// use LOOM_LOW_ASM_FORM_ORDINAL_NONE.
typedef struct loom_low_packet_asm_form_table_t {
  // Module containing the packetized low function.
  const loom_module_t* module;
  // Target-low function operation packetized by this table.
  const loom_op_t* function_op;
  // Resolved target context selected by |function_op|.
  loom_low_resolved_target_t target;
  // Selected asm-form ordinals indexed by scheduled packet ordinal.
  const uint32_t* asm_form_ordinals;
  // Number of records in |asm_form_ordinals|.
  iree_host_size_t asm_form_ordinal_count;
} loom_low_packet_asm_form_table_t;

// Verifies that |asm_forms| describes selected asm forms for |schedule|.
iree_status_t loom_low_packet_validate_asm_form_table(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms);

// Resolves the asm form for |packet|. A selected asm-form table overrides the
// descriptor canonical form when it names a valid form for the packet's
// descriptor; otherwise the descriptor must have a unique canonical form.
iree_status_t loom_low_packet_lookup_asm_form(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms,
    const loom_low_packet_view_t* packet, uint32_t* out_asm_form_ordinal);

// Returns the region-block index for |block|, or LOOM_LOW_PACKET_INDEX_NONE
// when |block| does not belong to |schedule|.
uint32_t loom_low_packet_block_index(const loom_low_schedule_table_t* schedule,
                                     const loom_block_t* block);

// Maps a hazard-gap scheduled ordinal within the gap block to a packet index.
// Returns LOOM_LOW_PACKET_INDEX_NONE when the gap block is invalid or the
// computed packet index cannot fit in the packet-index sentinel domain.
uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_H_
