// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/function_layout.h"

#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/emit/vm/function_call.h"
#include "loom/target/emit/vm/function_locals.h"

static const uint8_t kLoomVmInstructionRecordByteLengths[] = {
#define LOOM_VM_INSTRUCTION_ENCODING_LIMITS(maximum_record_byte_length)
#define LOOM_VM_INSTRUCTION_ENCODING_ROW(byte_length) byte_length,
#include "loom/target/arch/vm/encoding_rows.inl"
#undef LOOM_VM_INSTRUCTION_ENCODING_ROW
#undef LOOM_VM_INSTRUCTION_ENCODING_LIMITS
};

static const loom_vm_function_control_layout_t
    kLoomVmFunctionControlLayouts[LOOM_VM_FUNCTION_CONTROL_ENCODING_COUNT] = {
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE] = {0},
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S16] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16,
                .first_byte_length = 4,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S32,
                .first_byte_length = 8,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S16,
                .first_byte_length = 4,
                .first_uses_condition = true,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S32,
                .first_byte_length = 8,
                .first_uses_condition = true,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S16] =
            {
                .first_opcode =
                    IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_UNLESS_S16,
                .first_byte_length = 4,
                .first_uses_condition = true,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S32] =
            {
                .first_opcode =
                    IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_UNLESS_S32,
                .first_byte_length = 8,
                .first_uses_condition = true,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S16] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S16,
                .first_byte_length = 4,
                .first_uses_condition = true,
                .second_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16,
                .second_byte_length = 4,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S32] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S16,
                .first_byte_length = 4,
                .first_uses_condition = true,
                .second_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S32,
                .second_byte_length = 8,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S16] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S32,
                .first_byte_length = 8,
                .first_uses_condition = true,
                .second_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S16,
                .second_byte_length = 4,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S32] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_IF_S32,
                .first_byte_length = 8,
                .first_uses_condition = true,
                .second_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_S32,
                .second_byte_length = 8,
            },
        [LOOM_VM_FUNCTION_CONTROL_ENCODING_YIELD_S32] =
            {
                .first_opcode = IREE_VM_ISA_CORE_OPCODE_CONTROL_YIELD_S32,
                .first_byte_length = 8,
            },
};

uint8_t loom_vm_function_descriptor_record_byte_length(
    uint32_t descriptor_ordinal) {
  IREE_ASSERT_LT(descriptor_ordinal,
                 IREE_ARRAYSIZE(kLoomVmInstructionRecordByteLengths));
  return kLoomVmInstructionRecordByteLengths[descriptor_ordinal];
}

const loom_vm_function_control_layout_t*
loom_vm_function_control_encoding_layout(
    loom_vm_function_control_encoding_t encoding) {
  IREE_ASSERT_LT(encoding, LOOM_VM_FUNCTION_CONTROL_ENCODING_COUNT);
  return &kLoomVmFunctionControlLayouts[encoding];
}

static uint32_t loom_vm_function_move_group_byte_length(
    const loom_low_move_group_t* group) {
  if (group == NULL) return 0;
  IREE_ASSERT_LE(group->moves.count, UINT32_MAX / 4u);
  return (uint32_t)group->moves.count * 4u;
}

static uint32_t loom_vm_function_packet_move_byte_length(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet) {
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          &frame->allocation, packet->node->source_ordinal);
  return group != NULL
             ? loom_vm_function_move_group_byte_length(&group->move_group)
             : 0;
}

static uint32_t loom_vm_function_edge_move_byte_length(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet) {
  const loom_low_allocation_edge_copy_group_t* group =
      loom_low_allocation_find_edge_copy_group_by_source_ordinal(
          &frame->allocation, packet->node->source_ordinal);
  return group != NULL
             ? loom_vm_function_move_group_byte_length(&group->move_group)
             : 0;
}

static iree_status_t loom_vm_function_structural_packet_byte_length(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_vm_function_code_layout_t* code_layout, uint32_t* out_byte_length) {
  *out_byte_length = 0;
  const loom_op_t* op = packet->node->op;
  if (loom_low_return_isa(op)) {
    *out_byte_length = 4;
    return iree_ok_status();
  }
  if (loom_low_copy_isa(op) || loom_low_move_isa(op) ||
      loom_low_slice_isa(op) || loom_low_concat_isa(op)) {
    *out_byte_length = loom_vm_function_packet_move_byte_length(frame, packet);
    return iree_ok_status();
  }
  if (loom_low_storage_reserve_isa(op) || loom_low_storage_view_isa(op)) {
    return iree_ok_status();
  }
  if (loom_vm_function_local_transfer_is_packet(packet)) {
    *out_byte_length =
        loom_vm_function_local_transfer_byte_length(frame, packet);
    return iree_ok_status();
  }
  loom_vm_function_call_view_t call = {0};
  if (loom_vm_function_call_try_view(op, &call)) {
    loom_vm_call_abi_packet_layout_t call_layout = {0};
    IREE_RETURN_IF_ERROR(
        loom_vm_function_call_layout_build(frame->module, &call, &call_layout));
    *out_byte_length =
        loom_vm_function_call_record_byte_length(&call, &call_layout);
    return iree_ok_status();
  }
  if (loom_low_func_null_isa(op)) {
    *out_byte_length = sizeof(iree_vm_isa_func_null_record_t);
    return iree_ok_status();
  }
  if (loom_low_func_compare_null_isa(op)) {
    *out_byte_length = sizeof(iree_vm_isa_func_compare_null_record_t);
    return iree_ok_status();
  }
  if (loom_low_func_address_isa(op)) {
    *out_byte_length = sizeof(iree_vm_isa_func_address_record_t);
    return iree_ok_status();
  }
  if (loom_low_func_import_resolved_isa(op)) {
    *out_byte_length = sizeof(iree_vm_isa_func_import_resolved_record_t);
    return iree_ok_status();
  }
  const loom_vm_function_control_layout_t* control_layout =
      loom_vm_function_control_encoding_layout(
          code_layout->control_encodings[packet->packet_index]);
  if (loom_low_br_isa(op)) {
    *out_byte_length = loom_vm_function_edge_move_byte_length(frame, packet) +
                       control_layout->first_byte_length;
    return iree_ok_status();
  }
  if (loom_low_cond_br_isa(op)) {
    *out_byte_length =
        control_layout->first_byte_length + control_layout->second_byte_length;
    return iree_ok_status();
  }
  if (loom_low_switch_isa(op)) {
    *out_byte_length = sizeof(iree_vm_isa_control_switch_record_t) +
                       control_layout->first_byte_length;
    return iree_ok_status();
  }
  if (packet->descriptor != NULL) {
    const uint32_t descriptor_ordinal =
        (uint32_t)(packet->descriptor -
                   frame->target.descriptor_set->descriptors);
    *out_byte_length =
        loom_vm_function_descriptor_record_byte_length(descriptor_ordinal);
    return iree_ok_status();
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(frame->module, op);
  const iree_string_view_t op_name =
      vtable != NULL ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "VM function layout does not support structural "
                          "operation %.*s",
                          (int)op_name.size, op_name.data);
}

static uint32_t loom_vm_function_target_block_index(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block) {
  const uint32_t block_index = loom_low_packet_block_index(schedule, block);
  IREE_ASSERT_NE(block_index, LOOM_LOW_PACKET_INDEX_NONE);
  return block_index;
}

static loom_vm_function_control_encoding_t
loom_vm_function_initial_control_encoding(
    const loom_low_schedule_table_t* schedule, uint32_t block_index,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const uint32_t next_block_index = block_index + 1u;
  if (loom_low_br_isa(op)) {
    if (packet->descriptor != NULL) {
      IREE_ASSERT_EQ(packet->descriptor->carrier,
                     LOOM_LOW_DESCRIPTOR_CARRIER_BRANCH);
      IREE_ASSERT_EQ(packet->descriptor->encoding_id,
                     IREE_VM_ISA_CORE_OPCODE_CONTROL_YIELD_S32);
      return LOOM_VM_FUNCTION_CONTROL_ENCODING_YIELD_S32;
    }
    const uint32_t target_block_index =
        loom_vm_function_target_block_index(schedule, loom_low_br_dest(op));
    return target_block_index == next_block_index
               ? LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE
               : LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32;
  }
  if (!loom_low_cond_br_isa(op)) {
    if (!loom_low_switch_isa(op)) {
      return LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE;
    }
    const uint32_t target_block_index = loom_vm_function_target_block_index(
        schedule, loom_low_switch_default_dest(op));
    return target_block_index == next_block_index
               ? LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE
               : LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32;
  }
  const uint32_t true_block_index = loom_vm_function_target_block_index(
      schedule, loom_low_cond_br_true_dest(op));
  const uint32_t false_block_index = loom_vm_function_target_block_index(
      schedule, loom_low_cond_br_false_dest(op));
  if (true_block_index == false_block_index) {
    return true_block_index == next_block_index
               ? LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE
               : LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32;
  }
  if (false_block_index == next_block_index) {
    return LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32;
  }
  if (true_block_index == next_block_index) {
    return LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S32;
  }
  return LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S32;
}

static iree_status_t loom_vm_function_code_layout_calculate_offsets(
    const loom_low_emission_frame_t* frame,
    loom_vm_function_code_layout_t* layout) {
  uint64_t byte_offset = 0;
  for (uint32_t block_index = 0; block_index < frame->schedule.block_count;
       ++block_index) {
    if (byte_offset > UINT32_MAX - 4u) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM function bytecode length exceeds u32");
    }
    layout->block_offsets[block_index] = (uint32_t)byte_offset;
    byte_offset += 4u;

    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(&frame->schedule, block_index, i);
      layout->packet_offsets[packet.packet_index] = (uint32_t)byte_offset;
      if (loom_low_packet_is_compile_time_only(&packet)) continue;
      uint32_t packet_byte_length = 0;
      IREE_RETURN_IF_ERROR(loom_vm_function_structural_packet_byte_length(
          frame, &packet, layout, &packet_byte_length));
      if (byte_offset + packet_byte_length > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM function bytecode length exceeds u32");
      }
      byte_offset += packet_byte_length;
    }
  }
  layout->bytecode_length = (uint32_t)byte_offset;
  return iree_ok_status();
}

static bool loom_vm_function_wide_branch_can_shrink(uint32_t record_offset,
                                                    uint8_t record_byte_length,
                                                    uint32_t target_offset) {
  IREE_ASSERT_EQ(record_byte_length, 8u);
  int64_t displacement =
      (int64_t)target_offset - ((int64_t)record_offset + record_byte_length);
  if (target_offset <= record_offset) {
    displacement += 4;
  }
  IREE_ASSERT_EQ(displacement % 4, 0);
  const int64_t displacement_words = displacement / 4;
  return displacement_words >= INT16_MIN && displacement_words <= INT16_MAX;
}

static void loom_vm_function_select_narrow_branch(
    const loom_low_emission_frame_t* frame,
    loom_vm_function_code_layout_t* layout,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  loom_vm_function_control_encoding_t* encoding =
      &layout->control_encodings[packet->packet_index];
  const loom_vm_function_control_layout_t* control_layout =
      loom_vm_function_control_encoding_layout(*encoding);
  uint32_t first_target_block_index = LOOM_LOW_PACKET_INDEX_NONE;
  uint32_t second_target_block_index = LOOM_LOW_PACKET_INDEX_NONE;
  uint32_t first_record_offset = layout->packet_offsets[packet->packet_index];
  if (loom_low_br_isa(op)) {
    if (packet->descriptor != NULL) return;
    first_record_offset +=
        loom_vm_function_edge_move_byte_length(frame, packet);
    first_target_block_index = loom_vm_function_target_block_index(
        &frame->schedule, loom_low_br_dest(op));
  } else if (loom_low_cond_br_isa(op)) {
    const uint32_t true_block_index = loom_vm_function_target_block_index(
        &frame->schedule, loom_low_cond_br_true_dest(op));
    const uint32_t false_block_index = loom_vm_function_target_block_index(
        &frame->schedule, loom_low_cond_br_false_dest(op));
    if (control_layout->first_uses_condition) {
      const bool branches_unless =
          *encoding == LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S32 ||
          *encoding == LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S16;
      first_target_block_index =
          branches_unless ? false_block_index : true_block_index;
      if (control_layout->second_byte_length != 0) {
        second_target_block_index = false_block_index;
      }
    } else {
      first_target_block_index = true_block_index;
    }
  } else if (loom_low_switch_isa(op)) {
    first_record_offset += sizeof(iree_vm_isa_control_switch_record_t);
    first_target_block_index = loom_vm_function_target_block_index(
        &frame->schedule, loom_low_switch_default_dest(op));
  } else {
    return;
  }

  bool first_shrinks = false;
  if (control_layout->first_byte_length == 8u) {
    first_shrinks = loom_vm_function_wide_branch_can_shrink(
        first_record_offset, control_layout->first_byte_length,
        layout->block_offsets[first_target_block_index]);
  }
  bool second_shrinks = false;
  if (control_layout->second_byte_length == 8u) {
    const uint32_t second_record_offset =
        first_record_offset + control_layout->first_byte_length;
    second_shrinks = loom_vm_function_wide_branch_can_shrink(
        second_record_offset, control_layout->second_byte_length,
        layout->block_offsets[second_target_block_index]);
  }
  if (!first_shrinks && !second_shrinks) return;

  switch (*encoding) {
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32:
      *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S16;
      break;
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32:
      *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16;
      break;
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S32:
      *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S16;
      break;
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S32:
      *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S16;
      break;
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S16:
      *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S16;
      break;
    case LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S32:
      if (first_shrinks && second_shrinks) {
        *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S16;
      } else if (first_shrinks) {
        *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S32;
      } else {
        *encoding = LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S16;
      }
      break;
    default:
      IREE_ASSERT_UNREACHABLE("narrowable VM branch encoding");
      break;
  }
}

iree_status_t loom_vm_function_code_layout_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_vm_function_code_layout_t* out_layout) {
  *out_layout = (loom_vm_function_code_layout_t){0};
  if (frame->schedule.block_count == 0 ||
      frame->schedule.block_count > 65536u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM function block count must be in [1, 65536]");
  }
  uint32_t* block_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, frame->schedule.block_count, sizeof(*block_offsets),
      (void**)&block_offsets));
  uint32_t* packet_offsets = NULL;
  loom_vm_function_control_encoding_t* control_encodings = NULL;
  if (frame->schedule.scheduled_node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, frame->schedule.scheduled_node_count, sizeof(*packet_offsets),
        (void**)&packet_offsets));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, frame->schedule.scheduled_node_count, sizeof(*control_encodings),
        (void**)&control_encodings));
  }
  *out_layout = (loom_vm_function_code_layout_t){
      .block_offsets = block_offsets,
      .packet_offsets = packet_offsets,
      .control_encodings = control_encodings,
  };

  uint64_t switch_target_entry_count = 0;
  loom_vm_function_local_counts_t call_prefix = {0};
  for (uint32_t block_index = 0; block_index < frame->schedule.block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block =
        &frame->schedule.blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(&frame->schedule, block_index, i);
      control_encodings[packet.packet_index] =
          loom_vm_function_initial_control_encoding(&frame->schedule,
                                                    block_index, &packet);
      if (loom_low_switch_isa(packet.node->op)) {
        switch_target_entry_count +=
            loom_low_switch_target_dests(packet.node->op).count;
      }
      loom_vm_function_call_view_t call = {0};
      if (loom_vm_function_call_try_view(packet.node->op, &call)) {
        loom_vm_call_abi_packet_layout_t call_layout = {0};
        IREE_RETURN_IF_ERROR(loom_vm_function_call_layout_build(
            frame->module, &call, &call_layout));
        call_prefix.byte_length = iree_max(
            call_prefix.byte_length, (uint16_t)call_layout.local_byte_length);
        call_prefix.ref_count =
            iree_max(call_prefix.ref_count, call_layout.local_ref_count);
        call_prefix.function_count = iree_max(call_prefix.function_count,
                                              call_layout.local_function_count);
        out_layout->has_call = true;
      }
    }
  }
  if (switch_target_entry_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM function switch-target entry count exceeds u32");
  }
  out_layout->switch_target_entry_count = (uint32_t)switch_target_entry_count;
  IREE_RETURN_IF_ERROR(loom_vm_function_local_layout_build(
      frame, call_prefix, arena, &out_layout->locals));

  // Select narrow branches against a pessimistic all-wide layout. Shrinking a
  // record can only bring every other direct target closer or leave its
  // displacement unchanged, so every selected narrow form remains valid in
  // the final compact layout. A branch that crosses the narrow boundary only
  // because unrelated records also shrink deliberately stays wide; avoiding
  // iterative layout work is worth those exceptionally rare four bytes.
  IREE_RETURN_IF_ERROR(
      loom_vm_function_code_layout_calculate_offsets(frame, out_layout));
  for (iree_host_size_t packet_index = 0;
       packet_index < frame->schedule.scheduled_node_count; ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&frame->schedule, packet_index);
    loom_vm_function_select_narrow_branch(frame, out_layout, &packet);
  }
  return loom_vm_function_code_layout_calculate_offsets(frame, out_layout);
}
