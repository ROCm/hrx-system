// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"

static void loom_x86_function_append(loom_x86_function_t* function,
                                     loom_x86_encoding_form_t form,
                                     uint16_t encoding_id,
                                     loom_x86_encoding_operands_t operands,
                                     uint32_t branch_target) {
  function->instructions[function->instruction_count++] =
      (loom_x86_instruction_t){
          .operands = operands,
          .branch_target = branch_target,
          .form = form,
          .encoding_id = encoding_id,
      };
  // SysV RBX, RBP, R12..R15 are callee-preserved. Reads, unused entry
  // arguments, and coalesced moves require no preservation.
  const uint16_t preserved = (1u << 3) | (1u << 5) | (0xfu << 12);
  function->saved_registers |=
      loom_x86_encoding_gpr_writes(form, &operands) & preserved;
}

static iree_status_t loom_x86_function_move(loom_x86_function_t* function,
                                            uint16_t register_class,
                                            uint8_t destination,
                                            uint8_t source) {
  loom_x86_register_class_t logical_class =
      loom_x86_logical_register_class(register_class);
  if (logical_class != LOOM_X86_REGISTER_CLASS_GPR32 &&
      logical_class != LOOM_X86_REGISTER_CLASS_GPR64) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 native transport requires a scalar GPR");
  }
  if (destination != source) {
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_MOVE,
        0x8b | (logical_class == LOOM_X86_REGISTER_CLASS_GPR64
                    ? LOOM_X86_ENCODING_REX_W
                    : 0),
        (loom_x86_encoding_operands_t){.result = destination,
                                       .inputs = {source}},
        UINT32_MAX);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_moves(
    const loom_low_allocation_table_t* allocation, loom_low_move_range_t range,
    loom_x86_function_t* function) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range.count && iree_status_is_ok(status);
       ++i) {
    const loom_low_move_t* move = &allocation->moves[range.start + i];
    status = loom_x86_function_move(
        function, move->destination.descriptor_reg_class_id,
        (uint8_t)move->destination.location, (uint8_t)move->source.location);
  }
  return status;
}

static int64_t loom_x86_function_immediate(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, uint16_t index) {
  const loom_low_immediate_t* immediate =
      &frame->target.descriptor_set
           ->immediates[packet->descriptor->immediate_start + index];
  loom_attribute_t value = loom_low_packet_immediate_attr(packet, immediate);
  return value.kind == LOOM_ATTR_ABSENT ? immediate->default_value : value.i64;
}

static iree_status_t loom_x86_function_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, loom_x86_function_t* function) {
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  if (descriptor->encoding_format_id == LOOM_X86_ENCODING_FORM_NONE) {
    iree_string_view_t mnemonic = loom_low_descriptor_set_string(
        frame->target.descriptor_set, descriptor->mnemonic_string_ref);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 native encoding is unavailable for '%.*s'",
                            (int)mnemonic.size, mnemonic.data);
  }
  loom_x86_encoding_operands_t operands = {0};
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    operands.inputs[i] = (uint8_t)loom_low_packet_operand_assignment(
                             &frame->allocation, packet, i)
                             ->location_base;
  }
  if (packet->node->result_count) {
    operands.result = (uint8_t)loom_low_packet_result_assignment(
                          &frame->allocation, packet, 0)
                          ->location_base;
  }
  if (descriptor->immediate_count) {
    operands.immediate = loom_x86_function_immediate(frame, packet, 0);
  }
  if (descriptor->immediate_count == 2) {
    operands.scale = (uint8_t)iree_math_count_trailing_zeros_u32(
        (uint32_t)loom_x86_function_immediate(frame, packet, 1));
  }
  loom_x86_function_append(
      function, (loom_x86_encoding_form_t)descriptor->encoding_format_id,
      descriptor->encoding_id, operands, UINT32_MAX);
  return iree_ok_status();
}

static void loom_x86_function_jump(loom_x86_function_t* function,
                                   uint32_t target, uint32_t fallthrough) {
  if (target != fallthrough) {
    loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_JUMP, 0,
                             (loom_x86_encoding_operands_t){0}, target);
  }
}

static iree_status_t loom_x86_function_structural(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, loom_x86_function_t* function) {
  const loom_low_schedule_node_t* node = packet->node;
  const uint32_t block_index = node->block_index;
  const loom_cfg_graph_t* graph = &frame->schedule.cfg_graph;
  if (loom_low_return_isa(node->op)) {
    if (node->operand_count) {
      const loom_low_allocation_assignment_t* result =
          loom_low_packet_operand_assignment(&frame->allocation, packet, 0);
      IREE_RETURN_IF_ERROR(
          loom_x86_function_move(function, result->descriptor_reg_class_id, 0,
                                 (uint8_t)result->location_base));
    }
    loom_x86_function_jump(function, function->block_count, block_index + 1);
    return iree_ok_status();
  }
  if (loom_low_br_isa(node->op)) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            &frame->allocation, node->source_ordinal);
    if (group) {
      IREE_RETURN_IF_ERROR(loom_x86_function_moves(
          &frame->allocation, group->move_group.moves, function));
    }
    loom_x86_function_jump(
        function,
        graph->successor_indices[graph->blocks[block_index].successor_start],
        block_index + 1);
    return iree_ok_status();
  }
  if (loom_low_cond_br_isa(node->op)) {
    const uint16_t* targets =
        graph->successor_indices + graph->blocks[block_index].successor_start;
    const loom_low_allocation_assignment_t* condition =
        loom_low_packet_operand_assignment(&frame->allocation, packet, 0);
    const uint16_t width =
        loom_x86_logical_register_class(condition->descriptor_reg_class_id) ==
                LOOM_X86_REGISTER_CLASS_GPR64
            ? LOOM_X86_ENCODING_REX_W
            : 0;
    const bool true_falls_through = targets[0] == block_index + 1;
    loom_x86_function_append(function,
                             true_falls_through
                                 ? LOOM_X86_ENCODING_FORM_BRANCH_ZERO
                                 : LOOM_X86_ENCODING_FORM_BRANCH_NONZERO,
                             width,
                             (loom_x86_encoding_operands_t){
                                 .inputs = {(uint8_t)condition->location_base}},
                             targets[true_falls_through ? 1 : 0]);
    if (!true_falls_through) {
      loom_x86_function_jump(function, targets[1], block_index + 1);
    }
    return iree_ok_status();
  }
  if (loom_low_copy_isa(node->op) || loom_low_move_isa(node->op) ||
      loom_low_slice_isa(node->op) || loom_low_concat_isa(node->op)) {
    const loom_low_allocation_packet_move_group_t* group =
        loom_low_allocation_find_packet_move_group_by_source_ordinal(
            &frame->allocation, node->source_ordinal);
    return group ? loom_x86_function_moves(&frame->allocation,
                                           group->move_group.moves, function)
                 : iree_ok_status();
  }
  iree_string_view_t name = loom_op_name(frame->module, node->op);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 native instruction materialization does not "
                          "support '%.*s'",
                          (int)name.size, name.data);
}

iree_status_t loom_x86_function_prepare(const loom_low_emission_frame_t* frame,
                                        iree_arena_allocator_t* arena,
                                        loom_x86_function_t* out_function) {
  *out_function = (loom_x86_function_t){0};
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  loom_x86_function_t function = {.block_count =
                                      (uint32_t)schedule->block_count};
  // A descriptor is one encoding record; a return or conditional branch needs
  // at most two. Allocation retains the exact number of final physical moves,
  // including multi-unit transport and cycle scratch.
  const iree_host_size_t capacity = schedule->scheduled_node_count +
                                    schedule->block_count +
                                    frame->allocation.move_count;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, capacity, sizeof(*function.instructions),
                                (void**)&function.instructions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count + 1, sizeof(*function.block_starts),
      (void**)&function.block_starts));
  iree_status_t status = iree_ok_status();
  for (uint32_t b = 0; b < schedule->block_count && iree_status_is_ok(status);
       ++b) {
    function.block_starts[b] = function.instruction_count;
    const loom_low_schedule_block_t* block = &schedule->blocks[b];
    for (uint32_t i = 0;
         i < block->scheduled_node_count && iree_status_is_ok(status); ++i) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(schedule, b, i);
      if (loom_low_packet_is_compile_time_only(&packet)) {
        continue;
      }
      status = packet.descriptor
                   ? loom_x86_function_packet(frame, &packet, &function)
                   : loom_x86_function_structural(frame, &packet, &function);
    }
  }
  function.block_starts[schedule->block_count] = function.instruction_count;
  if (iree_status_is_ok(status)) {
    *out_function = function;
  }
  return status;
}

typedef struct loom_x86_branch_fixup_t {
  // Output displacement field offset.
  iree_io_stream_pos_t offset;
  // Retained block ordinal; block_count denotes the epilogue.
  uint32_t target;
} loom_x86_branch_fixup_t;

static iree_status_t loom_x86_function_write_stack(
    iree_io_stream_t* stream, loom_x86_encoding_form_t form, uint8_t reg) {
  const loom_x86_encoding_operands_t operands = {.inputs = {reg}};
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(form, 0, &operands, &instruction);
  return iree_io_stream_write(stream, instruction.length, instruction.bytes);
}

iree_status_t loom_x86_function_write(const loom_x86_function_t* function,
                                      iree_io_stream_t* stream,
                                      iree_arena_allocator_t* arena) {
  iree_io_stream_pos_t* block_offsets = NULL;
  loom_x86_branch_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function->block_count + 1, sizeof(*block_offsets),
      (void**)&block_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 2 * function->block_count, sizeof(*fixups), (void**)&fixups));
  iree_status_t status = iree_ok_status();
  for (uint8_t reg = 0; reg < 16 && iree_status_is_ok(status); ++reg) {
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(stream,
                                             LOOM_X86_ENCODING_FORM_PUSH, reg);
    }
  }
  iree_host_size_t fixup_count = 0;
  uint32_t block = 0;
  for (iree_host_size_t i = 0;
       i < function->instruction_count && iree_status_is_ok(status); ++i) {
    while (block < function->block_count &&
           function->block_starts[block] == i) {
      block_offsets[block++] = iree_io_stream_offset(stream);
    }
    const loom_x86_instruction_t* prepared = &function->instructions[i];
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction((loom_x86_encoding_form_t)prepared->form,
                                prepared->encoding_id, &prepared->operands,
                                &instruction);
    if (prepared->branch_target != UINT32_MAX) {
      fixups[fixup_count++] = (loom_x86_branch_fixup_t){
          .offset = iree_io_stream_offset(stream) + instruction.length - 4,
          .target = prepared->branch_target,
      };
    }
    status =
        iree_io_stream_write(stream, instruction.length, instruction.bytes);
  }
  while (block <= function->block_count) {
    block_offsets[block++] = iree_io_stream_offset(stream);
  }
  for (uint8_t i = 16; i > 0 && iree_status_is_ok(status); --i) {
    const uint8_t reg = i - 1;
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(stream, LOOM_X86_ENCODING_FORM_POP,
                                             reg);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_x86_function_write_stack(stream, LOOM_X86_ENCODING_FORM_RETURN, 0);
  }
  const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
  for (iree_host_size_t i = 0; i < fixup_count && iree_status_is_ok(status);
       ++i) {
    const int64_t displacement =
        block_offsets[fixups[i].target] - fixups[i].offset - 4;
    if (displacement < INT32_MIN || displacement > INT32_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "x86 branch exceeds signed 32-bit displacement");
    } else {
      uint8_t bytes[4];
      iree_unaligned_store_le_u32(bytes, (uint32_t)displacement);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                                   fixups[i].offset);
      if (iree_status_is_ok(status)) {
        status = iree_io_stream_write(stream, 4, bytes);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
  }
  return status;
}
