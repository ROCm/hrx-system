// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/function_encoder.h"

#include <string.h>

#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/value.h"
#include "iree/vm/module.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/emit/vm/function_layout.h"

#define LOOM_VM_INSTRUCTION_ENCODING_LIMITS(maximum_record_byte_length) \
  enum {                                                                \
    LOOM_VM_MAXIMUM_INSTRUCTION_RECORD_BYTE_LENGTH =                    \
        maximum_record_byte_length,                                     \
  };
#define LOOM_VM_INSTRUCTION_ENCODING_ROW(byte_length)
#include "loom/target/arch/vm/encoding_rows.inl"
#undef LOOM_VM_INSTRUCTION_ENCODING_ROW
#undef LOOM_VM_INSTRUCTION_ENCODING_LIMITS

static const loom_named_attr_t* loom_vm_function_find_packet_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    const iree_string_view_t attr_name = module->strings.entries[attr->name_id];
    if (iree_string_view_equal(attr_name, name)) return attr;
  }
  return NULL;
}

static int64_t loom_vm_function_resolve_immediate(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_immediate_t* immediate, const loom_named_attr_t* attr) {
  if (attr == NULL) {
    IREE_ASSERT(iree_any_bit_set(immediate->flags,
                                 LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE));
    return immediate->default_value;
  }
  if (attr->value.kind == LOOM_ATTR_I64) {
    return attr->value.i64;
  }
  if (attr->value.kind != LOOM_ATTR_STRING ||
      immediate->kind != LOOM_LOW_IMMEDIATE_KIND_ENUM ||
      attr->value.string_id >= module->strings.count) {
    IREE_ASSERT_UNREACHABLE("verified VM immediate kind");
    IREE_BUILTIN_UNREACHABLE();
  }
  int64_t value = 0;
  if (!loom_low_descriptor_set_lookup_enum_value_by_token(
          descriptor_set, immediate->enum_domain_id,
          module->strings.entries[attr->value.string_id], &value)) {
    IREE_ASSERT_UNREACHABLE("verified VM enum immediate token");
    IREE_BUILTIN_UNREACHABLE();
  }
  return value;
}

static void loom_vm_function_encode_immediate(uint8_t* record,
                                              uint16_t byte_offset,
                                              uint16_t bit_width,
                                              int64_t value) {
  const uint64_t bits = (uint64_t)value;
  const uint16_t byte_length = bit_width / 8u;
  for (uint16_t i = 0; i < byte_length; ++i) {
    record[byte_offset + i] = (uint8_t)(bits >> (i * 8u));
  }
}

static iree_status_t loom_vm_function_encode_descriptor_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, loom_bytecode_page_writer_t* writer) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  const uint32_t descriptor_ordinal =
      (uint32_t)(packet->descriptor - descriptor_set->descriptors);
  const uint8_t record_byte_length =
      loom_vm_function_descriptor_record_byte_length(descriptor_ordinal);
  uint8_t record[LOOM_VM_MAXIMUM_INSTRUCTION_RECORD_BYTE_LENGTH];
  memset(record, 0, record_byte_length);
  record[0] = (uint8_t)packet->descriptor->encoding_id;

  for (uint16_t i = 0; i < packet->descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[packet->descriptor->operand_start + i];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(&frame->allocation,
                                                      packet, i);
    IREE_ASSERT(assignment->location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    IREE_ASSERT(loom_low_operand_accepts_unit_count(
        operand, assignment->location_count));
    IREE_ASSERT_LT(assignment->location_base, 256u);
    IREE_ASSERT_LE(assignment->location_count,
                   256u - assignment->location_base);
    IREE_ASSERT_LT(operand->encoding_field_id, record_byte_length);
    record[operand->encoding_field_id] = (uint8_t)assignment->location_base;
  }

  const loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  for (uint16_t i = 0; i < packet->descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[packet->descriptor->immediate_start + i];
    const iree_string_view_t name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    const loom_named_attr_t* attr =
        loom_vm_function_find_packet_attr(frame->module, attrs, name);
    const int64_t value = loom_vm_function_resolve_immediate(
        frame->module, descriptor_set, immediate, attr);
    IREE_ASSERT_LE(immediate->encoding_field_id + immediate->bit_width / 8u,
                   record_byte_length);
    loom_vm_function_encode_immediate(record, immediate->encoding_field_id,
                                      immediate->bit_width, value);
  }

  return loom_bytecode_page_writer_write(writer, record, record_byte_length);
}

static iree_status_t loom_vm_function_encode_move_group(
    const loom_low_allocation_table_t* allocation,
    const loom_low_move_group_t* group, loom_bytecode_page_writer_t* writer) {
  if (group == NULL) return iree_ok_status();
  const iree_host_size_t move_end = group->moves.start + group->moves.count;
  for (iree_host_size_t i = group->moves.start; i < move_end; ++i) {
    const loom_low_move_t* move = &allocation->moves[i];
    IREE_ASSERT(move->source.location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    IREE_ASSERT(move->destination.location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    IREE_ASSERT_EQ(move->source.descriptor_reg_class_id,
                   VM_CORE_REG_CLASS_ID_VALUE);
    IREE_ASSERT_EQ(move->destination.descriptor_reg_class_id,
                   VM_CORE_REG_CLASS_ID_VALUE);
    IREE_ASSERT_LT(move->source.location, 256u);
    IREE_ASSERT_LT(move->destination.location, 256u);
    const iree_vm_isa_value_copy_record_t record = {
        .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_VALUE_COPY,
        .dst_v8 = (uint8_t)move->destination.location,
        .src_v8 = (uint8_t)move->source.location,
        .zero_padding_u8 = 0,
    };
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write(writer, &record, sizeof(record)));
  }
  return iree_ok_status();
}

static uint8_t loom_vm_function_value_operand_register(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, uint16_t operand_index) {
  IREE_ASSERT_LT(operand_index, packet->node->operand_count);
  const loom_value_ordinal_t operand_ordinal =
      loom_low_schedule_node_const_operand_ordinals(
          packet->node)[operand_index];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(&frame->allocation,
                                                       operand_ordinal, NULL);
  IREE_ASSERT(assignment != NULL);
  IREE_ASSERT_EQ(assignment->location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  IREE_ASSERT_EQ(assignment->descriptor_reg_class_id,
                 VM_CORE_REG_CLASS_ID_VALUE);
  IREE_ASSERT_EQ(assignment->location_count, 1u);
  IREE_ASSERT_LT(assignment->location_base, 256u);
  return (uint8_t)assignment->location_base;
}

static iree_status_t loom_vm_function_encode_direct_branch(
    uint8_t opcode, uint8_t record_byte_length, uint16_t condition_register,
    uint32_t record_offset, uint32_t target_offset,
    loom_bytecode_page_writer_t* writer) {
  IREE_ASSERT(record_byte_length == 4u || record_byte_length == 8u);
  IREE_ASSERT_EQ(record_offset % 4u, 0u);
  IREE_ASSERT_EQ(target_offset % 4u, 0u);
  const int64_t displacement =
      (int64_t)target_offset - ((int64_t)record_offset + record_byte_length);
  IREE_ASSERT_EQ(displacement % 4, 0);
  const int64_t displacement_words = displacement / 4;

  uint8_t record[8] = {0};
  record[0] = opcode;
  if (condition_register != UINT16_MAX) {
    IREE_ASSERT_LT(condition_register, 256u);
    record[1] = (uint8_t)condition_register;
  }
  if (record_byte_length == 4u) {
    IREE_ASSERT(displacement_words >= INT16_MIN &&
                displacement_words <= INT16_MAX);
    loom_vm_function_encode_immediate(record, /*byte_offset=*/2,
                                      /*bit_width=*/16, displacement_words);
  } else {
    IREE_ASSERT(displacement_words >= INT32_MIN &&
                displacement_words <= INT32_MAX);
    loom_vm_function_encode_immediate(record, /*byte_offset=*/4,
                                      /*bit_width=*/32, displacement_words);
  }
  return loom_bytecode_page_writer_write(writer, record, record_byte_length);
}

static iree_status_t loom_vm_function_encode_control_packet(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_code_layout_t* code_layout,
    const loom_low_packet_view_t* packet, loom_bytecode_page_writer_t* writer) {
  const loom_op_t* op = packet->node->op;
  const loom_vm_function_control_layout_t* control_layout =
      loom_vm_function_control_encoding_layout(
          code_layout->control_encodings[packet->packet_index]);
  uint32_t first_record_offset =
      code_layout->packet_offsets[packet->packet_index];
  if (loom_low_br_isa(op)) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            &frame->allocation, packet->node->source_ordinal);
    if (group != NULL) {
      IREE_RETURN_IF_ERROR(loom_vm_function_encode_move_group(
          &frame->allocation, &group->move_group, writer));
      first_record_offset += (uint32_t)group->move_group.moves.count * 4u;
    }
    if (control_layout->first_byte_length == 0) return iree_ok_status();
    const uint32_t target_block_index =
        loom_low_packet_block_index(&frame->schedule, loom_low_br_dest(op));
    IREE_ASSERT_NE(target_block_index, LOOM_LOW_PACKET_INDEX_NONE);
    return loom_vm_function_encode_direct_branch(
        control_layout->first_opcode, control_layout->first_byte_length,
        /*condition_register=*/UINT16_MAX, first_record_offset,
        code_layout->block_offsets[target_block_index], writer);
  }

  IREE_ASSERT(loom_low_cond_br_isa(op));
  if (control_layout->first_byte_length == 0) return iree_ok_status();
  const uint32_t true_block_index = loom_low_packet_block_index(
      &frame->schedule, loom_low_cond_br_true_dest(op));
  const uint32_t false_block_index = loom_low_packet_block_index(
      &frame->schedule, loom_low_cond_br_false_dest(op));
  IREE_ASSERT_NE(true_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  IREE_ASSERT_NE(false_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  const bool branches_unless =
      control_layout->first_opcode ==
          IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_UNLESS_S16 ||
      control_layout->first_opcode ==
          IREE_VM_ISA_CORE_OPCODE_CONTROL_BRANCH_UNLESS_S32;
  const uint32_t first_target_block_index =
      branches_unless ? false_block_index : true_block_index;
  const uint16_t condition_register =
      control_layout->first_uses_condition
          ? loom_vm_function_value_operand_register(frame, packet, 0)
          : UINT16_MAX;
  IREE_RETURN_IF_ERROR(loom_vm_function_encode_direct_branch(
      control_layout->first_opcode, control_layout->first_byte_length,
      condition_register, first_record_offset,
      code_layout->block_offsets[first_target_block_index], writer));
  if (control_layout->second_byte_length == 0) return iree_ok_status();
  return loom_vm_function_encode_direct_branch(
      control_layout->second_opcode, control_layout->second_byte_length,
      /*condition_register=*/UINT16_MAX,
      first_record_offset + control_layout->first_byte_length,
      code_layout->block_offsets[false_block_index], writer);
}

static iree_status_t loom_vm_function_encode_switch(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_code_layout_t* code_layout,
    const loom_low_packet_view_t* packet, uint32_t switch_target_base,
    iree_vm_bytecode_v0_switch_target_entry_t* switch_targets,
    loom_bytecode_page_writer_t* writer) {
  const loom_successor_slice_t target_dests =
      loom_low_switch_target_dests(packet->node->op);
  IREE_ASSERT_GT(target_dests.count, 0u);
  IREE_ASSERT_LT(target_dests.count, UINT16_MAX);
  IREE_ASSERT_LE(target_dests.count, code_layout->switch_target_entry_count);
  IREE_ASSERT_LE(switch_target_base,
                 code_layout->switch_target_entry_count - target_dests.count);
  for (uint16_t i = 0; i < target_dests.count; ++i) {
    const uint32_t target_block_index =
        loom_low_packet_block_index(&frame->schedule, target_dests.blocks[i]);
    IREE_ASSERT_NE(target_block_index, LOOM_LOW_PACKET_INDEX_NONE);
    IREE_ASSERT_EQ(code_layout->block_offsets[target_block_index] % 4u, 0u);
    switch_targets[switch_target_base + i] =
        code_layout->block_offsets[target_block_index] / 4u;
  }

  const iree_vm_isa_control_switch_record_t record = {
      .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_CONTROL_SWITCH,
      .selector_v8 = loom_vm_function_value_operand_register(frame, packet, 0),
      .target_count_u16 = target_dests.count,
      .target_base_u32 = switch_target_base,
  };
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write(writer, &record, sizeof(record)));

  const loom_vm_function_control_layout_t* control_layout =
      loom_vm_function_control_encoding_layout(
          code_layout->control_encodings[packet->packet_index]);
  IREE_ASSERT_EQ(control_layout->second_byte_length, 0u);
  if (control_layout->first_byte_length == 0) return iree_ok_status();
  const uint32_t default_block_index = loom_low_packet_block_index(
      &frame->schedule, loom_low_switch_default_dest(packet->node->op));
  IREE_ASSERT_NE(default_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  return loom_vm_function_encode_direct_branch(
      control_layout->first_opcode, control_layout->first_byte_length,
      /*condition_register=*/UINT16_MAX,
      code_layout->packet_offsets[packet->packet_index] + sizeof(record),
      code_layout->block_offsets[default_block_index], writer);
}

static iree_status_t loom_vm_function_encode_structural_packet(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_code_layout_t* code_layout,
    const loom_low_packet_view_t* packet, uint32_t switch_target_base,
    iree_vm_bytecode_v0_switch_target_entry_t* switch_targets,
    loom_bytecode_page_writer_t* writer) {
  if (loom_low_return_isa(packet->node->op)) {
    const iree_vm_isa_control_return_record_t record = {
        .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN,
        .zero_padding_u8 = {0, 0, 0},
    };
    return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
  }
  if (loom_low_copy_isa(packet->node->op) ||
      loom_low_move_isa(packet->node->op) ||
      loom_low_slice_isa(packet->node->op) ||
      loom_low_concat_isa(packet->node->op)) {
    const loom_low_allocation_packet_move_group_t* group =
        loom_low_allocation_find_packet_move_group_by_source_ordinal(
            &frame->allocation, packet->node->source_ordinal);
    return loom_vm_function_encode_move_group(
        &frame->allocation, group != NULL ? &group->move_group : NULL, writer);
  }
  if (loom_low_br_isa(packet->node->op) ||
      loom_low_cond_br_isa(packet->node->op)) {
    return loom_vm_function_encode_control_packet(frame, code_layout, packet,
                                                  writer);
  }
  if (loom_low_switch_isa(packet->node->op)) {
    return loom_vm_function_encode_switch(
        frame, code_layout, packet, switch_target_base, switch_targets, writer);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "VM structural operation is not implemented");
}

static iree_status_t loom_vm_function_collect_fixed_values(
    const loom_module_t* module,
    const loom_vm_module_function_layout_t* function,
    iree_arena_allocator_t* arena,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count) {
  *out_fixed_values = NULL;
  *out_fixed_value_count = 0;
  loom_func_like_t function_like =
      loom_func_like_cast(module, function->function_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function_like, &argument_count);
  if (argument_count > IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM call ABI was not materialized before emission");
  }

  const loom_region_t* body =
      loom_low_function_const_body(function->function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM function must have a non-empty body");
  }
  iree_host_size_t fixed_value_capacity = argument_count;
  for (uint16_t i = 0; i < body->block_count; ++i) {
    const loom_op_t* terminator = body->blocks[i]->last_op;
    if (terminator == NULL || !loom_low_return_isa(terminator)) continue;
    const loom_value_slice_t results = loom_low_return_values(terminator);
    if (results.count > IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM call ABI was not materialized before emission");
    }
    fixed_value_capacity += results.count;
  }

  loom_low_allocation_fixed_value_t* fixed_values = NULL;
  if (fixed_value_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, fixed_value_capacity,
                                                   sizeof(*fixed_values),
                                                   (void**)&fixed_values));
  }
  iree_host_size_t fixed_value_count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    fixed_values[fixed_value_count++] = (loom_low_allocation_fixed_value_t){
        .value_id = arguments[i],
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = i,
        .location_count = 1,
    };
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_op_t* terminator = body->blocks[block_index]->last_op;
    if (terminator == NULL || !loom_low_return_isa(terminator)) continue;
    const loom_value_slice_t results = loom_low_return_values(terminator);
    for (iree_host_size_t i = 0; i < results.count; ++i) {
      fixed_values[fixed_value_count++] = (loom_low_allocation_fixed_value_t){
          .value_id = results.values[i],
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = (uint32_t)i,
          .location_count = 1,
      };
    }
  }
  IREE_ASSERT_EQ(fixed_value_count, fixed_value_capacity);
  *out_fixed_values = fixed_values;
  *out_fixed_value_count = fixed_value_count;
  return iree_ok_status();
}

iree_status_t loom_vm_function_encode(
    loom_module_t* module, const loom_vm_module_function_layout_t* function,
    const loom_vm_function_encoder_options_t* options,
    iree_arena_allocator_t* scratch_arena, loom_bytecode_page_writer_t* writer,
    loom_vm_function_encoding_t* out_encoding) {
  *out_encoding = (loom_vm_function_encoding_t){0};
  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_vm_function_collect_fixed_values(
      module, function, scratch_arena, &fixed_values, &fixed_value_count));

  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = options->descriptor_registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_budgets = options->allocation_budgets,
      .allocation_budget_count = options->allocation_budget_count,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .emitter = options->diagnostic_emitter,
  };
  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      module, function->function_op, &frame_options, scratch_arena, &frame));
  if (frame.schedule.error_count != 0 || frame.allocation.error_count != 0) {
    return iree_ok_status();
  }
  if (frame.target.descriptor_set != loom_vm_core_descriptor_set()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM emission requires descriptor set 'vm.core'");
  }
  if (frame.allocation.spill_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM registers cannot spill");
  }
  IREE_ASSERT_EQ(frame.allocation.physical_extents.count,
                 VM_CORE_REG_CLASS_ID_FUNCTION + 1u);
  const uint32_t value_register_count =
      frame.allocation.physical_extents
          .ends_by_reg_class[VM_CORE_REG_CLASS_ID_VALUE];
  const uint32_t ref_register_count =
      frame.allocation.physical_extents
          .ends_by_reg_class[VM_CORE_REG_CLASS_ID_REF];
  const uint32_t function_register_count =
      frame.allocation.physical_extents
          .ends_by_reg_class[VM_CORE_REG_CLASS_ID_FUNCTION];
  IREE_ASSERT_LE(value_register_count, 256u);
  IREE_ASSERT_LE(ref_register_count, 256u);
  IREE_ASSERT_LE(function_register_count, 256u);

  loom_vm_function_code_layout_t code_layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_vm_function_code_layout_build(&frame, scratch_arena, &code_layout));
  if (code_layout.switch_target_entry_count !=
      function->switch_target_entry_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "VM function layout switch-target count changed during emission");
  }
  iree_vm_bytecode_v0_switch_target_entry_t* switch_targets = NULL;
  if (code_layout.switch_target_entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, code_layout.switch_target_entry_count,
        sizeof(*switch_targets), (void**)&switch_targets));
  }

  const uint64_t bytecode_start = writer->total_written;
  const iree_vm_isa_control_block_record_t block_record = {
      .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK,
      .zero_padding_u8 = {0, 0, 0},
  };
  uint32_t switch_target_base = 0;
  for (uint32_t block_index = 0; block_index < frame.schedule.block_count;
       ++block_index) {
    IREE_ASSERT_EQ(writer->total_written - bytecode_start,
                   code_layout.block_offsets[block_index]);
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, &block_record,
                                                         sizeof(block_record)));
    const loom_low_schedule_block_t* block =
        &frame.schedule.blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(&frame.schedule, block_index, i);
      IREE_ASSERT_EQ(writer->total_written - bytecode_start,
                     code_layout.packet_offsets[packet.packet_index]);
      if (loom_low_packet_is_compile_time_only(&packet)) continue;
      if (packet.descriptor != NULL &&
          packet.descriptor->carrier == LOOM_LOW_DESCRIPTOR_CARRIER_PACKET) {
        IREE_RETURN_IF_ERROR(
            loom_vm_function_encode_descriptor_packet(&frame, &packet, writer));
      } else {
        IREE_RETURN_IF_ERROR(loom_vm_function_encode_structural_packet(
            &frame, &code_layout, &packet, switch_target_base, switch_targets,
            writer));
        if (loom_low_switch_isa(packet.node->op)) {
          switch_target_base +=
              loom_low_switch_target_dests(packet.node->op).count;
        }
      }
    }
  }
  IREE_ASSERT_EQ(switch_target_base, code_layout.switch_target_entry_count);

  const uint64_t bytecode_length = writer->total_written - bytecode_start;
  if (bytecode_length != code_layout.bytecode_length) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "VM function layout and serialization byte lengths disagree");
  }
  out_encoding->row.callable_type_ordinal_u16 = function->callable_type_ordinal;
  out_encoding->row.bytecode_length_u32 = (uint32_t)bytecode_length;
  out_encoding->row.value_register_count_u16 = (uint16_t)value_register_count;
  out_encoding->row.ref_register_count_u16 = (uint16_t)ref_register_count;
  out_encoding->row.function_register_count_u16 =
      (uint16_t)function_register_count;
  out_encoding->row.block_count_u32 = (uint32_t)frame.schedule.block_count;
  out_encoding->row.switch_target_entry_count_u32 =
      code_layout.switch_target_entry_count;
  out_encoding->switch_targets = switch_targets;
  out_encoding->is_complete = true;
  return iree_ok_status();
}
