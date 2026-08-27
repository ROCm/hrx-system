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

#define LOOM_VM_INSTRUCTION_ENCODING_LIMITS(maximum_record_byte_length) \
  enum {                                                                \
    LOOM_VM_MAXIMUM_INSTRUCTION_RECORD_BYTE_LENGTH =                    \
        maximum_record_byte_length,                                     \
  };
#define LOOM_VM_INSTRUCTION_ENCODING_ROW(byte_length)
#include "loom/target/arch/vm/encoding_rows.inl"
#undef LOOM_VM_INSTRUCTION_ENCODING_ROW
#undef LOOM_VM_INSTRUCTION_ENCODING_LIMITS

static const uint8_t kLoomVmInstructionRecordByteLengths[] = {
#define LOOM_VM_INSTRUCTION_ENCODING_LIMITS(maximum_record_byte_length)
#define LOOM_VM_INSTRUCTION_ENCODING_ROW(byte_length) byte_length,
#include "loom/target/arch/vm/encoding_rows.inl"
#undef LOOM_VM_INSTRUCTION_ENCODING_ROW
#undef LOOM_VM_INSTRUCTION_ENCODING_LIMITS
};

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
  IREE_ASSERT_LT(descriptor_ordinal,
                 IREE_ARRAYSIZE(kLoomVmInstructionRecordByteLengths));
  const uint8_t record_byte_length =
      kLoomVmInstructionRecordByteLengths[descriptor_ordinal];
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
    const loom_low_allocation_packet_move_group_t* group,
    loom_bytecode_page_writer_t* writer) {
  if (group == NULL) return iree_ok_status();
  const iree_host_size_t move_end =
      group->move_group.moves.start + group->move_group.moves.count;
  for (iree_host_size_t i = group->move_group.moves.start; i < move_end; ++i) {
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

static iree_status_t loom_vm_function_encode_structural_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet, loom_bytecode_page_writer_t* writer) {
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
    return loom_vm_function_encode_move_group(&frame->allocation, group,
                                              writer);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "VM structural operation is not implemented");
}

static iree_status_t loom_vm_function_validate_body(
    const loom_vm_module_function_layout_t* function,
    const loom_op_t** out_return_op) {
  const loom_region_t* body =
      loom_low_function_const_body(function->function_op);
  if (body == NULL || body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM function emission currently requires one basic block");
  }
  const loom_block_t* block = loom_region_const_entry_block(body);
  if (block->last_op == NULL || !loom_low_return_isa(block->last_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM function block must end in low.return");
  }
  *out_return_op = block->last_op;
  return iree_ok_status();
}

static iree_status_t loom_vm_function_collect_fixed_values(
    const loom_module_t* module,
    const loom_vm_module_function_layout_t* function,
    const loom_op_t* return_op, loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t* out_fixed_value_count) {
  loom_func_like_t function_like =
      loom_func_like_cast(module, function->function_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function_like, &argument_count);
  const loom_value_slice_t results = loom_low_return_values(return_op);
  if (argument_count > IREE_VM_CALL_DIRECT_REGISTER_COUNT ||
      results.count > IREE_VM_CALL_DIRECT_REGISTER_COUNT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM call ABI was not materialized before emission");
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
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    fixed_values[fixed_value_count++] = (loom_low_allocation_fixed_value_t){
        .value_id = results.values[i],
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = (uint32_t)i,
        .location_count = 1,
    };
  }
  *out_fixed_value_count = fixed_value_count;
  return iree_ok_status();
}

iree_status_t loom_vm_function_encode(
    loom_module_t* module, const loom_vm_module_function_layout_t* function,
    const loom_vm_function_encoder_options_t* options,
    iree_arena_allocator_t* scratch_arena, loom_bytecode_page_writer_t* writer,
    loom_vm_function_encoding_t* out_encoding) {
  *out_encoding = (loom_vm_function_encoding_t){0};
  const loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_function_validate_body(function, &return_op));

  loom_low_allocation_fixed_value_t
      fixed_values[IREE_VM_CALL_DIRECT_REGISTER_COUNT * 2];
  iree_host_size_t fixed_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_vm_function_collect_fixed_values(
      module, function, return_op, fixed_values, &fixed_value_count));

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

  const uint64_t bytecode_start = writer->total_written;
  const iree_vm_isa_control_block_record_t block_record = {
      .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK,
      .zero_padding_u8 = {0, 0, 0},
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, &block_record,
                                                       sizeof(block_record)));
  const iree_host_size_t packet_count = loom_low_packet_count(&frame.schedule);
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(&frame.schedule, i);
    if (loom_low_packet_is_compile_time_only(&packet)) continue;
    if (packet.descriptor != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_vm_function_encode_descriptor_packet(&frame, &packet, writer));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_vm_function_encode_structural_packet(&frame, &packet, writer));
    }
  }

  const uint64_t bytecode_length = writer->total_written - bytecode_start;
  if (bytecode_length == 0 || bytecode_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM function bytecode length exceeds u32");
  }
  out_encoding->row.callable_type_ordinal_u16 = function->callable_type_ordinal;
  out_encoding->row.bytecode_length_u32 = (uint32_t)bytecode_length;
  out_encoding->row.value_register_count_u16 = (uint16_t)value_register_count;
  out_encoding->row.ref_register_count_u16 = (uint16_t)ref_register_count;
  out_encoding->row.function_register_count_u16 =
      (uint16_t)function_register_count;
  out_encoding->row.block_count_u32 = 1;
  out_encoding->is_complete = true;
  return iree_ok_status();
}
