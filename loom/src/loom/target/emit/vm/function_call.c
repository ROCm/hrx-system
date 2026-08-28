// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/function_call.h"

#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "iree/vm/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors.h"

bool loom_vm_function_call_try_view(const loom_op_t* op,
                                    loom_vm_function_call_view_t* out_call) {
  *out_call = (loom_vm_function_call_view_t){0};
  if (loom_low_func_call_isa(op)) {
    *out_call = (loom_vm_function_call_view_t){
        .kind = LOOM_VM_FUNCTION_CALL_KIND_DIRECT,
        .argument_operand_base = 0,
        .arguments = loom_low_func_call_operands(op),
        .results = loom_low_func_call_results(op),
    };
    return true;
  }
  if (loom_low_func_call_indirect_isa(op)) {
    *out_call = (loom_vm_function_call_view_t){
        .kind = LOOM_VM_FUNCTION_CALL_KIND_INDIRECT,
        .argument_operand_base = 1,
        .arguments = loom_low_func_call_indirect_operands(op),
        .results = loom_low_func_call_indirect_results(op),
    };
    return true;
  }
  return false;
}

iree_status_t loom_vm_function_call_layout_build(
    const loom_module_t* module, const loom_vm_function_call_view_t* call,
    loom_vm_call_abi_packet_layout_t* out_layout) {
  return loom_vm_call_abi_packet_layout_build(
      module, call->arguments.values, call->arguments.count,
      call->results.values, call->results.count, out_layout);
}

uint32_t loom_vm_function_call_record_byte_length(
    const loom_vm_call_abi_packet_layout_t* layout) {
  return sizeof(iree_vm_isa_control_call_record_t) +
         loom_vm_call_abi_overflow_count(layout->arguments.value) *
             sizeof(iree_vm_isa_stack_store_record_t) +
         loom_vm_call_abi_overflow_count(layout->results.value) *
             sizeof(iree_vm_isa_stack_load_record_t) +
         loom_vm_call_abi_overflow_count(layout->arguments.ref) *
             sizeof(iree_vm_isa_ref_stack_store_move_record_t) +
         loom_vm_call_abi_overflow_count(layout->results.ref) *
             sizeof(iree_vm_isa_ref_stack_load_move_record_t) +
         loom_vm_call_abi_overflow_count(layout->arguments.function) *
             sizeof(iree_vm_isa_func_stack_store_record_t) +
         loom_vm_call_abi_overflow_count(layout->results.function) *
             sizeof(iree_vm_isa_func_stack_load_record_t);
}

static uint16_t* loom_vm_function_call_bank_ordinal(
    loom_vm_call_abi_bank_counts_t* ordinals, loom_vm_call_abi_bank_t bank) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return &ordinals->value;
    case LOOM_VM_CALL_ABI_BANK_REF:
      return &ordinals->ref;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      return &ordinals->function;
    default:
      IREE_ASSERT_UNREACHABLE("classified VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static uint8_t loom_vm_function_call_assigned_register(
    const loom_low_allocation_table_t* allocation,
    loom_value_ordinal_t value_ordinal, uint16_t expected_reg_class_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(allocation,
                                                       value_ordinal, NULL);
  IREE_ASSERT(assignment != NULL);
  IREE_ASSERT_EQ(assignment->location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  IREE_ASSERT_EQ(assignment->descriptor_reg_class_id, expected_reg_class_id);
  IREE_ASSERT_EQ(assignment->location_count, 1u);
  IREE_ASSERT_LT(assignment->location_base, 256u);
  return (uint8_t)assignment->location_base;
}

static uint16_t loom_vm_function_call_register_class_id(
    loom_vm_call_abi_bank_t bank) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      return VM_CORE_REG_CLASS_ID_VALUE;
    case LOOM_VM_CALL_ABI_BANK_REF:
      return VM_CORE_REG_CLASS_ID_REF;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      return VM_CORE_REG_CLASS_ID_FUNCTION;
    default:
      IREE_ASSERT_UNREACHABLE("classified VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_function_call_encode_argument_overflow(
    loom_vm_call_abi_bank_t bank, uint16_t overflow_ordinal,
    uint8_t source_register, loom_bytecode_page_writer_t* writer) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE: {
      const uint32_t byte_offset =
          (uint32_t)overflow_ordinal * sizeof(uint64_t);
      IREE_ASSERT_LE(byte_offset, UINT16_MAX);
      const iree_vm_isa_stack_store_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_STORE,
          .base_u16 = (uint16_t)byte_offset,
          .src_v8 = source_register,
          .format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1,
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    case LOOM_VM_CALL_ABI_BANK_REF: {
      const iree_vm_isa_ref_stack_store_move_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_MOVE,
          .src_r8 = source_register,
          .slot_u16 = overflow_ordinal,
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    case LOOM_VM_CALL_ABI_BANK_FUNCTION: {
      const iree_vm_isa_func_stack_store_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_STORE,
          .src_f8 = source_register,
          .local_ordinal_u16 = overflow_ordinal,
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    default:
      IREE_ASSERT_UNREACHABLE("classified VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_vm_function_call_encode_result_overflow(
    const loom_vm_call_abi_packet_layout_t* layout,
    loom_vm_call_abi_bank_t bank, uint16_t overflow_ordinal,
    uint8_t destination_register, loom_bytecode_page_writer_t* writer) {
  switch (bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE: {
      const uint32_t argument_overflow_count =
          loom_vm_call_abi_overflow_count(layout->arguments.value);
      const uint32_t byte_offset =
          (argument_overflow_count + overflow_ordinal) * sizeof(uint64_t);
      IREE_ASSERT_LE(byte_offset, UINT16_MAX);
      const iree_vm_isa_stack_load_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_STACK_LOAD,
          .dst_v8 = destination_register,
          .base_u16 = (uint16_t)byte_offset,
          .format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X1,
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    case LOOM_VM_CALL_ABI_BANK_REF: {
      const uint16_t argument_overflow_count =
          loom_vm_call_abi_overflow_count(layout->arguments.ref);
      const iree_vm_isa_ref_stack_load_move_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_MOVE,
          .dst_r8 = destination_register,
          .slot_u16 = (uint16_t)(argument_overflow_count + overflow_ordinal),
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    case LOOM_VM_CALL_ABI_BANK_FUNCTION: {
      const uint16_t argument_overflow_count =
          loom_vm_call_abi_overflow_count(layout->arguments.function);
      const iree_vm_isa_func_stack_load_record_t record = {
          .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_LOAD,
          .dst_f8 = destination_register,
          .local_ordinal_u16 =
              (uint16_t)(argument_overflow_count + overflow_ordinal),
      };
      return loom_bytecode_page_writer_write(writer, &record, sizeof(record));
    }
    default:
      IREE_ASSERT_UNREACHABLE("classified VM call ABI bank");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_vm_function_call_encode(
    const loom_low_emission_frame_t* frame,
    const loom_vm_module_layout_t* module_layout,
    const loom_low_packet_view_t* packet,
    const loom_vm_function_call_view_t* call,
    const loom_vm_call_abi_packet_layout_t* layout,
    loom_bytecode_page_writer_t* writer) {
  if (call->kind != LOOM_VM_FUNCTION_CALL_KIND_DIRECT) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM indirect calls are not implemented");
  }

  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(packet->node) +
      call->argument_operand_base;
  loom_vm_call_abi_bank_counts_t argument_ordinals = {0};
  for (uint16_t i = 0; i < call->arguments.count; ++i) {
    loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        frame->module,
        loom_module_value_type(frame->module, call->arguments.values[i]),
        &bank));
    uint16_t* bank_ordinal =
        loom_vm_function_call_bank_ordinal(&argument_ordinals, bank);
    const uint16_t ordinal = (*bank_ordinal)++;
    if (ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT) continue;
    const uint16_t expected_reg_class_id =
        loom_vm_function_call_register_class_id(bank);
    const uint8_t source_register = loom_vm_function_call_assigned_register(
        &frame->allocation, operand_ordinals[i], expected_reg_class_id);
    IREE_RETURN_IF_ERROR(loom_vm_function_call_encode_argument_overflow(
        bank, (uint16_t)(ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT),
        source_register, writer));
  }

  loom_vm_module_call_target_t target = {0};
  if (!loom_vm_module_layout_try_resolve_call_target(
          module_layout, loom_low_func_call_callee(packet->node->op),
          &target)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM runtime function imports are not implemented");
  }
  const iree_vm_isa_control_call_record_t call_record = {
      .opcode_u8 = IREE_VM_ISA_CORE_OPCODE_CONTROL_CALL,
      .target_kind_u8 = target.kind,
      .target_ordinal_u16 = target.ordinal,
      .direct_ref_move_mask_u16 = layout->direct_ref_move_mask,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, &call_record,
                                                       sizeof(call_record)));

  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(packet->node);
  loom_vm_call_abi_bank_counts_t result_bank_ordinals = {0};
  for (uint16_t i = 0; i < call->results.count; ++i) {
    loom_vm_call_abi_bank_t bank = LOOM_VM_CALL_ABI_BANK_NONE;
    IREE_RETURN_IF_ERROR(loom_vm_call_abi_classify_type(
        frame->module,
        loom_module_value_type(frame->module, call->results.values[i]), &bank));
    uint16_t* bank_ordinal =
        loom_vm_function_call_bank_ordinal(&result_bank_ordinals, bank);
    const uint16_t ordinal = (*bank_ordinal)++;
    if (ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT) continue;
    const uint16_t expected_reg_class_id =
        loom_vm_function_call_register_class_id(bank);
    const uint8_t destination_register =
        loom_vm_function_call_assigned_register(
            &frame->allocation, result_ordinals[i], expected_reg_class_id);
    IREE_RETURN_IF_ERROR(loom_vm_function_call_encode_result_overflow(
        layout, bank, (uint16_t)(ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT),
        destination_register, writer));
  }
  return iree_ok_status();
}
