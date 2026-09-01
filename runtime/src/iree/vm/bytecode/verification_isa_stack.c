// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification_isa_stack.h"

#include "iree/vm/bytecode/verification_isa_registers.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/stack.h"

static iree_status_t iree_vm_bytecode_verify_stack_lane_range(
    uint16_t base, uint8_t register_base, uint8_t format,
    const iree_vm_bytecode_v0_function_row_t* function) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_lane_register_range(
      register_base, format, function));
  const uint8_t access_length =
      (uint8_t)(1u << ((format >> 2) + (format & 3u)));
  return iree_vm_bytecode_verify_local_range(base, access_length,
                                             function->local_byte_length_u16);
}

static iree_status_t iree_vm_bytecode_verify_stack_fixed_range(
    uint16_t base, uint32_t length, uint8_t alignment,
    uint16_t local_byte_length) {
  if ((base & (alignment - 1u)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local byte range has invalid alignment");
  }
  return iree_vm_bytecode_verify_local_range(base, length, local_byte_length);
}

iree_status_t iree_vm_bytecode_verify_stack_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function,
    iree_const_byte_span_t record_data) {
  switch (record_data.data[0]) {
    case IREE_VM_ISA_CORE_OPCODE_STACK_LOAD: {
      const iree_vm_isa_stack_load_record_t* record =
          (const iree_vm_isa_stack_load_record_t*)record_data.data;
      if (record->zero_padding_u8[0] != 0 || record->zero_padding_u8[1] != 0 ||
          record->zero_padding_u8[2] != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.load padding is nonzero");
      }
      return iree_vm_bytecode_verify_stack_lane_range(
          record->base_u16, record->dst_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_STORE: {
      const iree_vm_isa_stack_store_record_t* record =
          (const iree_vm_isa_stack_store_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0 || record->zero_padding_u16 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.store padding is nonzero");
      }
      return iree_vm_bytecode_verify_stack_lane_range(
          record->base_u16, record->src_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_LOAD_INDEXED: {
      const iree_vm_isa_stack_load_indexed_record_t* record =
          (const iree_vm_isa_stack_load_indexed_record_t*)record_data.data;
      if (record->scale_u8 == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.load.indexed scale is zero");
      }
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.load.indexed padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->index_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_stack_lane_range(
          record->base_u16, record->dst_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_STORE_INDEXED: {
      const iree_vm_isa_stack_store_indexed_record_t* record =
          (const iree_vm_isa_stack_store_indexed_record_t*)record_data.data;
      if (record->scale_u8 == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.store.indexed scale is zero");
      }
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.store.indexed padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->index_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_stack_lane_range(
          record->base_u16, record->src_v8, record->format_u8, function);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_FILL: {
      const iree_vm_isa_stack_fill_record_t* record =
          (const iree_vm_isa_stack_fill_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.fill padding is nonzero");
      }
      if (record->pattern_width_u8 != 1 && record->pattern_width_u8 != 2 &&
          record->pattern_width_u8 != 4 && record->pattern_width_u8 != 8) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.fill pattern width is invalid");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->pattern_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_local_range(
          record->target_base_u16, record->length_u16,
          function->local_byte_length_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_COPY: {
      const iree_vm_isa_stack_copy_record_t* record =
          (const iree_vm_isa_stack_copy_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
          record->target_u16, record->length_u16,
          function->local_byte_length_u16));
      return iree_vm_bytecode_verify_local_range(
          record->source_u16, record->length_u16,
          function->local_byte_length_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_COMPARE: {
      const iree_vm_isa_stack_compare_record_t* record =
          (const iree_vm_isa_stack_compare_record_t*)record_data.data;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->dst_v8, function->value_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
          record->lhs_u16, record->length_u16,
          function->local_byte_length_u16));
      return iree_vm_bytecode_verify_local_range(
          record->rhs_u16, record->length_u16, function->local_byte_length_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_COPY_RODATA: {
      const iree_vm_isa_stack_copy_rodata_record_t* record =
          (const iree_vm_isa_stack_copy_rodata_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy.rodata padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
          record->target_u16, record->length_u16,
          function->local_byte_length_u16));
      if (record->rodata_u16 >= layout->rodata.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy.rodata ordinal is out of range");
      }
      const uint64_t rodata_length =
          layout->rodata.descriptors[record->rodata_u16].byte_length_u64;
      if (record->source_offset_u32 > rodata_length ||
          record->length_u16 > rodata_length - record->source_offset_u32) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy.rodata range is out of bounds");
      }
      return iree_ok_status();
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_COPY_FROM_BUFFER: {
      const iree_vm_isa_stack_copy_from_buffer_record_t* record =
          (const iree_vm_isa_stack_copy_from_buffer_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy.from.buffer padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
          record->target_u16, record->length_u16,
          function->local_byte_length_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      return iree_vm_bytecode_verify_value_register(
          record->source_offset_v8, function->value_register_count_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_COPY_TO_BUFFER: {
      const iree_vm_isa_stack_copy_to_buffer_record_t* record =
          (const iree_vm_isa_stack_copy_to_buffer_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.copy.to.buffer padding is nonzero");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
          record->buffer_r8, function->ref_register_count_u16));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
          record->target_offset_v8, function->value_register_count_u16));
      return iree_vm_bytecode_verify_local_range(
          record->source_u16, record->length_u16,
          function->local_byte_length_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_CONST_S16_I32:
    case IREE_VM_ISA_CORE_OPCODE_STACK_CONST_S16_I64: {
      const iree_vm_isa_stack_const_s16_i32_record_t* record =
          (const iree_vm_isa_stack_const_s16_i32_record_t*)record_data.data;
      if (record->zero_padding_u8 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.const.s16 padding is nonzero");
      }
      const uint8_t element_byte_length =
          record_data.data[0] == IREE_VM_ISA_CORE_OPCODE_STACK_CONST_S16_I32
              ? 4
              : 8;
      return iree_vm_bytecode_verify_stack_fixed_range(
          record->target_u16, (uint32_t)record->count_u16 * element_byte_length,
          element_byte_length, function->local_byte_length_u16);
    }
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X2:
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X4:
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X8:
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I64_U32_X2:
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I64_U32_X4:
    case IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I64_U32_X8: {
      if (record_data.data[1] != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stack.pack padding is nonzero");
      }
      const uint16_t target =
          ((const iree_vm_isa_stack_pack_i32_u16_x2_record_t*)record_data.data)
              ->target_u16;
      const bool is_i32 =
          record_data.data[0] ==
              IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X2 ||
          record_data.data[0] ==
              IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X4 ||
          record_data.data[0] == IREE_VM_ISA_CORE_OPCODE_STACK_PACK_I32_U16_X8;
      const uint8_t immediate_byte_length = is_i32 ? 2 : 4;
      const uint8_t destination_byte_length = is_i32 ? 4 : 8;
      const uint32_t lane_count =
          (uint32_t)(record_data.data_length - 4) / immediate_byte_length;
      return iree_vm_bytecode_verify_stack_fixed_range(
          target, lane_count * destination_byte_length, destination_byte_length,
          function->local_byte_length_u16);
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "generated stack verification form is invalid");
  }
}
