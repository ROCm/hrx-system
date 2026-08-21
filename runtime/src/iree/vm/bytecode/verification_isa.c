// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/value.h"

static iree_status_t iree_vm_bytecode_verify_value_register(
    uint8_t ordinal, uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value register ordinal is out of range");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_ref_register(
    uint8_t ordinal, uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref register ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_function_verify(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t ordinal) {
  const uint8_t* bytecode =
      layout->functions.bytecode_data + function->bytecode_offset_u32;
  const uint32_t record_count = function->bytecode_length_u32 / 4;
  if (bytecode[0] != IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32 " does not begin with control.block", ordinal);
  }

  uint8_t final_opcode = 0;
  for (uint32_t record_i = 0; record_i < record_count; ++record_i) {
    const uint8_t* record_data = bytecode + record_i * 4;
    const uint8_t opcode = record_data[0];
    final_opcode = opcode;
    switch (opcode) {
      case IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK: {
        const iree_vm_isa_control_block_record_t* record =
            (const iree_vm_isa_control_block_record_t*)record_data;
        if (record_i != 0 || record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.block must be the first record with "
                                  "canonical padding");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN: {
        const iree_vm_isa_control_return_record_t* record =
            (const iree_vm_isa_control_return_record_t*)record_data;
        if (record_i + 1 != record_count || record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "control.return must be the final record with "
              "canonical padding");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16: {
        const iree_vm_isa_constant_s16_record_t* record =
            (const iree_vm_isa_constant_s16_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_VALUE_COPY: {
        const iree_vm_isa_value_copy_record_t* record =
            (const iree_vm_isa_value_copy_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "value.copy padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_LOAD: {
        const iree_vm_isa_global_value_immutable_load_record_t* record =
            (const iree_vm_isa_global_value_immutable_load_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_value_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable value-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_STORE: {
        const iree_vm_isa_global_value_immutable_store_record_t* record =
            (const iree_vm_isa_global_value_immutable_store_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_value_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable value-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I32: {
        const iree_vm_isa_integer_add_i32_record_t* record =
            (const iree_vm_isa_integer_add_i32_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->lhs_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->rhs_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_INTEGER_MUL_I64: {
        const iree_vm_isa_integer_mul_i64_record_t* record =
            (const iree_vm_isa_integer_mul_i64_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->lhs_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->rhs_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_BUFFER_RODATA_LOAD: {
        const iree_vm_isa_buffer_rodata_load_record_t* record =
            (const iree_vm_isa_buffer_rodata_load_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        if (record->rodata_u16 >= layout->rodata.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "rodata ordinal is out of range");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER: {
        const iree_vm_isa_conversion_integer_record_t* record =
            (const iree_vm_isa_conversion_integer_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (record->selector_u8 != IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64 &&
            record->selector_u8 != IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "integer conversion selector is outside the B0 closure");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_EXTEND: {
        const iree_vm_isa_conversion_float_extend_record_t* record =
            (const iree_vm_isa_conversion_float_extend_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (record->selector_u8 != IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "float extension selector is outside the B0 closure");
        }
        break;
      }
      case IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER: {
        const iree_vm_isa_conversion_float_to_integer_record_t* record =
            (const iree_vm_isa_conversion_float_to_integer_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (record->selector_u8 != IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "float-to-integer selector is outside the B0 closure");
        }
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "opcode 0x%02" PRIx8
                                " is outside the B0 execution closure",
                                opcode);
    }
  }

  if (final_opcode != IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32 " falls through past its final record", ordinal);
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_module_verify_executable(
    const iree_vm_bytecode_module_plan_t* plan) {
  if (plan->layout.requirements.count != 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "extension page 0x%02" PRIx16
                            " is unavailable in this runtime",
                            plan->layout.requirements.rows[0].page_id_u16);
  }
  for (uint32_t i = 0; i < plan->layout.functions.count; ++i) {
    const iree_vm_bytecode_v0_function_row_t* function =
        &plan->layout.functions.rows[i];
    const iree_vm_bytecode_v0_signature_row_t* signature =
        &plan->layout.signatures.rows[function->signature_ordinal_u16];
    if (signature->argument_value_count_u16 > 16 ||
        signature->result_value_count_u16 > 16 ||
        signature->argument_ref_count_u16 > 16 ||
        signature->result_ref_count_u16 > 16 ||
        signature->argument_function_count_u16 != 0 ||
        signature->result_function_count_u16 != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "B0 bytecode functions support only direct value/ref banks");
    }
    if (function->switch_target_entry_count_u32 != 0 ||
        function->local_byte_length_u16 != 0 ||
        function->local_ref_count_u32 != 0 ||
        function->local_function_count_u32 != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "B0 bytecode functions do not support targets or local packets");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_function_verify(&plan->layout, function, i));
  }
  return iree_ok_status();
}
