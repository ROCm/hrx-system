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
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/value.h"

typedef enum iree_vm_bytecode_verification_form_e {
  IREE_VM_BYTECODE_VERIFICATION_FORM_NONE = 0,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BLOCK,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_ZERO,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_S16,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_UNARY_4,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_SELECT,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_BINARY_4,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_LEA,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MINMAX,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLASSIFY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLAMP,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_RODATA_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_INTEGER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_EXTEND,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_TO_INTEGER,
} iree_vm_bytecode_verification_form_t;

typedef struct iree_vm_bytecode_execution_info_t {
  // Complete encoded instruction length in bytes, or zero when unsupported.
  uint8_t record_length;
  // Handwritten verification form selected for the instruction.
  uint8_t verification_form;
} iree_vm_bytecode_execution_info_t;

#define IREE_VM_BYTECODE_EXECUTION_INFO_ROW(record_length, form) \
  {(record_length), IREE_VM_BYTECODE_VERIFICATION_FORM_##form},
static const iree_vm_bytecode_execution_info_t
    iree_vm_bytecode_execution_info[256] = {
#define IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS
#include "iree/vm/bytecode/execution_tables.inl"
#undef IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS
};
#undef IREE_VM_BYTECODE_EXECUTION_INFO_ROW

static_assert(sizeof(iree_vm_bytecode_execution_info_t) == 2,
              "VM execution rows must remain compact");
static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_execution_info) == 256,
              "VM execution table must cover every byte opcode");

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

static iree_status_t iree_vm_bytecode_verify_record_available(
    uint32_t remaining_length, iree_host_size_t required_length,
    uint8_t opcode) {
  if (required_length > remaining_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "opcode 0x%02" PRIx8 " record is truncated",
                            opcode);
  }
  return iree_ok_status();
}

// Verifies the shared opcode/dst/lhs/rhs value-register layout.
static iree_status_t iree_vm_bytecode_verify_value_binary_record(
    const uint8_t* record_data, uint16_t value_register_count) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
      record_data[1], value_register_count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
      record_data[2], value_register_count));
  return iree_vm_bytecode_verify_value_register(record_data[3],
                                                value_register_count);
}

// Verifies the shared opcode/dst/src/zero value-register layout.
static iree_status_t iree_vm_bytecode_verify_value_unary_record(
    const uint8_t* record_data, uint16_t value_register_count) {
  if (record_data[3] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value unary padding is nonzero");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
      record_data[1], value_register_count));
  return iree_vm_bytecode_verify_value_register(record_data[2],
                                                value_register_count);
}

// Verifies the shared opcode/dst/lhs/rhs/selector/zero layout.
static iree_status_t iree_vm_bytecode_verify_value_binary_selector_record(
    const uint8_t* record_data, uint16_t value_register_count,
    uint8_t maximum_selector, const char* record_name) {
  if (record_data[4] > maximum_selector) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s selector is invalid", record_name);
  }
  if (record_data[5] != 0 || record_data[6] != 0 || record_data[7] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s padding is nonzero", record_name);
  }
  return iree_vm_bytecode_verify_value_binary_record(record_data,
                                                     value_register_count);
}

iree_status_t iree_vm_bytecode_function_verify(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t ordinal) {
  const uint8_t* bytecode =
      layout->functions.bytecode_data + function->bytecode_offset_u32;
  if (bytecode[0] != IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32 " does not begin with control.block", ordinal);
  }

  uint8_t final_opcode = 0;
  uint32_t byte_offset = 0;
  while (byte_offset < function->bytecode_length_u32) {
    const uint8_t* record_data = bytecode + byte_offset;
    const uint8_t opcode = record_data[0];
    final_opcode = opcode;
    const uint32_t remaining_length =
        function->bytecode_length_u32 - byte_offset;
    const iree_vm_bytecode_execution_info_t execution_info =
        iree_vm_bytecode_execution_info[opcode];
    if (execution_info.verification_form ==
        IREE_VM_BYTECODE_VERIFICATION_FORM_NONE) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "opcode 0x%02" PRIx8 " is outside the executable subset", opcode);
    }
    const iree_host_size_t record_length = execution_info.record_length;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record_available(
        remaining_length, record_length, opcode));
    switch ((iree_vm_bytecode_verification_form_t)
                execution_info.verification_form) {
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BLOCK: {
        const iree_vm_isa_control_block_record_t* record =
            (const iree_vm_isa_control_block_record_t*)record_data;
        if (byte_offset != 0 || record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.block must be the first record with "
                                  "canonical padding");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN: {
        const iree_vm_isa_control_return_record_t* record =
            (const iree_vm_isa_control_return_record_t*)record_data;
        if (record_length != remaining_length ||
            record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "control.return must be the final record with "
              "canonical padding");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_ZERO: {
        const iree_vm_isa_constant_zero_record_t* record =
            (const iree_vm_isa_constant_zero_record_t*)record_data;
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "constant.zero padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_S16: {
        const iree_vm_isa_constant_s16_record_t* record =
            (const iree_vm_isa_constant_s16_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I32: {
        const iree_vm_isa_constant_i32_record_t* record =
            (const iree_vm_isa_constant_i32_record_t*)record_data;
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "constant.i32 padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I64: {
        const iree_vm_isa_constant_i64_record_t* record =
            (const iree_vm_isa_constant_i64_record_t*)record_data;
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "constant.i64 padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I32: {
        const iree_vm_isa_constant_pool_load_i32_record_t* record =
            (const iree_vm_isa_constant_pool_load_i32_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        if (record->pool_u16 >= layout->constants.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "constant-pool ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I64: {
        const iree_vm_isa_constant_pool_load_i64_record_t* record =
            (const iree_vm_isa_constant_pool_load_i64_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        if (record->pool_u16 >= layout->constants.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "constant-pool ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_UNARY_4: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_unary_record(
            record_data, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_SELECT: {
        const iree_vm_isa_value_select_record_t* record =
            (const iree_vm_isa_value_select_record_t*)record_data;
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "value.select padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->condition_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->true_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->false_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_LOAD: {
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_STORE: {
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_LOAD: {
        const iree_vm_isa_global_value_mutable_load_record_t* record =
            (const iree_vm_isa_global_value_mutable_load_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_value_count_u32 ||
            record->global_u16 >= layout->globals.header->value_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "mutable value-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_STORE: {
        const iree_vm_isa_global_value_mutable_store_record_t* record =
            (const iree_vm_isa_global_value_mutable_store_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_value_count_u32 ||
            record->global_u16 >= layout->globals.header->value_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "mutable value-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_BINARY_4: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_binary_record(
            record_data, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_COMPARE: {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_value_binary_selector_record(
                record_data, function->value_register_count_u16,
                IREE_VM_ISA_INTEGER_COMPARE_UGE, "integer.compare"));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_LEA: {
        const iree_vm_isa_integer_lea_i32_record_t* record =
            (const iree_vm_isa_integer_lea_i32_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "integer.lea padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->base_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->index_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U32: {
        const iree_vm_isa_integer_ceildiv_pow2_u32_record_t* record =
            (const iree_vm_isa_integer_ceildiv_pow2_u32_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (record->log2_u8 > 31) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "integer.ceildiv.pow2.u32 log2 is invalid");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U64: {
        const iree_vm_isa_integer_ceildiv_pow2_u64_record_t* record =
            (const iree_vm_isa_integer_ceildiv_pow2_u64_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        if (record->log2_u8 > 63) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "integer.ceildiv.pow2.u64 log2 is invalid");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MINMAX: {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_value_binary_selector_record(
                record_data, function->value_register_count_u16,
                IREE_VM_ISA_FLOAT_MINMAX_MAXNUM, "float.minmax"));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_COMPARE: {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_value_binary_selector_record(
                record_data, function->value_register_count_u16,
                IREE_VM_ISA_FLOAT_COMPARE_UNO, "float.compare"));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLASSIFY: {
        const iree_vm_isa_float_classify_f32_record_t* record =
            (const iree_vm_isa_float_classify_f32_record_t*)record_data;
        if (record->selector_u8 > IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.classify selector is invalid");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLAMP: {
        const iree_vm_isa_float_clamp_f32_record_t* record =
            (const iree_vm_isa_float_clamp_f32_record_t*)record_data;
        if (record->mode_u8 > IREE_VM_ISA_FLOAT_CLAMP_IEEE) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.clamp mode is invalid");
        }
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.clamp padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->value_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->lower_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->upper_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_RODATA_LOAD: {
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_INTEGER: {
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_EXTEND: {
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_TO_INTEGER: {
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
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "generated opcode verification form is invalid");
    }
    byte_offset += (uint32_t)record_length;
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
