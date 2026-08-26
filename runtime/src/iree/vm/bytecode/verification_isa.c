// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification.h"
#include "iree/vm/bytecode/wire/core/abi.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/float.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/stack.h"
#include "iree/vm/bytecode/wire/core/value.h"

typedef enum iree_vm_bytecode_verification_form_e {
  IREE_VM_BYTECODE_VERIFICATION_FORM_NONE = 0,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BLOCK,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_ARGUMENT_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_RESULT_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_ARGUMENT_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_RESULT_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_ZERO,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_S16,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_I64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONSTANT_POOL_LOAD_I64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_UNARY_4,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_SELECT,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_NULL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_COMPARE_NULL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_COPY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ADDRESS,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_IMPORT_RESOLVED,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_STACK_TRANSFER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_IMMUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_VALUE_MUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_IMMUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_IMMUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_MUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_MUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_BINARY_4,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_LEA,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MINMAX,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLASSIFY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_CLAMP,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_UNARY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_BINARY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_TERNARY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_CLEAR,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_COMPARE_NULL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_COMPARE_EQ,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_RETAIN,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_MOVE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_STACK_TRANSFER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_STACK_DISCARD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_RODATA_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_INTEGER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_EXTEND,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONVERSION_FLOAT_TO_INTEGER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD_INDEXED,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE_INDEXED,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_FILL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_RODATA,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I64,
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

static iree_status_t iree_vm_bytecode_verify_function_register(
    uint8_t ordinal, uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function register ordinal is out of range");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_function_local(
    uint16_t ordinal, uint32_t local_count) {
  if (ordinal >= local_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local function ordinal is out of range");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_ref_slot(uint16_t ordinal,
                                                      uint32_t slot_count) {
  if (ordinal >= slot_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local ref slot ordinal is out of range");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_value_register_range(
    uint8_t base, uint8_t count, uint16_t register_count) {
  if (base > register_count || count > register_count - base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value register range is out of bounds");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_local_range(
    uint16_t base, uint32_t length, uint16_t local_byte_length) {
  if (base > local_byte_length || length > local_byte_length - base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local byte range is out of bounds");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_stack_lane_range(
    uint16_t base, uint8_t register_base, uint8_t format,
    const iree_vm_bytecode_v0_function_row_t* function) {
  if (format > IREE_VM_ISA_MEMORY_FORMAT_I64_X8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stack memory format is invalid");
  }
  const uint8_t lane_count = (uint8_t)(1u << (format & 3u));
  const uint8_t access_length =
      (uint8_t)(1u << ((format >> 2) + (format & 3u)));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
      base, access_length, function->local_byte_length_u16));
  return iree_vm_bytecode_verify_value_register_range(
      register_base, lane_count, function->value_register_count_u16);
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

// Verifies a function address against its selected immutable declaration.
static iree_status_t iree_vm_bytecode_verify_function_address_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_isa_func_address_record_t* record) {
  if (record->zero_padding_u8 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.address padding is nonzero");
  }
  if (record->callable_type_ordinal_u16 >= layout->callable_types.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.address callable type is out of range");
  }

  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &layout->callable_types.rows[record->callable_type_ordinal_u16];
  switch (record->target_kind_u8) {
    case IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL: {
      if (record->target_ordinal_u16 >= layout->functions.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "func.address local target is out of range");
      }
      const iree_vm_bytecode_v0_function_row_t* target =
          &layout->functions.rows[record->target_ordinal_u16];
      if (target->signature_ordinal_u16 !=
              callable_type->signature_ordinal_u16 ||
          (iree_any_bit_set(target->flags_u16,
                            IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD) &&
           !iree_any_bit_set(callable_type->flags_u16,
                             IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "func.address callable contract does not match its local target");
      }
      return iree_ok_status();
    }
    case IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT:
    case IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT: {
      if (record->target_ordinal_u16 >= layout->imports.entry_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "func.address import target is out of range");
      }
      const iree_vm_bytecode_v0_import_entry_row_t* import =
          &layout->imports.entries[record->target_ordinal_u16];
      const bool is_optional = iree_any_bit_set(
          import->flags_u16, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL);
      const bool expects_optional =
          record->target_kind_u8 ==
          IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT;
      if (is_optional != expects_optional) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "func.address import target kind does not match its declaration");
      }
      if (import->callable_type_ordinal_u16 !=
          record->callable_type_ordinal_u16) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "func.address callable contract does not match its import target");
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.address target kind is invalid");
  }
}

iree_status_t iree_vm_bytecode_function_verify(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t ordinal) {
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &layout->signatures.rows[function->signature_ordinal_u16];
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_ARGUMENT_LOAD: {
        const iree_vm_isa_value_abi_argument_load_record_t* record =
            (const iree_vm_isa_value_abi_argument_load_record_t*)record_data;
        const uint16_t overflow_count =
            signature->argument_value_count_u16 > 16
                ? signature->argument_value_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "value argument overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_RESULT_STORE: {
        const iree_vm_isa_value_abi_result_store_record_t* record =
            (const iree_vm_isa_value_abi_result_store_record_t*)record_data;
        const uint16_t overflow_count =
            signature->result_value_count_u16 > 16
                ? signature->result_value_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "value result overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_ARGUMENT_LOAD: {
        const iree_vm_isa_ref_abi_argument_load_borrow_record_t* record =
            (const iree_vm_isa_ref_abi_argument_load_borrow_record_t*)
                record_data;
        const uint16_t overflow_count =
            signature->argument_ref_count_u16 > 16
                ? signature->argument_ref_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "ref argument overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_RESULT_STORE: {
        const iree_vm_isa_ref_abi_result_store_move_record_t* record =
            (const iree_vm_isa_ref_abi_result_store_move_record_t*)record_data;
        const uint16_t overflow_count =
            signature->result_ref_count_u16 > 16
                ? signature->result_ref_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "ref result overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->src_r8, function->ref_register_count_u16));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_NULL: {
        const iree_vm_isa_func_null_record_t* record =
            (const iree_vm_isa_func_null_record_t*)record_data;
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "func.null padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_COMPARE_NULL: {
        const iree_vm_isa_func_compare_null_record_t* record =
            (const iree_vm_isa_func_compare_null_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "func.compare.null padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->src_f8, function->function_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_COPY: {
        const iree_vm_isa_func_copy_record_t* record =
            (const iree_vm_isa_func_copy_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "func.copy padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->src_f8, function->function_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ADDRESS: {
        const iree_vm_isa_func_address_record_t* record =
            (const iree_vm_isa_func_address_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_function_address_record(layout, record));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_IMPORT_RESOLVED: {
        const iree_vm_isa_func_import_resolved_record_t* record =
            (const iree_vm_isa_func_import_resolved_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        if (record->import_ordinal_u16 >= layout->imports.entry_count ||
            !iree_any_bit_set(
                layout->imports.entries[record->import_ordinal_u16].flags_u16,
                IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "func.import.resolved requires an optional import ordinal");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_STACK_TRANSFER: {
        const iree_vm_isa_func_stack_load_record_t* record =
            (const iree_vm_isa_func_stack_load_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_local(
            record->local_ordinal_u16, function->local_function_count_u32));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_IMMUTABLE_LOAD: {
        const iree_vm_isa_global_ref_immutable_load_borrow_record_t* record =
            (const iree_vm_isa_global_ref_immutable_load_borrow_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_ref_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable ref-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_IMMUTABLE_STORE: {
        const iree_vm_isa_global_ref_immutable_store_move_record_t* record =
            (const iree_vm_isa_global_ref_immutable_store_move_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->src_r8, function->ref_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_ref_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable ref-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_MUTABLE_LOAD: {
        const iree_vm_isa_global_ref_mutable_load_retain_record_t* record =
            (const iree_vm_isa_global_ref_mutable_load_retain_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_ref_count_u32 ||
            record->global_u16 >= layout->globals.header->ref_count_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "mutable ref-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_REF_MUTABLE_STORE: {
        const iree_vm_isa_global_ref_mutable_store_move_record_t* record =
            (const iree_vm_isa_global_ref_mutable_store_move_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->src_r8, function->ref_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_ref_count_u32 ||
            record->global_u16 >= layout->globals.header->ref_count_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "mutable ref-global ordinal is out of range");
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_UNARY: {
        const iree_vm_isa_float_math_unary_f32_record_t* record =
            (const iree_vm_isa_float_math_unary_f32_record_t*)record_data;
        if (record->selector_u8 > IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.math.unary selector is invalid");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->src_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_BINARY: {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_value_binary_selector_record(
                record_data, function->value_register_count_u16,
                IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC,
                "float.math.binary"));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FLOAT_MATH_TERNARY: {
        const iree_vm_isa_float_math_ternary_f32_record_t* record =
            (const iree_vm_isa_float_math_ternary_f32_record_t*)record_data;
        if (record->selector_u8 != IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.math.ternary selector is invalid");
        }
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "float.math.ternary padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->a_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->b_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->c_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_CLEAR: {
        const iree_vm_isa_ref_null_record_t* record =
            (const iree_vm_isa_ref_null_record_t*)record_data;
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ref clear padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_COMPARE_NULL: {
        const iree_vm_isa_ref_compare_null_record_t* record =
            (const iree_vm_isa_ref_compare_null_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ref.compare.null padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->src_r8, function->ref_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_COMPARE_EQ: {
        const iree_vm_isa_ref_compare_eq_record_t* record =
            (const iree_vm_isa_ref_compare_eq_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->lhs_r8, function->ref_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->rhs_r8, function->ref_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_RETAIN:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_MOVE: {
        const iree_vm_isa_ref_retain_record_t* record =
            (const iree_vm_isa_ref_retain_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ref transfer padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->src_r8, function->ref_register_count_u16));
        if (execution_info.verification_form ==
                IREE_VM_BYTECODE_VERIFICATION_FORM_REF_MOVE &&
            record->dst_r8 == record->src_r8) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ref.move registers must be distinct");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_STACK_TRANSFER: {
        const iree_vm_isa_ref_stack_load_retain_record_t* record =
            (const iree_vm_isa_ref_stack_load_retain_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->dst_r8, function->ref_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_slot(
            record->slot_u16, function->local_ref_count_u32));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_REF_STACK_DISCARD: {
        const iree_vm_isa_ref_stack_discard_record_t* record =
            (const iree_vm_isa_ref_stack_discard_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ref.stack.discard padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_slot(
            record->slot_u16, function->local_ref_count_u32));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD: {
        const iree_vm_isa_stack_load_record_t* record =
            (const iree_vm_isa_stack_load_record_t*)record_data;
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.load padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_lane_range(
            record->base_u16, record->dst_v8, record->format_u8, function));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE: {
        const iree_vm_isa_stack_store_record_t* record =
            (const iree_vm_isa_stack_store_record_t*)record_data;
        if (record->zero_padding_u8 != 0 || record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.store padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_lane_range(
            record->base_u16, record->src_v8, record->format_u8, function));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD_INDEXED: {
        const iree_vm_isa_stack_load_indexed_record_t* record =
            (const iree_vm_isa_stack_load_indexed_record_t*)record_data;
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
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_lane_range(
            record->base_u16, record->dst_v8, record->format_u8, function));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE_INDEXED: {
        const iree_vm_isa_stack_store_indexed_record_t* record =
            (const iree_vm_isa_stack_store_indexed_record_t*)record_data;
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
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_lane_range(
            record->base_u16, record->src_v8, record->format_u8, function));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_FILL: {
        const iree_vm_isa_stack_fill_record_t* record =
            (const iree_vm_isa_stack_fill_record_t*)record_data;
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
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
            record->target_base_u16, record->length_u16,
            function->local_byte_length_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY: {
        const iree_vm_isa_stack_copy_record_t* record =
            (const iree_vm_isa_stack_copy_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.copy padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
            record->target_u16, record->length_u16,
            function->local_byte_length_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
            record->source_u16, record->length_u16,
            function->local_byte_length_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COMPARE: {
        const iree_vm_isa_stack_compare_record_t* record =
            (const iree_vm_isa_stack_compare_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->dst_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
            record->lhs_u16, record->length_u16,
            function->local_byte_length_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_local_range(
            record->rhs_u16, record->length_u16,
            function->local_byte_length_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_RODATA: {
        const iree_vm_isa_stack_copy_rodata_record_t* record =
            (const iree_vm_isa_stack_copy_rodata_record_t*)record_data;
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
            layout->rodata.lengths[record->rodata_u16];
        if (record->source_offset_u32 > rodata_length ||
            record->length_u16 > rodata_length - record->source_offset_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.copy.rodata range is out of bounds");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I64: {
        const iree_vm_isa_stack_const_s16_i32_record_t* record =
            (const iree_vm_isa_stack_const_s16_i32_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.const.s16 padding is nonzero");
        }
        const uint8_t element_byte_length =
            execution_info.verification_form ==
                    IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I32
                ? 4
                : 8;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_fixed_range(
            record->target_u16,
            (uint32_t)record->count_u16 * element_byte_length,
            element_byte_length, function->local_byte_length_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I64: {
        if (record_data[1] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack.pack padding is nonzero");
        }
        const uint16_t target_u16 =
            ((const iree_vm_isa_stack_pack_i32_u16_x2_record_t*)record_data)
                ->target_u16;
        const bool is_i32 = execution_info.verification_form ==
                            IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I32;
        const uint8_t immediate_byte_length = is_i32 ? 2 : 4;
        const uint8_t destination_byte_length = is_i32 ? 4 : 8;
        const uint32_t lane_count =
            (uint32_t)(record_length - 4) / immediate_byte_length;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_fixed_range(
            target_u16, lane_count * destination_byte_length,
            destination_byte_length, function->local_byte_length_u16));
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
    if (signature->argument_function_count_u16 > 16 ||
        signature->result_function_count_u16 > 16) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "function argument and result overflow is not executable");
    }
    if (function->switch_target_entry_count_u32 != 0) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "direct control targets are not executable");
    }
    if (function->local_ref_count_u32 > (uint32_t)UINT16_MAX + 1u) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "B0 bytecode functions support only direct local ref slots");
    }
    if (function->local_function_count_u32 > (uint32_t)UINT16_MAX + 1u) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "bytecode functions support only direct local function slots");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_function_verify(&plan->layout, function, i));
  }
  return iree_ok_status();
}
