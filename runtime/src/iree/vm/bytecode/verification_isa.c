// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/alignment.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/verification.h"
#include "iree/vm/bytecode/verification_isa_buffer.h"
#include "iree/vm/bytecode/verification_isa_registers.h"
#include "iree/vm/bytecode/verification_isa_stack.h"
#include "iree/vm/bytecode/wire/core/abi.h"
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
#include "iree/vm/bytecode/wire/core/value.h"

typedef enum iree_vm_bytecode_verification_form_e {
  IREE_VM_BYTECODE_VERIFICATION_FORM_NONE = 0,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BLOCK,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_YIELD_S32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S16,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S16,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_SWITCH,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_CALL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_CALL_INDIRECT,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_ASSERT,
  IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_FAIL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_ARGUMENT_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_ABI_RESULT_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_ARGUMENT_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_REF_ABI_RESULT_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ABI_ARGUMENT_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ABI_RESULT_STORE,
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
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_IMMUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_IMMUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_MUTABLE_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_MUTABLE_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_VALUE_BINARY_4,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_LEA,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U32,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_CEILDIV_POW2_U64,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_BITSTREAM_PACK,
  IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_BITSTREAM_UNPACK,
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
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ALLOCATE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_LENGTH,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_SUBSPAN,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_LOAD,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_STORE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_REDUCE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_RMW,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_CMPXCHG,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_FILL,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COPY,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COMPARE,
  IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COPY_RODATA,
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
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_FROM_BUFFER,
  IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_TO_BUFFER,
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

static bool iree_vm_bytecode_integer_carrier_width_is_valid(uint8_t width) {
  return width == 8 || width == 16 || width == 32 || width == 64;
}

static iree_status_t iree_vm_bytecode_verify_integer_bitstream_record(
    const iree_vm_isa_integer_bitstream_pack_record_t* record,
    uint16_t value_register_count, bool is_pack) {
  if (record->source_count_u8 == 0 || record->result_count_u8 == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer bitstream register range is empty");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register_range(
      record->source_base_v8, record->source_count_u8, value_register_count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register_range(
      record->result_base_v8, record->result_count_u8, value_register_count));
  if (record->field_width_u8 == 0 || record->field_width_u8 > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer bitstream field width is invalid");
  }
  if (!iree_vm_bytecode_integer_carrier_width_is_valid(
          record->source_width_u8) ||
      !iree_vm_bytecode_integer_carrier_width_is_valid(
          record->result_width_u8)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer bitstream carrier width is invalid");
  }
  const uint32_t source_bit_count =
      record->source_count_u8 *
      (is_pack ? record->field_width_u8 : record->source_width_u8);
  const uint32_t result_bit_count =
      record->result_count_u8 *
      (is_pack ? record->result_width_u8 : record->field_width_u8);
  const uint8_t field_carrier_width =
      is_pack ? record->source_width_u8 : record->result_width_u8;
  if (record->field_width_u8 > field_carrier_width ||
      source_bit_count != result_bit_count || source_bit_count > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "integer bitstream shape is invalid");
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

static bool iree_vm_bytecode_control_status_is_assigned(uint8_t status) {
  return (status >= IREE_VM_ISA_CONTROL_STATUS_CANCELLED &&
          status <= IREE_VM_ISA_CONTROL_STATUS_UNAUTHENTICATED) ||
         status == IREE_VM_ISA_CONTROL_STATUS_INCOMPATIBLE;
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
      const iree_vm_bytecode_v0_callable_type_row_t* target_callable_type =
          iree_vm_bytecode_function_callable_type(layout, target);
      if (target_callable_type->signature_ordinal_u16 !=
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

// Verifies that one caller frame covers the complete implicit call packet.
static iree_status_t iree_vm_bytecode_verify_call_packet(
    const iree_vm_bytecode_v0_function_row_t* caller,
    const iree_vm_bytecode_v0_callable_type_row_t* callable_type,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    uint16_t direct_ref_move_mask) {
  const uint16_t direct_value_count =
      iree_min(16u, iree_max(signature->argument_value_count_u16,
                             signature->result_value_count_u16));
  const uint16_t direct_ref_count =
      iree_min(16u, iree_max(signature->argument_ref_count_u16,
                             signature->result_ref_count_u16));
  const uint16_t direct_function_count =
      iree_min(16u, iree_max(signature->argument_function_count_u16,
                             signature->result_function_count_u16));
  if (caller->value_register_count_u16 < direct_value_count ||
      caller->ref_register_count_u16 < direct_ref_count ||
      caller->function_register_count_u16 < direct_function_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call register banks do not cover the direct packet prefixes");
  }

  const uint16_t direct_ref_argument_count =
      iree_min(16u, signature->argument_ref_count_u16);
  const uint16_t valid_ref_move_mask =
      direct_ref_argument_count == 16
          ? UINT16_MAX
          : (uint16_t)((1u << direct_ref_argument_count) - 1u);
  if ((direct_ref_move_mask & ~valid_ref_move_mask) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "call ref move mask exceeds its argument prefix");
  }

  const uint32_t argument_value_overflow =
      signature->argument_value_count_u16 > 16
          ? (uint32_t)signature->argument_value_count_u16 - 16u
          : 0;
  const uint32_t result_value_overflow =
      signature->result_value_count_u16 > 16
          ? (uint32_t)signature->result_value_count_u16 - 16u
          : 0;
  const uint32_t required_local_bytes =
      (argument_value_overflow + result_value_overflow) * sizeof(uint64_t);
  if (required_local_bytes > caller->local_byte_length_u16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call value overflow packet exceeds local byte storage");
  }

  const uint32_t argument_ref_overflow =
      signature->argument_ref_count_u16 > 16
          ? (uint32_t)signature->argument_ref_count_u16 - 16u
          : 0;
  const uint32_t result_ref_overflow =
      signature->result_ref_count_u16 > 16
          ? (uint32_t)signature->result_ref_count_u16 - 16u
          : 0;
  uint32_t required_local_refs = argument_ref_overflow + result_ref_overflow;
  if (direct_ref_move_mask != valid_ref_move_mask) {
    required_local_refs += direct_ref_argument_count;
  }
  if (required_local_refs > caller->local_ref_count_u32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call ref overflow and scratch packet exceeds local ref storage");
  }

  const uint32_t argument_function_overflow =
      signature->argument_function_count_u16 > 16
          ? (uint32_t)signature->argument_function_count_u16 - 16u
          : 0;
  const uint32_t result_function_overflow =
      signature->result_function_count_u16 > 16
          ? (uint32_t)signature->result_function_count_u16 - 16u
          : 0;
  if (argument_function_overflow + result_function_overflow >
      caller->local_function_count_u32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call function overflow packet exceeds local function storage");
  }

  if (iree_any_bit_set(callable_type->flags_u16,
                       IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD) &&
      !iree_any_bit_set(caller->flags_u16,
                        IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-yielding function calls a potentially yielding target");
  }
  return iree_ok_status();
}

// Verifies one direct local or import call against immutable module tables.
static iree_status_t iree_vm_bytecode_verify_control_call_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* caller,
    const iree_vm_isa_control_call_record_t* record) {
  if (record->zero_padding_u16 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control.call padding is nonzero");
  }

  const iree_vm_bytecode_v0_callable_type_row_t* callable_type = NULL;
  switch (record->target_kind_u8) {
    case IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL: {
      if (record->target_ordinal_u16 >= layout->functions.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "control.call local target is out of range");
      }
      callable_type = iree_vm_bytecode_function_callable_type(
          layout, &layout->functions.rows[record->target_ordinal_u16]);
      break;
    }
    case IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT:
    case IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT: {
      if (record->target_ordinal_u16 >= layout->imports.entry_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "control.call import target is out of range");
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
            "control.call import target kind does not match its declaration");
      }
      callable_type =
          &layout->callable_types.rows[import->callable_type_ordinal_u16];
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "control.call target kind is invalid");
  }

  const iree_vm_bytecode_v0_signature_row_t* signature =
      &layout->signatures.rows[callable_type->signature_ordinal_u16];
  return iree_vm_bytecode_verify_call_packet(caller, callable_type, signature,
                                             record->direct_ref_move_mask_u16);
}

// Verifies one indirect call's static contract and packet requirements.
static iree_status_t iree_vm_bytecode_verify_control_call_indirect_record(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* caller,
    const iree_vm_isa_control_call_indirect_record_t* record) {
  if (record->zero_padding_u16 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control.call.indirect padding is nonzero");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
      record->target_f8, caller->function_register_count_u16));
  if (record->callable_type_ordinal_u16 >= layout->callable_types.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "control.call.indirect callable type is out of range");
  }
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &layout->callable_types.rows[record->callable_type_ordinal_u16];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &layout->signatures.rows[callable_type->signature_ordinal_u16];
  return iree_vm_bytecode_verify_call_packet(caller, callable_type, signature,
                                             record->direct_ref_move_mask_u16);
}

typedef struct iree_vm_bytecode_function_verification_scratch_t {
  // Caller-owned decoded control.block offsets in ascending byte order.
  uint32_t* block_offsets;
  // Number of entries available in |block_offsets|.
  uint32_t block_capacity;
} iree_vm_bytecode_function_verification_scratch_t;

static iree_status_t iree_vm_bytecode_verify_control_target(
    uint32_t record_offset, uint8_t record_length, int64_t target_word_delta,
    uint32_t function_length, uint32_t* out_target_offset) {
  const int64_t target_offset =
      (int64_t)record_offset + record_length + target_word_delta * 4;
  if (target_offset < 0 || target_offset >= function_length ||
      (target_offset & 3) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct control target is out of range");
  }
  *out_target_offset = (uint32_t)target_offset;
  return iree_ok_status();
}

static bool iree_vm_bytecode_function_has_block_at(
    const uint32_t* block_offsets, uint32_t block_count,
    uint32_t target_offset) {
  uint32_t lower = 0;
  uint32_t upper = block_count;
  while (lower < upper) {
    const uint32_t middle = lower + (upper - lower) / 2;
    const uint32_t block_offset = block_offsets[middle];
    if (target_offset < block_offset) {
      upper = middle;
    } else if (target_offset > block_offset) {
      lower = middle + 1;
    } else {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_vm_bytecode_verify_direct_targets(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function,
    const uint32_t* block_offsets) {
  const uint8_t* bytecode =
      layout->functions.bytecode_data + function->bytecode_offset_u32;
  uint32_t byte_offset = 0;
  while (byte_offset < function->bytecode_length_u32) {
    const uint8_t* record_data = bytecode + byte_offset;
    const iree_vm_bytecode_execution_info_t execution_info =
        iree_vm_bytecode_execution_info[record_data[0]];
    int64_t target_word_delta = 0;
    bool has_target = true;
    switch ((iree_vm_bytecode_verification_form_t)
                execution_info.verification_form) {
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S16:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S16:
        target_word_delta =
            (int16_t)iree_unaligned_load_le_u16(record_data + 2);
        break;
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_YIELD_S32:
        target_word_delta =
            (int32_t)iree_unaligned_load_le_u32(record_data + 4);
        break;
      default:
        has_target = false;
        break;
    }
    if (has_target) {
      uint32_t target_offset = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_control_target(
          byte_offset, execution_info.record_length, target_word_delta,
          function->bytecode_length_u32, &target_offset));
      if (!iree_vm_bytecode_function_has_block_at(
              block_offsets, function->block_count_u32, target_offset)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "direct control target does not name a decoded control.block");
      }
    }
    byte_offset += execution_info.record_length;
  }

  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets =
      layout->functions.switch_targets + function->switch_target_base_u32;
  for (uint32_t i = 0; i < function->switch_target_entry_count_u32; ++i) {
    const uint32_t target_word_offset = switch_targets[i];
    if (target_word_offset > UINT32_MAX / 4u) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "switch target is out of range");
    }
    const uint32_t target_offset = target_word_offset * 4u;
    if (target_offset >= function->bytecode_length_u32 ||
        !iree_vm_bytecode_function_has_block_at(
            block_offsets, function->block_count_u32, target_offset)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "switch target does not name a decoded control.block");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_function_verify(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t ordinal,
    iree_vm_bytecode_function_verification_scratch_t scratch) {
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(layout, function);
  const uint8_t* bytecode =
      layout->functions.bytecode_data + function->bytecode_offset_u32;
  if (bytecode[0] != IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32 " does not begin with control.block", ordinal);
  }

  iree_vm_bytecode_verification_form_t final_form =
      IREE_VM_BYTECODE_VERIFICATION_FORM_NONE;
  bool has_call = false;
  uint32_t block_count = 0;
  uint32_t byte_offset = 0;
  while (byte_offset < function->bytecode_length_u32) {
    const uint8_t* record_data = bytecode + byte_offset;
    const uint8_t opcode = record_data[0];
    const uint32_t remaining_length =
        function->bytecode_length_u32 - byte_offset;
    const iree_vm_bytecode_execution_info_t execution_info =
        iree_vm_bytecode_execution_info[opcode];
    final_form =
        (iree_vm_bytecode_verification_form_t)execution_info.verification_form;
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
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.block padding is nonzero");
        }
        if (block_count >= function->block_count_u32 ||
            block_count >= scratch.block_capacity) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "decoded control.block count exceeds its "
                                  "function declaration");
        }
        scratch.block_offsets[block_count++] = byte_offset;
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN: {
        const iree_vm_isa_control_return_record_t* record =
            (const iree_vm_isa_control_return_record_t*)record_data;
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.return padding is nonzero");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_YIELD_S32: {
        const iree_vm_isa_control_yield_s32_record_t* record =
            (const iree_vm_isa_control_yield_s32_record_t*)record_data;
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.yield.s32 padding is nonzero");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S16: {
        const iree_vm_isa_control_branch_s16_record_t* record =
            (const iree_vm_isa_control_branch_s16_record_t*)record_data;
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.branch.s16 padding is nonzero");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S32: {
        const iree_vm_isa_control_branch_s32_record_t* record =
            (const iree_vm_isa_control_branch_s32_record_t*)record_data;
        if (record->zero_padding_u8[0] != 0 ||
            record->zero_padding_u8[1] != 0 ||
            record->zero_padding_u8[2] != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.branch.s32 padding is nonzero");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S16: {
        const iree_vm_isa_control_branch_if_s16_record_t* record =
            (const iree_vm_isa_control_branch_if_s16_record_t*)record_data;
        if (record_length == remaining_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "conditional branch requires a sequential record");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->condition_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_CONDITIONAL_S32: {
        const iree_vm_isa_control_branch_if_s32_record_t* record =
            (const iree_vm_isa_control_branch_if_s32_record_t*)record_data;
        if (record_length == remaining_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "conditional branch requires a sequential record");
        }
        if (record->zero_padding_u16 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "conditional branch padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->condition_v8, function->value_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_SWITCH: {
        const iree_vm_isa_control_switch_record_t* record =
            (const iree_vm_isa_control_switch_record_t*)record_data;
        if (record_length == remaining_length) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.switch requires a default record");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->selector_v8, function->value_register_count_u16));
        if (record->target_base_u32 > function->switch_target_entry_count_u32 ||
            record->target_count_u16 > function->switch_target_entry_count_u32 -
                                           record->target_base_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.switch target slice is out of "
                                  "range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_CALL: {
        has_call = true;
        if (record_length == remaining_length) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.call requires a sequential record");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_control_call_record(
            layout, function,
            (const iree_vm_isa_control_call_record_t*)record_data));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_CALL_INDIRECT: {
        has_call = true;
        if (record_length == remaining_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "control.call.indirect requires a sequential record");
        }
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_verify_control_call_indirect_record(
                layout, function,
                (const iree_vm_isa_control_call_indirect_record_t*)
                    record_data));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_ASSERT: {
        const iree_vm_isa_control_assert_record_t* record =
            (const iree_vm_isa_control_assert_record_t*)record_data;
        if (record_length == remaining_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "control.assert requires a sequential record");
        }
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.assert padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_value_register(
            record->condition_v8, function->value_register_count_u16));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->message_r8_nullable, function->ref_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_FAIL: {
        const iree_vm_isa_control_fail_record_t* record =
            (const iree_vm_isa_control_fail_record_t*)record_data;
        if (!iree_vm_bytecode_control_status_is_assigned(record->status_u8)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.fail status is invalid");
        }
        if (record->zero_padding_u8 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "control.fail padding is nonzero");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_register(
            record->message_r8_nullable, function->ref_register_count_u16));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ABI_ARGUMENT_LOAD: {
        const iree_vm_isa_func_abi_argument_load_record_t* record =
            (const iree_vm_isa_func_abi_argument_load_record_t*)record_data;
        const uint16_t overflow_count =
            signature->argument_function_count_u16 > 16
                ? signature->argument_function_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function argument overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_FUNC_ABI_RESULT_STORE: {
        const iree_vm_isa_func_abi_result_store_record_t* record =
            (const iree_vm_isa_func_abi_result_store_record_t*)record_data;
        const uint16_t overflow_count =
            signature->result_function_count_u16 > 16
                ? signature->result_function_count_u16 - 16
                : 0;
        if (record->slot_u16 >= overflow_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function result overflow slot ordinal is out of range");
        }
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->src_f8, function->function_register_count_u16));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_IMMUTABLE_LOAD: {
        const iree_vm_isa_global_func_immutable_load_record_t* record =
            (const iree_vm_isa_global_func_immutable_load_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_function_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable function-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_IMMUTABLE_STORE: {
        const iree_vm_isa_global_func_immutable_store_record_t* record =
            (const iree_vm_isa_global_func_immutable_store_record_t*)
                record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->src_f8, function->function_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 >=
                layout->globals.header->immutable_function_count_u32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "immutable function-global ordinal is out of range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_MUTABLE_LOAD: {
        const iree_vm_isa_global_func_mutable_load_record_t* record =
            (const iree_vm_isa_global_func_mutable_load_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->dst_f8, function->function_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_function_count_u32 ||
            record->global_u16 >= layout->globals.header->function_count_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "mutable function-global ordinal is out of "
                                  "range");
        }
        break;
      }
      case IREE_VM_BYTECODE_VERIFICATION_FORM_GLOBAL_FUNC_MUTABLE_STORE: {
        const iree_vm_isa_global_func_mutable_store_record_t* record =
            (const iree_vm_isa_global_func_mutable_store_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_register(
            record->src_f8, function->function_register_count_u16));
        if (!layout->globals.header ||
            record->global_u16 <
                layout->globals.header->immutable_function_count_u32 ||
            record->global_u16 >= layout->globals.header->function_count_u32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "mutable function-global ordinal is out of "
                                  "range");
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_BITSTREAM_PACK:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_BITSTREAM_UNPACK: {
        const iree_vm_isa_integer_bitstream_pack_record_t* record =
            (const iree_vm_isa_integer_bitstream_pack_record_t*)record_data;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_integer_bitstream_record(
            record, function->value_register_count_u16,
            execution_info.verification_form ==
                IREE_VM_BYTECODE_VERIFICATION_FORM_INTEGER_BITSTREAM_PACK));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ALLOCATE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_LENGTH:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_SUBSPAN:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_LOAD:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_STORE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_REDUCE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_RMW:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_ATOMIC_CMPXCHG:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_FILL:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COPY:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COMPARE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_COPY_RODATA:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_BUFFER_RODATA_LOAD: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_buffer_record(
            layout, function,
            iree_make_const_byte_span(record_data, record_length)));
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
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_LOAD_INDEXED:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_STORE_INDEXED:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_FILL:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COMPARE:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_RODATA:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_FROM_BUFFER:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_COPY_TO_BUFFER:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_CONST_S16_I64:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I32:
      case IREE_VM_BYTECODE_VERIFICATION_FORM_STACK_PACK_I64: {
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_stack_record(
            layout, function,
            iree_make_const_byte_span(record_data, record_length)));
        break;
      }
      default:
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "generated opcode verification form is invalid");
    }
    byte_offset += (uint32_t)record_length;
  }

  if (block_count != function->block_count_u32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function %" PRIu32
                            " decoded control.block count does not match its "
                            "declaration",
                            ordinal);
  }
  if (has_call != iree_any_bit_set(function->flags_u16,
                                   IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32
        " call records do not match its function-row declaration",
        ordinal);
  }
  if (final_form != IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_RETURN &&
      final_form != IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_YIELD_S32 &&
      final_form != IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S16 &&
      final_form != IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_BRANCH_S32 &&
      final_form != IREE_VM_BYTECODE_VERIFICATION_FORM_CONTROL_FAIL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32 " falls through past its final record", ordinal);
  }
  return iree_vm_bytecode_verify_direct_targets(layout, function,
                                                scratch.block_offsets);
}

iree_status_t iree_vm_bytecode_module_verify_executable(
    const iree_vm_bytecode_module_plan_t* plan,
    iree_allocator_t scratch_allocator) {
  if (plan->layout.requirements.count != 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "extension page 0x%02" PRIx16
                            " is unavailable in this runtime",
                            plan->layout.requirements.rows[0].page_id_u16);
  }
  uint32_t maximum_block_count = 0;
  for (uint32_t i = 0; i < plan->layout.functions.count; ++i) {
    const iree_vm_bytecode_v0_function_row_t* function =
        &plan->layout.functions.rows[i];
    if ((uint64_t)function->bytecode_length_u32 > (uint64_t)PTRDIFF_MAX) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "function bytecode exceeds the host pointer-difference range");
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
    maximum_block_count =
        iree_max(maximum_block_count, function->block_count_u32);
  }

  enum { IREE_VM_BYTECODE_INLINE_BLOCK_CAPACITY = 32 };
  uint32_t inline_block_offsets[IREE_VM_BYTECODE_INLINE_BLOCK_CAPACITY];
  uint32_t* allocated_block_offsets = NULL;
  iree_status_t status = iree_ok_status();
  if (maximum_block_count > IREE_VM_BYTECODE_INLINE_BLOCK_CAPACITY) {
    status = iree_allocator_malloc_array_uninitialized(
        scratch_allocator, maximum_block_count,
        sizeof(*allocated_block_offsets), (void**)&allocated_block_offsets);
  }
  uint32_t* block_offsets = allocated_block_offsets != NULL
                                ? allocated_block_offsets
                                : inline_block_offsets;
  const iree_vm_bytecode_function_verification_scratch_t scratch = {
      .block_offsets = block_offsets,
      .block_capacity = maximum_block_count,
  };
  for (uint32_t i = 0;
       i < plan->layout.functions.count && iree_status_is_ok(status); ++i) {
    status = iree_vm_bytecode_function_verify(
        &plan->layout, &plan->layout.functions.rows[i], i, scratch);
  }
  iree_allocator_free(scratch_allocator, allocated_block_offsets);
  return status;
}
