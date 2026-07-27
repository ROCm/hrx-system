// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/bitcode_writer.h"

#include <string.h>

#include "loom/target/emit/llvmir/bitcode_format.h"
#include "loom/target/emit/llvmir/bitcode_record_writer.h"
#include "loom/target/emit/llvmir/types.h"

#define LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID UINT64_MAX
#define LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT (UINT64_MAX - 1)

static uint64_t loom_llvmir_bitcode_function_type_id(
    const loom_llvmir_module_t* module, iree_host_size_t function_ordinal) {
  return module->type_count + function_ordinal;
}

static uint64_t loom_llvmir_bitcode_function_value_id(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id) {
  return module->global_count + function_id;
}

static uint64_t loom_llvmir_bitcode_first_module_constant_value_id(
    const loom_llvmir_module_t* module) {
  return module->global_count + module->function_count;
}

static iree_status_t loom_llvmir_bitcode_write_record_u64(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    uint64_t operand) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(writer, code,
                                                                 &operand, 1);
}

static iree_status_t loom_llvmir_bitcode_write_empty_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(writer, code,
                                                                 NULL, 0);
}

static uint64_t loom_llvmir_bitcode_encode_signed(int64_t value) {
  if (value >= 0) return ((uint64_t)value) << 1;
  return (((uint64_t)(-value)) << 1) | 1;
}

static bool loom_llvmir_bitcode_is_constant_value_kind(
    loom_llvmir_value_kind_t kind) {
  return kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_NULL ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_POISON;
}

static bool loom_llvmir_bitcode_value_is_global_initializer(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    if (module->globals[i]->initializer == value_id) return true;
  }
  return false;
}

static iree_host_size_t loom_llvmir_bitcode_module_constant_count(
    const loom_llvmir_module_t* module) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    if (!loom_llvmir_bitcode_is_constant_value_kind(module->values[i].kind)) {
      continue;
    }
    if (loom_llvmir_bitcode_value_is_global_initializer(
            module, (loom_llvmir_value_id_t)i)) {
      count += 1;
    }
  }
  return count;
}

static iree_status_t loom_llvmir_bitcode_module_constant_value_id(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id,
    uint64_t* out_bitcode_value_id) {
  uint64_t bitcode_value_id =
      loom_llvmir_bitcode_first_module_constant_value_id(module);
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    if (!loom_llvmir_bitcode_is_constant_value_kind(module->values[i].kind)) {
      continue;
    }
    if (!loom_llvmir_bitcode_value_is_global_initializer(
            module, (loom_llvmir_value_id_t)i)) {
      continue;
    }
    if (i == value_id) {
      *out_bitcode_value_id = bitcode_value_id;
      return iree_ok_status();
    }
    bitcode_value_id += 1;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "LLVM bitcode value is not a module constant");
}

static uint64_t loom_llvmir_bitcode_sign_extend_integer_constant(
    uint64_t value, uint32_t bit_width) {
  if (bit_width == 0 || bit_width >= 64) return value;
  uint64_t sign_bit = UINT64_C(1) << (bit_width - 1);
  uint64_t value_mask = (UINT64_C(1) << bit_width) - 1;
  uint64_t masked_value = value & value_mask;
  if ((masked_value & sign_bit) == 0) return masked_value;
  return masked_value | ~value_mask;
}

static bool loom_llvmir_bitcode_is_power_of_two_u64(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_llvmir_bitcode_check_attr_alignment(
    uint64_t alignment) {
  static const uint64_t kMaxAlignment = UINT64_C(1) << 32;
  if (!loom_llvmir_bitcode_is_power_of_two_u64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode align attribute is not a power of "
                            "two");
  }
  if (alignment > kMaxAlignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode align attribute is too large");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_encode_alignment(
    uint32_t alignment, uint64_t* out_encoded_alignment) {
  if (alignment == 0) {
    *out_encoded_alignment = 0;
    return iree_ok_status();
  }
  if (!loom_llvmir_bitcode_is_power_of_two_u64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode memory alignment is not a power of "
                            "two");
  }
  uint64_t exponent = 0;
  uint32_t remaining_alignment = alignment;
  while (remaining_alignment > 1) {
    remaining_alignment >>= 1;
    exponent += 1;
  }
  *out_encoded_alignment = exponent + 1;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_pack_alloca_flags(
    uint32_t alignment, uint64_t* out_packed_flags) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_encode_alignment(alignment, &encoded_alignment));
  *out_packed_flags =
      (encoded_alignment & LOOM_LLVMIR_BITCODE_ALLOCA_ALIGN_LOWER_MASK) |
      (UINT64_C(1) << LOOM_LLVMIR_BITCODE_ALLOCA_EXPLICIT_TYPE_BIT) |
      ((encoded_alignment >> 5)
       << LOOM_LLVMIR_BITCODE_ALLOCA_ALIGN_UPPER_SHIFT);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_memory_volatile(
    uint32_t flags, uint64_t* out_is_volatile) {
  if ((flags & ~LOOM_LLVMIR_MEMORY_VOLATILE) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "LLVM bitcode memory flag emission is not "
                            "implemented");
  }
  *out_is_volatile = (flags & LOOM_LLVMIR_MEMORY_VOLATILE) ? 1 : 0;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_atomic_rmw_op(
    loom_llvmir_atomic_rmw_op_t op, uint64_t* out_op) {
  switch (op) {
    case LOOM_LLVMIR_ATOMIC_RMW_XCHG:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_XCHG;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_ADD:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_ADD;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_SUB:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_SUB;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_AND:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_AND;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_OR:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_OR;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_XOR:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_XOR;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_MAX:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_MAX;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_MIN:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_MIN;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_UMAX:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_UMAX;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_UMIN:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_UMIN;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_FADD:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_FADD;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_FMAX:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_FMAX;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_FMIN:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_FMIN;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_FMAXIMUM:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_FMAXIMUM;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_RMW_FMINIMUM:
      *out_op = LOOM_LLVMIR_BITCODE_RMW_FMINIMUM;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM atomicrmw op");
  }
}

static iree_status_t loom_llvmir_bitcode_atomic_ordering(
    loom_llvmir_atomic_ordering_t ordering, uint64_t* out_ordering) {
  switch (ordering) {
    case LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC:
      *out_ordering = LOOM_LLVMIR_BITCODE_ORDERING_MONOTONIC;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQUIRE:
      *out_ordering = LOOM_LLVMIR_BITCODE_ORDERING_ACQUIRE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_ORDERING_RELEASE:
      *out_ordering = LOOM_LLVMIR_BITCODE_ORDERING_RELEASE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQ_REL:
      *out_ordering = LOOM_LLVMIR_BITCODE_ORDERING_ACQ_REL;
      return iree_ok_status();
    case LOOM_LLVMIR_ATOMIC_ORDERING_SEQ_CST:
      *out_ordering = LOOM_LLVMIR_BITCODE_ORDERING_SEQ_CST;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM atomic ordering");
  }
}

static iree_status_t loom_llvmir_bitcode_append_string_ref(
    iree_string_view_t string, uint64_t* operands,
    iree_host_size_t* inout_operand_count) {
  if (string.size > 0 && string.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode string attribute has null storage");
  }
  for (iree_host_size_t i = 0; i < string.size; ++i) {
    operands[(*inout_operand_count)++] = (uint8_t)string.data[i];
  }
  operands[(*inout_operand_count)++] = 0;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_attr_operand_count(
    const loom_llvmir_module_t* module, const loom_llvmir_attr_t* attr,
    iree_host_size_t* out_operand_count) {
  switch (attr->kind) {
    case LOOM_LLVMIR_ATTR_ALIGN: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_check_attr_alignment(attr->value));
      *out_operand_count = 3;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_NOALIAS:
    case LOOM_LLVMIR_ATTR_READONLY:
    case LOOM_LLVMIR_ATTR_WRITEONLY:
    case LOOM_LLVMIR_ATTR_READNONE:
    case LOOM_LLVMIR_ATTR_NOUNDEF:
    case LOOM_LLVMIR_ATTR_NONNULL:
    case LOOM_LLVMIR_ATTR_INREG:
    case LOOM_LLVMIR_ATTR_ALWAYSINLINE:
    case LOOM_LLVMIR_ATTR_NOCAPTURE:
    case LOOM_LLVMIR_ATTR_IMMARG:
      *out_operand_count = 2;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_RANGE: {
      if (attr->type_id >= module->type_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM range attribute references unknown type");
      }
      const loom_llvmir_type_t* type = &module->types[attr->type_id];
      if (type->kind != LOOM_LLVMIR_TYPE_INTEGER || type->bit_width > 64) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode range attribute emission is not implemented for this "
            "type");
      }
      *out_operand_count = 5;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_STRING_KEY:
      if (iree_string_view_is_empty(attr->key)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM string attribute key must not be empty");
      }
      if (attr->key.size > IREE_HOST_SIZE_MAX - 2) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "LLVM bitcode string attribute operand count overflows");
      }
      *out_operand_count = attr->key.size + 2;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_STRING_KEY_VALUE:
      if (iree_string_view_is_empty(attr->key)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM string key/value attribute key must not be empty");
      }
      if (attr->key.size > IREE_HOST_SIZE_MAX - attr->string_value.size ||
          attr->key.size + attr->string_value.size > IREE_HOST_SIZE_MAX - 3) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "LLVM bitcode string attribute operand count overflows");
      }
      *out_operand_count = attr->key.size + attr->string_value.size + 3;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM attribute kind");
  }
}

static iree_status_t loom_llvmir_bitcode_attr_kind(loom_llvmir_attr_kind_t kind,
                                                   uint64_t* out_attr_kind) {
  switch (kind) {
    case LOOM_LLVMIR_ATTR_ALIGN:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_ALIGNMENT;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NOALIAS:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NO_ALIAS;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_READONLY:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_READ_ONLY;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_WRITEONLY:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_WRITEONLY;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_READNONE:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_READ_NONE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NOUNDEF:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NOUNDEF;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NONNULL:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NON_NULL;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_INREG:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_IN_REG;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_ALWAYSINLINE:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_ALWAYS_INLINE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NOCAPTURE:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NO_CAPTURE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_IMMARG:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_IMMARG;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_RANGE:
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_RANGE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM enum attribute kind");
  }
}

static iree_status_t loom_llvmir_bitcode_append_attr_operands(
    const loom_llvmir_module_t* module, const loom_llvmir_attr_t* attr,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_attr_operand_count(module, attr, &operand_count));
  (void)operand_count;
  switch (attr->kind) {
    case LOOM_LLVMIR_ATTR_ALIGN: {
      uint64_t attr_kind = 0;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_attr_kind(attr->kind, &attr_kind));
      operands[(*inout_operand_count)++] = 1;
      operands[(*inout_operand_count)++] = attr_kind;
      operands[(*inout_operand_count)++] = attr->value;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_NOALIAS:
    case LOOM_LLVMIR_ATTR_READONLY:
    case LOOM_LLVMIR_ATTR_WRITEONLY:
    case LOOM_LLVMIR_ATTR_READNONE:
    case LOOM_LLVMIR_ATTR_NOUNDEF:
    case LOOM_LLVMIR_ATTR_NONNULL:
    case LOOM_LLVMIR_ATTR_INREG:
    case LOOM_LLVMIR_ATTR_NOCAPTURE:
    case LOOM_LLVMIR_ATTR_IMMARG:
    case LOOM_LLVMIR_ATTR_ALWAYSINLINE: {
      uint64_t attr_kind = 0;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_attr_kind(attr->kind, &attr_kind));
      operands[(*inout_operand_count)++] = 0;
      operands[(*inout_operand_count)++] = attr_kind;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_RANGE: {
      uint64_t attr_kind = 0;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_attr_kind(attr->kind, &attr_kind));
      const loom_llvmir_type_t* type = &module->types[attr->type_id];
      operands[(*inout_operand_count)++] = 7;
      operands[(*inout_operand_count)++] = attr_kind;
      operands[(*inout_operand_count)++] = type->bit_width;
      uint64_t lower = loom_llvmir_bitcode_sign_extend_integer_constant(
          attr->value, type->bit_width);
      uint64_t upper = loom_llvmir_bitcode_sign_extend_integer_constant(
          attr->value2, type->bit_width);
      operands[(*inout_operand_count)++] =
          loom_llvmir_bitcode_encode_signed((int64_t)lower);
      operands[(*inout_operand_count)++] =
          loom_llvmir_bitcode_encode_signed((int64_t)upper);
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_STRING_KEY:
      operands[(*inout_operand_count)++] = 3;
      return loom_llvmir_bitcode_append_string_ref(attr->key, operands,
                                                   inout_operand_count);
    case LOOM_LLVMIR_ATTR_STRING_KEY_VALUE:
      operands[(*inout_operand_count)++] = 4;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_append_string_ref(
          attr->key, operands, inout_operand_count));
      return loom_llvmir_bitcode_append_string_ref(attr->string_value, operands,
                                                   inout_operand_count);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM attribute kind");
  }
}

static iree_status_t loom_llvmir_bitcode_write_function_type(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_allocator_t allocator = iree_allocator_system();
  if (function->parameter_count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM function type operand count overflows");
  }
  iree_host_size_t operand_count = function->parameter_count + 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM function type operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator, operand_count * sizeof(*operands), (void**)&operands));
  }

  operands[0] = 0;
  operands[1] = function->return_type;
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    operands[i + 2] = function->parameters[i].type_id;
  }
  iree_status_t status =
      loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FUNCTION, operands,
          operand_count);

  if (operands != stack_operands) {
    iree_allocator_free(allocator, operands);
  }
  return status;
}

static iree_host_size_t loom_llvmir_bitcode_count_inline_asm_types(
    const loom_llvmir_module_t* module) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        if (block->instructions[k].kind == LOOM_LLVMIR_INST_INLINE_ASM) {
          count += 1;
        }
      }
    }
  }
  return count;
}

static iree_status_t loom_llvmir_bitcode_write_inline_asm_function_type(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->inline_asm.arg_count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "LLVM inline asm function type operand count overflows");
  }
  iree_host_size_t operand_count = instruction->inline_asm.arg_count + 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "LLVM inline asm function type operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  operands[0] = 0;
  operands[1] = instruction->inline_asm.result_type;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < instruction->inline_asm.arg_count && iree_status_is_ok(status);
       ++i) {
    loom_llvmir_value_id_t value_id = instruction->inline_asm.args[i];
    if (value_id >= module->value_count) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM inline asm function type references unknown argument");
    } else {
      operands[i + 2] = module->values[value_id].type_id;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FUNCTION, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_inline_asm_function_type_id(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* target_instruction,
    uint64_t* out_type_id) {
  uint64_t type_id = module->type_count + module->function_count;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (instruction->kind != LOOM_LLVMIR_INST_INLINE_ASM) continue;
        if (instruction == target_instruction) {
          *out_type_id = type_id;
          return iree_ok_status();
        }
        type_id += 1;
      }
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "LLVM inline asm instruction is not owned by module");
}

static iree_status_t loom_llvmir_bitcode_write_type_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_host_size_t inline_asm_type_count =
      loom_llvmir_bitcode_count_inline_asm_types(module);
  if (module->function_count > IREE_HOST_SIZE_MAX - module->type_count ||
      inline_asm_type_count >
          IREE_HOST_SIZE_MAX - module->type_count - module->function_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode type table is too large");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_TYPE_BLOCK,
      LOOM_LLVMIR_BITCODE_TYPE_ABBREV_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
      writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_NUMENTRY,
      module->type_count + module->function_count + inline_asm_type_count));

  for (iree_host_size_t i = 0; i < module->type_count; ++i) {
    const loom_llvmir_type_t* type = &module->types[i];
    switch (type->kind) {
      case LOOM_LLVMIR_TYPE_VOID: {
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VOID, NULL, 0));
        break;
      }
      case LOOM_LLVMIR_TYPE_INTEGER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_INTEGER, type->bit_width));
        break;
      }
      case LOOM_LLVMIR_TYPE_FLOAT: {
        switch (type->float_kind) {
          case LOOM_LLVMIR_FLOAT_F16: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_HALF, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_BF16: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_BFLOAT, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F32: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FLOAT, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F64: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_DOUBLE, NULL, 0));
            break;
          }
        }
        break;
      }
      case LOOM_LLVMIR_TYPE_POINTER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_OPAQUE_POINTER,
            type->address_space));
        break;
      }
      case LOOM_LLVMIR_TYPE_VECTOR: {
        uint64_t operands[] = {type->element_count, type->element_type};
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VECTOR, operands,
                IREE_ARRAYSIZE(operands)));
        break;
      }
      case LOOM_LLVMIR_TYPE_STRUCT: {
        iree_host_size_t operand_count = 1 + type->element_count;
        uint64_t stack_operands[8];
        uint64_t* operands = stack_operands;
        if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
          if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "LLVM bitcode struct operand storage "
                                    "overflows");
          }
          IREE_RETURN_IF_ERROR(iree_allocator_malloc(
              iree_allocator_system(), operand_count * sizeof(*operands),
              (void**)&operands));
        }
        operands[0] = 0;
        for (uint32_t j = 0; j < type->element_count; ++j) {
          operands[j + 1] = type->element_types[j];
        }
        iree_status_t status =
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_STRUCT_ANON, operands,
                operand_count);
        if (operands != stack_operands) {
          iree_allocator_free(iree_allocator_system(), operands);
        }
        IREE_RETURN_IF_ERROR(status);
        break;
      }
    }
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_write_function_type(module->functions[i], writer));
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (instruction->kind != LOOM_LLVMIR_INST_INLINE_ASM) continue;
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_inline_asm_function_type(
            module, instruction, writer));
      }
    }
  }

  return loom_llvmir_bitcode_record_writer_exit_block(writer);
}

static bool loom_llvmir_bitcode_attr_list_has_attrs(
    loom_llvmir_attr_list_t attrs) {
  return attrs.attr_count != 0;
}

static bool loom_llvmir_bitcode_function_has_parameter_attrs(
    const loom_llvmir_function_t* function) {
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (loom_llvmir_bitcode_attr_list_has_attrs(
            function->parameters[i].attrs)) {
      return true;
    }
  }
  return false;
}

static bool loom_llvmir_bitcode_function_has_attrs(
    const loom_llvmir_function_t* function) {
  return function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID ||
         loom_llvmir_bitcode_function_has_parameter_attrs(function);
}

static iree_status_t loom_llvmir_bitcode_function_attr_list_id(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    uint64_t* out_attr_list_id) {
  if (!loom_llvmir_bitcode_function_has_attrs(function)) {
    *out_attr_list_id = 0;
    return iree_ok_status();
  }
  uint64_t attr_list_id = 1;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (module->functions[i] == function) {
      *out_attr_list_id = attr_list_id;
      return iree_ok_status();
    }
    if (loom_llvmir_bitcode_function_has_attrs(module->functions[i])) {
      attr_list_id += 1;
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "LLVM bitcode function is not owned by module");
}

static bool loom_llvmir_bitcode_call_has_result_attrs(
    const loom_llvmir_instruction_t* instruction) {
  return instruction->kind == LOOM_LLVMIR_INST_CALL &&
         instruction->call.result_attrs.attr_count != 0;
}

static iree_status_t loom_llvmir_bitcode_call_attr_list_id(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* target_instruction,
    uint64_t* out_attr_list_id) {
  if (!loom_llvmir_bitcode_call_has_result_attrs(target_instruction)) {
    *out_attr_list_id = 0;
    return iree_ok_status();
  }
  uint64_t attr_list_id = 1;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (loom_llvmir_bitcode_function_has_attrs(module->functions[i])) {
      attr_list_id += 1;
    }
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* current_function = module->functions[i];
    for (iree_host_size_t j = 0; j < current_function->block_count; ++j) {
      const loom_llvmir_block_t* block = current_function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (instruction == target_instruction) {
          *out_attr_list_id = attr_list_id;
          return iree_ok_status();
        }
        if (loom_llvmir_bitcode_call_has_result_attrs(instruction)) {
          attr_list_id += 1;
        }
      }
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "LLVM bitcode call is not owned by module");
}

static bool loom_llvmir_bitcode_module_has_attrs(
    const loom_llvmir_module_t* module) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    if (loom_llvmir_bitcode_function_has_attrs(function)) return true;
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        if (loom_llvmir_bitcode_call_has_result_attrs(
                &block->instructions[k])) {
          return true;
        }
      }
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_attr_list_operand_count(
    const loom_llvmir_module_t* module, loom_llvmir_attr_list_t attrs,
    iree_host_size_t* out_operand_count) {
  iree_host_size_t operand_count = 0;
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    iree_host_size_t attr_operand_count = 0;
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_attr_operand_count(
        module, &attrs.attrs[i], &attr_operand_count));
    if (operand_count > IREE_HOST_SIZE_MAX - attr_operand_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode attribute group operand count "
                              "overflows");
    }
    operand_count += attr_operand_count;
  }
  *out_operand_count = operand_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_write_attr_group_record(
    const loom_llvmir_module_t* module, loom_llvmir_attr_list_t attrs,
    uint64_t group_id, uint64_t attribute_index,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_host_size_t attr_operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_attr_list_operand_count(
      module, attrs, &attr_operand_count));
  if (attr_operand_count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode attribute group operand count "
                            "overflows");
  }
  iree_host_size_t operand_capacity = 2 + attr_operand_count;
  uint64_t stack_operands[17];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode attribute group operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = group_id;
  operands[operand_count++] = attribute_index;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < attrs.attr_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_append_attr_operands(module, &attrs.attrs[i],
                                                      operands, &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_GROUP_CODE_ENTRY, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_attr_group_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_attrs(module)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_PARAMETERATTR_GROUP_BLOCK,
      LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_GROUP_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  uint64_t group_id = 1;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    if (function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID) {
      status = loom_llvmir_bitcode_write_attr_group_record(
          module, module->attr_groups[function->attr_group_id].attrs, group_id,
          UINT32_MAX, writer);
      if (iree_status_is_ok(status)) {
        group_id += 1;
      }
    }
    for (iree_host_size_t j = 0;
         j < function->parameter_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_parameter_t* parameter = &function->parameters[j];
      if (!loom_llvmir_bitcode_attr_list_has_attrs(parameter->attrs)) continue;
      status = loom_llvmir_bitcode_write_attr_group_record(
          module, parameter->attrs, group_id, j + 1, writer);
      if (iree_status_is_ok(status)) {
        group_id += 1;
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0;
         j < function->block_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0;
           k < block->instruction_count && iree_status_is_ok(status); ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (!loom_llvmir_bitcode_call_has_result_attrs(instruction)) continue;
        status = loom_llvmir_bitcode_write_attr_group_record(
            module, instruction->call.result_attrs, group_id, 0, writer);
        if (iree_status_is_ok(status)) {
          group_id += 1;
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static iree_host_size_t loom_llvmir_bitcode_count_function_attr_groups(
    const loom_llvmir_function_t* function) {
  iree_host_size_t count =
      function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID ? 1 : 0;
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (loom_llvmir_bitcode_attr_list_has_attrs(
            function->parameters[i].attrs)) {
      count += 1;
    }
  }
  return count;
}

static iree_status_t loom_llvmir_bitcode_write_attr_record(
    iree_host_size_t operand_count, uint64_t* operands,
    loom_llvmir_bitcode_record_writer_t* writer) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_CODE_ENTRY, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_function_attr_record(
    const loom_llvmir_function_t* function, uint64_t* inout_group_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_host_size_t operand_count =
      loom_llvmir_bitcode_count_function_attr_groups(function);
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode attribute list operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t written_operand_count = 0;
  if (function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID) {
    operands[written_operand_count++] = (*inout_group_id)++;
  }
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (!loom_llvmir_bitcode_attr_list_has_attrs(
            function->parameters[i].attrs)) {
      continue;
    }
    operands[written_operand_count++] = (*inout_group_id)++;
  }
  iree_status_t status = loom_llvmir_bitcode_write_attr_record(
      written_operand_count, operands, writer);
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_attr_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_attrs(module)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_PARAMETERATTR_BLOCK,
      LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  uint64_t group_id = 1;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    if (!loom_llvmir_bitcode_function_has_attrs(function)) continue;
    status = loom_llvmir_bitcode_write_function_attr_record(function, &group_id,
                                                            writer);
  }
  uint64_t call_group_id = group_id;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0;
         j < function->block_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0;
           k < block->instruction_count && iree_status_is_ok(status); ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (!loom_llvmir_bitcode_call_has_result_attrs(instruction)) continue;
        uint64_t operand = call_group_id++;
        status = loom_llvmir_bitcode_write_attr_record(1, &operand, writer);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

typedef struct loom_llvmir_bitcode_function_value_map_t {
  // Module-global value ID to LLVM bitcode value ID map.
  uint64_t* value_ids;
  // Number of entries in |value_ids|.
  iree_host_size_t value_count;
  // First function-local constant value ID after function parameters.
  uint64_t first_constant_value_id;
  // First function-local inline asm constant value ID after literal constants.
  uint64_t first_inline_asm_value_id;
  // First instruction value ID after this function's parameters.
  uint64_t first_instruction_value_id;
} loom_llvmir_bitcode_function_value_map_t;

static iree_status_t loom_llvmir_bitcode_check_value_id_range(
    uint64_t value_id) {
  if (value_id > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode value id exceeds 32-bit range");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_map_set_value(
    loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t bitcode_value_id) {
  if (value_id >= map->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode value map saw unknown value");
  }
  if (map->value_ids[value_id] != LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode value map saw duplicate value");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(bitcode_value_id));
  map->value_ids[value_id] = bitcode_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_map_mark_constant(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id) {
  if (value_id >= module->value_count || value_id >= map->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode constant scan saw unknown value");
  }
  if (!loom_llvmir_bitcode_is_constant_value_kind(
          module->values[value_id].kind)) {
    return iree_ok_status();
  }
  if (map->value_ids[value_id] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    map->value_ids[value_id] = LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_map_mark_instruction_constants(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_function_value_map_t* map,
    const loom_llvmir_instruction_t* instruction) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI: {
      for (iree_host_size_t i = 0; i < instruction->phi.incoming_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->phi.incoming[i].value));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_BINOP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->binop.lhs));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->binop.rhs);
    }
    case LOOM_LLVMIR_INST_UNOP:
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->unop.value);
    case LOOM_LLVMIR_INST_ICMP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->icmp.lhs));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->icmp.rhs);
    }
    case LOOM_LLVMIR_INST_FCMP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->fcmp.lhs));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->fcmp.rhs);
    }
    case LOOM_LLVMIR_INST_SELECT: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->select.condition));
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->select.true_value));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->select.false_value);
    }
    case LOOM_LLVMIR_INST_CAST:
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->cast.value);
    case LOOM_LLVMIR_INST_GEP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->gep.base));
      for (iree_host_size_t i = 0; i < instruction->gep.index_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->gep.indices[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_ALLOCA:
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->alloca.count);
    case LOOM_LLVMIR_INST_LOAD:
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->load.pointer);
    case LOOM_LLVMIR_INST_STORE: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->store.value));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->store.pointer);
    }
    case LOOM_LLVMIR_INST_ATOMIC_RMW: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->atomic_rmw.value));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->atomic_rmw.pointer);
    }
    case LOOM_LLVMIR_INST_CMPXCHG: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->cmpxchg.expected));
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->cmpxchg.replacement));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->cmpxchg.pointer);
    }
    case LOOM_LLVMIR_INST_EXTRACT_ELEMENT: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->extract_element.vector));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->extract_element.index);
    }
    case LOOM_LLVMIR_INST_EXTRACT_VALUE:
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->extract_value.aggregate);
    case LOOM_LLVMIR_INST_INSERT_ELEMENT: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->insert_element.vector));
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->insert_element.element));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->insert_element.index);
    }
    case LOOM_LLVMIR_INST_SHUFFLE_VECTOR: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->shuffle_vector.lhs));
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->shuffle_vector.rhs));
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->shuffle_vector.mask);
    }
    case LOOM_LLVMIR_INST_CALL: {
      for (iree_host_size_t i = 0; i < instruction->call.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->call.args[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_INLINE_ASM: {
      for (iree_host_size_t i = 0; i < instruction->inline_asm.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->inline_asm.args[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_RET:
      if (!instruction->ret.has_value) return iree_ok_status();
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->ret.value);
    case LOOM_LLVMIR_INST_BR:
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_COND_BR:
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->cond_br.condition);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_bitcode_map_mark_metadata_constants(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_function_value_map_t* map) {
  const loom_llvmir_module_t* module = function->module;
  for (iree_host_size_t i = 0; i < function->metadata_attachment_count; ++i) {
    const loom_llvmir_metadata_attachment_storage_t* attachment =
        &function->metadata_attachments[i];
    if (attachment->metadata_id >= module->metadata_node_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode metadata attachment references unknown node");
    }
    const loom_llvmir_metadata_node_t* node =
        &module->metadata_nodes[attachment->metadata_id];
    for (iree_host_size_t j = 0; j < node->i32_value_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, node->i32_value_ids[j]));
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_llvmir_bitcode_count_function_inline_asm_constants(
    const loom_llvmir_function_t* function) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      if (block->instructions[j].kind == LOOM_LLVMIR_INST_INLINE_ASM) {
        count += 1;
      }
    }
  }
  return count;
}

static iree_status_t loom_llvmir_bitcode_inline_asm_value_id(
    const loom_llvmir_function_t* function,
    const loom_llvmir_bitcode_function_value_map_t* map,
    const loom_llvmir_instruction_t* target_instruction,
    uint64_t* out_value_id) {
  uint64_t value_id = map->first_inline_asm_value_id;
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->kind != LOOM_LLVMIR_INST_INLINE_ASM) continue;
      if (instruction == target_instruction) {
        *out_value_id = value_id;
        return iree_ok_status();
      }
      value_id += 1;
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "LLVM inline asm instruction is not owned by "
                          "function");
}

static iree_status_t loom_llvmir_bitcode_function_value_map_initialize(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_function_value_map_t* out_map) {
  memset(out_map, 0, sizeof(*out_map));
  const loom_llvmir_module_t* module = function->module;
  if (module->value_count == 0) {
    uint64_t first_local_value_id =
        loom_llvmir_bitcode_first_module_constant_value_id(module) +
        loom_llvmir_bitcode_module_constant_count(module);
    iree_host_size_t inline_asm_count =
        loom_llvmir_bitcode_count_function_inline_asm_constants(function);
    if (inline_asm_count > UINT64_MAX - first_local_value_id) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode inline asm value ids overflow");
    }
    out_map->first_constant_value_id = first_local_value_id;
    out_map->first_inline_asm_value_id = first_local_value_id;
    out_map->first_instruction_value_id =
        first_local_value_id + inline_asm_count;
    return iree_ok_status();
  }
  if (module->value_count > IREE_HOST_SIZE_MAX / sizeof(*out_map->value_ids)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode value map storage overflows");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(iree_allocator_system(),
                            module->value_count * sizeof(*out_map->value_ids),
                            (void**)&out_map->value_ids));
  out_map->value_count = module->value_count;
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    out_map->value_ids[i] = LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID;
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < module->global_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_map_set_value(out_map,
                                               module->globals[i]->value_id, i);
  }
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    if (!loom_llvmir_bitcode_is_constant_value_kind(module->values[i].kind)) {
      continue;
    }
    if (!loom_llvmir_bitcode_value_is_global_initializer(
            module, (loom_llvmir_value_id_t)i)) {
      continue;
    }
    uint64_t bitcode_value_id = 0;
    status = loom_llvmir_bitcode_module_constant_value_id(
        module, (loom_llvmir_value_id_t)i, &bitcode_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_bitcode_map_set_value(
          out_map, (loom_llvmir_value_id_t)i, bitcode_value_id);
    }
  }
  uint64_t next_value_id =
      loom_llvmir_bitcode_first_module_constant_value_id(module) +
      loom_llvmir_bitcode_module_constant_count(module);
  for (iree_host_size_t i = 0;
       i < function->parameter_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_map_set_value(
        out_map, function->parameters[i].value_id, next_value_id++);
  }
  out_map->first_constant_value_id = next_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      status = loom_llvmir_bitcode_map_mark_instruction_constants(
          module, out_map, &block->instructions[j]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_map_mark_metadata_constants(function, out_map);
  }
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    if (out_map->value_ids[i] !=
        LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT) {
      continue;
    }
    status = loom_llvmir_bitcode_check_value_id_range(next_value_id);
    if (iree_status_is_ok(status)) {
      out_map->value_ids[i] = next_value_id++;
    }
  }
  out_map->first_inline_asm_value_id = next_value_id;
  if (iree_status_is_ok(status)) {
    iree_host_size_t inline_asm_count =
        loom_llvmir_bitcode_count_function_inline_asm_constants(function);
    if (inline_asm_count > UINT64_MAX - next_value_id) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "LLVM bitcode inline asm value ids overflow");
    } else {
      next_value_id += inline_asm_count;
    }
  }
  out_map->first_instruction_value_id = next_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        continue;
      }
      status = loom_llvmir_bitcode_map_set_value(
          out_map, instruction->result_value_id, next_value_id++);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), out_map->value_ids);
    *out_map = (loom_llvmir_bitcode_function_value_map_t){0};
  }
  return status;
}

static void loom_llvmir_bitcode_function_value_map_deinitialize(
    loom_llvmir_bitcode_function_value_map_t* map) {
  iree_allocator_free(iree_allocator_system(), map->value_ids);
}

static iree_status_t loom_llvmir_bitcode_map_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t* out_bitcode_value_id) {
  if (value_id >= map->value_count ||
      map->value_ids[value_id] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode instruction references unmapped "
                            "function value");
  }
  *out_bitcode_value_id = map->value_ids[value_id];
  return iree_ok_status();
}

static bool loom_llvmir_bitcode_function_has_constants(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* map) {
  if (map->first_inline_asm_value_id != map->first_instruction_value_id) {
    return true;
  }
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    if (!loom_llvmir_bitcode_is_constant_value_kind(module->values[i].kind)) {
      continue;
    }
    if (map->value_ids[i] != LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID &&
        map->value_ids[i] != LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT &&
        map->value_ids[i] >= map->first_constant_value_id &&
        map->value_ids[i] < map->first_inline_asm_value_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_push_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  if (bitcode_value_id >= instruction_value_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM bitcode instruction operand references a forward value");
  }
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)bitcode_value_id;
  operands[(*inout_operand_count)++] = relative_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_function(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id,
    uint64_t instruction_value_id, uint64_t* operands,
    iree_host_size_t* inout_operand_count) {
  if (function_id >= module->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode instruction references unknown "
                            "function");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  uint64_t function_value_id =
      loom_llvmir_bitcode_function_value_id(module, function_id);
  if (function_value_id >= instruction_value_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM bitcode instruction operand references a forward function");
  }
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)function_value_id;
  operands[(*inout_operand_count)++] = relative_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_value_type_pair(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)bitcode_value_id;
  operands[(*inout_operand_count)++] = relative_value_id;
  if (bitcode_value_id >= instruction_value_id) {
    if (value_id >= module->value_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode instruction references unknown "
                              "typed value");
    }
    operands[(*inout_operand_count)++] = module->values[value_id].type_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_signed_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  int64_t relative_value_id =
      (int64_t)instruction_value_id - (int64_t)bitcode_value_id;
  operands[(*inout_operand_count)++] =
      loom_llvmir_bitcode_encode_signed(relative_value_id);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_check_constant(
    const loom_llvmir_module_t* module, const loom_llvmir_value_t* constant) {
  if (constant->type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode constant references unknown type");
  }
  const loom_llvmir_type_t* type = &module->types[constant->type_id];
  switch (constant->kind) {
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER:
      if (type->kind != LOOM_LLVMIR_TYPE_INTEGER || type->bit_width > 64) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LLVM bitcode integer constant emission is not "
                                "implemented for this type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      if (type->kind != LOOM_LLVMIR_TYPE_FLOAT) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LLVM bitcode floating constant emission is "
                                "not implemented for this type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      if (type->kind == LOOM_LLVMIR_TYPE_VOID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode null constant has invalid type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_POISON:
      if (type->kind == LOOM_LLVMIR_TYPE_VOID) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM bitcode poison constant has invalid type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR:
      if (type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
          type->element_type >= module->type_count ||
          module->types[type->element_type].kind != LOOM_LLVMIR_TYPE_INTEGER ||
          module->types[type->element_type].bit_width > 64 ||
          constant->integer_vector.value_count != type->element_count) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode integer vector constant emission is not implemented "
            "for this type");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode value is not a constant");
  }
}

static iree_status_t loom_llvmir_bitcode_check_attr_list(
    const loom_llvmir_module_t* module, loom_llvmir_attr_list_t attrs) {
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    iree_host_size_t operand_count = 0;
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_attr_operand_count(
        module, &attrs.attrs[i], &operand_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_check_global(
    const loom_llvmir_module_t* module, const loom_llvmir_global_t* global) {
  if (global->value_type >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode global references unknown type");
  }
  if (global->initializer >= module->value_count ||
      !loom_llvmir_bitcode_is_constant_value_kind(
          module->values[global->initializer].kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode global initializer must be constant");
  }
  if (module->values[global->initializer].type_id != global->value_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode global initializer type mismatch");
  }
  const loom_llvmir_value_t* value = global->value_id < module->value_count
                                         ? &module->values[global->value_id]
                                         : NULL;
  if (!value || value->kind != LOOM_LLVMIR_VALUE_GLOBAL ||
      value->global.global_id != global->id ||
      value->type_id >= module->type_count ||
      module->types[value->type_id].kind != LOOM_LLVMIR_TYPE_POINTER ||
      module->types[value->type_id].address_space != global->address_space) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode global value is invalid");
  }
  switch (global->linkage) {
    case LOOM_LLVMIR_LINKAGE_DEFAULT:
    case LOOM_LLVMIR_LINKAGE_DSO_LOCAL:
    case LOOM_LLVMIR_LINKAGE_INTERNAL:
    case LOOM_LLVMIR_LINKAGE_PRIVATE:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM global linkage");
  }
  uint64_t encoded_alignment = 0;
  return loom_llvmir_bitcode_encode_alignment(global->alignment,
                                              &encoded_alignment);
}

static iree_status_t loom_llvmir_bitcode_icmp_predicate(
    loom_llvmir_icmp_predicate_t predicate, uint64_t* out_predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_ICMP_EQ:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_EQ;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_NE:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_NE;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_UGT:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_UGT;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_UGE:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_UGE;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_ULT:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_ULT;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_ULE:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_ULE;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_SGT:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_SGT;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_SGE:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_SGE;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_SLT:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_SLT;
      return iree_ok_status();
    case LOOM_LLVMIR_ICMP_SLE:
      *out_predicate = LOOM_LLVMIR_BITCODE_ICMP_SLE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM icmp predicate");
  }
}

static iree_status_t loom_llvmir_bitcode_fcmp_predicate(
    loom_llvmir_fcmp_predicate_t predicate, uint64_t* out_predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_FCMP_FALSE:
    case LOOM_LLVMIR_FCMP_OEQ:
    case LOOM_LLVMIR_FCMP_OGT:
    case LOOM_LLVMIR_FCMP_OGE:
    case LOOM_LLVMIR_FCMP_OLT:
    case LOOM_LLVMIR_FCMP_OLE:
    case LOOM_LLVMIR_FCMP_ONE:
    case LOOM_LLVMIR_FCMP_ORD:
    case LOOM_LLVMIR_FCMP_UNO:
    case LOOM_LLVMIR_FCMP_UEQ:
    case LOOM_LLVMIR_FCMP_UGT:
    case LOOM_LLVMIR_FCMP_UGE:
    case LOOM_LLVMIR_FCMP_ULT:
    case LOOM_LLVMIR_FCMP_ULE:
    case LOOM_LLVMIR_FCMP_UNE:
    case LOOM_LLVMIR_FCMP_TRUE:
      *out_predicate = (uint64_t)predicate;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM fcmp predicate");
  }
}

static iree_status_t loom_llvmir_bitcode_cast_op(loom_llvmir_cast_op_t op,
                                                 uint64_t* out_op) {
  switch (op) {
    case LOOM_LLVMIR_CAST_TRUNCATE:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_TRUNCATE;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_ZERO_EXTEND:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_ZERO_EXTEND;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_SIGN_EXTEND:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_SIGN_EXTEND;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_FP_TO_UNSIGNED_INT;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_FP_TO_SIGNED_INT;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_UNSIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_SIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_FP_TRUNCATE:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_FP_TRUNCATE;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_FP_EXTEND:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_FP_EXTEND;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_PTR_TO_INT:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_PTR_TO_INT;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_INT_TO_PTR:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_INT_TO_PTR;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_BITCAST:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_BITCAST;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_ADDRESS_SPACE_CAST;
      return iree_ok_status();
    case LOOM_LLVMIR_CAST_PTR_TO_ADDR:
      *out_op = LOOM_LLVMIR_BITCODE_CAST_PTR_TO_ADDR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM cast op");
  }
}

static bool loom_llvmir_bitcode_binop_is_float(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_FADD:
    case LOOM_LLVMIR_BINOP_FSUB:
    case LOOM_LLVMIR_BINOP_FMUL:
    case LOOM_LLVMIR_BINOP_FDIV:
    case LOOM_LLVMIR_BINOP_FREM:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_bitcode_binop_allows_no_wrap_flags(
    loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_SHL:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_bitcode_binop_allows_exact_flag(
    loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_UDIV:
    case LOOM_LLVMIR_BINOP_SDIV:
    case LOOM_LLVMIR_BINOP_LSHR:
    case LOOM_LLVMIR_BINOP_ASHR:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_bitcode_fast_math_flags(
    loom_llvmir_fast_math_flags_t flags, uint64_t* out_flags) {
  const uint32_t known_flags = LOOM_LLVMIR_FAST_MATH_FAST;
  uint32_t flags_bits = (uint32_t)flags;
  if (flags_bits & ~known_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM fast-math flags");
  }
  uint64_t bitcode_flags = 0;
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_REASSOC) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_ALLOW_REASSOC;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_NANS) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_NO_NANS;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_INFS) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_NO_INFS;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_NO_SIGNED_ZEROS;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_RECIPROCAL) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_ALLOW_RECIPROCAL;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_CONTRACT) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_ALLOW_CONTRACT;
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_APPROX_FUNC) {
    bitcode_flags |= LOOM_LLVMIR_BITCODE_FAST_MATH_APPROX_FUNC;
  }
  *out_flags = bitcode_flags;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_integer_arithmetic_flags(
    loom_llvmir_binop_t op, loom_llvmir_integer_arithmetic_flags_t flags,
    uint64_t* out_flags) {
  const uint32_t known_flags = LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
                               LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP |
                               LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT;
  uint32_t flags_bits = (uint32_t)flags;
  if (flags_bits & ~known_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM integer arithmetic flags");
  }
  const uint32_t no_wrap_flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  if ((flags_bits & no_wrap_flags) &&
      !loom_llvmir_bitcode_binop_allows_no_wrap_flags(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry no-wrap flags");
  }
  if ((flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT) &&
      !loom_llvmir_bitcode_binop_allows_exact_flag(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry exact flag");
  }
  uint64_t bitcode_flags = 0;
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP) {
    bitcode_flags |= UINT64_C(1)
                     << LOOM_LLVMIR_BITCODE_OVERFLOWING_BINOP_NO_UNSIGNED_WRAP;
  }
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP) {
    bitcode_flags |= UINT64_C(1)
                     << LOOM_LLVMIR_BITCODE_OVERFLOWING_BINOP_NO_SIGNED_WRAP;
  }
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT) {
    bitcode_flags |= UINT64_C(1) << LOOM_LLVMIR_BITCODE_EXACT_BINOP_EXACT;
  }
  *out_flags = bitcode_flags;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_binop_flags(
    const loom_llvmir_instruction_t* instruction, uint64_t* out_flags) {
  if (instruction->binop.integer_flags != LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE &&
      instruction->binop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop mixes integer and fast-math flags");
  }
  if (instruction->binop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE) {
    if (!loom_llvmir_bitcode_binop_is_float(instruction->binop.op)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM integer binop cannot carry fast-math "
                              "flags");
    }
    return loom_llvmir_bitcode_fast_math_flags(
        instruction->binop.fast_math_flags, out_flags);
  }
  if (instruction->binop.integer_flags != LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE) {
    if (loom_llvmir_bitcode_binop_is_float(instruction->binop.op)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM floating-point binop cannot carry integer "
                              "arithmetic flags");
    }
    return loom_llvmir_bitcode_integer_arithmetic_flags(
        instruction->binop.op, instruction->binop.integer_flags, out_flags);
  }
  *out_flags = 0;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_unop_flags(
    const loom_llvmir_instruction_t* instruction, uint64_t* out_flags) {
  if (instruction->unop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE &&
      instruction->unop.op != LOOM_LLVMIR_UNOP_FNEG) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM unary op cannot carry fast-math flags");
  }
  return loom_llvmir_bitcode_fast_math_flags(instruction->unop.fast_math_flags,
                                             out_flags);
}

static iree_status_t loom_llvmir_bitcode_check_instruction(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_RET:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BR:
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_PHI:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BINOP: {
      uint64_t binop_flags = 0;
      return loom_llvmir_bitcode_binop_flags(instruction, &binop_flags);
    }
    case LOOM_LLVMIR_INST_UNOP: {
      if (instruction->unop.op != LOOM_LLVMIR_UNOP_FNEG) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM unary op");
      }
      uint64_t unop_flags = 0;
      return loom_llvmir_bitcode_unop_flags(instruction, &unop_flags);
    }
    case LOOM_LLVMIR_INST_ICMP: {
      uint64_t predicate = 0;
      return loom_llvmir_bitcode_icmp_predicate(instruction->icmp.predicate,
                                                &predicate);
    }
    case LOOM_LLVMIR_INST_FCMP: {
      uint64_t predicate = 0;
      return loom_llvmir_bitcode_fcmp_predicate(instruction->fcmp.predicate,
                                                &predicate);
    }
    case LOOM_LLVMIR_INST_SELECT:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_CAST: {
      uint64_t op = 0;
      return loom_llvmir_bitcode_cast_op(instruction->cast.op, &op);
    }
    case LOOM_LLVMIR_INST_GEP:
      if (instruction->gep.index_count > (IREE_HOST_SIZE_MAX - 4) / 2) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "LLVM bitcode gep operand count overflows");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_ALLOCA: {
      if (instruction->alloca.element_type >= module->type_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM bitcode alloca references unknown element type");
      }
      if (module->types[instruction->alloca.element_type].kind ==
          LOOM_LLVMIR_TYPE_VOID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode alloca element type is invalid");
      }
      if (instruction->alloca.count >= module->value_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode alloca references unknown count");
      }
      loom_llvmir_type_id_t count_type_id =
          module->values[instruction->alloca.count].type_id;
      if (count_type_id >= module->type_count ||
          module->types[count_type_id].kind != LOOM_LLVMIR_TYPE_INTEGER) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode alloca count must be an integer");
      }
      const loom_llvmir_value_t* result_value =
          instruction->result_value_id < module->value_count
              ? &module->values[instruction->result_value_id]
              : NULL;
      if (!result_value || result_value->type_id >= module->type_count ||
          module->types[result_value->type_id].kind !=
              LOOM_LLVMIR_TYPE_POINTER) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode alloca result must be a pointer");
      }
      uint64_t packed_flags = 0;
      return loom_llvmir_bitcode_pack_alloca_flags(
          instruction->alloca.alignment, &packed_flags);
    }
    case LOOM_LLVMIR_INST_LOAD: {
      if (instruction->load.metadata_attachment_count != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode instruction metadata attachment emission is not "
            "implemented");
      }
      uint64_t encoded_alignment = 0;
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->load.alignment, &encoded_alignment));
      return loom_llvmir_bitcode_memory_volatile(instruction->load.flags,
                                                 &is_volatile);
    }
    case LOOM_LLVMIR_INST_STORE: {
      if (instruction->store.metadata_attachment_count != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode instruction metadata attachment emission is not "
            "implemented");
      }
      uint64_t encoded_alignment = 0;
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->store.alignment, &encoded_alignment));
      return loom_llvmir_bitcode_memory_volatile(instruction->store.flags,
                                                 &is_volatile);
    }
    case LOOM_LLVMIR_INST_ATOMIC_RMW: {
      uint64_t encoded_op = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_rmw_op(
          instruction->atomic_rmw.op, &encoded_op));
      uint64_t encoded_alignment = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->atomic_rmw.alignment, &encoded_alignment));
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
          instruction->atomic_rmw.flags, &is_volatile));
      uint64_t encoded_ordering = 0;
      return loom_llvmir_bitcode_atomic_ordering(
          instruction->atomic_rmw.ordering, &encoded_ordering);
    }
    case LOOM_LLVMIR_INST_CMPXCHG: {
      uint64_t encoded_alignment = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->cmpxchg.alignment, &encoded_alignment));
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
          instruction->cmpxchg.flags, &is_volatile));
      uint64_t encoded_success_ordering = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_ordering(
          instruction->cmpxchg.success_ordering, &encoded_success_ordering));
      uint64_t encoded_failure_ordering = 0;
      return loom_llvmir_bitcode_atomic_ordering(
          instruction->cmpxchg.failure_ordering, &encoded_failure_ordering);
    }
    case LOOM_LLVMIR_INST_EXTRACT_ELEMENT:
    case LOOM_LLVMIR_INST_EXTRACT_VALUE:
    case LOOM_LLVMIR_INST_INSERT_ELEMENT:
    case LOOM_LLVMIR_INST_SHUFFLE_VECTOR:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_CALL:
      if (instruction->call.result_attrs.attr_count != 0 &&
          instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM call result attributes require a result value");
      }
      return loom_llvmir_bitcode_check_attr_list(
          module, instruction->call.result_attrs);
    case LOOM_LLVMIR_INST_INLINE_ASM: {
      if (instruction->inline_asm.result_type >= module->type_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM inline asm references unknown result type");
      }
      const loom_llvmir_type_t* result_type =
          &module->types[instruction->inline_asm.result_type];
      if (result_type->kind == LOOM_LLVMIR_TYPE_VOID &&
          instruction->result_value_id != LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM void inline asm must not define a result value");
      }
      if (result_type->kind != LOOM_LLVMIR_TYPE_VOID &&
          instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM non-void inline asm requires a result value");
      }
      if ((instruction->inline_asm.flags &
           ~(LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT |
             LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK |
             LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT)) != 0) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LLVM bitcode inline asm flag emission is not "
                                "implemented");
      }
      if (iree_string_view_is_empty(instruction->inline_asm.asm_template)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM inline asm template must not be empty");
      }
      if (instruction->inline_asm.asm_template.size > 0 &&
          instruction->inline_asm.asm_template.data == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM inline asm template has null storage");
      }
      if (instruction->inline_asm.constraints.size > 0 &&
          instruction->inline_asm.constraints.data == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM inline asm constraints have null "
                                "storage");
      }
      bool has_generic_pointer_type = false;
      for (iree_host_size_t i = 0; i < module->type_count; ++i) {
        const loom_llvmir_type_t* type = &module->types[i];
        if (type->kind == LOOM_LLVMIR_TYPE_POINTER &&
            type->address_space == 0) {
          has_generic_pointer_type = true;
          break;
        }
      }
      if (!has_generic_pointer_type) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM bitcode inline asm emission requires a generic pointer type");
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_COND_BR:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_bitcode_check_function(
    const loom_llvmir_module_t* module,
    const loom_llvmir_function_t* function) {
  if (function->kind != LOOM_LLVMIR_FUNCTION_DECLARATION &&
      function->kind != LOOM_LLVMIR_FUNCTION_DEFINITION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM function kind");
  }
  if (function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID) {
    if (function->attr_group_id >= module->attr_group_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM function references unknown attr group");
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_attr_list(
        module, module->attr_groups[function->attr_group_id].attrs));
  }
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_attr_list(
        module, function->parameters[i].attrs));
  }
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    if (function->metadata_attachment_count != 0) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode declaration metadata attachment "
                              "emission is not implemented");
    }
    if (function->block_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM declaration contains blocks");
    }
    return iree_ok_status();
  }
  if (function->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function definition has no blocks");
  }
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_instruction(
          module, &block->instructions[j]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_check_module(
    const loom_llvmir_module_t* module) {
  if (module->function_count > IREE_HOST_SIZE_MAX - module->type_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode type table is too large");
  }
  iree_host_size_t inline_asm_type_count =
      loom_llvmir_bitcode_count_inline_asm_types(module);
  if (module->type_count + module->function_count >
      IREE_HOST_SIZE_MAX - inline_asm_type_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode type table is too large");
  }
  for (iree_host_size_t i = 0; i < module->attr_group_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_attr_list(
        module, module->attr_groups[i].attrs));
  }
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    switch (module->values[i].kind) {
      case LOOM_LLVMIR_VALUE_GLOBAL:
        break;
      case LOOM_LLVMIR_VALUE_PARAMETER:
        break;
      case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER:
      case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      case LOOM_LLVMIR_VALUE_CONSTANT_POISON:
      case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR: {
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_check_constant(module, &module->values[i]));
        break;
      }
      case LOOM_LLVMIR_VALUE_INSTRUCTION:
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM value kind");
    }
  }
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_check_global(module, module->globals[i]));
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_check_function(module, module->functions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_linkage(loom_llvmir_linkage_t linkage,
                                                 uint64_t* out_linkage,
                                                 uint64_t* out_dso_local) {
  switch (linkage) {
    case LOOM_LLVMIR_LINKAGE_DEFAULT:
      *out_linkage = 0;
      *out_dso_local = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_DSO_LOCAL:
      *out_linkage = 0;
      *out_dso_local = 1;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_INTERNAL:
      *out_linkage = 3;
      *out_dso_local = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_PRIVATE:
      *out_linkage = 9;
      *out_dso_local = 0;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM linkage");
  }
}

static iree_status_t loom_llvmir_bitcode_function_calling_convention(
    loom_llvmir_calling_convention_t calling_convention,
    uint64_t* out_calling_convention) {
  switch (calling_convention) {
    case LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT:
      *out_calling_convention = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL:
      *out_calling_convention = 91;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM calling convention");
  }
}

static iree_status_t loom_llvmir_bitcode_unary_opcode(loom_llvmir_unop_t op,
                                                      uint64_t* out_opcode) {
  switch (op) {
    case LOOM_LLVMIR_UNOP_FNEG:
      *out_opcode = LOOM_LLVMIR_BITCODE_UNOP_FNEG;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM unary op");
  }
}

static iree_status_t loom_llvmir_bitcode_binary_opcode(loom_llvmir_binop_t op,
                                                       uint64_t* out_opcode) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_FADD:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_ADD;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_FSUB:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SUB;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_FMUL:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_MUL;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_UDIV:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_UDIV;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SDIV:
    case LOOM_LLVMIR_BINOP_FDIV:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SDIV;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_UREM:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_UREM;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SREM:
    case LOOM_LLVMIR_BINOP_FREM:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SREM;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_AND:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_AND;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_OR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_OR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_XOR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_XOR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SHL:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SHL;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_LSHR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_LSHR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_ASHR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_ASHR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM binary op");
  }
}

static iree_status_t loom_llvmir_bitcode_write_function_record(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    uint64_t name_offset, loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t linkage = 0;
  uint64_t dso_local = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_linkage(function->linkage, &linkage, &dso_local));
  uint64_t calling_convention = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_calling_convention(
      function->calling_convention, &calling_convention));
  uint64_t attr_list_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_attr_list_id(
      module, function, &attr_list_id));

  uint64_t operands[] = {
      name_offset,
      function->name.size,
      loom_llvmir_bitcode_function_type_id(module, function->id),
      calling_convention,
      function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION,
      linkage,
      attr_list_id,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      dso_local,
      0,
  };
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_FUNCTION, operands,
      IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_llvmir_bitcode_write_global_record(
    const loom_llvmir_module_t* module, const loom_llvmir_global_t* global,
    uint64_t name_offset, loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t linkage = 0;
  uint64_t dso_local = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_linkage(global->linkage, &linkage, &dso_local));
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      global->alignment, &encoded_alignment));
  uint64_t initializer_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_module_constant_value_id(
      module, global->initializer, &initializer_id));

  uint64_t operands[16];
  iree_host_size_t operand_count = 0;
  operands[operand_count++] = name_offset;
  operands[operand_count++] = global->name.size;
  operands[operand_count++] = global->value_type;
  operands[operand_count++] = ((uint64_t)global->address_space << 2) |
                              UINT64_C(2) |
                              (global->is_constant ? UINT64_C(1) : UINT64_C(0));
  operands[operand_count++] = initializer_id + 1;
  operands[operand_count++] = linkage;
  operands[operand_count++] = encoded_alignment;
  operands[operand_count++] = 0;
  if (dso_local != 0) {
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = 0;
    operands[operand_count++] = dso_local;
  }
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_GLOBALVAR, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_constant(
    const loom_llvmir_module_t* module, const loom_llvmir_value_t* constant,
    loom_llvmir_type_id_t* inout_current_type,
    loom_llvmir_bitcode_record_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_constant(module, constant));
  if (*inout_current_type != constant->type_id) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
        writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_SETTYPE, constant->type_id));
    *inout_current_type = constant->type_id;
  }

  const loom_llvmir_type_t* type = &module->types[constant->type_id];
  switch (constant->kind) {
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER: {
      uint64_t value = loom_llvmir_bitcode_sign_extend_integer_constant(
          constant->integer_value, type->bit_width);
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_INTEGER,
          loom_llvmir_bitcode_encode_signed((int64_t)value));
    }
    case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_FLOAT,
          constant->float_bits);
    case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      return loom_llvmir_bitcode_write_empty_record(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_NULL);
    case LOOM_LLVMIR_VALUE_CONSTANT_POISON:
      return loom_llvmir_bitcode_write_empty_record(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_POISON);
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR:
      return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_DATA,
          constant->integer_vector.values,
          constant->integer_vector.value_count);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode value is not a constant");
  }
}

static iree_status_t loom_llvmir_bitcode_write_module_constants_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (loom_llvmir_bitcode_module_constant_count(module) == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_CONSTANTS_BLOCK,
      LOOM_LLVMIR_BITCODE_CONSTANT_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  loom_llvmir_type_id_t current_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  uint64_t expected_value_id =
      loom_llvmir_bitcode_first_module_constant_value_id(module);
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_value_t* value = &module->values[i];
    if (!loom_llvmir_bitcode_is_constant_value_kind(value->kind)) continue;
    if (!loom_llvmir_bitcode_value_is_global_initializer(
            module, (loom_llvmir_value_id_t)i)) {
      continue;
    }
    uint64_t bitcode_value_id = 0;
    status = loom_llvmir_bitcode_module_constant_value_id(
        module, (loom_llvmir_value_id_t)i, &bitcode_value_id);
    if (iree_status_is_ok(status) && bitcode_value_id != expected_value_id) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode module constant value numbering is inconsistent");
    }
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_bitcode_write_constant(module, value, &current_type,
                                                  writer);
      expected_value_id += 1;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_inline_asm_pointer_type_id(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t* out_type_id) {
  for (iree_host_size_t i = 0; i < module->type_count; ++i) {
    const loom_llvmir_type_t* type = &module->types[i];
    if (type->kind == LOOM_LLVMIR_TYPE_POINTER && type->address_space == 0) {
      *out_type_id = (loom_llvmir_type_id_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "LLVM bitcode inline asm emission requires a generic pointer type");
}

static uint64_t loom_llvmir_bitcode_inline_asm_flags(
    loom_llvmir_inline_asm_flags_t flags) {
  uint64_t encoded_flags = 0;
  if (flags & LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT) encoded_flags |= 1;
  if (flags & LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK) encoded_flags |= 2;
  if (flags & LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT) encoded_flags |= 4;
  return encoded_flags;
}

static iree_status_t loom_llvmir_bitcode_write_inline_asm_constant(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction,
    loom_llvmir_type_id_t* inout_current_type,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if ((instruction->inline_asm.flags &
       ~(LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT |
         LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK |
         LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT)) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "LLVM bitcode inline asm flag emission is not "
                            "implemented");
  }
  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_inline_asm_pointer_type_id(module, &pointer_type));
  if (*inout_current_type != pointer_type) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
        writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_SETTYPE, pointer_type));
    *inout_current_type = pointer_type;
  }
  if (instruction->inline_asm.asm_template.size >
      IREE_HOST_SIZE_MAX - instruction->inline_asm.constraints.size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode inline asm operand count overflows");
  }
  iree_host_size_t string_operand_count =
      instruction->inline_asm.asm_template.size +
      instruction->inline_asm.constraints.size;
  if (string_operand_count > IREE_HOST_SIZE_MAX - 4) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode inline asm operand count overflows");
  }
  iree_host_size_t operand_count = string_operand_count + 4;
  uint64_t stack_operands[32];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "LLVM bitcode inline asm operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t written_operand_count = 0;
  iree_status_t status = loom_llvmir_bitcode_inline_asm_function_type_id(
      module, instruction, &operands[written_operand_count++]);
  if (iree_status_is_ok(status)) {
    operands[written_operand_count++] =
        loom_llvmir_bitcode_inline_asm_flags(instruction->inline_asm.flags);
    operands[written_operand_count++] =
        instruction->inline_asm.asm_template.size;
    for (iree_host_size_t i = 0; i < instruction->inline_asm.asm_template.size;
         ++i) {
      operands[written_operand_count++] =
          (uint8_t)instruction->inline_asm.asm_template.data[i];
    }
    operands[written_operand_count++] =
        instruction->inline_asm.constraints.size;
    for (iree_host_size_t i = 0; i < instruction->inline_asm.constraints.size;
         ++i) {
      operands[written_operand_count++] =
          (uint8_t)instruction->inline_asm.constraints.data[i];
    }
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_INLINEASM, operands,
        written_operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_constants_block(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_function_has_constants(module, value_map)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_CONSTANTS_BLOCK,
      LOOM_LLVMIR_BITCODE_CONSTANT_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  loom_llvmir_type_id_t current_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  uint64_t expected_value_id = value_map->first_constant_value_id;
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_value_t* value = &module->values[i];
    if (!loom_llvmir_bitcode_is_constant_value_kind(value->kind)) continue;
    if (value_map->value_ids[i] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
      continue;
    }
    if (value_map->value_ids[i] < value_map->first_constant_value_id) {
      continue;
    }
    if (value_map->value_ids[i] != expected_value_id) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode constant value numbering is inconsistent");
    } else {
      status = loom_llvmir_bitcode_write_constant(module, value, &current_type,
                                                  writer);
      expected_value_id += 1;
    }
  }
  uint64_t expected_inline_asm_value_id = value_map->first_inline_asm_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->kind != LOOM_LLVMIR_INST_INLINE_ASM) continue;
      uint64_t inline_asm_value_id = 0;
      status = loom_llvmir_bitcode_inline_asm_value_id(
          function, value_map, instruction, &inline_asm_value_id);
      if (iree_status_is_ok(status) &&
          inline_asm_value_id != expected_inline_asm_value_id) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "LLVM bitcode inline asm value numbering is inconsistent");
      }
      if (iree_status_is_ok(status)) {
        status = loom_llvmir_bitcode_write_inline_asm_constant(
            module, instruction, &current_type, writer);
        expected_inline_asm_value_id += 1;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static bool loom_llvmir_bitcode_metadata_kind_seen(
    const loom_llvmir_module_t* module,
    const loom_llvmir_function_t* stop_function,
    const loom_llvmir_metadata_attachment_storage_t* stop_attachment,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->metadata_attachment_count; ++j) {
      const loom_llvmir_metadata_attachment_storage_t* attachment =
          &function->metadata_attachments[j];
      if (function == stop_function && attachment == stop_attachment) {
        return false;
      }
      if (iree_string_view_equal(attachment->name, name)) return true;
    }
  }
  return false;
}

static bool loom_llvmir_bitcode_module_has_metadata_kinds(
    const loom_llvmir_module_t* module) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (module->functions[i]->metadata_attachment_count != 0) return true;
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_metadata_kind_id(
    const loom_llvmir_module_t* module, iree_string_view_t name,
    uint64_t* out_kind_id) {
  uint64_t kind_id = 0;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->metadata_attachment_count; ++j) {
      const loom_llvmir_metadata_attachment_storage_t* attachment =
          &function->metadata_attachments[j];
      if (loom_llvmir_bitcode_metadata_kind_seen(module, function, attachment,
                                                 attachment->name)) {
        continue;
      }
      if (iree_string_view_equal(attachment->name, name)) {
        *out_kind_id = kind_id;
        return iree_ok_status();
      }
      kind_id += 1;
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "LLVM bitcode metadata kind is unknown");
}

static iree_status_t loom_llvmir_bitcode_write_metadata_kind_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_metadata_kinds(module)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_METADATA_KIND_BLOCK,
      LOOM_LLVMIR_BITCODE_METADATA_KIND_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  uint64_t kind_id = 0;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0;
         j < function->metadata_attachment_count && iree_status_is_ok(status);
         ++j) {
      const loom_llvmir_metadata_attachment_storage_t* attachment =
          &function->metadata_attachments[j];
      if (loom_llvmir_bitcode_metadata_kind_seen(module, function, attachment,
                                                 attachment->name)) {
        continue;
      }
      if (attachment->name.size > IREE_HOST_SIZE_MAX - 1) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "LLVM bitcode metadata kind operand count overflows");
        continue;
      }
      iree_host_size_t operand_count = attachment->name.size + 1;
      uint64_t stack_operands[32];
      uint64_t* operands = stack_operands;
      if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
        if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "LLVM bitcode metadata kind operand storage overflows");
          continue;
        }
        status = iree_allocator_malloc(iree_allocator_system(),
                                       operand_count * sizeof(*operands),
                                       (void**)&operands);
      }
      if (iree_status_is_ok(status)) {
        operands[0] = kind_id++;
        for (iree_host_size_t k = 0; k < attachment->name.size; ++k) {
          operands[k + 1] = (uint8_t)attachment->name.data[k];
        }
        status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
            writer, LOOM_LLVMIR_BITCODE_METADATA_CODE_KIND, operands,
            operand_count);
      }
      if (operands != stack_operands) {
        iree_allocator_free(iree_allocator_system(), operands);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static bool loom_llvmir_bitcode_instruction_sync_scope(
    const loom_llvmir_instruction_t* instruction,
    iree_string_view_t* out_sync_scope) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_ATOMIC_RMW:
      *out_sync_scope = instruction->atomic_rmw.sync_scope;
      return true;
    case LOOM_LLVMIR_INST_CMPXCHG:
      *out_sync_scope = instruction->cmpxchg.sync_scope;
      return true;
    default:
      *out_sync_scope = iree_string_view_empty();
      return false;
  }
}

static bool loom_llvmir_bitcode_module_has_scoped_atomic(
    const loom_llvmir_module_t* module) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        iree_string_view_t sync_scope = iree_string_view_empty();
        if (loom_llvmir_bitcode_instruction_sync_scope(&block->instructions[k],
                                                       &sync_scope)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_llvmir_bitcode_sync_scope_seen(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* stop_instruction,
    iree_string_view_t sync_scope) {
  if (iree_string_view_equal(sync_scope, IREE_SV("singlethread")) ||
      iree_string_view_is_empty(sync_scope)) {
    return true;
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        if (instruction == stop_instruction) {
          return false;
        }
        iree_string_view_t current_scope = iree_string_view_empty();
        if (!loom_llvmir_bitcode_instruction_sync_scope(instruction,
                                                        &current_scope)) {
          continue;
        }
        if (iree_string_view_equal(current_scope, sync_scope)) {
          return true;
        }
      }
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_sync_scope_id(
    const loom_llvmir_module_t* module, iree_string_view_t sync_scope,
    uint64_t* out_scope_id) {
  if (iree_string_view_equal(sync_scope, IREE_SV("singlethread"))) {
    *out_scope_id = 0;
    return iree_ok_status();
  }
  if (iree_string_view_is_empty(sync_scope)) {
    *out_scope_id = 1;
    return iree_ok_status();
  }
  uint64_t scope_id = 2;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0; j < function->block_count; ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0; k < block->instruction_count; ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        iree_string_view_t current_scope = iree_string_view_empty();
        if (!loom_llvmir_bitcode_instruction_sync_scope(instruction,
                                                        &current_scope)) {
          continue;
        }
        if (iree_string_view_equal(current_scope, IREE_SV("singlethread")) ||
            iree_string_view_is_empty(current_scope) ||
            loom_llvmir_bitcode_sync_scope_seen(module, instruction,
                                                current_scope)) {
          continue;
        }
        if (iree_string_view_equal(current_scope, sync_scope)) {
          *out_scope_id = scope_id;
          return iree_ok_status();
        }
        scope_id += 1;
      }
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "LLVM bitcode syncscope name is unknown");
}

static iree_status_t loom_llvmir_bitcode_write_sync_scope_names_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_scoped_atomic(module)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_SYNC_SCOPE_NAMES_BLOCK,
      LOOM_LLVMIR_BITCODE_SYNC_SCOPE_NAMES_ABBREV_WIDTH));

  iree_status_t status = loom_llvmir_bitcode_record_writer_write_string_record(
      writer, LOOM_LLVMIR_BITCODE_SYNC_SCOPE_NAMES_CODE_NAME,
      IREE_SV("singlethread"));
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        writer, LOOM_LLVMIR_BITCODE_SYNC_SCOPE_NAMES_CODE_NAME,
        iree_string_view_empty());
  }
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0;
         j < function->block_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_block_t* block = function->blocks[j];
      for (iree_host_size_t k = 0;
           k < block->instruction_count && iree_status_is_ok(status); ++k) {
        const loom_llvmir_instruction_t* instruction = &block->instructions[k];
        iree_string_view_t sync_scope = iree_string_view_empty();
        if (!loom_llvmir_bitcode_instruction_sync_scope(instruction,
                                                        &sync_scope)) {
          continue;
        }
        if (iree_string_view_equal(sync_scope, IREE_SV("singlethread")) ||
            iree_string_view_is_empty(sync_scope) ||
            loom_llvmir_bitcode_sync_scope_seen(module, instruction,
                                                sync_scope)) {
          continue;
        }
        status = loom_llvmir_bitcode_record_writer_write_string_record(
            writer, LOOM_LLVMIR_BITCODE_SYNC_SCOPE_NAMES_CODE_NAME, sync_scope);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static bool loom_llvmir_bitcode_function_metadata_node_seen(
    const loom_llvmir_function_t* function, iree_host_size_t stop_index,
    loom_llvmir_metadata_id_t metadata_id) {
  for (iree_host_size_t i = 0; i < stop_index; ++i) {
    if (function->metadata_attachments[i].metadata_id == metadata_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_function_metadata_node_id(
    const loom_llvmir_function_t* function,
    loom_llvmir_metadata_id_t metadata_id, uint64_t* out_metadata_id) {
  const loom_llvmir_module_t* module = function->module;
  uint64_t next_metadata_id = 0;
  for (iree_host_size_t i = 0; i < function->metadata_attachment_count; ++i) {
    loom_llvmir_metadata_id_t current_id =
        function->metadata_attachments[i].metadata_id;
    if (loom_llvmir_bitcode_function_metadata_node_seen(function, i,
                                                        current_id)) {
      continue;
    }
    if (current_id >= module->metadata_node_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode metadata attachment references unknown node");
    }
    const loom_llvmir_metadata_node_t* node =
        &module->metadata_nodes[current_id];
    if (current_id == metadata_id) {
      *out_metadata_id = next_metadata_id + node->i32_value_count;
      return iree_ok_status();
    }
    next_metadata_id += node->i32_value_count + 1;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "LLVM bitcode metadata node is not attached");
}

static iree_status_t loom_llvmir_bitcode_write_metadata_block(
    const loom_llvmir_function_t* function,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (function->metadata_attachment_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_METADATA_BLOCK,
      LOOM_LLVMIR_BITCODE_METADATA_ABBREV_WIDTH));

  const loom_llvmir_module_t* module = function->module;
  iree_status_t status = iree_ok_status();
  uint64_t next_metadata_id = 0;
  for (iree_host_size_t i = 0;
       i < function->metadata_attachment_count && iree_status_is_ok(status);
       ++i) {
    loom_llvmir_metadata_id_t metadata_id =
        function->metadata_attachments[i].metadata_id;
    if (loom_llvmir_bitcode_function_metadata_node_seen(function, i,
                                                        metadata_id)) {
      continue;
    }
    const loom_llvmir_metadata_node_t* node =
        &module->metadata_nodes[metadata_id];
    for (iree_host_size_t j = 0;
         j < node->i32_value_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_value_t* value =
          &module->values[node->i32_value_ids[j]];
      uint64_t bitcode_value_id = 0;
      status = loom_llvmir_bitcode_map_value(value_map, node->i32_value_ids[j],
                                             &bitcode_value_id);
      if (iree_status_is_ok(status)) {
        uint64_t operands[] = {value->type_id, bitcode_value_id};
        status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
            writer, LOOM_LLVMIR_BITCODE_METADATA_CODE_VALUE, operands,
            IREE_ARRAYSIZE(operands));
      }
    }
    if (iree_status_is_ok(status)) {
      iree_host_size_t operand_count = node->i32_value_count;
      uint64_t stack_operands[16];
      uint64_t* operands = stack_operands;
      if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
        if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "LLVM bitcode metadata node operand storage overflows");
        } else {
          status = iree_allocator_malloc(iree_allocator_system(),
                                         operand_count * sizeof(*operands),
                                         (void**)&operands);
        }
      }
      if (iree_status_is_ok(status)) {
        for (iree_host_size_t j = 0; j < node->i32_value_count; ++j) {
          operands[j] = next_metadata_id + j + 1;
        }
        status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
            writer, LOOM_LLVMIR_BITCODE_METADATA_CODE_NODE, operands,
            operand_count);
      }
      if (operands != stack_operands) {
        iree_allocator_free(iree_allocator_system(), operands);
      }
      next_metadata_id += node->i32_value_count + 1;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_metadata_attachment_block(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (function->metadata_attachment_count == 0) return iree_ok_status();
  if (function->metadata_attachment_count >
      IREE_HOST_SIZE_MAX / (2 * sizeof(uint64_t))) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "LLVM bitcode metadata attachment operand storage overflows");
  }
  iree_host_size_t operand_count = function->metadata_attachment_count * 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_status_t status = loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_METADATA_ATTACHMENT_BLOCK,
      LOOM_LLVMIR_BITCODE_METADATA_ATTACHMENT_ABBREV_WIDTH);

  const loom_llvmir_module_t* module = function->module;
  iree_host_size_t written_operand_count = 0;
  for (iree_host_size_t i = 0;
       i < function->metadata_attachment_count && iree_status_is_ok(status);
       ++i) {
    const loom_llvmir_metadata_attachment_storage_t* attachment =
        &function->metadata_attachments[i];
    status = loom_llvmir_bitcode_metadata_kind_id(
        module, attachment->name, &operands[written_operand_count]);
    if (iree_status_is_ok(status)) {
      written_operand_count += 1;
      status = loom_llvmir_bitcode_function_metadata_node_id(
          function, attachment->metadata_id, &operands[written_operand_count]);
      written_operand_count += 1;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_METADATA_CODE_ATTACHMENT, operands,
        written_operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_return(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!instruction->ret.has_value) {
    return loom_llvmir_bitcode_write_empty_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_RET);
  }
  uint64_t operands[2];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->ret.value, instruction_value_id, operands,
      &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_RET, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_binop(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t opcode = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_binary_opcode(instruction->binop.op, &opcode));
  uint64_t flags = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_binop_flags(instruction, &flags));
  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->binop.lhs, instruction_value_id, operands,
      &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->binop.rhs, instruction_value_id, operands,
      &operand_count));
  operands[operand_count++] = opcode;
  if (flags != 0) {
    operands[operand_count++] = flags;
  }
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BINOP, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_unop(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t opcode = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_unary_opcode(instruction->unop.op, &opcode));
  uint64_t flags = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_unop_flags(instruction, &flags));
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->unop.value, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = opcode;
  if (flags != 0) {
    operands[operand_count++] = flags;
  }
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_UNOP, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_icmp(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t predicate = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_icmp_predicate(
      instruction->icmp.predicate, &predicate));
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->icmp.lhs, instruction_value_id, operands,
      &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->icmp.rhs, instruction_value_id, operands,
      &operand_count));
  operands[operand_count++] = predicate;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CMP2, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_fcmp(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t predicate = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_fcmp_predicate(
      instruction->fcmp.predicate, &predicate));
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->fcmp.lhs, instruction_value_id, operands,
      &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->fcmp.rhs, instruction_value_id, operands,
      &operand_count));
  operands[operand_count++] = predicate;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CMP2, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_select(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->select.true_value, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->select.false_value, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->select.condition, instruction_value_id,
      operands, &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_VSELECT, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_cast(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t op = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_cast_op(instruction->cast.op, &op));
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->cast.value, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] =
      module->values[instruction->result_value_id].type_id;
  operands[operand_count++] = op;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CAST, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_gep(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->gep.index_count > (IREE_HOST_SIZE_MAX - 4) / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode gep operand count overflows");
  }
  iree_host_size_t operand_capacity = 4 + instruction->gep.index_count * 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode gep operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = 0;
  operands[operand_count++] = instruction->gep.element_type;
  iree_status_t status = loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->gep.base, instruction_value_id, operands,
      &operand_count);
  for (iree_host_size_t i = 0;
       i < instruction->gep.index_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_value_type_pair(
        module, value_map, instruction->gep.indices[i], instruction_value_id,
        operands, &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_GEP, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_alloca(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->result_value_id >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode alloca references unknown result");
  }
  const loom_llvmir_value_t* result_value =
      &module->values[instruction->result_value_id];
  if (result_value->type_id >= module->type_count ||
      module->types[result_value->type_id].kind != LOOM_LLVMIR_TYPE_POINTER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode alloca result must be a pointer");
  }
  const loom_llvmir_type_t* result_type = &module->types[result_value->type_id];
  if (instruction->alloca.count >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode alloca references unknown count");
  }
  const loom_llvmir_value_t* count_value =
      &module->values[instruction->alloca.count];
  uint64_t count_value_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_value(
      value_map, instruction->alloca.count, &count_value_id));
  if (count_value_id >= instruction_value_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM bitcode alloca count references a forward value");
  }
  uint64_t packed_flags = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_pack_alloca_flags(
      instruction->alloca.alignment, &packed_flags));

  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  operands[operand_count++] = instruction->alloca.element_type;
  operands[operand_count++] = count_value->type_id;
  operands[operand_count++] = count_value_id;
  operands[operand_count++] = packed_flags;
  if (result_type->address_space != 0) {
    operands[operand_count++] = result_type->address_space;
  }
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_ALLOCA, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_load(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->load.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->load.flags, &is_volatile));

  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->load.pointer, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = instruction->load.result_type;
  operands[operand_count++] = encoded_alignment;
  operands[operand_count++] = is_volatile;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_LOAD, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_store(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->store.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->store.flags, &is_volatile));

  uint64_t operands[6];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->store.pointer, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->store.value, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = encoded_alignment;
  operands[operand_count++] = is_volatile;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_STORE, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_atomic_rmw(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_op = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_rmw_op(
      instruction->atomic_rmw.op, &encoded_op));
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->atomic_rmw.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->atomic_rmw.flags, &is_volatile));
  uint64_t encoded_ordering = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_ordering(
      instruction->atomic_rmw.ordering, &encoded_ordering));
  uint64_t sync_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_sync_scope_id(
      module, instruction->atomic_rmw.sync_scope, &sync_scope_id));

  uint64_t operands[7];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->atomic_rmw.pointer, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->atomic_rmw.value, instruction_value_id, operands,
      &operand_count));
  operands[operand_count++] = encoded_op;
  operands[operand_count++] = is_volatile;
  operands[operand_count++] = encoded_ordering;
  operands[operand_count++] = sync_scope_id;
  operands[operand_count++] = encoded_alignment;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_ATOMICRMW, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_cmpxchg(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->cmpxchg.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->cmpxchg.flags, &is_volatile));
  uint64_t encoded_success_ordering = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_ordering(
      instruction->cmpxchg.success_ordering, &encoded_success_ordering));
  uint64_t encoded_failure_ordering = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_atomic_ordering(
      instruction->cmpxchg.failure_ordering, &encoded_failure_ordering));
  uint64_t sync_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_sync_scope_id(
      module, instruction->cmpxchg.sync_scope, &sync_scope_id));

  uint64_t operands[9];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->cmpxchg.pointer, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->cmpxchg.expected, instruction_value_id, operands,
      &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->cmpxchg.replacement, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = is_volatile;
  operands[operand_count++] = encoded_success_ordering;
  operands[operand_count++] = sync_scope_id;
  operands[operand_count++] = encoded_failure_ordering;
  operands[operand_count++] = instruction->cmpxchg.is_weak ? 1 : 0;
  operands[operand_count++] = encoded_alignment;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CMPXCHG, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_extract_element(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->extract_element.vector,
      instruction_value_id, operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->extract_element.index,
      instruction_value_id, operands, &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_EXTRACTELT, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_extract_value(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  (void)module;
  uint64_t operands[2];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->extract_value.aggregate, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = instruction->extract_value.index;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_EXTRACTVAL, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_insert_element(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t operands[6];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->insert_element.vector,
      instruction_value_id, operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->insert_element.element, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->insert_element.index,
      instruction_value_id, operands, &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_INSERTELT, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_shuffle_vector(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->shuffle_vector.lhs, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->shuffle_vector.rhs, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->shuffle_vector.mask, instruction_value_id,
      operands, &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_SHUFFLEVEC, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_call(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->call.callee >= module->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode call references unknown callee");
  }
  const loom_llvmir_function_t* callee =
      module->functions[instruction->call.callee];
  if (instruction->call.arg_count != callee->parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode call arg count mismatch");
  }
  if (instruction->call.arg_count > IREE_HOST_SIZE_MAX - 5) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode call operand count overflows");
  }
  iree_host_size_t operand_capacity = 5 + instruction->call.arg_count;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode call operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  uint64_t calling_convention = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_calling_convention(
      callee->calling_convention, &calling_convention));
  uint64_t attr_list_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_call_attr_list_id(
      module, instruction, &attr_list_id));

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = attr_list_id;
  operands[operand_count++] =
      (calling_convention << LOOM_LLVMIR_BITCODE_CALL_FLAG_CCONV) |
      (UINT64_C(1) << LOOM_LLVMIR_BITCODE_CALL_FLAG_EXPLICIT_TYPE);
  operands[operand_count++] =
      loom_llvmir_bitcode_function_type_id(module, callee->id);
  iree_status_t status = loom_llvmir_bitcode_push_function(
      module, instruction->call.callee, instruction_value_id, operands,
      &operand_count);
  for (iree_host_size_t i = 0;
       i < instruction->call.arg_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_value(
        value_map, instruction->call.args[i], instruction_value_id, operands,
        &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CALL, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_inline_asm(
    const loom_llvmir_function_t* function,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->inline_asm.arg_count > IREE_HOST_SIZE_MAX - 5) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode inline asm call operand count "
                            "overflows");
  }
  iree_host_size_t operand_capacity = 5 + instruction->inline_asm.arg_count;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode inline asm call operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  const loom_llvmir_module_t* module = function->module;
  uint64_t inline_asm_type_id = 0;
  iree_status_t status = loom_llvmir_bitcode_inline_asm_function_type_id(
      module, instruction, &inline_asm_type_id);
  uint64_t inline_asm_value_id = 0;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_inline_asm_value_id(
        function, value_map, instruction, &inline_asm_value_id);
  }

  iree_host_size_t operand_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_check_value_id_range(instruction_value_id);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_check_value_id_range(inline_asm_value_id);
  }
  if (iree_status_is_ok(status)) {
    if (inline_asm_value_id >= instruction_value_id) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode inline asm call references a forward callee");
    } else {
      operands[operand_count++] = 0;
      operands[operand_count++] =
          UINT64_C(1) << LOOM_LLVMIR_BITCODE_CALL_FLAG_EXPLICIT_TYPE;
      operands[operand_count++] = inline_asm_type_id;
      operands[operand_count++] = instruction_value_id - inline_asm_value_id;
    }
  }
  for (iree_host_size_t i = 0;
       i < instruction->inline_asm.arg_count && iree_status_is_ok(status);
       ++i) {
    status = loom_llvmir_bitcode_push_value(
        value_map, instruction->inline_asm.args[i], instruction_value_id,
        operands, &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CALL, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_phi(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->result_value_id >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode phi references unknown result");
  }
  if (instruction->phi.incoming_count > (IREE_HOST_SIZE_MAX - 1) / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode phi operand count overflows");
  }
  iree_host_size_t operand_count = 1 + instruction->phi.incoming_count * 2;
  uint64_t stack_operands[17];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode phi operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t written_operand_count = 0;
  operands[written_operand_count++] =
      module->values[instruction->result_value_id].type_id;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < instruction->phi.incoming_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_signed_value(
        value_map, instruction->phi.incoming[i].value, instruction_value_id,
        operands, &written_operand_count);
    if (iree_status_is_ok(status)) {
      operands[written_operand_count++] =
          instruction->phi.incoming[i].predecessor;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_PHI, operands,
        written_operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_instruction(
    const loom_llvmir_function_t* function,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  const loom_llvmir_module_t* module = function->module;
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI:
      return loom_llvmir_bitcode_write_phi(module, value_map, instruction,
                                           instruction_value_id, writer);
    case LOOM_LLVMIR_INST_BINOP:
      return loom_llvmir_bitcode_write_binop(module, value_map, instruction,
                                             instruction_value_id, writer);
    case LOOM_LLVMIR_INST_UNOP:
      return loom_llvmir_bitcode_write_unop(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_ICMP:
      return loom_llvmir_bitcode_write_icmp(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_FCMP:
      return loom_llvmir_bitcode_write_fcmp(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_SELECT:
      return loom_llvmir_bitcode_write_select(module, value_map, instruction,
                                              instruction_value_id, writer);
    case LOOM_LLVMIR_INST_CAST:
      return loom_llvmir_bitcode_write_cast(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_GEP:
      return loom_llvmir_bitcode_write_gep(module, value_map, instruction,
                                           instruction_value_id, writer);
    case LOOM_LLVMIR_INST_ALLOCA:
      return loom_llvmir_bitcode_write_alloca(module, value_map, instruction,
                                              instruction_value_id, writer);
    case LOOM_LLVMIR_INST_LOAD:
      return loom_llvmir_bitcode_write_load(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_STORE:
      return loom_llvmir_bitcode_write_store(module, value_map, instruction,
                                             instruction_value_id, writer);
    case LOOM_LLVMIR_INST_ATOMIC_RMW:
      return loom_llvmir_bitcode_write_atomic_rmw(
          module, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_CMPXCHG:
      return loom_llvmir_bitcode_write_cmpxchg(module, value_map, instruction,
                                               instruction_value_id, writer);
    case LOOM_LLVMIR_INST_EXTRACT_ELEMENT:
      return loom_llvmir_bitcode_write_extract_element(
          module, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_EXTRACT_VALUE:
      return loom_llvmir_bitcode_write_extract_value(
          module, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_INSERT_ELEMENT:
      return loom_llvmir_bitcode_write_insert_element(
          module, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_SHUFFLE_VECTOR:
      return loom_llvmir_bitcode_write_shuffle_vector(
          module, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_CALL:
      return loom_llvmir_bitcode_write_call(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_INLINE_ASM:
      return loom_llvmir_bitcode_write_inline_asm(
          function, value_map, instruction, instruction_value_id, writer);
    case LOOM_LLVMIR_INST_RET:
      return loom_llvmir_bitcode_write_return(module, value_map, instruction,
                                              instruction_value_id, writer);
    case LOOM_LLVMIR_INST_BR:
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BR,
          instruction->br.target);
    case LOOM_LLVMIR_INST_COND_BR: {
      uint64_t operands[3];
      iree_host_size_t operand_count = 0;
      operands[operand_count++] = instruction->cond_br.true_block;
      operands[operand_count++] = instruction->cond_br.false_block;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
          value_map, instruction->cond_br.condition, instruction_value_id,
          operands, &operand_count));
      return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BR, operands,
          operand_count);
    }
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return loom_llvmir_bitcode_write_empty_record(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_UNREACHABLE);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unsupported LLVM instruction passed bitcode "
                              "preflight");
  }
}

static bool loom_llvmir_bitcode_function_has_symbol_names(
    const loom_llvmir_function_t* function) {
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (!iree_string_view_is_empty(function->parameters[i].name)) return true;
  }
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    if (!iree_string_view_is_empty(block->name)) return true;
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        continue;
      }
      const loom_llvmir_value_t* value =
          &function->module->values[instruction->result_value_id];
      if (!iree_string_view_is_empty(value->name)) return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_write_value_symtab_record(
    uint64_t code, uint64_t id, iree_string_view_t name,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (name.size > 0 && name.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode symbol name has null storage");
  }
  if (name.size > IREE_HOST_SIZE_MAX - 1) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode symbol record operand count "
                            "overflows");
  }
  iree_host_size_t operand_count = name.size + 1;
  uint64_t stack_operands[32];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode symbol record operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  operands[0] = id;
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    operands[i + 1] = (uint8_t)name.data[i];
  }
  iree_status_t status =
      loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, code, operands, operand_count);
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_value_symtab_block(
    const loom_llvmir_function_t* function,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_function_has_symbol_names(function)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_VALUE_SYMTAB_BLOCK,
      LOOM_LLVMIR_BITCODE_VALUE_SYMTAB_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < function->parameter_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_parameter_t* parameter = &function->parameters[i];
    if (iree_string_view_is_empty(parameter->name)) continue;
    uint64_t bitcode_value_id = 0;
    status = loom_llvmir_bitcode_map_value(value_map, parameter->value_id,
                                           &bitcode_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_bitcode_write_value_symtab_record(
          LOOM_LLVMIR_BITCODE_VALUE_SYMTAB_CODE_ENTRY, bitcode_value_id,
          parameter->name, writer);
    }
  }
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        continue;
      }
      const loom_llvmir_value_t* value =
          &function->module->values[instruction->result_value_id];
      if (iree_string_view_is_empty(value->name)) continue;
      uint64_t bitcode_value_id = 0;
      status = loom_llvmir_bitcode_map_value(
          value_map, instruction->result_value_id, &bitcode_value_id);
      if (iree_status_is_ok(status)) {
        status = loom_llvmir_bitcode_write_value_symtab_record(
            LOOM_LLVMIR_BITCODE_VALUE_SYMTAB_CODE_ENTRY, bitcode_value_id,
            value->name, writer);
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    if (iree_string_view_is_empty(block->name)) continue;
    status = loom_llvmir_bitcode_write_value_symtab_record(
        LOOM_LLVMIR_BITCODE_VALUE_SYMTAB_CODE_BBENTRY, block->id, block->name,
        writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_function_body(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return iree_ok_status();
  }
  loom_llvmir_bitcode_function_value_map_t value_map;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_function_value_map_initialize(function, &value_map));

  iree_status_t status = loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_BLOCK,
      LOOM_LLVMIR_BITCODE_FUNCTION_ABBREV_WIDTH);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_record_u64(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_DECLAREBLOCKS,
        function->block_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_constants_block(
        function->module, &value_map, function, writer);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_bitcode_write_metadata_block(function, &value_map, writer);
  }
  uint64_t instruction_value_id = value_map.first_instruction_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      status = loom_llvmir_bitcode_write_instruction(
          function, &value_map, instruction, instruction_value_id, writer);
      if (iree_status_is_ok(status) &&
          instruction->result_value_id != LOOM_LLVMIR_VALUE_ID_INVALID) {
        uint64_t result_value_id = 0;
        status = loom_llvmir_bitcode_map_value(
            &value_map, instruction->result_value_id, &result_value_id);
        if (iree_status_is_ok(status) &&
            result_value_id != instruction_value_id) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "LLVM bitcode instruction value numbering is inconsistent");
        }
        instruction_value_id += 1;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_value_symtab_block(function, &value_map,
                                                          writer);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_bitcode_write_metadata_attachment_block(function, writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  loom_llvmir_bitcode_function_value_map_deinitialize(&value_map);
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_global_and_function_records(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t name_offset = 0;
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    const loom_llvmir_global_t* global = module->globals[i];
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_global_record(
        module, global, name_offset, writer));
    name_offset += global->name.size;
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_function_record(
        module, function, name_offset, writer));
    name_offset += function->name.size;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_write_function_bodies(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_write_function_body(module->functions[i], writer));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_build_module_strtab(
    const loom_llvmir_module_t* module, char** out_storage,
    iree_host_size_t* out_storage_size) {
  *out_storage = NULL;
  *out_storage_size = 0;
  iree_host_size_t storage_size = 0;
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    if (module->globals[i]->name.size > IREE_HOST_SIZE_MAX - storage_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode string table is too large");
    }
    storage_size += module->globals[i]->name.size;
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (module->functions[i]->name.size > IREE_HOST_SIZE_MAX - storage_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode string table is too large");
    }
    storage_size += module->functions[i]->name.size;
  }
  if (storage_size == 0) return iree_ok_status();

  iree_allocator_t allocator = iree_allocator_system();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, storage_size, (void**)&storage));
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    iree_string_view_t name = module->globals[i]->name;
    memcpy(storage + offset, name.data, name.size);
    offset += name.size;
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    iree_string_view_t name = module->functions[i]->name;
    memcpy(storage + offset, name.data, name.size);
    offset += name.size;
  }
  *out_storage = storage;
  *out_storage_size = storage_size;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_write_strtab_block(
    iree_string_view_t string_table,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (iree_string_view_is_empty(string_table)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_STRTAB_BLOCK,
      LOOM_LLVMIR_BITCODE_STRTAB_ABBREV_WIDTH));
  uint32_t blob_abbrev = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_define_blob_abbrev(
      writer, LOOM_LLVMIR_BITCODE_STRTAB_CODE_BLOB, &blob_abbrev));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_write_blob_record(
      writer, blob_abbrev, string_table));
  return loom_llvmir_bitcode_record_writer_exit_block(writer);
}

iree_status_t loom_llvmir_bitcode_write_module(
    const loom_llvmir_module_t* module, iree_io_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_module(module));

  char* string_table_storage = NULL;
  iree_host_size_t string_table_size = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_build_module_strtab(
      module, &string_table_storage, &string_table_size));

  static const uint8_t magic[] = {
      LOOM_LLVMIR_BITCODE_MAGIC_0,
      LOOM_LLVMIR_BITCODE_MAGIC_1,
      LOOM_LLVMIR_BITCODE_MAGIC_2,
      LOOM_LLVMIR_BITCODE_MAGIC_3,
  };

  iree_status_t status = iree_io_stream_write(stream, sizeof(magic), magic);
  loom_llvmir_bitstream_writer_t bitstream;
  loom_llvmir_bitcode_record_writer_t record_writer;
  if (iree_status_is_ok(status)) {
    loom_llvmir_bitstream_writer_initialize(stream, &bitstream);
    loom_llvmir_bitcode_record_writer_initialize(&bitstream, &record_writer);

    status = loom_llvmir_bitcode_record_writer_enter_subblock(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_BLOCK,
        LOOM_LLVMIR_BITCODE_MODULE_ABBREV_WIDTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_record_u64(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_VERSION,
        LOOM_LLVMIR_BITCODE_MODULE_VERSION);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.source_name)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_SOURCE_FILENAME,
        module->target_config.source_name);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.target_triple)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_TRIPLE,
        module->target_config.target_triple);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.data_layout)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_DATALAYOUT,
        module->target_config.data_layout);
  }
  if (iree_status_is_ok(status) &&
      (module->type_count != 0 || module->function_count != 0)) {
    status = loom_llvmir_bitcode_write_type_block(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_attr_group_block(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_attr_block(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_global_and_function_records(
        module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_module_constants_block(module,
                                                              &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_bitcode_write_metadata_kind_block(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_sync_scope_names_block(module,
                                                              &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_function_bodies(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(&record_writer);
  }
  if (iree_status_is_ok(status) && string_table_size != 0) {
    status = loom_llvmir_bitcode_write_strtab_block(
        iree_make_string_view(string_table_storage, string_table_size),
        &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitstream_writer_finish(&bitstream);
  }
  iree_allocator_free(iree_allocator_system(), string_table_storage);
  return status;
}
