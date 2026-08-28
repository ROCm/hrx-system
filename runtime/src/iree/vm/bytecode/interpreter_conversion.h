// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_
#define IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core/conversion.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Failure produced by a verified float-to-integer conversion record.
typedef enum iree_vm_bytecode_conversion_failure_e {
  IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE = 0,
  IREE_VM_BYTECODE_CONVERSION_FAILURE_NAN,
  IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE,
} iree_vm_bytecode_conversion_failure_t;

// Returns the low |bit_count| bits of |value| with all higher bits clear.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_low_bits(uint64_t value, uint32_t bit_count) {
  return value & (UINT64_MAX >> (64 - bit_count));
}

// Sign-extends the low source bits through the destination width and clears all
// higher cell bits.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_sign_extend(uint64_t value, uint32_t source_bit_count,
                                     uint32_t destination_bit_count) {
  const uint64_t sign_bit = UINT64_C(1) << (source_bit_count - 1);
  value =
      (iree_vm_bytecode_integer_low_bits(value, source_bit_count) ^ sign_bit) -
      sign_bit;
  return iree_vm_bytecode_integer_low_bits(value, destination_bit_count);
}

// Executes one verified exact integer-width conversion record.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_conversion_integer(
    const iree_vm_isa_conversion_integer_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  uint64_t result = 0;
  switch (record->selector_u8) {
#define IREE_VM_BYTECODE_INTEGER_SIGN_EXTEND_CASE(selector, source_bit_count, \
                                                  destination_bit_count)      \
  case IREE_VM_ISA_INTEGER_CONVERT_##selector:                                \
    result = iree_vm_bytecode_integer_sign_extend(source, (source_bit_count), \
                                                  (destination_bit_count));   \
    break;
#define IREE_VM_BYTECODE_INTEGER_ZERO_EXTEND_CASE(selector, source_bit_count, \
                                                  destination_bit_count)      \
  case IREE_VM_ISA_INTEGER_CONVERT_##selector:                                \
    result = iree_vm_bytecode_integer_low_bits(source, source_bit_count);     \
    break;
#define IREE_VM_BYTECODE_INTEGER_TRUNCATE_CASE(selector, source_bit_count,     \
                                               destination_bit_count)          \
  case IREE_VM_ISA_INTEGER_CONVERT_##selector:                                 \
    result = iree_vm_bytecode_integer_low_bits(source, destination_bit_count); \
    break;
#define IREE_VM_BYTECODE_DEFINE_INTEGER_CONVERSION_CASES
#include "iree/vm/bytecode/execution_tables.inl"
#undef IREE_VM_BYTECODE_DEFINE_INTEGER_CONVERSION_CASES
#undef IREE_VM_BYTECODE_INTEGER_TRUNCATE_CASE
#undef IREE_VM_BYTECODE_INTEGER_ZERO_EXTEND_CASE
#undef IREE_VM_BYTECODE_INTEGER_SIGN_EXTEND_CASE
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
  values[record->dst_v8] = result;
}

// Executes one verified narrow-float extension record.
void iree_vm_bytecode_execute_conversion_float_extend(
    const iree_vm_isa_conversion_float_extend_record_t* record,
    uint64_t* values);

// Executes one verified direct float truncation record.
void iree_vm_bytecode_execute_conversion_float_truncate(
    const iree_vm_isa_conversion_float_truncate_record_t* record,
    uint64_t* values);

// Executes one verified f32/f64 width conversion record.
void iree_vm_bytecode_execute_conversion_float_width(
    const iree_vm_isa_conversion_float_width_record_t* record,
    uint64_t* values);

// Executes one verified direct integer-to-float conversion record.
void iree_vm_bytecode_execute_conversion_integer_to_float(
    const iree_vm_isa_conversion_integer_to_float_record_t* record,
    uint64_t* values);

// Executes one verified float-to-integer record. The destination is unchanged
// when the source is a NaN or lies outside the selected successful interval.
iree_vm_bytecode_conversion_failure_t
iree_vm_bytecode_execute_conversion_float_to_integer(
    const iree_vm_isa_conversion_float_to_integer_record_t* record,
    uint64_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_
