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

// Executes one verified exact integer-width conversion record.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_conversion_integer(
    const iree_vm_isa_conversion_integer_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  uint64_t result = 0;
  switch (record->selector_u8) {
    case IREE_VM_ISA_INTEGER_CONVERT_S8_TO_I32:
      result =
          (source & 0x80u) ? (uint32_t)source | 0xFFFFFF00u : (uint8_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_U8_TO_I32:
      result = (uint8_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_S16_TO_I32:
      result = (source & 0x8000u) ? (uint32_t)source | 0xFFFF0000u
                                  : (uint16_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_U16_TO_I32:
      result = (uint16_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64:
      result = (source & UINT32_C(0x80000000))
                   ? (uint32_t)source | UINT64_C(0xFFFFFFFF00000000)
                   : (uint32_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64:
    case IREE_VM_ISA_INTEGER_CONVERT_I64_TO_I32:
      result = (uint32_t)source;
      break;
    case IREE_VM_ISA_INTEGER_CONVERT_I32_TO_I8:
      result = (uint8_t)source;
      break;
    default:
      result = (uint16_t)source;
      break;
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
