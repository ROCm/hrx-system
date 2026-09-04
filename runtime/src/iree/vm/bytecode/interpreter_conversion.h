// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_
#define IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core.h"

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
void iree_vm_bytecode_execute_conversion_integer(
    const iree_vm_bytecode_conversion_integer_t* record, uint64_t* values);

// Executes one verified narrow-float extension record.
void iree_vm_bytecode_execute_conversion_float_extend(
    const iree_vm_bytecode_conversion_float_extend_t* record, uint64_t* values);

// Executes one verified direct float truncation record.
void iree_vm_bytecode_execute_conversion_float_truncate(
    const iree_vm_bytecode_conversion_float_truncate_t* record,
    uint64_t* values);

// Executes one verified f32/f64 width conversion record.
void iree_vm_bytecode_execute_conversion_float_width(
    const iree_vm_bytecode_conversion_float_width_t* record, uint64_t* values);

// Executes one verified direct integer-to-float conversion record.
void iree_vm_bytecode_execute_conversion_integer_to_float(
    const iree_vm_bytecode_conversion_integer_to_float_t* record,
    uint64_t* values);

// Executes one verified float-to-integer record. The destination is unchanged
// when the source is a NaN or lies outside the selected successful interval.
iree_vm_bytecode_conversion_failure_t
iree_vm_bytecode_execute_conversion_float_to_integer(
    const iree_vm_bytecode_conversion_float_to_integer_t* record,
    uint64_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_CONVERSION_H_
