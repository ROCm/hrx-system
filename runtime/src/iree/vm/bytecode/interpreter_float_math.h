// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_FLOAT_MATH_H_
#define IREE_VM_BYTECODE_INTERPRETER_FLOAT_MATH_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// These helpers consume and produce raw IEEE payloads. Invocation drive
// segments establish the architectural floating-point environment before
// reaching them, and f32 results occupy only the low 32 bits of a value cell.

// Evaluates a verified unary f32 selector over a raw IEEE payload.
IREE_ATTRIBUTE_NOINLINE uint32_t
iree_vm_bytecode_float_math_unary_f32(uint8_t selector, uint32_t source_bits);

// Evaluates a verified unary f64 selector over a raw IEEE payload.
IREE_ATTRIBUTE_NOINLINE uint64_t
iree_vm_bytecode_float_math_unary_f64(uint8_t selector, uint64_t source_bits);

// Evaluates a verified binary f32 selector over raw IEEE payloads.
IREE_ATTRIBUTE_NOINLINE uint32_t iree_vm_bytecode_float_math_binary_f32(
    uint8_t selector, uint32_t lhs_bits, uint32_t rhs_bits);

// Evaluates a verified binary f64 selector over raw IEEE payloads.
IREE_ATTRIBUTE_NOINLINE uint64_t iree_vm_bytecode_float_math_binary_f64(
    uint8_t selector, uint64_t lhs_bits, uint64_t rhs_bits);

// Evaluates a verified ternary f32 selector over raw IEEE payloads.
IREE_ATTRIBUTE_NOINLINE uint32_t iree_vm_bytecode_float_math_ternary_f32(
    uint8_t selector, uint32_t a_bits, uint32_t b_bits, uint32_t c_bits);

// Evaluates a verified ternary f64 selector over raw IEEE payloads.
IREE_ATTRIBUTE_NOINLINE uint64_t iree_vm_bytecode_float_math_ternary_f64(
    uint8_t selector, uint64_t a_bits, uint64_t b_bits, uint64_t c_bits);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_FLOAT_MATH_H_
