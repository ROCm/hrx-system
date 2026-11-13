// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_STRING_MODULE_H_
#define IREE_VM_MODULE_STRING_MODULE_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates the synchronous "string" module. Its codepoint operations use the
// runtime's Unicode 15.1 properties and locale-independent full case mappings:
//
//   codepoint.to_upper(i32) -> (i32 count, i32 first, i32 second, i32 third)
//   codepoint.to_lower(i32) -> (i32 count, i32 first, i32 second)
//   codepoint.is_decimal(i32) -> i32
//   codepoint.is_digit(i32) -> i32
//   codepoint.is_whitespace(i32) -> i32
//
// Codepoints are unsigned 32-bit values. Case mappings return at least one
// codepoint, zero unused slots, and preserve unmapped or invalid input.
// Predicates return 0 or 1; decimal selects Numeric_Type=Decimal, digit selects
// Decimal or Digit, and whitespace selects White_Space. Language-specific
// policies such as Python's additional whitespace controls remain with the
// caller.
//
// Calls allocate nothing, retain no references, and use no per-process state.
// The module is independent of a VM environment or serving host. It can be
// shared by programs and concurrent processes. |out_module| is set null before
// construction; the caller releases the returned owner after linking.
IREE_API_EXPORT iree_status_t iree_vm_string_module_create(
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_STRING_MODULE_H_
