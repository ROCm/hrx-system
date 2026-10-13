// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_NATIVE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_NATIVE_H_

#include "iree/vm/bytecode/assembler_lexer.h"
#include "iree/vm/bytecode/wire/module.h"
#include "iree/vm/native_call.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_vm_bytecode_assembler_native_operand_t
    iree_vm_bytecode_assembler_native_operand_t;

// Parsed grouped operands retained until symbolic imports/functions resolve.
// Selector domains and result directions remain explicit until encoding checks
// them against the imported layout. No physical count or offset is authored.
typedef struct iree_vm_bytecode_assembler_native_call_t {
  // Next call in source/ordinal order, or NULL at the end.
  struct iree_vm_bytecode_assembler_native_call_t* next;
  // Owned preorder operand tree, with an enclosing record at index zero.
  iree_vm_bytecode_assembler_native_operand_t* operands;
  // Derived word extent of all fixed records and range headers.
  uint32_t word_count;
} iree_vm_bytecode_assembler_native_call_t;

// Parses records `(fields...)`, ranges `[(fields...), ...]`, and scalar/ref
// selectors `%vN`, `%rN`, `#vN`, `#rN`, optionally prefixed with `out`.
// `#vN` is a local byte offset and `#rN` is a local ref slot, not an absolute
// ref-bank index. `[() * N]` compactly represents N zero-field records.
// The returned tree is owned by the caller; failure leaves the output NULL.
iree_status_t iree_vm_bytecode_assembler_native_parse(
    iree_vm_bytecode_assembler_lexer_t* lexer, iree_allocator_t host_allocator,
    iree_vm_bytecode_assembler_native_call_t** out_call);

// Frees one parsed call, but not its linked successor.
void iree_vm_bytecode_assembler_native_free(
    iree_vm_bytecode_assembler_native_call_t* call,
    iree_allocator_t host_allocator);

// Matches external grouped operands to a validated declaration and encodes
// canonical count/offset/selector words into |words| of call->word_count.
// Function storage extents are trusted assembler-produced declarations.
iree_status_t iree_vm_bytecode_assembler_native_encode(
    const iree_vm_bytecode_assembler_native_call_t* call,
    const iree_vm_native_call_schema_t* schema,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t* words,
    iree_allocator_t scratch_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_VM_BYTECODE_ASSEMBLER_NATIVE_H_
