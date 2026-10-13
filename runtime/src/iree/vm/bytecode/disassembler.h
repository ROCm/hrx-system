// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_DISASSEMBLER_H_
#define IREE_VM_BYTECODE_DISASSEMBLER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the canonical mnemonic for |opcode| or an empty view when unknown.
IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_disassembler_instruction_name(uint8_t opcode);

// Returns the canonical module-record name or an empty view when unknown.
IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_disassembler_module_record_name(uint8_t record_ordinal);

// Callback invoked for each ordered fragment of canonical module text.
//
// |fragment| is valid only for the duration of the callback. Returning a
// non-OK status stops disassembly immediately and transfers the status to the
// caller unchanged.
typedef iree_status_t(IREE_API_PTR* iree_vm_bytecode_disassembler_write_fn_t)(
    void* user_data, iree_string_view_t fragment);

// Streaming text sink used by the bytecode disassembler.
typedef struct iree_vm_bytecode_disassembler_write_callback_t {
  // Function receiving the next canonical text fragment.
  iree_vm_bytecode_disassembler_write_fn_t fn;
  // Unowned state passed to |fn|.
  void* user_data;
} iree_vm_bytecode_disassembler_write_callback_t;

// Disassembles one complete VM bytecode image into canonical UTF-8 text.
//
// |contents| is borrowed for the duration of the call and must be naturally
// aligned as required by the bytecode image format. The image is fully
// verified and checked for supported presentation before |write_callback| is
// invoked. Malformed or unsupported images therefore produce no text. Once
// output begins, a callback failure may leave its destination partially
// written and is returned unchanged.
//
// The emitted text uses LF line endings and is accepted by the matching VM
// bytecode assembler. |scratch_allocator| is used only for verification and
// unusually large function control-flow graphs or deeply nested native
// operands; common small functions do not allocate while being printed.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_disassemble_module(
    iree_const_byte_span_t contents,
    iree_vm_bytecode_disassembler_write_callback_t write_callback,
    iree_allocator_t scratch_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_DISASSEMBLER_H_
