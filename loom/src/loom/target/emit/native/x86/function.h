// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete native instructions and preservation for an allocated x86 function.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_

#include "iree/io/stream.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/emit/native/x86/encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_x86_instruction_t {
  // Fully allocated operands; branch displacement is resolved during layout.
  loom_x86_encoding_operands_t operands;
  // Shared CFG block ordinal, including the epilogue, or UINT32_MAX.
  uint32_t branch_target;
  // Native operand/encoding form.
  uint16_t form;
  // Immutable opcode fields from the descriptor.
  uint16_t encoding_id;
} loom_x86_instruction_t;

// Prepared instructions borrow no IR, schedule, or allocation storage. They
// remain valid until their arena is reset. The producer consumes each shared
// packet and its final transport once; byte encoding never revisits the IR.
typedef struct loom_x86_function_t {
  // Arena-owned instructions in final block order.
  loom_x86_instruction_t* instructions;
  // Number of instructions, excluding the preservation envelope.
  iree_host_size_t instruction_count;
  // First instruction of each shared CFG block and the common epilogue.
  iree_host_size_t* block_starts;
  // Number of CFG blocks, excluding the epilogue.
  uint32_t block_count;
  // Callee-preserved GPRs actually written by instructions or transport.
  uint16_t saved_registers;
} loom_x86_function_t;

// Materializes native instructions from an accepted spill-free scalar frame.
// SysV result transport uses RAX. The caller already applied the entry ABI and
// reserved RSP. No retained calls or stack storage are admitted by this leaf
// product. Unsupported authored instructions return UNIMPLEMENTED.
iree_status_t loom_x86_function_prepare(const loom_low_emission_frame_t* frame,
                                        iree_arena_allocator_t* arena,
                                        loom_x86_function_t* out_function);

// Encodes the prepared envelope and instructions, then resolves branch fields
// against the retained block map. Requires a writable, seekable stream. This
// routine has no access to source IR or allocation and makes no ABI decisions.
iree_status_t loom_x86_function_write(const loom_x86_function_t* function,
                                      iree_io_stream_t* stream,
                                      iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_
