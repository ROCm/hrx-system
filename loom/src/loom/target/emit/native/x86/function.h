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
#include "loom/target/emit/native/object.h"
#include "loom/target/emit/native/x86/abi.h"
#include "loom/target/emit/native/x86/encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_x86_instruction_t {
  // Fully allocated operands; branch displacement is resolved during layout.
  loom_x86_encoding_operands_t operands;
  // Shared CFG block ordinal for branches, module symbol ID for calls and
  // symbolic addresses, or UINT32_MAX without a reference. The encoding format
  // determines the namespace; this has no effect on instruction size or
  // register use.
  uint32_t reference;
  // Native scalar form or packed vector recipe.
  uint16_t encoding_format_id;
  // Immutable opcode fields from the descriptor.
  uint16_t encoding_id;
} loom_x86_instruction_t;

// Prepared instructions borrow no IR, schedule, or allocation storage. They
// remain valid until their arena is reset. The producer consumes each shared
// packet and its final transport once; byte encoding never revisits the IR.
typedef struct loom_x86_function_t {
  // Arena-owned entry transport followed by instructions in final block order.
  loom_x86_instruction_t* instructions;
  // Number of instructions, excluding the preservation envelope.
  iree_host_size_t instruction_count;
  // Number of native symbol relocation rows written by this function.
  iree_host_size_t symbol_fixup_count;
  // First instruction of each shared CFG block and the common epilogue. Entry
  // transport precedes every block and runs only when the function is invoked.
  iree_host_size_t* block_starts;
  // Number of CFG blocks, excluding the epilogue.
  uint32_t block_count;
  // Callee-preserved GPRs actually written by instructions or transport.
  uint16_t saved_registers;
  // True when the function may leave the upper state of YMM/ZMM0-15 dirty.
  bool may_dirty_upper_vector_state;
  // True when the public result occupies live YMM/ZMM0-15 upper state.
  bool has_upper_vector_result;
  // Instruction ordinals of calls that admit VZEROUPPER immediately before
  // CALL because no register argument occupies live YMM/ZMM0-15 upper state.
  const uint32_t* upper_vector_call_cleanup_indices;
  // Number of entries in |upper_vector_call_cleanup_indices|.
  iree_host_size_t upper_vector_call_cleanup_count;
  // Fixed stack allocation after callee saves. Byte emission consumes these
  // concrete adjustments without computing alignment or selecting scratch.
  struct {
    // Bytes subtracted from RSP after any alignment adjustment.
    uint32_t allocation_size;
    // Required alignment of the body RSP, or zero without local storage.
    uint32_t alignment;
    // Dynamic restoration for alignments stronger than the SysV entry promise.
    struct {
      // Immediate AND mask, or zero when static padding suffices.
      int32_t mask;
      // Byte offset of the saved pre-alignment RSP within the allocation.
      uint32_t saved_pointer_offset;
      // Caller-clobbered register used before invocation transport.
      uint8_t scratch_register;
    } realignment;
    // True when RBP retains the post-save RSP for incoming SIMD stack values.
    bool has_frame_pointer;
  } stack;
} loom_x86_function_t;

// Returns the allocator reservations needed by the native function envelope.
// Descriptor fragments without GPR values need no reservation: the envelope
// may still address RSP/RBP directly because those registers cannot be assigned
// from the fragment's value classes.
iree_host_size_t loom_x86_function_reserved_ranges(
    const loom_x86_function_abi_t* function_abi,
    loom_low_allocation_reserved_range_t out_ranges[3]);

// Materializes native instructions from an accepted allocated Low frame. The
// retained SysV plan supplies entry, call, and result transport. Stack,
// scratch, and private storage share the native stack; workgroup storage has no
// ordinary host-function ABI. Unsupported authored instructions emit
// diagnostics and leave |out_accepted| false. Status failures describe
// allocation, output, or diagnostic-sink failures.
iree_status_t loom_x86_function_prepare(
    const loom_low_emission_frame_t* frame,
    const loom_x86_function_abi_t* function_abi,
    const loom_x86_module_abi_t* module_abi, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_x86_function_t* out_function);

// Encodes the prepared envelope and instructions, then resolves branch fields
// against the retained block map. Requires a writable, seekable stream. This
// routine has no access to source IR or allocation and makes no ABI decisions.
// |symbol_indices| translates retained module symbol IDs into object symbols.
// |fixups| has space for |function->symbol_fixup_count| records relative to the
// supplied section contribution. Both may be NULL when the function has no
// symbol references.
iree_status_t loom_x86_function_write(const loom_x86_function_t* function,
                                      const uint32_t* symbol_indices,
                                      iree_host_size_t section_index,
                                      loom_native_object_fixup_t* fixups,
                                      iree_io_stream_t* stream,
                                      iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_FUNCTION_H_
