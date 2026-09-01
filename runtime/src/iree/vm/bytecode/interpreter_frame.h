// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_FRAME_H_
#define IREE_VM_BYTECODE_INTERPRETER_FRAME_H_

#include <string.h>

#include "iree/vm/bytecode/module_storage.h"

// Caller-owned ref packet regions kept live across one suspended child call.
typedef struct iree_vm_bytecode_call_cleanup_t {
  // First direct argument state not replaced by direct results.
  iree_vm_ref_t* direct_arguments;
  // Number of contiguous direct argument states to release.
  uint16_t direct_argument_count;
  // Number of argument states at the start of the local-ref bank to release.
  uint16_t overflow_argument_count;
} iree_vm_bytecode_call_cleanup_t;

// Bytecode execution state followed by exact function-owned banks when stored
// in a durable frame.
typedef struct iree_vm_bytecode_execution_state_t {
  // Canonical call packet whose bank bases remain live through completion.
  const iree_vm_call_packet_t* call;
  // Next verified instruction record to execute.
  const uint8_t* program_counter;
  // Owning durable frame, or null for one synchronous transient execution.
  iree_vm_frame_t* frame;
  // Function value-register bank.
  uint64_t* values;
  // Function ref-register bank followed by local ref slots.
  iree_vm_ref_t* refs;
  // First function-local ref slot.
  iree_vm_ref_t* local_refs;
  // Function-register bank followed by local function slots.
  iree_vm_function_ref_t* functions;
  // First function-local function slot.
  iree_vm_function_ref_t* local_functions;
  // Function-local byte storage.
  uint8_t* local_bytes;
  // Number of ref states released by durable frame cleanup.
  uint32_t ref_count;
  // Ref packet ownership pending one suspended child completion.
  iree_vm_bytecode_call_cleanup_t pending_call_cleanup;
} iree_vm_bytecode_execution_state_t;

// Exact durable payload layout for one bytecode function frame.
typedef struct iree_vm_bytecode_frame_layout_t {
  // Generic frame payload request.
  iree_vm_frame_layout_t frame;
  // Byte offset of the durable call packet.
  iree_host_size_t call_offset;
  // Byte offset of the value-register bank.
  iree_host_size_t values_offset;
  // Byte offset of the ref-register bank.
  iree_host_size_t refs_offset;
  // Byte offset of the local ref slots.
  iree_host_size_t local_refs_offset;
  // Byte offset of the function-register bank.
  iree_host_size_t functions_offset;
  // Byte offset of the local function slots.
  iree_host_size_t local_functions_offset;
  // Byte offset of the local byte storage.
  iree_host_size_t local_bytes_offset;
} iree_vm_bytecode_frame_layout_t;

// Copies one direct physical value bank in fixed native-vector-sized chunks.
// Direct banks have a fixed small bound, so this remains a bounded sequence
// with no variable-length memcpy call or per-cell bounds branch.
static inline void iree_vm_bytecode_frame_copy_direct_values(
    uint64_t* target, const uint64_t* source, uint16_t count) {
  while (count >= 4) {
    memcpy(target, source, 4 * sizeof(uint64_t));
    target += 4;
    source += 4;
    count -= 4;
  }
  if (count >= 2) {
    memcpy(target, source, 2 * sizeof(uint64_t));
    target += 2;
    source += 2;
    count -= 2;
  }
  if (count) {
    *target = *source;
  }
}

// Initializes the exact register and local banks for one function entry.
void iree_vm_bytecode_frame_initialize(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    iree_vm_bytecode_execution_state_t* state);

// Releases and clears the first |ref_count| states in |refs|.
void iree_vm_bytecode_frame_reset_refs(iree_vm_ref_t* refs, uint32_t ref_count);

// Calculates the exact durable payload layout for |function|.
iree_status_t iree_vm_bytecode_query_frame_layout(
    const iree_vm_bytecode_v0_function_row_t* function,
    iree_vm_bytecode_frame_layout_t* out_layout);

// Releases every ref state owned by a durable bytecode frame.
void iree_vm_bytecode_frame_cleanup(iree_vm_frame_t* frame);

#endif  // IREE_VM_BYTECODE_INTERPRETER_FRAME_H_
