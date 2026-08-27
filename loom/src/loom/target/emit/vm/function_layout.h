// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM function record-stream layout.

#ifndef LOOM_TARGET_EMIT_VM_FUNCTION_LAYOUT_H_
#define LOOM_TARGET_EMIT_VM_FUNCTION_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selected direct-control encoding for one structural Low packet.
typedef uint8_t loom_vm_function_control_encoding_t;
enum loom_vm_function_control_encoding_e {
  LOOM_VM_FUNCTION_CONTROL_ENCODING_NONE = 0,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S16 = 1,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_S32 = 2,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16 = 3,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32 = 4,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S16 = 5,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_UNLESS_S32 = 6,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S16 = 7,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S16_BRANCH_S32 = 8,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S16 = 9,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_BRANCH_IF_S32_BRANCH_S32 = 10,
  LOOM_VM_FUNCTION_CONTROL_ENCODING_COUNT = 11,
};

// Physical records selected by one direct-control encoding.
typedef struct loom_vm_function_control_layout_t {
  // First control-record opcode, or zero when no record is emitted.
  uint8_t first_opcode;
  // First control-record byte length, or zero when absent.
  uint8_t first_byte_length;
  // Whether the first record reads the structural condition operand.
  bool first_uses_condition;
  // Second unconditional branch opcode, or zero when absent.
  uint8_t second_opcode;
  // Second unconditional branch byte length, or zero when absent.
  uint8_t second_byte_length;
} loom_vm_function_control_layout_t;

// Exact record-stream layout for one scheduled and allocated Low function.
// Arrays are indexed by schedule block or packet ordinal and owned by the
// caller-provided arena.
typedef struct loom_vm_function_code_layout_t {
  // Function-relative byte offsets of each control.block record.
  uint32_t* block_offsets;
  // Function-relative byte offsets of each scheduled packet's emitted range.
  uint32_t* packet_offsets;
  // Selected control encodings indexed by scheduled packet ordinal.
  loom_vm_function_control_encoding_t* control_encodings;
  // Exact complete function record-stream byte length.
  uint32_t bytecode_length;
} loom_vm_function_code_layout_t;

// Returns the fixed byte length of a sequential descriptor record.
uint8_t loom_vm_function_descriptor_record_byte_length(
    uint32_t descriptor_ordinal);

// Returns the physical record layout selected by |encoding|.
const loom_vm_function_control_layout_t*
loom_vm_function_control_encoding_layout(
    loom_vm_function_control_encoding_t encoding);

// Computes exact packet and block offsets and selects direct-branch encodings
// without serializing any instruction record. |arena| must outlive
// |out_layout|.
iree_status_t loom_vm_function_code_layout_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_vm_function_code_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_FUNCTION_LAYOUT_H_
