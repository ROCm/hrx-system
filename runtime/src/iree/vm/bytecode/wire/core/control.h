// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.control.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONTROL_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONTROL_H_

#include <stdint.h>

// Selects one non-OK architectural status code for control.fail; zero is
// deliberately absent because OK is not a failure.
typedef uint8_t iree_vm_isa_control_status_t;
enum {
  // Produces the canonical cancelled status code.
  IREE_VM_ISA_CONTROL_STATUS_CANCELLED = 0x01,
  // Produces the canonical unknown status code.
  IREE_VM_ISA_CONTROL_STATUS_UNKNOWN = 0x02,
  // Produces the canonical invalid_argument status code.
  IREE_VM_ISA_CONTROL_STATUS_INVALID_ARGUMENT = 0x03,
  // Produces the canonical deadline_exceeded status code.
  IREE_VM_ISA_CONTROL_STATUS_DEADLINE_EXCEEDED = 0x04,
  // Produces the canonical not_found status code.
  IREE_VM_ISA_CONTROL_STATUS_NOT_FOUND = 0x05,
  // Produces the canonical already_exists status code.
  IREE_VM_ISA_CONTROL_STATUS_ALREADY_EXISTS = 0x06,
  // Produces the canonical permission_denied status code.
  IREE_VM_ISA_CONTROL_STATUS_PERMISSION_DENIED = 0x07,
  // Produces the canonical resource_exhausted status code.
  IREE_VM_ISA_CONTROL_STATUS_RESOURCE_EXHAUSTED = 0x08,
  // Produces the canonical failed_precondition status code.
  IREE_VM_ISA_CONTROL_STATUS_FAILED_PRECONDITION = 0x09,
  // Produces the canonical aborted status code.
  IREE_VM_ISA_CONTROL_STATUS_ABORTED = 0x0A,
  // Produces the canonical out_of_range status code.
  IREE_VM_ISA_CONTROL_STATUS_OUT_OF_RANGE = 0x0B,
  // Produces the canonical unimplemented status code.
  IREE_VM_ISA_CONTROL_STATUS_UNIMPLEMENTED = 0x0C,
  // Produces the canonical internal status code.
  IREE_VM_ISA_CONTROL_STATUS_INTERNAL = 0x0D,
  // Produces the canonical unavailable status code.
  IREE_VM_ISA_CONTROL_STATUS_UNAVAILABLE = 0x0E,
  // Produces the canonical data_loss status code.
  IREE_VM_ISA_CONTROL_STATUS_DATA_LOSS = 0x0F,
  // Produces the canonical unauthenticated status code.
  IREE_VM_ISA_CONTROL_STATUS_UNAUTHENTICATED = 0x10,
  // Produces the canonical incompatible status code.
  IREE_VM_ISA_CONTROL_STATUS_INCOMPATIBLE = 0x12,
};

enum {
  IREE_VM_ISA_CONTROL_STATUS_CANCELLED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_UNKNOWN_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_INVALID_ARGUMENT_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_DEADLINE_EXCEEDED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_NOT_FOUND_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_ALREADY_EXISTS_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_PERMISSION_DENIED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_RESOURCE_EXHAUSTED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_FAILED_PRECONDITION_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_ABORTED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_OUT_OF_RANGE_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_UNIMPLEMENTED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_INTERNAL_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_UNAVAILABLE_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_DATA_LOSS_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_UNAUTHENTICATED_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_STATUS_INCOMPATIBLE_SINCE_MINOR = 0,
};

// Page 0x00, opcode 0x01: Marks the only legal direct control target.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_control_block_record_t;

// Page 0x00, opcode 0x02: Validates, publishes, and returns function results.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_control_return_record_t;

// Page 0x00, opcode 0x03: Suspends with one explicit wide continuation target.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Signed four-byte-word displacement from the end of this record.
  int32_t target_rel32;
} iree_vm_isa_control_yield_s32_record_t;

// Page 0x00, opcode 0x04: Unconditionally transfers to the verified direct
// target.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Signed four-byte-word displacement from the end of this record.
  int16_t target_rel16;
} iree_vm_isa_control_branch_s16_record_t;

// Page 0x00, opcode 0x05: Unconditionally transfers to the verified direct
// target.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
  // Signed four-byte-word displacement from the end of this record.
  int32_t target_rel32;
} iree_vm_isa_control_branch_s32_record_t;

// Page 0x00, opcode 0x06: Transfers to the verified target when the complete
// condition is nonzero; otherwise continues at the following record.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Complete 64-bit branch condition.
  uint8_t condition_v8;
  // Signed four-byte-word displacement from the end of this record.
  int16_t target_rel16;
} iree_vm_isa_control_branch_if_s16_record_t;

// Page 0x00, opcode 0x07: Transfers to the verified target when the complete
// condition is nonzero; otherwise continues at the following record.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Complete 64-bit branch condition.
  uint8_t condition_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Signed four-byte-word displacement from the end of this record.
  int32_t target_rel32;
} iree_vm_isa_control_branch_if_s32_record_t;

// Page 0x00, opcode 0x08: Transfers to the verified target when the complete
// condition is zero; otherwise continues at the following record.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Complete 64-bit branch condition.
  uint8_t condition_v8;
  // Signed four-byte-word displacement from the end of this record.
  int16_t target_rel16;
} iree_vm_isa_control_branch_unless_s16_record_t;

// Page 0x00, opcode 0x09: Transfers to the verified target when the complete
// condition is zero; otherwise continues at the following record.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Complete 64-bit branch condition.
  uint8_t condition_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Signed four-byte-word displacement from the end of this record.
  int32_t target_rel32;
} iree_vm_isa_control_branch_unless_s32_record_t;

// Page 0x00, opcode 0x0A: Branches through a function-local dense target table.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Unsigned complete-cell zero-based table selector.
  uint8_t selector_v8;
  // Number of switch-target entries in this table slice.
  uint16_t target_count_u16;
  // First entry relative to the owning function's target range.
  uint32_t target_base_u32;
} iree_vm_isa_control_switch_record_t;

// Page 0x00, opcode 0x0B: Calls a local or linked imported function.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Local, required-import, or optional-import target selector.
  uint8_t target_kind_u8;
  // Direct ordinal in the target-kind-selected table.
  uint16_t target_ordinal_u16;
  // Ownership-transfer bits for direct ref arguments.
  uint16_t direct_ref_move_mask_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_control_call_record_t;

// Page 0x00, opcode 0x0C: Calls a dynamic first-class function value.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Dynamic first-class function target.
  uint8_t target_f8;
  // Exact expected structural callable type.
  uint16_t callable_type_ordinal_u16;
  // Ownership-transfer bits for direct ref arguments.
  uint16_t direct_ref_move_mask_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_control_call_indirect_record_t;

// Page 0x00, opcode 0x0D: Continues on true and fails with failed_precondition
// on false.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Complete 64-bit assertion condition.
  uint8_t condition_v8;
  // Optional best-effort readable vm.buffer diagnostic message.
  uint8_t message_r8_nullable;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_control_assert_record_t;

// Page 0x00, opcode 0x0E: Terminates with one immediate architectural status.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Assigned non-OK architectural status selector.
  uint8_t status_u8;
  // Optional best-effort readable vm.buffer diagnostic message.
  uint8_t message_r8_nullable;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_control_fail_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONTROL_H_
// clang-format on
