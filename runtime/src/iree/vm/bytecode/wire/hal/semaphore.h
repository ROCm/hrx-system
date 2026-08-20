// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for hal.family.semaphore.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_HAL_SEMAPHORE_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_HAL_SEMAPHORE_H_

#include <stdint.h>

// Selects the satisfaction rule and successful result for a timepoint set.
typedef uint8_t iree_vm_isa_semaphore_await_mode_t;
enum {
  // Completes when every timepoint is satisfied and returns UINT64_MAX; an
  // empty set succeeds.
  IREE_VM_ISA_SEMAPHORE_AWAIT_MODE_ALL = 0x00,
  // Requires a nonempty set, completes when one timepoint is satisfied, and
  // returns a selected zero-based index.
  IREE_VM_ISA_SEMAPHORE_AWAIT_MODE_ANY = 0x01,
};

enum {
  IREE_VM_ISA_SEMAPHORE_AWAIT_MODE_ALL_SINCE_MINOR = 0,
  IREE_VM_ISA_SEMAPHORE_AWAIT_MODE_ANY_SINCE_MINOR = 0,
};

// Selects the signed-i64 nanosecond interpretation of the timeout register.
typedef uint8_t iree_vm_isa_semaphore_await_timeout_kind_t;
enum {
  // Zero polls, positive finite values become one saturating monotonic
  // deadline, INT64_MAX waits indefinitely, and negatives fail.
  IREE_VM_ISA_SEMAPHORE_AWAIT_TIMEOUT_KIND_RELATIVE = 0x00,
  // INT64_MIN polls, INT64_MAX waits indefinitely, and every other value is an
  // exact monotonic deadline.
  IREE_VM_ISA_SEMAPHORE_AWAIT_TIMEOUT_KIND_ABSOLUTE = 0x01,
};

enum {
  IREE_VM_ISA_SEMAPHORE_AWAIT_TIMEOUT_KIND_RELATIVE_SINCE_MINOR = 0,
  IREE_VM_ISA_SEMAPHORE_AWAIT_TIMEOUT_KIND_ABSOLUTE_SINCE_MINOR = 0,
};

// Page 0xF0, opcode 0x06: Creates one timeline semaphore on an explicit device.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.semaphore ref.
  uint8_t dst_r8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Initial architectural semaphore payload.
  uint8_t initial_value_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // HOST_INTERRUPT, EXPORTABLE, and EXPORTABLE_TIMEPOINTS flag bits.
  uint32_t flags_u32;
} iree_vm_isa_hal_semaphore_create_record_t;

// Page 0xF0, opcode 0x07: Queries a timeline payload or propagates its sticky
// failure.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Current architectural semaphore payload.
  uint8_t dst_v8;
  // required exact hal.semaphore ref.
  uint8_t semaphore_r8;
} iree_vm_isa_hal_semaphore_query_record_t;

// Page 0xF0, opcode 0x08: Publishes a host signal with explicit group causal
// state.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // required exact hal.device_group ref.
  uint8_t group_r8;
  // required exact hal.semaphore ref.
  uint8_t semaphore_r8;
  // Architectural semaphore signal payload.
  uint8_t payload_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_hal_semaphore_signal_record_t;

// Page 0xF0, opcode 0x09: Polls or asynchronously awaits an ALL/ANY timepoint
// set.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // UINT64_MAX for ALL or selected zero-based member index for ANY.
  uint8_t dst_v8;
  // ALL=0 or ANY=1 completion selector.
  uint8_t mode_u8;
  // RELATIVE=0 or ABSOLUTE=1 signed-nanosecond interpretation.
  uint8_t timeout_kind_u8;
  // Exact two's-complement signed-i64 timeout bits.
  uint8_t timeout_v8;
  // Base ref slot of exact non-null semaphore elements.
  uint16_t semaphore_base_u16;
  // Eight-byte-aligned local base of u64 payload elements.
  uint16_t payload_base_u16;
  // Shared semaphore and payload element count.
  uint16_t count_u16;
} iree_vm_isa_hal_semaphore_await_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_HAL_SEMAPHORE_H_
// clang-format on
