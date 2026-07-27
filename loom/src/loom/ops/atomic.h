// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared atomic and synchronization vocabulary used by scalar view atomics,
// vector atomics, and kernel barriers. Generated dialect APIs alias these enum
// types directly so verification, lowering, and builders use one semantic
// domain.

#ifndef LOOM_OPS_ATOMIC_H_
#define LOOM_OPS_ATOMIC_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Read-modify-write operation kind for view and vector atomics.
typedef enum loom_atomic_kind_e {
  // Integer exchange.
  LOOM_ATOMIC_KIND_XCHGI = 0,
  // Floating-point exchange.
  LOOM_ATOMIC_KIND_XCHGF = 1,
  // Integer addition.
  LOOM_ATOMIC_KIND_ADDI = 2,
  // Floating-point addition.
  LOOM_ATOMIC_KIND_ADDF = 3,
  // Integer subtraction.
  LOOM_ATOMIC_KIND_SUBI = 4,
  // Bitwise AND.
  LOOM_ATOMIC_KIND_ANDI = 5,
  // Bitwise OR.
  LOOM_ATOMIC_KIND_ORI = 6,
  // Bitwise XOR.
  LOOM_ATOMIC_KIND_XORI = 7,
  // Signed integer minimum.
  LOOM_ATOMIC_KIND_MINSI = 8,
  // Signed integer maximum.
  LOOM_ATOMIC_KIND_MAXSI = 9,
  // Unsigned integer minimum.
  LOOM_ATOMIC_KIND_MINUI = 10,
  // Unsigned integer maximum.
  LOOM_ATOMIC_KIND_MAXUI = 11,
  // IEEE 754 floating-point minimum.
  LOOM_ATOMIC_KIND_MINIMUMF = 12,
  // IEEE 754 floating-point maximum.
  LOOM_ATOMIC_KIND_MAXIMUMF = 13,
  // C99 fmin-style floating-point minimum.
  LOOM_ATOMIC_KIND_MINNUMF = 14,
  // C99 fmax-style floating-point maximum.
  LOOM_ATOMIC_KIND_MAXNUMF = 15,
  LOOM_ATOMIC_KIND_COUNT_,
} loom_atomic_kind_t;

// Memory ordering for atomic operations and explicit synchronization fences.
typedef enum loom_atomic_ordering_e {
  // No ordering constraints beyond atomicity.
  LOOM_ATOMIC_ORDERING_RELAXED = 0,
  // Acquire ordering on loads or read-modify-write results.
  LOOM_ATOMIC_ORDERING_ACQUIRE = 1,
  // Release ordering on stores or read-modify-write updates.
  LOOM_ATOMIC_ORDERING_RELEASE = 2,
  // Acquire-release ordering on read-modify-write success.
  LOOM_ATOMIC_ORDERING_ACQ_REL = 3,
  // Sequentially consistent ordering.
  LOOM_ATOMIC_ORDERING_SEQ_CST = 4,
  LOOM_ATOMIC_ORDERING_COUNT_,
} loom_atomic_ordering_t;

// Synchronization scope for atomic operations and explicit synchronization
// fences.
typedef enum loom_atomic_scope_e {
  // Current invocation or thread.
  LOOM_ATOMIC_SCOPE_THREAD = 0,
  // Current SIMD subgroup or wave.
  LOOM_ATOMIC_SCOPE_SUBGROUP = 1,
  // Current workgroup or block.
  LOOM_ATOMIC_SCOPE_WORKGROUP = 2,
  // Current device.
  LOOM_ATOMIC_SCOPE_DEVICE = 3,
  // Whole system.
  LOOM_ATOMIC_SCOPE_SYSTEM = 4,
  LOOM_ATOMIC_SCOPE_COUNT_,
} loom_atomic_scope_t;

// Compare-exchange ordering validation error.
typedef enum loom_atomic_cmpxchg_ordering_error_e {
  // The success/failure ordering pair is valid.
  LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE = 0,
  // Failure ordering cannot include release semantics.
  LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_RELEASE,
  // Failure ordering must not be stronger than success ordering.
  LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_STRONGER,
} loom_atomic_cmpxchg_ordering_error_t;

// Returns true when |kind| is a known loom_atomic_kind_t value.
bool loom_atomic_kind_is_valid(uint8_t kind);

// Returns true when |ordering| is a known loom_atomic_ordering_t value.
bool loom_atomic_ordering_is_valid(uint8_t ordering);

// Returns true when |scope| is a known loom_atomic_scope_t value.
bool loom_atomic_scope_is_valid(uint8_t scope);

// Validates the success/failure ordering pair for compare-exchange atomics.
loom_atomic_cmpxchg_ordering_error_t loom_atomic_cmpxchg_ordering_validate(
    uint8_t success_ordering, uint8_t failure_ordering);

// Returns the attribute name responsible for |error|.
iree_string_view_t loom_atomic_cmpxchg_ordering_error_attr_name(
    loom_atomic_cmpxchg_ordering_error_t error);

// Returns a diagnostic constraint phrase for |error|.
iree_string_view_t loom_atomic_cmpxchg_ordering_error_expected_constraint(
    loom_atomic_cmpxchg_ordering_error_t error);

// Returns true when |kind| operates on integer element types.
bool loom_atomic_kind_accepts_integer(uint8_t kind);

// Returns true when |kind| operates on floating-point element types.
bool loom_atomic_kind_accepts_float(uint8_t kind);

// Returns true when |kind| is an atomic exchange operation.
bool loom_atomic_kind_is_exchange(uint8_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ATOMIC_H_
