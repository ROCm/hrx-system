// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ATOMIC_H_
#define IREE_HAL_ATOMIC_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Width in bits of an atomic memory operation.
typedef uint8_t iree_hal_atomic_width_t;
typedef enum iree_hal_atomic_width_e {
  IREE_HAL_ATOMIC_WIDTH_32 = 32,
  IREE_HAL_ATOMIC_WIDTH_64 = 64,
} iree_hal_atomic_width_e;

// Comparison applied by an atomic wait operation.
typedef uint8_t iree_hal_atomic_wait_condition_t;
typedef enum iree_hal_atomic_wait_condition_e {
  // Waits until `(observed & mask) == value`.
  IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL = 0,
  // Waits until `(observed & mask) != value`.
  IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL = 1,
  // Waits until `(observed & mask) >= value` using an unsigned comparison.
  IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL = 2,
} iree_hal_atomic_wait_condition_e;

// Operation applied by an atomic read-modify-write command.
typedef uint8_t iree_hal_atomic_rmw_operation_t;
typedef enum iree_hal_atomic_rmw_operation_e {
  IREE_HAL_ATOMIC_RMW_OPERATION_ADD = 0,
  IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT = 1,
  IREE_HAL_ATOMIC_RMW_OPERATION_AND = 2,
  IREE_HAL_ATOMIC_RMW_OPERATION_OR = 3,
  IREE_HAL_ATOMIC_RMW_OPERATION_XOR = 4,
} iree_hal_atomic_rmw_operation_e;

// Ordering and visibility semantics requested for an atomic operation.
//
// These flags specify minimum semantics. Implementations may provide stronger
// ordering or visibility when that is inherent in their memory model. All
// known flags are accepted for every operation so callers may conservatively
// reuse a common ordering mask. Operation-inapplicable flags have no effect.
typedef uint32_t iree_hal_atomic_flags_t;
typedef enum iree_hal_atomic_flag_bits_e {
  IREE_HAL_ATOMIC_FLAG_NONE = 0u,
  // Makes writes published before the observed atomic modification visible to
  // commands in the target execution stages.
  IREE_HAL_ATOMIC_FLAG_ACQUIRE = 1u << 0,
  // Publishes writes from the source execution stages before making the atomic
  // modification visible.
  IREE_HAL_ATOMIC_FLAG_RELEASE = 1u << 1,
  // Extends the atomic access and requested ordering beyond the default
  // device coherence domain to the system coherence domain.
  IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE = 1u << 2,
} iree_hal_atomic_flag_bits_e;

#define IREE_HAL_ATOMIC_FLAGS_KNOWN                         \
  ((iree_hal_atomic_flags_t)(IREE_HAL_ATOMIC_FLAG_ACQUIRE | \
                             IREE_HAL_ATOMIC_FLAG_RELEASE | \
                             IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE))

// Parameters for an atomic wait operation.
typedef struct iree_hal_atomic_wait_params_t {
  // Unsigned value compared against the masked value loaded from memory.
  uint64_t value;
  // Mask applied to each value loaded from memory before comparison.
  uint64_t mask;
  // Ordering and visibility flags. RELEASE has no effect on waits.
  iree_hal_atomic_flags_t flags;
  // Width of the memory location and comparison values.
  iree_hal_atomic_width_t width;
  // Comparison that determines when the wait is satisfied.
  iree_hal_atomic_wait_condition_t condition;
  // Reserved for future use and must be zero.
  uint16_t reserved;
} iree_hal_atomic_wait_params_t;

// Parameters for an atomic store operation.
typedef struct iree_hal_atomic_store_params_t {
  // Unsigned value stored to memory.
  uint64_t value;
  // Ordering and visibility flags. ACQUIRE has no effect on stores.
  iree_hal_atomic_flags_t flags;
  // Width of the memory location and stored value.
  iree_hal_atomic_width_t width;
  // Reserved for future use and must be zero.
  uint8_t reserved[3];
} iree_hal_atomic_store_params_t;

// Parameters for a no-result atomic read-modify-write operation.
typedef struct iree_hal_atomic_rmw_params_t {
  // Unsigned right-hand operand applied to the memory value.
  uint64_t operand;
  // Ordering and visibility flags.
  iree_hal_atomic_flags_t flags;
  // Width of the memory location and operand.
  iree_hal_atomic_width_t width;
  // Read-modify-write operation to perform.
  iree_hal_atomic_rmw_operation_t operation;
  // Reserved for future use and must be zero.
  uint16_t reserved;
} iree_hal_atomic_rmw_params_t;

// Atomic operation capabilities for one width and coherence-domain cell.
typedef uint32_t iree_hal_atomic_operation_flags_t;
typedef enum iree_hal_atomic_operation_flag_bits_e {
  IREE_HAL_ATOMIC_OPERATION_FLAG_NONE = 0u,
  IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT = 1u << 0,
  IREE_HAL_ATOMIC_OPERATION_FLAG_STORE = 1u << 1,
  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD = 1u << 2,
  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT = 1u << 3,
  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND = 1u << 4,
  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR = 1u << 5,
  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR = 1u << 6,
} iree_hal_atomic_operation_flag_bits_e;

#define IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL                                          \
  ((iree_hal_atomic_operation_flags_t)(IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |         \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |        \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD |      \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT | \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND |      \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR |       \
                                       IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR))

// Atomic wait conditions supported by one width and coherence-domain cell.
typedef uint32_t iree_hal_atomic_wait_condition_flags_t;
typedef enum iree_hal_atomic_wait_condition_flag_bits_e {
  IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE = 0u,
  IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL = 1u << 0,
  IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NOT_EQUAL = 1u << 1,
  IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_UNSIGNED_GREATER_EQUAL = 1u << 2,
} iree_hal_atomic_wait_condition_flag_bits_e;

#define IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL                                            \
  ((iree_hal_atomic_wait_condition_flags_t)(IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL |     \
                                            IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NOT_EQUAL | \
                                            IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_UNSIGNED_GREATER_EQUAL))

// Atomic operation capabilities partitioned by width and coherence domain.
typedef struct iree_hal_atomic_operation_capabilities_t {
  // Operations supported for 32-bit values in the device coherence domain.
  iree_hal_atomic_operation_flags_t device_scope_32;
  // Operations supported for 64-bit values in the device coherence domain.
  iree_hal_atomic_operation_flags_t device_scope_64;
  // Operations supported for 32-bit values in the system coherence domain.
  iree_hal_atomic_operation_flags_t system_scope_32;
  // Operations supported for 64-bit values in the system coherence domain.
  iree_hal_atomic_operation_flags_t system_scope_64;
} iree_hal_atomic_operation_capabilities_t;

// Atomic wait condition capabilities partitioned by width and coherence
// domain.
typedef struct iree_hal_atomic_wait_condition_capabilities_t {
  // Conditions supported for 32-bit waits in the device coherence domain.
  iree_hal_atomic_wait_condition_flags_t device_scope_32;
  // Conditions supported for 64-bit waits in the device coherence domain.
  iree_hal_atomic_wait_condition_flags_t device_scope_64;
  // Conditions supported for 32-bit waits in the system coherence domain.
  iree_hal_atomic_wait_condition_flags_t system_scope_32;
  // Conditions supported for 64-bit waits in the system coherence domain.
  iree_hal_atomic_wait_condition_flags_t system_scope_64;
} iree_hal_atomic_wait_condition_capabilities_t;

// Atomic execution capabilities for a queue family.
typedef struct iree_hal_atomic_capabilities_t {
  // Supported atomic operations.
  iree_hal_atomic_operation_capabilities_t operations;
  // Supported predicates for operations advertising WAIT.
  iree_hal_atomic_wait_condition_capabilities_t wait_conditions;
} iree_hal_atomic_capabilities_t;

// Builds memory operation capabilities for lock-free host atomic widths.
// Naturally aligned host memory supports both device and system scope.
IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_atomic_operation_capabilities_for_host(
    iree_hal_atomic_operation_flags_t allowed_operations);

// Returns the width in bytes or zero when |width| is not valid.
static inline iree_device_size_t iree_hal_atomic_width_byte_count(
    iree_hal_atomic_width_t width) {
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return 4;
    case IREE_HAL_ATOMIC_WIDTH_64:
      return 8;
    default:
      return 0;
  }
}

// Validates target-independent atomic wait parameter semantics.
IREE_API_EXPORT iree_status_t
iree_hal_atomic_wait_params_validate(iree_hal_atomic_wait_params_t params);

// Validates target-independent atomic store parameter semantics.
IREE_API_EXPORT iree_status_t
iree_hal_atomic_store_params_validate(iree_hal_atomic_store_params_t params);

// Validates target-independent atomic RMW parameter semantics.
IREE_API_EXPORT iree_status_t
iree_hal_atomic_rmw_params_validate(iree_hal_atomic_rmw_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ATOMIC_H_
