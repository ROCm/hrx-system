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

// Controls only the status classification of target validation failures.
// Alignment requirements, capability checks, successful execution, and all
// non-target failures are identical in every mode.
typedef uint8_t iree_hal_atomic_target_error_mode_t;
typedef enum iree_hal_atomic_target_error_mode_e {
  // Uses the backend's established target-validation status contract.
  IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT = 0,
  // Reports target misalignment or missing target memory capability as
  // IREE_STATUS_INCOMPATIBLE. This is useful for API adapters whose public
  // contract distinguishes an incompatible target from other preconditions.
  IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE = 1,
} iree_hal_atomic_target_error_mode_e;

// Language-level object alignment. On SysV i386 Clang's __alignof__ reports
// the preferred alignment of uint64_t (8) instead of its ABI alignment (4).
#if defined(__cplusplus)
#define IREE_HAL_ATOMIC_NATURAL_ALIGNOF(x) alignof(x)
#elif defined(IREE_COMPILER_MSVC)
#define IREE_HAL_ATOMIC_NATURAL_ALIGNOF(x) __alignof(x)
#else
#define IREE_HAL_ATOMIC_NATURAL_ALIGNOF(x) _Alignof(x)
#endif

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
  // Status classification used only for target validation failures.
  iree_hal_atomic_target_error_mode_t target_error_mode;
  // Reserved for future use and must be zero.
  uint8_t reserved;
} iree_hal_atomic_wait_params_t;
static_assert(sizeof(iree_hal_atomic_wait_params_t) == 24,
              "atomic wait parameter ABI size must remain stable");
static_assert(IREE_HAL_ATOMIC_NATURAL_ALIGNOF(iree_hal_atomic_wait_params_t) ==
                  IREE_HAL_ATOMIC_NATURAL_ALIGNOF(uint64_t),
              "atomic wait parameter ABI alignment must remain stable");
static_assert(offsetof(iree_hal_atomic_wait_params_t, target_error_mode) == 22,
              "atomic wait target error mode must consume reserved storage");

// Parameters for an atomic store operation.
typedef struct iree_hal_atomic_store_params_t {
  // Unsigned value stored to memory.
  uint64_t value;
  // Ordering and visibility flags. ACQUIRE has no effect on stores.
  iree_hal_atomic_flags_t flags;
  // Width of the memory location and stored value.
  iree_hal_atomic_width_t width;
  // Status classification used only for target validation failures.
  iree_hal_atomic_target_error_mode_t target_error_mode;
  // Reserved for future use and must be zero.
  uint8_t reserved[2];
} iree_hal_atomic_store_params_t;
static_assert(sizeof(iree_hal_atomic_store_params_t) == 16,
              "atomic store parameter ABI size must remain stable");
static_assert(IREE_HAL_ATOMIC_NATURAL_ALIGNOF(iree_hal_atomic_store_params_t) ==
                  IREE_HAL_ATOMIC_NATURAL_ALIGNOF(uint64_t),
              "atomic store parameter ABI alignment must remain stable");
static_assert(offsetof(iree_hal_atomic_store_params_t, target_error_mode) == 13,
              "atomic store target error mode must consume reserved storage");

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
  // Status classification used only for target validation failures.
  iree_hal_atomic_target_error_mode_t target_error_mode;
  // Reserved for future use and must be zero.
  uint8_t reserved;
} iree_hal_atomic_rmw_params_t;
static_assert(sizeof(iree_hal_atomic_rmw_params_t) == 16,
              "atomic RMW parameter ABI size must remain stable");
static_assert(IREE_HAL_ATOMIC_NATURAL_ALIGNOF(iree_hal_atomic_rmw_params_t) ==
                  IREE_HAL_ATOMIC_NATURAL_ALIGNOF(uint64_t),
              "atomic RMW parameter ABI alignment must remain stable");
static_assert(offsetof(iree_hal_atomic_rmw_params_t, target_error_mode) == 14,
              "atomic RMW target error mode must consume reserved storage");

#undef IREE_HAL_ATOMIC_NATURAL_ALIGNOF

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

// Returns true when no operations are advertised or required.
static inline bool iree_hal_atomic_operation_capabilities_is_empty(
    iree_hal_atomic_operation_capabilities_t value) {
  return !(value.device_scope_32 | value.device_scope_64 |
           value.system_scope_32 | value.system_scope_64);
}

// Returns operations supported by both the backing and the executing site.
static inline iree_hal_atomic_operation_capabilities_t
iree_hal_atomic_operation_capabilities_intersect(
    iree_hal_atomic_operation_capabilities_t lhs,
    iree_hal_atomic_operation_capabilities_t rhs) {
  const iree_hal_atomic_operation_capabilities_t result = {
      lhs.device_scope_32 & rhs.device_scope_32,
      lhs.device_scope_64 & rhs.device_scope_64,
      lhs.system_scope_32 & rhs.system_scope_32,
      lhs.system_scope_64 & rhs.system_scope_64,
  };
  return result;
}

// Combines requirements for operations used at either site.
static inline iree_hal_atomic_operation_capabilities_t
iree_hal_atomic_operation_capabilities_union(
    iree_hal_atomic_operation_capabilities_t lhs,
    iree_hal_atomic_operation_capabilities_t rhs) {
  const iree_hal_atomic_operation_capabilities_t result = {
      lhs.device_scope_32 | rhs.device_scope_32,
      lhs.device_scope_64 | rhs.device_scope_64,
      lhs.system_scope_32 | rhs.system_scope_32,
      lhs.system_scope_64 | rhs.system_scope_64,
  };
  return result;
}

// Returns true when every operation in |required| is present in the matching
// width and coherence-domain cell of |available|.
static inline bool iree_hal_atomic_operation_capabilities_support(
    iree_hal_atomic_operation_capabilities_t available,
    iree_hal_atomic_operation_capabilities_t required) {
  return iree_all_bits_set(available.device_scope_32,
                           required.device_scope_32) &&
         iree_all_bits_set(available.device_scope_64,
                           required.device_scope_64) &&
         iree_all_bits_set(available.system_scope_32,
                           required.system_scope_32) &&
         iree_all_bits_set(available.system_scope_64, required.system_scope_64);
}

// Returns true when every capability cell contains only known operations.
static inline bool iree_hal_atomic_operation_capabilities_are_valid(
    iree_hal_atomic_operation_capabilities_t capabilities) {
  return !iree_any_bit_set(capabilities.device_scope_32,
                           ~IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL) &&
         !iree_any_bit_set(capabilities.device_scope_64,
                           ~IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL) &&
         !iree_any_bit_set(capabilities.system_scope_32,
                           ~IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL) &&
         !iree_any_bit_set(capabilities.system_scope_64,
                           ~IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
}

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
