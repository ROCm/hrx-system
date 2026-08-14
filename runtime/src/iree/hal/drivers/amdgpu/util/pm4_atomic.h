// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PM4 packet emission for HAL atomic memory operations.
//
// These helpers only encode the operation packet. Callers own any release or
// acquire barriers required by the HAL flags and must gate use on an exact
// target whose packet forms have been validated.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_ATOMIC_H_

#include "iree/base/api.h"
#include "iree/hal/atomic.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT = 7,
  IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT = 9,
  IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT = 9,
  IREE_HAL_AMDGPU_PM4_ATOMIC_TARGET_DWORD_OFFSET = 2,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_ATOMIC_MEM = 0x1E,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM = 0x3C,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM64 = 0x93,
  IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_EQUAL = 3,
  IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_NOT_EQUAL = 4,
  IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_UNSIGNED_GREATER_EQUAL = 5,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SWAP_32 = 0x07,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_ADD_32 = 0x0F,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SUBTRACT_32 = 0x10,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_AND_32 = 0x15,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_OR_32 = 0x16,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_XOR_32 = 0x17,
  IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_64_OFFSET = 0x20,
};

// Returns the PM4 WAIT_REG_MEM comparison function for |condition|, or zero
// when the condition is invalid.
static inline uint32_t iree_hal_amdgpu_pm4_atomic_wait_function(
    iree_hal_atomic_wait_condition_t condition) {
  switch (condition) {
    case IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL:
      return IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_EQUAL;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL:
      return IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_NOT_EQUAL;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL:
      return IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_UNSIGNED_GREATER_EQUAL;
    default:
      return 0;
  }
}

// Returns the PM4 ATOMIC_MEM operation for |operation| and |width|, or zero
// when either argument is invalid.
static inline uint32_t iree_hal_amdgpu_pm4_atomic_rmw_operation(
    iree_hal_atomic_width_t width, iree_hal_atomic_rmw_operation_t operation) {
  uint32_t encoded_operation = 0;
  switch (operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
      encoded_operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_ADD_32;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
      encoded_operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SUBTRACT_32;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
      encoded_operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_AND_32;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
      encoded_operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_OR_32;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      encoded_operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_XOR_32;
      break;
    default:
      return 0;
  }
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return encoded_operation;
    case IREE_HAL_ATOMIC_WIDTH_64:
      return encoded_operation + IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_64_OFFSET;
    default:
      return 0;
  }
}

// Returns the exact PM4 dword count for an atomic wait of |width|, or zero
// when the width is invalid.
static inline uint32_t iree_hal_amdgpu_pm4_atomic_wait_dword_count(
    iree_hal_atomic_width_t width) {
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT;
    case IREE_HAL_ATOMIC_WIDTH_64:
      return IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT;
    default:
      return 0;
  }
}

// Emits a WAIT_REG_MEM packet implementing a HAL atomic wait.
static inline bool iree_hal_amdgpu_pm4_atomic_wait_emit(
    iree_hal_atomic_width_t width, iree_hal_atomic_wait_condition_t condition,
    uint64_t target_address, uint64_t value, uint64_t mask, uint32_t capacity,
    uint32_t* target_dwords, uint32_t* out_dword_count) {
  *out_dword_count = 0;
  const uint32_t dword_count =
      iree_hal_amdgpu_pm4_atomic_wait_dword_count(width);
  const uint32_t function = iree_hal_amdgpu_pm4_atomic_wait_function(condition);
  const uint32_t alignment = (uint32_t)iree_hal_atomic_width_byte_count(width);
  if (dword_count == 0 || function == 0 ||
      (target_address & (alignment - 1)) != 0 || capacity < dword_count) {
    return false;
  }

  memset(target_dwords, 0, dword_count * sizeof(*target_dwords));
  const bool is_64_bit = width == IREE_HAL_ATOMIC_WIDTH_64;
  target_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      is_64_bit ? IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM64
                : IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM,
      dword_count);
  target_dwords[1] = iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
      function, IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM);
  target_dwords[2] = is_64_bit ? (uint32_t)(target_address & 0xFFFFFFF8u)
                               : (uint32_t)(target_address & 0xFFFFFFFCu);
  target_dwords[3] = (uint32_t)(target_address >> 32);
  target_dwords[4] = (uint32_t)value;
  if (is_64_bit) {
    target_dwords[5] = (uint32_t)(value >> 32);
    target_dwords[6] = (uint32_t)mask;
    target_dwords[7] = (uint32_t)(mask >> 32);
    target_dwords[8] =
        4 | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE;
  } else {
    target_dwords[5] = (uint32_t)mask;
    target_dwords[6] =
        4 | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE;
  }
  *out_dword_count = dword_count;
  return true;
}

// Emits a single-pass ATOMIC_MEM packet with the encoded |operation|.
static inline bool iree_hal_amdgpu_pm4_atomic_mem_emit(
    iree_hal_atomic_width_t width, uint32_t operation, uint64_t target_address,
    uint64_t operand, uint32_t capacity, uint32_t* target_dwords,
    uint32_t* out_dword_count) {
  *out_dword_count = 0;
  const uint32_t alignment = (uint32_t)iree_hal_atomic_width_byte_count(width);
  if (alignment == 0 || operation == 0 ||
      (target_address & (alignment - 1)) != 0 ||
      capacity < IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT) {
    return false;
  }

  memset(target_dwords, 0,
         IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT * sizeof(*target_dwords));
  target_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_ATOMIC_MEM,
      IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT);
  target_dwords[1] = operation;
  target_dwords[2] = width == IREE_HAL_ATOMIC_WIDTH_64
                         ? (uint32_t)(target_address & 0xFFFFFFF8u)
                         : (uint32_t)(target_address & 0xFFFFFFFCu);
  target_dwords[3] = (uint32_t)(target_address >> 32);
  target_dwords[4] = (uint32_t)operand;
  target_dwords[5] = (uint32_t)(operand >> 32);
  *out_dword_count = IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT;
  return true;
}

// Emits a single-pass ATOMIC_MEM swap implementing a HAL atomic store.
static inline bool iree_hal_amdgpu_pm4_atomic_store_emit(
    iree_hal_atomic_width_t width, uint64_t target_address, uint64_t value,
    uint32_t capacity, uint32_t* target_dwords, uint32_t* out_dword_count) {
  uint32_t operation = IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SWAP_32;
  if (width == IREE_HAL_ATOMIC_WIDTH_64) {
    operation += IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_64_OFFSET;
  } else if (width != IREE_HAL_ATOMIC_WIDTH_32) {
    *out_dword_count = 0;
    return false;
  }
  return iree_hal_amdgpu_pm4_atomic_mem_emit(width, operation, target_address,
                                             value, capacity, target_dwords,
                                             out_dword_count);
}

// Emits a single-pass ATOMIC_MEM packet implementing a no-result HAL RMW.
static inline bool iree_hal_amdgpu_pm4_atomic_rmw_emit(
    iree_hal_atomic_width_t width, iree_hal_atomic_rmw_operation_t operation,
    uint64_t target_address, uint64_t operand, uint32_t capacity,
    uint32_t* target_dwords, uint32_t* out_dword_count) {
  const uint32_t encoded_operation =
      iree_hal_amdgpu_pm4_atomic_rmw_operation(width, operation);
  return iree_hal_amdgpu_pm4_atomic_mem_emit(width, encoded_operation,
                                             target_address, operand, capacity,
                                             target_dwords, out_dword_count);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_ATOMIC_H_
