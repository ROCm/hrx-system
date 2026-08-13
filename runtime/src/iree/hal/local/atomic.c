// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/atomic.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/processor.h"

bool iree_hal_local_atomic_width_is_lock_free(iree_hal_atomic_width_t width) {
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      return iree_atomic_int32_is_lock_free();
    case IREE_HAL_ATOMIC_WIDTH_64:
      return iree_atomic_int64_is_lock_free();
    default:
      return false;
  }
}

iree_hal_atomic_operation_capabilities_t
iree_hal_local_atomic_operation_capabilities(
    iree_hal_atomic_operation_flags_t allowed_operations) {
  iree_hal_atomic_operation_capabilities_t capabilities = {0};
  if (iree_atomic_int32_is_lock_free()) {
    capabilities.device_scope_32 = allowed_operations;
    capabilities.system_scope_32 = allowed_operations;
  }
  if (iree_atomic_int64_is_lock_free()) {
    capabilities.device_scope_64 = allowed_operations;
    capabilities.system_scope_64 = allowed_operations;
  }
  return capabilities;
}

iree_hal_atomic_capabilities_t iree_hal_local_atomic_capabilities(
    iree_hal_atomic_operation_flags_t allowed_operations) {
  iree_hal_atomic_capabilities_t capabilities = {
      .operations =
          iree_hal_local_atomic_operation_capabilities(allowed_operations),
  };
  if (iree_any_bit_set(allowed_operations,
                       IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)) {
    if (iree_any_bit_set(capabilities.operations.device_scope_32,
                         IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)) {
      capabilities.wait_conditions.device_scope_32 =
          IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
      capabilities.wait_conditions.system_scope_32 =
          IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
    }
    if (iree_any_bit_set(capabilities.operations.device_scope_64,
                         IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT)) {
      capabilities.wait_conditions.device_scope_64 =
          IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
      capabilities.wait_conditions.system_scope_64 =
          IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
    }
  }
  return capabilities;
}

static iree_memory_order_t iree_hal_local_atomic_memory_order(
    iree_hal_atomic_flags_t flags) {
  const bool acquire = iree_any_bit_set(flags, IREE_HAL_ATOMIC_FLAG_ACQUIRE);
  const bool release = iree_any_bit_set(flags, IREE_HAL_ATOMIC_FLAG_RELEASE);
  if (acquire && release) return iree_memory_order_acq_rel;
  if (acquire) return iree_memory_order_acquire;
  if (release) return iree_memory_order_release;
  return iree_memory_order_relaxed;
}

static bool iree_hal_local_atomic_wait_condition_matches(
    uint64_t observed, iree_hal_atomic_wait_params_t params) {
  const uint64_t masked_value = observed & params.mask;
  switch (params.condition) {
    case IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL:
      return masked_value == params.value;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL:
      return masked_value != params.value;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL:
      return masked_value >= params.value;
    default:
      IREE_ASSERT_UNREACHABLE("atomic wait parameters must be validated");
      return false;
  }
}

void iree_hal_local_atomic_wait(void* target,
                                iree_hal_atomic_wait_params_t params) {
  IREE_ASSERT(iree_hal_local_atomic_width_is_lock_free(params.width),
              "atomic width must be lock-free");
  const iree_memory_order_t order = iree_hal_local_atomic_memory_order(
      params.flags & IREE_HAL_ATOMIC_FLAG_ACQUIRE);
  switch (params.width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      while (!iree_hal_local_atomic_wait_condition_matches(
          (uint32_t)iree_atomic_load((iree_atomic_uint32_t*)target, order),
          params)) {
        iree_processor_yield();
      }
      return;
    case IREE_HAL_ATOMIC_WIDTH_64:
      while (!iree_hal_local_atomic_wait_condition_matches(
          (uint64_t)iree_atomic_load((iree_atomic_uint64_t*)target, order),
          params)) {
        iree_processor_yield();
      }
      return;
    default:
      IREE_ASSERT_UNREACHABLE("atomic wait parameters must be validated");
      return;
  }
}

void iree_hal_local_atomic_store(void* target,
                                 iree_hal_atomic_store_params_t params) {
  IREE_ASSERT(iree_hal_local_atomic_width_is_lock_free(params.width),
              "atomic width must be lock-free");
  const iree_memory_order_t order = iree_hal_local_atomic_memory_order(
      params.flags & IREE_HAL_ATOMIC_FLAG_RELEASE);
  switch (params.width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      iree_atomic_store((iree_atomic_uint32_t*)target, (uint32_t)params.value,
                        order);
      return;
    case IREE_HAL_ATOMIC_WIDTH_64:
      iree_atomic_store((iree_atomic_uint64_t*)target, params.value, order);
      return;
    default:
      IREE_ASSERT_UNREACHABLE("atomic store parameters must be validated");
      return;
  }
}

static void iree_hal_local_atomic_rmw_32(iree_atomic_uint32_t* target,
                                         iree_hal_atomic_rmw_params_t params,
                                         iree_memory_order_t order) {
  const uint32_t operand = (uint32_t)params.operand;
  switch (params.operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
      (void)iree_atomic_fetch_add(target, operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
      (void)iree_atomic_fetch_sub(target, operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
      (void)iree_atomic_fetch_and(target, operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
      (void)iree_atomic_fetch_or(target, operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      (void)iree_atomic_fetch_xor(target, operand, order);
      return;
    default:
      IREE_ASSERT_UNREACHABLE("atomic RMW parameters must be validated");
      return;
  }
}

static void iree_hal_local_atomic_rmw_64(iree_atomic_uint64_t* target,
                                         iree_hal_atomic_rmw_params_t params,
                                         iree_memory_order_t order) {
  switch (params.operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
      (void)iree_atomic_fetch_add(target, params.operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
      (void)iree_atomic_fetch_sub(target, params.operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
      (void)iree_atomic_fetch_and(target, params.operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
      (void)iree_atomic_fetch_or(target, params.operand, order);
      return;
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      (void)iree_atomic_fetch_xor(target, params.operand, order);
      return;
    default:
      IREE_ASSERT_UNREACHABLE("atomic RMW parameters must be validated");
      return;
  }
}

void iree_hal_local_atomic_rmw(void* target,
                               iree_hal_atomic_rmw_params_t params) {
  IREE_ASSERT(iree_hal_local_atomic_width_is_lock_free(params.width),
              "atomic width must be lock-free");
  const iree_memory_order_t order =
      iree_hal_local_atomic_memory_order(params.flags);
  switch (params.width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      iree_hal_local_atomic_rmw_32((iree_atomic_uint32_t*)target, params,
                                   order);
      return;
    case IREE_HAL_ATOMIC_WIDTH_64:
      iree_hal_local_atomic_rmw_64((iree_atomic_uint64_t*)target, params,
                                   order);
      return;
    default:
      IREE_ASSERT_UNREACHABLE("atomic RMW parameters must be validated");
      return;
  }
}
