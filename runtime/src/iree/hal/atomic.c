// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/atomic.h"

#include <inttypes.h>

#include "iree/base/internal/atomics.h"

IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_atomic_operation_capabilities_for_host(
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

static iree_status_t iree_hal_atomic_validate_width_and_value(
    iree_hal_atomic_width_t width, uint64_t value, const char* value_name) {
  switch (width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      if (IREE_UNLIKELY(value > UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "32-bit atomic %s has nonzero upper bits: 0x%016" PRIx64,
            value_name, value);
      }
      return iree_ok_status();
    case IREE_HAL_ATOMIC_WIDTH_64:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported atomic width: %u", width);
  }
}

static iree_status_t iree_hal_atomic_validate_flags(
    iree_hal_atomic_flags_t flags, iree_hal_atomic_flags_t allowed_flags) {
  const iree_hal_atomic_flags_t unsupported_flags = flags & ~allowed_flags;
  if (IREE_UNLIKELY(unsupported_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported atomic flags: 0x%08" PRIx32,
                            unsupported_flags);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_atomic_wait_params_validate(iree_hal_atomic_wait_params_t params) {
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_width_and_value(
      params.width, params.value, "wait value"));
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_width_and_value(
      params.width, params.mask, "wait mask"));
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_flags(
      params.flags, IREE_HAL_ATOMIC_FLAGS_KNOWN));
  switch (params.condition) {
    case IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL:
    case IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL:
    case IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported atomic wait condition: %u",
                              params.condition);
  }
  if (IREE_UNLIKELY(params.reserved != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "atomic wait reserved fields must be zero");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_atomic_store_params_validate(iree_hal_atomic_store_params_t params) {
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_width_and_value(
      params.width, params.value, "store value"));
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_flags(
      params.flags, IREE_HAL_ATOMIC_FLAGS_KNOWN));
  if (IREE_UNLIKELY(params.reserved[0] != 0 || params.reserved[1] != 0 ||
                    params.reserved[2] != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "atomic store reserved fields must be zero");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_atomic_rmw_params_validate(iree_hal_atomic_rmw_params_t params) {
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_width_and_value(
      params.width, params.operand, "read-modify-write operand"));
  IREE_RETURN_IF_ERROR(iree_hal_atomic_validate_flags(
      params.flags, IREE_HAL_ATOMIC_FLAGS_KNOWN));
  switch (params.operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported atomic RMW operation: %u",
                              params.operation);
  }
  if (IREE_UNLIKELY(params.reserved != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "atomic RMW reserved fields must be zero");
  }
  return iree_ok_status();
}
