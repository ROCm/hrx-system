// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/status_conversion.h"

hipError_t iree_status_to_hip_result(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }

  const iree_status_code_t code = iree_status_consume_code(status);
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    // Hardware execution failures use data loss when no invalid memory access
    // was identified. This status is non-fatal because other producers also
    // use it for malformed input and protocol data.
    case IREE_STATUS_DATA_LOSS:
      return hipErrorLaunchFailure;
    // Device-side memory access faults abort the submitted operation and make
    // the device context unsafe to continue using.
    case IREE_STATUS_ABORTED:
      return hipErrorIllegalAddress;
    default:
      return hipErrorUnknown;
  }
}
