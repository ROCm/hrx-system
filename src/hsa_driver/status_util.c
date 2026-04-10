// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Direct HSA linkage.

#include "hsa_driver/status_util.h"

#include <stddef.h>

#include "iree/base/status.h"

// Converts HSA status code to the corresponding IREE status code.
static iree_status_code_t
iree_hal_hsa_status_to_iree_code(hsa_status_t status) {
  switch (status) {
  case HSA_STATUS_SUCCESS:
    return IREE_STATUS_OK;
  case HSA_STATUS_ERROR:
    return IREE_STATUS_UNKNOWN;
  case HSA_STATUS_ERROR_INVALID_ARGUMENT:
    return IREE_STATUS_INVALID_ARGUMENT;
  case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
  case HSA_STATUS_ERROR_INVALID_QUEUE:
    return IREE_STATUS_FAILED_PRECONDITION;
  case HSA_STATUS_ERROR_INVALID_ALLOCATION:
  case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  case HSA_STATUS_ERROR_INVALID_AGENT:
  case HSA_STATUS_ERROR_INVALID_REGION:
  case HSA_STATUS_ERROR_INVALID_SIGNAL:
    return IREE_STATUS_INVALID_ARGUMENT;
  case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
  case HSA_STATUS_ERROR_FATAL:
    return IREE_STATUS_INTERNAL;
  case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
  case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
  case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
  case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
    return IREE_STATUS_FAILED_PRECONDITION;
  case HSA_STATUS_ERROR_INVALID_INDEX:
  case HSA_STATUS_ERROR_INVALID_ISA:
  case HSA_STATUS_ERROR_INVALID_ISA_NAME:
    return IREE_STATUS_OUT_OF_RANGE;
  case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
  case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
  case HSA_STATUS_ERROR_INVALID_CACHE:
  case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
  case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
  case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
    return IREE_STATUS_INVALID_ARGUMENT;
  case HSA_STATUS_ERROR_RESOURCE_FREE:
    return IREE_STATUS_FAILED_PRECONDITION;
  case HSA_STATUS_ERROR_NOT_INITIALIZED:
    return IREE_STATUS_FAILED_PRECONDITION;
  case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
    return IREE_STATUS_INVALID_ARGUMENT;
  case HSA_STATUS_ERROR_INVALID_FILE:
    return IREE_STATUS_NOT_FOUND;
  case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
    return IREE_STATUS_FAILED_PRECONDITION;
  case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
  case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
    return IREE_STATUS_ALREADY_EXISTS;
  case HSA_STATUS_ERROR_EXCEPTION:
    return IREE_STATUS_INTERNAL;
  default:
    return IREE_STATUS_UNKNOWN;
  }
}

iree_status_t iree_hal_hsa_result_to_status(hsa_status_t result,
                                            const char *file, uint32_t line) {
  if (IREE_LIKELY(result == HSA_STATUS_SUCCESS)) {
    return iree_ok_status();
  }

  const char *status_string = NULL;
  hsa_status_string(result, &status_string);
  if (!status_string) {
    status_string = "unknown HSA error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hsa_status_to_iree_code(result),
      "HSA driver error '%s' (%d)", status_string, (int)result);
}
