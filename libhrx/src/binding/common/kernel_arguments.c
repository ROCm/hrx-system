// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/kernel_arguments.h"

iree_status_t iree_hal_streaming_validate_prepacked_kernel_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  if (params->buffer_size > 0 && !params->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pre-packed kernel arguments require storage when length is non-zero");
  }

  const iree_host_size_t required_size = symbol->parameters.direct_arg_bytes;
  if (params->buffer_size < required_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pre-packed kernel arguments are shorter than the native argument "
        "extent (%" PRIhsz " < %" PRIhsz ")",
        params->buffer_size, required_size);
  }
  return iree_ok_status();
}
