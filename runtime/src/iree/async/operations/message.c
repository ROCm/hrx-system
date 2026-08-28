// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/operations/message.h"

iree_status_t iree_async_message_operation_validate(
    const iree_async_message_operation_t* message) {
  if (!message->target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MESSAGE target proactor is NULL");
  }

  bool skip_source_completion = iree_any_bit_set(
      message->message_flags, IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION);
  if (skip_source_completion) {
    if (iree_any_bit_set(message->base.flags,
                         IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "MESSAGE with SKIP_SOURCE_COMPLETION must be a LINKED chain tail");
    }
    if (message->base.completion_fn) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "MESSAGE with SKIP_SOURCE_COMPLETION must not have a completion "
          "callback");
    }
    if (message->base.pool) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "MESSAGE with SKIP_SOURCE_COMPLETION must not have an operation "
          "pool");
    }
  } else if (!message->base.completion_fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MESSAGE requires a completion callback");
  }
  return iree_ok_status();
}
