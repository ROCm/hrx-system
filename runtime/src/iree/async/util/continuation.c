// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/continuation.h"

#include "iree/async/util/operation_completion.h"

iree_status_t iree_async_continuation_prepare_batch(
    iree_async_operation_list_t operations) {
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (!operations.values[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "operation %" PRIhsz " in batch is NULL", i);
    }
  }
  if (operations.count > 0 &&
      iree_any_bit_set(operations.values[operations.count - 1]->flags,
                       IREE_ASYNC_OPERATION_FLAG_LINKED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LINKED flag set on last operation in batch (no successor)");
  }

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    operation->linked_next =
        iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED)
            ? operations.values[i + 1]
            : NULL;
  }
  return iree_ok_status();
}

iree_host_size_t iree_async_continuation_cancel(
    iree_async_operation_t* chain_head) {
  iree_host_size_t callback_count = 0;
  iree_async_operation_t* operation = chain_head;
  while (operation) {
    iree_async_operation_t* next = operation->linked_next;
    operation->linked_next = NULL;
    callback_count += iree_async_operation_complete(
        operation, iree_status_from_code(IREE_STATUS_CANCELLED),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
    operation = next;
  }
  return callback_count;
}

iree_async_continuation_t iree_async_continuation_begin(
    iree_async_continuation_submit_fn_t submit_fn, void* submit_user_data,
    iree_async_operation_t* chain_head,
    iree_status_code_t trigger_status_code) {
  iree_async_continuation_t continuation = {
      .chain_head = NULL,
      .submit_status = iree_ok_status(),
      .cancel_chain = false,
  };
  if (!chain_head) return continuation;
  if (trigger_status_code != IREE_STATUS_OK) {
    // A deliberately suppressed tail has no successor callback to order after
    // the trigger. Consume it now so the trigger callback may release its
    // storage under the MESSAGE(SKIP_SOURCE_COMPLETION) lifetime contract.
    if (!chain_head->completion_fn && !chain_head->linked_next) {
      iree_async_operation_complete(
          chain_head, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
      return continuation;
    }
    continuation.chain_head = chain_head;
    continuation.cancel_chain = true;
    return continuation;
  }

  iree_status_t submit_status = submit_fn(submit_user_data, chain_head);
  if (iree_status_is_ok(submit_status)) return continuation;

  // As above, a suppressed chain tail has no observable callback to defer.
  // Free its handled submission error while its storage is still valid.
  if (!chain_head->completion_fn && !chain_head->linked_next) {
    iree_async_operation_complete(chain_head, submit_status,
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
    return continuation;
  }

  continuation.chain_head = chain_head;
  continuation.submit_status = submit_status;
  return continuation;
}

iree_host_size_t iree_async_continuation_finish(
    iree_async_continuation_t* continuation) {
  iree_async_operation_t* chain_head = continuation->chain_head;
  iree_status_t submit_status = continuation->submit_status;
  bool cancel_chain = continuation->cancel_chain;
  continuation->chain_head = NULL;
  continuation->submit_status = iree_ok_status();
  continuation->cancel_chain = false;

  if (!chain_head) return 0;
  if (cancel_chain) {
    return iree_async_continuation_cancel(chain_head);
  }

  iree_async_operation_t* remaining_chain = chain_head->linked_next;
  chain_head->linked_next = NULL;
  iree_host_size_t callback_count = iree_async_operation_complete(
      chain_head, submit_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
  callback_count += iree_async_continuation_cancel(remaining_chain);
  return callback_count;
}
