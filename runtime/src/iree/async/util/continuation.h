// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common emulation for IREE_ASYNC_OPERATION_FLAG_LINKED operation chains.

#ifndef IREE_ASYNC_UTIL_CONTINUATION_H_
#define IREE_ASYNC_UTIL_CONTINUATION_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Submits the head of an intrusive continuation chain.
//
// On success, the backend owns eventual completion of |chain_head| and its
// remaining linked_next chain. On failure, the function returns an owned
// status and leaves the chain intact for the dispatcher to complete.
typedef iree_status_t (*iree_async_continuation_submit_fn_t)(
    void* user_data, iree_async_operation_t* chain_head);

// Pending work produced by iree_async_continuation_begin.
typedef struct iree_async_continuation_t {
  // Chain to complete after the triggering callback, if any.
  iree_async_operation_t* chain_head;
  // Owned submission error transferred to the chain head, if any.
  iree_status_t submit_status;
  // True when the trigger failed and the entire chain must be cancelled.
  bool cancel_chain;
} iree_async_continuation_t;

// Validates LINKED flags and constructs intrusive linked_next chains for a
// submission batch. All linked_next fields are cleared and rebuilt from the
// batch; the caller's operation pointer array is not retained.
//
// Returns INVALID_ARGUMENT if an operation pointer is NULL or the final
// operation has LINKED set. On failure, linked_next fields are not modified.
iree_status_t iree_async_continuation_prepare_batch(
    iree_async_operation_list_t operations);

// Detaches and returns |operation|'s continuation chain.
//
// This must happen before invoking the operation's callback because the
// callback may free or recycle the operation.
static inline iree_async_operation_t* iree_async_continuation_take(
    iree_async_operation_t* operation) {
  iree_async_operation_t* chain_head = operation->linked_next;
  operation->linked_next = NULL;
  return chain_head;
}

// Completes an unsubmitted continuation chain with fresh CANCELLED statuses.
// Returns the number of user callbacks invoked directly.
//
// Thread safety: must be called from the proactor's poll owner thread.
iree_host_size_t iree_async_continuation_cancel(
    iree_async_operation_t* chain_head);

// Begins dispatch of a detached continuation chain.
//
// A successful trigger submits the chain head through |submit_fn| before the
// trigger callback can release successor storage. Successful submission returns
// an empty result. Submission failure returns the intact chain and owned error
// for later completion. A failed trigger returns the chain marked for later
// cancellation without calling |submit_fn|.
//
// A single tail with a suppressed completion (completion_fn == NULL) is
// consumed during this call on trigger or submission failure. It has no user
// callback to order, and consuming it before the trigger callback preserves
// the fire-and-forget tail lifetime contract.
//
// |trigger_status_code| is deliberately scalar: the caller retains ownership
// of the trigger's full status until transferring it to the trigger callback.
//
// Thread safety: must be called from the proactor's poll owner thread.
iree_async_continuation_t iree_async_continuation_begin(
    iree_async_continuation_submit_fn_t submit_fn, void* submit_user_data,
    iree_async_operation_t* chain_head, iree_status_code_t trigger_status_code);

// Finishes a continuation after the triggering operation's callback.
//
// On trigger failure, cancels the entire chain. On continuation submission
// failure, completes the head with the owned submission error and cancels the
// remaining tail. Clears |continuation| before invoking callbacks so recursive
// poll/submit paths cannot complete it twice.
//
// Returns the number of user callbacks invoked directly. Successfully
// submitted operations complete through the backend's normal poll path and are
// not included.
//
// Thread safety: must be called from the proactor's poll owner thread.
iree_host_size_t iree_async_continuation_finish(
    iree_async_continuation_t* continuation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_CONTINUATION_H_
