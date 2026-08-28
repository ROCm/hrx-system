// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_OPERATION_COMPLETION_H_
#define IREE_ASYNC_UTIL_OPERATION_COMPLETION_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Transfers |status| to an operation's completion callback and returns the
// operation to its optional pool after a final completion.
//
// The caller must release retained operation resources and detach any data it
// needs after the callback before calling this function. The callback may free
// or recycle the operation. If completion is deliberately suppressed and
// completion_fn is NULL, the status is freed as handled.
//
// Returns 1 when a user callback was invoked and 0 when completion was
// suppressed.
iree_host_size_t iree_async_operation_complete(
    iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_OPERATION_COMPLETION_H_
