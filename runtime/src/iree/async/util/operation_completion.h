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

// Backend-only input to completion. Once a checked syscall has run, even
// ECANCELED is its native result. The callback receives the original status.
#define IREE_ASYNC_COMPLETION_FLAG_NATIVE_RESULT (1u << 31)

// Transfers |status| to an operation's completion callback and returns the
// operation to its optional pool after a final completion. Final completion
// clears backend-private flags before the callback so the callback receives a
// caller-owned operation ready for immediate reuse. Multishot completions with
// MORE preserve those flags until the final completion.
//
// On final completion this releases acquired non-region resources before the
// callback, detaches retained span regions before the callback, and releases
// those regions after it returns. The callback may therefore use registered
// memory and immediately free, recycle, or resubmit the operation. If
// completion is deliberately suppressed and completion_fn is NULL, the status
// is freed as handled.
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
