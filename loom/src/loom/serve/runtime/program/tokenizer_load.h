// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_LOAD_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_LOAD_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/runtime/program/tokenizer_cache.h"
#include "loomc/task.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent cold-path source read and tokenizer construction. The load owner
// joins this operation before linking an input module or retiring model inputs.
typedef struct loom_serve_tokenizer_load_t loom_serve_tokenizer_load_t;

// Resolves a code-facing tokenizer identity from sources->card, or a legacy
// path relative to directory when sources have no card, and schedules
// construction on the caller's shared task sink. Named inputs validate their
// format before submission; explicit formats permit arbitrary file names and
// inference accepts .json only. An empty directory means the path is already a
// locator in the source namespace (including embedded packages).
// The resolved locator is copied; source views, package, and wake payload
// remain borrowed until take.
// Successful submission owes one nonblocking wake, possibly before return.
// Failure leaves out_load NULL and owes no wake. The owner serializes
// submission and take; wake only schedules the owner and must not reenter this
// operation.
iree_status_t loom_serve_tokenizer_load_start(
    loom_serve_tokenizer_cache_t* cache, loomc_task_sink_t sink,
    const loom_serve_sources_t* sources, iree_string_view_t directory,
    iree_string_view_t name, void (*wake)(void* user_data), void* user_data,
    iree_allocator_t host_allocator, loom_serve_tokenizer_load_t** out_load);

// Returns false without modifying arguments while construction is pending.
// Otherwise joins the result into status, transfers tokenizer ownership, and
// frees and clears the operation. A NULL operation is already joined and leaves
// status/tokenizer unchanged. Even cancellation and other preparation failures
// must take accepted work before releasing borrowed inputs or the wake payload.
bool loom_serve_tokenizer_load_take(loom_serve_tokenizer_load_t** load,
                                    iree_status_t* status,
                                    loom_serve_tokenizer_t** out_tokenizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_TOKENIZER_LOAD_H_
