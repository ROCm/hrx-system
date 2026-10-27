// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Root-local rewrites proved against one coherent fact snapshot.

#ifndef LOOM_REWRITE_ACTION_H_
#define LOOM_REWRITE_ACTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_rewrite_action_t loom_rewrite_action_t;

// One prepared, meaning-preserving rewrite. Providers embed this header before
// their payload. Apply may replace only its own regionless root and must
// preserve result counts and types. It reads surviving operands from the live
// root, never from captured value IDs, and does not query facts. Other planned
// roots remain live until their own actions execute. Failure is limited to
// allocation or another infrastructure failure; matching has already made every
// decision.
struct loom_rewrite_action_t {
  // Next action in the temporary batch; assigned by the shared collector.
  loom_rewrite_action_t* next;
  // Live root to rewrite; assigned by the shared collector.
  loom_op_t* root;
  // Provider callback consuming this action and its trailing payload.
  iree_status_t (*apply)(const loom_rewrite_action_t* action,
                         loom_rewriter_t* rewriter);
};

// Matches against coherent read-only IR and facts. A match allocates its action
// and payload in |arena| and initializes the apply callback. A nonmatch returns
// NULL without allocating. No IR mutation occurs until all roots are prepared.
typedef struct loom_rewrite_prepare_callback_t {
  // Caller-owned policy borrowed during preparation only.
  void* user_data;
  // Optional action for this root, with every semantic decision already made.
  iree_status_t (*fn)(void* user_data, const loom_rewriter_t* rewriter,
                      const loom_op_t* op, iree_arena_allocator_t* arena,
                      loom_rewrite_action_t** out_action);
} loom_rewrite_prepare_callback_t;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_ACTION_H_
