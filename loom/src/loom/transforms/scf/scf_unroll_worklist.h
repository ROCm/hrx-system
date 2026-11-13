// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SCF_SCF_UNROLL_WORKLIST_H_
#define LOOM_TRANSFORMS_SCF_SCF_UNROLL_WORKLIST_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_scf_unroll_work_item_t {
  // Borrowed source loop, in parent-before-child traversal order.
  loom_op_t* op;
  // An enclosing recurrence needs this loop's contents exposed before its cut.
  bool prepare;
} loom_scf_unroll_work_item_t;

typedef struct loom_scf_unroll_worklist_t {
  // Source loops and their original scheduling ancestry.
  loom_scf_unroll_work_item_t* items;
  // Number of populated source loops.
  iree_host_size_t count;
  // Number of annotated descendants eligible for full-linear preparation.
  iree_host_size_t preparation_candidate_count;
  // At least one source loop requests unrolling.
  bool has_policy;
} loom_scf_unroll_worklist_t;

// Whether the source loop has any local unroll intent, including a schedule.
bool loom_scf_unroll_policy_present(const loom_op_t* op);

// Captures loop handles and recurrence ancestry in one source traversal. The
// ordinary rewrite order stays parent-first: inner bounds may need an outer
// iteration's specialization. Preparation candidates still require an original
// fact proof of a full linear unroll before they may precede their parent.
// Storage belongs to |arena| and source handles remain valid until rewritten.
iree_status_t loom_scf_unroll_worklist_build(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, loom_scf_unroll_worklist_t* out_worklist);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_SCF_UNROLL_WORKLIST_H_
