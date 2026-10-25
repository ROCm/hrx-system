// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Logical kernel launch resolution.

#ifndef LOOM_TRANSFORMS_KERNEL_RESOLVE_LAUNCHES_H_
#define LOOM_TRANSFORMS_KERNEL_RESOLVE_LAUNCHES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Results retained while resolving logical kernel launches.
typedef struct loom_kernel_launch_resolution_t {
  // Scratch-arena-owned entry operation pointers indexed by pre-resolution
  // logical symbol ID. NULL slots were not live. Operation pointers remain
  // stable across symbol-table compaction; the entry's callee accessor returns
  // its current module-local identity after each compaction.
  loom_op_t** entry_ops;

  // Number of pre-resolution module symbol slots.
  iree_host_size_t entry_count;
} loom_kernel_launch_resolution_t;

// Resolves every logical kernel launch recorded in |references|.
//
// |configuration_functions| is a dense prefix projection indexed by the
// module-local logical-kernel symbol IDs present when linking completed. Each
// valid entry names a private pure or deterministic read-only function
// accepting the kernel workload signature and returning exact XYZ workgroup
// counts. Caller-owned target specialization may append symbols while
// preserving every linked symbol ID; those appended symbols intentionally lie
// outside the projection. The transform inserts ordinary calls at their
// caller-owned CFG sites, derives one executable-entry declaration per logical
// kernel, and replaces each kernel.launch with a kernel.dispatch targeting that
// entry.
// Kernel-scoped pipelines retain their logical launches and declaration in
// the entry table: their workload prefix specializes an independently compiled
// artifact and does not compute a physical dispatch grid.
//
// Pure calls cross containing launch-schedule regions only while every workload
// is defined outside the crossed region. Read-only calls remain at the lexical
// launch so they cannot move ahead of command producers. Ordinary CFG parents
// keep both kinds of calls in the launch's original control-flow scope.
//
// The immutable reference table must describe |module| immediately before the
// transform begins. It is stale after any launch is replaced. Callers may
// continue using unaffected occurrence records, but must rebuild it before any
// analysis requiring a complete post-transform reference graph.
//
// User-authored contract failures emit diagnostics, set |out_valid| to false,
// and return OK. The module may be partially transformed in that case and must
// not be used as a successful product. Infrastructure failures return non-OK.
iree_status_t loom_kernel_resolve_launches(
    loom_module_t* module, const loom_symbol_reference_table_t* references,
    const loom_symbol_ref_t* configuration_functions,
    iree_host_size_t configuration_function_count,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_kernel_launch_resolution_t* out_resolution, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_RESOLVE_LAUNCHES_H_
