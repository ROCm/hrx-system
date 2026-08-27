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

// Resolves every logical kernel launch recorded in |references|.
//
// |configuration_functions| is a dense projection indexed by module-local
// logical-kernel symbol ID. Each valid entry names a private pure function
// accepting the kernel workload signature and returning exact XYZ workgroup
// counts. The transform inserts ordinary pure calls at their caller-owned CFG
// sites, derives one executable-entry declaration per logical kernel, and
// replaces each kernel.launch with a kernel.dispatch targeting that entry.
//
// Calls for launches nested only in kernel launch-schedule regions are placed
// immediately before the outermost schedule. Schedule regions describe command
// ordering and cannot contain ordinary computation; their launch operands are
// necessarily defined outside the schedule. Calls otherwise remain in the
// launch's original control-flow scope.
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
    iree_arena_allocator_t* scratch_arena, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_RESOLVE_LAUNCHES_H_
