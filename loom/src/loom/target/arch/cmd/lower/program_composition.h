// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-program composition flattening.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_COMPOSITION_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_COMPOSITION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/error/emitter.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Flattens command-program calls reachable from |root_programs|.
//
// Calls are clone-inlined in callee-before-caller order so each selected root
// becomes a closed command body. Reachable non-root command definitions are
// consumed after inlining; independently selected roots remain intact. Other
// call-like kinds are not changed.
//
// |references| must describe |module| immediately before any logical launches
// were resolved. Launch resolution does not alter command-program call sites,
// so the composition edges remain valid and avoid a redundant module walk.
// The table is stale for every other post-resolution analysis.
//
// Recursive composition emits a diagnostic against the selected definition,
// sets |out_valid| to false, leaves |module| unchanged, and returns OK.
iree_status_t loom_cmd_program_composition_flatten(
    loom_module_t* module, const loom_symbol_reference_table_t* references,
    const loom_func_like_t* root_programs, iree_host_size_t root_program_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_COMPOSITION_H_
