// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural equivalence for independently materialized symbol definitions.

#ifndef LOOM_TRANSFORMS_SYMBOL_SYMBOL_EQUIVALENCE_H_
#define LOOM_TRANSFORMS_SYMBOL_SYMBOL_EQUIVALENCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns whether two concrete private symbols define the same program.
//
// SSA values, block labels, private symbol names, and source locations may
// differ. Private symbol references are compared recursively, while references
// to global-identity symbols must resolve to the same module symbol. The
// comparison is conservative: definitions that cannot be proven equivalent
// return false.
//
// |module| must be verified and both refs must name concrete private symbols in
// that module. Transient comparison state is allocated from |scratch_arena|.
iree_status_t loom_symbol_definitions_equivalent(
    const loom_module_t* module, loom_symbol_ref_t lhs_ref,
    loom_symbol_ref_t rhs_ref, iree_arena_allocator_t* scratch_arena,
    bool* out_equivalent);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_SYMBOL_EQUIVALENCE_H_
