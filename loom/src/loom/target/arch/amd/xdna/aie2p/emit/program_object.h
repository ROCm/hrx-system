// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-tile native object assembly from an entry and its direct helpers.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Joins an entry contribution followed by its direct helper contributions.
//
// All executable fragments are normalized into the entry's code section. The
// entry remains first, while helper placement minimizes alignment padding with
// stable order for the helpers that remain. Ordinary contribution assembly
// performs the final aligned concatenation. Direct-call imports are resolved
// to exactly one helper entry. Helpers must be code-only leaves; the entry
// continues to own all resources, immutable data, and function storage for the
// tile program. The result retains its exact combined code requirement even
// when it exceeds one core's program memory; program compilation owns target
// capacity admission after composition.
iree_status_t loom_aie2p_program_object_merge(
    const loom_aie2p_leaf_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_
