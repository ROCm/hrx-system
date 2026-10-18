// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct-helper closure compilation for one AIE2P tile program.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiles |entry_function_op| and its direct, nonrecursive, module-internal
// helper callees into one tile contribution. Helpers use scalar and pointer
// register ABI values, remain spill-free and code-only, and cannot call other
// functions. Call-free entries retain ordinary leaf compilation behavior.
iree_status_t loom_aie2p_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_aie2p_leaf_compile_options_t* entry_options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_
