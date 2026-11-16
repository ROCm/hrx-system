// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPLACEMENT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPLACEMENT_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/program_clone.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares lifetime groups for an oversized source program and its resident
// loading helper. Group identities are fixed before native compilation learns
// exact sizes, allowing physical descriptor admission without guessing code.
iree_status_t loom_aie2p_program_replacement_prepare(
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_object_t* native, iree_arena_allocator_t* arena,
    loom_aie2p_program_layout_t* out_layout);

// Compiles a placement-specific program with ordinary loading calls. Only
// affected controllers and their callers are recompiled; unchanged helpers
// borrow the source compiler's detached native results and exact call
// contracts. Physical admission supplies one descriptor per non-root group and
// the queue header/command. The prepared layout receives exact native code and
// placement. The source pipeline scope owns lifetime/execution semantics
// throughout.
iree_status_t loom_aie2p_program_replacement_compile(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, uint32_t queue_header,
    uint32_t queue_command, const uint32_t* descriptors,
    loom_aie2p_program_layout_t* layout, bool* out_compiled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPLACEMENT_H_
