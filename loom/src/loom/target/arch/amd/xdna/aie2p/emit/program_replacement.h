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

// Retains the execution owner of each function before physical placement.
// Optional per-owner lengths describe one shared native body per equivalence
// class. The worker scratch is zero on entry and exit. Maximums accumulate
// the largest total seen on any single worker for each construction.
iree_status_t loom_aie2p_program_replacement_owners(
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_object_t* native, uint64_t* worker_lengths,
    uint64_t* maximum_lengths, iree_arena_allocator_t* arena,
    const loom_aie2p_program_census_t** out_census);

// Projects the entry/loader/helper closure into physical code groups.
// A NULL code owner map preserves the source construction boundaries.
iree_status_t loom_aie2p_program_replacement_prepare(
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const iree_host_size_t* owners, const iree_host_size_t* code_owners,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* out_layout);

// Compiles a placement-specific program with ordinary loading calls. Only
// affected controllers and their callers are recompiled; unchanged helpers
// borrow the source compiler's detached native results and exact call
// contracts. Physical selection supplies an upper bound of one descriptor per
// non-root group and the queue header/command. Exact native retention may
// remove whole loading groups. One bounded recompilation removes those calls
// and compacts descriptors; final capacity failure preserves the original
// loading program. The resulting layout contains only the selected native
// groups. The source pipeline scope owns lifetime/execution semantics
// throughout. A pending_report receives the owned native evidence for a
// provisional placement. The caller publishes it only after physical admission.
iree_status_t loom_aie2p_program_replacement_compile(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, uint32_t queue_header,
    uint32_t queue_command, const uint32_t* descriptors,
    loom_aie2p_program_layout_t* layout,
    loom_target_compile_report_t* pending_report, bool* out_compiled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPLACEMENT_H_
