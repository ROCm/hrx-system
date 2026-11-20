// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Installs linked entry sections using temporary circuits at the boundary
// between resetting/loading workers and establishing application routes.
// The inline program remains available to constrained product directories.
iree_status_t loom_aie2p_worker_upload_install(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_xdna_tile_t* tiles, iree_host_size_t tile_count,
    iree_host_size_t first_record, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_WORKER_UPLOAD_H_
