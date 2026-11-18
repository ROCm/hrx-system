// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPORT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPORT_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Arena-owned inventory of linked code and the work of its logical callers.
typedef struct loom_aie2p_program_report_t {
  // Logical functions after final code-group selection and relocation.
  loom_target_compile_report_placed_function_t* functions;
  // Number of linked function rows.
  iree_host_size_t function_count;
  // Allocated function row capacity.
  iree_host_size_t function_capacity;
  // Counted loops for those functions, in physical order within each.
  loom_target_compile_report_packet_loop_t* loops;
  // Number of populated loop rows.
  iree_host_size_t loop_count;
  // Allocated loop row capacity.
  iree_host_size_t loop_capacity;
} loom_aie2p_program_report_t;

// Records a final linked worker. Native aliases retain their logical names and
// independent call multiplicities, including when other groups reuse the PC.
iree_status_t loom_aie2p_program_report_worker(
    loom_aie2p_program_report_t* report, iree_arena_allocator_t* arena,
    uint32_t worker_index,
    loom_target_compile_report_pipeline_placement_t placement,
    const loom_aie2p_program_object_t* program,
    const loom_aie2p_program_layout_t* layout,
    const loom_aie2p_linked_tile_t* linked);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_REPORT_H_
