// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materialization of physically planned AIE2P workers as resident Low CFGs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// One materialized resident worker program.
typedef struct loom_aie2p_array_resident_worker_t {
  // Index of the logical and physically planned worker.
  uint32_t worker_index;
  // Module-local symbol naming the materialized worker function.
  loom_symbol_ref_t entry;
  // Private retained core Low function implementing the resident loop.
  loom_op_t* function_op;
  // Invocation target facts borrowed from the source worker's compiler version.
  const loom_target_facts_t* function_target_facts;
  // Module-owned proof bindings translated into the resident invocation.
  loom_low_memory_access_map_t* memory_accesses;
} loom_aie2p_array_resident_worker_t;

// Materialized resident core programs for one physical array plan.
typedef struct loom_aie2p_array_resident_program_t {
  // Arena-owned worker programs in logical worker order.
  const loom_aie2p_array_resident_worker_t* workers;
  // Number of entries in workers.
  iree_host_size_t worker_count;
} loom_aie2p_array_resident_program_t;

// Materializes every planned worker as an independently compilable Low CFG.
//
// |source_module| owns the selected array and core leaf IR retained by |plan|.
// |resident_module| receives all generated functions and is the module used to
// compile and inspect them. The modules must share a finalized context and may
// not alias. Materialization does not mutate |source_module|.
//
// Each source worker function represents one channel firing. The materializer
// clones its arbitrary CFG once, replaces resource imports with loop-carried
// local-address values, surrounds the firing with the channel lock protocol,
// and advances every channel ring independently after the firing completes.
// Single-predecessor block chains are fused so scheduling can overlap firing
// and protocol work while preserving lock effects and shared loop headers.
// The resulting functions have no imported resources or register ABI and are
// retained as final array-image roots.
iree_status_t loom_aie2p_array_materialize_resident_program(
    const loom_module_t* source_module, loom_module_t* resident_module,
    const loom_aie2p_array_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_array_resident_program_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_H_
