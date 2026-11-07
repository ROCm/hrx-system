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
#include "loom/codegen/low/read_only_data.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// One module-internal native helper cloned into the resident module.
typedef struct loom_aie2p_array_resident_helper_t {
  // Module-local symbol naming the cloned helper definition.
  loom_symbol_ref_t entry;
  // Invocation target facts borrowed from the source helper version.
  const loom_target_facts_t* function_target_facts;
  // Module-owned memory proofs translated to cloned helper operations.
  loom_low_memory_access_map_t* memory_accesses;
  // Module-owned loop facts translated to cloned helper controls.
  loom_low_loop_execution_map_t* loop_execution;
} loom_aie2p_array_resident_helper_t;

// One materialized resident worker program shared by compatible instances.
typedef struct loom_aie2p_array_resident_program_t {
  // First logical worker instance represented by this program.
  uint32_t representative_worker_index;
  // Module-local symbol naming the materialized worker function.
  loom_symbol_ref_t entry;
  // Private retained core Low function implementing the resident loop.
  loom_op_t* function_op;
  // Invocation target facts borrowed from the source worker's compiler version.
  const loom_target_facts_t* function_target_facts;
  // Module-owned proof bindings translated into the resident invocation.
  loom_low_memory_access_map_t* memory_accesses;
  // Module-owned loop facts translated into the resident invocation.
  loom_low_loop_execution_map_t* loop_execution;
  // Complete immutable inventory for helper composition, translated to resident
  // symbols in physical-plan order. Empty when there are no direct helpers.
  loom_low_read_only_data_set_t program_data;
  // Direct helpers remapped into the resident module in source call order.
  const loom_aie2p_array_resident_helper_t* direct_helpers;
  // Number of records in |direct_helpers|.
  iree_host_size_t direct_helper_count;
} loom_aie2p_array_resident_program_t;

// Materialized resident core program set for one physical array plan.
typedef struct loom_aie2p_array_resident_program_set_t {
  // Arena-owned programs in first-instance order.
  const loom_aie2p_array_resident_program_t* programs;
  // Number of entries in programs.
  iree_host_size_t program_count;
  // Arena-owned diagnostic names indexed by physical worker instance.
  const iree_string_view_t* instance_names;
} loom_aie2p_array_resident_program_set_t;

// Materializes every unique planned worker program as a compilable Low CFG.
//
// |source_module| owns the selected array and core leaf IR retained by |plans|.
// |resident_module| receives the retained read-only data definitions and all
// generated functions and is the module used to compile and inspect them. The
// modules must share a finalized context and may not alias. Materialization
// does not mutate |source_module|.
//
// All plans sharing a resident module are materialized in one call. This keeps
// one source-symbol projection across the batch, so a read-only data definition
// referenced by several entries or workers is cloned exactly once. Each source
// worker function represents one channel firing. The materializer clones its
// arbitrary CFG once, replaces resource imports with loop-carried local-address
// values, surrounds the firing with the channel lock protocol, and advances
// every channel ring independently after the firing completes.
// Source scheduling contracts become explicit controls around the cloned body,
// ending at each source return before the generated latch and fold protocol.
// The source stays immutable; generated protocol work retains its own ordering.
// Single-predecessor block chains are fused so scheduling can overlap firing
// and protocol work while preserving lock effects and shared loop headers. The
// resulting functions have no imported resources or register ABI and are
// retained as final array-image roots.
iree_status_t loom_aie2p_array_materialize_resident_programs(
    const loom_module_t* source_module, loom_module_t* resident_module,
    const loom_aie2p_array_plan_t* plans, iree_host_size_t plan_count,
    iree_arena_allocator_t* arena,
    loom_aie2p_array_resident_program_set_t* out_program_sets);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_H_
