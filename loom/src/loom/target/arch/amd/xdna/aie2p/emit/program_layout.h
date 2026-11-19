// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LAYOUT_H_

#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Aligned demand of this worker's unique native bodies at one construction.
typedef struct loom_aie2p_program_demand_t {
  // Construction ordinal in the shared pipeline lifetime tree.
  iree_host_size_t owner;
  // Aligned native bytes attributed to this owner on this worker.
  uint64_t length;
} loom_aie2p_program_demand_t;

typedef struct loom_aie2p_program_census_t {
  // Native entry, synthetic loader, and helper construction owners.
  const iree_host_size_t* function_owners;
  // Number of native functions, including the synthetic loader.
  iree_host_size_t function_count;
  // Only constructions with nonempty native demand on this worker.
  const loom_aie2p_program_demand_t* demands;
  // Number of construction demands.
  iree_host_size_t demand_count;
} loom_aie2p_program_census_t;

// Lifetime-local native code, loaded as one image when its owner starts.
typedef struct loom_aie2p_program_layout_group_t {
  // Construction ordinal in the retained lifetime tree.
  iree_host_size_t owner;
  // Stable request in the original common projection, including empty groups.
  iree_host_size_t request_index;
  // Exact native section and packed byte requirement.
  loom_aie2p_program_code_group_t native;
  // Offset in this core's program memory, including transport alignment.
  uint64_t offset;
} loom_aie2p_program_layout_group_t;

typedef struct loom_aie2p_program_layout_t {
  // Complete detached program with independently relocatable code sections.
  loom_aie2p_program_object_t program;
  // Original code group to load at each call boundary, in helper then entry
  // order. Zero means the caller can invoke that boundary directly.
  const iree_host_size_t* const* call_groups;
  // Dense code group for each function in the program's retained input order.
  iree_host_size_t* function_groups;
  // Root group first, then other reached construction owners in ordinal order.
  // The root's code is resident; a child group's code survives its descendants.
  loom_aie2p_program_layout_group_t* groups;
  // Number of reached construction owners, not the whole pipeline's size.
  iree_host_size_t group_count;
  // Prepared common topology shared by retention and exact byte placement.
  loom_pipeline_lifetime_projection_t* projection;
  // Group requests in stable construction order; numeric requirements vary.
  loom_pipeline_lifetime_request_t* requests;
  // Original projection size, preserved when native groups become empty.
  iree_host_size_t request_count;
  // Maximum simultaneous instruction footprint, including write padding.
  uint64_t byte_length;
} loom_aie2p_program_layout_t;

// Returns the construction-to-code map when this worker needs fewer code-load
// descriptors than the incumbent placement. The original native image already
// exceeded capacity, so the worker still needs at least one replaceable group.
// All results and temporary storage have the lifetime of one worker's scratch.
iree_status_t loom_aie2p_program_layout_reduce_catalog(
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t owner_count,
    const loom_aie2p_program_census_t* census, uint64_t capacity,
    iree_host_size_t current_group_count, iree_arena_allocator_t* scratch,
    const iree_host_size_t** out_owners);

// Groups the admitted entry/helper closure by producer-retained construction
// owners before native composition. Functions have entry-then-helper order.
// Result storage belongs to arena; temporary indexes belong to scratch_arena.
// No source traversal or target-private dependence solve occurs.
iree_status_t loom_aie2p_program_layout_initialize(
    const loom_pipeline_lifetimes_t* lifetimes,
    const iree_host_size_t* function_owners, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_layout_t* out_layout);

// Copies mutable placement decisions while sharing their prepared topology.
// Each candidate can change requirements without invalidating another layout.
iree_status_t loom_aie2p_program_layout_copy(
    const loom_aie2p_program_layout_t* source, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* out_layout);

// Promotes native-equivalent bodies into resident storage using the common
// lifetime policy. A zero minimum permits removing entire loading groups;
// otherwise each group retains at least that many aligned native units.
// Native composition follows selection.
iree_status_t loom_aie2p_program_layout_retain(
    const loom_aie2p_program_native_t* native, uint64_t minimum_group_length,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* layout);

// Removes empty native groups while preserving their common lifetime requests.
// Returns the old-to-dense group map; removed groups map to the resident group.
// Functions retain their order, and remaining groups retain construction order.
iree_status_t loom_aie2p_program_layout_compact(
    iree_host_size_t function_count, iree_arena_allocator_t* arena,
    loom_aie2p_program_layout_t* layout, iree_host_size_t** out_group_indices);

// Places exact compiled group sizes with the retained common projection. Parent
// code coexists with descendants; concurrent groups remain disjoint and serial
// groups reuse instruction memory. Native composition has populated program;
// groups has one exact native record per initialized construction group.
// Transport writes require four-byte lengths; entry alignment is unchanged.
iree_status_t loom_aie2p_program_layout_finish(
    const loom_aie2p_program_code_group_t* groups,
    loom_aie2p_program_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_LAYOUT_H_
