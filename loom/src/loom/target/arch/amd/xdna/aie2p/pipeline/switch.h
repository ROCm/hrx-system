// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical selection consumed by switch admission. These arrays may belong to
// source construction or an independently retained execution plan.
typedef struct loom_aie2p_native_switch_input_t {
  // Canonical physical edges, tile facts, and indexed route counts.
  const loom_aie2p_array_route_builder_t* routing;
  // Packet/control identity paired one-for-one with routing's edges.
  const loom_aie2p_native_route_t* routes;
  // Endpoint ownership of each core or tile-control sender.
  const loom_aie2p_native_control_source_t* control_sources;
  // Sole sender per destination; UINT32_MAX is unused, UINT32_MAX - 1 shared.
  const uint32_t* control_destination_sources;
} loom_aie2p_native_switch_input_t;

// Admits physical route edges against packet/circuit ownership and switch
// resources, retaining the complete initialization banks in |arena|.
// Destination sets connected by a shared output require one arbiter. Disjoint
// sets share only when fed exclusively by the same input, so arbitration adds
// no blocking dependency between independent streams. This does not prove
// progress for dependencies already present in the selected routes.
// Allocation errors return a status. A resource conflict returns a static
// |out_requirement| for the caller's source diagnostic; otherwise it is empty.
// |out_switches| receives complete banks only when admission succeeds.
iree_status_t loom_aie2p_native_switches_build(
    const loom_aie2p_native_switch_input_t* input,
    iree_arena_allocator_t* arena, loom_aie2p_native_switch_t** out_switches,
    iree_string_view_t* out_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
