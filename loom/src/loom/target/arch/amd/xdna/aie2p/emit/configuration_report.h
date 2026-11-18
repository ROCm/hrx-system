// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CONFIGURATION_REPORT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CONFIGURATION_REPORT_H_

#include "loom/analysis/source_storage_packing.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional emission-scoped inventory. The configuration evaluator records
// admitted reservations and linked placements at their owning boundaries.
// Collection never inspects IR or reconstructs logical channels from addresses.
typedef struct loom_aie2p_configuration_report_t
    loom_aie2p_configuration_report_t;

// Creates arena-owned collection only for an explicitly requested report.
iree_status_t loom_aie2p_configuration_report_allocate(
    loom_target_compile_report_t* destination,
    const loom_xdna_array_family_t* family, uint32_t column_count,
    iree_string_view_t root_name, iree_arena_allocator_t* arena,
    loom_aie2p_configuration_report_t** out_report);

// Records each distinct compiled function once, before any placement copies.
void loom_aie2p_configuration_report_program(
    loom_aie2p_configuration_report_t* report, uint32_t code_byte_count);

// Records the complete admitted fixed-range set for one physical store. Ranges
// may alias; their union is occupied once regardless of the number of owners.
iree_status_t loom_aie2p_configuration_report_reservations(
    loom_aie2p_configuration_report_t* report,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_source_storage_packing_range_t* ranges,
    iree_host_size_t range_count);

// Records one successfully linked worker. Its placements are disjoint from all
// fixed reservations and prior placements by the canonical packing contract.
iree_status_t loom_aie2p_configuration_report_worker(
    loom_aie2p_configuration_report_t* report, iree_string_view_t entry_name,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_aie2p_program_object_t* program,
    const loom_aie2p_tile_link_layout_t* layout,
    const loom_aie2p_linked_tile_t* linked);

// Copies the complete physical inventory to the destination report. Borrowed
// names have the destination report's module lifetime; no arena strings escape.
iree_status_t loom_aie2p_configuration_report_finish(
    loom_aie2p_configuration_report_t* report, uint32_t binding_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CONFIGURATION_REPORT_H_
