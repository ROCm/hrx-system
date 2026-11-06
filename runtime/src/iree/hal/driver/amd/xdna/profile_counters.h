// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_PROFILE_COUNTERS_H_
#define IREE_HAL_DRIVER_AMD_XDNA_PROFILE_COUNTERS_H_

#include "iree/hal/driver/amd/xdna/executable.h"

#ifdef __cplusplus
extern "C" {
#endif

// Finite queue-ordered attribution and optional partition-shim timer capture.
// Only the exclusive native publisher mutates the ledger. Begin/end run with
// the device idle; flush cannot collect or reset the native append position.
typedef struct iree_hal_amd_xdna_profile_counters_t
    iree_hal_amd_xdna_profile_counters_t;

// Copies selection strings and reserves bounded attribution/result storage.
// Counter capture attaches its destination once. Metadata-only capture uses no
// native controls. Context, arena, and sink are borrowed through end.
iree_status_t iree_hal_amd_xdna_profile_counters_begin(
    const iree_hal_device_profiling_options_t* options,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_profile_counters_t** out_counters);

// Selects measured bytes once per prepared command, before its first native
// submit attempt. Retry preserves this descriptor and does not call again.
// Exhausted storage leaves the ordinary command unchanged and reports loss
// only if that command is accepted. Previously captured data remains valid.
// Failures stop capture, preserve ordinary execution, and surface at end.
void iree_hal_amd_xdna_profile_counters_prepare(
    iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_profile_queue_event_t* event, uint32_t command_index,
    iree_hal_amd_xdna_invocation_t* invocation,
    amdf_xdna_kernel_command_t* command);

// Commits attribution only after native acceptance, in native command order.
// Retains the executable until end; rejected commands never enter the ledger.
void iree_hal_amd_xdna_profile_counters_accept(
    iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_profile_queue_event_t* event, uint32_t command_index,
    iree_hal_amd_xdna_invocation_t* invocation,
    const amdf_xdna_kernel_command_t* command);

// Collects after all writers retire, validates the complete native protocol,
// and emits retained executable metadata and cycle deltas. Capacity loss emits
// a TRUNCATED chunk after the retained data without failing the session.
// Consumes counters on every path. Ambiguous native controls retain backing
// and mark the context's parent graph for retention; neither sink failure nor
// invalid bytes require a leak.
iree_status_t iree_hal_amd_xdna_profile_counters_end(
    iree_hal_amd_xdna_profile_counters_t* counters);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_PROFILE_COUNTERS_H_
