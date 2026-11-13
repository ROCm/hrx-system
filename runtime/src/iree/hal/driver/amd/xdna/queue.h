// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_XDNA_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/hal/driver/amd/xdna/command_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates one finite execution owner. Context and proactor are borrowed from
// the device. Ready operations publish directly from their caller; the proactor
// owns deferred causal admission and unattended checked native completion,
// while exact local waits may publish checked retirement on their caller.
// Private native-publication and host-transfer services isolate blocking work.
// Unresolved consumers occupy no native execution slot.
iree_status_t iree_hal_amd_xdna_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Returns the immutable native array limit, bounded by HAL pending capacity.
uint32_t iree_hal_amd_xdna_queue_maximum_command_count(iree_hal_queue_t* queue);

// Assigns or clears an idle queue's causal domain. The tracker is retained.
// Topology assignment is externally serialized against submission and teardown.
iree_status_t iree_hal_amd_xdna_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

// Attempts a checked wait for an exact epoch owned by |queue|. Returns true
// after consuming the wait and setting |out_status|. Returns false without
// modifying |out_status| when the epoch is not currently mapped to a pending
// native submission or queue-state ownership is contended.
bool iree_hal_amd_xdna_queue_try_wait_epoch(iree_hal_queue_t* queue,
                                            iree_async_axis_t producer_axis,
                                            uint64_t producer_epoch,
                                            iree_timeout_t timeout,
                                            iree_async_wait_flags_t flags,
                                            iree_status_t* out_status);

// Captures host queue and execution records under the device profiling
// contract. Native submission spans measure host preparation/publication, not
// NPU work.
iree_status_t iree_hal_amd_xdna_queue_profiling_begin(
    iree_hal_queue_t* queue, uint32_t physical_device_ordinal,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    const iree_hal_device_profiling_options_t* options,
    iree_hal_profile_session_id_t session_id);
iree_status_t iree_hal_amd_xdna_queue_profiling_flush(iree_hal_queue_t* queue);
iree_status_t iree_hal_amd_xdna_queue_profiling_end(iree_hal_queue_t* queue);

// Releases currently unused operation and payload capture blocks.
void iree_hal_amd_xdna_queue_trim(iree_hal_queue_t* queue);

// Stops an idle queue and joins its native services and persistent observer.
// The application has finished queue operations and keeps proactor progress
// running through this call. Called outside proactor/service callbacks. False
// diagnoses failed native teardown and preserves the remaining native owners.
bool iree_hal_amd_xdna_queue_shutdown(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_QUEUE_H_
