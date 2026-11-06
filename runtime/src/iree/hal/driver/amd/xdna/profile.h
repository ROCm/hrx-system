// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_PROFILE_H_
#define IREE_HAL_DRIVER_AMD_XDNA_PROFILE_H_

#include "iree/hal/driver/amd/xdna/profile_counters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bounded host-clock capture. Queue events describe public submissions. Host
// execution spans cover native preparation/publication and mapped host work;
// they exclude device execution and asynchronous file I/O latency.
typedef struct iree_hal_amd_xdna_profile_t iree_hal_amd_xdna_profile_t;

// Begins a session and emits its device/queue identity. Supports QUEUE_EVENTS,
// HOST_EXECUTION_EVENTS, EXECUTABLE_METADATA, and finite COUNTER_SAMPLES.
// LIGHTWEIGHT_STATISTICS selects only the two host event families. An empty
// request returns NULL. The sink is retained until end. Begin/end require the
// HAL idle-device contract; warm capture never allocates or invokes the sink.
iree_status_t iree_hal_amd_xdna_profile_begin(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_profile_session_id_t session_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_profile_t** out_profile);

// Returns the optional finite counter/metadata recorder, or NULL.
iree_hal_amd_xdna_profile_counters_t* iree_hal_amd_xdna_profile_counters(
    iree_hal_amd_xdna_profile_t* profile);

// Writes completed host event snapshots without holding the recording mutex.
// Native results are collected only at idle end; flush does not rearm them.
// Failed writes leave the snapshot available for a subsequent flush.
iree_status_t iree_hal_amd_xdna_profile_flush(
    iree_hal_amd_xdna_profile_t* profile);

// Flushes and ends a session, consuming profile even if sink callbacks fail.
iree_status_t iree_hal_amd_xdna_profile_end(
    iree_hal_amd_xdna_profile_t* profile);

// Assigns submission identity and host timestamp to a captured queue event.
void iree_hal_amd_xdna_profile_initialize_event(
    iree_hal_amd_xdna_profile_t* profile,
    iree_hal_profile_queue_event_t* event);

// Appends a finalized queue event. Full rings account dropped records for
// flush.
void iree_hal_amd_xdna_profile_append_queue(
    iree_hal_amd_xdna_profile_t* profile,
    const iree_hal_profile_queue_event_t* event);

// Samples the host clock only when host execution capture is enabled.
iree_time_t iree_hal_amd_xdna_profile_host_begin(
    iree_hal_amd_xdna_profile_t* profile);

// Records a completed span; a zero start denotes disabled host capture.
void iree_hal_amd_xdna_profile_host_end(
    iree_hal_amd_xdna_profile_t* profile,
    const iree_hal_profile_queue_event_t* queue_event, iree_time_t start,
    iree_hal_profile_host_execution_event_flags_t flags,
    iree_status_code_t status_code, uint32_t operation_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_PROFILE_H_
