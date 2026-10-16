// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_PROFILE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_PROFILE_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-owned session with bounded, independently synchronized queue streams.
// Begin/end require idle queues and external serialization with submission.
// Appending never allocates or calls a sink; only begin/flush/end call the
// sink.
typedef struct iree_hal_amd_gpu_profile_t iree_hal_amd_gpu_profile_t;

// Optional inline capture metadata, present only during a profiling session.
// The operation's publication owner alone updates this record. A native
// operation may publish several chunks but contributes one public queue event.
typedef struct iree_hal_amd_gpu_profile_operation_t {
  // Public identity and admission facts, copied to the stream at readiness.
  iree_hal_profile_queue_event_t event;
  // The stream has received this operation, including a possible ring drop.
  bool recorded;
} iree_hal_amd_gpu_profile_operation_t;

// Begins QUEUE_EVENTS capture and emits device/queue metadata. Unsupported
// families and filters fail before starting a sink session. An empty request
// returns NULL. The sink remains retained until end.
iree_status_t iree_hal_amd_gpu_profile_begin(
    const iree_hal_device_profiling_options_t* options, uint64_t session_id,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_t** out_profile);

// Writes stable stream snapshots outside their mutexes. Failed writes retain
// the snapshot for retry. The caller serializes flush/end with each other.
iree_status_t iree_hal_amd_gpu_profile_flush(
    iree_hal_amd_gpu_profile_t* profile);

// Flushes, ends, and frees the session even when sink callbacks fail.
iree_status_t iree_hal_amd_gpu_profile_end(iree_hal_amd_gpu_profile_t* profile);

// Initializes identity at acceptance. The caller fills type-specific fields
// before transferring the capture to its execution owner.
void iree_hal_amd_gpu_profile_operation_initialize(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t wait_count,
    iree_host_size_t signal_count, iree_host_size_t operation_count,
    iree_hal_amd_gpu_profile_operation_t* operation);

// Records actual software deferral before first native publication/readiness.
static inline void iree_hal_amd_gpu_profile_operation_defer(
    iree_hal_amd_gpu_profile_operation_t* operation) {
  if (operation && !operation->recorded) {
    operation->event.flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
    operation->event.dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  }
}

// Publishes an operation once. Readiness is host-observable admission to the
// executor, not GPU execution time. Failed admission uses ready=false. Native
// wait packets are counted independently of a prior host deferral.
void iree_hal_amd_gpu_profile_operation_record(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation, bool ready,
    uint32_t native_wait_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_PROFILE_H_
