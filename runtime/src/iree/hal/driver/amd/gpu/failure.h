// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_FAILURE_H_
#define IREE_HAL_DRIVER_AMD_GPU_FAILURE_H_

#include "amdf/amdf.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/device_event.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold terminal phases used by the native cleanup owner and its independent
// existing-worker monitor. Native removal has no cancellable deadline; expiry
// reports retained ownership without certifying that native access stopped.
typedef enum iree_hal_amd_gpu_failure_phase_e {
  IREE_HAL_AMD_GPU_FAILURE_CLOSING = 0,
  IREE_HAL_AMD_GPU_FAILURE_INACTIVATING,
  IREE_HAL_AMD_GPU_FAILURE_RETIRING,
  IREE_HAL_AMD_GPU_FAILURE_JOINING,
  IREE_HAL_AMD_GPU_FAILURE_FINISHED,
} iree_hal_amd_gpu_failure_phase_t;

typedef struct iree_hal_amd_gpu_failure_t {
  // Borrowed immutable native identity; no failed-device query is needed.
  const amdf_endpoint_info_t* endpoint;
  // Copied diagnostic destination, borrowed through the worker joins.
  iree_hal_device_event_sink_t event_sink;
  // Borrowed HAL identity, live through completion and publication worker
  // joins.
  iree_hal_device_t* device;
  // Native-stop software wait budget and report threshold in milliseconds.
  uint32_t timeout_ms;
  // Host progress/callback notification, independent of the failed native wake.
  iree_notification_t changed;
  // First native error, release-published after immutable timing facts.
  iree_atomic_int64_t status;
  // Original diagnostic operation, borrowed through terminal cleanup.
  const char* operation;
  // First-fault monotonic timestamp, immutable after status publication.
  iree_time_t started;
  // Absolute attempt/report deadline; expiry cannot detach live CPU borrowers.
  iree_time_t deadline;
  // Cached phase and logical queue ordinal for a possibly blocked native call.
  iree_atomic_int64_t position;
  // Number of native owners already joined in the current phase.
  iree_atomic_int32_t completed_count;
  // Total native owners captured before terminal traversal.
  iree_atomic_int32_t total_count;
  // Exactly one deadline diagnostic, shared by the monitor and cleanup owner.
  iree_atomic_int32_t reported;
} iree_hal_amd_gpu_failure_t;

void iree_hal_amd_gpu_failure_initialize(
    const amdf_endpoint_info_t* endpoint, iree_hal_device_t* device,
    iree_hal_device_event_sink_t event_sink, uint32_t timeout_ms,
    iree_hal_amd_gpu_failure_t* failure);
void iree_hal_amd_gpu_failure_deinitialize(iree_hal_amd_gpu_failure_t* failure);

// Records the first fault and its timing facts under caller serialization.
// True selects the one caller that arms independent monitoring and reports the
// original error. Later faults preserve that original outcome. operation stays
// live through the worker join.
bool iree_hal_amd_gpu_failure_begin(iree_hal_amd_gpu_failure_t* failure,
                                    amdf_status_t status, const char* operation,
                                    iree_time_t now);

// Updates cached diagnostics before a native call or software ownership join.
// Ordinals below 256 are compute queues. UINT32_MAX - 1 denotes the transfer
// queue; UINT32_MAX denotes the private code queue or initial device closure.
void iree_hal_amd_gpu_failure_set_phase(iree_hal_amd_gpu_failure_t* failure,
                                        iree_hal_amd_gpu_failure_phase_t phase,
                                        uint32_t queue_ordinal,
                                        uint32_t completed_count,
                                        uint32_t total_count);

// Reports a deadline once through DRIVER_FAILURE, returning the next monitor
// deadline or INFINITE_FUTURE when complete/already reported. This performs no
// native call, queue traversal, or wait on the native cleanup owner's locks.
iree_time_t iree_hal_amd_gpu_failure_poll(iree_hal_amd_gpu_failure_t* failure,
                                          iree_time_t now);

// Ends monitoring after CPU borrowers join, with either native access cessation
// or explicitly retained backing. This does not itself supply a native proof.
void iree_hal_amd_gpu_failure_finish(iree_hal_amd_gpu_failure_t* failure);

// Wakes software ownership joins only while terminal failure is active.
bool iree_hal_amd_gpu_failure_wake(iree_hal_amd_gpu_failure_t* failure);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_FAILURE_H_
