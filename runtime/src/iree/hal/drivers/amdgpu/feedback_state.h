// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/device_event.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/feedback_channel.h"

typedef struct iree_thread_t iree_thread_t;
typedef struct iree_hal_amdgpu_logical_device_options_t
    iree_hal_amdgpu_logical_device_options_t;
typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_feedback_state_t
//===----------------------------------------------------------------------===//

// Takes ownership of |status| produced by a feedback service thread.
typedef void(IREE_API_PTR* iree_hal_amdgpu_feedback_error_handler_fn_t)(
    void* user_data, iree_status_t status);

// Per-physical-device feedback service state.
typedef struct iree_hal_amdgpu_feedback_device_state_t {
  // Owning logical-device feedback state.
  struct iree_hal_amdgpu_feedback_state_t* parent;

  // Physical device ordinal associated with this feedback channel.
  iree_host_size_t physical_device_ordinal;

  // Serializes drains between the feedback service thread and queue retirement.
  iree_slim_mutex_t drain_mutex;

  // Host-owned channel shared with device producers.
  iree_hal_amdgpu_feedback_channel_t channel;

  // Thread draining ready packets from |channel|.
  iree_thread_t* service_thread;

  // HSA signal used to request |service_thread| shutdown.
  hsa_signal_t stop_signal;
} iree_hal_amdgpu_feedback_device_state_t;

// Logical-device feedback state shared by executable and service paths.
typedef struct iree_hal_amdgpu_feedback_state_t {
  // True when feedback support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for state-owned allocations.
  iree_allocator_t host_allocator;

  // Borrowed HAL device used for event source attribution.
  iree_hal_device_t* device;

  // Borrowed stable device identifier used for event source attribution.
  iree_string_view_t device_id;

  // Programmatic sink receiving decoded feedback events.
  iree_hal_device_event_sink_t event_sink;

  // Policy applied after a valid ASAN report is emitted.
  iree_hal_amdgpu_asan_report_policy_t asan_report_policy;

  // Policy applied after a valid TSAN report is emitted.
  iree_hal_amdgpu_tsan_report_policy_t tsan_report_policy;

  // Number of entries in |device_states|.
  iree_host_size_t device_state_count;

  // Per-physical-device feedback channels and service threads.
  iree_hal_amdgpu_feedback_device_state_t* device_states;

  // Callback consuming service-thread errors. Owns the passed status.
  iree_hal_amdgpu_feedback_error_handler_fn_t error_handler;

  // User data passed to |error_handler|.
  void* error_handler_user_data;
} iree_hal_amdgpu_feedback_state_t;

// Initializes |out_state| from logical-device options.
//
// When feedback is disabled this leaves |out_state| zeroed and returns OK.
// Sanitizers currently enable feedback because report packets need a serviced
// device-to-host transport; other feedback clients can enable the same state
// once they have packet schemas and handlers.
iree_status_t iree_hal_amdgpu_feedback_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_hal_device_t* device, iree_string_view_t device_id,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amdgpu_feedback_error_handler_fn_t error_handler,
    void* error_handler_user_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_state_t* out_state);

// Stops service threads and releases any feedback resources owned by |state|.
void iree_hal_amdgpu_feedback_state_deinitialize(
    iree_hal_amdgpu_feedback_state_t* state);

// Returns true when |state| owns enabled feedback resources.
bool iree_hal_amdgpu_feedback_state_is_enabled(
    const iree_hal_amdgpu_feedback_state_t* state);

// Drains ready packets from the channel for |physical_device_ordinal|.
//
// This reports channel or packet handling errors through the state error
// handler. It may be called by the feedback service thread after a device
// notification or by host queue retirement before queue-owned resources and
// signal semaphores are released.
void iree_hal_amdgpu_feedback_state_drain_physical_device(
    iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal);

// Populates |out_config| with the device-visible feedback configuration.
//
// Callers must only call this when feedback is enabled.
// |physical_device_ordinal| selects the physical device whose executable global
// is being assigned.
iree_status_t iree_hal_amdgpu_feedback_state_populate_config(
    const iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_feedback_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_FEEDBACK_STATE_H_
