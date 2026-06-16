// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_FEEDBACK_CHANNEL_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_FEEDBACK_CHANNEL_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

// Parameters used to initialize a per-physical-device feedback channel.
typedef struct iree_hal_amdgpu_feedback_channel_params_t {
  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Physical GPU agent that will publish packets into this channel.
  hsa_agent_t device_agent;
  // CPU fine-grained memory pool used for the control block.
  hsa_amd_memory_pool_t control_memory_pool;
  // GPU global memory pool used to create the pinned VMM packet ring.
  hsa_amd_memory_pool_t ring_memory_pool;
  // Topology used to grant ring access to CPU and GPU agents.
  const iree_hal_amdgpu_topology_t* topology;
  // Minimum packet ring capacity in bytes.
  iree_device_size_t minimum_capacity;
} iree_hal_amdgpu_feedback_channel_params_t;

// Host-owned feedback channel for one physical device.
typedef struct iree_hal_amdgpu_feedback_channel_t {
  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Device-visible configuration assigned to executable globals.
  iree_hal_amdgpu_feedback_config_t config;
  // Host interrupt signal used by device producers after publishing packets.
  hsa_signal_t notify_signal;
  // Device-visible control block allocated from host fine-grained memory.
  iree_hal_amdgpu_feedback_channel_header_t* control;
  // Mirrored VMM packet ring allocated from pinned host memory.
  iree_hal_amdgpu_vmem_ringbuffer_t ringbuffer;
} iree_hal_amdgpu_feedback_channel_t;

// Callback invoked for each ready packet drained from the channel.
typedef iree_status_t(IREE_API_PTR* iree_hal_amdgpu_feedback_drain_fn_t)(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data);

// Initializes |out_channel| with a host-pinned, device-writable packet ring.
iree_status_t iree_hal_amdgpu_feedback_channel_initialize(
    const iree_hal_amdgpu_feedback_channel_params_t* params,
    iree_hal_amdgpu_feedback_channel_t* out_channel);

// Deinitializes |channel| and releases all HSA resources.
void iree_hal_amdgpu_feedback_channel_deinitialize(
    iree_hal_amdgpu_feedback_channel_t* channel);

// Drains up to |max_packet_count| ready packets in reservation order.
//
// The callback observes packet storage that remains valid for the duration of
// the callback. The channel advances its read tail after each successful
// callback, allowing device producers to reuse that storage.
iree_status_t iree_hal_amdgpu_feedback_channel_drain(
    iree_hal_amdgpu_feedback_channel_t* channel,
    iree_host_size_t max_packet_count,
    iree_hal_amdgpu_feedback_drain_fn_t callback, void* user_data,
    iree_host_size_t* out_packet_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_FEEDBACK_CHANNEL_H_
