// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/timestamp.h"

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

//===----------------------------------------------------------------------===//
// Profiling signal initialization
//===----------------------------------------------------------------------===//

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_timestamp_arm_completion_signal(
    iree_amd_signal_t* IREE_AMDGPU_RESTRICT signal) {
  signal->kind = IREE_AMD_SIGNAL_KIND_USER;
  // AQL completion signals are armed with a positive value and decremented by
  // the CP on completion. Some command processors only populate start/end
  // timestamps for armed user signals, even though the timestamp fields are
  // independent of host-side signal waits.
  signal->value = 1;
  signal->event_mailbox_ptr = 0;
  signal->event_id = 0;
  signal->reserved1 = 0;
  signal->start_ts = 0;
  signal->end_ts = 0;
  signal->queue_ptr = NULL;
  signal->reserved3[0] = 0;
  signal->reserved3[1] = 0;
}

void iree_hal_amdgpu_device_timestamp_emplace_signal_initialization(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        signal_initialization_kernel_args,
    iree_amd_signal_t* signals, uint32_t signal_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_dispatch_timestamp_signal_initialize_args_t*
      IREE_AMDGPU_RESTRICT kernargs =
          (iree_hal_amdgpu_dispatch_timestamp_signal_initialize_args_t*)
              kernarg_ptr;
  kernargs->signals = signals;
  kernargs->signal_count = signal_count;
  kernargs->reserved0 = 0;

  const uint32_t workgroup_size =
      signal_initialization_kernel_args->workgroup_size[0];
  const uint32_t workgroup_count[3] = {
      (uint32_t)IREE_AMDGPU_CEIL_DIV(signal_count, workgroup_size), 1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      signal_initialization_kernel_args, workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr);
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_timestamp_initialize_completion_signals(
    iree_amd_signal_t* IREE_AMDGPU_RESTRICT signals, uint32_t signal_count) {
  const size_t signal_index = iree_hal_amdgpu_device_global_linear_id_1d();
  if (signal_index >= signal_count) return;

  iree_amd_signal_t* IREE_AMDGPU_RESTRICT signal = &signals[signal_index];
  iree_hal_amdgpu_device_timestamp_arm_completion_signal(signal);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Dispatch timestamp harvest
//===----------------------------------------------------------------------===//

iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*
iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        harvest_kernel_args,
    uint32_t source_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_dispatch_timestamp_harvest_args_t* IREE_AMDGPU_RESTRICT
      kernargs =
          (iree_hal_amdgpu_dispatch_timestamp_harvest_args_t*)kernarg_ptr;
  iree_hal_amdgpu_dispatch_timestamp_harvest_source_t* sources =
      iree_hal_amdgpu_device_timestamp_dispatch_harvest_sources(kernarg_ptr);
  kernargs->sources = sources;
  kernargs->source_count = source_count;
  kernargs->reserved0 = 0;

  const uint32_t harvest_workgroup_size =
      harvest_kernel_args->workgroup_size[0];
  const uint32_t harvest_workgroup_count[3] = {
      (uint32_t)IREE_AMDGPU_CEIL_DIV(source_count, harvest_workgroup_size), 1,
      1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      harvest_kernel_args, harvest_workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr);
  return sources;
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_timestamp_harvest_dispatch_records(
    const iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*
        IREE_AMDGPU_RESTRICT sources,
    uint32_t source_count) {
  const size_t source_index = iree_hal_amdgpu_device_global_linear_id_1d();
  if (source_index >= source_count) return;

  const iree_hal_amdgpu_dispatch_timestamp_harvest_source_t source =
      sources[source_index];
  source.ticks->start_tick = source.completion_signal->start_ts;
  source.ticks->end_tick = source.completion_signal->end_ts;
  iree_hal_amdgpu_device_timestamp_arm_completion_signal(
      source.completion_signal);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
