// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/completion.h"

#include "iree/hal/drivers/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_COMPLETION_RUNNING = 0,
  IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED = 1,
  IREE_HAL_AMD_GPU_COMPLETION_STOPPED = 2,
};

static int iree_hal_amd_gpu_completion_main(void* user_data) {
  iree_hal_amd_gpu_completion_t* completion = user_data;
  const iree_hal_amd_gpu_context_t* context = completion->context;
  while (iree_atomic_load(&completion->state, iree_memory_order_acquire) ==
         IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_poll, "iree_hal_amd_gpu_completion_poll");
    bool ready = false;
    for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
      const iree_hal_amd_gpu_completion_source_t* source =
          &completion->sources[i];
      ready |= source->fn(source->user_data, AMDF_STATUS_OK);
    }
    IREE_TRACE_ZONE_END(z_poll);
    if (ready) {
      continue;
    }
    if (iree_atomic_load(&completion->state, iree_memory_order_acquire) !=
        IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
      break;
    }

    // Signals before entry remain pending, closing the predicate-to-wait gap.
    // A stale or coalesced wake simply causes another authoritative scan.
    IREE_TRACE_ZONE_BEGIN_NAMED(z_wait, "iree_hal_amd_gpu_completion_wait");
    amdf_status_t status =
        context->gpu->event_wait(context->event, AMDF_TIMEOUT_INFINITE);
    IREE_TRACE_ZONE_END(z_wait);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(context, status, "gpu.event_wait");
      for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
        const iree_hal_amd_gpu_completion_source_t* source =
            &completion->sources[i];
        source->fn(source->user_data, status);
      }
      break;
    }
  }
  iree_atomic_store(&completion->state, IREE_HAL_AMD_GPU_COMPLETION_STOPPED,
                    iree_memory_order_release);
  return 0;
}

iree_status_t iree_hal_amd_gpu_completion_initialize(
    const iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    const iree_hal_amd_gpu_completion_source_t* sources,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_completion, 0, sizeof(*out_completion));
  out_completion->context = context;
  out_completion->source_count = source_count;
  out_completion->sources = sources;
  const iree_thread_create_params_t params = {
      .name = IREE_SV("hal-gpu-retire"),
      .create_suspended = true,
      .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
      .initial_affinity = affinity,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_thread_create(iree_hal_amd_gpu_completion_main, out_completion,
                             params, host_allocator, &out_completion->thread));
  iree_thread_resume(out_completion->thread);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_completion_deinitialize(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!completion->thread) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  int32_t expected = IREE_HAL_AMD_GPU_COMPLETION_RUNNING;
  iree_atomic_compare_exchange_strong(
      &completion->state, &expected, IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
  if (expected != IREE_HAL_AMD_GPU_COMPLETION_STOPPED) {
    // A failed wake preserves the join obligation. Retrying must signal again
    // even when a previous call has already published STOP_REQUESTED.
    const iree_hal_amd_gpu_context_t* context = completion->context;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        IREE_HAL_AMD_STATUS_FROM_AMDF(
            context->gpu->event_signal(context->event), "gpu.event_signal"));
  }
  iree_thread_release(completion->thread);
  completion->thread = NULL;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
