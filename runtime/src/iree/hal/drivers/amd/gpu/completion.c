// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/completion.h"

#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_COMPLETION_RUNNING = 0,
  IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED = 1,
  IREE_HAL_AMD_GPU_COMPLETION_STOPPED = 2,
};

static bool iree_hal_amd_gpu_completion_drain(
    iree_hal_amd_gpu_completion_t* completion, amdf_status_t status) {
  bool ready = false;
  for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
    iree_hal_amd_gpu_completion_source_t* source = &completion->sources[i];
    source->wait = (iree_hal_amd_gpu_completion_wait_t){0};
    ready |= source->fn(source->user_data, status, &source->wait);
  }
  return ready;
}

bool iree_hal_amd_gpu_completion_try_drain(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!iree_slim_mutex_try_lock(&completion->drain_mutex)) {
    return false;
  }
  bool ready = false;
  if (iree_atomic_load(&completion->state, iree_memory_order_acquire) ==
      IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
    ready = iree_hal_amd_gpu_completion_drain(completion, AMDF_STATUS_OK);
  }
  iree_slim_mutex_unlock(&completion->drain_mutex);
  return ready;
}

// The source drain establishes these addresses once. Unchanged native work
// performs only coherent loads here, never callbacks, locks, or native queries.
static bool iree_hal_amd_gpu_completion_observe(
    iree_hal_amd_gpu_completion_t* completion, iree_time_t deadline) {
  while (iree_atomic_load(&completion->state, iree_memory_order_relaxed) ==
             IREE_HAL_AMD_GPU_COMPLETION_RUNNING &&
         iree_time_now() < deadline) {
    bool has_address = false;
    for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
      const iree_hal_amd_gpu_completion_wait_t* wait =
          &completion->sources[i].wait;
      if (wait->address) {
        has_address = true;
        if (iree_atomic_load(wait->address, iree_memory_order_acquire) !=
            wait->value) {
          return true;
        }
      }
    }
    if (!has_address) {
      break;
    }
    iree_processor_yield();
  }
  return false;
}

static int iree_hal_amd_gpu_completion_main(void* user_data) {
  iree_hal_amd_gpu_completion_t* completion = user_data;
  const iree_hal_amd_gpu_context_t* context = completion->context;
  iree_time_t spin_deadline = IREE_TIME_INFINITE_PAST;
  while (iree_atomic_load(&completion->state, iree_memory_order_acquire) ==
         IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
    iree_slim_mutex_lock(&completion->drain_mutex);
    if (iree_atomic_load(&completion->state, iree_memory_order_acquire) !=
        IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
      iree_slim_mutex_unlock(&completion->drain_mutex);
      break;
    }
    bool ready = iree_hal_amd_gpu_completion_drain(completion, AMDF_STATUS_OK);
    if (ready) {
      if (completion->spin_ns) {
        const iree_time_t now = iree_time_now();
        spin_deadline = completion->spin_ns < IREE_TIME_INFINITE_FUTURE - now
                            ? now + completion->spin_ns
                            : IREE_TIME_INFINITE_FUTURE;
      }
      iree_slim_mutex_unlock(&completion->drain_mutex);
      continue;
    }
    if (completion->spin_ns &&
        iree_hal_amd_gpu_completion_observe(completion, spin_deadline)) {
      iree_slim_mutex_unlock(&completion->drain_mutex);
      continue;
    }
    iree_slim_mutex_unlock(&completion->drain_mutex);
    if (iree_atomic_load(&completion->state, iree_memory_order_acquire) !=
        IREE_HAL_AMD_GPU_COMPLETION_RUNNING) {
      break;
    }

    // Signals before entry remain pending, closing the predicate-to-wait gap.
    // A stale or coalesced wake simply causes another authoritative scan.
    amdf_status_t status = AMDF_STATUS_OK;
    iree_status_t host_wait_status = iree_ok_status();
    if (context->event) {
      status = context->gpu->event_wait(context->event, AMDF_TIMEOUT_INFINITE);
    } else {
      host_wait_status = iree_async_event_native_wait(&context->kernel_event,
                                                      iree_infinite_timeout());
      if (!iree_status_is_ok(host_wait_status)) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      }
    }
    if (!amdf_status_is_ok(status)) {
      // Exclude new helper drains before reporting failure. An already active
      // source round joins before the terminal notification reaches sources.
      iree_atomic_store(&completion->state,
                        IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED,
                        iree_memory_order_release);
      if (context->event) {
        iree_hal_amd_gpu_context_report(context, status, "gpu.event_wait");
      } else {
        iree_hal_amd_gpu_context_report_status(context, NULL, host_wait_status);
        iree_status_free(host_wait_status);
      }
      iree_slim_mutex_lock(&completion->drain_mutex);
      iree_hal_amd_gpu_completion_drain(completion, status);
      iree_slim_mutex_unlock(&completion->drain_mutex);
      break;
    }
  }
  // Join an active helper before returning to the destruction owner; stopped
  // helpers cannot enter another source round.
  iree_slim_mutex_lock(&completion->drain_mutex);
  iree_atomic_store(&completion->state, IREE_HAL_AMD_GPU_COMPLETION_STOPPED,
                    iree_memory_order_release);
  iree_slim_mutex_unlock(&completion->drain_mutex);
  return 0;
}

iree_status_t iree_hal_amd_gpu_completion_initialize(
    const iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources, iree_duration_t spin_ns,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_completion, 0, sizeof(*out_completion));
  out_completion->context = context;
  out_completion->source_count = source_count;
  out_completion->sources = sources;
  out_completion->spin_ns = spin_ns;
  iree_slim_mutex_initialize(&out_completion->drain_mutex);
  const iree_thread_create_params_t params = {
      .name = IREE_SV("amd-gpu-retire"),
      .create_suspended = true,
      .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
      .initial_affinity = affinity,
  };
  iree_status_t status =
      iree_thread_create(iree_hal_amd_gpu_completion_main, out_completion,
                         params, host_allocator, &out_completion->thread);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&out_completion->drain_mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_thread_resume(out_completion->thread);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

amdf_status_t iree_hal_amd_gpu_completion_request_stop(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!completion->thread) {
    return AMDF_STATUS_OK;
  }
  int32_t expected = IREE_HAL_AMD_GPU_COMPLETION_RUNNING;
  iree_atomic_compare_exchange_strong(
      &completion->state, &expected, IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
  if (expected != IREE_HAL_AMD_GPU_COMPLETION_STOPPED) {
    // A failed wake preserves the join obligation. Retrying must signal again
    // even when a previous call has already published STOP_REQUESTED.
    const iree_hal_amd_gpu_context_t* context = completion->context;
    return iree_hal_amd_gpu_context_signal(context);
  }
  return AMDF_STATUS_OK;
}

iree_status_t iree_hal_amd_gpu_completion_deinitialize(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!completion->thread) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              iree_hal_amd_gpu_completion_request_stop(completion),
              "gpu.event_signal"));
  iree_thread_release(completion->thread);
  completion->thread = NULL;
  iree_slim_mutex_deinitialize(&completion->drain_mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
