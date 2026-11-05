// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/completion.h"

#include "iree/base/threading/processor.h"
#include "iree/base/threading/wait_address.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
#include "iree/hal/drivers/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_COMPLETION_RUNNING = 0,
  IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED = 1 << 0,
  IREE_HAL_AMD_GPU_COMPLETION_STOPPED = 1 << 1,
  IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES = 1 << 2,
  IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES = 1 << 3,
};

static bool iree_hal_amd_gpu_completion_stop_requested(
    iree_hal_amd_gpu_completion_t* completion) {
  return iree_any_bit_set(
      iree_atomic_load(&completion->state, iree_memory_order_acquire),
      IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED |
          IREE_HAL_AMD_GPU_COMPLETION_STOPPED);
}

static bool iree_hal_amd_gpu_completion_drain(
    iree_hal_amd_gpu_completion_t* completion, amdf_status_t status,
    iree_host_size_t first_source, bool* out_has_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy) {
  bool ready = false;
  const iree_host_size_t count = completion->source_count;
  iree_host_size_t index = first_source < count ? first_source : 0;
  for (iree_host_size_t i = 0; i < count;
       ++i, index = index + 1 < count ? index + 1 : 0) {
    iree_hal_amd_gpu_completion_source_t* source = &completion->sources[index];
    if (source == completion->active_source) {
      continue;
    }
    iree_hal_amd_gpu_completion_wait_t ignored_wait;
    iree_hal_amd_gpu_completion_wait_t* wait =
        out_has_wait ? &source->wait : &ignored_wait;
    *wait = (iree_hal_amd_gpu_completion_wait_t){0};
    iree_hal_amd_gpu_completion_copy_t* copy =
        out_copy && !out_copy->fn ? out_copy : NULL;
    ready |= source->fn(source->user_data, status, wait, copy);
    if (copy && copy->fn) {
      completion->active_source = source;
    }
    if (out_has_wait) {
      *out_has_wait |= wait->address != NULL;
    }
  }
  return ready;
}

bool iree_hal_amd_gpu_completion_try_drain(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!iree_slim_mutex_try_lock(&completion->drain_mutex)) {
    return false;
  }
  bool ready = false;
  if (!iree_any_bit_set(
          iree_atomic_load(&completion->state, iree_memory_order_acquire),
          IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED |
              IREE_HAL_AMD_GPU_COMPLETION_STOPPED |
              IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES)) {
    ready = iree_hal_amd_gpu_completion_drain(completion, AMDF_STATUS_OK, 0,
                                              NULL, NULL);
  }
  iree_slim_mutex_unlock(&completion->drain_mutex);
  return ready;
}

// Called under the drain mutex before any source access can escape it. Atomic
// admission excludes a setter that requested replacement before taking the
// mutex. Only the worker borrows sources; helpers still drain under the mutex.
static bool iree_hal_amd_gpu_completion_begin_borrow(
    iree_hal_amd_gpu_completion_t* completion) {
  int32_t expected = IREE_HAL_AMD_GPU_COMPLETION_RUNNING;
  return iree_atomic_compare_exchange_strong(
      &completion->state, &expected,
      IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES, iree_memory_order_acq_rel,
      iree_memory_order_relaxed);
}

static void iree_hal_amd_gpu_completion_end_borrow(
    iree_hal_amd_gpu_completion_t* completion) {
  // No further unlocked access to the source table or its borrowed storage.
  // Only cold source replacement waits here; steady-state retirement never
  // makes an address-wake syscall.
  const int32_t previous = iree_atomic_fetch_and(
      &completion->state, ~IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES,
      iree_memory_order_release);
  if (iree_any_bit_set(previous,
                       IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES)) {
    iree_wait_address_wake_all(&completion->state);
  }
}

static bool iree_hal_amd_gpu_completion_begin_observation(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!iree_hal_amd_gpu_completion_begin_borrow(completion)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
    iree_hal_amd_gpu_signal_retain(completion->sources[i].wait.signal);
  }
  return true;
}

static void iree_hal_amd_gpu_completion_end_observation(
    iree_hal_amd_gpu_completion_t* completion) {
  for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
    iree_hal_amd_gpu_signal_release(completion->sources[i].wait.signal);
  }
  iree_hal_amd_gpu_completion_end_borrow(completion);
}

// The source drain establishes these addresses once. Unchanged native work
// performs only coherent loads here, never callbacks, locks, or native queries.
static bool iree_hal_amd_gpu_completion_observe(
    iree_hal_amd_gpu_completion_t* completion, iree_time_t deadline) {
  while (iree_atomic_load(&completion->state, iree_memory_order_relaxed) ==
             IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES &&
         iree_time_now() < deadline) {
    for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
      const iree_hal_amd_gpu_completion_wait_t* wait =
          &completion->sources[i].wait;
      if (wait->address) {
        if (iree_atomic_load(wait->address, iree_memory_order_acquire) !=
            wait->value) {
          return true;
        }
      }
    }
    iree_processor_yield();
  }
  return false;
}

static int iree_hal_amd_gpu_completion_main(void* user_data) {
  iree_hal_amd_gpu_completion_t* completion = user_data;
  const iree_hal_amd_gpu_context_t* context = completion->context;
  iree_time_t spin_deadline = IREE_TIME_INFINITE_PAST;
  // Continue after the last copy source so a busy download cannot monopolize
  // the worker. Source replacement may shrink the table; drain wraps the index.
  iree_host_size_t first_source = 0;
  while (!iree_hal_amd_gpu_completion_stop_requested(completion)) {
    iree_slim_mutex_lock(&completion->drain_mutex);
    const int32_t state =
        iree_atomic_load(&completion->state, iree_memory_order_acquire);
    if (iree_any_bit_set(state, IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED |
                                    IREE_HAL_AMD_GPU_COMPLETION_STOPPED)) {
      iree_slim_mutex_unlock(&completion->drain_mutex);
      break;
    }
    bool has_wait = false;
    iree_hal_amd_gpu_completion_copy_t copy = {0};
    bool ready = iree_hal_amd_gpu_completion_drain(
        completion, AMDF_STATUS_OK, first_source, &has_wait,
        state == IREE_HAL_AMD_GPU_COMPLETION_RUNNING ? &copy : NULL);
    if (copy.fn) {
      first_source = completion->active_source - completion->sources + 1;
      const bool borrowed =
          iree_hal_amd_gpu_completion_begin_borrow(completion);
      if (!borrowed) {
        completion->active_source = NULL;
      }
      iree_slim_mutex_unlock(&completion->drain_mutex);
      if (borrowed) {
        copy.fn(copy.user_data);
        iree_slim_mutex_lock(&completion->drain_mutex);
        completion->active_source = NULL;
        iree_hal_amd_gpu_completion_end_borrow(completion);
        iree_slim_mutex_unlock(&completion->drain_mutex);
      }
      // Native captures and staging credit still belong to their source. Its
      // next ordinary drain performs retirement after the final CPU write.
      continue;
    }
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
    const bool observe =
        has_wait && completion->spin_ns && iree_time_now() < spin_deadline &&
        iree_hal_amd_gpu_completion_begin_observation(completion);
    iree_slim_mutex_unlock(&completion->drain_mutex);
    if (observe) {
      const bool changed =
          iree_hal_amd_gpu_completion_observe(completion, spin_deadline);
      iree_hal_amd_gpu_completion_end_observation(completion);
      if (changed) {
        continue;
      }
    }
    if (iree_hal_amd_gpu_completion_stop_requested(completion)) {
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
      if (context->event) {
        iree_hal_amd_gpu_context_fail_execution(context, status,
                                                "gpu.event_wait");
      } else {
        iree_hal_amd_gpu_context_report_status(context, NULL, host_wait_status);
        iree_status_free(host_wait_status);
        iree_hal_amd_gpu_context_fail_execution(context, status,
                                                "kernel completion wait");
      }
      if (!context->execution_failure.notify) {
        // A construction/test context may have no enclosing terminal owner.
        iree_hal_amd_gpu_completion_request_failure(completion, status);
      }
      break;
    }
  }
  iree_slim_mutex_lock(&completion->drain_mutex);
  const amdf_status_t failure =
      iree_hal_amd_gpu_completion_query_status(completion);
  if (!amdf_status_is_ok(failure)) {
    // Join active helpers before the one terminal notification round. Native
    // inactivation runs outside this mutex so CPU publishers can hand off.
    iree_hal_amd_gpu_completion_drain(completion, failure, 0, NULL, NULL);
  }
  iree_slim_mutex_unlock(&completion->drain_mutex);
  if (!amdf_status_is_ok(failure) && context->execution_failure.join) {
    context->execution_failure.join(context->execution_failure.user_data,
                                    failure);
  }
  // Final release joins the terminal owner too; stopped helpers cannot enter
  // another source round.
  iree_slim_mutex_lock(&completion->drain_mutex);
  iree_atomic_fetch_or(&completion->state, IREE_HAL_AMD_GPU_COMPLETION_STOPPED,
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
  out_completion->spin_ns = spin_ns;
  iree_slim_mutex_initialize(&out_completion->drain_mutex);
  iree_hal_amd_gpu_completion_set_sources(out_completion, source_count,
                                          sources);
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

void iree_hal_amd_gpu_completion_set_sources(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources) {
  // Stop new borrows, join unlocked reads/copies, then join callbacks and swap
  // the table. Helpers can finish while we join the worker's unlocked access.
  iree_atomic_fetch_or(&completion->state,
                       IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES,
                       iree_memory_order_release);
  int32_t state =
      iree_atomic_load(&completion->state, iree_memory_order_acquire);
  while (
      iree_any_bit_set(state, IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES)) {
    iree_wait_address_wait_int32(&completion->state, state,
                                 IREE_TIME_INFINITE_FUTURE);
    state = iree_atomic_load(&completion->state, iree_memory_order_acquire);
  }
  iree_slim_mutex_lock(&completion->drain_mutex);
  completion->sources = sources;
  completion->source_count = source_count;
  iree_atomic_fetch_and(&completion->state,
                        ~IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES,
                        iree_memory_order_release);
  iree_slim_mutex_unlock(&completion->drain_mutex);
}

amdf_status_t iree_hal_amd_gpu_completion_query_status(
    iree_hal_amd_gpu_completion_t* completion) {
  return (amdf_status_t)iree_atomic_load(&completion->failure,
                                         iree_memory_order_acquire);
}

void iree_hal_amd_gpu_completion_request_failure(
    iree_hal_amd_gpu_completion_t* completion, amdf_status_t status) {
  int64_t expected = 0;
  iree_atomic_compare_exchange_strong(&completion->failure, &expected, status,
                                      iree_memory_order_release,
                                      iree_memory_order_relaxed);
  iree_atomic_fetch_or(&completion->state,
                       IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED,
                       iree_memory_order_release);
}

amdf_status_t iree_hal_amd_gpu_completion_request_stop(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!completion->thread) {
    return AMDF_STATUS_OK;
  }
  const int32_t previous = iree_atomic_fetch_or(
      &completion->state, IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED,
      iree_memory_order_acq_rel);
  if (!iree_any_bit_set(previous, IREE_HAL_AMD_GPU_COMPLETION_STOPPED)) {
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
