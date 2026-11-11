// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/completion.h"

#include "iree/base/threading/processor.h"
#include "iree/base/threading/wait_address.h"
#include "iree/hal/driver/amd/gpu/queue/signal.h"
#include "iree/hal/driver/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED = 1 << 0,
  IREE_HAL_AMD_GPU_COMPLETION_STOPPED = 1 << 1,
  IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES = 1 << 2,
  IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES = 1 << 3,
  IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT = 1 << 4,
  IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT = 1 << 5,
};

static bool iree_hal_amd_gpu_completion_stop_requested(
    iree_hal_amd_gpu_completion_t* completion) {
  return iree_any_bit_set(
      iree_atomic_load(&completion->state, iree_memory_order_acquire),
      IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED |
          IREE_HAL_AMD_GPU_COMPLETION_STOPPED);
}

static void iree_hal_amd_gpu_completion_scope_borrow(
    iree_hal_amd_gpu_completion_scope_t* scope) {
  if (!scope->previous) {
    scope->source->fn = NULL;
    scope->next = scope->completion->active_scopes;
    scope->previous = &scope->completion->active_scopes;
    if (scope->next) {
      scope->next->previous = &scope->next;
    }
    *scope->previous = scope;
  }
}

void iree_hal_amd_gpu_completion_scope_unlock(
    iree_hal_amd_gpu_completion_scope_t* scope) {
  if (scope->unlocked) {
    return;
  }
  iree_hal_amd_gpu_completion_scope_borrow(scope);
  scope->unlocked = true;
  iree_slim_mutex_unlock(&scope->completion->drain_mutex);
}

void iree_hal_amd_gpu_completion_scope_lock(
    iree_hal_amd_gpu_completion_scope_t* scope) {
  if (!scope->unlocked) {
    return;
  }
  iree_slim_mutex_lock(&scope->completion->drain_mutex);
  scope->unlocked = false;
}

bool iree_hal_amd_gpu_completion_scope_finish(
    iree_hal_amd_gpu_completion_scope_t* scope) {
  iree_hal_amd_gpu_completion_scope_lock(scope);
  if (!scope->previous) {
    return false;
  }
  scope->source->fn = scope->fn;
  *scope->previous = scope->next;
  if (scope->next) {
    scope->next->previous = scope->previous;
  }
  scope->previous = NULL;
  iree_hal_amd_gpu_completion_t* completion = scope->completion;
  int32_t state =
      iree_atomic_load(&completion->state, iree_memory_order_acquire);
  const int32_t waiting =
      IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT |
      (scope->is_worker ? 0 : IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT);
  if (state & waiting) {
    // Ending observation may clear BORROWING after the load above. Decide
    // whether to wake from this atomic handoff, not that earlier snapshot.
    state = iree_atomic_fetch_and(&completion->state, ~waiting,
                                  iree_memory_order_release);
    if (state & IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT) {
      iree_notification_post(&completion->callout_joined, IREE_ALL_WAITERS);
    }
  }
  // An active observer sees the cleared enrollment in its existing state
  // load. Only a worker that can park needs the native event wake.
  return !scope->is_worker &&
         (state & IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT) != 0 &&
         !(state & IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES);
}

// Only a worker-owned staging copy survives its source invocation. Move its
// existing stack links into the worker's longer-lived scope before returning.
static void iree_hal_amd_gpu_completion_scope_move(
    iree_hal_amd_gpu_completion_scope_t* source,
    iree_hal_amd_gpu_completion_scope_t* target) {
  *target = *source;
  *target->previous = target;
  if (target->next) {
    target->next->previous = &target->next;
  }
  source->previous = NULL;
}

// Membership changes only; the ordinary source round uses indexed table access.
static iree_hal_amd_gpu_completion_source_t*
iree_hal_amd_gpu_completion_find_source(
    iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources,
    iree_hal_amd_gpu_completion_fn_t fn, void* user_data) {
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    if (sources[i].fn == fn && sources[i].user_data == user_data) {
      return &sources[i];
    }
  }
  return NULL;
}

static bool iree_hal_amd_gpu_completion_has_removed_scope(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources) {
  for (iree_hal_amd_gpu_completion_scope_t* scope = completion->active_scopes;
       scope; scope = scope->next) {
    if (!iree_hal_amd_gpu_completion_find_source(
            source_count, sources, scope->fn, scope->source->user_data)) {
      return true;
    }
  }
  return false;
}

// Enters and returns under drain_mutex, which protects the caller's predicate
// through enrollment. A different joiner may rearm the hint before this thread
// sleeps; the notification epoch preserves the intervening callback return.
static void iree_hal_amd_gpu_completion_wait_callout(
    iree_hal_amd_gpu_completion_t* completion) {
  const iree_wait_token_t token =
      iree_notification_prepare_wait(&completion->callout_joined);
  iree_atomic_fetch_or(&completion->state,
                       IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT,
                       iree_memory_order_acq_rel);
  iree_slim_mutex_unlock(&completion->drain_mutex);
  iree_notification_commit_wait(&completion->callout_joined, token,
                                IREE_DURATION_ZERO, IREE_TIME_INFINITE_FUTURE);
  iree_slim_mutex_lock(&completion->drain_mutex);
}

// The caller holds drain_mutex and has excluded new worker rounds. Retained
// callbacks need no join; one may itself be acquiring another queue. Empty
// replacement joins every active invocation during terminal shutdown.
static void iree_hal_amd_gpu_completion_join_removed(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources) {
  while (iree_hal_amd_gpu_completion_has_removed_scope(completion, source_count,
                                                       sources)) {
    iree_hal_amd_gpu_completion_wait_callout(completion);
  }
}

static void iree_hal_amd_gpu_completion_clear_hints(
    iree_hal_amd_gpu_completion_t* completion) {
  for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
    iree_hal_amd_gpu_signal_release(completion->sources[i].wait.signal);
    completion->sources[i].wait = (iree_hal_amd_gpu_completion_wait_t){0};
  }
}

static void iree_hal_amd_gpu_completion_relocate_sources(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources) {
  // A source being called out has no function in the old table. Its stack
  // scope carries that identity and becomes bound to the new entry instead.
  for (iree_hal_amd_gpu_completion_scope_t* scope = completion->active_scopes;
       scope; scope = scope->next) {
    iree_hal_amd_gpu_completion_source_t* target =
        iree_hal_amd_gpu_completion_find_source(
            source_count, sources, scope->fn, scope->source->user_data);
    scope->source = target;
    scope->relocated = true;
    target->fn = NULL;
  }
  // Observation has joined and a relocated worker restarts its source round.
  // Fresh hints avoid searching for every inactive source in the new table.
  iree_hal_amd_gpu_completion_clear_hints(completion);
}

typedef enum iree_hal_amd_gpu_completion_caller_e {
  IREE_HAL_AMD_GPU_COMPLETION_CALLER_HELPER,
  IREE_HAL_AMD_GPU_COMPLETION_CALLER_WORKER,
} iree_hal_amd_gpu_completion_caller_t;

static bool iree_hal_amd_gpu_completion_drain(
    iree_hal_amd_gpu_completion_t* completion, amdf_status_t status,
    iree_hal_amd_gpu_completion_caller_t caller, iree_host_size_t first_source,
    bool* out_has_wait, iree_hal_amd_gpu_completion_copy_t* out_copy,
    iree_hal_amd_gpu_completion_scope_t* out_copy_scope, bool* out_wake) {
  const bool is_worker = caller == IREE_HAL_AMD_GPU_COMPLETION_CALLER_WORKER;
  bool ready = false;
  iree_host_size_t count = completion->source_count;
  iree_host_size_t index = first_source < count ? first_source : 0;
  for (iree_host_size_t remaining = count; remaining;
       --remaining, index = index + 1 < count ? index + 1 : 0) {
    iree_hal_amd_gpu_completion_source_t* source = &completion->sources[index];
    if (!source->fn) {
      if (is_worker && completion->active_scopes &&
          (!out_copy || !out_copy->fn)) {
        // Enroll before another source can unlock this round. The skipped
        // helper may return during that callout, after its native wake has
        // already been consumed but before the worker is ready to park.
        // A pending worker copy guarantees another round and may itself be
        // the only active scope when this source is permanently disabled.
        iree_atomic_fetch_or(&completion->state,
                             IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT,
                             iree_memory_order_release);
      }
      continue;
    }
    // The callback may replace and free its original table while unlocked.
    // Its output stays on this stack until metadata exclusion is restored.
    iree_hal_amd_gpu_completion_wait_t wait = {0};
    iree_hal_amd_gpu_completion_copy_t* copy =
        out_copy && !out_copy->fn ? out_copy : NULL;
    iree_hal_amd_gpu_completion_scope_t scope = {.completion = completion,
                                                 .source = source,
                                                 .fn = source->fn,
                                                 .is_worker = is_worker};
    ready |= scope.fn(source->user_data, status, &scope, &wait, copy);
    iree_hal_amd_gpu_completion_scope_lock(&scope);
    if (out_has_wait) {
      // Another source can yield to a helper before observation begins. Pin
      // this hint while its source is still exclusively owned.
      iree_hal_amd_gpu_signal_retain(wait.signal);
      scope.source->wait = wait;
    }
    if (copy && copy->fn) {
      if (iree_atomic_load(&completion->state, iree_memory_order_acquire) &
          IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES) {
        *copy = (iree_hal_amd_gpu_completion_copy_t){0};
      } else {
        iree_hal_amd_gpu_completion_scope_borrow(&scope);
        iree_hal_amd_gpu_completion_scope_move(&scope, out_copy_scope);
      }
    }
    if (out_has_wait) {
      *out_has_wait |= wait.pending;
    }
    if ((!copy || !copy->fn) &&
        iree_hal_amd_gpu_completion_scope_finish(&scope)) {
      *out_wake = true;
      if (amdf_status_is_ok(status)) {
        // Return the wake before entering another potentially long callout.
        // The worker resumes this round; an unrelated source cannot delay it.
        return ready;
      }
    }
    if (scope.relocated) {
      // A healthy round can restart on the new table. Terminal delivery must
      // remain once-only: retained sources preserve their relative order, so
      // continue past this invocation in its replacement position. New sources
      // independently observe the latched failure during their installation.
      if (amdf_status_is_ok(status)) {
        return true;
      }
      count = completion->source_count;
      index = scope.source - completion->sources;
      remaining = count - index;
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
  bool wake = false;
  if (!iree_any_bit_set(
          iree_atomic_load(&completion->state, iree_memory_order_acquire),
          IREE_HAL_AMD_GPU_COMPLETION_STOP_REQUESTED |
              IREE_HAL_AMD_GPU_COMPLETION_STOPPED)) {
    ready = iree_hal_amd_gpu_completion_drain(
        completion, AMDF_STATUS_OK, IREE_HAL_AMD_GPU_COMPLETION_CALLER_HELPER,
        0, NULL, NULL, NULL, &wake);
  }
  iree_slim_mutex_unlock(&completion->drain_mutex);
  if (wake) {
    const amdf_status_t status =
        iree_hal_amd_gpu_context_signal(completion->context);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_fail_execution(completion->context, status,
                                              "GPU callout completion wake");
    }
  }
  return ready;
}

// Called under the drain mutex before any source access can escape it. Atomic
// admission excludes a setter that requested replacement before taking the
// mutex. This borrow protects only the worker's coherent hint observation;
// source callbacks have independent stack ownership.
static bool iree_hal_amd_gpu_completion_begin_borrow(
    iree_hal_amd_gpu_completion_t* completion) {
  int32_t expected =
      iree_atomic_load(&completion->state, iree_memory_order_relaxed);
  if (expected & ~(IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT |
                   IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT)) {
    return false;
  }
  return iree_atomic_compare_exchange_strong(
      &completion->state, &expected,
      expected | IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES,
      iree_memory_order_acq_rel, iree_memory_order_relaxed);
}

static int32_t iree_hal_amd_gpu_completion_end_borrow(
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
  return previous;
}

static bool iree_hal_amd_gpu_completion_begin_observation(
    iree_hal_amd_gpu_completion_t* completion) {
  if (!iree_hal_amd_gpu_completion_begin_borrow(completion)) {
    return false;
  }
  return true;
}

static bool iree_hal_amd_gpu_completion_end_observation(
    iree_hal_amd_gpu_completion_t* completion, bool has_callouts) {
  iree_hal_amd_gpu_completion_clear_hints(completion);
  const int32_t previous = iree_hal_amd_gpu_completion_end_borrow(completion);
  // The atomic handoff covers a callout returning after the final observation
  // load. Before it, that caller clears enrollment without a native wake;
  // after it, the caller sees a parkable worker and signals the native event.
  return has_callouts &&
         !(previous & IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT);
}

// The source drain establishes these addresses once. Unchanged native work
// performs only coherent loads here, never callbacks, locks, or native queries.
static bool iree_hal_amd_gpu_completion_observe(
    iree_hal_amd_gpu_completion_t* completion, iree_time_t deadline,
    int64_t wake_epoch, bool has_callouts) {
  const int32_t observing_state =
      IREE_HAL_AMD_GPU_COMPLETION_BORROWING_SOURCES |
      (has_callouts ? IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT : 0);
  // A returning callout clears its enrolled wait even if every observed native
  // word is unchanged. Rejoin source observation before waiting on its event.
  while ((iree_atomic_load(&completion->state, iree_memory_order_relaxed) &
          ~IREE_HAL_AMD_GPU_COMPLETION_JOINING_CALLOUT) == observing_state &&
         iree_time_now() < deadline) {
    if (iree_atomic_load(&completion->context->wake_epoch,
                         iree_memory_order_acquire) != wake_epoch) {
      return true;
    }
    for (iree_host_size_t i = 0; i < completion->source_count; ++i) {
      const iree_hal_amd_gpu_completion_wait_t* wait =
          &completion->sources[i].wait;
      if (wait->address) {
        const int64_t value =
            wait->is_64_bit
                ? iree_atomic_load((const iree_atomic_int64_t*)wait->address,
                                   iree_memory_order_acquire)
                : iree_atomic_load((const iree_atomic_int32_t*)wait->address,
                                   iree_memory_order_acquire);
        if (value != wait->value) {
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
    bool wake = false;
    iree_hal_amd_gpu_completion_copy_t copy = {0};
    iree_hal_amd_gpu_completion_scope_t copy_scope = {0};
    bool ready = false;
    // A host wake during the drain must remain visible to observation even
    // when its native event arrives after the source was already inspected.
    const int64_t wake_epoch =
        iree_atomic_load(&context->wake_epoch, iree_memory_order_acquire);
    if (!(state & IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES)) {
      ready = iree_hal_amd_gpu_completion_drain(
          completion, AMDF_STATUS_OK, IREE_HAL_AMD_GPU_COMPLETION_CALLER_WORKER,
          first_source, &has_wait, &copy, &copy_scope, &wake);
    }
    if (copy.fn) {
      first_source = copy_scope.source - completion->sources + 1;
      iree_hal_amd_gpu_completion_clear_hints(completion);
      iree_hal_amd_gpu_completion_scope_unlock(&copy_scope);
      copy.fn(copy.user_data);
      iree_hal_amd_gpu_completion_scope_finish(&copy_scope);
      iree_slim_mutex_unlock(&completion->drain_mutex);
      // Native captures and staging credit still belong to their source. Its
      // next ordinary drain performs retirement after the final CPU write.
      continue;
    }
    if (ready || wake) {
      if (completion->spin_ns) {
        const iree_time_t now = iree_time_now();
        spin_deadline = completion->spin_ns < IREE_TIME_INFINITE_FUTURE - now
                            ? now + completion->spin_ns
                            : IREE_TIME_INFINITE_FUTURE;
      }
      iree_hal_amd_gpu_completion_clear_hints(completion);
      iree_slim_mutex_unlock(&completion->drain_mutex);
      continue;
    }
    const bool observe =
        has_wait && completion->spin_ns && iree_time_now() < spin_deadline &&
        iree_hal_amd_gpu_completion_begin_observation(completion);
    if (!observe) {
      iree_hal_amd_gpu_completion_clear_hints(completion);
    }
    const bool has_callouts = completion->active_scopes != NULL;
    if (has_callouts) {
      iree_atomic_fetch_or(&completion->state,
                           IREE_HAL_AMD_GPU_COMPLETION_WAITING_FOR_CALLOUT,
                           iree_memory_order_release);
    }
    iree_slim_mutex_unlock(&completion->drain_mutex);
    if (observe) {
      const bool changed = iree_hal_amd_gpu_completion_observe(
          completion, spin_deadline, wake_epoch, has_callouts);
      const bool callout_returned =
          iree_hal_amd_gpu_completion_end_observation(completion, has_callouts);
      if (changed || callout_returned) {
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
  iree_hal_amd_gpu_completion_join_removed(completion, 0, NULL);
  const amdf_status_t failure =
      iree_hal_amd_gpu_completion_query_status(completion);
  if (!amdf_status_is_ok(failure)) {
    // Join active helpers before the one terminal notification round. Native
    // inactivation runs outside this mutex so CPU publishers can hand off.
    bool wake = false;
    iree_hal_amd_gpu_completion_drain(completion, failure,
                                      IREE_HAL_AMD_GPU_COMPLETION_CALLER_WORKER,
                                      0, NULL, NULL, NULL, &wake);
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
    iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources, iree_duration_t spin_ns,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_completion, 0, sizeof(*out_completion));
  out_completion->context = context;
  out_completion->spin_ns = spin_ns;
  iree_slim_mutex_initialize(&out_completion->drain_mutex);
  iree_notification_initialize(&out_completion->callout_joined);
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
    iree_notification_deinitialize(&out_completion->callout_joined);
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
  // Join coherent observation and removed invocations. Retained callbacks may
  // be acquiring another queue here; their stack ownership survives the swap.
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
  iree_hal_amd_gpu_completion_join_removed(completion, source_count, sources);
  iree_hal_amd_gpu_completion_relocate_sources(completion, source_count,
                                               sources);
  completion->sources = sources;
  completion->source_count = source_count;
  iree_atomic_fetch_and(&completion->state,
                        ~IREE_HAL_AMD_GPU_COMPLETION_REPLACING_SOURCES,
                        iree_memory_order_release);
  iree_slim_mutex_unlock(&completion->drain_mutex);
}

bool iree_hal_amd_gpu_completion_remove_source(
    iree_hal_amd_gpu_completion_t* completion,
    iree_hal_amd_gpu_completion_fn_t fn, void* user_data) {
  iree_slim_mutex_lock(&completion->drain_mutex);
  for (iree_hal_amd_gpu_completion_scope_t* scope = completion->active_scopes;
       scope; scope = scope->next) {
    if (scope->fn == fn && scope->source->user_data == user_data) {
      // The current entry is already borrowed. Leave it disabled when the
      // invocation returns, retaining its position through source relocation.
      scope->fn = NULL;
      iree_slim_mutex_unlock(&completion->drain_mutex);
      return true;
    }
  }
  iree_hal_amd_gpu_completion_find_source(completion->source_count,
                                          completion->sources, fn, user_data)
      ->fn = NULL;
  iree_slim_mutex_unlock(&completion->drain_mutex);
  return false;
}

static bool iree_hal_amd_gpu_completion_has_disabled_scope(
    iree_hal_amd_gpu_completion_t* completion, void* user_data) {
  for (iree_hal_amd_gpu_completion_scope_t* scope = completion->active_scopes;
       scope; scope = scope->next) {
    if (!scope->fn && scope->source->user_data == user_data) {
      return true;
    }
  }
  return false;
}

void iree_hal_amd_gpu_completion_join_source(
    iree_hal_amd_gpu_completion_t* completion, void* user_data) {
  iree_slim_mutex_lock(&completion->drain_mutex);
  while (
      iree_hal_amd_gpu_completion_has_disabled_scope(completion, user_data)) {
    iree_hal_amd_gpu_completion_wait_callout(completion);
  }
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
    iree_hal_amd_gpu_context_t* context = completion->context;
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
  iree_notification_deinitialize(&completion->callout_joined);
  iree_slim_mutex_deinitialize(&completion->drain_mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
