// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/publication.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/gpu/failure.h"

enum {
  IREE_HAL_AMD_GPU_PUBLICATION_STOP_REQUESTED = 1 << 0,
  IREE_HAL_AMD_GPU_PUBLICATION_WATCH_FAILURE = 1 << 1,
};

static bool iree_hal_amd_gpu_publication_is_ready(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  if (iree_atomic_load(&publication->state, iree_memory_order_acquire)) {
    return true;
  }
  const uint32_t word_count =
      iree_atomic_load(&publication->word_count, iree_memory_order_acquire);
  for (uint32_t i = 0; i < word_count; ++i) {
    if (iree_atomic_load(&publication->ready[i], iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_amd_gpu_publication_run_source(
    iree_hal_amd_gpu_publication_t* publication, uint32_t ordinal) {
  iree_hal_amd_gpu_publication_source_t source = {0};
  iree_slim_mutex_lock(&publication->sources.mutex);
  if (ordinal < publication->sources.count) {
    source = publication->sources.values[ordinal];
    if (source.publish) {
      publication->sources.active_ordinal = ordinal;
    }
  }
  iree_slim_mutex_unlock(&publication->sources.mutex);
  if (!source.publish) {
    return false;
  }

  // The table may be replaced and freed while this retained callback runs.
  const bool ready = source.publish(source.user_data);

  iree_slim_mutex_lock(&publication->sources.mutex);
  publication->sources.active_ordinal = UINT32_MAX;
  const bool joining = publication->sources.joining;
  publication->sources.joining = false;
  iree_slim_mutex_unlock(&publication->sources.mutex);
  if (joining) {
    iree_notification_post(&publication->sources.joined, 1);
  }
  return ready;
}

static int iree_hal_amd_gpu_publication_main(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  iree_hal_amd_gpu_failure_t* watch = NULL;
  iree_time_t deadline = IREE_TIME_INFINITE_FUTURE;
  while (true) {
    iree_notification_await(&publication->notification,
                            iree_hal_amd_gpu_publication_is_ready, publication,
                            iree_make_deadline(deadline));
    const int32_t state =
        iree_atomic_load(&publication->state, iree_memory_order_acquire);
    if (iree_any_bit_set(state, IREE_HAL_AMD_GPU_PUBLICATION_STOP_REQUESTED)) {
      break;
    }
    if (iree_any_bit_set(state, IREE_HAL_AMD_GPU_PUBLICATION_WATCH_FAILURE)) {
      watch = publication->failure_watch;
      iree_atomic_fetch_and(&publication->state,
                            ~IREE_HAL_AMD_GPU_PUBLICATION_WATCH_FAILURE,
                            iree_memory_order_release);
    }
    if (watch) {
      deadline = iree_hal_amd_gpu_failure_poll(watch, iree_time_now());
      if (deadline == IREE_TIME_INFINITE_FUTURE) {
        watch = NULL;
      }
    }
    const uint32_t word_count =
        iree_atomic_load(&publication->word_count, iree_memory_order_acquire);
    for (uint32_t i = 0; i < word_count; ++i) {
      uint64_t ready = (uint64_t)iree_atomic_exchange(
          &publication->ready[i], 0, iree_memory_order_acq_rel);
      while (ready) {
        const uint32_t bit = iree_math_count_trailing_zeros_u64(ready);
        const uint64_t mask = UINT64_C(1) << bit;
        ready &= ~mask;
        const uint32_t ordinal = i * 64 + bit;
        // The queue traces actual publication after claiming its producer.
        // Stale readiness hints and idle waits create no flame-chart scopes.
        if (iree_hal_amd_gpu_publication_run_source(publication, ordinal)) {
          // Keep one bounded turn per ready queue in each round. A source with
          // a long ready suffix cannot starve another notified publisher.
          iree_atomic_fetch_or(&publication->ready[i], (int64_t)mask,
                               iree_memory_order_release);
        }
      }
    }
  }
  return 0;
}

iree_status_t iree_hal_amd_gpu_publication_initialize(
    uint32_t source_count, const iree_hal_amd_gpu_publication_source_t* sources,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_publication_t* out_publication) {
  *out_publication = (iree_hal_amd_gpu_publication_t){
      .sources =
          {
              .values = sources,
              .count = source_count,
              .active_ordinal = UINT32_MAX,
          },
  };
  iree_atomic_store(&out_publication->word_count, (source_count + 63) / 64,
                    iree_memory_order_relaxed);
  const iree_thread_create_params_t params = {
      .name = IREE_SV("amd-gpu-submit"),
      .create_suspended = true,
      .initial_affinity = affinity,
  };
  IREE_RETURN_IF_ERROR(
      iree_thread_create(iree_hal_amd_gpu_publication_main, out_publication,
                         params, host_allocator, &out_publication->thread));
  iree_notification_initialize(&out_publication->notification);
  iree_slim_mutex_initialize(&out_publication->sources.mutex);
  iree_notification_initialize(&out_publication->sources.joined);
  iree_thread_resume(out_publication->thread);
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_publication_source_joined(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  iree_slim_mutex_lock(&publication->sources.mutex);
  const bool joined = !publication->sources.joining;
  iree_slim_mutex_unlock(&publication->sources.mutex);
  return joined;
}

void iree_hal_amd_gpu_publication_set_sources(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_count,
    const iree_hal_amd_gpu_publication_source_t* sources) {
  iree_slim_mutex_lock(&publication->sources.mutex);
  const uint32_t ordinal = publication->sources.active_ordinal;
  bool joining = false;
  if (ordinal != UINT32_MAX) {
    const iree_hal_amd_gpu_publication_source_t* active =
        &publication->sources.values[ordinal];
    joining = ordinal >= source_count ||
              active->publish != sources[ordinal].publish ||
              active->user_data != sources[ordinal].user_data;
  }
  publication->sources.joining = joining;
  const uint32_t previous_word_count = (publication->sources.count + 63) / 64;
  const uint32_t word_count = (source_count + 63) / 64;
  publication->sources.values = sources;
  publication->sources.count = source_count;
  iree_atomic_store(&publication->word_count, word_count,
                    iree_memory_order_release);
  iree_slim_mutex_unlock(&publication->sources.mutex);
  if (word_count > previous_word_count) {
    // A newly observed word may contain a stale hint whose producer would
    // coalesce without posting. Make the expanded predicate visible now.
    iree_notification_post(&publication->notification, 1);
  }
  if (joining) {
    iree_notification_await(&publication->sources.joined,
                            iree_hal_amd_gpu_publication_source_joined,
                            publication, iree_infinite_timeout());
  }
}

void iree_hal_amd_gpu_publication_notify(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal) {
  const uint64_t mask = UINT64_C(1) << (source_ordinal % 64);
  const uint64_t previous =
      (uint64_t)iree_atomic_fetch_or(&publication->ready[source_ordinal / 64],
                                     (int64_t)mask, iree_memory_order_acq_rel);
  if (!(previous & mask)) {
    iree_notification_post(&publication->notification, 1);
  }
}

void iree_hal_amd_gpu_publication_enqueue(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal,
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  if (iree_hal_amd_gpu_publisher_enqueue(publisher, entry)) {
    iree_hal_amd_gpu_publication_entry_t* ready =
        iree_hal_amd_gpu_publisher_next(publisher);
    if (iree_hal_amd_gpu_publisher_handoff(publisher, ready)) {
      iree_hal_amd_gpu_publication_notify(publication, source_ordinal);
    }
  }
}

void iree_hal_amd_gpu_publication_watch_failure(
    iree_hal_amd_gpu_publication_t* publication,
    iree_hal_amd_gpu_failure_t* failure) {
  publication->failure_watch = failure;
  iree_atomic_fetch_or(&publication->state,
                       IREE_HAL_AMD_GPU_PUBLICATION_WATCH_FAILURE,
                       iree_memory_order_release);
  iree_notification_post(&publication->notification, 1);
}

void iree_hal_amd_gpu_publication_deinitialize(
    iree_hal_amd_gpu_publication_t* publication) {
  if (!publication->thread) {
    return;
  }
  iree_atomic_fetch_or(&publication->state,
                       IREE_HAL_AMD_GPU_PUBLICATION_STOP_REQUESTED,
                       iree_memory_order_release);
  iree_notification_post(&publication->notification, 1);
  iree_thread_release(publication->thread);
  publication->thread = NULL;
  iree_notification_deinitialize(&publication->notification);
  iree_notification_deinitialize(&publication->sources.joined);
  iree_slim_mutex_deinitialize(&publication->sources.mutex);
}
