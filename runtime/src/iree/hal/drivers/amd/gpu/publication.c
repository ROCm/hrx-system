// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/publication.h"

#include "iree/base/internal/math.h"

enum {
  IREE_HAL_AMD_GPU_PUBLICATION_STOP_REQUESTED = 1 << 0,
  IREE_HAL_AMD_GPU_PUBLICATION_REPLACE_SOURCES = 1 << 1,
};

static bool iree_hal_amd_gpu_publication_is_ready(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  if (iree_atomic_load(&publication->state, iree_memory_order_acquire)) {
    return true;
  }
  for (uint32_t i = 0; i < publication->word_count; ++i) {
    if (iree_atomic_load(&publication->ready[i], iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

static int iree_hal_amd_gpu_publication_main(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  while (true) {
    iree_notification_await(&publication->notification,
                            iree_hal_amd_gpu_publication_is_ready, publication,
                            iree_infinite_timeout());
    const int32_t state =
        iree_atomic_load(&publication->state, iree_memory_order_acquire);
    if (iree_any_bit_set(state, IREE_HAL_AMD_GPU_PUBLICATION_STOP_REQUESTED)) {
      break;
    }
    if (iree_any_bit_set(state, IREE_HAL_AMD_GPU_PUBLICATION_REPLACE_SOURCES)) {
      publication->sources = publication->replacement.sources;
      publication->source_count = publication->replacement.source_count;
      publication->word_count = (publication->source_count + 63) / 64;
      iree_atomic_fetch_and(&publication->state,
                            ~IREE_HAL_AMD_GPU_PUBLICATION_REPLACE_SOURCES,
                            iree_memory_order_release);
      iree_notification_post(&publication->replacement.completed, 1);
    }
    for (uint32_t i = 0; i < publication->word_count; ++i) {
      uint64_t ready = (uint64_t)iree_atomic_exchange(
          &publication->ready[i], 0, iree_memory_order_acq_rel);
      while (ready) {
        const uint32_t bit = iree_math_count_trailing_zeros_u64(ready);
        const uint64_t mask = UINT64_C(1) << bit;
        ready &= ~mask;
        const uint32_t ordinal = i * 64 + bit;
        if (ordinal >= publication->source_count) {
          continue;
        }
        const iree_hal_amd_gpu_publication_source_t* source =
            &publication->sources[ordinal];
        // The queue traces actual publication after claiming its producer.
        // Stale readiness hints and idle waits create no flame-chart scopes.
        if (source->publish && source->publish(source->user_data)) {
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
      .sources = sources,
      .source_count = source_count,
      .word_count = (source_count + 63) / 64,
  };
  const iree_thread_create_params_t params = {
      .name = IREE_SV("amd-gpu-submit"),
      .create_suspended = true,
      .initial_affinity = affinity,
  };
  IREE_RETURN_IF_ERROR(
      iree_thread_create(iree_hal_amd_gpu_publication_main, out_publication,
                         params, host_allocator, &out_publication->thread));
  iree_notification_initialize(&out_publication->notification);
  iree_notification_initialize(&out_publication->replacement.completed);
  iree_thread_resume(out_publication->thread);
  return iree_ok_status();
}

static bool iree_hal_amd_gpu_publication_replacement_completed(
    void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  return !iree_any_bit_set(
      iree_atomic_load(&publication->state, iree_memory_order_acquire),
      IREE_HAL_AMD_GPU_PUBLICATION_REPLACE_SOURCES);
}

void iree_hal_amd_gpu_publication_set_sources(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_count,
    const iree_hal_amd_gpu_publication_source_t* sources) {
  publication->replacement.sources = sources;
  publication->replacement.source_count = source_count;
  iree_atomic_fetch_or(&publication->state,
                       IREE_HAL_AMD_GPU_PUBLICATION_REPLACE_SOURCES,
                       iree_memory_order_release);
  iree_notification_post(&publication->notification, 1);
  iree_notification_await(&publication->replacement.completed,
                          iree_hal_amd_gpu_publication_replacement_completed,
                          publication, iree_infinite_timeout());
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
  iree_notification_deinitialize(&publication->replacement.completed);
}
