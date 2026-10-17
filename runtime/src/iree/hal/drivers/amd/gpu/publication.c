// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/publication.h"

#include "iree/base/internal/math.h"

static bool iree_hal_amd_gpu_publication_is_ready(void* user_data) {
  iree_hal_amd_gpu_publication_t* publication = user_data;
  if (iree_atomic_load(&publication->stopping, iree_memory_order_acquire)) {
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
    if (iree_atomic_load(&publication->stopping, iree_memory_order_acquire)) {
      break;
    }
    for (uint32_t i = 0; i < publication->word_count; ++i) {
      uint64_t ready = (uint64_t)iree_atomic_exchange(
          &publication->ready[i], 0, iree_memory_order_acq_rel);
      while (ready) {
        const uint32_t bit = iree_math_count_trailing_zeros_u64(ready);
        const uint64_t mask = UINT64_C(1) << bit;
        ready &= ~mask;
        const iree_hal_amd_gpu_publication_source_t* source =
            &publication->sources[i * 64 + bit];
        // The queue traces actual publication after claiming its producer.
        // Stale readiness hints and idle waits create no flame-chart scopes.
        if (source->publish(source->user_data)) {
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
      .word_count = (source_count + 63) / 64,
  };
  const iree_thread_create_params_t params = {
      .name = IREE_SV("hal-gpu-submit"),
      .create_suspended = true,
      .initial_affinity = affinity,
  };
  IREE_RETURN_IF_ERROR(
      iree_thread_create(iree_hal_amd_gpu_publication_main, out_publication,
                         params, host_allocator, &out_publication->thread));
  iree_notification_initialize(&out_publication->notification);
  iree_thread_resume(out_publication->thread);
  return iree_ok_status();
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

void iree_hal_amd_gpu_publication_deinitialize(
    iree_hal_amd_gpu_publication_t* publication) {
  if (!publication->thread) {
    return;
  }
  iree_atomic_store(&publication->stopping, 1, iree_memory_order_release);
  iree_notification_post(&publication->notification, 1);
  iree_thread_release(publication->thread);
  publication->thread = NULL;
  iree_notification_deinitialize(&publication->notification);
}
