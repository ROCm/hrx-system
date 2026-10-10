// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_SUBMITTED_SIGNAL_H_
#define IREE_HAL_UTILS_SUBMITTED_SIGNAL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint8_t iree_hal_submitted_signal_flags_t;
enum iree_hal_submitted_signal_flag_bits_e {
  IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE = 0u,

  // The cache contains a producer axis/epoch/value snapshot from at least one
  // signal submission.
  IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID = 1u << 0,

  // The semaphore's post-publish frontier is exactly the producer queue's
  // frontier at |epoch|. Waiting on |producer_axis| at |epoch| therefore
  // observes every transitive dependency carried by the signal.
  IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT = 1u << 1,
};

// Seqlock-protected metadata for the most recently submitted semaphore signal.
//
// Writers are serialized by the containing semaphore or a single-producer
// contract. Readers may load concurrently without taking that lock. Payload
// fields remain atomic because retrying a seqlock does not make concurrent
// non-atomic C accesses data-race-free.
typedef struct iree_hal_submitted_signal_t {
  // Seqlock sequence counter; odd means a writer is updating payload fields.
  iree_atomic_int32_t sequence;

  // Cached signal validity and producer-frontier precision flags.
  iree_atomic_int32_t flags;

  // Producer queue axis that submitted the cached signal.
  iree_atomic_int64_t producer_axis;

  // Producer queue epoch associated with the cached signal.
  iree_atomic_int64_t epoch;

  // Semaphore payload value submitted at |producer_axis| and |epoch|.
  iree_atomic_int64_t value;
} iree_hal_submitted_signal_t;

// Stores one complete signal snapshot. Callers must serialize writers.
static inline void iree_hal_submitted_signal_store(
    iree_hal_submitted_signal_t* cache, iree_hal_submitted_signal_flags_t flags,
    iree_async_axis_t producer_axis, uint64_t epoch, uint64_t value) {
  // Publish the odd sequence before any payload field. The release fence pairs
  // with a reader that observes a concurrent payload store and forces its
  // closing sequence load to observe this write in progress.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_relaxed);
  iree_atomic_thread_fence(iree_memory_order_release);
  iree_atomic_store(&cache->flags, (int32_t)flags, iree_memory_order_relaxed);
  iree_atomic_store(&cache->producer_axis, (int64_t)producer_axis,
                    iree_memory_order_relaxed);
  iree_atomic_store(&cache->epoch, (int64_t)epoch, iree_memory_order_relaxed);
  iree_atomic_store(&cache->value, (int64_t)value, iree_memory_order_relaxed);
  // Publish the completed payload to readers beginning a new snapshot.
  iree_atomic_fetch_add(&cache->sequence, 1, iree_memory_order_release);
}

// Loads one complete signal snapshot. Returns false when the cache is invalid.
static inline bool iree_hal_submitted_signal_load(
    const iree_hal_submitted_signal_t* cache,
    iree_hal_submitted_signal_flags_t* out_flags,
    iree_async_axis_t* out_producer_axis, uint64_t* out_epoch,
    uint64_t* out_value) {
  for (;;) {
    const int32_t sequence =
        iree_atomic_load(&cache->sequence, iree_memory_order_acquire);
    if (IREE_UNLIKELY(sequence & 1)) {
      continue;
    }
    *out_flags = (iree_hal_submitted_signal_flags_t)iree_atomic_load(
        &cache->flags, iree_memory_order_relaxed);
    *out_producer_axis = (iree_async_axis_t)iree_atomic_load(
        &cache->producer_axis, iree_memory_order_relaxed);
    *out_epoch =
        (uint64_t)iree_atomic_load(&cache->epoch, iree_memory_order_relaxed);
    *out_value =
        (uint64_t)iree_atomic_load(&cache->value, iree_memory_order_relaxed);
    // Keep payload reads ahead of the closing sequence check. If any read
    // observed a concurrent writer this fence pairs with the writer's opening
    // release fence and the closing check must observe its odd sequence.
    iree_atomic_thread_fence(iree_memory_order_acquire);
    if (IREE_LIKELY(iree_atomic_load(&cache->sequence,
                                     iree_memory_order_relaxed) == sequence)) {
      return (*out_flags & IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID) != 0;
    }
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_SUBMITTED_SIGNAL_H_
