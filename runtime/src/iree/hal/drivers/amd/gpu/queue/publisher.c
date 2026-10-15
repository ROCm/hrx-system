// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/publisher.h"

static_assert(sizeof(iree_hal_amd_gpu_publisher_t) <= 64,
              "publisher metadata must fit within one cache line");

enum {
  IREE_HAL_AMD_GPU_PUBLISHER_OWNED = 1u << 0,
  IREE_HAL_AMD_GPU_PUBLISHER_SERVICE = 1u << 1,
  IREE_HAL_AMD_GPU_PUBLISHER_NOTIFIED = 1u << 2,
};

void iree_hal_amd_gpu_publisher_initialize(
    iree_hal_amd_gpu_publisher_t* out_publisher) {
  memset(out_publisher, 0, sizeof(*out_publisher));
  out_publisher->head = &out_publisher->stub;
  iree_atomic_store(&out_publisher->tail, (intptr_t)&out_publisher->stub,
                    iree_memory_order_relaxed);
}

void iree_hal_amd_gpu_publisher_deinitialize(
    iree_hal_amd_gpu_publisher_t* publisher) {
  IREE_ASSERT(
      iree_atomic_load(&publisher->state, iree_memory_order_relaxed) == 0 &&
          publisher->head == &publisher->stub && !publisher->ready &&
          iree_atomic_load(&publisher->tail, iree_memory_order_relaxed) ==
              (intptr_t)&publisher->stub,
      "publication must be joined before destroying its owner");
}

// A producer owns its predecessor's link until the release-store. The
// consumer never detaches that predecessor across an incomplete link.
static void iree_hal_amd_gpu_publisher_append(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  iree_atomic_store(&entry->next, 0, iree_memory_order_relaxed);
  iree_hal_amd_gpu_publication_entry_t* previous =
      (iree_hal_amd_gpu_publication_entry_t*)iree_atomic_exchange(
          &publisher->tail, (intptr_t)entry, iree_memory_order_acq_rel);
  iree_atomic_store(&previous->next, (intptr_t)entry,
                    iree_memory_order_release);
}

// Only the exclusive publication owner detaches entries. The stub closes a
// final real entry so that it can retire independently of future enqueues.
static iree_hal_amd_gpu_publication_entry_t* iree_hal_amd_gpu_publisher_detach(
    iree_hal_amd_gpu_publisher_t* publisher) {
  iree_hal_amd_gpu_publication_entry_t* head = publisher->head;
  iree_hal_amd_gpu_publication_entry_t* next =
      (iree_hal_amd_gpu_publication_entry_t*)iree_atomic_load(
          &head->next, iree_memory_order_acquire);
  if (head == &publisher->stub) {
    if (!next) {
      return NULL;
    }
    publisher->head = head = next;
    next = (iree_hal_amd_gpu_publication_entry_t*)iree_atomic_load(
        &head->next, iree_memory_order_acquire);
  }
  if (next) {
    publisher->head = next;
    return head;
  }
  if ((intptr_t)head !=
      iree_atomic_load(&publisher->tail, iree_memory_order_acquire)) {
    return NULL;
  }
  iree_hal_amd_gpu_publisher_append(publisher, &publisher->stub);
  next = (iree_hal_amd_gpu_publication_entry_t*)iree_atomic_load(
      &head->next, iree_memory_order_acquire);
  if (!next) {
    return NULL;
  }
  publisher->head = next;
  return head;
}

bool iree_hal_amd_gpu_publisher_enqueue(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  int32_t expected = 0;
  if (iree_atomic_load(&publisher->state, iree_memory_order_relaxed) == 0 &&
      iree_atomic_compare_exchange_strong(
          &publisher->state, &expected, IREE_HAL_AMD_GPU_PUBLISHER_OWNED,
          iree_memory_order_acquire, iree_memory_order_relaxed)) {
    publisher->ready = entry;
    return true;
  }
  iree_hal_amd_gpu_publisher_append(publisher, entry);
  // No entry access follows linking: another owner may already have retired
  // it. Idle ownership transfers to this caller; a service handoff stays
  // reserved so capacity pressure does not make every caller retry publication.
  const int32_t previous = iree_atomic_fetch_or(
      &publisher->state,
      IREE_HAL_AMD_GPU_PUBLISHER_OWNED | IREE_HAL_AMD_GPU_PUBLISHER_NOTIFIED,
      iree_memory_order_acq_rel);
  return !(previous & (IREE_HAL_AMD_GPU_PUBLISHER_OWNED |
                       IREE_HAL_AMD_GPU_PUBLISHER_SERVICE));
}

bool iree_hal_amd_gpu_publisher_acquire_service(
    iree_hal_amd_gpu_publisher_t* publisher) {
  if (!(iree_atomic_load(&publisher->state, iree_memory_order_acquire) &
        IREE_HAL_AMD_GPU_PUBLISHER_SERVICE)) {
    return false;
  }
  // Only this service executor removes SERVICE. Producers may set notification
  // bits, but cannot acquire a reserved turn, so no retry loop is needed.
  iree_atomic_exchange(&publisher->state, IREE_HAL_AMD_GPU_PUBLISHER_OWNED,
                       iree_memory_order_acq_rel);
  return true;
}

iree_hal_amd_gpu_publication_entry_t* iree_hal_amd_gpu_publisher_next(
    iree_hal_amd_gpu_publisher_t* publisher) {
  iree_hal_amd_gpu_publication_entry_t* entry = publisher->ready;
  if (entry) {
    publisher->ready = NULL;
    return entry;
  }
  return iree_hal_amd_gpu_publisher_detach(publisher);
}

bool iree_hal_amd_gpu_publisher_handoff(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* remainder) {
  // Join notifications preceding this check. A link completed afterward either
  // defeats the idle CAS below or acquires ownership in its enqueue operation.
  iree_atomic_fetch_and(&publisher->state, ~IREE_HAL_AMD_GPU_PUBLISHER_NOTIFIED,
                        iree_memory_order_acq_rel);
  publisher->ready =
      remainder ? remainder : iree_hal_amd_gpu_publisher_detach(publisher);
  if (!publisher->ready) {
    int32_t expected = IREE_HAL_AMD_GPU_PUBLISHER_OWNED;
    if (iree_atomic_compare_exchange_strong(&publisher->state, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return false;
    }
  }
  iree_atomic_store(&publisher->state, IREE_HAL_AMD_GPU_PUBLISHER_SERVICE,
                    iree_memory_order_release);
  return true;
}

bool iree_hal_amd_gpu_publisher_is_idle(
    iree_hal_amd_gpu_publisher_t* publisher) {
  return iree_atomic_load(&publisher->state, iree_memory_order_acquire) == 0;
}
