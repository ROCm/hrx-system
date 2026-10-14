// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/publisher.h"

static_assert(sizeof(iree_hal_amd_gpu_publisher_t) <= 64,
              "publisher metadata must fit within one cache line");

enum {
  IREE_HAL_AMD_GPU_PUBLISHER_IDLE = 0,
  IREE_HAL_AMD_GPU_PUBLISHER_CALLER,
  IREE_HAL_AMD_GPU_PUBLISHER_SERVICE,
};

void iree_hal_amd_gpu_publisher_initialize(
    iree_hal_amd_gpu_publisher_t* out_publisher) {
  memset(out_publisher, 0, sizeof(*out_publisher));
  iree_slim_mutex_initialize(&out_publisher->mutex);
}

void iree_hal_amd_gpu_publisher_deinitialize(
    iree_hal_amd_gpu_publisher_t* publisher) {
  IREE_ASSERT(publisher->owner == IREE_HAL_AMD_GPU_PUBLISHER_IDLE,
              "publication must be joined before destroying its owner");
  iree_slim_mutex_deinitialize(&publisher->mutex);
}

bool iree_hal_amd_gpu_publisher_enqueue(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry) {
  entry->next = NULL;
  iree_slim_mutex_lock(&publisher->mutex);
  const bool acquired = publisher->owner == IREE_HAL_AMD_GPU_PUBLISHER_IDLE;
  if (acquired) {
    publisher->owner = IREE_HAL_AMD_GPU_PUBLISHER_CALLER;
  } else {
    if (publisher->pending.tail) {
      publisher->pending.tail->next = entry;
    } else {
      publisher->pending.head = entry;
    }
    publisher->pending.tail = entry;
  }
  iree_slim_mutex_unlock(&publisher->mutex);
  return acquired;
}

bool iree_hal_amd_gpu_publisher_handoff(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_list_t remainder) {
  iree_slim_mutex_lock(&publisher->mutex);
  IREE_ASSERT(publisher->owner != IREE_HAL_AMD_GPU_PUBLISHER_IDLE,
              "handoff requires exclusive publication ownership");
  if (remainder.head) {
    remainder.tail->next = publisher->pending.head;
    if (!publisher->pending.tail) {
      publisher->pending.tail = remainder.tail;
    }
    publisher->pending.head = remainder.head;
  }
  const bool has_pending = publisher->pending.head != NULL;
  publisher->owner = has_pending ? IREE_HAL_AMD_GPU_PUBLISHER_SERVICE
                                 : IREE_HAL_AMD_GPU_PUBLISHER_IDLE;
  iree_slim_mutex_unlock(&publisher->mutex);
  return has_pending;
}

iree_hal_amd_gpu_publication_list_t iree_hal_amd_gpu_publisher_take(
    iree_hal_amd_gpu_publisher_t* publisher) {
  iree_slim_mutex_lock(&publisher->mutex);
  iree_hal_amd_gpu_publication_list_t work = {0};
  if (publisher->owner == IREE_HAL_AMD_GPU_PUBLISHER_SERVICE) {
    work = publisher->pending;
    publisher->pending = (iree_hal_amd_gpu_publication_list_t){0};
  }
  iree_slim_mutex_unlock(&publisher->mutex);
  return work;
}
