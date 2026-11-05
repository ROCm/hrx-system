// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/variant_pool.h"

#include "iree/base/threading/mutex.h"

struct loom_runtime_variant_entry_t {
  // Pool owning this stable entry through all producer and consumer borrows.
  loom_runtime_variant_pool_t* pool;
  // Next cached entry in owner-selected traversal order.
  struct loom_runtime_variant_entry_t* next;
  // Previous cached entry, allowing eviction without another identity search.
  struct loom_runtime_variant_entry_t* previous;
  // Process-lifetime identity, independent of a reused allocation address.
  loom_runtime_variant_id_t id;
  // Construction provenance retained even if that cached base is later removed.
  loom_runtime_variant_id_t base;
  // Owner-published readiness; compiler callbacks never change this field.
  loom_runtime_variant_phase_t phase;
  // Owner-thread consumer borrows protecting a ready cached payload.
  iree_host_size_t users;
  // Result publication protected by pool->mutex until collection. Immutable
  // terminal evidence thereafter.
  struct {
    // Producer has relinquished its entry and model borrows.
    bool ready;
    // Owned terminal status, retained for observation through removal.
    iree_status_t status;
    // Owned compiled payload, disposed immediately when construction fails.
    void* value;
    // Copied evidence, independent of compiler/source storage.
    loom_runtime_compilation_report_t report;
  } completion;
};

struct loom_runtime_variant_pool_t {
  // Allocator owning this pool and its individually stable entries.
  iree_allocator_t host_allocator;
  // Modality payload teardown; the enclosing model outlives all calls.
  loom_runtime_variant_dispose_fn_t dispose;
  // Protects completion publication and wake lifetime, never model execution.
  iree_slim_mutex_t mutex;
  // Nonblocking owner scheduling target, protected by mutex.
  loom_runtime_wake_callback_t observer;
  // First retained entry in owner-selected traversal order.
  loom_runtime_variant_entry_t* head;
  // Last cached entry; appending never moves existing producer payloads.
  loom_runtime_variant_entry_t* tail;
  // Retained entries, including the original and pending/failed attempts.
  iree_host_size_t count;
  // Maximum simultaneous retained entries.
  iree_host_size_t capacity;
  // Accepted producers not yet collected by the owner.
  iree_host_size_t pending;
};

static iree_atomic_uint64_t loom_runtime_variant_next_id =
    IREE_ATOMIC_VAR_INIT(1);

static loom_runtime_variant_entry_t* loom_runtime_variant_pool_find(
    const loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t id) {
  for (loom_runtime_variant_entry_t* entry = pool->head; entry;
       entry = entry->next) {
    if (entry->id == id) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_runtime_variant_pool_append(
    loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t base,
    loom_runtime_variant_entry_t** out_entry) {
  loom_runtime_variant_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*entry), (void**)&entry));
  entry->pool = pool;
  entry->base = base;
  entry->previous = pool->tail;
  entry->id = iree_atomic_fetch_add(&loom_runtime_variant_next_id, 1,
                                    iree_memory_order_relaxed);
  if (pool->tail) {
    pool->tail->next = entry;
  } else {
    pool->head = entry;
  }
  pool->tail = entry;
  ++pool->count;
  *out_entry = entry;
  return iree_ok_status();
}

iree_status_t loom_runtime_variant_pool_allocate(
    iree_host_size_t capacity, void* original_value,
    loom_runtime_variant_dispose_fn_t dispose, iree_allocator_t host_allocator,
    loom_runtime_variant_entry_t** out_original,
    loom_runtime_variant_pool_t** out_pool) {
  *out_original = NULL;
  *out_pool = NULL;
  if (!capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant capacity must include the original");
  }
  loom_runtime_variant_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  pool->host_allocator = host_allocator;
  pool->dispose = dispose;
  pool->capacity = capacity;
  iree_slim_mutex_initialize(&pool->mutex);
  loom_runtime_variant_entry_t* entry = NULL;
  iree_status_t status = loom_runtime_variant_pool_append(pool, 0, &entry);
  if (iree_status_is_ok(status)) {
    entry->phase = LOOM_RUNTIME_VARIANT_READY;
    entry->completion.value = original_value;
    *out_original = entry;
    *out_pool = pool;
  } else {
    loom_runtime_variant_pool_free(pool);
  }
  return status;
}

void loom_runtime_variant_pool_free(loom_runtime_variant_pool_t* pool) {
  if (!pool) {
    return;
  }
  IREE_ASSERT(!pool->pending);
  loom_runtime_variant_entry_t* entry = pool->head;
  while (entry) {
    loom_runtime_variant_entry_t* next = entry->next;
    IREE_ASSERT(!entry->users);
    pool->dispose(entry->completion.value);
    iree_status_free(entry->completion.status);
    iree_allocator_free(pool->host_allocator, entry);
    entry = next;
  }
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_free(pool->host_allocator, pool);
}

void loom_runtime_variant_pool_set_observer(
    loom_runtime_variant_pool_t* pool, loom_runtime_wake_callback_t observer) {
  iree_slim_mutex_lock(&pool->mutex);
  pool->observer = observer;
  iree_slim_mutex_unlock(&pool->mutex);
}

iree_host_size_t loom_runtime_variant_pool_count(
    const loom_runtime_variant_pool_t* pool) {
  return pool->count;
}

iree_host_size_t loom_runtime_variant_pool_storage_size(
    const loom_runtime_variant_pool_t* pool) {
  return sizeof(*pool) + pool->count * sizeof(*pool->head);
}

loom_runtime_variant_entry_t* loom_runtime_variant_pool_first(
    const loom_runtime_variant_pool_t* pool) {
  return pool->head;
}

loom_runtime_variant_entry_t* loom_runtime_variant_entry_next(
    const loom_runtime_variant_entry_t* entry) {
  return entry->next;
}

void loom_runtime_variant_entry_touch(loom_runtime_variant_entry_t* entry) {
  loom_runtime_variant_pool_t* pool = entry->pool;
  if (pool->tail == entry) {
    return;
  }
  if (entry->previous) {
    entry->previous->next = entry->next;
  } else {
    pool->head = entry->next;
  }
  entry->next->previous = entry->previous;
  entry->previous = pool->tail;
  entry->next = NULL;
  pool->tail->next = entry;
  pool->tail = entry;
}

iree_status_t loom_runtime_variant_entry_status(
    const loom_runtime_variant_entry_t* entry) {
  return iree_status_clone(entry->completion.status);
}

bool loom_runtime_variant_pool_pending(
    const loom_runtime_variant_pool_t* pool) {
  return pool->pending != 0;
}

iree_status_t loom_runtime_variant_pool_begin(
    loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t base,
    loom_runtime_variant_entry_t** out_entry) {
  *out_entry = NULL;
  const loom_runtime_variant_entry_t* original =
      loom_runtime_variant_pool_find(pool, base);
  if (!original || original->phase != LOOM_RUNTIME_VARIANT_READY) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "base variant is not ready in this model pool");
  }
  if (pool->count == pool->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "variant pool is full; remove an unused choice");
  }
  IREE_RETURN_IF_ERROR(loom_runtime_variant_pool_append(pool, base, out_entry));
  ++pool->pending;
  return iree_ok_status();
}

void loom_runtime_variant_entry_complete(
    loom_runtime_variant_entry_t* entry, iree_status_t status, void* value,
    const loom_runtime_compilation_report_t* report) {
  loom_runtime_variant_pool_t* pool = entry->pool;
  iree_slim_mutex_lock(&pool->mutex);
  entry->completion.status = status;
  entry->completion.value = value;
  if (report) {
    entry->completion.report = *report;
  }
  entry->completion.ready = true;
  if (pool->observer.wake) {
    pool->observer.wake(pool->observer.user_data);
  }
  iree_slim_mutex_unlock(&pool->mutex);
}

bool loom_runtime_variant_pool_collect(loom_runtime_variant_pool_t* pool) {
  if (!pool->pending) {
    return false;
  }
  bool progress = false;
  for (loom_runtime_variant_entry_t* entry = pool->head; entry;
       entry = entry->next) {
    if (entry->phase != LOOM_RUNTIME_VARIANT_COMPILING) {
      continue;
    }
    iree_slim_mutex_lock(&pool->mutex);
    const bool ready = entry->completion.ready;
    iree_slim_mutex_unlock(&pool->mutex);
    if (!ready) {
      continue;
    }
    --pool->pending;
    if (iree_status_is_ok(entry->completion.status)) {
      entry->phase = LOOM_RUNTIME_VARIANT_READY;
    } else {
      entry->phase = LOOM_RUNTIME_VARIANT_FAILED;
      pool->dispose(entry->completion.value);
      entry->completion.value = NULL;
    }
    progress = true;
  }
  return progress;
}

iree_status_t loom_runtime_variant_pool_acquire(
    loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t id,
    loom_runtime_variant_entry_t** out_entry) {
  *out_entry = NULL;
  loom_runtime_variant_entry_t* entry =
      loom_runtime_variant_pool_find(pool, id);
  if (!entry) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "variant does not belong to this model pool");
  }
  if (entry->phase != LOOM_RUNTIME_VARIANT_READY) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "variant is not ready");
  }
  ++entry->users;
  *out_entry = entry;
  return iree_ok_status();
}

void loom_runtime_variant_entry_release(loom_runtime_variant_entry_t* entry) {
  if (entry) {
    --entry->users;
  }
}

void loom_runtime_variant_entry_retain(loom_runtime_variant_entry_t* entry) {
  if (entry) {
    ++entry->users;
  }
}

loom_runtime_variant_id_t loom_runtime_variant_entry_id(
    const loom_runtime_variant_entry_t* entry) {
  return entry->id;
}

void* loom_runtime_variant_entry_value(
    const loom_runtime_variant_entry_t* entry) {
  return entry->completion.value;
}

iree_status_t loom_runtime_variant_pool_remove(
    loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t id) {
  loom_runtime_variant_entry_t* entry =
      loom_runtime_variant_pool_find(pool, id);
  if (!entry) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "variant does not belong to this model pool");
  }
  if (entry->users || entry->phase == LOOM_RUNTIME_VARIANT_COMPILING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "variant still owns a producer or consumer");
  }
  loom_runtime_variant_entry_discard(entry);
  return iree_ok_status();
}

void loom_runtime_variant_entry_discard(loom_runtime_variant_entry_t* entry) {
  loom_runtime_variant_pool_t* pool = entry->pool;
  IREE_ASSERT(!entry->users && entry->phase != LOOM_RUNTIME_VARIANT_COMPILING);
  if (entry->previous) {
    entry->previous->next = entry->next;
  } else {
    pool->head = entry->next;
  }
  if (entry->next) {
    entry->next->previous = entry->previous;
  } else {
    pool->tail = entry->previous;
  }
  --pool->count;
  pool->dispose(entry->completion.value);
  iree_status_free(entry->completion.status);
  iree_allocator_free(pool->host_allocator, entry);
}

loom_runtime_variant_snapshot_t loom_runtime_variant_entry_snapshot(
    const loom_runtime_variant_entry_t* entry) {
  const bool terminal = entry->phase != LOOM_RUNTIME_VARIANT_COMPILING;
  return (loom_runtime_variant_snapshot_t){
      .id = entry->id,
      .base = entry->base,
      .phase = entry->phase,
      .users = entry->users,
      .code = terminal ? iree_status_code(entry->completion.status)
                       : IREE_STATUS_OK,
      .message = terminal ? iree_status_message(entry->completion.status)
                          : iree_string_view_empty(),
      .compilation = terminal ? &entry->completion.report : NULL};
}

bool loom_runtime_variant_pool_snapshot(
    const loom_runtime_variant_pool_t* pool, loom_runtime_variant_id_t id,
    loom_runtime_variant_snapshot_t* out_snapshot) {
  const loom_runtime_variant_entry_t* entry =
      loom_runtime_variant_pool_find(pool, id);
  if (!entry) {
    return false;
  }
  *out_snapshot = loom_runtime_variant_entry_snapshot(entry);
  return true;
}

bool loom_runtime_variant_pool_at(
    const loom_runtime_variant_pool_t* pool, iree_host_size_t index,
    loom_runtime_variant_snapshot_t* out_snapshot) {
  const loom_runtime_variant_entry_t* entry = pool->head;
  for (iree_host_size_t i = 0; entry && i < index; ++i) {
    entry = entry->next;
  }
  if (!entry) {
    return false;
  }
  *out_snapshot = loom_runtime_variant_entry_snapshot(entry);
  return true;
}
