// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"

#include <stdio.h>

#include "iree/async/proactor_platform.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

// The thread runner is available on platforms with C threading support.
// On wasm, the JS event loop drives proactors — no runner needed.
#if !IREE_PLATFORM_WASM
#include "iree/async/util/proactor_thread_runner.h"
#define IREE_ASYNC_PROACTOR_POOL_HAVE_RUNNER_THREAD 1
#endif  // !IREE_PLATFORM_WASM

//===----------------------------------------------------------------------===//
// iree_async_proactor_pool_options_default
//===----------------------------------------------------------------------===//

iree_async_proactor_pool_options_t iree_async_proactor_pool_options_default(
    void) {
  iree_async_proactor_pool_options_t options;
  memset(&options, 0, sizeof(options));
  options.proactor_options = iree_async_proactor_options_default();
#if IREE_ASYNC_PROACTOR_POOL_HAVE_RUNNER_THREAD
  options.runner = iree_async_proactor_pool_thread_runner_factory();
#endif  // IREE_ASYNC_PROACTOR_POOL_HAVE_RUNNER_THREAD
  return options;
}

//===----------------------------------------------------------------------===//
// iree_async_proactor_pool_t
//===----------------------------------------------------------------------===//

// Initialized proactor/runner pair that may outlive its aggregate pool.
struct iree_async_proactor_pool_entry_t {
  // Reference count for the pool-owned and consumer-owned entry references.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for this entry and its proactor/runner resources.
  iree_allocator_t allocator;

  // Runner callbacks and user data retained for independent teardown.
  iree_async_proactor_pool_runner_factory_t runner_factory;

  // NUMA node ID for this entry, or UINT32_MAX if unspecified.
  uint32_t node_id;

  // Proactor instance owned for the lifetime of this entry.
  iree_async_proactor_t* proactor;

  // Opaque poll runner handle, created alongside the proactor by the runner
  // factory. NULL if no runner factory is configured.
  void* runner;
};

// Lightweight per-node slot allocated inline with the aggregate pool.
typedef struct iree_async_proactor_pool_slot_t {
  // NUMA node ID for this slot, or UINT32_MAX if unspecified.
  uint32_t node_id;

  // Lazily allocated entry owned by the pool, or NULL until first access.
  iree_async_proactor_pool_entry_t* entry;
} iree_async_proactor_pool_slot_t;

struct iree_async_proactor_pool_t {
  // Reference count for aggregate pool ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the pool and lazy entries.
  iree_allocator_t allocator;

  // Options stored for deferred proactor/runner creation.
  iree_async_proactor_pool_options_t options;

  // Mutex protecting lazy initialization of entries.
  iree_slim_mutex_t mutex;

  // Number of slots in the pool.
  iree_host_size_t count;

  // Inline slots containing immutable topology and optional lazy entries.
  iree_async_proactor_pool_slot_t slots[];
};

// Claims final ownership of |entry| when the caller releases its reference.
static bool iree_async_proactor_pool_entry_release_claim(
    iree_async_proactor_pool_entry_t* entry) {
  return iree_atomic_ref_count_dec(&entry->ref_count) == 1;
}

// Requests the entry's runner to stop without waiting for it.
static void iree_async_proactor_pool_entry_request_stop(
    iree_async_proactor_pool_entry_t* entry) {
  if (entry->runner && entry->runner_factory.request_stop) {
    entry->runner_factory.request_stop(entry->runner_factory.user_data,
                                       entry->runner);
  }
}

// Destroys an entry after its runner has been requested to stop.
static void iree_async_proactor_pool_entry_destroy(
    iree_async_proactor_pool_entry_t* entry) {
  if (entry->runner && entry->runner_factory.destroy) {
    entry->runner_factory.destroy(entry->runner_factory.user_data,
                                  entry->runner);
  }
  iree_async_proactor_release(entry->proactor);

  iree_allocator_t allocator = entry->allocator;
  iree_allocator_free(allocator, entry);
}

void iree_async_proactor_pool_entry_retain(
    iree_async_proactor_pool_entry_t* entry) {
  if (IREE_LIKELY(entry)) {
    iree_atomic_ref_count_inc(&entry->ref_count);
  }
}

void iree_async_proactor_pool_entry_release(
    iree_async_proactor_pool_entry_t* entry) {
  if (IREE_LIKELY(entry) &&
      iree_async_proactor_pool_entry_release_claim(entry)) {
    iree_async_proactor_pool_entry_request_stop(entry);
    iree_async_proactor_pool_entry_destroy(entry);
  }
}

iree_async_proactor_t* iree_async_proactor_pool_entry_proactor(
    const iree_async_proactor_pool_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);
  return entry->proactor;
}

uint32_t iree_async_proactor_pool_entry_node_id(
    const iree_async_proactor_pool_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);
  return entry->node_id;
}

static void iree_async_proactor_pool_destroy(iree_async_proactor_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)pool->count);

  // Drop all pool-owned entry references and request every entry claimed for
  // destruction to stop before waiting for any runner. Consumer-retained
  // entries detach and perform their own teardown on final release.
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    iree_async_proactor_pool_entry_t* entry = pool->slots[i].entry;
    if (!entry) {
      continue;
    }
    if (iree_async_proactor_pool_entry_release_claim(entry)) {
      iree_async_proactor_pool_entry_request_stop(entry);
    } else {
      pool->slots[i].entry = NULL;
    }
  }

  // All claimed runners have received stop requests and can now be joined and
  // destroyed without serializing their stop latency.
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    iree_async_proactor_pool_entry_t* entry = pool->slots[i].entry;
    if (!entry) {
      continue;
    }
    iree_async_proactor_pool_entry_destroy(entry);
    pool->slots[i].entry = NULL;
  }

  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_t allocator = pool->allocator;
  iree_allocator_free(allocator, pool);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_proactor_pool_create(
    iree_host_size_t node_count, const uint32_t* node_ids,
    iree_async_proactor_pool_options_t options, iree_allocator_t allocator,
    iree_async_proactor_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)node_count);
  *out_pool = NULL;

  if (node_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node_count must be >= 1");
  }

  const iree_thread_affinity_set_t* processor_set =
      options.proactor_options.worker_processor_set;
  if (processor_set && iree_thread_affinity_set_is_empty(processor_set)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "processor budget contains no CPUs");
  }

  // Allocate the pool and its lightweight slot table together. Proactors,
  // runners, and independently retained entries remain lazily allocated.
  iree_host_size_t total_size = 0;
  iree_host_size_t processor_set_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          iree_sizeof_struct(iree_async_proactor_pool_t), &total_size,
          IREE_STRUCT_FIELD(node_count, iree_async_proactor_pool_slot_t,
                            /*out_offset=*/NULL),
          IREE_STRUCT_FIELD(processor_set ? 1 : 0, iree_thread_affinity_set_t,
                            &processor_set_offset)));
  iree_async_proactor_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&pool));
  memset(pool, 0, total_size);

  iree_atomic_ref_count_init(&pool->ref_count);
  pool->allocator = allocator;
  pool->options = options;
  if (processor_set) {
    iree_thread_affinity_set_t* retained_set =
        (iree_thread_affinity_set_t*)((uint8_t*)pool + processor_set_offset);
    *retained_set = *processor_set;
    pool->options.proactor_options.worker_processor_set = retained_set;
  }
  iree_slim_mutex_initialize(&pool->mutex);
  pool->count = node_count;

  // Initialize immutable slot topology. Entries, proactors, and runners are
  // created on-demand when a slot is first accessed.
  for (iree_host_size_t i = 0; i < node_count; ++i) {
    pool->slots[i].node_id = node_ids ? node_ids[i] : UINT32_MAX;
  }

  *out_pool = pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_proactor_pool_retain(iree_async_proactor_pool_t* pool) {
  if (IREE_LIKELY(pool)) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void iree_async_proactor_pool_release(iree_async_proactor_pool_t* pool) {
  if (IREE_LIKELY(pool) && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_async_proactor_pool_destroy(pool);
  }
}

iree_host_size_t iree_async_proactor_pool_count(
    const iree_async_proactor_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->count;
}

// Creates the proactor and runner for |entry| if not already initialized.
// Must be called with pool->mutex held.
static iree_status_t iree_async_proactor_pool_ensure_entry_locked(
    iree_async_proactor_pool_t* pool, iree_host_size_t index) {
  iree_async_proactor_pool_slot_t* slot = &pool->slots[index];
  if (slot->entry) {
    return iree_ok_status();
  }

  iree_async_proactor_pool_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(pool->allocator, sizeof(*entry), (void**)&entry));
  memset(entry, 0, sizeof(*entry));
  iree_atomic_ref_count_init(&entry->ref_count);
  entry->allocator = pool->allocator;
  entry->runner_factory = pool->options.runner;
  entry->node_id = slot->node_id;

  // The pool creates proactors here but polls from dedicated threads.
  iree_async_proactor_options_t proactor_options =
      pool->options.proactor_options;
  proactor_options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
  if (entry->node_id != UINT32_MAX) {
    iree_thread_affinity_set_group_any(entry->node_id,
                                       &proactor_options.worker_affinity);
  }
  char name_buffer[32];
  if (iree_string_view_is_empty(proactor_options.debug_name)) {
    snprintf(name_buffer, sizeof(name_buffer), "proactor-%zu", index);
    proactor_options.debug_name =
        iree_make_string_view(name_buffer, strlen(name_buffer));
  }

  iree_async_proactor_pool_proactor_create_fn_t proactor_create =
      pool->options.proactor_create ? pool->options.proactor_create
                                    : iree_async_proactor_create_platform;
  iree_status_t status =
      proactor_create(proactor_options, entry->allocator, &entry->proactor);

  // Create a poll runner if the factory is configured.
  if (iree_status_is_ok(status) && entry->runner_factory.create) {
    status = entry->runner_factory.create(entry->runner_factory.user_data,
                                          entry->proactor, entry->node_id,
                                          proactor_options.worker_processor_set,
                                          entry->allocator, &entry->runner);
  }

  if (iree_status_is_ok(status)) {
    slot->entry = entry;
  } else {
    iree_async_proactor_release(entry->proactor);
    iree_allocator_free(entry->allocator, entry);
  }
  return status;
}

iree_status_t iree_async_proactor_pool_acquire(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_pool_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  if (IREE_UNLIKELY(index >= pool->count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "proactor pool index %" PRIhsz
                            " out of range (pool has %" PRIhsz " entries)",
                            index, pool->count);
  }

  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status =
      iree_async_proactor_pool_ensure_entry_locked(pool, index);
  if (iree_status_is_ok(status)) {
    *out_entry = pool->slots[index].entry;
    iree_async_proactor_pool_entry_retain(*out_entry);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

iree_status_t iree_async_proactor_pool_get(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_t** out_proactor) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;
  if (IREE_UNLIKELY(index >= pool->count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "proactor pool index %" PRIhsz
                            " out of range (pool has %" PRIhsz " entries)",
                            index, pool->count);
  }
  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status =
      iree_async_proactor_pool_ensure_entry_locked(pool, index);
  if (iree_status_is_ok(status)) {
    *out_proactor = pool->slots[index].entry->proactor;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

uint32_t iree_async_proactor_pool_node_id(
    const iree_async_proactor_pool_t* pool, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(pool);
  if (IREE_UNLIKELY(index >= pool->count)) {
    return UINT32_MAX;
  }
  return pool->slots[index].node_id;
}

// Resolves an explicit node exactly. An unspecified request can use any slot.
// Slot topology is immutable after pool creation.
static iree_status_t iree_async_proactor_pool_resolve_node_index(
    const iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_host_size_t* out_index) {
  if (node_id == UINT32_MAX) {
    *out_index = 0;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    if (pool->slots[i].node_id == node_id) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "proactor pool has no entry for NUMA node %u",
                          node_id);
}

iree_status_t iree_async_proactor_pool_acquire_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_pool_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  iree_host_size_t index = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_pool_resolve_node_index(pool, node_id, &index));
  return iree_async_proactor_pool_acquire(pool, index, out_entry);
}

iree_status_t iree_async_proactor_pool_get_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_t** out_proactor) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;
  iree_host_size_t index = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_pool_resolve_node_index(pool, node_id, &index));
  return iree_async_proactor_pool_get(pool, index, out_proactor);
}
