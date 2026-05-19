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

// Per-node entry in the pool.
struct iree_async_proactor_pool_entry_t {
  // Reference count for the pool-owned and consumer-owned entry references.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the entry and any proactor/runner resources it creates.
  iree_allocator_t allocator;

  // Options stored for deferred proactor/runner creation.
  iree_async_proactor_pool_options_t options;

  // NUMA node ID for this entry, or UINT32_MAX if unspecified.
  uint32_t node_id;

  // Proactor instance, created on first access (retained by the pool).
  iree_async_proactor_t* proactor;

  // Opaque poll runner handle, created alongside the proactor by the runner
  // factory. NULL if no runner factory is configured.
  void* runner;
};

struct iree_async_proactor_pool_t {
  // Reference count for the aggregate pool handle.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the pool and its entry table.
  iree_allocator_t allocator;

  // Mutex protecting lazy initialization of entries.
  iree_slim_mutex_t mutex;

  // Number of entries in the pool.
  iree_host_size_t count;

  // Pointers to ref-counted per-node entries.
  iree_async_proactor_pool_entry_t* entries[];
};

static void iree_async_proactor_pool_entry_destroy(
    iree_async_proactor_pool_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);

  if (entry->runner && entry->options.runner.request_stop) {
    void* runners[1] = {entry->runner};
    entry->options.runner.request_stop(entry->options.runner.user_data, runners,
                                       IREE_ARRAYSIZE(runners));
  }
  if (entry->runner && entry->options.runner.destroy) {
    entry->options.runner.destroy(entry->options.runner.user_data,
                                  entry->runner);
    entry->runner = NULL;
  }

  iree_async_proactor_release(entry->proactor);
  entry->proactor = NULL;

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
  if (IREE_LIKELY(entry) && iree_atomic_ref_count_dec(&entry->ref_count) == 1) {
    iree_async_proactor_pool_entry_destroy(entry);
  }
}

iree_async_proactor_t* iree_async_proactor_pool_entry_proactor(
    iree_async_proactor_pool_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);
  return entry->proactor;
}

uint32_t iree_async_proactor_pool_entry_node_id(
    const iree_async_proactor_pool_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(entry);
  return entry->node_id;
}

static void iree_async_proactor_pool_destroy(iree_async_proactor_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)pool->count);

  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    iree_async_proactor_pool_entry_release(pool->entries[i]);
    pool->entries[i] = NULL;
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

  // Allocate pool with trailing entry pointer array.
  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_async_proactor_pool_t), &total_size,
      IREE_STRUCT_FIELD(node_count, iree_async_proactor_pool_entry_t*,
                        /*out_offset=*/NULL));

  iree_async_proactor_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&pool);
  }
  if (iree_status_is_ok(status)) {
    memset(pool, 0, total_size);

    iree_atomic_ref_count_init(&pool->ref_count);
    pool->allocator = allocator;
    iree_slim_mutex_initialize(&pool->mutex);
    pool->count = node_count;
  }

  // Initialize entries. Proactors and runners are created on-demand when
  // pool_get, pool_acquire, or their _for_node variants first access an entry.
  for (iree_host_size_t i = 0; i < node_count && iree_status_is_ok(status);
       ++i) {
    iree_async_proactor_pool_entry_t* entry = NULL;
    status = iree_allocator_malloc(allocator, sizeof(*entry), (void**)&entry);
    if (iree_status_is_ok(status)) {
      memset(entry, 0, sizeof(*entry));
      iree_atomic_ref_count_init(&entry->ref_count);
      entry->allocator = allocator;
      entry->options = options;
      entry->node_id = node_ids ? node_ids[i] : UINT32_MAX;
      pool->entries[i] = entry;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else if (pool) {
    iree_async_proactor_pool_destroy(pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
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
  iree_async_proactor_pool_entry_t* entry = pool->entries[index];
  if (entry->proactor) return iree_ok_status();

  // The pool creates proactors here but polls from dedicated threads.
  iree_async_proactor_options_t proactor_options =
      entry->options.proactor_options;
  proactor_options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
  char name_buffer[32];
  if (iree_string_view_is_empty(proactor_options.debug_name)) {
    snprintf(name_buffer, sizeof(name_buffer), "proactor-%zu", index);
    proactor_options.debug_name =
        iree_make_string_view(name_buffer, strlen(name_buffer));
  }

  iree_async_proactor_pool_proactor_create_fn_t proactor_create =
      entry->options.proactor_create ? entry->options.proactor_create
                                     : iree_async_proactor_create_platform;
  IREE_RETURN_IF_ERROR(
      proactor_create(proactor_options, pool->allocator, &entry->proactor));

  // Create a poll runner if the factory is configured.
  if (entry->options.runner.create) {
    iree_status_t status = entry->options.runner.create(
        entry->options.runner.user_data, entry->proactor, entry->node_id,
        entry->allocator, &entry->runner);
    if (!iree_status_is_ok(status)) {
      iree_async_proactor_release(entry->proactor);
      entry->proactor = NULL;
      return status;
    }
  }

  return iree_ok_status();
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
    *out_entry = pool->entries[index];
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

  iree_async_proactor_pool_entry_t* entry = NULL;
  iree_status_t status = iree_async_proactor_pool_acquire(pool, index, &entry);
  if (iree_status_is_ok(status)) {
    *out_proactor = iree_async_proactor_pool_entry_proactor(entry);
    iree_async_proactor_pool_entry_release(entry);
  }
  return status;
}

uint32_t iree_async_proactor_pool_node_id(
    const iree_async_proactor_pool_t* pool, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(pool);
  if (IREE_UNLIKELY(index >= pool->count)) return UINT32_MAX;
  return pool->entries[index]->node_id;
}

iree_status_t iree_async_proactor_pool_acquire_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_pool_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  // Linear scan for the matching node. N is small (1-8 NUMA nodes).
  iree_host_size_t index = 0;  // Fallback to first entry if no match.
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    if (pool->entries[i]->node_id == node_id) {
      index = i;
      break;
    }
  }
  return iree_async_proactor_pool_acquire(pool, index, out_entry);
}

iree_status_t iree_async_proactor_pool_get_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_t** out_proactor) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  iree_async_proactor_pool_entry_t* entry = NULL;
  iree_status_t status =
      iree_async_proactor_pool_acquire_for_node(pool, node_id, &entry);
  if (iree_status_is_ok(status)) {
    *out_proactor = iree_async_proactor_pool_entry_proactor(entry);
    iree_async_proactor_pool_entry_release(entry);
  }
  return status;
}
