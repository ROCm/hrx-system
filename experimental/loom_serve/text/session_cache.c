// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/session_cache.h"

#include <string.h>

iree_status_t loom_serve_text_session_cache_initialize(
    iree_host_size_t capacity, loom_serve_text_session_cache_t* cache,
    iree_allocator_t allocator) {
  *cache = (loom_serve_text_session_cache_t){.allocator = allocator,
                                             .capacity = capacity};
  if (!capacity) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(allocator, capacity + 1,
                                     sizeof(*cache->entries),
                                     (void**)&cache->entries);
}

void loom_serve_text_session_cache_reset(loom_serve_text_session_cache_t* cache,
                                         loom_serve_text_telemetry_t* telemetry,
                                         iree_status_code_t terminal_code) {
  for (iree_host_size_t i = 0; cache->entries && i <= cache->capacity; ++i) {
    loom_serve_text_session_cache_entry_t* entry = &cache->entries[i];
    loom_serve_text_telemetry_session_end(telemetry, &entry->lifetime);
    if (terminal_code == IREE_STATUS_OK) {
      loom_serve_text_checkpoint_entry_release(&entry->endpoint);
    } else {
      loom_serve_text_chat_completion_deinitialize(&entry->endpoint.completion);
      memset(&entry->endpoint, 0, sizeof(entry->endpoint));
    }
  }
  cache->count = 0;
}

void loom_serve_text_session_cache_deinitialize(
    loom_serve_text_session_cache_t* cache) {
  iree_allocator_free(cache->allocator, cache->entries);
  memset(cache, 0, sizeof(*cache));
}

loom_serve_text_session_cache_entry_t* loom_serve_text_session_cache_find(
    loom_serve_text_session_cache_t* cache, iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       cache->entries && name.size && i <= cache->capacity; ++i) {
    if (iree_string_view_equal(
            name, iree_make_cstring_view(cache->entries[i].lifetime.name))) {
      return &cache->entries[i];
    }
  }
  return NULL;
}

loom_serve_text_session_cache_entry_t* loom_serve_text_session_cache_oldest(
    loom_serve_text_session_cache_t* cache,
    const loom_serve_text_session_cache_entry_t* protected_entry) {
  loom_serve_text_session_cache_entry_t* oldest = NULL;
  for (iree_host_size_t i = 0; cache->entries && i <= cache->capacity; ++i) {
    loom_serve_text_session_cache_entry_t* entry = &cache->entries[i];
    if (entry != protected_entry && entry->lifetime.id &&
        (!oldest || entry->lifetime.serial < oldest->lifetime.serial)) {
      oldest = entry;
    }
  }
  return oldest;
}

iree_status_t loom_serve_text_session_cache_submit(
    loom_serve_text_session_cache_t* cache, loom_serve_text_session_t* session,
    loom_serve_text_session_cache_entry_t** out_entry,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_entry = NULL;
  // A single admission owns the exchange slot; another capture cannot begin
  // until its incoming owner is consumed or the excess endpoint is evicted.
  IREE_ASSERT(cache->count <= cache->capacity);
  iree_host_size_t index = 0;
  while (cache->entries[index].endpoint.state) {
    ++index;
  }
  loom_serve_text_session_cache_entry_t* entry = &cache->entries[index];
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_entry_submit_pin(
      &entry->endpoint, session->row, &session->completion, out_frontier));
  // Service construction reserves disjoint cache/pin quotas. One exchange
  // record permits a full-cache handoff without stealing the incoming endpoint.
  IREE_ASSERT(entry->endpoint.state);
  ++cache->count;
  *out_entry = entry;
  return iree_ok_status();
}

void loom_serve_text_session_cache_publish(
    loom_serve_text_session_cache_entry_t* entry,
    loom_serve_text_session_t* session) {
  entry->lifetime = session->lifetime;
  memset(&session->lifetime, 0, sizeof(session->lifetime));
  loom_serve_text_chat_completion_deinitialize(&session->completion);
}

void loom_serve_text_session_cache_release(
    loom_serve_text_session_cache_t* cache,
    loom_serve_text_session_cache_entry_t* entry,
    loom_serve_text_telemetry_t* telemetry) {
  loom_serve_text_telemetry_session_end(telemetry, &entry->lifetime);
  loom_serve_text_checkpoint_entry_release(&entry->endpoint);
  --cache->count;
}

bool loom_serve_text_session_cache_evict(
    loom_serve_text_session_cache_t* cache, iree_host_size_t row_count,
    loom_serve_text_session_t* rows,
    const loom_serve_text_session_t* protected_row,
    const loom_serve_text_session_cache_entry_t* protected_entry,
    loom_serve_text_telemetry_t* telemetry) {
  loom_serve_text_session_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    loom_serve_text_session_t* row = &rows[i];
    if (row != protected_row && !row->request.connection &&
        loom_serve_text_row_position(row->row) &&
        (!oldest || row->lifetime.serial < oldest->lifetime.serial)) {
      oldest = row;
    }
  }
  loom_serve_text_session_cache_entry_t* cached =
      loom_serve_text_session_cache_oldest(cache, protected_entry);
  if (cached &&
      (!oldest || cached->lifetime.serial < oldest->lifetime.serial)) {
    loom_serve_text_session_cache_release(cache, cached, telemetry);
  } else if (oldest) {
    loom_serve_text_telemetry_session_end(telemetry, &oldest->lifetime);
    loom_serve_text_row_reset_retired(oldest->row);
    loom_serve_text_chat_completion_deinitialize(&oldest->completion);
  } else {
    return false;
  }
  return true;
}
