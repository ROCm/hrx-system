// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/prefix_cache.h"

#include <string.h>

iree_status_t loom_serve_text_prefix_cache_initialize(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    loom_serve_text_prefix_cache_t* out_cache) {
  *out_cache = (loom_serve_text_prefix_cache_t){
      .host_allocator = host_allocator, .capacity = capacity};
  if (!capacity) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(host_allocator, capacity,
                                     sizeof(*out_cache->entries),
                                     (void**)&out_cache->entries);
}

void loom_serve_text_prefix_cache_reset(loom_serve_text_prefix_cache_t* cache,
                                        iree_status_code_t terminal_code) {
  for (iree_host_size_t i = 0; cache->entries && i < cache->capacity; ++i) {
    loom_serve_text_prefix_entry_t* entry = &cache->entries[i];
    if (terminal_code == IREE_STATUS_OK) {
      loom_serve_text_checkpoint_release(entry->state);
    }
    iree_allocator_free(cache->host_allocator, entry->tokens);
    loom_serve_text_chat_media_identity_deinitialize(&entry->media);
    memset(entry, 0, sizeof(*entry));
  }
  cache->count = 0;
}

void loom_serve_text_prefix_cache_deinitialize(
    loom_serve_text_prefix_cache_t* cache) {
  iree_allocator_free(cache->host_allocator, cache->entries);
  memset(cache, 0, sizeof(*cache));
}

loom_serve_text_prefix_entry_t* loom_serve_text_prefix_cache_find(
    loom_serve_text_prefix_cache_t* cache, const loom_serve_text_chat_t* chat,
    iree_host_size_t count, const int32_t* tokens) {
  loom_serve_text_prefix_entry_t* selected = NULL;
  for (iree_host_size_t i = 0; i < cache->capacity; ++i) {
    loom_serve_text_prefix_entry_t* entry = &cache->entries[i];
    if (entry->state && entry->count < count &&
        (!selected || entry->count > selected->count) &&
        loom_serve_text_chat_media_identity_equal_chat(&entry->media, chat) &&
        !memcmp(entry->tokens, tokens, entry->count * sizeof(*tokens))) {
      selected = entry;
    }
  }
  return selected;
}

void loom_serve_text_prefix_cache_touch(loom_serve_text_prefix_cache_t* cache,
                                        loom_serve_text_prefix_entry_t* entry) {
  entry->serial = ++cache->serial;
}

void loom_serve_text_prefix_cache_release(
    loom_serve_text_prefix_cache_t* cache,
    loom_serve_text_prefix_entry_t* entry) {
  loom_serve_text_checkpoint_release(entry->state);
  iree_allocator_free(cache->host_allocator, entry->tokens);
  loom_serve_text_chat_media_identity_deinitialize(&entry->media);
  memset(entry, 0, sizeof(*entry));
  --cache->count;
}

static loom_serve_text_prefix_entry_t* loom_serve_text_prefix_cache_oldest(
    loom_serve_text_prefix_cache_t* cache,
    const loom_serve_text_prefix_entry_t* protected_entry) {
  loom_serve_text_prefix_entry_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < cache->capacity; ++i) {
    loom_serve_text_prefix_entry_t* entry = &cache->entries[i];
    if (entry->state && entry != protected_entry &&
        (!oldest || entry->serial < oldest->serial)) {
      oldest = entry;
    }
  }
  return oldest;
}

bool loom_serve_text_prefix_cache_evict(
    loom_serve_text_prefix_cache_t* cache,
    const loom_serve_text_prefix_entry_t* protected_entry) {
  loom_serve_text_prefix_entry_t* entry =
      loom_serve_text_prefix_cache_oldest(cache, protected_entry);
  if (!entry) {
    return false;
  }
  loom_serve_text_prefix_cache_release(cache, entry);
  return true;
}

iree_status_t loom_serve_text_prefix_cache_submit(
    loom_serve_text_prefix_cache_t* cache, const loom_serve_text_chat_t* chat,
    loom_serve_text_row_t* row, iree_host_size_t count, const int32_t* tokens,
    const loom_serve_text_prefix_entry_t* protected_entry, bool* out_evicted,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_evicted = false;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  // Include equality when checking ancestors; admission itself is strict.
  if (loom_serve_text_prefix_cache_find(cache, chat, count + 1, tokens)) {
    return iree_ok_status();
  }
  loom_serve_text_prefix_entry_t* entry = NULL;
  for (iree_host_size_t i = 0; i < cache->capacity && !entry; ++i) {
    if (!cache->entries[i].state) {
      entry = &cache->entries[i];
    }
  }
  if (!entry) {
    entry = loom_serve_text_prefix_cache_oldest(cache, protected_entry);
  }
  if (!entry) {
    return iree_ok_status();
  }
  int32_t* copy = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      cache->host_allocator,
      iree_make_const_byte_span(tokens, count * sizeof(*tokens)),
      (void**)&copy));
  loom_serve_text_chat_media_identity_t media = {0};
  iree_status_t status = loom_serve_text_chat_media_identity_initialize(
      chat, cache->host_allocator, &media);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(cache->host_allocator, copy);
    return status;
  }
  if (entry->state) {
    loom_serve_text_prefix_cache_release(cache, entry);
    *out_evicted = true;
  }
  status = loom_serve_text_row_submit_pin(row, &entry->state, out_frontier);
  if (iree_status_is_ok(status) && entry->state) {
    entry->tokens = copy;
    entry->count = count;
    entry->media = media;
    loom_serve_text_prefix_cache_touch(cache, entry);
    ++cache->count;
  } else {
    iree_allocator_free(cache->host_allocator, copy);
    loom_serve_text_chat_media_identity_deinitialize(&media);
  }
  return status;
}
