// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/tokenizer_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"

typedef struct loom_serve_tokenizer_cache_entry_t {
  // Retained immutable source establishing exact tokenizer identity.
  loomc_source_t* source;
  // Retained immutable tokenizer, or NULL while its producer is in flight.
  loom_serve_tokenizer_t* tokenizer;
  // Caller-owned subscribers detached atomically at terminal publication.
  loom_serve_tokenizer_subscription_t* subscriptions;
} loom_serve_tokenizer_cache_entry_t;

struct loom_serve_tokenizer_t {
  // Cache ownership and each active model/load retain independently.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy owning this reference and its immutable value.
  iree_allocator_t host_allocator;
  // Owned thread-safe tokenizer shared by independent encode/decode states.
  iree_tokenizer_t* value;
};

struct loom_serve_tokenizer_cache_t {
  // Allocation policy for the cache and shared tokenizer references.
  iree_allocator_t host_allocator;
  // Serializes lookup, publication, and trimming.
  iree_slim_mutex_t mutex;
  // Unordered exact-content entries; expected counts are small.
  loom_serve_tokenizer_cache_entry_t* entries;
  // Number of initialized entries, including in-flight producers.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
};

static iree_host_size_t loom_serve_tokenizer_cache_find(
    const loom_serve_tokenizer_cache_t* cache, const loomc_source_t* source) {
  const loomc_byte_span_t contents = loomc_source_contents(source);
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    const loomc_source_t* candidate = cache->entries[i].source;
    if (candidate == source) {
      return i;
    }
    const loomc_byte_span_t candidate_contents =
        loomc_source_contents(candidate);
    if (candidate_contents.data_length == contents.data_length &&
        (!contents.data_length || memcmp(candidate_contents.data, contents.data,
                                         (size_t)contents.data_length) == 0)) {
      return i;
    }
  }
  return cache->count;
}

static iree_status_t loom_serve_tokenizer_create(
    iree_tokenizer_t* value, iree_allocator_t host_allocator,
    loom_serve_tokenizer_t** out_tokenizer) {
  *out_tokenizer = NULL;
  loom_serve_tokenizer_t* tokenizer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*tokenizer),
                                             (void**)&tokenizer));
  iree_atomic_ref_count_init(&tokenizer->ref_count);
  tokenizer->host_allocator = host_allocator;
  tokenizer->value = value;
  *out_tokenizer = tokenizer;
  return iree_ok_status();
}

void loom_serve_tokenizer_retain(loom_serve_tokenizer_t* tokenizer) {
  if (tokenizer) {
    iree_atomic_ref_count_inc(&tokenizer->ref_count);
  }
}

void loom_serve_tokenizer_release(loom_serve_tokenizer_t* tokenizer) {
  if (!tokenizer) {
    return;
  }
  if (iree_atomic_ref_count_dec(&tokenizer->ref_count) != 1) {
    return;
  }
  iree_tokenizer_free(tokenizer->value);
  iree_allocator_free(tokenizer->host_allocator, tokenizer);
}

const iree_tokenizer_t* loom_serve_tokenizer_value(
    const loom_serve_tokenizer_t* tokenizer) {
  return tokenizer ? tokenizer->value : NULL;
}

iree_status_t loom_serve_tokenizer_cache_allocate(
    iree_allocator_t host_allocator, loom_serve_tokenizer_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_tokenizer_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

void loom_serve_tokenizer_cache_free(loom_serve_tokenizer_cache_t* cache) {
  if (!cache) {
    return;
  }
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    IREE_ASSERT(cache->entries[i].tokenizer);
    IREE_ASSERT(!cache->entries[i].subscriptions);
    loom_serve_tokenizer_release(cache->entries[i].tokenizer);
    loomc_source_release(cache->entries[i].source);
  }
  iree_allocator_free(cache->host_allocator, cache->entries);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
}

void loom_serve_tokenizer_cache_trim(loom_serve_tokenizer_cache_t* cache) {
  iree_slim_mutex_lock(&cache->mutex);
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    loom_serve_tokenizer_cache_entry_t entry = cache->entries[i];
    if (entry.tokenizer &&
        iree_atomic_ref_count_load(&entry.tokenizer->ref_count) == 1) {
      loom_serve_tokenizer_release(entry.tokenizer);
      loomc_source_release(entry.source);
    } else {
      cache->entries[count++] = entry;
    }
  }
  cache->count = count;
  iree_slim_mutex_unlock(&cache->mutex);
}

iree_status_t loom_serve_tokenizer_cache_acquire(
    loom_serve_tokenizer_cache_t* cache, loomc_source_t* source,
    loom_serve_tokenizer_subscription_t* subscription,
    loom_serve_tokenizer_acquisition_t* out_acquisition,
    loom_serve_tokenizer_t** out_tokenizer) {
  IREE_ASSERT(subscription && subscription->completion);
  *out_acquisition = LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER;
  *out_tokenizer = NULL;
  iree_slim_mutex_lock(&cache->mutex);
  const iree_host_size_t index = loom_serve_tokenizer_cache_find(cache, source);
  if (index < cache->count) {
    loom_serve_tokenizer_cache_entry_t* entry = &cache->entries[index];
    if (entry->tokenizer) {
      loom_serve_tokenizer_retain(entry->tokenizer);
      *out_tokenizer = entry->tokenizer;
      *out_acquisition = LOOM_SERVE_TOKENIZER_ACQUISITION_COMPLETE;
    } else {
      subscription->next = entry->subscriptions;
      entry->subscriptions = subscription;
      *out_acquisition = LOOM_SERVE_TOKENIZER_ACQUISITION_PENDING;
    }
    iree_slim_mutex_unlock(&cache->mutex);
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (cache->count == cache->capacity) {
    status = iree_allocator_grow_array(
        cache->host_allocator, cache->count + 1, sizeof(*cache->entries),
        &cache->capacity, (void**)&cache->entries);
  }
  if (iree_status_is_ok(status)) {
    loomc_source_retain(source);
    cache->entries[cache->count++] =
        (loom_serve_tokenizer_cache_entry_t){.source = source};
  }
  iree_slim_mutex_unlock(&cache->mutex);
  return status;
}

iree_status_t loom_serve_tokenizer_cache_complete(
    loom_serve_tokenizer_cache_t* cache, const loomc_source_t* source,
    iree_status_t status, iree_tokenizer_t* value,
    loom_serve_tokenizer_t** out_tokenizer) {
  *out_tokenizer = NULL;
  loom_serve_tokenizer_t* tokenizer = NULL;
  iree_status_t publication_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    IREE_ASSERT(value);
    publication_status =
        loom_serve_tokenizer_create(value, cache->host_allocator, &tokenizer);
    if (!iree_status_is_ok(publication_status)) {
      iree_tokenizer_free(value);
    }
  } else {
    iree_tokenizer_free(value);
  }
  const bool succeeded =
      iree_status_is_ok(status) && iree_status_is_ok(publication_status);

  loom_serve_tokenizer_subscription_t* subscriptions = NULL;
  iree_slim_mutex_lock(&cache->mutex);
  const iree_host_size_t index = loom_serve_tokenizer_cache_find(cache, source);
  IREE_ASSERT(index < cache->count);
  loom_serve_tokenizer_cache_entry_t* entry = &cache->entries[index];
  IREE_ASSERT(!entry->tokenizer);
  subscriptions = entry->subscriptions;
  if (succeeded) {
    entry->tokenizer = tokenizer;
    entry->subscriptions = NULL;
    loom_serve_tokenizer_retain(tokenizer);
    *out_tokenizer = tokenizer;
  } else {
    loomc_source_release(entry->source);
    memmove(entry, entry + 1, (--cache->count - index) * sizeof(*entry));
  }
  iree_slim_mutex_unlock(&cache->mutex);

  while (subscriptions) {
    loom_serve_tokenizer_subscription_t* subscription = subscriptions;
    subscriptions = subscription->next;
    if (tokenizer) {
      loom_serve_tokenizer_retain(tokenizer);
    }
    iree_status_t delivered_status = iree_status_join(
        iree_status_clone(status), iree_status_clone(publication_status));
    subscription->completion(subscription->user_data, delivered_status,
                             tokenizer);
  }
  return publication_status;
}
