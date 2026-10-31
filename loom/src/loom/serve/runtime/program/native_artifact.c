// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/native_artifact.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/util/digest.h"

// Both indexes compare full byte spans. Hashes select buckets, never identity.
typedef struct loom_serve_native_index_entry_t {
  // Next collision-chain entry, or NULL.
  struct loom_serve_native_index_entry_t* next;
  // Immutable owned-key or emitted-image span.
  iree_const_byte_span_t key;
  // Retained table hash so growth does not rescan source or image bytes.
  uint64_t hash;
} loom_serve_native_index_entry_t;

typedef struct loom_serve_native_index_t {
  // Non-owning collision-chain heads, protected by the domain mutex.
  loom_serve_native_index_entry_t** buckets;
  // Power-of-two bucket capacity, zero before the first insertion.
  iree_host_size_t capacity;
  // Number of indexed entries; failed outcomes are removed before fan-out.
  iree_host_size_t count;
} loom_serve_native_index_t;

typedef struct loom_serve_native_image_t {
  // First-member exact byte index; the key borrows sequence storage below.
  loom_serve_native_index_entry_t index;
  // Number of immutable artifact results retaining these bytes.
  uint32_t references;
  // Borrowed from those results; each retains its own sequence reference.
  iree_byte_sequence_t* sequence;
} loom_serve_native_image_t;

struct loom_serve_native_cache_t {
  // Host allocation policy, outliving every artifact and loaded image.
  iree_allocator_t host_allocator;
  // Serializes domain acquisition and final removal, not native lookup.
  iree_slim_mutex_t mutex;
  // Non-owning domains retained by catalogs and their native artifacts.
  loom_serve_native_domain_t* domains;
};

struct loom_serve_native_domain_t {
  // Catalog and artifact ownership; final release removes the weak index.
  iree_atomic_ref_count_t ref_count;
  // Borrowed host cache, outliving all dependent catalogs and work.
  loom_serve_native_cache_t* cache;
  // Next exact profile/options domain, protected by the cache mutex.
  loom_serve_native_domain_t* next;
  // Complete immutable compilation policy; retains its profile.
  loom_serve_native_domain_options_t options;
  // Serializes artifact and image indexing, references, and publication.
  iree_slim_mutex_t mutex;
  // Weak exact-input index, independent of device loading.
  loom_serve_native_index_t artifacts;
  // Weak emitted-image index, independent of input roots and launch facts.
  loom_serve_native_index_t images;
};

struct loom_serve_native_artifact_t {
  // First-member exact input index; owns its framed key storage.
  loom_serve_native_index_entry_t index;
  // Retained compilation domain, including during callback-side teardown.
  loom_serve_native_domain_t* domain;
  // Consumer and producer references, protected by the domain mutex.
  uint32_t references;
  // Allocated key storage, including builder capacity and terminator.
  iree_host_size_t key_capacity;
  // Pending device-local continuations, detached before callbacks.
  loom_serve_native_subscription_t* waiters;
  // Interned emitted bytes, or NULL before successful publication.
  loom_serve_native_image_t* image;
  // Immutable result has been published under the domain mutex.
  bool complete;
  // The weak input index still includes this entry.
  bool indexed;
  // Producer-owned until publication, then immutable for all consumers.
  loom_serve_native_result_t result;
  // Exact root-order views, with each name allocated by the producer.
  iree_string_view_t roots[];
};

static uint64_t loom_serve_native_hash(iree_const_byte_span_t key) {
  loom_sha256_t digest;
  loom_sha256_compute(key, &digest);
  return iree_unaligned_load_le_u64(digest.bytes);
}

static loom_serve_native_index_entry_t* loom_serve_native_index_find(
    const loom_serve_native_index_t* index, uint64_t hash,
    iree_const_byte_span_t key) {
  if (!index->capacity) {
    return NULL;
  }
  for (loom_serve_native_index_entry_t* entry =
           index->buckets[hash & (index->capacity - 1)];
       entry; entry = entry->next) {
    if (entry->hash == hash && entry->key.data_length == key.data_length &&
        memcmp(entry->key.data, key.data, key.data_length) == 0) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_serve_native_index_reserve(
    loom_serve_native_index_t* index, iree_allocator_t allocator) {
  if (index->count < index->capacity) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = 0;
  if (!iree_host_size_checked_mul(index->capacity ? index->capacity : 16, 2,
                                  &capacity)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "native artifact index exceeds addressable storage");
  }
  loom_serve_native_index_entry_t** buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, capacity, sizeof(*buckets), (void**)&buckets));
  for (iree_host_size_t i = 0; i < index->capacity; ++i) {
    loom_serve_native_index_entry_t* entry = index->buckets[i];
    while (entry) {
      loom_serve_native_index_entry_t* next = entry->next;
      const iree_host_size_t bucket = entry->hash & (capacity - 1);
      entry->next = buckets[bucket];
      buckets[bucket] = entry;
      entry = next;
    }
  }
  iree_allocator_free(allocator, index->buckets);
  index->buckets = buckets;
  index->capacity = capacity;
  return iree_ok_status();
}

static void loom_serve_native_index_insert(
    loom_serve_native_index_t* index, loom_serve_native_index_entry_t* entry) {
  const iree_host_size_t bucket = entry->hash & (index->capacity - 1);
  entry->next = index->buckets[bucket];
  index->buckets[bucket] = entry;
  ++index->count;
}

static void loom_serve_native_index_remove(
    loom_serve_native_index_t* index, loom_serve_native_index_entry_t* entry) {
  loom_serve_native_index_entry_t** link =
      &index->buckets[entry->hash & (index->capacity - 1)];
  while (*link != entry) {
    link = &(*link)->next;
  }
  *link = entry->next;
  --index->count;
}

iree_status_t loom_serve_native_cache_allocate(
    iree_allocator_t host_allocator, loom_serve_native_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_native_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

void loom_serve_native_cache_free(loom_serve_native_cache_t* cache) {
  if (!cache) {
    return;
  }
  IREE_ASSERT(cache->domains == NULL);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
}

iree_allocator_t loom_serve_native_cache_allocator(
    const loom_serve_native_cache_t* cache) {
  return cache->host_allocator;
}

static bool loom_serve_native_domain_equal(
    const loom_serve_native_domain_options_t* lhs,
    const loom_serve_native_domain_options_t* rhs) {
  return lhs->control_flow_lowering == rhs->control_flow_lowering &&
         lhs->sanitizer_checks == rhs->sanitizer_checks &&
         lhs->sanitizer_flags == rhs->sanitizer_flags &&
         lhs->sanitizer_reporting_mode == rhs->sanitizer_reporting_mode &&
         iree_string_view_equal(lhs->artifact_format, rhs->artifact_format) &&
         loomc_target_profile_equal(lhs->profile, rhs->profile);
}

iree_status_t loom_serve_native_domain_acquire(
    loom_serve_native_cache_t* cache,
    const loom_serve_native_domain_options_t* options,
    loom_serve_native_domain_t** out_domain) {
  *out_domain = NULL;
  iree_slim_mutex_lock(&cache->mutex);
  loom_serve_native_domain_t* domain = cache->domains;
  while (domain && !loom_serve_native_domain_equal(&domain->options, options)) {
    domain = domain->next;
  }
  iree_status_t status = iree_ok_status();
  if (domain) {
    iree_atomic_ref_count_inc(&domain->ref_count);
  } else {
    status = iree_allocator_malloc(cache->host_allocator, sizeof(*domain),
                                   (void**)&domain);
    if (iree_status_is_ok(status)) {
      iree_atomic_ref_count_init(&domain->ref_count);
      domain->cache = cache;
      domain->options = *options;
      loomc_target_profile_retain(domain->options.profile);
      iree_slim_mutex_initialize(&domain->mutex);
      domain->next = cache->domains;
      cache->domains = domain;
    }
  }
  *out_domain = domain;
  iree_slim_mutex_unlock(&cache->mutex);
  return status;
}

void loom_serve_native_domain_release(loom_serve_native_domain_t* domain) {
  if (!domain) {
    return;
  }
  loom_serve_native_cache_t* cache = domain->cache;
  iree_slim_mutex_lock(&cache->mutex);
  const bool last = iree_atomic_ref_count_dec(&domain->ref_count) == 1;
  if (last) {
    loom_serve_native_domain_t** link = &cache->domains;
    while (*link != domain) {
      link = &(*link)->next;
    }
    *link = domain->next;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (last) {
    iree_allocator_free(cache->host_allocator, domain->artifacts.buckets);
    iree_allocator_free(cache->host_allocator, domain->images.buckets);
    loomc_target_profile_release(domain->options.profile);
    iree_slim_mutex_deinitialize(&domain->mutex);
    iree_allocator_free(cache->host_allocator, domain);
  }
}

iree_allocator_t loom_serve_native_domain_allocator(
    const loom_serve_native_domain_t* domain) {
  return domain->cache->host_allocator;
}

iree_status_t loom_serve_native_artifact_acquire(
    loom_serve_native_domain_t* domain, iree_string_builder_t* key,
    iree_host_size_t root_count, loom_serve_native_subscription_t* subscription,
    loom_serve_native_artifact_t** out_artifact, bool* out_created,
    bool* out_ready) {
  *out_artifact = NULL;
  *out_created = false;
  *out_ready = false;
  const iree_string_view_t view = iree_string_builder_view(key);
  const iree_const_byte_span_t bytes =
      iree_make_const_byte_span(view.data, view.size);
  const uint64_t hash = loom_serve_native_hash(bytes);
  const iree_allocator_t allocator = domain->cache->host_allocator;
  iree_slim_mutex_lock(&domain->mutex);
  loom_serve_native_artifact_t* artifact =
      (loom_serve_native_artifact_t*)loom_serve_native_index_find(
          &domain->artifacts, hash, bytes);
  iree_status_t status = iree_ok_status();
  if (artifact) {
    ++artifact->references;
  } else {
    status = loom_serve_native_index_reserve(&domain->artifacts, allocator);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_struct_array(
          allocator, sizeof(*artifact), root_count, sizeof(*artifact->roots),
          (void**)&artifact);
    }
    if (iree_status_is_ok(status)) {
      artifact->domain = domain;
      iree_atomic_ref_count_inc(&domain->ref_count);
      artifact->references = 2;  // Consumer and pending producer.
      artifact->key_capacity = iree_string_builder_capacity(key);
      artifact->index.key = iree_make_const_byte_span(
          iree_string_builder_take_storage(key), bytes.data_length);
      artifact->index.hash = hash;
      artifact->indexed = true;
      artifact->result.root_count = root_count;
      artifact->result.roots = artifact->roots;
      loom_serve_native_index_insert(&domain->artifacts, &artifact->index);
      *out_created = true;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_artifact = artifact;
    *out_ready = artifact->complete;
    if (!artifact->complete) {
      subscription->next = artifact->waiters;
      artifact->waiters = subscription;
    }
  }
  iree_slim_mutex_unlock(&domain->mutex);
  return status;
}

void loom_serve_native_artifact_retain(loom_serve_native_artifact_t* artifact) {
  iree_slim_mutex_lock(&artifact->domain->mutex);
  ++artifact->references;
  iree_slim_mutex_unlock(&artifact->domain->mutex);
}

void loom_serve_native_artifact_release(
    loom_serve_native_artifact_t* artifact) {
  if (!artifact) {
    return;
  }
  loom_serve_native_domain_t* domain = artifact->domain;
  const iree_allocator_t allocator = domain->cache->host_allocator;
  loom_serve_native_image_t* retired_image = NULL;
  iree_slim_mutex_lock(&domain->mutex);
  const bool last = --artifact->references == 0;
  if (last) {
    if (artifact->indexed) {
      loom_serve_native_index_remove(&domain->artifacts, &artifact->index);
    }
    if (artifact->image && --artifact->image->references == 0) {
      retired_image = artifact->image;
      loom_serve_native_index_remove(&domain->images, &retired_image->index);
    }
  }
  iree_slim_mutex_unlock(&domain->mutex);
  if (last) {
    iree_allocator_free(allocator, retired_image);
    for (iree_host_size_t i = 0; i < artifact->result.root_count; ++i) {
      iree_allocator_free(allocator, (void*)artifact->roots[i].data);
    }
    iree_status_free(artifact->result.status);
    iree_allocator_free(allocator, artifact->result.diagnostics);
    iree_byte_sequence_release(artifact->result.image);
    iree_allocator_free(allocator, (void*)artifact->index.key.data);
    iree_allocator_free(allocator, artifact);
    loom_serve_native_domain_release(domain);
  }
}

loom_serve_native_result_t* loom_serve_native_artifact_result(
    loom_serve_native_artifact_t* artifact) {
  return &artifact->result;
}

iree_const_byte_span_t loom_serve_native_artifact_key(
    const loom_serve_native_artifact_t* artifact) {
  return artifact->index.key;
}

// Coalesce once before all device loads and retain equal output bytes once,
// even when distinct input specializations emit the same image.
static iree_status_t loom_serve_native_artifact_prepare_image(
    loom_serve_native_artifact_t* artifact) {
  loom_serve_native_domain_t* domain = artifact->domain;
  const iree_allocator_t allocator = domain->cache->host_allocator;
  loom_serve_native_result_t* result = &artifact->result;
  iree_const_byte_span_t data = iree_const_byte_span_empty();
  if (!iree_byte_sequence_try_get_contiguous_span(result->image, &data)) {
    iree_byte_span_t bytes = iree_byte_span_empty();
    IREE_RETURN_IF_ERROR(
        iree_byte_sequence_clone(result->image, allocator, &bytes));
    data = iree_make_const_byte_span(bytes.data, bytes.data_length);
    iree_byte_sequence_t* image = NULL;
    iree_status_t status =
        iree_byte_sequence_create_from_span_move(&bytes, allocator, &image);
    iree_allocator_free(allocator, bytes.data);
    IREE_RETURN_IF_ERROR(status);
    iree_byte_sequence_release(result->image);
    result->image = image;
  }
  const uint64_t hash = loom_serve_native_hash(data);
  iree_byte_sequence_t* duplicate = NULL;
  iree_slim_mutex_lock(&domain->mutex);
  loom_serve_native_image_t* image =
      (loom_serve_native_image_t*)loom_serve_native_index_find(&domain->images,
                                                               hash, data);
  iree_status_t status = iree_ok_status();
  if (image) {
    ++image->references;
    iree_byte_sequence_retain(image->sequence);
    duplicate = result->image;
    result->image = image->sequence;
  } else {
    status = loom_serve_native_index_reserve(&domain->images, allocator);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(allocator, sizeof(*image), (void**)&image);
    }
    if (iree_status_is_ok(status)) {
      image->index =
          (loom_serve_native_index_entry_t){.key = data, .hash = hash};
      image->references = 1;
      image->sequence = result->image;
      loom_serve_native_index_insert(&domain->images, &image->index);
    }
  }
  artifact->image = image;
  iree_slim_mutex_unlock(&domain->mutex);
  iree_byte_sequence_release(duplicate);
  return status;
}

void loom_serve_native_artifact_publish(
    loom_serve_native_artifact_t* artifact) {
  if (iree_status_is_ok(artifact->result.status)) {
    artifact->result.status =
        loom_serve_native_artifact_prepare_image(artifact);
  }
  loom_serve_native_domain_t* domain = artifact->domain;
  iree_slim_mutex_lock(&domain->mutex);
  artifact->complete = true;
  if (!iree_status_is_ok(artifact->result.status)) {
    loom_serve_native_index_remove(&domain->artifacts, &artifact->index);
    artifact->indexed = false;
  }
  loom_serve_native_subscription_t* waiters = artifact->waiters;
  artifact->waiters = NULL;
  iree_slim_mutex_unlock(&domain->mutex);
  while (waiters) {
    loom_serve_native_subscription_t* subscription = waiters;
    waiters = subscription->next;
    subscription->ready(subscription->user_data, &artifact->result);
  }
  loom_serve_native_artifact_release(artifact);
}

iree_host_size_t loom_serve_native_artifact_storage_size(
    const loom_serve_native_artifact_t* artifact) {
  iree_host_size_t size =
      sizeof(*artifact) + artifact->key_capacity +
      artifact->result.root_count * sizeof(*artifact->roots) +
      loom_serve_compilation_diagnostics_storage_size(
          artifact->result.diagnostics);
  for (iree_host_size_t i = 0; i < artifact->result.root_count; ++i) {
    size += artifact->roots[i].size;
  }
  return size + (artifact->image ? sizeof(*artifact->image) : 0);
}
