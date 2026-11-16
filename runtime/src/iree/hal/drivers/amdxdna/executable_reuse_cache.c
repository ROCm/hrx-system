// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/executable_reuse_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/program_image.h"
#include "iree/hal/drivers/amdxdna/util.h"

typedef struct iree_hal_amdxdna_executable_reuse_entry_t
    iree_hal_amdxdna_executable_reuse_entry_t;
struct iree_hal_amdxdna_executable_reuse_entry_t {
  // LRU list; the cache head is the most recently used entry. Evicted entries
  // reuse |lru_next| to form a release list.
  iree_hal_amdxdna_executable_reuse_entry_t* lru_prev;
  iree_hal_amdxdna_executable_reuse_entry_t* lru_next;
  iree_hal_amdxdna_executable_reuse_entry_t* bucket_next;
  uint64_t hash;
  // Retained.
  iree_hal_executable_t* executable;
  // Aliases |executable|'s control code.
  iree_const_byte_span_t control_code;
  // Exclusive executable bytes plus this entry's own storage.
  iree_host_size_t host_bytes;
  iree_host_size_t image_count;
  const iree_hal_amdxdna_program_image_t*
      images[IREE_HAL_AMDXDNA_EXECUTABLE_FOOTPRINT_MAX_IMAGES];
  iree_host_size_t descriptor_length;
  // Followed by |descriptor_length| descriptor bytes.
};

// One program image referenced by at least one cached entry.
typedef struct iree_hal_amdxdna_executable_reuse_image_charge_t {
  const iree_hal_amdxdna_program_image_t* image;
  iree_host_size_t entry_count;
  iree_host_size_t host_bytes;
} iree_hal_amdxdna_executable_reuse_image_charge_t;

struct iree_hal_amdxdna_executable_reuse_cache_t {
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_executable_reuse_cache_limits_t limits;
  iree_slim_mutex_t mutex;
  iree_host_size_t bucket_count;  // Power of two.
  iree_hal_amdxdna_executable_reuse_entry_t** buckets;
  iree_hal_amdxdna_executable_reuse_entry_t* lru_head;
  iree_hal_amdxdna_executable_reuse_entry_t* lru_tail;
  iree_host_size_t entry_count;
  iree_host_size_t host_bytes;
  iree_host_size_t image_bytes;
  iree_hal_amdxdna_executable_reuse_image_charge_t* image_charges;
  iree_host_size_t image_charge_count;
  iree_host_size_t image_charge_capacity;
  uint64_t hit_count;
  uint64_t miss_count;
  uint64_t eviction_count;
};

static const uint8_t* iree_hal_amdxdna_executable_reuse_entry_descriptor(
    const iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  return (const uint8_t*)entry + iree_sizeof_struct(*entry);
}

static uint64_t iree_hal_amdxdna_executable_reuse_key_hash(
    const iree_hal_amdxdna_executable_reuse_key_t* key) {
  return iree_hal_amdxdna_hash_bytes(
      iree_hal_amdxdna_hash_bytes(/*seed=*/0, key->descriptor),
      key->control_code);
}

static bool iree_hal_amdxdna_byte_spans_equal(iree_const_byte_span_t lhs,
                                              iree_const_byte_span_t rhs) {
  return lhs.data_length == rhs.data_length &&
         (lhs.data_length == 0 ||
          memcmp(lhs.data, rhs.data, lhs.data_length) == 0);
}

static bool iree_hal_amdxdna_executable_reuse_entry_matches(
    const iree_hal_amdxdna_executable_reuse_entry_t* entry, uint64_t hash,
    const iree_hal_amdxdna_executable_reuse_key_t* key) {
  return entry->hash == hash &&
         iree_hal_amdxdna_byte_spans_equal(
             iree_make_const_byte_span(
                 iree_hal_amdxdna_executable_reuse_entry_descriptor(entry),
                 entry->descriptor_length),
             key->descriptor) &&
         iree_hal_amdxdna_byte_spans_equal(entry->control_code,
                                           key->control_code);
}

static iree_hal_amdxdna_executable_reuse_entry_t**
iree_hal_amdxdna_executable_reuse_cache_bucket(
    iree_hal_amdxdna_executable_reuse_cache_t* cache, uint64_t hash) {
  return &cache->buckets[hash & (cache->bucket_count - 1)];
}

static iree_hal_amdxdna_executable_reuse_entry_t*
iree_hal_amdxdna_executable_reuse_cache_find_locked(
    iree_hal_amdxdna_executable_reuse_cache_t* cache, uint64_t hash,
    const iree_hal_amdxdna_executable_reuse_key_t* key) {
  for (iree_hal_amdxdna_executable_reuse_entry_t* entry =
           *iree_hal_amdxdna_executable_reuse_cache_bucket(cache, hash);
       entry; entry = entry->bucket_next) {
    if (iree_hal_amdxdna_executable_reuse_entry_matches(entry, hash, key)) {
      return entry;
    }
  }
  return NULL;
}

static void iree_hal_amdxdna_executable_reuse_cache_lru_unlink(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  if (entry->lru_prev) {
    entry->lru_prev->lru_next = entry->lru_next;
  } else {
    cache->lru_head = entry->lru_next;
  }
  if (entry->lru_next) {
    entry->lru_next->lru_prev = entry->lru_prev;
  } else {
    cache->lru_tail = entry->lru_prev;
  }
  entry->lru_prev = NULL;
  entry->lru_next = NULL;
}

static void iree_hal_amdxdna_executable_reuse_cache_lru_push_head(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  entry->lru_prev = NULL;
  entry->lru_next = cache->lru_head;
  if (cache->lru_head) {
    cache->lru_head->lru_prev = entry;
  } else {
    cache->lru_tail = entry;
  }
  cache->lru_head = entry;
}

static iree_hal_amdxdna_executable_reuse_image_charge_t*
iree_hal_amdxdna_executable_reuse_cache_find_image_charge(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_program_image_t* image) {
  for (iree_host_size_t i = 0; i < cache->image_charge_count; ++i) {
    if (cache->image_charges[i].image == image) {
      return &cache->image_charges[i];
    }
  }
  return NULL;
}

// Returns the image bytes |entry| would add to the cache charge.
static iree_host_size_t iree_hal_amdxdna_executable_reuse_cache_new_image_bytes(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_entry_t* entry,
    iree_host_size_t* out_new_image_count) {
  iree_host_size_t bytes = 0;
  *out_new_image_count = 0;
  for (iree_host_size_t i = 0; i < entry->image_count; ++i) {
    if (!iree_hal_amdxdna_executable_reuse_cache_find_image_charge(
            cache, entry->images[i])) {
      bytes += iree_hal_amdxdna_program_image_host_bytes(entry->images[i]);
      ++*out_new_image_count;
    }
  }
  return bytes;
}

static iree_status_t
iree_hal_amdxdna_executable_reuse_cache_reserve_image_charges(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      cache->image_charge_count + additional_count;
  if (required_count <= cache->image_charge_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      iree_max(required_count, cache->image_charge_capacity * 2);
  new_capacity = iree_max(new_capacity, 8);
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      cache->host_allocator, new_capacity, sizeof(*cache->image_charges),
      (void**)&cache->image_charges));
  cache->image_charge_capacity = new_capacity;
  return iree_ok_status();
}

// Requires capacity reserved by
// iree_hal_amdxdna_executable_reuse_cache_reserve_image_charges.
static void iree_hal_amdxdna_executable_reuse_cache_charge_images(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  for (iree_host_size_t i = 0; i < entry->image_count; ++i) {
    iree_hal_amdxdna_executable_reuse_image_charge_t* charge =
        iree_hal_amdxdna_executable_reuse_cache_find_image_charge(
            cache, entry->images[i]);
    if (!charge) {
      charge = &cache->image_charges[cache->image_charge_count++];
      charge->image = entry->images[i];
      charge->entry_count = 0;
      charge->host_bytes =
          iree_hal_amdxdna_program_image_host_bytes(entry->images[i]);
      cache->image_bytes += charge->host_bytes;
      cache->host_bytes += charge->host_bytes;
    }
    ++charge->entry_count;
  }
}

static void iree_hal_amdxdna_executable_reuse_cache_uncharge_images(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  for (iree_host_size_t i = 0; i < entry->image_count; ++i) {
    iree_hal_amdxdna_executable_reuse_image_charge_t* charge =
        iree_hal_amdxdna_executable_reuse_cache_find_image_charge(
            cache, entry->images[i]);
    IREE_ASSERT(charge);
    if (--charge->entry_count != 0) continue;
    cache->image_bytes -= charge->host_bytes;
    cache->host_bytes -= charge->host_bytes;
    *charge = cache->image_charges[--cache->image_charge_count];
  }
}

static void iree_hal_amdxdna_executable_reuse_cache_unlink_locked(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_entry_t* entry) {
  iree_hal_amdxdna_executable_reuse_entry_t** link =
      iree_hal_amdxdna_executable_reuse_cache_bucket(cache, entry->hash);
  while (*link != entry) link = &(*link)->bucket_next;
  *link = entry->bucket_next;
  entry->bucket_next = NULL;
  iree_hal_amdxdna_executable_reuse_cache_lru_unlink(cache, entry);
  iree_hal_amdxdna_executable_reuse_cache_uncharge_images(cache, entry);
  cache->host_bytes -= entry->host_bytes;
  --cache->entry_count;
}

static void iree_hal_amdxdna_executable_reuse_cache_release_entries(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_entry_t* list) {
  while (list) {
    iree_hal_amdxdna_executable_reuse_entry_t* next = list->lru_next;
    iree_hal_executable_release(list->executable);
    iree_allocator_free(cache->host_allocator, list);
    list = next;
  }
}

iree_status_t iree_hal_amdxdna_executable_reuse_cache_create(
    iree_hal_amdxdna_executable_reuse_cache_limits_t limits,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_executable_reuse_cache_t** out_cache) {
  IREE_ASSERT_ARGUMENT(out_cache);
  *out_cache = NULL;
  if (limits.max_entries == 0 || limits.max_host_bytes == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable reuse cache limits must be non-zero");
  }
  iree_host_size_t bucket_count = 16;
  while (bucket_count < limits.max_entries) bucket_count <<= 1;

  iree_hal_amdxdna_executable_reuse_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  memset(cache, 0, sizeof(*cache));
  cache->host_allocator = host_allocator;
  cache->limits = limits;
  cache->bucket_count = bucket_count;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, bucket_count, sizeof(*cache->buckets),
      (void**)&cache->buckets);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, cache);
    return status;
  }
  memset(cache->buckets, 0, bucket_count * sizeof(*cache->buckets));
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

void iree_hal_amdxdna_executable_reuse_cache_destroy(
    iree_hal_amdxdna_executable_reuse_cache_t* cache) {
  if (!cache) return;
  iree_hal_amdxdna_executable_reuse_cache_clear(cache);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache->image_charges);
  iree_allocator_free(cache->host_allocator, cache->buckets);
  iree_allocator_free(cache->host_allocator, cache);
}

iree_hal_executable_t* iree_hal_amdxdna_executable_reuse_cache_lookup(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_key_t* key) {
  if (!cache || key->control_code.data_length == 0) return NULL;
  const uint64_t hash = iree_hal_amdxdna_executable_reuse_key_hash(key);
  iree_slim_mutex_lock(&cache->mutex);
  iree_hal_amdxdna_executable_reuse_entry_t* entry =
      iree_hal_amdxdna_executable_reuse_cache_find_locked(cache, hash, key);
  iree_hal_executable_t* executable = NULL;
  if (entry) {
    ++cache->hit_count;
    iree_hal_amdxdna_executable_reuse_cache_lru_unlink(cache, entry);
    iree_hal_amdxdna_executable_reuse_cache_lru_push_head(cache, entry);
    executable = entry->executable;
    iree_hal_executable_retain(executable);
  } else {
    ++cache->miss_count;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  return executable;
}

bool iree_hal_amdxdna_executable_reuse_cache_insert(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_key_t* key,
    iree_hal_executable_t* executable) {
  if (!cache || !executable) return false;
  const iree_const_byte_span_t control_code =
      iree_hal_amdxdna_executable_single_run_control_code(executable);
  if (control_code.data_length == 0 ||
      !iree_hal_amdxdna_byte_spans_equal(control_code, key->control_code)) {
    return false;
  }
  iree_hal_amdxdna_executable_footprint_t footprint;
  if (!iree_hal_amdxdna_executable_query_footprint(executable, &footprint)) {
    return false;
  }

  iree_hal_amdxdna_executable_reuse_entry_t* entry = NULL;
  const iree_host_size_t entry_size =
      iree_sizeof_struct(*entry) + key->descriptor.data_length;
  if (!iree_status_is_ok(iree_allocator_malloc(cache->host_allocator,
                                               entry_size, (void**)&entry))) {
    return false;
  }
  memset(entry, 0, sizeof(*entry));
  entry->hash = iree_hal_amdxdna_executable_reuse_key_hash(key);
  entry->executable = executable;
  entry->control_code = control_code;
  entry->host_bytes = footprint.exclusive_host_bytes + entry_size;
  entry->image_count = footprint.image_count;
  for (iree_host_size_t i = 0; i < footprint.image_count; ++i) {
    entry->images[i] = footprint.images[i];
  }
  entry->descriptor_length = key->descriptor.data_length;
  if (key->descriptor.data_length != 0) {
    memcpy((uint8_t*)entry + iree_sizeof_struct(*entry), key->descriptor.data,
           key->descriptor.data_length);
  }

  iree_slim_mutex_lock(&cache->mutex);
  iree_host_size_t new_image_count = 0;
  const iree_host_size_t new_image_bytes =
      iree_hal_amdxdna_executable_reuse_cache_new_image_bytes(cache, entry,
                                                              &new_image_count);
  const bool admissible =
      !iree_hal_amdxdna_executable_reuse_cache_find_locked(cache, entry->hash,
                                                           key) &&
      entry->host_bytes + new_image_bytes <= cache->limits.max_host_bytes &&
      iree_status_is_ok(
          iree_hal_amdxdna_executable_reuse_cache_reserve_image_charges(
              cache, new_image_count));
  if (!admissible) {
    iree_slim_mutex_unlock(&cache->mutex);
    iree_allocator_free(cache->host_allocator, entry);
    return false;
  }

  iree_hal_executable_retain(executable);
  iree_hal_amdxdna_executable_reuse_entry_t** bucket =
      iree_hal_amdxdna_executable_reuse_cache_bucket(cache, entry->hash);
  entry->bucket_next = *bucket;
  *bucket = entry;
  iree_hal_amdxdna_executable_reuse_cache_lru_push_head(cache, entry);
  iree_hal_amdxdna_executable_reuse_cache_charge_images(cache, entry);
  cache->host_bytes += entry->host_bytes;
  ++cache->entry_count;

  iree_hal_amdxdna_executable_reuse_entry_t* evicted = NULL;
  while (cache->lru_tail != entry &&
         (cache->entry_count > cache->limits.max_entries ||
          cache->host_bytes > cache->limits.max_host_bytes)) {
    iree_hal_amdxdna_executable_reuse_entry_t* victim = cache->lru_tail;
    iree_hal_amdxdna_executable_reuse_cache_unlink_locked(cache, victim);
    victim->lru_next = evicted;
    evicted = victim;
    ++cache->eviction_count;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  iree_hal_amdxdna_executable_reuse_cache_release_entries(cache, evicted);
  return true;
}

void iree_hal_amdxdna_executable_reuse_cache_clear(
    iree_hal_amdxdna_executable_reuse_cache_t* cache) {
  if (!cache) return;
  iree_slim_mutex_lock(&cache->mutex);
  iree_hal_amdxdna_executable_reuse_entry_t* evicted = NULL;
  while (cache->lru_tail) {
    iree_hal_amdxdna_executable_reuse_entry_t* victim = cache->lru_tail;
    iree_hal_amdxdna_executable_reuse_cache_unlink_locked(cache, victim);
    victim->lru_next = evicted;
    evicted = victim;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  iree_hal_amdxdna_executable_reuse_cache_release_entries(cache, evicted);
}

void iree_hal_amdxdna_executable_reuse_cache_query_stats(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_cache_stats_t* out_stats) {
  memset(out_stats, 0, sizeof(*out_stats));
  if (!cache) return;
  iree_slim_mutex_lock(&cache->mutex);
  out_stats->entry_count = cache->entry_count;
  out_stats->host_bytes = cache->host_bytes;
  out_stats->image_bytes = cache->image_bytes;
  out_stats->hit_count = cache->hit_count;
  out_stats->miss_count = cache->miss_count;
  out_stats->eviction_count = cache->eviction_count;
  iree_slim_mutex_unlock(&cache->mutex);
}
