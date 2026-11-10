// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "amdxdna_executable_cache.h"

#include <limits.h>
#include <string.h>

#include "iree/base/assert.h"

#define HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_TOMBSTONE UINT32_MAX

typedef struct hrx_amdxdna_executable_cache_key_t {
  uint64_t hash_1;
  uint64_t hash_2;
} hrx_amdxdna_executable_cache_key_t;

static void hrx_amdxdna_executable_cache_key_append(
    hrx_amdxdna_executable_cache_key_t* key, const void* data,
    size_t data_length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (size_t i = 0; i < data_length; ++i) {
    key->hash_1 ^= bytes[i];
    key->hash_1 *= UINT64_C(1099511628211);
    key->hash_2 += bytes[i];
    key->hash_2 += key->hash_2 << 10;
    key->hash_2 ^= key->hash_2 >> 6;
  }
}

bool hrx_amdxdna_executable_cache_descriptor_initialize(
    const hrx_amdxdna_xclbin_identity_t* xclbin_identity,
    const hrx_amdxdna_executable_entry_point_t* entry_point,
    const hrx_amdxdna_executable_run_t* run,
    hrx_amdxdna_executable_cache_descriptor_t* out_descriptor) {
  if (!xclbin_identity || !entry_point || !run || !out_descriptor) return false;
  size_t host_bytes = (size_t)xclbin_identity->data_length;
  if (entry_point->name.size > SIZE_MAX - host_bytes ||
      run->transaction.data_length >
          SIZE_MAX - host_bytes - entry_point->name.size) {
    return false;
  }

  hrx_amdxdna_executable_cache_key_t key = {UINT64_C(1469598103934665603),
                                            UINT64_C(0xcbf29ce484222325)};
  hrx_amdxdna_executable_cache_key_append(&key, xclbin_identity,
                                          sizeof(*xclbin_identity));
  hrx_amdxdna_executable_cache_key_append(&key, &entry_point->context_mode,
                                          sizeof(entry_point->context_mode));
  hrx_amdxdna_executable_cache_key_append(&key, &entry_point->xclbin_ordinal,
                                          sizeof(entry_point->xclbin_ordinal));
  hrx_amdxdna_executable_cache_key_append(&key, &entry_point->pdi_ordinal,
                                          sizeof(entry_point->pdi_ordinal));
  hrx_amdxdna_executable_cache_key_append(&key, &entry_point->name.size,
                                          sizeof(entry_point->name.size));
  hrx_amdxdna_executable_cache_key_append(&key, entry_point->name.data,
                                          entry_point->name.size);
  hrx_amdxdna_executable_cache_key_append(&key, &run->transaction.data_length,
                                          sizeof(run->transaction.data_length));
  hrx_amdxdna_executable_cache_key_append(&key, run->transaction.data,
                                          run->transaction.data_length);
  key.hash_2 += key.hash_2 << 3;
  key.hash_2 ^= key.hash_2 >> 11;
  key.hash_2 += key.hash_2 << 15;

  *out_descriptor = (hrx_amdxdna_executable_cache_descriptor_t){
      .key_hash_1 = key.hash_1,
      .key_hash_2 = key.hash_2,
      .xclbin_identity = *xclbin_identity,
      .transaction_size = run->transaction.data_length,
      .entry_point_name_size = entry_point->name.size,
      .host_bytes = host_bytes + entry_point->name.size +
                    run->transaction.data_length,
      .context_mode = entry_point->context_mode,
      .xclbin_ordinal = entry_point->xclbin_ordinal,
      .pdi_ordinal = entry_point->pdi_ordinal,
  };
  return true;
}

void hrx_amdxdna_executable_cache_initialize(
    size_t max_entries, size_t max_host_bytes,
    hrx_amdxdna_executable_cache_s* out_cache) {
  memset(out_cache, 0, sizeof(*out_cache));
  out_cache->max_entries =
      max_entries < HRX_AMDXDNA_EXECUTABLE_CACHE_CAPACITY
          ? max_entries
          : HRX_AMDXDNA_EXECUTABLE_CACHE_CAPACITY;
  out_cache->max_host_bytes = max_host_bytes;
}

static bool hrx_amdxdna_executable_cache_entry_matches(
    const hrx_amdxdna_executable_cache_entry_t* entry,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor) {
  const hrx_amdxdna_executable_cache_descriptor_t* stored =
      &entry->descriptor;
  return stored->key_hash_1 == descriptor->key_hash_1 &&
         stored->key_hash_2 == descriptor->key_hash_2 &&
         stored->xclbin_identity.hash_1 ==
             descriptor->xclbin_identity.hash_1 &&
         stored->xclbin_identity.hash_2 ==
             descriptor->xclbin_identity.hash_2 &&
         stored->xclbin_identity.data_length ==
             descriptor->xclbin_identity.data_length &&
         stored->context_mode == descriptor->context_mode &&
         stored->xclbin_ordinal == descriptor->xclbin_ordinal &&
         stored->pdi_ordinal == descriptor->pdi_ordinal &&
         stored->entry_point_name_size == descriptor->entry_point_name_size &&
         stored->transaction_size == descriptor->transaction_size;
}

static size_t hrx_amdxdna_executable_cache_hash_slot(
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor) {
  const uint64_t mixed =
      descriptor->key_hash_1 ^ (descriptor->key_hash_2 << 32) ^
      (descriptor->key_hash_2 >> 32);
  return (size_t)mixed & (HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY - 1);
}

static bool hrx_amdxdna_executable_cache_find(
    hrx_amdxdna_executable_cache_s* cache,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor,
    size_t* out_index) {
  const size_t base_slot =
      hrx_amdxdna_executable_cache_hash_slot(descriptor);
  for (size_t probe = 0; probe < HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY;
       ++probe) {
    const uint32_t value =
        cache->hash_table[(base_slot + probe) &
                          (HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY - 1)];
    if (value == 0) return false;
    if (value == HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_TOMBSTONE) continue;
    const size_t index = (size_t)value - 1;
    if (hrx_amdxdna_executable_cache_entry_matches(&cache->entries[index],
                                                   descriptor)) {
      *out_index = index;
      return true;
    }
  }
  return false;
}

static void hrx_amdxdna_executable_cache_insert_hash(
    hrx_amdxdna_executable_cache_s* cache, size_t entry_index) {
  const hrx_amdxdna_executable_cache_descriptor_t* descriptor =
      &cache->entries[entry_index].descriptor;
  const size_t base_slot =
      hrx_amdxdna_executable_cache_hash_slot(descriptor);
  size_t tombstone_slot = SIZE_MAX;
  for (size_t probe = 0; probe < HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY;
       ++probe) {
    const size_t slot =
        (base_slot + probe) & (HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY - 1);
    const uint32_t value = cache->hash_table[slot];
    if (value == HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_TOMBSTONE) {
      if (tombstone_slot == SIZE_MAX) tombstone_slot = slot;
      continue;
    }
    if (value != 0) continue;
    const size_t insert_slot =
        tombstone_slot == SIZE_MAX ? slot : tombstone_slot;
    cache->hash_table[insert_slot] = (uint32_t)entry_index + 1;
    if (tombstone_slot != SIZE_MAX) --cache->hash_tombstone_count;
    return;
  }
  IREE_ASSERT(tombstone_slot != SIZE_MAX);
  cache->hash_table[tombstone_slot] = (uint32_t)entry_index + 1;
  --cache->hash_tombstone_count;
}

static void hrx_amdxdna_executable_cache_rebuild_hash(
    hrx_amdxdna_executable_cache_s* cache) {
  memset(cache->hash_table, 0, sizeof(cache->hash_table));
  cache->hash_tombstone_count = 0;
  for (size_t i = 0; i < cache->max_entries; ++i) {
    if (cache->entries[i].hal_executable) {
      hrx_amdxdna_executable_cache_insert_hash(cache, i);
    }
  }
}

static void hrx_amdxdna_executable_cache_remove_entry(
    hrx_amdxdna_executable_cache_s* cache, size_t index) {
  hrx_amdxdna_executable_cache_entry_t* entry = &cache->entries[index];
  const size_t base_slot =
      hrx_amdxdna_executable_cache_hash_slot(&entry->descriptor);
  for (size_t probe = 0; probe < HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY;
       ++probe) {
    const size_t slot =
        (base_slot + probe) & (HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY - 1);
    const uint32_t value = cache->hash_table[slot];
    if (value == 0) break;
    if (value == (uint32_t)index + 1) {
      cache->hash_table[slot] = HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_TOMBSTONE;
      ++cache->hash_tombstone_count;
      break;
    }
  }
  cache->host_bytes -= entry->descriptor.host_bytes;
  iree_hal_executable_release(entry->hal_executable);
  memset(entry, 0, sizeof(*entry));
  --cache->entry_count;
  if (cache->hash_tombstone_count >
      HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY / 4) {
    hrx_amdxdna_executable_cache_rebuild_hash(cache);
  }
}

static size_t hrx_amdxdna_executable_cache_lru_index(
    hrx_amdxdna_executable_cache_s* cache) {
  size_t lru_index = SIZE_MAX;
  for (size_t i = 0; i < cache->max_entries; ++i) {
    if (!cache->entries[i].hal_executable) continue;
    if (lru_index == SIZE_MAX ||
        cache->entries[i].last_use < cache->entries[lru_index].last_use) {
      lru_index = i;
    }
  }
  return lru_index;
}

static size_t hrx_amdxdna_executable_cache_free_entry_index(
    hrx_amdxdna_executable_cache_s* cache) {
  for (size_t i = 0; i < cache->max_entries; ++i) {
    if (!cache->entries[i].hal_executable) return i;
  }
  return SIZE_MAX;
}

iree_hal_executable_t* hrx_amdxdna_executable_cache_lookup(
    hrx_amdxdna_executable_cache_s* cache,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor) {
  size_t entry_index = SIZE_MAX;
  if (!hrx_amdxdna_executable_cache_find(cache, descriptor, &entry_index)) {
    return NULL;
  }
  hrx_amdxdna_executable_cache_entry_t* entry = &cache->entries[entry_index];
  entry->last_use = ++cache->use_clock;
  iree_hal_executable_retain(entry->hal_executable);
  return entry->hal_executable;
}

bool hrx_amdxdna_executable_cache_insert(
    hrx_amdxdna_executable_cache_s* cache,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor,
    iree_hal_executable_t* hal_executable) {
  if (cache->max_entries == 0 ||
      descriptor->host_bytes > cache->max_host_bytes) {
    return false;
  }
  while (cache->entry_count > 0 &&
         (cache->entry_count >= cache->max_entries ||
          cache->host_bytes + descriptor->host_bytes >
              cache->max_host_bytes)) {
    const size_t lru_index = hrx_amdxdna_executable_cache_lru_index(cache);
    IREE_ASSERT(lru_index != SIZE_MAX);
    hrx_amdxdna_executable_cache_remove_entry(cache, lru_index);
  }

  const size_t entry_index =
      hrx_amdxdna_executable_cache_free_entry_index(cache);
  IREE_ASSERT(entry_index != SIZE_MAX);
  hrx_amdxdna_executable_cache_entry_t* entry = &cache->entries[entry_index];
  entry->descriptor = *descriptor;
  entry->hal_executable = hal_executable;
  iree_hal_executable_retain(entry->hal_executable);
  entry->last_use = ++cache->use_clock;
  ++cache->entry_count;
  cache->host_bytes += descriptor->host_bytes;
  hrx_amdxdna_executable_cache_insert_hash(cache, entry_index);
  return true;
}

void hrx_amdxdna_executable_cache_deinitialize(
    hrx_amdxdna_executable_cache_s* cache) {
  for (size_t i = 0; i < cache->max_entries; ++i) {
    if (cache->entries[i].hal_executable) {
      iree_hal_executable_release(cache->entries[i].hal_executable);
    }
  }
  memset(cache, 0, sizeof(*cache));
}
