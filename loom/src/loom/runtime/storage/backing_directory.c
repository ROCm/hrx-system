// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_directory.h"

static iree_host_size_t loom_runtime_backing_directory_hash(
    iree_host_size_t index) {
  return index * (iree_host_size_t)UINT64_C(11400714819323198485);
}

static void loom_runtime_backing_directory_release_storage(
    loom_runtime_backing_directory_t* directory) {
  iree_allocator_free(directory->host_allocator, directory->entries);
  directory->entries = NULL;
  directory->capacity = 0;
  directory->tombstone_count = 0;
}

static const loom_runtime_backing_entry_t* loom_runtime_backing_directory_probe(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    bool for_insert) {
  if (!directory->capacity) {
    return NULL;
  }
  const iree_host_size_t mask = directory->capacity - 1;
  iree_host_size_t position = loom_runtime_backing_directory_hash(index) & mask;
  const loom_runtime_backing_entry_t* first_tombstone = NULL;
  for (;;) {
    const loom_runtime_backing_entry_t* entry = &directory->entries[position];
    if (entry->occupied) {
      if (entry->index == index) {
        return entry;
      }
    } else if (entry->tombstone) {
      if (for_insert && !first_tombstone) {
        first_tombstone = entry;
      }
    } else {
      if (!for_insert) {
        return NULL;
      }
      return first_tombstone ? first_tombstone : entry;
    }
    position = (position + 1) & mask;
  }
}

const loom_runtime_backing_entry_t* loom_runtime_backing_directory_find_const(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  return loom_runtime_backing_directory_probe(directory, index, false);
}

loom_runtime_backing_entry_t* loom_runtime_backing_directory_find(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  return (loom_runtime_backing_entry_t*)loom_runtime_backing_directory_probe(
      directory, index, false);
}

static void loom_runtime_backing_directory_insert_raw(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_entry_t value) {
  loom_runtime_backing_entry_t* entry =
      (loom_runtime_backing_entry_t*)loom_runtime_backing_directory_probe(
          directory, value.index, true);
  IREE_ASSERT(entry && !entry->occupied);
  if (entry->tombstone) {
    --directory->tombstone_count;
  }
  value.occupied = true;
  value.tombstone = false;
  *entry = value;
  ++directory->count;
}

static iree_status_t loom_runtime_backing_directory_rehash(
    loom_runtime_backing_directory_t* directory, iree_host_size_t capacity) {
  loom_runtime_backing_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      directory->host_allocator, capacity, sizeof(*entries), (void**)&entries));
  loom_runtime_backing_entry_t* old_entries = directory->entries;
  const iree_host_size_t old_capacity = directory->capacity;
  directory->entries = entries;
  directory->capacity = capacity;
  directory->count = 0;
  directory->tombstone_count = 0;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_entries[i].occupied) {
      loom_runtime_backing_directory_insert_raw(directory, old_entries[i]);
    }
  }
  iree_allocator_free(directory->host_allocator, old_entries);
  return iree_ok_status();
}

static iree_status_t loom_runtime_backing_directory_reserve(
    loom_runtime_backing_directory_t* directory,
    iree_host_size_t additional_count) {
  if (additional_count > IREE_HOST_SIZE_MAX - directory->count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sparse backing directory is too large");
  }
  const iree_host_size_t required = directory->count + additional_count;
  const iree_host_size_t occupied =
      directory->count + iree_max(directory->tombstone_count, additional_count);
  if (directory->capacity && occupied <= directory->capacity / 4 * 3) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = directory->capacity ? directory->capacity : 8;
  while (required > capacity / 4 * 3) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "sparse backing directory is too large");
    }
    capacity *= 2;
  }
  return loom_runtime_backing_directory_rehash(directory, capacity);
}

void loom_runtime_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_runtime_backing_directory_t* out_directory) {
  *out_directory = (loom_runtime_backing_directory_t){
      .host_allocator = host_allocator,
  };
}

void loom_runtime_backing_directory_deinitialize(
    loom_runtime_backing_directory_t* directory) {
  iree_allocator_free(directory->host_allocator, directory->entries);
  *directory = (loom_runtime_backing_directory_t){0};
}

iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_entry_t** out_entry) {
  loom_runtime_backing_entry_t* entry =
      loom_runtime_backing_directory_find(directory, index);
  if (!entry) {
    IREE_RETURN_IF_ERROR(loom_runtime_backing_directory_reserve(directory, 1));
    loom_runtime_backing_directory_insert_raw(
        directory, (loom_runtime_backing_entry_t){.index = index});
    entry = loom_runtime_backing_directory_find(directory, index);
  }
  *out_entry = entry;
  return iree_ok_status();
}

void loom_runtime_backing_directory_remove(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_entry_t* entry) {
  IREE_ASSERT(entry->occupied);
  *entry = (loom_runtime_backing_entry_t){.tombstone = true};
  --directory->count;
  ++directory->tombstone_count;
}

void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory) {
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    loom_runtime_backing_entry_t* entry = &directory->entries[i];
    if (entry->occupied && !entry->physical && !entry->in_flight) {
      IREE_ASSERT(!entry->pending_commit && !entry->counted);
      loom_runtime_backing_directory_remove(directory, entry);
    }
  }
  if (!directory->count) {
    loom_runtime_backing_directory_release_storage(directory);
  }
}

iree_status_t loom_runtime_backing_directory_shrink(
    loom_runtime_backing_directory_t* directory) {
  if (!directory->count) {
    loom_runtime_backing_directory_release_storage(directory);
    return iree_ok_status();
  }
  iree_host_size_t capacity = 8;
  while (directory->count > capacity / 4 * 3) {
    capacity *= 2;
  }
  if (capacity >= directory->capacity) {
    return iree_ok_status();
  }
  return loom_runtime_backing_directory_rehash(directory, capacity);
}
