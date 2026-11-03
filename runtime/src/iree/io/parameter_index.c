// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index.h"

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

typedef struct iree_io_parameter_index_slot_t {
  // Cached key hash, retained across table growth.
  uint64_t hash;
  // First entry with this key, or NULL when the slot is empty.
  const iree_io_parameter_index_entry_t* entry;
} iree_io_parameter_index_slot_t;

struct iree_io_parameter_index_t {
  // Number of owners retaining this index and its immutable entries.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the index, entries, and lookup table.
  iree_allocator_t host_allocator;
  // Lazily allocated small blocks for immutable entries and their copied bytes.
  iree_arena_block_pool_t entry_block_pool;
  // Stable entry storage, released together after all retained files retire.
  iree_arena_allocator_t entry_arena;

  // Guards mutation of the entries list and lookup table.
  // NOTE: this does not guard the entries themselves as we assume they are
  // immutable (today).
  iree_slim_mutex_t mutex;

  // Total capacity of the entries list in elements.
  iree_host_size_t entry_capacity;
  // Currently used entry count in elements.
  iree_host_size_t entry_count;
  // Dense list of entries in the index. Grows as needed.
  iree_io_parameter_index_entry_t** entries;

  // Power-of-two lookup capacity, with at least half the slots empty.
  iree_host_size_t slot_capacity;
  // Open-addressed lookup table borrowing immutable entries from the dense
  // list.
  iree_io_parameter_index_slot_t* slots;
};

// FNV-1a over length-delimited keys, including any embedded NUL bytes.
static uint64_t iree_io_parameter_index_hash(iree_string_view_t key) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    hash ^= (uint8_t)key.data[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

// Returns the matching slot or an empty slot in a non-full lookup table.
static iree_io_parameter_index_slot_t* iree_io_parameter_index_find_slot(
    iree_io_parameter_index_slot_t* slots, iree_host_size_t capacity,
    uint64_t hash, iree_string_view_t key) {
  iree_host_size_t position = (iree_host_size_t)hash & (capacity - 1);
  while (slots[position].entry &&
         (slots[position].hash != hash ||
          !iree_string_view_equal(slots[position].entry->key, key))) {
    position = (position + 1) & (capacity - 1);
  }
  return &slots[position];
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_create(
    iree_allocator_t host_allocator, iree_io_parameter_index_t** out_index) {
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*index), (void**)&index));
  iree_atomic_ref_count_init(&index->ref_count);
  index->host_allocator = host_allocator;
  iree_arena_block_pool_initialize(4096, host_allocator,
                                   &index->entry_block_pool);
  iree_arena_initialize(&index->entry_block_pool, &index->entry_arena);

  iree_slim_mutex_initialize(&index->mutex);

  // Grown on first use. We could allocate a bit of inline storage or take an
  // optional initial capacity for callers that know.
  index->entry_capacity = 0;
  index->entry_count = 0;
  index->entries = NULL;
  index->slot_capacity = 0;
  index->slots = NULL;

  *out_index = index;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_io_parameter_index_destroy(iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = index->host_allocator;

  for (iree_host_size_t i = 0; i < index->entry_count; ++i) {
    iree_io_parameter_index_entry_t* entry = index->entries[i];
    switch (entry->type) {
      default:
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        iree_io_file_handle_release(entry->storage.file.handle);
        break;
    }
  }
  iree_arena_deinitialize(&index->entry_arena);
  iree_arena_block_pool_deinitialize(&index->entry_block_pool);
  if (index->entries) {
    iree_allocator_free(host_allocator, index->entries);
  }
  iree_allocator_free(host_allocator, index->slots);

  iree_slim_mutex_deinitialize(&index->mutex);

  iree_allocator_free(host_allocator, index);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_io_parameter_index_retain(
    iree_io_parameter_index_t* index) {
  if (IREE_LIKELY(index)) {
    iree_atomic_ref_count_inc(&index->ref_count);
  }
}

IREE_API_EXPORT void iree_io_parameter_index_release(
    iree_io_parameter_index_t* index) {
  if (IREE_LIKELY(index) && iree_atomic_ref_count_dec(&index->ref_count) == 1) {
    iree_io_parameter_index_destroy(index);
  }
}

IREE_API_EXPORT iree_host_size_t
iree_io_parameter_index_count(iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  iree_slim_mutex_lock(&index->mutex);
  iree_host_size_t count = index->entry_count;
  iree_slim_mutex_unlock(&index->mutex);
  return count;
}

static iree_status_t iree_io_parameter_index_reserve_unsafe(
    iree_io_parameter_index_t* index, iree_host_size_t new_capacity) {
  IREE_ASSERT_ARGUMENT(index);
  if (new_capacity <= index->entry_capacity &&
      new_capacity <= index->slot_capacity / 2) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_capacity);

  iree_host_size_t slot_capacity = iree_max(index->slot_capacity, 32);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && new_capacity > slot_capacity / 2) {
    if (!iree_host_size_checked_mul(slot_capacity, 2, &slot_capacity)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter index capacity overflow");
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Each allocation can fail independently. Published entries and their lookup
  // remain intact even if only the dense array has grown when an error occurs.
  if (new_capacity > index->entry_capacity) {
    status = iree_allocator_grow_array(
        index->host_allocator, iree_max(new_capacity, 16),
        sizeof(index->entries[0]), &index->entry_capacity,
        (void**)&index->entries);
  }
  if (iree_status_is_ok(status) && slot_capacity > index->slot_capacity) {
    iree_io_parameter_index_slot_t* slots = NULL;
    status = iree_allocator_malloc_array(index->host_allocator, slot_capacity,
                                         sizeof(*slots), (void**)&slots);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < index->slot_capacity; ++i) {
        const iree_io_parameter_index_slot_t* old_slot = &index->slots[i];
        if (old_slot->entry) {
          *iree_io_parameter_index_find_slot(
              slots, slot_capacity, old_slot->hash, old_slot->entry->key) =
              *old_slot;
        }
      }
      iree_allocator_free(index->host_allocator, index->slots);
      index->slots = slots;
      index->slot_capacity = slot_capacity;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_reserve(
    iree_io_parameter_index_t* index, iree_host_size_t new_capacity) {
  IREE_ASSERT_ARGUMENT(index);
  iree_slim_mutex_lock(&index->mutex);
  iree_status_t status =
      iree_io_parameter_index_reserve_unsafe(index, new_capacity);
  iree_slim_mutex_unlock(&index->mutex);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_parameter_index_add(iree_io_parameter_index_t* index,
                            const iree_io_parameter_index_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(entry);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, entry->key.data, entry->key.size);
  iree_slim_mutex_lock(&index->mutex);

  iree_status_t status =
      iree_io_parameter_index_reserve_unsafe(index, index->entry_count + 1);

  // Entries, keys, and metadata share the index lifetime. Arena growth never
  // moves published entries, so callers may retain their pointers while more
  // entries are appended. Oversized metadata receives dedicated arena storage.
  iree_io_parameter_index_entry_t* cloned_entry = NULL;
  if (iree_status_is_ok(status)) {
    // Calculate trailing size with overflow checking. Entry sizes come from
    // file data (untrusted input).
    iree_host_size_t trailing_size = 0;
    if (!iree_host_size_checked_add(
            entry->key.size, entry->metadata.data_length, &trailing_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter entry size overflow");
    } else {
      status = iree_allocator_malloc_with_trailing(
          iree_arena_allocator(&index->entry_arena), sizeof(*cloned_entry),
          trailing_size, (void**)&cloned_entry);
    }
  }
  if (iree_status_is_ok(status)) {
    cloned_entry->key = iree_make_string_view(
        (char*)cloned_entry + sizeof(*cloned_entry), entry->key.size);
    cloned_entry->metadata =
        iree_const_byte_span_is_empty(entry->metadata)
            ? iree_const_byte_span_empty()
            : iree_make_const_byte_span(
                  (uint8_t*)cloned_entry->key.data + cloned_entry->key.size,
                  entry->metadata.data_length);
    cloned_entry->length = entry->length;
    cloned_entry->type = entry->type;
    switch (entry->type) {
      default:
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
        cloned_entry->storage.splat = entry->storage.splat;
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        cloned_entry->storage.file = entry->storage.file;
        iree_io_file_handle_retain(cloned_entry->storage.file.handle);
        break;
    }
    if (entry->key.size > 0) {
      memcpy((void*)cloned_entry->key.data, entry->key.data, entry->key.size);
    }
    if (entry->metadata.data_length > 0) {
      memcpy((void*)cloned_entry->metadata.data, entry->metadata.data,
             entry->metadata.data_length);
    }

    // Append the entry to the file index.
    index->entries[index->entry_count++] = cloned_entry;

    // Enumeration retains duplicates, while lookup returns the first entry.
    uint64_t hash = iree_io_parameter_index_hash(cloned_entry->key);
    iree_io_parameter_index_slot_t* slot = iree_io_parameter_index_find_slot(
        index->slots, index->slot_capacity, hash, cloned_entry->key);
    if (!slot->entry) {
      slot->hash = hash;
      slot->entry = cloned_entry;
    }
  }

  iree_slim_mutex_unlock(&index->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_get(
    iree_io_parameter_index_t* index, iree_host_size_t i,
    const iree_io_parameter_index_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  iree_slim_mutex_lock(&index->mutex);

  iree_status_t status = iree_ok_status();
  if (i < index->entry_count) {
    *out_entry = index->entries[i];
  } else {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "entry %" PRIhsz " out of range (have %" PRIhsz
                              " entries in the index)",
                              i, index->entry_count);
  }

  iree_slim_mutex_unlock(&index->mutex);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_lookup(
    iree_io_parameter_index_t* index, iree_string_view_t key,
    const iree_io_parameter_index_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, key.data, key.size);
  uint64_t hash = iree_io_parameter_index_hash(key);
  iree_slim_mutex_lock(&index->mutex);

  iree_status_t status = iree_ok_status();
  if (index->slot_capacity) {
    *out_entry = iree_io_parameter_index_find_slot(
                     index->slots, index->slot_capacity, hash, key)
                     ->entry;
  }
  if (*out_entry == NULL) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no parameter found in index with key '%.*s'",
                              (int)key.size, key.data);
  }

  iree_slim_mutex_unlock(&index->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_dump(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_string_builder_t* builder) {
  iree_host_size_t entry_count = iree_io_parameter_index_count(index);
  uint64_t total_bytes = 0;
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(iree_io_parameter_index_get(index, i, &entry));
    total_bytes += entry->length;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//"
      "===-----------------------------------------------------------------"
      "---------------------------------------------===//\n"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "// Parameter scope "));
  if (iree_string_view_is_empty(scope)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "<global>"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "`%.*s`", (int)scope.size, scope.data));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " (%" PRIhsz " entries, %" PRIu64 " total bytes)\n", entry_count,
      total_bytes));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//===------------+------------------+------------------+------------"
      "-----------------------------------------------===//\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//         Start |              End |           Length | Key\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//---------------+------------------+------------------+------------"
      "--------------------------------------------------//\n"));
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(iree_io_parameter_index_get(index, i, &entry));
    switch (entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "               - |                - | %16" PRIu64 " | `%.*s`\n",
            entry->length, (int)entry->key.size, entry->key.data));
        break;
      }
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%16" PRIu64 " | %16" PRIu64 " | %16" PRIu64 " | `%.*s`\n",
            entry->storage.file.offset,
            entry->storage.file.offset + entry->length, entry->length,
            (int)entry->key.size, entry->key.data));
        break;
      }
      default: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "               ? |               ? | %16" PRIu64 " | `%.*s`\n",
            entry->length, (int)entry->key.size, entry->key.data));
        break;
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_fprint(
    FILE* file, iree_string_view_t scope, iree_io_parameter_index_t* index) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(index->host_allocator, &builder);
  iree_status_t status = iree_io_parameter_index_dump(scope, index, &builder);
  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}
