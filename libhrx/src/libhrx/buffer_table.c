// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer table: maps device/host pointers to hrx_buffer_t handles.
// Unified implementation used by both libhrx and the HIP/CUDA bindings.

#include "buffer_table.h"

#include <stdlib.h>
#include <string.h>

#define HRX_BUFFER_TABLE_INITIAL_CAPACITY 256

void hrx_buffer_table_initialize(hrx_buffer_table_t* table) {
  memset(table, 0, sizeof(*table));
  iree_slim_mutex_initialize(&table->mutex);
}

void hrx_buffer_table_deinitialize(hrx_buffer_table_t* table) {
  IREE_ASSERT(table->reserved_insert_count == 0);
  iree_slim_mutex_deinitialize(&table->mutex);
  free(table->entries);
  free(table->range_index);
  memset(table, 0, sizeof(*table));
}

static size_t hrx_buffer_table_range_lower_bound(
    const hrx_buffer_table_t* table, uint64_t base) {
  size_t low = 0;
  size_t high = table->range_count;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    if (table->range_index[middle].base < base) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

static size_t hrx_buffer_table_range_upper_bound(
    const hrx_buffer_table_t* table, uint64_t address) {
  size_t low = 0;
  size_t high = table->range_count;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    if (table->range_index[middle].base <= address) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

static size_t hrx_buffer_table_find_index_locked(
    const hrx_buffer_table_t* table, uint64_t any_ptr) {
  const size_t upper_bound = hrx_buffer_table_range_upper_bound(table, any_ptr);
  if (upper_bound == 0) {
    return table->count;
  }
  const hrx_buffer_table_range_index_t* range =
      &table->range_index[upper_bound - 1];
  const hrx_buffer_table_entry_t* entry = &table->entries[range->entry_index];
  return any_ptr - range->base < (uint64_t)entry->size ? range->entry_index
                                                       : table->count;
}

static hrx_status_t hrx_buffer_table_grow(hrx_buffer_table_t* table) {
  size_t new_cap = HRX_BUFFER_TABLE_INITIAL_CAPACITY;
  if (table->capacity) {
    if (table->capacity > SIZE_MAX / 2) {
      return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                             "buffer table capacity overflow");
    }
    new_cap = table->capacity * 2;
  }
  if (new_cap > SIZE_MAX / sizeof(hrx_buffer_table_entry_t)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table allocation size overflow");
  }
  if (new_cap > SIZE_MAX / 2 ||
      new_cap * 2 > SIZE_MAX / sizeof(hrx_buffer_table_range_index_t)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table range index size overflow");
  }
  hrx_buffer_table_entry_t* new_entries =
      malloc(new_cap * sizeof(hrx_buffer_table_entry_t));
  if (!new_entries) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table grow failed");
  }
  hrx_buffer_table_range_index_t* new_range_index =
      malloc(new_cap * 2 * sizeof(hrx_buffer_table_range_index_t));
  if (!new_range_index) {
    free(new_entries);
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table range index grow failed");
  }
  if (table->count > 0) {
    memcpy(new_entries, table->entries,
           table->count * sizeof(hrx_buffer_table_entry_t));
  }
  if (table->range_count > 0) {
    memcpy(new_range_index, table->range_index,
           table->range_count * sizeof(hrx_buffer_table_range_index_t));
  }
  free(table->entries);
  free(table->range_index);
  table->entries = new_entries;
  table->range_index = new_range_index;
  table->capacity = new_cap;
  return hrx_ok_status();
}

static hrx_status_t hrx_buffer_table_ensure_insert_capacity_locked(
    hrx_buffer_table_t* table, size_t additional_reservations) {
  if (table->count > SIZE_MAX - table->reserved_insert_count ||
      table->count + table->reserved_insert_count >
          SIZE_MAX - additional_reservations) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "buffer table entry count overflow");
  }
  const size_t required_capacity =
      table->count + table->reserved_insert_count + additional_reservations;
  while (table->capacity < required_capacity) {
    hrx_status_t status = hrx_buffer_table_grow(table);
    if (!hrx_status_is_ok(status)) {
      return status;
    }
  }
  return hrx_ok_status();
}

static bool hrx_buffer_table_range_overlaps_locked(
    const hrx_buffer_table_t* table, uint64_t base, size_t size) {
  const size_t position = hrx_buffer_table_range_lower_bound(table, base);
  if (position > 0) {
    const hrx_buffer_table_range_index_t* previous =
        &table->range_index[position - 1];
    const hrx_buffer_table_entry_t* previous_entry =
        &table->entries[previous->entry_index];
    if (base - previous->base < (uint64_t)previous_entry->size) {
      return true;
    }
  }
  return position < table->range_count &&
         table->range_index[position].base - base < (uint64_t)size;
}

static hrx_status_t hrx_buffer_table_validate_insert_locked(
    hrx_buffer_table_t* table, uint64_t device_ptr, void* host_ptr,
    size_t size) {
  if (size == 0 || device_ptr == 0 ||
      (uint64_t)size > UINT64_MAX - device_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device allocation range is invalid");
  }
  const uint64_t host_address = (uint64_t)(uintptr_t)host_ptr;
  if (host_ptr && host_address != device_ptr &&
      ((uint64_t)size > UINT64_MAX - host_address ||
       (device_ptr < host_address + (uint64_t)size &&
        host_address < device_ptr + (uint64_t)size))) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "host allocation alias range is invalid");
  }
  if (hrx_buffer_table_range_overlaps_locked(table, device_ptr, size) ||
      (host_ptr && host_address != device_ptr &&
       hrx_buffer_table_range_overlaps_locked(table, host_address, size))) {
    return hrx_make_status(HRX_STATUS_ALREADY_EXISTS,
                           "allocation range already registered");
  }
  return hrx_ok_status();
}

static void hrx_buffer_table_insert_range_locked(hrx_buffer_table_t* table,
                                                 uint64_t base,
                                                 size_t entry_index) {
  const size_t position = hrx_buffer_table_range_lower_bound(table, base);
  IREE_ASSERT(table->range_count < table->capacity * 2);
  if (position < table->range_count) {
    memmove(&table->range_index[position + 1], &table->range_index[position],
            (table->range_count - position) * sizeof(*table->range_index));
  }
  table->range_index[position] = (hrx_buffer_table_range_index_t){
      .base = base,
      .entry_index = entry_index,
  };
  ++table->range_count;
}

// Removes an entry's aliases before the caller relocates the last dense entry.
// All alias bases are unique and remain stable while the table lock is held.
static void hrx_buffer_table_remove_entry_ranges_locked(
    hrx_buffer_table_t* table, size_t entry_index, size_t last_entry_index) {
  const hrx_buffer_table_entry_t* entry = &table->entries[entry_index];

  size_t first_range =
      hrx_buffer_table_range_lower_bound(table, entry->device_ptr);
  const uint64_t host_address = (uint64_t)(uintptr_t)entry->host_ptr;
  const bool has_host_alias =
      entry->host_ptr && host_address != entry->device_ptr;
  size_t second_range = table->range_count;
  if (has_host_alias) {
    second_range = hrx_buffer_table_range_lower_bound(table, host_address);
    if (second_range < first_range) {
      const size_t swap = first_range;
      first_range = second_range;
      second_range = swap;
    }
  }
  IREE_ASSERT(first_range < table->range_count &&
              table->range_index[first_range].entry_index == entry_index);

  // Preserve the prefix and move each surviving suffix record once. With two
  // aliases, the middle segment crosses one gap and the final segment crosses
  // both. Either segment may be empty, including removal at the end of the
  // index.
  if (has_host_alias) {
    IREE_ASSERT(second_range < table->range_count &&
                table->range_index[second_range].entry_index == entry_index);
    memmove(&table->range_index[first_range],
            &table->range_index[first_range + 1],
            (second_range - first_range - 1) * sizeof(*table->range_index));
    memmove(
        &table->range_index[second_range - 1],
        &table->range_index[second_range + 1],
        (table->range_count - second_range - 1) * sizeof(*table->range_index));
    table->range_count -= 2;
  } else {
    memmove(
        &table->range_index[first_range], &table->range_index[first_range + 1],
        (table->range_count - first_range - 1) * sizeof(*table->range_index));
    --table->range_count;
  }

  // Update range mappings for the last entry being moved into this slot.
  if (entry_index != last_entry_index) {
    const hrx_buffer_table_entry_t* moved = &table->entries[last_entry_index];
    size_t position =
        hrx_buffer_table_range_lower_bound(table, moved->device_ptr);
    table->range_index[position].entry_index = entry_index;
    const uint64_t moved_host_address = (uint64_t)(uintptr_t)moved->host_ptr;
    if (moved->host_ptr && moved_host_address != moved->device_ptr) {
      position = hrx_buffer_table_range_lower_bound(table, moved_host_address);
      table->range_index[position].entry_index = entry_index;
    }
  }
}

hrx_status_t hrx_buffer_table_insert(hrx_buffer_table_t* table,
                                     uint64_t device_ptr, void* host_ptr,
                                     size_t size, hrx_buffer_t buffer,
                                     void* user_data) {
  iree_slim_mutex_lock(&table->mutex);

  hrx_status_t status = hrx_buffer_table_validate_insert_locked(
      table, device_ptr, host_ptr, size);
  if (hrx_status_is_ok(status)) {
    status = hrx_buffer_table_ensure_insert_capacity_locked(
        table, /*additional_reservations=*/1);
  }
  if (!hrx_status_is_ok(status)) {
    iree_slim_mutex_unlock(&table->mutex);
    return status;
  }

  const size_t entry_index = table->count;
  table->entries[entry_index] = (hrx_buffer_table_entry_t){
      .device_ptr = device_ptr,
      .host_ptr = host_ptr,
      .size = size,
      .buffer = buffer,
      .user_data = user_data,
  };
  hrx_buffer_table_insert_range_locked(table, device_ptr, entry_index);
  const uint64_t host_address = (uint64_t)(uintptr_t)host_ptr;
  if (host_ptr && host_address != device_ptr) {
    hrx_buffer_table_insert_range_locked(table, host_address, entry_index);
  }
  ++table->count;

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_table_reserve_insert(hrx_buffer_table_t* table) {
  iree_slim_mutex_lock(&table->mutex);
  hrx_status_t status = hrx_buffer_table_ensure_insert_capacity_locked(
      table, /*additional_reservations=*/1);
  if (hrx_status_is_ok(status)) {
    ++table->reserved_insert_count;
  }
  iree_slim_mutex_unlock(&table->mutex);
  return status;
}

hrx_status_t hrx_buffer_table_insert_reserved(hrx_buffer_table_t* table,
                                              uint64_t device_ptr,
                                              void* host_ptr, size_t size,
                                              hrx_buffer_t buffer,
                                              void* user_data) {
  iree_slim_mutex_lock(&table->mutex);
  IREE_ASSERT(table->reserved_insert_count > 0);
  --table->reserved_insert_count;

  hrx_status_t status = hrx_buffer_table_validate_insert_locked(
      table, device_ptr, host_ptr, size);
  if (hrx_status_is_ok(status)) {
    IREE_ASSERT(table->count < table->capacity);
    const size_t entry_index = table->count;
    table->entries[entry_index] = (hrx_buffer_table_entry_t){
        .device_ptr = device_ptr,
        .host_ptr = host_ptr,
        .size = size,
        .buffer = buffer,
        .user_data = user_data,
    };
    hrx_buffer_table_insert_range_locked(table, device_ptr, entry_index);
    const uint64_t host_address = (uint64_t)(uintptr_t)host_ptr;
    if (host_ptr && host_address != device_ptr) {
      hrx_buffer_table_insert_range_locked(table, host_address, entry_index);
    }
    ++table->count;
  }

  iree_slim_mutex_unlock(&table->mutex);
  return status;
}

void hrx_buffer_table_cancel_reserved_insert(hrx_buffer_table_t* table) {
  iree_slim_mutex_lock(&table->mutex);
  IREE_ASSERT(table->reserved_insert_count > 0);
  --table->reserved_insert_count;
  iree_slim_mutex_unlock(&table->mutex);
}

hrx_status_t hrx_buffer_table_remove(hrx_buffer_table_t* table,
                                     uint64_t any_ptr) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = hrx_buffer_table_find_index_locked(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "pointer not found in buffer table");
  }

  const size_t last_index = table->count - 1;
  hrx_buffer_table_remove_entry_ranges_locked(table, idx, last_index);
  if (idx != last_index) {
    table->entries[idx] = table->entries[last_index];
  }
  --table->count;

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_ok_status();
}

static void hrx_buffer_table_fill_result(hrx_buffer_table_entry_t* e,
                                         uint64_t any_ptr,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data) {
  if (out_buffer) {
    *out_buffer = e->buffer;
  }
  if (out_offset) {
    if (any_ptr >= e->device_ptr &&
        any_ptr - e->device_ptr < (uint64_t)e->size) {
      *out_offset = (size_t)(any_ptr - e->device_ptr);
    } else {
      uint64_t host_addr = (uint64_t)(uintptr_t)e->host_ptr;
      *out_offset = (size_t)(any_ptr - host_addr);
    }
  }
  if (out_user_data) {
    *out_user_data = e->user_data;
  }
}

static void hrx_buffer_table_retain_entry(
    hrx_buffer_table_entry_t* entry, size_t offset,
    hrx_buffer_table_retained_ref_t* out_ref) {
  hrx_buffer_retain(entry->buffer);
  *out_ref = (hrx_buffer_table_retained_ref_t){
      .buffer = entry->buffer,
      .device_ptr = entry->device_ptr,
      .host_ptr = entry->host_ptr,
      .size = entry->size,
      .offset = offset,
      .user_data = entry->user_data,
  };
}

bool hrx_buffer_table_try_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                               hrx_buffer_t* out_buffer, size_t* out_offset,
                               void** out_user_data) {
  iree_slim_mutex_lock(&table->mutex);

  size_t idx = hrx_buffer_table_find_index_locked(table, any_ptr);
  if (idx >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    if (out_buffer) {
      *out_buffer = NULL;
    }
    if (out_offset) {
      *out_offset = 0;
    }
    if (out_user_data) {
      *out_user_data = NULL;
    }
    return false;
  }

  hrx_buffer_table_fill_result(&table->entries[idx], any_ptr, out_buffer,
                               out_offset, out_user_data);
  iree_slim_mutex_unlock(&table->mutex);
  return true;
}

hrx_status_t hrx_buffer_table_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                                   hrx_buffer_t* out_buffer, size_t* out_offset,
                                   void** out_user_data) {
  if (!hrx_buffer_table_try_find(table, any_ptr, out_buffer, out_offset,
                                 out_user_data)) {
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "pointer not found in buffer table");
  }
  return hrx_ok_status();
}

// Returns the matching entry while |table->mutex| is held by the caller.
static hrx_buffer_table_entry_t* hrx_buffer_table_find_range_locked(
    hrx_buffer_table_t* table, uint64_t any_ptr, size_t size) {
  const size_t upper_bound = hrx_buffer_table_range_upper_bound(table, any_ptr);
  if (upper_bound == 0) {
    return NULL;
  }
  const hrx_buffer_table_range_index_t* range =
      &table->range_index[upper_bound - 1];
  hrx_buffer_table_entry_t* entry = &table->entries[range->entry_index];
  const uint64_t offset = any_ptr - range->base;
  return offset <= (uint64_t)entry->size &&
                 (uint64_t)size <= (uint64_t)entry->size - offset
             ? entry
             : NULL;
}

hrx_status_t hrx_buffer_table_find_range(hrx_buffer_table_t* table,
                                         uint64_t any_ptr, size_t size,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data) {
  if (out_buffer) {
    *out_buffer = NULL;
  }
  if (out_offset) {
    *out_offset = 0;
  }
  if (out_user_data) {
    *out_user_data = NULL;
  }

  if (size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "range size must be > 0");
  }
  if ((uint64_t)size > UINT64_MAX - any_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "range would overflow");
  }
  iree_slim_mutex_lock(&table->mutex);
  hrx_buffer_table_entry_t* entry =
      hrx_buffer_table_find_range_locked(table, any_ptr, size);
  if (entry) {
    hrx_buffer_table_fill_result(entry, any_ptr, out_buffer, out_offset,
                                 out_user_data);
    iree_slim_mutex_unlock(&table->mutex);
    return hrx_ok_status();
  }

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_make_status(HRX_STATUS_NOT_FOUND,
                         "no buffer contains the requested range");
}

hrx_status_t hrx_buffer_table_find_range_retain(
    hrx_buffer_table_t* table, uint64_t any_ptr, size_t size,
    hrx_buffer_table_retained_ref_t* out_ref) {
  return hrx_buffer_table_find_range_retain_if(
      table, any_ptr, size, /*callback=*/NULL, /*callback_user_data=*/NULL,
      out_ref);
}

hrx_status_t hrx_buffer_table_find_range_retain_if(
    hrx_buffer_table_t* table, uint64_t any_ptr, size_t size,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    hrx_buffer_table_retained_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));

  if (size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "range size must be > 0");
  }
  if ((uint64_t)size > UINT64_MAX - any_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "range would overflow");
  }

  iree_slim_mutex_lock(&table->mutex);
  hrx_buffer_table_entry_t* entry =
      hrx_buffer_table_find_range_locked(table, any_ptr, size);
  if (entry) {
    size_t offset = 0;
    hrx_buffer_table_fill_result(entry, any_ptr, NULL, &offset, NULL);
    hrx_status_t status = callback ? callback(entry, offset, callback_user_data)
                                   : hrx_ok_status();
    if (hrx_status_is_ok(status)) {
      hrx_buffer_table_retain_entry(entry, offset, out_ref);
    }
    iree_slim_mutex_unlock(&table->mutex);
    return status;
  }

  iree_slim_mutex_unlock(&table->mutex);
  return hrx_make_status(HRX_STATUS_NOT_FOUND,
                         "no buffer contains the requested range");
}

hrx_status_t hrx_buffer_table_find_ranges_retain_if(
    hrx_buffer_table_t* table, size_t request_count,
    const hrx_buffer_table_range_request_t* requests,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    size_t ref_capacity, hrx_buffer_table_retained_ref_t* out_refs,
    size_t* out_ref_count, hrx_buffer_table_range_match_t* out_matches) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_ref_count);
  *out_ref_count = 0;
  if (request_count == 0 || !requests || !out_matches || !out_refs) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "bulk range lookup requires non-empty storage");
  }
  if (ref_capacity < request_count) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "retained reference storage is too small");
  }
  for (size_t i = 0; i < request_count; ++i) {
    out_matches[i] = (hrx_buffer_table_range_match_t){
        .ref_index = SIZE_MAX,
        .offset = 0,
    };
    if (requests[i].length == 0 ||
        (uint64_t)requests[i].length > UINT64_MAX - requests[i].address) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "bulk lookup range is invalid");
    }
  }

  iree_slim_mutex_lock(&table->mutex);
  uint64_t generation = ++table->bulk_lookup_generation;
  if (IREE_UNLIKELY(generation == 0)) {
    for (size_t i = 0; i < table->count; ++i) {
      table->entries[i].bulk_lookup_generation = 0;
    }
    generation = ++table->bulk_lookup_generation;
  }

  hrx_status_t status = hrx_ok_status();
  size_t ref_count = 0;
  for (size_t i = 0; i < request_count && hrx_status_is_ok(status); ++i) {
    hrx_buffer_table_entry_t* entry = hrx_buffer_table_find_range_locked(
        table, requests[i].address, requests[i].length);
    if (!entry) {
      continue;
    }

    size_t offset = 0;
    hrx_buffer_table_fill_result(entry, requests[i].address, NULL, &offset,
                                 NULL);
    if (entry->bulk_lookup_generation != generation) {
      status = callback ? callback(entry, offset, callback_user_data)
                        : hrx_ok_status();
      if (!hrx_status_is_ok(status)) {
        break;
      }
      hrx_buffer_table_retain_entry(entry, offset, &out_refs[ref_count]);
      entry->bulk_lookup_generation = generation;
      entry->bulk_lookup_ref_index = ref_count++;
    }
    out_matches[i].ref_index = entry->bulk_lookup_ref_index;
    out_matches[i].offset = offset;
  }
  *out_ref_count = ref_count;
  iree_slim_mutex_unlock(&table->mutex);
  return status;
}

hrx_status_t hrx_buffer_table_remove_reserved_if(
    hrx_buffer_table_t* table, uint64_t any_ptr,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    hrx_buffer_table_entry_t* out_entry, size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(out_entry);
  memset(out_entry, 0, sizeof(*out_entry));
  if (out_offset) {
    *out_offset = 0;
  }

  iree_slim_mutex_lock(&table->mutex);
  const size_t index = hrx_buffer_table_find_index_locked(table, any_ptr);
  if (index >= table->count) {
    iree_slim_mutex_unlock(&table->mutex);
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "pointer not found in buffer table");
  }

  hrx_buffer_table_entry_t* entry = &table->entries[index];
  size_t offset = 0;
  hrx_buffer_table_fill_result(entry, any_ptr, NULL, &offset, NULL);
  hrx_status_t status =
      callback ? callback(entry, offset, callback_user_data) : hrx_ok_status();
  if (hrx_status_is_ok(status)) {
    *out_entry = *entry;
    if (out_offset) {
      *out_offset = offset;
    }
    const size_t last_index = table->count - 1;
    hrx_buffer_table_remove_entry_ranges_locked(table, index, last_index);
    if (index != last_index) {
      table->entries[index] = table->entries[last_index];
    }
    --table->count;
    ++table->reserved_insert_count;
  }
  iree_slim_mutex_unlock(&table->mutex);
  return status;
}
