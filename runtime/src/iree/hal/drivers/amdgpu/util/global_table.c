// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/global_table.h"

#include "iree/base/internal/math.h"

typedef struct iree_hal_amdgpu_global_table_entry_t {
  // Executable-local handle value assigned when the entry is interned.
  uint64_t handle_value;

  // Persistent table-owned name storage.
  iree_string_view_t name;

  // Byte length verified from the first loaded physical device.
  iree_device_size_t byte_length;

  // One executable-owned buffer alias per physical device.
  iree_hal_buffer_t** device_buffers;
} iree_hal_amdgpu_global_table_entry_t;

static iree_status_t iree_hal_amdgpu_global_table_validate_params(
    const iree_hal_amdgpu_global_table_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);

  if (IREE_UNLIKELY(params->physical_device_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU global table requires at least one "
                            "physical device");
  }
  if (IREE_UNLIKELY(params->physical_device_count > IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU global table physical device count %" PRIhsz
                            " exceeds queue affinity capacity %" PRIhsz,
                            params->physical_device_count,
                            (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }
  if (IREE_UNLIKELY(params->loaded_physical_device_mask == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU global table requires a loaded device");
  }

  uint64_t valid_physical_device_mask = UINT64_MAX;
  if (params->physical_device_count < 64) {
    valid_physical_device_mask =
        (((uint64_t)1) << params->physical_device_count) - 1;
  }
  if (IREE_UNLIKELY(params->loaded_physical_device_mask &
                    ~valid_physical_device_mask)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU global table loaded device mask 0x%" PRIx64
                            " exceeds physical device count %" PRIhsz,
                            params->loaded_physical_device_mask,
                            params->physical_device_count);
  }

  if (IREE_UNLIKELY(!params->resolver.try_verify ||
                    !params->resolver.create_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU global table resolver is incomplete");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_global_table_first_loaded_device(
    const iree_hal_amdgpu_global_table_t* table,
    iree_host_size_t* out_physical_device_ordinal) {
  if (IREE_UNLIKELY(table->loaded_physical_device_mask == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU global table has no loaded devices");
  }
  *out_physical_device_ordinal =
      (iree_host_size_t)iree_math_count_trailing_zeros_u64(
          table->loaded_physical_device_mask);
  return iree_ok_status();
}

static iree_hal_amdgpu_global_table_entry_t*
iree_hal_amdgpu_global_table_find_entry_locked(
    iree_hal_amdgpu_global_table_t* table, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < table->entry_count; ++i) {
    iree_hal_amdgpu_global_table_entry_t* entry = table->entries[i];
    if (iree_string_view_equal(entry->name, name)) return entry;
  }
  return NULL;
}

static iree_hal_amdgpu_global_table_entry_t*
iree_hal_amdgpu_global_table_entry_from_handle_locked(
    iree_hal_amdgpu_global_table_t* table,
    iree_hal_executable_global_t global) {
  if (!iree_hal_executable_global_is_valid(global)) return NULL;
  if (global.value >= table->entry_count) return NULL;
  return table->entries[global.value];
}

static iree_status_t iree_hal_amdgpu_global_table_grow_entries_locked(
    iree_hal_amdgpu_global_table_t* table, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= table->entry_capacity) return iree_ok_status();

  iree_host_size_t new_capacity =
      table->entry_capacity ? table->entry_capacity : 4;
  while (new_capacity < minimum_capacity) {
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(new_capacity, 2, &new_capacity))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU global table capacity overflow");
    }
  }

  iree_hal_amdgpu_global_table_entry_t** entries = table->entries;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc_array(table->host_allocator, new_capacity,
                                   sizeof(entries[0]), (void**)&entries));
  memset(entries + table->entry_capacity, 0,
         (new_capacity - table->entry_capacity) * sizeof(entries[0]));
  table->entries = entries;
  table->entry_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_global_table_entry_allocate(
    iree_hal_amdgpu_global_table_t* table, iree_string_view_t name,
    iree_device_size_t byte_length,
    iree_hal_amdgpu_global_table_entry_t** out_entry) {
  *out_entry = NULL;

  iree_host_size_t name_storage_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(name.size, 1, &name_storage_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU global table name storage overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t device_buffers_offset = 0;
  iree_host_size_t name_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_global_table_entry_t), &total_size,
      IREE_STRUCT_FIELD(table->physical_device_count, iree_hal_buffer_t*,
                        &device_buffers_offset),
      IREE_STRUCT_FIELD(name_storage_size, char, &name_offset)));

  iree_hal_amdgpu_global_table_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(table->host_allocator, total_size, (void**)&entry));
  memset(entry, 0, total_size);

  entry->handle_value = IREE_HAL_EXECUTABLE_GLOBAL_INVALID_VALUE;
  entry->byte_length = byte_length;
  entry->device_buffers =
      (iree_hal_buffer_t**)((uint8_t*)entry + device_buffers_offset);

  char* name_storage = (char*)entry + name_offset;
  memcpy(name_storage, name.data, name.size);
  name_storage[name.size] = 0;
  entry->name = iree_make_string_view(name_storage, name.size);

  *out_entry = entry;
  return iree_ok_status();
}

static void iree_hal_amdgpu_global_table_entry_free(
    iree_hal_amdgpu_global_table_t* table,
    iree_hal_amdgpu_global_table_entry_t* entry) {
  if (!entry) return;
  for (iree_host_size_t i = 0; i < table->physical_device_count; ++i) {
    iree_hal_buffer_release(entry->device_buffers[i]);
  }
  iree_allocator_free(table->host_allocator, entry);
}

iree_status_t iree_hal_amdgpu_global_table_initialize(
    const iree_hal_amdgpu_global_table_params_t* params,
    iree_hal_amdgpu_global_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_validate_params(params));

  memset(out_table, 0, sizeof(*out_table));
  out_table->initialized = true;
  out_table->host_allocator = params->host_allocator;
  out_table->queue_affinity_domain = params->queue_affinity_domain;
  out_table->loaded_physical_device_mask = params->loaded_physical_device_mask;
  out_table->physical_device_count = params->physical_device_count;
  out_table->resolver = params->resolver;
  iree_slim_mutex_initialize(&out_table->mutex);
  return iree_ok_status();
}

void iree_hal_amdgpu_global_table_deinitialize(
    iree_hal_amdgpu_global_table_t* table) {
  if (!table || !table->initialized) return;
  for (iree_host_size_t i = 0; i < table->entry_count; ++i) {
    iree_hal_amdgpu_global_table_entry_free(table, table->entries[i]);
  }
  iree_allocator_free(table->host_allocator, table->entries);
  iree_slim_mutex_deinitialize(&table->mutex);
  memset(table, 0, sizeof(*table));
}

iree_status_t iree_hal_amdgpu_global_table_try_lookup(
    iree_hal_amdgpu_global_table_t* table, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_found);
  IREE_ASSERT_ARGUMENT(out_global);
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();

  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable global name is empty");
  }

  iree_slim_mutex_lock(&table->mutex);
  iree_hal_amdgpu_global_table_entry_t* entry =
      iree_hal_amdgpu_global_table_find_entry_locked(table, name);
  if (entry) {
    *out_found = true;
    *out_global = iree_hal_executable_global_from_value(entry->handle_value);
    iree_slim_mutex_unlock(&table->mutex);
    return iree_ok_status();
  }
  iree_slim_mutex_unlock(&table->mutex);

  iree_host_size_t verification_physical_device_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_first_loaded_device(
      table, &verification_physical_device_ordinal));

  iree_device_size_t byte_length = 0;
  bool resolved = false;
  IREE_RETURN_IF_ERROR(table->resolver.try_verify(
      table->resolver.user_data, name, verification_physical_device_ordinal,
      &resolved, &byte_length));
  if (!resolved) return iree_ok_status();

  iree_hal_amdgpu_global_table_entry_t* new_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_global_table_entry_allocate(
      table, name, byte_length, &new_entry));

  iree_slim_mutex_lock(&table->mutex);
  entry = iree_hal_amdgpu_global_table_find_entry_locked(table, name);
  if (entry) {
    *out_found = true;
    *out_global = iree_hal_executable_global_from_value(entry->handle_value);
  } else {
    const uint64_t handle_value = table->entry_count;
    iree_status_t status = iree_hal_amdgpu_global_table_grow_entries_locked(
        table, table->entry_count + 1);
    if (iree_status_is_ok(status)) {
      new_entry->handle_value = handle_value;
      table->entries[table->entry_count++] = new_entry;
      *out_found = true;
      *out_global = iree_hal_executable_global_from_value(handle_value);
      new_entry = NULL;
    }
    iree_slim_mutex_unlock(&table->mutex);
    if (new_entry) {
      iree_hal_amdgpu_global_table_entry_free(table, new_entry);
    }
    return status;
  }
  iree_slim_mutex_unlock(&table->mutex);

  iree_hal_amdgpu_global_table_entry_free(table, new_entry);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_global_table_lookup(
    iree_hal_amdgpu_global_table_t* table, iree_string_view_t name,
    iree_hal_executable_global_t* out_global) {
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_global_table_try_lookup(table, name, &found, out_global));
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable global `%.*s` not found",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_global_table_info(
    iree_hal_amdgpu_global_table_t* table, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_info);
  memset(out_info, 0, sizeof(*out_info));

  iree_slim_mutex_lock(&table->mutex);
  iree_hal_amdgpu_global_table_entry_t* entry =
      iree_hal_amdgpu_global_table_entry_from_handle_locked(table, global);
  if (!entry) {
    iree_slim_mutex_unlock(&table->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU executable global handle");
  }
  out_info->name = entry->name;
  out_info->byte_length = entry->byte_length;
  iree_slim_mutex_unlock(&table->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_global_table_buffer(
    iree_hal_amdgpu_global_table_t* table, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_hal_queue_affinity_t requested_queue_affinity = queue_affinity;
  if (iree_hal_queue_affinity_is_empty(requested_queue_affinity)) {
    requested_queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  }

  iree_hal_queue_affinity_t selected_queue_affinity = 0;
  iree_host_size_t physical_device_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
          table->queue_affinity_domain, requested_queue_affinity,
          &selected_queue_affinity, &physical_device_ordinal));

  if (IREE_UNLIKELY(((((uint64_t)1) << physical_device_ordinal) &
                     table->loaded_physical_device_mask) == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU executable global buffer requested on unloaded device %" PRIhsz,
        physical_device_ordinal);
  }

  iree_slim_mutex_lock(&table->mutex);
  iree_hal_amdgpu_global_table_entry_t* entry =
      iree_hal_amdgpu_global_table_entry_from_handle_locked(table, global);
  if (!entry) {
    iree_slim_mutex_unlock(&table->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU executable global handle");
  }
  iree_hal_buffer_t* cached_buffer =
      entry->device_buffers[physical_device_ordinal];
  if (cached_buffer) {
    *out_buffer = cached_buffer;
    iree_slim_mutex_unlock(&table->mutex);
    return iree_ok_status();
  }
  iree_string_view_t name = entry->name;
  iree_device_size_t byte_length = entry->byte_length;
  iree_slim_mutex_unlock(&table->mutex);

  iree_hal_buffer_t* new_buffer = NULL;
  IREE_RETURN_IF_ERROR(table->resolver.create_buffer(
      table->resolver.user_data, name, byte_length, selected_queue_affinity,
      physical_device_ordinal, &new_buffer));

  iree_slim_mutex_lock(&table->mutex);
  entry = iree_hal_amdgpu_global_table_entry_from_handle_locked(table, global);
  if (!entry) {
    iree_slim_mutex_unlock(&table->mutex);
    iree_hal_buffer_release(new_buffer);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU executable global handle");
  }
  cached_buffer = entry->device_buffers[physical_device_ordinal];
  if (cached_buffer) {
    *out_buffer = cached_buffer;
    iree_slim_mutex_unlock(&table->mutex);
    iree_hal_buffer_release(new_buffer);
  } else {
    entry->device_buffers[physical_device_ordinal] = new_buffer;
    *out_buffer = new_buffer;
    iree_slim_mutex_unlock(&table->mutex);
  }
  return iree_ok_status();
}
