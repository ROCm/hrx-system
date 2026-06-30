// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_cache_provider.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/threading/mutex.h"

typedef struct id4_pipeline_parameter_cache_entry_t {
  // Source scope identifying the upstream parameter table.
  iree_string_view_t source_scope;
  // Source key identifying the upstream parameter.
  iree_string_view_t key;
  // Byte offset within the source parameter.
  uint64_t parameter_offset;
  // Number of source bytes cached in |buffer|.
  iree_device_size_t length;
  // Device that owns |buffer| and |ready_semaphore|.
  iree_hal_device_t* device;
  // Queue affinity used to fill and copy |buffer|.
  iree_hal_queue_affinity_t queue_affinity;
  // Device-local cache buffer containing the exact source span.
  iree_hal_buffer_t* buffer;
  // Timeline semaphore signaled when |buffer| is ready to copy.
  iree_hal_semaphore_t* ready_semaphore;
  // Payload value on |ready_semaphore| that marks |buffer| ready.
  uint64_t ready_payload_value;
} id4_pipeline_parameter_cache_entry_t;

typedef struct id4_pipeline_parameter_cache_provider_t {
  // Base IREE parameter provider interface.
  iree_io_parameter_provider_t base;
  // Host allocator used for provider-owned storage.
  iree_allocator_t host_allocator;
  // Upstream provider containing the authoritative parameter data.
  iree_io_parameter_provider_t* source_provider;
  // Buffer parameters used for source cache allocations.
  iree_hal_buffer_params_t cache_params;
  // Maximum live cached source bytes, or zero for unbounded cache growth.
  iree_device_size_t maximum_cached_byte_length;
  // Miss policy controlling how uncached source spans are served.
  id4_pipeline_parameter_cache_miss_mode_t miss_mode;
  // Guards |entries| and entry metadata mutation.
  iree_slim_mutex_t mutex;
  // Number of live cache entries.
  iree_host_size_t entry_count;
  // Allocated capacity of |entries|.
  iree_host_size_t entry_capacity;
  // Sum of live source-resident cache buffer byte lengths.
  iree_device_size_t cached_byte_length;
  // Largest observed value of |cached_byte_length|.
  iree_device_size_t peak_cached_byte_length;
  // Number of upstream gather calls issued to fill cache entries.
  iree_host_size_t source_gather_count;
  // Number of upstream source bytes gathered to fill cache entries.
  iree_device_size_t source_gather_byte_length;
  // Number of caller gather requests served from existing cache entries.
  iree_host_size_t cache_reuse_count;
  // Number of caller source bytes served from existing cache entries.
  iree_device_size_t cache_reuse_byte_length;
  // Number of caller gather requests served directly under budget pressure.
  iree_host_size_t direct_miss_count;
  // Number of caller source bytes served directly under budget pressure.
  iree_device_size_t direct_miss_byte_length;
  // Number of cache entries evicted by budget pressure or notifications.
  iree_host_size_t evicted_entry_count;
  // Number of cached source bytes evicted by budget pressure or notifications.
  iree_device_size_t evicted_byte_length;
  // Source-resident exact-span cache entries.
  id4_pipeline_parameter_cache_entry_t* entries;
} id4_pipeline_parameter_cache_provider_t;

typedef struct id4_pipeline_parameter_cache_request_t {
  // Source key returned by the caller enumerator.
  iree_string_view_t key;
  // Source span and caller target span returned by the enumerator.
  iree_io_parameter_span_t span;
} id4_pipeline_parameter_cache_request_t;

typedef struct id4_pipeline_parameter_cache_ready_entry_t {
  // Cache buffer retained for recording a copy command, or NULL for direct.
  iree_hal_buffer_t* buffer;
  // Semaphore retained for the copy submission wait list.
  iree_hal_semaphore_t* ready_semaphore;
  // Payload value paired with |ready_semaphore|.
  uint64_t ready_payload_value;
} id4_pipeline_parameter_cache_ready_entry_t;

typedef struct id4_pipeline_parameter_cache_single_enumerator_t {
  // Key supplied to the upstream source provider.
  iree_string_view_t key;
  // Source span copied into a cache buffer beginning at offset zero.
  iree_io_parameter_span_t span;
} id4_pipeline_parameter_cache_single_enumerator_t;

static iree_status_t id4_pipeline_parameter_cache_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_parameter_cache_copy_string(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter cache string is too large to copy");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_parameter_cache_free_string(
    iree_string_view_t* value, iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static void id4_pipeline_parameter_cache_entry_deinitialize(
    id4_pipeline_parameter_cache_provider_t* provider,
    id4_pipeline_parameter_cache_entry_t* entry) {
  id4_pipeline_parameter_cache_free_string(&entry->source_scope,
                                           provider->host_allocator);
  id4_pipeline_parameter_cache_free_string(&entry->key,
                                           provider->host_allocator);
  iree_hal_semaphore_release(entry->ready_semaphore);
  iree_hal_buffer_release(entry->buffer);
  iree_hal_device_release(entry->device);
  memset(entry, 0, sizeof(*entry));
}

static void id4_pipeline_parameter_cache_evict_oldest_entry_locked(
    id4_pipeline_parameter_cache_provider_t* provider) {
  if (provider->entry_count == 0) return;
  const iree_device_size_t evicted_length = provider->entries[0].length;
  id4_pipeline_parameter_cache_entry_deinitialize(provider,
                                                  &provider->entries[0]);
  const iree_host_size_t remaining_count = provider->entry_count - 1;
  if (remaining_count > 0) {
    memmove(&provider->entries[0], &provider->entries[1],
            remaining_count * sizeof(provider->entries[0]));
  }
  --provider->entry_count;
  if (evicted_length > provider->cached_byte_length) {
    provider->cached_byte_length = 0;
  } else {
    provider->cached_byte_length -= evicted_length;
  }
  ++provider->evicted_entry_count;
  provider->evicted_byte_length += evicted_length;
}

static void id4_pipeline_parameter_cache_provider_evict_all(
    id4_pipeline_parameter_cache_provider_t* provider) {
  iree_slim_mutex_lock(&provider->mutex);
  provider->evicted_entry_count += provider->entry_count;
  provider->evicted_byte_length += provider->cached_byte_length;
  for (iree_host_size_t i = 0; i < provider->entry_count; ++i) {
    id4_pipeline_parameter_cache_entry_deinitialize(provider,
                                                    &provider->entries[i]);
  }
  provider->entry_count = 0;
  provider->cached_byte_length = 0;
  iree_slim_mutex_unlock(&provider->mutex);
}

static iree_status_t id4_pipeline_parameter_cache_validate_budget(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_device_size_t length) {
  if (provider->maximum_cached_byte_length == 0) return iree_ok_status();
  if (length <= provider->maximum_cached_byte_length) return iree_ok_status();
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "parameter cache request length %" PRIu64
                          " exceeds configured cache budget %" PRIu64,
                          length, provider->maximum_cached_byte_length);
}

static iree_status_t id4_pipeline_parameter_cache_evict_until_available_locked(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_cache_validate_budget(provider, length));
  if (provider->maximum_cached_byte_length == 0) return iree_ok_status();
  iree_device_size_t required_byte_length = 0;
  if (!iree_device_size_checked_add(provider->cached_byte_length, length,
                                    &required_byte_length)) {
    required_byte_length = IREE_DEVICE_SIZE_MAX;
  }
  while (provider->entry_count > 0 &&
         required_byte_length > provider->maximum_cached_byte_length) {
    id4_pipeline_parameter_cache_evict_oldest_entry_locked(provider);
    if (!iree_device_size_checked_add(provider->cached_byte_length, length,
                                      &required_byte_length)) {
      required_byte_length = IREE_DEVICE_SIZE_MAX;
    }
  }
  return iree_ok_status();
}

static void id4_pipeline_parameter_cache_ready_entries_release(
    iree_host_size_t count,
    id4_pipeline_parameter_cache_ready_entry_t* ready_entries) {
  if (!ready_entries) return;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_semaphore_release(ready_entries[i].ready_semaphore);
    iree_hal_buffer_release(ready_entries[i].buffer);
    memset(&ready_entries[i], 0, sizeof(ready_entries[i]));
  }
}

static bool id4_pipeline_parameter_cache_entry_matches(
    const id4_pipeline_parameter_cache_entry_t* entry,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_string_view_t source_scope, iree_string_view_t key,
    uint64_t parameter_offset, iree_device_size_t length) {
  return entry->device == device && entry->queue_affinity == queue_affinity &&
         entry->parameter_offset == parameter_offset &&
         entry->length == length &&
         iree_string_view_equal(entry->source_scope, source_scope) &&
         iree_string_view_equal(entry->key, key);
}

static iree_host_size_t id4_pipeline_parameter_cache_find_entry_locked(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_string_view_t source_scope, iree_string_view_t key,
    uint64_t parameter_offset, iree_device_size_t length) {
  for (iree_host_size_t i = 0; i < provider->entry_count; ++i) {
    if (id4_pipeline_parameter_cache_entry_matches(
            &provider->entries[i], device, queue_affinity, source_scope, key,
            parameter_offset, length)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static void id4_pipeline_parameter_cache_retain_ready_entry(
    const id4_pipeline_parameter_cache_entry_t* entry,
    id4_pipeline_parameter_cache_ready_entry_t* out_ready_entry) {
  iree_hal_buffer_retain(entry->buffer);
  iree_hal_semaphore_retain(entry->ready_semaphore);
  out_ready_entry->buffer = entry->buffer;
  out_ready_entry->ready_semaphore = entry->ready_semaphore;
  out_ready_entry->ready_payload_value = entry->ready_payload_value;
}

static bool id4_pipeline_parameter_cache_try_retain_ready_entry(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_string_view_t source_scope, iree_string_view_t key,
    uint64_t parameter_offset, iree_device_size_t length,
    id4_pipeline_parameter_cache_ready_entry_t* out_ready_entry) {
  iree_slim_mutex_lock(&provider->mutex);
  const iree_host_size_t entry_index =
      id4_pipeline_parameter_cache_find_entry_locked(
          provider, device, queue_affinity, source_scope, key, parameter_offset,
          length);
  if (entry_index != IREE_HOST_SIZE_MAX) {
    ++provider->cache_reuse_count;
    provider->cache_reuse_byte_length += length;
    id4_pipeline_parameter_cache_retain_ready_entry(
        &provider->entries[entry_index], out_ready_entry);
  }
  iree_slim_mutex_unlock(&provider->mutex);
  return entry_index != IREE_HOST_SIZE_MAX;
}

static bool id4_pipeline_parameter_cache_should_gather_direct(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_device_size_t length) {
  bool should_gather_direct = false;
  iree_slim_mutex_lock(&provider->mutex);
  if (provider->miss_mode ==
          ID4_PIPELINE_PARAMETER_CACHE_MISS_MODE_DIRECT_ON_PRESSURE &&
      provider->maximum_cached_byte_length != 0) {
    iree_device_size_t required_byte_length = 0;
    should_gather_direct =
        length > provider->maximum_cached_byte_length ||
        !iree_device_size_checked_add(provider->cached_byte_length, length,
                                      &required_byte_length) ||
        required_byte_length > provider->maximum_cached_byte_length;
  }
  iree_slim_mutex_unlock(&provider->mutex);
  return should_gather_direct;
}

static iree_status_t id4_pipeline_parameter_cache_single_enumerator(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  if (i != 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "single cache fill enumerator index %" PRIhsz " is out of range", i);
  }
  id4_pipeline_parameter_cache_single_enumerator_t* state =
      (id4_pipeline_parameter_cache_single_enumerator_t*)user_data;
  *out_key = state->key;
  *out_span = state->span;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_cache_fill_entry(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t key,
    uint64_t parameter_offset, iree_device_size_t length,
    iree_string_view_t* out_source_scope, iree_string_view_t* out_key,
    iree_hal_buffer_t** out_buffer, iree_hal_semaphore_t** out_ready_semaphore,
    uint64_t* out_ready_payload_value) {
  *out_source_scope = iree_string_view_empty();
  *out_key = iree_string_view_empty();
  *out_buffer = NULL;
  *out_ready_semaphore = NULL;
  *out_ready_payload_value = 1;

  iree_status_t status = id4_pipeline_parameter_cache_copy_string(
      source_scope, provider->host_allocator, out_source_scope);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_cache_copy_string(
        key, provider->host_allocator, out_key);
  }

  iree_hal_buffer_t* buffer = NULL;
  iree_hal_buffer_params_t cache_params = provider->cache_params;
  cache_params.queue_affinity = queue_affinity;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), cache_params, length, &buffer);
  }

  iree_hal_semaphore_t* ready_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, queue_affinity,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready_semaphore);
  }

  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_cache_single_enumerator_t enumerator_state = {
        .key = key,
        .span =
            {
                .parameter_offset = parameter_offset,
                .buffer_offset = 0,
                .length = length,
            },
    };
    iree_io_parameter_enumerator_t enumerator = {
        .fn = id4_pipeline_parameter_cache_single_enumerator,
        .user_data = &enumerator_state,
    };
    iree_hal_semaphore_t* ready_semaphore_ptr = ready_semaphore;
    uint64_t ready_payload_value = *out_ready_payload_value;
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &ready_semaphore_ptr,
        .payload_values = &ready_payload_value,
    };
    status = iree_io_parameter_provider_gather(
        provider->source_provider, device, queue_affinity, wait_semaphore_list,
        signal_list, source_scope, buffer, /*count=*/1, enumerator);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&provider->mutex);
      ++provider->source_gather_count;
      provider->source_gather_byte_length += length;
      iree_slim_mutex_unlock(&provider->mutex);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    *out_ready_semaphore = ready_semaphore;
  } else {
    iree_hal_semaphore_release(ready_semaphore);
    iree_hal_buffer_release(buffer);
    id4_pipeline_parameter_cache_free_string(out_key, provider->host_allocator);
    id4_pipeline_parameter_cache_free_string(out_source_scope,
                                             provider->host_allocator);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_cache_gather_direct(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t key,
    iree_io_parameter_span_t span, iree_hal_buffer_t* target_buffer,
    id4_pipeline_parameter_cache_ready_entry_t* out_ready_entry) {
  memset(out_ready_entry, 0, sizeof(*out_ready_entry));

  iree_hal_semaphore_t* ready_semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      device, queue_affinity,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready_semaphore);
  uint64_t ready_payload_value = 1;
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_cache_single_enumerator_t enumerator_state = {
        .key = key,
        .span = span,
    };
    iree_io_parameter_enumerator_t enumerator = {
        .fn = id4_pipeline_parameter_cache_single_enumerator,
        .user_data = &enumerator_state,
    };
    iree_hal_semaphore_t* ready_semaphore_ptr = ready_semaphore;
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &ready_semaphore_ptr,
        .payload_values = &ready_payload_value,
    };
    status = iree_io_parameter_provider_gather(
        provider->source_provider, device, queue_affinity, wait_semaphore_list,
        signal_list, source_scope, target_buffer, /*count=*/1, enumerator);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&provider->mutex);
    ++provider->direct_miss_count;
    provider->direct_miss_byte_length += span.length;
    iree_slim_mutex_unlock(&provider->mutex);
    out_ready_entry->ready_semaphore = ready_semaphore;
    out_ready_entry->ready_payload_value = ready_payload_value;
  } else {
    iree_hal_semaphore_release(ready_semaphore);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_cache_insert_or_retain_entry(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t parameter_offset, iree_device_size_t length,
    iree_string_view_t* inout_source_scope, iree_string_view_t* inout_key,
    iree_hal_buffer_t** inout_buffer,
    iree_hal_semaphore_t** inout_ready_semaphore, uint64_t ready_payload_value,
    id4_pipeline_parameter_cache_ready_entry_t* out_ready_entry) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&provider->mutex);
  const iree_host_size_t existing_index =
      id4_pipeline_parameter_cache_find_entry_locked(
          provider, device, queue_affinity, *inout_source_scope, *inout_key,
          parameter_offset, length);
  if (existing_index != IREE_HOST_SIZE_MAX) {
    ++provider->cache_reuse_count;
    provider->cache_reuse_byte_length += length;
    id4_pipeline_parameter_cache_retain_ready_entry(
        &provider->entries[existing_index], out_ready_entry);
  } else {
    if (provider->entry_count == provider->entry_capacity) {
      status = iree_allocator_grow_array(
          provider->host_allocator, provider->entry_count + 1,
          sizeof(provider->entries[0]), &provider->entry_capacity,
          (void**)&provider->entries);
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_cache_evict_until_available_locked(
          provider, length);
    }
    iree_device_size_t next_cached_byte_length = 0;
    if (iree_status_is_ok(status)) {
      if (!iree_device_size_checked_add(provider->cached_byte_length, length,
                                        &next_cached_byte_length)) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "parameter cache byte accounting overflow");
      }
    }
    if (iree_status_is_ok(status)) {
      id4_pipeline_parameter_cache_entry_t* entry =
          &provider->entries[provider->entry_count++];
      memset(entry, 0, sizeof(*entry));
      entry->source_scope = *inout_source_scope;
      entry->key = *inout_key;
      entry->parameter_offset = parameter_offset;
      entry->length = length;
      iree_hal_device_retain(device);
      entry->device = device;
      entry->queue_affinity = queue_affinity;
      entry->buffer = *inout_buffer;
      entry->ready_semaphore = *inout_ready_semaphore;
      entry->ready_payload_value = ready_payload_value;
      provider->cached_byte_length = next_cached_byte_length;
      if (provider->cached_byte_length > provider->peak_cached_byte_length) {
        provider->peak_cached_byte_length = provider->cached_byte_length;
      }
      *inout_source_scope = iree_string_view_empty();
      *inout_key = iree_string_view_empty();
      *inout_buffer = NULL;
      *inout_ready_semaphore = NULL;
      id4_pipeline_parameter_cache_retain_ready_entry(entry, out_ready_entry);
    }
  }
  iree_slim_mutex_unlock(&provider->mutex);
  return status;
}

static iree_status_t id4_pipeline_parameter_cache_get_ready_entry(
    id4_pipeline_parameter_cache_provider_t* provider,
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t key,
    iree_io_parameter_span_t span, iree_hal_buffer_t* target_buffer,
    id4_pipeline_parameter_cache_ready_entry_t* out_ready_entry) {
  memset(out_ready_entry, 0, sizeof(*out_ready_entry));
  if (id4_pipeline_parameter_cache_try_retain_ready_entry(
          provider, device, queue_affinity, source_scope, key,
          span.parameter_offset, span.length, out_ready_entry)) {
    return iree_ok_status();
  }
  if (id4_pipeline_parameter_cache_should_gather_direct(provider,
                                                        span.length)) {
    return id4_pipeline_parameter_cache_gather_direct(
        provider, device, queue_affinity, wait_semaphore_list, source_scope,
        key, span, target_buffer, out_ready_entry);
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_cache_validate_budget(provider, span.length));

  iree_string_view_t cached_source_scope = iree_string_view_empty();
  iree_string_view_t cached_key = iree_string_view_empty();
  iree_hal_buffer_t* cached_buffer = NULL;
  iree_hal_semaphore_t* ready_semaphore = NULL;
  uint64_t ready_payload_value = 0;
  iree_status_t status = id4_pipeline_parameter_cache_fill_entry(
      provider, device, queue_affinity, wait_semaphore_list, source_scope, key,
      span.parameter_offset, span.length, &cached_source_scope, &cached_key,
      &cached_buffer, &ready_semaphore, &ready_payload_value);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_cache_insert_or_retain_entry(
        provider, device, queue_affinity, span.parameter_offset, span.length,
        &cached_source_scope, &cached_key, &cached_buffer, &ready_semaphore,
        ready_payload_value, out_ready_entry);
  }

  iree_hal_semaphore_release(ready_semaphore);
  iree_hal_buffer_release(cached_buffer);
  id4_pipeline_parameter_cache_free_string(&cached_key,
                                           provider->host_allocator);
  id4_pipeline_parameter_cache_free_string(&cached_source_scope,
                                           provider->host_allocator);
  return status;
}

static iree_status_t id4_pipeline_parameter_cache_read_requests(
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    id4_pipeline_parameter_cache_request_t* requests) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(enumerator.fn(enumerator.user_data, i,
                                       &requests[i].key, &requests[i].span));
    if (iree_string_view_is_empty(requests[i].key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter cache request %" PRIhsz " key is required", i);
    }
    if (requests[i].span.length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter cache request %" PRIhsz " byte length must be nonzero", i);
    }
  }
  return iree_ok_status();
}

static bool id4_pipeline_parameter_cache_wait_contains(
    iree_host_size_t count, const iree_hal_semaphore_t* const* semaphores,
    iree_hal_semaphore_t* semaphore) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (semaphores[i] == semaphore) return true;
  }
  return false;
}

static iree_status_t id4_pipeline_parameter_cache_make_wait_list(
    const iree_hal_semaphore_list_t caller_wait_list,
    iree_host_size_t ready_entry_count,
    const id4_pipeline_parameter_cache_ready_entry_t* ready_entries,
    iree_allocator_t host_allocator, iree_hal_semaphore_list_t* out_wait_list) {
  memset(out_wait_list, 0, sizeof(*out_wait_list));
  const iree_host_size_t max_wait_count =
      caller_wait_list.count + ready_entry_count;
  if (max_wait_count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, max_wait_count,
                                  sizeof(semaphores[0]), (void**)&semaphores));
  uint64_t* payload_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, max_wait_count, sizeof(payload_values[0]),
      (void**)&payload_values);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, semaphores);
    return status;
  }

  iree_host_size_t wait_count = 0;
  for (iree_host_size_t i = 0; i < caller_wait_list.count; ++i) {
    semaphores[wait_count] = caller_wait_list.semaphores[i];
    payload_values[wait_count] = caller_wait_list.payload_values[i];
    ++wait_count;
  }
  for (iree_host_size_t i = 0; i < ready_entry_count; ++i) {
    if (id4_pipeline_parameter_cache_wait_contains(
            wait_count, (const iree_hal_semaphore_t* const*)semaphores,
            ready_entries[i].ready_semaphore)) {
      continue;
    }
    semaphores[wait_count] = ready_entries[i].ready_semaphore;
    payload_values[wait_count] = ready_entries[i].ready_payload_value;
    ++wait_count;
  }
  out_wait_list->count = wait_count;
  out_wait_list->semaphores = semaphores;
  out_wait_list->payload_values = payload_values;
  return iree_ok_status();
}

static void id4_pipeline_parameter_cache_free_wait_list(
    iree_allocator_t host_allocator, iree_hal_semaphore_list_t* wait_list) {
  iree_allocator_free(host_allocator, (void*)wait_list->payload_values);
  iree_allocator_free(host_allocator, wait_list->semaphores);
  memset(wait_list, 0, sizeof(*wait_list));
}

static iree_status_t id4_pipeline_parameter_cache_record_copies(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t request_count,
    const id4_pipeline_parameter_cache_request_t* requests,
    const id4_pipeline_parameter_cache_ready_entry_t* ready_entries,
    iree_hal_buffer_t* target_buffer,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue_affinity,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    if (!ready_entries[i].buffer) continue;
    status = iree_hal_command_buffer_copy_buffer(
        command_buffer,
        iree_hal_make_buffer_ref(ready_entries[i].buffer, /*offset=*/0,
                                 requests[i].span.length),
        iree_hal_make_buffer_ref(target_buffer, requests[i].span.buffer_offset,
                                 requests[i].span.length),
        IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

static void id4_pipeline_parameter_cache_provider_destroy(
    iree_io_parameter_provider_t* base_provider) {
  id4_pipeline_parameter_cache_provider_t* provider =
      (id4_pipeline_parameter_cache_provider_t*)base_provider;
  iree_allocator_t host_allocator = provider->host_allocator;
  id4_pipeline_parameter_cache_provider_evict_all(provider);
  iree_allocator_free(host_allocator, provider->entries);
  iree_slim_mutex_deinitialize(&provider->mutex);
  iree_io_parameter_provider_release(provider->source_provider);
  iree_allocator_free(host_allocator, provider);
}

static iree_status_t id4_pipeline_parameter_cache_provider_notify(
    iree_io_parameter_provider_t* base_provider,
    iree_io_parameter_provider_signal_t signal) {
  id4_pipeline_parameter_cache_provider_t* provider =
      (id4_pipeline_parameter_cache_provider_t*)base_provider;
  iree_status_t status =
      iree_io_parameter_provider_notify(provider->source_provider, signal);
  switch (signal) {
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND:
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY:
      id4_pipeline_parameter_cache_provider_evict_all(provider);
      break;
    default:
      break;
  }
  return status;
}

static bool id4_pipeline_parameter_cache_provider_query_support(
    iree_io_parameter_provider_t* base_provider, iree_string_view_t scope) {
  id4_pipeline_parameter_cache_provider_t* provider =
      (id4_pipeline_parameter_cache_provider_t*)base_provider;
  return iree_io_parameter_provider_query_support(provider->source_provider,
                                                  scope);
}

static iree_status_t id4_pipeline_parameter_cache_provider_load(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  id4_pipeline_parameter_cache_provider_t* provider =
      (id4_pipeline_parameter_cache_provider_t*)base_provider;
  return iree_io_parameter_provider_load(
      provider->source_provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_scope, target_params, count, enumerator,
      emitter);
}

static iree_status_t id4_pipeline_parameter_cache_provider_gather(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  id4_pipeline_parameter_cache_provider_t* provider =
      (id4_pipeline_parameter_cache_provider_t*)base_provider;

  id4_pipeline_parameter_cache_request_t* requests = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      provider->host_allocator, count, sizeof(requests[0]), (void**)&requests);
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_cache_read_requests(count, enumerator, requests);
  }

  id4_pipeline_parameter_cache_ready_entry_t* ready_entries = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(provider->host_allocator, count,
                                         sizeof(ready_entries[0]),
                                         (void**)&ready_entries);
  }
  iree_host_size_t ready_entry_count = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = id4_pipeline_parameter_cache_get_ready_entry(
        provider, device, queue_affinity, wait_semaphore_list, source_scope,
        requests[i].key, requests[i].span, target_buffer, &ready_entries[i]);
    if (iree_status_is_ok(status)) {
      ready_entry_count = i + 1;
    }
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_cache_record_copies(
        device, queue_affinity, count, requests, ready_entries, target_buffer,
        &command_buffer);
  }

  iree_hal_semaphore_list_t target_wait_list;
  memset(&target_wait_list, 0, sizeof(target_wait_list));
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_cache_make_wait_list(
        wait_semaphore_list, count, ready_entries, provider->host_allocator,
        &target_wait_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, target_wait_list, signal_semaphore_list,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  id4_pipeline_parameter_cache_free_wait_list(provider->host_allocator,
                                              &target_wait_list);
  iree_hal_command_buffer_release(command_buffer);
  id4_pipeline_parameter_cache_ready_entries_release(ready_entry_count,
                                                     ready_entries);
  iree_allocator_free(provider->host_allocator, ready_entries);
  iree_allocator_free(provider->host_allocator, requests);
  return status;
}

static iree_status_t id4_pipeline_parameter_cache_provider_scatter(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  (void)base_provider;
  (void)device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)target_scope;
  (void)count;
  (void)enumerator;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "source-resident parameter cache provider does not support scatter");
}

static const iree_io_parameter_provider_vtable_t
    id4_pipeline_parameter_cache_provider_vtable = {
        .destroy = id4_pipeline_parameter_cache_provider_destroy,
        .notify = id4_pipeline_parameter_cache_provider_notify,
        .query_support = id4_pipeline_parameter_cache_provider_query_support,
        .load = id4_pipeline_parameter_cache_provider_load,
        .gather = id4_pipeline_parameter_cache_provider_gather,
        .scatter = id4_pipeline_parameter_cache_provider_scatter,
};

static iree_status_t id4_pipeline_parameter_cache_provider_cast(
    iree_io_parameter_provider_t* base_provider,
    id4_pipeline_parameter_cache_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (!base_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter cache provider is required");
  }
  if (base_provider->vtable != &id4_pipeline_parameter_cache_provider_vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter provider is not a source-resident "
                            "parameter cache provider");
  }
  *out_provider = (id4_pipeline_parameter_cache_provider_t*)base_provider;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_cache_provider_validate_options(
    const id4_pipeline_parameter_cache_provider_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter cache provider options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_cache_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("parameter cache provider")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter cache provider extension structures are not supported");
  }
  if (!options->source_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter cache provider source provider is required");
  }
  if (options->cache_params.queue_affinity != IREE_HAL_QUEUE_AFFINITY_ANY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter cache provider cache params must use "
                            "IREE_HAL_QUEUE_AFFINITY_ANY");
  }
  if (!iree_any_bit_set(options->cache_params.type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter cache provider cache memory must be "
                            "device visible");
  }
  if (!iree_all_bits_set(options->cache_params.usage,
                         IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter cache provider cache usage must include "
                            "transfer source and target");
  }
  switch (options->miss_mode) {
    case ID4_PIPELINE_PARAMETER_CACHE_MISS_MODE_RETAIN:
    case ID4_PIPELINE_PARAMETER_CACHE_MISS_MODE_DIRECT_ON_PRESSURE:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter cache provider miss mode is invalid");
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_cache_provider_create(
    const id4_pipeline_parameter_cache_provider_options_t* options,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_cache_provider_validate_options(options));

  id4_pipeline_parameter_cache_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &id4_pipeline_parameter_cache_provider_vtable;
  provider->host_allocator = host_allocator;
  provider->cache_params = options->cache_params;
  provider->maximum_cached_byte_length = options->maximum_cached_byte_length;
  provider->miss_mode = options->miss_mode;
  iree_slim_mutex_initialize(&provider->mutex);
  iree_io_parameter_provider_retain(options->source_provider);
  provider->source_provider = options->source_provider;
  *out_provider = &provider->base;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_cache_provider_query_statistics(
    iree_io_parameter_provider_t* base_provider,
    id4_pipeline_parameter_cache_provider_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));

  id4_pipeline_parameter_cache_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_cache_provider_cast(base_provider, &provider));

  iree_slim_mutex_lock(&provider->mutex);
  out_statistics->entry_count = provider->entry_count;
  out_statistics->cached_byte_length = provider->cached_byte_length;
  out_statistics->peak_cached_byte_length = provider->peak_cached_byte_length;
  out_statistics->maximum_cached_byte_length =
      provider->maximum_cached_byte_length;
  out_statistics->source_gather_count = provider->source_gather_count;
  out_statistics->source_gather_byte_length =
      provider->source_gather_byte_length;
  out_statistics->cache_reuse_count = provider->cache_reuse_count;
  out_statistics->cache_reuse_byte_length = provider->cache_reuse_byte_length;
  out_statistics->direct_miss_count = provider->direct_miss_count;
  out_statistics->direct_miss_byte_length = provider->direct_miss_byte_length;
  out_statistics->evicted_entry_count = provider->evicted_entry_count;
  out_statistics->evicted_byte_length = provider->evicted_byte_length;
  iree_slim_mutex_unlock(&provider->mutex);
  return iree_ok_status();
}
