// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_provider.h"

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"

void iree_hal_slab_provider_initialize(
    const iree_hal_slab_provider_vtable_t* vtable,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t* provider) {
  iree_atomic_ref_count_init(&provider->ref_count);
  provider->vtable = vtable;
  provider->host_allocator = host_allocator;
}

void iree_hal_slab_provider_retain(iree_hal_slab_provider_t* provider) {
  if (IREE_LIKELY(provider)) {
    iree_atomic_ref_count_inc(&provider->ref_count);
  }
}

void iree_hal_slab_provider_release(iree_hal_slab_provider_t* provider) {
  if (IREE_LIKELY(provider) &&
      iree_atomic_ref_count_dec(&provider->ref_count) == 1) {
    provider->vtable->destroy(provider);
  }
}

void iree_hal_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  request->operation = (iree_hal_slab_operation_t){
      .provider = provider,
      .type = IREE_HAL_SLAB_OPERATION_ACQUIRE,
  };
  provider->vtable->acquire_slabs(provider, request);
}

void iree_hal_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  request->operation = (iree_hal_slab_operation_t){
      .provider = provider,
      .type = IREE_HAL_SLAB_OPERATION_RELEASE,
  };
  provider->vtable->release_slabs(provider, request);
}

void iree_hal_slab_provider_acquire_slabs_inline(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request,
    iree_status_t (*acquire_slab)(iree_hal_slab_provider_t* provider,
                                  iree_device_size_t min_length,
                                  iree_device_size_t min_alignment,
                                  iree_hal_slab_t* out_slab),
    void (*release_slab)(iree_hal_slab_provider_t* provider,
                         const iree_hal_slab_t* slab)) {
  memset(request->slabs, 0, request->count * sizeof(*request->slabs));
  iree_status_t status = iree_ok_status();
  iree_host_size_t acquired_count = 0;
  while (acquired_count < request->count && iree_status_is_ok(status)) {
    status =
        acquire_slab(provider, request->requests[acquired_count].min_length,
                     request->requests[acquired_count].min_alignment,
                     &request->slabs[acquired_count]);
    if (iree_status_is_ok(status)) {
      ++acquired_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      release_slab(provider, &request->slabs[i]);
    }
    memset(request->slabs, 0, request->count * sizeof(*request->slabs));
  }
  request->completion.fn(request->completion.user_data, status);
}

void iree_hal_slab_provider_release_slabs_inline(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request,
    void (*release_slab)(iree_hal_slab_provider_t* provider,
                         const iree_hal_slab_t* slab)) {
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    release_slab(provider, &request->slabs[i]);
  }
  request->completion.fn(request->completion.user_data);
}

// The mutex joins the callback's final notification access before stack
// storage can disappear. Only explicit synchronous conveniences use this.
typedef struct iree_hal_slab_provider_sync_t {
  // Protects terminal publication and the notification's last access.
  iree_slim_mutex_t mutex;
  // Wakes the synchronous caller after completion.
  iree_notification_t notification;
  // True after the callback has published the terminal result.
  bool complete;
  // Owned acquisition failure, transferred to the caller.
  iree_status_t status;
} iree_hal_slab_provider_sync_t;

static void iree_hal_slab_provider_sync_acquired(void* user_data,
                                                 iree_status_t status) {
  iree_hal_slab_provider_sync_t* call = user_data;
  iree_slim_mutex_lock(&call->mutex);
  call->status = status;
  call->complete = true;
  iree_notification_post(&call->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&call->mutex);
}

static void iree_hal_slab_provider_sync_released(void* user_data) {
  iree_hal_slab_provider_sync_acquired(user_data, iree_ok_status());
}

static bool iree_hal_slab_provider_sync_is_complete(void* user_data) {
  iree_hal_slab_provider_sync_t* call = user_data;
  iree_slim_mutex_lock(&call->mutex);
  const bool complete = call->complete;
  iree_slim_mutex_unlock(&call->mutex);
  return complete;
}

iree_status_t iree_hal_slab_provider_acquire_slab_sync(
    iree_hal_slab_provider_t* provider, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
  iree_hal_slab_provider_sync_t call = {0};
  iree_slim_mutex_initialize(&call.mutex);
  iree_notification_initialize(&call.notification);
  const iree_hal_slab_request_t geometry = {min_length, min_alignment};
  iree_hal_slab_acquire_request_t request = {
      .count = 1,
      .requests = &geometry,
      .slabs = out_slab,
      .completion = {iree_hal_slab_provider_sync_acquired, &call},
  };
  iree_hal_slab_provider_acquire_slabs(provider, &request);
  iree_notification_await(&call.notification,
                          iree_hal_slab_provider_sync_is_complete, &call,
                          iree_infinite_timeout());
  iree_notification_deinitialize(&call.notification);
  iree_slim_mutex_deinitialize(&call.mutex);
  return call.status;
}

void iree_hal_slab_provider_release_slab_sync(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab) {
  iree_hal_slab_provider_sync_t call = {0};
  iree_slim_mutex_initialize(&call.mutex);
  iree_notification_initialize(&call.notification);
  iree_hal_slab_t consumed_slab = *slab;
  iree_hal_slab_release_request_t request = {
      .count = 1,
      .slabs = &consumed_slab,
      .completion = {iree_hal_slab_provider_sync_released, &call},
  };
  iree_hal_slab_provider_release_slabs(provider, &request);
  iree_notification_await(&call.notification,
                          iree_hal_slab_provider_sync_is_complete, &call,
                          iree_infinite_timeout());
  iree_notification_deinitialize(&call.notification);
  iree_slim_mutex_deinitialize(&call.mutex);
}

iree_hal_external_storage_t iree_hal_slab_provider_registration_view(
    const iree_hal_external_storage_t* storage) {
  const iree_device_size_t byte_offset = storage->maintenance_range.byte_offset;
  const iree_device_size_t byte_length =
      storage->maintenance_range.byte_length == IREE_HAL_WHOLE_BUFFER
          ? storage->byte_length - byte_offset
          : storage->maintenance_range.byte_length;
  return (iree_hal_external_storage_t){
      .storage_class = storage->storage_class,
      .payload = {.host_pointer =
                      storage->payload.host_pointer
                          ? (uint8_t*)storage->payload.host_pointer +
                                byte_offset
                          : NULL},
      .byte_length = byte_length,
      .host_cacheability = storage->host_cacheability,
      .maintenance_range = {0, byte_length},
      .physical_backing_id = storage->physical_backing_id,
  };
}

iree_status_t iree_hal_slab_provider_import_slab(
    iree_hal_slab_provider_t* provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(storage);
  IREE_ASSERT_ARGUMENT(out_import);
  memset(out_import, 0, sizeof(*out_import));
  if (!provider->vtable->import_slab) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "slab provider cannot import external storage");
  }
  return provider->vtable->import_slab(provider, storage, out_import);
}

bool iree_hal_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_requirements);
  memset(out_requirements, 0, sizeof(*out_requirements));
  return provider->vtable->query_import_requirements &&
         provider->vtable->query_import_requirements(provider, storage_class,
                                                     out_requirements);
}

iree_status_t iree_hal_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (slab_offset > slab->length ||
      allocation_size > slab->length - slab_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab buffer offset %" PRIdsz
                            " with length %" PRIdsz
                            " is outside slab length %" PRIdsz,
                            slab_offset, allocation_size, slab->length);
  }
  return provider->vtable->wrap_buffer(
      provider, slab, slab_offset, allocation_size, allowed_access,
      allowed_usage, release_callback, out_buffer);
}

static bool iree_hal_slab_provider_asan_range_is_valid(
    const iree_hal_slab_t* slab, iree_device_size_t backing_offset,
    const iree_hal_asan_allocation_layout_t* layout) {
  if (layout->backing_offset_alignment == 0 ||
      !iree_device_size_is_power_of_two(layout->backing_offset_alignment)) {
    return false;
  }
  if (layout->backing_length_alignment == 0 ||
      !iree_device_size_is_power_of_two(layout->backing_length_alignment)) {
    return false;
  }
  if (!iree_device_size_has_alignment(backing_offset,
                                      layout->backing_offset_alignment)) {
    return false;
  }
  if (layout->backing_length == 0 ||
      !iree_device_size_has_alignment(layout->backing_length,
                                      layout->backing_length_alignment)) {
    return false;
  }
  if (layout->user_offset > layout->backing_length ||
      layout->user_length > layout->backing_length - layout->user_offset) {
    return false;
  }
  if (backing_offset > slab->length ||
      layout->backing_length > slab->length - backing_offset) {
    return false;
  }
  return true;
}

static bool iree_hal_slab_provider_asan_advice_flags_are_valid(
    iree_hal_asan_range_advice_flags_t advice_flags) {
  switch (advice_flags) {
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED:
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED:
      return true;
    default:
      return false;
  }
}

iree_status_t iree_hal_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* provider,
    const iree_hal_asan_pool_options_t* options) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(options);
  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(options));
  if (!iree_hal_asan_pool_options_is_enabled(options)) {
    return iree_ok_status();
  }
  return provider->vtable->validate_asan_options(provider, options);
}

void iree_hal_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT(iree_hal_slab_provider_asan_advice_flags_are_valid(advice_flags),
              "unsupported ASAN range advice flags 0x%x", advice_flags);
  IREE_ASSERT(
      iree_hal_slab_provider_asan_range_is_valid(slab, backing_offset, layout),
      "invalid ASAN backing range advice");
  provider->vtable->advise_asan_range(provider, slab, backing_offset,
                                      advice_flags, layout);
}

void iree_hal_slab_provider_prefault(iree_hal_slab_provider_t* provider,
                                     const iree_hal_slab_t* slab,
                                     iree_device_size_t offset,
                                     iree_device_size_t length) {
  IREE_ASSERT(offset <= slab->length && length <= slab->length - offset);
  if (!length) {
    return;
  }
  provider->vtable->prefault(provider, slab, offset, length);
}

void iree_hal_slab_provider_trim(iree_hal_slab_provider_t* provider,
                                 iree_hal_pool_trim_flags_t flags) {
  provider->vtable->trim(provider, flags);
}

void iree_hal_slab_provider_query_stats(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  provider->vtable->query_stats(provider, visited, out_stats);
}

void iree_hal_slab_provider_query_properties(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  memset(out_properties, 0, sizeof(*out_properties));
  out_properties->allocation_alignment = 1;
  out_properties->max_allocation_alignment = 1;
  out_properties->maintenance_alignment = 1;
  provider->vtable->query_properties(provider, out_properties);
}

bool iree_hal_slab_provider_visited(
    iree_hal_slab_provider_visited_set_t* visited,
    const iree_hal_slab_provider_t* provider) {
  for (iree_host_size_t i = 0; i < visited->count; ++i) {
    if (visited->providers[i] == provider) {
      return true;
    }
  }
  if (visited->count >= IREE_HAL_SLAB_PROVIDER_MAX_VISITED) {
    return true;
  }
  visited->providers[visited->count++] = provider;
  return false;
}

static iree_status_t iree_hal_slab_buffer_validate_asan(
    void* user_data, const iree_hal_asan_pool_options_t* options) {
  const iree_hal_slab_buffer_backing_t* backing = user_data;
  return iree_hal_slab_provider_validate_asan_options(backing->provider,
                                                      options);
}

static void iree_hal_slab_buffer_advise_asan(
    void* user_data, iree_device_size_t offset,
    iree_hal_asan_range_advice_flags_t flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  const iree_hal_slab_buffer_backing_t* backing = user_data;
  iree_hal_slab_provider_advise_asan_range(backing->provider, backing->slab,
                                           offset, flags, layout);
}

static void iree_hal_slab_buffer_prefault(void* user_data,
                                          iree_device_size_t offset,
                                          iree_device_size_t length) {
  const iree_hal_slab_buffer_backing_t* backing = user_data;
  iree_hal_slab_provider_prefault(backing->provider, backing->slab, offset,
                                  length);
}

void iree_hal_slab_buffer_backing_initialize(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    const iree_hal_slab_provider_properties_t* properties,
    iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* tracker,
    iree_hal_slab_buffer_backing_t* out_backing) {
  out_backing->provider = provider;
  out_backing->slab = slab;
  out_backing->advice = (iree_hal_buffer_range_advice_t){
      .user_data = out_backing,
      .validate_asan = iree_hal_slab_buffer_validate_asan,
      .prefault = iree_hal_slab_buffer_prefault,
      .advise_asan = iree_hal_slab_buffer_advise_asan,
  };
  out_backing->facts = (iree_hal_buffer_backing_facts_t){
      .notification = notification,
      .tracker = tracker,
      .advice = &out_backing->advice,
      .maintenance_range = slab->maintenance_range,
      .host_cacheability = slab->host_cacheability,
      .physical_backing_id = slab->physical_backing_id,
      .native_allocation = slab->native_allocation,
      .allocation_alignment = slab->allocation_alignment,
      .maintenance_alignment = properties->maintenance_alignment,
      .atomic_operations = properties->atomic_operations,
  };
  if (out_backing->facts.maintenance_range.byte_length ==
      IREE_HAL_WHOLE_BUFFER) {
    out_backing->facts.maintenance_range.byte_length =
        slab->length - slab->maintenance_range.byte_offset;
  }
}
