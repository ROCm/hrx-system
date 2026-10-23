// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/gather.h"

#include <string.h>

typedef struct loom_serve_parameter_gather_t {
  // Shared ownership between the download source view and deferred host call.
  iree_hal_resource_t resource;
  // Allocator owning this record and its trailing snapshot/key storage.
  iree_allocator_t host_allocator;
  // Retained device used by the provider to import checkpoint files.
  iree_hal_device_t* device;
  // Retained immutable index and imported file cache, independent of the model.
  iree_io_parameter_provider_t* provider;
  // Retained destination until accepted reads have captured their references.
  iree_hal_buffer_t* target;
  // Source key stored after the descriptor snapshot in this allocation.
  iree_string_view_t key;
  // Number of 24-byte records in the trailing descriptor snapshot.
  iree_host_size_t count;
} loom_serve_parameter_gather_t;

static void loom_serve_parameter_gather_destroy(iree_hal_resource_t* resource) {
  loom_serve_parameter_gather_t* gather =
      (loom_serve_parameter_gather_t*)resource;
  iree_hal_buffer_release(gather->target);
  iree_io_parameter_provider_release(gather->provider);
  iree_hal_device_release(gather->device);
  iree_allocator_free(gather->host_allocator, gather);
}

static const iree_hal_resource_vtable_t loom_serve_parameter_gather_vtable = {
    .destroy = loom_serve_parameter_gather_destroy,
};

static void loom_serve_parameter_gather_download_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_resource_release((iree_hal_resource_t*)user_data);
}

static iree_status_t loom_serve_parameter_gather_enumerate(
    void* user_data, iree_host_size_t index, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_serve_parameter_gather_t* gather = user_data;
  const uint8_t* record = (const uint8_t*)(gather + 1) + index * 24;
  *out_key = gather->key;
  *out_span = (iree_io_parameter_span_t){
      .parameter_offset = iree_unaligned_load_le_u64(record),
      .buffer_offset = iree_unaligned_load_le_u64(record + 8),
      .length = iree_unaligned_load_le_u64(record + 16),
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_parameter_gather_invoke(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  loom_serve_parameter_gather_t* gather = user_data;
  IREE_RETURN_IF_ERROR(iree_io_parameter_provider_gather(
      gather->provider, gather->device, context->queue,
      iree_hal_semaphore_list_empty(), context->signal_semaphore_list,
      iree_string_view_empty(), gather->target, gather->count,
      (iree_io_parameter_enumerator_t){loom_serve_parameter_gather_enumerate,
                                       gather}));
  return iree_status_from_code(IREE_STATUS_DEFERRED);
}

iree_status_t loom_serve_parameter_gather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_string_view_t key, iree_hal_buffer_t* descriptors,
    iree_device_size_t descriptor_offset, iree_host_size_t count,
    iree_hal_buffer_t* target, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_ready) {
  *out_ready = NULL;
  if (key.size > IREE_HOST_SIZE_MAX - sizeof(loom_serve_parameter_gather_t) ||
      count > (IREE_HOST_SIZE_MAX - sizeof(loom_serve_parameter_gather_t) -
               key.size) /
                  24) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter gather descriptor storage overflows");
  }
  const iree_host_size_t snapshot_length = count * 24;
  const iree_host_size_t allocation_size =
      sizeof(loom_serve_parameter_gather_t) + snapshot_length + key.size;
  if (descriptor_offset > iree_hal_buffer_byte_length(descriptors) ||
      snapshot_length >
          iree_hal_buffer_byte_length(descriptors) - descriptor_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter gather descriptors exceed the buffer");
  }
  if (!count) {
    // An empty read still carries its producer dependency, but borrows no
    // payload or descriptor storage and submits no download or host callback.
    iree_hal_fence_t* ready = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_create(waits.count, host_allocator, &ready));
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < waits.count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_fence_insert(ready, waits.semaphores[i],
                                     waits.payload_values[i]);
    }
    if (iree_status_is_ok(status)) {
      *out_ready = ready;
    } else {
      iree_hal_fence_release(ready);
    }
    return status;
  }
  loom_serve_parameter_gather_t* gather = NULL;
  // Initialize the owner and key here; the download writes every descriptor
  // byte before enumeration. Clearing the snapshot would add a full write pass.
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, allocation_size, (void**)&gather));
  iree_hal_resource_initialize(&loom_serve_parameter_gather_vtable,
                               &gather->resource);
  gather->host_allocator = host_allocator;
  gather->device = device;
  iree_hal_device_retain(device);
  gather->provider = provider;
  iree_io_parameter_provider_retain(provider);
  gather->target = target;
  iree_hal_buffer_retain(target);
  gather->count = count;
  uint8_t* snapshot = (uint8_t*)(gather + 1);
  char* key_storage = (char*)snapshot + snapshot_length;
  if (key.size) {
    memcpy(key_storage, key.data, key.size);
  }
  gather->key = iree_make_string_view(key_storage, key.size);

  iree_hal_semaphore_t* downloaded = NULL;
  iree_hal_semaphore_t* completed = NULL;
  iree_hal_fence_t* ready = NULL;
  iree_hal_buffer_t* source = NULL;
  iree_status_t status =
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &downloaded);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &completed);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_fence_create_at(completed, 1, host_allocator, &ready);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_subspan_buffer_create_with_callback(
        descriptors,
        iree_hal_buffer_byte_offset(descriptors) + descriptor_offset,
        snapshot_length,
        (iree_hal_buffer_release_callback_t){
            loom_serve_parameter_gather_download_release, gather},
        host_allocator, &source);
    if (iree_status_is_ok(status)) {
      // The view owns a separate claim until the queued copy releases it,
      // independently of callback cancellation or failed semaphore readiness.
      iree_hal_resource_retain(&gather->resource);
    }
  }
  uint64_t value = 1;
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_download(
        queue, waits, (iree_hal_semaphore_list_t){1, &downloaded, &value},
        source, 0, snapshot, snapshot_length, /*barriers=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t args[4] = {0};
    status = iree_hal_queue_host_call(
        queue, (iree_hal_semaphore_list_t){1, &downloaded, &value},
        iree_hal_fence_semaphore_list(ready),
        iree_hal_make_host_call_with_resource(
            loom_serve_parameter_gather_invoke, gather, &gather->resource),
        args, IREE_HAL_HOST_CALL_FLAG_NONE);
  }
  iree_hal_buffer_release(source);
  iree_hal_resource_release(&gather->resource);
  iree_hal_semaphore_release(completed);
  iree_hal_semaphore_release(downloaded);
  if (iree_status_is_ok(status)) {
    *out_ready = ready;
  } else {
    iree_hal_fence_release(ready);
  }
  return status;
}
