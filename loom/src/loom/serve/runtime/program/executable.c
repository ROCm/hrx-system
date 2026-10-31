// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/executable.h"

#include <string.h>

#include "iree/base/threading/mutex.h"

typedef struct loom_serve_executable_image_t {
  // Cache ownership and each specialized view retain the image independently.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this record and the retained exact image bytes.
  iree_allocator_t host_allocator;
  // Owned immutable storage backing |data|.
  iree_byte_sequence_t* storage;
  // Full contiguous emitted bytes establish identity and feed the HAL loader.
  iree_const_byte_span_t data;
  // Owned HAL executable, published under the cache mutex after loading.
  iree_hal_executable_t* handle;
  // HAL references owned by direct descriptors in live specialized views.
  iree_atomic_int32_t dispatch_count;
  // Pending requests, detached before callbacks. A NULL handle denotes loading.
  loom_serve_executable_request_t* waiters;
} loom_serve_executable_image_t;

struct loom_serve_executable_cache_t {
  // Allocation policy for images, views, and lookup storage.
  iree_allocator_t host_allocator;
  // Serializes lookup/publication, never loading or consumer callbacks.
  iree_slim_mutex_t mutex;
  // Borrowed immutable loader queue family.
  const iree_hal_queue_family_t* queue_family;
  // Borrowed immutable target accepted by that queue family.
  const iree_hal_executable_target_t* target;
  // Stable image records sorted by exact emitted bytes.
  loom_serve_executable_image_t** images;
  // Indexed images, including pending loads.
  iree_host_size_t count;
  // Allocated lookup capacity, retained through individual image eviction.
  iree_host_size_t capacity;
};

struct loom_serve_executable_t {
  // Input-cache and stage ownership of this specialization's binding view.
  iree_atomic_ref_count_t ref_count;
  // Retained image shared with other specializations.
  loom_serve_executable_image_t* image;
  // Owned direct launch descriptor, or NULL for command children.
  loom_serve_dispatch_t* dispatch;
  // Number of functions in input root order.
  iree_host_size_t function_count;
  // Executable-local functions, independent of parent command slot order.
  iree_hal_executable_function_t functions[];
};

static void loom_serve_executable_image_release(
    loom_serve_executable_image_t* image) {
  if (!image) {
    return;
  }
  if (iree_atomic_ref_count_dec(&image->ref_count) == 1) {
    iree_hal_executable_release(image->handle);
    iree_byte_sequence_release(image->storage);
    iree_allocator_free(image->host_allocator, image);
  }
}

void loom_serve_executable_retain(loom_serve_executable_t* executable) {
  if (executable) {
    iree_atomic_ref_count_inc(&executable->ref_count);
  }
}

void loom_serve_executable_release(loom_serve_executable_t* executable) {
  if (!executable) {
    return;
  }
  if (iree_atomic_ref_count_dec(&executable->ref_count) != 1) {
    return;
  }
  loom_serve_executable_image_t* image = executable->image;
  if (executable->dispatch) {
    // Relinquish cache accounting before the HAL reference. A concurrent trim
    // can conservatively defer collection, never mistake a live reference for
    // one it owns. A separately retained descriptor becomes an external owner.
    iree_atomic_fetch_sub(&image->dispatch_count, 1, iree_memory_order_relaxed);
    loom_serve_dispatch_release(executable->dispatch);
  }
  iree_allocator_free(image->host_allocator, executable);
  loom_serve_executable_image_release(image);
}

static bool loom_serve_executable_image_is_shared(
    const loom_serve_executable_image_t* image) {
  const iree_hal_resource_t* resource =
      (const iree_hal_resource_t*)image->handle;
  const int32_t owned_references =
      1 + iree_atomic_load(&image->dispatch_count, iree_memory_order_relaxed);
  return iree_atomic_ref_count_load(&resource->ref_count) != owned_references;
}

bool loom_serve_executable_is_shared(
    const loom_serve_executable_t* executable) {
  return iree_atomic_ref_count_load(&executable->ref_count) != 1 ||
         (executable->dispatch &&
          loom_serve_dispatch_is_shared(executable->dispatch)) ||
         loom_serve_executable_image_is_shared(executable->image);
}

iree_host_size_t loom_serve_executable_storage_size(
    const loom_serve_executable_t* executable) {
  return sizeof(*executable) +
         executable->function_count * sizeof(*executable->functions);
}

iree_hal_executable_t* loom_serve_executable_handle(
    const loom_serve_executable_t* executable) {
  return executable->image->handle;
}

iree_hal_executable_function_t loom_serve_executable_function(
    const loom_serve_executable_t* executable, iree_host_size_t ordinal) {
  return executable->functions[ordinal];
}

loom_serve_dispatch_t* loom_serve_executable_dispatch(
    const loom_serve_executable_t* executable) {
  return executable->dispatch;
}

static iree_status_t loom_serve_executable_create_dispatch(
    const loomc_launch_config_t* launch, loom_serve_executable_t* executable) {
  loom_serve_executable_image_t* image = executable->image;
  iree_hal_executable_function_info_t info;
  IREE_RETURN_IF_ERROR(iree_hal_executable_function_info(
      image->handle, executable->functions[0], &info));
  if (launch->workgroup_cluster_size.x != 1 ||
      launch->workgroup_cluster_size.y != 1 ||
      launch->workgroup_cluster_size.z != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "HAL direct stage cannot express clustered launch");
  }
  const uint64_t fixed_storage =
      info.resource_usage.fixed_workgroup_local_memory_size;
  if (launch->workgroup_storage_bytes &&
      (!iree_any_bit_set(
           info.resource_usage.provided_flags,
           IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY) ||
       launch->workgroup_storage_bytes < fixed_storage ||
       launch->workgroup_storage_bytes - fixed_storage > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct stage requires representable reflected local storage");
  }
  iree_hal_dispatch_config_t config = {
      .workgroup_count = {launch->workgroup_count.x, launch->workgroup_count.y,
                          launch->workgroup_count.z},
      .dynamic_workgroup_local_memory =
          launch->workgroup_storage_bytes > fixed_storage
              ? (uint32_t)(launch->workgroup_storage_bytes - fixed_storage)
              : 0,
  };
  if (iree_any_bit_set(
          info.flags,
          IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC)) {
    config.workgroup_size[0] = launch->workgroup_size.x;
    config.workgroup_size[1] = launch->workgroup_size.y;
    config.workgroup_size[2] = launch->workgroup_size.z;
  }
  IREE_RETURN_IF_ERROR(loom_serve_dispatch_create(
      image->handle, executable->functions[0], config, image->host_allocator,
      &executable->dispatch));
  iree_atomic_fetch_add(&image->dispatch_count, 1, iree_memory_order_relaxed);
  return iree_ok_status();
}

static iree_status_t loom_serve_executable_create(
    loom_serve_executable_image_t* image,
    const loom_serve_executable_request_t* request,
    loom_serve_executable_t** out_executable) {
  *out_executable = NULL;
  loom_serve_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      image->host_allocator, sizeof(*executable), request->root_count,
      sizeof(*executable->functions), (void**)&executable));
  iree_atomic_ref_count_init(&executable->ref_count);
  executable->image = image;
  iree_atomic_ref_count_inc(&image->ref_count);
  executable->function_count = request->root_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < request->root_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_executable_lookup_function_by_name(
        image->handle, request->roots[i], &executable->functions[i]);
  }
  if (iree_status_is_ok(status) && request->launch_config) {
    status = loom_serve_executable_create_dispatch(request->launch_config,
                                                   executable);
  }
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
  } else {
    loom_serve_executable_release(executable);
  }
  return status;
}

iree_status_t loom_serve_executable_cache_allocate(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target, iree_allocator_t host_allocator,
    loom_serve_executable_cache_t** out_cache) {
  *out_cache = NULL;
  loom_serve_executable_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  cache->queue_family = queue_family;
  cache->target = target;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

void loom_serve_executable_cache_free(loom_serve_executable_cache_t* cache) {
  if (!cache) {
    return;
  }
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    loom_serve_executable_image_release(cache->images[i]);
  }
  iree_allocator_free(cache->host_allocator, cache->images);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
}

// The array is small and cold; exact byte ordering avoids an additional hash
// table and establishes equality even for differing images with common headers.
static iree_host_size_t loom_serve_executable_cache_position(
    const loom_serve_executable_cache_t* cache, iree_const_byte_span_t data,
    bool* out_found) {
  *out_found = false;
  iree_host_size_t begin = 0;
  iree_host_size_t end = cache->count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const iree_const_byte_span_t candidate = cache->images[middle]->data;
    const int order = data.data_length == candidate.data_length
                          ? memcmp(data.data, candidate.data, data.data_length)
                          : (data.data_length < candidate.data_length ? -1 : 1);
    if (order == 0) {
      *out_found = true;
      return middle;
    } else if (order < 0) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return begin;
}

static iree_status_t loom_serve_executable_cache_insert(
    loom_serve_executable_cache_t* cache, iree_host_size_t position,
    loom_serve_executable_request_t* request, iree_const_byte_span_t data,
    loom_serve_executable_image_t** out_image) {
  if (cache->count == cache->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        cache->host_allocator, cache->count + 1, sizeof(*cache->images),
        &cache->capacity, (void**)&cache->images));
  }
  loom_serve_executable_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(cache->host_allocator,
                                             sizeof(*image), (void**)&image));
  iree_atomic_ref_count_init(&image->ref_count);
  image->host_allocator = cache->host_allocator;
  image->storage = request->image;
  image->data = data;
  request->image = NULL;
  memmove(&cache->images[position + 1], &cache->images[position],
          (cache->count - position) * sizeof(*cache->images));
  cache->images[position] = image;
  ++cache->count;
  *out_image = image;
  return iree_ok_status();
}

static iree_status_t loom_serve_executable_cache_load_image(
    loom_serve_executable_cache_t* cache, loom_serve_executable_image_t* image,
    iree_hal_executable_t** out_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data = image->data;
  iree_status_t status = iree_hal_executable_load(
      cache->queue_family, cache->target, &params, out_handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_executable_complete(
    loom_serve_executable_image_t* image,
    loom_serve_executable_request_t* request) {
  loom_serve_executable_t* executable = NULL;
  iree_status_t status =
      loom_serve_executable_create(image, request, &executable);
  request->completion(request->user_data, status, executable);
}

void loom_serve_executable_cache_load(
    loom_serve_executable_cache_t* cache,
    loom_serve_executable_request_t* request) {
  iree_const_byte_span_t data = iree_const_byte_span_empty();
  const bool contiguous =
      iree_byte_sequence_try_get_contiguous_span(request->image, &data);
  IREE_ASSERT(contiguous && "native publication owns image coalescing");
  (void)contiguous;

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&cache->mutex);
  bool found = false;
  const iree_host_size_t position =
      loom_serve_executable_cache_position(cache, data, &found);
  loom_serve_executable_image_t* image = NULL;
  if (found) {
    image = cache->images[position];
  } else {
    status = loom_serve_executable_cache_insert(cache, position, request, data,
                                                &image);
  }
  const bool ready = image && image->handle;
  // Retire producer storage before publishing its continuation. The loader
  // can complete it immediately after unlock, including destroying the caller.
  iree_byte_sequence_release(request->image);
  request->image = NULL;
  if (image && !ready) {
    request->next = image->waiters;
    image->waiters = request;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  if (!iree_status_is_ok(status)) {
    request->completion(request->user_data, status, NULL);
    return;
  }
  if (found) {
    if (ready) {
      loom_serve_executable_complete(image, request);
    }
    return;
  }

  iree_hal_executable_t* handle = NULL;
  status = loom_serve_executable_cache_load_image(cache, image, &handle);
  iree_slim_mutex_lock(&cache->mutex);
  image->handle = handle;
  loom_serve_executable_request_t* waiters = image->waiters;
  image->waiters = NULL;
  if (!iree_status_is_ok(status)) {
    const iree_host_size_t index =
        loom_serve_executable_cache_position(cache, image->data, &found);
    memmove(&cache->images[index], &cache->images[index + 1],
            (--cache->count - index) * sizeof(*cache->images));
    loom_serve_executable_image_release(image);
  }
  iree_slim_mutex_unlock(&cache->mutex);
  // Pending consumers keep the cache/domain alive. The last callback can
  // destroy both; nothing below it may access cache, image, or its request.
  while (waiters) {
    request = waiters;
    waiters = request->next;
    if (iree_status_is_ok(status)) {
      loom_serve_executable_complete(image, request);
    } else {
      request->completion(request->user_data,
                          waiters ? iree_status_clone(status) : status, NULL);
    }
  }
}

loom_serve_executable_cache_storage_t loom_serve_executable_cache_storage(
    const loom_serve_executable_cache_t* cache) {
  loom_serve_executable_cache_storage_t storage = {
      .count = cache->count,
      .host_bytes = sizeof(*cache) + cache->capacity * sizeof(*cache->images),
      .index_bytes = sizeof(*cache) + cache->capacity * sizeof(*cache->images)};
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    storage.code_bytes += cache->images[i]->data.data_length;
    storage.host_bytes +=
        sizeof(*cache->images[i]) + cache->images[i]->data.data_length;
  }
  return storage;
}

void loom_serve_executable_cache_trim(loom_serve_executable_cache_t* cache) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < cache->count; ++i) {
    loom_serve_executable_image_t* image = cache->images[i];
    if (iree_atomic_ref_count_load(&image->ref_count) == 1 &&
        !loom_serve_executable_image_is_shared(image)) {
      loom_serve_executable_image_release(image);
    } else {
      cache->images[count++] = image;
    }
  }
  cache->count = count;
}
