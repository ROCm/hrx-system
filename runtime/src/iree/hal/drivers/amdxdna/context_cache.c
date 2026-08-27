// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/context_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/device.h"

// Bounds how many native hardware contexts the cache keeps alive. The amdxdna
// driver has only a small, process-global pool of concurrent hardware contexts,
// so an unbounded cache exhausts it after enough distinct kernels. On overflow
// the least-recently-used entry is evicted, dropping the cache's reference so
// the context is reclaimed once no executable or command buffer still holds it.
//
// The capacity is a soft cap: the device reports a per-architecture budget
// (max_hardware_contexts) that sizes the cache, but the true ceiling is not
// queryable and varies with the part and its array partitioning. So
// get_or_create_context treats it as a target and, on a creation failure from
// pool exhaustion, evicts an LRU context and retries, backing off to whatever
// the driver accepts.
//
// This default applies only when the device budget is unknown (0).
#define IREE_HAL_AMDXDNA_CONTEXT_CACHE_DEFAULT_CAPACITY 8

typedef struct iree_hal_amdxdna_context_cache_entry_t {
  iree_byte_span_t pdi;
  iree_byte_span_t xclbin;
  iree_string_view_t kernel_name;
  iree_hal_amdxdna_native_context_ref_t* context_ref;
  iree_host_size_t lease_count;
  struct iree_hal_amdxdna_context_cache_entry_t* next;
} iree_hal_amdxdna_context_cache_entry_t;

// Entries form a singly-linked most-recently-used list: `head` is the MRU entry
// and the tail is the LRU eviction target. A cache hit moves the entry to the
// front; inserts prepend and evict the tail while `count` exceeds `capacity`.
struct iree_hal_amdxdna_device_context_cache_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_context_cache_entry_t* head;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_hal_amdxdna_context_cache_ops_t ops;
  void* ops_user_data;
};

struct iree_hal_amdxdna_context_cache_lease_t {
  iree_hal_amdxdna_device_context_cache_t* context_cache;
  iree_hal_amdxdna_context_cache_entry_t* entry;
};

static iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_context_cache_retain_context(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  if (context_cache->ops.retain_context) {
    return context_cache->ops.retain_context(context_cache->ops_user_data,
                                             context_ref);
  }
  return iree_hal_amdxdna_native_context_ref_retain(context_ref);
}

static void iree_hal_amdxdna_context_cache_release_context(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_context_ref_t* context_ref) {
  if (context_cache->ops.release_context) {
    context_cache->ops.release_context(context_cache->ops_user_data,
                                       context_ref);
    return;
  }
  iree_hal_amdxdna_native_context_ref_release(context_ref);
}

static iree_status_t iree_hal_amdxdna_context_cache_create_context(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_amdxdna_native_c_context_image_t* context_image,
    bool* out_context_pool_exhausted,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  *out_context_pool_exhausted = false;
  *out_context_ref = NULL;
  if (context_cache->ops.create_context) {
    return context_cache->ops.create_context(
        context_cache->ops_user_data, context_image, out_context_pool_exhausted,
        out_context_ref);
  }
  return iree_hal_amdxdna_native_device_c_create_context_ref(
      native_device, context_image, out_context_pool_exhausted,
      out_context_ref);
}

iree_host_size_t iree_hal_amdxdna_context_cache_resolve_capacity(
    iree_host_size_t hardware_context_budget) {
  // Prefer the device-reported budget so the bound tracks the actual part; fall
  // back to the conservative default when the architecture is unknown (budget
  // 0), e.g. on backends that cannot resolve it yet.
  if (hardware_context_budget != 0) {
    return hardware_context_budget;
  }
  return IREE_HAL_AMDXDNA_CONTEXT_CACHE_DEFAULT_CAPACITY;
}

static bool iree_hal_amdxdna_byte_spans_equal(iree_const_byte_span_t lhs,
                                              iree_const_byte_span_t rhs) {
  return lhs.data_length == rhs.data_length &&
         (lhs.data_length == 0 ||
          memcmp(lhs.data, rhs.data, lhs.data_length) == 0);
}

bool iree_hal_amdxdna_context_cache_key_equal(
    const iree_hal_amdxdna_context_cache_key_t* lhs,
    const iree_hal_amdxdna_context_cache_key_t* rhs) {
  return iree_hal_amdxdna_byte_spans_equal(lhs->pdi, rhs->pdi) &&
         iree_hal_amdxdna_byte_spans_equal(lhs->xclbin, rhs->xclbin) &&
         iree_string_view_equal(lhs->kernel_name, rhs->kernel_name);
}

static iree_status_t iree_hal_amdxdna_context_cache_copy_bytes(
    iree_allocator_t host_allocator, iree_const_byte_span_t source,
    iree_byte_span_t* out_copy) {
  *out_copy = iree_byte_span_empty();
  if (source.data_length == 0) return iree_ok_status();
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, source.data_length,
                                             (void**)&storage));
  memcpy(storage, source.data, source.data_length);
  *out_copy = iree_make_byte_span(storage, source.data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_context_cache_copy_string(
    iree_allocator_t host_allocator, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  *out_copy = iree_string_view_empty();
  if (source.size == 0) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, source.size, (void**)&storage));
  memcpy(storage, source.data, source.size);
  *out_copy = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void iree_hal_amdxdna_context_cache_entry_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_context_cache_entry_t* entry) {
  if (!entry) return;
  iree_allocator_t host_allocator = context_cache->host_allocator;
  if (entry->context_ref) {
    if (context_cache->ops.before_release_context) {
      context_cache->ops.before_release_context(context_cache->ops_user_data,
                                                entry->context_ref);
    }
    iree_hal_amdxdna_context_cache_release_context(context_cache,
                                                   entry->context_ref);
  }
  iree_allocator_free(host_allocator, entry->pdi.data);
  iree_allocator_free(host_allocator, entry->xclbin.data);
  iree_allocator_free(host_allocator, (void*)entry->kernel_name.data);
  iree_allocator_free(host_allocator, entry);
}

// Evicts the least-recently-used unleased entry. Must be called with the cache
// mutex held. Releasing the entry's context reference reclaims the hardware
// context once no retained dispatch references remain.
static bool iree_hal_amdxdna_context_cache_evict_lru(
    iree_hal_amdxdna_device_context_cache_t* context_cache) {
  if (!context_cache->head) return false;
  iree_hal_amdxdna_context_cache_entry_t* prev = NULL;
  iree_hal_amdxdna_context_cache_entry_t* entry = context_cache->head;
  iree_hal_amdxdna_context_cache_entry_t* lru_prev = NULL;
  iree_hal_amdxdna_context_cache_entry_t* lru = NULL;
  while (entry) {
    if (entry->lease_count == 0) {
      lru_prev = prev;
      lru = entry;
    }
    prev = entry;
    entry = entry->next;
  }
  if (!lru) return false;
  if (lru_prev) {
    lru_prev->next = lru->next;
  } else {
    context_cache->head = lru->next;
  }
  if (context_cache->count > 0) context_cache->count--;
  iree_hal_amdxdna_context_cache_entry_destroy(context_cache, lru);
  return true;
}

static iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create_internal(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget,
    const iree_hal_amdxdna_context_cache_ops_t* ops, void* user_data) {
  iree_hal_amdxdna_device_context_cache_t* context_cache = NULL;
  if (!iree_status_is_ok(iree_allocator_malloc(
          host_allocator, sizeof(*context_cache), (void**)&context_cache))) {
    return NULL;
  }
  memset(context_cache, 0, sizeof(*context_cache));
  context_cache->host_allocator = host_allocator;
  context_cache->capacity =
      iree_hal_amdxdna_context_cache_resolve_capacity(hardware_context_budget);
  if (ops) context_cache->ops = *ops;
  context_cache->ops_user_data = user_data;
  iree_slim_mutex_initialize(&context_cache->mutex);
  return context_cache;
}

iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget) {
  return iree_hal_amdxdna_device_context_cache_create_internal(
      host_allocator, hardware_context_budget, NULL, NULL);
}

iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create_with_ops(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget,
    const iree_hal_amdxdna_context_cache_ops_t* ops, void* user_data) {
  IREE_ASSERT_ARGUMENT(ops);
  return iree_hal_amdxdna_device_context_cache_create_internal(
      host_allocator, hardware_context_budget, ops, user_data);
}

void iree_hal_amdxdna_device_context_cache_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache) {
  if (!context_cache) return;
  iree_allocator_t host_allocator = context_cache->host_allocator;
  iree_hal_amdxdna_device_context_cache_clear(context_cache);
  iree_slim_mutex_deinitialize(&context_cache->mutex);
  iree_allocator_free(host_allocator, context_cache);
}

void iree_hal_amdxdna_device_context_cache_clear(
    iree_hal_amdxdna_device_context_cache_t* context_cache) {
  if (!context_cache) return;
  iree_slim_mutex_lock(&context_cache->mutex);
  iree_hal_amdxdna_context_cache_entry_t* entry = context_cache->head;
  context_cache->head = NULL;
  context_cache->count = 0;
  iree_slim_mutex_unlock(&context_cache->mutex);

  while (entry) {
    iree_hal_amdxdna_context_cache_entry_t* next = entry->next;
    iree_hal_amdxdna_context_cache_entry_destroy(context_cache, entry);
    entry = next;
  }
}

static iree_status_t iree_hal_amdxdna_context_cache_create_lease_locked(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_context_cache_entry_t* entry,
    iree_hal_amdxdna_context_cache_lease_t** out_lease) {
  iree_hal_amdxdna_context_cache_lease_t* lease = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(context_cache->host_allocator,
                                             sizeof(*lease), (void**)&lease));
  lease->context_cache = context_cache;
  lease->entry = entry;
  entry->lease_count++;
  *out_lease = lease;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_context_cache_get_or_create_internal(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref,
    iree_hal_amdxdna_context_cache_lease_t** out_lease) {
  IREE_ASSERT_ARGUMENT(context_cache);
  if (out_context_ref) *out_context_ref = NULL;
  if (out_lease) *out_lease = NULL;
  if (pdi.data_length == 0 && xclbin.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache requires context "
                            "PDI or xclbin data");
  }
  if (pdi.data_length != 0 && !pdi.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache PDI is NULL");
  }
  if (xclbin.data_length != 0 && !xclbin.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control-packet context cache xclbin is NULL");
  }
  if (iree_string_view_is_empty(kernel_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "control-packet context cache requires a non-empty CU name");
  }

  const bool use_xclbin_context =
      xclbin.data_length != 0 &&
      (context_image_models &
       IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_XCLBIN);
  iree_const_byte_span_t key_pdi = iree_const_byte_span_empty();
  iree_const_byte_span_t key_xclbin = iree_const_byte_span_empty();
  iree_string_view_t key_kernel_name = iree_string_view_empty();
  iree_hal_amdxdna_native_c_context_image_t context_image;
  memset(&context_image, 0, sizeof(context_image));
  context_image.pdi = pdi;
  context_image.xclbin = xclbin;
  context_image.kernel_name = kernel_name;
  if (use_xclbin_context) {
    key_xclbin = xclbin;
    context_image.type = IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_XCLBIN;
  } else {
    if (pdi.data_length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "control-packet context cache requires PDI data for this native "
          "driver");
    }
    key_pdi = pdi;
    key_kernel_name = kernel_name;
    context_image.type = IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_TYPE_PDI;
    context_image.xclbin = iree_const_byte_span_empty();
  }

  const iree_hal_amdxdna_context_cache_key_t request_key = {
      /*.pdi=*/key_pdi,
      /*.xclbin=*/key_xclbin,
      /*.kernel_name=*/key_kernel_name,
  };

  iree_slim_mutex_lock(&context_cache->mutex);
  iree_hal_amdxdna_context_cache_entry_t* prev = NULL;
  for (iree_hal_amdxdna_context_cache_entry_t* entry = context_cache->head;
       entry; prev = entry, entry = entry->next) {
    const iree_hal_amdxdna_context_cache_key_t entry_key = {
        /*.pdi=*/iree_make_const_byte_span(entry->pdi.data,
                                           entry->pdi.data_length),
        /*.xclbin=*/
        iree_make_const_byte_span(entry->xclbin.data,
                                  entry->xclbin.data_length),
        /*.kernel_name=*/entry->kernel_name,
    };
    if (iree_hal_amdxdna_context_cache_key_equal(&entry_key, &request_key)) {
      // Cache hit: promote to MRU (front) so the LRU eviction order stays
      // meaningful, then hand out an additional reference to the caller.
      if (prev) {
        prev->next = entry->next;
        entry->next = context_cache->head;
        context_cache->head = entry;
      }
      iree_status_t status = iree_ok_status();
      if (out_context_ref) {
        *out_context_ref = iree_hal_amdxdna_context_cache_retain_context(
            context_cache, entry->context_ref);
      }
      if (out_lease) {
        status = iree_hal_amdxdna_context_cache_create_lease_locked(
            context_cache, entry, out_lease);
        if (!iree_status_is_ok(status) && out_context_ref) {
          iree_hal_amdxdna_context_cache_release_context(context_cache,
                                                         *out_context_ref);
          *out_context_ref = NULL;
        }
      }
      iree_slim_mutex_unlock(&context_cache->mutex);
      return status;
    }
  }

  // Make room before creating so we never momentarily exceed the cap, which on
  // an already-full driver pool would itself fail. Eviction drops only the
  // cache's reference; a context still held by a live executable or command
  // buffer is not reclaimed until that owner releases it.
  while (context_cache->capacity != 0 &&
         context_cache->count >= context_cache->capacity) {
    if (!iree_hal_amdxdna_context_cache_evict_lru(context_cache)) {
      iree_slim_mutex_unlock(&context_cache->mutex);
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "amdxdna context cache capacity is exhausted by leased entries");
    }
  }

  // Create the hardware context. If creation fails because the driver's pool is
  // exhausted, evict an LRU cached context (freeing its hwctx once no live
  // owner holds it) and retry until it succeeds or no cached context remains.
  // The budget is only a soft cap; this backoff is what keeps the cache safe on
  // parts whose real ceiling is below it.
  iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;
  iree_status_t status = iree_ok_status();
  for (;;) {
    bool context_pool_exhausted = false;
    status = iree_hal_amdxdna_context_cache_create_context(
        context_cache, native_device, &context_image, &context_pool_exhausted,
        &context_ref);
    if (iree_status_is_ok(status) || !context_pool_exhausted ||
        !context_cache->head) {
      break;
    }
    iree_status_ignore(status);
    if (!iree_hal_amdxdna_context_cache_evict_lru(context_cache)) break;
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&context_cache->mutex);
    return status;
  }

  iree_hal_amdxdna_context_cache_entry_t* entry = NULL;
  status = iree_allocator_malloc(context_cache->host_allocator, sizeof(*entry),
                                 (void**)&entry);
  if (iree_status_is_ok(status)) {
    memset(entry, 0, sizeof(*entry));
    status = iree_hal_amdxdna_context_cache_copy_bytes(
        context_cache->host_allocator, key_pdi, &entry->pdi);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_context_cache_copy_bytes(
        context_cache->host_allocator, key_xclbin, &entry->xclbin);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_context_cache_copy_string(
        context_cache->host_allocator, key_kernel_name, &entry->kernel_name);
  }
  if (iree_status_is_ok(status)) {
    entry->context_ref = context_ref;
    entry->next = context_cache->head;
    context_cache->head = entry;
    context_cache->count++;
    if (out_context_ref) {
      *out_context_ref = iree_hal_amdxdna_context_cache_retain_context(
          context_cache, context_ref);
    }
    if (out_lease) {
      status = iree_hal_amdxdna_context_cache_create_lease_locked(
          context_cache, entry, out_lease);
      if (!iree_status_is_ok(status) && out_context_ref) {
        iree_hal_amdxdna_context_cache_release_context(context_cache,
                                                       *out_context_ref);
        *out_context_ref = NULL;
      }
    }
    // Bound the number of cached hardware contexts. Eviction only drops the
    // cache's reference; leased or retained dispatch contexts survive until
    // their owners release them.
    while (context_cache->capacity != 0 &&
           context_cache->count > context_cache->capacity &&
           iree_status_is_ok(status)) {
      if (!iree_hal_amdxdna_context_cache_evict_lru(context_cache)) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "amdxdna context cache capacity is exhausted by leased entries");
      }
    }
    iree_slim_mutex_unlock(&context_cache->mutex);
    return status;
  }

  iree_hal_amdxdna_context_cache_release_context(context_cache, context_ref);
  iree_hal_amdxdna_context_cache_entry_destroy(context_cache, entry);
  iree_slim_mutex_unlock(&context_cache->mutex);
  return status;
}

iree_status_t iree_hal_amdxdna_context_cache_get_or_create(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  IREE_ASSERT_ARGUMENT(out_context_ref);
  return iree_hal_amdxdna_context_cache_get_or_create_internal(
      context_cache, native_device, context_image_models, pdi, xclbin,
      kernel_name, out_context_ref, NULL);
}

iree_status_t iree_hal_amdxdna_context_cache_pin(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_context_cache_lease_t** out_lease) {
  IREE_ASSERT_ARGUMENT(out_lease);
  return iree_hal_amdxdna_context_cache_get_or_create_internal(
      context_cache, native_device, context_image_models, pdi, xclbin,
      kernel_name, NULL, out_lease);
}

iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_context_cache_lease_retain_context(
    iree_hal_amdxdna_context_cache_lease_t* lease) {
  if (!lease) return NULL;
  iree_hal_amdxdna_device_context_cache_t* context_cache =
      lease->context_cache;
  iree_slim_mutex_lock(&context_cache->mutex);
  iree_hal_amdxdna_native_context_ref_t* context_ref =
      iree_hal_amdxdna_context_cache_retain_context(context_cache,
                                                    lease->entry->context_ref);
  iree_slim_mutex_unlock(&context_cache->mutex);
  return context_ref;
}

void iree_hal_amdxdna_context_cache_lease_release(
    iree_hal_amdxdna_context_cache_lease_t* lease) {
  if (!lease) return;
  iree_hal_amdxdna_device_context_cache_t* context_cache =
      lease->context_cache;
  iree_slim_mutex_lock(&context_cache->mutex);
  IREE_ASSERT(lease->entry->lease_count > 0,
              "amdxdna context cache lease count underflow");
  lease->entry->lease_count--;
  iree_slim_mutex_unlock(&context_cache->mutex);
  iree_allocator_free(context_cache->host_allocator, lease);
}

iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  IREE_ASSERT_ARGUMENT(device);
  return iree_hal_amdxdna_context_cache_get_or_create(
      device->context_cache, device->native_device,
      device->native_caps.context_image_models, pdi, xclbin, kernel_name,
      out_context_ref);
}

iree_status_t iree_hal_amdxdna_device_pin_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_context_cache_lease_t** out_lease) {
  IREE_ASSERT_ARGUMENT(device);
  return iree_hal_amdxdna_context_cache_pin(
      device->context_cache, device->native_device,
      device->native_caps.context_image_models, pdi, xclbin, kernel_name,
      out_lease);
}
