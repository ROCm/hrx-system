// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/context_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/device.h"

typedef struct iree_hal_amdxdna_context_cache_entry_t {
  iree_byte_span_t pdi;
  iree_byte_span_t xclbin;
  iree_string_view_t kernel_name;
  iree_hal_amdxdna_native_context_ref_t* context_ref;
  struct iree_hal_amdxdna_context_cache_entry_t* next;
} iree_hal_amdxdna_context_cache_entry_t;

struct iree_hal_amdxdna_device_context_cache_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_context_cache_entry_t* head;
};

static bool iree_hal_amdxdna_byte_spans_equal(iree_const_byte_span_t lhs,
                                              iree_const_byte_span_t rhs) {
  return lhs.data_length == rhs.data_length &&
         (lhs.data_length == 0 ||
          memcmp(lhs.data, rhs.data, lhs.data_length) == 0);
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
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_context_cache_entry_t* entry) {
  if (!entry) return;
  iree_hal_amdxdna_native_context_ref_release(entry->context_ref);
  iree_allocator_free(host_allocator, entry->pdi.data);
  iree_allocator_free(host_allocator, entry->xclbin.data);
  iree_allocator_free(host_allocator, (void*)entry->kernel_name.data);
  iree_allocator_free(host_allocator, entry);
}

iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create(iree_allocator_t host_allocator) {
  iree_hal_amdxdna_device_context_cache_t* context_cache = NULL;
  if (!iree_status_is_ok(iree_allocator_malloc(
          host_allocator, sizeof(*context_cache), (void**)&context_cache))) {
    return NULL;
  }
  memset(context_cache, 0, sizeof(*context_cache));
  context_cache->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context_cache->mutex);
  return context_cache;
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
  iree_slim_mutex_unlock(&context_cache->mutex);

  while (entry) {
    iree_hal_amdxdna_context_cache_entry_t* next = entry->next;
    iree_hal_amdxdna_context_cache_entry_destroy(context_cache->host_allocator,
                                                 entry);
    entry = next;
  }
}

iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
  *out_context_ref = NULL;
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
      (device->native_caps.context_image_models &
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

  iree_slim_mutex_lock(&device->context_cache->mutex);
  for (iree_hal_amdxdna_context_cache_entry_t* entry =
           device->context_cache->head;
       entry; entry = entry->next) {
    if (iree_hal_amdxdna_byte_spans_equal(
            iree_make_const_byte_span(entry->pdi.data, entry->pdi.data_length),
            key_pdi) &&
        iree_hal_amdxdna_byte_spans_equal(
            iree_make_const_byte_span(entry->xclbin.data,
                                      entry->xclbin.data_length),
            key_xclbin) &&
        iree_string_view_equal(entry->kernel_name, key_kernel_name)) {
      *out_context_ref =
          iree_hal_amdxdna_native_context_ref_retain(entry->context_ref);
      iree_slim_mutex_unlock(&device->context_cache->mutex);
      return iree_ok_status();
    }
  }

  iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;
  iree_status_t status = iree_hal_amdxdna_native_device_c_create_context_ref(
      device->native_device, &context_image, &context_ref);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&device->context_cache->mutex);
    return status;
  }

  iree_hal_amdxdna_context_cache_entry_t* entry = NULL;
  status = iree_allocator_malloc(device->context_cache->host_allocator,
                                 sizeof(*entry), (void**)&entry);
  if (iree_status_is_ok(status)) {
    memset(entry, 0, sizeof(*entry));
    status = iree_hal_amdxdna_context_cache_copy_bytes(
        device->context_cache->host_allocator, key_pdi, &entry->pdi);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_context_cache_copy_bytes(
        device->context_cache->host_allocator, key_xclbin, &entry->xclbin);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_context_cache_copy_string(
        device->context_cache->host_allocator, key_kernel_name,
        &entry->kernel_name);
  }
  if (iree_status_is_ok(status)) {
    entry->context_ref = context_ref;
    entry->next = device->context_cache->head;
    device->context_cache->head = entry;
    *out_context_ref = iree_hal_amdxdna_native_context_ref_retain(context_ref);
    iree_slim_mutex_unlock(&device->context_cache->mutex);
    return iree_ok_status();
  }

  iree_hal_amdxdna_native_context_ref_release(context_ref);
  iree_hal_amdxdna_context_cache_entry_destroy(
      device->context_cache->host_allocator, entry);
  iree_slim_mutex_unlock(&device->context_cache->mutex);
  return status;
}
