// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/program_image.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/util.h"

struct iree_hal_amdxdna_program_image_store_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  // Guards the image list and the counters below. Image reference drops that
  // may reach zero also run under this mutex so a concurrent acquire never
  // observes an image that is being destroyed.
  iree_slim_mutex_t mutex;
  iree_hal_amdxdna_program_image_t* head;
  iree_host_size_t image_count;
  iree_host_size_t content_bytes;
};

struct iree_hal_amdxdna_program_image_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  // Retained owning store, or NULL for a private image.
  iree_hal_amdxdna_program_image_store_t* store;
  iree_hal_amdxdna_program_image_t* prev;
  iree_hal_amdxdna_program_image_t* next;
  uint64_t content_hash;
  iree_host_size_t content_length;
  iree_host_size_t host_bytes;
};

static uint8_t* iree_hal_amdxdna_program_image_data(
    const iree_hal_amdxdna_program_image_t* image) {
  return (uint8_t*)image + iree_sizeof_struct(*image);
}

iree_status_t iree_hal_amdxdna_program_image_store_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_program_image_store_t** out_store) {
  IREE_ASSERT_ARGUMENT(out_store);
  *out_store = NULL;
  iree_hal_amdxdna_program_image_store_t* store = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*store), (void**)&store));
  memset(store, 0, sizeof(*store));
  iree_atomic_ref_count_init(&store->ref_count);
  store->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&store->mutex);
  *out_store = store;
  return iree_ok_status();
}

void iree_hal_amdxdna_program_image_store_retain(
    iree_hal_amdxdna_program_image_store_t* store) {
  if (store) iree_atomic_ref_count_inc(&store->ref_count);
}

void iree_hal_amdxdna_program_image_store_release(
    iree_hal_amdxdna_program_image_store_t* store) {
  if (!store || iree_atomic_ref_count_dec(&store->ref_count) != 1) return;
  // Every live image retains the store, so the list is empty here.
  IREE_ASSERT(!store->head);
  iree_slim_mutex_deinitialize(&store->mutex);
  iree_allocator_free(store->host_allocator, store);
}

void iree_hal_amdxdna_program_image_store_query_stats(
    iree_hal_amdxdna_program_image_store_t* store,
    iree_hal_amdxdna_program_image_store_stats_t* out_stats) {
  memset(out_stats, 0, sizeof(*out_stats));
  if (!store) return;
  iree_slim_mutex_lock(&store->mutex);
  out_stats->image_count = store->image_count;
  out_stats->content_bytes = store->content_bytes;
  iree_slim_mutex_unlock(&store->mutex);
}

static iree_status_t iree_hal_amdxdna_program_image_allocate(
    iree_const_byte_span_t contents, uint64_t content_hash,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_program_image_t** out_image) {
  iree_hal_amdxdna_program_image_t* image = NULL;
  iree_host_size_t host_bytes = 0;
  if (!iree_host_size_checked_add(iree_sizeof_struct(*image),
                                  contents.data_length, &host_bytes)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna program image is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, host_bytes, (void**)&image));
  memset(image, 0, sizeof(*image));
  iree_atomic_ref_count_init(&image->ref_count);
  image->host_allocator = host_allocator;
  image->content_hash = content_hash;
  image->content_length = contents.data_length;
  image->host_bytes = host_bytes;
  memcpy(iree_hal_amdxdna_program_image_data(image), contents.data,
         contents.data_length);
  *out_image = image;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_program_image_acquire(
    iree_hal_amdxdna_program_image_store_t* store,
    iree_const_byte_span_t contents, iree_allocator_t host_allocator,
    iree_hal_amdxdna_program_image_t** out_image) {
  IREE_ASSERT_ARGUMENT(out_image);
  *out_image = NULL;
  if (contents.data_length == 0) return iree_ok_status();
  const uint64_t content_hash =
      iree_hal_amdxdna_hash_bytes(/*seed=*/0, contents);
  if (!store) {
    return iree_hal_amdxdna_program_image_allocate(contents, content_hash,
                                                   host_allocator, out_image);
  }

  iree_slim_mutex_lock(&store->mutex);
  for (iree_hal_amdxdna_program_image_t* image = store->head; image;
       image = image->next) {
    if (image->content_hash == content_hash &&
        image->content_length == contents.data_length &&
        memcmp(iree_hal_amdxdna_program_image_data(image), contents.data,
               contents.data_length) == 0) {
      iree_atomic_ref_count_inc(&image->ref_count);
      iree_slim_mutex_unlock(&store->mutex);
      *out_image = image;
      return iree_ok_status();
    }
  }
  iree_hal_amdxdna_program_image_t* image = NULL;
  iree_status_t status = iree_hal_amdxdna_program_image_allocate(
      contents, content_hash, store->host_allocator, &image);
  if (iree_status_is_ok(status)) {
    image->store = store;
    iree_hal_amdxdna_program_image_store_retain(store);
    image->next = store->head;
    if (store->head) store->head->prev = image;
    store->head = image;
    ++store->image_count;
    store->content_bytes += contents.data_length;
    *out_image = image;
  }
  iree_slim_mutex_unlock(&store->mutex);
  return status;
}

void iree_hal_amdxdna_program_image_retain(
    iree_hal_amdxdna_program_image_t* image) {
  if (image) iree_atomic_ref_count_inc(&image->ref_count);
}

void iree_hal_amdxdna_program_image_release(
    iree_hal_amdxdna_program_image_t* image) {
  if (!image) return;
  iree_hal_amdxdna_program_image_store_t* store = image->store;
  if (!store) {
    if (iree_atomic_ref_count_dec(&image->ref_count) == 1) {
      iree_allocator_free(image->host_allocator, image);
    }
    return;
  }
  iree_slim_mutex_lock(&store->mutex);
  const bool last_reference = iree_atomic_ref_count_dec(&image->ref_count) == 1;
  if (last_reference) {
    if (image->prev) {
      image->prev->next = image->next;
    } else {
      store->head = image->next;
    }
    if (image->next) image->next->prev = image->prev;
    --store->image_count;
    store->content_bytes -= image->content_length;
  }
  iree_slim_mutex_unlock(&store->mutex);
  if (last_reference) {
    iree_allocator_free(image->host_allocator, image);
    iree_hal_amdxdna_program_image_store_release(store);
  }
}

iree_const_byte_span_t iree_hal_amdxdna_program_image_contents(
    const iree_hal_amdxdna_program_image_t* image) {
  if (!image) return iree_const_byte_span_empty();
  return iree_make_const_byte_span(iree_hal_amdxdna_program_image_data(image),
                                   image->content_length);
}

iree_host_size_t iree_hal_amdxdna_program_image_host_bytes(
    const iree_hal_amdxdna_program_image_t* image) {
  return image ? image->host_bytes : 0;
}
