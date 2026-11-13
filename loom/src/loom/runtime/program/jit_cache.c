// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/jit_cache.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/runtime/program/jit.h"

typedef struct loom_runtime_jit_cached_stage_t {
  // Exact stage root and configuration identity within the generation.
  loom_sha256_t key;
  // Retained immutable stage shared by subsequent model catalogs.
  loom_runtime_jit_stage_t* stage;
  // Owned compiler diagnostics replayed for every equivalent model.
  loom_runtime_compilation_diagnostics_t* diagnostics;
} loom_runtime_jit_cached_stage_t;

typedef struct loom_runtime_jit_cached_vm_t {
  // Exact primary source and ordered root identity within the generation.
  loom_sha256_t key;
  // Owned immutable VM bytecode cloned into each fresh module.
  iree_const_byte_span_t image;
  // Owned compiler diagnostics replayed for every equivalent model.
  loom_runtime_compilation_diagnostics_t* diagnostics;
} loom_runtime_jit_cached_vm_t;

struct loom_runtime_jit_generation_t {
  // Cache ownership and each active model catalog retain independently.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for this generation and all retained products.
  iree_allocator_t host_allocator;
  // Short product-table lock; never held across compiler or VM work.
  iree_slim_mutex_t mutex;
  // Exact source, selected configuration, and sanitizer identity.
  loom_sha256_t identity;
  // Published immutable native stages.
  loom_runtime_jit_cached_stage_t* stages;
  // Number of initialized stage entries.
  iree_host_size_t stage_count;
  // Allocated stage entry capacity.
  iree_host_size_t stage_capacity;
  // Published immutable VM images.
  loom_runtime_jit_cached_vm_t* vm_images;
  // Number of initialized VM image entries.
  iree_host_size_t vm_image_count;
  // Allocated VM image entry capacity.
  iree_host_size_t vm_image_capacity;
};

struct loom_runtime_jit_cache_t {
  // Allocation policy for the cache and new generations.
  iree_allocator_t host_allocator;
  // Borrowed fixed device domain owned beyond cache destruction.
  iree_hal_device_t* device;
  // Borrowed exact queue domain owned beyond cache destruction.
  iree_hal_queue_t* dispatch;
  // Serializes generation lookup, insertion, and trimming.
  iree_slim_mutex_t mutex;
  // Unordered exact source generations; expected counts are small.
  loom_runtime_jit_generation_t** generations;
  // Number of initialized generation entries.
  iree_host_size_t generation_count;
  // Allocated generation pointer capacity.
  iree_host_size_t generation_capacity;
};

static iree_status_t loom_runtime_jit_generation_create(
    iree_allocator_t host_allocator, const loom_sha256_t* identity,
    loom_runtime_jit_generation_t** out_generation) {
  *out_generation = NULL;
  loom_runtime_jit_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*generation), (void**)&generation));
  iree_atomic_ref_count_init(&generation->ref_count);
  generation->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&generation->mutex);
  generation->identity = *identity;
  *out_generation = generation;
  return iree_ok_status();
}

void loom_runtime_jit_generation_retain(
    loom_runtime_jit_generation_t* generation) {
  if (generation) {
    iree_atomic_ref_count_inc(&generation->ref_count);
  }
}

void loom_runtime_jit_generation_release(
    loom_runtime_jit_generation_t* generation) {
  if (!generation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&generation->ref_count) != 1) {
    return;
  }
  for (iree_host_size_t i = 0; i < generation->stage_count; ++i) {
    loom_runtime_jit_stage_free(generation->stages[i].stage);
    iree_allocator_free(generation->host_allocator,
                        generation->stages[i].diagnostics);
  }
  for (iree_host_size_t i = 0; i < generation->vm_image_count; ++i) {
    iree_allocator_free(generation->host_allocator,
                        (void*)generation->vm_images[i].image.data);
    iree_allocator_free(generation->host_allocator,
                        generation->vm_images[i].diagnostics);
  }
  iree_allocator_free(generation->host_allocator, generation->stages);
  iree_allocator_free(generation->host_allocator, generation->vm_images);
  iree_slim_mutex_deinitialize(&generation->mutex);
  iree_allocator_free(generation->host_allocator, generation);
}

void loom_runtime_jit_generation_trim(
    loom_runtime_jit_generation_t* generation) {
  if (!generation) {
    return;
  }
  iree_slim_mutex_lock(&generation->mutex);
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < generation->stage_count; ++i) {
    loom_runtime_jit_cached_stage_t entry = generation->stages[i];
    if (loom_runtime_jit_stage_is_shared(entry.stage)) {
      generation->stages[count++] = entry;
    } else {
      loom_runtime_jit_stage_free(entry.stage);
      iree_allocator_free(generation->host_allocator, entry.diagnostics);
    }
  }
  generation->stage_count = count;
  iree_slim_mutex_unlock(&generation->mutex);
}

iree_status_t loom_runtime_jit_cache_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* dispatch,
    iree_allocator_t host_allocator, loom_runtime_jit_cache_t** out_cache) {
  *out_cache = NULL;
  loom_runtime_jit_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache));
  cache->host_allocator = host_allocator;
  cache->device = device;
  cache->dispatch = dispatch;
  iree_slim_mutex_initialize(&cache->mutex);
  *out_cache = cache;
  return iree_ok_status();
}

iree_hal_device_t* loom_runtime_jit_cache_device(
    const loom_runtime_jit_cache_t* cache) {
  return cache->device;
}

iree_hal_queue_t* loom_runtime_jit_cache_dispatch(
    const loom_runtime_jit_cache_t* cache) {
  return cache->dispatch;
}

iree_allocator_t loom_runtime_jit_cache_allocator(
    const loom_runtime_jit_cache_t* cache) {
  return cache->host_allocator;
}

void loom_runtime_jit_cache_free(loom_runtime_jit_cache_t* cache) {
  if (!cache) {
    return;
  }
  for (iree_host_size_t i = 0; i < cache->generation_count; ++i) {
    loom_runtime_jit_generation_release(cache->generations[i]);
  }
  iree_allocator_free(cache->host_allocator, cache->generations);
  iree_slim_mutex_deinitialize(&cache->mutex);
  iree_allocator_free(cache->host_allocator, cache);
}

void loom_runtime_jit_cache_trim(loom_runtime_jit_cache_t* cache) {
  iree_slim_mutex_lock(&cache->mutex);
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < cache->generation_count; ++i) {
    loom_runtime_jit_generation_t* generation = cache->generations[i];
    if (iree_atomic_ref_count_load(&generation->ref_count) == 1) {
      loom_runtime_jit_generation_release(generation);
    } else {
      cache->generations[count++] = generation;
    }
  }
  cache->generation_count = count;
  iree_slim_mutex_unlock(&cache->mutex);
}

iree_status_t loom_runtime_jit_cache_acquire(
    loom_runtime_jit_cache_t* cache, const loom_sha256_t* identity,
    loom_runtime_jit_generation_t** out_generation) {
  *out_generation = NULL;
  iree_slim_mutex_lock(&cache->mutex);
  for (iree_host_size_t i = 0; i < cache->generation_count; ++i) {
    loom_runtime_jit_generation_t* generation = cache->generations[i];
    if (loom_sha256_equal(&generation->identity, identity)) {
      loom_runtime_jit_generation_retain(generation);
      *out_generation = generation;
      iree_slim_mutex_unlock(&cache->mutex);
      return iree_ok_status();
    }
  }
  iree_status_t status = iree_ok_status();
  if (cache->generation_count == cache->generation_capacity) {
    status = iree_allocator_grow_array(
        cache->host_allocator, cache->generation_count + 1,
        sizeof(*cache->generations), &cache->generation_capacity,
        (void**)&cache->generations);
  }
  loom_runtime_jit_generation_t* generation = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_runtime_jit_generation_create(cache->host_allocator, identity,
                                                &generation);
  }
  if (iree_status_is_ok(status)) {
    cache->generations[cache->generation_count++] = generation;
    loom_runtime_jit_generation_retain(generation);
    *out_generation = generation;
  }
  iree_slim_mutex_unlock(&cache->mutex);
  return status;
}

loom_runtime_jit_stage_t* loom_runtime_jit_generation_lookup_stage(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_compilation_report_t* report) {
  loom_runtime_jit_stage_t* stage = NULL;
  const loom_runtime_compilation_diagnostics_t* diagnostics = NULL;
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->stage_count; ++i) {
    if (loom_sha256_equal(&generation->stages[i].key, key)) {
      stage = generation->stages[i].stage;
      // The retained stage also pins its cache entry and diagnostics against
      // generation_trim after this lock is released.
      loom_runtime_jit_stage_retain(stage);
      diagnostics = generation->stages[i].diagnostics;
      break;
    }
  }
  iree_slim_mutex_unlock(&generation->mutex);
  loom_runtime_compilation_diagnostics_merge(diagnostics, report);
  return stage;
}

iree_status_t loom_runtime_jit_generation_publish_stage(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_jit_stage_t* stage,
    const loom_runtime_compilation_report_t* report) {
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->stage_count; ++i) {
    if (loom_sha256_equal(&generation->stages[i].key, key)) {
      iree_slim_mutex_unlock(&generation->mutex);
      return iree_ok_status();
    }
  }
  iree_slim_mutex_unlock(&generation->mutex);
  loom_runtime_compilation_diagnostics_t* diagnostics = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_compilation_diagnostics_clone(
      report, generation->host_allocator, &diagnostics));
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->stage_count; ++i) {
    if (loom_sha256_equal(&generation->stages[i].key, key)) {
      iree_slim_mutex_unlock(&generation->mutex);
      iree_allocator_free(generation->host_allocator, diagnostics);
      return iree_ok_status();
    }
  }
  iree_status_t status = iree_ok_status();
  if (generation->stage_count == generation->stage_capacity) {
    status = iree_allocator_grow_array(
        generation->host_allocator, generation->stage_count + 1,
        sizeof(*generation->stages), &generation->stage_capacity,
        (void**)&generation->stages);
  }
  if (iree_status_is_ok(status)) {
    loom_runtime_jit_stage_retain(stage);
    generation->stages[generation->stage_count++] =
        (loom_runtime_jit_cached_stage_t){
            .key = *key, .stage = stage, .diagnostics = diagnostics};
  }
  iree_slim_mutex_unlock(&generation->mutex);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(generation->host_allocator, diagnostics);
  }
  return status;
}

iree_status_t loom_runtime_jit_generation_lookup_vm(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    bool* out_found, iree_const_byte_span_t* out_image) {
  *out_found = false;
  *out_image = iree_const_byte_span_empty();
  bool found = false;
  iree_const_byte_span_t image = iree_const_byte_span_empty();
  const loom_runtime_compilation_diagnostics_t* diagnostics = NULL;
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->vm_image_count; ++i) {
    const loom_runtime_jit_cached_vm_t* cached = &generation->vm_images[i];
    if (loom_sha256_equal(&cached->key, key)) {
      // The caller's generation reference pins immutable VM entries. They
      // retire only with the complete generation, never during active use.
      found = true;
      image = cached->image;
      diagnostics = cached->diagnostics;
      break;
    }
  }
  iree_slim_mutex_unlock(&generation->mutex);
  if (!found) {
    return iree_ok_status();
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(host_allocator, image, (void**)&data));
  *out_image = iree_make_const_byte_span(data, image.data_length);
  loom_runtime_compilation_diagnostics_merge(diagnostics, report);
  *out_found = true;
  return iree_ok_status();
}

iree_status_t loom_runtime_jit_generation_publish_vm(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    iree_const_byte_span_t image,
    const loom_runtime_compilation_report_t* report) {
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->vm_image_count; ++i) {
    if (loom_sha256_equal(&generation->vm_images[i].key, key)) {
      iree_slim_mutex_unlock(&generation->mutex);
      return iree_ok_status();
    }
  }
  iree_slim_mutex_unlock(&generation->mutex);
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(generation->host_allocator, image, (void**)&data));
  loom_runtime_compilation_diagnostics_t* diagnostics = NULL;
  iree_status_t status = loom_runtime_compilation_diagnostics_clone(
      report, generation->host_allocator, &diagnostics);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(generation->host_allocator, data);
    return status;
  }
  iree_slim_mutex_lock(&generation->mutex);
  for (iree_host_size_t i = 0; i < generation->vm_image_count; ++i) {
    if (loom_sha256_equal(&generation->vm_images[i].key, key)) {
      iree_slim_mutex_unlock(&generation->mutex);
      iree_allocator_free(generation->host_allocator, diagnostics);
      iree_allocator_free(generation->host_allocator, data);
      return iree_ok_status();
    }
  }
  if (generation->vm_image_count == generation->vm_image_capacity) {
    status = iree_allocator_grow_array(
        generation->host_allocator, generation->vm_image_count + 1,
        sizeof(*generation->vm_images), &generation->vm_image_capacity,
        (void**)&generation->vm_images);
  }
  if (iree_status_is_ok(status)) {
    generation->vm_images[generation->vm_image_count++] =
        (loom_runtime_jit_cached_vm_t){
            .key = *key,
            .image = iree_make_const_byte_span(data, image.data_length),
            .diagnostics = diagnostics};
  } else {
    iree_allocator_free(generation->host_allocator, diagnostics);
    iree_allocator_free(generation->host_allocator, data);
  }
  iree_slim_mutex_unlock(&generation->mutex);
  return status;
}
