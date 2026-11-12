// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/backing.h"

#include "iree/hal/driver/amd/gpu/dispatch.h"
#include "iree/hal/driver/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_signal_cache_t) <=
                  3 * iree_hardware_destructive_interference_size,
              "signal capacity and stable wake ownership fit in three lines");

static void iree_hal_amd_gpu_signal_storage_release_metadata(void* user_data);

iree_status_t iree_hal_amd_gpu_grid_storage_initialize(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options,
    iree_hal_amd_gpu_grid_storage_t* out_storage) {
  *out_storage = (iree_hal_amd_gpu_grid_storage_t){0};
  amdf_memory_program_atomics_t program = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROGRAM_ATOMICS,
      .structure_size = sizeof(program),
  };
  amdf_memory_access_capabilities_t access = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access),
      .next = &program,
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_scope_query_device_profile(
          options->scope, options->profile.ordinal, 1, options->accesses,
          &profile, &access),
      "qualify cooperative grid backing"));
  if (!iree_all_bits_set(
          program.operations.device_scope_32,
          AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_ADD)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "resident cooperative grid backing lacks device-scope 32-bit LOAD/ADD");
  }
  uint64_t allocation_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocation_size(
      options, sizeof(iree_hal_amd_gpu_grid_sync_t), &allocation_length));
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = options->profile.ordinal,
      .access_count = 1,
      .required_flags = options->required_flags,
      .byte_length = allocation_length,
      .minimum_alignment = iree_max(8u, profile.allocation.minimum_alignment),
      .accesses = options->accesses,
      .placement = options->placement,
  };
  iree_hal_amd_gpu_grid_storage_t storage = {0};
  amdf_memory_release_info_t release_info = {0};
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_create(options->scope, &create_info, &storage.memory,
                                  &release_info),
      "allocate cooperative grid backing");
  if (!amdf_status_is_ok(release_info.status)) {
    iree_hal_amd_gpu_context_report(context, release_info.status,
                                    "cooperative grid acquisition cleanup");
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->memory_query_address(
            storage.memory, 0, AMDF_MEMORY_ADDRESS_GPU, &storage.address),
        "query cooperative grid address");
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_grid_storage_deinitialize(context, &storage);
  }
  return status;
}

void iree_hal_amd_gpu_grid_storage_deinitialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_grid_storage_t* storage) {
  if (storage->memory) {
    amdf_memory_release_info_t release_info = {0};
    const amdf_status_t status =
        context->api->memory_destroy(storage->memory, &release_info);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(context, status,
                                      "release cooperative grid backing");
    }
  }
  *storage = (iree_hal_amd_gpu_grid_storage_t){0};
}

iree_status_t iree_hal_amd_gpu_signal_cache_reserve_frontiers(
    iree_hal_amd_gpu_signal_cache_t* cache, uint32_t queue_count,
    iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  for (iree_hal_amd_gpu_signal_storage_t* storage = cache->storage;
       storage && iree_status_is_ok(status); storage = storage->next) {
    status = iree_hal_amd_gpu_signal_block_reserve_frontiers(
        &storage->block, queue_count, host_allocator);
  }
  return status;
}

void iree_hal_amd_gpu_frontier_request_process(
    iree_hal_amd_gpu_signal_cache_t* cache,
    iree_hal_amd_gpu_signal_storage_t* pending,
    iree_hal_amd_gpu_frontier_request_t* request) {
  iree_hal_amd_gpu_signal_storage_t* current = NULL;
  switch (request->phase) {
    case IREE_HAL_AMD_GPU_FRONTIER_BEGIN:
      cache->frontier_request = request;
      request->next = cache->storage;
      current = pending;
      break;
    case IREE_HAL_AMD_GPU_FRONTIER_ADVANCE:
      iree_hal_amd_gpu_signal_block_install_frontiers(&request->current->block,
                                                      request->frontiers);
      request->frontiers = NULL;
      break;
    case IREE_HAL_AMD_GPU_FRONTIER_CANCEL:
      request->next = NULL;
      break;
  }
  while (
      !current ||
      iree_hal_amd_gpu_signal_block_frontiers(&current->block)->queue_count >=
          request->queue_count) {
    current = request->next;
    if (!current) {
      break;
    }
    request->next = current->next;
  }
  request->current = current;
  if (current) {
    // Only the visited host metadata is retained. Trim/native failure may
    // still release its native backing while the caller allocates a table.
    iree_atomic_ref_count_inc(&current->metadata_ref_count);
  } else {
    cache->frontier_request = NULL;
  }
  iree_notification_t* notification = request->notification;
  iree_atomic_store(&request->completed, 1, iree_memory_order_release);
  // Queue release and frontier growth can share this notification. Wake every
  // enrolled predicate; an arbitrary waiter may belong to a different queue.
  iree_notification_post(notification, IREE_ALL_WAITERS);
}

static bool iree_hal_amd_gpu_frontier_request_completed(void* user_data) {
  iree_hal_amd_gpu_frontier_request_t* request = user_data;
  return iree_atomic_load(&request->completed, iree_memory_order_acquire);
}

static void iree_hal_amd_gpu_frontier_request_await(
    iree_hal_amd_gpu_frontier_request_t* request) {
  iree_notification_await(request->notification,
                          iree_hal_amd_gpu_frontier_request_completed, request,
                          iree_infinite_timeout());
}

iree_status_t iree_hal_amd_gpu_frontier_request_run(
    iree_hal_amd_gpu_frontier_request_t* request,
    iree_allocator_t host_allocator, void (*enqueue)(void*), void* user_data) {
  enqueue(user_data);
  iree_hal_amd_gpu_frontier_request_await(request);
  iree_allocator_free(host_allocator, request->vectors);
  request->vectors = NULL;
  iree_status_t status = iree_ok_status();
  while (request->current && iree_status_is_ok(status)) {
    iree_hal_amd_gpu_signal_storage_t* current = request->current;
    status = iree_hal_amd_gpu_signal_frontiers_allocate(
        current->capacity, request->queue_count, host_allocator,
        &request->frontiers);
    request->phase = iree_status_is_ok(status)
                         ? IREE_HAL_AMD_GPU_FRONTIER_ADVANCE
                         : IREE_HAL_AMD_GPU_FRONTIER_CANCEL;
    iree_atomic_store(&request->completed, 0, iree_memory_order_relaxed);
    enqueue(user_data);
    iree_hal_amd_gpu_frontier_request_await(request);
    iree_hal_amd_gpu_signal_storage_release_metadata(current);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_backing_query_address(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_t* memory, uint64_t* out_address) {
  if (!iree_all_bits_set(memory->host_properties,
                         IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GPU backing requires a coherent write-back mapping");
  }
  return out_address
             ? IREE_HAL_AMD_STATUS_FROM_AMDF(
                   context->api->memory_query_address(
                       memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, out_address),
                   "memory_query_address")
             : iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_backing_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, options, length, alignment, context->event_sink, memory,
      NULL));
  return iree_hal_amd_gpu_backing_query_address(context, memory, out_address);
}

iree_status_t iree_hal_amd_gpu_byte_storage_prepare(
    uint64_t capacity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_byte_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_byte_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  storage->host_allocator = host_allocator;
  storage->capacity = capacity;
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_byte_storage_initialize(
    iree_hal_amd_gpu_context_t* context, uint64_t begin,
    iree_hal_amd_gpu_byte_storage_t* storage) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_backing_query_address(
      context, &storage->memory, &storage->address));
  storage->begin = begin;
  storage->end = begin;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_byte_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_byte_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_byte_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_byte_storage_prepare(
      capacity, host_allocator, &storage));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, options, capacity, capacity, context->event_sink,
      &storage->memory, NULL);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_byte_storage_initialize(context, begin, storage);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_byte_storage_free(context, storage);
  }
  return status;
}

static void iree_hal_amd_gpu_byte_storage_released(void* user_data) {
  iree_hal_amd_gpu_byte_storage_t* storage = user_data;
  iree_allocator_free(storage->host_allocator, storage);
}

void iree_hal_amd_gpu_byte_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_byte_storage_t* storage) {
  iree_hal_amd_gpu_context_release_memory(
      context, 1, &storage->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_byte_storage_released, .user_data = storage},
      &storage->release);
}

iree_status_t iree_hal_amd_gpu_signal_storage_prepare(
    uint32_t capacity, uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  iree_host_size_t total_size = 0, signals_offset = 0, frontiers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*storage), &total_size,
      IREE_STRUCT_FIELD(capacity, iree_hal_amd_gpu_signal_t, &signals_offset),
      IREE_STRUCT_FIELD(1, iree_hal_amd_gpu_signal_frontiers_t,
                        &frontiers_offset),
      IREE_STRUCT_ARRAY_FIELD(capacity, queue_count, uint64_t, NULL)));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  storage->host_allocator = host_allocator;
  storage->capacity = capacity;
  iree_atomic_ref_count_init(&storage->metadata_ref_count);
  storage->frontiers = (void*)((uint8_t*)storage + frontiers_offset);
  iree_hal_amd_gpu_signal_block_initialize(
      (void*)((uint8_t*)storage + signals_offset), capacity, queue_count,
      storage->frontiers, &storage->block);
  *out_storage = storage;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_signal_storage_release_metadata(void* user_data) {
  iree_hal_amd_gpu_signal_storage_t* storage = user_data;
  if (iree_atomic_ref_count_dec(&storage->metadata_ref_count) != 1) {
    return;
  }
  iree_hal_amd_gpu_signal_frontiers_t* frontiers =
      iree_hal_amd_gpu_signal_block_frontiers(&storage->block);
  iree_hal_amd_gpu_signal_frontiers_free(frontiers, storage->frontiers,
                                         storage->host_allocator);
  iree_allocator_free(storage->host_allocator, storage);
}

void iree_hal_amd_gpu_signal_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_storage_t* storage) {
  if (storage->block.pool) {
    iree_hal_amd_gpu_signal_block_deinitialize(&storage->block);
  }
  iree_hal_amd_gpu_context_release_memory(
      context, 1, &storage->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_signal_storage_release_metadata,
          .user_data = storage},
      &storage->release);
}

static void iree_hal_amd_gpu_signal_cache_available(void* user_data) {
  iree_hal_amd_gpu_signal_cache_t* cache = user_data;
  if (iree_atomic_load(&cache->waiting, iree_memory_order_acquire) &&
      !iree_atomic_exchange(&cache->ready, 1, iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_publication_notify(cache->publication,
                                        cache->source_ordinal);
  }
}

void iree_hal_amd_gpu_signal_cache_initialize(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal,
    iree_hal_amd_gpu_signal_cache_t* out_cache) {
  *out_cache = (iree_hal_amd_gpu_signal_cache_t){
      .publication = publication, .source_ordinal = source_ordinal};
  iree_hal_amd_gpu_signal_pool_initialize(
      (iree_hal_amd_gpu_signal_pool_notification_t){
          .fn = iree_hal_amd_gpu_signal_cache_available,
          .user_data = out_cache},
      &out_cache->pool);
}

void iree_hal_amd_gpu_signal_cache_trim(
    iree_hal_amd_gpu_context_t* context, uint32_t minimum_block_count,
    iree_hal_amd_gpu_signal_cache_t* cache) {
  iree_hal_amd_gpu_signal_storage_t** link = &cache->storage;
  while (*link && cache->pool.block_count > minimum_block_count) {
    iree_hal_amd_gpu_signal_storage_t* storage = *link;
    if (iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block)) {
      *link = storage->next;
      if (cache->frontier_request && cache->frontier_request->next == storage) {
        cache->frontier_request->next = storage->next;
      }
      iree_hal_amd_gpu_signal_storage_free(context, storage);
    } else {
      link = &storage->next;
    }
  }
}

void iree_hal_amd_gpu_signal_cache_deinitialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_cache_t* cache) {
  iree_atomic_store(&cache->waiting, 0, iree_memory_order_release);
  iree_hal_amd_gpu_signal_cache_trim(context, 0, cache);
  iree_hal_amd_gpu_signal_pool_deinitialize(&cache->pool);
}
