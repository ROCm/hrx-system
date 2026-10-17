// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/memory_mapping.h"

typedef struct iree_hal_amd_gpu_memory_mapping_t {
  // Completed native result, detached before terminal capture release.
  iree_atomic_slist_entry_t entry;
  // Borrowed device metadata, dominated by the capture's device reference.
  iree_hal_amd_gpu_memory_mapping_context_t* context;
  // Owns this capture, cloned lists, wait tracker, and mapping batch storage.
  iree_arena_allocator_t arena;
  // Captured and retained prerequisites; mapping resolves after they complete.
  iree_hal_semaphore_list_t waits;
  // Captured and retained public completion points.
  iree_hal_semaphore_list_t signals;
  // Owned logical transaction, retaining original source and target views.
  iree_hal_memory_mapping_batch_t* batch;
  // Dependency registration; finalization relinquishes it before resuming.
  iree_hal_amd_gpu_host_wait_t wait;
  // Embedded ready native request and callback/context join.
  iree_hal_amd_gpu_memory_map_t native;
  // Owned native result consumed once by the device publisher.
  iree_status_t status;
} iree_hal_amd_gpu_memory_mapping_t;

void iree_hal_amd_gpu_memory_mapping_context_initialize(
    iree_hal_device_t* device, iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_memory_mapping_context_t* out_context) {
  *out_context = (iree_hal_amd_gpu_memory_mapping_context_t){
      .device = device,
      .context = context,
      .publication = publication,
      .publication_ordinal = publication_ordinal};
  iree_atomic_slist_initialize(&out_context->ready);
  iree_arena_block_pool_initialize(1024, host_allocator, &out_context->blocks);
  iree_async_semaphore_wait_context_initialize(&out_context->waits);
}

void iree_hal_amd_gpu_memory_mapping_context_deinitialize(
    iree_hal_amd_gpu_memory_mapping_context_t* context) {
  iree_atomic_slist_deinitialize(&context->ready);
  iree_async_semaphore_wait_context_deinitialize(&context->waits);
  iree_arena_block_pool_deinitialize(&context->blocks);
}

static void iree_hal_amd_gpu_memory_mapping_free(
    iree_hal_amd_gpu_memory_mapping_t* operation) {
  iree_hal_device_t* device = operation->context->device;
  iree_arena_allocator_t arena = operation->arena;
  iree_hal_memory_mapping_batch_release(operation->batch);
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_arena_deinitialize(&arena);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_memory_mapping_complete(
    iree_hal_amd_gpu_memory_mapping_t* operation, iree_status_t status) {
  iree_hal_amd_gpu_memory_mapping_context_t* context = operation->context;
  if (context->profile) {
    iree_hal_amd_gpu_profile_operation_record(context->profile,
                                              (void*)(operation + 1), false, 0);
  }
  // Releasing the captured views precedes public completion. Installed aliases
  // borrow their physical storage under the caller's explicit lifetime
  // contract.
  iree_hal_memory_mapping_batch_release(operation->batch);
  operation->batch = NULL;
  if (iree_status_is_ok(status) && context->failure) {
    const iree_status_code_t code = (iree_status_code_t)iree_atomic_load(
        context->failure, iree_memory_order_acquire);
    if (code != IREE_STATUS_OK) {
      status = iree_make_status(code, "GPU mapping completed on failed device");
    }
  }
  if (iree_status_is_ok(status)) {
    // All input waits and native translation publication have completed.
    status = iree_hal_semaphore_list_signal(operation->signals, NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(context->context, context->device,
                                           status);
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_amd_gpu_memory_mapping_free(operation);
}

static bool iree_hal_amd_gpu_memory_mapping_publish(void* user_data) {
  iree_hal_amd_gpu_memory_mapping_context_t* context = user_data;
  if (!context->ready_head) {
    iree_atomic_slist_flush(&context->ready,
                            IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                            &context->ready_head, NULL);
  }
  for (uint32_t i = 0; i < 32 && context->ready_head; ++i) {
    iree_atomic_slist_entry_t* entry = context->ready_head;
    context->ready_head = entry->next;
    iree_hal_amd_gpu_memory_mapping_t* operation =
        iree_containerof(entry, iree_hal_amd_gpu_memory_mapping_t, entry);
    iree_hal_amd_gpu_memory_mapping_complete(operation, operation->status);
  }
  return context->ready_head != NULL;
}

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_memory_mapping_context_publication_source(
    iree_hal_amd_gpu_memory_mapping_context_t* context) {
  return (iree_hal_amd_gpu_publication_source_t){
      .publish = iree_hal_amd_gpu_memory_mapping_publish, .user_data = context};
}

static void iree_hal_amd_gpu_memory_mapping_mapped(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_gpu_memory_mapping_t* operation = user_data;
  iree_hal_amd_gpu_memory_mapping_context_t* context = operation->context;
  operation->status = status;
  // The context's native callback join protects source/notification storage
  // even when publication completes and releases the final device reference
  // before notify returns. No capture access follows its push.
  iree_atomic_slist_push(&context->ready, &operation->entry);
  iree_hal_amd_gpu_publication_notify(context->publication,
                                      context->publication_ordinal);
}

static void iree_hal_amd_gpu_memory_mapping_ready(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_memory_mapping_t* operation = user_data;
  if (operation->context->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->context->profile,
                                              (void*)(operation + 1),
                                              iree_status_is_ok(status), 0);
  }
  if (iree_status_is_ok(status) && operation->batch) {
    status = iree_hal_memory_mapping_batch_resolve(operation->batch);
  }
  if (!iree_status_is_ok(status) || !operation->batch) {
    iree_hal_amd_gpu_memory_mapping_complete(operation, status);
    return;
  }
  iree_hal_amd_gpu_context_map_memory(
      operation->context->context, operation->batch,
      (iree_hal_slab_acquire_callback_t){
          .fn = iree_hal_amd_gpu_memory_mapping_mapped, .user_data = operation},
      &operation->native);
}

iree_status_t iree_hal_amd_gpu_queue_map(
    iree_hal_amd_gpu_memory_mapping_context_t* context, iree_hal_queue_t* queue,
    uint32_t queue_ordinal, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&context->blocks, &arena);
  iree_hal_amd_gpu_memory_mapping_t* operation = NULL;
  iree_status_t status = iree_arena_allocate(
      &arena,
      sizeof(*operation) +
          (context->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0),
      (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  *operation =
      (iree_hal_amd_gpu_memory_mapping_t){.context = context, .arena = arena};
  iree_hal_device_retain(context->device);
  const iree_allocator_t allocator = iree_arena_allocator(&operation->arena);
  status = iree_hal_semaphore_list_clone(&waits, allocator, &operation->waits);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_list_clone(&signals, allocator, &operation->signals);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_mapping_batch_create(
        queue, mapping_count, mappings, flags, allocator, &operation->batch);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_memory_mapping_free(operation);
    return status;
  }
  if (context->profile) {
    iree_hal_amd_gpu_profile_operation_t* profile_operation =
        (void*)(operation + 1);
    iree_hal_amd_gpu_profile_operation_initialize(
        context->profile, queue_ordinal, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_MAP,
        waits.count, signals.count, mapping_count, profile_operation);
    iree_hal_amd_gpu_profile_operation_defer(profile_operation);
  }
  if (!operation->waits.count) {
    iree_hal_amd_gpu_memory_mapping_ready(operation, iree_ok_status());
  } else {
    status = iree_hal_amd_gpu_host_wait_begin(
        &context->waits, operation->waits,
        iree_hal_amd_gpu_memory_mapping_ready, operation, allocator,
        &operation->wait);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_memory_mapping_complete(operation, status);
    }
  }
  return iree_ok_status();
}
