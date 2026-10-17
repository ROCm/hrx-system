// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/queues.h"

typedef struct loom_serve_execution_queue_t {
  // Provisioned queue borrowed from the set's retained device.
  iree_hal_queue_t* handle;
  // Live execution bindings; final returns may arrive from other threads.
  iree_atomic_int64_t borrowers;
} loom_serve_execution_queue_t;

struct loom_serve_execution_queues_t {
  // Device owner and every outstanding execution binding.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for the fixed inventory and its counters.
  iree_allocator_t host_allocator;
  // Retained owner of every provisioned queue in the inventory.
  iree_hal_device_t* device;
  // Number of entries in the dispatch family, starting at zero.
  iree_host_size_t dispatch_count;
  // Number of entries in the transfer family.
  iree_host_size_t transfer_count;
  // First transfer entry; zero when both roles share one family.
  iree_host_size_t transfer_offset;
  // Unique queue entries, dispatch followed by any distinct transfer family.
  loom_serve_execution_queue_t entries[];
};

iree_status_t loom_serve_execution_queues_create(
    const iree_hal_queue_family_t* dispatch_family,
    const iree_hal_queue_family_t* transfer_family,
    iree_allocator_t host_allocator,
    loom_serve_execution_queues_t** out_queues) {
  *out_queues = NULL;
  const iree_host_size_t dispatch_count =
      iree_hal_queue_family_spec(dispatch_family)->provisioned_queue_count;
  const iree_host_size_t transfer_count =
      iree_hal_queue_family_spec(transfer_family)->provisioned_queue_count;
  const iree_host_size_t transfer_offset =
      dispatch_family == transfer_family ? 0 : dispatch_count;
  loom_serve_execution_queues_t* queues = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*queues), transfer_offset + transfer_count,
      sizeof(queues->entries[0]), (void**)&queues));
  iree_atomic_ref_count_init(&queues->ref_count);
  queues->host_allocator = host_allocator;
  queues->device = iree_hal_queue_family_device(dispatch_family);
  iree_hal_device_retain(queues->device);
  queues->dispatch_count = dispatch_count;
  queues->transfer_count = transfer_count;
  queues->transfer_offset = transfer_offset;
  for (iree_host_size_t i = 0; i < dispatch_count; ++i) {
    queues->entries[i].handle = iree_hal_device_queue(
        queues->device, iree_hal_queue_family_ordinal(dispatch_family), i);
  }
  if (transfer_offset) {
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      queues->entries[transfer_offset + i].handle = iree_hal_device_queue(
          queues->device, iree_hal_queue_family_ordinal(transfer_family), i);
    }
  }
  *out_queues = queues;
  return iree_ok_status();
}

void loom_serve_execution_queues_retain(loom_serve_execution_queues_t* queues) {
  if (queues) {
    iree_atomic_ref_count_inc(&queues->ref_count);
  }
}

void loom_serve_execution_queues_release(
    loom_serve_execution_queues_t* queues) {
  if (queues && iree_atomic_ref_count_dec(&queues->ref_count) == 1) {
    iree_hal_device_release(queues->device);
    iree_allocator_free(queues->host_allocator, queues);
  }
}

static iree_hal_queue_ordinal_t loom_serve_execution_queues_select(
    iree_host_size_t count, loom_serve_execution_queue_t* entries) {
  iree_hal_queue_ordinal_t ordinal = 0;
  int64_t borrowers =
      iree_atomic_load(&entries[0].borrowers, iree_memory_order_relaxed);
  for (iree_hal_queue_ordinal_t i = 1; i < count; ++i) {
    const int64_t candidate =
        iree_atomic_load(&entries[i].borrowers, iree_memory_order_relaxed);
    if (candidate < borrowers) {
      ordinal = i;
      borrowers = candidate;
    }
  }
  iree_atomic_fetch_add(&entries[ordinal].borrowers, 1,
                        iree_memory_order_relaxed);
  return ordinal;
}

loom_serve_execution_binding_t loom_serve_execution_queues_acquire(
    loom_serve_execution_queues_t* queues) {
  loom_serve_execution_queues_retain(queues);
  const iree_hal_queue_ordinal_t dispatch = loom_serve_execution_queues_select(
      queues->dispatch_count, queues->entries);
  const iree_hal_queue_ordinal_t transfer =
      queues->transfer_offset ? loom_serve_execution_queues_select(
                                    queues->transfer_count,
                                    queues->entries + queues->transfer_offset)
                              : dispatch;
  return (loom_serve_execution_binding_t){
      .owner = queues,
      .dispatch = queues->entries[dispatch].handle,
      .transfer = queues->entries[queues->transfer_offset + transfer].handle,
      .dispatch_ordinal = dispatch,
      .transfer_ordinal = transfer,
  };
}

void loom_serve_execution_binding_release(
    loom_serve_execution_binding_t binding) {
  loom_serve_execution_queues_t* queues = binding.owner;
  iree_atomic_fetch_sub(&queues->entries[binding.dispatch_ordinal].borrowers, 1,
                        iree_memory_order_relaxed);
  if (queues->transfer_offset) {
    iree_atomic_fetch_sub(
        &queues->entries[queues->transfer_offset + binding.transfer_ordinal]
             .borrowers,
        1, iree_memory_order_relaxed);
  }
  loom_serve_execution_queues_release(queues);
}
