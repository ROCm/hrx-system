// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/queues.h"

typedef struct loom_runtime_execution_queue_t {
  // Retained exact queue, released before the inventory's parent device.
  iree_hal_queue_t* handle;
  // Live execution bindings; final returns may arrive from other threads.
  iree_atomic_int64_t borrowers;
} loom_runtime_execution_queue_t;

struct loom_runtime_execution_queues_t {
  // Device owner and every outstanding execution binding.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for the fixed inventory and its counters.
  iree_allocator_t host_allocator;
  // Retained parent of every queue in the inventory.
  iree_hal_device_t* device;
  // Number of entries in the dispatch family, starting at zero.
  iree_host_size_t dispatch_count;
  // Number of entries in the transfer family.
  iree_host_size_t transfer_count;
  // First transfer entry; zero when both roles share one family.
  iree_host_size_t transfer_offset;
  // Dispatch family if it provides a distinct, transfer-capable alternative.
  const iree_hal_queue_family_t* local_transfer_family;
  // Unique queue entries, dispatch followed by any distinct transfer family.
  loom_runtime_execution_queue_t entries[];
};

static iree_status_t loom_runtime_execution_queues_initialize_family(
    const iree_hal_queue_family_t* family,
    loom_runtime_execution_queue_t* entries) {
  const iree_host_size_t count =
      iree_hal_queue_family_spec(family)->provisioned_queue_count;
  if (!count) {
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    return iree_hal_queue_acquire(family, &params, &entries[0].handle);
  }
  iree_hal_device_t* device = iree_hal_queue_family_device(family);
  for (iree_host_size_t i = 0; i < count; ++i) {
    entries[i].handle =
        iree_hal_device_queue(device, iree_hal_queue_family_ordinal(family), i);
    iree_hal_queue_retain(entries[i].handle);
  }
  return iree_ok_status();
}

iree_status_t loom_runtime_execution_queues_create(
    const iree_hal_queue_family_t* dispatch_family,
    const iree_hal_queue_family_t* transfer_family,
    iree_allocator_t host_allocator,
    loom_runtime_execution_queues_t** out_queues) {
  *out_queues = NULL;
  const iree_host_size_t dispatch_count = iree_max(
      iree_hal_queue_family_spec(dispatch_family)->provisioned_queue_count, 1);
  const iree_host_size_t transfer_count = iree_max(
      iree_hal_queue_family_spec(transfer_family)->provisioned_queue_count, 1);
  const iree_host_size_t transfer_offset =
      dispatch_family == transfer_family ? 0 : dispatch_count;
  loom_runtime_execution_queues_t* queues = NULL;
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
  if (transfer_offset &&
      iree_any_bit_set(iree_hal_queue_family_spec(dispatch_family)->role_flags,
                       IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
    queues->local_transfer_family = dispatch_family;
  }
  iree_status_t status = loom_runtime_execution_queues_initialize_family(
      dispatch_family, queues->entries);
  if (iree_status_is_ok(status) && transfer_offset) {
    status = loom_runtime_execution_queues_initialize_family(
        transfer_family, queues->entries + transfer_offset);
  }
  if (!iree_status_is_ok(status)) {
    loom_runtime_execution_queues_release(queues);
    return status;
  }
  *out_queues = queues;
  return iree_ok_status();
}

void loom_runtime_execution_queues_retain(
    loom_runtime_execution_queues_t* queues) {
  if (queues) {
    iree_atomic_ref_count_inc(&queues->ref_count);
  }
}

void loom_runtime_execution_queues_release(
    loom_runtime_execution_queues_t* queues) {
  if (queues && iree_atomic_ref_count_dec(&queues->ref_count) == 1) {
    for (iree_host_size_t i = 0;
         i < queues->transfer_offset + queues->transfer_count; ++i) {
      iree_hal_queue_release(queues->entries[i].handle);
    }
    iree_hal_device_release(queues->device);
    iree_allocator_free(queues->host_allocator, queues);
  }
}

iree_hal_queue_t* loom_runtime_execution_queues_dispatch_queue(
    const loom_runtime_execution_queues_t* queues) {
  return queues->entries[0].handle;
}

iree_hal_queue_t* loom_runtime_execution_queues_transfer_queue(
    const loom_runtime_execution_queues_t* queues) {
  return queues->entries[queues->transfer_offset].handle;
}

static iree_hal_queue_ordinal_t loom_runtime_execution_queues_select(
    iree_host_size_t count, loom_runtime_execution_queue_t* entries) {
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

loom_runtime_execution_binding_t loom_runtime_execution_queues_acquire(
    loom_runtime_execution_queues_t* queues) {
  loom_runtime_execution_queues_retain(queues);
  const iree_hal_queue_ordinal_t dispatch =
      loom_runtime_execution_queues_select(queues->dispatch_count,
                                           queues->entries);
  const iree_hal_queue_ordinal_t transfer =
      queues->transfer_offset ? loom_runtime_execution_queues_select(
                                    queues->transfer_count,
                                    queues->entries + queues->transfer_offset)
                              : dispatch;
  return (loom_runtime_execution_binding_t){
      .owner = queues,
      .dispatch = queues->entries[dispatch].handle,
      .transfer = queues->entries[queues->transfer_offset + transfer].handle,
      .dispatch_ordinal = dispatch,
      .transfer_ordinal = transfer,
  };
}

// Bound the aggregate host capture and the transaction admitted beside model
// dispatches. Count payload bytes across all ranges, including device copies.
#define LOOM_RUNTIME_EXECUTION_LOCAL_TRANSFER_LIMIT (64 * 1024)

static bool loom_runtime_transfer_family_accessible(
    const iree_hal_queue_family_t* family,
    const iree_hal_transfer_operation_t* operation) {
  if (!iree_hal_transfer_operation_length(operation)) {
    return true;
  }
  iree_hal_buffer_t* source = NULL;
  iree_hal_buffer_t* target = NULL;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      target = operation->fill.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      target = operation->update.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      source = operation->copy.source_buffer;
      target = operation->copy.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      target = operation->upload.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      source = operation->download.source_buffer;
      break;
  }
  return (!source ||
          iree_any_bit_set(iree_hal_buffer_family_usage(source, family),
                           IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) &&
         (!target ||
          iree_any_bit_set(iree_hal_buffer_family_usage(target, family),
                           IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
}

loom_runtime_execution_transfer_plan_t
loom_runtime_execution_binding_plan_transfer(
    const loom_runtime_execution_binding_t* binding,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  loom_runtime_execution_transfer_plan_t plan = {.queue = binding->transfer};
  if (!operations && operation_count) {
    return plan;  // Diagnosed by HAL.
  }
  bool small = true;
  iree_device_size_t remaining = LOOM_RUNTIME_EXECUTION_LOCAL_TRANSFER_LIMIT;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const iree_device_size_t length =
        iree_hal_transfer_operation_length(&operations[i]);
    if (length > remaining) {
      small = false;
      plan.capture_uploads = false;
      break;
    }
    remaining -= length;
    plan.capture_uploads |=
        length && operations[i].type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  }
  const iree_hal_queue_family_t* family = binding->owner->local_transfer_family;
  if (!family) {
    return plan;
  }
  if (small) {
    plan.queue = binding->dispatch;
  } else {
    family = iree_hal_queue_family(binding->transfer);
  }
  // Pool and imported buffer scopes can restrict sources and targets to
  // different families. HAL validates the entire batch on the selected queue.
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    if (!loom_runtime_transfer_family_accessible(family, &operations[i])) {
      plan.queue = small ? binding->transfer : binding->dispatch;
      break;
    }
  }
  return plan;
}

void loom_runtime_execution_binding_release(
    loom_runtime_execution_binding_t binding) {
  loom_runtime_execution_queues_t* queues = binding.owner;
  iree_atomic_fetch_sub(&queues->entries[binding.dispatch_ordinal].borrowers, 1,
                        iree_memory_order_relaxed);
  if (queues->transfer_offset) {
    iree_atomic_fetch_sub(
        &queues->entries[queues->transfer_offset + binding.transfer_ordinal]
             .borrowers,
        1, iree_memory_order_relaxed);
  }
  loom_runtime_execution_queues_release(queues);
}
