// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/growth.h"

struct iree_hal_amd_gpu_aql_growth_t {
  // Typed native-result publication.
  iree_hal_amd_gpu_aql_entry_t entry;
  // Exact owner, kept live through the native callback join.
  iree_hal_amd_gpu_aql_growth_context_t* context;
  // Complete pooled extent in bytes.
  iree_host_size_t allocation_size;
  // First parked capture or NULL after an earlier reuse wake.
  iree_hal_amd_gpu_publication_entry_t* first;
  // Last parked capture for constant-time append.
  iree_hal_amd_gpu_publication_entry_t* last;
  // Native result transferred to the publisher.
  iree_status_t status;
  // Immutable acquisition geometry through completion.
  iree_hal_amd_fabric_memory_request_t request;
  // Embedded native lifetime/operation descriptor.
  iree_hal_amd_gpu_memory_acquire_t acquire;
  union {
    // Owned argument generation for ENTRY_ARGUMENTS.
    iree_hal_amd_gpu_argument_storage_t* arguments;
    // Owned descriptor generation for ENTRY_SIGNALS.
    iree_hal_amd_gpu_signal_storage_t* signals;
  };
};

static void iree_hal_amd_gpu_aql_growth_free(
    iree_hal_amd_gpu_aql_growth_t* growth) {
  iree_hal_amd_gpu_aql_growth_context_t* context = growth->context;
  iree_hal_amd_gpu_capture_pool_release(
      context->captures, iree_make_byte_span(growth, growth->allocation_size));
  iree_atomic_fetch_sub(&context->pending_count, 1, iree_memory_order_release);
}

static void iree_hal_amd_gpu_aql_growth_append(
    iree_hal_amd_gpu_aql_growth_t* growth,
    iree_hal_amd_gpu_aql_submission_t* submission) {
  iree_hal_amd_gpu_publication_entry_t* entry = &submission->entry.publication;
  iree_atomic_store(&entry->next, 0, iree_memory_order_relaxed);
  if (growth->last) {
    iree_atomic_store(&growth->last->next, (intptr_t)entry,
                      iree_memory_order_relaxed);
  } else {
    growth->first = entry;
  }
  growth->last = entry;
}

void iree_hal_amd_gpu_aql_growth_enqueue(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_t* entry) {
  if (iree_hal_amd_gpu_publisher_enqueue(context->publisher,
                                         &entry->publication)) {
    iree_hal_amd_gpu_publication_entry_t* ready =
        iree_hal_amd_gpu_publisher_next(context->publisher);
    if (iree_hal_amd_gpu_publisher_handoff(context->publisher, ready)) {
      iree_hal_amd_gpu_publication_notify(context->publication,
                                          context->queue_ordinal);
    }
  }
}

static void iree_hal_amd_gpu_aql_growth_completed(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_aql_growth_t* growth = user_data;
  growth->status = status;
  iree_hal_amd_gpu_aql_growth_enqueue(growth->context, &growth->entry);
}

void iree_hal_amd_gpu_aql_growth_signals_available(void* user_data) {
  iree_hal_amd_gpu_aql_growth_context_t* context = user_data;
  if (iree_atomic_load(&context->reuse.waiting, iree_memory_order_acquire) &&
      !iree_atomic_exchange(&context->reuse.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_growth_enqueue(context, &context->reuse.entry);
  }
}

static void iree_hal_amd_gpu_aql_growth_await_signals(
    iree_hal_amd_gpu_aql_growth_context_t* context) {
  iree_atomic_store(&context->reuse.waiting, 1, iree_memory_order_release);
  if (iree_hal_amd_gpu_signal_pool_has_available(context->signal_pool)) {
    iree_hal_amd_gpu_aql_growth_signals_available(context);
  }
}

iree_status_t iree_hal_amd_gpu_aql_growth_begin(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_kind_t kind,
    iree_hal_amd_gpu_aql_submission_t* submission, uint64_t capacity) {
  const bool arguments = kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS;
  iree_hal_amd_gpu_aql_growth_t** pending =
      arguments ? &context->arguments : &context->signals;
  if (*pending) {
    iree_hal_amd_gpu_aql_growth_append(*pending, submission);
    if (!arguments) {
      iree_hal_amd_gpu_aql_growth_await_signals(context);
    }
    return iree_ok_status();
  }
  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      context->captures, sizeof(iree_hal_amd_gpu_aql_growth_t), &storage));
  iree_hal_amd_gpu_aql_growth_t* growth = (void*)storage.data;
  *growth =
      (iree_hal_amd_gpu_aql_growth_t){.entry.kind = kind,
                                      .context = context,
                                      .allocation_size = storage.data_length};
  iree_atomic_fetch_add(&context->pending_count, 1, iree_memory_order_relaxed);
  iree_status_t status = iree_ok_status();
  iree_hal_amd_fabric_memory_t* memory = NULL;
  if (arguments) {
    status = iree_hal_amd_gpu_argument_storage_prepare(
        capacity, context->captures->host_allocator, &growth->arguments);
    if (iree_status_is_ok(status)) {
      memory = &growth->arguments->memory;
      growth->request =
          (iree_hal_amd_fabric_memory_request_t){.options = context->memory,
                                                 .byte_length = capacity,
                                                 .alignment = capacity};
    }
  } else {
    status = iree_hal_amd_gpu_signal_storage_prepare(
        context->signal_capacity, context->queue_count,
        context->captures->host_allocator, &growth->signals);
    if (iree_status_is_ok(status)) {
      memory = &growth->signals->memory;
      growth->request = (iree_hal_amd_fabric_memory_request_t){
          .options = context->memory,
          .byte_length = (uint64_t)context->signal_capacity *
                         sizeof(iree_hal_amd_gpu_aql_signal_t),
          .alignment = 64};
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_growth_append(growth, submission);
    *pending = growth;
    iree_hal_amd_gpu_context_acquire_memory(
        context->context, 1, &growth->request, memory,
        (iree_hal_amd_fabric_memory_callback_t){
            .fn = iree_hal_amd_gpu_aql_growth_completed, .user_data = growth},
        &growth->acquire);
    if (!arguments) {
      iree_hal_amd_gpu_aql_growth_await_signals(context);
    }
  } else {
    iree_hal_amd_gpu_aql_growth_free(growth);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_aql_growth_end(
    iree_hal_amd_gpu_aql_entry_t* entry, uint64_t argument_begin,
    iree_hal_amd_gpu_aql_growth_result_t* out_result) {
  iree_hal_amd_gpu_aql_growth_t* growth =
      iree_containerof(entry, iree_hal_amd_gpu_aql_growth_t, entry);
  iree_hal_amd_gpu_aql_growth_context_t* context = growth->context;
  *out_result = (iree_hal_amd_gpu_aql_growth_result_t){.first = growth->first,
                                                       .last = growth->last};
  iree_status_t status = growth->status;
  if (entry->kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS) {
    context->arguments = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_argument_storage_initialize(
          context->context, argument_begin, growth->arguments);
    }
    if (iree_status_is_ok(status)) {
      out_result->arguments = growth->arguments;
    } else {
      iree_hal_amd_gpu_argument_storage_free(context->context,
                                             growth->arguments);
    }
  } else {
    context->signals = NULL;
    iree_atomic_store(&context->reuse.waiting, 0, iree_memory_order_release);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_aql_signal_storage_initialize(
          context->context, context->signal_pool, growth->signals);
    }
    if (iree_status_is_ok(status)) {
      out_result->signals = growth->signals;
    } else {
      iree_hal_amd_gpu_signal_storage_free(context->context, growth->signals);
    }
  }
  iree_hal_amd_gpu_aql_growth_free(growth);
  return status;
}

void iree_hal_amd_gpu_aql_growth_reuse_signals(
    iree_hal_amd_gpu_aql_growth_context_t* context) {
  iree_atomic_store(&context->reuse.pending, 0, iree_memory_order_release);
  if (iree_hal_amd_gpu_signal_pool_has_available(context->signal_pool)) {
    iree_atomic_store(&context->reuse.waiting, 0, iree_memory_order_release);
    if (context->signals) {
      iree_hal_amd_gpu_publication_entry_t* first = context->signals->first;
      iree_hal_amd_gpu_publication_entry_t* last = context->signals->last;
      context->signals->first = context->signals->last = NULL;
      if (first) {
        iree_hal_amd_gpu_publisher_enqueue_batch(context->publisher, first,
                                                 last);
      }
    }
  }
}
