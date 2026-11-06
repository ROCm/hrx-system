// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/growth.h"

#include "iree/hal/driver/amd/gpu/queue/sdma/execution.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/state.h"
#include "iree/hal/driver/amd/status.h"

struct iree_hal_amd_gpu_sdma_growth_t {
  // Native completion and an enrolled reuse notification own separate uses.
  iree_atomic_ref_count_t ref_count;
  // Metadata-only result continuation under the same native publisher.
  iree_hal_amd_gpu_sdma_submission_t submission;
  // Exact queue, retained by cold host-owner accounting.
  iree_hal_amd_gpu_sdma_queue_t* queue;
  // Pool storage returned after the result and entire waiting cohort detach.
  iree_host_size_t allocation_size;
  // Native acquisition result, consumed only once on the publisher.
  iree_status_t status;
  // Native completion was installed; later turns drain only captured waiters.
  bool installed;
  // Retained independently of invocations resumed by an availability hint.
  iree_hal_command_buffer_t* recording;
  // Coalesced program-capacity notification, independent of native completion.
  iree_hal_amd_gpu_sdma_submission_t reuse_submission;
  // One while a program reuse hint owns a reference and publication entry.
  iree_atomic_int32_t reuse_pending;
  // First parked operation; every entry retains its publication reference.
  iree_hal_amd_gpu_sdma_operation_t* first;
  // Last parked operation for constant-time coalescing.
  iree_hal_amd_gpu_sdma_operation_t* last;
  // Program request, or an empty program for native signal growth.
  iree_hal_amd_gpu_program_growth_t program;
  // Signal backing prepared before asynchronous native acquisition.
  iree_hal_amd_gpu_signal_storage_t* signals;
  // Signal allocation geometry borrowed through the native callback.
  iree_hal_amd_fabric_memory_request_t request;
  // Existing allocation-domain acquisition descriptor.
  iree_hal_amd_gpu_memory_acquire_t acquire;
};

static void iree_hal_amd_gpu_sdma_growth_release(
    iree_hal_amd_gpu_sdma_growth_t* growth) {
  if (growth && iree_atomic_ref_count_dec(&growth->ref_count) == 1) {
    iree_hal_amd_gpu_sdma_queue_t* queue = growth->queue;
    iree_hal_command_buffer_release(growth->recording);
    iree_hal_amd_gpu_capture_pool_release(
        &queue->captures, iree_make_byte_span(growth, growth->allocation_size));
    iree_hal_amd_gpu_sdma_host_end(&queue->base);
  }
}

// Detach bounded cohorts under exclusive publication. Resumed operations may
// miss again, but cannot append to this list until this turn finishes.
static bool iree_hal_amd_gpu_sdma_growth_resume(
    iree_hal_amd_gpu_sdma_growth_t* growth, iree_status_t status) {
  for (uint32_t i = 0; i < 8 && growth->first; ++i) {
    iree_hal_amd_gpu_sdma_operation_t* operation = growth->first;
    growth->first = operation->next;
    if (!growth->first) {
      growth->last = NULL;
    }
    operation->next = NULL;
    if (iree_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_execution_resume(operation);
    } else {
      iree_hal_amd_gpu_sdma_execution_abandon(operation,
                                              iree_status_clone(status));
    }
  }
  return growth->first != NULL;
}

static bool iree_hal_amd_gpu_sdma_program_reuse_prepare(void* user_data,
                                                        uint64_t position) {
  iree_hal_amd_gpu_sdma_growth_t* growth = user_data;
  if (!growth->installed) {
    iree_hal_amd_gpu_program_grow_cancel_reuse(&growth->program);
    if (iree_hal_amd_gpu_sdma_growth_resume(growth, iree_ok_status())) {
      iree_hal_amd_gpu_sdma_native_queue_defer(growth->queue->native,
                                               &growth->reuse_submission);
      return false;
    }
  }
  iree_atomic_store(&growth->reuse_pending, 0, iree_memory_order_release);
  iree_hal_amd_gpu_sdma_growth_release(growth);
  return false;
}

static void iree_hal_amd_gpu_sdma_program_reuse_rejected(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_sdma_program_reuse_prepare(user_data, 0);
}

static void iree_hal_amd_gpu_sdma_program_available(void* user_data) {
  iree_hal_amd_gpu_sdma_growth_t* growth = user_data;
  if (!iree_atomic_exchange(&growth->reuse_pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_atomic_ref_count_inc(&growth->ref_count);
    iree_hal_amd_gpu_sdma_native_queue_defer(growth->queue->native,
                                             &growth->reuse_submission);
  }
}

static bool iree_hal_amd_gpu_sdma_signal_reuse_prepare(void* user_data,
                                                       uint64_t position) {
  iree_hal_amd_gpu_sdma_queue_t* queue = user_data;
  if (queue->signal_growth && iree_hal_amd_gpu_signal_pool_has_available(
                                  &queue->options.signals->pool)) {
    iree_atomic_store(&queue->options.signals->waiting, 0,
                      iree_memory_order_release);
    if (iree_hal_amd_gpu_sdma_growth_resume(queue->signal_growth,
                                            iree_ok_status())) {
      iree_hal_amd_gpu_sdma_native_queue_defer(queue->native,
                                               &queue->signal_reuse.submission);
      return false;
    }
  }
  iree_atomic_store(&queue->signal_reuse.pending, 0, iree_memory_order_release);
  return false;
}

static void iree_hal_amd_gpu_sdma_signal_reuse_rejected(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_sdma_signal_reuse_prepare(user_data, 0);
}

void iree_hal_amd_gpu_sdma_growth_signals_available(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  if (iree_atomic_load(&queue->options.signals->waiting,
                       iree_memory_order_acquire) &&
      !iree_atomic_exchange(&queue->signal_reuse.pending, 1,
                            iree_memory_order_acq_rel)) {
    queue->signal_reuse.submission = (iree_hal_amd_gpu_sdma_submission_t){
        .prepare = iree_hal_amd_gpu_sdma_signal_reuse_prepare,
        .completion = {.notify = iree_hal_amd_gpu_sdma_signal_reuse_rejected,
                       .user_data = queue},
        .user_data = queue};
    iree_hal_amd_gpu_sdma_native_queue_defer(queue->native,
                                             &queue->signal_reuse.submission);
  }
}

static void iree_hal_amd_gpu_sdma_growth_await_reuse(
    iree_hal_amd_gpu_sdma_growth_t* growth) {
  if (growth->program.program) {
    iree_hal_amd_gpu_program_grow_await_reuse(
        &growth->program, iree_hal_amd_gpu_sdma_program_available, growth);
  } else {
    iree_hal_amd_gpu_sdma_queue_t* queue = growth->queue;
    iree_atomic_store(&queue->options.signals->waiting, 1,
                      iree_memory_order_release);
    if (iree_hal_amd_gpu_signal_pool_has_available(
            &queue->options.signals->pool)) {
      iree_hal_amd_gpu_sdma_growth_signals_available(queue);
    }
  }
}

iree_hal_amd_gpu_signal_storage_t* iree_hal_amd_gpu_sdma_growth_pending_signals(
    iree_hal_amd_gpu_sdma_queue_t* queue) {
  return queue->signal_growth ? queue->signal_growth->signals : NULL;
}

static void iree_hal_amd_gpu_sdma_growth_install(
    iree_hal_amd_gpu_sdma_growth_t* growth) {
  iree_hal_amd_gpu_sdma_queue_t* queue = growth->queue;
  growth->status = iree_status_join(growth->status,
                                    iree_hal_amd_gpu_sdma_check_failure(queue));
  if (growth->program.program) {
    growth->status =
        iree_hal_amd_gpu_program_grow_end(&growth->program, growth->status);
  } else {
    queue->signal_growth = NULL;
    iree_atomic_store(&queue->options.signals->waiting, 0,
                      iree_memory_order_release);
    if (iree_status_is_ok(growth->status)) {
      growth->status = queue->options.signal_layout.initialize(
          queue->context, &queue->options.signals->pool, growth->signals);
    }
    if (iree_status_is_ok(growth->status)) {
      growth->signals->next = queue->options.signals->storage;
      queue->options.signals->storage = growth->signals;
    } else {
      iree_hal_amd_gpu_signal_storage_free(queue->context, growth->signals);
    }
    growth->signals = NULL;
  }
  growth->installed = true;
}

// Metadata processing is bounded independently of the number of waiting calls.
// Native acquisition never publishes GPU work or walks the waiting cohort.
static bool iree_hal_amd_gpu_sdma_growth_prepare(void* user_data,
                                                 uint64_t position) {
  iree_hal_amd_gpu_sdma_growth_t* growth = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = growth->queue;
  if (!growth->installed) {
    iree_hal_amd_gpu_sdma_growth_install(growth);
  }
  if (iree_hal_amd_gpu_sdma_growth_resume(growth, growth->status)) {
    iree_hal_amd_gpu_sdma_native_queue_defer(queue->native,
                                             &growth->submission);
  } else {
    iree_status_free(growth->status);
    growth->status = iree_ok_status();
    iree_hal_amd_gpu_sdma_growth_release(growth);
  }
  return false;
}

static void iree_hal_amd_gpu_sdma_growth_rejected(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_sdma_growth_t* growth = user_data;
  growth->status = iree_status_join(
      growth->status, IREE_HAL_AMD_STATUS_FROM_AMDF(status, "SDMA growth"));
  iree_hal_amd_gpu_sdma_growth_prepare(growth, 0);
}

static void iree_hal_amd_gpu_sdma_growth_ready(void* user_data,
                                               iree_status_t status) {
  iree_hal_amd_gpu_sdma_growth_t* growth = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = growth->queue;
  growth->status = status;
  iree_hal_amd_gpu_sdma_native_queue_defer(queue->native, &growth->submission);
  // No growth access follows handoff: the publisher may already have freed it.
  iree_hal_amd_gpu_sdma_host_end(&queue->base);
}

iree_status_t iree_hal_amd_gpu_sdma_growth_begin(
    iree_hal_amd_gpu_sdma_operation_t* operation,
    iree_hal_amd_gpu_program_t* program) {
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_hal_amd_gpu_sdma_growth_t* growth = queue->signal_growth;
  if (program) {
    iree_hal_amd_gpu_program_growth_t* pending =
        iree_hal_amd_gpu_program_pending_growth(program,
                                                queue->options.queue_ordinal);
    growth = pending ? iree_containerof(pending, iree_hal_amd_gpu_sdma_growth_t,
                                        program)
                     : NULL;
  }
  if (growth) {
    if (growth->last) {
      growth->last->next = operation;
    } else {
      growth->first = operation;
    }
    growth->last = operation;
    iree_hal_amd_gpu_sdma_growth_await_reuse(growth);
    return iree_ok_status();
  }
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      &queue->captures, sizeof(*growth), &storage));
  growth = (void*)storage.data;
  *growth = (iree_hal_amd_gpu_sdma_growth_t){
      .submission =
          {
              .prepare = iree_hal_amd_gpu_sdma_growth_prepare,
              .completion =
                  {
                      .notify = iree_hal_amd_gpu_sdma_growth_rejected,
                      .user_data = growth,
                  },
              .user_data = growth,
          },
      .queue = queue,
      .reuse_submission =
          {.prepare = iree_hal_amd_gpu_sdma_program_reuse_prepare,
           .completion = {.notify =
                              iree_hal_amd_gpu_sdma_program_reuse_rejected,
                          .user_data = growth},
           .user_data = growth},
      .allocation_size = storage.data_length,
      .first = operation,
      .last = operation,
  };
  iree_atomic_ref_count_init(&growth->ref_count);
  iree_hal_amd_gpu_sdma_host_begin(&queue->base);
  iree_hal_amd_gpu_sdma_host_begin(&queue->base);
  const iree_hal_amd_fabric_memory_callback_t callback = {
      iree_hal_amd_gpu_sdma_growth_ready, growth};
  iree_status_t status = iree_ok_status();
  if (program) {
    growth->recording = operation->execute.invocation.command_buffer;
    iree_hal_command_buffer_retain(growth->recording);
    status = iree_hal_amd_gpu_program_grow_begin(
        program, queue->options.queue_ordinal, callback, &growth->program);
  } else {
    status = iree_hal_amd_gpu_signal_storage_prepare(
        IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY, queue->producer.count,
        queue->host_allocator, &growth->signals);
    if (iree_status_is_ok(status)) {
      queue->signal_growth = growth;
      growth->request = (iree_hal_amd_fabric_memory_request_t){
          .options = queue->options.memory,
          .byte_length = (uint64_t)IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY *
                         queue->options.signal_layout.stride,
          .alignment = queue->options.signal_layout.stride,
      };
      iree_hal_amd_gpu_context_acquire_memory(
          queue->context, 1, &growth->request, &growth->signals->memory,
          callback, &growth->acquire);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_growth_release(growth);
    iree_hal_amd_gpu_sdma_host_end(&queue->base);
  } else {
    iree_hal_amd_gpu_sdma_growth_await_reuse(growth);
  }
  return status;
}
