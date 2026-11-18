// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"

#include "iree/async/file.h"
#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/fabric/buffer.h"
#include "iree/hal/driver/amd/fabric/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/file_completion.h"

typedef struct iree_hal_amd_gpu_file_pipeline_t
    iree_hal_amd_gpu_file_pipeline_t;

typedef struct iree_hal_amd_gpu_file_transfer_t {
  // Borrowed queue-owned services, live through operation retirement.
  const iree_hal_amd_gpu_file_executor_t* executor;
  // Exact capture extent returned to the reusable metadata pool.
  iree_host_size_t allocation_size;
  // Retained file/buffer, partial I/O state, and optional owned host staging.
  iree_hal_amd_fabric_file_transfer_t transfer;
  // Qualified address slot for staged copies, resolved after dependencies.
  iree_hal_amd_gpu_buffer_reference_t reference;
  // Copied and retained dependencies, joined before accessing payloads.
  iree_hal_semaphore_list_t waits;
  // Copied and retained public completion points.
  iree_hal_semaphore_list_t signals;
  // Prepared cache boundary preceding the first native buffer access.
  void* before;
  // Prepared cache boundary following every file and host effect.
  void* after;
  // Joins each submitting call with its potentially immediate completion.
  iree_atomic_int32_t pending;
  // Next pipeline step, run by the last submission/completion borrower.
  iree_hal_amd_gpu_host_wait_fn_t resume;
  // Terminal result published before the completion borrow is returned.
  iree_status_t status;
  // Last retired native epoch, preserved when publishing public completion.
  iree_async_single_frontier_t frontier;
  // Public dependencies, finalized before the first native submission.
  struct {
    // One tracker registration, finalized before file/native work starts.
    iree_hal_amd_gpu_host_wait_t wait;
    // Inline tracker allocation sized for the initial public waits.
    iree_allocator_inline_storage_t storage;
  } dependency;
} iree_hal_amd_gpu_file_transfer_t;

static void iree_hal_amd_gpu_file_transfer_join(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  iree_status_t status = operation->status;
  operation->status = iree_ok_status();
  operation->resume(operation, status);
}

static void iree_hal_amd_gpu_file_transfer_step_complete(void* user_data,
                                                         iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  operation->status = status;
  iree_hal_amd_gpu_file_transfer_join(operation);
}

static void iree_hal_amd_gpu_file_transfer_native_complete(
    void* user_data, iree_status_t status,
    const iree_async_single_frontier_t* frontier, bool retired) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (!retired) {
    // The native owner retains this failure through its final callback. An
    // early notification cannot publish completion while file bytes are live.
    iree_status_free(status);
    return;
  }
  if (frontier) {
    operation->frontier = *frontier;
  }
  iree_hal_amd_gpu_file_transfer_step_complete(operation, status);
}

// Native retirement may precede submit returning. Join both accesses before
// advancing mutable I/O state or releasing the file's borrowed bytes.
static void iree_hal_amd_gpu_file_transfer_submit_native(
    iree_hal_amd_gpu_file_transfer_t* operation,
    const iree_hal_amd_gpu_transfer_operation_t* transfer, const void* before,
    const void* after, iree_hal_amd_gpu_host_wait_fn_t resume) {
  operation->resume = resume;
  iree_atomic_store(&operation->pending, 2, iree_memory_order_relaxed);
  iree_status_t status = operation->executor->submit(
      operation->executor->queue, transfer ? 1 : 0, transfer, before, after,
      (iree_hal_amd_gpu_retirement_callback_t){
          .fn = iree_hal_amd_gpu_file_transfer_native_complete,
          .user_data = operation});
  if (!iree_status_is_ok(status)) {
    // Rejected work never acquired native ownership or its callback borrow.
    iree_hal_amd_gpu_file_transfer_step_complete(operation, status);
  }
  iree_hal_amd_gpu_file_transfer_join(operation);
}

static iree_status_t iree_hal_amd_gpu_file_transfer_copy_native(
    void* user_data, uint64_t source, uint64_t target, uint64_t length,
    iree_hal_memory_effects_t before, iree_hal_memory_effects_t after,
    iree_hal_amd_gpu_retirement_callback_t callback) {
  const iree_hal_amd_gpu_file_executor_t* executor = user_data;
  const iree_hal_amd_gpu_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .source.device = iree_hal_amd_gpu_buffer_reference_borrow(source),
      .target.device = iree_hal_amd_gpu_buffer_reference_borrow(target),
      .length = length};
  // The queue fixes barrier_size at construction; both values fit on the
  // submitting thread's stack and are captured by submit before returning.
  void* before_barrier = iree_alloca(executor->barrier_size);
  void* after_barrier = iree_alloca(executor->barrier_size);
  memset(before_barrier, 0, executor->barrier_size);
  memset(after_barrier, 0, executor->barrier_size);
  executor->add_effects(executor->queue, before, before_barrier);
  executor->add_effects(executor->queue, after, after_barrier);
  return executor->submit(executor->queue, 1, &copy, before_barrier,
                          after_barrier, callback);
}

static void iree_hal_amd_gpu_file_transfer_free(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  const iree_hal_amd_gpu_file_executor_t* executor = operation->executor;
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_gpu_capture_pool_release(
      operation->executor->captures,
      iree_make_byte_span(operation, operation->allocation_size));
  executor->end(executor->queue);
}

static void iree_hal_amd_gpu_file_transfer_retire(void* user_data,
                                                  iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->transfer.length);
  if (operation->executor->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->executor->profile,
                                              (void*)(operation + 1), false, 0);
  }
  // Native copies, file requests, and the final cache boundary have retired.
  // Keep backing live until dirty cache lines can no longer refer to it.
  iree_hal_amd_fabric_file_transfer_deinitialize(&operation->transfer);
  if (iree_status_is_ok(status)) {
    // Preserve the completed queue epoch after all buffer/pool uses have ended.
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        operation->frontier.entry_count
            ? iree_async_single_frontier_as_const_frontier(&operation->frontier)
            : NULL);
  }
  if (!iree_status_is_ok(status)) {
    // File/dependency failures belong to this operation. Native queue owners
    // report their own device failures; a propagated status is not a new one.
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  IREE_TRACE_ZONE_END(z0);
  iree_hal_amd_gpu_file_transfer_free(operation);
}

static void iree_hal_amd_gpu_file_transfer_complete(void* user_data,
                                                    iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_retire(operation, status);
    return;
  }
  iree_hal_amd_gpu_file_transfer_submit_native(
      operation, NULL, NULL, operation->after,
      iree_hal_amd_gpu_file_transfer_retire);
}

#define IREE_HAL_AMD_GPU_FILE_DRAINING (INT64_C(1) << 32)

typedef enum iree_hal_amd_gpu_file_window_phase_e {
  IREE_HAL_AMD_GPU_FILE_WINDOW_IDLE = 0,
  IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED,
  IREE_HAL_AMD_GPU_FILE_WINDOW_READ,
  IREE_HAL_AMD_GPU_FILE_WINDOW_COPY,
  IREE_HAL_AMD_GPU_FILE_WINDOW_WRITE,
  IREE_HAL_AMD_GPU_FILE_WINDOW_RELEASE,
} iree_hal_amd_gpu_file_window_phase_t;

typedef struct iree_hal_amd_gpu_file_window_t {
  // Enclosing finite pipeline owns this window through final release.
  iree_hal_amd_gpu_file_pipeline_t* pipeline;
  // Exact physical loan and reusable native registration membership.
  iree_hal_amd_gpu_file_staging_request_t staging;
  // A failed copy cannot start file I/O, so error delivery reuses request
  // space.
  union {
    // Independent native file request, live until its completion callback.
    iree_hal_file_request_t request;
    // Error-only native retirement handoff to the file's poll owner.
    iree_hal_amd_gpu_file_completion_t failure;
  };
  // Completion result published by the ready bit.
  iree_status_t status;
  // Single-owner next step; callbacks publish completion without changing it.
  iree_hal_amd_gpu_file_window_phase_t phase;
  // The first native error initializes the handoff before notifying its owner.
  bool failure_initialized;
} iree_hal_amd_gpu_file_window_t;

struct iree_hal_amd_gpu_file_pipeline_t {
  // Enclosing accepted file operation retains operands and executor services.
  iree_hal_amd_gpu_file_transfer_t* operation;
  // Capture extent returned after all native access and registrations retire.
  iree_host_size_t allocation_size;
  // First byte not yet assigned to an independent window.
  uint64_t cursor;
  // Qualified native destination/source address of transfer byte zero.
  uint64_t address;
  // Native file alignment, qualified before the first reservation.
  uint64_t alignment;
  // Completed window bits and the single inline drainer's ownership bit.
  iree_atomic_int64_t ready;
  // First failure stops new admission without releasing accepted accesses.
  iree_atomic_intptr_t failure;
  // Windows with outstanding storage, native I/O, or device-copy callbacks.
  uint32_t active;
  // Number of placed windows whose staging memberships were acquired.
  uint32_t count;
  // Final phase joins each foreign-proactor registration release.
  bool releasing;
  // Bounded independent lanes; count depends on concurrency, not file size.
  iree_hal_amd_gpu_file_window_t windows[];
};

static void iree_hal_amd_gpu_file_pipeline_drain(
    iree_hal_amd_gpu_file_pipeline_t* pipeline);

static void iree_hal_amd_gpu_file_pipeline_notify(
    iree_hal_amd_gpu_file_pipeline_t* pipeline, uint32_t completed) {
  const int64_t previous = iree_atomic_fetch_or(
      &pipeline->ready, IREE_HAL_AMD_GPU_FILE_DRAINING | completed,
      iree_memory_order_acq_rel);
  if (!(previous & IREE_HAL_AMD_GPU_FILE_DRAINING)) {
    iree_hal_amd_gpu_file_pipeline_drain(pipeline);
  }
  // Publication ends this callback's borrow; the pipeline may now be reclaimed.
}

static void iree_hal_amd_gpu_file_window_complete(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_file_window_t* window = user_data;
  window->status = status;
  iree_hal_amd_gpu_file_pipeline_t* pipeline = window->pipeline;
  iree_hal_amd_gpu_file_pipeline_notify(
      pipeline, UINT32_C(1) << (window - pipeline->windows));
}

// Retain the first failure for terminal publication. A later error adds no
// retirement proof and cannot replace the original failure's diagnostic.
static bool iree_hal_amd_gpu_file_pipeline_fail(
    iree_hal_amd_gpu_file_pipeline_t* pipeline, iree_status_t status) {
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &pipeline->failure, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_relaxed)) {
    return true;
  }
  iree_status_free(status);
  return false;
}

static void iree_hal_amd_gpu_file_window_failed(void* user_data,
                                                iree_status_t status,
                                                bool retired) {
  iree_hal_amd_gpu_file_window_t* window = user_data;
  iree_hal_amd_gpu_file_pipeline_t* pipeline = window->pipeline;
  if (iree_hal_amd_gpu_file_pipeline_fail(pipeline,
                                          iree_status_clone(status))) {
    const iree_hal_amd_gpu_file_executor_t* executor =
        pipeline->operation->executor;
    iree_hal_amd_gpu_context_report_status(
        executor->context, executor->queue->queue_family->device, status);
  }
  if (retired) {
    iree_hal_amd_gpu_file_window_complete(window, status);
  } else {
    iree_status_free(status);
    iree_hal_amd_gpu_file_pipeline_notify(pipeline, 0);
  }
}

static void iree_hal_amd_gpu_file_window_copied(
    void* user_data, iree_status_t status,
    const iree_async_single_frontier_t* frontier, bool retired) {
  iree_hal_amd_gpu_file_window_t* window = user_data;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_window_complete(window, status);
  } else {
    if (!window->failure_initialized) {
      window->failure_initialized = true;
      iree_hal_amd_gpu_file_completion_initialize(
          window->pipeline->operation->transfer.async.file->proactor,
          iree_hal_amd_gpu_file_window_failed, window, &window->failure);
    }
    iree_hal_amd_gpu_file_completion_notify(&window->failure, status, retired);
  }
}

static void iree_hal_amd_gpu_file_window_copy(
    iree_hal_amd_gpu_file_window_t* window) {
  const iree_hal_amd_gpu_file_pipeline_t* pipeline = window->pipeline;
  const iree_hal_amd_fabric_file_transfer_t* transfer =
      &pipeline->operation->transfer;
  const bool reading = transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ;
  const uint64_t address =
      pipeline->address + window->request.file_offset - transfer->file_offset;
  window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_COPY;
  window->failure_initialized = false;
  iree_status_t status = iree_hal_amd_gpu_file_transfer_copy_native(
      (void*)pipeline->operation->executor,
      reading ? window->staging.address : address,
      reading ? address : window->staging.address,
      iree_hal_file_request_span(&window->request).length,
      reading ? window->staging.effects.read : (iree_hal_memory_effects_t){0},
      reading ? (iree_hal_memory_effects_t){0} : window->staging.effects.write,
      (iree_hal_amd_gpu_retirement_callback_t){
          .fn = iree_hal_amd_gpu_file_window_copied,
          .user_data = window,
      });
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_window_copied(window, status, NULL, true);
  }
}

static void iree_hal_amd_gpu_file_window_io(
    iree_hal_amd_gpu_file_window_t* window) {
  window->phase = window->pipeline->operation->transfer.file_access ==
                          IREE_HAL_MEMORY_ACCESS_READ
                      ? IREE_HAL_AMD_GPU_FILE_WINDOW_READ
                      : IREE_HAL_AMD_GPU_FILE_WINDOW_WRITE;
  iree_status_t status = iree_hal_file_request_submit(
      &window->request, (iree_hal_file_transfer_callback_t){
                            iree_hal_amd_gpu_file_window_complete, window});
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_window_complete(window, status);
  }
}

static void iree_hal_amd_gpu_file_pipeline_finish(
    iree_hal_amd_gpu_file_pipeline_t* pipeline) {
  iree_hal_amd_gpu_file_transfer_t* operation = pipeline->operation;
  iree_status_t status = (iree_status_t)iree_atomic_load(
      &pipeline->failure, iree_memory_order_acquire);
  iree_hal_amd_gpu_capture_pool_release(
      operation->executor->captures,
      iree_make_byte_span(pipeline, pipeline->allocation_size));
  iree_hal_amd_gpu_file_transfer_complete(operation, status);
}

static void iree_hal_amd_gpu_file_pipeline_drain(
    iree_hal_amd_gpu_file_pipeline_t* pipeline) {
  const iree_hal_amd_fabric_file_transfer_t* transfer =
      &pipeline->operation->transfer;
  for (;;) {
    uint32_t completed = (uint32_t)iree_atomic_exchange(
        &pipeline->ready, IREE_HAL_AMD_GPU_FILE_DRAINING,
        iree_memory_order_acquire);
    pipeline->active &= ~completed;
    while (completed) {
      const uint32_t index = iree_math_count_trailing_zeros_u32(completed);
      const uint32_t bit = UINT32_C(1) << index;
      completed &= completed - 1;
      iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[index];
      if (!iree_status_is_ok(window->status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline, window->status);
        window->status = iree_ok_status();
      }
      const bool failed =
          iree_atomic_load(&pipeline->failure, iree_memory_order_acquire) != 0;
      if (!failed && !pipeline->releasing) {
        if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED &&
            pipeline->cursor < transfer->length) {
          iree_status_t status = iree_hal_amd_fabric_file_request_prepare(
              transfer, pipeline->cursor, window->staging.span,
              &window->request);
          if (iree_status_is_ok(status)) {
            pipeline->cursor +=
                iree_hal_file_request_span(&window->request).length;
            pipeline->active |= bit;
            if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
              iree_hal_amd_gpu_file_window_io(window);
            } else {
              iree_hal_amd_gpu_file_window_copy(window);
            }
            continue;
          }
          iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
        } else if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_READ) {
          pipeline->active |= bit;
          iree_hal_amd_gpu_file_window_copy(window);
          continue;
        } else if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_COPY &&
                   transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
          pipeline->active |= bit;
          iree_hal_amd_gpu_file_window_io(window);
          continue;
        }
      }
      if (window->staging.slot) {
        iree_hal_amd_gpu_file_staging_return(&window->staging);
      }
      window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_IDLE;
    }
    if (!pipeline->releasing &&
        !iree_atomic_load(&pipeline->failure, iree_memory_order_acquire)) {
      for (uint32_t i = 0; i < pipeline->count; ++i) {
        const uint32_t bit = UINT32_C(1) << i;
        if ((pipeline->active & bit) || pipeline->cursor == transfer->length) {
          continue;
        }
        iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[i];
        window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED;
        pipeline->active |= bit;
        iree_hal_amd_gpu_file_staging_reserve(
            &window->staging, transfer->length - pipeline->cursor,
            pipeline->alignment,
            (iree_hal_file_transfer_callback_t){
                iree_hal_amd_gpu_file_window_complete, window});
      }
    }
    if (!pipeline->active) {
      if (pipeline->releasing) {
        iree_hal_amd_gpu_file_pipeline_finish(pipeline);
        return;
      }
      pipeline->releasing = true;
      for (uint32_t i = 0; i < pipeline->count; ++i) {
        iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[i];
        window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_RELEASE;
        pipeline->active |= UINT32_C(1) << i;
        iree_hal_amd_gpu_file_staging_release(
            &window->staging, iree_ok_status(),
            (iree_hal_file_transfer_callback_t){
                iree_hal_amd_gpu_file_window_complete, window});
      }
      if (!pipeline->active) {
        iree_hal_amd_gpu_file_pipeline_finish(pipeline);
        return;
      }
    }
    int64_t expected = IREE_HAL_AMD_GPU_FILE_DRAINING;
    if (iree_atomic_compare_exchange_strong(&pipeline->ready, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return;
    }
  }
}

static void iree_hal_amd_gpu_file_pipeline_begin(
    iree_hal_amd_gpu_file_transfer_t* operation, uint64_t address) {
  const uint32_t count = iree_min(
      32, iree_hal_amd_gpu_file_staging_concurrency(
              operation->executor->staging, operation->transfer.length));
  iree_byte_span_t storage = iree_byte_span_empty();
  iree_status_t status = iree_hal_amd_gpu_capture_pool_acquire(
      operation->executor->captures,
      sizeof(iree_hal_amd_gpu_file_pipeline_t) +
          count * sizeof(iree_hal_amd_gpu_file_window_t),
      &storage);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  iree_hal_amd_gpu_file_pipeline_t* pipeline = (void*)storage.data;
  *pipeline = (iree_hal_amd_gpu_file_pipeline_t){
      .operation = operation,
      .allocation_size = storage.data_length,
      .address = address,
  };
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(operation->transfer.file);
  pipeline->alignment =
      operation->transfer.file_access == IREE_HAL_MEMORY_ACCESS_READ
          ? requirements->read.buffer_alignment
          : requirements->write.buffer_alignment;
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[i];
    *window = (iree_hal_amd_gpu_file_window_t){.pipeline = pipeline};
    status = iree_hal_amd_gpu_file_staging_acquire(
        operation->executor->staging, operation->transfer.async.file->proactor,
        &window->staging);
    if (iree_status_is_ok(status)) {
      ++pipeline->count;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
  }
  iree_hal_amd_gpu_file_pipeline_notify(pipeline, 0);
}

static void iree_hal_amd_gpu_file_transfer_advance(
    iree_hal_amd_gpu_file_transfer_t* operation);
static void iree_hal_amd_gpu_file_transfer_copy(
    iree_hal_amd_gpu_file_transfer_t* operation);

static void iree_hal_amd_gpu_file_transfer_io_complete(void* user_data,
                                                       iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
  } else {
    iree_hal_amd_gpu_file_transfer_advance(operation);
  }
}

static void iree_hal_amd_gpu_file_transfer_issue(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  operation->resume = iree_hal_amd_gpu_file_transfer_io_complete;
  iree_atomic_store(&operation->pending, 2, iree_memory_order_relaxed);
  iree_status_t status = iree_hal_file_request_submit(
      &operation->transfer.async.request,
      (iree_hal_file_transfer_callback_t){
          .fn = iree_hal_amd_gpu_file_transfer_step_complete,
          .user_data = operation,
      });
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_step_complete(operation, status);
  }
  iree_hal_amd_gpu_file_transfer_join(operation);
}

static void iree_hal_amd_gpu_file_transfer_copy(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  const iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  const iree_byte_span_t contents =
      iree_make_byte_span(transfer->file_contents.data + transfer->file_offset,
                          (iree_host_size_t)transfer->length);
  iree_hal_amd_gpu_buffer_reference_t reference;
  iree_status_t status = iree_hal_amd_gpu_buffer_reference_initialize(
      operation->executor->queue->queue_family,
      operation->executor->transfer_site, transfer->buffer,
      transfer->buffer_offset, &reference);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  iree_hal_amd_gpu_transfer_operation_t copy = {.length = contents.data_length};
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    copy.source.host = contents.data;
    copy.target.device = reference;
  } else {
    copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    copy.source.device = reference;
    copy.target.host = contents.data;
  }
  iree_hal_amd_gpu_file_transfer_submit_native(
      operation, &copy, operation->before, NULL,
      iree_hal_amd_gpu_file_transfer_complete);
}

static void iree_hal_amd_gpu_file_transfer_advance(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  if (iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer) ==
      transfer->length) {
    iree_status_t status = iree_ok_status();
    if (transfer->source.contents.data &&
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
      // Publish every host-written range once, after all I/O and native copies
      // have retired and before the final native consumer acquire.
      status = iree_hal_amd_fabric_file_transfer_flush(transfer);
    }
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  const iree_hal_amd_fabric_file_source_t* range = &transfer->source;
  if (!range->contents.data) {
    uint64_t address = 0;
    iree_status_t status = iree_hal_amd_gpu_buffer_reference_resolve(
        &operation->reference, &address);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_file_transfer_complete(operation, status);
      return;
    }
    iree_hal_amd_gpu_file_pipeline_begin(operation, address);
    return;
  }
  iree_status_t status =
      iree_hal_amd_fabric_file_transfer_prepare_range(transfer, range);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
  } else {
    iree_hal_amd_gpu_file_transfer_issue(operation);
  }
}

static void iree_hal_amd_gpu_file_transfer_host_start(void* user_data,
                                                      iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_advance(operation);
  } else {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
  }
}

static void iree_hal_amd_gpu_file_transfer_start(void* user_data,
                                                 iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  if (operation->executor->profile) {
    iree_hal_amd_gpu_profile_operation_record(operation->executor->profile,
                                              (void*)(operation + 1), true, 0);
  }
  iree_hal_amd_fabric_file_transfer_t* transfer = &operation->transfer;
  iree_hal_memory_effects_t before = {0}, after = {0};
  status = iree_hal_amd_fabric_file_transfer_prepare_source(
      transfer, operation->executor->context->api,
      operation->executor->queue->queue_family,
      operation->executor->native_family_ordinal, &before, &after);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  if (!transfer->async.file) {
    iree_hal_amd_gpu_file_transfer_copy(operation);
    return;
  }
  if (!transfer->source.contents.data) {
    const iree_async_file_requirements_t* requirements =
        iree_hal_file_requirements(transfer->file);
    status = iree_hal_amd_gpu_file_staging_prepare(
        operation->executor->staging,
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? requirements->read.buffer_alignment
            : requirements->write.buffer_alignment);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_buffer_reference_initialize(
          operation->executor->queue->queue_family,
          operation->executor->transfer_site, transfer->buffer,
          transfer->buffer_offset, &operation->reference);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  operation->executor->add_effects(operation->executor->queue, before,
                                   operation->before);
  operation->executor->add_effects(operation->executor->queue, after,
                                   operation->after);
  iree_hal_amd_gpu_file_transfer_submit_native(
      operation, NULL, operation->before, NULL,
      iree_hal_amd_gpu_file_transfer_host_start);
}

static void iree_hal_amd_gpu_file_transfer_begin(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  if (!operation->waits.count) {
    iree_hal_amd_gpu_file_transfer_start(operation, iree_ok_status());
    return;
  }
  iree_status_t status = iree_hal_amd_gpu_host_wait_begin(
      operation->executor->waits, operation->waits,
      iree_hal_amd_gpu_file_transfer_start, operation,
      iree_allocator_inline_arena(&operation->dependency.storage),
      &operation->dependency.wait);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_start(operation, status);
  }
}

iree_status_t iree_hal_amd_gpu_file_transfer_submit(
    const iree_hal_amd_gpu_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    const void* before, const void* after) {
  iree_host_size_t tracker_size = 0;
  if (waits.count) {
    IREE_RETURN_IF_ERROR(iree_async_semaphore_wait_tracker_storage_size(
        waits.count, &tracker_size));
  }
  iree_host_size_t total_size = 0;
  iree_host_size_t wait_values_offset = 0, wait_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0, signal_semaphores_offset = 0;
  iree_host_size_t tracker_offset = 0;
  iree_host_size_t before_offset = 0, after_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_file_transfer_t) +
          (executor->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t)
                             : 0),
      &total_size,
      IREE_STRUCT_FIELD_ALIGNED(executor->barrier_size, uint8_t,
                                iree_max_align_t, &before_offset),
      IREE_STRUCT_FIELD_ALIGNED(executor->barrier_size, uint8_t,
                                iree_max_align_t, &after_offset),
      IREE_STRUCT_FIELD_ALIGNED(waits.count, uint64_t, iree_alignof(uint64_t),
                                &wait_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(waits.count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                &wait_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, uint64_t, iree_alignof(uint64_t),
                                &signal_values_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, iree_hal_semaphore_t*,
                                iree_alignof(iree_hal_semaphore_t*),
                                &signal_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(tracker_size, uint8_t, iree_max_align_t,
                                &tracker_offset)));
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      executor->captures, total_size, &storage));
  iree_hal_amd_gpu_file_transfer_t* operation = (void*)storage.data;
  *operation = (iree_hal_amd_gpu_file_transfer_t){
      .executor = executor,
      .allocation_size = storage.data_length,
      .waits = {.count = waits.count,
                .semaphores = (void*)(storage.data + wait_semaphores_offset),
                .payload_values = (void*)(storage.data + wait_values_offset)},
      .signals =
          {
              .count = signals.count,
              .semaphores = (void*)(storage.data + signal_semaphores_offset),
              .payload_values = (void*)(storage.data + signal_values_offset),
          },
      .before = storage.data + before_offset,
      .after = storage.data + after_offset,
      .dependency = {.storage = {.capacity = tracker_size,
                                 .buffer = storage.data + tracker_offset}},
  };
  executor->begin(executor->queue);
  memcpy(storage.data + before_offset, before, executor->barrier_size);
  memcpy(storage.data + after_offset, after, executor->barrier_size);
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    operation->waits.semaphores[i] = waits.semaphores[i];
    operation->waits.payload_values[i] = waits.payload_values[i];
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    operation->signals.semaphores[i] = signals.semaphores[i];
    operation->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  iree_status_t status = iree_hal_amd_fabric_file_transfer_initialize(
      file, file_access, file_offset, buffer, buffer_offset, length,
      executor->captures->host_allocator, &operation->transfer);
  if (iree_status_is_ok(status)) {
    if (executor->profile) {
      iree_hal_amd_gpu_profile_operation_t* profile_operation =
          (void*)(operation + 1);
      iree_hal_amd_gpu_profile_operation_initialize(
          executor->profile, executor->queue_ordinal,
          file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
              : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE,
          waits.count, signals.count, 1, profile_operation);
      profile_operation->event.payload_length = length;
      iree_hal_amd_gpu_profile_operation_defer(profile_operation);
    }
    iree_hal_amd_gpu_file_transfer_begin(operation);
  } else {
    iree_hal_amd_fabric_file_transfer_deinitialize(&operation->transfer);
    iree_hal_amd_gpu_file_transfer_free(operation);
  }
  return status;
}
