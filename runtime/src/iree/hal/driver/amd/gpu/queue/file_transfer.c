// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"

#include "iree/async/proactor.h"
#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/fabric/buffer.h"
#include "iree/hal/driver/amd/fabric/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/file_completion.h"
#include "iree/hal/driver/amd/status.h"
#include "iree/hal/utils/memory_file.h"

#define IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY 32
#define IREE_HAL_AMD_GPU_FILE_DRAINING (INT64_C(1) << 32)
#define IREE_HAL_AMD_GPU_FILE_METADATA_READY (INT64_C(1) << 33)

typedef enum iree_hal_amd_gpu_file_binding_kind_e {
  IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC,
  IREE_HAL_AMD_GPU_FILE_BINDING_HOST,
  IREE_HAL_AMD_GPU_FILE_BINDING_DEVICE,
} iree_hal_amd_gpu_file_binding_kind_t;

// File handles and registrations are independent of row traversal. A file's
// bucket use spans all of its loans, including gaps between successive rows.
// Imported execution storage uses its native address without a host copy.
typedef struct iree_hal_amd_gpu_file_binding_t {
  // Qualified representation retained by the enclosing file capture.
  iree_hal_amd_gpu_file_binding_kind_t kind;
  union {
    // Native I/O ownership; never used for either memory representation.
    struct {
      // Borrowed native handle and its owning proactor.
      iree_async_file_t* handle;
      // Acquired for staged I/O and released after every window retires.
      iree_hal_amd_gpu_file_staging_bucket_t* bucket;
    } async;
    // Borrowed host-only memory contents, starting at the file binding origin.
    iree_byte_span_t host;
    // Qualified execution storage at the file binding origin, owned by file.
    iree_hal_amd_gpu_buffer_reference_t device;
  };
} iree_hal_amd_gpu_file_binding_t;

typedef struct iree_hal_amd_gpu_file_buffer_t {
  // Native address slot, qualified once and resolved only after waits.
  iree_hal_amd_gpu_buffer_reference_t reference;
  // Direct file view and visibility facts, independent of the selected file.
  iree_hal_amd_fabric_file_source_t source;
  // Queue effects required before the first native file access to this binding.
  iree_hal_memory_effects_t before;
  // The native address interface is qualified when this binding needs copies.
  bool reference_prepared;
  // File visibility has been qualified; memory-file copies need no such query.
  bool source_prepared;
} iree_hal_amd_gpu_file_buffer_t;

typedef struct iree_hal_amd_gpu_file_transfer_t {
  // Queue-owned execution and storage services, borrowed through retirement.
  const iree_hal_amd_gpu_file_executor_t* executor;
  // Complete capture extent returned to the reusable metadata pool.
  iree_host_size_t allocation_size;
  // Immutable retained file/buffer tables and host or indirect descriptions.
  iree_hal_file_transfer_batch_t batch;
  // Placed per-file native facts and registration memberships.
  iree_hal_amd_gpu_file_binding_t* files;
  // Placed per-buffer facts, prepared only for bindings actually accessed.
  iree_hal_amd_gpu_file_buffer_t* buffers;
  // Exactly READ for file-to-buffer, or WRITE for buffer-to-file.
  iree_hal_memory_access_t access;
  // Copied and retained public dependencies.
  iree_hal_semaphore_list_t waits;
  // Copied and retained terminal completion points.
  iree_hal_semaphore_list_t signals;
  // Captured producer boundary, consumed once by the first native submission.
  void* before;
  // Captured consumer boundary, augmented by qualified file visibility.
  void* after;
  // Joins serial initialization/finalization submissions with their callbacks.
  iree_atomic_int32_t pending;
  // Next serial step, invoked after submit and completion relinquish use.
  iree_hal_amd_gpu_host_wait_fn_t resume;
  // Result transferred by the serial completion borrow.
  iree_status_t status;
  // Last retired native epoch, attached to public completion.
  iree_async_single_frontier_t frontier;
  // Live count, downloaded or copied exactly once after producer waits.
  uint64_t count;
  // Native files require the producer boundary before host payload accesses.
  bool has_native_files;
  // Initial public dependency join with capture-local tracker storage.
  struct {
    // Initial public wait join, finalized before file/native work starts.
    iree_hal_amd_gpu_host_wait_t wait;
    // Inline tracker allocation sized to the captured wait list.
    iree_allocator_inline_storage_t storage;
  } dependency;
} iree_hal_amd_gpu_file_transfer_t;

static void iree_hal_amd_gpu_file_transfer_join(
    iree_hal_amd_gpu_file_transfer_t* operation) {
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_status_t status = operation->status;
    operation->status = iree_ok_status();
    operation->resume(operation, status);
  }
}

static void iree_hal_amd_gpu_file_transfer_native_complete(
    void* user_data, iree_status_t status,
    const iree_async_single_frontier_t* frontier, bool retired) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  if (!retired) {
    // The native owner retains the first failure through its final callback.
    iree_status_free(status);
    return;
  }
  if (frontier) {
    operation->frontier = *frontier;
  }
  operation->status = status;
  iree_hal_amd_gpu_file_transfer_join(operation);
}

static void iree_hal_amd_gpu_file_transfer_submit_native(
    iree_hal_amd_gpu_file_transfer_t* operation, iree_host_size_t count,
    const iree_hal_amd_gpu_transfer_operation_t* transfers, const void* after,
    iree_hal_amd_gpu_host_wait_fn_t resume) {
  const void* before = operation->before;
  operation->before = NULL;
  operation->resume = resume;
  iree_atomic_store(&operation->pending, 2, iree_memory_order_relaxed);
  iree_status_t status = operation->executor->submit(
      operation->executor->queue, count, transfers, before, after,
      (iree_hal_amd_gpu_retirement_callback_t){
          iree_hal_amd_gpu_file_transfer_native_complete, operation});
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_native_complete(operation, status, NULL,
                                                   true);
  }
  iree_hal_amd_gpu_file_transfer_join(operation);
}

static void iree_hal_amd_gpu_file_transfer_retire(void* user_data,
                                                  iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  const iree_hal_amd_gpu_file_executor_t* executor = operation->executor;
  if (executor->profile) {
    iree_hal_amd_gpu_profile_operation_record(executor->profile,
                                              (void*)(operation + 1), false, 0);
  }
  for (iree_host_size_t i = 0; i < operation->batch.files.count; ++i) {
    iree_hal_file_release(operation->batch.files.bindings[i].file);
  }
  for (iree_host_size_t i = 0; i < operation->batch.buffers.count; ++i) {
    iree_hal_buffer_release(operation->batch.buffers.bindings[i].buffer);
  }
  if (operation->batch.transfers.type ==
      IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_buffer_release(operation->batch.transfers.indirect.records.buffer);
    iree_hal_buffer_release(operation->batch.transfers.indirect.count_buffer);
  }
  // Captured backing uses end before public signaling and pool destruction.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(
        operation->signals,
        operation->frontier.entry_count
            ? iree_async_single_frontier_as_const_frontier(&operation->frontier)
            : NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
  iree_hal_semaphore_list_release(operation->waits);
  iree_hal_semaphore_list_release(operation->signals);
  iree_hal_amd_gpu_capture_pool_release(
      executor->captures,
      iree_make_byte_span(operation, operation->allocation_size));
  IREE_TRACE_ZONE_END(z0);
  executor->end(executor->queue);
}

static void iree_hal_amd_gpu_file_transfer_complete(
    iree_hal_amd_gpu_file_transfer_t* operation, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_submit_native(
        operation, 0, NULL, operation->after,
        iree_hal_amd_gpu_file_transfer_retire);
  } else {
    iree_hal_amd_gpu_file_transfer_retire(operation, status);
  }
}

typedef struct iree_hal_amd_gpu_file_pipeline_t
    iree_hal_amd_gpu_file_pipeline_t;

typedef enum iree_hal_amd_gpu_file_window_phase_e {
  IREE_HAL_AMD_GPU_FILE_WINDOW_IDLE = 0,
  IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED,
  IREE_HAL_AMD_GPU_FILE_WINDOW_BEFORE,
  IREE_HAL_AMD_GPU_FILE_WINDOW_IO,
  IREE_HAL_AMD_GPU_FILE_WINDOW_COPY,
  IREE_HAL_AMD_GPU_FILE_WINDOW_RELEASE,
} iree_hal_amd_gpu_file_window_phase_t;

typedef enum iree_hal_amd_gpu_file_window_kind_e {
  IREE_HAL_AMD_GPU_FILE_WINDOW_MEMORY = 0,
  IREE_HAL_AMD_GPU_FILE_WINDOW_DIRECT,
  IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED,
} iree_hal_amd_gpu_file_window_kind_t;

typedef struct iree_hal_amd_gpu_file_piece_t {
  union {
    // Native request with its own partial-I/O continuation.
    iree_hal_file_request_t request;
    // Memory files use the native upload/download path directly.
    iree_hal_amd_gpu_transfer_operation_t copy;
    // Error handoff can reuse the first request only outside native file I/O.
    iree_hal_amd_gpu_file_completion_t failure;
  };
  // Resolved payload address for a staged copy; never read back from the GPU.
  uint64_t address;
  // Buffer facts used by uncommon direct-host cache actions.
  uint32_t buffer_binding;
} iree_hal_amd_gpu_file_piece_t;

typedef struct iree_hal_amd_gpu_file_window_t {
  // Enclosing owner, live through every accepted callback and submit call.
  iree_hal_amd_gpu_file_pipeline_t* pipeline;
  // Physical loan; registration membership belongs to the file table.
  iree_hal_amd_gpu_file_staging_request_t staging;
  // One exact native I/O executor for all requests in this window.
  iree_async_proactor_t* proactor;
  // Placed request/copy storage; capacity is fixed for this finite operation.
  iree_hal_amd_gpu_file_piece_t* pieces;
  // Number of prepared requests/copies in this window.
  uint32_t count;
  // Next phase, modified only by the pipeline drainer.
  iree_hal_amd_gpu_file_window_phase_t phase;
  // Direct I/O, staged I/O, or native memory-file copies.
  iree_hal_amd_gpu_file_window_kind_t kind;
  // Qualified visibility accumulated over the window's newly used bindings.
  iree_hal_memory_effects_t before;
  // Joins the native I/O list's callbacks; the drainer joins submit itself.
  iree_atomic_int32_t pending;
  // Completion result published by this window's ready bit.
  iree_status_t status;
  // Error-only native handoff is initialized on the first failure notification.
  bool failure_initialized;
} iree_hal_amd_gpu_file_window_t;

struct iree_hal_amd_gpu_file_pipeline_t {
  // Retained immutable capture and native services.
  iree_hal_amd_gpu_file_transfer_t* operation;
  // Complete pooled allocation, including windows, pieces, and metadata page.
  iree_host_size_t allocation_size;
  // Constant-space traversal of captured or acquired records.
  iree_hal_file_transfer_iterator_t iterator;
  // Completed window bits, metadata bit, and single inline drainer ownership.
  iree_atomic_int64_t ready;
  // First error stops admission; every accepted access still drains.
  iree_atomic_intptr_t failure;
  // Native/storage callbacks still borrowing their windows.
  uint32_t active;
  // Maximum independently progressing windows, bounded by staging policy.
  uint32_t window_count;
  // Slots per window: one for linear bulk, bounded groups for row lists.
  uint32_t piece_capacity;
  // A pending physical loan owns the iterator head. A first-use host visibility
  // boundary also gates admission until later windows can safely reuse it.
  bool admission_pending;
  // True only after the last live description has been consumed.
  bool exhausted;
  // Final membership release cursor; no windows remain in payload use.
  iree_host_size_t release_index;
  // Final phase reuses window zero for asynchronous registration detach.
  bool releasing;
  // One bounded indirect-description download, independent of payload windows.
  struct {
    // Private download destination, absent for CPU-readable records.
    iree_hal_file_transfer_record_t* records;
    // Qualified source slot, resolved after producer dependencies.
    iree_hal_amd_gpu_buffer_reference_t reference;
    // Number of records in the outstanding bounded download.
    uint32_t count;
    // Native metadata use has not yet retired.
    bool pending;
    // Native callback result, published by the metadata-ready bit.
    iree_status_t status;
  } metadata;
  // Independent placed windows, followed by their variable-capacity pieces.
  iree_hal_amd_gpu_file_window_t windows[];
};

static void iree_hal_amd_gpu_file_pipeline_drain(
    iree_hal_amd_gpu_file_pipeline_t* pipeline);

static void iree_hal_amd_gpu_file_pipeline_notify(
    iree_hal_amd_gpu_file_pipeline_t* pipeline, int64_t completed) {
  const int64_t previous = iree_atomic_fetch_or(
      &pipeline->ready, IREE_HAL_AMD_GPU_FILE_DRAINING | completed,
      iree_memory_order_acq_rel);
  if (!(previous & IREE_HAL_AMD_GPU_FILE_DRAINING)) {
    iree_hal_amd_gpu_file_pipeline_drain(pipeline);
  }
  // Publication ends this callback's borrow; another drainer may reclaim it.
}

static bool iree_hal_amd_gpu_file_pipeline_fail(
    iree_hal_amd_gpu_file_pipeline_t* pipeline, iree_status_t status) {
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &pipeline->failure, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_relaxed)) {
    return true;
  }
  // The first terminal diagnostic remains authoritative for this operation.
  iree_status_free(status);
  return false;
}

static void iree_hal_amd_gpu_file_window_complete(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_file_window_t* window = user_data;
  window->status = status;
  iree_hal_amd_gpu_file_pipeline_t* pipeline = window->pipeline;
  iree_hal_amd_gpu_file_pipeline_notify(
      pipeline, INT64_C(1) << (window - pipeline->windows));
}

static void iree_hal_amd_gpu_file_window_io_complete(void* user_data,
                                                     iree_status_t status) {
  iree_hal_amd_gpu_file_window_t* window = user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_pipeline_fail(window->pipeline, status);
  }
  if (iree_atomic_fetch_sub(&window->pending, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_hal_amd_gpu_file_window_complete(window, iree_ok_status());
  }
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
  } else if (window->proactor) {
    if (!window->failure_initialized) {
      window->failure_initialized = true;
      iree_hal_amd_gpu_file_completion_initialize(
          window->proactor, iree_hal_amd_gpu_file_window_failed, window,
          &window->pieces[0].failure);
    }
    iree_hal_amd_gpu_file_completion_notify(&window->pieces[0].failure, status,
                                            retired);
  } else if (retired) {
    iree_hal_amd_gpu_file_window_complete(window, status);
  } else {
    // Memory-file ownership remains native until its terminal notification.
    iree_status_free(status);
  }
}

static iree_status_t iree_hal_amd_gpu_file_buffer_prepare(
    iree_hal_amd_gpu_file_transfer_t* operation, uint32_t index,
    bool native_file) {
  iree_hal_amd_gpu_file_buffer_t* state = &operation->buffers[index];
  const iree_hal_buffer_binding_t binding =
      operation->batch.buffers.bindings[index];
  const iree_hal_amd_gpu_file_executor_t* executor = operation->executor;
  if (native_file && !state->source_prepared) {
    iree_hal_memory_effects_t after = {0};
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_source_initialize(
        executor->context->api, executor->queue->queue_family,
        executor->native_family_ordinal, operation->access, binding,
        &state->source, &state->before, &after));
    if (after.bits) {
      executor->add_effects(executor->queue, after, operation->after);
    }
    state->source_prepared = true;
  }
  if ((!native_file || !state->source.contents.data) &&
      !state->reference_prepared) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_buffer_reference_initialize(
        executor->queue->queue_family, executor->transfer_site, binding.buffer,
        binding.offset, &state->reference));
    state->reference_prepared = true;
  }
  return iree_ok_status();
}

static iree_hal_amd_gpu_file_window_kind_t iree_hal_amd_gpu_file_window_kind(
    const iree_hal_amd_gpu_file_transfer_t* operation,
    iree_hal_file_transfer_range_t range) {
  return operation->files[range.file_binding].kind !=
                 IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC
             ? IREE_HAL_AMD_GPU_FILE_WINDOW_MEMORY
         : operation->buffers[range.buffer_binding].source.contents.data
             ? IREE_HAL_AMD_GPU_FILE_WINDOW_DIRECT
             : IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED;
}

// Host cache actions are needed only for qualified noncoherent direct views.
// The prepared span keeps the exact range; padding is never a payload access.
static iree_status_t iree_hal_amd_gpu_file_window_host_cache(
    iree_hal_amd_gpu_file_window_t* window,
    amdf_host_cache_operation_t action) {
  iree_hal_amd_gpu_file_transfer_t* operation = window->pipeline->operation;
  const uint32_t effect = action == AMDF_HOST_CACHE_OPERATION_FLUSH
                              ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH
                              : IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < window->count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_gpu_file_piece_t* piece = &window->pieces[i];
    const iree_hal_amd_fabric_file_source_t* source =
        &operation->buffers[piece->buffer_binding].source;
    if (!(source->effects.bits & effect)) {
      continue;
    }
    const iree_async_span_t span = iree_hal_file_request_span(&piece->request);
    const uint64_t offset = source->region
                                ? span.offset
                                : source->source_offset + span.offset -
                                      (uintptr_t)source->contents.data;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        source->api->host_mapping_cache_control(source->storage->mapping,
                                                action, offset, span.length),
        "host_mapping_cache_control(file window)");
  }
  return status;
}

static void iree_hal_amd_gpu_file_window_io(
    iree_hal_amd_gpu_file_window_t* window) {
  iree_async_operation_t* requests[IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY];
  iree_status_t status = iree_ok_status();
  if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_DIRECT) {
    status = iree_hal_amd_gpu_file_window_host_cache(
        window, AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_window_complete(window, status);
    return;
  }
  window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_IO;
  for (uint32_t i = 0; i < window->count; ++i) {
    requests[i] = iree_hal_file_request_bind(
        &window->pieces[i].request,
        (iree_hal_file_transfer_callback_t){
            iree_hal_amd_gpu_file_window_io_complete, window});
  }
  iree_atomic_store(&window->pending, window->count, iree_memory_order_relaxed);
  status = iree_async_proactor_submit(
      window->proactor, (iree_async_operation_list_t){requests, window->count});
  if (!iree_status_is_ok(status)) {
    // A rejected list owns no callbacks. Accepted earlier windows still drain.
    iree_hal_amd_gpu_file_window_complete(window, status);
  }
}

static void iree_hal_amd_gpu_file_window_copy(
    iree_hal_amd_gpu_file_window_t* window, bool boundary_only) {
  iree_hal_amd_gpu_file_transfer_t* operation = window->pipeline->operation;
  const iree_hal_amd_gpu_file_executor_t* executor = operation->executor;
  iree_hal_amd_gpu_transfer_operation_t
      copies[IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY];
  const bool reading = operation->access == IREE_HAL_MEMORY_ACCESS_READ;
  const uint32_t count = boundary_only ? 0 : window->count;
  for (uint32_t i = 0; i < count; ++i) {
    const iree_hal_amd_gpu_file_piece_t* piece = &window->pieces[i];
    if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_MEMORY) {
      copies[i] = piece->copy;
    } else {
      const iree_async_span_t span =
          iree_hal_file_request_span(&piece->request);
      const uint64_t staging =
          window->staging.address + span.offset - window->staging.span.offset;
      copies[i] = (iree_hal_amd_gpu_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .source.device = iree_hal_amd_gpu_buffer_reference_borrow(
              reading ? staging : piece->address),
          .target.device = iree_hal_amd_gpu_buffer_reference_borrow(
              reading ? piece->address : staging),
          .length = span.length};
    }
  }
  void* before = iree_alloca(executor->barrier_size);
  void* after = iree_alloca(executor->barrier_size);
  const bool has_before = operation->before != NULL;
  if (operation->before) {
    memcpy(before, operation->before, executor->barrier_size);
    operation->before = NULL;
  } else {
    memset(before, 0, executor->barrier_size);
  }
  memset(after, 0, executor->barrier_size);
  iree_hal_memory_effects_t before_effects = window->before,
                            after_effects = {0};
  if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED) {
    if (reading) {
      before_effects.bits |= window->staging.effects.read.bits;
    } else {
      after_effects = window->staging.effects.write;
    }
  }
  if (before_effects.bits) {
    executor->add_effects(executor->queue, before_effects, before);
  }
  if (after_effects.bits) {
    executor->add_effects(executor->queue, after_effects, after);
  }
  window->phase = boundary_only ? IREE_HAL_AMD_GPU_FILE_WINDOW_BEFORE
                                : IREE_HAL_AMD_GPU_FILE_WINDOW_COPY;
  if (boundary_only) {
    window->pipeline->admission_pending = true;
  }
  window->failure_initialized = false;
  iree_status_t status =
      executor->submit(executor->queue, count, copies,
                       has_before || before_effects.bits ? before : NULL,
                       after_effects.bits ? after : NULL,
                       (iree_hal_amd_gpu_retirement_callback_t){
                           iree_hal_amd_gpu_file_window_copied, window});
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_window_copied(window, status, NULL, true);
  }
}

static iree_status_t iree_hal_amd_gpu_file_window_fill(
    iree_hal_amd_gpu_file_window_t* window) {
  iree_hal_amd_gpu_file_pipeline_t* pipeline = window->pipeline;
  iree_hal_amd_gpu_file_transfer_t* operation = pipeline->operation;
  uint64_t position = 0;
  iree_status_t status = iree_ok_status();
  while (window->count < pipeline->piece_capacity &&
         iree_status_is_ok(status)) {
    iree_hal_file_transfer_range_t range;
    status = iree_hal_file_transfer_iterator_peek(&pipeline->iterator, &range);
    if (!iree_status_is_ok(status) || !range.length) {
      break;
    }
    iree_hal_amd_gpu_file_binding_t* file =
        &operation->files[range.file_binding];
    const bool native_file = file->kind == IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC;
    status = iree_hal_amd_gpu_file_buffer_prepare(
        operation, range.buffer_binding, native_file);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (iree_hal_amd_gpu_file_window_kind(operation, range) != window->kind ||
        (native_file && file->async.handle->proactor != window->proactor)) {
      break;
    }
    iree_hal_amd_gpu_file_buffer_t* buffer =
        &operation->buffers[range.buffer_binding];
    iree_hal_amd_gpu_file_piece_t* piece = &window->pieces[window->count];
    piece->buffer_binding = range.buffer_binding;
    const iree_hal_file_binding_t binding =
        operation->batch.files.bindings[range.file_binding];
    uint64_t length = range.length;
    if (!native_file) {
      iree_hal_amd_gpu_buffer_reference_t reference = buffer->reference;
      reference.offset += range.buffer_offset;
      const bool reading = operation->access == IREE_HAL_MEMORY_ACCESS_READ;
      piece->copy = (iree_hal_amd_gpu_transfer_operation_t){.length = length};
      if (file->kind == IREE_HAL_AMD_GPU_FILE_BINDING_DEVICE) {
        iree_hal_amd_gpu_buffer_reference_t storage = file->device;
        storage.offset += range.file_offset;
        piece->copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
        piece->copy.source.device = reading ? storage : reference;
        piece->copy.target.device = reading ? reference : storage;
      } else if (reading) {
        piece->copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
        piece->copy.source.host = file->host.data + range.file_offset;
        piece->copy.target.device = reference;
      } else {
        piece->copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
        piece->copy.source.device = reference;
        piece->copy.target.host = file->host.data + range.file_offset;
      }
    } else {
      iree_async_span_t span;
      if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED) {
        const iree_async_file_requirements_t* requirements =
            iree_hal_file_requirements(binding.file);
        const uint64_t alignment =
            operation->access == IREE_HAL_MEMORY_ACCESS_READ
                ? requirements->read.buffer_alignment
                : requirements->write.buffer_alignment;
        position = iree_host_align(position, alignment);
        if (position >= window->staging.span.length) {
          break;
        }
        span = window->staging.span;
        span.offset += position;
        span.length -= position;
        status = iree_hal_amd_gpu_buffer_reference_resolve(&buffer->reference,
                                                           &piece->address);
        if (iree_status_is_ok(status)) {
          piece->address += range.buffer_offset;
        }
      } else {
        const iree_hal_amd_fabric_file_source_t* source = &buffer->source;
        if (source->region &&
            source->region->proactor != file->async.handle->proactor) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "registered buffer and file use different I/O proactors");
          break;
        }
        span = source->region ? iree_async_span_make(
                                    source->region,
                                    source->source_offset + range.buffer_offset,
                                    range.length)
                              : iree_async_span_from_ptr(
                                    source->contents.data + range.buffer_offset,
                                    (iree_host_size_t)iree_min(
                                        range.length, UINT32_C(1) << 30));
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_file_request_prepare(
            binding.file, operation->access, binding.offset + range.file_offset,
            range.length, span, &piece->request);
      }
      if (iree_status_is_ok(status)) {
        length = iree_hal_file_request_span(&piece->request).length;
        position += length;
      }
    }
    if (iree_status_is_ok(status)) {
      window->before.bits |= buffer->before.bits;
      buffer->before.bits = 0;
      ++window->count;
      iree_hal_file_transfer_iterator_advance(&pipeline->iterator, length);
    }
  }
  return status;
}

static void iree_hal_amd_gpu_file_window_start(
    iree_hal_amd_gpu_file_window_t* window) {
  const bool reading =
      window->pipeline->operation->access == IREE_HAL_MEMORY_ACCESS_READ;
  if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_MEMORY ||
      (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED && !reading)) {
    iree_hal_amd_gpu_file_window_copy(window, false);
  } else if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_DIRECT &&
             window->before.bits) {
    iree_hal_amd_gpu_file_window_copy(window, true);
  } else {
    iree_hal_amd_gpu_file_window_io(window);
  }
}

static void iree_hal_amd_gpu_file_metadata_complete(
    void* user_data, iree_status_t status,
    const iree_async_single_frontier_t* frontier, bool retired) {
  iree_hal_amd_gpu_file_pipeline_t* pipeline = user_data;
  if (!retired) {
    iree_status_free(status);
    return;
  }
  pipeline->metadata.status = status;
  iree_hal_amd_gpu_file_pipeline_notify(pipeline,
                                        IREE_HAL_AMD_GPU_FILE_METADATA_READY);
}

static void iree_hal_amd_gpu_file_metadata_acquire(
    iree_hal_amd_gpu_file_pipeline_t* pipeline) {
  const iree_hal_amd_gpu_file_executor_t* executor =
      pipeline->operation->executor;
  pipeline->metadata.count =
      (uint32_t)iree_min(pipeline->iterator.count - pipeline->iterator.index,
                         IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY);
  pipeline->metadata.pending = true;
  iree_hal_amd_gpu_buffer_reference_t reference = pipeline->metadata.reference;
  reference.offset +=
      pipeline->iterator.index * sizeof(iree_hal_file_transfer_record_t);
  const iree_hal_amd_gpu_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .source.device = reference,
      .target.host = pipeline->metadata.records,
      .length =
          pipeline->metadata.count * sizeof(iree_hal_file_transfer_record_t)};
  iree_status_t status =
      executor->submit(executor->queue, 1, &copy, NULL, NULL,
                       (iree_hal_amd_gpu_retirement_callback_t){
                           iree_hal_amd_gpu_file_metadata_complete, pipeline});
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_metadata_complete(pipeline, status, NULL, true);
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
  iree_hal_amd_gpu_file_transfer_t* operation = pipeline->operation;
  const bool reading = operation->access == IREE_HAL_MEMORY_ACCESS_READ;
  for (;;) {
    const int64_t ready =
        iree_atomic_exchange(&pipeline->ready, IREE_HAL_AMD_GPU_FILE_DRAINING,
                             iree_memory_order_acquire);
    if (ready & IREE_HAL_AMD_GPU_FILE_METADATA_READY) {
      pipeline->metadata.pending = false;
      if (!iree_status_is_ok(pipeline->metadata.status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline,
                                            pipeline->metadata.status);
        pipeline->metadata.status = iree_ok_status();
      } else {
        iree_hal_file_transfer_iterator_acquire(&pipeline->iterator,
                                                pipeline->metadata.records,
                                                pipeline->metadata.count);
      }
    }
    uint32_t completed = (uint32_t)ready;
    pipeline->active &= ~completed;
    while (completed) {
      const uint32_t index = iree_math_count_trailing_zeros_u32(completed);
      const uint32_t bit = UINT32_C(1) << index;
      completed &= completed - 1;
      iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[index];
      if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED ||
          window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_BEFORE) {
        pipeline->admission_pending = false;
      }
      if (!iree_status_is_ok(window->status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline, window->status);
        window->status = iree_ok_status();
      }
      if (!iree_atomic_load(&pipeline->failure, iree_memory_order_acquire) &&
          !pipeline->releasing) {
        if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED) {
          iree_status_t status = iree_hal_amd_gpu_file_window_fill(window);
          if (!iree_status_is_ok(status)) {
            iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
          } else {
            pipeline->active |= bit;
            iree_hal_amd_gpu_file_window_start(window);
            continue;
          }
        } else if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_BEFORE ||
                   (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_COPY &&
                    window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED &&
                    !reading)) {
          pipeline->active |= bit;
          iree_hal_amd_gpu_file_window_io(window);
          continue;
        } else if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_IO &&
                   window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED &&
                   reading) {
          pipeline->active |= bit;
          iree_hal_amd_gpu_file_window_copy(window, false);
          continue;
        } else if (window->phase == IREE_HAL_AMD_GPU_FILE_WINDOW_IO &&
                   window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_DIRECT &&
                   reading) {
          iree_status_t status = iree_hal_amd_gpu_file_window_host_cache(
              window, AMDF_HOST_CACHE_OPERATION_FLUSH);
          if (!iree_status_is_ok(status)) {
            iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
          }
        }
      }
      if (window->staging.slot) {
        iree_hal_amd_gpu_file_staging_return(&window->staging);
      }
      window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_IDLE;
    }

    uint32_t available = pipeline->window_count == 32
                             ? UINT32_MAX
                             : (UINT32_C(1) << pipeline->window_count) - 1;
    available &= ~pipeline->active;
    while (available && !pipeline->admission_pending &&
           !pipeline->metadata.pending && !pipeline->releasing &&
           !pipeline->exhausted &&
           !iree_atomic_load(&pipeline->failure, iree_memory_order_acquire)) {
      iree_hal_file_transfer_range_t range;
      iree_status_t status =
          iree_hal_file_transfer_iterator_peek(&pipeline->iterator, &range);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
        break;
      }
      if (!range.length) {
        if (pipeline->iterator.index < pipeline->iterator.count) {
          iree_hal_amd_gpu_file_metadata_acquire(pipeline);
        } else {
          pipeline->exhausted = true;
        }
        break;
      }
      iree_hal_amd_gpu_file_binding_t* file =
          &operation->files[range.file_binding];
      const bool native_file =
          file->kind == IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC;
      status = iree_hal_amd_gpu_file_buffer_prepare(
          operation, range.buffer_binding, native_file);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
        break;
      }
      const uint32_t index = iree_math_count_trailing_zeros_u32(available);
      const uint32_t bit = UINT32_C(1) << index;
      available &= available - 1;
      iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[index];
      window->count = 0;
      window->before.bits = 0;
      window->proactor = native_file ? file->async.handle->proactor : NULL;
      window->kind = iree_hal_amd_gpu_file_window_kind(operation, range);
      if (window->kind == IREE_HAL_AMD_GPU_FILE_WINDOW_STAGED) {
        const iree_async_file_requirements_t* requirements =
            iree_hal_file_requirements(
                operation->batch.files.bindings[range.file_binding].file);
        const uint64_t alignment = reading
                                       ? requirements->read.buffer_alignment
                                       : requirements->write.buffer_alignment;
        status = iree_hal_amd_gpu_file_staging_prepare(
            operation->executor->staging, alignment);
        if (iree_status_is_ok(status) && !file->async.bucket) {
          status = iree_hal_amd_gpu_file_staging_acquire(
              operation->executor->staging, window->proactor,
              &file->async.bucket);
        }
        if (iree_status_is_ok(status)) {
          const uint64_t capacity = iree_hal_amd_gpu_file_staging_capacity(
              operation->executor->staging);
          const uint64_t desired = iree_min(
              iree_min(range.length, capacity) * pipeline->piece_capacity,
              capacity);
          window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_RESERVED;
          pipeline->active |= bit;
          pipeline->admission_pending = true;
          iree_hal_amd_gpu_file_staging_reserve(
              file->async.bucket, desired, alignment,
              (iree_hal_file_transfer_callback_t){
                  iree_hal_amd_gpu_file_window_complete, window},
              &window->staging);
        }
      } else {
        status = iree_hal_amd_gpu_file_window_fill(window);
        if (iree_status_is_ok(status)) {
          pipeline->active |= bit;
          iree_hal_amd_gpu_file_window_start(window);
        }
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_gpu_file_pipeline_fail(pipeline, status);
      }
    }

    if (!pipeline->active && !pipeline->metadata.pending &&
        (pipeline->exhausted ||
         iree_atomic_load(&pipeline->failure, iree_memory_order_acquire))) {
      pipeline->releasing = true;
      while (pipeline->release_index < operation->batch.files.count) {
        const iree_hal_amd_gpu_file_binding_t* file =
            &operation->files[pipeline->release_index];
        if (file->kind == IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC &&
            file->async.bucket) {
          break;
        }
        ++pipeline->release_index;
      }
      if (pipeline->release_index == operation->batch.files.count) {
        iree_hal_amd_gpu_file_pipeline_finish(pipeline);
        return;
      }
      iree_hal_amd_gpu_file_window_t* window = &pipeline->windows[0];
      window->phase = IREE_HAL_AMD_GPU_FILE_WINDOW_RELEASE;
      pipeline->active = 1;
      iree_hal_amd_gpu_file_staging_release(
          operation->files[pipeline->release_index++].async.bucket,
          iree_ok_status(),
          (iree_hal_file_transfer_callback_t){
              iree_hal_amd_gpu_file_window_complete, window},
          &window->staging);
    }
    int64_t expected = IREE_HAL_AMD_GPU_FILE_DRAINING;
    if (iree_atomic_compare_exchange_strong(&pipeline->ready, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return;
    }
  }
}

static void iree_hal_amd_gpu_file_pipeline_begin(void* user_data,
                                                 iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  iree_hal_file_transfer_iterator_t iterator;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_transfer_iterator_initialize(
        &operation->batch, operation->count, NULL, &iterator);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  if (!iterator.count) {
    iree_hal_amd_gpu_file_transfer_complete(operation, iree_ok_status());
    return;
  }
  iree_hal_amd_gpu_buffer_reference_t metadata_reference = {0};
  bool download = false;
  const iree_hal_file_transfer_list_t* list = &operation->batch.transfers;
  if (list->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    const iree_hal_buffer_binding_t binding = list->indirect.records;
    download = binding.buffer->host_binding_index ==
               IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
    if (download) {
      status = iree_hal_amd_gpu_buffer_reference_initialize(
          operation->executor->queue->queue_family,
          operation->executor->transfer_site, binding.buffer, binding.offset,
          &metadata_reference);
    } else {
      const uint64_t length =
          iterator.count * sizeof(iree_hal_file_transfer_record_t);
      iree_byte_span_t span;
      status = iree_hal_buffer_native_host_span(binding.buffer, binding.offset,
                                                length, &span);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_buffer_invalidate_host_range(
            binding.buffer, binding.offset, length);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_file_transfer_iterator_acquire(
            &iterator, (const void*)span.data, iterator.count);
      }
    }
  }
  iree_hal_file_transfer_range_t first = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_transfer_iterator_peek(&iterator, &first);
  }
  if (!iree_status_is_ok(status) ||
      (!first.length && iterator.index == iterator.count)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  if (operation->has_native_files && operation->before && first.length) {
    // A captured host list already identifies its first view. Qualify its
    // native route before publication and combine its host visibility with the
    // producer boundary. Indirect lists consumed that boundary while acquiring
    // their count, before any record was available to inspect.
    const iree_hal_amd_gpu_file_executor_t* executor = operation->executor;
    const bool native_file = operation->files[first.file_binding].kind ==
                             IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC;
    status = iree_hal_amd_gpu_file_buffer_prepare(
        operation, first.buffer_binding, native_file);
    iree_hal_amd_gpu_file_buffer_t* buffer =
        &operation->buffers[first.buffer_binding];
    if (iree_status_is_ok(status) && native_file &&
        !buffer->source.contents.data) {
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(
              operation->batch.files.bindings[first.file_binding].file);
      status = iree_hal_amd_gpu_file_staging_prepare(
          executor->staging, operation->access == IREE_HAL_MEMORY_ACCESS_READ
                                 ? requirements->read.buffer_alignment
                                 : requirements->write.buffer_alignment);
    }
    if (iree_status_is_ok(status)) {
      if (buffer->before.bits) {
        executor->add_effects(executor->queue, buffer->before,
                              operation->before);
        buffer->before.bits = 0;
      }
      iree_hal_amd_gpu_file_transfer_submit_native(
          operation, 0, NULL, NULL, iree_hal_amd_gpu_file_pipeline_begin);
    } else {
      iree_hal_amd_gpu_file_transfer_complete(operation, status);
    }
    return;
  }
  const bool linear = iterator.index == iterator.count &&
                      iterator.current.row_count == 1 &&
                      iterator.current.slice_count == 1;
  const uint32_t piece_capacity =
      linear ? 1 : IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY;
  const uint32_t window_count =
      operation->executor->staging
          ? iree_min(32, iree_hal_amd_gpu_file_staging_concurrency(
                             operation->executor->staging,
                             linear ? first.length : UINT64_MAX))
          : 1;
  iree_host_size_t total_size = 0, windows_offset = 0, pieces_offset = 0,
                   metadata_offset = 0;
  status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_file_pipeline_t), &total_size,
      IREE_STRUCT_FIELD(window_count, iree_hal_amd_gpu_file_window_t,
                        &windows_offset),
      IREE_STRUCT_FIELD(window_count * piece_capacity,
                        iree_hal_amd_gpu_file_piece_t, &pieces_offset),
      IREE_STRUCT_FIELD(download ? IREE_HAL_AMD_GPU_FILE_WINDOW_CAPACITY : 0,
                        iree_hal_file_transfer_record_t, &metadata_offset));
  iree_byte_span_t storage = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_capture_pool_acquire(
        operation->executor->captures, total_size, &storage);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_transfer_complete(operation, status);
    return;
  }
  iree_hal_amd_gpu_file_pipeline_t* pipeline = (void*)storage.data;
  *pipeline = (iree_hal_amd_gpu_file_pipeline_t){
      .operation = operation,
      .allocation_size = storage.data_length,
      .iterator = iterator,
      .window_count = window_count,
      .piece_capacity = piece_capacity,
      .metadata = {
          .records = download ? (void*)(storage.data + metadata_offset) : NULL,
          .reference = metadata_reference}};
  for (uint32_t i = 0; i < window_count; ++i) {
    pipeline->windows[i] = (iree_hal_amd_gpu_file_window_t){
        .pipeline = pipeline,
        .pieces =
            (iree_hal_amd_gpu_file_piece_t*)(storage.data + pieces_offset) +
            i * piece_capacity};
  }
  iree_hal_amd_gpu_file_pipeline_notify(pipeline, 0);
}

static void iree_hal_amd_gpu_file_count_ready(void* user_data,
                                              iree_status_t status) {
  iree_hal_amd_gpu_file_transfer_t* operation = user_data;
  const iree_hal_file_transfer_list_t* list = &operation->batch.transfers;
  if (iree_status_is_ok(status) &&
      list->indirect.count_buffer->host_binding_index !=
          IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    iree_byte_span_t span;
    status = iree_hal_buffer_native_host_span(
        list->indirect.count_buffer, list->indirect.count_offset, 8, &span);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_buffer_invalidate_host_range(
          list->indirect.count_buffer, list->indirect.count_offset, 8);
    }
    if (iree_status_is_ok(status)) {
      memcpy(&operation->count, span.data, 8);
    }
  }
  if (iree_status_is_ok(status)) {
    operation->count = iree_unaligned_load_le_u64(&operation->count);
  }
  iree_hal_amd_gpu_file_pipeline_begin(operation, status);
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
  const iree_hal_file_transfer_list_t* list = &operation->batch.transfers;
  if (list->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_amd_gpu_transfer_operation_t copy = {0};
    if (list->indirect.count_buffer->host_binding_index ==
        IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
      copy.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      copy.target.host = &operation->count;
      copy.length = 8;
      status = iree_hal_amd_gpu_buffer_reference_initialize(
          operation->executor->queue->queue_family,
          operation->executor->transfer_site, list->indirect.count_buffer,
          list->indirect.count_offset, &copy.source.device);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_file_transfer_complete(operation, status);
    } else {
      iree_hal_amd_gpu_file_transfer_submit_native(
          operation, copy.length ? 1 : 0, &copy, NULL,
          iree_hal_amd_gpu_file_count_ready);
    }
  } else {
    // Memory-file copies carry the before boundary on their first publication.
    iree_hal_amd_gpu_file_pipeline_begin(operation, iree_ok_status());
  }
}

// Classifies and qualifies each file once. A device-visible import follows the
// same family/access contract as queue_copy and remains owned by its file.
// The address itself is resolved by native submission after producer waits.
static iree_status_t iree_hal_amd_gpu_file_binding_initialize(
    const iree_hal_amd_gpu_file_executor_t* executor,
    iree_hal_memory_access_t access, iree_hal_file_binding_t binding,
    iree_hal_amd_gpu_file_binding_t* out_binding) {
  *out_binding = (iree_hal_amd_gpu_file_binding_t){0};
  iree_hal_buffer_t* storage = iree_hal_file_storage_buffer(binding.file);
  if (storage && iree_all_bits_set(iree_hal_buffer_memory_type(storage),
                                   IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    const uint64_t storage_length = iree_hal_buffer_byte_length(storage);
    if (binding.offset > storage_length ||
        binding.length > storage_length - binding.offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "file binding exceeds its execution storage");
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
        iree_hal_buffer_allowed_access(storage), access));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_family_usage(
        storage, executor->queue->queue_family,
        access == IREE_HAL_MEMORY_ACCESS_READ
            ? IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE
            : IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
    out_binding->kind = IREE_HAL_AMD_GPU_FILE_BINDING_DEVICE;
    return iree_hal_amd_gpu_buffer_reference_initialize(
        executor->queue->queue_family, executor->transfer_site, storage,
        (iree_device_size_t)binding.offset, &out_binding->device);
  }
  iree_async_file_t* native = iree_hal_file_async_handle(binding.file);
  if (native) {
    out_binding->kind = IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC;
    out_binding->async.handle = native;
    return iree_ok_status();
  }
  if (iree_hal_memory_file_isa(binding.file)) {
    out_binding->kind = IREE_HAL_AMD_GPU_FILE_BINDING_HOST;
    IREE_RETURN_IF_ERROR(
        iree_hal_memory_file_contents(binding.file, &out_binding->host));
    if (binding.offset) {
      out_binding->host.data += binding.offset;
    }
    out_binding->host.data_length = (iree_host_size_t)binding.length;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "AMD queue file transfers require execution "
                          "storage, host memory, or an async file handle");
}

iree_status_t iree_hal_amd_gpu_file_transfer_submit_batch(
    const iree_hal_amd_gpu_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_memory_access_t file_access, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers, const void* before,
    const void* after) {
  iree_host_size_t batch_size = 0, tracker_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_storage_size(
      files, buffers, transfers, &batch_size));
  if (waits.count) {
    IREE_RETURN_IF_ERROR(iree_async_semaphore_wait_tracker_storage_size(
        waits.count, &tracker_size));
  }
  iree_host_size_t total_size = 0, batch_offset = 0;
  iree_host_size_t files_offset = 0, buffers_offset = 0;
  iree_host_size_t wait_values_offset = 0, wait_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0, signal_semaphores_offset = 0;
  iree_host_size_t tracker_offset = 0, before_offset = 0, after_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_file_transfer_t) +
          (executor->profile ? sizeof(iree_hal_amd_gpu_profile_operation_t)
                             : 0),
      &total_size,
      IREE_STRUCT_FIELD_ALIGNED(batch_size, uint8_t, iree_max_align_t,
                                &batch_offset),
      IREE_STRUCT_FIELD(files.count, iree_hal_amd_gpu_file_binding_t,
                        &files_offset),
      IREE_STRUCT_FIELD(buffers.count, iree_hal_amd_gpu_file_buffer_t,
                        &buffers_offset),
      IREE_STRUCT_FIELD_ALIGNED(executor->barrier_size, uint8_t,
                                iree_max_align_t, &before_offset),
      IREE_STRUCT_FIELD_ALIGNED(executor->barrier_size, uint8_t,
                                iree_max_align_t, &after_offset),
      // Native barriers may end between timeline-value alignment boundaries.
      IREE_STRUCT_FIELD_ALIGNED(waits.count, uint64_t, iree_alignof(uint64_t),
                                &wait_values_offset),
      IREE_STRUCT_FIELD(waits.count, iree_hal_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, uint64_t, iree_alignof(uint64_t),
                                &signal_values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*,
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
      .files = (void*)(storage.data + files_offset),
      .buffers = (void*)(storage.data + buffers_offset),
      .access = file_access,
      .waits = {waits.count, (void*)(storage.data + wait_semaphores_offset),
                (void*)(storage.data + wait_values_offset)},
      .signals = {signals.count,
                  (void*)(storage.data + signal_semaphores_offset),
                  (void*)(storage.data + signal_values_offset)},
      .before = storage.data + before_offset,
      .after = storage.data + after_offset,
      .dependency = {.storage = {.capacity = tracker_size,
                                 .buffer = storage.data + tracker_offset}}};
  iree_status_t status = iree_hal_file_transfer_batch_capture(
      executor->queue, file_access, files, buffers, transfers,
      storage.data + batch_offset, &operation->batch);
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_file_binding_t* binding = &operation->files[i];
    status = iree_hal_amd_gpu_file_binding_initialize(
        executor, file_access, files.bindings[i], binding);
    if (binding->kind == IREE_HAL_AMD_GPU_FILE_BINDING_ASYNC) {
      operation->has_native_files = true;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_capture_pool_release(executor->captures, storage);
    return status;
  }
  memset(operation->buffers, 0, buffers.count * sizeof(*operation->buffers));
  memcpy(operation->before, before, executor->barrier_size);
  memcpy(operation->after, after, executor->barrier_size);
  for (iree_host_size_t i = 0; i < files.count; ++i) {
    iree_hal_file_retain(files.bindings[i].file);
  }
  for (iree_host_size_t i = 0; i < buffers.count; ++i) {
    iree_hal_buffer_retain(buffers.bindings[i].buffer);
  }
  if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_buffer_retain(transfers->indirect.records.buffer);
    iree_hal_buffer_retain(transfers->indirect.count_buffer);
  }
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
  if (executor->profile) {
    iree_hal_amd_gpu_profile_operation_t* profile = (void*)(operation + 1);
    iree_hal_amd_gpu_profile_operation_initialize(
        executor->profile, executor->queue_ordinal,
        file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
            : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE,
        waits.count, signals.count,
        transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST
            ? transfers->host.count
            : 0,
        profile);
    if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST) {
      // Admission events describe captured host payloads. Indirect payloads
      // are not available at admission and must not be read ahead for tracing.
      for (iree_host_size_t i = 0; i < transfers->host.count; ++i) {
        const iree_hal_file_transfer_t* transfer =
            &transfers->host.transfers[i];
        const uint64_t length = iree_math_saturating_mul_u64(
            iree_math_saturating_mul_u64(transfer->row_length,
                                         transfer->row_count),
            transfer->slice_count);
        profile->event.payload_length =
            iree_math_saturating_add_u64(profile->event.payload_length, length);
      }
    }
    iree_hal_amd_gpu_profile_operation_defer(profile);
  }
  executor->begin(executor->queue);
  if (waits.count) {
    status = iree_hal_amd_gpu_host_wait_begin(
        executor->waits, operation->waits, iree_hal_amd_gpu_file_transfer_start,
        operation, iree_allocator_inline_arena(&operation->dependency.storage),
        &operation->dependency.wait);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_file_transfer_start(operation, status);
    }
  } else {
    iree_hal_amd_gpu_file_transfer_start(operation, iree_ok_status());
  }
  return iree_ok_status();
}
