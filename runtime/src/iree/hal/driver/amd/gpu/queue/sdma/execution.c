// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/execution.h"

#include "iree/async/frontier_tracker.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/growth.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/state.h"
#include "iree/hal/driver/amd/status.h"

static bool iree_hal_amd_gpu_sdma_prepare(void* user_data, uint64_t position);

static void iree_hal_amd_gpu_sdma_reject(
    iree_hal_amd_gpu_sdma_operation_t* operation, iree_status_code_t code) {
  int32_t expected = IREE_STATUS_OK;
  iree_atomic_compare_exchange_strong(&operation->failure_code, &expected, code,
                                      iree_memory_order_relaxed,
                                      iree_memory_order_relaxed);
}

static void iree_hal_amd_gpu_sdma_end_publication(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_hal_amd_gpu_sdma_operation_release(operation);
  iree_hal_amd_gpu_sdma_host_end(&queue->base);
}

void iree_hal_amd_gpu_sdma_execution_abandon(
    iree_hal_amd_gpu_sdma_operation_t* operation, iree_status_t status) {
  iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(status));
  iree_status_free(status);
  // An asynchronous preparation error is not proof that captured inputs have
  // retired. Re-enter admission in failure mode to join those inputs using the
  // wait storage reserved at capture. Accepted spans already own that join;
  // a device-wide failure instead belongs to the parent's stop protocol.
  if (!operation->accepted && operation->wait_count &&
      !iree_atomic_load(&operation->queue->failure_code,
                        iree_memory_order_acquire) &&
      amdf_status_is_ok(iree_hal_amd_gpu_sdma_native_queue_query_status(
          operation->queue->native))) {
    operation->admitted = false;
    for (iree_host_size_t i = 0; i < operation->wait_count; ++i) {
      iree_hal_amd_gpu_signal_release(operation->waits[i].signal);
      operation->waits[i].signal = NULL;
    }
    iree_hal_amd_gpu_sdma_execution_resume(operation);
  } else {
    iree_hal_amd_gpu_sdma_end_publication(operation);
  }
}

static void iree_hal_amd_gpu_sdma_rejected(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_sdma_execution_abandon(
      user_data, IREE_HAL_AMD_STATUS_FROM_AMDF(status, "SDMA publication"));
}

void iree_hal_amd_gpu_sdma_execution_resume(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  operation->submission.prepare = iree_hal_amd_gpu_sdma_prepare;
  operation->submission.completion = (iree_hal_amd_gpu_sdma_completion_t){
      .notify = iree_hal_amd_gpu_sdma_rejected, .user_data = operation};
  iree_hal_amd_gpu_sdma_native_queue_defer(operation->queue->native,
                                           &operation->submission);
}

static void iree_hal_amd_gpu_sdma_pending_ready(void* user_data,
                                                iree_status_t status) {
  iree_hal_amd_gpu_sdma_pending_t* pending = user_data;
  iree_hal_amd_gpu_sdma_operation_t* operation = pending->operation;
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  if (!iree_status_is_ok(status)) {
    // Dependency failure still joins all sibling inputs. Queue cancellation
    // instead stops admission under the parent's native-quiescence contract.
    uint64_t value = 0;
    iree_status_t dependency =
        iree_hal_semaphore_query(pending->source, &value);
    if (!iree_status_is_ok(dependency)) {
      iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(dependency));
      iree_status_free(status);
      status = iree_ok_status();
    }
    iree_status_free(dependency);
  }
  iree_hal_semaphore_release(pending->semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_execution_resume(operation);
  } else {
    iree_hal_amd_gpu_sdma_execution_abandon(operation, status);
  }
  iree_hal_amd_gpu_sdma_host_end(&queue->base);
}

static void iree_hal_amd_gpu_sdma_wait(
    iree_hal_amd_gpu_sdma_operation_t* operation, iree_host_size_t index,
    bool submission_available) {
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_hal_amd_gpu_sdma_pending_t* pending = operation->pending;
  pending->value = operation->waits[index].value;
  pending->source = operation->waits[index].semaphore;
  iree_hal_amd_gpu_sdma_host_begin(&queue->base);
  iree_hal_amd_gpu_profile_operation_defer(
      iree_hal_amd_gpu_sdma_operation_profile(operation));
  iree_status_t status = iree_ok_status();
  if (submission_available) {
    status = iree_hal_amd_gpu_semaphore_acquire_submission_semaphore(
        pending->source, &pending->semaphore);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(status));
      iree_status_free(status);
      status = iree_ok_status();
      pending->semaphore = pending->source;
      iree_hal_semaphore_retain(pending->semaphore);
    }
  } else {
    pending->semaphore = pending->source;
    iree_hal_semaphore_retain(pending->semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_host_wait_set_begin(
        &queue->waits,
        (iree_hal_semaphore_list_t){1, &pending->semaphore, &pending->value},
        iree_hal_amd_gpu_sdma_pending_ready, pending,
        iree_allocator_inline_arena(&pending->tracker), &pending->entry);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  iree_hal_amd_gpu_sdma_pending_ready(pending, status);
  return;
}

static bool iree_hal_amd_gpu_sdma_admit(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  if (operation->admitted) {
    return true;
  }
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  const bool native_eligible =
      !queue->options.barriers.submission &&
      !(operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER &&
        operation->transfer.host_wait_required) &&
      !(operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION &&
        (!queue->frontier.tracker ||
         iree_hal_amd_gpu_allocation_requires_host_wait(
             operation->allocation)));
  iree_status_t status = iree_ok_status();
  iree_host_size_t deferred = operation->wait_count;
  bool available = false;
  for (iree_host_size_t i = 0;
       i < operation->wait_count && deferred == operation->wait_count &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_sdma_wait_t* wait = &operation->waits[i];
    if (native_eligible && iree_hal_amd_gpu_semaphore_isa(wait->semaphore) &&
        !iree_atomic_load(&operation->failure_code,
                          iree_memory_order_relaxed)) {
      iree_hal_amd_gpu_semaphore_wait_t acquired;
      status = iree_hal_amd_gpu_semaphore_acquire_wait(
          wait->semaphore, wait->value, queue->context, &acquired);
      if (iree_status_is_ok(status)) {
        wait->signal = acquired.signal;
        if (acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED ||
            acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST) {
          deferred = i;
          available =
              acquired.state == IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED;
        }
      }
    } else {
      uint64_t value = 0;
      status = iree_hal_semaphore_query(wait->semaphore, &value);
      if (iree_status_is_ok(status) && value < wait->value) {
        deferred = i;
      }
    }
  }
  if (iree_status_is_ok(status) && deferred == operation->wait_count &&
      !iree_atomic_load(&operation->failure_code, iree_memory_order_relaxed)) {
    operation->admitted = true;
    return true;
  }
  for (iree_host_size_t i = 0; i < operation->wait_count; ++i) {
    iree_hal_amd_gpu_signal_release(operation->waits[i].signal);
    operation->waits[i].signal = NULL;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(status));
    iree_status_free(status);
  }
  if (iree_atomic_load(&operation->failure_code, iree_memory_order_relaxed)) {
    // A failed producer does not permit us to return resources still being used
    // by another input. Consume only completed/failed edges and park on the
    // next.
    deferred = 0;
    available = false;
    bool unresolved = false;
    while (operation->wait_count && !unresolved) {
      iree_hal_amd_gpu_sdma_wait_t* wait = operation->waits;
      uint64_t value = 0;
      status = iree_hal_semaphore_query(wait->semaphore, &value);
      unresolved = iree_status_is_ok(status) && value < wait->value;
      iree_status_free(status);
      if (!unresolved) {
        iree_hal_semaphore_release(wait->semaphore);
        ++operation->waits;
        --operation->wait_count;
      }
    }
    if (!operation->wait_count) {
      iree_hal_amd_gpu_sdma_execution_abandon(
          operation, iree_status_from_code((iree_status_code_t)iree_atomic_load(
                         &operation->failure_code, iree_memory_order_relaxed)));
      return false;
    }
  }
  iree_hal_amd_gpu_sdma_wait(operation, deferred, available);
  return false;
}

static void iree_hal_amd_gpu_sdma_allocation_ready(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_gpu_sdma_operation_t* operation = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_execution_resume(operation);
  } else {
    iree_hal_amd_gpu_sdma_execution_abandon(operation, status);
  }
  iree_hal_amd_gpu_sdma_host_end(&queue->base);
}

// Native spans never own a new heap allocation. Download actions retain their
// exact fixed staging slot until the completion worker returns from this copy.
static void iree_hal_amd_gpu_sdma_download(void* user_data) {
  iree_hal_amd_gpu_download_t* download = user_data;
  iree_hal_amd_gpu_download_complete(download);
  download->length = 0;
}

static bool iree_hal_amd_gpu_sdma_ready(
    void* user_data, iree_hal_amd_gpu_completion_copy_t* out_copy) {
  iree_hal_amd_gpu_sdma_span_t* span = user_data;
  iree_hal_amd_gpu_sdma_operation_t* operation = span->operation;
  bool ready = true;
  for (iree_host_size_t i = 0; i < operation->wait_count && ready; ++i) {
    if (operation->waits[i].signal) {
      iree_status_code_t code = IREE_STATUS_OK;
      ready = iree_hal_amd_gpu_signal_query_retirement(
          operation->waits[i].signal, &code);
      if (ready && code != IREE_STATUS_OK) {
        iree_hal_amd_gpu_sdma_reject(operation, code);
      }
    }
  }
  if (!ready) {
    return false;
  }
  if (span->download.length &&
      !iree_atomic_load(&operation->failure_code, iree_memory_order_relaxed)) {
    if (out_copy) {
      *out_copy = (iree_hal_amd_gpu_completion_copy_t){
          .fn = iree_hal_amd_gpu_sdma_download, .user_data = &span->download};
    }
    return false;
  }
  return true;
}

static void iree_hal_amd_gpu_sdma_retired(
    void* user_data, amdf_status_t native_status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  iree_hal_amd_gpu_sdma_span_t* span = user_data;
  iree_hal_amd_gpu_sdma_operation_t* operation = span->operation;
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(native_status, "SDMA execution");
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(status));
    int32_t expected = 0;
    if (operation->callback.fn &&
        iree_atomic_compare_exchange_strong(
            &operation->notified_code, &expected, iree_status_code(status),
            iree_memory_order_relaxed, iree_memory_order_relaxed)) {
      operation->callback.fn(operation->callback.user_data,
                             iree_status_clone(status), NULL, false);
    }
  }
  iree_status_free(status);
  if (!amdf_status_is_ok(native_status) && span->owns_publication) {
    span->owns_publication = false;
    iree_hal_amd_gpu_sdma_end_publication(operation);
  }
  if (retirement != IREE_HAL_AMD_GPU_SDMA_RETIRED) {
    return;
  }
  if (amdf_status_is_ok(native_status)) {
    if (queue->frontier.tracker) {
      iree_async_single_frontier_initialize(&operation->frontier,
                                            queue->frontier.axis, span->epoch);
      iree_async_frontier_tracker_advance(queue->frontier.tracker,
                                          queue->frontier.axis, span->epoch);
    }
    iree_atomic_store(&queue->completed_epoch, span->epoch,
                      iree_memory_order_release);
  }
  if (span->owns_publication) {
    span->owns_publication = false;
    // This span retained both its native reference and the unpublished suffix.
    // Resumption is enqueued, never executed on the completion observer.
    if (iree_atomic_load(&operation->failure_code, iree_memory_order_relaxed)) {
      iree_hal_amd_gpu_sdma_execution_abandon(
          operation, iree_status_from_code((iree_status_code_t)iree_atomic_load(
                         &operation->failure_code, iree_memory_order_relaxed)));
    } else {
      iree_hal_amd_gpu_sdma_execution_resume(operation);
    }
  }
  iree_hal_amd_gpu_sdma_operation_release(operation);
}

static void iree_hal_amd_gpu_sdma_accepted(void* user_data) {
  iree_hal_amd_gpu_sdma_operation_t* operation = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_hal_amd_gpu_sdma_span_t* span = &queue->spans[operation->slot];
  const bool terminal = span->next.phase == IREE_HAL_AMD_GPU_SDMA_PHASE_END;
  span->owns_publication = !terminal && span->boundary;
  for (iree_host_size_t i = operation->cursor.wait; i < span->next.wait; ++i) {
    if (operation->waits[i].signal) {
      iree_hal_amd_gpu_signal_join_frontier(operation->waits[i].signal,
                                            queue->producer.count,
                                            queue->producer.published);
    }
  }
  queue->producer.published[queue->options.queue_ordinal] = span->epoch;
  operation->cursor = span->next;
  operation->accepted = true;
  if (operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER) {
    operation->transfer.cursor = operation->cursor.transfer;
  }
  if (terminal &&
      operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION) {
    iree_async_single_frontier_t death;
    iree_async_single_frontier_initialize(&death, queue->frontier.axis,
                                          span->epoch);
    iree_hal_amd_gpu_allocation_accept(
        operation->allocation,
        queue->frontier.tracker
            ? iree_async_single_frontier_as_const_frontier(&death)
            : NULL);
  }
  if (terminal && operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE &&
      operation->execute.instance) {
    iree_hal_amd_gpu_program_publish(operation->execute.storage,
                                     operation->execute.instance,
                                     queue->options.queue_ordinal, span->epoch);
    operation->program_published = true;
  }
  if (terminal && operation->signal) {
    iree_hal_amd_gpu_signal_publish_frontier(operation->signal,
                                             queue->producer.published);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < operation->signals.count && iree_status_is_ok(status); ++i) {
      if (iree_hal_amd_gpu_semaphore_isa(operation->signals.semaphores[i])) {
        status = iree_hal_amd_gpu_semaphore_publish_signal(
            operation->signals.semaphores[i],
            operation->signals.payload_values[i], operation->signal);
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_sdma_reject(operation, iree_status_code(status));
      iree_status_free(status);
    }
  }
  if (terminal) {
    iree_hal_amd_gpu_sdma_end_publication(operation);
  } else if (!span->boundary) {
    iree_hal_amd_gpu_sdma_execution_resume(operation);
  }
}

static uint32_t* iree_hal_amd_gpu_sdma_barrier_words(
    iree_hal_amd_gpu_sdma_queue_t* queue,
    iree_hal_amd_gpu_sdma_barrier_t barrier, uint32_t* words) {
  if (barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN) {
    *words++ = 0;
  }
  const uint32_t cache = ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE)
                              ? queue->options.barriers.acquire
                              : 0) |
                         ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE)
                              ? queue->options.barriers.release
                              : 0);
  return cache ? iree_hal_amd_gpu_sdma_emit_cache(words, cache) : words;
}

static iree_status_t iree_hal_amd_gpu_sdma_emit(
    iree_hal_amd_gpu_sdma_operation_t* operation,
    iree_hal_amd_gpu_sdma_span_t* span, uint32_t* out_word_count) {
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  iree_hal_amd_gpu_sdma_cursor_t* next = &span->next;
  uint32_t* words = span->words;
  uint32_t* const end = words + IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS;
  const uint64_t offset =
      (uint64_t)operation->slot * queue->options.staging_capacity;
  const uint64_t staging_address = queue->staging->address + offset;
  const iree_byte_span_t staging =
      iree_make_byte_span(queue->staging->memory.contents.data + offset,
                          queue->options.staging_capacity);
  iree_status_t status = iree_ok_status();
  bool full = false;
  while (!full && !span->boundary &&
         next->phase != IREE_HAL_AMD_GPU_SDMA_PHASE_END &&
         iree_status_is_ok(status)) {
    const uint32_t capacity = (uint32_t)(end - words);
    switch (next->phase) {
      case IREE_HAL_AMD_GPU_SDMA_PHASE_WAITS:
        while (next->wait < operation->wait_count && !full) {
          const iree_hal_amd_gpu_signal_t* signal =
              operation->waits[next->wait].signal;
          if (signal) {
            if (end - words < 6) {
              full = true;
              break;
            }
            words = iree_hal_amd_gpu_sdma_emit_wait(
                words, queue->features,
                signal->device_address +
                    queue->options.signal_layout.value_offset,
                1u - queue->options.signal_layout.pending_value);
          }
          ++next->wait;
        }
        if (!full) {
          next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_BEFORE;
        }
        break;
      case IREE_HAL_AMD_GPU_SDMA_PHASE_BEFORE:
      case IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER: {
        if (capacity < 6) {
          full = true;
          break;
        }
        const bool before = next->phase == IREE_HAL_AMD_GPU_SDMA_PHASE_BEFORE;
        words = iree_hal_amd_gpu_sdma_barrier_words(
            queue, before ? operation->before : operation->after, words);
        next->phase = before ? IREE_HAL_AMD_GPU_SDMA_PHASE_INITIALIZE
                             : IREE_HAL_AMD_GPU_SDMA_PHASE_SIGNAL;
        break;
      }
      case IREE_HAL_AMD_GPU_SDMA_PHASE_INITIALIZE:
        if (operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE &&
            operation->execute.initialize_length) {
          while (next->initialize_offset <
                     operation->execute.initialize_length &&
                 !full) {
            if (end - words < 7) {
              full = true;
              break;
            }
            const uint32_t length = (uint32_t)iree_min(
                operation->execute.initialize_length - next->initialize_offset,
                IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
            words = iree_hal_amd_gpu_sdma_emit_copy(
                words, queue->features,
                operation->execute.source_address + next->initialize_offset,
                operation->execute.instance->address + next->initialize_offset,
                length);
            next->initialize_offset += length;
          }
          if (!full) {
            if (words == end) {
              full = true;
            } else {
              *words++ =
                  0;  // Join the private initialization before any reader.
            }
          }
        }
        if (!full) {
          next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_BODY;
        }
        break;
      case IREE_HAL_AMD_GPU_SDMA_PHASE_BODY:
        switch (operation->kind) {
          case IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER: {
            const uint32_t release_words =
                operation->transfer.host_completion_required &&
                        queue->options.barriers.release
                    ? 5u
                    : 0u;
            if (capacity < release_words + 7) {
              full = true;
              break;
            }
            iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
            status = iree_hal_amd_gpu_dma_transfer_prepare_chunk(
                &queue->encoding, &operation->transfer,
                capacity - release_words, staging, staging_address, &chunk);
            if (iree_status_is_ok(status)) {
              memcpy(words, chunk.words, chunk.word_count * sizeof(uint32_t));
              words += chunk.word_count;
              next->transfer = chunk.next;
              span->download = chunk.download;
              if (span->download.length && release_words) {
                words = iree_hal_amd_gpu_sdma_emit_cache(
                    words, queue->options.barriers.release);
              }
              if (next->transfer.index == operation->transfer.count) {
                next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER;
              } else {
                full = true;
              }
            }
            break;
          }
          case IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE: {
            uint32_t count = 0;
            const iree_hal_amd_gpu_sdma_program_progress_t progress =
                iree_hal_amd_gpu_sdma_program_emit(
                    iree_hal_amd_gpu_sdma_command_buffer_program(
                        operation->execute.invocation.command_buffer),
                    operation->execute.invocation.values, &next->program,
                    capacity, words, &count);
            words += count;
            span->boundary = progress == IREE_HAL_AMD_GPU_SDMA_PROGRAM_BOUNDARY;
            if (progress == IREE_HAL_AMD_GPU_SDMA_PROGRAM_END) {
              next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER;
            } else {
              full = true;
            }
            break;
          }
          case IREE_HAL_AMD_GPU_SDMA_OPERATION_TIMESTAMP: {
            if (capacity < 11) {
              full = true;
              break;
            }
            uint64_t target = 0;
            status = iree_hal_amd_gpu_buffer_reference_resolve(
                &operation->timestamp, &target);
            if (iree_status_is_ok(status)) {
              words = iree_hal_amd_gpu_sdma_emit_timestamp(
                  words, queue->features, staging_address);
              *words++ = 0;
              words = iree_hal_amd_gpu_sdma_emit_copy(
                  words, queue->features, staging_address, target, 8);
              next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER;
            }
            break;
          }
          default:
            next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER;
            break;
        }
        break;
      case IREE_HAL_AMD_GPU_SDMA_PHASE_SIGNAL:
        if (operation->signal) {
          if (capacity < 4) {
            full = true;
            break;
          }
          words = iree_hal_amd_gpu_sdma_emit_fence(
              words, queue->features,
              operation->signal->device_address +
                  queue->options.signal_layout.value_offset,
              1u - queue->options.signal_layout.pending_value);
        }
        next->phase = IREE_HAL_AMD_GPU_SDMA_PHASE_END;
        break;
      case IREE_HAL_AMD_GPU_SDMA_PHASE_END:
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  *out_word_count = (uint32_t)(words - span->words);
  return status;
}

static bool iree_hal_amd_gpu_sdma_prepare(void* user_data, uint64_t position) {
  iree_hal_amd_gpu_sdma_operation_t* operation = user_data;
  iree_hal_amd_gpu_sdma_queue_t* queue = operation->queue;
  if (!iree_hal_amd_gpu_sdma_admit(operation)) {
    return false;
  }
  memcpy(queue->producer.pending, queue->producer.published,
         queue->producer.count * sizeof(uint64_t));
  for (iree_host_size_t i = 0; i < operation->wait_count; ++i) {
    if (operation->waits[i].signal) {
      iree_hal_amd_gpu_signal_join_frontier(operation->waits[i].signal,
                                            queue->producer.count,
                                            queue->producer.pending);
    }
  }
  iree_status_t status = iree_hal_amd_gpu_sdma_check_failure(queue);
  bool pending = false;
  if (iree_status_is_ok(status) && !operation->prepared) {
    const bool native_signal =
        operation->signals.count && !queue->options.barriers.submission &&
        !(operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER &&
          operation->transfer.host_completion_required);
    if (native_signal && !operation->signal) {
      operation->signal = iree_hal_amd_gpu_signal_pool_try_acquire(
          &queue->options.signals->pool,
          queue->options.signal_layout.pending_value);
      if (!operation->signal) {
        status = iree_hal_amd_gpu_sdma_growth_begin(operation, NULL);
        pending = iree_status_is_ok(status);
      }
    }
    if (iree_status_is_ok(status) && !pending &&
        operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE) {
      iree_hal_amd_gpu_program_t* storage = operation->execute.storage;
      if (storage && !operation->execute.instance) {
        operation->execute.instance = iree_hal_amd_gpu_program_try_acquire(
            storage, queue->options.queue_ordinal, queue->producer.pending);
        if (!operation->execute.instance) {
          status = iree_hal_amd_gpu_sdma_growth_begin(operation, storage);
          pending = iree_status_is_ok(status);
        } else if (!operation->execute.instance->initialized) {
          operation->execute.source_address =
              iree_hal_amd_gpu_program_source(storage).address;
          operation->execute.initialize_length = storage->byte_length;
        }
        if (operation->execute.instance &&
            operation->execute.instance->join_previous) {
          operation->before |= IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN;
        }
      }
      if (iree_status_is_ok(status) && !pending) {
        status = iree_hal_amd_gpu_sdma_command_invocation_resolve(
            &operation->execute.invocation, queue->options.queue_ordinal,
            storage ? operation->execute.instance->address : 0);
      }
    }
    if (iree_status_is_ok(status) && !pending &&
        operation->kind == IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION) {
      iree_hal_amd_gpu_sdma_host_begin(&queue->base);
      status = iree_hal_amd_gpu_allocation_prepare(
          operation->allocation, queue->frontier.axis, queue->producer.count,
          queue->producer.pending, iree_hal_amd_gpu_sdma_allocation_ready,
          operation, &pending);
      if (!pending) {
        iree_hal_amd_gpu_sdma_host_end(&queue->base);
      }
    }
    if (iree_status_is_ok(status) && pending) {
      return false;
    }
    operation->prepared = iree_status_is_ok(status);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_execution_abandon(operation, status);
    return false;
  }
  operation->slot =
      (uint32_t)(position % IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY);
  iree_hal_amd_gpu_sdma_span_t* span = &queue->spans[operation->slot];
  span->operation = operation;
  span->epoch = queue->options.initial_epoch + position + 1;
  span->next = operation->cursor;
  span->boundary = false;
  span->owns_publication = true;
  span->download = (iree_hal_amd_gpu_download_t){0};
  uint32_t word_count = 0;
  status = iree_hal_amd_gpu_sdma_emit(operation, span, &word_count);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_sdma_execution_abandon(operation, status);
    return false;
  }
  uint32_t native_waits = 0;
  for (iree_host_size_t i = operation->cursor.wait; i < span->next.wait; ++i) {
    native_waits += operation->waits[i].signal != NULL;
  }
  iree_hal_amd_gpu_profile_operation_record(
      queue->profile, iree_hal_amd_gpu_sdma_operation_profile(operation), true,
      native_waits);
  iree_hal_amd_gpu_sdma_operation_retain(operation);
  operation->submission.words = span->words;
  operation->submission.word_count = word_count;
  operation->submission.completion = (iree_hal_amd_gpu_sdma_completion_t){
      .ready = iree_hal_amd_gpu_sdma_ready,
      .notify = iree_hal_amd_gpu_sdma_retired,
      .user_data = span,
  };
  return true;
}

void iree_hal_amd_gpu_sdma_execution_submit(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  operation->submitted = true;
  operation->submission = (iree_hal_amd_gpu_sdma_submission_t){
      .prepare = iree_hal_amd_gpu_sdma_prepare,
      .accepted = iree_hal_amd_gpu_sdma_accepted,
      .completion = {.notify = iree_hal_amd_gpu_sdma_rejected,
                     .user_data = operation},
      .user_data = operation,
  };
  iree_hal_amd_gpu_sdma_host_begin(&operation->queue->base);
  iree_atomic_fetch_add(&operation->queue->operation_count, 1,
                        iree_memory_order_relaxed);
  iree_hal_amd_gpu_sdma_native_queue_submit(operation->queue->native,
                                            &operation->submission);
}
