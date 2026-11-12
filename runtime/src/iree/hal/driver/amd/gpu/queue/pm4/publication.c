// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/publication.h"

#include "iree/base/internal/memory_copy.h"
#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/control_program.h"
#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"

// Native acceptance is irrevocable here. Finish every capture access before
// exposing the retirement record, transferring the final publication reference
// directly to completion. Only a piece with an unpublished suffix needs another
// reference. No capture access follows count publication or native
// notification.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_accepted(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position,
    bool terminal) {
  if (queue->producer.frontier.pending != queue->producer.frontier.published) {
    queue->producer.frontier.scratch = queue->producer.frontier.published;
    queue->producer.frontier.published = queue->producer.frontier.pending;
  }
  queue->producer.frontier.published[queue->options.queue_ordinal] =
      position + 1;
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION) {
    iree_async_single_frontier_t death_frontier;
    iree_async_single_frontier_initialize(&death_frontier, queue->frontier.axis,
                                          position + 1);
    iree_hal_amd_gpu_allocation_accept(
        submission->allocation.operation,
        queue->frontier.tracker
            ? iree_async_single_frontier_as_const_frontier(&death_frontier)
            : NULL);
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation =
        &submission->execute;
    iree_hal_amd_gpu_program_publish(
        &invocation->program->resident, invocation->instance,
        queue->options.queue_ordinal, position + 1);
    invocation->published = true;
    iree_hal_amd_gpu_byte_storage_pool_accept(&queue->values);
  }
  if (submission->signal) {
    iree_hal_amd_gpu_signal_publish_frontier(
        submission->signal, queue->producer.frontier.published);
  }
  if (submission->profiled) {
    if (queue->kernel) {
      iree_hal_amd_gpu_byte_storage_pool_accept(queue->samples);
    }
    iree_hal_amd_gpu_profile_operation_record(
        queue->profile, iree_hal_amd_gpu_pm4_submission_profile(submission),
        true, submission->native_wait_count);
  }
  if (!terminal) {
    iree_hal_amd_gpu_pm4_submission_retain(submission);
  }
  queue->issued[position & queue->retirement.capacity_mask].action.scratch =
      iree_hal_amd_gpu_pm4_scratch_publish(
          &queue->producer.scratch, position + 1,
          iree_hal_amd_gpu_pm4_submission_scratch_required(submission));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       terminal && submission->signal && i < submission->signals.count &&
       iree_status_is_ok(status);
       ++i) {
    if (iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i])) {
      status = iree_hal_amd_gpu_semaphore_publish_signal(
          submission->signals.semaphores[i],
          submission->signals.payload_values[i], submission->signal);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_fail_submission(queue, submission, status);
  }
  const bool has_host_owner = terminal && submission->has_host_owner;
  if (has_host_owner) {
    submission->has_host_owner = false;
  }
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
  // Native ownership becomes observable before deferred host ownership ends.
  // The publishing caller's trailing accesses remain under publisher ownership.
  if (has_host_owner) {
    iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                          iree_memory_order_release);
  }
  if (queue->kernel) {
    if (iree_atomic_exchange(&queue->kernel->wake_required, 0,
                             iree_memory_order_acq_rel)) {
      iree_async_event_native_set(&queue->context->kernel_event);
    }
  } else {
    const uint64_t published_words = queue->producer.ring.reserved_words;
    if (queue->producer.ring.host_publish_executor !=
        AMDF_CACHE_TRANSITION_EXECUTOR_NONE) {
      const amdf_status_t publication_status =
          iree_hal_amd_gpu_user_queue_flush(&queue->storage);
      if (IREE_UNLIKELY(!amdf_status_is_ok(publication_status))) {
        iree_hal_amd_gpu_pm4_fail_native(queue, publication_status,
                                         "flush PM4 command publication");
        return terminal ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED
                        : IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE;
      }
    }
    iree_atomic_store(queue->producer.write_index, published_words,
                      iree_memory_order_release);
    const iree_hal_memory_transition_recipe_info_t publication_fence = {
        .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
    };
    iree_hal_memory_transition_execute_host_direct(&publication_fence, NULL, 1);
    *queue->producer.doorbell = published_words;
  }
  return terminal ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED
                  : IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE;
}

// The exclusive publisher keeps one prepared, unaccepted KERNEL piece. Native
// capacity leaves its bytes/cursors intact. Rejection rolls back storage only;
// accepted earlier pieces retain their independent references and retirement.
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_kernel(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t position) {
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel = queue->kernel;
  iree_hal_amd_gpu_pm4_retirement_t* action =
      &queue->issued[position & queue->retirement.capacity_mask].action;
  const amdf_status_t status =
      iree_hal_amd_gpu_pm4_kmq_storage_submit(kernel, position);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY) &&
      position != (uint64_t)iree_atomic_load(&queue->completed.count,
                                             iree_memory_order_acquire)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (!amdf_status_is_ok(status)) {
    queue->producer.ring.reserved_words = kernel->rollback.words;
    if (kernel->rollback.storage) {
      kernel->rollback.storage->end = kernel->rollback.bytes;
    }
    kernel->pending.byte_length = 0;
    iree_hal_amd_gpu_byte_storage_pool_rollback(queue->samples);
    iree_hal_amd_gpu_byte_storage_pool_rollback(&queue->values);
    iree_hal_amd_gpu_pm4_abandon(
        queue, iree_hal_amd_gpu_pm4_retirement_submission(action),
        IREE_HAL_AMD_STATUS_FROM_AMDF(status, "gpu.kernel_queue_submit"));
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  return iree_hal_amd_gpu_pm4_publish_accepted(
      queue, iree_hal_amd_gpu_pm4_retirement_submission(action), position,
      iree_hal_amd_gpu_pm4_retirement_is_terminal(action));
}

// Encodes one complete piece before native publication. USER publication is
// infallible; KERNEL acceptance can defer without rebuilding or replaying it.
static iree_hal_amd_gpu_pm4_publication_result_t
iree_hal_amd_gpu_pm4_publish_commit(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position,
    const iree_hal_amd_gpu_pm4_ring_reservation_t* reservation,
    uint32_t body_words, uint32_t entry_word_count, uint32_t* words,
    iree_hal_amd_gpu_pm4_barrier_t after, uint64_t argument_end,
    iree_hal_amd_gpu_download_t download, bool terminal) {
  iree_hal_amd_gpu_pm4_retirement_t* action =
      &queue->issued[position & queue->retirement.capacity_mask].action;
  *action = (iree_hal_amd_gpu_pm4_retirement_t){
      .tagged_submission = (uintptr_t)submission | (uintptr_t)terminal,
      .consumed_arguments = argument_end,
      .download = download,
  };
  uint32_t release_cache_control = 0;
  words = iree_hal_amd_gpu_pm4_barrier_emit_tail(words, after,
                                                 &release_cache_control);
  if (submission->signal) {
    words = iree_hal_amd_gpu_pm4_emit_release32(
        words, release_cache_control, submission->signal->device_address, 1);
    if (release_cache_control) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
      release_cache_control = 0;
    }
  }
  words = iree_hal_amd_gpu_pm4_retirement_emit(
      words, &queue->retirement, position, action, release_cache_control);
  if (queue->kernel) {
    // A generation with no bytes in this piece can retire during metadata
    // progress while native capacity is unavailable. Retain only actual
    // borrows.
    if (queue->kernel->rollback.storage &&
        queue->kernel->rollback.storage->end == queue->kernel->rollback.bytes) {
      queue->kernel->rollback.storage = NULL;
    }
    if (entry_word_count) {
      // The resident IB chains into this fixed 32-DWORD continuation. Root
      // padding precedes its terminal CHAIN; only continuation padding is here.
      const uint32_t padding =
          body_words - (uint32_t)(words - reservation->words);
      if (padding) {
        words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
      }
    }
    IREE_ASSERT_EQ(words - reservation->words, body_words);
    action->consumed_words = queue->producer.ring.reserved_words;
    queue->kernel->pending = (amdf_gpu_kernel_command_t){
        .memory = queue->kernel->memory.handle,
        .byte_offset = (reservation->words - queue->producer.ring.words) * 4,
        .byte_length = entry_word_count ? entry_word_count * 4
                                        : (queue->producer.ring.reserved_words -
                                           reservation->position) *
                                              4,
    };
    return iree_hal_amd_gpu_pm4_publish_kernel(queue, position);
  }
  action->consumed_words = reservation->position + (words - reservation->words);
  words = iree_hal_amd_gpu_pm4_emit_notify32(
      words, 0, queue->context->event_info.mailbox_address,
      queue->context->event_info.trigger_data,
      queue->context->event_info.interrupt_context);
  IREE_ASSERT_EQ(words - reservation->words, body_words);
  return iree_hal_amd_gpu_pm4_publish_accepted(queue, submission, position,
                                               terminal);
}

iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_transfer(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  iree_hal_amd_gpu_transfer_t* transfer = &submission->transfer;
  const bool first = !transfer->cursor.index && !transfer->cursor.offset;
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  // Independent copies can overlap on shaders. Scalar copies keep their
  // compact DMA packet; incompatible program views and rings too small for
  // one shader argument record retain the queue path without extra backing.
  const bool program =
      transfer->program_accessible &&
      arguments->capacity >= sizeof(iree_hal_amd_gpu_blit_arguments_t) &&
      transfer->operations[transfer->cursor.index].type ==
          IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  iree_hal_amd_gpu_byte_storage_t* storage =
      program ? arguments : queue->producer.transfers.current;
  if (!program && transfer->staging_capacity &&
      (!storage || storage->capacity < transfer->staging_capacity)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_TRANSFERS;
  }
  uint64_t offset = 0, start = 0, available = 0;
  if (storage) {
    const uint64_t relative =
        iree_align_uint64(storage->end - storage->begin, program ? 32 : 8);
    offset = relative & (storage->capacity - 1);
    start = storage->begin + relative;
    const uint64_t completed = (uint64_t)iree_atomic_load(
        program ? &queue->completed.arguments : &queue->completed.transfers,
        iree_memory_order_acquire);
    const uint64_t reusable =
        completed >= storage->end ? start : iree_max(completed, storage->begin);
    const uint64_t used = start - reusable;
    available =
        used >= storage->capacity
            ? 0
            : iree_min(storage->capacity - used, storage->capacity - offset);
  }
  // A source fitting one admitted span needs no publication boundary. Larger
  // uploads overlap preparation with native reads in two bounded regions.
  // Downloads keep short retirement copies so independent work can progress.
  const uint64_t maximum_staging_length =
      transfer->host_completion_required
          ? IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_DOWNLOAD_LENGTH
          : IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_UPLOAD_LENGTH;
  const bool overlap = transfer->host_completion_required ||
                       (transfer->staging_capacity &&
                        transfer->operations[transfer->cursor.index].length >
                            transfer->staging_capacity);
  const uint64_t staging_window_length =
      storage ? iree_min(overlap ? storage->capacity / 2 : storage->capacity,
                         maximum_staging_length)
              : 0;
  // Admission reserves launch capacity beyond native waits. Keep 96 words for
  // signals, retirement, notification, and alignment, leaving room for DMA
  // packets even at the maximum admitted native fan-in.
  const uint32_t word_budget =
      iree_min(queue->producer.ring.capacity / 2, 16384u);
  const uint32_t wait_words = first ? submission->native_wait_count * 7 : 0;
  iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
  const uint32_t word_capacity = iree_min(
      IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS, word_budget - wait_words - 96);
  const iree_byte_span_t bytes = iree_make_byte_span(
      storage ? storage->memory.contents.data + offset : NULL,
      program ? available : iree_min(available, staging_window_length));
  const uint64_t address = storage ? storage->address + offset : 0;
  uint32_t argument_length = 0;
  iree_status_t status =
      program ? iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
                    transfer, &queue->options.programs->blit, word_capacity,
                    bytes, address, &chunk, &argument_length)
              : iree_hal_amd_gpu_dma_transfer_prepare_chunk(
                    &iree_hal_amd_gpu_pm4_transfer_encoding, transfer,
                    word_capacity, bytes, address, &chunk);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  if (!chunk.word_count) {
    // Aligned fills and device copies need no payload storage. A narrow fill
    // edge first discovers its transport requirement at native address
    // resolution.
    if (!storage) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_TRANSFERS;
    }
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  const bool terminal = chunk.next.index == transfer->count;
  iree_hal_amd_gpu_pm4_barrier_t before =
      first ? submission->before : (iree_hal_amd_gpu_pm4_barrier_t){0};
  // Every preceding publication drains its own CP DMA before retirement.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  if (program) {
    // Publish resident builtin instructions.
    before.acquire_cache_control |= 1u;
  }
  if (argument_length || chunk.staging_length > chunk.download.length) {
    // Host-written ABI and transport bytes need visibility even with empty
    // payload barriers. Their storage and retirement owners are independent.
    before.acquire_cache_control |= queue->options.barriers.acquire;
  }
  iree_hal_amd_gpu_pm4_barrier_t after =
      terminal ? submission->after : (iree_hal_amd_gpu_pm4_barrier_t){0};
  after.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  if (chunk.download.length) {
    after.release_cache_control |= queue->options.barriers.release;
  }
  bool native_signal = false;
  for (iree_host_size_t i = 0;
       terminal && !transfer->host_completion_required &&
       i < submission->signals.count;
       ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words =
      wait_words + chunk.word_count +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      iree_hal_amd_gpu_pm4_barrier_completion_wait_word_count(after,
                                                              native_signal) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_signal_pool_try_acquire(
        &queue->options.signals->pool, 0);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  if (queue->kernel && !program) {
    queue->kernel->rollback.storage = storage;
    queue->kernel->rollback.bytes = storage ? storage->end : 0;
  }
  const uint64_t storage_length = argument_length + chunk.staging_length;
  if (storage_length) {
    storage->end = start + storage_length;
  }
  queue->issued[position & queue->retirement.capacity_mask].consumed_transfers =
      queue->producer.transfers.current ? queue->producer.transfers.current->end
                                        : 0;
  queue->producer.ring.reserved_words = candidate.reserved_words;
  transfer->cursor = chunk.next;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; first && i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  iree_memcpy_write_only(words, chunk.words, chunk.word_count * sizeof(*words));
  words += chunk.word_count;
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      arguments->end, chunk.download, terminal);
}

// One bounded primary sequence invokes the entire resident program. Native
// copy/fixup work scales with program size; host work scales only with its
// dense input table and semaphore edges. No recorded operation is visited here.
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_execute(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  iree_hal_amd_gpu_pm4_command_invocation_t* invocation = &submission->execute;
  iree_hal_amd_gpu_pm4_command_program_t* command = invocation->program;
  iree_hal_amd_gpu_program_t* program = &command->resident;
  iree_hal_amd_gpu_profile_operation_t* profile =
      iree_hal_amd_gpu_pm4_submission_profile(submission);
  const uint64_t sample_length = profile ? (uint64_t)profile->completion.count *
                                               profile->completion.sample_stride
                                         : 0;
  const uint64_t argument_length =
      sizeof(iree_hal_amd_gpu_pm4_program_arguments_t);
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  if (argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
  }
  uint64_t relative = iree_align_uint64(arguments->end - arguments->begin, 8);
  uint64_t offset = relative & (arguments->capacity - 1);
  if (argument_length > arguments->capacity - offset) {
    relative += arguments->capacity - offset;
    offset = 0;
  }
  const uint64_t start = arguments->begin + relative;
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.arguments, iree_memory_order_acquire);
  const uint64_t reusable = completed >= arguments->end
                                ? start
                                : iree_max(completed, arguments->begin);
  if (start - reusable + argument_length > arguments->capacity) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  uint64_t sample_offset = 0, sample_end = 0;
  if (sample_length) {
    const iree_hal_amd_gpu_byte_reservation_t sample =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            queue->samples, sample_length, 8, &sample_offset, &sample_end);
    if (sample != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return sample == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SAMPLES
                 : IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
  }
  if (!invocation->instance) {
    invocation->instance = iree_hal_amd_gpu_program_try_acquire(
        program, queue->options.queue_ordinal,
        queue->producer.frontier.pending);
    if (!invocation->instance) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM;
    }
  }
  iree_hal_amd_gpu_program_instance_t* instance = invocation->instance;
  // Acceptance installs a nonzero epoch only after this instance's initial
  // repairs. Exclusive acquisition has already proved every prior use ordered
  // or retired; fixed addresses need neither rewriting nor a reader join.
  const uint32_t fixup_count = command->invariant_fixups && instance->last_epoch
                                   ? 0
                                   : program->fixup_count;
  const uint32_t fixup_word_count = fixup_count ? command->fixup_word_count : 0;
  uint64_t values_offset = 0, values_end = 0;
  const bool shader_fixup = fixup_count && !fixup_word_count;
  if (shader_fixup) {
    const iree_hal_amd_gpu_byte_reservation_t reservation =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            &queue->values, (uint64_t)command->value_count * 8, 8,
            &values_offset, &values_end);
    if (reservation != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return reservation == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_VALUES
                 : IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
  }
  const uint64_t values_address =
      shader_fixup ? queue->values.current->address + values_offset : 0;
  const uint32_t program_words = iree_hal_amd_gpu_pm4_program_word_count(
      program, instance, &queue->options.barriers, fixup_count,
      fixup_word_count);
  const uint32_t scratch_words = iree_hal_amd_gpu_pm4_scratch_word_count(
      &queue->producer.scratch, command->scratch_wave_byte_length,
      queue->kernel != NULL);
  const uint32_t prefix_words =
      submission->native_wait_count * 7 + scratch_words + program_words - 4;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const iree_hal_amd_gpu_pm4_barrier_t after = {
      .release_cache_control =
          sample_length ? queue->options.barriers.release : 0,
  };
  const uint32_t tail_words =
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      iree_hal_amd_gpu_pm4_barrier_completion_wait_word_count(after,
                                                              native_signal) +
      (native_signal ? 8 : 0) + 14;
  uint32_t root_padding = 0;
  if (queue->kernel) {
    root_padding = (0u - prefix_words - 4) & 7u;
    if (root_padding == 1) {
      root_padding += 8;
    }
  }
  const uint32_t entry_words =
      queue->kernel ? prefix_words + root_padding + 4 : 0;
  const uint32_t body_words =
      queue->kernel ? entry_words + 32 : prefix_words + 4 + tail_words + 8;
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_signal_pool_try_acquire(
        &queue->options.signals->pool, 0);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  const uint64_t dispatch_id = queue->producer.dispatch_count;
  const uint64_t continuation =
      queue->kernel
          ? queue->kernel->address +
                (reservation.words - queue->producer.ring.words + entry_words) *
                    4
          : 0;
  uint8_t* upload = arguments->memory.contents.data + offset;
  const uint64_t sample_address =
      sample_length ? queue->samples->current->address + sample_offset : 0;
  uint32_t fixup_words[IREE_HAL_AMD_GPU_PM4_PACKET_FIXUP_WORD_CAPACITY];
  if (fixup_count) {
    iree_status_t status = iree_hal_amd_gpu_pm4_command_invocation_write(
        invocation, queue->options.queue_ordinal, dispatch_id, continuation,
        sample_address,
        shader_fixup ? (void*)(queue->values.current->memory.contents.data +
                               values_offset)
                     : NULL,
        fixup_words);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
  }
  arguments->end = start + argument_length;
  queue->issued[position & queue->retirement.capacity_mask].consumed_values =
      values_end;
  if (values_end) {
    iree_hal_amd_gpu_byte_storage_pool_commit(&queue->values, values_end,
                                              queue->kernel != NULL);
  }
  if (sample_length) {
    profile->completion.samples =
        (void*)(queue->samples->current->memory.contents.data + sample_offset);
    profile->completion.samples_end = sample_end;
    iree_hal_amd_gpu_byte_storage_pool_commit(queue->samples, sample_end,
                                              queue->kernel != NULL);
  }
  queue->producer.ring.reserved_words = candidate.reserved_words;
  queue->producer.dispatch_count = dispatch_id + command->dispatch_count;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_scratch_emit(words, &queue->producer.scratch,
                                            scratch_words);
  words = iree_hal_amd_gpu_pm4_program_emit(
      program, instance, command->entry, &queue->options.barriers,
      arguments->address + offset, values_address, (void*)upload, fixup_count,
      fixup_word_count, fixup_words, root_padding, words);
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, entry_words, words,
      after, arguments->end, (iree_hal_amd_gpu_download_t){0}, true);
}

iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_dispatch(
    iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t position,
    uint32_t scratch_required,
    const iree_hal_amd_gpu_pm4_dispatch_publication_t* request,
    uint64_t* addresses, iree_hal_amd_gpu_pm4_submission_t** inout_submission,
    iree_status_t* out_status) {
  const iree_hal_amd_gpu_pm4_dispatch_request_t* dispatch_request =
      request->dispatch;
  const iree_hal_amd_gpu_pm4_function_t* function = dispatch_request->function;
  const bool indirect_parameters =
      function &&
      iree_hal_dispatch_uses_indirect_parameters(dispatch_request->flags);
  const bool dispatch =
      function && (indirect_parameters ||
                   (dispatch_request->launch.geometry.workgroup_count[0] &&
                    dispatch_request->launch.geometry.workgroup_count[1] &&
                    dispatch_request->launch.geometry.workgroup_count[2]));
  const bool cooperative = dispatch && (dispatch_request->flags &
                                        IREE_HAL_DISPATCH_FLAG_COOPERATIVE);
  const bool indirect =
      iree_hal_dispatch_uses_indirect_arguments(dispatch_request->flags);
  if (dispatch && function->packet && !queue->packets) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PACKETS;
  }
  const iree_hal_amd_gpu_pm4_dispatch_storage_t argument_storage =
      dispatch ? iree_hal_amd_gpu_pm4_dispatch_storage(function,
                                                       dispatch_request->flags)
               : (iree_hal_amd_gpu_pm4_dispatch_storage_t){0};
  iree_hal_amd_gpu_profile_operation_t* profile = request->profile;
  const uint64_t sample_length = dispatch && profile
                                     ? (uint64_t)profile->completion.count *
                                           profile->completion.sample_stride
                                     : 0;
  uint64_t sample_offset = 0, sample_end = 0;
  if (sample_length) {
    const iree_hal_amd_gpu_byte_reservation_t sample =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            queue->samples, sample_length, 8, &sample_offset, &sample_end);
    if (sample != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return sample == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SAMPLES
                 : IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
  }
  const uint64_t argument_length = argument_storage.length;
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  uint64_t argument_offset = 0;
  uint64_t argument_end = arguments->end;
  uint64_t argument_address = 0;
  if (argument_length) {
    const uint64_t alignment = argument_storage.alignment;
    if (argument_length > arguments->capacity ||
        alignment > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
    }
    uint64_t relative =
        iree_align_uint64(arguments->end - arguments->begin, alignment);
    argument_offset = relative & (arguments->capacity - 1);
    if (argument_offset + argument_length > arguments->capacity) {
      relative += arguments->capacity - argument_offset;
      argument_offset = 0;
    }
    argument_end = arguments->begin + relative + argument_length;
    const uint64_t completed = (uint64_t)iree_atomic_load(
        &queue->completed.arguments, iree_memory_order_acquire);
    const uint64_t reusable = completed >= arguments->end
                                  ? arguments->begin + relative
                                  : iree_max(completed, arguments->begin);
    if (argument_end - reusable > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
    argument_address = arguments->address + argument_offset;
  }
  iree_hal_amd_gpu_pm4_barrier_t before = request->before;
  // Transfer chunks and resident-program exits join their own CP DMA. Direct
  // dispatch setup also joins its copies before launching, so neither boundary
  // of this publication has outstanding CP DMA to drain.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  iree_hal_amd_gpu_pm4_barrier_t after = request->after;
  after.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  if (sample_length) {
    after.release_cache_control |= queue->options.barriers.release;
  }
  if (dispatch) {
    if (cooperative) {
      // Exclusive queue admission prevents competition from other queues;
      // these joins also prevent two resident grids on this queue competing
      // for the resources each was independently admitted to occupy.
      before.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
      after.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
    }
    // Instruction upload and host argument publication remain mandatory even
    // when the caller supplies an empty optional payload boundary.
    before.acquire_cache_control |= 1u | queue->options.barriers.acquire;
    if (function->launch.requirements &
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_TSAN) {
      before.execution |= IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
      before.release_cache_control |= queue->options.barriers.shader;
      before.acquire_cache_control |= queue->options.barriers.shader;
    }
  }
  const uint16_t indirect_copy_count =
      dispatch && indirect_parameters && !indirect
          ? function->base.kernarg->indirect_copy_count
          : 0;
  const bool prepare_packet =
      dispatch && function->packet && indirect_parameters;
  const uint32_t launch_words =
      dispatch ? (prepare_packet ? IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS
                                 : 0u) +
                     (function->launch.user_data_count
                          ? 24 + function->launch.user_data_count
                          : 22) +
                     (indirect_parameters
                          ? iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(
                                indirect_copy_count)
                          : 5)
               : 0;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < request->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(request->signals.semaphores[i]);
  }
  const uint32_t scratch_words = iree_hal_amd_gpu_pm4_scratch_word_count(
      &queue->producer.scratch, scratch_required, queue->kernel != NULL);
  const uint32_t body_words =
      request->native_wait_count * 7 + launch_words + scratch_words +
      (cooperative ? 4 + sizeof(iree_hal_amd_gpu_grid_sync_t) / sizeof(uint32_t)
                   : 0) +
      (sample_length ? 16 : 0) +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      iree_hal_amd_gpu_pm4_barrier_completion_wait_word_count(after,
                                                              native_signal) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  uint64_t parameters_address = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < dispatch_request->bindings.count &&
       iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        dispatch_request->bindings.values[i],
        indirect ? IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS
                 : IREE_HAL_BUFFER_USAGE_STORAGE |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
        indirect ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (indirect_parameters && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_QUEUE,
        dispatch_request->parameters,
        IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        IREE_HAL_MEMORY_ACCESS_READ, &parameters_address);
    if (iree_status_is_ok(status) && (parameters_address & 3)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect parameter address is misaligned");
    }
  }
  if (dispatch && indirect && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (function->base.kernarg->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "indirect kernarg address is misaligned");
    }
  }
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  iree_hal_amd_gpu_signal_t* signal = NULL;
  if (native_signal) {
    signal = iree_hal_amd_gpu_signal_pool_try_acquire(
        &queue->options.signals->pool, 0);
    if (!signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = *inout_submission;
  if (!submission) {
    iree_hal_amd_gpu_pm4_retirement_slot_t* slot =
        &queue->issued[position & queue->retirement.capacity_mask];
    status = iree_hal_amd_gpu_pm4_submission_create_retirement(
        &queue->captures,
        iree_make_byte_span(slot->capture, sizeof(slot->capture)),
        dispatch_request, request->wait_count, request->waits,
        request->native_wait_count, request->signals, request->before,
        request->after, profile, &submission);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_signal_release(signal);
      *out_status = status;
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
    }
    *inout_submission = submission;
    profile = iree_hal_amd_gpu_pm4_submission_profile(submission);
  }
  submission->signal = signal;
  arguments->end = argument_end;
  // Reservation commits only the advancing position; ring geometry is fixed.
  queue->producer.ring.reserved_words = candidate.reserved_words;
  if (argument_storage.length) {
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        function, &dispatch_request->launch, dispatch_request->constants,
        addresses, dispatch_request->flags,
        arguments->memory.contents.data + argument_offset);
  }
  uint64_t packet_address = 0;
  if (dispatch && function->packet) {
    const uint64_t packet_offset =
        (position & queue->retirement.capacity_mask) *
        sizeof(iree_hal_amd_gpu_pm4_dispatch_packet_t);
    packet_address = queue->packets->address + packet_offset;
    iree_hal_amd_gpu_pm4_dispatch_packet_t* packet =
        (void*)(queue->packets->memory.contents.data + packet_offset);
    iree_hal_amd_gpu_pm4_dispatch_emplace_packet(
        function, &dispatch_request->launch, dispatch_request->flags, packet);
    packet->kernarg_address = argument_address;
    if (prepare_packet) {
      uint64_t* inputs =
          (void*)(arguments->memory.contents.data + argument_offset +
                  argument_storage.prepare_offset);
      inputs[0] = parameters_address;
      inputs[1] = packet_address;
    }
  }
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < request->wait_count; ++i) {
    if (request->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, request->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  if (cooperative) {
    // Reuse the record on the device timeline. The existing mandatory acquire
    // follows the confirmed TC/L2 write, making it visible to the next grid.
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
    before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE;
    iree_hal_amd_gpu_grid_sync_t grid;
    iree_hal_amd_gpu_dispatch_emplace_grid_sync(
        function->base.kernarg, &dispatch_request->launch.geometry,
        queue->grid.address, arguments->memory.contents.data + argument_offset,
        &grid);
    words = iree_hal_amd_gpu_pm4_emit_write(words, queue->grid.address,
                                            sizeof(grid) / sizeof(uint32_t),
                                            (const uint32_t*)&grid);
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  words = iree_hal_amd_gpu_pm4_scratch_emit(words, &queue->producer.scratch,
                                            scratch_words);
  if (dispatch) {
    if (prepare_packet) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
          words, function->packet->prepare,
          arguments->address + argument_offset +
              argument_storage.prepare_offset,
          queue->options.barriers.shader);
    }
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
        .dispatch_pointer = packet_address,
        .kernarg_pointer = argument_address,
        .dispatch_id = queue->producer.dispatch_count++,
        .private_segment_size =
            (function->base.metadata->private_segment_fixed_size + 3u) & ~3u};
    words = iree_hal_amd_gpu_pm4_launch_emit(
        words, &function->launch,
        dispatch_request->launch.geometry.workgroup_size,
        dispatch_request->launch.geometry.partial_size,
        dispatch_request->launch.geometry.group_segment_byte_length, &inputs);
    const uint32_t initiator = iree_hal_amd_gpu_pm4_launch_dispatch_initiator(
        &function->launch, dispatch_request->launch.geometry.partial_size);
    const uint64_t sample_address =
        sample_length ? queue->samples->current->address + sample_offset : 0;
    if (sample_length) {
      profile->completion.samples =
          (void*)(queue->samples->current->memory.contents.data +
                  sample_offset);
      profile->completion.samples_end = sample_end;
      iree_hal_amd_gpu_byte_storage_pool_commit(queue->samples, sample_end,
                                                queue->kernel != NULL);
      words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
      words = iree_hal_amd_gpu_pm4_emit_clock(words, sample_address, true);
    }
    if (indirect_parameters) {
      words = iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
          words, function->base.kernarg, indirect_copy_count,
          parameters_address, argument_address,
          queue->options.barriers.command_processor_bypasses_l2
              ? queue->options.barriers.acquire
              : queue->options.barriers.shader,
          initiator);
    } else {
      words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
          words, dispatch_request->launch.geometry.workgroup_count, initiator);
    }
    if (sample_length) {
      words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
      words = iree_hal_amd_gpu_pm4_emit_clock(words, sample_address + 8, true);
    }
  }
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      argument_end, (iree_hal_amd_gpu_download_t){0}, true);
}

iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_control(
    iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position) {
  uint64_t target = 0;
  iree_status_t status = iree_hal_amd_gpu_control_resolve(
      &submission->control.target, submission->control.command, &target);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_abandon(queue, submission, status);
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED;
  }
  const bool program = iree_hal_amd_gpu_control_uses_program(
      queue->base.queue_family, submission->control.target.buffer,
      submission->control.command);
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  uint64_t argument_offset = 0, argument_end = arguments->end;
  if (program) {
    const uint64_t length =
        sizeof(iree_hal_amd_gpu_control_program_arguments_t);
    if (length > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS;
    }
    uint64_t relative = iree_align_uint64(arguments->end - arguments->begin, 8);
    argument_offset = relative & (arguments->capacity - 1);
    if (length > arguments->capacity - argument_offset) {
      relative += arguments->capacity - argument_offset;
      argument_offset = 0;
    }
    const uint64_t start = arguments->begin + relative;
    const uint64_t completed = (uint64_t)iree_atomic_load(
        &queue->completed.arguments, iree_memory_order_acquire);
    const uint64_t reusable = completed >= arguments->end
                                  ? start
                                  : iree_max(completed, arguments->begin);
    if (start - reusable + length > arguments->capacity) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
    }
    argument_end = start + length;
  }
  iree_hal_amd_gpu_pm4_barrier_t before = submission->before;
  iree_hal_amd_gpu_pm4_barrier_t after = submission->after;
  iree_hal_amd_gpu_pm4_barrier_control(
      &queue->options.barriers, program, submission->control.command.flags,
      (submission->control.command.flags & IREE_HAL_ATOMIC_FLAG_RELEASE)
          ? IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE
          : 0,
      &before, &after);
  if (program) {
    before.acquire_cache_control |= queue->options.barriers.shader;
  }
  // Every earlier publication joins its own DMA before leaving the CP.
  before.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  after.execution &= ~IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  bool native_signal = false;
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    native_signal |=
        iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i]);
  }
  const uint32_t body_words =
      submission->native_wait_count * 7 +
      (program
           ? 31
           : iree_hal_amd_gpu_control_word_count(submission->control.command)) +
      iree_hal_amd_gpu_pm4_barrier_word_count(before) +
      iree_hal_amd_gpu_pm4_barrier_tail_word_count(after) +
      iree_hal_amd_gpu_pm4_barrier_completion_wait_word_count(after,
                                                              native_signal) +
      (native_signal ? 8 : 0) + 14 + (queue->kernel ? 0 : 8);
  iree_hal_amd_gpu_pm4_ring_t candidate = queue->producer.ring;
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  if (!iree_hal_amd_gpu_pm4_ring_try_reserve(
          &candidate,
          (uint64_t)iree_atomic_load(&queue->completed.words,
                                     iree_memory_order_acquire),
          body_words, &reservation)) {
    return IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY;
  }
  if (native_signal) {
    submission->signal = iree_hal_amd_gpu_signal_pool_try_acquire(
        &queue->options.signals->pool, 0);
    if (!submission->signal) {
      return IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS;
    }
  }
  queue->producer.ring.reserved_words = candidate.reserved_words;
  uint32_t* words = reservation.words;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      words = iree_hal_amd_gpu_pm4_emit_wait32(
          words, submission->waits[i].signal->device_address, 1, UINT32_MAX,
          IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    }
  }
  words = iree_hal_amd_gpu_pm4_barrier_emit(words, before);
  if (program) {
    *(iree_hal_amd_gpu_control_program_arguments_t*)(arguments->memory.contents
                                                         .data +
                                                     argument_offset) =
        iree_hal_amd_gpu_control_program_arguments(target,
                                                   submission->control.command);
    arguments->end = argument_end;
    words = iree_hal_amd_gpu_pm4_control_program_emit(
        words,
        &queue->options.programs
             ->control[iree_hal_amd_gpu_control_program_index(
                 submission->control.command)],
        arguments->address + argument_offset);
  } else {
    words = iree_hal_amd_gpu_control_emit(words, target,
                                          submission->control.command);
  }
  return iree_hal_amd_gpu_pm4_publish_commit(
      queue, submission, position, &reservation, body_words, 0, words, after,
      argument_end, (iree_hal_amd_gpu_download_t){0}, true);
}
