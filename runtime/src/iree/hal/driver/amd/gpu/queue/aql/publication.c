// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"

#include "iree/base/internal/memory_copy.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"

// Fresh isolated arguments need only the native near-cache acquisition and can
// publish packet by packet. Other routes amortize SYSTEM acquisition across
// eight prepared packets. Both retain a private tail for completion and
// release.
static uint32_t iree_hal_amd_gpu_aql_publication_group_mask(
    const iree_hal_amd_gpu_aql_queue_t* queue) {
  return queue->producer.argument_cache_isolated ? 0 : 7;
}

static amdf_status_t iree_hal_amd_gpu_aql_write_packet(
    iree_hal_amd_gpu_aql_ring_t* ring,
    const iree_hal_amd_gpu_user_queue_t* storage, uint64_t position,
    iree_hal_amd_gpu_aql_packet_t packet) {
  iree_hal_amd_gpu_aql_ring_write_body(ring, position, &packet);
  return iree_hal_amd_gpu_aql_ring_publish_header(ring, storage, position,
                                                  packet.words[0]);
}

// Accepted ownership is already visible to retirement even when publication
// fails. Wake the existing terminal path; no failed release proves completion.
static void iree_hal_amd_gpu_aql_publish_notify(
    iree_hal_amd_gpu_aql_queue_t* queue, amdf_status_t status,
    uint64_t published) {
  if (IREE_UNLIKELY(!amdf_status_is_ok(status))) {
    iree_hal_amd_gpu_aql_queue_fail_native(&queue->base, status,
                                           "flush AQL command publication");
  } else {
    iree_hal_amd_gpu_aql_ring_notify(&queue->producer.ring, published);
  }
}

amdf_status_t iree_hal_amd_gpu_aql_publish_run(
    iree_hal_amd_gpu_aql_queue_t* queue) {
  iree_hal_amd_gpu_signal_t* completion = queue->run.completion;
  if (!completion) {
    return AMDF_STATUS_OK;
  }
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  uint32_t* words = ring->packets[queue->run.last & (ring->capacity - 1)].words;
  iree_memcpy_write_only(words + 14, &completion->device_address,
                         sizeof(completion->device_address));
  queue->run.completion = NULL;
  iree_atomic_store(&queue->run.position, 0, iree_memory_order_release);
  const amdf_status_t status = iree_hal_amd_gpu_aql_ring_publish_header(
      ring, &queue->storage,
      queue->run.first +
          ((queue->run.last - queue->run.first) &
           ~(uint64_t)iree_hal_amd_gpu_aql_publication_group_mask(queue)),
      queue->run.group_header);
  iree_hal_amd_gpu_aql_publish_notify(queue, status, queue->run.last);
  iree_hal_amd_gpu_signal_release(completion);
  return status;
}

uint64_t iree_hal_amd_gpu_aql_argument_length(
    const iree_hal_amd_gpu_aql_dispatch_request_t* request) {
  if (!request->function) {
    return 0;
  }
  const uint64_t length =
      iree_hal_dispatch_uses_indirect_arguments(request->flags)
          ? 0
          : request->function->base.kernarg->byte_length;
  const uint64_t arguments =
      iree_hal_dispatch_uses_indirect_parameters(request->flags)
          ? iree_align_uint64(length, 8) +
                sizeof(iree_hal_amd_gpu_aql_indirect_arguments_t)
          : length;
  return iree_align_uint64(arguments, 4);
}

// A reservation calculation changes no ring state. Completion prefixes include
// wrap/alignment padding; successful publication alone commits the end.
static iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_reserve_bytes(
    iree_hal_amd_gpu_byte_storage_t* storage, iree_atomic_int64_t* consumed,
    uint64_t length, uint64_t alignment,
    iree_hal_amd_gpu_aql_publish_result_t growth, uint64_t* out_offset,
    uint64_t* out_end) {
  *out_offset = 0;
  *out_end = storage->end;
  if (!length) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  if (length > storage->capacity || alignment > storage->capacity) {
    return growth;
  }
  uint64_t relative =
      iree_align_uint64(storage->end - storage->begin, alignment);
  uint64_t offset = relative & (storage->capacity - 1);
  if (offset + length > storage->capacity) {
    relative += storage->capacity - offset;
    offset = 0;
  }
  const uint64_t end = storage->begin + relative + length;
  const uint64_t completed =
      (uint64_t)iree_atomic_load(consumed, iree_memory_order_acquire);
  const uint64_t reusable = completed >= storage->end
                                ? storage->begin + relative
                                : iree_max(completed, storage->begin);
  if (end - reusable > storage->capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  *out_offset = offset;
  *out_end = end;
  return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
}

static iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_arguments(
    iree_hal_amd_gpu_aql_queue_t* queue, uint64_t length, uint64_t alignment,
    uint64_t* out_offset, uint64_t* out_end) {
  return iree_hal_amd_gpu_aql_reserve_bytes(
      queue->producer.arguments, &queue->completed.arguments, length,
      iree_max(alignment, (uint64_t)IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS, out_offset, out_end);
}

// Records readiness and prepares native ownership before executable packet
// headers become valid. The retirement record remains private to the publisher
// until publish_retirement. All fallible preparation and native reservation
// have already succeeded.
static void iree_hal_amd_gpu_aql_accept(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission,
    iree_hal_amd_gpu_signal_t* signal, uint64_t position, uint64_t argument_end,
    bool terminal, bool native_signals) {
  if (IREE_UNLIKELY(submission->profiled)) {
    iree_hal_amd_gpu_profile_operation_record(
        queue->profile, iree_hal_amd_gpu_aql_submission_profile(submission),
        true, (submission->native_wait_count + 4) / 5);
  }
  queue->producer.frontier[queue->options.queue_ordinal] = position + 1;
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ALLOCATION) {
    iree_async_single_frontier_t death_frontier;
    iree_async_single_frontier_initialize(&death_frontier, queue->frontier.axis,
                                          position + 1);
    iree_hal_amd_gpu_allocation_accept(
        submission->allocation.operation,
        queue->frontier.tracker
            ? iree_async_single_frontier_as_const_frontier(&death_frontier)
            : NULL);
  }
  iree_hal_amd_gpu_signal_publish_frontier(signal, queue->producer.frontier);
  iree_hal_amd_gpu_aql_retirement_t* action =
      &queue->issued[position & (queue->options.retirement_capacity - 1)];
  action->submission = submission;
  action->signal = signal;
  action->arguments = argument_end;
  action->terminal = terminal;
  if (!terminal) {
    iree_atomic_ref_count_inc(submission->entry.kind ==
                                      IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE
                                  ? &submission->execute.ref_count
                                  : &submission->transfer.ref_count);
  } else {
    for (iree_host_size_t i = 0;
         native_signals && i < submission->signals.count; ++i) {
      if (iree_hal_amd_gpu_semaphore_isa(submission->signals.semaphores[i])) {
        iree_status_t status = iree_hal_amd_gpu_semaphore_publish_signal(
            submission->signals.semaphores[i],
            submission->signals.payload_values[i], signal);
        if (!iree_status_is_ok(status)) {
          iree_hal_amd_gpu_aql_queue_fail_submission(&queue->base, submission,
                                                     status);
        }
      }
    }
    if (submission->has_host_owner) {
      submission->has_host_owner = false;
      iree_atomic_fetch_sub(&queue->pending.host_owners, 1,
                            iree_memory_order_release);
    }
  }
}

// Transfers the publisher's capture access to retirement. All capture reads
// precede this release; the first completion-capable packet header follows it
// so native completion cannot outrun the host handoff.
static void iree_hal_amd_gpu_aql_publish_retirement(
    iree_hal_amd_gpu_aql_queue_t* queue, uint64_t position) {
  iree_atomic_store(&queue->producer.count, position + 1,
                    iree_memory_order_release);
}

static amdf_status_t iree_hal_amd_gpu_aql_emit_waits(
    iree_hal_amd_gpu_aql_ring_t* ring,
    const iree_hal_amd_gpu_user_queue_t* storage, iree_host_size_t wait_count,
    const iree_hal_amd_gpu_aql_wait_t* waits, uint64_t* inout_position) {
  uint64_t position = *inout_position;
  uint64_t dependencies[5];
  uint32_t count = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  for (iree_host_size_t i = 0; i < wait_count && amdf_status_is_ok(status);
       ++i) {
    dependencies[count++] = waits[i].signal->device_address;
    if (count == 5 || (i + 1 == wait_count && count)) {
      status = iree_hal_amd_gpu_aql_write_packet(
          ring, storage, position++,
          iree_hal_amd_gpu_aql_packet_barrier(
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
              count, dependencies, 0));
      count = 0;
    }
  }
  *inout_position = position;
  return status;
}

iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish(
    iree_hal_amd_gpu_aql_queue_t* queue,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, iree_hal_amd_gpu_aql_wait_t* waits,
    iree_hal_amd_gpu_aql_wait_counts_t counts,
    iree_hal_semaphore_list_t signals, uint32_t before, uint32_t after,
    uint64_t* addresses, const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_aql_submission_t** inout_submission,
    iree_status_t* out_status) {
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  const uint64_t completed = (uint64_t)iree_atomic_load(
      &queue->completed.count, iree_memory_order_acquire);
  if (position - completed == queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  const bool indirect =
      iree_hal_dispatch_uses_indirect_parameters(request->flags);
  const bool dispatch =
      request->function && (indirect || (request->geometry.workgroup_count[0] &&
                                         request->geometry.workgroup_count[1] &&
                                         request->geometry.workgroup_count[2]));
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(request->flags);
  const iree_hal_amd_gpu_kernarg_layout_t* layout =
      dispatch ? request->function->base.kernarg : NULL;
  const uint64_t kernarg_length =
      dispatch && !indirect_arguments ? layout->byte_length : 0;
  const iree_hal_amd_gpu_profile_operation_t* operation =
      *inout_submission
          ? iree_hal_amd_gpu_aql_submission_profile(*inout_submission)
          : profile;
  const bool sampled =
      dispatch && operation && operation->completion.sample_stride;
  const bool cooperative =
      dispatch && (request->flags & IREE_HAL_DISPATCH_FLAG_COOPERATIVE);
  if (cooperative && !queue->commands) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_COMMANDS;
  }
  const uint64_t argument_length =
      dispatch ? iree_hal_amd_gpu_aql_argument_length(request) : 0;
  const uint64_t argument_alignment =
      dispatch ? iree_max(layout->alignment, indirect ? 8u : 4u) : 4;
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  uint64_t argument_offset = 0, argument_end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t argument_result =
      iree_hal_amd_gpu_aql_arguments(queue, argument_length, argument_alignment,
                                     &argument_offset, &argument_end);
  if (argument_result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return argument_result;
  }
  uint64_t sample_offset = 0, sample_end = 0;
  if (sampled) {
    const iree_hal_amd_gpu_byte_reservation_t reservation =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            queue->samples, sizeof(iree_hal_amd_gpu_aql_signal_t), 64,
            &sample_offset, &sample_end);
    if (reservation != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return reservation == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_AQL_PUBLISH_SAMPLES
                 : IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  uint64_t argument_address =
      argument_length ? arguments->address + argument_offset : 0;
  const bool prefix = dispatch && (before & (3u << 11));
  const bool suffix = !dispatch || sampled || (after & (3u << 9));
  const bool run_eligible = dispatch && !indirect && !indirect_arguments &&
                            !cooperative && !prefix && !suffix &&
                            !counts.native_count && !operation;
  const bool run_release =
      queue->run.completion && !request->function && !operation &&
      !counts.native_count && !((before | after) & (3u << 9)) &&
      ((before | after) & (3u << 11)) &&
      (!*inout_submission || (*inout_submission)->entry.kind ==
                                 IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION);
  if (!run_eligible && !run_release) {
    const amdf_status_t seal_status = iree_hal_amd_gpu_aql_publish_run(queue);
    if (!amdf_status_is_ok(seal_status)) {
      *out_status = IREE_HAL_AMD_STATUS_FROM_AMDF(seal_status,
                                                  "publish preceding AQL run");
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
  }
  const uint32_t packet_count = run_release ? 0
                                            : (counts.native_count + 4) / 5 +
                                                  prefix + indirect + dispatch +
                                                  suffix + cooperative;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       dispatch && i < request->bindings.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        request->bindings.values[i],
        indirect_arguments ? IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS
                           : IREE_HAL_BUFFER_USAGE_STORAGE |
                                 IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
        indirect_arguments ? IREE_HAL_MEMORY_ACCESS_READ : 0, &addresses[i]);
  }
  if (dispatch && indirect_arguments && iree_status_is_ok(status)) {
    argument_address = addresses[0];
    if (argument_address & (layout->alignment - 1)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AQL argument address is misaligned");
    }
  }
  uint64_t counts_address = 0;
  if (indirect && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_dispatch_resolve_address(
        queue->base.queue_family, IREE_HAL_MEMORY_SITE_PROGRAM,
        request->parameters, IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
        IREE_HAL_MEMORY_ACCESS_READ, &counts_address);
  }
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  iree_hal_amd_gpu_signal_t* signal =
      (run_eligible || run_release) && queue->run.completion
          ? iree_hal_amd_gpu_signal_pool_try_acquire_alias(
                &queue->options.signals->pool, queue->run.completion)
          : iree_hal_amd_gpu_signal_pool_try_acquire(
                &queue->options.signals->pool, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  iree_hal_amd_gpu_aql_retirement_t* action =
      &queue->issued[position & (queue->options.retirement_capacity - 1)];
  iree_hal_amd_gpu_aql_submission_t* submission = *inout_submission;
  if (!submission) {
    status = iree_hal_amd_gpu_aql_submission_retirement(
        &queue->captures,
        iree_make_byte_span(action->capture, sizeof(action->capture)), profile,
        request, wait_count, waits, counts, signals, &submission);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_signal_release(signal);
      *out_status = status;
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
    *inout_submission = submission;
  }
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    if (waits[i].signal) {
      iree_hal_amd_gpu_signal_join_frontier(waits[i].signal,
                                            queue->producer.queue_count,
                                            queue->producer.frontier);
    }
  }
  if (run_release) {
    // The final group's INVALID header still guards the last dispatch. Its
    // ordered release covers the already published groups without another
    // packet. Preserve independent logical ownership on a completion alias.
    const uint32_t release = iree_max(before & (3u << 11), after & (3u << 11));
    const uint32_t header =
        (queue->run.last_header & ~(3u << 11)) |
        iree_max(queue->run.last_header & (3u << 11), release);
    queue->run.last_header = header;
    if (((queue->run.last - queue->run.first) &
         iree_hal_amd_gpu_aql_publication_group_mask(queue)) == 0) {
      queue->run.group_header = header;
    } else {
      iree_memcpy_write_only(
          ring->packets[queue->run.last & (ring->capacity - 1)].words, &header,
          sizeof(header));
    }
    iree_hal_amd_gpu_aql_accept(queue, submission, signal, position,
                                argument_end, true, true);
    iree_hal_amd_gpu_aql_publish_retirement(queue, position);
    iree_hal_amd_gpu_aql_publish_run(queue);
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  const uint64_t grid_address = cooperative ? queue->grid.address : 0;
  if (kernarg_length) {
    if (queue->producer.argument_streaming) {
      iree_hal_amd_gpu_dispatch_emplace_arguments_stream(
          &request->function->base, &request->geometry, request->constants,
          addresses, request->flags, grid_address,
          arguments->memory.contents.data + argument_offset);
    } else {
      iree_hal_amd_gpu_dispatch_emplace_arguments(
          &request->function->base, &request->geometry, request->constants,
          addresses, request->flags,
          arguments->memory.contents.data + argument_offset);
    }
  }
  iree_hal_amd_gpu_aql_packet_t grid_packet;
  if (cooperative) {
    iree_hal_amd_gpu_grid_sync_t grid =
        iree_hal_amd_gpu_dispatch_grid_sync(&request->geometry);
    if (!queue->producer.argument_streaming) {
      iree_hal_amd_gpu_dispatch_emplace_grid_sync(
          layout, &request->geometry, grid_address,
          arguments->memory.contents.data + argument_offset, &grid);
    }
    const uint64_t command_offset =
        (position & (queue->options.retirement_capacity - 1)) *
        IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
    grid_packet = iree_hal_amd_gpu_aql_grid_sync_write_packet(
        queue->context->target_info.topology.xcc_count, queue->grid.address,
        &grid, queue->commands->address + command_offset,
        (void*)(queue->commands->memory.contents.data + command_offset));
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    action->transfer.end =
        queue->producer.transfers ? queue->producer.transfers->end : 0;
    action->transfer.download = (iree_hal_amd_gpu_download_t){0};
  }
  uint64_t sample_address = 0;
  if (sampled) {
    iree_hal_amd_gpu_byte_storage_t* samples = queue->samples->current;
    iree_hal_amd_gpu_aql_signal_t* sample =
        (void*)(samples->memory.contents.data + sample_offset);
    *sample = (iree_hal_amd_gpu_aql_signal_t){.kind = 1, .value = 1};
    iree_hal_amd_gpu_profile_completion_t* completion =
        &iree_hal_amd_gpu_aql_submission_profile(submission)->completion;
    completion->samples = (const void*)&sample->start_timestamp;
    completion->samples_end = sample_end;
    sample_address = samples->address + sample_offset;
    iree_hal_amd_gpu_byte_storage_pool_commit(queue->samples, sample_end,
                                              false);
  }
  const bool grouped =
      run_eligible && (queue->run.completion || position != completed);
  const bool first_in_run = grouped && !queue->run.completion;
  const uint32_t group_mask =
      iree_hal_amd_gpu_aql_publication_group_mask(queue);
  const bool first_in_group =
      first_in_run || ((ring->reserved - queue->run.first) & group_mask) == 0;
  if (grouped) {
    if (first_in_run) {
      queue->run.completion = signal;
      iree_hal_amd_gpu_signal_retain(signal);
    }
  }
  // Prepare the terminal packet before a retirement slot can be consumed.
  // A ready request may borrow the same capture which native retirement frees.
  // The first record of every backing/traversal invalidates old argument
  // contents. Later isolated records have no overlapping consumer until this
  // dispatch. Native AGENT acquisition clears near caches; explicit payload
  // scopes remain stronger when requested. Other input producers acquire their
  // entire prepared group at SYSTEM scope before its first consumer.
  const uint32_t acquire =
      queue->producer.argument_cache_isolated && run_eligible
          ? iree_max(before & (3u << 9),
                     argument_length && argument_offset == 0
                         ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM
                         : IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT)
      : grouped && !first_in_group ? before & (3u << 9)
                                   : IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
  const uint32_t dispatch_header =
      IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | acquire | (after & (3u << 11));
  iree_hal_amd_gpu_aql_packet_t payload =
      dispatch ? iree_hal_amd_gpu_aql_dispatch_packet(
                     request->function, &request->geometry, dispatch_header,
                     argument_address,
                     sampled             ? sample_address
                     : suffix || grouped ? 0
                                         : signal->device_address,
                     request->flags)
               : iree_hal_amd_gpu_aql_packet_barrier(
                     IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                         IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                         iree_hal_amd_gpu_aql_barrier_join(before, after),
                     0, NULL, signal->device_address);
  iree_hal_amd_gpu_aql_packet_t patch = {0};
  if (indirect) {
    const uint64_t patch_offset =
        argument_offset + iree_align_uint64(kernarg_length, 8);
    const uint64_t target_position = ring->reserved +
                                     (counts.native_count + 4) / 5 + prefix +
                                     cooperative + 1;
    iree_hal_amd_gpu_aql_indirect_arguments_t indirect_inputs;
    iree_hal_amd_gpu_aql_indirect_prepare(
        layout, &request->geometry, request->flags,
        iree_hal_amd_gpu_aql_ring_packet_address(ring, target_position),
        counts_address, argument_address, payload.words[0], &indirect_inputs);
    if (queue->producer.argument_streaming) {
      iree_memcpy_stream_dst(arguments->memory.contents.data + patch_offset,
                             &indirect_inputs, sizeof(indirect_inputs));
    } else {
      memcpy(arguments->memory.contents.data + patch_offset, &indirect_inputs,
             sizeof(indirect_inputs));
    }
    patch = iree_hal_amd_gpu_aql_indirect_packet(
        queue->options.programs->indirect, arguments->address + patch_offset);
  }
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, packet_count, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  arguments->end = argument_end;
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position, argument_end,
                              true, true);
  amdf_status_t publication_status = iree_hal_amd_gpu_aql_emit_waits(
      ring, &queue->storage, counts.native_count, waits, &packet_position);
  const uint64_t signal_address = signal->device_address;
  iree_hal_amd_gpu_aql_publish_retirement(queue, position);
  if (grouped) {
    if (first_in_run) {
      queue->run.first = packet_position;
      queue->run.group_header = payload.words[0];
    }
    iree_hal_amd_gpu_aql_ring_write_body(ring, packet_position, &payload);
    if (!first_in_run) {
      if (first_in_group) {
        const amdf_status_t status = iree_hal_amd_gpu_aql_ring_publish_header(
            ring, &queue->storage, queue->run.last - group_mask,
            queue->run.group_header);
        iree_hal_amd_gpu_aql_publish_notify(queue, status, queue->run.last);
        queue->run.group_header = payload.words[0];
      } else {
        iree_memcpy_write_only(
            ring->packets[packet_position & (ring->capacity - 1)].words,
            payload.words, sizeof(uint32_t));
      }
    } else {
      iree_atomic_store(&queue->run.position, position + 1,
                        iree_memory_order_release);
    }
    queue->run.last = packet_position;
    queue->run.last_header = payload.words[0];
    if (queue->run.last - queue->run.first + 1 == 64) {
      iree_hal_amd_gpu_aql_publish_run(queue);
    } else if (first_in_run) {
      // A predecessor can retire between admission and this handoff. This
      // wake belongs to the new run and does not depend on another API call.
      const amdf_status_t wake_status =
          iree_hal_amd_gpu_context_signal(queue->context);
      if (!amdf_status_is_ok(wake_status)) {
        iree_hal_amd_gpu_aql_queue_fail_native(&queue->base, wake_status,
                                               "wake AQL run publication");
      }
    }
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  if (prefix && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | (before & (3u << 11)) |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
            0, NULL, 0));
  }
  if (cooperative && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++, grid_packet);
  }
  if (amdf_status_is_ok(publication_status)) {
    if (indirect) {
      // The producer may run as soon as its header becomes valid, before the
      // doorbell. Complete the target body first; only the GPU publishes it.
      iree_hal_amd_gpu_aql_ring_write_body(ring, packet_position + 1, &payload);
      publication_status = iree_hal_amd_gpu_aql_write_packet(
          ring, &queue->storage, packet_position, patch);
      packet_position += 2;
    } else {
      publication_status = iree_hal_amd_gpu_aql_write_packet(
          ring, &queue->storage, packet_position++, payload);
    }
  }
  if (dispatch && suffix && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND | (after & (3u << 9)) |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
            0, NULL, signal_address));
  }
  // No capture/signal/request access follows the terminal header publication.
  iree_hal_amd_gpu_aql_publish_notify(queue, publication_status,
                                      ring->reserved - 1);
  return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
}

iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_control(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  const amdf_status_t seal_status = iree_hal_amd_gpu_aql_publish_run(queue);
  if (!amdf_status_is_ok(seal_status)) {
    *out_status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(seal_status, "publish preceding AQL run");
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  uint64_t target = 0;
  iree_status_t status = iree_hal_amd_gpu_control_resolve(
      &submission->control.target, submission->control.command, &target);
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  const bool program = iree_hal_amd_gpu_control_uses_program(
      queue->base.queue_family, submission->control.target.buffer,
      submission->control.command);
  if (!program && !queue->commands) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_COMMANDS;
  }
  const uint32_t argument_length =
      program ? sizeof(iree_hal_amd_gpu_control_program_arguments_t) : 0;
  uint64_t offset = 0, end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t argument_result =
      iree_hal_amd_gpu_aql_arguments(queue, argument_length, 8, &offset, &end);
  if (argument_result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return argument_result;
  }
  const bool prefix = (submission->before & (3u << 11)) != 0;
  const bool suffix = (submission->after & (3u << 9)) != 0;
  const uint32_t packet_count =
      (submission->native_wait_count + 4) / 5 + prefix + 1 + suffix;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  iree_hal_amd_gpu_signal_t* signal = iree_hal_amd_gpu_signal_pool_try_acquire(
      &queue->options.signals->pool, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  const uint32_t header = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                          IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                          (submission->after & (3u << 11));
  const uint64_t completion = suffix ? 0 : signal->device_address;
  iree_hal_amd_gpu_aql_packet_t packet;
  if (program) {
    const iree_hal_amd_gpu_control_program_arguments_t inputs =
        iree_hal_amd_gpu_control_program_arguments(target,
                                                   submission->control.command);
    if (queue->producer.argument_streaming) {
      iree_memcpy_stream_dst(arguments->memory.contents.data + offset, &inputs,
                             sizeof(inputs));
    } else {
      memcpy(arguments->memory.contents.data + offset, &inputs, sizeof(inputs));
    }
    packet = iree_hal_amd_gpu_aql_control_program_packet(
        queue->options.programs->control[iree_hal_amd_gpu_control_program_index(
            submission->control.command)],
        arguments->address + offset, header, completion);
  } else {
    const uint64_t command_offset =
        (position & (queue->options.retirement_capacity - 1)) *
        IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
    uint32_t* words =
        (void*)(queue->commands->memory.contents.data + command_offset);
    uint32_t* cursor = iree_hal_amd_gpu_aql_emit_single_executor(
        words, queue->context->target_info.topology.xcc_count,
        iree_hal_amd_gpu_control_word_count(submission->control.command));
    cursor = iree_hal_amd_gpu_control_emit(cursor, target,
                                           submission->control.command);
    packet = iree_hal_amd_gpu_aql_packet_indirect_buffer(
        header, queue->commands->address + command_offset,
        (uint32_t)(cursor - words), completion);
  }
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, packet_count, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  arguments->end = end;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      iree_hal_amd_gpu_signal_join_frontier(submission->waits[i].signal,
                                            queue->producer.queue_count,
                                            queue->producer.frontier);
    }
  }
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position, end, true,
                              true);
  amdf_status_t publication_status = iree_hal_amd_gpu_aql_emit_waits(
      ring, &queue->storage, submission->native_wait_count, submission->waits,
      &packet_position);
  if (prefix && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                (submission->before & (3u << 11)),
            0, NULL, 0));
  }
  const uint32_t after = submission->after;
  const uint64_t signal_address = signal->device_address;
  // The native signal joins all XCCs, even when only XCC0 performs the scalar
  // operation. Its host owner must be published before this packet can run.
  iree_hal_amd_gpu_aql_publish_retirement(queue, position);
  if (amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++, packet);
  }
  if (suffix && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (after & (3u << 9)),
            0, NULL, signal_address));
  }
  iree_hal_amd_gpu_aql_publish_notify(queue, publication_status,
                                      ring->reserved - 1);
  return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
}

uint64_t iree_hal_amd_gpu_aql_execute_argument_length(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation) {
  const iree_hal_amd_gpu_aql_host_template_t* host_template =
      invocation->program->host_template;
  if (host_template) {
    if (!host_template->fixup_count) {
      return 0;
    }
    const iree_hal_amd_gpu_aql_program_block_t block =
        invocation->program->blocks[invocation->block_ordinal];
    return iree_align_uint64(
        (block.data.offset & (host_template->alignment - 1)) +
            block.data.length,
        4);
  }
  uint64_t length = invocation->block_ordinal
                        ? sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t)
                        : sizeof(iree_hal_amd_gpu_aql_program_arguments_t);
  if (invocation->program->profile) {
    length += sizeof(iree_hal_amd_gpu_aql_materialize_profile_arguments_t) -
              sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t);
  }
  return length;
}

uint64_t iree_hal_amd_gpu_aql_execute_argument_alignment(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation) {
  const iree_hal_amd_gpu_aql_host_template_t* host_template =
      invocation->program->host_template;
  return host_template ? iree_max(host_template->alignment, 8u) : 8;
}

// Publish one dispatch-only block. A dominating final dispatch can carry the
// block's completion; its private CPU copy preserves the sealed template.
static IREE_ATTRIBUTE_NOINLINE amdf_status_t
iree_hal_amd_gpu_aql_publish_host_template(
    iree_hal_amd_gpu_aql_queue_t* queue,
    const iree_hal_amd_gpu_aql_command_program_t* program,
    iree_hal_amd_gpu_aql_program_block_t block, uint64_t packet_position,
    const void* upload, uint64_t completion_address, uint64_t sample_address,
    iree_hal_amd_gpu_aql_signal_t* samples) {
  const iree_hal_amd_gpu_aql_packet_t* packets = upload;
  if (completion_address) {
    amdf_status_t status = AMDF_STATUS_OK;
    if (block.packet_count > 1) {
      status = iree_hal_amd_gpu_aql_ring_write_packets(
          &queue->producer.ring, &queue->storage, packet_position,
          block.packet_count - 1, packets);
    }
    if (amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_aql_packet_t last = packets[block.packet_count - 1];
      last.words[14] = (uint32_t)completion_address;
      last.words[15] = (uint32_t)(completion_address >> 32);
      status = iree_hal_amd_gpu_aql_write_packet(
          &queue->producer.ring, &queue->storage,
          packet_position + block.packet_count - 1, last);
    }
    return status;
  }
  if (program->profile) {
    if (samples) {
      for (uint32_t i = 0; i < block.packet_count; ++i) {
        samples[i] = (iree_hal_amd_gpu_aql_signal_t){.kind = 1, .value = 1};
      }
    }
    return iree_hal_amd_gpu_aql_ring_write_packets_with_completion(
        &queue->producer.ring, &queue->storage, packet_position,
        block.packet_count, packets, sample_address, sizeof(*samples));
  }
  return iree_hal_amd_gpu_aql_ring_write_packets(
      &queue->producer.ring, &queue->storage, packet_position,
      block.packet_count, packets);
}

static IREE_ATTRIBUTE_NOINLINE iree_hal_amd_gpu_aql_publish_result_t
iree_hal_amd_gpu_aql_publish_execute_prepared(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, void* host_contents,
    iree_status_t* out_status) {
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  iree_hal_amd_gpu_aql_command_invocation_t* invocation = &submission->execute;
  iree_hal_amd_gpu_program_t* program = &invocation->program->resident;
  const bool first = !invocation->block_ordinal;
  const bool terminal =
      invocation->block_ordinal + 1 == invocation->program->block_count;
  const iree_hal_amd_gpu_profile_operation_t* operation =
      iree_hal_amd_gpu_aql_submission_profile(submission);
  const bool sampled = operation && operation->completion.sample_stride;
  const iree_hal_amd_gpu_aql_host_template_t* host_template =
      invocation->program->host_template;
  const bool command_data =
      host_template &&
      host_template->command_data_offset < host_template->byte_length;
  if (command_data && !queue->commands) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_COMMANDS;
  }
  const uint64_t argument_length =
      iree_hal_amd_gpu_aql_execute_argument_length(invocation);
  uint64_t argument_offset = 0, argument_end = 0;
  const iree_hal_amd_gpu_aql_publish_result_t argument_result =
      iree_hal_amd_gpu_aql_arguments(
          queue, argument_length,
          iree_hal_amd_gpu_aql_execute_argument_alignment(invocation),
          &argument_offset, &argument_end);
  if (argument_result != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return argument_result;
  }
  const iree_hal_amd_gpu_aql_program_block_t block =
      invocation->program->blocks[invocation->block_ordinal];
  uint64_t sample_offset = 0, sample_end = 0;
  if (sampled && block.dispatches.count) {
    const iree_hal_amd_gpu_byte_reservation_t reservation =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            queue->samples,
            (uint64_t)block.dispatches.count *
                sizeof(iree_hal_amd_gpu_aql_signal_t),
            64, &sample_offset, &sample_end);
    if (reservation != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return reservation == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_AQL_PUBLISH_SAMPLES
                 : IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  // The accepted prefix already includes same-queue dependencies. Cross-queue
  // proofs join a private candidate until preparation succeeds, so rejection
  // cannot advance the queue's causal prefix.
  uint64_t* frontier = queue->producer.frontier;
  if (first && submission->native_wait_count) {
    frontier += queue->producer.queue_count;
    memcpy(frontier, queue->producer.frontier,
           queue->producer.queue_count * sizeof(uint64_t));
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(
            submission->waits[i].signal, queue->producer.queue_count, frontier);
      }
    }
  }
  if (!host_template && !invocation->instance) {
    invocation->instance = iree_hal_amd_gpu_program_try_acquire(
        program, queue->options.queue_ordinal, frontier);
    if (!invocation->instance) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM;
    }
  }
  const uint32_t waits = first ? (submission->native_wait_count + 4) / 5 : 0;
  const iree_hal_amd_gpu_aql_program_flags_t program_flags =
      (host_template ? 0
                     : iree_hal_amd_gpu_aql_program_prepare(
                           program, invocation->instance,
                           invocation->program->value_count, first)) |
      (invocation->program->has_control ? IREE_HAL_AMD_GPU_AQL_PROGRAM_CONTROL
                                        : 0) |
      (invocation->program->profile ? IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE : 0);
  uint64_t values_offset = 0, values_end = 0;
  const bool fixup = (program_flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_FIXUP) != 0;
  if (fixup) {
    const iree_hal_amd_gpu_byte_reservation_t reservation =
        iree_hal_amd_gpu_byte_storage_pool_reserve(
            &queue->values, (uint64_t)invocation->program->value_count * 8, 8,
            &values_offset, &values_end);
    if (reservation != IREE_HAL_AMD_GPU_BYTE_RESERVED) {
      return reservation == IREE_HAL_AMD_GPU_BYTE_GROW
                 ? IREE_HAL_AMD_GPU_AQL_PUBLISH_VALUES
                 : IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  const uint64_t values_address =
      fixup ? queue->values.current->address + values_offset : 0;
  const uint32_t prefix =
      host_template ? 0
                    : iree_hal_amd_gpu_aql_program_prefix_count(program_flags);
  const iree_hal_amd_gpu_aql_packet_t* host_packets =
      host_template
          ? (const void*)((const uint8_t*)(host_template + 1) + block.offset)
          : NULL;
  // An ordered final dispatch joins the complete pure block. Profiling owns
  // dispatch completion operands independently and therefore keeps its join.
  const bool dispatch_completion =
      host_packets && !invocation->program->profile &&
      block.packet_count == block.dispatches.count &&
      (host_packets[block.packet_count - 1].words[0] &
       IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED);
  const uint32_t packet_count =
      waits + prefix + block.packet_count + !dispatch_completion;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  void* upload = arguments->memory.contents.data + argument_offset;
  const void* packets =
      host_contents
          ? host_contents
          : (host_template ? (const uint8_t*)(host_template + 1) + block.offset
                           : NULL);
  const uint64_t command_offset =
      (position & (queue->options.retirement_capacity - 1)) *
      IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
  if ((first || host_template) && (argument_length || host_contents)) {
    *out_status = iree_hal_amd_gpu_aql_command_invocation_write(
        invocation, queue->options.queue_ordinal, queue->grid.address,
        arguments->address + argument_offset, upload,
        queue->producer.argument_streaming,
        command_data ? queue->commands->address + command_offset : 0,
        command_data ? queue->commands->memory.contents.data + command_offset
                     : NULL,
        host_contents,
        fixup ? (void*)(queue->values.current->memory.contents.data +
                        values_offset)
              : NULL);
    if (!iree_status_is_ok(*out_status)) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
  }
  iree_hal_amd_gpu_signal_t* signal = iree_hal_amd_gpu_signal_pool_try_acquire(
      &queue->options.signals->pool, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, packet_count, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  if (frontier != queue->producer.frontier) {
    memcpy(queue->producer.frontier, frontier,
           queue->producer.queue_count * sizeof(uint64_t));
  }
  arguments->end = argument_end;
  queue->issued[position & (queue->options.retirement_capacity - 1)]
      .execute.values_end = values_end;
  if (values_end) {
    iree_hal_amd_gpu_byte_storage_pool_commit(&queue->values, values_end,
                                              false);
  }
  if (terminal && !host_template) {
    // Install the reusable writer coordinate before native completion.
    invocation->published = true;
    iree_hal_amd_gpu_program_publish(program, invocation->instance,
                                     queue->options.queue_ordinal,
                                     position + 1);
  }
  uint64_t sample_address = 0;
  iree_hal_amd_gpu_aql_signal_t* samples = NULL;
  if (operation) {
    iree_hal_amd_gpu_profile_completion_t* completion =
        &queue->issued[position & (queue->options.retirement_capacity - 1)]
             .execute.profile;
    *completion = (iree_hal_amd_gpu_profile_completion_t){
        .first = block.dispatches.first,
        .count = operation->commands ? block.dispatches.count : 0,
        .sample_stride = operation->completion.sample_stride,
    };
    if (sampled && completion->count) {
      iree_hal_amd_gpu_byte_storage_t* storage = queue->samples->current;
      completion->samples =
          (const void*)(storage->memory.contents.data + sample_offset +
                        offsetof(iree_hal_amd_gpu_aql_signal_t,
                                 start_timestamp));
      completion->samples_end = sample_end;
      sample_address = storage->address + sample_offset;
      samples = (void*)(storage->memory.contents.data + sample_offset);
      iree_hal_amd_gpu_byte_storage_pool_commit(queue->samples, sample_end,
                                                false);
    }
  }
  ++invocation->block_ordinal;
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position, argument_end,
                              terminal, true);
  amdf_status_t publication_status = iree_hal_amd_gpu_aql_emit_waits(
      ring, &queue->storage, first ? submission->native_wait_count : 0,
      submission->waits, &packet_position);
  // publish() changes initialization state, so native prelude emission must
  // consume the pre-publication facts captured above.
  if (amdf_status_is_ok(publication_status)) {
    if (host_template) {
      if (block.packet_count == block.dispatches.count) {
        publication_status = iree_hal_amd_gpu_aql_publish_host_template(
            queue, invocation->program, block, packet_position, packets,
            dispatch_completion ? signal->device_address : 0, sample_address,
            samples);
      } else {
        publication_status = iree_hal_amd_gpu_aql_program_emit_host(
            block.packet_count, packets, ring, &queue->storage, packet_position,
            sample_address, samples);
      }
    } else {
      publication_status = iree_hal_amd_gpu_aql_program_emit(
          queue->options.programs, program, invocation->instance, block.offset,
          block.packet_count, ring, &queue->storage, packet_position,
          arguments->address + argument_offset, values_address, sample_address,
          upload, program_flags);
    }
  }
  packet_position += prefix + block.packet_count;
  const uint64_t signal_address = signal->device_address;
  // A terminal dispatch may already have completed. Publishing the retirement
  // count hands off capture ownership; no invocation or signal access follows.
  iree_hal_amd_gpu_aql_publish_retirement(queue, position);
  if (amdf_status_is_ok(publication_status) && !dispatch_completion) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
            0, NULL, signal_address));
  }
  iree_hal_amd_gpu_aql_publish_notify(queue, publication_status,
                                      ring->reserved - 1);
  return terminal ? IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE
                  : IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE;
}

// CPU preparation is private to the exclusive publisher. A larger ABI grows
// this reusable host workspace without changing its native execution path.
static iree_hal_amd_gpu_aql_publish_result_t
iree_hal_amd_gpu_aql_publish_execute_host(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  const iree_host_size_t required =
      submission->execute.program->host_template->preparation_byte_length;
  iree_byte_span_t* workspace = &queue->producer.preparation;
  if (required > workspace->data_length) {
    const iree_host_size_t capacity = iree_max(required, 16384u);
    *out_status = iree_allocator_realloc(queue->host_allocator, capacity,
                                         (void**)&workspace->data);
    if (!iree_status_is_ok(*out_status)) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
    }
    workspace->data_length = capacity;
  }
  return iree_hal_amd_gpu_aql_publish_execute_prepared(
      queue, submission, workspace->data, out_status);
}

iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_execute(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  const amdf_status_t seal_status = iree_hal_amd_gpu_aql_publish_run(queue);
  if (!amdf_status_is_ok(seal_status)) {
    *out_status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(seal_status, "publish preceding AQL run");
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  const iree_hal_amd_gpu_aql_host_template_t* host_template =
      submission->execute.program->host_template;
  return host_template && host_template->fixup_count
             ? iree_hal_amd_gpu_aql_publish_execute_host(queue, submission,
                                                         out_status)
             : iree_hal_amd_gpu_aql_publish_execute_prepared(queue, submission,
                                                             NULL, out_status);
}

iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_transfer(
    iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status) {
  const amdf_status_t seal_status = iree_hal_amd_gpu_aql_publish_run(queue);
  if (!amdf_status_is_ok(seal_status)) {
    *out_status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(seal_status, "publish preceding AQL run");
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  if (!submission->transfer.capture.count) {
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {0};
    return iree_hal_amd_gpu_aql_publish(
        queue, &request, submission->wait_count, submission->waits,
        (iree_hal_amd_gpu_aql_wait_counts_t){.native_count =
                                                 submission->native_wait_count},
        submission->signals, submission->before, submission->after, NULL, NULL,
        &submission, out_status);
  }
  const uint64_t position = (uint64_t)iree_atomic_load(
      &queue->producer.count, iree_memory_order_relaxed);
  if (position - (uint64_t)iree_atomic_load(&queue->completed.count,
                                            iree_memory_order_acquire) ==
      queue->options.retirement_capacity) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
  }
  iree_hal_amd_gpu_transfer_t* transfer = &submission->transfer.capture;
  const bool first = !transfer->cursor.index && !transfer->cursor.offset;
  iree_hal_amd_gpu_byte_storage_t* arguments = queue->producer.arguments;
  iree_hal_amd_gpu_byte_storage_t* transfers = queue->producer.transfers;
  iree_hal_amd_gpu_aql_ring_t* ring = &queue->producer.ring;
  const uint32_t waits = first ? (submission->native_wait_count + 4) / 5 : 0;
  const uint32_t capacity =
      iree_min(iree_min(IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS,
                        transfer->count - transfer->cursor.index),
               ring->capacity / 2 - waits - 2);
  const uint64_t argument_length =
      capacity * sizeof(iree_hal_amd_gpu_blit_arguments_t);
  const uint64_t staging_length = iree_hal_amd_gpu_aql_transfer_staging_length(
      transfer, transfers ? transfers->capacity : 0);
  uint64_t argument_offset = 0, argument_end = 0;
  iree_hal_amd_gpu_aql_publish_result_t prepared =
      iree_hal_amd_gpu_aql_arguments(queue, argument_length, 8,
                                     &argument_offset, &argument_end);
  if (prepared != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
    return prepared;
  }
  uint64_t staging_offset = 0, staging_end = 0;
  if (staging_length && !transfers) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_TRANSFERS;
  }
  if (transfers) {
    prepared = iree_hal_amd_gpu_aql_reserve_bytes(
        transfers, &queue->completed.transfers, staging_length, 8,
        IREE_HAL_AMD_GPU_AQL_PUBLISH_TRANSFERS, &staging_offset, &staging_end);
    if (prepared != IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE) {
      return prepared;
    }
  }
  const uint32_t before = first ? submission->before : 0;
  const bool prefix = (before & (3u << 11)) != 0;
  // Reserve enough credit for the bounded candidate before touching staging.
  const uint32_t maximum = waits + prefix + capacity + 1;
  if (ring->reserved - ring->consumed + maximum > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + maximum > ring->capacity) {
      return IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY;
    }
  }
  iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
  *out_status = iree_hal_amd_gpu_aql_transfer_prepare_chunk(
      transfer, queue->options.programs->blit, capacity,
      iree_make_byte_span(arguments->memory.contents.data + argument_offset,
                          argument_length),
      arguments->address + argument_offset, queue->producer.argument_streaming,
      transfers ? iree_make_byte_span(
                      transfers->memory.contents.data + staging_offset,
                      staging_length)
                : iree_byte_span_empty(),
      transfers ? transfers->address + staging_offset : 0, &chunk);
  if (!iree_status_is_ok(*out_status)) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE;
  }
  iree_hal_amd_gpu_signal_t* signal = iree_hal_amd_gpu_signal_pool_try_acquire(
      &queue->options.signals->pool, 1);
  if (!signal) {
    return IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS;
  }
  const bool terminal = chunk.next.index == transfer->count;
  const uint32_t after = iree_hal_amd_gpu_aql_barrier_join(
      terminal ? submission->after : 0,
      chunk.download.length ? IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM : 0);
  // All host arguments precede the first acquire. Sibling transfers have no
  // implicit dependencies; the ordered terminal packet joins the entire group.
  chunk.packets[0].words[0] = chunk.packets[0].words[0] |
                              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                              IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
  const iree_hal_amd_gpu_aql_packet_t completion =
      iree_hal_amd_gpu_aql_packet_barrier(
          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | after,
          0, NULL, signal->device_address);
  uint64_t packet_position = 0;
  const bool reserved = iree_hal_amd_gpu_aql_ring_try_reserve(
      ring, waits + prefix + chunk.packet_count + 1, &packet_position);
  IREE_ASSERT(reserved, "exclusive publication retains admitted packet credit");
  (void)reserved;
  arguments->end = argument_end - argument_length + chunk.argument_length;
  if (chunk.staging_length) {
    transfers->end = staging_end - staging_length + chunk.staging_length;
  }
  if (first) {
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      if (submission->waits[i].signal) {
        iree_hal_amd_gpu_signal_join_frontier(submission->waits[i].signal,
                                              queue->producer.queue_count,
                                              queue->producer.frontier);
      }
    }
  }
  transfer->cursor = chunk.next;
  iree_hal_amd_gpu_aql_retirement_t* action =
      &queue->issued[position & (queue->options.retirement_capacity - 1)];
  action->transfer.end = transfers ? transfers->end : 0;
  action->transfer.download = chunk.download;
  iree_hal_amd_gpu_aql_accept(queue, submission, signal, position,
                              arguments->end, terminal,
                              !transfer->host_completion_required);
  amdf_status_t publication_status = iree_hal_amd_gpu_aql_emit_waits(
      ring, &queue->storage, first ? submission->native_wait_count : 0,
      submission->waits, &packet_position);
  iree_hal_amd_gpu_aql_publish_retirement(queue, position);
  if (prefix && amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++,
        iree_hal_amd_gpu_aql_packet_barrier(
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (before & (3u << 11)),
            0, NULL, 0));
  }
  for (uint32_t i = 0;
       i < chunk.packet_count && amdf_status_is_ok(publication_status); ++i) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position++, chunk.packets[i]);
  }
  if (amdf_status_is_ok(publication_status)) {
    publication_status = iree_hal_amd_gpu_aql_write_packet(
        ring, &queue->storage, packet_position, completion);
  }
  iree_hal_amd_gpu_aql_publish_notify(queue, publication_status,
                                      ring->reserved - 1);
  return terminal ? IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE
                  : IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE;
}
