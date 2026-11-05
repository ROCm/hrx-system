// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/state_memory.h"

#include "loom/serve/service/text/state.h"

iree_status_t loom_serve_text_memory_allocate(loom_serve_text_state_t* state) {
  loom_serve_pool_t* pool = state->pool;
  loom_serve_execution_t* execution = state->execution;
  state->memory.pool = loom_serve_pool_memory(pool);
  if (state->memory.pool) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        state->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state->memory.backing.semaphore));
  }
  // Binding roles are the private adapter contract. Sizes, views, initial
  // contents and zero extents are produced by the state's source bootstrap.
  iree_hal_pool_reservation_request_t
      requests[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {0};
  iree_hal_buffer_t** targets[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  loom_serve_memory_range_t
      commitments[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_host_size_t allocation_count = 0;
  iree_host_size_t commitment_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    state->allocations[i].execution = execution;
    iree_hal_buffer_t** allocation = &state->allocations[i].buffer;
    const loom_serve_text_allocation_t* record = &state->storage.allocations[i];
    const iree_device_size_t length = record->length;
    const iree_device_size_t alignment = record->alignment;
    const iree_device_size_t clear_length = record->clear_length;
    // Carry belongs to the source's retained state, not the drafting policy.
    // Speculative feedback and optional draft cache roots still require
    // drafting.
    const bool required =
        i < 2 ||
        (i < 6
             ? iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)
         : i == 6
             ? state->storage.carry_stride != 0
             : i < 9 && iree_any_bit_set(state->flags,
                                         LOOM_SERVE_TEXT_STATE_FLAG_DRAFT));
    const bool optional =
        i >= 9 &&
        iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT);
    if ((length != 0) != required && !optional) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid source allocation role %zu", i);
    } else if (length) {
      if (state->memory.pool && record->cache) {
        const loom_serve_memory_geometry_t geometry =
            loom_serve_pool_state_geometry(pool);
        status = loom_serve_virtual_buffer_allocate(
            state->memory.pool, geometry,
            iree_device_align(length, geometry.mapping_size), alignment,
            &state->memory.statistics, &state->memory.buffers[i]);
        if (iree_status_is_ok(status)) {
          *allocation =
              loom_serve_virtual_buffer_handle(state->memory.buffers[i]);
          iree_hal_buffer_retain(*allocation);
          status = loom_serve_retirement_track(&state->retirement, allocation,
                                               state->host_allocator);
          if (iree_status_is_ok(status) && clear_length && i != 1) {
            state->memory.private_lengths[i] = clear_length;
            commitments[commitment_count++] = (loom_serve_memory_range_t){
                state->memory.buffers[i], 0, clear_length};
          }
        }
      } else {
        requests[allocation_count] =
            iree_hal_make_pool_reservation_request(length, alignment);
        targets[allocation_count++] = allocation;
      }
    }
  }
  if (iree_status_is_ok(status) && allocation_count) {
    status = loom_serve_execution_alloca_from_pool(
        execution, loom_serve_pool_storage(pool), allocation_count, requests,
        buffers);
    for (iree_host_size_t i = 0; i < allocation_count; ++i) {
      *targets[i] = buffers[i];
      if (iree_status_is_ok(status)) {
        status = loom_serve_retirement_track(&state->retirement, targets[i],
                                             state->host_allocator);
      }
    }
  }
  if (iree_status_is_ok(status) && commitment_count) {
    loom_serve_memory_admission_t admission;
    status = loom_serve_memory_pool_commit_async(
        state->memory.pool, commitment_count, commitments,
        state->memory.backing.semaphore, state->memory.backing.value + 1,
        &admission, &state->memory.backing.operation);
    if (iree_status_is_ok(status) &&
        admission != LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "physical pool cannot back private text state");
    }
    if (iree_status_is_ok(status)) {
      ++state->memory.backing.value;
      status = loom_serve_execution_barrier(
          execution,
          (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                      &state->memory.backing.value});
    }
  }
  return status;
}

iree_status_t loom_serve_text_memory_initialize(
    loom_serve_text_state_t* state) {
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t
      transfers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 2];
  iree_host_size_t transfer_count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    const iree_device_size_t clear_length =
        state->storage.allocations[i].clear_length;
    if (clear_length && !(i == 1 && state->memory.buffers[i])) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {.target_buffer = state->allocations[i].buffer,
                   .length = clear_length,
                   .pattern = &zero,
                   .pattern_length = sizeof(zero)},
      };
    }
  }
  if (iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)) {
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS];
    if (origins.data_length) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->epoch.buffers[2],
                     .length = origins.data_length},
      };
    }
  }
  if (state->draft.row_table) {
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS];
    if (origins.data_length) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->draft.row_table,
                     .length = origins.data_length},
      };
    }
  }
  uint64_t completion = 0;
  return loom_serve_execution_transfer(state->execution, transfer_count,
                                       transfers, &completion);
}

void loom_serve_text_memory_set_observer(loom_serve_text_memory_t* memory,
                                         void (*wake)(void*), void* user_data) {
  loom_serve_memory_operation_set_observer(memory->backing.operation, wake,
                                           user_data);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    loom_serve_memory_operation_set_observer(memory->growth.operations[i], wake,
                                             user_data);
  }
}

bool loom_serve_text_memory_is_retired(const loom_serve_text_memory_t* memory) {
  bool complete =
      loom_serve_memory_operation_is_complete(memory->backing.operation);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    complete &=
        loom_serve_memory_operation_is_complete(memory->growth.operations[i]);
  }
  return complete;
}

void loom_serve_text_memory_await_retirement(loom_serve_text_memory_t* memory) {
  loom_serve_memory_operation_await(memory->backing.operation);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    loom_serve_memory_operation_await(memory->growth.operations[i]);
  }
}

// Successful model readiness includes the native signal; join its final access
// before reusing the state-owned operation slot or editing its slabs again.
// A prefix pin can leave completed backing here until the next capacity change.
void loom_serve_text_memory_complete_backing(loom_serve_text_memory_t* memory) {
  const bool complete =
      loom_serve_memory_operation_is_complete(memory->backing.operation);
  IREE_ASSERT(complete);
  loom_serve_memory_operation_free(memory->backing.operation);
  memory->backing.operation = NULL;
}

void loom_serve_text_memory_complete_growth(loom_serve_text_memory_t* memory) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    const bool complete =
        loom_serve_memory_operation_is_complete(memory->growth.operations[i]);
    IREE_ASSERT(complete);
    loom_serve_memory_operation_free(memory->growth.operations[i]);
    memory->growth.operations[i] = NULL;
    iree_hal_semaphore_release(memory->growth.semaphores[i]);
    memory->growth.semaphores[i] = NULL;
  }
}

iree_status_t loom_serve_text_memory_deinitialize(
    loom_serve_text_state_t* state) {
  loom_serve_memory_operation_free(state->memory.backing.operation);
  state->memory.backing.operation = NULL;
  loom_serve_text_memory_complete_growth(&state->memory);
  loom_serve_text_storage_release_payloads(&state->storage);
  iree_hal_semaphore_release(state->memory.backing.semaphore);
  state->memory.backing.semaphore = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    status = loom_serve_virtual_buffer_free(state->memory.buffers[i]);
    if (iree_status_is_ok(status)) {
      state->memory.buffers[i] = NULL;
    }
  }
  return status;
}

iree_status_t loom_serve_text_memory_trim(loom_serve_text_state_t* state) {
  loom_serve_virtual_buffer_t*
      buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    if (state->memory.buffers[i]) {
      buffers[count++] = state->memory.buffers[i];
    }
  }
  IREE_ASSERT(!state->memory.backing.operation);
  IREE_RETURN_IF_ERROR(loom_serve_memory_pool_trim_async(
      state->memory.pool, count, buffers, state->memory.backing.semaphore,
      state->memory.backing.value + 1, &state->memory.backing.operation));
  ++state->memory.backing.value;
  if (!state->memory.backing.operation) {
    return iree_ok_status();
  }
  return loom_serve_execution_barrier(
      state->execution,
      (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                  &state->memory.backing.value});
}
