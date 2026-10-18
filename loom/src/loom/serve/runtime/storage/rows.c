// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/rows.h"

void loom_serve_rows_initialize(iree_allocator_t host_allocator,
                                loom_serve_rows_t* out_state) {
  *out_state = (loom_serve_rows_t){.host_allocator = host_allocator};
  loom_serve_retirement_initialize(&out_state->retirement);
}

iree_status_t loom_serve_rows_allocate(
    loom_serve_rows_t* state, loom_serve_device_t* device,
    loom_serve_execution_t* execution, iree_host_size_t row_count,
    iree_device_size_t input_stride, iree_device_size_t stream_stride,
    iree_device_size_t conditioning_length, iree_device_size_t alignment) {
  const iree_allocator_t host_allocator = state->host_allocator;
  state->pool = loom_serve_device_memory_pool(device);
  state->execution = execution;
  if (state->pool) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state->backing.semaphore));
  }
  state->row_count = row_count;
  state->strides[LOOM_SERVE_ROWS_INPUT] = input_stride;
  state->strides[LOOM_SERVE_ROWS_CONDITIONING] = conditioning_length;
  state->strides[LOOM_SERVE_ROWS_STREAM] = stream_stride;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, row_count, sizeof(*state->rows), (void**)&state->rows);
  if (iree_status_is_ok(status) && !state->pool) {
    iree_hal_pool_reservation_request_t requests[LOOM_SERVE_ROWS_BUFFER_COUNT];
    for (iree_host_size_t i = 0; i < LOOM_SERVE_ROWS_BUFFER_COUNT; ++i) {
      const iree_device_size_t length =
          state->strides[i] *
          (i == LOOM_SERVE_ROWS_CONDITIONING ? 1 : row_count);
      requests[i] = iree_hal_make_pool_reservation_request(length, alignment);
    }
    status = loom_serve_execution_alloca_from_pool(
        execution, loom_serve_device_storage_pool(device),
        LOOM_SERVE_ROWS_BUFFER_COUNT, requests, state->buffers);
  }
  for (iree_host_size_t i = 0;
       i < LOOM_SERVE_ROWS_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    if (state->pool) {
      const iree_device_size_t length =
          state->strides[i] *
          (i == LOOM_SERVE_ROWS_CONDITIONING ? 1 : row_count);
      status = loom_serve_virtual_buffer_allocate(state->pool, length,
                                                  alignment, &state->statistics,
                                                  &state->reservations[i]);
      if (iree_status_is_ok(status)) {
        state->buffers[i] =
            loom_serve_virtual_buffer_handle(state->reservations[i]);
        iree_hal_buffer_retain(state->buffers[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_retirement_track(&state->retirement,
                                           &state->buffers[i], host_allocator);
    }
  }
  return status;
}

iree_hal_buffer_t* loom_serve_rows_buffer(const loom_serve_rows_t* state,
                                          iree_host_size_t index) {
  return state->buffers[index];
}

iree_status_t loom_serve_rows_try_reserve(loom_serve_rows_t* state,
                                          iree_host_size_t row_index,
                                          bool* out_reserved) {
  *out_reserved = false;
  bool reserved = true;
  if (state->pool) {
    loom_serve_memory_range_t ranges[LOOM_SERVE_ROWS_BUFFER_COUNT];
    for (iree_host_size_t i = 0; i < LOOM_SERVE_ROWS_BUFFER_COUNT; ++i) {
      ranges[i] = (loom_serve_memory_range_t){
          .buffer = state->reservations[i],
          .offset = i == LOOM_SERVE_ROWS_CONDITIONING
                        ? 0
                        : row_index * state->strides[i],
          .length = state->strides[i],
      };
    }
    IREE_ASSERT(!state->backing.operation);
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_commit_async(
        state->pool, IREE_ARRAYSIZE(ranges), ranges, state->backing.semaphore,
        state->backing.value + 1, &reserved, &state->backing.operation));
    if (reserved) {
      ++state->backing.value;
    }
  }
  if (reserved) {
    state->rows[row_index] = true;
  }
  *out_reserved = reserved;
  if (state->backing.operation) {
    return loom_serve_execution_barrier(
        state->execution,
        (iree_hal_semaphore_list_t){1, &state->backing.semaphore,
                                    &state->backing.value});
  }
  return iree_ok_status();
}

void loom_serve_rows_complete_backing(loom_serve_rows_t* state) {
  const bool complete =
      loom_serve_memory_operation_is_complete(state->backing.operation);
  IREE_ASSERT(complete);
  loom_serve_memory_operation_free(state->backing.operation);
  state->backing.operation = NULL;
}

iree_status_t loom_serve_rows_release_row(loom_serve_rows_t* state,
                                          iree_host_size_t row_index,
                                          bool* out_complete) {
  *out_complete = false;
  if (state->backing.operation) {
    loom_serve_rows_complete_backing(state);
    *out_complete = true;
    return iree_ok_status();
  }
  state->rows[row_index] = false;
  loom_serve_virtual_buffer_t* buffers[2];
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; state->pool && i < LOOM_SERVE_ROWS_BUFFER_COUNT;
       ++i) {
    if (i == LOOM_SERVE_ROWS_CONDITIONING) {
      continue;
    }
    loom_serve_virtual_buffer_begin_trim(state->reservations[i]);
    for (iree_host_size_t row = 0; row < state->row_count; ++row) {
      if (state->rows[row]) {
        loom_serve_virtual_buffer_keep(
            state->reservations[i], row * state->strides[i], state->strides[i]);
      }
    }
    buffers[count++] = state->reservations[i];
  }
  if (count) {
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_trim_async(
        state->pool, count, buffers, state->backing.semaphore,
        state->backing.value + 1, &state->backing.operation));
    ++state->backing.value;
  }
  *out_complete = state->backing.operation == NULL;
  if (!*out_complete) {
    return loom_serve_execution_barrier(
        state->execution,
        (iree_hal_semaphore_list_t){1, &state->backing.semaphore,
                                    &state->backing.value});
  }
  return iree_ok_status();
}

void loom_serve_rows_retire(loom_serve_rows_t* state,
                            void (*wake)(void* user_data), void* user_data) {
  loom_serve_retirement_set_observer(&state->retirement, wake, user_data);
  loom_serve_memory_operation_set_observer(state->backing.operation, wake,
                                           user_data);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_ROWS_BUFFER_COUNT; ++i) {
    iree_hal_buffer_release(state->buffers[i]);
    state->buffers[i] = NULL;
  }
}

bool loom_serve_rows_is_retired(loom_serve_rows_t* state) {
  return loom_serve_memory_operation_is_complete(state->backing.operation) &&
         loom_serve_retirement_is_idle(&state->retirement);
}

void loom_serve_rows_await_retirement(loom_serve_rows_t* state) {
  loom_serve_memory_operation_await(state->backing.operation);
  loom_serve_retirement_await(&state->retirement);
}

iree_status_t loom_serve_rows_deinitialize(loom_serve_rows_t* state) {
  IREE_ASSERT(loom_serve_rows_is_retired(state));
  loom_serve_memory_operation_free(state->backing.operation);
  state->backing.operation = NULL;
  iree_hal_semaphore_release(state->backing.semaphore);
  state->backing.semaphore = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < LOOM_SERVE_ROWS_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    status = loom_serve_virtual_buffer_free(state->reservations[i]);
    if (iree_status_is_ok(status)) {
      state->reservations[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_serve_retirement_deinitialize(&state->retirement);
    iree_allocator_free(state->host_allocator, state->rows);
    state->rows = NULL;
  }
  return status;
}
