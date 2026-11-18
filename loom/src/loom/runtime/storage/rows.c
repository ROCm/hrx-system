// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/rows.h"

void loom_runtime_rows_initialize(iree_allocator_t host_allocator,
                                  loom_runtime_rows_t* out_state) {
  *out_state = (loom_runtime_rows_t){.host_allocator = host_allocator};
  loom_runtime_retirement_initialize(&out_state->retirement);
}

iree_status_t loom_runtime_rows_allocate(
    loom_runtime_rows_t* state, loom_runtime_device_t* device,
    loom_runtime_pool_t* pool, loom_runtime_execution_t* execution,
    iree_host_size_t row_count, iree_device_size_t input_stride,
    iree_device_size_t stream_stride, iree_device_size_t conditioning_length,
    iree_device_size_t alignment) {
  const iree_allocator_t host_allocator = state->host_allocator;
  state->pool = loom_runtime_pool_memory(pool);
  state->execution = execution;
  if (state->pool) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        loom_runtime_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state->backing.semaphore));
  }
  state->rows.bit_count = row_count;
  state->strides[LOOM_RUNTIME_ROWS_INPUT] = input_stride;
  state->strides[LOOM_RUNTIME_ROWS_CONDITIONING] = conditioning_length;
  state->strides[LOOM_RUNTIME_ROWS_STREAM] = stream_stride;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, iree_bitmap_calculate_words(row_count),
      sizeof(*state->rows.words), (void**)&state->rows.words);
  if (iree_status_is_ok(status) && !state->pool) {
    iree_hal_pool_reservation_request_t
        requests[LOOM_RUNTIME_ROWS_BUFFER_COUNT];
    for (iree_host_size_t i = 0; i < LOOM_RUNTIME_ROWS_BUFFER_COUNT; ++i) {
      const iree_device_size_t length =
          state->strides[i] *
          (i == LOOM_RUNTIME_ROWS_CONDITIONING ? 1 : row_count);
      requests[i] = iree_hal_make_pool_reservation_request(length, alignment);
    }
    status = loom_runtime_execution_alloca_from_pool(
        execution, loom_runtime_pool_storage(pool),
        LOOM_RUNTIME_ROWS_BUFFER_COUNT, requests, state->buffers);
  }
  for (iree_host_size_t i = 0;
       i < LOOM_RUNTIME_ROWS_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    if (state->pool) {
      const iree_device_size_t length =
          state->strides[i] *
          (i == LOOM_RUNTIME_ROWS_CONDITIONING ? 1 : row_count);
      status = loom_runtime_virtual_buffer_allocate(
          state->pool, loom_runtime_pool_state_geometry(pool), length,
          alignment, &state->statistics, &state->reservations[i]);
      if (iree_status_is_ok(status)) {
        state->buffers[i] =
            loom_runtime_virtual_buffer_handle(state->reservations[i]);
        iree_hal_buffer_retain(state->buffers[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_retirement_track(
          &state->retirement, &state->buffers[i], host_allocator);
    }
  }
  return status;
}

iree_hal_buffer_t* loom_runtime_rows_buffer(const loom_runtime_rows_t* state,
                                            iree_host_size_t index) {
  return state->buffers[index];
}

iree_status_t loom_runtime_rows_try_reserve(
    loom_runtime_rows_t* state, iree_host_size_t row_index,
    loom_runtime_residency_t* residency,
    loom_runtime_residency_admission_t* out_admission) {
  loom_runtime_memory_range_t ranges[LOOM_RUNTIME_ROWS_BUFFER_COUNT];
  iree_host_size_t range_count = 0;
  if (state->pool) {
    for (iree_host_size_t i = 0; i < LOOM_RUNTIME_ROWS_BUFFER_COUNT; ++i) {
      ranges[range_count++] = (loom_runtime_memory_range_t){
          .buffer = state->reservations[i],
          .offset = i == LOOM_RUNTIME_ROWS_CONDITIONING
                        ? 0
                        : row_index * state->strides[i],
          .length = state->strides[i],
      };
    }
  }
  IREE_ASSERT(!state->backing.operation);
  IREE_RETURN_IF_ERROR(loom_runtime_residency_try_admit_state(
      residency, range_count, ranges, NULL, state->backing.semaphore,
      state->backing.value + 1, out_admission, &state->backing.operation));
  if (*out_admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED) {
    if (state->pool) {
      ++state->backing.value;
    }
    iree_bitmap_set(state->rows, row_index);
  }
  if (state->backing.operation) {
    return loom_runtime_execution_barrier(
        state->execution,
        (iree_hal_semaphore_list_t){1, &state->backing.semaphore,
                                    &state->backing.value});
  }
  return iree_ok_status();
}

void loom_runtime_rows_complete_backing(loom_runtime_rows_t* state) {
  const bool complete =
      loom_runtime_memory_operation_is_complete(state->backing.operation);
  IREE_ASSERT(complete);
  loom_runtime_memory_operation_free(state->backing.operation);
  state->backing.operation = NULL;
}

// Contiguous rows share only their boundary mappings. Limit each query to the
// rows overlapping that boundary rather than scanning the source capacity.
static bool loom_runtime_rows_any_reserved(const loom_runtime_rows_t* state,
                                           iree_host_size_t begin,
                                           iree_host_size_t end) {
  return begin < end &&
         iree_bitmap_find_first_set((iree_bitmap_t){end, state->rows.words},
                                    begin) < end;
}

iree_status_t loom_runtime_rows_release_row(loom_runtime_rows_t* state,
                                            iree_host_size_t row_index,
                                            bool* out_complete) {
  *out_complete = false;
  if (state->backing.operation) {
    loom_runtime_rows_complete_backing(state);
    *out_complete = true;
    return iree_ok_status();
  }
  iree_bitmap_reset(state->rows, row_index);
  loom_runtime_memory_range_t ranges[2];
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0;
       state->pool && i < LOOM_RUNTIME_ROWS_BUFFER_COUNT; ++i) {
    if (i == LOOM_RUNTIME_ROWS_CONDITIONING) {
      continue;
    }
    loom_runtime_virtual_buffer_t* buffer = state->reservations[i];
    const iree_device_size_t mapping_size =
        loom_runtime_virtual_buffer_mapping_size(buffer);
    const iree_device_size_t length =
        iree_hal_buffer_byte_length(state->buffers[i]);
    const iree_device_size_t stride = state->strides[i];
    const iree_device_size_t row_begin = row_index * stride;
    const iree_device_size_t row_end = row_begin + stride;
    iree_device_size_t begin = row_begin - row_begin % mapping_size;
    iree_device_size_t end =
        row_end +
        iree_min((mapping_size - row_end % mapping_size) % mapping_size,
                 length - row_end);
    if (loom_runtime_rows_any_reserved(state, begin / stride, row_index)) {
      begin = row_begin +
              iree_min((mapping_size - row_begin % mapping_size) % mapping_size,
                       end - row_begin);
    }
    const iree_host_size_t end_row =
        iree_min(end / stride + (end % stride != 0), state->rows.bit_count);
    if (loom_runtime_rows_any_reserved(state, row_index + 1, end_row)) {
      end = row_end - row_end % mapping_size;
    }
    if (begin < end) {
      ranges[count++] =
          (loom_runtime_memory_range_t){buffer, begin, end - begin};
    }
  }
  if (count) {
    IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_trim_async(
        state->pool, count, ranges, state->backing.semaphore,
        state->backing.value + 1, &state->backing.operation));
    ++state->backing.value;
  }
  *out_complete = state->backing.operation == NULL;
  if (!*out_complete) {
    return loom_runtime_execution_barrier(
        state->execution,
        (iree_hal_semaphore_list_t){1, &state->backing.semaphore,
                                    &state->backing.value});
  }
  return iree_ok_status();
}

void loom_runtime_rows_retire(loom_runtime_rows_t* state,
                              void (*wake)(void* user_data), void* user_data) {
  loom_runtime_retirement_set_observer(&state->retirement, wake, user_data);
  loom_runtime_memory_operation_set_observer(state->backing.operation, wake,
                                             user_data);
  for (iree_host_size_t i = 0; i < LOOM_RUNTIME_ROWS_BUFFER_COUNT; ++i) {
    iree_hal_buffer_release(state->buffers[i]);
    state->buffers[i] = NULL;
  }
}

bool loom_runtime_rows_is_retired(loom_runtime_rows_t* state) {
  return loom_runtime_memory_operation_is_complete(state->backing.operation) &&
         loom_runtime_retirement_is_idle(&state->retirement);
}

void loom_runtime_rows_await_retirement(loom_runtime_rows_t* state) {
  loom_runtime_memory_operation_await(state->backing.operation);
  loom_runtime_retirement_await(&state->retirement);
}

iree_status_t loom_runtime_rows_deinitialize(loom_runtime_rows_t* state) {
  IREE_ASSERT(loom_runtime_rows_is_retired(state));
  loom_runtime_memory_operation_free(state->backing.operation);
  state->backing.operation = NULL;
  iree_hal_semaphore_release(state->backing.semaphore);
  state->backing.semaphore = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < LOOM_RUNTIME_ROWS_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    status = loom_runtime_virtual_buffer_free(state->reservations[i]);
    if (iree_status_is_ok(status)) {
      state->reservations[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_runtime_retirement_deinitialize(&state->retirement);
    iree_allocator_free(state->host_allocator, state->rows.words);
    state->rows = (iree_bitmap_t){0};
  }
  return status;
}
