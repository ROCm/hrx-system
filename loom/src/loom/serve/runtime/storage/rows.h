// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_ROWS_H_
#define LOOM_SERVE_RUNTIME_STORAGE_ROWS_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/parameter/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_SERVE_ROWS_INPUT = 0,
  LOOM_SERVE_ROWS_CONDITIONING = 1,
  LOOM_SERVE_ROWS_STREAM = 2,
  LOOM_SERVE_ROWS_BUFFER_COUNT = 3,
};

// Stable source-addressed input/history rows and shared conditioning. Batch
// membership changes only the work packet, never these roots or row origins.
typedef struct loom_serve_rows_t {
  // Allocator owning the row ownership array.
  iree_allocator_t host_allocator;
  // Borrowed shared commitment owner; NULL selects fixed backing.
  loom_serve_memory_pool_t* pool;
  // Borrowed model timeline, enclosing state mapping and release readiness.
  loom_serve_execution_t* execution;
  // Accounting shared by all virtual reservations below.
  loom_serve_memory_statistics_t statistics;
  // Final queue ownership, independent of success/failure readiness.
  loom_serve_retirement_t retirement;
  // One native capacity transition, serialized with other model mutation.
  struct {
    // Owned immutable native borrower, joined independently of queue failure.
    loom_serve_memory_operation_t* operation;
    // Owned readiness timeline imported into model execution.
    iree_hal_semaphore_t* semaphore;
    // Last admitted native readiness value.
    uint64_t value;
  } backing;
  // Whole-arena roots lent to numerical commands and transfers.
  iree_hal_buffer_t* buffers[LOOM_SERVE_ROWS_BUFFER_COUNT];
  // Optional stable address reservations, owned through actual retirement.
  loom_serve_virtual_buffer_t* reservations[LOOM_SERVE_ROWS_BUFFER_COUNT];
  // Input/state row strides and the complete conditioning extent.
  iree_device_size_t strides[LOOM_SERVE_ROWS_BUFFER_COUNT];
  // Addressable row count, independent of current cohort membership.
  iree_host_size_t row_count;
  // Owner-only row reservations, retained until release_row.
  bool* rows;
} loom_serve_rows_t;

void loom_serve_rows_initialize(iree_allocator_t host_allocator,
                                loom_serve_rows_t* out_state);

// Geometry is validated by the source-loading boundary. Input and stream
// strides are positive alignment multiples; row_count * stride fits the device
// address domain.
// Elastic backing reserves addresses only; fixed backing allocates all rows.
// The shared device and physical pool outlive state. On failure, state owns
// partial construction and remains safe to retire and deinitialize.
iree_status_t loom_serve_rows_allocate(
    loom_serve_rows_t* state, loom_serve_device_t* device,
    loom_serve_execution_t* execution, iree_host_size_t row_count,
    iree_device_size_t input_stride, iree_device_size_t stream_stride,
    iree_device_size_t conditioning_length, iree_device_size_t alignment);

// Borrows a tracked whole-arena root until state retirement. The index names
// one of the state buffers above; row offsets remain source-owned facts.
iree_hal_buffer_t* loom_serve_rows_buffer(const loom_serve_rows_t* state,
                                          iree_host_size_t index);

// At a retired model cut, admits parameters, one unowned row, and conditioning
// in one commitment transaction. CAPACITY/DEFERRED own neither a row nor a
// parameter pin and submit no activation or destination backing. The residency
// and state belong to the same device and serialized model owner. Its prepared
// private high-water covers the shared cohort work, not each resident row.
// Mapping readiness precedes initialization on the model execution timeline.
// Actual platform failures retain any partial backing for terminal teardown.
iree_status_t loom_serve_rows_try_reserve(
    loom_serve_rows_t* state, iree_host_size_t row_index,
    loom_serve_residency_t* residency,
    loom_serve_residency_admission_t* out_admission);

// Joins a native backing change after successful model readiness, or a no-op.
void loom_serve_rows_complete_backing(loom_serve_rows_t* state);

// Releases a reserved row only after its accepted work and host borrowers
// retire. Surviving rows retain their contents and addresses. Shared slabs stay
// backed while any row needs them; conditioning remains reusable model state.
// False requires successful model readiness before calling again for this row.
// No other state mutation occurs until completion; failure retains all owners.
iree_status_t loom_serve_rows_release_row(loom_serve_rows_t* state,
                                          iree_host_size_t row_index,
                                          bool* out_complete);

// Relinquishes caller roots after all CPU borrowers stop. Accepted queues can
// still retain tracked views. The observer only schedules the serialized owner.
void loom_serve_rows_retire(loom_serve_rows_t* state,
                            void (*wake)(void* user_data), void* user_data);
bool loom_serve_rows_is_retired(loom_serve_rows_t* state);

// Explicit outer-thread join; scheduler tasks observe is_retired instead.
void loom_serve_rows_await_retirement(loom_serve_rows_t* state);

// Requires actual retirement, independently of failed semaphore readiness.
// A platform release failure preserves remaining reservations for retry.
iree_status_t loom_serve_rows_deinitialize(loom_serve_rows_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_STORAGE_ROWS_H_
