// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_BARRIER_H_
#define LIBHRX_SRC_LIBHRX_BARRIER_H_

#include "hrx_runtime.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Call-local translation storage. Ordinary before/after records and short range
// lists stay on the stack; native enqueue/recording captures them before
// return. Callers bypass this storage completely when retaining NULL defaults.
typedef struct hrx_barrier_storage_t {
  // Native boundary wrapper borrowing the lists below.
  iree_hal_queue_barriers_t barriers;
  // Native before and after lists, preserving NULL versus explicitly empty.
  iree_hal_barrier_list_t lists[2];
  // Common case: one record at each boundary.
  iree_hal_barrier_t inline_barriers[2];
  // Common global dependencies shared by the two boundaries.
  iree_hal_memory_barrier_t inline_memory_barriers[4];
  // Common concrete resource ranges shared by the two boundaries.
  iree_hal_buffer_barrier_t inline_buffer_barriers[4];
  // Optional contiguous overflow allocation, owned through deinitialize.
  void* allocation;
} hrx_barrier_storage_t;

// Validates public array/handle storage and translates descriptors without
// querying memory contracts. Requires a non-NULL input wrapper. Semantic and
// native capability validation belongs to the receiving HAL boundary.
iree_status_t hrx_barrier_storage_initialize(
    const hrx_queue_barriers_t* barriers, hrx_barrier_storage_t* out_storage);

void hrx_barrier_storage_deinitialize(hrx_barrier_storage_t* storage);

#ifdef __cplusplus
}
#endif
#endif  // LIBHRX_SRC_LIBHRX_BARRIER_H_
