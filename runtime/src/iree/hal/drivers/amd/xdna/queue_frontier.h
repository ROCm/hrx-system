// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_FRONTIER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_FRONTIER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_amd_xdna_frontier_t,
                               IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY);

// Fixed causal lower bound carried by an operation and its native queue.
typedef struct iree_hal_amd_xdna_frontier_state_t {
  // Sorted causal axes retained without allocation.
  iree_hal_amd_xdna_frontier_t frontier;

  // True when the frontier contains the complete causal history.
  bool exact;
} iree_hal_amd_xdna_frontier_state_t;

// Controls which physical ordering facts may resolve a wait.
enum iree_hal_amd_xdna_wait_resolution_flag_bits_e {
  IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE = 0u,

  // Allows an exact producer already dominated by the accepted native
  // frontier to rely on the queue's physical FIFO.
  IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO = 1u << 0,
};
typedef uint32_t iree_hal_amd_xdna_wait_resolution_flags_t;

// Result of resolving all waits for one operation.
typedef enum iree_hal_amd_xdna_wait_resolution_e {
  // Every wait is reached or proven by an accepted native FIFO predecessor.
  IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY = 0,

  // At least one wait requires an asynchronous software timepoint.
  IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER = 1,
} iree_hal_amd_xdna_wait_resolution_t;

// Initializes an exact empty frontier state.
void iree_hal_amd_xdna_frontier_state_initialize(
    iree_hal_amd_xdna_frontier_state_t* out_state);

// Copies |source| into |target|.
void iree_hal_amd_xdna_frontier_state_copy(
    const iree_hal_amd_xdna_frontier_state_t* source,
    iree_hal_amd_xdna_frontier_state_t* target);

// Merges |source| into |target|. Capacity overflow preserves the previous
// lower bound and makes |target| inexact.
void iree_hal_amd_xdna_frontier_state_merge(
    iree_hal_amd_xdna_frontier_state_t* target,
    const iree_hal_amd_xdna_frontier_state_t* source);

// Adds one accepted queue epoch. If the first self axis would overflow the
// fixed capacity, the self epoch becomes the retained lower bound and the
// state becomes inexact.
void iree_hal_amd_xdna_frontier_state_advance(
    iree_hal_amd_xdna_frontier_state_t* state, iree_async_axis_t axis,
    uint64_t epoch);

// Returns true when |state| contains |axis| at or beyond |epoch|.
bool iree_hal_amd_xdna_frontier_state_dominates(
    const iree_hal_amd_xdna_frontier_state_t* state, iree_async_axis_t axis,
    uint64_t epoch);

// Returns the variable-length frontier view of |state|.
const iree_async_frontier_t* iree_hal_amd_xdna_frontier_state_as_frontier(
    const iree_hal_amd_xdna_frontier_state_t* state);

// Resolves |waits| and captures their causal lower bound.
//
// |device| identifies the exact XDNA semaphore domain. |accepted_state| is the
// current native queue frontier, or NULL for host work that cannot inherit
// native FIFO order. A failed semaphore is returned as a status. A reached but
// tainted value or a fixed-capacity overflow remains executable but makes the
// resulting state inexact.
iree_status_t iree_hal_amd_xdna_frontier_resolve_waits(
    iree_hal_device_t* device, iree_hal_semaphore_list_t waits,
    const iree_hal_amd_xdna_frontier_state_t* accepted_state,
    iree_hal_amd_xdna_wait_resolution_flags_t flags,
    iree_hal_amd_xdna_frontier_state_t* out_state,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_FRONTIER_H_
