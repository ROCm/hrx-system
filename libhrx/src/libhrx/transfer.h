// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_TRANSFER_H_
#define HRX_TRANSFER_H_

#include "hrx_runtime.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Converts call-time HRX geometry without sharing C effective types across the
// public API boundary. The caller establishes that |region| is non-NULL.
static inline iree_hal_transfer_region_t hrx_transfer_region_to_hal(
    const hrx_transfer_region_t* region) {
  return (iree_hal_transfer_region_t){
      .row_length = region->row_length,
      .row_count = region->row_count,
      .slice_count = region->slice_count,
      .source_strides = {region->source_strides.row_stride,
                         region->source_strides.slice_stride},
      .target_strides = {region->target_strides.row_stride,
                         region->target_strides.slice_stride},
  };
}

// Converts call-time fill geometry; the caller supplies a non-NULL region.
static inline iree_hal_fill_region_t hrx_fill_region_to_hal(
    const hrx_fill_region_t* region) {
  return (iree_hal_fill_region_t){
      .row_length = region->row_length,
      .row_count = region->row_count,
      .slice_count = region->slice_count,
      .target_strides = {region->target_strides.row_stride,
                         region->target_strides.slice_stride},
  };
}

// Submits a transfer transaction to the caller's selected |queue| and waits for
// terminal completion. The queue, its device, the operation array, and all
// borrowed host ranges remain live until the function returns. The caller
// supplies its host scheduling policy and owns queue placement and priority.
iree_status_t hrx_hal_queue_transfer_and_wait(
    iree_hal_queue_t* queue, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    iree_async_wait_flags_t wait_flags);

#ifdef __cplusplus
}
#endif

#endif  // HRX_TRANSFER_H_
