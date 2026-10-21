// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_GROWTH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_GROWTH_H_

#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/scratch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_pm4_growth_t iree_hal_amd_gpu_pm4_growth_t;

// Cold capacity requests belonging to one exact native publisher. Non-atomic
// fields belong to that publisher. Callbacks only enqueue typed results or
// capacity hints; they never install backing, walk captures, or publish GPU
// work.
typedef struct iree_hal_amd_gpu_pm4_growth_context_t {
  // Borrowed native owner, joined before destroying this context.
  iree_hal_amd_gpu_context_t* context;
  // Borrowed immutable coherent argument/signal memory policy.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed physical-local read/write scratch policy, with no host mapping.
  const iree_hal_amd_fabric_memory_options_t* scratch_memory;
  // Borrowed exact publisher's immutable scratch slot geometry.
  const iree_hal_amd_gpu_pm4_scratch_profile_t* scratch_profile;
  // Borrowed reusable host captures, live through every completed request.
  iree_hal_amd_gpu_capture_pool_t* captures;
  // Queue's sole publication FIFO and ownership transfer.
  iree_hal_amd_gpu_publisher_t* publisher;
  // Device publisher receiving completed acquisitions.
  iree_hal_amd_gpu_publication_t* publication;
  // Exact source index in publication, also the native queue ordinal.
  uint32_t queue_ordinal;
  // Installed native frontier width, updated only by the exact publisher.
  uint32_t queue_count;
  // Number of cells in each acquired signal generation.
  uint32_t signal_capacity;
  // Available cells from every acquired block, independent of pending growth.
  iree_hal_amd_gpu_signal_cache_t* signal_cache;
  // Coalesced returned-capacity publication with explicit wake enrollment.
  struct {
    // Stable typed record; initialized to ENTRY_REUSE_SIGNALS by the owner.
    iree_hal_amd_gpu_pm4_entry_t entry;
    // One while the record is linked or being consumed.
    iree_atomic_int32_t pending;
  } reuse;
  // One coalesced argument acquisition, or NULL without pending growth.
  iree_hal_amd_gpu_pm4_growth_t* arguments;
  // One coalesced signal acquisition, or NULL without pending growth.
  iree_hal_amd_gpu_pm4_growth_t* signals;
  // One coalesced scratch acquisition, or NULL without pending growth.
  iree_hal_amd_gpu_pm4_growth_t* scratch;
  // Live growth records, including native requests and queued reuse hints after
  // their entire waiting cohort has resumed. Part of cold queue idle.
  iree_atomic_int32_t pending_count;
} iree_hal_amd_gpu_pm4_growth_context_t;

// Transfers a detached capture to a coalesced growth request. The caller owns
// the queue publisher and has released any unpublished program reservation.
// kind is a GROW variant. capacity is the admitted power-of-two argument
// extent or per-wave scratch extent, and is ignored for other kinds.
// Success parks the capture outside the ready FIFO. Failure leaves ownership
// with the caller and starts no callback for this capture. Pending program
// requests are indexed directly by program and queue, without a registry walk.
iree_status_t iree_hal_amd_gpu_pm4_growth_begin(
    iree_hal_amd_gpu_pm4_growth_context_t* context,
    iree_hal_amd_gpu_pm4_entry_kind_t kind,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t capacity);

// Acquired backing and the detached waiting cohort. The publisher installs
// successful backing, then batch-splices first..last into its ready FIFO. On
// failure it retires the cohort through the normal submission failure path.
typedef struct iree_hal_amd_gpu_pm4_growth_result_t {
  // First waiting capture, or NULL when existing capacity already released it.
  iree_hal_amd_gpu_publication_entry_t* first;
  // Last waiting capture with a NULL next link, or NULL for an empty cohort.
  iree_hal_amd_gpu_publication_entry_t* last;
  // Owned acquired arguments, or NULL for another kind or failure.
  iree_hal_amd_gpu_argument_storage_t* arguments;
  // Owned acquired scratch generation, or NULL for another kind or failure.
  iree_hal_amd_gpu_pm4_scratch_storage_t* scratch;
  // Owned acquired signal block, or NULL for another kind or failure.
  iree_hal_amd_gpu_signal_storage_t* signals;
} iree_hal_amd_gpu_pm4_growth_result_t;

// Consumes a detached completed growth entry on its exact queue publisher.
// Initializes acquired addresses, clears the indexed pending request, and
// returns the waiting cohort without traversing it. argument_begin comes from
// the current generation at installation, after any intervening publication.
// Consumes the native-result reference; an already-queued program reuse hint
// independently retains its record. Partial backing is released asynchronously
// on failure; no native allocation/free or wait runs here. Native producer
// callbacks are joined before context/queue destruction.
iree_status_t iree_hal_amd_gpu_pm4_growth_end(
    iree_hal_amd_gpu_pm4_entry_t* entry, uint64_t argument_begin,
    iree_hal_amd_gpu_pm4_growth_result_t* out_result);

// Called with a live queue borrow by its publisher or service source.
// Enqueues a typed hint only while demand is enrolled; native signal returns
// notify the device-owned cache without reaching this queue state.
void iree_hal_amd_gpu_pm4_growth_signals_available(void* user_data);

// Consumes REUSE_SIGNALS under exclusive publication ownership. Available cells
// release the waiting cohort without waiting for native growth. Batch-splices
// without traversing captures; growth retains its own independent lifetime.
void iree_hal_amd_gpu_pm4_growth_reuse_signals(
    iree_hal_amd_gpu_pm4_growth_context_t* context);

// Consumes a program's enrolled reuse hint on its exact publisher. The hint
// owns its growth record even after native completion has consumed the clone.
// Requeues only the waiting cohort; native acquisition retains the program
// independently until its own result has been consumed.
void iree_hal_amd_gpu_pm4_growth_reuse_program(
    iree_hal_amd_gpu_pm4_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_GROWTH_H_
