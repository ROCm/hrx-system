// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_

#include "iree/hal/driver/amd/gpu/completion.h"
#include "iree/hal/driver/amd/gpu/publication.h"
#include "iree/hal/driver/amd/gpu/queue/publisher.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_sdma_queue_t iree_hal_amd_gpu_sdma_queue_t;

// A native failure reports logical completion independently of storage safety.
// Unretired operands remain owned by the caller even after the error is
// reported.
typedef enum iree_hal_amd_gpu_sdma_retirement_e {
  IREE_HAL_AMD_GPU_SDMA_UNRETIRED = 0,
  IREE_HAL_AMD_GPU_SDMA_RETIRED,
} iree_hal_amd_gpu_sdma_retirement_t;

// Cold protocol admission from captured native family facts.
amdf_queue_publication_modes_t iree_hal_amd_gpu_sdma_queue_publication_modes(
    const amdf_queue_family_info_t* family);

// One ready copy with already-resolved addresses and visibility effects.
// The pipeline owns this record and both operands through completion. A copy
// never exceeds MAX_COPY_LENGTH; callers split longer ranges into bounded
// copies and choose their own pipelining depth.
#define IREE_HAL_AMD_GPU_SDMA_MAX_COPY_LENGTH (UINT32_C(64) << 20)
typedef struct iree_hal_amd_gpu_sdma_copy_t {
  // Intrusive publication ownership, reusable only after completion.
  iree_hal_amd_gpu_publication_entry_t entry;
  // First device byte read by the selected SDMA family.
  uint64_t source;
  // First device byte written by the selected SDMA family.
  uint64_t target;
  // Nonzero exact extent, including byte-aligned tails.
  uint32_t length;
  // Qualified queue-side acquire/release operations for this copy.
  iree_hal_memory_effects_t effects;
  // Reports failure independently of final operand retirement. Successful
  // retirement runs on the completion owner. An UNRETIRED failure may precede
  // the one RETIRED notification. Callbacks only enqueue a prepared result;
  // they never issue I/O or run user code. RETIRED includes rejected work that
  // never acquired a native consumer.
  void (*notify)(void* user_data, amdf_status_t status,
                 iree_hal_amd_gpu_sdma_retirement_t retirement);
  // Borrowed slot or operation, live through the callback.
  void* user_data;
} iree_hal_amd_gpu_sdma_copy_t;

// Cold constructor for a device's explicitly selected transfer queue. All
// command and completion storage is provisioned here. The device installs
// the returned publication/completion sources before submitting copies.
// No HAL device reference, worker, or payload staging allocation is acquired.
iree_status_t iree_hal_amd_gpu_sdma_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_sdma_queue_t** out_queue);

// Whether this queue can realize global acquire/release around each copy.
// The selected publication protocol supplies either explicit stream packets
// or native submission boundaries; callers establish dependencies first.
bool iree_hal_amd_gpu_sdma_queue_supports_cache_control(
    const iree_hal_amd_gpu_sdma_queue_t* queue);

// Transfers one ready, validated copy without allocation. An idle caller
// publishes one entry immediately. The completion callback may precede return;
// the caller joins its submitting borrow before reusing the record.
void iree_hal_amd_gpu_sdma_queue_submit(iree_hal_amd_gpu_sdma_queue_t* queue,
                                        iree_hal_amd_gpu_sdma_copy_t* copy);

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_sdma_queue_publication_source(
    iree_hal_amd_gpu_sdma_queue_t* queue);
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_sdma_queue_completion_source(
    iree_hal_amd_gpu_sdma_queue_t* queue);

// Cold query after the device has joined its copy producers.
bool iree_hal_amd_gpu_sdma_queue_is_idle(iree_hal_amd_gpu_sdma_queue_t* queue);

// Joins CPU publication and failure-notification borrowers after admission has
// closed. Unretired native copies retain both operands and are not certified
// safe by this predicate. NULL is idle.
bool iree_hal_amd_gpu_sdma_queue_is_host_idle(
    iree_hal_amd_gpu_sdma_queue_t* queue);

// Stops native access while retaining command, mapping, and operand ownership.
// The parent excludes completion observers first. False means a CPU publisher
// is still active and leaves out_status unchanged; true supplies the native
// stop result. A failed native stop remains sticky and preserves backing.
bool iree_hal_amd_gpu_sdma_queue_try_inactivate(
    iree_hal_amd_gpu_sdma_queue_t* queue, amdf_status_t* out_status);

// Returns accepted operands with the original failure after all affected
// native queues have stopped. The parent exclusively owns completion. False
// means a concurrent CPU publisher or failure notifier must join first.
bool iree_hal_amd_gpu_sdma_queue_retire_failed(
    iree_hal_amd_gpu_sdma_queue_t* queue);

// Joins the final notification tail and releases native storage. Called after
// both sources and every submitting caller have joined. Failure preserves the
// owner and context; it never certifies reclamation after a failed native join.
amdf_status_t iree_hal_amd_gpu_sdma_queue_free(
    iree_hal_amd_gpu_sdma_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_
