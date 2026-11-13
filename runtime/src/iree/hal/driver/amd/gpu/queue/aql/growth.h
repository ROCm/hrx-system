// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_GROWTH_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_GROWTH_H_

#include "iree/hal/driver/amd/gpu/publication.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/aql/submission.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_aql_growth_t iree_hal_amd_gpu_aql_growth_t;

// Native acquisition is independent of ready publication. Only the exact
// publisher changes cohorts and installs results; callbacks enqueue results.
typedef struct iree_hal_amd_gpu_aql_growth_context_t {
  // Native backing owner, joined after every callback's final access.
  iree_hal_amd_gpu_context_t* context;
  // Borrowed coherent control-memory policy.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed cached ABI policy, identical to initial argument allocation.
  const iree_hal_amd_fabric_memory_options_t* argument_memory;
  // Borrowed read/write transfer policy, independent of native control.
  const iree_hal_amd_fabric_memory_options_t* transfer_memory;
  // Borrowed CPU-write/shader-read replay value policy.
  const iree_hal_amd_fabric_memory_options_t* input_memory;
  // Reusable cold request records.
  iree_hal_amd_gpu_capture_pool_t* captures;
  // Exact queue publication ownership and ready FIFO.
  iree_hal_amd_gpu_publisher_t* publisher;
  // Existing device service owner.
  iree_hal_amd_gpu_publication_t* publication;
  // Queue's stable native causal coordinate.
  uint32_t queue_ordinal;
  // Exact deferred service source, independent of the vector coordinate.
  uint32_t source_ordinal;
  // Installed native frontier width, updated only by the exact publisher.
  uint32_t queue_count;
  // Number of native descriptors in each acquired block.
  uint32_t signal_capacity;
  // Existing native cells can satisfy demand while acquisition is pending.
  iree_hal_amd_gpu_signal_cache_t* signal_cache;
  // Coalesced argument acquisition or NULL.
  iree_hal_amd_gpu_aql_growth_t* arguments;
  // Coalesced transfer payload acquisition or NULL.
  iree_hal_amd_gpu_aql_growth_t* transfers;
  // Coalesced profiling sample acquisition or NULL.
  iree_hal_amd_gpu_aql_growth_t* samples;
  // Coalesced replay value-table acquisition or NULL.
  iree_hal_amd_gpu_aql_growth_t* values;
  // Coalesced first indirect-command allocation or NULL.
  iree_hal_amd_gpu_aql_growth_t* commands;
  // Coalesced signal acquisition or NULL.
  iree_hal_amd_gpu_aql_growth_t* signals;
  // Native result owners, including results with an empty waiting cohort.
  iree_atomic_int32_t pending_count;
  // Returned signal capacity wakes only explicitly enrolled demand.
  struct {
    // Typed coalesced capacity hint.
    iree_hal_amd_gpu_aql_entry_t entry;
    // One while the hint is queued or being consumed.
    iree_atomic_int32_t pending;
  } reuse;
} iree_hal_amd_gpu_aql_growth_context_t;

// Widens pending host metadata independently of native allocation completion.
// The exact publisher installs prepared tables; subsequent acquisitions use
// the new extent. Existing pending backing participates in the caller's visit.
void iree_hal_amd_gpu_aql_growth_process_frontiers(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_frontier_request_t* request);

// Enqueues a completed native result without running GPU work on its producer.
void iree_hal_amd_gpu_aql_growth_enqueue(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_t* entry);

// Transfers a detached submission to its typed native acquisition. capacity is
// the admitted power-of-two byte length for a byte-storage request.
// Success parks outside the ready FIFO; failure transfers nothing.
iree_status_t iree_hal_amd_gpu_aql_growth_begin(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_kind_t kind,
    iree_hal_amd_gpu_aql_submission_t* submission, uint64_t capacity);

typedef struct iree_hal_amd_gpu_aql_growth_result_t {
  // First detached waiting capture, or NULL after an earlier capacity return.
  iree_hal_amd_gpu_publication_entry_t* first;
  // Last capture for constant-time requeue, or NULL for an empty cohort.
  iree_hal_amd_gpu_publication_entry_t* last;
  // Acquired purpose-specific byte generation, or NULL for other kinds/failure.
  iree_hal_amd_gpu_byte_storage_t* bytes;
  // Acquired signal generation, or NULL for another kind/failure.
  iree_hal_amd_gpu_signal_storage_t* signals;
} iree_hal_amd_gpu_aql_growth_result_t;

// Installs native metadata on the exact publisher and consumes the request.
// Failure returns the cohort and asynchronously releases partial backing.
iree_status_t iree_hal_amd_gpu_aql_growth_end(
    iree_hal_amd_gpu_aql_entry_t* entry, uint64_t byte_begin,
    iree_hal_amd_gpu_aql_growth_result_t* out_result);

// Called with a live queue borrow by its publisher or service source.
// Enqueues a typed hint only while demand is enrolled; native signal returns
// notify the device-owned cache without reaching this queue state.
void iree_hal_amd_gpu_aql_growth_signals_available(void* user_data);

// Requeues the parked cohort immediately when already-owned cells return.
void iree_hal_amd_gpu_aql_growth_reuse_signals(
    iree_hal_amd_gpu_aql_growth_context_t* context);

// Consumes the separately retained program availability hint. Native clone
// acquisition may still be running; ready captures can reuse released backing.
void iree_hal_amd_gpu_aql_growth_reuse_program(
    iree_hal_amd_gpu_aql_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_GROWTH_H_
