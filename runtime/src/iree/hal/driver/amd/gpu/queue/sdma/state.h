// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_STATE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_STATE_H_

#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/host_call.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/operation.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/queue.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_sdma_growth_t iree_hal_amd_gpu_sdma_growth_t;

// Parallel to one native retirement slot. The native publisher returns capacity
// only after this record's readiness/copy/notification tail has joined.
typedef struct iree_hal_amd_gpu_sdma_span_t {
  // Retained capture, independent of its reusable publication entry.
  iree_hal_amd_gpu_sdma_operation_t* operation;
  // Native-domain publication epoch, including prior owners of this queue.
  uint64_t epoch;
  // Cursor committed by acceptance, not by speculative preparation.
  iree_hal_amd_gpu_sdma_cursor_t next;
  // KERNEL visibility requires this invocation to resume after retirement.
  bool boundary;
  // This slot owns the unpublished suffix until acceptance or boundary return.
  bool owns_publication;
  // Worker-owned bulk copy; zero length after its final host access.
  iree_hal_amd_gpu_download_t download;
  // Complete immutable command span, retained through capacity retries.
  uint32_t words[IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS];
} iree_hal_amd_gpu_sdma_span_t;

struct iree_hal_amd_gpu_sdma_queue_t {
  // Public exact queue; the family and device remain borrowed.
  iree_hal_queue_t base;
  // Allocator for cold ownership and oversized CPU captures.
  iree_allocator_t host_allocator;
  // Borrowed native domain, events, and allocation service.
  iree_hal_amd_gpu_context_t* context;
  // Immutable device-owned services and qualified native facts.
  iree_hal_amd_gpu_sdma_queue_options_t options;
  // Sole native publisher and retirement observer, shared by all operations.
  iree_hal_amd_gpu_sdma_native_queue_t* native;
  // Native packet writers injected into the shared transfer traversal.
  iree_hal_amd_gpu_dma_transfer_encoding_t encoding;
  // Format feature row selected by native family, independent of shader ISA.
  amdf_queue_format_features_t features;
  // Optional final-release join installed by device membership.
  struct {
    // Joins sources, captures, and native access before returning ownership.
    bool (*fn)(void* user_data, iree_hal_queue_t* queue);
    // Stable device-owned membership slot.
    void* user_data;
  } release;
  // Optional profile session, stable while any operation is in flight.
  iree_hal_amd_gpu_profile_t* profile;
  // Shared CPU capture pool; ordinary calls allocate no native storage.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Coherent fixed staging, partitioned by native retirement slot.
  iree_hal_amd_gpu_byte_storage_t* staging;
  // Matching fixed command/retirement metadata.
  iree_hal_amd_gpu_sdma_span_t spans[IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY];
  // Exact shared file executor; file ownership creates no execution queue.
  iree_hal_amd_gpu_file_executor_t file_executor;
  // Exact shared host-callback executor.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Deferred dependency associations on the existing async infrastructure.
  iree_async_semaphore_wait_context_t wait_context;
  // Cancellation joins every dependency callback before releasing captures.
  iree_hal_amd_gpu_host_wait_set_t waits;
  // Joins callback tails with cold idle observation and shutdown notification.
  iree_slim_mutex_t host_mutex;
  // Unpublished operations and allocation/wait/file callback borrowers.
  iree_atomic_int64_t host_owners;
  // Public captures live through their final native or failed retirement.
  iree_atomic_int64_t operation_count;
  // Public acceptance is closed; internal accepted continuations still run.
  iree_atomic_int32_t closed;
  // Device-wide failure cancels unpublished work without claiming retirement.
  iree_atomic_int32_t failure_code;
  // Native coordinate advanced only after payload and callback retirement.
  iree_atomic_int64_t completed_epoch;
  // Cold-assigned canonical frontier namespace, borrowed from device
  // membership.
  struct {
    // Tracker retained by the device; NULL before group assignment.
    iree_async_frontier_tracker_t* tracker;
    // Canonical axis corresponding to options.queue_ordinal.
    iree_async_axis_t axis;
  } frontier;
  // Exclusive-publisher causal state; cold expansion installs both vectors.
  struct {
    // Installed vector width, including vacant stable coordinates.
    uint32_t count;
    // Accepted native prefix; initial storage follows this queue allocation.
    uint64_t* published;
    // Prospective dependency merge, reused by one preparation at a time.
    uint64_t* pending;
    // Owned replacement allocation, or NULL for initial inline storage.
    uint64_t* allocation;
  } producer;
  // Coalesced signal growth, owned by the exclusive native publisher.
  iree_hal_amd_gpu_sdma_growth_t* signal_growth;
  // Returned signal capacity can resume demand before native growth completes.
  struct {
    // Metadata-only entry serialized with signal acquisition and attachment.
    iree_hal_amd_gpu_sdma_metadata_t metadata;
    // One while this exact entry is queued or being consumed.
    iree_atomic_int32_t pending;
  } signal_reuse;
  // Coalesced trim command owns its entry until its publication turn ends.
  struct {
    // Metadata-only publication entry; it emits no native span.
    iree_hal_amd_gpu_sdma_metadata_t metadata;
    // One while this exact entry is linked or executing.
    iree_atomic_int32_t pending;
  } trim;
};

// Operation ownership uses the same callback-tail accounting as file and host
// services. End performs its final queue access under host_mutex.
void iree_hal_amd_gpu_sdma_host_begin(iree_hal_queue_t* base);
void iree_hal_amd_gpu_sdma_host_end(iree_hal_queue_t* base);
void iree_hal_amd_gpu_sdma_wake_shutdown(iree_hal_amd_gpu_sdma_queue_t* queue);

// Failure checks distinguish new public calls from accepted continuations.
iree_status_t iree_hal_amd_gpu_sdma_check_failure(
    iree_hal_amd_gpu_sdma_queue_t* queue);
iree_status_t iree_hal_amd_gpu_sdma_check_acceptance(
    iree_hal_amd_gpu_sdma_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_STATE_H_
