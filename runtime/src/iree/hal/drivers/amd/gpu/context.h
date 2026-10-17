// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CONTEXT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CONTEXT_H_

#include "amdf/gpu.h"
#include "iree/async/event.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/device_event.h"
#include "iree/hal/drivers/amd/fabric/allocation_queue.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cold native ownership dominated by one HAL GPU device. Queues, executable
// storage and memory backends borrow these resources; they never activate a
// second native device. The HAL owner joins its workers and removes all native
// queues before freeing this context. Sharing a fabric preserves native
// identity without granting access between unrelated HAL device groups.
//
// Captured facts are immutable. Each queue selects an advertised protocol and
// publication mode independently of the endpoint's ISA, then copies the facts
// needed by its hot path. Native backing uses the fabric's shared NUMA
// allocation domain. This owner creates no execution queue or allocation pool
// and does not advertise HAL execution capabilities by itself.
typedef struct iree_hal_amd_gpu_context_t {
  // Allocator owning this structure and its captured arrays.
  iree_allocator_t host_allocator;
  // Retained owner of the native instance and ordinary system storage scope.
  iree_hal_amd_fabric_t* fabric;
  // Borrowed immutable core API table, dominated by fabric.
  const amdf_api_t* api;
  // Borrowed negotiated GPU extension table, including native event support.
  const amdf_gpu_api_t* gpu;
  // Owned endpoint, released after every device resource.
  amdf_endpoint_t* endpoint;
  // Owned execution/address domain, released after every queue and backing.
  amdf_device_t* device;
  // USER publication's owned GPU/host wake, or NULL for KERNEL publication.
  amdf_gpu_event_t* event;
  // KERNEL publication's owned native event, empty for USER publication.
  iree_async_event_native_t kernel_event;
  // Borrowed native destination dominated by kernel_event.
  amdf_native_event_t kernel_notification;
  // Physical identity and native family count captured at opening.
  amdf_endpoint_info_t endpoint_info;
  // Compiler identity and compute geometry, independent of queue protocol.
  amdf_gpu_endpoint_info_t target_info;
  // Activated native identity, epoch and achieved device services.
  amdf_gpu_device_info_t device_info;
  // Immutable producer operands; event return remains only a wake hint.
  amdf_gpu_event_info_t event_info;
  // Owned facts in native ordinal order, with endpoint_info.queue_family_count
  // entries. HAL family admission may select a strict subset of these.
  amdf_queue_family_info_t* families;
  // Number of physical-local storage scopes, excluding fabric's system scope.
  uint32_t memory_scope_count;
  // Owned array of borrowed scopes whose lifetime is dominated by endpoint.
  amdf_memory_scope_t** memory_scopes;
  // Borrowed diagnostic destination, live through final native cleanup.
  iree_hal_device_event_sink_t event_sink;
  // Native backing ownership, initialized after selecting host NUMA placement.
  struct {
    // Retained fabric allocation domain shared with other engines on this node.
    iree_hal_amd_fabric_allocation_queue_t* queue;
    // Protects the outstanding count and the final notification access.
    iree_slim_mutex_t mutex;
    // Wakes the cold teardown owner after the last native callback returns.
    iree_notification_t notification;
    // Accepted native memory callbacks that still borrow this context.
    iree_host_size_t pending_count;
  } memory;
  // A child failed its native ownership join. The cold owner serializes this
  // latch with final cleanup; no context resource may be released afterward.
  bool cleanup_blocked;
} iree_hal_amd_gpu_context_t;

// Opens one exact GPU endpoint, creates its address domain,
// and captures family/storage facts. The caller transfers this unique owner
// to the HAL device only after complete device construction. Failure leaves
// out_context NULL and releases acquired resources; native cleanup failures
// are diagnosed and preserve the still-live ownership graph.
iree_status_t iree_hal_amd_gpu_context_allocate(
    iree_hal_amd_fabric_t* fabric, const amdf_endpoint_id_t* endpoint_id,
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_context_t** out_context);

// Selects the existing shared native allocation domain before any backing is
// acquired. Called once by device construction; UINT32_MAX is unspecified NUMA
// placement. Failure leaves no allocation domain attached to the context.
iree_status_t iree_hal_amd_gpu_context_prepare_memory(
    iree_hal_amd_gpu_context_t* context, uint32_t node_id);

// Acquisition storage embedded in the owner being prepared. Completion may
// reclaim or reuse this descriptor; the context join outlives that callback.
typedef struct iree_hal_amd_gpu_memory_acquire_t {
  // Typed native batch and private outputs, borrowed until completion.
  iree_hal_amd_fabric_memory_acquire_request_t request;
  // Native owner borrowed through the callback's final access.
  iree_hal_amd_gpu_context_t* context;
  // Publishes the acquired backing or failure to its actual consumer.
  iree_hal_slab_acquire_callback_t completion;
} iree_hal_amd_gpu_memory_acquire_t;

// Release storage secured alongside acquisition, never allocated at retirement.
typedef struct iree_hal_amd_gpu_memory_release_t {
  // Typed native release and consumed backing array.
  iree_hal_amd_fabric_memory_release_request_t request;
  // Native owner borrowed through the callback's final access.
  iree_hal_amd_gpu_context_t* context;
  // Reclaims retired host metadata after native release has finished.
  iree_hal_slab_release_callback_t completion;
} iree_hal_amd_gpu_memory_release_t;

// A resolved mapping batch borrows its targets until native publication ends.
typedef struct iree_hal_amd_gpu_memory_map_t {
  // Typed mapping request borrowed by the native allocation domain.
  iree_hal_amd_fabric_memory_map_request_t request;
  // Native owner joined after the callback's final publication access.
  iree_hal_amd_gpu_context_t* context;
  // Transfers the native result to the device's mapping completion source.
  iree_hal_slab_acquire_callback_t completion;
} iree_hal_amd_gpu_memory_map_t;

// Submits dependency-ready native mapping. The caller owns the resolved batch
// through completion; final context teardown joins the callback's final access.
void iree_hal_amd_gpu_context_map_memory(
    iree_hal_amd_gpu_context_t* context, iree_hal_memory_mapping_batch_t* batch,
    iree_hal_slab_acquire_callback_t completion,
    iree_hal_amd_gpu_memory_map_t* operation);

// Submits only a ready native backing batch. All options and output storage
// remain live through completion. The callback runs on the allocation domain:
// it publishes results, without submitting GPU work or waiting for
// dependencies. It may run before this call returns and may reclaim/reuse
// operation storage.
void iree_hal_amd_gpu_context_acquire_memory(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    const iree_hal_amd_fabric_memory_request_t* requests,
    iree_hal_amd_fabric_memory_t* memories,
    iree_hal_slab_acquire_callback_t completion,
    iree_hal_amd_gpu_memory_acquire_t* operation);

// Consumes fully retired backing without allocation or native calls on the
// caller. Completion has the same descriptor lifetime contract as acquisition.
void iree_hal_amd_gpu_context_release_memory(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    iree_hal_amd_fabric_memory_t* memories,
    iree_hal_slab_release_callback_t completion,
    iree_hal_amd_gpu_memory_release_t* operation);

// Cold construction/destruction adapters. These wait only for their own native
// batch and belong outside queue publication, retirement, and callback paths.
iree_status_t iree_hal_amd_gpu_context_acquire_memory_sync(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    const iree_hal_amd_fabric_memory_request_t* requests,
    iree_hal_amd_fabric_memory_t* memories);
void iree_hal_amd_gpu_context_release_memory_sync(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    iree_hal_amd_fabric_memory_t* memories);

// Cold ownership join after all producers have stopped and retired owners have
// submitted their releases. Native callbacks may publish nested releases before
// returning. Teardown remains on this caller, never on the allocation worker.
void iree_hal_amd_gpu_context_join_memory(iree_hal_amd_gpu_context_t* context);

// Joins backing callbacks and detaches the allocation domain after every native
// backing owner has submitted release. The empty state is safe for constructor
// rollback and a later device-construction attempt using the same context.
void iree_hal_amd_gpu_context_deinitialize_memory(
    iree_hal_amd_gpu_context_t* context);

// Prepares the device's shared completion wake after selecting publication.
// Called once under device construction ownership, before any queue or worker.
// Partial acquisition remains owned by context on failure.
iree_status_t iree_hal_amd_gpu_context_prepare_completion(
    iree_hal_amd_gpu_context_t* context,
    amdf_queue_publication_modes_t publication_mode);

// Releases completion resources after all native queues and host users join.
// Device construction rollback uses this before returning the context so a
// later construction can select a different publication mode. Failure preserves
// the remaining owner and forbids freeing the borrowed context.
amdf_status_t iree_hal_amd_gpu_context_deinitialize_completion(
    iree_hal_amd_gpu_context_t* context);

// Signals the shared completion observer without allocating or advancing any
// completion frontier. The caller retains context through the final access.
amdf_status_t iree_hal_amd_gpu_context_signal(
    const iree_hal_amd_gpu_context_t* context);

// Releases an idle owner after all queue removals, notification tails, host
// event calls/waits, backing and worker uses have joined. Joins outstanding
// native backing releases before destroying the address domain; performs no
// execution wait or cancellation. Native cleanup failure is diagnosed and
// preserves the remaining ownership graph. NULL is ignored.
void iree_hal_amd_gpu_context_free(iree_hal_amd_gpu_context_t* context);

// Reports a native failure without manufacturing an IREE status to discard.
void iree_hal_amd_gpu_context_report(const iree_hal_amd_gpu_context_t* context,
                                     amdf_status_t status,
                                     const char* operation);

// Reports a captured HAL failure without consuming it. The caller still owns
// propagation to its operation's completion signals.
void iree_hal_amd_gpu_context_report_status(
    const iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    const iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CONTEXT_H_
