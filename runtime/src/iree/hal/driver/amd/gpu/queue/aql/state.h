// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_STATE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_STATE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/executable.h"
#include "iree/hal/driver/amd/gpu/queue/aql/growth.h"
#include "iree/hal/driver/amd/gpu/queue/aql/queue.h"
#include "iree/hal/driver/amd/gpu/queue/aql/ring.h"
#include "iree/hal/driver/amd/gpu/queue/aql/scratch.h"
#include "iree/hal/driver/amd/gpu/queue/aql/transfer.h"
#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/host_call.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/sample_storage.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_aql_queue_t iree_hal_amd_gpu_aql_queue_t;

// Retirement credit owns both the native point and the inline host capture.
// Firmware consumption alone never returns this storage.
typedef struct iree_hal_amd_gpu_aql_retirement_t {
  // Capture dominated by the native point, immutable after publication.
  iree_alignas(64) iree_hal_amd_gpu_aql_submission_t* submission;
  // Independent native reader/completion point for this chunk.
  iree_hal_amd_gpu_signal_t* signal;
  // Argument prefix retired with this native completion.
  uint64_t arguments;
  // Final chunk releases invocation payload and publishes public completion.
  bool terminal;
  union {
    // Ready direct calls retain resources without another pool access.
    iree_alignas(8) uint8_t capture[480];
    // Transfer-only ownership leaves the ordinary inline capture unchanged.
    struct {
      // Payload prefix retired after native use and any host completion copy.
      uint64_t end;
      // Host destinations completed before returning payload credit.
      iree_hal_amd_gpu_download_t download;
    } transfer;
    // Recorded chunks own sample ranges independently of the shared capture.
    iree_hal_amd_gpu_profile_completion_t profile;
  };
} iree_hal_amd_gpu_aql_retirement_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_retirement_t) == 512,
              "AQL retirement has a power-of-two stride");

struct iree_hal_amd_gpu_aql_queue_t {
  // Refcounted HAL queue; family/device are borrowed.
  iree_hal_queue_t base;
  // Allocator owning this wrapper and cold captures.
  iree_allocator_t host_allocator;
  // Native device/address/event owner.
  iree_hal_amd_gpu_context_t* context;
  // Immutable construction policies.
  iree_hal_amd_gpu_aql_queue_options_t options;
  // Optional device membership join, armed only after successful construction.
  struct {
    // Synchronously joins this queue before its host wrapper is freed.
    bool (*fn)(void* user_data, iree_hal_queue_t* queue);
    // Borrowed device-owned slot, live through final device destruction.
    void* user_data;
  } release;
  // Native queue and independently mapped host/device operands.
  iree_hal_amd_gpu_user_queue_t storage;
  // Device-only state for memory-strategy cooperative grids; empty for GWS.
  iree_hal_amd_gpu_grid_storage_t grid;
  // Device-owned completion facts, borrowed until final queue cleanup.
  struct {
    // Borrowed tracker or NULL before group assignment.
    iree_async_frontier_tracker_t* tracker;
    // Exact queue completion axis.
    iree_async_axis_t axis;
  } frontier;
  // Current publication owner; producers and the service executor share it.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_hal_amd_gpu_publisher_t publisher;
  // Deferred captures and typed acquisition result ownership.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Native boundary executor borrowed by accepted host callbacks.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Native barriers/transfers borrowed by the shared direct-I/O/SDMA pipeline.
  iree_hal_amd_gpu_file_executor_t file_executor;
  // Device-owned optional session, changed only with idle queues.
  iree_hal_amd_gpu_profile_t* profile;
  // Timestamp-session native owner, independent of launch argument capacity.
  iree_hal_amd_gpu_sample_storage_t* samples;
  // Fixed sidecar extent computed at construction.
  iree_host_size_t pending_size;
  // Native backing acquisition, independent of execution publication.
  iree_hal_amd_gpu_aql_growth_context_t growth;
  // Demand-driven firmware private memory.
  iree_hal_amd_gpu_aql_scratch_t scratch;
  // Publisher-owned native storage and accepted causal prefix.
  struct {
    // Native slot credit and write-only packet/doorbell operands.
    iree_hal_amd_gpu_aql_ring_t ring;
    // Current argument generation.
    iree_hal_amd_gpu_byte_storage_t* arguments;
    // Replaced generations, newest first; visited only at explicit trim.
    iree_hal_amd_gpu_byte_storage_t* retired_arguments;
    // Lazily acquired host transfer payloads, independent of launch arguments.
    iree_hal_amd_gpu_byte_storage_t* transfers;
    // Replaced payload generations retained through their final host consumer.
    iree_hal_amd_gpu_byte_storage_t* retired_transfers;
    // Dense immutable-domain causal prefix, updated only at acceptance.
    uint64_t* frontier;
    // Frontier width captured at construction.
    uint32_t queue_count;
    // Host-complete publication prefix, released before terminal headers.
    iree_atomic_int64_t count;
  } producer;
  // Completion credit is isolated from ordinary publisher writes.
  struct {
    // Fully retired native prefix, including final resource release.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_atomic_int64_t count;
    // Retired argument prefix including alignment/wrap padding.
    iree_atomic_int64_t arguments;
    // Payload prefix including completed download copies and wrap padding.
    iree_atomic_int64_t transfers;
    // Publisher enrolled for actual completion credit.
    iree_atomic_int32_t waiting;
  } completed;
  // Captures waiting for execution credit, outside the ready FIFO so native
  // scratch results cannot get stuck behind the work requiring that scratch.
  struct {
    // First parked capture, exclusively owned by the publisher.
    iree_hal_amd_gpu_publication_entry_t* first;
    // Last parked capture for constant-time append/requeue.
    iree_hal_amd_gpu_publication_entry_t* last;
    // Stable coalesced capacity-return entry.
    iree_hal_amd_gpu_aql_entry_t entry;
    // One while entry is linked or being consumed.
    iree_atomic_int32_t pending;
  } capacity;
  // Coalesced explicit trim request.
  struct {
    // Stable trim entry.
    iree_hal_amd_gpu_aql_entry_t entry;
    // One while entry is linked or being consumed.
    iree_atomic_int32_t pending;
  } trim;
  // Wait tracker associations used only on dependency deferral.
  iree_async_semaphore_wait_context_t wait_context;
  // Independent callback-return and accepted-host ownership.
  struct {
    // Deferred work plus callback borrowers, separate from native retirement.
    iree_atomic_int64_t host_owners;
    // Failure-cancellable dependency registrations.
    iree_hal_amd_gpu_host_wait_set_t waits;
    // Joins a callback's final count drop and shutdown notification.
    iree_slim_mutex_t completion_mutex;
  } pending;
  // Public acceptance has closed; previously accepted waits remain live.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t closed;
  // Sticky terminal failure; it never supplies resource-reclamation credit.
  iree_atomic_int32_t failure;
  // Issued captures have recorded the native error after publication joined.
  // Public failure waits for native stop and each capture's final release.
  bool failure_delivered;
  // Logical shader failure has been reported without stopping retirement.
  bool feedback_failure_delivered;
  // Host metadata parallel to the finite native execution credit.
  iree_hal_amd_gpu_aql_retirement_t* issued;
};

// Frees joined queue metadata after the device releases native ownership.
void iree_hal_amd_gpu_aql_queue_destroy(iree_hal_queue_t* queue);

// Records issued failures after publication joins without publishing public
// completion. Only the exclusive completion owner inspects issued captures.
void iree_hal_amd_gpu_aql_queue_fail_issued(iree_hal_amd_gpu_aql_queue_t* queue,
                                            iree_status_code_t code);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_STATE_H_
