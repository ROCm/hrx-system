// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_STATE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_STATE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/cpu_data.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/dispatch.h"
#include "iree/hal/driver/amd/gpu/executable.h"
#include "iree/hal/driver/amd/gpu/queue/buffer.h"
#include "iree/hal/driver/amd/gpu/queue/capture_pool.h"
#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/host_call.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/growth.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/retirement.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/submission.h"
#include "iree/hal/driver/amd/gpu/queue/program.h"
#include "iree/hal/driver/amd/gpu/queue/sample_storage.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_pm4_queue_t iree_hal_amd_gpu_pm4_queue_t;

// One cache line of host retirement data per native publication. A capture
// can publish several pieces without overwriting a still-live piece's credit.
typedef struct iree_hal_amd_gpu_pm4_retirement_t {
  // Owned aligned capture pointer with bit zero marking terminal publication.
  // The tag reuses alignment bits so the entire action remains one cache line.
  iree_alignas(64) uintptr_t tagged_submission;
  // Reclaimable command prefix: USER release end or KERNEL chunk end.
  uint64_t consumed_words;
  // Exclusive argument prefix reusable after this host action.
  uint64_t consumed_arguments;
  // Optional bounded host copy, completed before returning staging credit.
  iree_hal_amd_gpu_download_t download;
  // Replaced scratch generations dominated by this FIFO completion.
  iree_hal_amd_gpu_pm4_scratch_storage_t* scratch;
} iree_hal_amd_gpu_pm4_retirement_t;
static_assert(iree_alignof(iree_hal_amd_gpu_pm4_submission_t) >= 2,
              "submission alignment reserves the terminal pointer bit");
static_assert(sizeof(iree_hal_amd_gpu_pm4_retirement_t) == 64,
              "retirement records occupy one cache line");

static inline iree_hal_amd_gpu_pm4_submission_t*
iree_hal_amd_gpu_pm4_retirement_submission(
    const iree_hal_amd_gpu_pm4_retirement_t* action) {
  return (iree_hal_amd_gpu_pm4_submission_t*)(action->tagged_submission &
                                              ~(uintptr_t)1);
}

static inline bool iree_hal_amd_gpu_pm4_retirement_is_terminal(
    const iree_hal_amd_gpu_pm4_retirement_t* action) {
  return (action->tagged_submission & 1) != 0;
}

// A ready single-piece dispatch borrows its capture from the reserved native
// retirement slot. Deferred or larger captures retain independent pool storage.
// Reuse follows completed.count, published only after final resource release.
typedef struct iree_hal_amd_gpu_pm4_retirement_slot_t {
  // Native retirement action, initialized without overwriting capture storage.
  iree_hal_amd_gpu_pm4_retirement_t action;
  union {
    // Exclusively held through publication, including KERNEL capacity deferral,
    // and final retirement. The action's alignment also aligns this span.
    uint8_t capture[448];
    // Transfer payload prefix reusable after native access and the host copy.
    // Transfer captures always use the pool, leaving inline storage available.
    uint64_t consumed_transfers;
  };
} iree_hal_amd_gpu_pm4_retirement_slot_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_retirement_slot_t) == 512,
              "retirement slots have a power-of-two stride");
static_assert(offsetof(iree_hal_amd_gpu_pm4_retirement_slot_t, capture) %
                      iree_alignof(iree_hal_amd_gpu_pm4_submission_t) ==
                  0,
              "retirement actions preserve capture alignment");

struct iree_hal_amd_gpu_pm4_queue_t {
  // Refcounted exact HAL queue; its family/device remain borrowed.
  iree_hal_queue_t base;
  // Allocator for the wrapper, cold backing owners, and oversized captures.
  iree_allocator_t host_allocator;
  // Borrowed native domain and event owner.
  iree_hal_amd_gpu_context_t* context;
  // Immutable native construction and memory facts.
  iree_hal_amd_gpu_pm4_queue_options_t options;
  // Optional device membership join, armed only after successful construction.
  struct {
    // Synchronously joins this queue before its host wrapper is freed.
    bool (*fn)(void* user_data, iree_hal_queue_t* queue);
    // Borrowed device-owned slot, live through final device destruction.
    void* user_data;
  } release;
  // Optional device-owned session, changed only under idle begin/end.
  iree_hal_amd_gpu_profile_t* profile;
  // Persistent native boundary submission borrowed by host calls.
  iree_hal_amd_gpu_host_call_executor_t host_executor;
  // Persistent services borrowed by file transfers.
  iree_hal_amd_gpu_file_executor_t file_executor;
  // Group-assigned completed progress; immutable after the first submission.
  struct {
    // Borrowed device-owned tracker, or NULL before group construction.
    iree_async_frontier_tracker_t* tracker;
    // Registered queue identity; epochs are native publication positions + 1.
    iree_async_axis_t axis;
  } frontier;
  // Sole caller/deferred native publication owner.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_hal_amd_gpu_publisher_t publisher;
  // Reusable host captures, independent of native ring capacity.
  iree_hal_amd_gpu_capture_pool_t captures;
  // Required sidecar bytes, including one tracker, computed at construction.
  iree_host_size_t pending_size;
  // Typed native acquisition and coalesced waiting captures.
  iree_hal_amd_gpu_pm4_growth_context_t growth;
  // Coalesced explicit trim request, consumed under publication ownership.
  struct {
    // Stable typed record; only the first requester links it into the FIFO.
    iree_hal_amd_gpu_pm4_entry_t entry;
    // One while the trim entry is enqueued or executing.
    iree_atomic_int32_t pending;
  } trim;
  // Parent has closed public acceptance; accepted dependencies remain live.
  // Admission facts stay separate from publication, retirement, and callbacks.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t closed;
  // Sticky raw native failure; failure never supplies reclamation credit.
  iree_atomic_int64_t native_failure;
  // Timestamp-session native owner, independent of launch argument capacity.
  iree_hal_amd_gpu_sample_storage_t* samples;
  // Tracker associations shared by cold waits on this queue.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_async_semaphore_wait_context_t wait_context;
  // Deferred admission and its independent callback-return lifetime.
  struct {
    // Deferred unpublished captures and admission callbacks. Native pieces
    // use the published/completed prefixes instead of modifying this count.
    iree_atomic_int64_t host_owners;
    // Failure-cancellable waits, independent of publication ownership.
    iree_hal_amd_gpu_host_wait_set_t waits;
    // Joins each callback's final count drop and shutdown notification. Neither
    // wait registration nor resumed publication holds this cold mutex.
    iree_slim_mutex_t completion_mutex;
  } pending;
  // Issued captures have recorded the native error after publication joined.
  // Public failure waits for native stop and each capture's final release.
  bool issued_failure_delivered;
  // Completion-owned delivery of logical failure while native retirement runs.
  bool feedback_failure_delivered;
  // Native queue and mapping, joined after the final notification tail.
  iree_hal_amd_gpu_user_queue_t storage;
  // Finite command owner, present only for KERNEL publication.
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel;
  // Exclusive native publication state and MMIO operands.
  struct {
    // Primary command ring, accessed only by the current publisher.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_hal_amd_gpu_pm4_ring_t ring;
    // Naturally aligned native monotonic DWORD index.
    iree_atomic_int64_t* write_index;
    // Write-only 64-bit MMIO doorbell.
    volatile uint64_t* doorbell;
    // Current argument generation, owned by the publisher.
    iree_hal_amd_gpu_byte_storage_t* arguments;
    // Replaced generations ordered by their final consumed byte position.
    iree_hal_amd_gpu_byte_storage_t* retired_arguments;
    // Constant-time append boundary; NULL when no generation awaits retirement.
    iree_hal_amd_gpu_byte_storage_t* retired_arguments_tail;
    // Accepted native publication count, release-published for completion.
    iree_atomic_int64_t count;
    // Next native dispatch ID, independent of publication chunk boundaries.
    uint64_t dispatch_count;
    // Dense causal epochs over the device's allocated native coordinates.
    struct {
      // Installed coordinate capacity in this context's signal domain.
      uint32_t count;
      // Accepted FIFO prefix, indexed by native queue ordinal.
      uint64_t* published;
      // Current candidate, aliasing published when no waits are added.
      uint64_t* pending;
      // Second inline vector for candidates containing new native waits.
      uint64_t* scratch;
    } frontier;
    // Lazy private storage; only the exact publisher mutates active ownership.
    iree_hal_amd_gpu_pm4_scratch_t scratch;
    // DMA payload generations, independent of launch argument capacity.
    struct {
      // Lazily acquired current generation, or NULL before host transport.
      iree_hal_amd_gpu_byte_storage_t* current;
      // Replaced generations in final consumed-byte order.
      iree_hal_amd_gpu_byte_storage_t* retired;
      // Constant-time append boundary, or NULL when retired is empty.
      iree_hal_amd_gpu_byte_storage_t* tail;
    } transfers;
  } producer;
  // Native waiters leave metadata eligible under the same exclusive publisher.
  struct {
    // Oldest waiter; cold idle queries observe it independently of the FIFO.
    iree_atomic_intptr_t first;
    // Last native waiter for constant-time publisher-owned append.
    iree_hal_amd_gpu_publication_entry_t* last;
    // Coalesced retirement/failure continuation in the normal ready FIFO.
    iree_hal_amd_gpu_pm4_entry_t entry;
    // One while the continuation is queued for the publisher.
    iree_atomic_int32_t pending;
  } capacity;
  // Single completion observer's release-published capacity credit.
  struct {
    // Number of consumed retirement records.
    iree_alignas(iree_hardware_destructive_interference_size)
        iree_atomic_int64_t count;
    // Reclaimable command prefix established by the last retired action.
    iree_atomic_int64_t words;
    // Last consumed argument position, including wrap/alignment.
    iree_atomic_int64_t arguments;
    // Last consumed transfer payload position, after any host download copy.
    iree_atomic_int64_t transfers;
    // Publisher enrolled for capacity return. Only a capacity-blocked
    // publisher writes this observer-owned line; retirement consumes it once.
    iree_atomic_int32_t waiting_for_capacity;
  } completed;
  // Native host-visible records, immutable after preparation.
  iree_hal_amd_fabric_memory_t control;
  // Device-only barrier state reused by ordered direct cooperative grids.
  iree_hal_amd_gpu_grid_storage_t grid;
  // Indexed native record geometry shared by publisher and observer.
  iree_hal_amd_gpu_pm4_retirement_ring_t retirement;
  // Fixed host actions and inline captures parallel to native slots.
  // GPU-carried pointers name these actions directly on normal retirement.
  iree_hal_amd_gpu_pm4_retirement_slot_t* issued;
};

static_assert(
    sizeof(((iree_hal_amd_gpu_pm4_queue_t*)0)->completed) ==
        iree_hardware_destructive_interference_size,
    "completion credit and enrollment occupy one observer cache line");

// Frees joined queue metadata after the device releases native ownership.
void iree_hal_amd_gpu_pm4_queue_destroy(iree_hal_queue_t* queue);

// Records issued failures without publishing public completion.
// The completion owner excludes every other capture observer and has joined
// publication. Native access may still be live; this grants no lifetime credit.
void iree_hal_amd_gpu_pm4_queue_fail_issued(iree_hal_amd_gpu_pm4_queue_t* queue,
                                            iree_status_code_t code);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_STATE_H_
