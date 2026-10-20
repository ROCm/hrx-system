// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PROGRAM_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/fixup.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_program_t iree_hal_amd_gpu_program_t;
typedef struct iree_hal_amd_gpu_program_growth_t
    iree_hal_amd_gpu_program_growth_t;

// Device-owned native program storage. Recorders acquire backing and register
// sealed programs on their callers. Invocation growth and retired releases use
// the fabric allocation domain asynchronously. Submission reads prepared launch
// and placement facts. No worker belongs to this pool.
typedef struct iree_hal_amd_gpu_program_pool_t {
  // Borrowed native device dominated by the containing HAL device.
  iree_hal_amd_gpu_context_t* context;
  // Allocator for program metadata and recording-time storage.
  iree_allocator_t host_allocator;
  // Persistent access requirements borrowed by both native policies.
  amdf_memory_device_access_t access;
  // Cached host-visible upload and immutable builtin instruction placement.
  iree_hal_amd_fabric_memory_options_t upload;
  // Physical-local placement when advertised, otherwise cached system memory.
  iree_hal_amd_fabric_memory_options_t resident;
  // Protects registered-program membership during explicit trim and removal.
  iree_slim_mutex_t registry_mutex;
  // Registered programs, protected by registry_mutex.
  iree_hal_amd_gpu_program_t* head;
  // Immutable number of provisioned queues in this native domain.
  uint32_t queue_count;
} iree_hal_amd_gpu_program_pool_t;

// Captures placement for one native device; failure leaves a
// deinitializable partial owner.
iree_status_t iree_hal_amd_gpu_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_program_pool_t* out_pool);
void iree_hal_amd_gpu_program_pool_trim(iree_hal_amd_gpu_program_pool_t* pool);
void iree_hal_amd_gpu_program_pool_deinitialize(
    iree_hal_amd_gpu_program_pool_t* pool);

// One independently writable native program placement. Pool ownership keeps
// the backing alive; invocation leases cover every native command/data reader.
typedef struct iree_hal_amd_gpu_program_instance_t {
  // Native address of instance byte zero.
  uint64_t address;
  // Last ordered writer's accepted publication epoch, protected by mutex.
  uint64_t last_epoch;
  // Number of accepted or preparing invocations, including causal reuses.
  uint32_t use_count;
  // Stable slot in the program's small instance table.
  uint32_t ordinal;
  // Native-domain queue index of the last ordered writer.
  uint32_t last_queue_ordinal;
  // Immutable bytes are initialized, or a causal predecessor will initialize
  // them before this instance is next entered.
  bool initialized;
  // One publisher owns an unaccepted invocation. No causal proof can cover
  // that invocation until publication installs its writer coordinate.
  bool reserved;
  // A same-FIFO predecessor may still read private data before host retirement.
  bool join_previous;
  // Cold native ownership, following the compact invocation metadata.
  iree_hal_amd_fabric_memory_t memory;
  // Presecured native release, independent of the containing command buffer.
  iree_hal_amd_gpu_memory_release_t release;
  // Device allocator owning this instance through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_program_instance_t;

typedef struct iree_hal_amd_gpu_program_upload_t
    iree_hal_amd_gpu_program_upload_t;

// Stable state belonging to one exact publisher, beside its hot candidate.
typedef struct iree_hal_amd_gpu_program_queue_t {
  // Preferred physical instance for direct steady-state FIFO reuse.
  iree_hal_amd_gpu_program_instance_t* preferred;
  // Coalesced native acquisition, accessed only by this queue's publisher.
  iree_hal_amd_gpu_program_growth_t* growth;
  // Availability epoch observed by this publisher's most recent miss.
  uint64_t observed_epoch;
} iree_hal_amd_gpu_program_queue_t;

// Resident program storage owned independently from caller resources. The
// owner borrows the device and outlives native growth and invocation leases.
// Metadata selection/release is hot. Native backing acquisition and release
// run outside the metadata mutex; last invocation release never calls a driver.
struct iree_hal_amd_gpu_program_t {
  // Borrowed device-owned placement and registry.
  iree_hal_amd_gpu_program_pool_t* pool;
  // Previous registered program, protected by pool->registry_mutex.
  iree_hal_amd_gpu_program_t* previous;
  // Next registered program, protected by pool->registry_mutex.
  iree_hal_amd_gpu_program_t* next;
  // Optional staging when physical-local storage has no host view. Retained
  // through every initial copy reader, then released by an idle trim.
  iree_hal_amd_gpu_program_upload_t* upload;
  // Host view used once by the recorder before sealing.
  iree_byte_span_t contents;
  // GPU source for cold initialization before the first instance retires.
  uint64_t upload_address;
  // Initial native immutable bytes are complete and can serve other queues.
  iree_atomic_int32_t source_ready;
  // Protects instance selection, table growth, coordinates, and lease counts.
  iree_slim_mutex_t mutex;
  // Advances when return/cancellation can enable another queue's invocation.
  // Own-queue publication cannot race that queue's miss/enrollment interval.
  uint64_t availability_epoch;
  // Enrolled clone requests only; no instance or queue scan during release.
  iree_hal_amd_gpu_program_growth_t* waiters;
  // Stable initial source, independent of instance-table growth. Owned until
  // destruction; invocation emission reads it without a metadata lock.
  iree_hal_amd_gpu_program_instance_t* initial;
  // Number of allocated instance pointer slots, grown under mutex.
  uint32_t capacity;
  // Growable pointer table; NULL entries are unallocated or explicitly trimmed.
  iree_hal_amd_gpu_program_instance_t** instances;
  // Stable directly indexed publisher state and initial metadata allocation.
  iree_hal_amd_gpu_program_queue_t* queues;
  // Resident command/private-data extent copied into new instances.
  uint32_t byte_length;
  // Required power-of-two alignment of every instance's base.
  uint32_t alignment;
  // Immutable fixup count, independent of the invocation's value count.
  uint32_t fixup_count;
  // Immutable table offset in the initial instance and optional upload.
  uint32_t fixup_offset;
};

// Allocates an initial placement and writable upload view. The recorder writes
// commands/data at contents.data and the immutable table at fixup_offset, then
// seals. No native queue is blocked for upload: an unmapped local instance is
// initialized by the first invocation's ordered device copy. Called only by
// the recording owner, outside native publication and retirement.
iree_status_t iree_hal_amd_gpu_program_initialize(
    iree_hal_amd_gpu_program_pool_t* pool, uint32_t alignment,
    uint32_t byte_length, uint32_t fixup_count,
    iree_hal_amd_gpu_program_t* out_program);

// Flushes the completed upload and registers the immutable program. Failure
// leaves the owner deinitializable; no invocation can observe it before
// success.
iree_status_t iree_hal_amd_gpu_program_seal(
    iree_hal_amd_gpu_program_t* program);

// Immutable native sources selected once for an invocation. Before the initial
// local copy retires, other queues use the retained upload. Thereafter they
// copy the permanent resident instance, including while its mutable words
// change.
typedef struct iree_hal_amd_gpu_program_source_t {
  // Base of the initialized commands/private data.
  uint64_t address;
  // Shared immutable fixup table, never modified by invocation fixups.
  uint64_t fixups;
} iree_hal_amd_gpu_program_source_t;
iree_hal_amd_gpu_program_source_t iree_hal_amd_gpu_program_source(
    const iree_hal_amd_gpu_program_t* program);

// Reserves an idle or causally reusable instance exclusively until acceptance
// or rejection. Frontier includes the exact native waits to be emitted before
// this invocation, plus the queue's published FIFO prefix. It contains capacity
// publication epochs indexed by the device's immutable queue ordinals, so
// dominance of the last writer is a single indexed comparison.
// NULL requests cold growth; it never waits for completion or allocates. The
// caller retains its lease across native publication retries and releases it
// on rejection or proven completion. Copied mutable bytes need no source wait.
iree_hal_amd_gpu_program_instance_t* iree_hal_amd_gpu_program_try_acquire(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal,
    const uint64_t* frontier);

// Caller-owned native acquisition, borrowed until completion. The queue uses
// queues[queue_ordinal].growth to coalesce its captures behind one request.
// No per-invocation state or resource ownership is added to the native ring.
struct iree_hal_amd_gpu_program_growth_t {
  // Borrowed program retained independently of any waiting invocations.
  iree_hal_amd_gpu_program_t* program;
  // Reserved instance; NULL when an idle instance made acquisition unnecessary.
  iree_hal_amd_gpu_program_instance_t* instance;
  // Exact publisher owning this program's pending-growth entry.
  uint32_t queue_ordinal;
  // Link into the enrolled-only list; NULL when no reuse notification is armed.
  iree_hal_amd_gpu_program_growth_t** previous_wait;
  // Next enrolled growth operation, protected by the program mutex.
  iree_hal_amd_gpu_program_growth_t* next_wait;
  // Bounded notification that only enqueues its retained publication record.
  void (*notify_reuse)(void* user_data);
  // Borrowed notification target, joined by cancellation or grow_end.
  void* reuse_user_data;
  // Native policy with host mapping removed for device-only clone storage.
  iree_hal_amd_fabric_memory_options_t options;
  // Acquisition geometry borrowed by the native allocation domain.
  iree_hal_amd_fabric_memory_request_t request;
  // Embedded native request and context lifetime join.
  iree_hal_amd_gpu_memory_acquire_t acquire;
};

// Reserves an instance and starts native acquisition without waiting. The
// requesting queue has no pending growth or unpublished instance reservation.
// Success transfers operation storage through callback, which may run before
// this returns. Failure starts no callback. The callback publishes the result
// to the requesting queue; it never emits GPU work on the allocation worker.
iree_status_t iree_hal_amd_gpu_program_grow_begin(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal,
    iree_hal_amd_fabric_memory_callback_t callback,
    iree_hal_amd_gpu_program_growth_t* operation);

// Arms one reuse hint after a failed acquisition. Epoch comparison closes the
// miss/enrollment race without another instance walk. notify runs under the
// program mutex and may only retain/enqueue its own record: no program access,
// blocking, native call, allocation, or GPU publication. The owner retains the
// program and operation independently of any invocations released by the hint.
void iree_hal_amd_gpu_program_grow_await_reuse(
    iree_hal_amd_gpu_program_growth_t* operation,
    void (*notify)(void* user_data), void* user_data);

// Disarms the hint and joins its final callback access. Already-enqueued
// records retain their own storage. The requesting queue's publisher owns this
// call.
void iree_hal_amd_gpu_program_grow_cancel_reuse(
    iree_hal_amd_gpu_program_growth_t* operation);

// Consumes native completion on the requesting queue's publisher, clears its
// pending slot, and either installs the instance or releases failed backing.
// Native address query reads the acquired memory's prepared address. No native
// allocation/free or wait occurs here. The caller owns operation storage again.
iree_status_t iree_hal_amd_gpu_program_grow_end(
    iree_hal_amd_gpu_program_growth_t* operation, iree_status_t status);

// Marks the first copy as ordered before the leased instance's last writer. The
// caller has passed every fallible admission step and publishes this state
// before exposing native completion. This does not claim host completion.
void iree_hal_amd_gpu_program_publish(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance, uint32_t queue_ordinal,
    uint64_t epoch);

// Cancels an unaccepted reservation, preserving the last published coordinate
// and initialization state. No later invocation can depend on rejected work.
void iree_hal_amd_gpu_program_cancel(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance);

// Returns a published lease after native command fetch and data use join.
// Ordinary release is constant-time; a last use additionally notifies only
// enrolled clone demand. This operation never frees native memory.
void iree_hal_amd_gpu_program_release(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance);

// Releases idle spare instances without waiting. The initial instance remains
// available as the clone source and owns the shared fixup table. Native release
// uses storage secured with each instance; this call performs no driver work.
void iree_hal_amd_gpu_program_trim(iree_hal_amd_gpu_program_t* program);

// Releases all backing after the containing command buffer's last invocation.
// No native call, GPU wait, or allocation is hidden in teardown.
void iree_hal_amd_gpu_program_deinitialize(iree_hal_amd_gpu_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PROGRAM_H_
