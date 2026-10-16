// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/blit.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"
#include "iree/hal/memory/maintenance.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_pm4_program_t iree_hal_amd_gpu_pm4_program_t;

// Device-owned native program storage. Cold preparation, registration, trim,
// and cleanup run on the existing memory service. Submission only reads the
// prepared launches and placement facts; no worker belongs to this pool.
typedef struct iree_hal_amd_gpu_pm4_program_pool_t {
  // Borrowed native device dominated by the containing HAL device.
  const iree_hal_amd_gpu_context_t* context;
  // Borrowed cold executor, also dominated by the HAL device.
  iree_hal_memory_maintenance_t* service;
  // Allocator for program metadata and recording-time storage.
  iree_allocator_t host_allocator;
  // Persistent access requirements borrowed by both native policies.
  amdf_memory_device_access_t access;
  // Cached host-visible upload and immutable builtin instruction placement.
  iree_hal_amd_fabric_memory_options_t upload;
  // Physical-local placement when advertised, otherwise cached system memory.
  iree_hal_amd_fabric_memory_options_t resident;
  // Owned instruction bytes shared by every recording and invocation.
  iree_hal_amd_fabric_memory_t code;
  // Prepared parallel relocation kernel.
  iree_hal_amd_gpu_pm4_launch_t fixup;
  // Prepared dynamic-extent copy/fill kernel.
  iree_hal_amd_gpu_pm4_launch_t blit;
  // Registered programs; links are accessed only on service.
  iree_hal_amd_gpu_pm4_program_t* head;
  // Immutable number of provisioned queues in this native domain.
  uint32_t queue_count;
  // KERNEL submission enters IB1 and needs an explicit exit continuation.
  bool chain;
} iree_hal_amd_gpu_pm4_program_pool_t;

// Cold initialization for the selected native PM4 family; failure leaves a
// deinitializable partial owner.
iree_status_t iree_hal_amd_gpu_pm4_program_pool_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    iree_hal_memory_maintenance_t* service, uint32_t queue_count, bool chain,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool);
void iree_hal_amd_gpu_pm4_program_pool_trim(
    iree_hal_amd_gpu_pm4_program_pool_t* pool);
void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool);

// A sealed block's native entry. Block identities and extents are independent
// of invocation placement; successor packets and private-data pointers are
// ordinary fixups relative to the selected instance's base. An exit can return
// to a USER ring or chain to an invocation-private KERNEL continuation.
typedef struct iree_hal_amd_gpu_pm4_program_block_t {
  // DWORD-aligned entry relative to instance byte zero.
  uint32_t offset;
  // Native extent including padding before the final exit packet.
  uint32_t word_count;
} iree_hal_amd_gpu_pm4_program_block_t;

// One independently writable native program placement. Pool ownership keeps
// the backing alive; invocation leases cover every native command/data reader.
typedef struct iree_hal_amd_gpu_pm4_program_instance_t {
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
} iree_hal_amd_gpu_pm4_program_instance_t;

// Resident program storage embedded in its owning HAL command buffer. The
// owner retains the device and all native operands; leases retain that owner.
// Metadata selection/release is hot. Allocation, trim, and final teardown run
// on the shared device service, outside this mutex and after native use joins.
struct iree_hal_amd_gpu_pm4_program_t {
  // Borrowed device-owned placement and builtin programs.
  iree_hal_amd_gpu_pm4_program_pool_t* pool;
  // Previous registered program, accessed only on the cold service.
  iree_hal_amd_gpu_pm4_program_t* previous;
  // Next registered program, accessed only on the cold service.
  iree_hal_amd_gpu_pm4_program_t* next;
  // Optional staging when physical-local storage has no host view. Retained
  // through every initial copy reader, then released by an idle trim.
  iree_hal_amd_fabric_memory_t upload;
  // Host view used once by the recorder before sealing.
  iree_byte_span_t contents;
  // GPU source for cold initialization before the first instance retires.
  uint64_t upload_address;
  // Initial native immutable bytes are complete and can serve other queues.
  iree_atomic_int32_t source_ready;
  // Protects instance selection, last-use coordinates, and lease counts only.
  iree_slim_mutex_t mutex;
  // Number of instance slots, bounded by the device's provisioned queues.
  uint32_t capacity;
  // Dense stable slots; NULL entries are unallocated or explicitly trimmed.
  iree_hal_amd_gpu_pm4_program_instance_t** instances;
  // Per-queue preferred candidates; steady FIFO reuse never scans instances.
  iree_hal_amd_gpu_pm4_program_instance_t** preferred;
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
// initialized by the first invocation's ordered device copy. Runs on service.
iree_status_t iree_hal_amd_gpu_pm4_program_initialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool, uint32_t alignment,
    uint32_t byte_length, uint32_t fixup_count,
    iree_hal_amd_gpu_pm4_program_t* out_program);

// Flushes the completed upload and registers the immutable program. Failure
// leaves the owner deinitializable; no invocation can observe it before
// success.
iree_status_t iree_hal_amd_gpu_pm4_program_seal(
    iree_hal_amd_gpu_pm4_program_t* program);

// Immutable native sources selected once for an invocation. Before the initial
// local copy retires, other queues use the retained upload. Thereafter they
// copy the permanent resident instance, including while its mutable words
// change.
typedef struct iree_hal_amd_gpu_pm4_program_source_t {
  // Base of the initialized commands/private data.
  uint64_t address;
  // Shared immutable fixup table, never modified by invocation fixups.
  uint64_t fixups;
} iree_hal_amd_gpu_pm4_program_source_t;
iree_hal_amd_gpu_pm4_program_source_t iree_hal_amd_gpu_pm4_program_source(
    const iree_hal_amd_gpu_pm4_program_t* program);

// Fixed invocation header followed by the dense uint64_t input vector. The
// queue owns this upload until native retirement; program emission initializes
// the builtin arguments after the recorder resolves the input vector.
typedef struct iree_hal_amd_gpu_pm4_program_arguments_t {
  // Independent full-word replacements for this instance.
  iree_hal_amd_gpu_pm4_fixup_arguments_t fixup;
  // Cold initialization or overlap clone, used only for a large native copy.
  iree_hal_amd_gpu_pm4_blit_arguments_t copy;
} iree_hal_amd_gpu_pm4_program_arguments_t;

// Native invocation extent, excluding caller-owned waits, optional padding,
// and retirement. The reservation remains exclusive through emission, so the
// same initialized/join facts govern both sizing and encoding.
uint32_t iree_hal_amd_gpu_pm4_program_word_count(
    const iree_hal_amd_gpu_pm4_program_t* program,
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile);

// Publishes host input, initializes a cold placement, applies all fixups, and
// enters the resident block. Actual producer/consumer cache paths determine
// visibility. Padding precedes the final entry, including terminal CHAIN.
// No command, binding, or fixup entry is walked on the host here.
uint32_t* iree_hal_amd_gpu_pm4_program_emit(
    const iree_hal_amd_gpu_pm4_program_t* program,
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
    iree_hal_amd_gpu_pm4_program_block_t entry,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint64_t argument_address,
    iree_hal_amd_gpu_pm4_program_arguments_t* arguments, uint32_t padding,
    uint32_t* words);

// Reserves an idle or causally reusable instance exclusively until acceptance
// or rejection. Frontier includes the exact native waits to be emitted before
// this invocation, plus the queue's published FIFO prefix. It contains capacity
// publication epochs indexed by the device's immutable queue ordinals, so
// dominance of the last writer is a single indexed comparison.
// NULL requests cold growth; it never waits for completion or allocates. The
// caller retains its lease across native publication retries and releases it
// on rejection or proven completion. Copied mutable bytes need no source wait.
iree_hal_amd_gpu_pm4_program_instance_t*
iree_hal_amd_gpu_pm4_program_try_acquire(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t queue_ordinal,
    const uint64_t* frontier);

// Allocates one unused instance on service. Publication later copies the
// initial instance device-to-device and runs all fixups before entry. No native
// operation or allocation occurs while holding the program metadata mutex.
iree_status_t iree_hal_amd_gpu_pm4_program_grow(
    iree_hal_amd_gpu_pm4_program_t* program);

// Marks the first copy as ordered before the leased instance's last writer. The
// caller has passed every fallible admission step and publishes this state
// before exposing native completion. This does not claim host completion.
void iree_hal_amd_gpu_pm4_program_publish(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance, uint32_t queue_ordinal,
    uint64_t epoch);

// Cancels an unaccepted reservation, preserving the last published coordinate
// and initialization state. No later invocation can depend on rejected work.
void iree_hal_amd_gpu_pm4_program_cancel(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance);

// Returns a published lease after native command fetch and data use join.
// This constant-time operation never frees native memory.
void iree_hal_amd_gpu_pm4_program_release(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance);

// Releases idle spare instances without waiting. The initial instance remains
// available as the clone source and owns the shared fixup table. Runs on
// service.
void iree_hal_amd_gpu_pm4_program_trim(iree_hal_amd_gpu_pm4_program_t* program);

// Releases all backing after the containing command buffer's last invocation.
// Runs on service; no GPU wait or retry is hidden in teardown.
void iree_hal_amd_gpu_pm4_program_deinitialize(
    iree_hal_amd_gpu_pm4_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_
