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
#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
  // Cold native ownership, following the compact invocation metadata.
  iree_hal_amd_fabric_memory_t memory;
} iree_hal_amd_gpu_pm4_program_instance_t;

// Resident program storage embedded in its owning HAL command buffer. The
// owner retains the device and all native operands; leases retain that owner.
// Metadata selection/release is hot. Allocation, trim, and final teardown run
// on the shared device service, outside this mutex and after native use joins.
typedef struct iree_hal_amd_gpu_pm4_program_t {
  // Borrowed native identity, dominated by the command buffer's device.
  const iree_hal_amd_gpu_context_t* context;
  // Host allocator for the bounded instance table and cold native owners.
  iree_allocator_t host_allocator;
  // Captured native placement policy; pointed-to accesses outlive this owner.
  iree_hal_amd_fabric_memory_options_t memory;
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
  // Immutable table in the initial instance's allocation, after copied bytes.
  uint64_t fixup_address;
} iree_hal_amd_gpu_pm4_program_t;

// Allocates the initial placement and uploads the complete sealed program plus
// fixup table once. The input is trusted recorder output. The initial instance
// is also the permanent clone source; its immutable bytes never change.
// Native failures leave a deinitializable partial owner. Runs on service.
iree_status_t iree_hal_amd_gpu_pm4_program_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* memory, uint32_t queue_count,
    uint32_t alignment, iree_const_byte_span_t contents, uint32_t fixup_count,
    const iree_hal_amd_gpu_pm4_fixup_t* fixups, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_t* out_program);

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
