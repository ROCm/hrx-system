// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_STAGING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_STAGING_H_

#include "iree/hal/drivers/amd/fabric/file_transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_file_staging_t iree_hal_amd_gpu_file_staging_t;
typedef struct iree_hal_amd_gpu_file_staging_bucket_t
    iree_hal_amd_gpu_file_staging_bucket_t;

// Placed staging use for one complete file operation. Physical ranges submit
// serially through this state without releasing its prepared proactor view.
// The parent file capture retains the device and immutable transfer.
// Cursor, membership, status, and in_flight are protected by the staging lock;
// slot I/O and GPU operations run outside that lock.
typedef struct iree_hal_amd_gpu_file_staging_request_t {
  // Previous range awaiting a slot or backing generation, or NULL.
  struct iree_hal_amd_gpu_file_staging_request_t* previous;
  // Next range awaiting the same resource, or NULL.
  struct iree_hal_amd_gpu_file_staging_request_t* next;
  // Borrowed device-owned engine, dominated by the parent file capture.
  iree_hal_amd_gpu_file_staging_t* staging;
  // Exact file-proactor views over device-owned physical staging.
  iree_hal_amd_gpu_file_staging_bucket_t* bucket;
  // Borrowed file and buffer facts, immutable through the entire operation.
  const iree_hal_amd_fabric_file_transfer_t* transfer;
  // Borrowed public outcomes, failed even when native retirement is unproven.
  const iree_hal_semaphore_list_t* signals;
  // Next unassigned transfer-relative byte.
  uint64_t cursor;
  // Exclusive transfer-relative end of this physical range.
  uint64_t end;
  // GPU address corresponding to transfer-relative byte zero.
  uint64_t device_address;
  // Native destination's already-qualified SDMA visibility effects.
  iree_hal_memory_effects_t effects;
  // Terminal callback after every accepted I/O and GPU use has retired.
  iree_hal_amd_fabric_file_transfer_callback_t completion;
  // First terminal failure, owned until final callback delivery.
  iree_status_t status;
  // Active slots plus the submitting caller's temporary ownership.
  uint32_t in_flight;
  // Eligible physical block slots; zero while awaiting backing or attachment.
  uint32_t block_mask;
  // Requested lane capacity, used only for cache selection and growth.
  uint32_t chunk_capacity;
} iree_hal_amd_gpu_file_staging_request_t;

// Selects coherent system backing and allocates only metadata. File-capable
// backing prepares I/O registrations; ordinary backing admits buffered files.
// Payload storage is acquired lazily and shared across file proactors. All
// constructing, live, and retiring blocks share the native-aligned byte budget
// for lane_count * lane_capacity. Per-proactor views add no physical payload.
// The device proactor owns canonical preparation and cold result publication.
// The native SDMA queue and HAL device outlive this engine. Cached backing owns
// no device reference; parent file captures retain the device. Cold destruction
// joins asynchronous detach and native release before releasing metadata.
iree_status_t iree_hal_amd_gpu_file_staging_allocate(
    iree_hal_device_t* device, iree_hal_amd_gpu_context_t* context,
    uint32_t family_ordinal, iree_hal_amd_gpu_sdma_queue_t* queue,
    iree_async_proactor_t* proactor, uint32_t lane_count,
    uint32_t lane_capacity, uint32_t numa_node, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_file_staging_t** out_staging);

// Qualifies each captured physical copy range once, before touching a prefix.
// Combines originating-queue effects into the operation's two boundaries and
// stores copy-engine effects with each range. Direct host ranges are unchanged.
iree_status_t iree_hal_amd_gpu_file_staging_prepare(
    iree_hal_amd_gpu_file_staging_t* staging,
    const iree_hal_queue_family_t* family, uint32_t native_family_ordinal,
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_hal_memory_effects_t* before, iree_hal_memory_effects_t* after);

// Acquires one staging use spanning all physical ranges of a qualified file
// operation. The parent capture retains the device, transfer, and signals
// through release completion. Acquisition allocates only per-proactor metadata;
// native backing and registration attachment are demand-driven by submit.
iree_status_t iree_hal_amd_gpu_file_staging_acquire(
    iree_hal_amd_gpu_file_staging_t* staging,
    const iree_hal_amd_fabric_file_transfer_t* transfer,
    const iree_hal_semaphore_list_t* signals,
    iree_hal_amd_gpu_file_staging_request_t* request);

// Begins one dependency-ready nonempty copy range on an acquired staging use.
// The previous range must have completed. Native I/O uses the file's exact
// proactor; completion leaves its prepared views available for subsequent
// ranges. Completion may run before return; the caller joins its own submission
// borrow.
void iree_hal_amd_gpu_file_staging_submit(
    iree_hal_amd_gpu_file_staging_request_t* request, uint64_t offset,
    uint64_t length, uint64_t device_address, iree_hal_memory_effects_t effects,
    iree_hal_amd_fabric_file_transfer_callback_t completion);

// Releases the complete file operation after every range retires. Consumes
// status and delivers it after relinquishing foreign-proactor registrations.
// Completion may run before return and permits the caller to stop polling its
// file owner. The request storage remains live through this final completion.
void iree_hal_amd_gpu_file_staging_release(
    iree_hal_amd_gpu_file_staging_request_t* request, iree_status_t status,
    iree_hal_amd_fabric_file_transfer_callback_t completion);

// Returns cached backing when whole file operations quiesce without
// invalidating reusable live storage. Already prepared work progresses while
// independent growth is pending. Native release is asynchronous and returns
// physical budget only after every registered view detaches and native
// ownership is relinquished.
void iree_hal_amd_gpu_file_staging_trim(
    iree_hal_amd_gpu_file_staging_t* staging);

// Cold teardown after all file operations have joined. Trims cached backing
// and joins detach/free callbacks. False preserves the owner after a failed
// detach; it never permits releasing memory still registered with the proactor.
bool iree_hal_amd_gpu_file_staging_free(
    iree_hal_amd_gpu_file_staging_t* staging);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_STAGING_H_
