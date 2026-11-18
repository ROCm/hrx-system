// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_STAGING_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_STAGING_H_

#include "iree/hal/driver/amd/gpu/context.h"
#include "iree/hal/utils/file_request.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_file_staging_t iree_hal_amd_gpu_file_staging_t;
typedef struct iree_hal_amd_gpu_file_staging_bucket_t
    iree_hal_amd_gpu_file_staging_bucket_t;

typedef struct iree_hal_amd_gpu_file_staging_slot_t
    iree_hal_amd_gpu_file_staging_slot_t;

// One placed loan or final registration-release notification. A reservation
// borrows one physical lane until return; it performs no file I/O or device
// copy. The enclosing file operation owns the bucket membership independently
// of its loans, so replacing rows or files never detaches a live registration.
typedef struct iree_hal_amd_gpu_file_staging_request_t {
  // Previous request awaiting backing or final foreign registration detach.
  struct iree_hal_amd_gpu_file_staging_request_t* previous;
  // Next request awaiting the same resource, or NULL.
  struct iree_hal_amd_gpu_file_staging_request_t* next;
  // Borrowed device-owned storage, dominated by the enclosing operation.
  iree_hal_amd_gpu_file_staging_t* staging;
  // Borrowed exact proactor's views, kept live by the enclosing operation.
  iree_hal_amd_gpu_file_staging_bucket_t* bucket;
  // Borrowed physical lane, non-NULL only between reservation and return.
  iree_hal_amd_gpu_file_staging_slot_t* slot;
  // Native I/O span, valid only while slot is borrowed.
  iree_async_span_t span;
  // Queue address of span byte zero, valid while slot is borrowed.
  uint64_t address;
  struct {
    // Acquire before a queue copy consumes native file-read output.
    iree_hal_memory_effects_t read;
    // Release before native file writes consume queue-copy output.
    iree_hal_memory_effects_t write;
  } effects;
  // Reservation or final release notification, invoked outside the lock.
  iree_hal_file_transfer_callback_t completion;
  // Reservation/release failure transferred by the notification.
  iree_status_t status;
  // Eligible block bits; zero while awaiting backing or attachment.
  uint32_t block_mask;
  // Minimum lane capacity for the queued reservation, in bytes.
  uint32_t chunk_capacity;
} iree_hal_amd_gpu_file_staging_request_t;

// Resolves file-capable system backing without acquiring a native queue.
// Absence is reported through out_available; native query failures propagate.
// The profile is consumed directly by allocate on the same context. Staging
// bytes are accessed by native file I/O and the queue, without a CPU access
// site or virtual aliases. Their visibility follows FILE_IO/DEVICE pair
// recipes.
iree_status_t iree_hal_amd_gpu_file_staging_query_profile(
    iree_hal_amd_gpu_context_t* context, amdf_memory_profile_t* out_profile,
    bool* out_available);

// Consumes a resolved file-capable profile and allocates only metadata.
// Physical blocks prepare I/O registrations for both buffered and DIRECT files.
// Payload storage is acquired lazily and shared across file proactors. All
// constructing, live, and retiring blocks share the native-aligned byte budget
// for lane_count * lane_capacity. Per-proactor views add no physical payload.
// The device proactor owns canonical preparation and cold result publication.
// The HAL device outlives this engine. Cached backing owns no device reference;
// parent file captures also borrow the device. Destruction
// joins asynchronous detach and native release before releasing metadata.
iree_status_t iree_hal_amd_gpu_file_staging_allocate(
    iree_hal_device_t* device, iree_hal_amd_gpu_context_t* context,
    uint32_t family_ordinal, const amdf_memory_profile_t* profile,
    bool queue_cache_control, iree_async_proactor_t* proactor,
    uint32_t lane_count, uint32_t lane_capacity, uint32_t numa_node,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_file_staging_t** out_staging);

// Qualifies staging availability and file alignment before touching a prefix.
// Copies use the same queue as public buffer access; no cross-family ownership
// transition is needed. Direct host ranges are unchanged.
iree_status_t iree_hal_amd_gpu_file_staging_prepare(
    iree_hal_amd_gpu_file_staging_t* staging, uint64_t alignment);

// Maximum bytes in one physical loan, established at device construction.
uint32_t iree_hal_amd_gpu_file_staging_capacity(
    const iree_hal_amd_gpu_file_staging_t* staging);

// Number of useful concurrent lanes for a nonempty byte extent. This bounds
// caller metadata independently of file size and does not acquire storage.
uint32_t iree_hal_amd_gpu_file_staging_concurrency(
    const iree_hal_amd_gpu_file_staging_t* staging, uint64_t length);

// Acquires one use of an exact I/O proactor's views. Independent loans share
// this membership without acquiring additional references. The caller retains
// its file through release; native backing and attachments are demand-driven.
iree_status_t iree_hal_amd_gpu_file_staging_acquire(
    iree_hal_amd_gpu_file_staging_t* staging, iree_async_proactor_t* proactor,
    iree_hal_amd_gpu_file_staging_bucket_t** out_bucket);

// Reserves one lane of up to length bytes, at least alignment bytes. The caller
// already qualified nonzero length and file alignment. Success publishes span,
// address, and visibility effects through request before invoking completion.
// The callback may run before return. Only one reservation may be outstanding
// on request; returning its lane permits reuse with any acquired bucket.
void iree_hal_amd_gpu_file_staging_reserve(
    iree_hal_amd_gpu_file_staging_bucket_t* bucket, uint64_t length,
    uint64_t alignment, iree_hal_file_transfer_callback_t completion,
    iree_hal_amd_gpu_file_staging_request_t* request);

// Returns a lane after every accepted I/O/copy using it has retired. Native
// backing remains cached, and another unrelated request may use it immediately.
void iree_hal_amd_gpu_file_staging_return(
    iree_hal_amd_gpu_file_staging_request_t* request);

// Releases one membership after every loan borrowing it returns. Consumes
// status and delivers it after relinquishing foreign-proactor registrations.
// Completion may run before return and permits the caller to stop polling its
// file owner. The request storage, which may reuse a returned loan, remains
// live through this final completion.
void iree_hal_amd_gpu_file_staging_release(
    iree_hal_amd_gpu_file_staging_bucket_t* bucket, iree_status_t status,
    iree_hal_file_transfer_callback_t completion,
    iree_hal_amd_gpu_file_staging_request_t* request);

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

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_STAGING_H_
