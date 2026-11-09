// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_QUEUE_H_

#include "iree/hal/driver/amd/gpu/completion.h"
#include "iree/hal/driver/amd/gpu/feedback.h"
#include "iree/hal/driver/amd/gpu/profile.h"
#include "iree/hal/driver/amd/gpu/publication.h"
#include "iree/hal/driver/amd/gpu/queue/allocation.h"
#include "iree/hal/driver/amd/gpu/queue/aql/program.h"
#include "iree/hal/driver/amd/gpu/queue/backing.h"
#include "iree/hal/driver/amd/gpu/queue/file_staging.h"
#include "iree/hal/driver/amd/gpu/queue/memory_mapping.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact AQL family construction, independent of the physical shader ISA.
typedef struct iree_hal_amd_gpu_aql_queue_options_t {
  // Native family selected by the device.
  uint32_t family_ordinal;
  // Stable native frontier coordinate owned by the device.
  uint32_t queue_ordinal;
  // Stable service source, independent of the native frontier coordinate.
  uint32_t source_ordinal;
  // Last completed epoch from an earlier owner of this coordinate.
  uint64_t initial_epoch;
  // Installed native coordinate extent, including previously used slots.
  uint32_t queue_count;
  // Device-owned native cells and return wake target, live beyond this queue.
  iree_hal_amd_gpu_signal_cache_t* signals;
  // Power-of-two native packet storage extent in bytes.
  uint64_t command_byte_capacity;
  // Borrowed coherent host-readable/writable control and argument policy.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed local private-storage policy with no host mapping.
  const iree_hal_amd_fabric_memory_options_t* scratch_memory;
  // Borrowed resident placement and native builtins shared by peer queues.
  iree_hal_amd_gpu_aql_program_pool_t* programs;
  // Existing device deferred publisher.
  iree_hal_amd_gpu_publication_t* publication;
  // Shared queue-ordered allocation owner.
  iree_hal_amd_gpu_allocation_context_t* allocations;
  // Shared queue-ordered mapping admission.
  iree_hal_amd_gpu_memory_mapping_context_t* mappings;
  // Device-owned direct-I/O fallback through independent native copy queues.
  iree_hal_amd_gpu_file_staging_t* file_staging;
  // Optional device-owned shader channel, drained before public completion.
  iree_hal_amd_gpu_feedback_t* feedback;
  // Power-of-two simultaneously published completion capacity.
  uint32_t retirement_capacity;
  // Initial argument generation capacity, in power-of-two bytes.
  uint32_t argument_capacity;
  // Reusable deferred capture block extent in bytes.
  iree_host_size_t capture_block_size;
  // Initial number of reusable host capture blocks.
  iree_host_size_t capture_block_count;
} iree_hal_amd_gpu_aql_queue_options_t;

// Returns supported publication modes from the passive native family contract.
// AQL requires host publication, device-produced resident packets, and native
// scratch service. Shader architecture does not determine packet support.
amdf_queue_publication_modes_t iree_hal_amd_gpu_aql_queue_publication_modes(
    const amdf_queue_family_info_t* family);

// Creates a refcounted queue borrowing its family, context, and service owners.
// The parent registers publication/completion sources before exposing it and
// joins accepted work before removing those sources. Cold construction acquires
// backing on the calling thread. Failure leaves out_queue NULL.
iree_status_t iree_hal_amd_gpu_aql_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_aql_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Native block bound for resident recording, leaving at least half the ring
// for waits and publication packets. Family queues share construction geometry.
uint32_t iree_hal_amd_gpu_aql_queue_maximum_program_packet_count(
    iree_hal_queue_t* queue);

// Called with idle queues and submission externally serialized. Joins final
// host retirement before native timestamp configuration; native failure makes
// the queue terminal. The device owns the session; captures borrow it until
// their publication has finished.
iree_status_t iree_hal_amd_gpu_aql_queue_set_profile(
    iree_hal_queue_t* queue, iree_hal_amd_gpu_profile_t* profile);

struct iree_hal_amd_gpu_aql_submission_t;

// Records failure without releasing accepted capture or native ownership.
// Consumes status; the existing retirement path publishes final failure.
void iree_hal_amd_gpu_aql_queue_fail_submission(
    iree_hal_queue_t* queue,
    struct iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t status);

// Records a native terminal failure and wakes existing admission/retirement.
// Accepted native uses remain unproven until the normal failure join completes.
void iree_hal_amd_gpu_aql_queue_fail_native(iree_hal_queue_t* queue,
                                            amdf_status_t status,
                                            const char* operation);

// Wakes pending admission after the device latches a logical shader failure.
// Accepted native work retains its resources until actual completion.
void iree_hal_amd_gpu_aql_queue_notify_failure(iree_hal_queue_t* queue);

// Borrows a device-registered completion axis until queue destruction. The
// device retains the coordinate across replacement; NULL clears the borrow.
void iree_hal_amd_gpu_aql_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_aql_queue_publication_source(iree_hal_queue_t* queue);
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_aql_queue_completion_source(iree_hal_queue_t* queue);

// Coalesced, nonblocking trim of unused host/backing storage. Native scratch
// detachment is a separate cold idle operation because it enters the kernel.
void iree_hal_amd_gpu_aql_queue_trim(iree_hal_queue_t* queue);

// Cold public trim caller attempts exclusive idle publication, joins the final
// consumed packet, and detaches native scratch. Busy queues retain their
// binding. Native release is asynchronous; acquisition may resume after
// publication handoff. Failure is terminal and preserves unproven backing.
amdf_status_t iree_hal_amd_gpu_aql_queue_trim_scratch(iree_hal_queue_t* queue);

// Extends host dependency metadata. The lifecycle caller serializes requests
// for this queue, owns its lifetime, and prepares memory outside publication.
// Typed publisher turns copy/install the prepared tables without allocating.
// Native work and backing acquisition may remain outstanding. notification
// belongs to the device and stays live through callback tails. Partial growth
// leaves existing queues usable; new coordinates are admitted only on success.
iree_status_t iree_hal_amd_gpu_aql_queue_reserve_frontiers(
    iree_hal_queue_t* queue, uint32_t queue_count,
    iree_notification_t* notification);

// Arms synchronous final-release ownership after both worker sources are live.
// The callback joins accepted work and native ownership before returning true;
// false preserves this queue when native quiescence cannot be proven.
void iree_hal_amd_gpu_aql_queue_set_release(iree_hal_queue_t* queue,
                                            bool (*release)(void*,
                                                            iree_hal_queue_t*),
                                            void* user_data);

// Completed coordinate after accepted work and callback tails have joined.
uint64_t iree_hal_amd_gpu_aql_queue_completed_epoch(iree_hal_queue_t* queue);

// Closes public acceptance; accepted dependencies keep their normal meaning.
void iree_hal_amd_gpu_aql_queue_close(iree_hal_queue_t* queue);

// After external callers join, proves all accepted host/native uses retired.
bool iree_hal_amd_gpu_aql_queue_is_idle(iree_hal_queue_t* queue);

// After close/failure, joins CPU publishers, scratch/growth callbacks, and host
// borrowers without requiring native capture retirement. Native access and
// capture lifetimes remain unproven until successful inactivation.
bool iree_hal_amd_gpu_aql_queue_is_host_idle(iree_hal_queue_t* queue);

// Attempts exclusive publication ownership and stops native access while
// retaining mappings and captures. False leaves out_status unchanged when a
// CPU publisher still owns the queue. True supplies the native stop result;
// failure preserves backing. The parent closes/fails admission and excludes
// completion observers before calling this cold operation.
bool iree_hal_amd_gpu_aql_queue_try_inactivate(iree_hal_queue_t* queue,
                                               amdf_status_t* out_status);

// Releases issued captures with failure after every affected native queue has
// stopped. No completion signal, timestamp, download, or successful frontier is
// consumed. The parent exclusively owns completion; false means a remaining
// CPU publisher must join before retrying this cold lifetime operation.
bool iree_hal_amd_gpu_aql_queue_retire_failed(iree_hal_queue_t* queue,
                                              iree_status_code_t code);

// Cold native tail/mapping/queue teardown after this queue is idle and its
// service/completion sources and native callbacks have joined. Signal readers
// may remain on other queues; their device-owned backing stays live. Failure
// keeps unproven native users and backing alive; the parent preserves the
// context.
amdf_status_t iree_hal_amd_gpu_aql_queue_shutdown(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_QUEUE_H_
