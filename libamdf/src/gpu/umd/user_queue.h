// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_USER_QUEUE_H_
#define AMDF_SRC_GPU_UMD_USER_QUEUE_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_umd_user_queue_t amdf_gpu_umd_user_queue_t;
typedef struct amdf_gpu_umd_user_queue_mapping_t
    amdf_gpu_umd_user_queue_mapping_t;

// Queue-device scratch range resolved by the shared GPU layer. The caller
// keeps its backing live until a scratch request or successful destruction.
typedef struct amdf_gpu_umd_queue_scratch_t {
  // Queue-device address of the first scratch byte, or zero when disabled.
  uint64_t device_address;
  // Scratch backing length in bytes, or zero when disabled.
  uint64_t byte_length;
  // Maximum private-segment bytes per workitem accepted by the queue.
  uint32_t maximum_private_segment_byte_length;
  // Physical scratch slot capacity, including shader-engine rounding.
  uint32_t maximum_wave_count;
} amdf_gpu_umd_queue_scratch_t;

// Validated properties used to create one native user queue.
typedef struct amdf_gpu_umd_user_queue_create_info_t {
  // Native command representation selected from the endpoint profile.
  amdf_queue_command_type_t command_type;
  // Version defining command and publication semantics.
  uint32_t format_version;
  // Requested native scheduling priority.
  amdf_queue_priority_t priority;
  // Required primary-ring locality, or DEFAULT for the native policy.
  amdf_queue_placement_t placement;
  // Selected reservation protocol.
  amdf_queue_producer_mode_t producer_mode;
  // Enables native dispatch timestamps in each AQL completion signal.
  bool dispatch_timestamps;
  // Enables caller servicing of native scratch requests.
  bool scratch_service;
  // Optional native wake operands for scratch requests and queue errors.
  amdf_gpu_event_info_t inactive_event;
  // Direct producer operations required by the caller.
  amdf_user_queue_capabilities_t required_capabilities;
  // Semantic operations advertised for the selected family.
  amdf_queue_roles_t roles;
  // Requested ring capacity, or zero for native policy.
  uint64_t ring_byte_length;
  // Optional queue-device scratch range.
  amdf_gpu_umd_queue_scratch_t scratch;
} amdf_gpu_umd_user_queue_create_info_t;

// Immutable facts published after native queue construction succeeds.
typedef struct amdf_gpu_umd_user_queue_result_t {
  // Opaque identity of the live native queue.
  amdf_queue_id_t queue_id;
  // Direct producer operations achieved by this queue.
  amdf_user_queue_capabilities_t capabilities;
  // Achieved primary-ring locality guarantee.
  amdf_queue_placement_t placement;
  // Actual primary ring capacity in bytes.
  uint64_t ring_byte_length;
  // Actual metadata sidecar capacity in bytes, or zero when absent.
  uint64_t metadata_ring_byte_length;
} amdf_gpu_umd_user_queue_result_t;

// Producer-local addresses published after native mapping succeeds.
typedef struct amdf_gpu_umd_user_queue_mapping_result_t {
  // Producer-local base address of the writable primary ring.
  uint64_t ring_address;
  // Producer-local address of the consumer-owned read index.
  uint64_t read_index_address;
  // Producer-local address of the producer-owned write index.
  uint64_t write_index_address;
  // Producer-local address of the write-only notification doorbell.
  uint64_t doorbell_address;
  // Storage width of both queue indices.
  uint32_t index_bits;
  // Atomic write width of the notification aperture.
  uint32_t doorbell_bits;
  // Release executor before publishing a valid header or producer index.
  amdf_cache_transition_executor_t host_publish_executor;
  // Host fence for HOST_DIRECT; NONE for other executors.
  amdf_host_cache_fence_t host_publish_fence;
  // Producer-local base address of the writable metadata ring.
  uint64_t metadata_ring_address;
} amdf_gpu_umd_user_queue_mapping_result_t;

// Creates a complete native user queue. Failure releases or retains every
// partial resource internally and leaves both outputs unchanged.
amdf_status_t amdf_gpu_umd_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result);

// Maps one queue into the host when `producer_device` is NULL, or into the
// exact producer device otherwise. Failure leaves both outputs unchanged.
// A host view borrows queue-owned mappings without allocation or native calls.
// The common mapping owner keeps the queue live for the borrow's lifetime.
amdf_status_t amdf_gpu_umd_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result);

// Releases one producer-local native queue mapping.
amdf_status_t amdf_gpu_umd_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping);

// Completes host stores through a mapping admitted for HOST_API publication.
// Native failure is terminal and sticky, never completion or reuse permission.
amdf_status_t amdf_gpu_umd_user_queue_mapping_flush(
    amdf_gpu_umd_user_queue_mapping_t* mapping);

// Samples native publication, consumption, and sticky terminal state.
amdf_status_t amdf_gpu_umd_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status);

// Samples a pending request; zero byte_length means none. Does not enter the
// kernel or allocate. The caller serializes this with scratch installation.
amdf_status_t amdf_gpu_umd_user_queue_query_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    amdf_gpu_aql_scratch_request_t* out_request);

// Installs backing into a paused scratch-service queue and resumes firmware.
amdf_status_t amdf_gpu_umd_user_queue_resume_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    const amdf_gpu_umd_queue_scratch_t* scratch);

// Detaches scratch from a caller-quiesced service queue. Native failure is
// terminal and preserves the caller's backing obligation through destruction.
amdf_status_t amdf_gpu_umd_user_queue_trim_scratch(
    amdf_gpu_umd_user_queue_t* queue);

// Selects timestamp writes on a caller-quiesced AQL queue. Native failure is
// terminal and preserves borrowed resources through successful destruction.
amdf_status_t amdf_gpu_umd_user_queue_set_dispatch_timestamps(
    amdf_gpu_umd_user_queue_t* queue, bool enabled);

// Waits for one already-published format index under `deadline`.
amdf_status_t amdf_gpu_umd_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t* deadline);

// Rejects unconsumed publication with API-domain BUSY and no native mutation.
// Every other result consumes the queue, returning native cleanup failure and
// preserving any backing whose safe release cannot be established. No native
// identifier is retried, regardless of whether the driver consumed it.
amdf_status_t amdf_gpu_umd_user_queue_destroy(amdf_gpu_umd_user_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_USER_QUEUE_H_
