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

// Queue-device scratch range resolved and retained by the shared GPU layer.
typedef struct amdf_gpu_umd_queue_scratch_t {
  // Queue-device address of the first scratch byte, or zero when disabled.
  uint64_t device_address;
  // Scratch backing length in bytes, or zero when disabled.
  uint64_t byte_length;
  // Maximum private-segment bytes per workitem accepted by the queue.
  uint32_t maximum_private_segment_byte_length;
  // Number of simultaneously scratch-backed waves.
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
  // Selected reservation protocol.
  amdf_queue_producer_mode_t producer_mode;
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
amdf_status_t amdf_gpu_umd_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result);

// Releases one producer-local native queue mapping.
amdf_status_t amdf_gpu_umd_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping);

// Samples native publication, consumption, and sticky terminal state.
amdf_status_t amdf_gpu_umd_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status);

// Waits for one already-published format index under `deadline`.
amdf_status_t amdf_gpu_umd_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t* deadline);

// Releases an idle native user queue. A consumed native identity retains its
// storage until target retirement or reset recovery proves the device no
// longer references it.
amdf_status_t amdf_gpu_umd_user_queue_destroy(amdf_gpu_umd_user_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_USER_QUEUE_H_
