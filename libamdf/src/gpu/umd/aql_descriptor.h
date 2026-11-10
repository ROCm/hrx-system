// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_AQL_DESCRIPTOR_H_
#define AMDF_SRC_GPU_UMD_AQL_DESCRIPTOR_H_

#include <stddef.h>

#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/umd/user_queue.h"

// Native AMD queue descriptor consumed by CP firmware and kernel queue-pointer
// SGPRs. This is the fixed 256-byte prefix of amd_queue_t/amd_queue_v2_t;
// asynchronous scratch reclamation is not enabled. Storage is 64-byte aligned.
typedef struct amdf_gpu_aql_descriptor_t {
  // HSA queue type: zero for multi-producer, one for single, two cooperative.
  uint32_t queue_type;
  // Native HSA queue features (kernel dispatch).
  uint32_t features;
  // GPU address of the first 64-byte packet slot.
  uint64_t ring_address;
  // HSA doorbell signal handle; zero for host-only raw doorbell publication.
  uint64_t doorbell_signal;
  // Power-of-two packet capacity.
  uint32_t packet_count;
  // Reserved HSA queue field.
  uint32_t reserved_queue;
  // Native descriptor identity; independent of the public opaque queue ID.
  uint64_t id;
  // Firmware capabilities; software asynchronous reclaim remains disabled.
  uint32_t capabilities;
  // Reserved native descriptor fields.
  uint32_t reserved_capabilities[3];
  // Producer reservation frontier in packets.
  uint64_t write_dispatch_id;
  // High word of the native flat LDS aperture.
  uint32_t group_segment_aperture_base_hi;
  // High word of the native flat scratch aperture.
  uint32_t private_segment_aperture_base_hi;
  // Highest CU identifier used by firmware scratch allocation.
  uint32_t maximum_compute_unit_id;
  // Highest wave identifier within one CU.
  uint32_t maximum_wave_id;
  // Unused legacy doorbell protocol storage.
  uint64_t reserved_legacy_dispatch_id;
  // Unused legacy doorbell lock and reserved words.
  uint32_t reserved_legacy[10];
  // Firmware-owned monotonic consumption frontier in packets.
  uint64_t read_dispatch_id;
  // Offset of read_dispatch_id from the descriptor base.
  uint32_t read_dispatch_id_byte_offset;
  // COMPUTE_TMPRING_SIZE encoding for the currently bound scratch.
  uint32_t compute_temporary_ring_size;
  // Native scratch buffer resource descriptor.
  uint32_t scratch_resource_descriptor[4];
  // GPU base address of the complete scratch allocation across all XCCs.
  uint64_t scratch_backing_address;
  // V2 scratch reclamation storage, unused by this owner.
  uint64_t reserved_scratch_byte_length;
  // Scratch capacity per lane of a 64-lane wave.
  uint32_t scratch_wave64_lane_byte_length;
  // Native properties, including 64-bit pointers and dispatch timestamps.
  uint32_t queue_properties;
  // V2 scratch reclamation frontier, unused by this owner.
  uint64_t reserved_scratch_frontier;
  // GPU address of the queue-owned native inactive signal block.
  uint64_t inactive_signal_address;
  // Reserved tail of the fixed descriptor prefix.
  uint32_t reserved_tail[14];
} amdf_gpu_aql_descriptor_t;

// Native AMD signal prefix. CP writes value directly; a zero mailbox disables
// interrupt notification. Storage is 64-byte aligned and lives with the queue.
typedef struct amdf_gpu_aql_signal_t {
  // AMD_SIGNAL_KIND_USER is one.
  int64_t kind;
  // Firmware-owned inactive/error value, initially zero.
  uint64_t value;
  // Optional native event mailbox; zero for polled status.
  uint64_t event_mailbox_address;
  // Optional native event identifier; zero without a mailbox.
  uint32_t event_id;
  // Reserved native signal field.
  uint32_t reserved_event;
  // Optional dispatch start timestamp, unused for this signal.
  uint64_t start_timestamp;
  // Optional dispatch end timestamp, unused for this signal.
  uint64_t end_timestamp;
  // Native doorbell queue pointer, unused for a user signal.
  uint64_t reserved_queue_address;
  // Reserved native signal tail.
  uint64_t reserved_tail;
} amdf_gpu_aql_signal_t;

// Target descriptor geometry shared by native and simulated AMD queues.
typedef struct amdf_gpu_aql_descriptor_profile_t {
  // Number of equal scratch partitions addressed by the command processors.
  uint32_t xcc_count;
  // Highest physical CU identifier accepted by the firmware descriptor.
  uint32_t maximum_compute_unit_id;
  // Highest resident wave identifier within one CU.
  uint32_t maximum_wave_id;
  // Flat-address apertures established by the process owner.
  struct {
    // High 32 bits of the LDS aperture base.
    uint32_t group_base_hi;
    // High 32 bits of the private aperture base.
    uint32_t private_base_hi;
  } apertures;
  // Fixed scratch slot provisioning and target register encodings.
  struct {
    // Physical slots per XCC, including shader-engine rounding.
    uint32_t slot_count_per_xcc;
    // COMPUTE_TMPRING_SIZE.WAVES, in the target's scheduling unit.
    uint32_t temporary_ring_wave_count;
    // Log2 byte granularity of COMPUTE_TMPRING_SIZE.WAVESIZE.
    uint32_t wave_size_shift;
    // Largest supported scratch allocation in bytes per 64-lane wave.
    uint32_t maximum_wave_byte_length;
    // Address-free buffer descriptor template; WORD2 capacity is initially 0.
    uint32_t resource_descriptor[4];
  } scratch;
  // Native firmware signal block byte offset in control storage.
  size_t inactive_signal_byte_offset;
} amdf_gpu_aql_descriptor_profile_t;

#ifdef __cplusplus
extern "C" {
#endif

// Qualifies externally discovered compute geometry for descriptor construction.
// Apertures are supplied by the process owner. Failure leaves output unchanged.
bool amdf_gpu_aql_descriptor_profile_initialize(
    const amdf_gpu_endpoint_properties_t* properties, uint32_t group_base_hi,
    uint32_t private_base_hi, amdf_gpu_aql_descriptor_profile_t* out_profile);

// Encodes the caller's initial scratch before any native allocation. Failure is
// a nonrepresentable or unsupported caller request and leaves output unchanged.
// The remaining descriptor fields are populated after addresses are assigned.
amdf_status_t amdf_gpu_aql_descriptor_initialize(
    const amdf_gpu_aql_descriptor_profile_t* profile, uint64_t ring_byte_length,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_aql_descriptor_t* out_descriptor);

// Samples the firmware's inactive signal and paused packet range. Ring points
// to the host view of the primary ring. A zero request means no scratch fault;
// other firmware errors remain errors. The caller serializes scratch service.
amdf_status_t amdf_gpu_aql_scratch_query(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    const amdf_gpu_aql_descriptor_t* descriptor, const void* ring,
    amdf_gpu_aql_scratch_request_t* out_request);

// Resolves the target's complete physical scratch requirement for a firmware
// request. Input private bytes originate in the caller-published dispatch.
amdf_status_t amdf_gpu_aql_scratch_requirements(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    uint32_t private_segment_byte_length, uint32_t wavefront_size,
    amdf_gpu_aql_scratch_request_t* out_request);

// Encodes a validated target wave size and caller backing into only the
// descriptor's scratch fields. Failure leaves the descriptor unchanged.
amdf_status_t amdf_gpu_aql_scratch_initialize(
    const amdf_gpu_aql_descriptor_profile_t* profile,
    const amdf_gpu_umd_queue_scratch_t* scratch, uint64_t wave_byte_length,
    amdf_gpu_aql_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_AQL_DESCRIPTOR_H_
