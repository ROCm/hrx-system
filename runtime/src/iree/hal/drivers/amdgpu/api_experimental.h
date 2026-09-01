// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_API_EXPERIMENTAL_H_
#define IREE_HAL_DRIVERS_AMDGPU_API_EXPERIMENTAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Temporary native AMDGPU bridge for libhrx HIP execution contexts.
//
// This restricted API accepts only an exact native in-process AMDGPU logical
// device. It has no replay, remoting, or cross-backend representation and is
// not a supported HAL extension. The AMDGPU queue replacement is expected to
// delete it rather than preserve its shape.

// Native queue topology required to provision an exact HSA queue mask.
typedef struct iree_hal_amdgpu_experimental_execution_queue_topology_t {
  // First physical queue ordinal reserved for caller-managed execution queues.
  iree_host_size_t first_private_physical_queue_ordinal;
  // Number of consecutive caller-managed physical queue ordinals.
  iree_host_size_t private_physical_queue_count;
  // Number of native compute units addressable by an HSA queue mask.
  uint32_t native_compute_unit_count;
  // Native mask group size and alignment in bits. Groups begin at bit zero and
  // must each be wholly enabled or wholly disabled.
  uint32_t native_compute_unit_mask_alignment;
  // Number of hardware partitions interleaved across native mask bits. Native
  // bit N belongs to partition N modulo this count, and a confining mask must
  // enable at least one compute unit in every partition.
  uint32_t native_compute_unit_mask_partition_count;
} iree_hal_amdgpu_experimental_execution_queue_topology_t;

// Queries the fixed queue range and execution-unit topology for one physical
// device. |device| must be an exact native in-process AMDGPU logical device.
// The returned facts are immutable for the logical-device lifetime.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_experimental_execution_queue_query(
    iree_hal_device_t* device, iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_experimental_execution_queue_topology_t* out_topology);

// Binds one caller-selected fixed physical execution queue to an immutable
// native compute-unit mask.
//
// |physical_device_ordinal| selects an exact physical device in |device|.
// |physical_queue_ordinal| selects an exact queue in the private range returned
// by iree_hal_amdgpu_experimental_execution_queue_query. AMDGPU does not choose
// a queue, share equal masks, or recycle the queue for another mask.
//
// The mask must describe the selected physical device's complete native
// execution-unit domain. Each alignment-sized group reported by the topology,
// beginning at native bit zero, must be wholly enabled or wholly disabled. The
// queue is created normally and the mask is applied before the queue is
// published to any submitter. ROCr may reject masks it cannot represent
// exactly. In particular, each interleaved hardware partition reported by the
// topology must retain at least one enabled compute unit; some KFD targets
// interpret an all-zero per-partition mask as unconfined. A process-wide
// HSA_CU_MASK is incompatible with this exact-mask interface. Any failure
// destroys the unpublished queue. On success |out_queue_affinity| is the stable
// private affinity for that physical queue. The queue remains bound to the mask
// until device teardown; once configuration succeeds, every later attempt for
// that queue fails, including one with an equal mask. Failed creation leaves
// the slot unconfigured and may be retried. The caller owns all mapping,
// sharing, capacity, and reuse policy above this boundary.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_experimental_execution_queue_configure(
    iree_hal_device_t* device, iree_host_size_t physical_device_ordinal,
    iree_host_size_t physical_queue_ordinal,
    iree_host_size_t native_mask_bit_count, const uint32_t* native_mask,
    iree_hal_queue_affinity_t* out_queue_affinity);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_API_EXPERIMENTAL_H_
