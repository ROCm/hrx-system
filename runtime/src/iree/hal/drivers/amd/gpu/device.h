// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_DEVICE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_DEVICE_H_

#include "iree/hal/device.h"
#include "iree/hal/drivers/amd/fabric/asan.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/feedback.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amd_gpu_asan_shadow_backing_e {
  // Local VRAM/HBM keeps instrumented reads on the device's memory interface.
  IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL = 0,
  // Pinned system backing, with host publication before instrumented uses.
  IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_HOST_LOCAL = 1,
} iree_hal_amd_gpu_asan_shadow_backing_t;

// Cold provisioning of a PM4 family with USER or KERNEL publication. Native
// protocol and publication capability determine admission, independently of
// RDNA/CDNA.
typedef struct iree_hal_amd_gpu_device_options_t {
  // Exact native family ordinal, or UINT32_MAX to select the first supported
  // PM4 family. No implicit SDMA queue is provisioned.
  uint32_t queue_family_ordinal;
  // Number of provisioned queues in this family, between 1 and 256.
  uint32_t queue_count;
  // Host NUMA node for both device workers and the shared proactor. UINT32_MAX
  // leaves placement unspecified; GPU memory nodes are not CPU NUMA nodes.
  uint32_t host_numa_node;
  // Initial per-queue storage. Growth is placed on the shared service worker.
  struct {
    // Native command storage capacity in bytes.
    uint64_t ring_byte_length;
    // Power-of-two number of live native completion records.
    uint32_t retirement_capacity;
    // Power-of-two initial native argument-byte capacity.
    uint32_t argument_capacity;
    // Fixed CPU block size for captures and deferred wait records. Construction
    // rejects sizes unable to hold either header; oversized calls allocate.
    iree_host_size_t capture_block_size;
    // Number of blocks preallocated for each queue.
    iree_host_size_t capture_block_count;
  } pm4;
  // Provisioning used when the device's runtime features enable ASAN.
  struct {
    // Placement of physical shadow backing; reservations consume only VA.
    iree_hal_amd_gpu_asan_shadow_backing_t shadow_backing;
    // Effect of a shader access report on subsequent device use.
    iree_hal_amd_gpu_sanitizer_report_policy_t report_policy;
    // Shadow geometry, redzone policy, and freed-backing retention.
    iree_hal_amd_fabric_asan_options_t memory;
  } asan;
} iree_hal_amd_gpu_device_options_t;

// Initializes ordinary provisioning sizes and automatic PM4 family selection.
// The caller may select an exact native family or explicit host NUMA placement.
void iree_hal_amd_gpu_device_options_initialize(
    iree_hal_amd_gpu_device_options_t* out_options);

// Creates a refcounted HAL device with provisioned queues, one completion
// worker, and one shared submission/memory service. Pools remain explicit.
// Success transfers the unique context and returns one device reference.
// Failure leaves out_device NULL and the context with the caller. Failed
// native rollback latches context cleanup before returning an error.
//
// Families, queues, and resources borrow their parent device. The final device
// release closes admission and transfers cleanup to its completion worker;
// accepted operations retain their normal dependency semantics. An unresolved
// external dependency may therefore keep the owner alive. Native failure never
// permits releasing storage whose native uses have not been proven complete.
iree_status_t iree_hal_amd_gpu_device_create(
    iree_hal_amd_gpu_context_t* context, iree_string_view_t display_name,
    const iree_hal_amd_gpu_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_DEVICE_H_
