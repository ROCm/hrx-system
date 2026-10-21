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
#include "iree/hal/drivers/amd/gpu/tsan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amd_gpu_asan_shadow_backing_e {
  // Prefer host-publishable local VRAM/HBM; otherwise use system storage.
  IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL = 0,
  // Pinned system backing, with host publication before instrumented uses.
  IREE_HAL_AMD_GPU_ASAN_SHADOW_BACKING_HOST_LOCAL = 1,
} iree_hal_amd_gpu_asan_shadow_backing_t;

// Cold provisioning of a native packet family. Protocol and publication
// capability determine admission, independently of RDNA/CDNA.
typedef struct iree_hal_amd_gpu_device_options_t {
  // Exact native family ordinal, or UINT32_MAX to prefer AQL on CDNA and PM4
  // on RDNA among supported families. File transfers independently select SDMA.
  uint32_t queue_family_ordinal;
  // Number of provisioned queues in this family, between 1 and 256.
  uint32_t queue_count;
  // Host NUMA node for both device workers and the shared proactor. UINT32_MAX
  // leaves placement unspecified; GPU memory nodes are not CPU NUMA nodes.
  uint32_t host_numa_node;
  // Active observation budget after real completion progress, in nanoseconds.
  // Zero uses native events immediately. Applies only to coherent USER queue
  // records; KERNEL queues retain native event/status observation.
  uint32_t completion_spin_ns;
  // Initial per-queue storage. Native growth uses the fabric allocation owner.
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
  } queue;
  // Provisioning used when the device's runtime features enable ASAN.
  struct {
    // Placement of physical shadow backing; reservations consume only VA.
    iree_hal_amd_gpu_asan_shadow_backing_t shadow_backing;
    // Effect of a shader access report on subsequent device use.
    iree_hal_amd_gpu_sanitizer_report_policy_t report_policy;
    // Shadow geometry, redzone policy, and freed-backing retention.
    iree_hal_amd_fabric_asan_options_t memory;
  } asan;
  // Provisioning used when the device's runtime features enable TSAN.
  struct {
    // Effect of a shader race report on subsequent device use.
    iree_hal_amd_gpu_sanitizer_report_policy_t report_policy;
    // Queue-local workgroup shadow geometry.
    iree_hal_amd_gpu_tsan_options_t memory;
  } tsan;
} iree_hal_amd_gpu_device_options_t;

// Initializes ordinary provisioning sizes and automatic PM4 family selection.
// The caller may select an exact native family or explicit host NUMA placement.
void iree_hal_amd_gpu_device_options_initialize(
    iree_hal_amd_gpu_device_options_t* out_options);

// Queries whether a passive endpoint advertises a native queue family usable
// by default device construction. Reads cached native descriptors without
// creating a device, queue, event, or allocation. Native construction still
// validates installed services and resource availability.
iree_status_t iree_hal_amd_gpu_device_query_support(const amdf_api_t* api,
                                                    amdf_endpoint_t* endpoint,
                                                    bool* out_supported);

// Creates a refcounted HAL device with provisioned queues, one completion
// worker and one deferred publisher. Blocking allocation uses the fabric's
// shared NUMA allocation owner; file I/O uses its proactor. Pools remain
// explicit. Consumes the unique context on every outcome. Success returns one
// device reference; failure leaves out_device NULL. Native rollback failure
// preserves ownership whose retirement cannot be proven.
//
// Families, queues, and resources borrow their parent device. The final device
// release joins workers and native cleanup on the releasing application thread.
// The application completes work and releases child resources first. Native
// failure never permits releasing storage whose uses have not been proven
// complete.
iree_status_t iree_hal_amd_gpu_device_create(
    iree_hal_amd_gpu_context_t* context, iree_string_view_t display_name,
    const iree_hal_amd_gpu_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_DEVICE_H_
