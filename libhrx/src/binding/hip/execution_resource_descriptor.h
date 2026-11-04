// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_DESCRIPTOR_H_
#define LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_DESCRIPTOR_H_

#include "binding/hip/api.h"
#include "common/execution_resource.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;

// One-shot resource specification consumed by execution-context creation.
//
// The public handle is the address of this allocation but is only dereferenced
// after a live-handle lookup has retained it. All referenced
// execution-resource sets remain owned by the device table for its complete
// incarnation.
typedef struct iree_hip_execution_resource_descriptor_t {
  // Reference count including the ownership represented by the public handle.
  iree_atomic_ref_count_t ref_count;

  // Host allocator owning this descriptor allocation.
  iree_allocator_t host_allocator;

  // Device ordinal encoded by every resource in |resources|.
  iree_host_size_t device_ordinal;

  // Device execution-resource table incarnation owning |sm_resource_set_id|.
  uint64_t table_incarnation;

  // Queue family shared by all SM resources in this descriptor.
  iree_hal_queue_family_ordinal_t queue_family_ordinal;

  // Canonical union of every SM resource in |resources|.
  iree_hal_streaming_execution_resource_set_id_t sm_resource_set_id;

  // Number of copied public resource values in |resources|.
  iree_host_size_t resource_count;

  // Validated resource values with all caller-owned links cleared.
  hipDevResource resources[];
} iree_hip_execution_resource_descriptor_t;

// Creates a one-shot descriptor from disjoint exact resources on |device|.
//
// All resources must resolve to one queue family. Exact resource identity,
// rather than split-call provenance, defines whether the sets can be safely
// combined. |out_descriptor| is unchanged on failure.
hipError_t iree_hip_execution_resource_descriptor_create(
    iree_hal_streaming_device_t* device, const hipDevResource* resources,
    iree_host_size_t resource_count, hipDevResourceDesc_t* out_descriptor);

// Looks up and retains |handle| while it is known to be live. The caller must
// release |*out_descriptor|. |out_descriptor| is unchanged on failure.
bool iree_hip_execution_resource_descriptor_lookup_retain(
    hipDevResourceDesc_t handle,
    iree_hip_execution_resource_descriptor_t** out_descriptor);

// Atomically invalidates |handle| and transfers descriptor ownership to the
// caller. |out_descriptor| is unchanged if the handle is not live.
bool iree_hip_execution_resource_descriptor_take(
    hipDevResourceDesc_t handle,
    iree_hip_execution_resource_descriptor_t** out_descriptor);

// Releases an owning descriptor reference.
void iree_hip_execution_resource_descriptor_release(
    iree_hip_execution_resource_descriptor_t* descriptor);

// Reclaims empty public-handle tables after runtime teardown.
void iree_hip_execution_resource_descriptor_trim(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_DESCRIPTOR_H_
