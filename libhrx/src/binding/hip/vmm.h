// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_VMM_H_
#define LIBHRX_SRC_BINDING_HIP_VMM_H_

#include "binding/hip/api.h"

hipError_t iree_hip_vmm_is_supported(int device_ordinal, bool* out_supported);
hipError_t iree_hip_vmm_address_reserve(void** ptr, size_t size,
                                        size_t alignment, void* address,
                                        unsigned long long flags);
hipError_t iree_hip_vmm_address_free(void* device_ptr, size_t size);
hipError_t iree_hip_vmm_create(hipMemGenericAllocationHandle_t* handle,
                               size_t size,
                               const hipMemAllocationProp* properties,
                               unsigned long long flags);
hipError_t iree_hip_vmm_release(hipMemGenericAllocationHandle_t handle);
hipError_t iree_hip_vmm_map(void* ptr, size_t size, size_t offset,
                            hipMemGenericAllocationHandle_t handle,
                            unsigned long long flags);
hipError_t iree_hip_vmm_unmap(void* ptr, size_t size);
hipError_t iree_hip_vmm_set_access(void* ptr, size_t size,
                                   const hipMemAccessDesc* descriptors,
                                   size_t count);
hipError_t iree_hip_vmm_get_access(unsigned long long* flags,
                                   const hipMemLocation* location, void* ptr);
hipError_t iree_hip_vmm_get_allocation_granularity(
    size_t* granularity, const hipMemAllocationProp* properties,
    hipMemAllocationGranularity_flags option);
hipError_t iree_hip_vmm_get_allocation_properties(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle);
hipError_t iree_hip_vmm_retain_allocation_handle(
    hipMemGenericAllocationHandle_t* handle, void* address);

#endif  // LIBHRX_SRC_BINDING_HIP_VMM_H_
