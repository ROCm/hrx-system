// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_DISPATCH_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared prefix of protocol-specific placed functions. The executable owns
// reflection and argument layout through every captured native execution.
typedef struct iree_hal_amd_gpu_function_t {
  // Complete cold reflection and resource constraints owned by the image.
  const iree_hal_amd_gpu_metadata_kernel_t* metadata;
  // Immutable argument packing and public parameter reflection.
  iree_hal_amd_gpu_kernarg_layout_t* kernarg;
} iree_hal_amd_gpu_function_t;

// Admitted launch geometry; protocol resource admission happens separately.
typedef struct iree_hal_amd_gpu_dispatch_geometry_t {
  // Work-items in each complete workgroup.
  uint32_t workgroup_size[3];
  // Number of dispatched workgroups, including a possible partial group.
  uint32_t workgroup_count[3];
  // Final group dimensions, all zero when every group is complete.
  uint32_t partial_size[3];
  // Fixed and dynamic LDS allocation in bytes.
  uint32_t group_segment_byte_length;
} iree_hal_amd_gpu_dispatch_geometry_t;

// Admits external geometry and argument representation against trusted image
// facts. workitem_grid requires the product of group size and count to fit the
// native 32-bit workitem grid. The protocol first admits required services and
// resource limits. Success initializes out_dispatch; failure leaves it
// unchanged.
iree_status_t iree_hal_amd_gpu_dispatch_prepare(
    const iree_hal_amd_gpu_function_t* function,
    uint32_t maximum_group_segment_byte_length, bool workitem_grid,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_dispatch_geometry_t* out_dispatch);

// Validates caller buffer ranges after argument representation is admitted.
// Queue-ordered backing need not have committed. Recording accumulates binding
// requirements separately; indirect native arguments cover the complete ABI.
iree_status_t iree_hal_amd_gpu_dispatch_validate_direct_bindings(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_hal_buffer_ref_t parameters, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags);

// Writes admitted dense or native explicit arguments, hidden geometry, and
// padding. External native arguments bypass host writes. target covers the
// layout's complete byte length and does not overlap inputs. No allocation,
// validation, native call, or packet access occurs.
void iree_hal_amd_gpu_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_function_t* function,
    const iree_hal_amd_gpu_dispatch_geometry_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_DISPATCH_H_
