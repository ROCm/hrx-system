// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_DISPATCH_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/executable.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Admitted dispatch geometry, independent of packet publication mode.
typedef struct iree_hal_amd_gpu_pm4_dispatch_t {
  // Work-items in each complete workgroup.
  uint32_t workgroup_size[3];
  // Number of dispatched workgroups, including a possible partial group.
  uint32_t workgroup_count[3];
  // Final group dimensions, all zero when every group is complete.
  uint32_t partial_size[3];
  // Fixed and dynamic LDS allocation in bytes.
  uint32_t group_segment_byte_length;
} iree_hal_amd_gpu_pm4_dispatch_t;

// One launch, including any device-side ABI copies, fits the smallest queue's
// bounded publication window alongside its dependencies and retirement tail.
enum { IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS = 96 };

// Copies and their visibility boundary precede one four-DWORD MEC dispatch.
static inline uint32_t iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(
    uint16_t copy_count) {
  return copy_count ? (uint32_t)copy_count * 7 + 7 + 8 + 4 : 4;
}

// Admits public launch geometry and argument representation against a prepared
// executable. Does not access buffers, capture caller storage, or resolve
// native addresses. Direct calls and command recording use the same ABI
// contract; their respective public boundaries validate direct or indirect
// references.
iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch);

// Validates direct API buffer ranges after argument representation is admitted.
// Indirect native arguments additionally cover the complete native ABI extent.
// This does not require queue-ordered backing to have committed. Recording uses
// its own accumulated binding requirements instead of this direct-only check.
iree_status_t iree_hal_amd_gpu_pm4_dispatch_validate_direct_bindings(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_hal_buffer_ref_t parameters, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags);

// Writes admitted native kernargs from captured or recording-time inputs. Dense
// arguments consume one resolved address per binding; custom direct arguments
// preserve native explicit offsets. Hidden geometry and padding are initialized
// here. Indirect native arguments bypass emplacement. Target covers the
// layout's byte_length and does not overlap the inputs. This transform cannot
// fail.
void iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target);

// Emits admitted device-time geometry. copy_count is zero for caller-owned
// indirect native arguments; otherwise it is layout->indirect_copy_count.
// Exact count bytes are copied into private kernargs, joined, and made visible
// using the prepared CP-to-shader cache control before the indirect launch.
// Zero base addresses produce relocation templates containing relative offsets.
// The input tuple remains stable through all CP and shader uses.
uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
    uint32_t* words, const iree_hal_amd_gpu_kernarg_layout_t* layout,
    uint16_t copy_count, uint64_t parameters_address, uint64_t argument_address,
    uint32_t cache_control, uint32_t initiator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_DISPATCH_H_
