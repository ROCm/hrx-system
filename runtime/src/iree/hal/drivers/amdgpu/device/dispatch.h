// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_

#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Dispatch Kernarg Layout
//===----------------------------------------------------------------------===//

// Device-visible kernarg byte layout for one dispatch.
//
// This is intentionally a prevalidated data contract instead of a status-
// producing API: once device-side replay is emitting packets there is no sane
// way to report malformed ABI metadata or recover from partially-written
// packet/kernarg storage. Host recording/submission code must validate kernel
// metadata and user-provided arguments before passing a layout here.
typedef struct iree_hal_amdgpu_device_dispatch_kernarg_layout_t {
  // Size in bytes of the explicitly provided dispatch arguments.
  size_t explicit_kernarg_size;
  // Offset in bytes of the implicit HIP/OpenCL suffix, if present.
  size_t implicit_args_offset;
  // Total kernarg reservation size in bytes required for this dispatch.
  size_t total_kernarg_size;
  // True if a HIP/OpenCL implicit args suffix is appended at
  // |implicit_args_offset| and must be populated during emplace.
  bool has_implicit_args;
} iree_hal_amdgpu_device_dispatch_kernarg_layout_t;

//===----------------------------------------------------------------------===//
// Dispatch Packet/Kernarg Emission
//===----------------------------------------------------------------------===//

// Kernel arguments for the builtin indirect-parameter patch dispatch.
typedef struct iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t {
  // Device pointer to a uint32_t[3] workgroup-count parameter buffer.
  const uint32_t* workgroup_count;
  // Device pointer to the AQL dispatch packet to publish after patching.
  iree_hsa_kernel_dispatch_packet_t* dispatch_packet;
  // Optional device pointer to the dispatch's implicit args suffix.
  iree_amdgpu_kernel_implicit_args_t* implicit_args;
  // Final 32-bit header/setup word to publish with a release store.
  uint32_t dispatch_header_setup;
} iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t) == 32,
    "indirect dispatch patch args must match the kernel ABI");

// Kernel arguments for the builtin PM4 command-buffer binding fixup dispatch.
typedef struct iree_hal_amdgpu_device_dispatch_patch_pm4_bindings_args_t {
  // Device pointer to queue_execute binding pointers indexed by binding slot.
  const uint64_t* binding_ptrs;
  // Device pointer to immutable fixup entries owned by the command buffer.
  const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries;
  // Device pointer to the command buffer's resident kernarg-template base.
  uint8_t* target_base;
  // Number of valid entries in |entries|.
  uint32_t entry_count;
  // Reserved padding that must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_device_dispatch_patch_pm4_bindings_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_dispatch_patch_pm4_bindings_args_t) == 32,
    "PM4 binding patch args must match the kernel ABI");

// Populates a kernel dispatch packet body in already-reserved storage.
//
// The caller owns packet header commit, completion-signal assignment, and
// doorbell writes. Zero workgroup counts are preserved verbatim and produce a
// valid zero-grid dispatch packet.
//
// Preconditions:
//   - |kernel_args|, |workgroup_count|, |dispatch_packet|, and |kernarg_ptr|
//     are non-NULL.
//   - |kernel_args->workgroup_size| and
//     |kernel_args->group_segment_size + dynamic_workgroup_local_memory| are
//     valid for the target kernel.
//   - Each grid dimension product
//     |workgroup_count[i] * kernel_args->workgroup_size[i]| fits in uint32_t.
//   - |kernarg_ptr| satisfies |kernel_args->kernarg_alignment|.
void iree_hal_amdgpu_device_dispatch_emplace_packet(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates the HIP/OpenCL implicit args suffix in already-reserved storage.
//
// This must be called after explicit HAL/custom kernargs have been populated
// whenever |layout->has_implicit_args| is true.
//
// Preconditions:
//   - |kernel_args|, |workgroup_count|, |layout|, and |kernarg_ptr| are
//     non-NULL.
//   - |layout| describes a reservation with an implicit suffix.
//   - |kernarg_ptr| points to at least |layout->total_kernarg_size| bytes of
//     writable storage.
void iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates custom direct explicit kernargs in already-reserved storage.
//
// |custom_kernarg_ptr| provides up to |layout->total_kernarg_size| bytes in the
// final kernel ABI shape expected by the target kernel. Missing trailing
// padding bytes remain zeroed.
//
// Preconditions:
//   - |layout| and |kernarg_ptr| are non-NULL.
//   - |layout| describes either a fixed custom-direct reservation with optional
//     implicit suffix storage or a dynamic custom-direct reservation where
//     |layout->total_kernarg_size == 0| and |custom_kernarg_length| determines
//     the reservation size.
//   - |custom_kernarg_ptr| is non-NULL when |custom_kernarg_length| > 0.
void iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    const void* IREE_AMDGPU_RESTRICT custom_kernarg_ptr,
    size_t custom_kernarg_length, void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates the builtin patch dispatch that updates an indirect-parameter
// dispatch packet and then publishes its header.
//
// The target dispatch packet must already contain every non-header field. The
// patch dispatch reads |workgroup_count| on device, updates the target packet's
// grid-size fields and optional implicit args, then atomically publishes the
// provided final dispatch header/setup word.
void iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        patch_kernel_args,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    uint16_t dispatch_header, uint16_t dispatch_setup,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT patch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates the builtin patch dispatch that updates resident PM4 command-buffer
// kernarg templates from a queue_execute binding table.
void iree_hal_amdgpu_device_dispatch_emplace_pm4_binding_patch(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        patch_kernel_args,
    const uint64_t* IREE_AMDGPU_RESTRICT binding_ptrs,
    const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* IREE_AMDGPU_RESTRICT
        entries,
    uint8_t* IREE_AMDGPU_RESTRICT target_base, uint32_t entry_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT patch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that patches a following dispatch packet from indirect
// workgroup-count parameters. Launched as a single work-item dispatch.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_dispatch_patch_indirect_params(
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    uint32_t dispatch_header_setup);

// Device builtin that patches PM4 command-buffer resident kernarg templates
// from a compact queue_execute binding pointer table.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_dispatch_patch_pm4_bindings(
    const uint64_t* IREE_AMDGPU_RESTRICT binding_ptrs,
    const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* IREE_AMDGPU_RESTRICT
        entries,
    uint8_t* IREE_AMDGPU_RESTRICT target_base, uint32_t entry_count);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_
