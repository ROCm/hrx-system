// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_DISPATCH_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executable-owned packet inputs for a function requesting the dispatch ABI.
typedef struct iree_hal_amd_gpu_pm4_dispatch_packet_info_t {
  // Placed AMDHSA descriptor address, independent of its instruction entry.
  uint64_t kernel_object;
  // Borrowed executable-owned program converting indirect groups to items.
  const iree_hal_amd_gpu_pm4_launch_t* prepare;
} iree_hal_amd_gpu_pm4_dispatch_packet_info_t;

// One placed PM4 function. The executable dominates every borrowed pointer;
// captured dispatches keep that executable alive through native completion.
typedef struct iree_hal_amd_gpu_pm4_function_t {
  // Complete cold reflection and resource constraints owned by the image.
  const iree_hal_amd_gpu_metadata_kernel_t* metadata;
  // Immutable argument packing and public parameter reflection.
  iree_hal_amd_gpu_kernarg_layout_t* kernarg;
  // Placed register/SGPR state prepared once for this exact family.
  iree_hal_amd_gpu_pm4_launch_t launch;
  // Borrowed executable-owned packet inputs, or NULL when unused by the ABI.
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t* packet;
} iree_hal_amd_gpu_pm4_function_t;
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_hal_amd_gpu_pm4_function_t) == 64,
              "placed functions occupy one cache line");

// Shader-readable AMDHSA packet. It is never submitted to the PM4 firmware;
// command packets independently own execution and cache transitions.
typedef struct iree_hal_amd_gpu_pm4_dispatch_packet_t {
  // KERNEL_DISPATCH packet type with no implicit ordering or fence requests.
  uint16_t header;
  // Number of active grid axes, from one through three.
  uint16_t setup;
  // Nominal workitems per group along each axis.
  uint16_t workgroup_size[3];
  // Reserved AMDHSA field, initialized to zero.
  uint16_t reserved0;
  // Exact workitems along each axis, including any partial final group.
  uint32_t grid_size[3];
  // Fixed private bytes per workitem; dynamic scratch needs separate admission.
  uint32_t private_segment_size;
  // Fixed and dynamic LDS bytes per workgroup.
  uint32_t group_segment_size;
  // Placed 64-byte AMDHSA descriptor address.
  uint64_t kernel_object;
  // Invocation's actual native kernarg address, including external arguments.
  uint64_t kernarg_address;
  // Reserved AMDHSA field, initialized to zero.
  uint64_t reserved1;
  // Zero: completion belongs to the native PM4 retirement protocol.
  uint64_t completion_signal;
} iree_hal_amd_gpu_pm4_dispatch_packet_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_dispatch_packet_t) == 64,
              "AMDHSA dispatch packets occupy one cache line");

// Captured storage geometry, independent of the invocation's native addresses.
typedef struct iree_hal_amd_gpu_pm4_dispatch_storage_t {
  // Complete private extent, including kernargs, packet, and preparation
  // inputs.
  uint32_t length;
  // Power-of-two alignment of the complete allocation.
  uint32_t alignment;
  // Packet byte offset within the allocation; unused without a packet ABI.
  uint32_t packet_offset;
} iree_hal_amd_gpu_pm4_dispatch_storage_t;

// Queries bounded argument storage for an admitted function/argument mode.
iree_hal_amd_gpu_pm4_dispatch_storage_t iree_hal_amd_gpu_pm4_dispatch_storage(
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_dispatch_flags_t flags);

// Native program bytes and descriptor facts; both have static lifetime.
typedef struct iree_hal_amd_gpu_pm4_dispatch_program_t {
  // Relocation-free instructions; the owner appends speculative fetch padding.
  iree_const_byte_span_t code;
  // Descriptor register facts, with entry_offset zero before placement.
  iree_hal_amd_gpu_image_kernel_t kernel;
} iree_hal_amd_gpu_pm4_dispatch_program_t;

// Selects the indirect packet preparation program for a physical instruction
// set. Its two native arguments are the count tuple and destination packet
// addresses. One workitem writes exactly the three packet grid fields.
iree_status_t iree_hal_amd_gpu_pm4_dispatch_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_dispatch_program_t* out_program);

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
enum {
  IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS = 160,
  // Preparation launch, direct dispatch, completion join, and cache acquire.
  IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS = 26 + 5 + 2 + 8,
  // Instruction extent, including alignment and speculative fetch padding.
  IREE_HAL_AMD_GPU_PM4_DISPATCH_PROGRAM_LENGTH = 512,
};

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
// preserve native explicit offsets. Hidden geometry, packet fields, and padding
// are initialized here. External native arguments still use a private packet
// when requested by the shader. Target covers dispatch_storage().length and
// does not overlap the inputs. This transform cannot fail.
void iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target);

// Initializes the immutable packet fields. Device preparation fills indirect
// grid counts; publication binds the invocation's actual kernarg address. The
// supplied target may be a recording template; the final invocation storage
// is 64-byte aligned and owned through native retirement.
void iree_hal_amd_gpu_pm4_dispatch_emplace_packet(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_packet_t* target);

// Emits 41 DWORDs preparing one indirect packet and joining its writes before
// the consumer's launch state is bound. The two-pointer argument block follows
// the packet and remains owned through the consuming dispatch's retirement.
uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* prepare,
    uint64_t argument_address, uint32_t cache_control);

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
