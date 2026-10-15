// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_LAUNCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_LAUNCH_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/gpu/image/image.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Runtime services required by a prepared program, independent of its argument
// metadata. The executable/queue owner must establish these services before
// admitting dispatch. Preparing register values does not provide the services.
enum iree_hal_amd_gpu_pm4_launch_requirement_bits_e {
  // Architected scratch backing and its native queue register state.
  IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH = 1u << 0,
  // A stack allocation policy in addition to the fixed private segment.
  IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DYNAMIC_STACK = 1u << 1,
  // An initialized AQL-compatible dispatch packet in GPU-readable storage.
  IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER = 1u << 2,
  // An initialized AMD queue ABI object in GPU-readable storage.
  IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_QUEUE_POINTER = 1u << 3,
};
typedef uint16_t iree_hal_amd_gpu_pm4_launch_requirements_t;

// Immutable register and user-input layout prepared once per placed kernel.
// Contains no borrowed pointers or mutable binding state. The enclosing
// executable owns image and instruction-fetch backing through final use.
typedef struct iree_hal_amd_gpu_pm4_launch_t {
  // COMPUTE_PGM_LO/HI, encoding the placed entry address in 256-byte units.
  uint32_t program[2];
  // Compiler resource registers, with LDS_SIZE cleared for each binding.
  struct {
    // COMPUTE_PGM_RSRC1, including compiler allocation and execution modes.
    uint32_t resource1;
    // COMPUTE_PGM_RSRC2, preserving user/system inputs and private enable.
    uint32_t resource2;
    // COMPUTE_PGM_RSRC3, including the backed instruction-prefetch request.
    uint32_t resource3;
  } resources;
  // Target-selected COMPUTE_PGM_RSRC3 register address.
  uint32_t resource3_register;
  // Enabled zero-origin dispatch in group units with the compiled wave mode.
  uint32_t dispatch_initiator;
  // Runtime service requirements derived from the descriptor.
  iree_hal_amd_gpu_pm4_launch_requirements_t requirements;
  // Complete user-register interval, including zero-filled compiler padding.
  uint8_t user_data_count;
  // Log2 of compiled workitems per wave, avoiding division during binding.
  uint8_t wavefront_size_log2;
  // Dense SGPR indices, or UINT8_MAX when the corresponding input is absent.
  struct {
    // Two-word AQL-compatible dispatch packet pointer.
    uint8_t dispatch_pointer;
    // Two-word AMD queue ABI pointer.
    uint8_t queue_pointer;
    // Two-word captured kernarg segment pointer.
    uint8_t kernarg_pointer;
    // Two-word dispatch ID.
    uint8_t dispatch_id;
    // One-word per-workitem private byte count, rounded to four bytes.
    uint8_t private_segment_size;
  } user_data;
} iree_hal_amd_gpu_pm4_launch_t;

// Resolves target-dependent register state and the descriptor's initial SGPR
// layout. This is cold executable-load work. The image has already passed
// structural parsing and target compatibility; |gfxip| is the physical target.
// |image_address| is the placed image base. Failure leaves |out_launch|
// unchanged.
//
// Success establishes encoding, not dispatch admission. The owner additionally
// supplies the required native services, validates metadata/dispatch resource
// limits, and retains code plus the descriptor's instruction-fetch extent.
iree_status_t iree_hal_amd_gpu_pm4_launch_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    const iree_hal_amd_gpu_image_kernel_t* kernel, uint64_t image_address,
    iree_hal_amd_gpu_pm4_launch_t* out_launch);

// Runtime values for the enabled initial user inputs. Only requested fields
// are read. Pointed-to objects and kernargs are initialized before publication
// and remain accessible through final shader use. In particular, a dispatch
// pointer is an AQL ABI packet, not the address of a PM4 dispatch command.
typedef struct iree_hal_amd_gpu_pm4_launch_inputs_t {
  // GPU address of the initialized AQL-compatible dispatch packet.
  uint64_t dispatch_pointer;
  // GPU address of the initialized AMD queue ABI object.
  uint64_t queue_pointer;
  // GPU address of captured kernel arguments.
  uint64_t kernarg_pointer;
  // Runtime-assigned dispatch identifier.
  uint64_t dispatch_id;
  // Complete per-workitem private byte count, rounded to four bytes.
  uint32_t private_segment_size;
} iree_hal_amd_gpu_pm4_launch_inputs_t;

// DWORD offset of an enabled user SGPR within launch_emit's packet sequence.
// UINT8_MAX denotes an absent input and produces UINT16_MAX. Recorders retain
// these offsets instead of decoding register packets during invocation.
static inline uint16_t iree_hal_amd_gpu_pm4_launch_user_data_word_offset(
    uint8_t index) {
  return index == UINT8_MAX ? UINT16_MAX : (uint16_t)(24 + index);
}

// Emits complete launch binding into caller-owned capacity and returns the next
// DWORD. Emits 22 DWORDs when user_data_count is zero, otherwise
// 24 + user_data_count. This hot transform is infallible: the caller has
// admitted geometry, total fixed+dynamic LDS bytes, runtime services and all
// storage lifetimes.
//
// Each workgroup dimension is positive and fits the target's local geometry.
// |partial_size| contains the final group's dimensions, or zeros for full
// groups. A partial launch supplies all three final dimensions (including full
// sizes on divisible axes) and enables PARTIAL_TG_EN in the subsequent dispatch
// initiator. |group_segment_byte_length| is the complete admitted LDS
// allocation.
//
// Every binding replaces prior program/LDS state. The following dispatch uses
// the prepared wave mode. Scratch-base, scheduler, trap-handler-address,
// profiling and dispatch-pointer context registers remain native-owned.
// Binding does not issue a dispatch or establish an execution/cache dependency.
uint32_t* iree_hal_amd_gpu_pm4_launch_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    const uint32_t workgroup_size[3], const uint32_t partial_size[3],
    uint32_t group_segment_byte_length,
    const iree_hal_amd_gpu_pm4_launch_inputs_t* inputs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_LAUNCH_H_
