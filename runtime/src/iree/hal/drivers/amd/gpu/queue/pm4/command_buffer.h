// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a recorder whose end operation seals device-resident native blocks.
// Its independently retainable program holds the device for the recording; the
// device-owned program pool provides placement and trim; native sealing
// allocations run on the caller. UNRETAINED controls only the caller's direct
// buffers/executables, not native program ownership.
iree_status_t iree_hal_amd_gpu_pm4_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_pm4_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity,
    uint32_t maximum_group_segment_byte_length,
    uint32_t maximum_scratch_wave_byte_length,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_allocator_t host_allocator, iree_hal_command_buffer_t** out_buffer);

// Separately retained cold recording facts read directly by publication. The
// public recording owns user resources; this object contains sealed execution
// facts but no caller resource set, and may outlive the final invocation during
// cold growth.
typedef struct iree_hal_amd_gpu_pm4_command_program_t {
  // Internal owners spanning native growth independently of the recording.
  iree_atomic_ref_count_t ref_count;
  // Device retaining the program pool through final native growth completion.
  iree_hal_device_t* device;
  // Allocator owning this separately retained program record.
  iree_allocator_t host_allocator;
  // Owned native instances and shared immutable fixup table.
  iree_hal_amd_gpu_pm4_program_t resident;
  // Entry block; native successors and exits are encoded in resident commands.
  iree_hal_amd_gpu_pm4_program_block_t entry;
  // Dense uint64_t inputs uploaded once per invocation.
  uint32_t value_count;
  // Complete invocation upload: copy/fixup kernargs followed by input values.
  uint32_t argument_length;
  // Number of public dispatch IDs consumed by one complete execution.
  uint32_t dispatch_count;
  // Maximum admitted private bytes per hardware wave across all dispatches.
  uint32_t scratch_wave_byte_length;
  // Encoded payload operation count used only by optional queue profiling.
  uint32_t operation_count;
  // Optional host-only dispatch attribution, independently retainable by a
  // sink.
  iree_hal_amd_gpu_profile_commands_t* profile;
} iree_hal_amd_gpu_pm4_command_program_t;

// Retains/releases sealed native program state independently of the recording
// and its user resources. The recording owns one reference; cold native growth
// may hold another after every invocation has retired.
void iree_hal_amd_gpu_pm4_command_program_retain(
    iree_hal_amd_gpu_pm4_command_program_t* program);
void iree_hal_amd_gpu_pm4_command_program_release(
    iree_hal_amd_gpu_pm4_command_program_t* program);

iree_host_size_t iree_hal_amd_gpu_pm4_command_buffer_operation_count(
    iree_hal_command_buffer_t* command_buffer);

// Captured, qualified binding. Address commitment waits for queue dependencies;
// repeated uses in recorded commands share one dense native input.
typedef struct iree_hal_amd_gpu_pm4_command_binding_t {
  // Owned buffer unless the caller explicitly borrowed invocation lifetime.
  iree_hal_buffer_t* buffer;
  // Byte offset from the logical beginning of buffer.
  iree_device_size_t offset;
  // Logical extent after resolving WHOLE_BUFFER.
  iree_device_size_t length;
  // Prepared native slots for the queue (0) and shader program (1) sites.
  iree_hal_buffer_native_binding_slot_t slots[2];
} iree_hal_amd_gpu_pm4_command_binding_t;

typedef struct iree_hal_amd_gpu_pm4_command_invocation_t {
  // Retained recording dominating its native program and direct resources.
  iree_hal_command_buffer_t* command_buffer;
  // Borrowed sealed facts retained by command_buffer.
  iree_hal_amd_gpu_pm4_command_program_t* program;
  // Captured table in submission-owned storage.
  iree_hal_amd_gpu_pm4_command_binding_t* bindings;
  // Reserved or published native placement, returned on rejection/retirement.
  iree_hal_amd_gpu_pm4_program_instance_t* instance;
  // Native acceptance transferred the reservation to a causal lease.
  bool published;
  // Binding buffers are borrowed; the table itself is always copied.
  bool borrow_bindings;
} iree_hal_amd_gpu_pm4_command_invocation_t;

// Captures an already validated public table. storage covers binding_count
// entries. Failure leaves a deinitializable prefix; success retains used
// buffers unless borrowed, plus the command buffer independently of caller
// ownership.
iree_status_t iree_hal_amd_gpu_pm4_command_invocation_initialize(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, void* storage,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_command_invocation_t* out_invocation);

// Releases the binding capture and native instance. Published uses require
// proven native retirement; cancellation preserves earlier causal state.
void iree_hal_amd_gpu_pm4_command_invocation_deinitialize(
    iree_hal_amd_gpu_pm4_command_invocation_t* invocation);

// Writes the dense input vector following program_arguments_t after
// dependencies have made backing addresses available. Cost is proportional to
// unique inputs, never to commands or fixups. Program emission fills the
// header.
iree_status_t iree_hal_amd_gpu_pm4_command_invocation_write(
    const iree_hal_amd_gpu_pm4_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t dispatch_id, uint64_t continuation,
    uint64_t profile_address, void* upload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_COMMAND_BUFFER_H_
