// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_PROGRAM_SET_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_PROGRAM_SET_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/abi/timestamp.h"
#include "iree/hal/drivers/amdgpu/util/pm4_program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-resident command-buffer storage and dynamic fixup records owned by a
// finalized PM4 command buffer.
typedef struct iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t {
  // Device pointer to immutable fixup records, or NULL when no fixup runs.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*
      entries;
  // Number of entries in |entries|.
  uint32_t entry_count;
  // Reserved padding that must be zero.
  uint32_t reserved0;
  // Device pointer to the resident allocation base patched by fixup offsets.
  IREE_AMDGPU_DEVICE_PTR uint8_t* target_base;
  // Allocated byte length of |target_base|.
  iree_host_size_t target_byte_length;
  // Device pointer to resident kernarg-template storage referenced by PM4.
  IREE_AMDGPU_DEVICE_PTR uint8_t* template_base;
  // Allocated byte length of |template_base|.
  iree_host_size_t template_byte_length;
} iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t;

// Device-resident PM4 program and fixup plan used only while
// dispatch-attributed profiling is enabled.
typedef struct iree_hal_amdgpu_pm4_command_buffer_profile_plan_t {
  // Device-visible PM4 IB with per-dispatch timestamp packets.
  iree_hal_amdgpu_pm4_program_t program;
  // Device pointer to immutable profile fixup records.
  IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t*
      entries;
  // Number of entries in |entries|.
  uint32_t entry_count;
  // First synthetic binding slot used for dispatch timestamp destinations.
  uint32_t timestamp_binding_base;
  // Total binding-table entries consumed by profile fixup.
  uint32_t binding_count;
  // Number of profile-visible operations represented in |program|.
  uint32_t operation_count;
  // Device pointer to the resident allocation base patched by fixup offsets.
  IREE_AMDGPU_DEVICE_PTR uint8_t* target_base;
  // Device-visible fallback timestamp range used for unselected dispatches.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_timestamp_range_t* dummy_ticks;
} iree_hal_amdgpu_pm4_command_buffer_profile_plan_t;

typedef enum iree_hal_amdgpu_pm4_command_program_set_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_NONE = 0u,
  // The HAL lifecycle prohibits overlapping profile-plan mutation. This holds
  // for one-shot command buffers and indirect command buffers that this
  // implementation does not make concurrently schedulable.
  IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE = 1u << 0,
  // Dispatch-attributed profiling materializes mutable profile programs.
  IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE = 1u << 1,
} iree_hal_amdgpu_pm4_command_program_set_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_command_program_set_flags_t;

// Resident PM4 plans produced by command-buffer finalization.
//
// Normal execution is immutable and shared by every eligible queue. Profile
// execution patches timestamp destinations before each submission, so reusable
// direct command buffers own one profile plan per eligible physical queue.
// One-shot and indirect command buffers retain one shared profile plan because
// their HAL lifecycle prohibits overlapping execution.
typedef struct iree_hal_amdgpu_pm4_command_program_set_t {
  // Shared normal execution PM4 program.
  iree_hal_amdgpu_pm4_program_t program;
  // Shared dynamic binding fixup plan for |program|.
  iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t fixup;
  // Caller-owned profile plans indexed by compact profile-plan ordinal.
  iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plans;
  // Local physical queue bits eligible to execute the command buffer.
  uint64_t eligible_queue_mask;
  // Number of physical queues addressable by |eligible_queue_mask|.
  uint32_t physical_queue_count;
  // Number of entries in |profile_plans| active after finalization.
  uint32_t profile_plan_count;
  // iree_hal_amdgpu_pm4_command_program_set_flag_bits_t mask.
  iree_hal_amdgpu_pm4_command_program_set_flags_t flags;
} iree_hal_amdgpu_pm4_command_program_set_t;

// Byte layout of one contiguous resident PM4 command-buffer allocation.
typedef struct iree_hal_amdgpu_pm4_command_program_layout_t {
  // Total resident allocation byte length.
  iree_host_size_t total_byte_length;
  // Byte offset of the shared normal program image.
  iree_host_size_t program_offset;
  // Byte length of the shared normal program image.
  iree_host_size_t program_byte_length;
  // Byte offset of the first profiling program image.
  iree_host_size_t profile_program_offset;
  // Byte stride between profiling program images.
  iree_host_size_t profile_program_stride;
  // Byte length of each profiling program image.
  iree_host_size_t profile_program_byte_length;
  // Byte offset of shared kernarg template storage.
  iree_host_size_t template_offset;
  // Byte length of shared kernarg template storage.
  iree_host_size_t template_byte_length;
  // Byte offset of the shared normal fixup array.
  iree_host_size_t fixup_offset;
  // Byte length of the shared normal fixup array.
  iree_host_size_t fixup_byte_length;
  // Byte offset of the first profiling fixup array.
  iree_host_size_t profile_fixup_offset;
  // Byte stride between profiling fixup arrays.
  iree_host_size_t profile_fixup_stride;
  // Byte length of each profiling fixup array.
  iree_host_size_t profile_fixup_byte_length;
  // Byte offset of the first profiling dummy timestamp range.
  iree_host_size_t dummy_ticks_offset;
} iree_hal_amdgpu_pm4_command_program_layout_t;

// Initializes |out_program_set| and clears all caller-owned |profile_plans|.
iree_status_t iree_hal_amdgpu_pm4_command_program_set_initialize(
    iree_host_size_t physical_queue_count,
    iree_hal_amdgpu_pm4_command_program_set_flags_t flags,
    iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plans,
    iree_hal_amdgpu_pm4_command_program_set_t* out_program_set);

// Calculates the single-allocation resident layout for |program_set|.
iree_status_t iree_hal_amdgpu_pm4_command_program_layout_calculate(
    const iree_hal_amdgpu_pm4_command_program_set_t* program_set,
    uint32_t program_dword_count, uint32_t profile_program_dword_count,
    iree_host_size_t template_byte_length, uint32_t fixup_entry_count,
    uint32_t profile_fixup_entry_count,
    iree_hal_amdgpu_pm4_command_program_layout_t* out_layout);

// Returns the profiling program byte offset for |profile_plan_ordinal|.
static inline iree_host_size_t
iree_hal_amdgpu_pm4_command_program_layout_profile_program_offset(
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint32_t profile_plan_ordinal) {
  return layout->profile_program_offset +
         layout->profile_program_stride * profile_plan_ordinal;
}

// Returns the profiling fixup-array byte offset for |profile_plan_ordinal|.
static inline iree_host_size_t
iree_hal_amdgpu_pm4_command_program_layout_profile_fixup_offset(
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint32_t profile_plan_ordinal) {
  return layout->profile_fixup_offset +
         layout->profile_fixup_stride * profile_plan_ordinal;
}

// Returns the dummy timestamp byte offset for |profile_plan_ordinal|.
static inline iree_host_size_t
iree_hal_amdgpu_pm4_command_program_layout_dummy_ticks_offset(
    const iree_hal_amdgpu_pm4_command_program_layout_t* layout,
    uint32_t profile_plan_ordinal) {
  return layout->dummy_ticks_offset +
         sizeof(iree_hal_amdgpu_timestamp_range_t) * profile_plan_ordinal;
}

// Returns the profile plan selected for |physical_queue_ordinal|, or NULL when
// profiling is absent or that queue cannot execute the command buffer.
const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t*
iree_hal_amdgpu_pm4_command_program_set_select_profile(
    const iree_hal_amdgpu_pm4_command_program_set_t* program_set,
    uint32_t physical_queue_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_PROGRAM_SET_H_
