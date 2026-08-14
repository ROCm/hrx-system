// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_program_set.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"

static iree_status_t iree_hal_amdgpu_pm4_command_program_layout_append_region(
    iree_host_size_t alignment, iree_host_size_t element_byte_length,
    uint32_t element_count, iree_host_size_t* inout_cursor,
    iree_host_size_t* out_offset, iree_host_size_t* out_stride) {
  *out_offset = 0;
  *out_stride = 0;
  if (element_count == 0 || element_byte_length == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment) ||
                    *inout_cursor > UINTPTR_MAX - (alignment - 1) ||
                    element_byte_length > UINTPTR_MAX - (alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 resident region alignment overflows");
  }

  const iree_host_size_t offset = iree_host_align(*inout_cursor, alignment);
  const iree_host_size_t stride =
      iree_host_align(element_byte_length, alignment);
  iree_host_size_t region_byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(stride, element_count,
                                                &region_byte_length) ||
                    !iree_host_size_checked_add(offset, region_byte_length,
                                                inout_cursor))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 resident region size overflows");
  }
  *out_offset = offset;
  *out_stride = stride;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_command_program_set_initialize(
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t physical_device_ordinal,
    iree_host_size_t physical_queue_count,
    iree_hal_amdgpu_pm4_command_program_set_flags_t flags,
    iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plans,
    iree_hal_amdgpu_pm4_command_program_set_t* out_program_set) {
  memset(out_program_set, 0, sizeof(*out_program_set));
  const iree_hal_amdgpu_pm4_command_program_set_flags_t supported_flags =
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE |
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported PM4 program-set flags 0x%08" PRIx32,
                            flags & ~supported_flags);
  }
  if (IREE_UNLIKELY(physical_queue_count == 0 ||
                    physical_queue_count > IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "physical queue count %" PRIhsz " must be in [1, %" PRIhsz "]",
        physical_queue_count, (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "physical device ordinal %" PRIhsz
                            " exceeds queue affinity capacity %" PRIhsz,
                            physical_device_ordinal,
                            (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }

  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      .supported_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .physical_device_count = physical_device_ordinal + 1,
      .queue_count_per_physical_device = physical_queue_count,
  };
  iree_hal_queue_affinity_t physical_device_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
      domain, physical_device_ordinal, &physical_device_affinity));
  iree_hal_queue_affinity_t selected_affinity = queue_affinity;
  if (iree_hal_queue_affinity_is_any(selected_affinity)) {
    selected_affinity = physical_device_affinity;
  }
  if (IREE_UNLIKELY(
          iree_hal_queue_affinity_is_empty(selected_affinity) ||
          iree_any_bit_set(selected_affinity, ~physical_device_affinity))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 command-buffer affinity 0x%" PRIx64
                            " must select queues on physical device %" PRIhsz,
                            queue_affinity, physical_device_ordinal);
  }

  uint64_t eligible_queue_mask = 0;
  for (iree_host_size_t physical_queue_ordinal = 0;
       physical_queue_ordinal < physical_queue_count;
       ++physical_queue_ordinal) {
    iree_hal_queue_affinity_t singleton_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_queue(
        domain, physical_device_ordinal, physical_queue_ordinal,
        &singleton_affinity));
    if (iree_any_bit_set(selected_affinity, singleton_affinity)) {
      eligible_queue_mask |= (uint64_t)1 << physical_queue_ordinal;
    }
  }

  const bool has_profile = iree_any_bit_set(
      flags, IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_PROFILE);
  const bool has_serial_profile = iree_any_bit_set(
      flags, IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE);
  const uint32_t profile_plan_count =
      has_profile
          ? (has_serial_profile
                 ? 1u
                 : (uint32_t)iree_math_count_ones_u64(eligible_queue_mask))
          : 0u;
  if (IREE_UNLIKELY(profile_plan_count != 0 && !profile_plans)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 profile plan storage is required");
  }

  if (profile_plans) {
    memset(profile_plans, 0, physical_queue_count * sizeof(*profile_plans));
  }
  out_program_set->profile_plans = profile_plans;
  out_program_set->eligible_queue_mask = eligible_queue_mask;
  out_program_set->physical_queue_count = (uint32_t)physical_queue_count;
  out_program_set->profile_plan_count = profile_plan_count;
  out_program_set->flags = flags;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_command_program_layout_calculate(
    const iree_hal_amdgpu_pm4_command_program_set_t* program_set,
    uint32_t program_dword_count, uint32_t profile_program_dword_count,
    iree_host_size_t template_byte_length, uint32_t fixup_entry_count,
    uint32_t profile_fixup_entry_count,
    iree_hal_amdgpu_pm4_command_program_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  if (IREE_UNLIKELY(program_dword_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 resident program is empty");
  }
  if (IREE_UNLIKELY((program_set->profile_plan_count == 0) !=
                    (profile_program_dword_count == 0))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 profile plan and program counts are inconsistent");
  }

  iree_host_size_t program_byte_length = 0;
  iree_host_size_t profile_program_byte_length = 0;
  iree_host_size_t fixup_byte_length = 0;
  iree_host_size_t profile_fixup_byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &program_byte_length,
      IREE_STRUCT_FIELD(program_dword_count, uint32_t, NULL)));
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &profile_program_byte_length,
      IREE_STRUCT_FIELD(profile_program_dword_count, uint32_t, NULL)));
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &fixup_byte_length,
      IREE_STRUCT_FIELD(fixup_entry_count,
                        iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t,
                        NULL)));
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &profile_fixup_byte_length,
      IREE_STRUCT_FIELD(profile_fixup_entry_count,
                        iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t,
                        NULL)));

  iree_host_size_t cursor = 0;
  iree_host_size_t ignored_stride = 0;
  out_layout->program_byte_length = program_byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      sizeof(uint64_t), program_byte_length, /*element_count=*/1, &cursor,
      &out_layout->program_offset, &ignored_stride));
  out_layout->profile_program_byte_length = profile_program_byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      sizeof(uint64_t), profile_program_byte_length,
      program_set->profile_plan_count, &cursor,
      &out_layout->profile_program_offset,
      &out_layout->profile_program_stride));
  out_layout->template_byte_length = template_byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      iree_max_align_t, template_byte_length, /*element_count=*/1, &cursor,
      &out_layout->template_offset, &ignored_stride));
  out_layout->fixup_byte_length = fixup_byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      iree_alignof(iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t),
      fixup_byte_length, /*element_count=*/1, &cursor,
      &out_layout->fixup_offset, &ignored_stride));
  out_layout->profile_fixup_byte_length = profile_fixup_byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      iree_alignof(iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t),
      profile_fixup_byte_length, program_set->profile_plan_count, &cursor,
      &out_layout->profile_fixup_offset, &out_layout->profile_fixup_stride));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_command_program_layout_append_region(
      iree_alignof(iree_hal_amdgpu_timestamp_range_t),
      sizeof(iree_hal_amdgpu_timestamp_range_t),
      program_set->profile_plan_count, &cursor, &out_layout->dummy_ticks_offset,
      &ignored_stride));
  out_layout->total_byte_length = cursor;
  if (IREE_UNLIKELY(
          (fixup_entry_count != 0 || profile_fixup_entry_count != 0) &&
          cursor > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 resident fixup target offsets exceed uint32_t storage");
  }
  return iree_ok_status();
}

const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t*
iree_hal_amdgpu_pm4_command_program_set_select_profile(
    const iree_hal_amdgpu_pm4_command_program_set_t* program_set,
    uint32_t physical_queue_ordinal) {
  if (program_set->profile_plan_count == 0 ||
      physical_queue_ordinal >= program_set->physical_queue_count ||
      !iree_all_bits_set(program_set->eligible_queue_mask,
                         (uint64_t)1 << physical_queue_ordinal)) {
    return NULL;
  }
  const bool has_serial_profile = iree_any_bit_set(
      program_set->flags,
      IREE_HAL_AMDGPU_PM4_COMMAND_PROGRAM_SET_FLAG_SERIAL_PROFILE);
  if (has_serial_profile) return &program_set->profile_plans[0];
  const uint64_t preceding_queue_mask =
      physical_queue_ordinal == 0
          ? 0
          : program_set->eligible_queue_mask &
                (((uint64_t)1 << physical_queue_ordinal) - 1u);
  const uint32_t profile_plan_ordinal =
      (uint32_t)iree_math_count_ones_u64(preceding_queue_mask);
  return &program_set->profile_plans[profile_plan_ordinal];
}
