// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/descriptor.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/arch/amdgpu/target_info.h"

//===----------------------------------------------------------------------===//
// AMDHSA descriptor bit fields
//===----------------------------------------------------------------------===//

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH 4u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_16_64_SHIFT 18u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT 21u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT 23u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT 29u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_MEM_ORDERED_SHIFT 30u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT 31u

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_PRIVATE_SEGMENT_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT 1u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH 5u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_GFX125_WIDTH 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_X_SHIFT 7u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Y_SHIFT 8u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Z_SHIFT 9u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_INFO_SHIFT 10u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_SHIFT 11u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH 2u

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_WIDTH 6u

#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_BUFFER_SHIFT 0u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_PTR_SHIFT 1u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_QUEUE_PTR_SHIFT 2u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_KERNARG_SEGMENT_PTR_SHIFT 3u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_ID_SHIFT 4u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_FLAT_SCRATCH_INIT_SHIFT 5u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_SIZE_SHIFT 6u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_WAVEFRONT_SIZE32_SHIFT 10u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT 11u

#define LOOM_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_USER_SGPR_COUNT 2u

#define LOOM_AMDGPU_ELF_FEATURE_XNACK_V4 UINT32_C(0x300)
#define LOOM_AMDGPU_ELF_FEATURE_XNACK_ANY_V4 UINT32_C(0x100)
#define LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4 UINT32_C(0x300)

static const loom_amdgpu_kernel_descriptor_flags_t
    kAmdgpuKernelDescriptorKnownFlags =
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32 |
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK;

static iree_status_t loom_amdgpu_kernel_descriptor_resolve_target(
    iree_string_view_t processor_name,
    const loom_amdgpu_processor_info_t** out_target) {
  *out_target = NULL;
  const loom_amdgpu_processor_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor_name, &target));
  switch (target->kernel_descriptor.profile) {
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
      *out_target = target;
      return iree_ok_status();
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
      break;
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU kernel descriptor processor '%.*s' is not supported yet",
      (int)processor_name.size, processor_name.data);
}

static uint32_t loom_amdgpu_kernel_descriptor_bit_mask(uint32_t width) {
  return (UINT32_C(1) << width) - 1u;
}

static void loom_amdgpu_kernel_descriptor_set_bits_u32(uint32_t* inout_value,
                                                       uint32_t shift,
                                                       uint32_t width,
                                                       uint32_t value) {
  const uint32_t mask = loom_amdgpu_kernel_descriptor_bit_mask(width);
  *inout_value &= ~(mask << shift);
  *inout_value |= (value & mask) << shift;
}

static void loom_amdgpu_kernel_descriptor_set_bit_u32(uint32_t* inout_value,
                                                      uint32_t shift,
                                                      bool value) {
  if (value) {
    *inout_value |= UINT32_C(1) << shift;
  } else {
    *inout_value &= ~(UINT32_C(1) << shift);
  }
}

static void loom_amdgpu_kernel_descriptor_set_bit_u16(uint16_t* inout_value,
                                                      uint32_t shift,
                                                      bool value) {
  if (value) {
    *inout_value |= (uint16_t)(UINT16_C(1) << shift);
  } else {
    *inout_value &= (uint16_t)~(UINT16_C(1) << shift);
  }
}

static iree_status_t loom_amdgpu_kernel_descriptor_granulated_blocks(
    uint32_t register_count, uint32_t granule, uint32_t field_width,
    uint32_t* out_block_count) {
  *out_block_count = 0;
  uint64_t normalized_count = register_count == 0 ? 1u : register_count;
  if (normalized_count > UINT64_MAX - (granule - 1u)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor register count overflows");
  }
  uint64_t granulated_count = normalized_count + granule - 1u;
  granulated_count /= granule;
  const uint64_t encoded_count = granulated_count - 1u;
  const uint32_t max_value =
      loom_amdgpu_kernel_descriptor_bit_mask(field_width);
  if (encoded_count > max_value) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor register block count %" PRIu64
        " exceeds field capacity %" PRIu32,
        encoded_count, max_value);
  }
  *out_block_count = (uint32_t)encoded_count;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_kernel_descriptor_implied_user_sgpr_count(
    const loom_amdgpu_kernel_descriptor_t* descriptor) {
  uint32_t count = 0;
  if (iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
    count += 4;
  }
  if (iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR)) {
    count += 2;
  }
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR)) {
    count += 2;
  }
  if (iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR)) {
    count += 2;
  }
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID)) {
    count += 2;
  }
  if (iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
    count += 2;
  }
  if (iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
    count += 1;
  }
  return count;
}

static bool loom_amdgpu_kernel_descriptor_is_xnack_on_or_any(
    const loom_amdgpu_processor_info_t* target) {
  const uint32_t xnack_selection =
      target->elf.feature_flags & LOOM_AMDGPU_ELF_FEATURE_XNACK_V4;
  return xnack_selection == LOOM_AMDGPU_ELF_FEATURE_XNACK_ANY_V4 ||
         xnack_selection == LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4;
}

static bool loom_amdgpu_kernel_descriptor_supports_wgp_mode(
    const loom_amdgpu_processor_info_t* target) {
  switch (target->kernel_descriptor.profile) {
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
      return true;
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
      return false;
  }
  return false;
}

static uint32_t loom_amdgpu_kernel_descriptor_wavefront_size(
    const loom_amdgpu_kernel_descriptor_t* descriptor) {
  return iree_any_bit_set(descriptor->flags,
                          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32)
             ? 32
             : 64;
}

static uint32_t loom_amdgpu_kernel_descriptor_user_sgpr_count_width(
    const loom_amdgpu_processor_info_t* target) {
  if (target->kernel_descriptor.profile ==
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125) {
    return LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_GFX125_WIDTH;
  }
  return LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH;
}

static uint32_t loom_amdgpu_kernel_descriptor_legacy_extra_sgpr_count(
    const loom_amdgpu_processor_info_t* target) {
  uint32_t count = 2u;
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          target,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH)) {
    count = 6u;
  } else if (loom_amdgpu_kernel_descriptor_is_xnack_on_or_any(target)) {
    count = 4u;
  }
  return count;
}

iree_status_t loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
    loom_amdgpu_kernel_descriptor_flags_t flags, uint32_t* out_mode) {
  *out_mode = 0;
  const loom_amdgpu_kernel_descriptor_flags_t workitem_id_flags =
      flags & (LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
               LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
               LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z);
  const bool has_x =
      iree_any_bit_set(workitem_id_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X);
  const bool has_y =
      iree_any_bit_set(workitem_id_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y);
  const bool has_z =
      iree_any_bit_set(workitem_id_flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z);
  if (has_z && (!has_x || !has_y)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor workitem-id-z requires workitem-id-x/y "
        "flags");
  }
  if (has_y && !has_x) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor workitem-id-y requires the workitem-id-x "
        "flag");
  }
  if (has_z) {
    *out_mode = 2;
  } else if (has_y) {
    *out_mode = 1;
  } else {
    *out_mode = 0;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_descriptor_validate(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    const loom_amdgpu_processor_info_t** out_target) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_resolve_target(
      descriptor->processor, out_target));
  if (iree_any_bit_set(descriptor->flags, ~kAmdgpuKernelDescriptorKnownFlags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor flags contain unknown bits 0x%08" PRIx32,
        descriptor->flags & ~kAmdgpuKernelDescriptorKnownFlags);
  }
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor->flags, &system_vgpr_workitem_id));
  (void)system_vgpr_workitem_id;
  const uint32_t implied_user_sgpr_count =
      loom_amdgpu_kernel_descriptor_implied_user_sgpr_count(descriptor);
  if (descriptor->user_sgpr_count < implied_user_sgpr_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel descriptor user SGPR count %" PRIu32
                            " is smaller than implied count %" PRIu32,
                            descriptor->user_sgpr_count,
                            implied_user_sgpr_count);
  }
  const uint32_t user_sgpr_count_width =
      loom_amdgpu_kernel_descriptor_user_sgpr_count_width(*out_target);
  if (descriptor->user_sgpr_count >
      loom_amdgpu_kernel_descriptor_bit_mask(user_sgpr_count_width)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor user SGPR count exceeds target capacity");
  }
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          *out_target,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH) &&
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor private segment buffer user SGPR is invalid "
        "with architected flat scratch");
  }
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          *out_target,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH) &&
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor flat scratch init user SGPR is invalid with "
        "architected flat scratch");
  }
  const uint32_t wavefront_size =
      loom_amdgpu_kernel_descriptor_wavefront_size(descriptor);
  if (!loom_amdgpu_processor_supports_wavefront_size(*out_target,
                                                     wavefront_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' kernel descriptors do not support "
        "wavefront-size-%" PRIu32 " metadata",
        (int)(*out_target)->name.size, (*out_target)->name.data,
        wavefront_size);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata_kernel(
    const loom_amdgpu_processor_info_t* target,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel) {
  if (metadata_kernel->wavefront_size != 32 &&
      metadata_kernel->wavefront_size != 64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata wavefront size must be either 32 or 64");
  }
  if (!loom_amdgpu_processor_supports_wavefront_size(
          target, metadata_kernel->wavefront_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU processor '%.*s' metadata does not support "
                            "wavefront-size-%" PRIu32,
                            (int)target->name.size, target->name.data,
                            metadata_kernel->wavefront_size);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_amdgpu_kernel_descriptor_initialize_from_metadata(
    iree_string_view_t processor_name,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel,
    int64_t kernel_code_entry_byte_offset,
    loom_amdgpu_kernel_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_amdgpu_kernel_descriptor_t){0};
  const loom_amdgpu_processor_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_resolve_target(processor_name, &target));
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_validate_metadata_kernel(
      target, metadata_kernel));
  *out_descriptor = (loom_amdgpu_kernel_descriptor_t){
      .processor = processor_name,
      .group_segment_fixed_size = metadata_kernel->group_segment_fixed_size,
      .private_segment_fixed_size = metadata_kernel->private_segment_fixed_size,
      .kernarg_size = metadata_kernel->kernarg_segment_size,
      .kernel_code_entry_byte_offset = kernel_code_entry_byte_offset,
      .next_free_sgpr = metadata_kernel->sgpr_count,
      .next_free_vgpr = metadata_kernel->vgpr_count,
      .user_sgpr_count =
          metadata_kernel->kernarg_segment_size != 0
              ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_KERNARG_USER_SGPR_COUNT
              : 0,
      .flags =
          (metadata_kernel->kernarg_segment_size != 0
               ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR
               : 0) |
          (metadata_kernel->private_segment_fixed_size != 0
               ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT
               : 0) |
          (metadata_kernel->wavefront_size == 32
               ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32
               : 0),
  };
  return loom_amdgpu_kernel_descriptor_validate(out_descriptor, &target);
}

iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel) {
  const loom_amdgpu_processor_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate(descriptor, &target));
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_validate_metadata_kernel(
      target, metadata_kernel));
  if (descriptor->group_segment_fixed_size !=
      metadata_kernel->group_segment_fixed_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor group segment size does not "
                            "match metadata");
  }
  if (descriptor->private_segment_fixed_size !=
      metadata_kernel->private_segment_fixed_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor private segment size does not "
                            "match metadata");
  }
  if (descriptor->kernarg_size != metadata_kernel->kernarg_segment_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor kernarg size does not match "
                            "metadata");
  }
  if (descriptor->next_free_sgpr != metadata_kernel->sgpr_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor SGPR count does not match "
                            "metadata");
  }
  if (descriptor->next_free_vgpr != metadata_kernel->vgpr_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor VGPR count does not match "
                            "metadata");
  }
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32) !=
      (metadata_kernel->wavefront_size == 32)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor wavefront size does not match "
                            "metadata");
  }
  return iree_ok_status();
}

static void loom_amdgpu_kernel_descriptor_store_le_u16(uint8_t* target,
                                                       uint16_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
}

static void loom_amdgpu_kernel_descriptor_store_le_u32(uint8_t* target,
                                                       uint32_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
  target[2] = (uint8_t)(value >> 16);
  target[3] = (uint8_t)(value >> 24);
}

static void loom_amdgpu_kernel_descriptor_store_le_i64(uint8_t* target,
                                                       int64_t value) {
  uint64_t unsigned_value = (uint64_t)value;
  for (iree_host_size_t i = 0; i < 8; ++i) {
    target[i] = (uint8_t)(unsigned_value >> (i * 8));
  }
}

iree_status_t loom_amdgpu_kernel_descriptor_write(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    iree_byte_span_t target_bytes) {
  const loom_amdgpu_processor_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate(descriptor, &target));
  if (target_bytes.data == NULL ||
      target_bytes.data_length < LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor output requires at least %u bytes",
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  }

  const uint32_t vgpr_granule =
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32)
          ? target->kernel_descriptor.vgpr_granules.wave32
          : target->kernel_descriptor.vgpr_granules.wave64;
  uint32_t vgpr_block_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_granulated_blocks(
      descriptor->next_free_vgpr, vgpr_granule,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH, &vgpr_block_count));

  uint32_t sgpr_block_count = 0;
  if (!loom_amdgpu_processor_kernel_descriptor_has_flags(
          target, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING)) {
    const uint32_t extra_sgpr_count =
        loom_amdgpu_kernel_descriptor_legacy_extra_sgpr_count(target);
    if (descriptor->next_free_sgpr > UINT32_MAX - extra_sgpr_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU kernel descriptor SGPR count overflows after target "
          "reserved SGPRs");
    }
    const uint32_t encoded_sgpr_count =
        descriptor->next_free_sgpr + extra_sgpr_count;
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_granulated_blocks(
        encoded_sgpr_count, 8, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH,
        &sgpr_block_count));
  }

  uint32_t compute_pgm_rsrc3 = 0;
  if (loom_amdgpu_processor_kernel_descriptor_has_flags(
          target, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET)) {
    uint32_t accum_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_granulated_blocks(
        descriptor->next_free_vgpr, 4,
        LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_WIDTH, &accum_offset));
    loom_amdgpu_kernel_descriptor_set_bits_u32(
        &compute_pgm_rsrc3, LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_SHIFT,
        LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_WIDTH, accum_offset);
  }

  uint32_t compute_pgm_rsrc1 = 0;
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH, vgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH, sgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_16_64_SHIFT, 2,
      3);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          target,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          target,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT,
      loom_amdgpu_kernel_descriptor_supports_wgp_mode(target));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_MEM_ORDERED_SHIFT,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          target, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          target, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING));

  uint32_t compute_pgm_rsrc2 = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_PRIVATE_SEGMENT_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT));
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT,
      loom_amdgpu_kernel_descriptor_user_sgpr_count_width(target),
      descriptor->user_sgpr_count);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_X_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Y_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Z_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_INFO_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO));
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor->flags, &system_vgpr_workitem_id));
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH, system_vgpr_workitem_id);

  uint16_t kernel_code_properties = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_BUFFER_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_PTR_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties, LOOM_AMDGPU_KERNEL_CODE_PROPERTY_QUEUE_PTR_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_KERNARG_SEGMENT_PTR_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_ID_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_FLAT_SCRATCH_INIT_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_SIZE_SHIFT,
      iree_any_bit_set(
          descriptor->flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_WAVEFRONT_SIZE32_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT,
      iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK));

  memset(target_bytes.data, 0, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 0, descriptor->group_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 4, descriptor->private_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 8,
                                             descriptor->kernarg_size);
  loom_amdgpu_kernel_descriptor_store_le_i64(
      target_bytes.data + 16, descriptor->kernel_code_entry_byte_offset);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 44,
                                             compute_pgm_rsrc3);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 48,
                                             compute_pgm_rsrc1);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 52,
                                             compute_pgm_rsrc2);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 56,
                                             kernel_code_properties);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 58, 0);
  return iree_ok_status();
}
