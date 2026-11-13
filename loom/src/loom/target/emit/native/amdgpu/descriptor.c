// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/descriptor.h"

#include <string.h>

#include "loom/target/arch/amdgpu/target_info.h"

//===----------------------------------------------------------------------===//
// AMDHSA descriptor bit fields
//===----------------------------------------------------------------------===//

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH 4u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_32_SHIFT 16u
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

typedef struct loom_amdgpu_kernel_descriptor_fields_t {
  // Fixed LDS/group segment size in bytes.
  uint32_t group_segment_fixed_size;
  // Fixed scratch/private segment size in bytes.
  uint32_t private_segment_fixed_size;
  // Kernel kernarg segment size in bytes.
  uint32_t kernarg_size;
  // Byte offset from the descriptor symbol to the kernel entry instruction.
  int64_t kernel_code_entry_byte_offset;
  // SGPR high-water count including body, ABI and entry requirements.
  uint32_t next_free_sgpr;
  // Total vector register allocation, including AGPRs on CDNA.
  uint32_t next_free_vgpr;
  // Total user SGPR count encoded in COMPUTE_PGM_RSRC2.
  uint32_t user_sgpr_count;
  // Descriptor flags controlling AMDHSA setup and code properties.
  loom_amdgpu_kernel_descriptor_flags_t flags;
} loom_amdgpu_kernel_descriptor_fields_t;

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

static uint32_t loom_amdgpu_kernel_descriptor_granulated_blocks(
    uint32_t register_count, uint32_t granule) {
  const uint32_t normalized_count = register_count == 0 ? 1u : register_count;
  return (normalized_count + granule - 1u) / granule - 1u;
}

static bool loom_amdgpu_kernel_descriptor_is_xnack_on_or_any(
    const loom_amdgpu_processor_info_t* processor) {
  const uint32_t xnack_selection = processor->properties.elf.feature_flags &
                                   LOOM_AMDGPU_ELF_FEATURE_XNACK_MASK_V4;
  return xnack_selection == LOOM_AMDGPU_ELF_FEATURE_XNACK_ANY_V4 ||
         xnack_selection == LOOM_AMDGPU_ELF_FEATURE_XNACK_ON_V4;
}

static bool loom_amdgpu_kernel_descriptor_supports_wgp_mode(
    const loom_amdgpu_processor_info_t* processor) {
  switch (processor->properties.kernel_descriptor.profile) {
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

static uint32_t loom_amdgpu_kernel_descriptor_user_sgpr_count_width(
    const loom_amdgpu_processor_info_t* processor) {
  if (processor->properties.kernel_descriptor.profile ==
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125) {
    return LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_GFX125_WIDTH;
  }
  return LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH;
}

static uint32_t loom_amdgpu_kernel_descriptor_legacy_extra_sgpr_count(
    const loom_amdgpu_processor_info_t* processor,
    loom_amdgpu_kernel_descriptor_flags_t flags) {
  uint32_t count = 2u;
  if (loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH) ||
      iree_any_bit_set(
          flags, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
    count = 6u;
  } else if (loom_amdgpu_kernel_descriptor_is_xnack_on_or_any(processor)) {
    count = 4u;
  }
  return count;
}

uint32_t loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
    loom_amdgpu_kernel_descriptor_flags_t flags) {
  if (iree_any_bit_set(
          flags, LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z)) {
    return 2;
  }
  return iree_any_bit_set(
             flags, LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y)
             ? 1u
             : 0u;
}

//===----------------------------------------------------------------------===//
// Byte encoding helpers
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void loom_amdgpu_kernel_descriptor_encode(
    const loom_amdgpu_processor_info_t* processor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel,
    loom_amdgpu_kernel_descriptor_options_t options,
    int64_t kernel_code_entry_byte_offset, iree_byte_span_t target_bytes) {
  const loom_amdgpu_kernel_descriptor_fields_t descriptor = {
      .group_segment_fixed_size = metadata_kernel->group_segment_fixed_size,
      .private_segment_fixed_size = metadata_kernel->private_segment_fixed_size,
      .kernarg_size = metadata_kernel->kernarg_segment_size,
      .kernel_code_entry_byte_offset = kernel_code_entry_byte_offset,
      .next_free_sgpr = metadata_kernel->sgpr_count,
      .next_free_vgpr = metadata_kernel->vgpr_count,
      .user_sgpr_count = options.minimum_user_sgpr_count,
      .flags = options.flags |
               (metadata_kernel->private_segment_fixed_size != 0
                    ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT
                    : 0) |
               (metadata_kernel->wavefront_size == 32
                    ? LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32
                    : 0),
  };
  const uint32_t user_sgpr_count_width =
      loom_amdgpu_kernel_descriptor_user_sgpr_count_width(processor);
  const uint32_t system_vgpr_workitem_id =
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor.flags);

  const uint32_t vgpr_granule =
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32)
          ? processor->properties.kernel_descriptor.vgpr_granules.wave32
          : processor->properties.kernel_descriptor.vgpr_granules.wave64;
  const uint32_t vgpr_block_count =
      loom_amdgpu_kernel_descriptor_granulated_blocks(descriptor.next_free_vgpr,
                                                      vgpr_granule);

  uint32_t sgpr_block_count = 0;
  if (!loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING)) {
    const uint32_t extra_sgpr_count =
        loom_amdgpu_kernel_descriptor_legacy_extra_sgpr_count(processor,
                                                              descriptor.flags);
    const uint32_t encoded_sgpr_count =
        descriptor.next_free_sgpr + extra_sgpr_count;
    sgpr_block_count =
        loom_amdgpu_kernel_descriptor_granulated_blocks(encoded_sgpr_count, 8);
  }

  uint32_t compute_pgm_rsrc3 = 0;
  if (loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET)) {
    loom_amdgpu_kernel_descriptor_set_bits_u32(
        &compute_pgm_rsrc3, LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_SHIFT,
        LOOM_AMDGPU_COMPUTE_PGM_RSRC3_ACCUM_OFFSET_WIDTH, options.accum_offset);
  }

  uint32_t compute_pgm_rsrc1 = 0;
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH, vgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH, sgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_32_SHIFT, 2, 3);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_16_64_SHIFT, 2,
      3);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT,
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT,
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT,
      loom_amdgpu_kernel_descriptor_supports_wgp_mode(processor));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_MEM_ORDERED_SHIFT,
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT,
      loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
          &processor->properties,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING));

  uint32_t compute_pgm_rsrc2 = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_PRIVATE_SEGMENT_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT));
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT,
      user_sgpr_count_width, descriptor.user_sgpr_count);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_X_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Y_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Z_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z));
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_INFO_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO));
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH, system_vgpr_workitem_id);

  uint16_t kernel_code_properties = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_BUFFER_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_PTR_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties, LOOM_AMDGPU_KERNEL_CODE_PROPERTY_QUEUE_PTR_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_KERNARG_SEGMENT_PTR_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_ID_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_FLAT_SCRATCH_INIT_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_SIZE_SHIFT,
      iree_any_bit_set(
          descriptor.flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_WAVEFRONT_SIZE32_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32));
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT,
      iree_any_bit_set(descriptor.flags,
                       LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK));

  memset(target_bytes.data, 0, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 0, descriptor.group_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 4, descriptor.private_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 8,
                                             descriptor.kernarg_size);
  loom_amdgpu_kernel_descriptor_store_le_i64(
      target_bytes.data + 16, descriptor.kernel_code_entry_byte_offset);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 44,
                                             compute_pgm_rsrc3);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 48,
                                             compute_pgm_rsrc1);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 52,
                                             compute_pgm_rsrc2);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 56,
                                             kernel_code_properties);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 58, 0);
}
