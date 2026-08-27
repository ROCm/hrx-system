// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/descriptor.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

loom_amdgpu_metadata_kernel_t MinimalMetadataKernel() {
  return loom_amdgpu_metadata_kernel_t{
      /*.name=*/IREE_SV("loom_kernel"),
      /*.descriptor_symbol=*/IREE_SV("loom_kernel.kd"),
      /*.kernarg_segment_size=*/0,
      /*.kernarg_segment_alignment=*/8,
      /*.wavefront_size=*/32,
      /*.group_segment_fixed_size=*/0,
      /*.private_segment_fixed_size=*/0,
      /*.sgpr_count=*/0,
      /*.vgpr_count=*/0,
      /*.max_flat_workgroup_size=*/64,
      /*.required_workgroup_size=*/{},
      /*.has_required_workgroup_size=*/false,
      /*.workgroup_cluster_size=*/{},
      /*.has_workgroup_cluster_size=*/false,
      /*.target_extensions=*/{},
  };
}

uint16_t LoadLeU16(const std::array<uint8_t, 65>& bytes, size_t offset) {
  return (uint16_t)bytes[offset] | ((uint16_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::array<uint8_t, 65>& bytes, size_t offset) {
  return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
         ((uint32_t)bytes[offset + 2] << 16) |
         ((uint32_t)bytes[offset + 3] << 24);
}

int64_t LoadLeI64(const std::array<uint8_t, 65>& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)bytes[offset + i] << (8 * i);
  }
  return (int64_t)value;
}

bool Bit(uint32_t value, uint32_t bit) { return ((value >> bit) & 1u) != 0; }

uint32_t Field(uint32_t value, uint32_t shift, uint32_t width) {
  return (value >> shift) & ((1u << width) - 1u);
}

void ExpectZeroRange(const std::array<uint8_t, 65>& bytes, size_t begin,
                     size_t end) {
  for (size_t i = begin; i < end; ++i) {
    EXPECT_EQ(bytes[i], 0u) << "offset " << i;
  }
}

constexpr uint32_t kComputePgmRsrc1VgprCountShift = 0;
constexpr uint32_t kComputePgmRsrc1VgprCountWidth = 6;
constexpr uint32_t kComputePgmRsrc1SgprCountShift = 6;
constexpr uint32_t kComputePgmRsrc1SgprCountWidth = 4;
constexpr uint32_t kComputePgmRsrc1Denorm16_64Shift = 18;
constexpr uint32_t kComputePgmRsrc1Denorm16_64Width = 2;
constexpr uint32_t kComputePgmRsrc1Dx10ClampShift = 21;
constexpr uint32_t kComputePgmRsrc1IeeeModeShift = 23;
constexpr uint32_t kComputePgmRsrc1WgpModeShift = 29;
constexpr uint32_t kComputePgmRsrc1MemoryOrderedShift = 30;
constexpr uint32_t kComputePgmRsrc1ForwardProgressShift = 31;

constexpr uint32_t kComputePgmRsrc2PrivateSegmentShift = 0;
constexpr uint32_t kComputePgmRsrc2UserSgprCountShift = 1;
constexpr uint32_t kComputePgmRsrc2UserSgprCountWidth = 5;
constexpr uint32_t kComputePgmRsrc2WorkgroupIdXShift = 7;
constexpr uint32_t kComputePgmRsrc2WorkgroupIdYShift = 8;
constexpr uint32_t kComputePgmRsrc2WorkgroupIdZShift = 9;
constexpr uint32_t kComputePgmRsrc2WorkgroupInfoShift = 10;
constexpr uint32_t kComputePgmRsrc2WorkitemIdShift = 11;
constexpr uint32_t kComputePgmRsrc2WorkitemIdWidth = 2;

constexpr uint32_t kComputePgmRsrc3AccumOffsetShift = 0;
constexpr uint32_t kComputePgmRsrc3AccumOffsetWidth = 6;

constexpr uint32_t kKernelCodePropertyPrivateSegmentBufferShift = 0;
constexpr uint32_t kKernelCodePropertyDispatchPtrShift = 1;
constexpr uint32_t kKernelCodePropertyQueuePtrShift = 2;
constexpr uint32_t kKernelCodePropertyKernargSegmentPtrShift = 3;
constexpr uint32_t kKernelCodePropertyDispatchIdShift = 4;
constexpr uint32_t kKernelCodePropertyFlatScratchInitShift = 5;
constexpr uint32_t kKernelCodePropertyPrivateSegmentSizeShift = 6;
constexpr uint32_t kKernelCodePropertyWavefrontSize32Shift = 10;
constexpr uint32_t kKernelCodePropertyUsesDynamicStackShift = 11;

bool SupportsWgpMode(const loom_amdgpu_processor_info_t* processor) {
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

TEST(AmdgpuDescriptorTest, WritesNoArgGfx1100Descriptor) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, -64, &descriptor));
  IREE_ASSERT_OK(
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));

  std::array<uint8_t, 65> bytes;
  bytes.fill(0xcc);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 0), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 4), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 8), 0u);
  ExpectZeroRange(bytes, 12, 16);
  EXPECT_EQ(LoadLeI64(bytes, 16), -64);
  ExpectZeroRange(bytes, 24, 44);
  EXPECT_EQ(LoadLeU32(bytes, 44), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xe0ac0000u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0400u);
  EXPECT_EQ(LoadLeU16(bytes, 58), 0u);
  ExpectZeroRange(bytes, 60, 64);
  EXPECT_EQ(bytes[64], 0xccu);
}

TEST(AmdgpuDescriptorTest, EncodesNamedComputePgmRsrc1TargetFeatures) {
  struct Case {
    const char* processor;
    uint32_t wavefront_size;
    uint32_t expected_vgpr_blocks;
    uint32_t expected_sgpr_blocks;
    bool expected_dx10_and_ieee;
    bool expected_wgp_mode;
    bool expected_memory_ordered;
    bool expected_forward_progress;
    bool expected_accum_offset;
  };
  static constexpr Case kCases[] = {
      {"gfx942", 64, 1, 3, true, false, false, false, true},
      {"gfx950", 64, 1, 3, true, false, false, false, true},
      {"gfx1100", 32, 1, 0, true, true, true, true, false},
      {"gfx1200", 32, 1, 0, false, true, true, true, false},
      {"gfx1250", 32, 0, 0, false, false, true, true, false},
  };
  for (const Case& c : kCases) {
    loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
    metadata.wavefront_size = c.wavefront_size;
    metadata.sgpr_count = 20;
    metadata.vgpr_count = 9;

    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        iree_make_cstring_view(c.processor), &metadata, 0, &descriptor));

    std::array<uint8_t, 65> bytes;
    bytes.fill(0);
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
        &descriptor, iree_make_byte_span(bytes.data(), bytes.size())))
        << c.processor;

    const uint32_t compute_pgm_rsrc1 = LoadLeU32(bytes, 48);
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1VgprCountShift,
                    kComputePgmRsrc1VgprCountWidth),
              c.expected_vgpr_blocks)
        << c.processor;
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1SgprCountShift,
                    kComputePgmRsrc1SgprCountWidth),
              c.expected_sgpr_blocks)
        << c.processor;
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1Denorm16_64Shift,
                    kComputePgmRsrc1Denorm16_64Width),
              3u)
        << c.processor;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1Dx10ClampShift),
              c.expected_dx10_and_ieee)
        << c.processor;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1IeeeModeShift),
              c.expected_dx10_and_ieee)
        << c.processor;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1WgpModeShift),
              c.expected_wgp_mode)
        << c.processor;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1MemoryOrderedShift),
              c.expected_memory_ordered)
        << c.processor;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1ForwardProgressShift),
              c.expected_forward_progress)
        << c.processor;

    const uint32_t compute_pgm_rsrc3 = LoadLeU32(bytes, 44);
    EXPECT_EQ(Field(compute_pgm_rsrc3, kComputePgmRsrc3AccumOffsetShift,
                    kComputePgmRsrc3AccumOffsetWidth),
              c.expected_accum_offset ? 2u : 0u)
        << c.processor;
  }
}

TEST(AmdgpuDescriptorTest, EncodesComputePgmFieldsForEveryDescriptorProfile) {
  iree_host_size_t checked_count = 0;
  for (iree_host_size_t i = 0; i < loom_amdgpu_target_info_processor_count();
       ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);
    if (processor->properties.kernel_descriptor.profile ==
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE) {
      continue;
    }
    ++checked_count;

    loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
    metadata.wavefront_size = processor->properties.wavefront.default_size;
    metadata.sgpr_count = 20;
    metadata.vgpr_count = 9;

    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        processor->name, &metadata, 0, &descriptor))
        << processor->name.data;

    std::array<uint8_t, 65> bytes;
    bytes.fill(0);
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
        &descriptor, iree_make_byte_span(bytes.data(), bytes.size())))
        << processor->name.data;

    const uint32_t compute_pgm_rsrc1 = LoadLeU32(bytes, 48);
    const uint32_t vgpr_granule =
        metadata.wavefront_size == 32
            ? processor->properties.kernel_descriptor.vgpr_granules.wave32
            : processor->properties.kernel_descriptor.vgpr_granules.wave64;
    const uint32_t expected_vgpr_blocks =
        (metadata.vgpr_count + vgpr_granule - 1) / vgpr_granule - 1;
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1VgprCountShift,
                    kComputePgmRsrc1VgprCountWidth),
              expected_vgpr_blocks)
        << processor->name.data;

    const uint32_t expected_sgpr_blocks =
        loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
            &processor->properties,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING)
            ? 0u
            : 3u;
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1SgprCountShift,
                    kComputePgmRsrc1SgprCountWidth),
              expected_sgpr_blocks)
        << processor->name.data;
    EXPECT_EQ(Field(compute_pgm_rsrc1, kComputePgmRsrc1Denorm16_64Shift,
                    kComputePgmRsrc1Denorm16_64Width),
              3u)
        << processor->name.data;
    const bool has_dx10_clamp_and_ieee_mode =
        loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
            &processor->properties,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE);
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1Dx10ClampShift),
              has_dx10_clamp_and_ieee_mode)
        << processor->name.data;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1IeeeModeShift),
              has_dx10_clamp_and_ieee_mode)
        << processor->name.data;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1WgpModeShift),
              SupportsWgpMode(processor))
        << processor->name.data;
    const bool uses_gfx10_sgpr_encoding =
        loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
            &processor->properties,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING);
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1MemoryOrderedShift),
              uses_gfx10_sgpr_encoding)
        << processor->name.data;
    EXPECT_EQ(Bit(compute_pgm_rsrc1, kComputePgmRsrc1ForwardProgressShift),
              uses_gfx10_sgpr_encoding)
        << processor->name.data;

    const uint32_t compute_pgm_rsrc3 = LoadLeU32(bytes, 44);
    EXPECT_EQ(Field(compute_pgm_rsrc3, kComputePgmRsrc3AccumOffsetShift,
                    kComputePgmRsrc3AccumOffsetWidth),
              loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
                  &processor->properties,
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET)
                  ? 2u
                  : 0u)
        << processor->name.data;
  }
  EXPECT_GT(checked_count, 0u);
}

TEST(AmdgpuDescriptorTest, EncodesNamedSetupAndCodePropertyFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.private_segment_fixed_size = 16;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.user_sgpr_count = 9;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  const uint32_t compute_pgm_rsrc2 = LoadLeU32(bytes, 52);
  EXPECT_TRUE(Bit(compute_pgm_rsrc2, kComputePgmRsrc2PrivateSegmentShift));
  EXPECT_EQ(Field(compute_pgm_rsrc2, kComputePgmRsrc2UserSgprCountShift,
                  kComputePgmRsrc2UserSgprCountWidth),
            9u);
  EXPECT_TRUE(Bit(compute_pgm_rsrc2, kComputePgmRsrc2WorkgroupIdXShift));
  EXPECT_TRUE(Bit(compute_pgm_rsrc2, kComputePgmRsrc2WorkgroupIdYShift));
  EXPECT_TRUE(Bit(compute_pgm_rsrc2, kComputePgmRsrc2WorkgroupIdZShift));
  EXPECT_TRUE(Bit(compute_pgm_rsrc2, kComputePgmRsrc2WorkgroupInfoShift));
  EXPECT_EQ(Field(compute_pgm_rsrc2, kComputePgmRsrc2WorkitemIdShift,
                  kComputePgmRsrc2WorkitemIdWidth),
            2u);

  const uint16_t kernel_code_properties = LoadLeU16(bytes, 56);
  EXPECT_FALSE(Bit(kernel_code_properties,
                   kKernelCodePropertyPrivateSegmentBufferShift));
  EXPECT_TRUE(Bit(kernel_code_properties, kKernelCodePropertyDispatchPtrShift));
  EXPECT_TRUE(Bit(kernel_code_properties, kKernelCodePropertyQueuePtrShift));
  EXPECT_TRUE(
      Bit(kernel_code_properties, kKernelCodePropertyKernargSegmentPtrShift));
  EXPECT_TRUE(Bit(kernel_code_properties, kKernelCodePropertyDispatchIdShift));
  EXPECT_FALSE(
      Bit(kernel_code_properties, kKernelCodePropertyFlatScratchInitShift));
  EXPECT_TRUE(
      Bit(kernel_code_properties, kKernelCodePropertyPrivateSegmentSizeShift));
  EXPECT_TRUE(
      Bit(kernel_code_properties, kKernelCodePropertyWavefrontSize32Shift));
  EXPECT_TRUE(
      Bit(kernel_code_properties, kKernelCodePropertyUsesDynamicStackShift));
}

TEST(AmdgpuDescriptorTest, EnablesKernargSegmentPointerFromMetadata) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.kernarg_segment_size = 8;
  metadata.sgpr_count = 2;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, -64, &descriptor));
  IREE_ASSERT_OK(
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 8), 8u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000004u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0408u);
}

TEST(AmdgpuDescriptorTest, EncodesDispatchAndKernargUserSgprs) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.kernarg_segment_size = 16;
  metadata.sgpr_count = 4;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, -64, &descriptor));
  descriptor.user_sgpr_count = 4;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000008u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x040au);
}

TEST(AmdgpuDescriptorTest, EncodesResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.group_segment_fixed_size = 128;
  metadata.private_segment_fixed_size = 16;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 256, &descriptor));
  descriptor.user_sgpr_count = 2;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 0), 128u);
  EXPECT_EQ(LoadLeU32(bytes, 4), 16u);
  EXPECT_EQ(LoadLeU32(bytes, 8), 24u);
  EXPECT_EQ(LoadLeI64(bytes, 16), 256);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xe0ac0001u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00001085u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0c08u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx942ResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.wavefront_size = 64;
  metadata.group_segment_fixed_size = 128;
  metadata.private_segment_fixed_size = 16;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx942"), &metadata, 256, &descriptor));
  descriptor.user_sgpr_count = 2;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 0), 128u);
  EXPECT_EQ(LoadLeU32(bytes, 4), 16u);
  EXPECT_EQ(LoadLeU32(bytes, 8), 24u);
  EXPECT_EQ(LoadLeI64(bytes, 16), 256);
  EXPECT_EQ(LoadLeU32(bytes, 44), 0x00000002u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0x00ac00c1u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00001085u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0808u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx942SmallWorkgroupIdBoundary) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.wavefront_size = 64;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 12;
  metadata.vgpr_count = 3;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx942"), &metadata, 0, &descriptor));
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 44), 0x00000000u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0x00ac0080u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000084u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0008u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx950ResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.wavefront_size = 64;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx950"), &metadata, 0, &descriptor));
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 44), 0x00000002u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0x00ac00c1u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000084u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0008u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx1200ResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1200"), &metadata, 0, &descriptor));
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 44), 0x00000000u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xe00c0001u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000084u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0408u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx1250ResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1250"), &metadata, 0, &descriptor));
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 44), 0x00000000u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xc00c0000u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000084u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0408u);
}

TEST(AmdgpuDescriptorTest, EncodesGfx1250SixBitUserSgprCount) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1250"), &metadata, 0, &descriptor));
  descriptor.user_sgpr_count = 40;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000050u);
}

TEST(AmdgpuDescriptorTest, RejectsWave32MetadataOnGfx942) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_kernel_descriptor_initialize_from_metadata(
                            IREE_SV("gfx942"), &metadata, 0, &descriptor));
}

TEST(AmdgpuDescriptorTest, SupportsWave64MetadataOnGfx1100AndGfx1200) {
  static const iree_string_view_t processors[] = {
      IREE_SVL("gfx1100"),
      IREE_SVL("gfx1200"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(processors); ++i) {
    loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
    metadata.wavefront_size = 64;
    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        processors[i], &metadata, 0, &descriptor));
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor,
                                                                   &metadata));
  }
}

TEST(AmdgpuDescriptorTest, RejectsWave64MetadataOnGfx1250) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.wavefront_size = 64;
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_kernel_descriptor_initialize_from_metadata(
                            IREE_SV("gfx1250"), &metadata, 0, &descriptor));
}

TEST(AmdgpuDescriptorTest, RejectsWave64DescriptorOnGfx1250) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1250"), &metadata, 0, &descriptor));
  descriptor.flags &= ~LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32;

  std::array<uint8_t, 64> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest, RejectsSparseWorkitemIdFlags) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));

  std::array<uint8_t, 64> bytes;
  descriptor.flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  descriptor.flags = LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                     LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
                     LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest, SupportsGfx11ProcessorVariants) {
  static const iree_string_view_t processors[] = {
      IREE_SVL("gfx1100"), IREE_SVL("gfx1101"), IREE_SVL("gfx1102"),
      IREE_SVL("gfx1103"), IREE_SVL("gfx1150"), IREE_SVL("gfx1151"),
      IREE_SVL("gfx1152"), IREE_SVL("gfx1153"),
  };
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(processors); ++i) {
    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        processors[i], &metadata, -64, &descriptor));
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor,
                                                                   &metadata));
  }
}

TEST(AmdgpuDescriptorTest, SupportsEnabledProcessorVariants) {
  const struct {
    iree_string_view_t processor;
    uint32_t wavefront_size;
  } cases[] = {
      {IREE_SV("gfx942"), 64},  {IREE_SV("gfx950"), 64},
      {IREE_SV("gfx1200"), 32}, {IREE_SV("gfx1201"), 32},
      {IREE_SV("gfx1250"), 32}, {IREE_SV("gfx1251"), 32},
  };
  for (const auto& c : cases) {
    loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
    metadata.wavefront_size = c.wavefront_size;
    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        c.processor, &metadata, -64, &descriptor));
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor,
                                                                   &metadata));
  }
}

TEST(AmdgpuDescriptorTest, RejectsMetadataMismatch) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.kernarg_size = 4;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));
}

TEST(AmdgpuDescriptorTest, RejectsTooFewUserSgprs) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.user_sgpr_count = 1;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  std::array<uint8_t, 64> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest,
     RejectsLegacyFlatScratchUserSgprsOnArchitectedFlatScratchTargets) {
  const loom_amdgpu_kernel_descriptor_flags_t legacy_flat_scratch_flags[] = {
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT,
  };
  iree_host_size_t checked_count = 0;
  for (iree_host_size_t i = 0; i < loom_amdgpu_target_info_processor_count();
       ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);
    if (processor->properties.kernel_descriptor.profile ==
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE ||
        !loom_amdgpu_processor_properties_kernel_descriptor_has_flags(
            &processor->properties,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH)) {
      continue;
    }
    ++checked_count;

    for (const loom_amdgpu_kernel_descriptor_flags_t flag :
         legacy_flat_scratch_flags) {
      loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
      metadata.wavefront_size = processor->properties.wavefront.default_size;

      loom_amdgpu_kernel_descriptor_t descriptor = {};
      IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
          processor->name, &metadata, 0, &descriptor))
          << processor->name.data;
      descriptor.user_sgpr_count = 16;
      descriptor.flags |= flag;

      std::array<uint8_t, 64> bytes;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          loom_amdgpu_kernel_descriptor_write(
              &descriptor, iree_make_byte_span(bytes.data(), bytes.size())))
          << processor->name.data;
    }
  }
  EXPECT_GT(checked_count, 0u);
}

TEST(AmdgpuDescriptorTest, RejectsUnsupportedTarget) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_amdgpu_kernel_descriptor_initialize_from_metadata(
                            IREE_SV("gfx900"), &metadata, 0, &descriptor));
}

TEST(AmdgpuDescriptorTest, RejectsSmallOutputBuffer) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));

  std::array<uint8_t, 63> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

}  // namespace
}  // namespace loom
