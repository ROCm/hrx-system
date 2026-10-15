// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    function_.metadata = &metadata_;
    function_.kernarg = &layout_;
  }
  iree_status_t Prepare(
      iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags = 0,
      iree_const_byte_span_t constants = iree_const_byte_span_empty(),
      iree_host_size_t binding_count = 0) {
    return iree_hal_amd_gpu_pm4_dispatch_prepare(
        &function_, 65536, config, constants, binding_count, flags, &dispatch_);
  }
  // Normalized image metadata; launch inputs remain public caller values.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Argument representation used by launch admission.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Prepared executable function borrowing this fixture's metadata.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Admitted native launch geometry.
  iree_hal_amd_gpu_pm4_dispatch_t dispatch_ = {};
};

TEST_F(DispatchTest, DefaultGeometryUsesRequiredMetadata) {
  auto config = iree_hal_make_static_dispatch_config(7, 3, 2);
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 1u);
  metadata_.has_required_workgroup_size = true;
  metadata_.required_workgroup_size[0] = 32;
  metadata_.required_workgroup_size[1] = 2;
  metadata_.required_workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 32u);
  EXPECT_EQ(dispatch_.workgroup_size[1], 2u);
  EXPECT_EQ(dispatch_.workgroup_count[0], 7u);
  config.workgroup_size[0] = 16;
  config.workgroup_size[1] = 2;
  config.workgroup_size[2] = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, ExactItemsAdmitPartialGroupsAndUniformRequirements) {
  iree_hal_dispatch_config_t config = {};
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.workitem_count[0] = 65;
  config.workitem_count[1] = 8;
  config.workitem_count[2] = 1;
  const auto flags = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.workgroup_count[0], 3u);
  EXPECT_EQ(dispatch_.workgroup_count[1], 2u);
  EXPECT_EQ(dispatch_.partial_size[0], 1u);
  EXPECT_EQ(dispatch_.partial_size[1], 4u);
  EXPECT_EQ(dispatch_.partial_size[2], 1u);
  metadata_.uniform_workgroup_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config, flags));
  config.workitem_count[0] = 64;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.partial_size[0], 0u);
  EXPECT_EQ(dispatch_.partial_size[1], 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, GeometryAndLdsRespectExecutableLimits) {
  auto config = iree_hal_make_static_dispatch_config(5, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  metadata_.max_flat_workgroup_size = 16;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_flat_workgroup_size = 32;
  metadata_.max_workgroup_count[0] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_workgroup_count[0] = 5;
  metadata_.group_segment_fixed_size = 32768;
  config.dynamic_workgroup_local_memory = 32768;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.group_segment_byte_length, 65536u);
  ++config.dynamic_workgroup_local_memory;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
  config.dynamic_workgroup_local_memory = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
}

TEST_F(DispatchTest, UnprovisionedRuntimeServicesFailBeforeCapture) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  function_.launch.requirements =
      IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  function_.launch.requirements = 0;
  layout_.requirements = IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HOSTCALL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  layout_.requirements = 0;
  metadata_.has_workgroup_cluster_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
}

TEST_F(DispatchTest, DirectArgumentsRespectDenseAndNativeExtents) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  std::array<uint8_t, 24> constants = {};
  layout_.byte_length = 24;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 8;
  IREE_ASSERT_OK(Prepare(config, 0, {constants.data(), 8}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, 0, {constants.data(), 12}));
  const auto native = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 12}));
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 24}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, native, {constants.data(), 11}));
}

TEST_F(DispatchTest, PacketGeometryIncludesExactItemsAndRejectsOverflow) {
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t packet_info = {
      .kernel_object = UINT64_C(0x123456780040),
  };
  function_.packet = &packet_info;
  function_.launch.requirements =
      IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER;
  metadata_.group_segment_fixed_size = 128;
  auto config = iree_hal_make_static_dispatch_config(3, 2, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.dynamic_workgroup_local_memory = 384;
  IREE_ASSERT_OK(Prepare(config));
  iree_hal_amd_gpu_pm4_dispatch_packet_t packet;
  iree_hal_amd_gpu_pm4_dispatch_emplace_packet(&function_, &dispatch_, 0,
                                               &packet);
  std::array<uint32_t, 16> words;
  memcpy(words.data(), &packet, sizeof(packet));
  EXPECT_EQ(words, (std::array<uint32_t, 16>{0x00020002, 0x00040020, 1, 96, 8,
                                             1, 0, 512, 0x56780040, 0x1234, 0,
                                             0, 0, 0, 0, 0}));
  config.workgroup_count[0] = UINT32_MAX / 32;
  IREE_ASSERT_OK(Prepare(config));
  ++config.workgroup_count[0];
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  config.workitem_count[0] = UINT32_MAX;
  config.workitem_count[1] = 7;
  config.workitem_count[2] = 1;
  const auto exact = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, exact));
  iree_hal_amd_gpu_pm4_dispatch_emplace_packet(&function_, &dispatch_, exact,
                                               &packet);
  EXPECT_EQ(packet.grid_size[0], UINT32_MAX);
  EXPECT_EQ(packet.grid_size[1], 7u);
  EXPECT_EQ(packet.grid_size[2], 1u);
  config.workitem_count[0] = config.workitem_count[1] =
      config.workitem_count[2] = 0;
  const auto indirect = IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
  IREE_ASSERT_OK(Prepare(config, indirect));
  iree_hal_amd_gpu_pm4_dispatch_emplace_packet(&function_, &dispatch_, indirect,
                                               &packet);
  EXPECT_EQ(packet.setup, 3u);
  EXPECT_EQ(packet.grid_size[0] | packet.grid_size[1] | packet.grid_size[2],
            0u);
}

TEST_F(DispatchTest, InvocationStoragePreservesArgumentAlignmentAndOwnership) {
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t packet_info = {};
  layout_.byte_length = 257;
  for (bool packet : {false, true}) {
    function_.packet = packet ? &packet_info : nullptr;
    for (uint32_t alignment : {8u, 64u, 256u}) {
      layout_.alignment = alignment;
      for (auto parameters : std::array<iree_hal_dispatch_flags_t, 3>{
               0, IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
               IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
        for (auto arguments : std::array<iree_hal_dispatch_flags_t, 4>{
                 0, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
                 IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS,
                 IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS}) {
          const auto flags = parameters | arguments;
          const bool external =
              iree_hal_dispatch_uses_indirect_arguments(flags);
          const auto storage =
              iree_hal_amd_gpu_pm4_dispatch_storage(&function_, flags);
          const uint32_t kernarg_length = external ? 0 : 257;
          const uint32_t argument_alignment = external ? 1 : alignment;
          EXPECT_EQ(storage.alignment, packet
                                           ? std::max(argument_alignment, 64u)
                                           : argument_alignment);
          EXPECT_EQ(storage.packet_offset, packet && !external ? 320u : 0u);
          EXPECT_EQ(storage.length,
                    packet ? storage.packet_offset + 64 + (parameters ? 16 : 0)
                           : kernarg_length);
        }
      }
    }
  }
}

TEST_F(DispatchTest, PacketTemplatesInitializePaddingAndPreparationInputs) {
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t packet_info = {
      .kernel_object = UINT64_C(0x123456780040),
  };
  function_.packet = &packet_info;
  layout_.byte_length = layout_.native_byte_length = 17;
  std::array<uint8_t, 17> constants;
  constants.fill(0x42);
  for (bool external : {false, true}) {
    const auto flags =
        IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
        (external ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS
                  : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
    IREE_ASSERT_OK(Prepare(iree_hal_make_static_dispatch_config(1, 1, 1), flags,
                           external ? iree_const_byte_span_empty()
                                    : iree_make_const_byte_span(
                                          constants.data(), constants.size()),
                           external ? 1 : 0));
    const auto storage =
        iree_hal_amd_gpu_pm4_dispatch_storage(&function_, flags);
    alignas(64) std::array<uint8_t, 192> bytes;
    bytes.fill(0xCD);
    iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
        &function_, &dispatch_,
        iree_make_const_byte_span(constants.data(), constants.size()), nullptr,
        flags, bytes.data());
    if (!external) {
      EXPECT_EQ(memcmp(bytes.data(), constants.data(), constants.size()), 0);
      for (size_t i = constants.size(); i < storage.packet_offset; ++i) {
        EXPECT_EQ(bytes[i], 0u) << i;
      }
    }
    const auto* packet =
        reinterpret_cast<const iree_hal_amd_gpu_pm4_dispatch_packet_t*>(
            bytes.data() + storage.packet_offset);
    EXPECT_EQ(packet->kernel_object, packet_info.kernel_object);
    EXPECT_EQ(packet->kernarg_address, 0u);
    EXPECT_EQ(packet->completion_signal, 0u);
    for (size_t i = storage.packet_offset + 64; i < storage.length; ++i) {
      EXPECT_EQ(bytes[i], 0u) << i;
    }
    for (size_t i = storage.length; i < bytes.size(); ++i) {
      EXPECT_EQ(bytes[i], 0xCD) << i;
    }
  }
}

TEST_F(DispatchTest, PreparationProgramEmitsCompleteOrderedLaunch) {
  for (auto gfxip : {iree_hal_amd_gpu_gfxip_version_t{11, 0, 0},
                     {11, 5, 1},
                     {11, 7, 0},
                     {12, 0, 1},
                     {12, 5, 0}}) {
    iree_hal_amd_gpu_pm4_dispatch_program_t program;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_dispatch_program_query(gfxip, &program));
    EXPECT_GT(program.code.data_length, 0u);
    EXPECT_LE(program.code.data_length + 192,
              IREE_HAL_AMD_GPU_PM4_DISPATCH_PROGRAM_LENGTH);
    iree_hal_amd_gpu_pm4_launch_t launch;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        gfxip, &program.kernel, UINT64_C(0x123456780000), &launch));
    EXPECT_EQ(launch.requirements, 0u);
    EXPECT_EQ(launch.user_data_count, 2u);
    EXPECT_EQ(launch.user_data.kernarg_pointer, 0u);
    std::array<uint32_t, IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS + 2> words;
    words.fill(0xCAFEBABE);
    auto* end = iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
        words.data() + 1, &launch, UINT64_C(0x56789ABCDEF0), 0x103E0);
    const uint32_t resource1 = gfxip.major == 11  ? 0xE0AF0000
                               : gfxip.minor == 0 ? 0xE00F0000
                                                  : 0xC00F0000;
    const uint32_t resource3_register =
        gfxip.major == 12 && gfxip.minor == 5 ? 0x223 : 0x228;
    const std::vector<uint32_t> expected = {
        0xC0027602, 0x20C,      0x34567800, 0x12,       0xC0027602,
        0x212,      resource1,  4,          0xC0017602, resource3_register,
        0,          0xC0017602, 0x215,      0,          0xC0067602,
        0x204,      0,          0,          0,          1,
        1,          1,          0xC0027602, 0x240,      0x9ABCDEF0,
        0x5678,     0xC0031502, 1,          1,          1,
        0x8005,     0xC0004600, 0x407,      0xC0065800, 0,
        UINT32_MAX, 0xFF,       0,          0,          0xA,
        0x103E0};
    EXPECT_EQ(end - words.data() - 1,
              IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS);
    EXPECT_EQ(std::vector<uint32_t>(words.data() + 1, end), expected);
    EXPECT_EQ(words.front(), 0xCAFEBABEu);
    EXPECT_EQ(words.back(), 0xCAFEBABEu);
  }
  iree_hal_amd_gpu_pm4_dispatch_program_t program;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_pm4_dispatch_program_query({9, 4, 2}, &program));
}

TEST_F(DispatchTest, IndirectGeometryIgnoresHostCountsAndRejectsExactItems) {
  auto config =
      iree_hal_make_static_dispatch_config(UINT32_MAX, UINT32_MAX, UINT32_MAX);
  for (int i = 0; i < 3; ++i) {
    metadata_.max_workgroup_count[i] = 1;
  }
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    IREE_ASSERT_OK(Prepare(config, flags));
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(dispatch_.workgroup_count[i], 0u);
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Prepare(config, flags | IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT));
    layout_.indirect_copy_count = 32;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          Prepare(config, flags));
    layout_.indirect_copy_count = 0;
  }
}

TEST_F(DispatchTest, IndirectParameterReferencesCoverThreeAlignedCounts) {
  std::array<uint32_t, 4> storage = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS, sizeof(storage),
      iree_make_byte_span(storage.data(), sizeof(storage)), {},
      iree_allocator_system(), &buffer));
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    const auto validate = [&](iree_hal_buffer_ref_t reference) {
      return iree_hal_amd_gpu_pm4_dispatch_validate_direct_bindings(
          &layout_, reference, {}, flags);
    };
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, validate({}));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          validate(iree_hal_make_buffer_ref(buffer, 1, 12)));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          validate(iree_hal_make_buffer_ref(buffer, 4, 8)));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        validate(iree_hal_make_buffer_ref(buffer, 8, IREE_HAL_WHOLE_BUFFER)));
    IREE_ASSERT_OK(validate(iree_hal_make_buffer_ref(buffer, 4, 12)));
    IREE_ASSERT_OK(
        validate(iree_hal_make_buffer_ref(buffer, 4, IREE_HAL_WHOLE_BUFFER)));
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(DispatchTest,
       IndirectPacketsCopyOnlyAdmittedCountBytesAndJoinVisibility) {
  std::array<iree_hal_amd_gpu_metadata_argument_t, 3> arguments = {};
  for (uint32_t i = 0; i < 3; ++i) {
    arguments[i].kind = static_cast<iree_hal_amd_gpu_metadata_argument_kind_t>(
        IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X + i);
    arguments[i].offset = 16 + i * 4;
    arguments[i].size = 4;
  }
  metadata_.has_argument_metadata = true;
  metadata_.arguments = arguments.data();
  metadata_.argument_count = arguments.size();
  metadata_.kernarg_segment_size = 32;
  metadata_.kernarg_segment_alignment = 8;
  iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
      &metadata_, iree_allocator_system(), &layout));
  ASSERT_EQ(layout->indirect_copy_count, 1u);
  std::array<uint32_t, 28> words;
  words.fill(0xCAFEBABE);
  auto* end = iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
      words.data() + 1, layout, layout->indirect_copy_count, 0x1234FFFFFFF4,
      0x567800001000, 0x103E0, 0x8001);
  EXPECT_EQ(end - words.data() - 1,
            iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(1));
  const std::vector<uint32_t> expected = {
      0xC0055000, 0x60300000, 0xFFFFFFF4, 0x1234, 0x1010, 0x5678, 12,
      0xC0055000, 0,          0,          0,      0,      0,      0,
      0xC0065800, 0,          UINT32_MAX, 0xFF,   0,      0,      0xA,
      0x103E0,    0xC0021602, 0xFFFFFFF4, 0x1234, 0x8001};
  EXPECT_EQ(std::vector<uint32_t>(words.begin() + 1, words.end() - 1),
            expected);
  EXPECT_EQ(words.front(), 0xCAFEBABEu);
  EXPECT_EQ(words.back(), 0xCAFEBABEu);
  // Caller-owned native arguments receive no writes or inferred geometry.
  end = iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
      words.data(), layout, 0, 0x123400000010, 0, 0, 0x8001);
  EXPECT_EQ(end - words.data(), 4);
  EXPECT_EQ((std::array<uint32_t, 4>{words[0], words[1], words[2], words[3]}),
            (std::array<uint32_t, 4>{0xC0021602, 0x10, 0x1234, 0x8001}));
  iree_hal_amd_gpu_kernarg_layout_free(layout);
}

}  // namespace
