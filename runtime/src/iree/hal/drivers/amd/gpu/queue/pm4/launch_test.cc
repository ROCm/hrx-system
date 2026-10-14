// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_test_util.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

static_assert(sizeof(iree_hal_amd_gpu_pm4_launch_t) <= 64);

class LaunchTest : public ::testing::Test {
 protected:
  // Representative compiler descriptor with only a kernarg pointer enabled.
  iree_hal_amd_gpu_image_kernel_t kernel_ = {
      .entry_offset = 0x200,
      .compute_pgm_rsrc1 = 0xe0af0000,
      .compute_pgm_rsrc2 = 0x84,
      .compute_pgm_rsrc3 = 0x20,
      .code_properties = 0x408,
  };
  // A nontrivial placed address exercising both PGM register words.
  static constexpr uint64_t kImageAddress = UINT64_C(0x123456780000);
  // Guard outside the exact binding/dispatch output extent.
  static constexpr uint32_t kGuard = 0xd8c37f91;

  iree_hal_amd_gpu_pm4_launch_t Prepare(
      iree_hal_amd_gpu_gfxip_version_t target = {11, 0, 0}) {
    iree_hal_amd_gpu_pm4_launch_t launch;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        target, &kernel_, kImageAddress, &launch));
    return launch;
  }
};

TEST_F(LaunchTest, CompleteDirectBinding) {
  auto launch = Prepare();
  EXPECT_EQ(launch.requirements, 0);
  EXPECT_EQ(launch.user_data.kernarg_pointer, 0);
  EXPECT_EQ(launch.user_data_count, 2);
  EXPECT_EQ(launch.wavefront_size_log2, 5);
  const uint32_t workgroup_size[] = {32, 1, 1};
  const uint32_t partial_size[] = {0, 0, 0};
  const uint32_t groups[] = {7, 3, 1};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = UINT64_C(0xabcdef123450),
  };
  std::array<uint32_t, 64> storage;
  storage.fill(kGuard);
  auto* words = storage.data() + 1;
  auto* end = iree_hal_amd_gpu_pm4_launch_emit(words, &launch, workgroup_size,
                                               partial_size, 513, &inputs);
  end = iree_hal_amd_gpu_pm4_emit_dispatch_direct(end, groups,
                                                  launch.dispatch_initiator);
  const std::vector<uint32_t> expected = {
      0xc0027602, 0x20c,   0x34567802, 0x12,  0xc0027602, 0x212,
      0xe0af0000, 0x10084, 0xc0017602, 0x228, 0x20,       0xc0017602,
      0x215,      0,       0xc0067602, 0x204, 0,          0,
      0,          32,      1,          1,     0xc0027602, 0x240,
      0xef123450, 0xabcd,  0xc0031502, 7,     3,          1,
      0x8005,
  };
  ASSERT_EQ(end - words, expected.size());
  EXPECT_EQ(std::vector<uint32_t>(words, end), expected);
  EXPECT_EQ(storage.front(), kGuard);
  for (auto* word = end; word != storage.data() + storage.size(); ++word) {
    EXPECT_EQ(*word, kGuard);
  }
}

TEST_F(LaunchTest, PartialWave64GeometryAndIndirectControl) {
  kernel_.code_properties = 8;
  auto launch = Prepare({12, 0, 1});
  const uint32_t workgroup_size[] = {32, 4, 2};
  const uint32_t partial_size[] = {17, 4, 1};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = UINT64_C(0x12340000),
  };
  std::array<uint32_t, 64> words = {};
  auto* end = iree_hal_amd_gpu_pm4_launch_emit(
      words.data(), &launch, workgroup_size, partial_size, 65536, &inputs);
  end = iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
      end, UINT64_C(0x567812340004), launch.dispatch_initiator | 2u);
  const std::vector<uint32_t> expected = {
      0xc0027602, 0x20c,    0x34567802, 0x12,       0xc0027602, 0x212,
      0xe0af0000, 0x400084, 0xc0017602, 0x228,      0x20,       0xc0017602,
      0x215,      0x400000, 0xc0067602, 0x204,      0,          0,
      0,          0x110020, 0x40004,    0x10002,    0xc0027602, 0x240,
      0x12340000, 0,        0xc0021602, 0x12340004, 0x5678,     7,
  };
  EXPECT_EQ(std::vector<uint32_t>(words.data(), end), expected);
  EXPECT_EQ(launch.wavefront_size_log2, 6);
}

TEST_F(LaunchTest, TargetRowsAndWaveModes) {
  for (auto target : {iree_hal_amd_gpu_gfxip_version_t{11, 0, 0},
                      {11, 5, 1},
                      {11, 7, 0},
                      {12, 0, 0},
                      {12, 5, 0}}) {
    for (uint32_t wavefront_size : {32u, 64u}) {
      SCOPED_TRACE(::testing::Message() << target.major << '.' << target.minor
                                        << " wave" << wavefront_size);
      kernel_.code_properties = wavefront_size == 32 ? 0x408 : 8;
      iree_hal_amd_gpu_pm4_launch_t launch;
      if (target.major == 12 && target.minor == 5 && wavefront_size == 64) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                              iree_hal_amd_gpu_pm4_launch_initialize(
                                  target, &kernel_, kImageAddress, &launch));
      } else {
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
            target, &kernel_, kImageAddress, &launch));
        EXPECT_EQ(launch.resource3_register,
                  target.major == 12 && target.minor == 5 ? 0x2e23u : 0x2e28u);
        EXPECT_EQ(launch.dispatch_initiator,
                  wavefront_size == 32 ? 0x8005u : 5u);
      }
    }
  }
}

TEST_F(LaunchTest, AllInputCombinationsAndReservedRegisters) {
  // The table is the AMDHSA initial-state order, including distinct high words.
  const uint16_t properties[] = {2, 4, 8, 16, 64};
  const std::vector<uint32_t> payloads[] = {
      {0x11111140, 0x2222},
      {0x33333380, 0x4444},
      {0x55555500, 0x6666},
      {0x77777777, 0x88888888},
      {256},
  };
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .dispatch_pointer = UINT64_C(0x222211111140),
      .queue_pointer = UINT64_C(0x444433333380),
      .kernarg_pointer = UINT64_C(0x666655555500),
      .dispatch_id = UINT64_C(0x8888888877777777),
      .private_segment_size = 256,
  };
  const uint32_t workgroup_size[] = {32, 1, 1};
  const uint32_t partial_size[] = {0, 0, 0};
  for (uint32_t mask = 0; mask < 32; ++mask) {
    SCOPED_TRACE(mask);
    kernel_.code_properties = 0x400;
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < IREE_ARRAYSIZE(properties); ++i) {
      if (mask & (1u << i)) {
        kernel_.code_properties |= properties[i];
        expected.insert(expected.end(), payloads[i].begin(), payloads[i].end());
      }
    }
    const size_t input_count = expected.size();
    for (size_t register_count : {input_count, size_t{15}, size_t{16}}) {
      SCOPED_TRACE(register_count);
      kernel_.compute_pgm_rsrc2 = 0x80u | (register_count << 1);
      auto launch = Prepare();
      const uint32_t requirements = (mask & 1 ? 4 : 0) | (mask & 2 ? 8 : 0);
      EXPECT_EQ(launch.requirements, requirements);
      std::array<uint32_t, 64> words;
      words.fill(kGuard);
      auto* end = iree_hal_amd_gpu_pm4_launch_emit(
          words.data(), &launch, workgroup_size, partial_size, 0, &inputs);
      if (register_count) {
        ASSERT_EQ(end - words.data(), 24 + register_count);
        EXPECT_EQ(words[22], 0xc0007602 | (register_count << 16));
        EXPECT_EQ(words[23], 0x240u);
        auto padded = expected;
        padded.resize(register_count, 0);
        EXPECT_EQ(std::vector<uint32_t>(words.data() + 24, end), padded);
      } else {
        EXPECT_EQ(end - words.data(), 22);
      }
      EXPECT_EQ(*end, kGuard);
    }
  }
}

TEST_F(LaunchTest, ScratchAndStackRequirementsRemainVisible) {
  kernel_.compute_pgm_rsrc2 |= 1u;
  auto launch = Prepare();
  EXPECT_EQ(launch.requirements,
            IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH);
  EXPECT_EQ(launch.resources.resource2 & 1u, 1u);
  kernel_.code_properties |= 0x800;
  launch = Prepare();
  EXPECT_EQ(launch.requirements,
            IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH |
                IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DYNAMIC_STACK);
}

TEST_F(LaunchTest, LdsRebindingReplacesAndCanShrink) {
  kernel_.compute_pgm_rsrc2 |= 0xff8000;
  auto launch = Prepare();
  const uint32_t workgroup_size[] = {33, 1, 1};
  const uint32_t partial_size[] = {0, 0, 0};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {};
  struct Case {
    // Total admitted workgroup allocation in bytes.
    uint32_t length;
    // Hardware allocation in 512-byte units.
    uint32_t units;
  };
  for (const Case& test_case : {Case{0, 0},
                                {1, 1},
                                {512, 1},
                                {513, 2},
                                {65536, 128},
                                {1024, 2},
                                {0, 0}}) {
    std::array<uint32_t, 64> words = {};
    iree_hal_amd_gpu_pm4_launch_emit(words.data(), &launch, workgroup_size,
                                     partial_size, test_case.length, &inputs);
    EXPECT_EQ(words[7], 0x84u | (test_case.units << 15));
    EXPECT_EQ(words[13], 0u);
  }
  EXPECT_EQ(launch.resources.resource2, 0x84u);
}

TEST_F(LaunchTest, WaveDistributionUsesRoundedWorkgroupSize) {
  const uint32_t partial_size[] = {0, 0, 0};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {};
  for (uint32_t wavefront_size : {32u, 64u}) {
    kernel_.code_properties = wavefront_size == 32 ? 0x408 : 8;
    auto launch = Prepare();
    for (uint32_t count :
         {1u, 3 * wavefront_size, 3 * wavefront_size + 1, 4 * wavefront_size,
          4 * wavefront_size + 1, 8 * wavefront_size}) {
      const uint32_t workgroup_size[] = {count, 1, 1};
      std::array<uint32_t, 64> words = {};
      iree_hal_amd_gpu_pm4_launch_emit(words.data(), &launch, workgroup_size,
                                       partial_size, 0, &inputs);
      const bool distribute = count == 3 * wavefront_size + 1 ||
                              count == 4 * wavefront_size ||
                              count == 8 * wavefront_size;
      EXPECT_EQ(words[13], distribute ? 0x400000u : 0u);
    }
  }
}

TEST_F(LaunchTest, RejectsUnrepresentableDescriptorsWithoutPublishingOutput) {
  iree_hal_amd_gpu_pm4_launch_t launch;
  std::memset(&launch, 0xa5, sizeof(launch));
  std::array<uint8_t, sizeof(launch)> untouched;
  std::memcpy(untouched.data(), &launch, sizeof(launch));
  for (uint16_t invalid : {1u, 32u, 128u, 256u, 512u, 4096u, 32768u}) {
    kernel_.code_properties = 0x408 | invalid;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_amd_gpu_pm4_launch_initialize(
                              {11, 0, 0}, &kernel_, kImageAddress, &launch));
    EXPECT_EQ(std::memcmp(&launch, untouched.data(), sizeof(launch)), 0);
  }
  kernel_.code_properties = 0x408;
  kernel_.kernarg_preload = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amd_gpu_pm4_launch_initialize(
                            {11, 0, 0}, &kernel_, kImageAddress, &launch));
  EXPECT_EQ(std::memcmp(&launch, untouched.data(), sizeof(launch)), 0);
  kernel_.kernarg_preload = 0;
  for (uint32_t count : {0u, 1u, 17u, 31u}) {
    kernel_.compute_pgm_rsrc2 = 0x80u | (count << 1);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_amd_gpu_pm4_launch_initialize(
                              {11, 0, 0}, &kernel_, kImageAddress, &launch));
  }
  // GFX12.5's sixth count bit must not be mistaken for GFX11's trap enable.
  kernel_.compute_pgm_rsrc2 = 0xc4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_gpu_pm4_launch_initialize(
                            {12, 5, 0}, &kernel_, kImageAddress, &launch));
  EXPECT_EQ(std::memcmp(&launch, untouched.data(), sizeof(launch)), 0);
}

TEST_F(LaunchTest, RejectsUnsupportedTargetsAndInvalidPlacement) {
  iree_hal_amd_gpu_pm4_launch_t launch;
  for (auto target : {iree_hal_amd_gpu_gfxip_version_t{9, 4, 2},
                      {10, 3, 0},
                      {11, 1, 0},
                      {12, 1, 0},
                      {13, 0, 0}}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          iree_hal_amd_gpu_pm4_launch_initialize(
                              target, &kernel_, kImageAddress, &launch));
  }
  for (uint64_t address : {uint64_t{1}, UINT64_C(0xffffffffff00),
                           UINT64_C(0x1000000000000), UINT64_MAX}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          iree_hal_amd_gpu_pm4_launch_initialize(
                              {11, 0, 0}, &kernel_, address, &launch));
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
      {11, 0, 0}, &kernel_, UINT64_C(0xffffffffff00) - kernel_.entry_offset,
      &launch));
  EXPECT_EQ(launch.program[0], 0xffffffffu);
  EXPECT_EQ(launch.program[1], 0xffu);
}

TEST_F(LaunchTest, PreparesActualHipCompilerArtifacts) {
  const auto* files = iree_hal_amd_gpu_hip_images_create();
  size_t prepared_count = 0;
  for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
    auto* source = testing::ByteSequence::Create(
        iree_make_const_byte_span(files[i].data, files[i].size), 257);
    iree_hal_amd_gpu_image_t image = {};
    iree_status_t status = iree_hal_amd_gpu_hsaco_image_initialize(
        source, iree_allocator_system(), &image);
    iree_byte_sequence_release(source);
    IREE_ASSERT_OK(status);
    if (image.target.version.major == 11 || image.target.version.major == 12) {
      SCOPED_TRACE(files[i].name);
      for (size_t j = 0; j < image.metadata.kernel_count; ++j) {
        iree_hal_amd_gpu_pm4_launch_t launch = {};
        IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
            image.target.version, &image.kernels[j], kImageAddress, &launch));
        EXPECT_EQ(launch.requirements, 0);
        EXPECT_EQ(launch.user_data.kernarg_pointer, 0);
        EXPECT_GE(launch.user_data_count, 2);
        ++prepared_count;
      }
    }
    iree_hal_amd_gpu_image_deinitialize(&image);
  }
  EXPECT_GT(prepared_count, 0);
}

}  // namespace
}  // namespace iree::hal::amd::gpu
