// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

#include <array>
#include <vector>

#include "iree/hal/device_spec.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

struct ProfileCase {
  // Physical target's independent ISA identity.
  iree_hal_amd_gpu_gfxip_version_t gfxip;
  // Whole shader-cache action in ACQUIRE_MEM layout.
  uint32_t shader;
  // System release in ACQUIRE_MEM layout.
  uint32_t release;
  // System acquire in ACQUIRE_MEM layout.
  uint32_t acquire;
  // Whole system boundary in RELEASE_MEM layout.
  uint32_t tail;
};

class BarrierTest : public ::testing::TestWithParam<ProfileCase> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_barrier_profile_initialize(
        GetParam().gfxip, &profile_));
  }

  iree_hal_amd_gpu_pm4_barrier_t Prepare(const iree_hal_barrier_t& value) {
    const iree_hal_barrier_list_t list = {1, &value};
    const iree_hal_queue_family_spec_t spec = {};
    const iree_hal_queue_family_t family = {.spec = &spec};
    IREE_CHECK_OK(iree_hal_barrier_validate(&value, &family));
    return iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, &list);
  }

  iree_hal_amd_gpu_pm4_barrier_t Access(iree_hal_access_scope_t source,
                                        iree_hal_access_scope_t target) {
    const iree_hal_memory_barrier_t memory = {source, target};
    return Prepare({.memory_barrier_count = 1, .memory_barriers = &memory});
  }

  // Immutable target policy, constructed at the same boundary as a family.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile_;
};

TEST_P(BarrierTest, TargetLayoutAndCompactCapture) {
  EXPECT_EQ(sizeof(iree_hal_amd_gpu_pm4_barrier_profile_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amd_gpu_pm4_barrier_t), 8u);
  EXPECT_EQ(profile_.shader, GetParam().shader);
  EXPECT_EQ(profile_.release, GetParam().release);
  EXPECT_EQ(profile_.acquire, GetParam().acquire);
  EXPECT_EQ(profile_.command_processor_bypasses_l2,
            GetParam().gfxip.major == 12);
}

TEST_P(BarrierTest, DefaultsAndExplicitEmptyAreDistinct) {
  const auto defaults =
      iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, nullptr);
  EXPECT_EQ(defaults.cache_control, GetParam().acquire);
  EXPECT_EQ(defaults.execution, 3u);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_barrier_word_count(defaults), 17u);
  const iree_hal_barrier_list_t empty = {};
  const auto omitted = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, &empty);
  EXPECT_EQ(omitted.cache_control, 0u);
  EXPECT_EQ(omitted.execution, 0u);
  std::array<uint32_t, 1> storage = {0x12345678};
  EXPECT_EQ(iree_hal_amd_gpu_pm4_barrier_emit(storage.data(), omitted),
            storage.data());
  EXPECT_EQ(storage[0], 0x12345678u);
}

TEST_P(BarrierTest, DirectionComesFromActionNotBoundaryPosition) {
  const auto release = Prepare({
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  });
  EXPECT_EQ(release.cache_control, GetParam().release);
  EXPECT_EQ(release.execution, 1u);
  const auto acquire = Prepare({
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  });
  EXPECT_EQ(acquire.cache_control, GetParam().acquire);
  EXPECT_EQ(acquire.execution, 0u);
  const auto visibility_only = Prepare({
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  });
  EXPECT_EQ(visibility_only.cache_control, GetParam().release);
  EXPECT_EQ(visibility_only.execution, 3u);
}

TEST_P(BarrierTest, EveryExecutionStagePreservesItsCompletionDomain) {
  const uint32_t expected[] = {0, 3, 1, 3, 3, 0, 0};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    SCOPED_TRACE(i);
    const auto barrier = Prepare({
        .source_stage_mask = 1u << i,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
    });
    EXPECT_EQ(barrier.cache_control, 0u);
    EXPECT_EQ(barrier.execution, expected[i]);
  }
}

TEST_P(BarrierTest, ExecutionOnlyUsesNoCacheCommands) {
  const auto barrier = Prepare({
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
  });
  std::array<uint32_t, 3> words = {0, 0, 0x12345678};
  EXPECT_EQ(iree_hal_amd_gpu_pm4_barrier_emit(words.data(), barrier),
            words.data() + 2);
  EXPECT_EQ(words, (std::array<uint32_t, 3>{0xc0004600, 0x407, 0x12345678}));
}

TEST_P(BarrierTest, LocalShaderDependenciesStayInsideShaderCaches) {
  for (auto source : {IREE_HAL_ACCESS_SCOPE_CONSTANT_READ,
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE}) {
    for (auto target : {IREE_HAL_ACCESS_SCOPE_CONSTANT_READ,
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE}) {
      const auto barrier = Access(source, target);
      EXPECT_EQ(barrier.cache_control, GetParam().shader);
      EXPECT_EQ(barrier.execution, 1u);
    }
  }
}

TEST_P(BarrierTest, CommandProcessorCrossingsIncludeReadReadDirtyHistory) {
  for (auto cp : {IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ,
                  IREE_HAL_ACCESS_SCOPE_ATOMIC_READ,
                  IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE}) {
    for (auto shader : {IREE_HAL_ACCESS_SCOPE_CONSTANT_READ,
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE}) {
      const auto to_shader = Access(cp, shader);
      const auto to_cp = Access(shader, cp);
      if (GetParam().gfxip.major == 12) {
        EXPECT_EQ(to_shader.cache_control, GetParam().acquire);
        EXPECT_EQ(to_cp.cache_control, GetParam().shader | GetParam().release);
      } else {
        EXPECT_EQ(to_shader.cache_control, GetParam().shader);
        EXPECT_EQ(to_cp.cache_control, GetParam().shader);
      }
      EXPECT_EQ(to_shader.execution, 0u);
      EXPECT_EQ(to_cp.execution, 1u);
    }
  }
}

TEST_P(BarrierTest, TransferCoversShaderAndCommandProcessorImplementations) {
  for (auto source : {IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE}) {
    const auto barrier = Access(source, IREE_HAL_ACCESS_SCOPE_TRANSFER_READ);
    EXPECT_EQ(barrier.cache_control, GetParam().gfxip.major == 12
                                         ? GetParam().acquire
                                         : GetParam().shader);
    EXPECT_EQ(barrier.execution, 3u);
  }
}

TEST_P(BarrierTest, ExternalScopesUseSystemMaintenance) {
  for (auto external :
       {IREE_HAL_ACCESS_SCOPE_HOST_READ, IREE_HAL_ACCESS_SCOPE_HOST_WRITE,
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
        IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE}) {
    EXPECT_EQ(
        Access(external, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ).cache_control,
        GetParam().acquire);
    EXPECT_EQ(
        Access(IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE, external).cache_control,
        GetParam().shader | GetParam().release);
  }
}

TEST_P(BarrierTest, QualifiedRangesPromoteWithoutRetainingStorage) {
  for (bool acquire : {false, true}) {
    const uint32_t range_effect =
        acquire ? IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM
                : IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM;
    const uint32_t global_effect =
        acquire ? IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM
                : IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
    const iree_hal_memory_transition_recipe_info_t operation = {
        .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
        .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
        .operation =
            acquire ? IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM
                    : IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
        .range_granularity = 64,
    };
    const iree_hal_memory_transition_recipe_t recipe = {
        .effects = {range_effect},
        .operation_count = 1,
        .operations = &operation,
    };
    const iree_hal_buffer_barrier_t buffer = {.recipe = &recipe};
    const auto ranged = Prepare({
        .effects = {range_effect},
        .buffer_barrier_count = 1,
        .buffer_barriers = &buffer,
    });
    const auto global = Prepare({.effects = {global_effect}});
    EXPECT_EQ(ranged.cache_control, global.cache_control);
    EXPECT_EQ(ranged.execution, global.execution);
  }
}

TEST_P(BarrierTest, CombinesIndependentRecordsAtOneBoundary) {
  const iree_hal_barrier_t values[] = {
      {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
          .effects = {IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM},
      },
      {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .effects = {IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM},
      },
  };
  const iree_hal_barrier_list_t list = {IREE_ARRAYSIZE(values), values};
  const auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, &list);
  EXPECT_EQ(barrier.cache_control, GetParam().acquire);
  EXPECT_EQ(barrier.execution, 3u);
}

TEST_P(BarrierTest, OrdinaryEmissionJoinsBeforeMaintenance) {
  const auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, nullptr);
  std::array<uint32_t, 18> words = {};
  words.back() = 0x12345678;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_barrier_emit(words.data(), barrier),
            words.data() + 17);
  EXPECT_EQ((std::vector<uint32_t>(words.begin(), words.end())),
            (std::vector<uint32_t>{0xc0004600, 0x407, 0xc0055000, 0, 0, 0, 0, 0,
                                   0, 0xc0065800, 0, 0xffffffff, 0xff, 0, 0,
                                   0xa, GetParam().acquire, 0x12345678}));
}

TEST_P(BarrierTest, TailFoldsComputeAndCachesIntoExistingRelease) {
  const auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, nullptr);
  std::array<uint32_t, 16> words = {};
  words.back() = 0x12345678;
  uint32_t cache_control = 0;
  auto* end = iree_hal_amd_gpu_pm4_barrier_emit_tail(words.data(), barrier,
                                                     &cache_control);
  ASSERT_EQ(end, words.data() + 7);
  EXPECT_EQ(cache_control, GetParam().tail);
  end = iree_hal_amd_gpu_pm4_emit_release32(end, cache_control, 0x100000040, 9);
  EXPECT_EQ(end, words.data() + 15);
  EXPECT_EQ((std::vector<uint32_t>(words.begin(), words.end())),
            (std::vector<uint32_t>{0xc0055000, 0, 0, 0, 0, 0, 0, 0xc0064900,
                                   0x528u | (GetParam().tail << 12), 0x23010000,
                                   0x40, 1, 9, 0, 0, 0x12345678}));
}

TEST_P(BarrierTest, EmptyTailPreservesMandatoryCompletion) {
  const iree_hal_barrier_list_t empty = {};
  const auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, &empty);
  std::array<uint32_t, 8> words = {};
  uint32_t cache_control = UINT32_MAX;
  auto* end = iree_hal_amd_gpu_pm4_barrier_emit_tail(words.data(), barrier,
                                                     &cache_control);
  EXPECT_EQ(end, words.data());
  EXPECT_EQ(cache_control, 0u);
  end = iree_hal_amd_gpu_pm4_emit_release64(end, cache_control, 0x100000040, 9);
  EXPECT_EQ(end, words.data() + words.size());
  EXPECT_EQ(words[1], 0x528u);
}

TEST_P(BarrierTest, InstructionPublicationSurvivesEmptyPayload) {
  const iree_hal_barrier_list_t empty = {};
  auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(&profile_, &empty);
  barrier.cache_control |= 1;
  std::array<uint32_t, 8> words = {};
  EXPECT_EQ(iree_hal_amd_gpu_pm4_barrier_emit(words.data(), barrier),
            words.data() + words.size());
  EXPECT_EQ(words, (std::array<uint32_t, 8>{0xc0065800, 0, 0xffffffff, 0xff, 0,
                                            0, 0xa, 1}));
}

INSTANTIATE_TEST_SUITE_P(
    PhysicalTargets, BarrierTest,
    ::testing::Values(
        ProfileCase{{11, 0, 0}, 0x103e0, 0x18040, 0x1c3e0, 0x4170e},
        ProfileCase{{11, 5, 1}, 0x103e0, 0x18040, 0x1c3e0, 0x4170e},
        ProfileCase{{11, 7, 2}, 0x103e0, 0x18040, 0x1c3e0, 0x4170e},
        ProfileCase{{12, 0, 1}, 0x101c0, 0x18040, 0x1c1c0, 0x41704},
        ProfileCase{{12, 5, 0}, 0x101c0, 0x18060, 0x1c1e0, 0x41706}));

TEST(BarrierProfileTest, UnqualifiedIsaFailsDuringFamilyConstruction) {
  for (auto gfxip : {iree_hal_amd_gpu_gfxip_version_t{9, 4, 2},
                     {10, 3, 0},
                     {11, 1, 0},
                     {12, 1, 0},
                     {13, 0, 0}}) {
    iree_hal_amd_gpu_pm4_barrier_profile_t profile = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        iree_hal_amd_gpu_pm4_barrier_profile_initialize(gfxip, &profile));
    EXPECT_EQ(profile.shader, 0u);
    EXPECT_EQ(profile.release, 0u);
    EXPECT_EQ(profile.acquire, 0u);
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu
