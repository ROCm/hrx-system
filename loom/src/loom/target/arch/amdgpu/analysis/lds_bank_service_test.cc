// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

using LaneAddresses =
    std::array<uint64_t, LOOM_AMDGPU_LDS_BANK_SERVICE_MAX_WAVE_SIZE>;

static uint64_t FullWaveMask(uint8_t wave_size) {
  return wave_size == 64 ? UINT64_MAX
                         : (UINT64_C(1) << wave_size) - UINT64_C(1);
}

static LaneAddresses MakeLinearLaneAddresses(uint8_t wave_size,
                                             uint32_t lane_stride_bytes,
                                             uint32_t translation_bytes = 0) {
  LaneAddresses addresses = {};
  for (uint8_t lane = 0; lane < wave_size; ++lane) {
    addresses[lane] = translation_bytes + (uint64_t)lane * lane_stride_bytes;
  }
  return addresses;
}

static loom_amdgpu_lds_bank_service_result_t EvaluateLinear(
    const loom_amdgpu_lds_bank_service_model_t* model,
    uint32_t lane_stride_bytes, uint32_t translation_bytes = 0) {
  const LaneAddresses addresses = MakeLinearLaneAddresses(
      model->wave_size, lane_stride_bytes, translation_bytes);
  loom_amdgpu_lds_bank_service_result_t result = {};
  loom_amdgpu_lds_bank_service_evaluate(model, FullWaveMask(model->wave_size),
                                        addresses.data(), &result);
  return result;
}

static void ExpectProfile(const loom_amdgpu_lds_bank_service_result_t& result,
                          uint16_t required_rounds, uint16_t uncontended_rounds,
                          uint16_t maximum_multiplicity) {
  EXPECT_EQ(result.required_rounds, required_rounds);
  EXPECT_EQ(result.uncontended_rounds, uncontended_rounds);
  EXPECT_EQ(result.extra_rounds, required_rounds - uncontended_rounds);
  EXPECT_EQ(result.maximum_request_multiplicity, maximum_multiplicity);
}

TEST(AmdgpuLdsBankServiceTest, Gfx1250B128MatchesReferenceProfiles) {
  struct Case {
    uint32_t lane_stride_bytes;
    uint16_t required_rounds;
    uint16_t maximum_multiplicity;
  };
  static constexpr Case kCases[] = {
      {16, 8, 1},
      {64, 16, 2},
      {80, 8, 1},
      {96, 8, 1},
  };
  for (const auto direction : {LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
                               LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE}) {
    const auto* model =
        loom_amdgpu_lds_bank_service_gfx1250_b128_model(direction);
    ASSERT_NE(model, nullptr);
    for (const Case& test_case : kCases) {
      const auto result = EvaluateLinear(model, test_case.lane_stride_bytes);
      ExpectProfile(result, test_case.required_rounds,
                    /*uncontended_rounds=*/8, test_case.maximum_multiplicity);
      EXPECT_EQ(result.base_residue_count, 32);
      for (uint32_t base_residue = 1; base_residue < 32; ++base_residue) {
        const auto translated =
            EvaluateLinear(model, test_case.lane_stride_bytes,
                           /*translation_bytes=*/base_residue * 4);
        EXPECT_EQ(translated.required_rounds, result.required_rounds);
        EXPECT_EQ(translated.uncontended_rounds, result.uncontended_rounds);
        EXPECT_EQ(translated.maximum_request_multiplicity,
                  result.maximum_request_multiplicity);
      }
    }
  }
}

static constexpr uint64_t LaneRangeMask(uint8_t begin, uint8_t count) {
  return ((UINT64_C(1) << count) - UINT64_C(1)) << begin;
}

TEST(AmdgpuLdsBankServiceTest, ModelPhasesChangeStride96Result) {
  const auto* gfx1250_model = loom_amdgpu_lds_bank_service_gfx1250_b128_model(
      LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ);
  ASSERT_NE(gfx1250_model, nullptr);
  const auto gfx1250_result = EvaluateLinear(gfx1250_model, 96);
  ExpectProfile(gfx1250_result, /*required_rounds=*/8,
                /*uncontended_rounds=*/8, /*maximum_multiplicity=*/1);

  const loom_amdgpu_lds_bank_service_model_t ck_wave64_model = {
      /*.key=*/IREE_SV("ck-wave64.ds_read_b128"),
      /*.revision=*/
      IREE_SV("ROCm/rocm-libraries@e370a4f28f42:lds_bank_conflicts.rst"),
      /*.evidence_class=*/
      LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
      /*.direction=*/LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
      /*.request_policy=*/
      LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
      /*.wave_size=*/64,
      /*.bank_count=*/32,
      /*.bank_word_byte_count=*/4,
      /*.packet_word_count=*/4,
      /*.phase_count=*/8,
      /*.phase_lane_masks=*/
      {
          LaneRangeMask(0, 4) | LaneRangeMask(20, 4),
          LaneRangeMask(4, 4) | LaneRangeMask(16, 4),
          LaneRangeMask(8, 4) | LaneRangeMask(28, 4),
          LaneRangeMask(12, 4) | LaneRangeMask(24, 4),
          LaneRangeMask(32, 4) | LaneRangeMask(52, 4),
          LaneRangeMask(36, 4) | LaneRangeMask(48, 4),
          LaneRangeMask(40, 4) | LaneRangeMask(60, 4),
          LaneRangeMask(44, 4) | LaneRangeMask(56, 4),
      },
  };
  const auto ck_result = EvaluateLinear(&ck_wave64_model, 96);
  ExpectProfile(ck_result, /*required_rounds=*/16,
                /*uncontended_rounds=*/8, /*maximum_multiplicity=*/2);
}

TEST(AmdgpuLdsBankServiceTest, ReadRequestPolicyChangesBroadcastProfile) {
  const auto* count_each_model =
      loom_amdgpu_lds_bank_service_gfx1250_b128_model(
          LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ);
  ASSERT_NE(count_each_model, nullptr);
  const auto count_each_result = EvaluateLinear(count_each_model, 0);
  ExpectProfile(count_each_result, /*required_rounds=*/32,
                /*uncontended_rounds=*/8, /*maximum_multiplicity=*/4);

  auto coalescing_model = *count_each_model;
  coalescing_model.request_policy =
      LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS;
  const auto coalesced_result = EvaluateLinear(&coalescing_model, 0);
  ExpectProfile(coalesced_result, /*required_rounds=*/8,
                /*uncontended_rounds=*/8, /*maximum_multiplicity=*/1);
}

}  // namespace
}  // namespace loom
