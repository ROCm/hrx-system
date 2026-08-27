// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/target_info.h"

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

static const loom_amdgpu_lds_bank_service_model_t* LookupB128Model(
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  IREE_ASSERT(processor != nullptr);
  return loom_amdgpu_lds_bank_service_model_lookup(
      processor->properties.features.lds_bank_service_model_set_ordinal,
      descriptor_ref);
}

TEST(AmdgpuLdsBankServiceTest, RegisteredB128ModelsMatchReferenceProfiles) {
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
  for (const auto descriptor_ref : {LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128}) {
    const auto* model = LookupB128Model(descriptor_ref);
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

TEST(AmdgpuLdsBankServiceTest, LookupRequiresModelAndDescriptorBinding) {
  EXPECT_EQ(loom_amdgpu_lds_bank_service_model_lookup(
                LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE,
                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128),
            nullptr);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  ASSERT_NE(processor, nullptr);
  EXPECT_EQ(
      loom_amdgpu_lds_bank_service_model_lookup(
          processor->properties.features.lds_bank_service_model_set_ordinal,
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE),
      nullptr);
}

static constexpr uint64_t LaneRangeMask(uint8_t begin, uint8_t count) {
  return ((UINT64_C(1) << count) - UINT64_C(1)) << begin;
}

TEST(AmdgpuLdsBankServiceTest, ModelPhasesChangeStride96Result) {
  const auto* registered_model =
      LookupB128Model(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128);
  ASSERT_NE(registered_model, nullptr);
  const auto registered_result = EvaluateLinear(registered_model, 96);
  ExpectProfile(registered_result, /*required_rounds=*/8,
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
      LookupB128Model(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128);
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
