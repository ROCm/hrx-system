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
  EXPECT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
      model, FullWaveMask(model->wave_size), addresses.data(),
      /*common_base_byte_residues=*/1, &result));
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

static const loom_amdgpu_lds_bank_service_model_t* LookupModel(
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t processor_name = IREE_SV("gfx1250"),
    uint8_t wave_size = 32) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  IREE_ASSERT(processor != nullptr);
  return loom_amdgpu_lds_bank_service_model_lookup(
      processor->properties.features.lds_bank_service_model_set_ordinal,
      descriptor_ref, wave_size);
}

TEST(AmdgpuLdsBankServiceTest, RegisteredB128ModelsMatchReferenceProfiles) {
  struct Case {
    // Byte distance between consecutive lanes.
    uint32_t lane_stride_bytes;
    // Total service rounds under the registered model.
    uint16_t required_rounds;
    // Largest number of distinct requests served by one bank in a phase.
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
    const auto* model = LookupModel(descriptor_ref);
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
                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128, 32),
            nullptr);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  ASSERT_NE(processor, nullptr);
  EXPECT_EQ(
      loom_amdgpu_lds_bank_service_model_lookup(
          processor->properties.features.lds_bank_service_model_set_ordinal,
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE, 32),
      nullptr);
}

// The permutations distinguish the read and write service groups. A stride
// sweep alone cannot distinguish octets made from different lane quads.
TEST(AmdgpuLdsBankServiceTest, QualifiedOctetsDistinguishReadsAndWrites) {
  for (const auto processor_name : {IREE_SV("gfx940"), IREE_SV("gfx942"),
                                    IREE_SV("gfx1100"), IREE_SV("gfx1151")}) {
    for (uint8_t wave_size : {32, 64}) {
      const auto* read_model = LookupModel(
          LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128, processor_name, wave_size);
      const auto* write_model = LookupModel(
          LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128, processor_name, wave_size);
      if (wave_size == 32 &&
          (iree_string_view_equal(processor_name, IREE_SV("gfx940")) ||
           iree_string_view_equal(processor_name, IREE_SV("gfx942")))) {
        EXPECT_EQ(read_model, nullptr);
        EXPECT_EQ(write_model, nullptr);
        continue;
      }
      ASSERT_NE(read_model, nullptr);
      ASSERT_NE(write_model, nullptr);
      const uint16_t phases = wave_size / 8;
      ExpectProfile(EvaluateLinear(read_model, 16), phases, phases, 1);
      ExpectProfile(EvaluateLinear(write_model, 16), phases, phases, 1);
      ExpectProfile(EvaluateLinear(read_model, 96), phases * 2, phases, 2);
      ExpectProfile(EvaluateLinear(write_model, 96), phases * 2, phases, 2);
      ExpectProfile(EvaluateLinear(read_model, 0), phases, phases, 1);
      for (unsigned permutation = 0; permutation < 2; ++permutation) {
        LaneAddresses addresses = {};
        for (uint8_t lane = 0; lane < wave_size; ++lane) {
          const unsigned half = (lane % 32) / 16;
          const unsigned bank_block = permutation == 0 ? (lane % 8) ^ (half * 4)
                                                       : (lane % 4) + half * 4;
          addresses[lane] = lane * 128 + bank_block * 16;
        }
        loom_amdgpu_lds_bank_service_result_t read_result = {};
        loom_amdgpu_lds_bank_service_result_t write_result = {};
        ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
            read_model, FullWaveMask(wave_size), addresses.data(),
            /*common_base_byte_residues=*/1, &read_result));
        ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
            write_model, FullWaveMask(wave_size), addresses.data(),
            /*common_base_byte_residues=*/1, &write_result));
        ExpectProfile(read_result, phases * (permutation == 0 ? 2 : 1), phases,
                      permutation == 0 ? 2 : 1);
        ExpectProfile(write_result, phases * (permutation == 0 ? 1 : 2), phases,
                      permutation == 0 ? 1 : 2);
      }
    }
  }
}

TEST(AmdgpuLdsBankServiceTest, ReadRequestPolicyChangesBroadcastProfile) {
  const auto* count_each_model =
      LookupModel(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128);
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

// Native controls separate 16-lane and 32-lane phases, same-word combining,
// and the two independent halves of wave64 without relying on cycle counts.
TEST(AmdgpuLdsBankServiceTest, NarrowPacketsUseQualifiedThirtyTwoLanePhases) {
  for (auto processor :
       {IREE_SV("gfx1100"), IREE_SV("gfx1151"), IREE_SV("gfx942")}) {
    for (uint8_t wave_size : {32, 64}) {
      for (auto descriptor : {LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32}) {
        const auto* model = LookupModel(descriptor, processor, wave_size);
        if (iree_string_view_equal(processor, IREE_SV("gfx942")) &&
            (wave_size == 32 ||
             descriptor == LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16 ||
             descriptor == LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI)) {
          EXPECT_EQ(model, nullptr);
          continue;
        }
        ASSERT_NE(model, nullptr);
        const uint16_t phases = wave_size / 32;
        ExpectProfile(EvaluateLinear(model, model->packet_byte_count), phases,
                      phases, 1);
        ExpectProfile(EvaluateLinear(model, 128), phases * 32, phases, 32);
        for (uint8_t partition : {0, 1, 2}) {
          LaneAddresses addresses = {};
          for (uint8_t lane = 0; lane < wave_size; ++lane) {
            const uint8_t bank_offset = partition == 1   ? (lane % 32) / 16 * 4
                                        : partition == 2 ? lane / 32 * 4
                                                         : 0;
            addresses[lane] = lane * 128 + ((lane % 4) + bank_offset) * 4;
          }
          loom_amdgpu_lds_bank_service_result_t result;
          ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
              model, FullWaveMask(wave_size), addresses.data(), 1, &result));
          const uint16_t multiplicity = partition == 1 ? 4 : 8;
          ExpectProfile(result, phases * multiplicity, phases, multiplicity);
        }
      }
    }
  }
}

TEST(AmdgpuLdsBankServiceTest, DoublewordPacketsUseQualifiedSixteenLanePhases) {
  struct Case {
    // Distinct adjacent bank pairs used by each lane partition.
    uint8_t bank_pair_count;
    // Lanes selecting one bank subset before switching to the other subset.
    uint8_t partition_lane_count;
    // Largest number of requests served by one bank within a service phase.
    uint16_t maximum_multiplicity;
  };
  // These native controls distinguish crossed octets, whole 32-lane phases,
  // and independent halves of wave64. Linear strides alone are insufficient.
  static constexpr Case kCases[] = {{8, 16, 2}, {4, 16, 4}, {4, 32, 4}};
  for (auto processor_name :
       {IREE_SV("gfx1100"), IREE_SV("gfx1151"), IREE_SV("gfx942")}) {
    SCOPED_TRACE(processor_name.data);
    for (uint8_t wave_size : {32, 64}) {
      SCOPED_TRACE(wave_size);
      for (auto descriptor : {LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64,
                              LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B64}) {
        SCOPED_TRACE(descriptor);
        const auto* model = LookupModel(descriptor, processor_name, wave_size);
        if (wave_size == 32 &&
            iree_string_view_equal(processor_name, IREE_SV("gfx942"))) {
          EXPECT_EQ(model, nullptr);
          continue;
        }
        ASSERT_NE(model, nullptr);
        const uint16_t phases = wave_size / 16;
        ExpectProfile(EvaluateLinear(model, 8), phases, phases, 1);
        ExpectProfile(EvaluateLinear(model, 128), phases * 16, phases, 16);
        if (descriptor == LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64) {
          ExpectProfile(EvaluateLinear(model, 0), phases, phases, 1);
        }
        for (const Case& test_case : kCases) {
          LaneAddresses addresses = {};
          for (uint8_t lane = 0; lane < wave_size; ++lane) {
            const uint8_t bank_pair =
                lane % test_case.bank_pair_count +
                (lane / test_case.partition_lane_count % 2) *
                    test_case.bank_pair_count;
            addresses[lane] = lane * 128 + bank_pair * 8;
          }
          loom_amdgpu_lds_bank_service_result_t result;
          ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
              model, FullWaveMask(wave_size), addresses.data(),
              /*common_base_byte_residues=*/1, &result));
          ExpectProfile(result, phases * test_case.maximum_multiplicity, phases,
                        test_case.maximum_multiplicity);
        }
      }
    }
  }
}

TEST(AmdgpuLdsBankServiceTest, HalfwordCombiningPreservesByteIdentity) {
  for (auto descriptor : {LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16}) {
    const auto* model = LookupModel(descriptor, IREE_SV("gfx1151"));
    ASSERT_NE(model, nullptr);
    // BF16 GEMM stores pair distinct halves of each bank word.
    LaneAddresses addresses = {};
    for (uint8_t lane = 0; lane < 32; ++lane) {
      addresses[lane] = 2 * (lane % 16) + 192 * (lane / 16);
    }
    loom_amdgpu_lds_bank_service_result_t result;
    ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
        model, FullWaveMask(32), addresses.data(),
        /*common_base_byte_residues=*/0b0101, &result));
    ExpectProfile(result, 1, 1, 1);
    EXPECT_EQ(result.base_residue_count, 64);
    // Distinct words that share a bank still require separate rounds.
    for (uint8_t lane = 0; lane < 32; ++lane) {
      addresses[lane] = (lane / 2) * 128 + (lane % 2) * 2;
    }
    ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
        model, FullWaveMask(32), addresses.data(), 1, &result));
    ExpectProfile(result, 16, 1, 16);
    // Read broadcast does not authorize combining overlapping writes.
    const bool is_read =
        model->direction == LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ;
    ExpectProfile(EvaluateLinear(model, 0), is_read ? 1 : 32, 1,
                  is_read ? 1 : 32);
  }
}

TEST(AmdgpuLdsBankServiceTest, UnresolvedSubwordBaseCanPreventExactProof) {
  for (auto descriptor : {LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI,
                          LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16}) {
    const auto* model = LookupModel(descriptor, IREE_SV("gfx1151"));
    ASSERT_NE(model, nullptr);
    LaneAddresses addresses = MakeLinearLaneAddresses(32, 4);
    addresses[1] = 130;
    loom_amdgpu_lds_bank_service_result_t result;
    ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
        model, FullWaveMask(32), addresses.data(), 1, &result));
    ExpectProfile(result, 2, 1, 2);
    ASSERT_TRUE(loom_amdgpu_lds_bank_service_evaluate(
        model, FullWaveMask(32), addresses.data(),
        /*common_base_byte_residues=*/0b0100, &result));
    ExpectProfile(result, 1, 1, 1);
    EXPECT_FALSE(loom_amdgpu_lds_bank_service_evaluate(
        model, FullWaveMask(32), addresses.data(),
        /*common_base_byte_residues=*/0b0101, &result));
    EXPECT_EQ(result.phase_count, 0);
    EXPECT_EQ(result.required_rounds, 0);
  }
}

}  // namespace
}  // namespace loom
