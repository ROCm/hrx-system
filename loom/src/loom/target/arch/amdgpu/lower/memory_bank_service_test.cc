// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

static const loom_amdgpu_lds_bank_service_model_t* WriteModel(
    uint8_t wave_size = 32) {
  const auto* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  return loom_amdgpu_lds_bank_service_model_lookup(
      processor->properties.features.lds_bank_service_model_set_ordinal,
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128, wave_size);
}

static loom_low_source_memory_access_plan_t CoordinateSource(
    int64_t x_stride, int64_t y_stride, int64_t z_stride = 0) {
  loom_low_source_memory_access_plan_t source = {};
  source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  source.root_uniform_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
  source.root_minimum_alignment = 16;
  source.minimum_alignment = 16;
  const int64_t strides[] = {x_stride, y_stride, z_stride};
  for (uint8_t axis = 0; axis < 3; ++axis) {
    if (strides[axis] == 0) {
      continue;
    }
    auto& term = source.dynamic_terms[source.dynamic_term_count++];
    term.source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
    term.dimension = static_cast<loom_kernel_dimension_t>(axis);
    term.byte_stride = strides[axis];
    term.byte_facts = loom_value_facts_make(0, 65520, strides[axis]);
    loom_value_facts_mark_lane_varying(&term.byte_facts);
  }
  return source;
}

static loom_low_lower_memory_bank_service_report_t Calculate(
    const loom_low_source_memory_access_plan_t& source,
    loom_target_workgroup_size_t workgroup_size, uint8_t wave_size = 32) {
  loom_low_lower_memory_bank_service_report_t report = {};
  loom_amdgpu_memory_calculate_source_bank_service(
      WriteModel(wave_size), &source, &workgroup_size, &report);
  return report;
}

static void ExpectExact(
    const loom_low_lower_memory_bank_service_report_t& report,
    uint16_t required_rounds, uint16_t uncontended_rounds) {
  EXPECT_TRUE(iree_string_view_equal(report.proof, IREE_SV("exact")));
  EXPECT_TRUE(iree_string_view_is_empty(report.unknown_reason));
  EXPECT_EQ(report.required_rounds, required_rounds);
  EXPECT_EQ(report.uncontended_rounds, uncontended_rounds);
  EXPECT_EQ(report.extra_rounds, required_rounds - uncontended_rounds);
}

static void ExpectUnknown(
    const loom_low_lower_memory_bank_service_report_t& report,
    iree_string_view_t reason) {
  EXPECT_TRUE(iree_string_view_equal(report.proof, IREE_SV("unknown")));
  EXPECT_TRUE(iree_string_view_equal(report.unknown_reason, reason));
  EXPECT_EQ(report.required_rounds, 0);
}

TEST(AmdgpuMemoryBankServiceTest, CoordinatesMustFollowNativeWaveMembership) {
  const auto source = CoordinateSource(16, 512);
  // With 32 columns each wave has one row; four columns put two rows in each
  // eight-lane service phase, targeting distinct words in the same banks.
  ExpectExact(Calculate(source, {32, 2, 1}), 4, 4);
  ExpectExact(Calculate(source, {4, 8, 1}), 8, 4);
  ExpectExact(Calculate(source, {64, 2, 1}, 64), 8, 8);
  ExpectExact(Calculate(source, {4, 16, 1}, 64), 16, 8);

  const auto volume = CoordinateSource(16, 64, 512);
  ExpectExact(Calculate(volume, {4, 2, 4}), 4, 4);
  const auto separated_rows = CoordinateSource(16, 512, 1024);
  ExpectExact(Calculate(separated_rows, {2, 2, 8}), 16, 4);
}

TEST(AmdgpuMemoryBankServiceTest, ExactProfileMustCoverEveryWave) {
  // These three waves require five, six, and five rounds respectively.
  // A wave-zero proof would understate the second wave's service.
  ExpectUnknown(Calculate(CoordinateSource(16, 512), {12, 8, 1}),
                IREE_SV("address-wave-profiles-differ"));
}

TEST(AmdgpuMemoryBankServiceTest, UniformContributionsPreserveBankTranslation) {
  auto source = CoordinateSource(16, 512);
  auto& stage = source.dynamic_terms[source.dynamic_term_count++];
  stage.source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  stage.byte_stride = 32768;
  stage.byte_facts = loom_value_facts_make(0, 32768, 32768);
  loom_value_facts_mark_subgroup_uniform(&stage.byte_facts);
  // The byte contribution is already scaled by its runtime factors. A uniform
  // runtime term is legal even though a varying coordinate with such a stride
  // cannot be evaluated at compile time.
  stage.stride_value_count = 1;
  ExpectExact(Calculate(source, {32, 2, 1}), 4, 4);

  loom_value_facts_mark_lane_varying(&stage.byte_facts);
  ExpectUnknown(Calculate(source, {32, 2, 1}),
                IREE_SV("address-varying-term-unproven"));
}

TEST(AmdgpuMemoryBankServiceTest, UniformReadStillUsesRequestPolicy) {
  const auto source = CoordinateSource(0, 0);
  const auto* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  const auto* model = loom_amdgpu_lds_bank_service_model_lookup(
      processor->properties.features.lds_bank_service_model_set_ordinal,
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128, 32);
  const loom_target_workgroup_size_t workgroup_size = {32, 1, 1};
  loom_low_lower_memory_bank_service_report_t report = {};
  loom_amdgpu_memory_calculate_source_bank_service(model, &source,
                                                   &workgroup_size, &report);
  ExpectExact(report, 4, 4);
}

TEST(AmdgpuMemoryBankServiceTest, OpaqueOrUnalignedAddressesStayUnknown) {
  auto source = CoordinateSource(16, 512);
  source.dynamic_terms[1].stride_value_count = 1;
  ExpectUnknown(Calculate(source, {32, 2, 1}),
                IREE_SV("address-dynamic-stride"));
  source.dynamic_terms[1].stride_value_count = 0;
  source.dynamic_terms[1].source =
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  ExpectUnknown(Calculate(source, {32, 2, 1}),
                IREE_SV("address-varying-term-unproven"));

  source = CoordinateSource(8, 512);
  ExpectUnknown(Calculate(source, {32, 2, 1}),
                IREE_SV("address-packet-alignment-unproven"));
  source = CoordinateSource(16, 512);
  source.root_uniform_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
  ExpectUnknown(Calculate(source, {32, 2, 1}),
                IREE_SV("address-root-not-subgroup-uniform"));
}

TEST(AmdgpuMemoryBankServiceTest, SubwordUniformOffsetRetainsItsResidues) {
  auto source = CoordinateSource(2, 0);
  source.minimum_alignment = 2;
  auto& stage = source.dynamic_terms[source.dynamic_term_count++];
  stage.source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  stage.byte_stride = 2;
  stage.byte_facts = loom_value_facts_make(0, 2, 2);
  loom_value_facts_mark_subgroup_uniform(&stage.byte_facts);
  const auto* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  const auto* model = loom_amdgpu_lds_bank_service_model_lookup(
      processor->properties.features.lds_bank_service_model_set_ordinal,
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16, 32);
  const loom_target_workgroup_size_t workgroup_size = {64, 1, 1};
  loom_low_lower_memory_bank_service_report_t report = {};
  loom_amdgpu_memory_calculate_source_bank_service(model, &source,
                                                   &workgroup_size, &report);
  ExpectExact(report, 1, 1);
  EXPECT_EQ(report.base_residue_count, 64);

  stage.byte_facts = loom_value_facts_exact_i64(2);
  loom_amdgpu_memory_calculate_source_bank_service(model, &source,
                                                   &workgroup_size, &report);
  ExpectExact(report, 1, 1);
  EXPECT_EQ(report.base_residue_count, 32);
}

}  // namespace
}  // namespace loom
