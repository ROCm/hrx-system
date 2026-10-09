// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/placement.h"

#include <algorithm>

#include "iree/testing/gtest.h"
#include "loom/codegen/low/allocation/preference.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

loom_low_placement_instruction_preferences_t Preferences(
    const char* target_name, uint32_t subgroup_size,
    const loom_low_descriptor_set_t* descriptors) {
  loom_amdgpu_target_facts_t facts = {};
  facts.base.fact_type = &loom_amdgpu_target_fact_type;
  loom_amdgpu_target_identity_initialize(
      loom_amdgpu_target_info_find_target(iree_make_cstring_view(target_name)),
      &facts.identity);
  facts.base.storage.snapshot.subgroup_size = subgroup_size;
  loom_target_fact_field_set_insert(&facts.base.explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  loom_target_bundle_storage_rebind(&facts.base.storage);
  loom_amdgpu_target_facts_initialize(&facts);
  loom_low_resolved_target_t target = {.target_facts = &facts.base,
                                       .descriptor_set = descriptors};
  return loom_amdgpu_placement_instruction_preferences(&target);
}

class AmdgpuPlacementBindingTest
    : public ::testing::TestWithParam<const char*> {};

TEST_P(AmdgpuPlacementBindingTest, OnlyQualifiedRegisterFormAndSubgroup) {
  const char* name = GetParam();
  const auto* info =
      loom_amdgpu_target_info_find_target(iree_make_cstring_view(name));
  ASSERT_NE(info, nullptr);
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const auto* descriptors = loom_low_descriptor_registry_lookup(
      &registry.registry, info->descriptor_set_key);
  if (descriptors == nullptr) {
    GTEST_SKIP()
        << "Descriptor set disabled by the selected build configuration";
  }
  const bool qualified =
      iree_string_view_equal(iree_make_cstring_view(name),
                             IREE_SV("gfx1100")) ||
      iree_string_view_equal(iree_make_cstring_view(name), IREE_SV("gfx1151"));
  for (uint32_t subgroup : {32u, 64u}) {
    const auto* processor =
        loom_amdgpu_target_info_processor_at(info->processor_ordinal);
    if (!loom_amdgpu_processor_properties_support_wavefront_size(
            &processor->properties, subgroup)) {
      continue;
    }
    const auto view = Preferences(name, subgroup, descriptors);
    if (!qualified || subgroup != 32) {
      EXPECT_EQ(view.indices_by_descriptor, nullptr);
      EXPECT_EQ(view.preferences, nullptr);
      continue;
    }
    ASSERT_NE(view.indices_by_descriptor, nullptr);
    uint32_t bound = 0;
    for (uint32_t i = 0; i < descriptors->descriptor_count; ++i) {
      if (!view.indices_by_descriptor[i]) {
        continue;
      }
      ++bound;
      EXPECT_EQ(i,
                loom_low_descriptor_set_lookup_descriptor(
                    descriptors, IREE_SV("amdgpu.v_wmma_f32_16x16x16_bf16")));
      const auto& preference =
          view.preferences[view.indices_by_descriptor[i] - 1];
      ASSERT_EQ(preference.value_count, 3);
      for (uint16_t j = 0; j < preference.value_count; ++j) {
        EXPECT_EQ(preference.values[j].operation_index, 0);
        EXPECT_EQ(preference.values[j].kind, LOOM_LOW_PLACEMENT_VALUE_OPERAND);
        EXPECT_EQ(preference.values[j].index, j);
      }
    }
    EXPECT_EQ(bound, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, AmdgpuPlacementBindingTest,
                         ::testing::Values("gfx1100", "gfx1101", "gfx1150",
                                           "gfx1151", "gfx1201", "gfx942"));

TEST(AmdgpuPlacementTest, FlatClausesMatchEveryPartialMinimum) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const auto* descriptors = loom_low_descriptor_registry_lookup(
      &registry.registry, IREE_SV("amdgpu.rdna3_5.core"));
  if (descriptors == nullptr) {
    GTEST_SKIP()
        << "Descriptor set disabled by the selected build configuration";
  }
  const auto view = Preferences("gfx1151", 32, descriptors);
  ASSERT_NE(view.indices_by_descriptor, nullptr);
  const uint32_t ordinal = loom_low_descriptor_set_lookup_descriptor(
      descriptors, IREE_SV("amdgpu.v_wmma_f32_16x16x16_bf16"));
  ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  ASSERT_NE(view.indices_by_descriptor[ordinal], 0);
  const auto& preference =
      view.preferences[view.indices_by_descriptor[ordinal] - 1];
  loom_low_placement_preference_use_t use = {&preference, 0, 1};
  loom_low_placement_preference_index_t index = {.uses = &use};
  const uint32_t use_index = 0;
  loom_low_allocation_preference_location_t locations[3] = {};
  loom_low_allocation_preference_query_t query = {.index = &index,
                                                  .use_indices = &use_index,
                                                  .locations = locations,
                                                  .use_count = 1};
  loom_low_allocation_assignment_t assignments[3] = {};
  uint16_t register_class = LOOM_LOW_REG_CLASS_NONE;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptors, IREE_SV("amdgpu.vgpr"), &register_class, nullptr));
  for (auto& assignment : assignments) {
    assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    assignment.descriptor_reg_class_id = register_class;
    assignment.unit_count = assignment.location_count = 8;
  }

  // Enumerate all mandatory-identity partitions and all known/unknown group
  // locations. The independent oracle minimizes the observed full rank over
  // every compatible completion, not over the factored predicates.
  const uint32_t partitions[][3] = {
      {0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  uint32_t checked = 0;
  for (const auto& partition : partitions) {
    for (uint32_t state = 0; state < 125; ++state) {
      uint32_t digits[3] = {state % 5, state / 5 % 5, state / 25};
      if ((partition[1] != 1 && digits[1] != 0) ||
          (partition[2] != 2 && digits[2] != 0)) {
        continue;
      }
      for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t known = digits[partition[i]];
        assignments[i].location_base = known - 1;
        locations[i] = {known ? &assignments[i] : nullptr, partition[i], 0};
      }
      uint32_t best = UINT32_MAX;
      for (uint32_t completion = 0; completion < 64; ++completion) {
        const uint32_t bases[] = {completion % 4, completion / 4 % 4,
                                  completion / 16};
        bool compatible = true;
        for (uint32_t i = 0; i < 3; ++i) {
          const uint32_t known = digits[partition[i]];
          compatible &= bases[i] == bases[partition[i]] &&
                        (!known || bases[i] == known - 1);
        }
        if (!compatible) {
          continue;
        }
        uint32_t counts[4] = {};
        for (uint32_t value :
             {bases[0], bases[1], bases[2], (bases[2] + 1) % 4}) {
          ++counts[value];
        }
        best = std::min(best, *std::max_element(counts, counts + 4) - 1);
      }
      ASSERT_NE(best, UINT32_MAX);
      EXPECT_EQ(loom_low_allocation_preference_penalty(descriptors, &query,
                                                       nullptr, nullptr),
                best)
          << "state=" << state << " partition=" << partition[0] << partition[1]
          << partition[2];
      ++checked;
    }
  }
  EXPECT_EQ(checked, 205u);
}

}  // namespace
}  // namespace loom
