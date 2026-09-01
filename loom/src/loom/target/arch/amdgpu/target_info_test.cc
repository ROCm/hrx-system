// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_info.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AmdgpuTargetInfoTest, IteratesProcessors) {
  const iree_host_size_t count = loom_amdgpu_target_info_processor_count();
  ASSERT_GT(count, 0u);
  EXPECT_EQ(loom_amdgpu_target_info_processor_at(count), nullptr);

  for (iree_host_size_t index = 0; index < count; ++index) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(index);
    ASSERT_NE(processor, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(processor->name));
    EXPECT_EQ(processor->ordinal, index);

    const loom_amdgpu_processor_info_t* found_processor = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_processor(processor->name,
                                                            &found_processor));
    EXPECT_EQ(found_processor, processor);
    EXPECT_EQ(processor->properties.elf.generic_version != 0,
              iree_string_view_ends_with(processor->name, IREE_SV("-generic")));
  }
}

TEST(AmdgpuTargetInfoTest, IteratesCanonicalTargets) {
  const iree_host_size_t count = loom_amdgpu_target_info_target_count();
  ASSERT_GT(count, 0u);
  EXPECT_EQ(loom_amdgpu_target_info_target_at(count), nullptr);

  for (iree_host_size_t index = 0; index < count; ++index) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(index);
    ASSERT_NE(target, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(target->name));
    EXPECT_EQ(target->target_kind, index + 1);
    EXPECT_EQ(loom_amdgpu_target_info_find_target(target->name), target);
    EXPECT_EQ(loom_amdgpu_target_info_find_target_by_kind(target->target_kind),
              target);
    EXPECT_NE(loom_amdgpu_target_info_target_processor(target), nullptr);
  }
}

TEST(AmdgpuTargetInfoTest, ExhaustsCodeObjectSatisfactionRelation) {
  const iree_host_size_t count = loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t effective_index = 0; effective_index < count;
       ++effective_index) {
    const loom_amdgpu_processor_info_t* effective_processor =
        loom_amdgpu_target_info_processor_at(effective_index);
    ASSERT_NE(effective_processor, nullptr);

    const uint16_t generic_processor_ordinal =
        effective_processor->generic_code_object.processor_ordinal;
    if (generic_processor_ordinal == LOOM_AMDGPU_PROCESSOR_ORDINAL_NONE) {
      EXPECT_EQ(effective_processor->generic_code_object.introduction_version,
                0u);
    } else {
      const loom_amdgpu_processor_info_t* generic_processor =
          loom_amdgpu_target_info_processor_at(generic_processor_ordinal);
      ASSERT_NE(generic_processor, nullptr);
      EXPECT_GT(generic_processor->properties.elf.generic_version, 0u);
      EXPECT_GT(effective_processor->generic_code_object.introduction_version,
                0u);
      EXPECT_LE(effective_processor->generic_code_object.introduction_version,
                generic_processor->properties.elf.generic_version);
    }

    for (iree_host_size_t required_index = 0; required_index < count;
         ++required_index) {
      const loom_amdgpu_processor_info_t* required_processor =
          loom_amdgpu_target_info_processor_at(required_index);
      ASSERT_NE(required_processor, nullptr);
      const bool expected =
          effective_processor->ordinal == required_processor->ordinal ||
          (generic_processor_ordinal == required_processor->ordinal &&
           effective_processor->generic_code_object.introduction_version <=
               required_processor->properties.elf.generic_version);
      EXPECT_EQ(loom_amdgpu_processor_satisfies_code_object_requirement(
                    effective_processor, required_processor),
                expected)
          << "effective ordinal " << effective_processor->ordinal
          << ", required ordinal " << required_processor->ordinal;
    }
  }
}

TEST(AmdgpuTargetInfoTest, MatchesNamedCodeObjectRelationWitnesses) {
  const loom_amdgpu_processor_info_t* gfx1151 =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  const loom_amdgpu_processor_info_t* gfx1170 =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1170"));
  const loom_amdgpu_processor_info_t* gfx11_generic =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx11-generic"));
  const loom_amdgpu_processor_info_t* gfx1250 =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  const loom_amdgpu_processor_info_t* gfx1251 =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1251"));
  const loom_amdgpu_processor_info_t* gfx12_generic =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx12-generic"));
  const loom_amdgpu_processor_info_t* gfx12_5_generic =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx12-5-generic"));
  ASSERT_NE(gfx1151, nullptr);
  ASSERT_NE(gfx1170, nullptr);
  ASSERT_NE(gfx11_generic, nullptr);
  ASSERT_NE(gfx1250, nullptr);
  ASSERT_NE(gfx1251, nullptr);
  ASSERT_NE(gfx12_generic, nullptr);
  ASSERT_NE(gfx12_5_generic, nullptr);

  EXPECT_TRUE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1151, gfx11_generic));
  EXPECT_FALSE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1170, gfx11_generic));
  EXPECT_FALSE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1151, gfx1170));
  EXPECT_TRUE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1250, gfx12_5_generic));
  EXPECT_TRUE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1251, gfx12_5_generic));
  EXPECT_FALSE(loom_amdgpu_processor_satisfies_code_object_requirement(
      gfx1250, gfx12_generic));
}

TEST(AmdgpuTargetInfoTest, HonorsGenericCodeObjectIntroductionVersion) {
  const loom_amdgpu_processor_info_t* gfx1151 =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  const loom_amdgpu_processor_info_t* gfx11_generic =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx11-generic"));
  ASSERT_NE(gfx1151, nullptr);
  ASSERT_NE(gfx11_generic, nullptr);

  loom_amdgpu_processor_info_t effective_processor = *gfx1151;
  loom_amdgpu_processor_info_t required_processor = *gfx11_generic;
  effective_processor.generic_code_object.introduction_version = 2;

  required_processor.properties.elf.generic_version = 1;
  EXPECT_FALSE(loom_amdgpu_processor_satisfies_code_object_requirement(
      &effective_processor, &required_processor));

  required_processor.properties.elf.generic_version = 2;
  EXPECT_TRUE(loom_amdgpu_processor_satisfies_code_object_requirement(
      &effective_processor, &required_processor));
}

TEST(AmdgpuTargetInfoTest, IteratesDescriptorSets) {
  const iree_host_size_t count = loom_amdgpu_target_info_descriptor_set_count();
  ASSERT_GT(count, 0u);
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at((uint16_t)count),
            nullptr);

  for (uint16_t ordinal = 0; ordinal < count; ++ordinal) {
    const loom_amdgpu_descriptor_set_info_t* descriptor_set =
        loom_amdgpu_target_info_descriptor_set_at(ordinal);
    ASSERT_NE(descriptor_set, nullptr);
    EXPECT_EQ(descriptor_set->ordinal, ordinal);
    EXPECT_FALSE(iree_string_view_is_empty(descriptor_set->key));

    const loom_amdgpu_descriptor_set_info_t* found_by_key = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set(
        descriptor_set->key, &found_by_key));
    EXPECT_EQ(found_by_key, descriptor_set);

    const loom_amdgpu_descriptor_set_info_t* found_by_ordinal = nullptr;
    IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
        ordinal, &found_by_ordinal));
    EXPECT_EQ(found_by_ordinal, descriptor_set);
  }
}

TEST(AmdgpuTargetInfoTest, DescriptorSetAtReturnsNullForUnknownOrdinal) {
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(
                LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE),
            nullptr);
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(UINT16_MAX - 1), nullptr);
}

TEST(AmdgpuTargetInfoTest, RejectsUnsupportedDescriptorSetKey) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_amdgpu_target_info_lookup_descriptor_set(
          IREE_SV("amdgpu.unsupported.core"), &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST(AmdgpuTargetInfoTest, RejectsUnsupportedDescriptorSetOrdinal) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
          UINT16_MAX - 1, &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST(AmdgpuTargetInfoTest, MatchesAmdhsaGfx9PlusProcessorElfFlags) {
  const struct {
    iree_string_view_t processor;
    uint32_t elf_flags;
  } cases[] = {
      {IREE_SV("gfx900"), 0x12Cu},
      {IREE_SV("gfx902"), 0x12Du},
      {IREE_SV("gfx904"), 0x12Eu},
      {IREE_SV("gfx906"), 0x52Fu},
      {IREE_SV("gfx908"), 0x530u},
      {IREE_SV("gfx909"), 0x131u},
      {IREE_SV("gfx90a"), 0x53Fu},
      {IREE_SV("gfx90c"), 0x132u},
      {IREE_SV("gfx940"), 0x540u},
      {IREE_SV("gfx941"), 0x54Bu},
      {IREE_SV("gfx942"), 0x54Cu},
      {IREE_SV("gfx950"), 0x54Fu},
      {IREE_SV("gfx1010"), 0x133u},
      {IREE_SV("gfx1011"), 0x134u},
      {IREE_SV("gfx1012"), 0x135u},
      {IREE_SV("gfx1013"), 0x142u},
      {IREE_SV("gfx1030"), 0x036u},
      {IREE_SV("gfx1031"), 0x037u},
      {IREE_SV("gfx1032"), 0x038u},
      {IREE_SV("gfx1033"), 0x039u},
      {IREE_SV("gfx1034"), 0x03Eu},
      {IREE_SV("gfx1035"), 0x03Du},
      {IREE_SV("gfx1036"), 0x045u},
      {IREE_SV("gfx1100"), 0x041u},
      {IREE_SV("gfx1101"), 0x046u},
      {IREE_SV("gfx1102"), 0x047u},
      {IREE_SV("gfx1103"), 0x044u},
      {IREE_SV("gfx1150"), 0x043u},
      {IREE_SV("gfx1151"), 0x04Au},
      {IREE_SV("gfx1152"), 0x055u},
      {IREE_SV("gfx1153"), 0x058u},
      {IREE_SV("gfx1170"), 0x05Du},
      {IREE_SV("gfx1171"), 0x05Eu},
      {IREE_SV("gfx1172"), 0x05Cu},
      {IREE_SV("gfx1200"), 0x048u},
      {IREE_SV("gfx1201"), 0x04Eu},
      {IREE_SV("gfx1250"), 0x549u},
      {IREE_SV("gfx1251"), 0x55Au},
      {IREE_SV("gfx1310"), 0x050u},
      {IREE_SV("gfx9-generic"), 0x01000151u},
      {IREE_SV("gfx10-1-generic"), 0x01000152u},
      {IREE_SV("gfx10-3-generic"), 0x01000053u},
      {IREE_SV("gfx11-generic"), 0x01000054u},
      {IREE_SV("gfx12-generic"), 0x01000059u},
      {IREE_SV("gfx9-4-generic"), 0x0100055Fu},
      {IREE_SV("gfx12-5-generic"), 0x0100055Bu},
  };
  ASSERT_EQ(IREE_ARRAYSIZE(cases), loom_amdgpu_target_info_processor_count());
  for (const auto& c : cases) {
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_target_info_lookup_processor(c.processor, &processor));
    ASSERT_NE(processor, nullptr);
    EXPECT_TRUE(iree_string_view_equal(processor->name, c.processor));
    EXPECT_EQ(processor->properties.elf.feature_flags &
                  LOOM_AMDGPU_ELF_GENERIC_VERSION_MASK_V6,
              0u);
    EXPECT_EQ(processor->properties.elf.machine_flags |
                  processor->properties.elf.feature_flags |
                  (processor->properties.elf.generic_version
                   << LOOM_AMDGPU_ELF_GENERIC_VERSION_OFFSET_V6),
              c.elf_flags);
  }
}

TEST(AmdgpuTargetInfoTest, ResolvesPhysicalObservationsToCanonicalTargets) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1250"));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(loom_amdgpu_target_info_requires_physical_resolution(processor));
  const loom_amdgpu_target_info_t* a0 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_physical_target(processor, 0, &a0));
  const loom_amdgpu_target_info_t* b0 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_physical_target(processor, 1, &b0));
  ASSERT_NE(a0, nullptr);
  ASSERT_NE(b0, nullptr);
  EXPECT_TRUE(iree_string_view_equal(a0->name, IREE_SV("gfx1250-a0")));
  EXPECT_TRUE(iree_string_view_equal(b0->name, IREE_SV("gfx1250")));

  const loom_amdgpu_target_info_t* unknown = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_info_lookup_physical_target(processor, 2, &unknown));
  EXPECT_EQ(unknown, nullptr);

  const loom_amdgpu_processor_info_t* same_named_processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx1151"));
  ASSERT_NE(same_named_processor, nullptr);
  EXPECT_FALSE(loom_amdgpu_target_info_requires_physical_resolution(
      same_named_processor));
  const loom_amdgpu_target_info_t* same_named_target = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_physical_target(
      same_named_processor, UINT32_MAX, &same_named_target));
  ASSERT_NE(same_named_target, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(same_named_target->name, IREE_SV("gfx1151")));
}

TEST(AmdgpuTargetInfoTest, ResolvesOverlayTargetSemantics) {
  const loom_amdgpu_target_info_t* a0 =
      loom_amdgpu_target_info_find_target(IREE_SV("gfx1250-a0"));
  const loom_amdgpu_target_info_t* b0 =
      loom_amdgpu_target_info_find_target(IREE_SV("gfx1250"));
  ASSERT_NE(a0, nullptr);
  ASSERT_NE(b0, nullptr);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(a0);
  ASSERT_NE(processor, nullptr);
  EXPECT_EQ(processor, loom_amdgpu_target_info_target_processor(b0));
  EXPECT_NE(a0->lds_bank_service_model_set_ordinal,
            LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE);
  EXPECT_NE(a0->instruction_constraints, 0u);
  EXPECT_EQ(a0->instruction_constraints &
                ~LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
            0u);
  ASSERT_EQ(a0->kernel_metadata_extensions.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(a0->kernel_metadata_extensions.entries[0].key,
                             IREE_SV(".gfx1250_revision")));
  EXPECT_TRUE(iree_string_view_equal(
      a0->kernel_metadata_extensions.entries[0].value, IREE_SV("A0")));
  EXPECT_EQ(b0->instruction_constraints, 0u);
  ASSERT_EQ(b0->kernel_metadata_extensions.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(b0->kernel_metadata_extensions.entries[0].key,
                             IREE_SV(".gfx1250_revision")));
  EXPECT_TRUE(iree_string_view_equal(
      b0->kernel_metadata_extensions.entries[0].value, IREE_SV("B0")));
}

TEST(AmdgpuTargetInfoTest, RejectsUnknownProcessor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx9999"), &processor));
  EXPECT_EQ(processor, nullptr);
}

}  // namespace
