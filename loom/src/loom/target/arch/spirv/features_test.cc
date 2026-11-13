// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/features.h"

#include "iree/testing/gtest.h"
#include "loom/target/arch/spirv/records/target_records.h"

namespace {

bool ContainsExtension(const loom_spirv_feature_set_t& feature_set,
                       iree_string_view_t extension_name) {
  for (uint8_t i = 0; i < feature_set.extension_count; ++i) {
    if (iree_string_view_equal(feature_set.extension_names[i],
                               extension_name)) {
      return true;
    }
  }
  return false;
}

bool ContainsCapability(const loom_spirv_feature_set_t& feature_set,
                        uint32_t capability) {
  for (uint8_t i = 0; i < feature_set.capability_count; ++i) {
    if (feature_set.capabilities[i] == capability) {
      return true;
    }
  }
  return false;
}

bool ContainsOpcode(const loom_spirv_feature_set_t& feature_set,
                    uint32_t opcode) {
  for (uint8_t i = 0; i < feature_set.opcode_count; ++i) {
    if (feature_set.opcodes[i] == opcode) {
      return true;
    }
  }
  return false;
}

bool ContainsStorageClass(const loom_spirv_feature_set_t& feature_set,
                          uint32_t storage_class) {
  for (uint8_t i = 0; i < feature_set.storage_class_count; ++i) {
    if (feature_set.storage_classes[i] == storage_class) {
      return true;
    }
  }
  return false;
}

bool ContainsDecoration(const loom_spirv_feature_set_t& feature_set,
                        uint32_t decoration) {
  for (uint8_t i = 0; i < feature_set.decoration_count; ++i) {
    if (feature_set.decorations[i] == decoration) {
      return true;
    }
  }
  return false;
}

TEST(SpirvFeaturesTest, KnownAtomsHaveDirectBitsAndDescriptors) {
  EXPECT_EQ(loom_spirv_feature_atom_bit(LOOM_SPIRV_FEATURE_ATOM_UNKNOWN), 0u);
  EXPECT_EQ(loom_spirv_feature_atom_bit(LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER),
            LOOM_SPIRV_FEATURE_VULKAN_SHADER);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER),
            LOOM_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV),
            LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV),
            LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV);
  EXPECT_EQ(loom_spirv_feature_atom_bit(LOOM_SPIRV_FEATURE_ATOM_INT64),
            LOOM_SPIRV_FEATURE_INT64);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR),
            LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR);
  EXPECT_EQ(
      loom_spirv_feature_atom_bit(LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR),
      LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_DOT_PRODUCT_KHR),
            LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR);
  EXPECT_EQ(loom_spirv_feature_atom_bit(
                LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR),
            LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR);

  const loom_spirv_feature_atom_descriptor_t* bda_descriptor =
      loom_spirv_feature_atom_descriptor(
          LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER);
  ASSERT_NE(bda_descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(bda_descriptor->name,
                                     IREE_SV("spirv.physical_storage_buffer")));
  EXPECT_EQ(bda_descriptor->required_atom_bits,
            LOOM_SPIRV_FEATURE_VULKAN_SHADER);
}

TEST(SpirvFeaturesTest, PreparesVulkanBdaProfile) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
                                 &feature_set);

  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(&feature_set,
                                              LOOM_SPIRV_FEATURE_ATOM_INT64));
  EXPECT_EQ(feature_set.addressing_model,
            LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64);
  EXPECT_EQ(feature_set.memory_model, LOOM_SPIRV_MEMORY_MODEL_VULKAN);
  EXPECT_EQ(feature_set.minimum_spirv_version, UINT32_C(0x00010300));
  EXPECT_TRUE(
      ContainsExtension(feature_set, IREE_SV("SPV_KHR_vulkan_memory_model")));
  EXPECT_TRUE(ContainsExtension(feature_set,
                                IREE_SV("SPV_KHR_physical_storage_buffer")));
  EXPECT_TRUE(ContainsCapability(feature_set, LOOM_SPIRV_CAPABILITY_SHADER));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES));
  EXPECT_TRUE(ContainsCapability(feature_set, LOOM_SPIRV_CAPABILITY_INT64));
  EXPECT_TRUE(ContainsStorageClass(
      feature_set, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER));
  EXPECT_TRUE(
      ContainsDecoration(feature_set, LOOM_SPIRV_DECORATION_RESTRICT_POINTER));
  EXPECT_TRUE(
      ContainsDecoration(feature_set, LOOM_SPIRV_DECORATION_ALIASED_POINTER));
}

TEST(SpirvFeaturesTest, PreparesSubgroupBallotCapability) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
          LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
      &feature_set);

  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_GROUP_NON_UNIFORM_BALLOT));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_BALLOT));
}

TEST(SpirvFeaturesTest, PreparesSubgroupVotes) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
                                     LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
                                 &feature_set);
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_GROUP_NON_UNIFORM_VOTE));
  EXPECT_TRUE(ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_ALL));
  EXPECT_TRUE(ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_ANY));
  EXPECT_EQ(loom_spirv_feature_bits_with_dependencies(
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE),
            LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM |
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE);
}

TEST(SpirvFeaturesTest, PreparesSubgroupShuffleCapabilities) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA |
          LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE |
          LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
      &feature_set);

  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_SHUFFLE));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_SHUFFLE_XOR));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_GROUP_NON_UNIFORM_SHUFFLE_UP));
  EXPECT_TRUE(ContainsOpcode(feature_set,
                             LOOM_SPIRV_OP_GROUP_NON_UNIFORM_SHUFFLE_DOWN));
}

TEST(SpirvFeaturesTest, ClosesTransitiveSubgroupBallotDependencies) {
  EXPECT_EQ(loom_spirv_feature_bits_with_dependencies(
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT),
            LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM |
                LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT);
}

TEST(SpirvFeaturesTest, PreparesAllFloatControlFeatures) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(LOOM_SPIRV_FEATURE_KNOWN_BITS, &feature_set);
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_FLOAT32_DENORM_PRESERVE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_DENORM_PRESERVE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_FLOAT16_64_ROUNDING_RTE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_FLOAT32_ROUNDING_RTE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set,
      LOOM_SPIRV_FEATURE_ATOM_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set,
      LOOM_SPIRV_FEATURE_ATOM_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set,
      LOOM_SPIRV_FEATURE_ATOM_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_FLOAT64_TO_FLOAT16_CONVERSION));
  EXPECT_TRUE(
      ContainsExtension(feature_set, IREE_SV("SPV_KHR_float_controls")));
  EXPECT_TRUE(
      ContainsCapability(feature_set, LOOM_SPIRV_CAPABILITY_DENORM_PRESERVE));
  EXPECT_TRUE(
      ContainsCapability(feature_set, LOOM_SPIRV_CAPABILITY_ROUNDING_MODE_RTE));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_SIGNED_ZERO_INF_NAN_PRESERVE));
}

TEST(SpirvFeaturesTest, PreparesCooperativeFeatureAtomsWithoutBda) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                                     LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
                                     LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                                 &feature_set);

  EXPECT_EQ(feature_set.addressing_model, LOOM_SPIRV_ADDRESSING_MODEL_LOGICAL);
  EXPECT_EQ(feature_set.memory_model, LOOM_SPIRV_MEMORY_MODEL_VULKAN);
  EXPECT_EQ(feature_set.minimum_spirv_version, UINT32_C(0x00010300));
  EXPECT_TRUE(
      ContainsExtension(feature_set, IREE_SV("SPV_KHR_cooperative_matrix")));
  EXPECT_TRUE(
      ContainsExtension(feature_set, IREE_SV("SPV_NV_cooperative_vector")));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_NV));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_TYPE_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(ContainsOpcode(feature_set,
                             LOOM_SPIRV_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR));
  EXPECT_TRUE(
      ContainsOpcode(feature_set, LOOM_SPIRV_OP_TYPE_COOPERATIVE_VECTOR_NV));
  EXPECT_TRUE(ContainsOpcode(
      feature_set, LOOM_SPIRV_OP_COOPERATIVE_VECTOR_MATRIX_MUL_ADD_NV));
}

TEST(SpirvFeaturesTest, PreparesCooperativeVectorTrainingAtom) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_VULKAN_SHADER |
          LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |
          LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV,
      &feature_set);

  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_NV));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_TRAINING_NV));
  EXPECT_TRUE(ContainsOpcode(
      feature_set,
      LOOM_SPIRV_OP_COOPERATIVE_VECTOR_OUTER_PRODUCT_ACCUMULATE_NV));
  EXPECT_TRUE(ContainsOpcode(
      feature_set, LOOM_SPIRV_OP_COOPERATIVE_VECTOR_REDUCE_SUM_ACCUMULATE_NV));
}

TEST(SpirvFeaturesTest, PreparesBfloat16TypeAtom) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_VULKAN_SHADER | LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
      &feature_set);

  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR));
  EXPECT_TRUE(ContainsExtension(feature_set, IREE_SV("SPV_KHR_bfloat16")));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_B_FLOAT16_TYPE_KHR));
}

TEST(SpirvFeaturesTest, PreparesBfloat16CooperativeMatrixAtom) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_VULKAN_SHADER |
          LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
          LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
          LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR,
      &feature_set);

  EXPECT_TRUE(
      ContainsExtension(feature_set, IREE_SV("SPV_KHR_cooperative_matrix")));
  EXPECT_TRUE(ContainsExtension(feature_set, IREE_SV("SPV_KHR_bfloat16")));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_B_FLOAT16_TYPE_KHR));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_B_FLOAT16_COOPERATIVE_MATRIX_KHR));
}

TEST(SpirvFeaturesTest, PreparesBfloat16DotProductAtom) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_VULKAN_SHADER | LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
          LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR,
      &feature_set);

  EXPECT_TRUE(ContainsExtension(feature_set, IREE_SV("SPV_KHR_bfloat16")));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_B_FLOAT16_TYPE_KHR));
  EXPECT_TRUE(ContainsCapability(
      feature_set, LOOM_SPIRV_CAPABILITY_B_FLOAT16_DOT_PRODUCT_KHR));
}

TEST(SpirvFeaturesTest, ClosesTransitiveDependencies) {
  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(
      LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR, &feature_set);
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER));
  EXPECT_TRUE(loom_spirv_feature_set_has_atom(
      &feature_set, LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR));
  EXPECT_TRUE(ContainsExtension(feature_set, IREE_SV("SPV_KHR_bfloat16")));
  EXPECT_TRUE(ContainsCapability(feature_set,
                                 LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR));
}

TEST(SpirvFeaturesTest, TargetRecordSelectsPreparedVulkanBdaProfile) {
  const loom_target_bundle_t& target = loom_spirv_low_target_bundle_vulkan1_3;
  ASSERT_NE(target.config, nullptr);
  EXPECT_EQ(target.config->contract_feature_bits,
            LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA);

  loom_spirv_feature_set_t feature_set;
  loom_spirv_feature_set_prepare(target.config->contract_feature_bits,
                                 &feature_set);
  EXPECT_EQ(feature_set.addressing_model,
            LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64);
}

}  // namespace
