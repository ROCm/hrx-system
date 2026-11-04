// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/fragment_memory/transpose.h"

#include <stdint.h>

#include <string>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/packet.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

const loom_low_descriptor_set_t* DescriptorSet(
    loom_target_low_descriptor_registry_t* registry,
    const char* processor_name) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(
          iree_make_cstring_view(processor_name));
  if (processor == nullptr) {
    return nullptr;
  }
  return loom_low_descriptor_registry_lookup(
      &registry->registry, processor->properties.descriptor_set.key);
}

loom_amdgpu_matrix_feature_bits_t MatrixFeatureBits(
    const char* processor_name) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(
          iree_make_cstring_view(processor_name));
  if (processor == nullptr) {
    return 0;
  }
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  EXPECT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
      processor->properties.features.matrix, &feature_bits));
  return feature_bits;
}

loom_amdgpu_fragment_memory_plan_t TransposedPlan(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_value_fact_memory_space_t memory_space =
        LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  const uint16_t element_byte_count = role_layout->element_bit_count / 8u;
  loom_amdgpu_fragment_memory_plan_t plan = {
      .operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
      .role = role,
      .layout_kind =
          static_cast<loom_amdgpu_matrix_fragment_layout_kind_t>(layout->kind),
      .dynamic_base_is_subgroup_uniform = true,
      .view_rank = 2,
      .register_count = role_layout->register_count,
      .payload_register_count = role_layout->register_count,
      .element_byte_count = element_byte_count,
      .payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE,
      .packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE,
  };
  plan.source.memory_space = memory_space;
  plan.source.minimum_alignment = 8;
  if (role == LOOM_CONTRACT_OPERAND_ROLE_LHS) {
    // A is logically [M, K]. Column-major storage makes M contiguous and
    // requires the hardware exchange to recover the native lane layout.
    plan.static_axis_byte_strides[0] = element_byte_count;
    plan.static_axis_byte_strides[1] = 16u * element_byte_count;
  } else {
    // B is logically [K, N]. Row-major storage makes N contiguous.
    plan.static_axis_byte_strides[0] = 16u * element_byte_count;
    plan.static_axis_byte_strides[1] = element_byte_count;
  }
  return plan;
}

void SetOrdinaryMajorStrides(const loom_amdgpu_matrix_fragment_layout_t* layout,
                             loom_contract_operand_role_t role,
                             loom_amdgpu_fragment_memory_plan_t* plan) {
  if (role == LOOM_CONTRACT_OPERAND_ROLE_LHS) {
    plan->static_axis_byte_strides[0] =
        layout->tile_shape.reduction_count * plan->element_byte_count;
    plan->static_axis_byte_strides[1] = plan->element_byte_count;
  } else {
    plan->static_axis_byte_strides[0] = plan->element_byte_count;
    plan->static_axis_byte_strides[1] =
        layout->tile_shape.reduction_count * plan->element_byte_count;
  }
}

TEST(AmdgpuFragmentMemoryTransposeTest, Gfx1200FamilyMatchesSemanticLayouts) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx1200");
  ASSERT_NE(descriptor_set, nullptr);
  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      MatrixFeatureBits("gfx1200");
  ASSERT_NE(feature_bits, 0u);

  bool visited[LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {};
  uint32_t selected_role_count = 0;
  uint32_t wave32_role_count = 0;
  uint32_t wave64_role_count = 0;
  uint32_t repeated_packet_role_count = 0;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* contract =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(contract);
    if (layout == nullptr || visited[layout->kind] ||
        !loom_amdgpu_matrix_contract_is_available(contract, feature_bits,
                                                  layout->wave_size)) {
      continue;
    }
    visited[layout->kind] = true;
    for (loom_contract_operand_role_t role :
         {LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_OPERAND_ROLE_RHS}) {
      const loom_matrix_fragment_role_layout_t* role_layout =
          loom_matrix_fragment_role_layout(layout, role);
      const bool applicable =
          (role_layout->element_bit_count == 8 ||
           role_layout->element_bit_count == 16) &&
          role_layout->coordinate_element_stride == 1 &&
          role_layout->reduction_group.storage_element_count == 0;
      if (!applicable) {
        continue;
      }

      SCOPED_TRACE(std::to_string(layout->kind) +
                   (role == LOOM_CONTRACT_OPERAND_ROLE_LHS ? " lhs" : " rhs"));
      loom_amdgpu_fragment_memory_plan_t plan = TransposedPlan(layout, role);
      loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
      ASSERT_TRUE(loom_amdgpu_fragment_memory_select_transpose_load(
          descriptor_set, layout, &plan, &choice));
      ASSERT_GT(choice.packet_count, 0u);
      const bool uses_repeated_packets = choice.packet_count > 1;
      EXPECT_EQ(choice.packet_count * choice.packets[0].packet_register_count,
                role_layout->register_count);
      EXPECT_TRUE(choice.packet_address_sources_consumed_at_issue);
      for (uint16_t packet_index = 0; packet_index < choice.packet_count;
           ++packet_index) {
        EXPECT_TRUE(iree_all_bits_set(
            choice.packets[packet_index].flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_NATIVE_TRANSPOSE_LOAD));
        EXPECT_EQ(
            iree_all_bits_set(
                choice.packets[packet_index].flags,
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FIRST_32_LANES_ONLY),
            layout->wave_size == 64);
        EXPECT_EQ(choice.packets[packet_index].issued_element_count, 8u);
      }

      loom_amdgpu_fragment_memory_apply_transpose_load(&choice, &plan);
      loom_amdgpu_fragment_memory_issued_access_t
          accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] =
              {};
      ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(
                    &plan, &plan.packets[0], accesses),
                1u);
      EXPECT_EQ(accesses[0].element_count, 8u);
      EXPECT_EQ(
          accesses[0].flags,
          layout->wave_size == 64
              ? LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_FIRST_32_LANES_ONLY
              : LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_NONE);

      SetOrdinaryMajorStrides(layout, role, &plan);
      EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
          descriptor_set, layout, &plan, &choice));

      ++selected_role_count;
      wave32_role_count += layout->wave_size == 32;
      wave64_role_count += layout->wave_size == 64;
      repeated_packet_role_count += uses_repeated_packets;
    }
  }
  EXPECT_GT(selected_role_count, 12u);
  EXPECT_GT(wave32_role_count, 4u);
  EXPECT_GT(wave64_role_count, 2u);
  EXPECT_GT(repeated_packet_role_count, 0u);
}

TEST(AmdgpuFragmentMemoryTransposeTest, Gfx1250ExtendsEveryEightBitRole) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx1250");
  ASSERT_NE(descriptor_set, nullptr);
  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      MatrixFeatureBits("gfx1250");
  ASSERT_NE(feature_bits, 0u);

  bool visited[LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {};
  uint32_t mixed_format_eight_bit_role_count = 0;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* contract =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(contract);
    if (layout == nullptr || visited[layout->kind] ||
        !loom_amdgpu_matrix_contract_is_available(contract, feature_bits,
                                                  layout->wave_size)) {
      continue;
    }
    visited[layout->kind] = true;
    const bool mixed_format =
        layout->lhs->element_bit_count != layout->rhs->element_bit_count;
    if (!mixed_format) {
      continue;
    }
    for (loom_contract_operand_role_t role :
         {LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_OPERAND_ROLE_RHS}) {
      const loom_matrix_fragment_role_layout_t* role_layout =
          loom_matrix_fragment_role_layout(layout, role);
      if (role_layout->element_bit_count != 8) {
        continue;
      }
      SCOPED_TRACE(std::to_string(layout->kind));
      const loom_amdgpu_fragment_memory_plan_t plan =
          TransposedPlan(layout, role);
      loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
      EXPECT_TRUE(loom_amdgpu_fragment_memory_select_transpose_load(
          descriptor_set, layout, &plan, &choice));
      ++mixed_format_eight_bit_role_count;
    }
  }
  EXPECT_GT(mixed_format_eight_bit_role_count, 0u);
}

TEST(AmdgpuFragmentMemoryTransposeTest,
     Wave32B16RetainsExactLaneAndPacketAddresses) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx1200");
  ASSERT_NE(descriptor_set, nullptr);
  const auto* layout = loom_amdgpu_matrix_fragment_layout_for_kind(
      LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X32_F16);
  ASSERT_NE(layout, nullptr);
  const loom_amdgpu_fragment_memory_plan_t plan =
      TransposedPlan(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS);

  loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
  ASSERT_TRUE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &plan, &choice));
  ASSERT_EQ(choice.packet_count, 2u);
  EXPECT_EQ(choice.address_layout.payload_elements_per_register, 2u);
  EXPECT_EQ(choice.address_layout.payload_registers_per_element, 1u);
  EXPECT_EQ(choice.address_layout.packed_element_byte_stride, 2u);
  EXPECT_EQ(choice.address_layout.primary_lane_divisor, 8u);
  EXPECT_EQ(choice.address_layout.linear_lane_byte_stride, 0u);
  ASSERT_EQ(choice.address_layout.lane_term_count, 3u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].divisor, 1u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].modulus, 8u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].byte_stride, 32u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].divisor, 8u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].modulus, 2u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].byte_stride, 16u);
  EXPECT_EQ(choice.address_layout.lane_terms[2].divisor, 16u);
  EXPECT_EQ(choice.address_layout.lane_terms[2].modulus, 0u);
  EXPECT_EQ(choice.address_layout.lane_terms[2].byte_stride, 512u);
  for (uint16_t register_index = 0; register_index < 4; ++register_index) {
    EXPECT_EQ(choice.address_layout.register_byte_offsets[register_index], 0u);
  }
  for (uint16_t register_index = 4; register_index < 8; ++register_index) {
    EXPECT_EQ(choice.address_layout.register_byte_offsets[register_index],
              256u);
  }
  EXPECT_EQ(choice.packets[0].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B128_SADDR);
  EXPECT_EQ(choice.packets[0].register_index, 0u);
  EXPECT_EQ(choice.packets[0].result_register_count, 4u);
  EXPECT_EQ(choice.packets[1].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B128_SADDR);
  EXPECT_EQ(choice.packets[1].register_index, 4u);
  EXPECT_EQ(choice.packets[1].result_register_count, 4u);
}

TEST(AmdgpuFragmentMemoryTransposeTest, Gfx950B16FamilyMatchesSemanticLayouts) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx950");
  ASSERT_NE(descriptor_set, nullptr);
  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      MatrixFeatureBits("gfx950");
  ASSERT_NE(feature_bits, 0u);

  bool visited[LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {};
  uint32_t selected_role_count = 0;
  uint32_t selected_16x16_role_count = 0;
  uint32_t selected_32x32_role_count = 0;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* contract =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(contract);
    if (layout == nullptr || visited[layout->kind] ||
        !loom_amdgpu_matrix_contract_is_available(contract, feature_bits,
                                                  layout->wave_size)) {
      continue;
    }
    visited[layout->kind] = true;
    for (loom_contract_operand_role_t role :
         {LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_OPERAND_ROLE_RHS}) {
      const loom_matrix_fragment_role_layout_t* role_layout =
          loom_matrix_fragment_role_layout(layout, role);
      if (role_layout->element_bit_count != 16 ||
          layout->tile_shape.block_count != 1 ||
          role_layout->coordinate_element_stride != 1 ||
          role_layout->reduction_group.storage_element_count != 0 ||
          role_layout->register_count % 2 != 0) {
        continue;
      }

      SCOPED_TRACE(std::to_string(layout->kind) +
                   (role == LOOM_CONTRACT_OPERAND_ROLE_LHS ? " lhs" : " rhs"));
      loom_amdgpu_fragment_memory_plan_t plan =
          TransposedPlan(layout, role, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);
      loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
      ASSERT_TRUE(loom_amdgpu_fragment_memory_select_transpose_load(
          descriptor_set, layout, &plan, &choice));
      ASSERT_EQ(choice.packet_count, role_layout->register_count / 2u);
      for (uint16_t packet_index = 0; packet_index < choice.packet_count;
           ++packet_index) {
        EXPECT_EQ(
            choice.packets[packet_index].flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_NATIVE_TRANSPOSE_LOAD);
        EXPECT_EQ(choice.packets[packet_index].issued_element_count, 4u);
        EXPECT_EQ(choice.packets[packet_index].descriptor_ref,
                  LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64_TR_B16);
      }

      loom_amdgpu_fragment_memory_apply_transpose_load(&choice, &plan);
      loom_amdgpu_fragment_memory_issued_access_t
          accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] =
              {};
      ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(
                    &plan, &plan.packets[0], accesses),
                1u);
      EXPECT_EQ(accesses[0].element_count, 4u);
      EXPECT_EQ(accesses[0].flags,
                LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_NONE);

      ++selected_role_count;
      selected_16x16_role_count += layout->tile_shape.result_row_count == 16 &&
                                   layout->tile_shape.result_column_count == 16;
      selected_32x32_role_count += layout->tile_shape.result_row_count == 32 &&
                                   layout->tile_shape.result_column_count == 32;
    }
  }
  // Enumerate every available single-block, unpacked 16-bit role whose payload
  // fills whole B64 packets, including the uncompressed sides of sparse
  // contracts.
  EXPECT_EQ(selected_role_count, 18u);
  EXPECT_EQ(selected_16x16_role_count, 10u);
  EXPECT_EQ(selected_32x32_role_count, 8u);
}

TEST(AmdgpuFragmentMemoryTransposeTest,
     Gfx950B16RetainsExactLaneAndPacketAddresses) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx950");
  ASSERT_NE(descriptor_set, nullptr);
  const auto* layout = loom_amdgpu_matrix_fragment_layout_for_kind(
      LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_BF16);
  ASSERT_NE(layout, nullptr);
  loom_amdgpu_fragment_memory_plan_t plan =
      TransposedPlan(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
                     LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);

  loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
  ASSERT_TRUE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &plan, &choice));
  ASSERT_EQ(choice.packet_count, 2u);
  EXPECT_EQ(choice.address_layout.primary_lane_divisor, 16u);
  EXPECT_EQ(choice.address_layout.linear_lane_byte_stride, 0u);
  ASSERT_EQ(choice.address_layout.lane_term_count, 2u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].divisor, 1u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].modulus, 16u);
  EXPECT_EQ(choice.address_layout.lane_terms[0].byte_stride, 8u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].divisor, 16u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].modulus, 0u);
  EXPECT_EQ(choice.address_layout.lane_terms[1].byte_stride, 256u);
  for (uint16_t register_index = 0; register_index < 2; ++register_index) {
    EXPECT_EQ(choice.address_layout.register_byte_offsets[register_index], 0u);
  }
  for (uint16_t register_index = 2; register_index < 4; ++register_index) {
    EXPECT_EQ(choice.address_layout.register_byte_offsets[register_index],
              128u);
  }
  EXPECT_EQ(choice.packets[0].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64_TR_B16);
  EXPECT_EQ(choice.packets[0].register_index, 0u);
  EXPECT_EQ(choice.packets[0].result_register_count, 2u);
  EXPECT_EQ(choice.packets[1].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64_TR_B16);
  EXPECT_EQ(choice.packets[1].register_index, 2u);
  EXPECT_EQ(choice.packets[1].result_register_count, 2u);

  plan.source.minimum_alignment = 4;
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &plan, &choice));
  plan.source.minimum_alignment = 8;
  plan.source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &plan, &choice));

  const auto* b8_layout = loom_amdgpu_matrix_fragment_layout_for_kind(
      LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_PACKED8);
  ASSERT_NE(b8_layout, nullptr);
  const loom_amdgpu_fragment_memory_plan_t b8_plan =
      TransposedPlan(b8_layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
                     LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, b8_layout, &b8_plan, &choice));
}

TEST(AmdgpuFragmentMemoryTransposeTest, RejectsCompressedAndDynamicLayouts) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx1200");
  ASSERT_NE(descriptor_set, nullptr);
  const auto* layout = loom_amdgpu_matrix_fragment_layout_for_kind(
      LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED16_W64);
  ASSERT_NE(layout, nullptr);
  ASSERT_NE(layout->lhs->reduction_group.storage_element_count, 0u);

  loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
  loom_amdgpu_fragment_memory_plan_t lhs_plan =
      TransposedPlan(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &lhs_plan, &choice));

  loom_amdgpu_fragment_memory_plan_t rhs_plan =
      TransposedPlan(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS);
  rhs_plan.runtime_axes[0].byte_stride.kind =
      LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC;
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &rhs_plan, &choice));
}

TEST(AmdgpuFragmentMemoryTransposeTest, RequiresTargetDescriptorFamily) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_low_descriptor_set_t* descriptor_set =
      DescriptorSet(&registry, "gfx1100");
  ASSERT_NE(descriptor_set, nullptr);
  const auto* layout = loom_amdgpu_matrix_fragment_layout_for_kind(
      LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_F16);
  ASSERT_NE(layout, nullptr);
  const loom_amdgpu_fragment_memory_plan_t plan =
      TransposedPlan(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS);
  loom_amdgpu_fragment_memory_transpose_choice_t choice = {};
  EXPECT_FALSE(loom_amdgpu_fragment_memory_select_transpose_load(
      descriptor_set, layout, &plan, &choice));
}

}  // namespace
