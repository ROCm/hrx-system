// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/fragment_memory/packet.h"

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/layout.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

TEST(AmdgpuFragmentMemoryPacketTest, OriginPartitionPreservesStaticBytes) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.source.static_byte_offset = 256;
  plan.source.static_view_base_byte_offset = 256;
  plan.source.dynamic_view_base_value_id = 1;
  plan.source.dynamic_view_base_term_count = 1;
  plan.source.dynamic_term_count = 1;
  plan.register_count = 2;
  plan.address_layout.register_byte_offsets[1] = 16;

  // A recovered dynamic value excludes the static view contribution.
  uint64_t offset = 0;
  ASSERT_TRUE(loom_amdgpu_fragment_memory_vaddr_static_offset_u32(&plan, 1, 0,
                                                                  &offset));
  EXPECT_EQ(offset, 272u);

  // An authored complete origin already includes that contribution.
  plan.source.dynamic_view_base_value_static_byte_offset = 256;
  ASSERT_TRUE(loom_amdgpu_fragment_memory_vaddr_static_offset_u32(&plan, 1, 0,
                                                                  &offset));
  EXPECT_EQ(offset, 16u);

  // Moving the canonical origin to SADDR leaves the same relative packet
  // displacement without changing the source-relative reporting address.
  plan.scalar_base.byte_offset = 256;
  plan.scalar_base.dynamic_term_mask = 1;
  ASSERT_TRUE(loom_amdgpu_fragment_memory_vaddr_static_offset_u32(&plan, 1, 0,
                                                                  &offset));
  EXPECT_EQ(offset, 16u);
  int64_t source_offset = 0;
  ASSERT_TRUE(loom_amdgpu_fragment_memory_static_offset_i64(&plan, 1, 0,
                                                            &source_offset));
  EXPECT_EQ(source_offset, 272);
}

loom_amdgpu_fragment_memory_packet_plan_t Packet(
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint16_t register_index,
    uint16_t result_register_count, uint16_t packet_register_count) {
  loom_amdgpu_fragment_memory_packet_plan_t packet = {
      .descriptor_ref = descriptor_ref,
      .register_index = register_index,
      .result_register_count = result_register_count,
      .packet_register_count = packet_register_count};
  return packet;
}

TEST(AmdgpuFragmentMemoryPacketTest, NativePacketIsOneContiguousAccess) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan.payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  plan.address_layout.payload_elements_per_register = 2;
  plan.address_layout.payload_registers_per_element = 1;
  const auto packet = Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64, 3, 2, 2);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            1);
  EXPECT_EQ(accesses[0].descriptor_ref, LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64);
  EXPECT_EQ(accesses[0].register_index, 3);
  EXPECT_EQ(accesses[0].element_index, 0);
  EXPECT_EQ(accesses[0].element_count, 4);
  EXPECT_EQ(accesses[0].flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_NONE);
}

TEST(AmdgpuFragmentMemoryPacketTest, PackedB16UsesSeparateElementAddresses) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16;
  plan.payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  plan.packed_b16_high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI;
  const auto packet =
      Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16, 4, 1, 1);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            2);
  EXPECT_EQ(accesses[0].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16);
  EXPECT_EQ(accesses[0].register_index, 4);
  EXPECT_EQ(accesses[0].element_index, 0);
  EXPECT_EQ(accesses[0].element_count, 1);
  EXPECT_EQ(accesses[1].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI);
  EXPECT_EQ(accesses[1].register_index, 4);
  EXPECT_EQ(accesses[1].element_index, 1);
  EXPECT_EQ(accesses[1].element_count, 1);
}

TEST(AmdgpuFragmentMemoryPacketTest,
     PackedResultLoadUsesOneAddressPerResultRegister) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan.payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const auto packet = Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16, 5, 2, 1);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            2);
  for (uint16_t i = 0; i < 2; ++i) {
    EXPECT_EQ(accesses[i].descriptor_ref,
              LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16);
    EXPECT_EQ(accesses[i].register_index, 5 + i);
    EXPECT_EQ(accesses[i].element_index, 0);
    EXPECT_EQ(accesses[i].element_count, 1);
  }
}

TEST(AmdgpuFragmentMemoryPacketTest, Fp8DecodeRetainsPhysicalLoadWidth) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan.payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16;
  const auto packet = Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32, 2, 2, 1);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            1);
  EXPECT_EQ(accesses[0].register_index, 2);
  EXPECT_EQ(accesses[0].element_index, 0);
  EXPECT_EQ(accesses[0].element_count, 4);
}

TEST(AmdgpuFragmentMemoryPacketTest, ScalarB16PacketAddressesOneElement) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16;
  plan.payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  plan.address_layout.payload_elements_per_register = 2;
  plan.address_layout.payload_registers_per_element = 1;
  const auto packet = Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16, 3, 1, 1);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            1);
  EXPECT_EQ(accesses[0].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16);
  EXPECT_EQ(accesses[0].register_index, 3);
  EXPECT_EQ(accesses[0].element_index, 0);
  EXPECT_EQ(accesses[0].element_count, 1);
}

TEST(AmdgpuFragmentMemoryPacketTest,
     DirectNarrowedStoreAddressesOneElementPerResultRegister) {
  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan.payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
  plan.address_layout.payload_elements_per_register = 2;
  plan.address_layout.payload_registers_per_element = 1;
  const auto packet = Packet(LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B64, 4, 4, 2);

  loom_amdgpu_fragment_memory_issued_access_t
      accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] = {};
  ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(&plan, &packet,
                                                              accesses),
            1);
  EXPECT_EQ(accesses[0].descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B64);
  EXPECT_EQ(accesses[0].register_index, 4);
  EXPECT_EQ(accesses[0].element_index, 0);
  EXPECT_EQ(accesses[0].element_count, 4);
}

TEST(AmdgpuFragmentMemoryPacketTest,
     Gfx950DppPublicationRestrictsObservationToPublishers) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(IREE_SV("gfx950"));
  ASSERT_NE(processor, nullptr);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(
          &registry.registry, processor->properties.descriptor_set.key);
  ASSERT_NE(descriptor_set, nullptr);

  const loom_amdgpu_matrix_contract_descriptor_t* contract = nullptr;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* candidate =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (iree_string_view_equal(candidate->name,
                               IREE_SV("mfma.f32.16x16x16.bf16.1k"))) {
      contract = candidate;
      break;
    }
  }
  ASSERT_NE(contract, nullptr);
  const loom_amdgpu_matrix_result_representation_t* representation =
      loom_amdgpu_matrix_result_representation_at(
          contract->realization.canonical_result_representation_id);
  ASSERT_NE(representation, nullptr);
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(
          static_cast<loom_amdgpu_matrix_fragment_layout_kind_t>(
              representation->fragment_layout_kind));
  ASSERT_NE(layout, nullptr);

  loom_low_source_memory_axis_byte_stride_t axis_strides[2] = {};
  axis_strides[0].kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC;
  axis_strides[0].static_byte_coefficient = 32;
  axis_strides[1].kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC;
  axis_strides[1].static_byte_coefficient = 2;
  loom_amdgpu_fragment_memory_address_layout_t address_layout = {};
  loom_amdgpu_fragment_memory_runtime_axis_t runtime_axes[3] = {};
  ASSERT_TRUE(loom_amdgpu_fragment_memory_compile_address_layout(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, &layout->result, representation->flags,
      2, axis_strides, &address_layout, runtime_axes, nullptr));
  const uint32_t static_axis_byte_strides[2] = {32, 2};
  const loom_amdgpu_fragment_memory_publication_query_t query = {
      .descriptor_set = descriptor_set,
      .layout = layout,
      .address_layout = &address_layout,
      .runtime_axes = runtime_axes,
      .static_axis_byte_strides = static_axis_byte_strides,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      .role = LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      .representation_flags = representation->flags,
      .payload_form =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16,
      .register_count = layout->result.register_count,
      .element_byte_count = 2,
      .view_rank = 2,
      .source_flags = LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_NONE,
  };
  loom_amdgpu_fragment_memory_publication_choice_t choice = {};
  ASSERT_TRUE(loom_amdgpu_fragment_memory_select_publication(&query, &choice));
  ASSERT_EQ(choice.strategy,
            LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE);

  loom_amdgpu_fragment_memory_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
  plan.role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
  plan.source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  plan.view_rank = 2;
  plan.representation_flags = representation->flags;
  plan.register_count = layout->result.register_count;
  plan.element_byte_count = 2;
  plan.address_layout = address_layout;
  plan.payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
  plan.packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan.static_axis_byte_strides[0] = 32;
  plan.static_axis_byte_strides[1] = 2;
  ASSERT_TRUE(loom_amdgpu_fragment_memory_plan_packets(
      descriptor_set, layout, &choice, &plan, nullptr));
  ASSERT_EQ(plan.packet_count, 4);
  for (uint16_t i = 0; i < plan.packet_count; ++i) {
    loom_amdgpu_fragment_memory_issued_access_t
        accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET] =
            {};
    ASSERT_EQ(loom_amdgpu_fragment_memory_query_issued_accesses(
                  &plan, &plan.packets[i], accesses),
              1);
    EXPECT_EQ(accesses[0].descriptor_ref,
              LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32);
    EXPECT_EQ(accesses[0].register_index, i);
    EXPECT_EQ(accesses[0].element_index, 0);
    EXPECT_EQ(accesses[0].element_count, 2);
    EXPECT_EQ(accesses[0].flags,
              LOOM_AMDGPU_FRAGMENT_MEMORY_ISSUED_ACCESS_FLAG_PUBLISHERS_ONLY);
  }
}

}  // namespace
