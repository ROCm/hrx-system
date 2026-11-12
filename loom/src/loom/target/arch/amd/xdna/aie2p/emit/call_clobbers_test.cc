// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/call_clobbers.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

namespace loom {
namespace {

class Aie2pCallClobbersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_aie2p_physical_register_id_t Register(iree_string_view_t name) {
    const auto value = loom_aie2p_machine_find_physical_register(name);
    EXPECT_NE(value, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
    return value;
  }

  static void AddWrite(iree_string_view_t name,
                       loom_aie2p_register_unit_set_t* writes) {
    const auto physical_register = Register(name);
    loom_aie2p_physical_register_info_t info;
    ASSERT_TRUE(
        loom_aie2p_machine_query_physical_register(physical_register, &info));
    for (uint8_t i = 0; i < info.atomic_unit_count; ++i) {
      const auto unit = loom_aie2p_machine_physical_register_atomic_unit(
          physical_register, i);
      writes->words[unit / 64] |= UINT64_C(1) << (unit % 64);
    }
  }

  static bool Contains(loom_low_call_clobber_list_t clobbers,
                       iree_string_view_t name) {
    const auto physical_register = Register(name);
    for (iree_host_size_t i = 0; i < clobbers.count; ++i) {
      if (clobbers.values[i].location == physical_register) {
        return true;
      }
    }
    return false;
  }

  static const loom_low_call_contract_t* Query(void* user_data,
                                               loom_symbol_ref_t symbol) {
    return &static_cast<const loom_low_call_contract_t*>(
        user_data)[symbol.symbol_id];
  }

  // Backing storage shared by the composed contracts in one test.
  iree_arena_block_pool_t pool_;
  // Owner of immutable contract rows.
  iree_arena_allocator_t arena_;
};

TEST_F(Aie2pCallClobbersTest, CombinesNativeWritesTransportAndRestoredShadow) {
  loom_aie2p_register_unit_set_t writes = {};
  AddWrite(IREE_SV("r4"), &writes);
  AddWrite(IREE_SV("r23"), &writes);
  const loom_low_allocation_abi_location_t argument = {
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
      .location_base = Register(IREE_SV("r0")),
  };
  const loom_low_allocation_abi_location_t result = {
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_EP,
      .location_base = Register(IREE_SV("p2")),
  };
  const loom_low_call_contract_t carriers = {
      .arguments = &argument,
      .argument_count = 1,
      .results = &result,
      .result_count = 1,
  };
  loom_low_call_clobber_list_t clobbers;
  IREE_ASSERT_OK(loom_aie2p_call_clobbers_build(
      writes, Register(IREE_SV("r23")), &carriers, {}, {}, &arena_, &clobbers));
  EXPECT_TRUE(Contains(clobbers, IREE_SV("r4")));
  EXPECT_TRUE(Contains(clobbers, IREE_SV("r0")));
  EXPECT_TRUE(Contains(clobbers, IREE_SV("p2")));
  EXPECT_FALSE(Contains(clobbers, IREE_SV("r23")));
  EXPECT_FALSE(Contains(clobbers, IREE_SV("r5")));
}

TEST_F(Aie2pCallClobbersTest, DiamondKeepsCalleeViewsWithoutAliasExpansion) {
  // Each pair names one written view and an adjacent untouched view. Scalar
  // pairs, vector halves, and FIFO positions must retain the same precision
  // through two generations of callers that have no writes of their own.
  const iree_string_view_t written[] = {
      IREE_SV("r26"),
      IREE_SV("wl0"),
      IREE_SV("r24"),
  };
  const iree_string_view_t untouched[] = {
      IREE_SV("r27"),
      IREE_SV("wh0"),
      IREE_SV("r25"),
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(written); ++i) {
    SCOPED_TRACE(i);
    loom_aie2p_register_unit_set_t writes = {};
    AddWrite(written[i], &writes);
    const loom_low_call_contract_t carriers = {};
    loom_low_call_contract_t contracts[2] = {};
    IREE_ASSERT_OK(loom_aie2p_call_clobbers_build(
        writes, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID, &carriers, {}, {},
        &arena_, &contracts[0].clobbers));
    const loom_symbol_ref_t callees[] = {{.symbol_id = 0}, {.symbol_id = 1}};
    const loom_low_call_contract_provider_t provider = {
        .query = Query,
        .user_data = contracts,
    };
    IREE_ASSERT_OK(loom_aie2p_call_clobbers_build(
        {}, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID, &carriers,
        loom_make_symbol_ref_array(callees, 1), provider, &arena_,
        &contracts[1].clobbers));
    loom_low_call_clobber_list_t root;
    IREE_ASSERT_OK(loom_aie2p_call_clobbers_build(
        {}, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID, &carriers,
        loom_make_symbol_ref_array(callees, 2), provider, &arena_, &root));
    ASSERT_EQ(root.count, contracts[0].clobbers.count);
    EXPECT_TRUE(Contains(root, written[i]));
    EXPECT_FALSE(Contains(root, untouched[i]));
    for (iree_host_size_t j = 0; j < root.count; ++j) {
      EXPECT_EQ(root.values[j].register_class,
                contracts[0].clobbers.values[j].register_class);
      EXPECT_EQ(root.values[j].location,
                contracts[0].clobbers.values[j].location);
      EXPECT_EQ(root.values[j].count, 1u);
    }
  }
}

TEST_F(Aie2pCallClobbersTest, EmptyBoundaryHasNoClobbers) {
  const loom_low_call_contract_t carriers = {};
  loom_low_call_clobber_list_t clobbers;
  IREE_ASSERT_OK(loom_aie2p_call_clobbers_build(
      {}, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID, &carriers, {}, {}, &arena_,
      &clobbers));
  EXPECT_EQ(clobbers.count, 0u);
}

}  // namespace
}  // namespace loom
