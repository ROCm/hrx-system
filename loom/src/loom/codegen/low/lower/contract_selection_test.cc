// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/contract_selection.h"

#include <cstdint>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

namespace {

constexpr uint16_t kCaseStart = 100;
constexpr loom_target_contract_op_entry_t kEntry = {kCaseStart, 64};

class ContractSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t scalar_op_count = 0;
    const loom_op_vtable_t* const* scalar_vtables =
        loom_scalar_dialect_vtables(&scalar_op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, scalar_vtables,
        static_cast<uint16_t>(scalar_op_count)));
    iree_host_size_t vector_op_count = 0;
    const loom_op_vtable_t* const* vector_vtables =
        loom_vector_dialect_vtables(&vector_op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_VECTOR, vector_vtables,
        static_cast<uint16_t>(vector_op_count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("contract_selection_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);

    loom_op_t* lhs_op = nullptr;
    loom_op_t* rhs_op = nullptr;
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_ASSERT_OK(loom_scalar_constant_build(
        &builder_, loom_attr_i64(1), i32_type, LOOM_LOCATION_UNKNOWN, &lhs_op));
    IREE_ASSERT_OK(loom_scalar_constant_build(
        &builder_, loom_attr_i64(2), i32_type, LOOM_LOCATION_UNKNOWN, &rhs_op));
    IREE_ASSERT_OK(loom_scalar_cmpi_build(
        &builder_, LOOM_SCALAR_CMPI_PREDICATE_EQ, loom_op_results(lhs_op)[0],
        loom_op_results(rhs_op)[0], LOOM_LOCATION_UNKNOWN, &compare_op_));
    IREE_ASSERT_OK(loom_vector_splat_build(
        &builder_, loom_op_results(lhs_op)[0],
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(4), 0),
        LOOM_LOCATION_UNKNOWN, &vector_op_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::vector<uint16_t> Iterate(
      const uint32_t* selection_data,
      loom_low_lower_contract_case_iteration_mode_t mode,
      const loom_op_t* source_op = nullptr,
      loom_target_contract_vector_lane_projection_t vector_lane_projection =
          {}) {
    const loom_target_contract_index_t index = {
        0, 0, nullptr, 0, nullptr, 0, 0, nullptr, selection_data, nullptr,
    };
    loom_low_lower_contract_case_iterator_t iterator;
    loom_low_lower_contract_case_iterator_initialize(
        module_, &index, kEntry, source_op != nullptr ? source_op : compare_op_,
        vector_lane_projection, mode, &iterator);
    std::vector<uint16_t> cases;
    uint16_t case_index = UINT16_MAX;
    while (loom_low_lower_contract_case_iterator_next(&iterator, &case_index)) {
      cases.push_back(case_index);
    }
    return cases;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_op_t* compare_op_ = nullptr;
  loom_op_t* vector_op_ = nullptr;
};

TEST_F(ContractSelectionTest, SelectsExactOperandTypeList) {
  constexpr uint32_t kSelectionData[] = {
      2u | (1u << 16),
      64u,
      2u,
      0u,
      1u | (static_cast<uint32_t>(LOOM_OP_SCALAR_ADDI) << 16),
      2u,
      1u,
      1u | (static_cast<uint32_t>(LOOM_OP_SCALAR_CMPI) << 16),
      (1u << 28) | (LOOM_SCALAR_TYPE_I32 << 16),
      1u | (2u << 16),
      7u,
      4u | (9u << 16),
  };

  EXPECT_EQ(Iterate(kSelectionData,
                    LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES),
            (std::vector<uint16_t>{104, 109}));

  const auto unindexed_cases =
      Iterate(kSelectionData, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
              vector_op_);
  EXPECT_EQ(unindexed_cases.size(), 64u);
  EXPECT_EQ(unindexed_cases.front(), kCaseStart);
  EXPECT_EQ(unindexed_cases.back(), kCaseStart + 63);
}

TEST_F(ContractSelectionTest, SelectsExactResultTypeList) {
  constexpr uint32_t kSelectionData[] = {
      1u | (1u << 16),
      64u,
      3u,
      1u,
      1u | (static_cast<uint32_t>(LOOM_OP_SCALAR_CMPI) << 16),
      (1u << 28) | (LOOM_SCALAR_TYPE_I1 << 16),
      1u | (2u << 16),
      7u,
      4u | (9u << 16),
  };

  EXPECT_EQ(Iterate(kSelectionData,
                    LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES),
            (std::vector<uint16_t>{104, 109}));
}

TEST_F(ContractSelectionTest, SelectsExactVectorTypeList) {
  constexpr uint32_t kSelectionData[] = {
      1u | (1u << 16),
      64u,
      3u,
      1u,
      1u | (static_cast<uint32_t>(LOOM_OP_VECTOR_SPLAT) << 16),
      (2u << 28) | (LOOM_SCALAR_TYPE_I32 << 16) | 4u,
      1u | (2u << 16),
      7u,
      4u | (9u << 16),
  };

  EXPECT_EQ(
      Iterate(kSelectionData, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
              vector_op_),
      (std::vector<uint16_t>{104, 109}));
}

TEST_F(ContractSelectionTest,
       ProjectsOperandAndResultTypeKeysWithoutMutatingIr) {
  const loom_value_id_t vector_value = loom_vector_splat_result(vector_op_);
  const loom_type_t authored_type =
      loom_module_value_type(module_, vector_value);
  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(
      &builder_, /*instance_flags=*/0, vector_value, vector_value,
      authored_type, LOOM_LOCATION_UNKNOWN, &add_op));

  for (const uint32_t selector : {2u, 3u}) {
    SCOPED_TRACE(selector == 2u ? "operand type" : "result type");
    const uint32_t selection_data[] = {
        1u | (2u << 16),
        64u,
        selector,
        2u,
        static_cast<uint32_t>(LOOM_OP_VECTOR_ADDI) << 16,
        (2u << 28) | (LOOM_SCALAR_TYPE_I32 << 16) | 2u,
        1u << 16,
        (2u << 28) | (LOOM_SCALAR_TYPE_I32 << 16) | 4u,
        1u | (1u << 16),
        7u,
        9u,
    };

    EXPECT_EQ(Iterate(selection_data,
                      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add_op,
                      {/*source_lane_count=*/4, /*projected_lane_count=*/2}),
              (std::vector<uint16_t>{107}));
    EXPECT_EQ(Iterate(selection_data,
                      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add_op,
                      {/*source_lane_count=*/8, /*projected_lane_count=*/2}),
              (std::vector<uint16_t>{109}));
    EXPECT_EQ(
        Iterate(selection_data,
                LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add_op),
        (std::vector<uint16_t>{109}));
    EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, vector_value),
                                authored_type));
    EXPECT_TRUE(loom_type_equal(
        loom_module_value_type(module_, loom_vector_addi_result(add_op)),
        authored_type));
  }
}

TEST_F(ContractSelectionTest, SelectsExactEnumBitmap) {
  constexpr uint32_t kSelectionData[] = {
      1u | (1u << 16),
      64u,
      1u,
      1u,
      1u | (static_cast<uint32_t>(LOOM_OP_SCALAR_CMPI) << 16),
      LOOM_SCALAR_CMPI_PREDICATE_EQ,
      1u | (0x8002u << 16),
      5u,
      (1u << 0) | (1u << 3),
      1u << 1,
  };

  EXPECT_EQ(Iterate(kSelectionData,
                    LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES),
            (std::vector<uint16_t>{100, 103, 133}));
}

TEST_F(ContractSelectionTest, SelectsProjectedOperandAndResultTypes) {
  const loom_value_id_t value = loom_op_results(vector_op_)[0];
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(
      &builder_, /*instance_flags=*/0, value, value,
      loom_module_value_type(module_, value), LOOM_LOCATION_UNKNOWN, &add));
  for (uint32_t selector : {2u, 3u}) {
    const uint32_t selection_data[] = {
        1u | (2u << 16),
        64u,
        selector,
        2u,
        1u | (static_cast<uint32_t>(LOOM_OP_VECTOR_ADDI) << 16),
        (2u << 28) | (LOOM_SCALAR_TYPE_I32 << 16) | 2u,
        1u | (1u << 16),
        (2u << 28) | (LOOM_SCALAR_TYPE_I32 << 16) | 4u,
        2u | (1u << 16),
        7u,
        4u,
        9u,
    };
    EXPECT_EQ(Iterate(selection_data,
                      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add,
                      {/*source_lane_count=*/4, /*projected_lane_count=*/2}),
              (std::vector<uint16_t>{104}));
    EXPECT_EQ(Iterate(selection_data,
                      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add,
                      {/*source_lane_count=*/8, /*projected_lane_count=*/2}),
              (std::vector<uint16_t>{109}));
    EXPECT_EQ(Iterate(selection_data,
                      LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES, add),
              (std::vector<uint16_t>{109}));
  }
}

TEST_F(ContractSelectionTest, UsesFallbackWhenExactKeyIsAbsent) {
  constexpr uint32_t kSelectionData[] = {
      1u | (1u << 16),
      64u,
      1u,
      1u,
      1u | (static_cast<uint32_t>(LOOM_OP_SCALAR_CMPI) << 16),
      LOOM_SCALAR_CMPI_PREDICATE_NE,
      1u | (1u << 16),
      5u,
      7u,
  };

  EXPECT_EQ(Iterate(kSelectionData,
                    LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES),
            (std::vector<uint16_t>{105}));
}

TEST_F(ContractSelectionTest, AllModePreservesCompleteComposedOrder) {
  const auto cases =
      Iterate(nullptr, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL);
  EXPECT_EQ(cases.size(), 64u);
  EXPECT_EQ(cases.front(), kCaseStart);
  EXPECT_EQ(cases.back(), kCaseStart + 63);
}

}  // namespace
