// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class LivenessSelectionTest : public ::testing::TestWithParam<
                                  std::tuple<loom_storage_space_t, uint16_t>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("selection"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_local_value_domain_release(&domain_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Build(loom_storage_space_t space, uint16_t block_count) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder, IREE_SV("addresses"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_string_id_t contract = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder, IREE_SV("test.low.core"), &contract));
    const auto register_type = loom_low_register_type(1, 0, 1);
    const loom_type_t result_types[] = {register_type, register_type};
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, contract, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), {0, symbol},
        nullptr, 0, result_types, 2, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function));
    auto* body = loom_low_func_def_body(function);
    body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_set_block(&builder, loom_region_entry_block(body));
    loom_builder_enter_region(&builder, function, body);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_low_storage_reserve_build(
        &builder, 16, 8, loom_type_storage(space), LOOM_LOCATION_UNKNOWN, &op));
    storage_ = loom_low_storage_reserve_storage(op);
    IREE_ASSERT_OK(loom_low_storage_address_build(
        &builder, storage_, 0, register_type, LOOM_LOCATION_UNKNOWN, &op));
    first_address_ = loom_low_storage_address_result(op);
    for (uint16_t i = 1; i < block_count; ++i) {
      loom_block_t* next = nullptr;
      IREE_ASSERT_OK(loom_region_append_block(module_, body, &next));
      IREE_ASSERT_OK(loom_low_br_build(&builder, next, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_set_block(&builder, next);
    }
    IREE_ASSERT_OK(loom_low_storage_view_build(&builder, storage_, 8, 8,
                                               loom_type_storage(space),
                                               LOOM_LOCATION_UNKNOWN, &op));
    view_ = loom_low_storage_view_result(op);
    IREE_ASSERT_OK(loom_low_storage_address_build(
        &builder, view_, 0, register_type, LOOM_LOCATION_UNKNOWN, &op));
    second_address_ = loom_low_storage_address_result(op);
    const loom_value_id_t results[] = {first_address_, second_address_};
    IREE_ASSERT_OK(loom_low_return_build(&builder, results, 2,
                                         LOOM_LOCATION_UNKNOWN, &op));
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, body, &arena_, &domain_));
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, body, &arena_, &graph_));
  }

  // Shared backing for the module and the two immutable analysis results.
  iree_arena_block_pool_t pool_ = {};
  // Analysis lifetime, independent of module construction storage.
  iree_arena_allocator_t arena_ = {};
  // Low dialect definitions used to construct the analyzed CFG.
  loom_context_t context_ = {};
  // Owns the function and its SSA values.
  loom_module_t* module_ = nullptr;
  // Complete ordinal domain shared by both selections.
  loom_local_value_domain_t domain_ = {};
  // The same immutable CFG is consumed by both analyses.
  loom_cfg_graph_t graph_ = {};
  // Frame-owned reservation used in the entry and final blocks.
  loom_value_id_t storage_ = LOOM_VALUE_ID_INVALID;
  // Static subspan used by the final address materialization.
  loom_value_id_t view_ = LOOM_VALUE_ID_INVALID;
  // Runtime address held across all CFG blocks.
  loom_value_id_t first_address_ = LOOM_VALUE_ID_INVALID;
  // Runtime address defined in the final block.
  loom_value_id_t second_address_ = LOOM_VALUE_ID_INVALID;
};

TEST_P(LivenessSelectionTest,
       RegisterProjectionPreservesExactRegisterLifetimes) {
  const auto [space, block_count] = GetParam();
  ASSERT_NO_FATAL_FAILURE(Build(space, block_count));
  loom_liveness_analysis_t all = {};
  loom_liveness_analysis_t registers = {};
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain(
      &domain_, loom_liveness_order_empty(), &arena_, &all));
  loom_liveness_dataflow_t dataflow = {};
  IREE_ASSERT_OK(loom_liveness_dataflow_analyze(
      &domain_, &graph_, LOOM_LIVENESS_VALUES_REGISTERS, &arena_, &dataflow));
  IREE_ASSERT_OK(loom_liveness_analyze_local_value_domain_with_dataflow(
      &domain_, &dataflow, loom_liveness_order_empty(), &arena_, &registers));
  EXPECT_EQ(all.value_ids, registers.value_ids);
  ASSERT_EQ(all.value_count, 4u);
  ASSERT_EQ(registers.value_count, all.value_count);
  ASSERT_EQ(all.interval_count, 4u);
  ASSERT_EQ(registers.interval_count, 2u);
  EXPECT_EQ(all.segment_count, 2u * block_count + 2u);
  EXPECT_EQ(registers.segment_count, block_count + 1u);
  EXPECT_EQ(registers.operation_count, all.operation_count);
  EXPECT_EQ(all.operation_use_count, 5u);
  EXPECT_EQ(registers.operation_use_count, 2u);
  for (uint16_t i = 0; i < block_count; ++i) {
    EXPECT_EQ(all.blocks[i].live_in_count, i == 0 ? 0u : 2u);
    EXPECT_EQ(registers.blocks[i].live_in_count, i == 0 ? 0u : 1u);
    EXPECT_EQ(all.blocks[i].live_out_count, i + 1 == block_count ? 0u : 2u);
    EXPECT_EQ(registers.blocks[i].live_out_count,
              i + 1 == block_count ? 0u : 1u);
  }
  for (auto value : {storage_, view_}) {
    const auto ordinal = loom_local_value_domain_ordinal(&domain_, value);
    EXPECT_EQ(loom_liveness_interval_for_value_ordinal(&registers, ordinal),
              nullptr);
    EXPECT_EQ(registers.value_segment_ranges[ordinal].count, 0u);
  }
  for (auto value : {first_address_, second_address_}) {
    const auto ordinal = loom_local_value_domain_ordinal(&domain_, value);
    const auto* before =
        loom_liveness_interval_for_value_ordinal(&all, ordinal);
    const auto* after =
        loom_liveness_interval_for_value_ordinal(&registers, ordinal);
    ASSERT_NE(before, nullptr);
    ASSERT_NE(after, nullptr);
    EXPECT_EQ(after->start_point, before->start_point);
    EXPECT_EQ(after->end_point, before->end_point);
    const auto original = all.value_segment_ranges[ordinal];
    const auto selected = registers.value_segment_ranges[ordinal];
    ASSERT_EQ(selected.count, original.count);
    for (uint32_t i = 0; i < selected.count; ++i) {
      EXPECT_EQ(registers.segments[selected.start + i].start_point,
                all.segments[original.start + i].start_point);
      EXPECT_EQ(registers.segments[selected.start + i].end_point,
                all.segments[original.start + i].end_point);
    }
  }
  ASSERT_EQ(registers.pressure_summary_count, 1u);
  EXPECT_EQ(registers.pressure_summaries[0].peak_live_units, 2u);
  for (uint32_t i = 0; i < registers.operation_use_count; ++i) {
    const auto ordinal = loom_liveness_operation_use_ordinal(&registers, i);
    EXPECT_TRUE(loom_type_is_register(
        loom_module_value_type(module_, registers.value_ids[ordinal])));
    const auto segment =
        loom_liveness_operation_use_segment_index(&registers, i);
    EXPECT_LT(segment, registers.segment_count);
  }
}

INSTANTIATE_TEST_SUITE_P(
    StorageSpacesAndCfgSizes, LivenessSelectionTest,
    ::testing::Combine(::testing::Values(LOOM_STORAGE_SPACE_STACK,
                                         LOOM_STORAGE_SPACE_SCRATCH,
                                         LOOM_STORAGE_SPACE_PRIVATE,
                                         LOOM_STORAGE_SPACE_WORKGROUP),
                       ::testing::Values(1, 3, 129)));

}  // namespace
}  // namespace loom
