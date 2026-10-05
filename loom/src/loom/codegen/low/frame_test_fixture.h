// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SRC_LOOM_CODEGEN_LOW_FRAME_TEST_FIXTURE_H_
#define LOOM_SRC_LOOM_CODEGEN_LOW_FRAME_TEST_FIXTURE_H_

#include <algorithm>
#include <initializer_list>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom::testing {

class LowEmissionFrameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("frame_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr ParseModule() {
    return ParseModule(R"(
low.func.def target<test.low.core> @structural_model() -> (reg<test.i32 x4>) asm {
  %storage = storage {byte_alignment = 16, byte_length = 64} : low.storage<workgroup>
  %address = storage_address %storage : low.storage<workgroup> -> reg<test.ptr>
  %value = test.load.v4i32 %address
  return %value
}
)");
  }

  iree_status_t BuildFrame(
      loom_module_t* module,
      loom_low_schedule_structural_model_list_t structural_models,
      loom_low_emission_frame_t* out_frame,
      loom_low_schedule_strategy_t schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    loom_block_t* module_block = loom_module_block(module);
    IREE_ASSERT_EQ(module_block->op_count, 1);
    return BuildFunctionFrame(module, loom_block_op(module_block, 0),
                              structural_models, out_frame, schedule_strategy);
  }

  iree_status_t BuildFunctionFrame(
      loom_module_t* module, loom_op_t* function,
      loom_low_schedule_structural_model_list_t structural_models,
      loom_low_emission_frame_t* out_frame,
      loom_low_schedule_strategy_t schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    loom_low_emission_frame_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.schedule_structural_models = structural_models;
    options.schedule_strategy = schedule_strategy;
    bool accepted = false;
    iree_status_t status = loom_low_emission_frame_build(
        module, function, &options, &arena_, out_frame, &accepted);
    if (iree_status_is_ok(status)) {
      EXPECT_TRUE(accepted);
    }
    return status;
  }

  const loom_low_schedule_node_t* FindNode(
      const loom_low_emission_frame_t& frame, loom_op_kind_t op_kind) {
    for (iree_host_size_t i = 0; i < frame.schedule.node_count; ++i) {
      if (frame.schedule.nodes[i].op->kind == op_kind) {
        return &frame.schedule.nodes[i];
      }
    }
    return nullptr;
  }

  void ExpectDependencyTiming(
      const loom_low_emission_frame_t& frame, uint32_t producer_node,
      uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
      int32_t minimum_issue_separation_cycles,
      loom_low_schedule_separation_source_t separation_source,
      loom_low_model_quality_t model_quality) {
    const loom_low_schedule_dependency_t* match = nullptr;
    for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies,
                                                i);
      if (dependency->producer_node == producer_node &&
          dependency->consumer_node == consumer_node &&
          dependency->kind == kind) {
        EXPECT_EQ(match, nullptr);
        match = dependency;
      }
    }
    ASSERT_NE(match, nullptr);
    EXPECT_EQ(match->minimum_issue_separation_cycles,
              minimum_issue_separation_cycles);
    EXPECT_EQ(match->separation_source, separation_source);
    EXPECT_EQ(match->model_quality, model_quality);
  }

  void ExpectDependencyPredecessors(
      const loom_low_emission_frame_t& frame,
      loom_low_schedule_dependency_kind_t kind, uint32_t consumer_node,
      std::initializer_list<uint32_t> expected_producer_nodes) {
    std::vector<uint32_t> actual_producer_nodes;
    for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies,
                                                i);
      if (dependency->kind == kind &&
          dependency->consumer_node == consumer_node) {
        actual_producer_nodes.push_back(dependency->producer_node);
      }
    }
    std::sort(actual_producer_nodes.begin(), actual_producer_nodes.end());
    std::vector<uint32_t> sorted_expected_producer_nodes(
        expected_producer_nodes);
    std::sort(sorted_expected_producer_nodes.begin(),
              sorted_expected_producer_nodes.end());
    EXPECT_EQ(actual_producer_nodes, sorted_expected_producer_nodes);
  }

  void ExpectEffectPredecessors(
      const loom_low_emission_frame_t& frame, uint32_t consumer_node,
      std::initializer_list<uint32_t> expected_producer_nodes) {
    ExpectDependencyPredecessors(frame, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
                                 consumer_node, expected_producer_nodes);
  }

  void ExpectStorageLifetimePredecessors(
      const loom_low_emission_frame_t& frame, uint32_t consumer_node,
      std::initializer_list<uint32_t> expected_producer_nodes) {
    std::vector<uint32_t> actual_producer_nodes;
    for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies,
                                                i);
      if (dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE &&
          dependency->consumer_node == consumer_node &&
          dependency->minimum_issue_separation_cycles > 0) {
        actual_producer_nodes.push_back(dependency->producer_node);
      }
    }
    std::sort(actual_producer_nodes.begin(), actual_producer_nodes.end());
    std::vector<uint32_t> sorted_expected_producer_nodes(
        expected_producer_nodes);
    std::sort(sorted_expected_producer_nodes.begin(),
              sorted_expected_producer_nodes.end());
    EXPECT_EQ(actual_producer_nodes, sorted_expected_producer_nodes);
  }

  // Block storage shared by parsed modules and emission frames in each test.
  iree_arena_block_pool_t block_pool_ = {};
  // IR context containing the Low dialect used by fixture source modules.
  loom_context_t context_ = {};
  // Test target descriptors used to resolve Low instruction packets.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Arena owning the frame under test.
  iree_arena_allocator_t arena_ = {};
};

}  // namespace loom::testing

#endif  // LOOM_SRC_LOOM_CODEGEN_LOW_FRAME_TEST_FIXTURE_H_
