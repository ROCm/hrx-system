// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/codegen/low/testing/source_workload_pipeline.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/registry.h"
#include "loom/pass/types.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SourceLoweringStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringStressTest, GeneratedSupportedSourceLowersAndPacketizes) {
  static constexpr uint32_t kScales[] = {1, 2, 4};
  static constexpr uint64_t kSeedCountPerScale = 32;
  loom_low_source_workload_pipeline_counters_t aggregate = {};
  const loom_low_source_workload_pipeline_options_t pipeline_options = {
      /*.pass_registry=*/loom_pass_builtin_registry(),
      /*.descriptor_registry=*/&descriptor_registry_.registry,
      /*.policy_registry=*/&policy_registry_,
      /*.schedule_strategy=*/LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
  };

  for (uint32_t scale : kScales) {
    for (uint64_t seed = 0; seed < kSeedCountPerScale; ++seed) {
      SCOPED_TRACE(::testing::Message()
                   << "scale=" << scale << " seed=" << seed);
      loom_module_t* module_raw = nullptr;
      loom_low_source_workload_config_t workload_config =
          loom_low_source_workload_config_make(scale);
      IREE_ASSERT_OK(loom_low_source_workload_generate_seeded_module(
          seed, &workload_config, &context_, &block_pool_, &module_raw));
      ModulePtr module(module_raw);

      loom_low_source_workload_pipeline_counters_t counters = {};
      bool pipeline_accepted = false;
      IREE_ASSERT_OK(loom_low_source_workload_run_pipeline(
          module.get(), &pipeline_options, &block_pool_, &counters,
          &pipeline_accepted));
      ASSERT_TRUE(pipeline_accepted);
      EXPECT_EQ(counters.lower_error_count, 0u);
      loom_low_source_workload_counts_accumulate(&aggregate.source_counts,
                                                 &counters.source_counts);
      aggregate.low_descriptor_op_count += counters.low_descriptor_op_count;
      aggregate.schedule_node_count += counters.schedule_node_count;
      aggregate.allocation_assignment_count +=
          counters.allocation_assignment_count;
      aggregate.allocation_check_count += counters.allocation_check_count;
    }
  }

  EXPECT_GT(aggregate.source_counts.scalar_integer_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.scalar_float_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.scalar_constant_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_integer_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_float_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_reduce_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_float_reduce_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_dot_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_extract_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_shuffle_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_cmpi_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_select_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_load_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_float_load_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_dynamic_load_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_store_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_float_store_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.vector_dynamic_store_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.index_madd_op_count, 0u);
  EXPECT_GT(aggregate.source_counts.cfg_cond_branch_count, 0u);
  EXPECT_GT(aggregate.source_counts.cfg_branch_count, 0u);
  EXPECT_GT(aggregate.low_descriptor_op_count, 0u);
  EXPECT_GT(aggregate.schedule_node_count, 0u);
  EXPECT_GT(aggregate.allocation_assignment_count, 0u);
  EXPECT_EQ(aggregate.allocation_check_count,
            IREE_ARRAYSIZE(kScales) * kSeedCountPerScale);
}

TEST_F(SourceLoweringStressTest, CopiedDestructiveResultKeepsStorageIdentity) {
  // Generates a materialized copy, destructive result, and coalesced copy
  // whose storage reservations overlap without overwriting a live value.
  static constexpr uint8_t kInput[] = {0x29, 0xca, 0x4a};
  const loom_low_source_workload_config_t workload_config =
      loom_low_source_workload_config_make(4);
  loom_module_t* module_raw = nullptr;
  IREE_ASSERT_OK(loom_low_source_workload_generate_fuzz_module(
      kInput, IREE_ARRAYSIZE(kInput), &workload_config, &context_, &block_pool_,
      &module_raw));
  ModulePtr module(module_raw);
  const loom_low_source_workload_pipeline_options_t pipeline_options = {
      /*.pass_registry=*/loom_pass_builtin_registry(),
      /*.descriptor_registry=*/&descriptor_registry_.registry,
      /*.policy_registry=*/&policy_registry_,
      /*.schedule_strategy=*/LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
  };
  loom_low_source_workload_pipeline_counters_t counters = {};
  bool pipeline_accepted = false;
  IREE_ASSERT_OK(loom_low_source_workload_run_pipeline(
      module.get(), &pipeline_options, &block_pool_, &counters,
      &pipeline_accepted));
  ASSERT_TRUE(pipeline_accepted);
  EXPECT_EQ(counters.allocation_check_count, 1u);
}

TEST_F(SourceLoweringStressTest, PreparationDiagnosticStopsBeforeAllocation) {
  // Replace a preparation pass, not the pipeline being tested. Diagnostics
  // and infrastructure failures are separate pass results: emitting an error
  // can succeed even though the function is not ready for the next stage.
  const loom_pass_registry_t* builtin = loom_pass_builtin_registry();
  std::vector<loom_pass_descriptor_t> descriptors(
      builtin->descriptors, builtin->descriptors + builtin->descriptor_count);
  bool replaced = false;
  for (auto& descriptor : descriptors) {
    if (!iree_string_view_equal(descriptor.key, IREE_SV("canonicalize"))) {
      continue;
    }
    descriptor.function_run = [](loom_pass_t* pass, loom_module_t* module,
                                 loom_func_like_t function) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("canonicalize")),
          loom_param_string(IREE_SV("function")),
          loom_param_string(IREE_SV("generated")),
          loom_param_string(IREE_SV("packetization preparation")),
      };
      loom_diagnostic_emission_t emission = {};
      emission.op = function.op;
      emission.error = LOOM_ERR_STRUCTURE_028;
      emission.params = params;
      emission.param_count = IREE_ARRAYSIZE(params);
      return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
    };
    replaced = true;
  }
  ASSERT_TRUE(replaced);
  const loom_pass_registry_t registry = {descriptors.data(),
                                         descriptors.size()};
  const loom_low_source_workload_pipeline_options_t pipeline_options = {
      &registry, &descriptor_registry_.registry, &policy_registry_,
      LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE};
  const loom_low_source_workload_config_t workload_config =
      loom_low_source_workload_config_make(1);
  loom_module_t* module_raw = nullptr;
  IREE_ASSERT_OK(loom_low_source_workload_generate_seeded_module(
      0, &workload_config, &context_, &block_pool_, &module_raw));
  ModulePtr module(module_raw);
  loom_low_source_workload_pipeline_counters_t counters = {};
  bool pipeline_accepted = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_source_workload_run_pipeline(
                            module.get(), &pipeline_options, &block_pool_,
                            &counters, &pipeline_accepted));
  EXPECT_FALSE(pipeline_accepted);
  EXPECT_EQ(counters.lower_error_count, 0u);
  EXPECT_EQ(counters.low_descriptor_op_count, 0u);
  EXPECT_EQ(counters.allocation_check_count, 0u);
}

}  // namespace
}  // namespace loom
