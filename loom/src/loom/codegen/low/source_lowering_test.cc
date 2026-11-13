// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "loom/codegen/low/source_lowering_test_util.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/pass/registry.h"
#include "loom/pass/types.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SourceLoweringTest : public ::loom::testing::SourceLoweringTestBase {
 protected:
  void ExpectFuzzWorkloadPasses(const uint8_t* input, size_t input_size,
                                uint32_t scale) {
    const loom_low_source_workload_config_t workload_config =
        loom_low_source_workload_config_make(scale);
    loom_module_t* module_raw = nullptr;
    IREE_ASSERT_OK(loom_low_source_workload_generate_fuzz_module(
        input, input_size, &workload_config, &context_, &block_pool_,
        &module_raw));
    ModulePtr module(module_raw);
    const loom_low_source_workload_pipeline_options_t pipeline_options = {
        .pass_registry = loom_pass_builtin_registry(),
        .descriptor_registry = &descriptor_registry_.registry,
        .policy_registry = &policy_registry_,
        .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
    };
    loom_low_source_workload_pipeline_counters_t counters = {};
    bool pipeline_accepted = false;
    IREE_ASSERT_OK(loom_low_source_workload_run_pipeline(
        module.get(), &pipeline_options, &block_pool_, &counters,
        &pipeline_accepted));
    ASSERT_TRUE(pipeline_accepted);
    EXPECT_EQ(counters.allocation_check_count, 1u);
  }
};

TEST_F(SourceLoweringTest, CopiedDestructiveResultKeepsStorageIdentity) {
  // Generates a materialized copy, destructive result, and coalesced copy
  // whose storage reservations overlap without overwriting a live value.
  static constexpr uint8_t kInput[] = {0x29, 0xca, 0x4a};
  ExpectFuzzWorkloadPasses(kInput, IREE_ARRAYSIZE(kInput), /*scale=*/4);
}

TEST_F(SourceLoweringTest, RequiredRegisterAssignmentPassesAllocationCheck) {
  // Generates a target-ID assignment whose immutable register requirement is
  // retained for later victim searches. The independent checker must accept
  // that assignment flag while still checking the assignment's full shape.
  static constexpr uint8_t kInput[] = {0xb9, 0x21};
  ExpectFuzzWorkloadPasses(kInput, IREE_ARRAYSIZE(kInput), /*scale=*/3);
}

TEST_F(SourceLoweringTest, PreparationDiagnosticStopsBeforeAllocation) {
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
      (void)module;
      loom_diagnostic_emission_t emission = {
          .op = function.op,
          .error = LOOM_ERR_EXPECT_001,
      };
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
