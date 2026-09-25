// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/vm/module.h"
#include "loom/target/arch/vm/provider.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/input/input.h"
#include "loom/tooling/target/vm/emission_test_data.h"
#include "loom/tooling/target/vm/native_references_bytecode.h"
#include "loom/transforms/cleanup/configured.h"

namespace {

struct EmissionAllocator {
  // Largest permitted system request, including fixed arena blocks.
  iree_host_size_t maximum_request = IREE_HOST_SIZE_MAX;
  // Zero-based allocation attempt to fail, or the host maximum for no failure.
  iree_host_size_t fail_at = IREE_HOST_SIZE_MAX;
  // Allocation attempts since arming the failure injection point.
  iree_host_size_t attempts = 0;
  // Successful allocations still owned by pools, streams or returned artifacts.
  iree_host_size_t live = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto* state = static_cast<EmissionAllocator*>(self);
    const bool had_allocation = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                                 command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                                *pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      const auto size =
          static_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length;
      if (state->attempts++ == state->fail_at ||
          size > state->maximum_request) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "emission allocation rejected");
      }
    }
    const auto system = iree_allocator_system();
    iree_status_t status = system.ctl(system.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        state->live -= had_allocation;
      } else if (!had_allocation) {
        ++state->live;
      }
    }
    return status;
  }
};

class VMEmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &pool_);
    static const loom_target_provider_t* const providers[] = {
        &loom_vm_target_provider};
    static const auto provider_set =
        loom_target_provider_set_make(providers, 1);
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&provider_set, &environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(
        loom_target_environment_register_context(&environment_, &context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_target_environment_initialize_low_descriptor_registry(
        &environment_, &registry_));
  }

  void TearDown() override {
    loom_compile_pipeline_result_deinitialize(&pipeline_);
    loom_input_module_deinitialize(&input_);
    loom_context_deinitialize(&context_);
    loom_target_environment_deinitialize(&environment_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Prepare(iree_string_view_t source) {
    loom_input_request_t request = {};
    request.source = source;
    request.path = IREE_SV("emission.loom");
    IREE_ASSERT_OK(loom_input_module_load(&loom_input_text_provider, &request,
                                          &context_, &pool_,
                                          iree_allocator_system(), &input_));
    ASSERT_NE(input_.module, nullptr);
    const loom_target_profile_t* profile = nullptr;
    IREE_ASSERT_OK(
        loom_vm_target_provider.select_profile(IREE_SV("core"), &profile));
    std::vector<loom_target_specialization_request_t> specializations;
    for (uint32_t i = 0; i < input_.module->symbols.count; ++i) {
      const auto& symbol = input_.module->symbols.entries[i];
      if (!loom_func_def_isa(symbol.defining_op)) {
        continue;
      }
      const auto function =
          loom_func_like_cast(input_.module, symbol.defining_op);
      if (!loom_func_like_is_exported(function)) {
        continue;
      }
      loom_target_specialization_request_t specialization = {};
      specialization.function_name =
          loom_string_table_get(&input_.module->strings, symbol.name_id);
      specialization.target_profile = profile;
      specializations.push_back(specialization);
    }
    loom_compile_pipeline_options_t options;
    loom_compile_pipeline_options_initialize(&options);
    options.target_environment = &environment_;
    options.low_descriptor_registry = &registry_;
    options.cleanup_pattern_provider_set =
        loom_cleanup_configured_pattern_provider_set();
    options.target_specializations = {specializations.data(),
                                      specializations.size()};
    IREE_ASSERT_OK(
        loom_compile_run_pipeline(input_.module, &options, &pool_, &pipeline_));
    ASSERT_EQ(pipeline_.pass.error_count, 0u);
  }

  iree_status_t Emit(EmissionAllocator* allocations, iree_host_size_t fail_at,
                     bool* out_emitted,
                     loom_target_emit_artifact_t* out_artifact) {
    *out_emitted = false;
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(128 * 1024, allocations->allocator(),
                                     &pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    uint32_t* prefix = nullptr;
    iree_status_t status =
        iree_arena_allocate(&arena, 256, reinterpret_cast<void**>(&prefix));
    if (iree_status_is_ok(status)) {
      std::fill_n(prefix, 64, 0xC011AB1Eu);
      const auto checkpoint = iree_arena_checkpoint_save(&arena);
      allocations->attempts = 0;
      allocations->fail_at = fail_at;
      loom_target_emit_request_t request = {};
      request.target_environment = &environment_;
      request.low_descriptor_registry = &registry_.registry;
      request.module = input_.module;
      request.function_versions = &pipeline_.function_versions.list;
      request.scratch_arena = &arena;
      request.allocator = allocations->allocator();
      status = loom_vm_module_emit(&request, out_emitted, out_artifact);
      EXPECT_EQ(arena.used_allocation_size, checkpoint.used_allocation_size);
      EXPECT_EQ(arena.total_allocation_size, checkpoint.total_allocation_size);
      for (unsigned i = 0; i < 64; ++i) {
        EXPECT_EQ(prefix[i], 0xC011AB1Eu);
      }
    }
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
    return status;
  }

  std::vector<uint8_t> Clone(const loom_target_emit_artifact_t& artifact) {
    iree_byte_span_t bytes;
    IREE_CHECK_OK(iree_byte_sequence_clone(artifact.contents,
                                           iree_allocator_system(), &bytes));
    std::vector<uint8_t> result(bytes.data, bytes.data + bytes.data_length);
    iree_allocator_free(iree_allocator_system(), bytes.data);
    return result;
  }

  // Compiler input and pipeline storage, independent of emission scratch.
  iree_arena_block_pool_t pool_ = {};
  // Registered source and target type/operation declarations.
  loom_context_t context_ = {};
  // Target services retained throughout compilation and emission.
  loom_target_environment_t environment_ = {};
  // Target-Low descriptors used by preparation and emission.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Admitted source and its mutable compiler module.
  loom_input_module_t input_ = {};
  // Prepared function versions and immutable target facts.
  loom_compile_pipeline_result_t pipeline_ = {};
};

TEST_F(VMEmissionTest, AllocationFailuresPreserveScratchAndPublishNoArtifact) {
  const auto* data = loom_vm_emission_test_data_create();
  ASSERT_NO_FATAL_FAILURE(
      Prepare({reinterpret_cast<const char*>(data[0].data), data[0].size}));
  const auto* compiled = loom_vm_native_references_bytecode_create();
  const std::vector<uint8_t> expected(compiled[0].data,
                                      compiled[0].data + compiled[0].size);
  for (iree_host_size_t fail_at = 0;; ++fail_at) {
    SCOPED_TRACE(fail_at);
    EmissionAllocator allocations;
    bool emitted = false;
    loom_target_emit_artifact_t artifact = {};
    iree_status_t status = Emit(&allocations, fail_at, &emitted, &artifact);
    const bool succeeded = iree_status_is_ok(status);
    if (succeeded) {
      EXPECT_TRUE(emitted);
      // Emission scratch and its entire pool are already destroyed.
      EXPECT_EQ(Clone(artifact), expected);
      EXPECT_GT(fail_at, 0u);
    } else {
      EXPECT_FALSE(emitted);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(artifact.contents, nullptr);
      EXPECT_EQ(artifact.storage, nullptr);
      EXPECT_EQ(artifact.sidecar_count, 0u);
    }
    loom_target_emit_artifact_release(&artifact);
    EXPECT_EQ(allocations.live, 0u);
    if (succeeded) {
      break;
    }
    ASSERT_GT(allocations.attempts, fail_at)
        << "emission failed without reaching the injected allocation failure";
  }
}

class VMEmissionReferenceScalingTest
    : public VMEmissionTest,
      public ::testing::WithParamInterface<unsigned> {};

TEST_P(VMEmissionReferenceScalingTest, WideSignaturesNeedNoOversizedScratch) {
  // 6240 fields: zero refs, one argument/result pair, or all repeated refs.
  // Each function remains public, preserving its complete entry signature.
  std::string source;
  for (unsigned function = 0; function < 96; ++function) {
    source += "func.def public @function_" + std::to_string(function) + "(";
    for (unsigned argument = 0; argument < 64; ++argument) {
      if (argument) {
        source += ", ";
      }
      const bool reference =
          GetParam() == 2 ||
          (GetParam() == 1 && function == 0 && argument == 0);
      source += "%a" + std::to_string(argument) + ": " +
                (reference ? "hal.buffer" : "i32");
    }
    const char* type = GetParam() == 2 || (GetParam() == 1 && function == 0)
                           ? "hal.buffer"
                           : "i32";
    source += ") -> (" + std::string(type) +
              ") {\n  func.return %a0 : " + type + "\n}\n";
  }
  ASSERT_NO_FATAL_FAILURE(Prepare({source.data(), source.size()}));
  EmissionAllocator allocations;
  allocations.maximum_request = 128 * 1024;
  bool emitted = false;
  loom_target_emit_artifact_t artifact = {};
  IREE_ASSERT_OK(Emit(&allocations, IREE_HOST_SIZE_MAX, &emitted, &artifact));
  ASSERT_TRUE(emitted);
  EXPECT_FALSE(Clone(artifact).empty());
  loom_target_emit_artifact_release(&artifact);
  EXPECT_EQ(allocations.live, 0u);
}

INSTANTIATE_TEST_SUITE_P(ReferenceDensity, VMEmissionReferenceScalingTest,
                         ::testing::Values(0u, 1u, 2u));

}  // namespace
