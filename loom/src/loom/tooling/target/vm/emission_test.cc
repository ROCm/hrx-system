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
#include "loom/target/emit/vm/module_compiler.h"
#include "loom/tooling/target/vm/emission_test_data.h"
#include "loom/tooling/target/vm/native_references_bytecode.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"

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
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        compiler_allocator_, &target_environment_)));
    environment_ =
        loomc_target_environment_get_interop_view(target_environment_);
    ASSERT_NE(environment_, nullptr);
    registry_ = loom_target_environment_low_descriptor_registry(environment_);

    loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .target_environment = target_environment_,
    };
    loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_context_create(
        &context_options, compiler_allocator_, &context_)));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_workspace_create(
        /*options=*/nullptr, compiler_allocator_, &workspace_)));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_compiler_create(
        context_, /*options=*/nullptr, compiler_allocator_, &compiler_)));
    loomc_result_t* program_result = nullptr;
    IREE_ASSERT_OK(
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            context_, /*options=*/nullptr, compiler_allocator_, &pass_program_,
            &program_result)));
    ASSERT_NE(program_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(program_result));
    loomc_result_release(program_result);
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_profile_select(
        target_environment_, loomc_make_cstring_view("vm:core"),
        compiler_allocator_, &target_profile_)));
  }

  void TearDown() override {
    loomc_module_release(module_);
    loomc_target_profile_release(target_profile_);
    loomc_pass_program_release(pass_program_);
    loomc_compiler_release(compiler_);
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(target_environment_);
  }

  void Prepare(iree_string_view_t source) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .next = nullptr,
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("emission.loom"),
        .contents = loomc_make_byte_span(source.data, source.size),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* compiler_source = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_source_create(
        &source_options, compiler_allocator_, &compiler_source)));
    loomc_result_t* deserialize_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_deserialize_from_source(
        context_, workspace_, compiler_source, /*options=*/nullptr,
        compiler_allocator_, &module_, &deserialize_result)));
    loomc_source_release(compiler_source);
    ASSERT_NE(deserialize_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(deserialize_result));
    loomc_result_release(deserialize_result);

    const loomc_module_function_query_options_t query_options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
        .structure_size = sizeof(query_options),
        .next = nullptr,
        .function_symbol = loomc_string_view_empty(),
        .kind = LOOMC_MODULE_FUNCTION_KIND_FUNCTION,
    };
    loomc_host_size_t function_count = 0;
    loomc_result_t* count_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_query_functions(
        module_, &query_options, compiler_allocator_, /*function_capacity=*/0,
        /*out_functions=*/nullptr, &function_count, &count_result)));
    ASSERT_NE(count_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(count_result));
    loomc_result_release(count_result);

    std::vector<loomc_module_function_t> functions(function_count);
    loomc_result_t* query_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_query_functions(
        module_, &query_options, compiler_allocator_, functions.size(),
        functions.data(), &function_count, &query_result)));
    ASSERT_NE(query_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(query_result));
    loomc_result_release(query_result);

    std::vector<loomc_target_specialization_t> specializations;
    for (const auto& function : functions) {
      if ((function.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC) == 0) {
        continue;
      }
      specializations.push_back({
          .function_symbol = function.symbol_name,
          .target_profile = target_profile_,
      });
    }
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .specializations = specializations.data(),
        .specialization_count = specializations.size(),
        .target_bindings = nullptr,
        .target_binding_count = 0,
    };
    const loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = &target_options,
        .module_name = loomc_string_view_empty(),
        .artifact_flags = 0,
        .config_flags = 0,
        .config_module = nullptr,
    };
    loomc_result_t* compile_result = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_compile_module(
        compiler_, workspace_, pass_program_, module_, &compile_options,
        compiler_allocator_, &compile_result)));
    ASSERT_NE(compile_result, nullptr);
    ASSERT_TRUE(loomc_result_succeeded(compile_result));
    loomc_result_release(compile_result);

    loomc_module_t* projected_module = nullptr;
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_module_clone(
        module_, workspace_, compiler_allocator_, &projected_module)));
    loomc_module_release(module_);
    module_ = projected_module;
    const loomc_module_mutable_interop_view_t view =
        loomc_module_get_mutable_interop_view(module_);
    ASSERT_NE(view.module, nullptr);
    native_module_ = view.module;
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
      loom_target_emit_request_t request = {
          .target_environment = environment_,
          .low_descriptor_registry = &registry_.registry,
          .module = native_module_,
          .scratch_arena = &arena,
          .allocator = allocations->allocator()};
      status = loom_vm_module_emitter.emit(&request, out_emitted, out_artifact);
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

  // Host storage for the reusable compiler and its products.
  loomc_allocator_t compiler_allocator_ = loomc_allocator_system();
  // Public target package owning the native environment below.
  loomc_target_environment_t* target_environment_ = nullptr;
  // Native target services borrowed for the direct emitter assertion.
  const loom_target_environment_t* environment_ = nullptr;
  // Target-Low descriptors used by program planning and emission.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Reusable public compiler state independent of emission scratch.
  loomc_context_t* context_ = nullptr;
  loomc_workspace_t* workspace_ = nullptr;
  loomc_compiler_t* compiler_ = nullptr;
  loomc_pass_program_t* pass_program_ = nullptr;
  loomc_target_profile_t* target_profile_ = nullptr;
  // Prepared module and its exact-version native projection.
  loomc_module_t* module_ = nullptr;
  loom_module_t* native_module_ = nullptr;
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
