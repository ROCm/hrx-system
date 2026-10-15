// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/preparation.h"

#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, boundary_source, "", "Preparation boundary callers.");

namespace {

class PreparationTest : public ::testing::Test {
 protected:
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation = {};

  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator), &target_environment)));
    IREE_ASSERT_OK(iree_vm_environment_allocate(host_allocator, &environment));
  }

  void TearDown() override {
    loom_serve_preparation_free(preparation);
    iree_vm_environment_free(environment);
    loomc_target_environment_release(target_environment);
  }

  // Allocation policy for the environment and owned declaration copies.
  const iree_allocator_t host_allocator = iree_allocator_system();
  // Explicit local namespace for the authored boundary fixture.
  const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
  // Environment owning the reference types used during preparation.
  iree_vm_environment_t* environment = nullptr;
  // VM-only grammar for the authored capability fixtures.
  loomc_target_environment_t* target_environment = nullptr;
  // Owned declarations independent of the temporary source program.
  loom_serve_preparation_t* preparation = nullptr;
};

TEST_F(PreparationTest, TextConfigurationCopiesValueSpelling) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("text_config"),
      iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
      iree_vm_module_span_empty(), &compilation, host_allocator, &preparation));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->kind, LOOM_SERVE_PREPARATION_STAGE_COMMAND);
  ASSERT_EQ(stage->config.binding_count, 1);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, DirectDeclarationRetainsProductAndSpecialization) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("direct_config"),
      iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
      iree_vm_module_span_empty(), &compilation, host_allocator, &preparation));
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 1u);
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->kind, LOOM_SERVE_PREPARATION_STAGE_DISPATCH);
  EXPECT_EQ(stage->tag, 17);
  EXPECT_EQ(std::string(stage->root.data, stage->root.size), "stage_root");
  ASSERT_EQ(stage->config.binding_count, 1u);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, WorkspaceLimitConstrainsCompilerReflection) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source),
      IREE_SV("workspace_constrained"), iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &compilation,
      host_allocator, &preparation));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->workspace_limit_bytes, 1073741824u);
  IREE_EXPECT_OK(loom_serve_preparation_stage_check_workspace(
      stage, stage->workspace_limit_bytes));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_preparation_stage_check_workspace(
                            stage, stage->workspace_limit_bytes + 1));
}

TEST_F(PreparationTest, CatalogGrowthPreservesEveryDeclaration) {
  // Bound the regression's failure mode: unconditional doubling would ask for
  // gigabytes before reporting a catalog that only needs kilobytes.
  const iree_allocator_t bounded_allocator = {
      nullptr,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** inout_pointer) -> iree_status_t {
        if (command != IREE_ALLOCATOR_COMMAND_FREE &&
            static_cast<const iree_allocator_alloc_params_t*>(params)
                    ->byte_length > 64 * 1024 * 1024) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "catalog allocation exceeded 64 MiB");
        }
        const auto system = iree_allocator_system();
        return system.ctl(system.self, command, params, inout_pointer);
      }};
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("large_catalog"),
      iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
      iree_vm_module_span_empty(), &compilation, bounded_allocator,
      &preparation));
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 33);
  for (size_t i = 0; i < 33; ++i) {
    const auto* stage = loom_serve_preparation_stage(preparation, i);
    EXPECT_EQ(std::string(stage->root.data, stage->root.size), "stage_root");
    EXPECT_EQ(stage->tag, i);
    ASSERT_EQ(stage->config.binding_count, 33);
    ASSERT_EQ(stage->parameter_count, 33);
    for (size_t j = 0; j < 33; ++j) {
      const auto& binding = stage->config.bindings[j];
      EXPECT_EQ(std::string(binding.key.data, binding.key.size),
                std::string(j < 10 ? "k0" : "k") + std::to_string(j));
      EXPECT_EQ(std::string(binding.value.data, binding.value.size),
                std::to_string(j));
      const auto& parameter = stage->parameters[j];
      EXPECT_EQ(parameter.binding, j);
      ASSERT_EQ(parameter.paths.count, 3);
      const auto* paths = parameter.paths.values;
      EXPECT_EQ(std::string(paths[0].data, paths[0].size),
                "/checkpoints/model.safetensors");
      EXPECT_EQ(std::string(paths[1].data, paths[1].size),
                std::string(j < 10 ? "/checkpoints/k0" : "/checkpoints/k") +
                    std::to_string(j));
      EXPECT_EQ(std::string(paths[2].data, paths[2].size),
                "/checkpoints/weights.loom");
      EXPECT_EQ(std::string(parameter.policy.data, parameter.policy.size),
                "weights.loom");
    }
  }
}

TEST_F(PreparationTest, CheckpointIdentityIncludesEveryFileAndPolicy) {
  const iree_string_view_t files[] = {IREE_SV("first.gguf"),
                                      IREE_SV("second.gguf")};
  const loom_serve_preparation_parameter_t base = {
      0, {IREE_ARRAYSIZE(files), files}, IREE_SV("weights.loom")};
  auto peer = base;
  peer.binding = 1;
  EXPECT_TRUE(loom_serve_preparation_parameter_equal(&base, &peer));
  const iree_string_view_t different[] = {files[0], IREE_SV("other.gguf")};
  peer.paths.values = different;
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer = base;
  peer.paths.count = 1;
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  const iree_string_view_t reversed[] = {files[1], files[0]};
  peer = base;
  peer.paths.values = reversed;
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer = base;
  peer.policy = IREE_SV("adapter.loom");
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
}

TEST_F(PreparationTest, InvalidDeclarationsReleasePartialOwnership) {
  struct Case {
    // Source entry that reaches a public declaration failure.
    const char* entry;
    // Terminal error from the declaration boundary.
    iree_status_code_t code;
  };
  const Case cases[] = {
      {"missing_stage", IREE_STATUS_OUT_OF_RANGE},
      {"conflicting_parameter_policy", IREE_STATUS_INVALID_ARGUMENT},
      {"direct_parameter", IREE_STATUS_INVALID_ARGUMENT},
      {"missing_workspace_stage", IREE_STATUS_OUT_OF_RANGE},
      {"duplicate_workspace_limit", IREE_STATUS_ALREADY_EXISTS},
      {"direct_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"zero_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"negative_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"no_stages", IREE_STATUS_INVALID_ARGUMENT}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.entry);
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_preparation_allocate(
            target_environment, environment, &sources,
            iree_make_cstring_view(FLAG_boundary_source),
            iree_make_cstring_view(test.entry), iree_vm_variant_span_empty(),
            iree_vm_variant_span_empty(), iree_vm_module_span_empty(),
            &compilation, host_allocator, &preparation));
    EXPECT_EQ(preparation, nullptr);
  }
}

TEST_F(PreparationTest, EmbeddedNulCannotTruncateDeclarationStrings) {
  for (int32_t field = 0; field < 6; ++field) {
    SCOPED_TRACE(field);
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(field)};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_preparation_allocate(
            target_environment, environment, &sources,
            iree_make_cstring_view(FLAG_boundary_source), IREE_SV("nul_string"),
            iree_vm_variant_span_from_array(arguments),
            iree_vm_variant_span_empty(), iree_vm_module_span_empty(),
            &compilation, host_allocator, &preparation));
    EXPECT_EQ(preparation, nullptr);
  }
}

}  // namespace
