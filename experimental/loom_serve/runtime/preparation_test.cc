// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/preparation.h"

#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, boundary_source, "", "Preparation boundary callers.");

namespace {

class PreparationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
  }

  void TearDown() override {
    loom_serve_preparation_destroy(preparation);
    iree_vm_environment_free(environment);
  }

  // Allocation policy for the environment and owned declaration copies.
  const iree_allocator_t allocator = iree_allocator_system();
  // Environment owning the reference types used during preparation.
  iree_vm_environment_t* environment = nullptr;
  // Owned declarations independent of the temporary source program.
  loom_serve_preparation_t* preparation = nullptr;
};

TEST_F(PreparationTest, TextConfigurationCopiesValueSpelling) {
  IREE_ASSERT_OK(loom_serve_preparation_create(
      environment, iree_make_cstring_view(FLAG_boundary_source),
      IREE_SV("text_config"), iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &preparation,
      allocator));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  ASSERT_EQ(stage->config.binding_count, 1);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, CatalogGrowthPreservesEveryDeclaration) {
  // Bound the regression's failure mode: unconditional doubling would ask for
  // gigabytes before reporting a catalog that only needs kilobytes.
  const iree_allocator_t bounded_allocator = {
      nullptr,
      +[](void*, iree_allocator_command_t command, const void* params,
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
  IREE_ASSERT_OK(loom_serve_preparation_create(
      environment, iree_make_cstring_view(FLAG_boundary_source),
      IREE_SV("large_catalog"), iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &preparation,
      bounded_allocator));
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
      EXPECT_EQ(std::string(parameter.path.data, parameter.path.size),
                "/checkpoints/model.safetensors");
      EXPECT_EQ(std::string(parameter.policy.data, parameter.policy.size),
                "weights.loom");
    }
  }
}

TEST_F(PreparationTest, InvalidDeclarationsReleasePartialOwnership) {
  struct Case {
    // Source entry that reaches a public declaration failure.
    const char* entry;
    // Terminal error from the declaration boundary.
    iree_status_code_t code;
  };
  const Case cases[] = {{"missing_stage", IREE_STATUS_OUT_OF_RANGE},
                        {"duplicate_parameter", IREE_STATUS_ALREADY_EXISTS},
                        {"no_stages", IREE_STATUS_INVALID_ARGUMENT}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.entry);
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_preparation_create(
            environment, iree_make_cstring_view(FLAG_boundary_source),
            iree_make_cstring_view(test.entry), iree_vm_variant_span_empty(),
            iree_vm_variant_span_empty(), iree_vm_module_span_empty(),
            &preparation, allocator));
    EXPECT_EQ(preparation, nullptr);
  }
}

}  // namespace
