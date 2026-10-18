// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/configured_ir.h"

#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

TEST(ConfiguredIrTest, ContextRetainsPortableCommandIrSupport) {
  loomc_target_environment_t* environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_configured_ir(
      loomc_allocator_system(), &environment));
  HandlePtr<loomc_target_environment_t, loomc_target_environment_release>
      environment_owner(environment);
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = environment,
  };
  const loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &context));
  HandlePtr<loomc_context_t, loomc_context_release> context_owner(context);
  environment_owner.reset();

  loomc_workspace_t* workspace = nullptr;
  LOOMC_ASSERT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace_owner(
      workspace);
  constexpr char kInput[] = R"(
low.func.def target<cmd.core> abi(command_program) @empty() asm {
  return
}
)";
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .contents = loomc_make_byte_span(kInput, sizeof(kInput) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  HandlePtr<loomc_source_t, loomc_source_release> source_owner(source);
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result));
  HandlePtr<loomc_module_t, loomc_module_release> module_owner(module);
  HandlePtr<loomc_result_t, loomc_result_release> result_owner(result);
  ASSERT_TRUE(loomc_result_succeeded(result));

  LOOMC_ASSERT_OK(
      loomc_module_verify(module, nullptr, loomc_allocator_system(), &result));
  result_owner.reset(result);
  EXPECT_TRUE(loomc_result_succeeded(result));
}

}  // namespace
