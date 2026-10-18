// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/target_environment.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "test/util.h"

namespace {
using loomc::testing::HandlePtr;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

void ExpectIncompatibleTarget(const loomc_result_t* result) {
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
  EXPECT_TRUE(
      loomc_string_view_equal(loomc_result_diagnostic_at(result, 0)->code,
                              loomc_make_cstring_view("TARGET/065")));
}

TEST(TestIrEnvironmentTest,
     StructuralVerificationDoesNotAdmitIncompatibleTarget) {
  loomc_target_environment_t* environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_test_ir(
      loomc_allocator_system(), &environment));
  const loomc_context_target_options_t targets = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(targets),
      .target_environment = environment,
  };
  const loomc_context_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &targets,
  };
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &raw_context));
  HandlePtr<loomc_context_t, loomc_context_release> context(raw_context);
  loomc_target_environment_release(environment);
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace);
  constexpr char kSource[] = R"(
test.target<low_core> @test_target {contract_set_key = "missing.low"}
low.func.def target<test.low.core>(@test_target) @entry() asm {
  return
}
)";
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .contents = loomc_make_byte_span(kSource, sizeof(kSource) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* parse_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source, nullptr, loomc_allocator_system(),
      &raw_module, &parse_result));
  HandlePtr<loomc_module_t, loomc_module_release> module(raw_module);
  ResultPtr parse_owner(parse_result);
  ASSERT_TRUE(loomc_result_succeeded(parse_result));
  loomc_source_release(source);
  const loomc_module_verify_options_t structural_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
      .structure_size = sizeof(structural_options),
      .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
  };
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_verify(module.get(), &structural_options,
                                      loomc_allocator_system(), &raw_result));
  ResultPtr result(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));

  // Pipeline admission must still check target invariants after structural
  // success, even for an empty pipeline that would otherwise do no work.
  loomc_pass_program_t* raw_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &raw_program));
  HandlePtr<loomc_pass_program_t, loomc_pass_program_release> program(
      raw_program);
  LOOMC_ASSERT_OK(
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             nullptr, loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ExpectIncompatibleTarget(result.get());

  // The original ABI prefix may have nonzero tail padding. Its size must not
  // cause appended fields to be read from that padding.
  struct VerifyOptionsPrefix {
    // Structure discriminator in the original prefix.
    loomc_structure_type_t type;
    // Caller-provided prefix size.
    loomc_host_size_t structure_size;
    // Original extension chain.
    const void* next;
    // Original error limit, before the appended scope.
    uint32_t max_errors;
  } prefix;
  static_assert(sizeof(prefix) <=
                offsetof(loomc_module_verify_options_t, scope));
  memset(&prefix, 0xFF, sizeof(prefix));
  prefix.type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS;
  prefix.structure_size = sizeof(prefix);
  prefix.next = nullptr;
  prefix.max_errors = 1;
  LOOMC_ASSERT_OK(loomc_module_verify(
      module.get(),
      reinterpret_cast<const loomc_module_verify_options_t*>(&prefix),
      loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ExpectIncompatibleTarget(result.get());

  // A zero-sized descriptor retains the original default verification scope.
  const loomc_module_verify_options_t unsized_options = {
      .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
  };
  LOOMC_ASSERT_OK(loomc_module_verify(module.get(), &unsized_options,
                                      loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ExpectIncompatibleTarget(result.get());
}

}  // namespace
