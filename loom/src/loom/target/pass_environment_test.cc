// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pass_environment.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/facts_builder.h"
#include "loom/target/pass_requirements.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetPassFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule() {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(
        IREE_SV("test.target<low_core> @authored_target\n"
                "func.def target(@authored_target) @authored() {\n"
                "  func.return\n"
                "}\n"
                "func.def @targetless() {\n"
                "  func.return\n"
                "}\n"),
        IREE_SV("target_pass_environment_test.loom"), &context_, &block_pool_,
        &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(op != nullptr);
    return loom_func_like_cast(module, op);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST(TargetPassEnvironmentTest, EnvironmentCarriesProvidersAndVersions) {
  const loom_target_environment_t* target_environment =
      reinterpret_cast<const loom_target_environment_t*>(uintptr_t{1});
  const loom_function_version_list_t function_versions = {};
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make(target_environment, &function_versions);
  const loom_pass_environment_capability_t* capabilities[] = {
      &target_capability.base,
  };
  const loom_pass_environment_t environment =
      loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));

  IREE_ASSERT_OK(loom_pass_environment_verify(&environment));
  const loom_target_pass_capability_t* found_capability =
      loom_target_pass_capability_from_environment(&environment);
  ASSERT_EQ(found_capability, &target_capability);
  EXPECT_EQ(loom_target_pass_capability_target_environment(found_capability),
            target_environment);
  EXPECT_EQ(loom_target_pass_capability_function_versions(found_capability),
            &function_versions);
  EXPECT_EQ(
      loom_target_pass_capability_function_version_owner(found_capability),
      nullptr);
  EXPECT_FALSE(loom_pass_environment_capability_satisfies_requirement(
      &target_capability.base,
      IREE_SV(LOOM_TARGET_PASS_REQUIREMENT_MUTABLE_FUNCTION_VERSIONS)));
}

TEST(TargetPassEnvironmentTest, MutableEnvironmentCarriesVersionOwner) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_function_version_owner_t function_version_owner = {};
  loom_function_version_owner_initialize(&arena, &function_version_owner);

  const loom_target_environment_t* target_environment =
      reinterpret_cast<const loom_target_environment_t*>(uintptr_t{1});
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make_mutable(target_environment,
                                               &function_version_owner);

  EXPECT_EQ(loom_target_pass_capability_target_environment(&target_capability),
            target_environment);
  EXPECT_EQ(loom_target_pass_capability_function_versions(&target_capability),
            &function_version_owner.list);
  EXPECT_EQ(
      loom_target_pass_capability_function_version_owner(&target_capability),
      &function_version_owner);
  EXPECT_TRUE(loom_pass_environment_capability_satisfies_requirement(
      &target_capability.base,
      IREE_SV(LOOM_TARGET_PASS_REQUIREMENT_MUTABLE_FUNCTION_VERSIONS)));

  loom_function_version_t appended_version = {};
  IREE_ASSERT_OK(loom_function_version_owner_append(&function_version_owner,
                                                    &appended_version));
  const loom_function_version_list_t* visible_versions =
      loom_target_pass_capability_function_versions(&target_capability);
  ASSERT_EQ(visible_versions->count, 1u);
  EXPECT_EQ(visible_versions->values[0], &appended_version);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(TargetPassEnvironmentTest,
     MutableEnvironmentDeclaresRuntimeOwnerBeforeInvocation) {
  const loom_target_environment_t* target_environment =
      reinterpret_cast<const loom_target_environment_t*>(uintptr_t{1});
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make_mutable(
          target_environment, /*function_version_owner=*/nullptr);

  EXPECT_EQ(loom_target_pass_capability_target_environment(&target_capability),
            target_environment);
  EXPECT_EQ(loom_target_pass_capability_function_versions(&target_capability),
            nullptr);
  EXPECT_EQ(
      loom_target_pass_capability_function_version_owner(&target_capability),
      nullptr);
  EXPECT_TRUE(loom_pass_environment_capability_satisfies_requirement(
      &target_capability.base,
      IREE_SV(LOOM_TARGET_PASS_REQUIREMENT_MUTABLE_FUNCTION_VERSIONS)));
}

TEST(TargetPassEnvironmentTest, MissingCapabilityHasEmptyAccessors) {
  EXPECT_EQ(loom_target_pass_capability_from_environment(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_target_environment(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_function_versions(nullptr), nullptr);
  EXPECT_EQ(loom_target_pass_capability_function_version_owner(nullptr),
            nullptr);
}

TEST_F(TargetPassFactsTest, RefinedVersionSuppliesFunctionTargetFactsDirectly) {
  ModulePtr module = ParseModule();
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("authored"));
  loom_target_facts_t function_target_facts = {};
  loom_target_facts_builder_initialize(&loom_test_target_fact_type,
                                       loom_test_target_bundles.values[2],
                                       &function_target_facts);
  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = function;
  function_version.function_target_facts = &function_target_facts;

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_t pass = {};
  pass.instance_arena = &arena;
  pass.arena = &arena;
  pass.function_version = &function_version.base;
  bool resolved = false;
  const loom_target_facts_t* resolved_facts = nullptr;
  IREE_ASSERT_OK(loom_target_pass_resolve_function_facts(
      &pass, module.get(), function, &resolved, &resolved_facts));

  EXPECT_TRUE(resolved);
  EXPECT_EQ(resolved_facts, &function_target_facts);
  iree_arena_deinitialize(&arena);
}

TEST_F(TargetPassFactsTest, UnrefinedFunctionProjectsAuthoredFacts) {
  ModulePtr module = ParseModule();
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("authored"));
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(&block_pool_, &instance_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_pass_t pass = {};
  pass.instance_arena = &instance_arena;
  pass.arena = &scratch_arena;
  bool resolved = false;
  const loom_target_facts_t* resolved_facts = nullptr;
  IREE_ASSERT_OK(loom_target_pass_resolve_function_facts(
      &pass, module.get(), function, &resolved, &resolved_facts));

  ASSERT_TRUE(resolved);
  ASSERT_NE(resolved_facts, nullptr);
  EXPECT_EQ(resolved_facts->fact_type, &loom_test_target_fact_type);
  const loom_target_bundle_t* bundle = loom_target_facts_bundle(resolved_facts);
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(bundle->snapshot->index_bitwidth, 64u);
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&instance_arena);
}

TEST_F(TargetPassFactsTest, ProjectedFactsSurviveScratchArenaReset) {
  ModulePtr module = ParseModule();
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("authored"));
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(&block_pool_, &instance_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_pass_t pass = {};
  pass.instance_arena = &instance_arena;
  pass.arena = &scratch_arena;

  bool resolved = false;
  const loom_target_facts_t* resolved_facts = nullptr;
  IREE_ASSERT_OK(loom_target_pass_resolve_function_facts(
      &pass, module.get(), function, &resolved, &resolved_facts));
  ASSERT_TRUE(resolved);
  ASSERT_NE(resolved_facts, nullptr);

  iree_arena_reset(&scratch_arena);
  void* worklist_storage = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate_array(&scratch_arena, 128, sizeof(void*),
                                           &worklist_storage));
  std::memset(worklist_storage, 0, 128 * sizeof(void*));

  const loom_target_bundle_t* bundle = loom_target_facts_bundle(resolved_facts);
  ASSERT_NE(bundle, nullptr);
  ASSERT_NE(bundle->snapshot, nullptr);
  EXPECT_EQ(bundle->snapshot->index_bitwidth, 64u);
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&instance_arena);
}

TEST_F(TargetPassFactsTest, TargetlessFunctionHasNoTargetFacts) {
  ModulePtr module = ParseModule();
  const loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("targetless"));
  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(&block_pool_, &instance_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_pass_t pass = {};
  pass.instance_arena = &instance_arena;
  pass.arena = &scratch_arena;
  bool resolved = false;
  const loom_target_facts_t* resolved_facts = nullptr;
  IREE_ASSERT_OK(loom_target_pass_resolve_function_facts(
      &pass, module.get(), function, &resolved, &resolved_facts));

  EXPECT_FALSE(resolved);
  EXPECT_EQ(resolved_facts, nullptr);
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&instance_arena);
}

}  // namespace
}  // namespace loom
