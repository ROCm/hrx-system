// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_version.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetFunctionVersionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_function_version_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_symbol_id_t symbol_id = FindSymbol(module, name);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(op != nullptr);
    loom_func_like_t function = loom_func_like_cast(module, op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  iree::Status BuildSnapshot(
      const loom_module_t* module,
      const loom_function_version_list_t* function_versions) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_target_function_version_snapshot_t snapshot = {};
    iree::Status status(loom_target_function_version_snapshot_build(
        module, function_versions, &arena, &snapshot));
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TargetFunctionVersionTest, SnapshotMapsTargetVersionsBySymbol) {
  ModulePtr module = ParseModule(R"(
test.func @first() {
  test.yield
}
test.func @unversioned() {
  test.yield
}
test.func @other_version_type() {
  test.yield
}
test.func @second() {
  test.yield
}
)");

  loom_target_facts_t first_facts = {};
  loom_target_function_version_t first_version = {};
  first_version.base.type = &loom_target_function_version_type;
  first_version.base.function = FindFunction(module.get(), IREE_SV("first"));
  first_version.function_target_facts = &first_facts;

  loom_target_facts_t second_facts = {};
  loom_target_function_version_t second_version = {};
  second_version.base.type = &loom_target_function_version_type;
  second_version.base.function = FindFunction(module.get(), IREE_SV("second"));
  second_version.function_target_facts = &second_facts;

  const loom_function_version_type_t other_type = {
      /*.name=*/IREE_SVL("other"),
  };
  loom_function_version_t other_version = {
      /*.type=*/&other_type,
      /*.function=*/FindFunction(module.get(), IREE_SV("other_version_type")),
  };
  loom_function_version_t* version_values[] = {
      &second_version.base,
      &other_version,
      &first_version.base,
  };
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_target_function_version_snapshot_t snapshot = {};
  IREE_ASSERT_OK(loom_target_function_version_snapshot_build(
      module.get(), &versions, &arena, &snapshot));

  EXPECT_EQ(snapshot.symbol_count, module->symbols.count);
  EXPECT_EQ(loom_target_function_version_snapshot_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("first"))),
            &first_version);
  EXPECT_EQ(loom_target_function_version_snapshot_handle_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("first"))),
            &first_version.base);
  EXPECT_EQ(loom_target_function_version_snapshot_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("unversioned"))),
            nullptr);
  EXPECT_EQ(loom_target_function_version_snapshot_handle_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("unversioned"))),
            nullptr);
  EXPECT_EQ(
      loom_target_function_version_snapshot_at(
          &snapshot, FindSymbol(module.get(), IREE_SV("other_version_type"))),
      nullptr);
  EXPECT_EQ(loom_target_function_version_snapshot_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("second"))),
            &second_version);
  iree_arena_deinitialize(&arena);
}

TEST_F(TargetFunctionVersionTest, EmptySnapshotRetainsModuleBounds) {
  ModulePtr module = ParseModule(R"(
test.func @only() {
  test.yield
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_target_function_version_snapshot_t snapshot = {};
  IREE_ASSERT_OK(loom_target_function_version_snapshot_build(
      module.get(), /*function_versions=*/nullptr, &arena, &snapshot));

  EXPECT_EQ(snapshot.symbol_count, module->symbols.count);
  EXPECT_EQ(loom_target_function_version_snapshot_at(
                &snapshot, FindSymbol(module.get(), IREE_SV("only"))),
            nullptr);
  iree_arena_deinitialize(&arena);
}

TEST_F(TargetFunctionVersionTest, RejectsFunctionFromAnotherModule) {
  ModulePtr module = ParseModule(R"(
test.func @only() {
  test.yield
}
)");
  ModulePtr foreign_module = ParseModule(R"(
test.func @foreign() {
  test.yield
}
)");

  loom_target_function_version_t version = {};
  version.base.type = &loom_target_function_version_type;
  version.base.function =
      FindFunction(foreign_module.get(), IREE_SV("foreign"));
  loom_function_version_t* version_values[] = {&version.base};
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  iree::Status status = BuildSnapshot(module.get(), &versions);

  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("does not name the live definition"));
}

TEST_F(TargetFunctionVersionTest, RejectsOutOfRangeFunctionSymbol) {
  ModulePtr module = ParseModule("");
  ModulePtr foreign_module = ParseModule(R"(
test.func @foreign() {
  test.yield
}
)");

  loom_target_function_version_t version = {};
  version.base.type = &loom_target_function_version_type;
  version.base.function =
      FindFunction(foreign_module.get(), IREE_SV("foreign"));
  loom_function_version_t* version_values[] = {&version.base};
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  iree::Status status = BuildSnapshot(module.get(), &versions);

  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("does not name a module-local function"));
}

TEST_F(TargetFunctionVersionTest, RejectsDuplicateFunctionVersions) {
  ModulePtr module = ParseModule(R"(
test.func @only() {
  test.yield
}
)");

  loom_target_function_version_t first_version = {};
  first_version.base.type = &loom_target_function_version_type;
  first_version.base.function = FindFunction(module.get(), IREE_SV("only"));
  loom_target_function_version_t second_version = {};
  second_version.base.type = &loom_target_function_version_type;
  second_version.base.function = FindFunction(module.get(), IREE_SV("only"));
  loom_function_version_t* version_values[] = {
      &first_version.base,
      &second_version.base,
  };
  const loom_function_version_list_t versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  iree::Status status = BuildSnapshot(module.get(), &versions);

  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("multiple target function versions"));
}

}  // namespace
}  // namespace loom
