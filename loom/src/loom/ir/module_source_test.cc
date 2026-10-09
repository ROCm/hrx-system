// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module_source.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleSourceTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleSourceTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected source allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(4096, {this, Allocate}, &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    loom_context_deinitialize(&context_);
  }

  // Registry shared by independent source-owning modules.
  loom_context_t context_ = {};
  // Arena block owner retained until every test module has been released.
  iree_arena_block_pool_t block_pool_ = {};
  // Backing allocation attempts since fixture initialization.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for ordinary allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

TEST_F(ModuleSourceTest, RegisterSourceDeduplicatesBySpelling) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_source_id_t first_id = LOOM_SOURCE_ID_INVALID;
  loom_source_id_t second_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("model.loom"), &first_id));
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("model.loom"), &second_id));

  EXPECT_EQ(first_id, 0u);
  EXPECT_EQ(second_id, first_id);
  ASSERT_EQ(module->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module->sources.entries[0],
                                     IREE_SV("model.loom")));
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, RegisterEmptySource) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(module, iree_string_view_empty(),
                                             &source_id));

  EXPECT_EQ(source_id, 0u);
  ASSERT_EQ(module->sources.count, 1u);
  EXPECT_EQ(module->sources.entries[0].size, 0u);
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, AppendSourcePreservesInsertionOrder) {
  loom_module_size_hints_t hints = {.source_count = 2};
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      &hints, iree_allocator_system(),
                                      &module));

  loom_source_id_t first_id = LOOM_SOURCE_ID_INVALID;
  loom_source_id_t second_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_append_source(module, IREE_SV("a.loom"), &first_id));
  IREE_ASSERT_OK(
      loom_module_append_source(module, IREE_SV("b.loom"), &second_id));

  EXPECT_EQ(first_id, 0u);
  EXPECT_EQ(second_id, 1u);
  EXPECT_GE(module->sources.capacity, 2u);
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, RegisterSourceRejectsInvalidSentinelId) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));

  // Trusted append requires unique names, including at the ID boundary.
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  for (iree_host_size_t i = 0; i < LOOM_SOURCE_ID_INVALID; ++i) {
    const std::string name = i == 0 ? "present" : std::to_string(i);
    IREE_ASSERT_OK(loom_module_append_source(
        module, iree_make_string_view(name.data(), name.size()), &source_id));
  }
  EXPECT_EQ(source_id, LOOM_SOURCE_ID_INVALID - 1);
  EXPECT_EQ(module->sources.count, LOOM_SOURCE_ID_INVALID);

  // An existing source remains resolvable even when no new ID is available.
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("present"), &source_id));
  EXPECT_EQ(source_id, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_module_register_source(module, IREE_SV("overflow"), &source_id));
  EXPECT_EQ(source_id, LOOM_SOURCE_ID_INVALID);
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, IndexesOnlyNewlyAppendedNames) {
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("sources"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  for (uint32_t batch = 0; batch < 4; ++batch) {
    for (uint32_t i = batch * 128; i < (batch + 1) * 128; ++i) {
      std::string name = "source_" + std::to_string(i);
      loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
      IREE_ASSERT_OK(loom_module_append_source(
          module, iree_make_string_view(name.data(), name.size()), &source_id));
      EXPECT_EQ(source_id, i);
      // The input string is temporary; the module retains its own copy.
      name.assign(name.size(), '!');
    }
    if (batch == 0) {
      EXPECT_EQ(module->sources.name_index, nullptr);
    } else {
      EXPECT_EQ(module->sources.name_index->count, batch * 128u);
    }
    for (uint32_t i = 0; i < (batch + 1) * 128; ++i) {
      const std::string name = "source_" + std::to_string(i);
      loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
      IREE_ASSERT_OK(loom_module_register_source(
          module, iree_make_string_view(name.data(), name.size()), &source_id));
      EXPECT_EQ(source_id, i);
    }
    EXPECT_EQ(module->sources.name_index->count, module->sources.count);
  }
  const auto used_before = module->arena.used_allocation_size;
  const auto total_before = module->arena.total_allocation_size;
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("source_511"), &source_id));
  EXPECT_EQ(source_id, 511u);
  EXPECT_EQ(module->arena.used_allocation_size, used_before);
  EXPECT_EQ(module->arena.total_allocation_size, total_before);
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, RegistrationPreservesExactByteIdentityAcrossGrowth) {
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("sources"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  const iree_string_view_t names[] = {
      IREE_SV(""),
      IREE_SV("kernel.loom"),
      IREE_SV("Kernel.loom"),
      IREE_SV("./kernel.loom"),
      IREE_SV("a/../kernel.loom"),
      iree_make_string_view("kernel.loom\0suffix", 18)};
  for (uint32_t i = 0; i < 512; ++i) {
    const std::string generated = "source_" + std::to_string(i);
    const auto name =
        i < IREE_ARRAYSIZE(names)
            ? names[i]
            : iree_make_string_view(generated.data(), generated.size());
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_register_source(module, name, &source_id));
    EXPECT_EQ(source_id, i);
    if (i < 16) {
      EXPECT_EQ(module->sources.name_index, nullptr);
    }
  }
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_register_source(module, names[i], &source_id));
    EXPECT_EQ(source_id, i);
    EXPECT_TRUE(iree_string_view_equal(module->sources.entries[i], names[i]));
  }
  EXPECT_EQ(module->sources.count, 512u);
  EXPECT_EQ(module->sources.name_index->count, 512u);
  loom_module_free(module);
}

TEST_F(ModuleSourceTest, IndexAllocationFailurePreservesSourceIdentity) {
  // A small pool makes the index header and bucket segment backing requests
  // independently fallible, without manufacturing invalid source-table state.
  iree_arena_block_pool_deinitialize(&block_pool_);
  iree_arena_block_pool_initialize(128, {this, Allocate}, &block_pool_);
  for (iree_host_size_t failed_allocation = 0; failed_allocation < 2;
       ++failed_allocation) {
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("sources"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module));
    for (uint32_t i = 0; i < 128; ++i) {
      const std::string name = std::to_string(i);
      loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
      IREE_ASSERT_OK(loom_module_append_source(
          module, iree_make_string_view(name.data(), name.size()), &source_id));
    }
    const auto used_before = module->arena.used_allocation_size;
    const auto total_before = module->arena.total_allocation_size;
    failure_index_ = allocation_count_ + failed_allocation;
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_module_register_source(module, IREE_SV("7"), &source_id));
    EXPECT_EQ(source_id, LOOM_SOURCE_ID_INVALID);
    EXPECT_EQ(module->sources.count, 128u);
    EXPECT_EQ(module->sources.name_index, nullptr);
    EXPECT_EQ(module->arena.used_allocation_size, used_before);
    EXPECT_EQ(module->arena.total_allocation_size, total_before);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(
        loom_module_register_source(module, IREE_SV("7"), &source_id));
    EXPECT_EQ(source_id, 7u);
    loom_module_free(module);
  }
}

}  // namespace
}  // namespace loom
