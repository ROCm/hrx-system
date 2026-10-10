// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"

namespace loom {
namespace {

class ModuleEncodingTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleEncodingTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected encoding allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(128, {this, Allocate}, &pool_);
    ASSERT_NO_FATAL_FAILURE(Prepare(0));
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  void Prepare(uint16_t count) {
    failure_index_ = SIZE_MAX;
    loom_module_free(module_);
    module_ = nullptr;
    iree_arena_block_pool_trim(&pool_);
    const loom_module_size_hints_t hints = {};
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("encodings"), &pool_,
                                        &hints, iree_allocator_system(),
                                        &module_));
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, IREE_SV("encoding.layout.strided"), &family_id_));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("strides"), &strides_id_));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("layout"), &alias_id_));
    for (uint16_t i = 0; i < count; ++i) {
      uint16_t id = 0;
      IREE_ASSERT_OK(Add(i + 1, LOOM_STRING_ID_INVALID, &id));
      ASSERT_EQ(id, i + 1);
    }
    allocation_count_ = 0;
  }

  iree_status_t Add(int64_t stride, loom_string_id_t alias_id,
                    uint16_t* out_id) {
    int64_t strides[] = {stride, 1};
    const loom_named_attr_t parameters[] = {{
        .name_id = strides_id_,
        .reserved = {},
        .value = loom_attr_i64_array(strides, IREE_ARRAYSIZE(strides)),
    }};
    const loom_encoding_t encoding = {
        .name_id = family_id_,
        .alias_id = alias_id,
        .attribute_count = IREE_ARRAYSIZE(parameters),
        .family = {},
        .attributes = parameters,
    };
    return loom_module_add_encoding(module_, &encoding, out_id);
  }

  // Backing allocation ordinal to fail, or SIZE_MAX to allow every request.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Backing allocation requests since the last catalog preparation.
  iree_host_size_t allocation_count_ = 0;
  // Small blocks expose payload and table growth to the allocator above.
  iree_arena_block_pool_t pool_ = {};
  // Shipping encoding families used by every construction in these tests.
  loom_context_t context_ = {};
  // Owner of all encoding candidates and canonical rows under test.
  loom_module_t* module_ = nullptr;
  // Interned name of the registered strided layout family.
  loom_string_id_t family_id_ = LOOM_STRING_ID_INVALID;
  // Interned name of its stride-array parameter.
  loom_string_id_t strides_id_ = LOOM_STRING_ID_INVALID;
  // Interned display name used for alias promotion and conflict checks.
  loom_string_id_t alias_id_ = LOOM_STRING_ID_INVALID;
};

TEST_F(ModuleEncodingTest, DuplicateAndAliasPromotionReleaseCandidates) {
  uint16_t id = 0;
  IREE_ASSERT_OK(Add(16, LOOM_STRING_ID_INVALID, &id));
  const auto* parameters = loom_module_encoding(module_, id)->attributes;
  const auto retained_bytes = module_->arena.used_allocation_size;
  const auto owned_bytes = module_->arena.total_allocation_size;
  for (int i = 0; i < 8; ++i) {
    uint16_t duplicate_id = 0;
    IREE_ASSERT_OK(Add(16, alias_id_, &duplicate_id));
    EXPECT_EQ(duplicate_id, id);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    EXPECT_EQ(module_->arena.total_allocation_size, owned_bytes);
    const auto* encoding = loom_module_encoding(module_, duplicate_id);
    EXPECT_TRUE(loom_encoding_static_is_valid(encoding));
    EXPECT_EQ(encoding->alias_id, alias_id_);
    EXPECT_EQ(encoding->attributes, parameters);
    EXPECT_EQ(parameters[0].value.i64_array[0], 16);
    EXPECT_EQ(parameters[0].value.i64_array[1], 1);
  }
}

TEST_F(ModuleEncodingTest, RejectedParametersAndAliasesReleaseCandidates) {
  uint16_t id = 0;
  IREE_ASSERT_OK(Add(16, alias_id_, &id));
  const auto retained_bytes = module_->arena.used_allocation_size;
  const auto owned_bytes = module_->arena.total_allocation_size;
  int64_t strides[] = {32, 1};
  const loom_named_attr_t parameters[] = {
      {strides_id_, {}, loom_attr_i64_array(strides, 2)},
      {strides_id_, {}, loom_attr_i64_array(strides, 2)},
  };
  const loom_encoding_t duplicate_keys = {
      .name_id = family_id_,
      .alias_id = LOOM_STRING_ID_INVALID,
      .attribute_count = IREE_ARRAYSIZE(parameters),
      .family = {},
      .attributes = parameters,
  };
  for (int i = 0; i < 8; ++i) {
    uint16_t rejected_id = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_module_add_encoding(module_, &duplicate_keys, &rejected_id));
    EXPECT_EQ(rejected_id, 0);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Add(32, alias_id_, &rejected_id));
    EXPECT_EQ(rejected_id, 0);
    EXPECT_EQ(module_->encodings.count, 1u);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    EXPECT_EQ(module_->arena.total_allocation_size, owned_bytes);
  }
  IREE_ASSERT_OK(Add(32, LOOM_STRING_ID_INVALID, &id));
  EXPECT_EQ(id, 2);
}

TEST_F(ModuleEncodingTest, FamilyAliasExpansionRetainsCanonicalParameters) {
  loom_string_id_t family_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("encoding.i8"), &family_id));
  const loom_encoding_t encoding = {
      .name_id = family_id,
      .alias_id = alias_id_,
  };
  uint16_t id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &id));
  EXPECT_TRUE(loom_encoding_static_is_valid(loom_module_encoding(module_, id)));
  const auto retained_bytes = module_->arena.used_allocation_size;
  const auto string_count = module_->strings.count;
  const auto* parameters = loom_module_encoding(module_, id)->attributes;
  for (int i = 0; i < 8; ++i) {
    uint16_t duplicate_id = 0;
    IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &duplicate_id));
    EXPECT_EQ(duplicate_id, id);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    EXPECT_EQ(module_->strings.count, string_count);
    EXPECT_EQ(loom_module_encoding(module_, id)->attributes, parameters);
  }
}

TEST_F(ModuleEncodingTest, OnlyCanonicalDisplayNamesReserveAliases) {
  loom_string_id_t second_alias = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("second_layout"),
                                           &second_alias));
  uint16_t first_id = 0;
  IREE_ASSERT_OK(Add(16, alias_id_, &first_id));
  uint16_t id = 0;
  IREE_ASSERT_OK(Add(16, second_alias, &id));
  EXPECT_EQ(id, first_id);
  EXPECT_EQ(loom_module_encoding(module_, id)->alias_id, alias_id_);

  // An alternate spelling for an existing row does not become its display
  // name or reserve another name in the module's canonical alias namespace.
  IREE_ASSERT_OK(Add(32, second_alias, &id));
  EXPECT_NE(id, first_id);
  EXPECT_EQ(loom_module_encoding(module_, id)->alias_id, second_alias);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Add(16, second_alias, &id));
  EXPECT_EQ(module_->encodings.count, 2u);
}

TEST_F(ModuleEncodingTest, AliasLookupSurvivesGrowthAndSymbolCompaction) {
  std::array<loom_string_id_t, 128> aliases;
  for (uint16_t i = 0; i < aliases.size(); ++i) {
    const std::string name = "layout_" + std::to_string(i);
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_string_view(name.data(), name.size()), &aliases[i]));
    uint16_t id = 0;
    IREE_ASSERT_OK(Add(i + 1, aliases[i], &id));
    EXPECT_EQ(id, i + 1);
  }

  // Removing a tombstone rebuilds the encoding index even when no encoding
  // parameter itself contains a symbol reference.
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, alias_id_, &symbol_id));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&pool_, &scratch_arena);
  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(
      loom_module_compact_symbols(module_, &scratch_arena, &removed_count));
  iree_arena_deinitialize(&scratch_arena);
  EXPECT_EQ(removed_count, 1u);

  const auto retained_bytes = module_->arena.used_allocation_size;
  for (uint16_t i = 0; i < aliases.size(); ++i) {
    SCOPED_TRACE(i);
    uint16_t id = 0;
    IREE_ASSERT_OK(Add(i + 1, aliases[i], &id));
    EXPECT_EQ(id, i + 1);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Add(aliases.size() + 1, aliases[i], &id));
  }
  EXPECT_EQ(module_->encodings.count, aliases.size());
  EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
}

TEST_F(ModuleEncodingTest, FailedAliasPromotionPreservesAnonymousRow) {
  for (iree_host_size_t failure_index = 0;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    ASSERT_NO_FATAL_FAILURE(Prepare(96));
    const auto interner = module_->encoding_intern;
    const auto retained_bytes = module_->arena.used_allocation_size;
    uint16_t id = 0;
    failure_index_ = failure_index;
    iree_status_t status = Add(1, alias_id_, &id);
    failure_index_ = SIZE_MAX;
    if (iree_status_is_ok(status)) {
      EXPECT_LE(allocation_count_, failure_index);
      EXPECT_EQ(id, 1);
      EXPECT_EQ(loom_module_encoding(module_, id)->alias_id, alias_id_);
      break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(id, 0);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    EXPECT_EQ(module_->encoding_intern.count, interner.count);
    EXPECT_EQ(module_->encoding_intern.capacity, interner.capacity);
    EXPECT_EQ(loom_module_encoding(module_, 1)->alias_id,
              LOOM_STRING_ID_INVALID);
    IREE_ASSERT_OK(Add(1, alias_id_, &id));
    EXPECT_EQ(id, 1);
    EXPECT_EQ(loom_module_encoding(module_, id)->alias_id, alias_id_);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Add(2, alias_id_, &id));
  }
}

class ModuleEncodingFailureTest
    : public ModuleEncodingTest,
      public ::testing::WithParamInterface<uint16_t> {};

TEST_P(ModuleEncodingFailureTest, FailedPublicationPreservesRowsAndRetries) {
  for (iree_host_size_t failure_index = 0;; ++failure_index) {
    SCOPED_TRACE(failure_index);
    ASSERT_NO_FATAL_FAILURE(Prepare(GetParam()));
    const auto encodings = module_->encodings;
    const auto interner = module_->encoding_intern;
    const auto retained_bytes = module_->arena.used_allocation_size;
    const auto owned_bytes = module_->arena.total_allocation_size;
    uint16_t id = 0;
    failure_index_ = failure_index;
    iree_status_t status = Add(GetParam() + 1, alias_id_, &id);
    failure_index_ = SIZE_MAX;
    if (iree_status_is_ok(status)) {
      EXPECT_LE(allocation_count_, failure_index);
      EXPECT_EQ(id, GetParam() + 1);
      break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(allocation_count_, failure_index + 1);
    EXPECT_EQ(id, 0);
    EXPECT_EQ(module_->encodings.count, encodings.count);
    EXPECT_EQ(module_->encodings.capacity, encodings.capacity);
    EXPECT_EQ(module_->encodings.entries, encodings.entries);
    EXPECT_EQ(module_->encoding_intern.count, interner.count);
    EXPECT_EQ(module_->encoding_intern.capacity, interner.capacity);
    EXPECT_EQ(module_->arena.used_allocation_size, retained_bytes);
    EXPECT_EQ(module_->arena.total_allocation_size, owned_bytes);
    for (uint16_t i = 0; i < GetParam(); ++i) {
      IREE_ASSERT_OK(Add(i + 1, LOOM_STRING_ID_INVALID, &id));
      EXPECT_EQ(id, i + 1);
      EXPECT_TRUE(
          loom_encoding_static_is_valid(loom_module_encoding(module_, id)));
    }
    IREE_ASSERT_OK(Add(GetParam() + 1, alias_id_, &id));
    EXPECT_EQ(id, GetParam() + 1);
    EXPECT_EQ(module_->encodings.count, encodings.count + 1);
    EXPECT_EQ(loom_module_encoding(module_, id)->alias_id, alias_id_);
  }
}

INSTANTIATE_TEST_SUITE_P(Growth, ModuleEncodingFailureTest,
                         ::testing::Values(0, 8, 95, 96));

}  // namespace
}  // namespace loom
