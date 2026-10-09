// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <random>
#include <set>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/types.h"

namespace loom {
namespace {

class ModuleTypeUsesTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleTypeUsesTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(1024, {this, Allocate}, &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    ResetModule();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void ResetModule() {
    failure_index_ = SIZE_MAX;
    loom_module_free(module_);
    module_ = nullptr;
    iree_arena_block_pool_trim(&pool_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("references"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    width_ = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
    allocation_count_ = 0;
  }

  loom_value_id_t AddArgument(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &value));
    IREE_CHECK_OK(
        loom_block_add_arg(module_, loom_module_block(module_), value));
    return value;
  }

  loom_type_id_t Intern(loom_type_t type) {
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_type_id(module_, type, &id));
    return id;
  }

  // Immediate canonical children follow the production topological constructor
  // contract. The temporary FAM is data for the type API, not an IR program.
  loom_type_id_t Pair(loom_type_id_t first, loom_type_id_t second) {
    alignas(loom_func_type_data_t) uint8_t
        storage[sizeof(loom_func_type_data_t) + 2 * sizeof(loom_type_t)] = {};
    auto* data = reinterpret_cast<loom_func_type_data_t*>(storage);
    data->arg_count = 2;
    data->types[0] = loom_type_table_get(&module_->types, first);
    data->types[1] = loom_type_table_get(&module_->types, second);
    const loom_type_id_t children[] = {first, second};
    loom_type_id_t id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_topological_type_id(
        module_, loom_type_function(data), children, IREE_ARRAYSIZE(children),
        &id));
    return id;
  }

  static loom_type_t Matrix(loom_value_id_t rows, loom_value_id_t columns) {
    return loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                               loom_dim_pack_dynamic(rows),
                               loom_dim_pack_dynamic(columns), 0);
  }

  std::vector<loom_value_id_t> Dependencies(loom_value_id_t value) {
    std::vector<loom_value_id_t> result;
    loom_type_use_iterator_t iterator;
    loom_module_value_type_dependencies(module_, value, &iterator);
    for (auto provider = loom_type_dependencies_next(&iterator);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&iterator)) {
      result.push_back(provider);
    }
    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));
    EXPECT_EQ(std::adjacent_find(result.begin(), result.end()), result.end());
    return result;
  }

  std::vector<loom_value_id_t> Users(loom_value_id_t provider) {
    std::vector<loom_value_id_t> result;
    loom_type_use_iterator_t iterator;
    loom_module_value_type_users(module_, provider, &iterator);
    for (auto carrier = loom_type_users_next(&iterator);
         carrier != LOOM_VALUE_ID_INVALID;
         carrier = loom_type_users_next(&iterator)) {
      result.push_back(carrier);
    }
    std::sort(result.begin(), result.end());
    EXPECT_EQ(std::adjacent_find(result.begin(), result.end()), result.end());
    EXPECT_EQ(loom_module_value_has_type_uses(module_, provider),
              !result.empty());
    return result;
  }

  void CheckEdges(iree_host_size_t expected_count) {
    EXPECT_EQ(module_->type_uses.active_carrier_count, expected_count);
    const auto users = Users(width_);
    EXPECT_EQ(users.size(), expected_count);
    for (auto user : users) {
      EXPECT_EQ(Dependencies(user), std::vector<loom_value_id_t>{width_});
    }
  }

  // Backing allocation ordinal to reject, or SIZE_MAX when disabled.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Backing requests since the current attempt's baseline.
  iree_host_size_t allocation_count_ = 0;
  // Small blocks expose record-page and type-payload allocation failures.
  iree_arena_block_pool_t pool_ = {};
  // Minimal context for value and canonical type APIs.
  loom_context_t context_ = {};
  // Module owning the dependency facts and active carriers.
  loom_module_t* module_ = nullptr;
  // Index-typed argument referenced by dependent argument types.
  loom_value_id_t width_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(ModuleTypeUsesTest, StaticTypesDoNotAllocateDependencyState) {
  for (uint32_t extent = 1; extent <= 256; ++extent) {
    AddArgument(loom_type_group_1d(loom_dim_pack_static(extent)));
  }
  EXPECT_EQ(module_->type_uses.index, nullptr);
  EXPECT_EQ(module_->type_uses.arena.used_allocation_size, 0u);
  EXPECT_EQ(module_->type_uses.arena.total_allocation_size, 0u);
  for (iree_host_size_t i = 0; i < module_->types.count; ++i) {
    EXPECT_EQ(loom_type_table_dependencies(&module_->types, i), 0u);
  }
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_FALSE(loom_module_has_active_type_uses(module_));
}

TEST_F(ModuleTypeUsesTest, RebuildAndReinsertReuseFragmentedCapacity) {
  std::array<loom_value_id_t, 256> arguments;
  const auto type = loom_type_group_1d(loom_dim_pack_dynamic(width_));
  for (auto& argument : arguments) {
    argument = AddArgument(type);
  }
  const auto used_bytes = module_->type_uses.arena.used_allocation_size;
  for (int iteration = 0; iteration < 3; ++iteration) {
    IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
    EXPECT_EQ(module_->type_uses.arena.used_allocation_size, used_bytes);
    CheckEdges(arguments.size());
  }
  for (size_t i = 0; i < arguments.size(); i += 2) {
    IREE_ASSERT_OK(loom_module_set_value_type(
        module_, arguments[i], loom_type_group_1d(loom_dim_pack_static(4))));
  }
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  CheckEdges(arguments.size() / 2);
  for (size_t i = 0; i < arguments.size(); i += 2) {
    EXPECT_TRUE(Dependencies(arguments[i]).empty());
    IREE_ASSERT_OK(loom_module_set_value_type(module_, arguments[i], type));
  }
  EXPECT_EQ(module_->type_uses.arena.used_allocation_size, used_bytes);
  CheckEdges(arguments.size());
}

TEST_F(ModuleTypeUsesTest, RepeatedSharedChildrenHaveOneDependency) {
  auto type_id = Intern(loom_type_group_1d(loom_dim_pack_dynamic(width_)));
  const auto singleton = loom_type_table_dependencies(&module_->types, type_id);
  const auto used_bytes = module_->type_uses.arena.used_allocation_size;
  for (int depth = 0; depth < 48; ++depth) {
    type_id = Pair(type_id, type_id);
    EXPECT_EQ(loom_type_table_dependencies(&module_->types, type_id),
              singleton);
  }
  EXPECT_EQ(module_->type_uses.arena.used_allocation_size, used_bytes);
  const auto carrier =
      AddArgument(loom_type_table_get(&module_->types, type_id));
  EXPECT_EQ(Dependencies(carrier), std::vector<loom_value_id_t>{width_});
  CheckEdges(1);
  loom_module_drop_value_type_uses(module_, carrier);
  EXPECT_TRUE(Dependencies(carrier).empty());
  EXPECT_TRUE(Users(width_).empty());
  IREE_ASSERT_OK(loom_module_refresh_value_type_uses(module_, carrier));
  CheckEdges(1);
  loom_module_drop_value_type_uses(module_, carrier);
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  CheckEdges(1);
}

TEST_F(ModuleTypeUsesTest, SharedStaticChildrenRetainEmptyMembership) {
  auto type_id = Intern(loom_type_group_1d(loom_dim_pack_static(4)));
  for (int depth = 0; depth < 48; ++depth) {
    type_id = Pair(type_id, type_id);
    EXPECT_EQ(loom_type_table_dependencies(&module_->types, type_id), 0u);
  }
  const auto carrier =
      AddArgument(loom_type_table_get(&module_->types, type_id));
  EXPECT_TRUE(Dependencies(carrier).empty());
  loom_module_drop_value_type_uses(module_, carrier);
  IREE_ASSERT_OK(loom_module_refresh_value_type_uses(module_, carrier));
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_FALSE(loom_module_has_active_type_uses(module_));
  EXPECT_EQ(module_->type_uses.index, nullptr);
  EXPECT_EQ(module_->type_uses.arena.used_allocation_size, 0u);
  EXPECT_EQ(module_->type_uses.arena.total_allocation_size, 0u);
}

TEST_F(ModuleTypeUsesTest, MembershipIsCanonicalButOccurrenceWalkingIsOrdered) {
  const auto height = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const auto forward = Intern(Matrix(width_, height));
  const auto reverse = Intern(Matrix(height, width_));
  EXPECT_EQ(loom_type_table_dependencies(&module_->types, forward),
            loom_type_table_dependencies(&module_->types, reverse));
  auto pair = Pair(reverse, reverse);
  const auto carrier = AddArgument(loom_type_table_get(&module_->types, pair));
  EXPECT_EQ(Dependencies(carrier),
            (std::vector<loom_value_id_t>{width_, height}));
  std::vector<loom_value_id_t> occurrences;
  IREE_ASSERT_OK(loom_type_walk_value_refs(
      module_, loom_type_table_get(&module_->types, pair),
      [](loom_value_id_t provider, void* user_data) {
        static_cast<std::vector<loom_value_id_t>*>(user_data)->push_back(
            provider);
        return iree_ok_status();
      },
      &occurrences));
  EXPECT_EQ(occurrences,
            (std::vector<loom_value_id_t>{height, width_, height, width_}));
}

TEST_F(ModuleTypeUsesTest, OverlapDoesNotMultiplyIncomingCarriers) {
  const auto height = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const auto depth = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const auto left = Intern(Matrix(width_, height));
  const auto right = Intern(Matrix(height, depth));
  const auto common = Pair(left, right);
  auto root = common;
  std::vector<loom_value_id_t> carriers;
  for (int i = 0; i < 128; ++i) {
    root = Pair(root, common);
    carriers.push_back(AddArgument(loom_type_table_get(&module_->types, root)));
    EXPECT_EQ(Dependencies(carriers.back()),
              (std::vector<loom_value_id_t>{width_, height, depth}));
  }
  for (auto provider : {width_, height, depth}) {
    EXPECT_EQ(Users(provider), carriers);
  }
  for (size_t i = 0; i < carriers.size(); i += 2) {
    loom_module_drop_value_type_uses(module_, carriers[i]);
  }
  std::vector<loom_value_id_t> retained;
  for (size_t i = 1; i < carriers.size(); i += 2) {
    retained.push_back(carriers[i]);
  }
  for (auto provider : {width_, height, depth}) {
    EXPECT_EQ(Users(provider), retained);
  }
}

TEST_F(ModuleTypeUsesTest, ForwardProvidersAndHighIdsRemainDeclared) {
  auto left = Intern(Matrix(width_, 4));
  auto right = Intern(Matrix(8, UINT32_MAX - 1));
  auto root = Pair(left, right);
  const auto carrier = AddArgument(loom_type_table_get(&module_->types, root));
  EXPECT_EQ(Dependencies(carrier), std::vector<loom_value_id_t>{width_});
  for (int i = 0; i < 3; ++i) {
    AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }
  IREE_ASSERT_OK(loom_module_refresh_value_type_uses(module_, carrier));
  EXPECT_EQ(Dependencies(carrier), (std::vector<loom_value_id_t>{width_, 4}));
  for (int i = 0; i < 4; ++i) {
    AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }
  loom_module_drop_value_type_uses(module_, carrier);
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_EQ(Dependencies(carrier),
            (std::vector<loom_value_id_t>{width_, 4, 8}));
  EXPECT_TRUE(Users(UINT32_MAX - 1).empty());
  EXPECT_TRUE(Dependencies(LOOM_VALUE_ID_INVALID).empty());
}

TEST_F(ModuleTypeUsesTest, DefinitionIncludesTheNewlyPublishedValue) {
  // Construction indexes references before semantic verification. The declared
  // set must match the published value range, including the new slot itself.
  const auto next_value = static_cast<loom_value_id_t>(module_->values.count);
  loom_value_id_t carrier = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_define_value(module_, Matrix(width_, next_value), &carrier));
  ASSERT_EQ(carrier, next_value);
  EXPECT_EQ(Dependencies(carrier),
            (std::vector<loom_value_id_t>{width_, carrier}));
  EXPECT_EQ(Users(carrier), std::vector<loom_value_id_t>{carrier});
  loom_module_drop_value_type_uses(module_, carrier);
  EXPECT_TRUE(Users(carrier).empty());
}

TEST_F(ModuleTypeUsesTest, RandomizedAssignmentDropAndRefreshMatchSets) {
  constexpr uint32_t kProviderCount = 64;
  constexpr uint32_t kCarrierCount = 24;
  for (uint32_t i = 1; i < kProviderCount; ++i) {
    AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }
  std::array<loom_value_id_t, kCarrierCount> carriers;
  for (auto& carrier : carriers) {
    carrier = AddArgument(loom_type_group_1d(loom_dim_pack_static(4)));
  }
  std::array<std::set<loom_value_id_t>, kCarrierCount> declared;
  std::array<std::set<loom_value_id_t>, kCarrierCount> active;
  std::mt19937 random(927141);
  for (uint32_t step = 0; step < 1024; ++step) {
    SCOPED_TRACE(step);
    const uint32_t index = random() % kCarrierCount;
    const auto carrier = carriers[index];
    switch (random() % 4) {
      case 0:
        loom_module_drop_value_type_uses(module_, carrier);
        active[index].clear();
        break;
      case 1:
        IREE_ASSERT_OK(loom_module_refresh_value_type_uses(module_, carrier));
        active[index] = declared[index];
        break;
      default: {
        const loom_value_id_t providers[] = {
            static_cast<loom_value_id_t>(random() % kProviderCount),
            static_cast<loom_value_id_t>(random() % kProviderCount),
            static_cast<loom_value_id_t>(random() % kProviderCount),
            static_cast<loom_value_id_t>(random() % kProviderCount)};
        const auto left = Intern(Matrix(providers[0], providers[1]));
        const auto right = Intern(Matrix(providers[2], providers[3]));
        const auto pair = Pair(left, right);
        IREE_ASSERT_OK(loom_module_set_value_type(
            module_, carrier, loom_type_table_get(&module_->types, pair)));
        declared[index] = std::set<loom_value_id_t>(std::begin(providers),
                                                    std::end(providers));
        active[index] = declared[index];
        break;
      }
    }
    EXPECT_EQ(Dependencies(carrier),
              std::vector<loom_value_id_t>(active[index].begin(),
                                           active[index].end()));
    for (uint32_t provider = 0; provider < kProviderCount; ++provider) {
      std::vector<loom_value_id_t> expected;
      for (uint32_t i = 0; i < kCarrierCount; ++i) {
        if (active[i].count(provider)) {
          expected.push_back(carriers[i]);
        }
      }
      EXPECT_EQ(Users(provider), expected);
    }
  }
}

TEST_F(ModuleTypeUsesTest, SharedTypeDagsMatchIndependentMembership) {
  constexpr uint32_t kProviderCount = 32;
  constexpr uint32_t kCarrierCount = 16;
  std::mt19937 random(410197);
  for (uint32_t trial = 0; trial < 16; ++trial) {
    SCOPED_TRACE(trial);
    ASSERT_NO_FATAL_FAILURE(ResetModule());
    for (uint32_t i = 1; i < kProviderCount; ++i) {
      AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
    }
    std::array<loom_value_id_t, kCarrierCount> carriers;
    std::array<std::set<loom_value_id_t>, kCarrierCount> active;
    for (auto& carrier : carriers) {
      carrier = AddArgument(loom_type_group_1d(loom_dim_pack_static(4)));
    }
    std::vector<loom_type_id_t> types;
    std::vector<std::set<loom_value_id_t>> members;
    for (uint32_t provider = 0; provider < kProviderCount; ++provider) {
      types.push_back(
          Intern(loom_type_group_1d(loom_dim_pack_dynamic(provider))));
      members.push_back({provider});
    }
    for (uint32_t step = 0; step < 256; ++step) {
      const auto left = random() % types.size();
      const auto right = random() % types.size();
      const auto root = Pair(types[left], types[right]);
      auto expected = members[left];
      expected.insert(members[right].begin(), members[right].end());
      types.push_back(root);
      members.push_back(expected);
      const auto owner = random() % kCarrierCount;
      IREE_ASSERT_OK(loom_module_set_value_type(
          module_, carriers[owner],
          loom_type_table_get(&module_->types, root)));
      active[owner] = expected;
      EXPECT_EQ(Dependencies(carriers[owner]),
                std::vector<loom_value_id_t>(expected.begin(), expected.end()));
      if (step % 8 == 0) {
        const auto dropped = random() % kCarrierCount;
        loom_module_drop_value_type_uses(module_, carriers[dropped]);
        active[dropped].clear();
      }
      for (uint32_t provider = 0; provider < kProviderCount; ++provider) {
        std::vector<loom_value_id_t> users;
        for (uint32_t i = 0; i < kCarrierCount; ++i) {
          if (active[i].count(provider)) {
            users.push_back(carriers[i]);
          }
        }
        EXPECT_EQ(Users(provider), users);
      }
    }
  }
}

TEST_F(ModuleTypeUsesTest, FailedAssignmentPreservesTypeAndOwnership) {
  iree_host_size_t requests = 0;
  for (iree_host_size_t attempt = 0; attempt <= requests; ++attempt) {
    ASSERT_NO_FATAL_FAILURE(ResetModule());
    const auto height = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
    const auto carrier =
        AddArgument(loom_type_group_1d(loom_dim_pack_dynamic(width_)));
    const auto old_type = loom_module_value_type(module_, carrier);
    const auto replacement = Matrix(width_, height);
    allocation_count_ = 0;
    failure_index_ = attempt == 0 ? SIZE_MAX : attempt - 1;
    auto status = loom_module_set_value_type(module_, carrier, replacement);
    failure_index_ = SIZE_MAX;
    if (attempt == 0) {
      IREE_ASSERT_OK(status);
      requests = allocation_count_;
      ASSERT_GT(requests, 0u);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_TRUE(
          loom_type_equal(loom_module_value_type(module_, carrier), old_type));
      CheckEdges(1);
      EXPECT_TRUE(Users(height).empty());
      IREE_ASSERT_OK(loom_module_set_value_type(module_, carrier, replacement));
    }
    EXPECT_EQ(Dependencies(carrier),
              (std::vector<loom_value_id_t>{width_, height}));
    EXPECT_EQ(Users(height), std::vector<loom_value_id_t>{carrier});
  }
}

TEST_F(ModuleTypeUsesTest, FailedCarrierGrowthDoesNotDefineAValue) {
  const auto type = loom_type_group_1d(loom_dim_pack_dynamic(width_));
  for (uint32_t i = 0; i < 128; ++i) {
    AddArgument(type);
  }
  const auto count = module_->values.count;
  allocation_count_ = 0;
  failure_index_ = 0;
  loom_value_id_t carrier = LOOM_VALUE_ID_INVALID;
  auto status = loom_module_define_value(module_, type, &carrier);
  failure_index_ = SIZE_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(module_->values.count, count);
  CheckEdges(128);
  IREE_ASSERT_OK(loom_module_define_value(module_, type, &carrier));
  EXPECT_EQ(carrier, count);
  CheckEdges(129);
}

TEST_F(ModuleTypeUsesTest, FailedBulkPrefixGrowthPreservesAllOwnership) {
  const auto left = Intern(Matrix(width_, 4));
  const auto right = Intern(Matrix(8, 12));
  const auto root = Pair(left, right);
  const auto first = AddArgument(loom_type_table_get(&module_->types, root));
  const auto second = AddArgument(loom_type_table_get(&module_->types, root));
  // Seven canonical set nodes plus these forward singletons fill the first
  // 128-record page. The new [0, 9) prefix requires another canonical node.
  for (uint32_t i = 0; i < 121; ++i) {
    Intern(loom_type_group_1d(loom_dim_pack_dynamic(1024 + i)));
  }
  while (module_->values.count < 9) {
    AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }
  CheckEdges(2);
  allocation_count_ = 0;
  failure_index_ = 0;
  auto status = loom_module_recompute_type_uses(module_);
  failure_index_ = SIZE_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  CheckEdges(2);
  EXPECT_TRUE(Users(4).empty());
  EXPECT_TRUE(Users(8).empty());
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_EQ(Dependencies(first), (std::vector<loom_value_id_t>{width_, 4, 8}));
  EXPECT_EQ(Dependencies(second), Dependencies(first));
  EXPECT_EQ(Users(8), (std::vector<loom_value_id_t>{first, second}));
}

TEST_F(ModuleTypeUsesTest,
       ParameterizedRollbackCannotInvalidateDependencyFacts) {
  iree_host_size_t requests = 0;
  for (iree_host_size_t attempt = 0; attempt <= requests; ++attempt) {
    ASSERT_NO_FATAL_FAILURE(ResetModule());
    const auto height = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
    const auto depth = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
    const auto first = Intern(Matrix(width_, height));
    const auto second = Intern(Matrix(height, depth));
    for (uint32_t i = 0; i < 123; ++i) {
      Intern(loom_type_group_1d(loom_dim_pack_dynamic(1024 + i)));
    }
    Intern(loom_type_group_1d(loom_dim_pack_static(4)));
    loom_string_id_t key = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("nested"), &key));
    const loom_named_attr_t metadata[] = {
        {.name_id = key, .reserved = 0, .value = loom_attr_type(second)}};
    const loom_attribute_t parameters[] = {
        loom_attr_type(first), loom_attr_absent(),
        loom_make_canonical_attr_dict(metadata, IREE_ARRAYSIZE(metadata))};
    const auto types = module_->types;
    const auto payload_bytes = module_->arena.used_allocation_size;
    allocation_count_ = 0;
    failure_index_ = attempt == 0 ? SIZE_MAX : attempt - 1;
    loom_type_t result = {};
    loom_type_id_t result_id = LOOM_TYPE_ID_INVALID;
    auto status = loom_module_make_parameterized_type(
        module_, &loom_test_array_type_parameterized_descriptor, parameters,
        IREE_ARRAYSIZE(parameters), &result, &result_id);
    failure_index_ = SIZE_MAX;
    if (attempt == 0) {
      IREE_ASSERT_OK(status);
      requests = allocation_count_;
      ASSERT_GT(requests, 0u);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(module_->types.segments.segment_count,
                types.segments.segment_count);
      EXPECT_EQ(loom_type_table_entry(&module_->types, first),
                loom_type_table_entry(&types, first));
      EXPECT_EQ(loom_type_table_dependencies(&module_->types, first),
                loom_type_table_dependencies(&types, first));
      EXPECT_EQ(module_->types.count, types.count);
      EXPECT_EQ(module_->arena.used_allocation_size, payload_bytes);
      EXPECT_FALSE(loom_module_has_active_type_uses(module_));
      IREE_ASSERT_OK(loom_module_make_parameterized_type(
          module_, &loom_test_array_type_parameterized_descriptor, parameters,
          IREE_ARRAYSIZE(parameters), &result, &result_id));
    }
    const auto carrier = AddArgument(result);
    EXPECT_EQ(Dependencies(carrier),
              (std::vector<loom_value_id_t>{width_, height, depth}));
    EXPECT_EQ(Users(width_), std::vector<loom_value_id_t>{carrier});
    const auto retained_bytes = module_->type_uses.arena.used_allocation_size;
    const auto retained_root =
        loom_type_table_dependencies(&module_->types, result_id);
    IREE_ASSERT_OK(loom_module_make_parameterized_type(
        module_, &loom_test_array_type_parameterized_descriptor, parameters,
        IREE_ARRAYSIZE(parameters), &result, &result_id));
    EXPECT_EQ(loom_type_table_dependencies(&module_->types, result_id),
              retained_root);
    EXPECT_EQ(module_->type_uses.arena.used_allocation_size, retained_bytes);
  }
}

}  // namespace
}  // namespace loom
