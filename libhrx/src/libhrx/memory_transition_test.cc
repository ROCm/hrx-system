// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"

namespace {

testing::AssertionResult Check(hrx_status_t status) {
  auto result = hrx_status_is_ok(status) ? testing::AssertionSuccess()
                                         : testing::AssertionFailure();
  if (!hrx_status_is_ok(status)) {
    result << hrx::runtime::format_status(status);
  }
  hrx_status_ignore(status);
  return result;
}

class MemoryTransitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                         runtime_.for_output())));
    const hrx_device_create_params_t devices[] = {
        {.uri = "task://"},
        {.uri = "task://"},
    };
    ASSERT_TRUE(Check(
        hrx_device_group_create(runtime_, 2, devices, group_.for_output())));
    for (size_t i = 0; i < 2; ++i) {
      families_[i].device = hrx_device_group_get_device(group_, i);
      families_[i].usage = HRX_BUFFER_USAGE_DEFAULT;
      ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
          families_[i].device, 0, HRX_MEMORY_SITE_QUEUE, &scopes_[i])));
    }
    ASSERT_TRUE(Check(CreatePool(2, pool_.for_output())));
  }

  hrx_status_t CreatePool(size_t family_count, hrx_mem_pool_t* out_pool) {
    const hrx_pool_scope_t scope = {
        .family_count = family_count,
        .families = families_,
        .host =
            {
                .access = HRX_MEMORY_ACCESS_ALL,
                .modes = HRX_MAPPING_MODE_SCOPED,
            },
    };
    return hrx_slab_pool_create(group_, scope, nullptr, out_pool);
  }

  // Runtime owner shared by the two explicitly created CPU devices.
  hrx::runtime::runtime_ptr runtime_;
  // Sealed domain defining both queue coordinates.
  hrx::runtime::device_group_ptr group_;
  // Pool retaining its complete contract independently of application owners.
  hrx::runtime::mem_pool_ptr pool_;
  // Construction access for the two task queue families.
  hrx_pool_family_access_t families_[2] = {};
  // Prepared exact queue sites in the same domain.
  hrx_memory_scope_t scopes_[2] = {};
};

TEST_F(MemoryTransitionTest, BorrowedContractSurvivesApplicationOwnerRelease) {
  auto table = hrx_mem_pool_transition_table(pool_);
  ASSERT_NE(table.cells, nullptr);
  hrx_memory_transition_pair_t pair;
  ASSERT_TRUE(Check(hrx_memory_transition_prepare_pair(
      table, scopes_[0], scopes_[1], HRX_MEMORY_TRANSITION_RELEASE, &pair)));
  group_.reset();
  runtime_.reset();
  const auto transition = hrx_memory_transition_query(table, pair);
  EXPECT_TRUE(hrx_memory_effects_is_supported(transition.release));
  EXPECT_TRUE(hrx_memory_effects_is_empty(transition.release));
  EXPECT_TRUE(hrx_memory_effects_is_empty(transition.acquire));
  EXPECT_FALSE(hrx_memory_effects_requires_resources(transition.acquire));
  EXPECT_FALSE(hrx_memory_effects_requires_program(transition.acquire));
  EXPECT_EQ(
      hrx_memory_transition_recipe(table, pair, HRX_MEMORY_TRANSITION_RELEASE),
      nullptr);
}

TEST_F(MemoryTransitionTest, SameGroupKeyKeepsEachPoolsQualification) {
  auto table = hrx_mem_pool_transition_table(pool_);
  hrx_memory_transition_pair_t pair;
  ASSERT_TRUE(Check(hrx_memory_transition_prepare_pair(
      table, scopes_[0], scopes_[1], HRX_MEMORY_TRANSITION_ACQUIRE, &pair)));
  hrx::runtime::mem_pool_ptr narrow_pool;
  ASSERT_TRUE(Check(CreatePool(1, narrow_pool.for_output())));
  const auto narrow = hrx_memory_transition_query(
      hrx_mem_pool_transition_table(narrow_pool), pair);
  EXPECT_FALSE(hrx_memory_effects_is_supported(narrow.release));
  EXPECT_FALSE(hrx_memory_effects_is_supported(narrow.acquire));
  const auto wide = hrx_memory_transition_query(table, pair);
  EXPECT_FALSE(hrx_memory_effects_is_supported(
      hrx_memory_effects_combine(wide.acquire, narrow.acquire)));
}

TEST_F(MemoryTransitionTest, WildcardsKeepNamespaceAndExactLocalAction) {
  auto table = hrx_mem_pool_transition_table(pool_);
  auto any = hrx_memory_scope_any(scopes_[0]);
  hrx_memory_transition_pair_t pair;
  ASSERT_TRUE(Check(hrx_memory_transition_prepare_pair(
      table, any, scopes_[1], HRX_MEMORY_TRANSITION_ACQUIRE, &pair)));
  auto transition = hrx_memory_transition_query(table, pair);
  EXPECT_TRUE(hrx_memory_effects_is_empty(transition.acquire));
  EXPECT_FALSE(hrx_memory_effects_is_supported(transition.release));
  ASSERT_TRUE(Check(hrx_memory_transition_prepare_pair(
      table, scopes_[0], any, HRX_MEMORY_TRANSITION_RELEASE, &pair)));
  transition = hrx_memory_transition_query(table, pair);
  EXPECT_TRUE(hrx_memory_effects_is_empty(transition.release));
  EXPECT_FALSE(hrx_memory_effects_is_supported(transition.acquire));
  const hrx_memory_transition_action_t actions[] = {
      HRX_MEMORY_TRANSITION_RELEASE,
      HRX_MEMORY_TRANSITION_ACQUIRE,
  };
  for (auto action : actions) {
    auto status =
        hrx_memory_transition_prepare_pair(table, any, any, action, &pair);
    EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
    hrx_status_ignore(status);
  }
  auto status = hrx_memory_transition_prepare_pair(
      table, {}, scopes_[1], HRX_MEMORY_TRANSITION_ACQUIRE, &pair);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_PERMISSION_DENIED);
  hrx_status_ignore(status);
}

TEST_F(MemoryTransitionTest, ForeignGroupAndMissingOutputAreRejected) {
  const hrx_device_create_params_t device = {.uri = "task://"};
  hrx::runtime::device_group_ptr other_group;
  ASSERT_TRUE(Check(
      hrx_device_group_create(runtime_, 1, &device, other_group.for_output())));
  hrx_memory_scope_t other_scope;
  ASSERT_TRUE(Check(hrx_device_query_queue_family_memory_scope(
      hrx_device_group_get_device(other_group, 0), 0, HRX_MEMORY_SITE_QUEUE,
      &other_scope)));
  hrx_memory_transition_pair_t pair;
  const auto table = hrx_mem_pool_transition_table(pool_);
  auto status = hrx_memory_transition_prepare_pair(
      table, scopes_[0], other_scope, HRX_MEMORY_TRANSITION_RELEASE, &pair);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_PERMISSION_DENIED);
  hrx_status_ignore(status);
  status = hrx_memory_transition_prepare_pair(
      table, scopes_[0], scopes_[1], HRX_MEMORY_TRANSITION_RELEASE, nullptr);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
}

TEST_F(MemoryTransitionTest, HostMaintenanceRequiresActiveMappings) {
  const hrx_buffer_mapping_transition_t missing = {};
  auto status = hrx_buffer_mapping_memory_barrier({}, 1, &missing);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_FAILED_PRECONDITION);
  hrx_status_ignore(status);
  status = hrx_buffer_mapping_memory_barrier({}, 1, nullptr);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_TRUE(Check(hrx_buffer_mapping_memory_barrier({}, 0, nullptr)));
}

}  // namespace
