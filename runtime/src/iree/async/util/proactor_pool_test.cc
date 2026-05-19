// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t RejectProactorCreation(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  (void)options;
  (void)allocator;
  *out_proactor = nullptr;
  return iree_make_status(IREE_STATUS_ABORTED, "selected creator invoked");
}

class ProactorPoolTest : public ::testing::Test {
 protected:
  iree_async_proactor_pool_options_t default_options() {
    return iree_async_proactor_pool_options_default();
  }
};

struct TestRunnerState {
  // Number of runner handles created by the factory.
  int create_count = 0;

  // Number of runner handles passed to request_stop.
  int request_stop_count = 0;

  // Number of runner handles destroyed by the factory.
  int destroy_count = 0;
};

static iree_status_t TestRunnerCreate(void* user_data,
                                      iree_async_proactor_t* proactor,
                                      uint32_t node_id,
                                      iree_allocator_t allocator,
                                      void** out_runner) {
  (void)proactor;
  (void)node_id;
  (void)allocator;
  TestRunnerState* state = (TestRunnerState*)user_data;
  ++state->create_count;
  *out_runner = user_data;
  return iree_ok_status();
}

static void TestRunnerRequestStop(void* user_data, void** runners,
                                  iree_host_size_t count) {
  (void)user_data;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (runners[i]) {
      TestRunnerState* state = (TestRunnerState*)runners[i];
      ++state->request_stop_count;
    }
  }
}

static void TestRunnerDestroy(void* user_data, void* runner) {
  (void)user_data;
  TestRunnerState* state = (TestRunnerState*)runner;
  ++state->destroy_count;
}

TEST_F(ProactorPoolTest, CreateZeroNodesFails) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_pool_create(
                            0, /*node_ids=*/nullptr, default_options(),
                            iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(ProactorPoolTest, CreateSingleNodeNoAffinity) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 0), UINT32_MAX);

  // Out of bounds returns error.
  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_async_proactor_pool_get(pool, 1, &proactor));
  EXPECT_EQ(proactor, nullptr);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 1), UINT32_MAX);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, CreateAndReleaseWithoutGet) {
  // Pool creation is lazy — creating and releasing without ever calling get
  // should be free (no threads spawned).
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);
  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, SelectedCreatorIsLazyAndPropagatesFailure) {
  iree_async_proactor_pool_options_t options = default_options();
  options.proactor_create = RejectProactorCreation;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, options, iree_allocator_system(), &pool));

  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_async_proactor_pool_get(pool, 0, &proactor));
  EXPECT_EQ(proactor, nullptr);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, OnDemandGet) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));

  // First get creates the proactor on-demand.
  iree_async_proactor_t* proactor = nullptr;
  iree_status_t status = iree_async_proactor_pool_get(pool, 0, &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  EXPECT_NE(proactor, nullptr);

  // Second get returns the same proactor (already created).
  iree_async_proactor_t* proactor_again = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor_again));
  EXPECT_EQ(proactor, proactor_again);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, CreateWithNodeIds) {
  uint32_t node_ids[] = {0, 1};
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      2, node_ids, default_options(), iree_allocator_system(), &pool));
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(iree_async_proactor_pool_count(pool), 2u);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 0), 0u);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 1), 1u);

  // Each proactor is distinct (on-demand creation).
  iree_async_proactor_t* proactor_0 = nullptr;
  iree_async_proactor_t* proactor_1 = nullptr;
  iree_status_t status = iree_async_proactor_pool_get(pool, 0, &proactor_0);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 1, &proactor_1));
  EXPECT_NE(proactor_0, nullptr);
  EXPECT_NE(proactor_1, nullptr);
  EXPECT_NE(proactor_0, proactor_1);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, GetForNodeExactMatch) {
  uint32_t node_ids[] = {3, 7};
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      2, node_ids, default_options(), iree_allocator_system(), &pool));

  // Exact match returns the right proactor.
  iree_async_proactor_t* proactor_3 = nullptr;
  iree_async_proactor_t* proactor_7 = nullptr;
  iree_status_t status =
      iree_async_proactor_pool_get_for_node(pool, 3, &proactor_3);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(iree_async_proactor_pool_get_for_node(pool, 7, &proactor_7));

  iree_async_proactor_t* proactor_0 = nullptr;
  iree_async_proactor_t* proactor_1 = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor_0));
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 1, &proactor_1));
  EXPECT_EQ(proactor_3, proactor_0);
  EXPECT_EQ(proactor_7, proactor_1);

  // No match falls back to the first proactor.
  iree_async_proactor_t* proactor_99 = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get_for_node(pool, 99, &proactor_99));
  EXPECT_EQ(proactor_99, proactor_0);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, RetainRelease) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));

  // Extra retain keeps the pool alive.
  iree_async_proactor_pool_retain(pool);
  iree_async_proactor_pool_release(pool);  // Drops to ref count 1.

  // Pool should still be usable.
  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);

  iree_async_proactor_pool_release(pool);  // Final release, destroys.
}

TEST_F(ProactorPoolTest, AcquiredEntryKeepsRunnerAfterPoolRelease) {
  TestRunnerState runner_state;
  iree_async_proactor_pool_options_t options = default_options();
  memset(&options.runner, 0, sizeof(options.runner));
  options.runner.user_data = &runner_state;
  options.runner.create = TestRunnerCreate;
  options.runner.request_stop = TestRunnerRequestStop;
  options.runner.destroy = TestRunnerDestroy;

  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, options, iree_allocator_system(), &pool));

  // Trigger on-demand creation and acquire the entry.
  iree_async_proactor_pool_entry_t* entry = nullptr;
  iree_status_t status = iree_async_proactor_pool_acquire(pool, 0, &entry);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  ASSERT_NE(entry, nullptr);
  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(entry);
  ASSERT_NE(proactor, nullptr);
  EXPECT_EQ(runner_state.create_count, 1);

  // Release the aggregate pool. The acquired entry owns the runner lifetime, so
  // the pool release must not stop it.
  iree_async_proactor_pool_release(pool);
  EXPECT_EQ(runner_state.request_stop_count, 0);
  EXPECT_EQ(runner_state.destroy_count, 0);

  iree_async_proactor_pool_entry_release(entry);
  EXPECT_EQ(runner_state.request_stop_count, 1);
  EXPECT_EQ(runner_state.destroy_count, 1);
}

}  // namespace
