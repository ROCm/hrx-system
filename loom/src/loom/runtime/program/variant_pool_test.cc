// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/variant_pool.h"

#include <atomic>
#include <string>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class VariantPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_runtime_variant_pool_allocate(
        4, nullptr, Dispose, iree_allocator_system(), &original_, &pool_));
    original_id_ = loom_runtime_variant_entry_id(original_);
  }

  void TearDown() override { loom_runtime_variant_pool_free(pool_); }

  static void Dispose(void* value) {
    if (value) {
      ++*static_cast<unsigned*>(value);
    }
  }

  // Owner-thread pool, freed after all accepted producers and consumers retire.
  loom_runtime_variant_pool_t* pool_ = nullptr;
  // Borrowed initial entry, with no implicit consumer pin.
  loom_runtime_variant_entry_t* original_ = nullptr;
  // Nonrecycled provenance, valid even after the entry retires.
  loom_runtime_variant_id_t original_id_ = 0;
  // Payload disposer count; each test releases payloads before this retires.
  unsigned disposed_ = 0;
};

TEST_F(VariantPoolTest,
       CallbackPublishesWithoutSelectingOrExposingPartialData) {
  std::atomic<unsigned> wakes = 0;
  loom_runtime_variant_pool_set_observer(
      pool_, {[](void* user_data) {
                ++*static_cast<std::atomic<unsigned>*>(user_data);
              },
              &wakes});
  loom_runtime_variant_entry_t* pending = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_variant_pool_begin(pool_, original_id_, &pending));
  const auto id = loom_runtime_variant_entry_id(pending);
  EXPECT_TRUE(loom_runtime_variant_pool_pending(pool_));
  EXPECT_FALSE(loom_runtime_variant_pool_collect(pool_));
  loom_runtime_variant_entry_t* borrowed = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_runtime_variant_pool_acquire(pool_, id, &borrowed));
  EXPECT_EQ(borrowed, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_runtime_variant_pool_remove(pool_, id));

  std::thread producer([&] {
    loom_runtime_compilation_report_t report = {.kernels = 17};
    loom_runtime_variant_entry_complete(pending, iree_ok_status(), &disposed_,
                                        &report);
  });
  producer.join();
  EXPECT_EQ(wakes, 1u);
  loom_runtime_variant_snapshot_t snapshot = {};
  ASSERT_TRUE(loom_runtime_variant_pool_snapshot(pool_, id, &snapshot));
  EXPECT_EQ(snapshot.phase, LOOM_RUNTIME_VARIANT_COMPILING);
  EXPECT_EQ(snapshot.compilation, nullptr);
  EXPECT_EQ(snapshot.users, 0u);
  EXPECT_TRUE(loom_runtime_variant_pool_pending(pool_));
  EXPECT_TRUE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_FALSE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_FALSE(loom_runtime_variant_pool_pending(pool_));
  ASSERT_TRUE(loom_runtime_variant_pool_snapshot(pool_, id, &snapshot));
  EXPECT_EQ(snapshot.phase, LOOM_RUNTIME_VARIANT_READY);
  EXPECT_EQ(snapshot.compilation->kernels, 17u);
  EXPECT_EQ(snapshot.code, IREE_STATUS_OK);
  EXPECT_EQ(snapshot.users, 0u);
  IREE_ASSERT_OK(loom_runtime_variant_pool_acquire(pool_, id, &borrowed));
  EXPECT_EQ(loom_runtime_variant_entry_value(borrowed), &disposed_);
  loom_runtime_variant_entry_retain(borrowed);
  loom_runtime_variant_entry_release(borrowed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_runtime_variant_pool_remove(pool_, id));
  loom_runtime_variant_entry_release(borrowed);
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(pool_, id));
  EXPECT_EQ(disposed_, 1u);
  EXPECT_EQ(loom_runtime_variant_pool_count(pool_), 1u);
  loom_runtime_variant_pool_set_observer(pool_, {});
}

TEST_F(VariantPoolTest, CapacityIncludesProducersAndRetainedFailures) {
  loom_runtime_variant_entry_t* entries[3] = {};
  for (auto& entry : entries) {
    IREE_ASSERT_OK(
        loom_runtime_variant_pool_begin(pool_, original_id_, &entry));
  }
  loom_runtime_variant_entry_t* extra = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_variant_pool_begin(pool_, original_id_, &extra));
  EXPECT_EQ(extra, nullptr);
  const auto failed_id = loom_runtime_variant_entry_id(entries[0]);
  loom_runtime_compilation_report_t report = {.diagnostic_count = 1};
  report.diagnostics[0].line = 23;
  loom_runtime_variant_entry_complete(
      entries[0], iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "bad program"),
      &disposed_, &report);
  EXPECT_TRUE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_EQ(disposed_, 1u);
  loom_runtime_variant_snapshot_t snapshot = {};
  ASSERT_TRUE(loom_runtime_variant_pool_snapshot(pool_, failed_id, &snapshot));
  EXPECT_EQ(snapshot.phase, LOOM_RUNTIME_VARIANT_FAILED);
  EXPECT_EQ(snapshot.code, IREE_STATUS_INVALID_ARGUMENT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_variant_entry_status(entries[0]));
  EXPECT_EQ(std::string(snapshot.message.data, snapshot.message.size),
            "bad program");
  EXPECT_EQ(snapshot.compilation->diagnostics[0].line, 23u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_variant_pool_begin(pool_, original_id_, &extra));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_runtime_variant_pool_begin(pool_, failed_id, &extra));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_runtime_variant_pool_acquire(pool_, failed_id, &extra));
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(pool_, failed_id));
  EXPECT_EQ(disposed_, 1u);
  IREE_ASSERT_OK(loom_runtime_variant_pool_begin(pool_, original_id_, &extra));
  EXPECT_NE(loom_runtime_variant_entry_id(extra), failed_id);
  for (auto* entry : {entries[1], entries[2], extra}) {
    loom_runtime_variant_entry_complete(entry, iree_ok_status(), nullptr,
                                        nullptr);
  }
  EXPECT_TRUE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_FALSE(loom_runtime_variant_pool_pending(pool_));
}

TEST_F(VariantPoolTest, ProvenanceDoesNotImplicitlyPinCodeIncludingOriginal) {
  loom_runtime_variant_entry_t* child = nullptr;
  IREE_ASSERT_OK(loom_runtime_variant_pool_begin(pool_, original_id_, &child));
  const auto child_id = loom_runtime_variant_entry_id(child);
  loom_runtime_variant_entry_complete(child, iree_ok_status(), &disposed_,
                                      nullptr);
  ASSERT_TRUE(loom_runtime_variant_pool_collect(pool_));
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(pool_, original_id_));
  loom_runtime_variant_snapshot_t snapshot = {};
  ASSERT_TRUE(loom_runtime_variant_pool_at(pool_, 0, &snapshot));
  EXPECT_EQ(snapshot.id, child_id);
  EXPECT_EQ(snapshot.base, original_id_);
  EXPECT_FALSE(loom_runtime_variant_pool_at(pool_, 1, &snapshot));
  EXPECT_FALSE(
      loom_runtime_variant_pool_snapshot(pool_, original_id_, &snapshot));

  loom_runtime_variant_entry_t* grandchild = nullptr;
  IREE_ASSERT_OK(loom_runtime_variant_pool_begin(pool_, child_id, &grandchild));
  const auto grandchild_id = loom_runtime_variant_entry_id(grandchild);
  // Construction owns its resource borrows separately. Provenance is not a
  // hidden borrow of the previous executable's payload.
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(pool_, child_id));
  EXPECT_EQ(disposed_, 1u);
  loom_runtime_variant_entry_complete(grandchild, iree_ok_status(), nullptr,
                                      nullptr);
  ASSERT_TRUE(loom_runtime_variant_pool_collect(pool_));
  ASSERT_TRUE(loom_runtime_variant_pool_at(pool_, 0, &snapshot));
  EXPECT_EQ(snapshot.id, grandchild_id);
  EXPECT_EQ(snapshot.base, child_id);
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(pool_, grandchild_id));
  EXPECT_EQ(loom_runtime_variant_pool_count(pool_), 0u);
}

TEST_F(VariantPoolTest,
       OtherPoolsAndRecreatedRegistrationsCannotReuseIdentity) {
  loom_runtime_variant_id_t previous = original_id_;
  for (unsigned i = 0; i < 2; ++i) {
    loom_runtime_variant_pool_t* other = nullptr;
    loom_runtime_variant_entry_t* original = nullptr;
    IREE_ASSERT_OK(loom_runtime_variant_pool_allocate(
        1, nullptr, Dispose, iree_allocator_system(), &original, &other));
    const auto id = loom_runtime_variant_entry_id(original);
    EXPECT_GT(id, previous);
    previous = id;
    loom_runtime_variant_entry_t* borrowed = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_NOT_FOUND,
        loom_runtime_variant_pool_acquire(pool_, id, &borrowed));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_runtime_variant_pool_begin(other, original_id_, &borrowed));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_NOT_FOUND,
        loom_runtime_variant_pool_remove(other, original_id_));
    loom_runtime_variant_pool_free(other);
  }
}

TEST_F(VariantPoolTest, EmptyCapacityRejectsWithoutTakingPayload) {
  loom_runtime_variant_pool_t* rejected = nullptr;
  loom_runtime_variant_entry_t* original = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_variant_pool_allocate(
                            0, &disposed_, Dispose, iree_allocator_system(),
                            &original, &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_EQ(original, nullptr);
  EXPECT_EQ(disposed_, 0u);
  loom_runtime_variant_entry_retain(original);
  loom_runtime_variant_entry_release(original);
  loom_runtime_variant_pool_free(rejected);
}

TEST_F(VariantPoolTest, OwnerTraversalAndEvictionPreserveStableEntries) {
  const auto original_bytes = loom_runtime_variant_pool_storage_size(pool_);
  loom_runtime_variant_entry_t* entries[3] = {};
  for (auto& entry : entries) {
    IREE_ASSERT_OK(
        loom_runtime_variant_pool_begin(pool_, original_id_, &entry));
    loom_runtime_variant_entry_complete(entry, iree_ok_status(), &disposed_,
                                        nullptr);
  }
  ASSERT_TRUE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_GT(loom_runtime_variant_pool_storage_size(pool_), original_bytes);
  loom_runtime_variant_entry_retain(entries[1]);
  auto* entry = loom_runtime_variant_pool_first(pool_);
  unsigned visited = 0;
  while (entry) {
    auto* next = loom_runtime_variant_entry_next(entry);
    const auto snapshot = loom_runtime_variant_entry_snapshot(entry);
    EXPECT_EQ(snapshot.phase, LOOM_RUNTIME_VARIANT_READY);
    if (!snapshot.users) {
      loom_runtime_variant_entry_discard(entry);
    }
    ++visited;
    entry = next;
  }
  EXPECT_EQ(visited, 4u);
  EXPECT_EQ(disposed_, 2u);
  EXPECT_EQ(loom_runtime_variant_pool_first(pool_), entries[1]);
  EXPECT_EQ(loom_runtime_variant_entry_next(entries[1]), nullptr);
  EXPECT_EQ(loom_runtime_variant_pool_storage_size(pool_), original_bytes);
  loom_runtime_variant_entry_release(entries[1]);
  const auto base = loom_runtime_variant_entry_id(entries[1]);
  IREE_ASSERT_OK(loom_runtime_variant_pool_begin(pool_, base, &entry));
  loom_runtime_variant_entry_complete(entry, iree_ok_status(), &disposed_,
                                      nullptr);
  ASSERT_TRUE(loom_runtime_variant_pool_collect(pool_));
  EXPECT_EQ(loom_runtime_variant_entry_next(entries[1]), entry);
  loom_runtime_variant_entry_discard(entries[1]);
  loom_runtime_variant_entry_discard(entry);
  EXPECT_EQ(disposed_, 4u);
  EXPECT_EQ(loom_runtime_variant_pool_first(pool_), nullptr);
  EXPECT_EQ(loom_runtime_variant_pool_count(pool_), 0u);
}

TEST_F(VariantPoolTest, OwnerRecencyDoesNotChangeIdentityOrBorrows) {
  loom_runtime_variant_entry_t* entries[2] = {};
  for (auto& entry : entries) {
    IREE_ASSERT_OK(
        loom_runtime_variant_pool_begin(pool_, original_id_, &entry));
    loom_runtime_variant_entry_complete(entry, iree_ok_status(), nullptr,
                                        nullptr);
  }
  ASSERT_TRUE(loom_runtime_variant_pool_collect(pool_));
  loom_runtime_variant_entry_retain(entries[0]);
  const auto before = loom_runtime_variant_entry_snapshot(entries[0]);
  loom_runtime_variant_entry_touch(entries[0]);
  loom_runtime_variant_entry_touch(entries[0]);
  EXPECT_EQ(loom_runtime_variant_entry_next(original_), entries[1]);
  EXPECT_EQ(loom_runtime_variant_entry_next(entries[1]), entries[0]);
  EXPECT_EQ(loom_runtime_variant_entry_next(entries[0]), nullptr);
  const auto after = loom_runtime_variant_entry_snapshot(entries[0]);
  EXPECT_EQ(before.id, after.id);
  EXPECT_EQ(before.users, after.users);
  IREE_EXPECT_OK(loom_runtime_variant_entry_status(entries[0]));
  loom_runtime_variant_entry_release(entries[0]);
}

}  // namespace
