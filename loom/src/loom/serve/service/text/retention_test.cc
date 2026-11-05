// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/retention.h"

#include "loom/serve/service/text/reservation.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/state.h"
#include "loom/serve/testing/native_allocator.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class TextRetentionTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const auto options = ModelOptions();
    IREE_ASSERT_OK(
        loom_serve_text_model_allocate(1, &device_, &storage_owner_, &options,
                                       allocations_.allocator(), &model_));
  }

  void TearDown() override {
    if (model_) {
      loom_serve_text_retention_clear(model_);
    }
    loom_serve_text_reservation_deinitialize(&reservation_);
    loom_serve_text_checkpoint_release(checkpoint_);
    SourceTextServiceTest::TearDown();
    allocations_.AwaitReleased();
  }

  void Capture() {
    auto* row = loom_serve_text_model_row(model_, 0);
    const int32_t tokens[] = {0, 1, 2};
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint_));
    ASSERT_NE(checkpoint_, nullptr);
    IREE_ASSERT_OK(loom_serve_text_row_reset(row));
  }

  void SeparatePrivateMappings() {
    const auto mapping_size =
        loom_serve_pool_state_geometry(storage_owner_).mapping_size;
    if (!mapping_size) {
      return;
    }
    // Keep the live token sequence tiny, but place each row's input across a
    // mapping boundary. Recurrent/KV bounds alone cannot cover that exact
    // private request range as they do in the compact fixture.
    const loom_serve_packing_shape_t shapes[] = {
        {4, 1}, {mapping_size / sizeof(int32_t) + 1, 4}};
    auto options = ModelOptions();
    options.prefill_capacity = shapes[1].token_capacity;
    options.context_capacity = 2 * options.prefill_capacity;
    options.pool_capacity = 4 * options.prefill_capacity;
    options.epoch_count = IREE_ARRAYSIZE(shapes);
    options.epoch_shapes = shapes;
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    IREE_ASSERT_OK(
        loom_serve_text_model_allocate(1, &device_, &storage_owner_, &options,
                                       allocations_.allocator(), &model_));
  }

  void Continue() {
    auto* row = loom_serve_text_model_row(model_, 0);
    bool restored = false;
    IREE_ASSERT_OK(
        loom_serve_text_row_try_restore(row, checkpoint_, &restored));
    ASSERT_TRUE(restored);
    const int32_t suffix = 1;
    IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &suffix));
    Oracle oracle;
    oracle.Consume("abxb");
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
  }

  // Observes/fails actual host allocation, not the retention implementation.
  loom_serve::testing::NativeAllocator allocations_;
  // Owned endpoint preserving actual source data across forecasts.
  loom_serve_text_checkpoint_t* checkpoint_ = nullptr;
  // Optional exact producer plan, retired before its source endpoint.
  loom_serve_text_reservation_t reservation_ = {};
};

TEST_F(TextRetentionTest, AliasedExplicitHandlesProtectSharedBacking) {
  ASSERT_NO_FATAL_FAILURE(Capture());
  const auto before = loom_serve_text_model_memory_statistics(model_);
  loom_serve_residency_projection_t retained;
  IREE_ASSERT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &retained));
  EXPECT_LE(retained.minimum_bytes, retained.resident_bytes);
  EXPECT_EQ(retained.resident_bytes,
            before.committed_bytes + before.pending_bytes);
  const bool elastic = before.reserved_bytes != 0;
  EXPECT_EQ(retained.minimum_bytes != 0, elastic);

  loom_serve_text_checkpoint_retain(checkpoint_);
  loom_serve_text_retention_add_checkpoint(checkpoint_);
  loom_serve_residency_projection_t aliased;
  IREE_EXPECT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &aliased));
  EXPECT_EQ(aliased.minimum_bytes, retained.minimum_bytes);
  // Both handle references are eligible now, but no native ownership changes.
  loom_serve_text_retention_add_checkpoint(checkpoint_);
  IREE_EXPECT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &aliased));
  EXPECT_EQ(aliased.minimum_bytes, 0u);
  loom_serve_text_retention_clear(model_);
  loom_serve_text_checkpoint_release(checkpoint_);
  const auto after = loom_serve_text_model_memory_statistics(model_);
  EXPECT_EQ(after.committed_bytes, before.committed_bytes);
  EXPECT_EQ(after.pending_bytes, before.pending_bytes);
  EXPECT_EQ(after.released_bytes, before.released_bytes);
  ASSERT_NO_FATAL_FAILURE(Continue());
}

TEST_F(TextRetentionTest, PreparedRequestProtectsItsBorrowAndExactRanges) {
  ASSERT_NO_FATAL_FAILURE(SeparatePrivateMappings());
  ASSERT_NO_FATAL_FAILURE(Capture());
  auto* row = loom_serve_text_model_row(model_, 1);
  allocations_.ResetAllocationMetrics();
  IREE_ASSERT_OK(loom_serve_text_reservation_prepare(
      row, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint_, 16, &reservation_));
  const auto preparation = allocations_.allocation_metrics();
  RecordProperty("prepare_allocations", preparation.request_count);
  RecordProperty("prepare_requested_bytes", preparation.requested_bytes);
  ASSERT_NE(reservation_.state, nullptr);
  loom_serve_text_retention_add_checkpoint(checkpoint_);
  loom_serve_residency_projection_t retained, required, movable;
  IREE_ASSERT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &retained));
  IREE_ASSERT_OK(loom_serve_text_retention_forecast(model_, reservation_.state,
                                                    false, &required));
  IREE_ASSERT_OK(loom_serve_text_retention_forecast(model_, reservation_.state,
                                                    true, &movable));
  EXPECT_LE(retained.minimum_bytes, retained.resident_bytes);
  EXPECT_LE(movable.minimum_bytes, required.minimum_bytes);
  EXPECT_GE(required.minimum_bytes, retained.minimum_bytes);
  // This request borrows the only retained endpoint. Its exact source bound
  // must match the native commitment union, including COW readers and tails.
  EXPECT_EQ(required.minimum_bytes,
            loom_serve_text_state_reservation_footprint(reservation_.state));
  if (retained.resident_bytes) {
    EXPECT_GT(retained.minimum_bytes, 0u);
  }
  loom_serve_text_retention_clear(model_);
  loom_serve_text_reservation_deinitialize(&reservation_);
  ASSERT_NO_FATAL_FAILURE(Continue());
}

TEST_F(TextRetentionTest,
       FailedScratchAllocationLeavesOwnershipAndMarksIntact) {
  ASSERT_NO_FATAL_FAILURE(Capture());
  if (!loom_serve_text_model_memory_statistics(model_).reserved_bytes) {
    return;
  }
  loom_serve_residency_projection_t before;
  IREE_ASSERT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &before));
  // Reference scratch fails before traversal; constraint growth fails after
  // actual backing has been selected. Neither can leave physical union marks.
  for (auto command :
       {IREE_ALLOCATOR_COMMAND_CALLOC, IREE_ALLOCATOR_COMMAND_REALLOC}) {
    allocations_.FailNextAllocation(command);
    loom_serve_residency_projection_t failed;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_text_retention_forecast(model_, nullptr, false, &failed));
    loom_serve_text_retention_clear(model_);
    loom_serve_residency_projection_t after;
    IREE_ASSERT_OK(
        loom_serve_text_retention_forecast(model_, nullptr, false, &after));
    EXPECT_EQ(after.resident_bytes, before.resident_bytes);
    EXPECT_EQ(after.minimum_bytes, before.minimum_bytes);
  }
  ASSERT_NO_FATAL_FAILURE(Continue());
}

TEST_F(TextRetentionTest, QueryScratchTracksOwnershipNotVirtualCapacity) {
  if (!loom_serve_text_model_memory_statistics(model_).reserved_bytes) {
    return;
  }
  loom_serve::testing::NativeAllocator::AllocationMetrics baseline = {};
  for (uint32_t capacity : {256u, 4u * 1024 * 1024}) {
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    auto options = ModelOptions();
    options.pool_capacity = capacity;
    IREE_ASSERT_OK(
        loom_serve_text_model_allocate(1, &device_, &storage_owner_, &options,
                                       allocations_.allocator(), &model_));
    ASSERT_NO_FATAL_FAILURE(Capture());
    allocations_.ResetAllocationMetrics();
    loom_serve_residency_projection_t retained;
    IREE_ASSERT_OK(
        loom_serve_text_retention_forecast(model_, nullptr, false, &retained));
    const auto metrics = allocations_.allocation_metrics();
    if (capacity == 256) {
      baseline = metrics;
      EXPECT_GT(metrics.request_count, 0u);
      RecordProperty("query_allocations", metrics.request_count);
      RecordProperty("query_requested_bytes", metrics.requested_bytes);
      RecordProperty("query_largest_allocation_bytes",
                     metrics.largest_request_bytes);
    } else {
      EXPECT_EQ(metrics.request_count, baseline.request_count);
      EXPECT_EQ(metrics.requested_bytes, baseline.requested_bytes);
      EXPECT_EQ(metrics.largest_request_bytes, baseline.largest_request_bytes);
    }
    loom_serve_text_checkpoint_release(checkpoint_);
    checkpoint_ = nullptr;
  }
}

TEST_F(TextRetentionTest, ColdRestoreDoesNotChargeUnacceptedDeclarations) {
  ASSERT_NO_FATAL_FAILURE(SeparatePrivateMappings());
  ASSERT_NO_FATAL_FAILURE(Capture());
  if (!loom_serve_text_model_memory_statistics(model_).reserved_bytes) {
    return;
  }
  IREE_ASSERT_OK(loom_serve_text_checkpoint_suspend(checkpoint_));
  loom_serve_text_trim_result_t trim;
  IREE_ASSERT_OK(loom_serve_text_model_trim(model_, &trim));
  auto* row = loom_serve_text_model_row(model_, 1);
  allocations_.ResetAllocationMetrics();
  IREE_ASSERT_OK(loom_serve_text_reservation_prepare(
      row, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint_, 16, &reservation_));
  const auto preparation = allocations_.allocation_metrics();
  RecordProperty("prepare_allocations", preparation.request_count);
  RecordProperty("prepare_requested_bytes", preparation.requested_bytes);
  ASSERT_NE(reservation_.state, nullptr);
  loom_serve_residency_projection_t idle, request;
  IREE_ASSERT_OK(
      loom_serve_text_retention_forecast(model_, nullptr, false, &idle));
  EXPECT_EQ(idle.minimum_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_text_retention_forecast(model_, reservation_.state,
                                                    false, &request));
  EXPECT_GT(request.minimum_bytes, idle.minimum_bytes);
  EXPECT_EQ(request.resident_bytes, idle.resident_bytes);
  EXPECT_EQ(request.minimum_bytes,
            loom_serve_text_state_reservation_footprint(reservation_.state));
  loom_serve_text_reservation_deinitialize(&reservation_);
  ASSERT_NO_FATAL_FAILURE(Continue());
}

}  // namespace
