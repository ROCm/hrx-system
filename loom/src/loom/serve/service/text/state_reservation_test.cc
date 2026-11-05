// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/device/device.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/state.h"

namespace {

// The same source bootstrap and native storage as model creation, without
// parameter activation or execution kernels. Existing checkpoint tests exercise
// the immediate wrapper through complete model inference.
class TextStateReservationTest
    : public loom_serve::testing::text::SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const auto options = ModelOptions();
    loom_serve_text_state_initialize(&options, host_allocator_, &state_);
    IREE_ASSERT_OK(loom_runtime_device_create_jit(device_, &options.sources,
                                                  &options.kernel_sanitizer,
                                                  &compilation_, &jit_));
    IREE_ASSERT_OK(loom_serve_text_source_initialize(
        jit_, &options, host_allocator_, &source_, &state_.storage));
    IREE_ASSERT_OK(loom_runtime_device_create_execution(device_, storage_owner_,
                                                        &execution_));
    IREE_ASSERT_OK(loom_serve_text_state_allocate(&state_, device_,
                                                  storage_owner_, execution_));
    IREE_ASSERT_OK(loom_runtime_execution_drain(execution_));
  }

  void TearDown() override {
    loom_serve_text_state_reservation_free(reservation_);
    loom_serve_text_state_retire(&state_, nullptr, nullptr);
    IREE_EXPECT_OK(loom_runtime_execution_drain(execution_));
    loom_serve_text_state_await_retirement(&state_);
    IREE_EXPECT_OK(loom_serve_text_state_deinitialize(&state_));
    loom_runtime_execution_release(execution_);
    loom_runtime_jit_release(jit_);
    loom_serve_text_source_deinitialize(&source_);
    SourceTextServiceTest::TearDown();
  }

  void Reserve(size_t index, size_t extent) {
    loom_runtime_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_text_state_row_submit_reserve(
        state_.rows[index], LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, extent,
        nullptr, nullptr, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
    IREE_ASSERT_OK(loom_runtime_execution_drain(execution_));
    loom_serve_text_state_row_complete_reserve(state_.rows[index]);
  }

  // Owned source description outliving native state.
  loom_serve_text_source_t source_ = {};
  // Model-local compilation evidence outliving its source JIT.
  loom_runtime_compilation_report_t compilation_ = {};
  // Owned production source catalog supplying cached VM compilation.
  loom_runtime_jit_t* jit_ = nullptr;
  // Owned source-defined pooled state under test.
  loom_serve_text_state_t state_ = {};
  // Owned device timeline, shared by all rows of this state.
  loom_runtime_execution_t* execution_ = nullptr;
  // Prepared continuation, cancelled by teardown if an assertion aborts.
  loom_serve_text_state_reservation_t* reservation_ = nullptr;
};

TEST_F(TextStateReservationTest, PreparationOwnsIdsWithoutSubmittingOrMapping) {
  auto* row = state_.rows[0];
  const auto before = loom_runtime_execution_frontier(execution_);
  const auto memory = state_.memory.statistics;
  ASSERT_EQ(state_.cache.pool.capacity, 0u);
  const auto recurrent = state_.recurrent.pool.available;
  IREE_ASSERT_OK(loom_serve_text_state_row_prepare_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &reservation_));
  ASSERT_NE(reservation_, nullptr);
  EXPECT_GE(state_.cache.pool.capacity, 4u);
  EXPECT_EQ(state_.cache.pool.available, state_.cache.pool.capacity - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent - 1);
  EXPECT_EQ(row->block_count, 0u);
  EXPECT_EQ(row->reserved_extent, 0u);
  EXPECT_EQ(row->recurrent.slot, UINT32_MAX);
  EXPECT_EQ(state_.memory.statistics.committed_bytes, memory.committed_bytes);
  EXPECT_EQ(state_.memory.statistics.pending_bytes, memory.pending_bytes);
  const auto after = loom_runtime_execution_frontier(execution_);
  for (size_t i = 0; i < IREE_ARRAYSIZE(before.values); ++i) {
    EXPECT_EQ(after.semaphores[i], before.semaphores[i]);
    EXPECT_EQ(after.values[i], before.values[i]);
  }

  loom_serve_text_state_reservation_free(reservation_);
  reservation_ = nullptr;
  EXPECT_EQ(state_.cache.pool.available, state_.cache.pool.capacity);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);

  // Cancellation returns the plan's IDs in acquisition order; an immediate
  // request can use the full pool with no stale continuation ownership.
  ASSERT_NO_FATAL_FAILURE(Reserve(0, 96));
  EXPECT_EQ(row->reserved_extent, 96u);
  EXPECT_GE(state_.cache.pool.capacity, 24u);
  EXPECT_EQ(state_.cache.pool.available, state_.cache.pool.capacity - 24);
}

TEST_F(TextStateReservationTest,
       DeniedPreparedContinuationKeepsItsExactPlanThroughPeerProgress) {
  ASSERT_NO_FATAL_FAILURE(Reserve(0, 16));
  auto* row = state_.rows[0];
  const std::vector<uint32_t> old_blocks(state_.cache.maps,
                                         state_.cache.maps + row->block_count);
  const auto pages = state_.cache.pool.available;
  const auto recurrent = state_.recurrent.pool.available;
  IREE_ASSERT_OK(loom_serve_text_state_row_prepare_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_CONTINUE, nullptr, 32, &reservation_));
  ASSERT_NE(reservation_, nullptr);
  // The unconsumed private pages fund their replacement, not a second full
  // reservation. The prepared plan owns four additional pages only.
  EXPECT_EQ(state_.cache.pool.available, pages - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);
  struct Admission {
    // Borrowed physical owner used by ordinary state commitment.
    loom_runtime_memory_pool_t* pool;
    // Exact ranges observed at the first real admission attempt.
    std::vector<loom_runtime_memory_range_t> planned;
  } context = {state_.memory.pool, {}};
  const auto deny = [](void* user_data, iree_host_size_t count,
                       const loom_runtime_memory_range_t* ranges,
                       iree_hal_semaphore_t* readiness, uint64_t value,
                       loom_runtime_memory_admission_t* out_admission,
                       loom_runtime_memory_operation_t** out_operation) {
    auto* context = static_cast<Admission*>(user_data);
    EXPECT_EQ(count != 0, context->pool != nullptr);
    if (count) {
      context->planned.assign(ranges, ranges + count);
    }
    for (size_t i = 0; i < count; ++i) {
      EXPECT_NE(ranges[i].buffer, nullptr);
      EXPECT_GT(ranges[i].length, 0u);
    }
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
    *out_operation = nullptr;
    return iree_ok_status();
  };
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_text_state_reservation_submit(
      reservation_, deny, &context, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(row->reserved_extent, 16u);
  EXPECT_EQ(std::vector<uint32_t>(state_.cache.maps,
                                  state_.cache.maps + row->block_count),
            old_blocks);

  // Independent rows can acquire/submit/retire between planning and commit.
  // This does not replan the protected destination or steal its selected IDs.
  ASSERT_NO_FATAL_FAILURE(Reserve(1, 16));
  loom_serve_text_state_row_reset(state_.rows[1]);
  const auto commit = [](void* user_data, iree_host_size_t count,
                         const loom_runtime_memory_range_t* ranges,
                         iree_hal_semaphore_t* readiness, uint64_t value,
                         loom_runtime_memory_admission_t* out_admission,
                         loom_runtime_memory_operation_t** out_operation) {
    auto* context = static_cast<Admission*>(user_data);
    EXPECT_EQ(count, context->planned.size());
    for (size_t i = 0; i < iree_min(count, context->planned.size()); ++i) {
      EXPECT_EQ(ranges[i].buffer, context->planned[i].buffer);
      EXPECT_EQ(ranges[i].offset, context->planned[i].offset);
      EXPECT_EQ(ranges[i].length, context->planned[i].length);
    }
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
    *out_operation = nullptr;
    return context->pool ? loom_runtime_memory_pool_commit_async(
                               context->pool, count, ranges, readiness, value,
                               out_admission, out_operation)
                         : iree_ok_status();
  };
  IREE_ASSERT_OK(loom_serve_text_state_reservation_submit(
      reservation_, commit, &context, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  loom_serve_text_state_reservation_free(reservation_);
  reservation_ = nullptr;
  IREE_ASSERT_OK(loom_runtime_execution_drain(execution_));
  loom_serve_text_state_row_complete_reserve(row);
  EXPECT_EQ(row->reserved_extent, 32u);
  EXPECT_EQ(row->block_count, 8u);
  EXPECT_EQ(state_.cache.pool.available, pages - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);
  EXPECT_TRUE(
      std::equal(old_blocks.begin(), old_blocks.end(), state_.cache.maps));
}

}  // namespace
