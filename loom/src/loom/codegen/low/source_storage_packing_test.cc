// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_storage_packing.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct NoninterferencePair {
  loom_value_id_t lhs_root_value_id;
  loom_value_id_t rhs_root_value_id;
};

struct InterferenceQueryState {
  NoninterferencePair noninterference_pairs[8];
  iree_host_size_t noninterference_pair_count = 0;
  loom_value_id_t failure_lhs_root_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t failure_rhs_root_value_id = LOOM_VALUE_ID_INVALID;
};

static bool PairMatches(const NoninterferencePair& pair,
                        loom_value_id_t lhs_root_value_id,
                        loom_value_id_t rhs_root_value_id) {
  return (pair.lhs_root_value_id == lhs_root_value_id &&
          pair.rhs_root_value_id == rhs_root_value_id) ||
         (pair.lhs_root_value_id == rhs_root_value_id &&
          pair.rhs_root_value_id == lhs_root_value_id);
}

static iree_status_t QueryInterference(void* user_data,
                                       loom_value_id_t lhs_root_value_id,
                                       loom_value_id_t rhs_root_value_id,
                                       bool* out_interferes) {
  auto* state = static_cast<InterferenceQueryState*>(user_data);
  *out_interferes = true;
  const NoninterferencePair failure_pair = {
      state->failure_lhs_root_value_id,
      state->failure_rhs_root_value_id,
  };
  if (PairMatches(failure_pair, lhs_root_value_id, rhs_root_value_id)) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected interference query failure");
  }
  for (iree_host_size_t i = 0; i < state->noninterference_pair_count; ++i) {
    if (PairMatches(state->noninterference_pairs[i], lhs_root_value_id,
                    rhs_root_value_id)) {
      *out_interferes = false;
      break;
    }
  }
  return iree_ok_status();
}

class SourceStoragePackingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_source_storage_packing_create(
        loom_source_storage_packing_interference_callback_make(
            QueryInterference, &query_state_),
        &arena_, &packing_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void AddNoninterference(loom_value_id_t lhs_root_value_id,
                          loom_value_id_t rhs_root_value_id) {
    ASSERT_LT(query_state_.noninterference_pair_count,
              IREE_ARRAYSIZE(query_state_.noninterference_pairs));
    query_state_
        .noninterference_pairs[query_state_.noninterference_pair_count++] = {
        lhs_root_value_id,
        rhs_root_value_id,
    };
  }

  uint64_t Append(loom_value_id_t root_value_id, uint64_t byte_length,
                  uint64_t byte_alignment) {
    uint64_t byte_offset = UINT64_MAX;
    IREE_CHECK_OK(loom_source_storage_packing_append(
        packing_, root_value_id, byte_length, byte_alignment, &byte_offset));
    return byte_offset;
  }

  InterferenceQueryState query_state_;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_source_storage_packing_t* packing_ = nullptr;
};

TEST_F(SourceStoragePackingTest, InterferingAllocationsPackWithAlignment) {
  EXPECT_EQ(Append(/*root_value_id=*/1, /*byte_length=*/12,
                   /*byte_alignment=*/4),
            0u);
  EXPECT_EQ(Append(/*root_value_id=*/2, /*byte_length=*/4,
                   /*byte_alignment=*/16),
            16u);

  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(requirement.byte_length, 20u);
  EXPECT_EQ(requirement.byte_alignment, 16u);
}

TEST_F(SourceStoragePackingTest, DisjointAllocationReusesSeveralRanges) {
  AddNoninterference(/*lhs_root_value_id=*/3, /*rhs_root_value_id=*/1);
  AddNoninterference(/*lhs_root_value_id=*/3, /*rhs_root_value_id=*/2);

  EXPECT_EQ(Append(/*root_value_id=*/1, /*byte_length=*/10240,
                   /*byte_alignment=*/16),
            0u);
  EXPECT_EQ(Append(/*root_value_id=*/2, /*byte_length=*/10240,
                   /*byte_alignment=*/16),
            10240u);
  EXPECT_EQ(Append(/*root_value_id=*/3, /*byte_length=*/32768,
                   /*byte_alignment=*/16),
            0u);

  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(requirement.byte_length, 32768u);
  EXPECT_EQ(requirement.byte_alignment, 16u);
}

TEST_F(SourceStoragePackingTest, AllocationPartiallyReusesOnePriorRange) {
  AddNoninterference(/*lhs_root_value_id=*/3, /*rhs_root_value_id=*/2);

  EXPECT_EQ(Append(/*root_value_id=*/1, /*byte_length=*/64,
                   /*byte_alignment=*/16),
            0u);
  EXPECT_EQ(Append(/*root_value_id=*/2, /*byte_length=*/64,
                   /*byte_alignment=*/16),
            64u);
  EXPECT_EQ(Append(/*root_value_id=*/3, /*byte_length=*/64,
                   /*byte_alignment=*/16),
            64u);

  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(requirement.byte_length, 128u);
  EXPECT_EQ(requirement.byte_alignment, 16u);
}

TEST_F(SourceStoragePackingTest, QueryFailureDoesNotPublishAllocation) {
  EXPECT_EQ(Append(/*root_value_id=*/1, /*byte_length=*/64,
                   /*byte_alignment=*/16),
            0u);
  query_state_.failure_lhs_root_value_id = 2;
  query_state_.failure_rhs_root_value_id = 1;
  uint64_t failed_offset = UINT64_MAX;
  iree_status_t status = loom_source_storage_packing_append(
      packing_, /*root_value_id=*/2, /*byte_length=*/64,
      /*byte_alignment=*/16, &failed_offset);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_ABORTED);
  iree_status_free(status);
  EXPECT_EQ(failed_offset, 0u);

  const loom_source_storage_packing_requirement_t failed_requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(failed_requirement.byte_length, 64u);
  EXPECT_EQ(failed_requirement.byte_alignment, 16u);

  query_state_.failure_lhs_root_value_id = LOOM_VALUE_ID_INVALID;
  query_state_.failure_rhs_root_value_id = LOOM_VALUE_ID_INVALID;
  EXPECT_EQ(Append(/*root_value_id=*/3, /*byte_length=*/64,
                   /*byte_alignment=*/16),
            64u);
}

TEST_F(SourceStoragePackingTest, ExtentOverflowDoesNotChangeRequirement) {
  EXPECT_EQ(Append(/*root_value_id=*/1, /*byte_length=*/INT64_MAX,
                   /*byte_alignment=*/1),
            0u);
  uint64_t failed_offset = UINT64_MAX;
  iree_status_t status = loom_source_storage_packing_append(
      packing_, /*root_value_id=*/2, /*byte_length=*/1,
      /*byte_alignment=*/1, &failed_offset);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_free(status);
  EXPECT_EQ(failed_offset, 0u);

  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(requirement.byte_length, static_cast<uint64_t>(INT64_MAX));
  EXPECT_EQ(requirement.byte_alignment, 1u);
}

TEST_F(SourceStoragePackingTest, InvalidAlignmentDoesNotChangeRequirement) {
  uint64_t failed_offset = UINT64_MAX;
  iree_status_t status = loom_source_storage_packing_append(
      packing_, /*root_value_id=*/1, /*byte_length=*/64,
      /*byte_alignment=*/3, &failed_offset);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_free(status);
  EXPECT_EQ(failed_offset, 0u);

  const loom_source_storage_packing_requirement_t requirement =
      loom_source_storage_packing_requirement(packing_);
  EXPECT_EQ(requirement.byte_length, 0u);
  EXPECT_EQ(requirement.byte_alignment, 0u);
}

}  // namespace
}  // namespace loom
