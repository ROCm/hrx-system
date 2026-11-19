// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_layout.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class PipelineLifetimeLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes_));
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // Parent-owned state coexists with two children of one serial scope.
  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Retained producer shape: every construction has a separate OWNER leaf.
  const loom_pipeline_lifetime_t nodes_[7] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  // Direct owner identities retained by lifetime construction.
  const iree_host_size_t owners_[3] = {1, 5, 6};
  // Complete graph consumed by both storage and semaphore placement.
  loom_pipeline_lifetimes_t lifetimes_ = {
      .nodes = nodes_,
      .count = 7,
      .owners = owners_,
      .has_serial = true,
  };
  // Backing for the retained layout result.
  iree_arena_block_pool_t blocks_;
  // Allocation owner for requests and placements.
  iree_arena_allocator_t arena_;
};

TEST_F(PipelineLifetimeLayoutTest, AlignmentAndOwnerLocalRangesAcrossPools) {
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 1, .pool = 0, .length = 3, .alignment = 1},
      {.lifetime = 5, .pool = 0, .length = 5, .alignment = 4},
      {.lifetime = 6, .pool = 0, .length = 11, .alignment = 8},
      {.lifetime = 5, .pool = 0, .length = 2, .alignment = 2},
      {.lifetime = 1, .pool = 1, .length = 1, .alignment = 1},
      {.lifetime = 5, .pool = 1, .length = 3, .alignment = 8},
      {.lifetime = 6, .pool = 1, .length = 17, .alignment = 16},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), 3, &arena_, &extents));
  const uint64_t offsets[] = {0, 8, 8, 14, 0, 16, 16};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]);
    EXPECT_EQ(requests[i].offset % requests[i].alignment, 0u);
  }
  EXPECT_EQ(extents[0].length, 19u);
  EXPECT_EQ(extents[0].alignment, 8u);
  EXPECT_EQ(extents[1].length, 33u);
  EXPECT_EQ(extents[1].alignment, 16u);
  EXPECT_EQ(extents[2].length, 0u);
  EXPECT_EQ(extents[2].alignment, 1u);
}

TEST_F(PipelineLifetimeLayoutTest, ASubsetKeepsTheCompleteOwnerLifetime) {
  // An unused serial child contributes no space. The selected child still
  // cannot overlap parent state, even when its request is enumerated first.
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 6, .pool = 0, .length = 2, .alignment = 1},
      {.lifetime = 1, .pool = 0, .length = 1, .alignment = 1},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), 1, &arena_, &extents));
  EXPECT_EQ(requests[0].offset, 0u);
  EXPECT_EQ(requests[1].offset, 2u);
  EXPECT_EQ(extents[0].length, 3u);
}

TEST_F(PipelineLifetimeLayoutTest, EmptyPoolsAndZeroLengthAlignment) {
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 5, .pool = 4095, .length = 0, .alignment = 64},
      {.lifetime = 1, .pool = 4095, .length = 3, .alignment = 1},
      {.lifetime = 6, .pool = 4095, .length = 5, .alignment = 2},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(&lifetimes_, requests,
                                               IREE_ARRAYSIZE(requests), 4096,
                                               &arena_, &extents));
  EXPECT_EQ(requests[0].offset, 0u);
  EXPECT_EQ(requests[1].offset, 5u);
  EXPECT_EQ(requests[2].offset, 0u);
  EXPECT_EQ(extents[4095].length, 8u);
  EXPECT_EQ(extents[4095].alignment, 64u);
  for (size_t i = 0; i < 4095; ++i) {
    EXPECT_EQ(extents[i].length, 0u);
    EXPECT_EQ(extents[i].alignment, 1u);
  }
}

TEST_F(PipelineLifetimeLayoutTest, EmptyProjection) {
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(&lifetimes_, nullptr, 0, 2,
                                               &arena_, &extents));
  EXPECT_EQ(extents[0].length, 0u);
  EXPECT_EQ(extents[1].alignment, 1u);
}

TEST_F(PipelineLifetimeLayoutTest, MixedPoolDemandsKeepIndependentAlignment) {
  // Interleaved requests span several scratch groups. Some pools only retain
  // parent state; the others also reuse space between both serial children.
  std::vector<loom_pipeline_lifetime_request_t> requests;
  for (size_t owner : {6u, 1u, 5u}) {
    for (size_t pool = 0; pool < 33; ++pool) {
      if (owner != 1 && pool % 3 == 0) {
        continue;
      }
      requests.push_back({
          .lifetime = owner,
          .pool = pool,
          .length = owner == 1   ? 3u
                    : owner == 5 ? 5u
                                 : 11u,
          .alignment = owner == 1   ? 1u
                       : owner == 5 ? 4u
                                    : 8u,
      });
    }
  }
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests.data(), requests.size(), 33, &arena_, &extents));
  for (const auto& request : requests) {
    EXPECT_EQ(request.offset,
              request.lifetime == 1 && request.pool % 3 ? 11u : 0u);
  }
  for (size_t pool = 0; pool < 33; ++pool) {
    EXPECT_EQ(extents[pool].length, pool % 3 ? 14u : 3u);
    EXPECT_EQ(extents[pool].alignment, pool % 3 ? 8u : 1u);
  }
}

TEST_F(PipelineLifetimeLayoutTest,
       RepeatedProjectionUsesCurrentSizesAndAlignment) {
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 6},
      {.lifetime = 5},
      {.lifetime = 1},
      {.lifetime = 5},
  };
  loom_pipeline_lifetime_projection_t* projection = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_projection_prepare(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), &arena_, &projection));
  const size_t prepared_bytes = arena_.used_allocation_size;
  const struct {
    // Current lengths in input order, including repeated owner five.
    uint64_t lengths[4];
    // Current alignment requirements, including empty aligned resources.
    uint64_t alignments[4];
    // Absolute offsets after serial reuse and concurrent concatenation.
    uint64_t offsets[4];
    // Complete pool requirement including alignment gaps.
    loom_pipeline_lifetime_extent_t extent;
  } cases[] = {
      {{11, 5, 3, 2}, {8, 4, 1, 2}, {0, 0, 11, 6}, {14, 8}},
      {{0, 2, 3, 5}, {64, 2, 4, 4}, {0, 0, 12, 4}, {15, 64}},
      {{12, 2, 4, 0}, {4, 8, 16, 32}, {0, 0, 16, 0}, {20, 32}},
      {{0, 0, 0, 0}, {64, 16, 8, 32}, {0, 0, 0, 0}, {0, 64}},
  };
  // An unrepresentable proposal cannot destroy the retained topology or poison
  // the next evaluation. Both accumulation and alignment can exceed the bound.
  for (bool alignment_overflow : {false, true}) {
    requests[0].length = INT64_MAX;
    requests[0].alignment = 1;
    requests[1].length = 0;
    requests[1].alignment = 1;
    requests[2].length = 1;
    requests[2].alignment = alignment_overflow ? 8 : 1;
    requests[3].length = 0;
    requests[3].alignment = 1;
    loom_pipeline_lifetime_extent_t extent;
    EXPECT_THAT(iree::Status(loom_pipeline_lifetime_projection_evaluate(
                    projection, requests, &extent)),
                iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
    for (const auto& test_case : cases) {
      for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
        requests[i].length = test_case.lengths[i];
        requests[i].alignment = test_case.alignments[i];
      }
      IREE_ASSERT_OK(loom_pipeline_lifetime_projection_evaluate(
          projection, requests, &extent));
      for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
        EXPECT_EQ(requests[i].offset, test_case.offsets[i]);
      }
      EXPECT_EQ(extent.length, test_case.extent.length);
      EXPECT_EQ(extent.alignment, test_case.extent.alignment);
      EXPECT_EQ(arena_.used_allocation_size, prepared_bytes);
    }
  }
}

TEST_F(PipelineLifetimeLayoutTest, EmptyPreparedProjection) {
  loom_pipeline_lifetime_projection_t* projection = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_projection_prepare(
      &lifetimes_, nullptr, 0, &arena_, &projection));
  for (int i = 0; i < 2; ++i) {
    loom_pipeline_lifetime_extent_t extent = {64, 8};
    IREE_ASSERT_OK(loom_pipeline_lifetime_projection_evaluate(
        projection, nullptr, &extent));
    EXPECT_EQ(extent.length, 0u);
    EXPECT_EQ(extent.alignment, 1u);
  }
}

TEST_F(PipelineLifetimeLayoutTest, SingletonSparsePoolHonorsExtentBound) {
  loom_pipeline_lifetime_request_t request = {
      .lifetime = 1,
      .pool = 4095,
      .length = uint64_t(INT64_MAX) + 1,
      .alignment = 8,
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  EXPECT_THAT(iree::Status(loom_pipeline_lifetime_layout(
                  &lifetimes_, &request, 1, 4096, &arena_, &extents)),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
  request.length = 3;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(&lifetimes_, &request, 1, 4096,
                                               &arena_, &extents));
  EXPECT_EQ(request.offset, 0u);
  EXPECT_EQ(extents[4095].length, 3u);
  EXPECT_EQ(extents[4095].alignment, 8u);
}

TEST_F(PipelineLifetimeLayoutTest, ConcurrentExtentOverflow) {
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 1, .pool = 0, .length = uint64_t(INT64_MAX), .alignment = 1},
      {.lifetime = 6, .pool = 0, .length = 1, .alignment = 1},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  EXPECT_THAT(iree::Status(loom_pipeline_lifetime_layout(
                  &lifetimes_, requests, 2, 1, &arena_, &extents)),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
}

TEST_F(PipelineLifetimeLayoutTest, NestedBranchesFollowRequestOrder) {
  // A serial choice between (A, B concurrently) and C runs concurrently with
  // D. The larger alignment of B applies to the entire first serial choice.
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, 2},
      {LOOM_PIPELINE_LIFETIME_OWNER, 3, 0, none, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, 1, 5, 4},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 1, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 3, none, 6},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
  };
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes,
      .count = IREE_ARRAYSIZE(nodes),
      .has_serial = true,
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes));
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 6, .pool = 0, .length = 1, .alignment = 8},
      {.lifetime = 2, .pool = 0, .length = 2, .alignment = 4},
      {.lifetime = 5, .pool = 0, .length = 3, .alignment = 1},
      {.lifetime = 4, .pool = 0, .length = 5, .alignment = 2},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes, requests, IREE_ARRAYSIZE(requests), 1, &arena_, &extents));
  const uint64_t offsets[] = {0, 8, 1, 0};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]);
  }
  EXPECT_EQ(extents[0].length, 10u);
  EXPECT_EQ(extents[0].alignment, 8u);
  // Compare the indexed relation against the definition of common ancestry.
  for (size_t first = 0; first < lifetimes.count; ++first) {
    for (size_t second = 0; second < lifetimes.count; ++second) {
      size_t expected = none;
      for (size_t a = first; a != none && expected == none;
           a = nodes[a].parent) {
        for (size_t b = second; b != none; b = nodes[b].parent) {
          if (a == b) {
            expected = a;
            break;
          }
        }
      }
      EXPECT_EQ(
          loom_pipeline_lifetimes_common_ancestor(&lifetimes, first, second),
          expected);
      if (first != second &&
          nodes[first].kind == LOOM_PIPELINE_LIFETIME_OWNER &&
          nodes[second].kind == LOOM_PIPELINE_LIFETIME_OWNER) {
        const auto& a = lifetimes.ancestry[first];
        const auto& b = lifetimes.ancestry[second];
        EXPECT_EQ((a.preorder < b.preorder) !=
                      (a.concurrent_preorder < b.concurrent_preorder),
                  nodes[expected].kind == LOOM_PIPELINE_LIFETIME_CONCURRENT);
      }
    }
  }
}

TEST_F(PipelineLifetimeLayoutTest, DeepUnaryPathsDoNotChangePlacement) {
  // Two deep branches require only their common concurrent boundary in this
  // resource projection. Nesting is unbounded by the native resource domain.
  constexpr size_t depth = 4096;
  std::vector<loom_pipeline_lifetime_t> nodes(2 * depth + 1);
  nodes[0] = {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none};
  for (size_t branch = 0; branch < 2; ++branch) {
    for (size_t i = 0; i < depth; ++i) {
      size_t index = 1 + branch * depth + i;
      nodes[index] = {i + 1 == depth ? LOOM_PIPELINE_LIFETIME_OWNER
                                     : LOOM_PIPELINE_LIFETIME_CONCURRENT,
                      branch, i ? index - 1 : 0,
                      i + 1 == depth ? none : index + 1,
                      !i && !branch ? depth + 1 : none};
    }
  }
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes.data(),
      .count = nodes.size(),
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes));
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 2 * depth, .pool = 0, .length = 3, .alignment = 16},
      {.lifetime = depth, .pool = 0, .length = 1, .alignment = 1},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(&lifetimes, requests, 2, 1,
                                               &arena_, &extents));
  EXPECT_EQ(requests[0].offset, 0u);
  EXPECT_EQ(requests[1].offset, 3u);
  EXPECT_EQ(extents[0].length, 4u);
}

TEST_F(PipelineLifetimeLayoutTest, IndependentSerialBranches) {
  std::vector<loom_pipeline_lifetime_t> nodes = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 5, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, 1, 7, 4},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 1, none, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, 2, 9, 6},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 2, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 3, none, 8},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 5, none, 10},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 5, none, none},
  };
  // Unrequested siblings still belong to the complete graph. The requested
  // serial branches retain their placement in this sparse projection.
  nodes[2].next_sibling = nodes.size();
  for (size_t i = 0; i < 64; ++i) {
    nodes.push_back({LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none,
                     i + 1 == 64 ? none : nodes.size() + 1});
  }
  loom_pipeline_lifetimes_t lifetimes = {
      .nodes = nodes.data(),
      .count = nodes.size(),
      .has_serial = true,
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes));
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 6, .pool = 0, .length = 32, .alignment = 1},
      {.lifetime = 10, .pool = 0, .length = 16, .alignment = 1},
      {.lifetime = 9, .pool = 0, .length = 8, .alignment = 1},
      {.lifetime = 4, .pool = 0, .length = 4, .alignment = 1},
      {.lifetime = 8, .pool = 0, .length = 2, .alignment = 1},
      {.lifetime = 7, .pool = 0, .length = 1, .alignment = 1},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes, requests, IREE_ARRAYSIZE(requests), 1, &arena_, &extents));
  const uint64_t offsets[] = {0, 0, 16, 32, 32, 34};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]);
  }
  EXPECT_EQ(extents[0].length, 36u);
}

TEST_F(PipelineLifetimeLayoutTest, EachPoolPreservesItsConcurrentPreferences) {
  // Two serial phases enumerate their concurrent owners in opposite orders.
  // Independent resource pools can prefer opposite owners without changing
  // either phase's coexistence relation or the complete tree.
  const std::vector<loom_pipeline_lifetime_t> requested_nodes = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 7, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, 6},
      {LOOM_PIPELINE_LIFETIME_OWNER, 3, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, 8},
      {LOOM_PIPELINE_LIFETIME_OWNER, 4, 4, none, none},
  };
  // Unrequested nesting changes projection density without changing any
  // resource lifetime. Both layout paths must preserve the same preferences.
  for (size_t unused_depth : {0u, 32u}) {
    SCOPED_TRACE(unused_depth);
    auto nodes = requested_nodes;
    if (unused_depth) {
      const size_t first = nodes.size();
      nodes[2].next_sibling = first;
      for (size_t i = 0; i < unused_depth; ++i) {
        nodes.push_back({i + 1 == unused_depth
                             ? LOOM_PIPELINE_LIFETIME_OWNER
                             : LOOM_PIPELINE_LIFETIME_CONCURRENT,
                         5, i ? first + i - 1 : 0,
                         i + 1 == unused_depth ? none : first + i + 1, none});
      }
    }
    loom_pipeline_lifetimes_t lifetimes = {
        .nodes = nodes.data(),
        .count = nodes.size(),
        .has_serial = true,
    };
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes));
    for (size_t pool_count : {3u, 4096u}) {
      SCOPED_TRACE(pool_count);
      const size_t last_pool = pool_count - 1;
      loom_pipeline_lifetime_request_t requests[] = {
          {.lifetime = 5, .pool = 0, .length = 1, .alignment = 1},
          {.lifetime = 6, .pool = 0, .length = 1, .alignment = 1},
          {.lifetime = 7, .pool = 0, .length = 1, .alignment = 1},
          {.lifetime = 8, .pool = 0, .length = 1, .alignment = 1},
          {.lifetime = 6, .pool = 1, .length = 1, .alignment = 1},
          {.lifetime = 7, .pool = 1, .length = 1, .alignment = 1},
          {.lifetime = 5, .pool = last_pool, .length = 1, .alignment = 1},
          {.lifetime = 8, .pool = last_pool, .length = 1, .alignment = 1},
          {.lifetime = 5, .pool = 1, .length = 1, .alignment = 1},
          {.lifetime = 8, .pool = 1, .length = 1, .alignment = 1},
          {.lifetime = 6, .pool = last_pool, .length = 1, .alignment = 1},
          {.lifetime = 7, .pool = last_pool, .length = 1, .alignment = 1},
      };
      const loom_pipeline_lifetime_extent_t* extents = nullptr;
      IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
          &lifetimes, requests, IREE_ARRAYSIZE(requests), pool_count, &arena_,
          &extents));
      const uint64_t offsets[] = {0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1};
      for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
        EXPECT_EQ(requests[i].offset, offsets[i]);
      }
      EXPECT_EQ(extents[0].length, 2u);
      EXPECT_EQ(extents[1].length, 2u);
      EXPECT_EQ(extents[last_pool].length, 2u);
    }
  }
}

}  // namespace
