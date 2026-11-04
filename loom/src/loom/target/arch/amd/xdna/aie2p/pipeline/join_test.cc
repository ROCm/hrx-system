// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/join.h"

#include <array>
#include <cstdlib>
#include <map>
#include <random>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class JoinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  size_t Tile(uint16_t column, uint16_t row) const {
    return column * family_->row_count + row;
  }
  loom_xdna_tile_coordinate_t Coordinate(size_t tile) const {
    return {static_cast<uint16_t>(tile / family_->row_count),
            static_cast<uint16_t>(tile % family_->row_count)};
  }
  void Build() {
    IREE_ASSERT_OK(loom_aie2p_join_plan_build(family_, participants_.data(),
                                              allocated_.data(), &arena_,
                                              &plan_, &valid_));
  }

  // Execute the selected instruction sequences with independently progressing
  // cores. A core may enter the next visit while another still consumes the
  // preceding departure. Every exit must observe all arrivals for its visit.
  void CheckRepeatedProgress(uint32_t seed) {
    constexpr size_t kVisits = 17;
    std::map<std::pair<size_t, uint16_t>, int> locks;
    for (size_t i = 0; i < plan_.edge_count; ++i) {
      const auto& pair = plan_.locks[i];
      ASSERT_TRUE(
          locks.emplace(std::make_pair(pair.owner, pair.first), 0).second);
      ASSERT_TRUE(
          locks.emplace(std::make_pair(pair.owner, pair.first + 1), 0).second);
    }
    std::array<size_t, 48> positions{}, visits{}, entered{};
    size_t remaining = 0;
    for (size_t i = 0; i < participants_.size(); ++i) {
      if (participants_[i]) {
        remaining += kVisits;
      }
    }
    std::mt19937 random(seed);
    while (remaining) {
      bool progressed = false;
      const size_t start = random() % participants_.size();
      for (size_t j = 0; j < participants_.size(); ++j) {
        const size_t node = (start + j) % participants_.size();
        if (!participants_[node] || visits[node] == kVisits) {
          continue;
        }
        const auto& program = plan_.programs[node];
        entered[node] = visits[node] + 1;
        if (positions[node] == program.count) {
          for (size_t other = 0; other < participants_.size(); ++other) {
            if (participants_[other]) {
              ASSERT_GE(entered[other], visits[node] + 1);
            }
          }
          ++visits[node];
          positions[node] = 0;
          --remaining;
          progressed = true;
          break;
        }
        const auto& action = program.actions[positions[node]];
        const auto coordinate = Coordinate(node);
        const auto* facts = loom_xdna_array_tile_facts(family_, coordinate);
        bool resolved = false;
        for (size_t w = 0; w < facts->memory.window_count; ++w) {
          const auto& window =
              family_->address_windows[facts->memory.window_start + w];
          if (action.selector < window.lock_selector_base ||
              action.selector >=
                  window.lock_selector_base + facts->lock_count) {
            continue;
          }
          const auto owner = loom_xdna_tile_coordinate_t{
              static_cast<uint16_t>(coordinate.column +
                                    window.owner_column_delta),
              static_cast<uint16_t>(coordinate.row + window.owner_row_delta)};
          ASSERT_LT(owner.column, family_->column_count);
          ASSERT_LT(owner.row, family_->row_count);
          const uint16_t lock = action.selector - window.lock_selector_base;
          auto it = locks.find({Tile(owner.column, owner.row), lock});
          ASSERT_NE(it, locks.end());
          resolved = true;
          if (action.acquire && !it->second) {
            break;
          }
          it->second += action.acquire ? -1 : 1;
          ASSERT_GE(it->second, 0);
          ASSERT_LE(it->second, 1);
          ++positions[node];
          progressed = true;
          break;
        }
        ASSERT_TRUE(resolved);
        if (progressed) {
          break;
        }
      }
      ASSERT_TRUE(progressed) << "all participating cores blocked";
    }
    for (const auto& [identity, value] : locks) {
      EXPECT_EQ(value, 0);
    }
  }

  // Generated target topology used by selection and instruction interpretation.
  const loom_xdna_array_family_t* family_ = loom_xdna_npu2_array_family();
  // Compute engines active in this join, indexed by physical tile.
  std::array<bool, 48> participants_{};
  // Lock capacity already owned by channels, DMA, and terminal completion.
  std::array<uint16_t, 48> allocated_{};
  // Backing pages for the selection arena.
  iree_arena_block_pool_t pool_ = {};
  // Owns all selected physical plans for the test.
  iree_arena_allocator_t arena_ = {};
  // Selected executable semaphore sequences and initialization requirements.
  loom_aie2p_join_plan_t plan_ = {};
  // Whether the requested participants fit a connected physical lock tree.
  bool valid_ = false;
};

TEST_F(JoinTest, SingletonNeedsNoLocksOrInstructions) {
  participants_[Tile(0, 2)] = true;
  allocated_[Tile(0, 2)] = 16;
  Build();
  ASSERT_TRUE(valid_);
  EXPECT_EQ(plan_.edge_count, 0u);
  EXPECT_EQ(plan_.programs[Tile(0, 2)].count, 0u);
  EXPECT_EQ(allocated_[Tile(0, 2)], 16u);
}

TEST_F(JoinTest, SharedAperturesCoverBothAxesAndDiagonalPeers) {
  for (auto pair :
       {std::pair{Tile(0, 3), Tile(1, 3)}, std::pair{Tile(0, 2), Tile(0, 3)},
        std::pair{Tile(0, 2), Tile(0, 4)}, std::pair{Tile(0, 2), Tile(1, 3)}}) {
    participants_.fill(false);
    allocated_.fill(0);
    participants_[pair.first] = participants_[pair.second] = true;
    Build();
    ASSERT_TRUE(valid_);
    EXPECT_EQ(plan_.edge_count, 1u);
    CheckRepeatedProgress(42);
  }
}

TEST_F(JoinTest, FullArrayFitsAroundOccupiedLocalLocks) {
  // Controllers consume their entire local bank; spare neighbor banks must
  // carry the join without requiring any private semaphore on those cores.
  for (uint16_t column = 0; column < 8; ++column) {
    for (uint16_t row = 2; row < 6; ++row) {
      const size_t tile = Tile(column, row);
      participants_[tile] = true;
      allocated_[tile] = row % 2 ? 9 : 16;
    }
  }
  Build();
  ASSERT_TRUE(valid_);
  EXPECT_EQ(plan_.edge_count, 31u);
  for (uint32_t seed = 0; seed < 32; ++seed) {
    CheckRepeatedProgress(seed);
  }
}

TEST_F(JoinTest, EveryTwoColumnSubsetEitherConnectsOrPreservesResources) {
  for (uint32_t subset = 1; subset < 256; ++subset) {
    participants_.fill(false);
    allocated_.fill(0);
    size_t count = 0;
    for (uint16_t i = 0; i < 8; ++i) {
      if (subset & (1u << i)) {
        participants_[Tile(i / 4, 2 + i % 4)] = true;
        ++count;
      }
    }
    // With empty banks, aperture connectivity alone determines feasibility.
    // Same-column peers can share the intermediate module two rows apart;
    // adjacent columns share the western module within one row vertically.
    std::array<bool, 48> reached{};
    for (size_t i = 0; i < participants_.size(); ++i) {
      if (participants_[i]) {
        reached[i] = true;
        break;
      }
    }
    for (size_t pass = 0; pass < count; ++pass) {
      for (size_t i = 0; i < participants_.size(); ++i) {
        if (!reached[i]) {
          continue;
        }
        for (size_t j = 0; j < participants_.size(); ++j) {
          if (!participants_[j]) {
            continue;
          }
          const auto a = Coordinate(i), b = Coordinate(j);
          const int columns = std::abs(int(a.column) - int(b.column));
          const int rows = std::abs(int(a.row) - int(b.row));
          if (columns <= 1 && rows <= 2 - columns) {
            reached[j] = true;
          }
        }
      }
    }
    Build();
    EXPECT_EQ(valid_, reached == participants_);
    if (valid_) {
      EXPECT_EQ(plan_.edge_count, count - 1);
      if (count > 1) {
        CheckRepeatedProgress(subset);
      }
    } else {
      EXPECT_EQ(allocated_, (std::array<uint16_t, 48>{}));
    }
  }
}

TEST_F(JoinTest, DisconnectedCoresAndExhaustedOwnersFailWithoutReservation) {
  participants_[Tile(0, 2)] = participants_[Tile(7, 5)] = true;
  Build();
  EXPECT_FALSE(valid_);
  EXPECT_EQ(allocated_, (std::array<uint16_t, 48>{}));
  participants_[Tile(7, 5)] = false;
  participants_[Tile(1, 2)] = true;
  allocated_[Tile(0, 2)] = 15;
  const auto before = allocated_;
  Build();
  EXPECT_FALSE(valid_);
  EXPECT_EQ(allocated_, before);
}

}  // namespace
